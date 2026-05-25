"""
T4: Tipping Points & Cascading Bifurcation Analysis
====================================================
Extends Paper1 U1 bistability to multi-contaminant tipping framework.

Analyses:
  1. GMM bistability testing with bootstrap CI on delta-BIC
  2. Tipping threshold (saddle point) identification per contaminant x depth zone
  3. Cascading analysis — conditional P(j in high | i in high), chi-squared
  4. Phase diagrams — PO4 vs ORP colored by As-state; ORP vs Depth colored by Mn-state
  5. Cusp catastrophe fitting — dV/dx = x^3 - alpha*x - beta
  6. Seasonal tipping shift — component weight changes, well-level state transitions

Outputs:
  Figures: T4_F01 .. T4_F05
  Tables:  T4_bistability_all_contaminants.csv, T4_cascade_conditional.csv,
           T4_cusp_parameters.csv
"""

import matplotlib
matplotlib.use('Agg')

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent.parent))

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from sklearn.mixture import GaussianMixture
from scipy.signal import argrelmin
from scipy.optimize import minimize
from scipy.stats import chi2_contingency, gaussian_kde
from itertools import combinations

from config import (
    DATA_FILE, DEPTH_ZONES, PHYSIOGRAPHIC_ZONES,
    TABLES_DIR, FIGURES_DIR, RANDOM_STATE, DPI, assign_zones
)

np.random.seed(RANDOM_STATE)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONTAMINANTS = ['As', 'Mn2+', 'Fe2+', 'Cr3+', 'PO43-', 'NO3-']
CONTAMINANT_LABELS = {
    'As': 'As (µg/L)', 'Mn2+': 'Mn²⁺ (mg/L)', 'Fe2+': 'Fe²⁺ (mg/L)',
    'Cr3+': 'Cr³⁺ (mg/L)', 'PO43-': 'PO₄³⁻ (mg/L)', 'NO3-': 'NO₃⁻ (mg/L)',
}
MAX_COMPONENTS = 4
N_BOOTSTRAP = 500
N_INIT = 50
MIN_SAMPLES = 50


# ═══════════════════════════════════════════════════════════════════════════
# 0. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_data():
    """Load dataset and assign depth + physiographic zones via config."""
    df = pd.read_csv(DATA_FILE)
    df = assign_zones(df)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 1. GMM BISTABILITY WITH BOOTSTRAP CI
# ═══════════════════════════════════════════════════════════════════════════

def fit_gmm(X, n_components, n_init=N_INIT):
    """Fit a single GMM and return it."""
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type='full',
        n_init=n_init,
        max_iter=500,
        random_state=RANDOM_STATE,
    )
    gmm.fit(X)
    return gmm


def test_bistability(values, n_bootstrap=N_BOOTSTRAP):
    """Test for bimodality using GMM-BIC comparison with bootstrap CI.

    Returns dict with best_n, delta_bic, bootstrap CI, bimodal flag, and
    the fitted models.
    """
    X = values.dropna().values.reshape(-1, 1)
    if len(X) < MIN_SAMPLES:
        return None

    # Fit GMM for 1..MAX_COMPONENTS
    models = {}
    for n in range(1, MAX_COMPONENTS + 1):
        gmm = fit_gmm(X, n)
        models[n] = {'bic': gmm.bic(X), 'aic': gmm.aic(X), 'model': gmm}

    best_n = min(models, key=lambda k: models[k]['bic'])
    delta_bic = models[1]['bic'] - models[best_n]['bic']

    # Bootstrap CI on delta-BIC (BIC_1 - BIC_best_n)
    boot_deltas = []
    n_data = len(X)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n_data, size=n_data, replace=True)
        Xb = X[idx]
        try:
            gmm1 = fit_gmm(Xb, 1, n_init=10)
            gmmk = fit_gmm(Xb, best_n, n_init=10)
            boot_deltas.append(gmm1.bic(Xb) - gmmk.bic(Xb))
        except Exception:
            continue

    boot_deltas = np.array(boot_deltas)
    ci_lo, ci_hi = np.nan, np.nan
    if len(boot_deltas) > 10:
        ci_lo = np.percentile(boot_deltas, 2.5)
        ci_hi = np.percentile(boot_deltas, 97.5)

    return {
        'best_n': best_n,
        'delta_bic': delta_bic,
        'ci_lo': ci_lo,
        'ci_hi': ci_hi,
        'bimodal': best_n >= 2 and delta_bic > 10,
        'models': models,
        'X': X,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2. TIPPING THRESHOLD (SADDLE POINT) IDENTIFICATION
# ═══════════════════════════════════════════════════════════════════════════

def find_saddle_point(gmm, X):
    """Find the saddle (local minimum of GMM density) between two components.

    Returns saddle_x (in transformed space) or None.
    """
    if gmm.n_components < 2:
        return None

    lo, hi = X.min(), X.max()
    margin = 0.05 * (hi - lo)
    x = np.linspace(lo - margin, hi + margin, 2000).reshape(-1, 1)
    log_prob = gmm.score_samples(x)
    prob = np.exp(log_prob)

    minima = argrelmin(prob, order=30)[0]
    if len(minima) == 0:
        return None

    # Return the deepest minimum (most prominent saddle)
    deepest = minima[np.argmin(prob[minima])]
    return float(x[deepest, 0])


def saddle_to_original(saddle_log, transform='log1p'):
    """Convert saddle in log1p space back to original units."""
    if saddle_log is None:
        return None
    return float(np.expm1(saddle_log))


# ═══════════════════════════════════════════════════════════════════════════
# 3. CASCADING ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def assign_gmm_states(df, contaminant, gmm, transform='log1p'):
    """Assign each sample to high/low state based on 2-component GMM.

    High state = component with higher mean.
    Returns a Series of 'high'/'low' aligned with df index, NaN where missing.
    """
    col = contaminant
    mask = df[col].notna()
    vals = np.log1p(df.loc[mask, col].values).reshape(-1, 1)
    labels = gmm.predict(vals)

    # Determine which component is 'high'
    means = gmm.means_.flatten()
    high_comp = np.argmax(means)

    states = pd.Series(np.nan, index=df.index, dtype=object)
    state_arr = np.where(labels == high_comp, 'high', 'low')
    states.loc[mask] = state_arr
    return states


def cascading_analysis(df, state_dict):
    """Compute conditional probability matrix and chi-squared tests.

    P(j in high | i in high) for all contaminant pairs.
    """
    contaminants = list(state_dict.keys())
    n = len(contaminants)
    cond_prob = pd.DataFrame(np.nan, index=contaminants, columns=contaminants)
    chi2_pval = pd.DataFrame(np.nan, index=contaminants, columns=contaminants)

    for i_name in contaminants:
        for j_name in contaminants:
            si = state_dict[i_name]
            sj = state_dict[j_name]
            valid = si.notna() & sj.notna()
            if valid.sum() < 20:
                continue

            si_v = si[valid]
            sj_v = sj[valid]

            # P(j=high | i=high)
            i_high = si_v == 'high'
            if i_high.sum() > 0:
                cond_prob.loc[i_name, j_name] = (sj_v[i_high] == 'high').mean()
            else:
                cond_prob.loc[i_name, j_name] = np.nan

            # Chi-squared
            if i_name != j_name:
                ct = pd.crosstab(si_v, sj_v)
                if ct.shape == (2, 2) and ct.values.min() >= 5:
                    _, p, _, _ = chi2_contingency(ct)
                    chi2_pval.loc[i_name, j_name] = p
                else:
                    chi2_pval.loc[i_name, j_name] = np.nan
            else:
                chi2_pval.loc[i_name, j_name] = np.nan

    return cond_prob, chi2_pval


# ═══════════════════════════════════════════════════════════════════════════
# 5. CUSP CATASTROPHE FITTING
# ═══════════════════════════════════════════════════════════════════════════

def cusp_potential_gradient(x, alpha, beta):
    """Gradient of cusp potential: dV/dx = x^3 - alpha*x - beta."""
    return x**3 - alpha * x - beta


def fit_cusp(x_obs, alpha_proxy, beta_proxy):
    """Fit cusp catastrophe via regularised least-squares on equilibrium manifold.

    The cusp canonical form: dV/dx = x^3 - alpha*x - beta = 0 at equilibrium.
    Linear maps:
        x     = a0 + a1 * x_obs       (state variable)
        alpha = b0 + b1 * alpha_proxy  (normal factor)
        beta  = c0 + c1 * beta_proxy   (splitting factor)

    Minimises sum of (x^3 - alpha*x - beta)^2 with a penalty on small |a1|
    to prevent the degenerate a1->0 trivial solution.
    """
    mask = np.isfinite(x_obs) & np.isfinite(alpha_proxy) & np.isfinite(beta_proxy)
    x_obs = x_obs[mask]
    alpha_proxy = alpha_proxy[mask]
    beta_proxy = beta_proxy[mask]

    if len(x_obs) < 30:
        return None

    n = len(x_obs)

    # Standardise inputs for numerical stability
    x_mu, x_sd = x_obs.mean(), x_obs.std() + 1e-12
    a_mu, a_sd = alpha_proxy.mean(), alpha_proxy.std() + 1e-12
    b_mu, b_sd = beta_proxy.mean(), beta_proxy.std() + 1e-12

    x_s = (x_obs - x_mu) / x_sd
    a_s = (alpha_proxy - a_mu) / a_sd
    b_s = (beta_proxy - b_mu) / b_sd

    # Penalty weight to prevent a1->0 (degenerate solution)
    lambda_reg = n * 10.0

    def objective(params):
        a0, a1, b0, b1, c0, c1 = params
        x = a0 + a1 * x_s
        alpha = b0 + b1 * a_s
        beta = c0 + c1 * b_s
        residual = x**3 - alpha * x - beta
        # Main loss + penalty for small a1 (encourage a1 away from 0)
        loss = np.sum(residual**2) + lambda_reg / (a1**2 + 1e-6)
        if not np.isfinite(loss):
            return 1e15
        return loss

    best_res = None
    for trial in range(15):
        p0 = np.random.randn(6) * 0.5
        p0[1] = np.random.uniform(0.5, 2.0)  # a1 > 0
        try:
            res = minimize(objective, p0, method='Nelder-Mead',
                           options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8})
            if best_res is None or res.fun < best_res.fun:
                best_res = res
        except Exception:
            continue

    if best_res is None or best_res.fun >= 1e14:
        return None

    a0, a1, b0, b1, c0, c1 = best_res.x

    # Reject degenerate solutions
    if abs(a1) < 0.1:
        return None

    # Compute fitted values
    x_fit = a0 + a1 * x_s
    alpha_fit = b0 + b1 * a_s
    beta_fit = c0 + c1 * b_s

    # Equilibrium residual
    eq_residual = x_fit**3 - alpha_fit * x_fit - beta_fit
    ss_res = np.sum(eq_residual**2)
    rmse = np.sqrt(np.mean(eq_residual**2))

    # Pseudo-R^2: compare cusp manifold fit to a linear null (x^3 ~ constant)
    x_cubed = x_fit**3
    ss_null = np.sum((x_cubed - x_cubed.mean())**2)
    pseudo_r2 = 1.0 - ss_res / (ss_null + 1e-12) if ss_null > 1e-12 else 0.0

    return {
        'a0': a0, 'a1': a1,
        'b0_normal': b0, 'b1_normal': b1,
        'c0_splitting': c0, 'c1_splitting': c1,
        'pseudo_r2': pseudo_r2,
        'eq_rmse': rmse,
        'loss': best_res.fun,
        'n_samples': n,
        'x_fit': x_fit,
        'alpha_fit': alpha_fit,
        'beta_fit': beta_fit,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 6. SEASONAL TIPPING SHIFT
# ═══════════════════════════════════════════════════════════════════════════

def seasonal_shift(df, contaminant, gmm):
    """Compare GMM state assignments between dry and wet season.

    Returns dict with counts of wells transitioning.
    """
    state_col = f'{contaminant}_state'
    states = assign_gmm_states(df, contaminant, gmm)
    df_tmp = df.copy()
    df_tmp[state_col] = states

    dry = df_tmp[df_tmp['Season'].str.lower().str.contains('dry', na=False)]
    wet = df_tmp[df_tmp['Season'].str.lower().str.contains('wet', na=False)]

    return {
        'contaminant': contaminant,
        'dry_n': len(dry),
        'wet_n': len(wet),
        'dry_high_frac': (dry[state_col] == 'high').mean() if len(dry) > 0 else np.nan,
        'wet_high_frac': (wet[state_col] == 'high').mean() if len(wet) > 0 else np.nan,
        'dry_high_count': (dry[state_col] == 'high').sum(),
        'wet_high_count': (wet[state_col] == 'high').sum(),
        'dry_low_count': (dry[state_col] == 'low').sum(),
        'wet_low_count': (wet[state_col] == 'low').sum(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def plot_gmm_densities(all_results, df):
    """T4_F01: GMM density plots for 6 contaminants (2x3 grid)."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, contam in enumerate(CONTAMINANTS):
        ax = axes[idx]
        if contam not in df.columns:
            ax.set_visible(False)
            continue

        values = np.log1p(df[contam].dropna().values)
        X = values.reshape(-1, 1)

        # Use the overall (all zones) best GMM
        overall_key = (contam, 'Overall')
        if overall_key not in all_results or all_results[overall_key] is None:
            # Fit overall
            result = test_bistability(pd.Series(values), n_bootstrap=0)
            if result is None:
                ax.text(0.5, 0.5, f'{contam}\nInsufficient data',
                        transform=ax.transAxes, ha='center')
                continue
            all_results[overall_key] = result
        result = all_results[overall_key]

        best_n = result['best_n']
        gmm = result['models'][best_n]['model']

        # Data histogram
        ax.hist(values, bins=60, density=True, alpha=0.3, color='steelblue',
                edgecolor='none', label='Data')

        # GMM total density
        x_plot = np.linspace(values.min() - 0.5, values.max() + 0.5, 1000).reshape(-1, 1)
        log_prob = gmm.score_samples(x_plot)
        total_density = np.exp(log_prob)
        ax.plot(x_plot.flatten(), total_density, 'k-', lw=2, label='GMM total')

        # Individual components
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
        for k in range(gmm.n_components):
            weight = gmm.weights_[k]
            mean = gmm.means_[k, 0]
            var = gmm.covariances_[k, 0, 0]
            comp_density = weight * (1 / np.sqrt(2 * np.pi * var)) * \
                np.exp(-0.5 * (x_plot.flatten() - mean)**2 / var)
            ax.plot(x_plot.flatten(), comp_density, '--', color=colors[k % len(colors)],
                    lw=1.5, label=f'Comp {k+1} (w={weight:.2f})')

        # Saddle point
        saddle = find_saddle_point(gmm, X)
        if saddle is not None:
            saddle_density = np.exp(gmm.score_samples(np.array([[saddle]])))
            ax.axvline(saddle, color='red', ls=':', lw=1.5, alpha=0.8)
            ax.plot(saddle, saddle_density[0], 'rv', ms=10, zorder=5,
                    label=f'Saddle={np.expm1(saddle):.2f}')

        label = CONTAMINANT_LABELS.get(contam, contam)
        ax.set_title(f'{label}  (k={best_n}, ΔBIC={result["delta_bic"]:.0f})',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel(f'log₁ₚ({contam})')
        ax.set_ylabel('Density')
        ax.legend(fontsize=7, loc='upper right')

    plt.suptitle('T4-F01: GMM Bistability — Density & Saddle Points',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'T4_F01_gmm_densities.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved {FIGURES_DIR / 'T4_F01_gmm_densities.png'}")


def plot_phase_diagram_as(df, state_dict):
    """T4_F02: PO4 vs ORP colored by As GMM state."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: PO4 vs ORP colored by As state
    ax = axes[0]
    if 'As' in state_dict and 'PO43-' in df.columns and 'ORP' in df.columns:
        mask = state_dict['As'].notna() & df['PO43-'].notna() & df['ORP'].notna()
        states = state_dict['As'][mask]
        po4 = df.loc[mask, 'PO43-']
        orp = df.loc[mask, 'ORP']

        colors = np.where(states == 'high', '#e74c3c', '#3498db')
        ax.scatter(po4[states == 'low'], orp[states == 'low'],
                   c='#3498db', alpha=0.4, s=15, label='As low-state', edgecolors='none')
        ax.scatter(po4[states == 'high'], orp[states == 'high'],
                   c='#e74c3c', alpha=0.5, s=15, label='As high-state', edgecolors='none')
        ax.set_xlabel('PO₄³⁻ (mg/L)', fontsize=12)
        ax.set_ylabel('ORP (mV)', fontsize=12)
        ax.set_title('(a) PO₄ vs ORP — As state', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
    else:
        ax.text(0.5, 0.5, 'Data not available', transform=ax.transAxes, ha='center')

    # Panel B: ORP vs Depth colored by Mn state
    ax = axes[1]
    if 'Mn2+' in state_dict and 'ORP' in df.columns:
        mask = state_dict['Mn2+'].notna() & df['ORP'].notna() & df['Depth'].notna()
        states = state_dict['Mn2+'][mask]
        orp = df.loc[mask, 'ORP']
        depth = df.loc[mask, 'Depth']

        ax.scatter(orp[states == 'low'], depth[states == 'low'],
                   c='#3498db', alpha=0.4, s=15, label='Mn low-state', edgecolors='none')
        ax.scatter(orp[states == 'high'], depth[states == 'high'],
                   c='#e74c3c', alpha=0.5, s=15, label='Mn high-state', edgecolors='none')
        ax.set_xlabel('ORP (mV)', fontsize=12)
        ax.set_ylabel('Depth (m)', fontsize=12)
        ax.invert_yaxis()
        ax.set_title('(b) ORP vs Depth — Mn state', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
    else:
        ax.text(0.5, 0.5, 'Data not available', transform=ax.transAxes, ha='center')

    plt.suptitle('T4-F02: Phase Diagrams by GMM State',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'T4_F02_phase_diagrams.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved {FIGURES_DIR / 'T4_F02_phase_diagrams.png'}")


def plot_cascade_heatmap(cond_prob, chi2_pval):
    """T4_F03: Cascading conditional probability heatmap."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: Conditional probabilities
    ax = axes[0]
    data = cond_prob.astype(float).values
    mask_nan = np.isnan(data)
    data_plot = np.where(mask_nan, 0, data)

    im = ax.imshow(data_plot, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(cond_prob.columns)))
    ax.set_yticks(range(len(cond_prob.index)))
    ax.set_xticklabels(cond_prob.columns, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(cond_prob.index, fontsize=9)
    ax.set_xlabel('Contaminant j', fontsize=11)
    ax.set_ylabel('Contaminant i (conditioned on high)', fontsize=11)
    ax.set_title('(a) P(j in high | i in high)', fontsize=11, fontweight='bold')

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not mask_nan[i, j]:
                ax.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center',
                        fontsize=8, color='black' if data[i, j] < 0.6 else 'white')
            else:
                ax.text(j, i, '—', ha='center', va='center', fontsize=8, color='gray')

    plt.colorbar(im, ax=ax, fraction=0.046)

    # Panel B: Chi-squared p-values (log10)
    ax = axes[1]
    pdata = chi2_pval.astype(float).values
    mask_nan2 = np.isnan(pdata)
    log_p = np.where(mask_nan2, 0, -np.log10(pdata + 1e-300))

    im2 = ax.imshow(log_p, cmap='RdPu', vmin=0, vmax=max(10, np.nanmax(log_p)),
                    aspect='auto')
    ax.set_xticks(range(len(chi2_pval.columns)))
    ax.set_yticks(range(len(chi2_pval.index)))
    ax.set_xticklabels(chi2_pval.columns, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(chi2_pval.index, fontsize=9)
    ax.set_xlabel('Contaminant j', fontsize=11)
    ax.set_ylabel('Contaminant i', fontsize=11)
    ax.set_title('(b) −log₁₀(p) Chi-squared', fontsize=11, fontweight='bold')

    for i in range(pdata.shape[0]):
        for j in range(pdata.shape[1]):
            if i == j:
                ax.text(j, i, '—', ha='center', va='center', fontsize=8, color='gray')
            elif not mask_nan2[i, j]:
                sig = '***' if pdata[i, j] < 0.001 else ('**' if pdata[i, j] < 0.01
                        else ('*' if pdata[i, j] < 0.05 else 'ns'))
                ax.text(j, i, sig, ha='center', va='center', fontsize=8,
                        color='black' if log_p[i, j] < 5 else 'white')
            else:
                ax.text(j, i, '—', ha='center', va='center', fontsize=8, color='gray')

    plt.colorbar(im2, ax=ax, fraction=0.046)

    plt.suptitle('T4-F03: Cascading Bifurcation — Conditional Probability Matrix',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'T4_F03_cascade_heatmap.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved {FIGURES_DIR / 'T4_F03_cascade_heatmap.png'}")


def plot_cusp_surface(cusp_result, df):
    """T4_F04: Cusp catastrophe surface for As-PO4-ORP system."""
    fig = plt.figure(figsize=(14, 6))

    if cusp_result is None:
        fig.text(0.5, 0.5, 'Cusp fitting failed — insufficient data or convergence',
                 ha='center', fontsize=12)
        plt.savefig(FIGURES_DIR / 'T4_F04_cusp_surface.png', dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"  Saved {FIGURES_DIR / 'T4_F04_cusp_surface.png'} (placeholder)")
        return

    # Panel A: 3D scatter of fitted cusp surface
    ax1 = fig.add_subplot(121, projection='3d')
    alpha_fit = cusp_result['alpha_fit']
    beta_fit = cusp_result['beta_fit']
    x_fit = cusp_result['x_fit']

    # Create theoretical surface
    a_range = np.linspace(alpha_fit.min(), alpha_fit.max(), 50)
    b_range = np.linspace(beta_fit.min(), beta_fit.max(), 50)
    A, B = np.meshgrid(a_range, b_range)

    # For each (alpha, beta), find equilibrium x from x^3 - alpha*x - beta = 0
    # Just plot data points colored by x
    sc = ax1.scatter(alpha_fit, beta_fit, x_fit, c=x_fit, cmap='RdYlBu_r',
                     s=5, alpha=0.5)
    ax1.set_xlabel('Normal (α, ORP-derived)', fontsize=9)
    ax1.set_ylabel('Splitting (β, PO₄-derived)', fontsize=9)
    ax1.set_zlabel('State (x, As-derived)', fontsize=9)
    ax1.set_title(f'(a) Cusp manifold\nR²={cusp_result["pseudo_r2"]:.3f}',
                  fontsize=11, fontweight='bold')
    plt.colorbar(sc, ax=ax1, shrink=0.5, label='As state variable')

    # Panel B: Bifurcation set in (alpha, beta) plane
    ax2 = fig.add_subplot(122)

    # Bifurcation curve: 4*alpha^3 = 27*beta^2
    alpha_curve = np.linspace(0, max(3, alpha_fit.max()), 200)
    beta_pos = np.sqrt(4 * alpha_curve**3 / 27)
    beta_neg = -beta_pos

    ax2.fill_between(alpha_curve, beta_neg, beta_pos, alpha=0.15, color='red',
                     label='Bistable region')
    ax2.plot(alpha_curve, beta_pos, 'r-', lw=2)
    ax2.plot(alpha_curve, beta_neg, 'r-', lw=2)
    ax2.scatter(alpha_fit, beta_fit, c=x_fit, cmap='RdYlBu_r', s=8, alpha=0.5,
                edgecolors='none')
    ax2.set_xlabel('Normal factor α (ORP-derived)', fontsize=11)
    ax2.set_ylabel('Splitting factor β (PO₄-derived)', fontsize=11)
    ax2.set_title('(b) Bifurcation set & data', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=10)

    plt.suptitle('T4-F04: Cusp Catastrophe — As × PO₄ × ORP System',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'T4_F04_cusp_surface.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved {FIGURES_DIR / 'T4_F04_cusp_surface.png'}")


def plot_seasonal_shift(seasonal_results):
    """T4_F05: Seasonal tipping shift bar chart."""
    sr = [r for r in seasonal_results if r is not None]
    if not sr:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No seasonal data available', transform=ax.transAxes, ha='center')
        plt.savefig(FIGURES_DIR / 'T4_F05_seasonal_shift.png', dpi=DPI, bbox_inches='tight')
        plt.close()
        return

    df_s = pd.DataFrame(sr)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Fraction in high state by season
    ax = axes[0]
    x_pos = np.arange(len(df_s))
    width = 0.35
    ax.bar(x_pos - width/2, df_s['dry_high_frac'], width, label='Dry season',
           color='#f39c12', edgecolor='black', linewidth=0.5)
    ax.bar(x_pos + width/2, df_s['wet_high_frac'], width, label='Wet season',
           color='#3498db', edgecolor='black', linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df_s['contaminant'], rotation=30, ha='right')
    ax.set_ylabel('Fraction in high-state', fontsize=11)
    ax.set_title('(a) High-state fraction by season', fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1)

    # Add delta labels
    for i, row in df_s.iterrows():
        delta = row['wet_high_frac'] - row['dry_high_frac']
        if np.isfinite(delta):
            sign = '+' if delta >= 0 else ''
            ax.text(i, max(row['dry_high_frac'], row['wet_high_frac']) + 0.02,
                    f'{sign}{delta:.1%}', ha='center', fontsize=8, fontweight='bold')

    # Panel B: Absolute counts transitioning
    ax = axes[1]
    ax.bar(x_pos - width/2, df_s['dry_high_count'], width, label='Dry high',
           color='#e67e22', edgecolor='black', linewidth=0.5)
    ax.bar(x_pos + width/2, df_s['wet_high_count'], width, label='Wet high',
           color='#2980b9', edgecolor='black', linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df_s['contaminant'], rotation=30, ha='right')
    ax.set_ylabel('Number of wells in high-state', fontsize=11)
    ax.set_title('(b) High-state well counts by season', fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)

    plt.suptitle('T4-F05: Seasonal Tipping Shift — Dry vs Wet Monsoon',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'T4_F05_seasonal_shift.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved {FIGURES_DIR / 'T4_F05_seasonal_shift.png'}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("T4: TIPPING POINTS & CASCADING BIFURCATION ANALYSIS")
    print("=" * 70)

    df = load_data()
    print(f"Loaded {len(df)} samples with zones assigned.")
    print(f"Depth zones: {df['depth_zone'].value_counts().to_dict()}")
    print(f"Seasons: {df['Season'].value_counts().to_dict()}")

    # ── 1 & 2: GMM BISTABILITY + SADDLE POINTS ──────────────────────────
    print("\n" + "─" * 70)
    print("1–2. GMM BISTABILITY TESTING & TIPPING THRESHOLDS")
    print("─" * 70)

    all_results = {}   # keyed by (contaminant, zone)
    table_rows = []

    for contam in CONTAMINANTS:
        if contam not in df.columns:
            print(f"  {contam}: column not found, skipping.")
            continue
        print(f"\n  ▸ {contam}")

        # Per depth zone
        for zone in list(DEPTH_ZONES.keys()) + ['Overall']:
            if zone == 'Overall':
                values = np.log1p(df[contam].dropna())
            else:
                mask = df['depth_zone'] == zone
                values = np.log1p(df.loc[mask, contam].dropna())

            if len(values) < MIN_SAMPLES:
                print(f"    {zone}: n={len(values)} < {MIN_SAMPLES}, skipped")
                continue

            result = test_bistability(values, n_bootstrap=N_BOOTSTRAP)
            if result is None:
                continue

            all_results[(contam, zone)] = result

            # Saddle point
            best_gmm = result['models'][result['best_n']]['model']
            saddle_log = find_saddle_point(best_gmm, result['X'])
            saddle_orig = saddle_to_original(saddle_log)

            print(f"    {zone}: k={result['best_n']}, "
                  f"ΔBIC={result['delta_bic']:.1f} "
                  f"[{result['ci_lo']:.1f}, {result['ci_hi']:.1f}], "
                  f"bimodal={result['bimodal']}, "
                  f"saddle={'%.3f' % saddle_orig if saddle_orig else 'None'}")

            # Component means and weights in original space
            means_orig = np.expm1(best_gmm.means_.flatten())
            weights = best_gmm.weights_.flatten()
            sorted_idx = np.argsort(means_orig)

            row = {
                'contaminant': contam,
                'depth_zone': zone,
                'n_samples': len(result['X']),
                'best_n_components': result['best_n'],
                'delta_bic': round(result['delta_bic'], 1),
                'delta_bic_ci_lo': round(result['ci_lo'], 1),
                'delta_bic_ci_hi': round(result['ci_hi'], 1),
                'bimodal': result['bimodal'],
                'saddle_log1p': round(saddle_log, 4) if saddle_log else None,
                'saddle_original': round(saddle_orig, 4) if saddle_orig else None,
            }
            # Add component info (up to 4)
            for k_idx, si in enumerate(sorted_idx):
                row[f'comp{k_idx+1}_mean'] = round(float(means_orig[si]), 4)
                row[f'comp{k_idx+1}_weight'] = round(float(weights[si]), 4)

            table_rows.append(row)

    bistability_df = pd.DataFrame(table_rows)
    bistability_df.to_csv(TABLES_DIR / 'T4_bistability_all_contaminants.csv', index=False)
    print(f"\n  → Saved {TABLES_DIR / 'T4_bistability_all_contaminants.csv'}")

    # ── 3. CASCADING ANALYSIS ────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("3. CASCADING ANALYSIS — CONDITIONAL PROBABILITIES")
    print("─" * 70)

    # Assign GMM states for each contaminant using overall model
    state_dict = {}
    for contam in CONTAMINANTS:
        key = (contam, 'Overall')
        if key in all_results and all_results[key] is not None:
            result = all_results[key]
            best_n = result['best_n']
            if best_n >= 2:
                gmm = result['models'][best_n]['model']
                state_dict[contam] = assign_gmm_states(df, contam, gmm)
                n_high = (state_dict[contam] == 'high').sum()
                n_low = (state_dict[contam] == 'low').sum()
                print(f"  {contam}: {n_high} high, {n_low} low")
            else:
                # For unimodal: use median split as fallback
                vals = df[contam].dropna()
                median_val = vals.median()
                states = pd.Series(np.nan, index=df.index, dtype=object)
                states.loc[df[contam] > median_val] = 'high'
                states.loc[df[contam] <= median_val] = 'low'
                state_dict[contam] = states
                print(f"  {contam}: unimodal — using median split "
                      f"(threshold={median_val:.3f})")

    if len(state_dict) >= 2:
        cond_prob, chi2_pval = cascading_analysis(df, state_dict)
        print(f"\n  Conditional probability matrix:\n{cond_prob.to_string()}")
        print(f"\n  Chi-squared p-values:\n{chi2_pval.to_string()}")

        # Save cascade table
        cascade_rows = []
        contams = list(state_dict.keys())
        for i_name in contams:
            for j_name in contams:
                cascade_rows.append({
                    'contaminant_i': i_name,
                    'contaminant_j': j_name,
                    'P_j_high_given_i_high': cond_prob.loc[i_name, j_name]
                        if i_name in cond_prob.index and j_name in cond_prob.columns
                        else np.nan,
                    'chi2_pvalue': chi2_pval.loc[i_name, j_name]
                        if i_name in chi2_pval.index and j_name in chi2_pval.columns
                        else np.nan,
                })
        cascade_df = pd.DataFrame(cascade_rows)
        cascade_df.to_csv(TABLES_DIR / 'T4_cascade_conditional.csv', index=False)
        print(f"\n  → Saved {TABLES_DIR / 'T4_cascade_conditional.csv'}")
    else:
        cond_prob, chi2_pval = None, None
        print("  Insufficient contaminants with bimodal states for cascading analysis.")

    # ── 5. CUSP CATASTROPHE FITTING ──────────────────────────────────────
    print("\n" + "─" * 70)
    print("5. CUSP CATASTROPHE FITTING — As × PO₄ × ORP")
    print("─" * 70)

    cusp_result = None
    required_cols = {'As', 'PO43-', 'ORP'}
    if required_cols.issubset(df.columns):
        mask = df[['As', 'PO43-', 'ORP']].notna().all(axis=1)
        x_obs = np.log1p(df.loc[mask, 'As'].values)
        alpha_proxy = df.loc[mask, 'ORP'].values   # Normal factor
        beta_proxy = df.loc[mask, 'PO43-'].values  # Splitting factor

        cusp_result = fit_cusp(x_obs, alpha_proxy, beta_proxy)

        if cusp_result is not None:
            print(f"  Cusp fit: n={cusp_result['n_samples']}, "
                  f"pseudo-R²={cusp_result['pseudo_r2']:.4f}, "
                  f"eq_RMSE={cusp_result['eq_rmse']:.3f}")
            print(f"    a0={cusp_result['a0']:.4f}, a1={cusp_result['a1']:.4f}")
            print(f"    Normal:   b0={cusp_result['b0_normal']:.4f}, "
                  f"b1={cusp_result['b1_normal']:.4f}")
            print(f"    Splitting: c0={cusp_result['c0_splitting']:.4f}, "
                  f"c1={cusp_result['c1_splitting']:.4f}")

            # Also fit As-Fe-ORP and As-PO4-Depth for comparison
            cusp_rows = []
            systems = [
                ('As', 'ORP', 'PO43-', 'As × ORP × PO₄'),
                ('As', 'ORP', 'Fe2+', 'As × ORP × Fe²⁺'),
                ('As', 'Depth', 'PO43-', 'As × Depth × PO₄'),
            ]
            for x_col, a_col, b_col, label in systems:
                cols_needed = {x_col, a_col, b_col}
                if not cols_needed.issubset(df.columns):
                    continue
                m = df[[x_col, a_col, b_col]].notna().all(axis=1)
                if m.sum() < 50:
                    continue
                xv = np.log1p(df.loc[m, x_col].values)
                av = df.loc[m, a_col].values
                bv = df.loc[m, b_col].values
                cr = fit_cusp(xv, av, bv)
                if cr is not None:
                    cusp_rows.append({
                        'system': label,
                        'state_var': x_col,
                        'normal_factor': a_col,
                        'splitting_factor': b_col,
                        'n_samples': cr['n_samples'],
                        'a0': round(cr['a0'], 4),
                        'a1': round(cr['a1'], 4),
                        'b0_normal': round(cr['b0_normal'], 4),
                        'b1_normal': round(cr['b1_normal'], 4),
                        'c0_splitting': round(cr['c0_splitting'], 4),
                        'c1_splitting': round(cr['c1_splitting'], 4),
                        'pseudo_r2': round(cr['pseudo_r2'], 4),
                        'eq_rmse': round(cr['eq_rmse'], 4),
                    })
                    print(f"  {label}: pseudo-R²={cr['pseudo_r2']:.4f}, "
                          f"eq_RMSE={cr['eq_rmse']:.3f}")

            cusp_df = pd.DataFrame(cusp_rows)
            cusp_df.to_csv(TABLES_DIR / 'T4_cusp_parameters.csv', index=False)
            print(f"\n  → Saved {TABLES_DIR / 'T4_cusp_parameters.csv'}")
        else:
            print("  Cusp fitting did not converge.")
            # Save empty table
            pd.DataFrame(columns=['system', 'state_var', 'normal_factor',
                                  'splitting_factor', 'n_samples', 'pseudo_r2']
                         ).to_csv(TABLES_DIR / 'T4_cusp_parameters.csv', index=False)
    else:
        missing = required_cols - set(df.columns)
        print(f"  Missing columns for cusp: {missing}")
        pd.DataFrame().to_csv(TABLES_DIR / 'T4_cusp_parameters.csv', index=False)

    # ── 6. SEASONAL TIPPING SHIFT ────────────────────────────────────────
    print("\n" + "─" * 70)
    print("6. SEASONAL TIPPING SHIFT")
    print("─" * 70)

    seasonal_results = []
    for contam in CONTAMINANTS:
        key = (contam, 'Overall')
        if key in all_results and all_results[key] is not None:
            best_n = all_results[key]['best_n']
            gmm = all_results[key]['models'][best_n]['model']
            sr = seasonal_shift(df, contam, gmm)
            seasonal_results.append(sr)
            delta = sr['wet_high_frac'] - sr['dry_high_frac']
            print(f"  {contam}: dry_high={sr['dry_high_frac']:.3f}, "
                  f"wet_high={sr['wet_high_frac']:.3f}, "
                  f"Δ={delta:+.3f} ({delta*100:+.1f} pp)")

    # ── FIGURES ──────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("GENERATING FIGURES")
    print("─" * 70)

    # F01: GMM density plots
    plot_gmm_densities(all_results, df)

    # F02: Phase diagrams
    plot_phase_diagram_as(df, state_dict)

    # F03: Cascade heatmap
    if cond_prob is not None:
        plot_cascade_heatmap(cond_prob, chi2_pval)
    else:
        print("  Skipping T4_F03 (no cascade data).")

    # F04: Cusp surface
    plot_cusp_surface(cusp_result, df)

    # F05: Seasonal shift
    plot_seasonal_shift(seasonal_results)

    # ── SUMMARY ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("T4 ANALYSIS COMPLETE")
    print("=" * 70)
    n_bimodal = bistability_df['bimodal'].sum() if len(bistability_df) > 0 else 0
    n_total = len(bistability_df)
    print(f"  Bimodal tests: {n_bimodal}/{n_total} show significant bistability (ΔBIC>10)")

    if len(bistability_df) > 0:
        saddle_count = bistability_df['saddle_original'].notna().sum()
        print(f"  Saddle points identified: {saddle_count}/{n_total}")

    if cond_prob is not None:
        n_sig = (chi2_pval.astype(float) < 0.05).sum().sum()
        n_pairs = len(state_dict) * (len(state_dict) - 1)
        print(f"  Significant cascade pairs (p<0.05): {n_sig}/{n_pairs}")

    if cusp_result is not None:
        print(f"  Cusp pseudo-R²: {cusp_result['pseudo_r2']:.4f}, "
              f"eq_RMSE: {cusp_result['eq_rmse']:.3f}")

    print(f"\n  Tables: {TABLES_DIR}")
    print(f"  Figures: {FIGURES_DIR}")


if __name__ == '__main__':
    main()
