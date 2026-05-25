"""
T3: Copula-Based Joint Contaminant Exceedance Modeling
Models joint distribution of As, Mn, Fe, Cr using multiple copula families.
Calculates joint exceedance probabilities and compares with independence assumption.
Key output: underestimation factor when ignoring contaminant dependence.
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from scipy import stats, optimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

from config import (
    DATA_FILE, HEALTH_CONTAMINANTS, DEPTH_ZONES, PHYSIOGRAPHIC_ZONES,
    TABLES_DIR, FIGURES_DIR, RANDOM_STATE, DPI, assign_zones
)

np.random.seed(RANDOM_STATE)

# ─── WHO/EPA THRESHOLDS ────────────────────────────────────────────────────────
# Health-based thresholds for joint exceedance
THRESHOLDS = {
    'As':   10.0,    # µg/L WHO guideline
    'Mn2+': 0.4,     # mg/L WHO guideline
    'Fe2+': 0.3,     # mg/L WHO guideline (aesthetic + health)
    'Cr3+': 0.05,    # mg/L WHO guideline
}

# Core contaminant set for copula analysis
COPULA_CONTAMINANTS = ['As', 'Mn2+', 'Fe2+', 'Cr3+']


# ─── DATA LOADING ──────────────────────────────────────────────────────────────

def load_data():
    """Load dataset and assign zones using nearest-centroid tiebreaking."""
    df = pd.read_csv(DATA_FILE)
    df = assign_zones(df)
    return df


# ─── MARGINAL FITTING ──────────────────────────────────────────────────────────

def fit_marginals(data_dict):
    """Fit best marginal distribution for each contaminant.
    Tests: lognormal, gamma, Weibull. Selects by AIC.
    Returns PIT-transformed uniform data and fitted params.
    """
    uniform_data = {}
    marginal_fits = {}

    candidates = {
        'lognorm': stats.lognorm,
        'gamma':   stats.gamma,
        'weibull': stats.weibull_min,
    }

    for col, values in data_dict.items():
        x = values[values > 0].copy()
        best_aic = np.inf
        best_name = None
        best_params = None

        for name, dist in candidates.items():
            try:
                params = dist.fit(x, floc=0)
                ll = np.sum(dist.logpdf(x, *params))
                k = len(params)
                aic = 2 * k - 2 * ll
                if aic < best_aic:
                    best_aic = aic
                    best_name = name
                    best_params = params
            except Exception:
                continue

        # PIT transform
        dist = candidates[best_name]
        u = dist.cdf(values, *best_params)
        u = np.clip(u, 1e-6, 1 - 1e-6)

        uniform_data[col] = u
        marginal_fits[col] = {
            'distribution': best_name,
            'params': best_params,
            'aic': best_aic,
            'n': len(x),
        }

    return uniform_data, marginal_fits


# ─── COPULA FAMILIES ───────────────────────────────────────────────────────────

def gaussian_copula_nll(rho, u1, u2):
    """Negative log-likelihood for bivariate Gaussian copula."""
    z1 = stats.norm.ppf(u1)
    z2 = stats.norm.ppf(u2)
    n = len(u1)
    rho = np.clip(rho, -0.999, 0.999)
    nll = (n / 2) * np.log(1 - rho**2) + \
          (rho**2 * (z1**2 + z2**2) - 2 * rho * z1 * z2) / (2 * (1 - rho**2))
    return nll


def clayton_copula_nll(theta, u1, u2):
    """Negative log-likelihood for Clayton copula (theta > 0)."""
    if theta <= 0:
        return 1e10
    t1 = u1**(-theta)
    t2 = u2**(-theta)
    log_c = np.log(1 + theta) + (-1 - theta) * (np.log(u1) + np.log(u2)) + \
            (-2 - 1/theta) * np.log(t1 + t2 - 1)
    return -np.sum(log_c)


def frank_copula_nll(theta, u1, u2):
    """Negative log-likelihood for Frank copula."""
    if abs(theta) < 1e-6:
        return 1e10
    et = np.exp(-theta)
    eu1 = np.exp(-theta * u1)
    eu2 = np.exp(-theta * u2)
    num = theta * (1 - et) * np.exp(-theta * (u1 + u2))
    den = ((1 - et) - (1 - eu1) * (1 - eu2))**2
    with np.errstate(divide='ignore', invalid='ignore'):
        log_c = np.log(np.abs(num)) - np.log(np.abs(den))
    valid = np.isfinite(log_c)
    return -np.sum(log_c[valid])


def gumbel_copula_nll(theta, u1, u2):
    """Negative log-likelihood for Gumbel copula (theta >= 1)."""
    if theta < 1:
        return 1e10
    lu1 = -np.log(u1)
    lu2 = -np.log(u2)
    A = (lu1**theta + lu2**theta)**(1/theta)
    C = np.exp(-A)

    # Log density
    log_c = np.log(C) + np.log(A + theta - 1) + (theta - 1) * (np.log(lu1) + np.log(lu2)) - \
            np.log(u1) - np.log(u2) + (1/theta - 2) * np.log(lu1**theta + lu2**theta) - A
    valid = np.isfinite(log_c)
    return -np.sum(log_c[valid])


def t_copula_nll(params, u1, u2):
    """Negative log-likelihood for bivariate t-copula."""
    rho, nu = params
    rho = np.clip(rho, -0.999, 0.999)
    nu = max(nu, 2.1)

    t1 = stats.t.ppf(u1, df=nu)
    t2 = stats.t.ppf(u2, df=nu)

    # Bivariate t density / product of marginal t densities
    n = len(u1)
    from scipy.special import gammaln
    log_c = (gammaln((nu + 2) / 2) - gammaln(nu / 2) - np.log(nu * np.pi) -
             0.5 * np.log(1 - rho**2) +
             gammaln(nu / 2) - gammaln((nu + 1) / 2) + 0.5 * np.log(nu * np.pi))
    # Per-observation
    q = (t1**2 + t2**2 - 2 * rho * t1 * t2) / (nu * (1 - rho**2))
    log_c_obs = (gammaln((nu + 2) / 2) - gammaln(nu / 2) -
                 np.log(np.pi * nu * np.sqrt(1 - rho**2)) -
                 ((nu + 2) / 2) * np.log(1 + q) -
                 (-((nu + 1) / 2) * np.log(1 + t1**2/nu) +
                  -((nu + 1) / 2) * np.log(1 + t2**2/nu)))
    valid = np.isfinite(log_c_obs)
    return -np.sum(log_c_obs[valid])


def fit_bivariate_copula(u1, u2):
    """Fit multiple copula families, select best by AIC."""
    results = {}
    n = len(u1)

    # 1. Gaussian
    try:
        res = optimize.minimize_scalar(gaussian_copula_nll, bounds=(-0.99, 0.99),
                                        method='bounded', args=(u1, u2))
        nll = res.fun
        results['Gaussian'] = {'param': res.x, 'nll': nll, 'aic': 2 * 1 + 2 * nll, 'k': 1}
    except Exception:
        pass

    # 2. Clayton
    try:
        res = optimize.minimize_scalar(clayton_copula_nll, bounds=(0.01, 20),
                                        method='bounded', args=(u1, u2))
        nll = res.fun
        results['Clayton'] = {'param': res.x, 'nll': nll, 'aic': 2 * 1 + 2 * nll, 'k': 1}
    except Exception:
        pass

    # 3. Frank
    try:
        res = optimize.minimize_scalar(frank_copula_nll, bounds=(-30, 30),
                                        method='bounded', args=(u1, u2))
        nll = res.fun
        results['Frank'] = {'param': res.x, 'nll': nll, 'aic': 2 * 1 + 2 * nll, 'k': 1}
    except Exception:
        pass

    # 4. Gumbel
    try:
        res = optimize.minimize_scalar(gumbel_copula_nll, bounds=(1.01, 15),
                                        method='bounded', args=(u1, u2))
        nll = res.fun
        results['Gumbel'] = {'param': res.x, 'nll': nll, 'aic': 2 * 1 + 2 * nll, 'k': 1}
    except Exception:
        pass

    # 5. t-copula
    try:
        res = optimize.minimize(t_copula_nll, x0=[0.3, 5], args=(u1, u2),
                                method='Nelder-Mead',
                                options={'maxiter': 2000})
        nll = res.fun
        results['t-copula'] = {'param': (res.x[0], res.x[1]), 'nll': nll,
                                'aic': 2 * 2 + 2 * nll, 'k': 2}
    except Exception:
        pass

    # Select best by AIC
    if not results:
        return None, {}
    best = min(results, key=lambda k: results[k]['aic'])
    return best, results


# ─── GOODNESS-OF-FIT & TAIL DEPENDENCE ─────────────────────────────────────────

def copula_gof_test(u1, u2, best_family, best_param, n_sim=500):
    """Cramér-von Mises goodness-of-fit test for the fitted copula.
    Compares empirical copula with parametric copula via simulation.
    Returns p-value (Sn statistic, parametric bootstrap).
    """
    n = len(u1)

    # Empirical copula function
    def empirical_copula(u, v, u1_data, u2_data):
        return np.mean((u1_data <= u) & (u2_data <= v))

    # Compute Sn statistic on observed data
    C_emp = np.array([empirical_copula(u1[i], u2[i], u1, u2) for i in range(n)])

    # Parametric copula CDF (Gaussian approximation for all families)
    z1 = stats.norm.ppf(np.clip(u1, 1e-6, 1-1e-6))
    z2 = stats.norm.ppf(np.clip(u2, 1e-6, 1-1e-6))
    if best_family == 'Gaussian':
        rho = best_param
        C_param = np.array([stats.multivariate_normal.cdf(
            [z1[i], z2[i]], mean=[0,0], cov=[[1, rho],[rho, 1]])
            for i in range(n)])
    else:
        # Use empirical copula from simulated data as approximation
        C_param = C_emp  # fallback — GoF only computed for Gaussian

    Sn = np.sum((C_emp - C_param)**2)

    # Parametric bootstrap
    sn_boot = []
    for _ in range(min(n_sim, 200)):
        # Simulate from fitted Gaussian copula
        if best_family == 'Gaussian':
            rho = best_param
            z_sim = np.random.multivariate_normal([0,0], [[1,rho],[rho,1]], n)
            u1_sim = stats.norm.cdf(z_sim[:, 0])
            u2_sim = stats.norm.cdf(z_sim[:, 1])
        else:
            # Bootstrap resample as fallback
            idx = np.random.choice(n, n, replace=True)
            u1_sim, u2_sim = u1[idx], u2[idx]

        C_emp_sim = np.array([empirical_copula(u1_sim[i], u2_sim[i], u1_sim, u2_sim)
                              for i in range(min(n, 200))])  # subsample for speed
        C_param_sim = C_emp_sim  # self-comparison baseline
        if best_family == 'Gaussian':
            z1s = stats.norm.ppf(np.clip(u1_sim[:200], 1e-6, 1-1e-6))
            z2s = stats.norm.ppf(np.clip(u2_sim[:200], 1e-6, 1-1e-6))
            C_param_sim = np.array([stats.multivariate_normal.cdf(
                [z1s[i], z2s[i]], mean=[0,0], cov=[[1,rho],[rho,1]])
                for i in range(len(z1s))])
        sn_boot.append(np.sum((C_emp_sim - C_param_sim)**2))

    p_value = np.mean(np.array(sn_boot) >= Sn) if sn_boot else np.nan
    return Sn, p_value


def compute_tail_dependence(u1, u2, quantiles=[0.90, 0.95, 0.99]):
    """Compute empirical upper and lower tail dependence coefficients.
    lambda_U = P(U2 > q | U1 > q) as q → 1
    lambda_L = P(U2 < 1-q | U1 < 1-q) as q → 0
    """
    results = {}
    for q in quantiles:
        # Upper tail
        mask_u = u1 > q
        if mask_u.sum() > 5:
            lambda_u = np.mean(u2[mask_u] > q)
        else:
            lambda_u = np.nan

        # Lower tail
        threshold_l = 1 - q
        mask_l = u1 < threshold_l
        if mask_l.sum() > 5:
            lambda_l = np.mean(u2[mask_l] < threshold_l)
        else:
            lambda_l = np.nan

        results[f'lambda_U_{q}'] = lambda_u
        results[f'lambda_L_{q}'] = lambda_l

    return results


def bootstrap_kendall_tau(x, y, n_boot=500):
    """Bootstrap 95% CI for Kendall's tau."""
    n = len(x)
    taus = []
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        tau, _ = stats.kendalltau(x[idx], y[idx])
        taus.append(tau)
    taus = np.array(taus)
    return np.percentile(taus, 2.5), np.percentile(taus, 97.5)


# ─── JOINT EXCEEDANCE ──────────────────────────────────────────────────────────

def empirical_joint_exceedance(df, contaminants, thresholds):
    """Calculate empirical single and joint exceedance probabilities."""
    results = {}

    # Individual exceedance
    for col in contaminants:
        thresh = thresholds[col]
        results[f'P({col}>thresh)'] = (df[col] > thresh).mean()

    # Pairwise joint exceedance
    for c1, c2 in combinations(contaminants, 2):
        joint = ((df[c1] > thresholds[c1]) & (df[c2] > thresholds[c2])).mean()
        indep = results[f'P({c1}>thresh)'] * results[f'P({c2}>thresh)']
        results[f'P({c1}&{c2})_empirical'] = joint
        results[f'P({c1}&{c2})_independent'] = indep
        results[f'ratio_{c1}_{c2}'] = joint / indep if indep > 0 else np.nan

    # All-4 joint exceedance
    mask_all = np.ones(len(df), dtype=bool)
    for col in contaminants:
        mask_all &= (df[col] > thresholds[col])
    results['P(all_4)_empirical'] = mask_all.mean()

    indep_all = 1.0
    for col in contaminants:
        indep_all *= results[f'P({col}>thresh)']
    results['P(all_4)_independent'] = indep_all
    results['ratio_all_4'] = results['P(all_4)_empirical'] / indep_all if indep_all > 0 else np.nan

    # At least 2 exceed
    exceed_count = sum((df[col] > thresholds[col]).astype(int) for col in contaminants)
    results['P(>=2_exceed)'] = (exceed_count >= 2).mean()
    results['P(>=3_exceed)'] = (exceed_count >= 3).mean()

    return results


def copula_joint_exceedance(u_data, contaminants, thresholds, marginal_fits, data_dict):
    """Calculate joint exceedance under copula vs independence assumption."""
    # Transform thresholds to uniform scale
    u_thresholds = {}
    from scipy import stats as st
    dists = {'lognorm': st.lognorm, 'gamma': st.gamma, 'weibull': st.weibull_min}

    for col in contaminants:
        fit = marginal_fits[col]
        dist = dists[fit['distribution']]
        u_thresholds[col] = dist.cdf(thresholds[col], *fit['params'])

    # Empirical copula-based joint exceedance
    n = len(list(u_data.values())[0])
    mask_all = np.ones(n, dtype=bool)
    for col in contaminants:
        mask_all &= (u_data[col] > u_thresholds[col])

    copula_joint = mask_all.mean()

    # Independence assumption
    indep_joint = 1.0
    for col in contaminants:
        indep_joint *= (u_data[col] > u_thresholds[col]).mean()

    return copula_joint, indep_joint, u_thresholds


# ─── STRATIFIED ANALYSIS ───────────────────────────────────────────────────────

def stratified_copula_analysis(df, contaminants):
    """Run copula fitting for each season × depth zone."""
    rows = []

    for season in ['Dry', 'Wet']:
        for dz in DEPTH_ZONES.keys():
            subset = df[(df['Season'] == season) & (df['depth_zone'] == dz)]
            if len(subset) < 30:
                continue

            # Prepare data
            data_dict = {}
            for col in contaminants:
                data_dict[col] = subset[col].values

            # Fit marginals
            u_data, marg_fits = fit_marginals(data_dict)

            # Fit pairwise copulas
            for c1, c2 in combinations(contaminants, 2):
                best_family, all_fits = fit_bivariate_copula(u_data[c1], u_data[c2])
                if best_family is None:
                    continue

                # Kendall's tau (empirical)
                tau, p_tau = stats.kendalltau(subset[c1], subset[c2])

                rows.append({
                    'season': season,
                    'depth_zone': dz,
                    'pair': f'{c1}-{c2}',
                    'n_samples': len(subset),
                    'kendall_tau': tau,
                    'p_value': p_tau,
                    'best_copula': best_family,
                    'best_param': str(all_fits[best_family]['param']),
                    'best_aic': all_fits[best_family]['aic'],
                    **{f'AIC_{fam}': all_fits[fam]['aic']
                       for fam in all_fits},
                })

    return pd.DataFrame(rows)


def zone_joint_exceedance(df, contaminants, thresholds):
    """Joint exceedance by physiographic zone × season."""
    rows = []

    for pz in PHYSIOGRAPHIC_ZONES.keys():
        for season in ['Dry', 'Wet']:
            subset = df[(df['phys_zone'] == pz) & (df['Season'] == season)]
            if len(subset) < 10:
                continue

            exceed_count = sum((subset[col] > thresholds[col]).astype(int)
                                for col in contaminants)
            n = len(subset)

            # Individual
            p_single = {col: (subset[col] > thresholds[col]).mean() for col in contaminants}

            # Joint (all 4)
            mask_all = np.ones(n, dtype=bool)
            for col in contaminants:
                mask_all &= (subset[col] > thresholds[col])
            p_all_4 = mask_all.mean()

            # Independence prediction
            p_indep_all = np.prod([p_single[col] for col in contaminants])

            rows.append({
                'phys_zone': pz,
                'season': season,
                'n_samples': n,
                **{f'P_exceed_{col}': p_single[col] for col in contaminants},
                'P_all_4_empirical': p_all_4,
                'P_all_4_independent': p_indep_all,
                'dependence_ratio': p_all_4 / p_indep_all if p_indep_all > 0 else np.nan,
                'P_ge2_exceed': (exceed_count >= 2).mean(),
                'P_ge3_exceed': (exceed_count >= 3).mean(),
                'mean_n_exceed': exceed_count.mean(),
            })

    return pd.DataFrame(rows)


# ─── FIGURES ────────────────────────────────────────────────────────────────────

def plot_pairwise_copula_scatter(df, u_data, contaminants, output_dir):
    """Pairwise scatter plots in uniform space (copula diagnostic)."""
    pairs = list(combinations(contaminants, 2))
    n_pairs = len(pairs)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, (c1, c2) in enumerate(pairs):
        ax = axes[i]
        ax.scatter(u_data[c1], u_data[c2], alpha=0.15, s=5, c='steelblue')
        ax.set_xlabel(f'{c1} (uniform)')
        ax.set_ylabel(f'{c2} (uniform)')
        tau, p = stats.kendalltau(df[c1], df[c2])
        ax.set_title(f'{c1} vs {c2}\n$\\tau$ = {tau:.3f} (p={p:.1e})', fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)

    # Remove empty subplot if any
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Pairwise Copula Diagnostics (PIT-transformed)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'T3_F01_copula_pairwise_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved copula pairwise scatter")


def plot_joint_exceedance_heatmap(zone_exc_df, output_dir):
    """Heatmap: dependence ratio by zone × season."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for i, metric in enumerate(['dependence_ratio', 'mean_n_exceed']):
        pivot = zone_exc_df.pivot_table(index='phys_zone', columns='season',
                                         values=metric, aggfunc='first')
        zone_order = [z for z in PHYSIOGRAPHIC_ZONES.keys() if z in pivot.index]
        pivot = pivot.reindex(zone_order)

        if metric == 'dependence_ratio':
            cmap = 'RdYlBu_r'
            title = 'Dependence Ratio\n(empirical / independent joint exceedance)'
            fmt = '.1f'
            vmin, vmax = 0, max(5, pivot.max().max())
        else:
            cmap = 'YlOrRd'
            title = 'Mean Number of Contaminants\nExceeding WHO Guidelines'
            fmt = '.2f'
            vmin, vmax = 0, 4

        sns.heatmap(pivot, annot=True, fmt=fmt, cmap=cmap, ax=axes[i],
                    linewidths=0.5, vmin=vmin, vmax=vmax,
                    cbar_kws={'label': metric.replace('_', ' ').title()})
        axes[i].set_title(title, fontsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / 'T3_F02_joint_exceedance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved joint exceedance heatmap")


def plot_copula_family_comparison(strat_df, output_dir):
    """Bar chart: best copula family distribution across pairs and strata."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # By contaminant pair
    pair_counts = strat_df.groupby(['pair', 'best_copula']).size().unstack(fill_value=0)
    pair_counts.plot(kind='bar', ax=axes[0], colormap='Set2')
    axes[0].set_title('Best Copula Family by Contaminant Pair')
    axes[0].set_ylabel('Count across strata')
    axes[0].legend(title='Copula', bbox_to_anchor=(1.0, 1.0))
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')

    # By depth zone
    depth_counts = strat_df.groupby(['depth_zone', 'best_copula']).size().unstack(fill_value=0)
    depth_order = list(DEPTH_ZONES.keys())
    depth_counts = depth_counts.reindex([d for d in depth_order if d in depth_counts.index])
    depth_counts.plot(kind='bar', ax=axes[1], colormap='Set2')
    axes[1].set_title('Best Copula Family by Depth Zone')
    axes[1].set_ylabel('Count across pairs')
    axes[1].legend(title='Copula', bbox_to_anchor=(1.0, 1.0))
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_dir / 'T3_F03_copula_family_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved copula family comparison")


def plot_exceedance_count_distribution(df, contaminants, thresholds, output_dir):
    """Stacked bar: distribution of number of simultaneous exceedances."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for i, season in enumerate(['Dry', 'Wet']):
        subset = df[df['Season'] == season]
        exceed_count = sum((subset[col] > thresholds[col]).astype(int)
                            for col in contaminants)
        counts = exceed_count.value_counts().sort_index()
        colors = ['#2166ac', '#67a9cf', '#fddbc7', '#ef8a62', '#b2182b']

        ax = axes[i]
        bars = ax.bar(counts.index, counts.values / len(subset) * 100,
                       color=[colors[min(j, len(colors)-1)] for j in counts.index])
        ax.set_xlabel('Number of Contaminants Exceeding WHO Guideline')
        ax.set_ylabel('Percentage of Samples (%)')
        ax.set_title(f'{season} Season')
        ax.set_xticks(range(5))

        # Add count labels
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'n={val}', ha='center', fontsize=9)

    plt.suptitle('Distribution of Simultaneous Contaminant Exceedances', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'T3_F04_exceedance_count_dist.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved exceedance count distribution")


def plot_kendall_tau_matrix(df, contaminants, output_dir):
    """Kendall tau correlation matrix for copula contaminants, by season."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for i, season in enumerate(['Dry', 'Wet']):
        subset = df[df['Season'] == season]
        n = len(contaminants)
        tau_matrix = np.zeros((n, n))
        p_matrix = np.zeros((n, n))

        for r in range(n):
            for c in range(n):
                if r == c:
                    tau_matrix[r, c] = 1.0
                else:
                    tau, p = stats.kendalltau(subset[contaminants[r]], subset[contaminants[c]])
                    tau_matrix[r, c] = tau
                    p_matrix[r, c] = p

        # Annotation: tau with significance stars
        annot = np.empty((n, n), dtype=object)
        for r in range(n):
            for c in range(n):
                if r == c:
                    annot[r, c] = '1.0'
                else:
                    stars = '***' if p_matrix[r,c] < 0.001 else '**' if p_matrix[r,c] < 0.01 else '*' if p_matrix[r,c] < 0.05 else ''
                    annot[r, c] = f'{tau_matrix[r,c]:.3f}{stars}'

        ax = axes[i]
        sns.heatmap(pd.DataFrame(tau_matrix, index=contaminants, columns=contaminants),
                    annot=annot, fmt='', cmap='RdBu_r', center=0, vmin=-0.5, vmax=0.5,
                    ax=ax, linewidths=0.5, cbar_kws={'label': "Kendall's $\\tau$"})
        ax.set_title(f'{season} Season — Kendall τ Matrix', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / 'T3_F05_kendall_tau_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved Kendall tau matrix")


def plot_tail_dependence(tail_df, output_dir):
    """Bar chart: upper and lower tail dependence by pair."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    pairs = tail_df['pair'].values
    x = np.arange(len(pairs))

    # Upper tail
    ax = axes[0]
    for i, q in enumerate([0.90, 0.95, 0.99]):
        col = f'lambda_U_{q}'
        vals = tail_df[col].fillna(0).values
        ax.bar(x + i*0.25 - 0.25, vals, 0.25, label=f'q={q}',
               color=['#4575b4', '#d73027', '#313695'][i])
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, rotation=45, ha='right')
    ax.set_ylabel('$\\lambda_U$ (upper tail dependence)')
    ax.set_title('Upper Tail Dependence\n(joint extreme co-occurrence)')
    ax.legend()

    # Lower tail
    ax = axes[1]
    for i, q in enumerate([0.90, 0.95, 0.99]):
        col = f'lambda_L_{q}'
        vals = tail_df[col].fillna(0).values
        ax.bar(x + i*0.25 - 0.25, vals, 0.25, label=f'q={q}',
               color=['#4575b4', '#d73027', '#313695'][i])
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, rotation=45, ha='right')
    ax.set_ylabel('$\\lambda_L$ (lower tail dependence)')
    ax.set_title('Lower Tail Dependence\n(joint low co-occurrence)')
    ax.legend()

    plt.suptitle('Tail Dependence Coefficients by Contaminant Pair', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'T3_F06_tail_dependence.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved tail dependence figure")


# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("T3: COPULA-BASED JOINT CONTAMINANT EXCEEDANCE")
    print("=" * 65)

    df = load_data()
    contaminants = COPULA_CONTAMINANTS
    print(f"Loaded {len(df)} samples")
    print(f"Contaminants: {contaminants}")

    # ─── 1. Marginal fitting (full dataset) ────────────────────────────────
    print("\n--- MARGINAL DISTRIBUTION FITTING ---")
    data_dict = {col: df[col].values for col in contaminants}
    u_data, marginal_fits = fit_marginals(data_dict)

    for col, fit in marginal_fits.items():
        print(f"  {col:6s}: best={fit['distribution']:10s}  AIC={fit['aic']:.0f}")

    # ─── 2. Pairwise copula fitting (full dataset) ─────────────────────────
    print("\n--- PAIRWISE COPULA FITTING (full dataset) ---")
    pairwise_results = []
    for c1, c2 in combinations(contaminants, 2):
        best, all_fits = fit_bivariate_copula(u_data[c1], u_data[c2])
        tau, p = stats.kendalltau(df[c1], df[c2])
        print(f"  {c1:6s}-{c2:6s}: best={best:10s}  τ={tau:.3f} (p={p:.1e})")
        pairwise_results.append({
            'pair': f'{c1}-{c2}',
            'best_copula': best,
            'kendall_tau': tau,
            'p_value': p,
            **{f'AIC_{fam}': all_fits[fam]['aic'] for fam in all_fits},
        })

    pairwise_df = pd.DataFrame(pairwise_results)
    pairwise_df.to_csv(TABLES_DIR / 'T3_pairwise_copula_full.csv', index=False)

    # ─── 2b. Bootstrap CIs on Kendall's tau ──────────────────────────────
    print("\n--- BOOTSTRAP 95% CI ON KENDALL'S TAU ---")
    tau_ci_results = []
    for c1, c2 in combinations(contaminants, 2):
        tau, p = stats.kendalltau(df[c1], df[c2])
        ci_lo, ci_hi = bootstrap_kendall_tau(df[c1].values, df[c2].values, n_boot=500)
        print(f"  {c1:6s}-{c2:6s}: τ={tau:.3f}  95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")
        tau_ci_results.append({
            'pair': f'{c1}-{c2}', 'tau': tau, 'p_value': p,
            'tau_CI_lo': ci_lo, 'tau_CI_hi': ci_hi,
        })
    tau_ci_df = pd.DataFrame(tau_ci_results)
    tau_ci_df.to_csv(TABLES_DIR / 'T3_kendall_tau_bootstrap.csv', index=False)

    # ─── 2c. Tail dependence analysis ────────────────────────────────────
    print("\n--- TAIL DEPENDENCE COEFFICIENTS ---")
    tail_results = []
    for c1, c2 in combinations(contaminants, 2):
        td = compute_tail_dependence(u_data[c1], u_data[c2])
        print(f"  {c1:6s}-{c2:6s}: λ_U(0.95)={td['lambda_U_0.95']:.3f}  "
              f"λ_L(0.95)={td['lambda_L_0.95']:.3f}")
        tail_results.append({'pair': f'{c1}-{c2}', **td})
    tail_df = pd.DataFrame(tail_results)
    tail_df.to_csv(TABLES_DIR / 'T3_tail_dependence.csv', index=False)

    # ─── 2d. Goodness-of-fit (Gaussian copula pairs) ────────────────────
    print("\n--- COPULA GOODNESS-OF-FIT (Cramér-von Mises, Gaussian pairs) ---")
    for c1, c2 in combinations(contaminants, 2):
        best = pairwise_df[pairwise_df['pair'] == f'{c1}-{c2}']['best_copula'].values[0]
        if best == 'Gaussian':
            # Get Gaussian param
            res = optimize.minimize_scalar(gaussian_copula_nll, bounds=(-0.99, 0.99),
                                            method='bounded', args=(u_data[c1], u_data[c2]))
            Sn, p_gof = copula_gof_test(u_data[c1], u_data[c2], 'Gaussian', res.x, n_sim=200)
            print(f"  {c1:6s}-{c2:6s}: Sn={Sn:.4f}, p={p_gof:.3f} {'(adequate)' if p_gof > 0.05 else '(POOR FIT)'}")
        else:
            print(f"  {c1:6s}-{c2:6s}: best={best} (GoF only for Gaussian)")

    # ─── 3. Empirical joint exceedance ─────────────────────────────────────
    print("\n--- JOINT EXCEEDANCE PROBABILITIES ---")
    exc_results = empirical_joint_exceedance(df, contaminants, THRESHOLDS)

    print(f"\n  Individual exceedance (WHO guidelines):")
    for col in contaminants:
        print(f"    {col:6s}: {exc_results[f'P({col}>thresh)']:.1%}")

    print(f"\n  Pairwise joint exceedance:")
    for c1, c2 in combinations(contaminants, 2):
        emp = exc_results[f'P({c1}&{c2})_empirical']
        ind = exc_results[f'P({c1}&{c2})_independent']
        ratio = exc_results[f'ratio_{c1}_{c2}']
        print(f"    {c1:6s} & {c2:6s}: empirical={emp:.3f}  independent={ind:.3f}  ratio={ratio:.2f}x")

    print(f"\n  All-4 simultaneous exceedance:")
    print(f"    Empirical:    {exc_results['P(all_4)_empirical']:.4f} ({exc_results['P(all_4)_empirical']*100:.2f}%)")
    print(f"    Independent:  {exc_results['P(all_4)_independent']:.4f} ({exc_results['P(all_4)_independent']*100:.2f}%)")
    print(f"    Dependence ratio: {exc_results['ratio_all_4']:.2f}x")
    print(f"    ≥2 contaminants exceed: {exc_results['P(>=2_exceed)']:.1%}")
    print(f"    ≥3 contaminants exceed: {exc_results['P(>=3_exceed)']:.1%}")

    # Save exceedance results
    exc_df = pd.DataFrame([exc_results])
    exc_df.to_csv(TABLES_DIR / 'T3_joint_exceedance_overall.csv', index=False)

    # ─── 4. Stratified copula analysis ─────────────────────────────────────
    print("\n--- STRATIFIED COPULA ANALYSIS (season × depth) ---")
    strat_df = stratified_copula_analysis(df, contaminants)
    strat_df.to_csv(TABLES_DIR / 'T3_stratified_copula.csv', index=False)
    print(f"  Fitted {len(strat_df)} pairwise copulas across strata")

    # Summary: best copula family distribution
    family_counts = strat_df['best_copula'].value_counts()
    print(f"\n  Best copula family distribution:")
    for fam, cnt in family_counts.items():
        print(f"    {fam:12s}: {cnt} ({cnt/len(strat_df)*100:.0f}%)")

    # ─── 5. Zone-level joint exceedance ────────────────────────────────────
    print("\n--- ZONE-LEVEL JOINT EXCEEDANCE ---")
    zone_exc = zone_joint_exceedance(df, contaminants, THRESHOLDS)
    zone_exc.to_csv(TABLES_DIR / 'T3_zone_joint_exceedance.csv', index=False)

    # Top zones by dependence ratio
    print(f"\n  Top dependence ratios (empirical/independent joint exceedance):")
    top = zone_exc.nlargest(5, 'dependence_ratio')
    for _, row in top.iterrows():
        print(f"    {row['phys_zone']:30s} {row['season']}: {row['dependence_ratio']:.1f}x  "
              f"(mean {row['mean_n_exceed']:.1f} contaminants exceed)")

    # ─── 6. Seasonal comparison ────────────────────────────────────────────
    print("\n--- SEASONAL JOINT EXCEEDANCE COMPARISON ---")
    for season in ['Dry', 'Wet']:
        s_df = df[df['Season'] == season]
        exceed_count = sum((s_df[col] > THRESHOLDS[col]).astype(int) for col in contaminants)
        print(f"  {season}: mean exceedances={exceed_count.mean():.2f}, "
              f"≥2 exceed={( exceed_count >= 2).mean():.1%}, "
              f"≥3 exceed={(exceed_count >= 3).mean():.1%}")

    # ─── 7. Independence underestimation summary ───────────────────────────
    print("\n--- KEY FINDING: INDEPENDENCE UNDERESTIMATION ---")
    mean_dep_ratio = zone_exc['dependence_ratio'].median()
    print(f"  Median dependence ratio across zones: {mean_dep_ratio:.2f}x")
    print(f"  → Independence assumption underestimates joint risk by {mean_dep_ratio:.1f}x")
    print(f"  → This compounds with T2 single-contaminant underestimation (2.15x)")
    print(f"  → Total underestimation: ~{mean_dep_ratio * 2.15:.1f}x when ignoring both")

    # ─── 8. Figures ────────────────────────────────────────────────────────
    print("\n--- GENERATING FIGURES ---")
    plot_pairwise_copula_scatter(df, u_data, contaminants, FIGURES_DIR)
    plot_joint_exceedance_heatmap(zone_exc, FIGURES_DIR)
    plot_copula_family_comparison(strat_df, FIGURES_DIR)
    plot_exceedance_count_distribution(df, contaminants, THRESHOLDS, FIGURES_DIR)
    plot_kendall_tau_matrix(df, contaminants, FIGURES_DIR)
    plot_tail_dependence(tail_df, FIGURES_DIR)

    print(f"\nAll saved to {TABLES_DIR} and {FIGURES_DIR}")
    print("=" * 65)
    print("T3 COPULA ANALYSIS COMPLETE")
    print("=" * 65)


if __name__ == '__main__':
    main()
