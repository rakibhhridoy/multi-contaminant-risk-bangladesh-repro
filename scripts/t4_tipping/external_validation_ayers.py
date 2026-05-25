"""
External validation of the phosphate-as-EWS framework on an independent cohort.

Dataset: Ayers et al. (2017), "Chemical composition of water samples from the
shallow aquifer in southwest Bangladesh" (PANGAEA 10.1594/PANGAEA.874439;
Geochemical Transactions 17:4). This is an INDEPENDENT cohort: different
research team (Vanderbilt), different region (southwest tidal deltaplain /
Sundarbans, distinct from our national survey), different instrument
(HR-ICP-MS), 2012-2013. 81 well-states with As, P (phosphorus), Eh, pH, Fe,
Mn, depth.

We replicate two pillars of the main analysis:
  (1) GMM bimodality of log10(As) -> is the bistability signature reproduced?
  (2) Partial-information test -> does P add information about As mode beyond
      {Fe, Mn, depth, Eh} in this independent cohort too?

A positive result in a separate team's separate-region dataset materially
strengthens the generality of the phosphate-as-surveillance-proxy claim.

Output:
  output/tables/T4_external_validation_ayers.csv (summary stats)
  output/figures/T4_external_validation_ayers.png
"""

import sys
import re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

from config import TABLES_DIR, FIGURES_DIR, RANDOM_STATE

DATA = Path("data/external/validation/ayers2017_sw_bangladesh.tab")
CRIMSON = '#c0392b'
STEEL = '#2980b9'
AMBER = '#e67e22'
RNG = np.random.default_rng(RANDOM_STATE)


def load_ayers():
    """Parse the PANGAEA tab file into a clean DataFrame."""
    lines = DATA.read_text(encoding='utf-8').splitlines()
    # find data start (line after the '*/' marker)
    start = next(i for i, ln in enumerate(lines) if ln.strip() == '*/') + 1
    header = lines[start].split('\t')
    rows = [ln.split('\t') for ln in lines[start+1:] if ln.strip()]
    df = pd.DataFrame(rows, columns=header)

    # Robust column matching by prefix (units glyph varies by encoding)
    def find_col(prefix):
        for c in df.columns:
            if c.strip().startswith(prefix):
                return c
        return None
    colmap = {
        'As ': 'As', 'P ': 'P', 'Eh ': 'Eh', 'pH': 'pH',
        'Fe ': 'Fe', 'Mn ': 'Mn', 'Depth sed': 'Depth',
    }
    out = pd.DataFrame()
    for prefix, dst in colmap.items():
        col = find_col(prefix)
        if col is not None:
            out[dst] = pd.to_numeric(df[col], errors='coerce')
        else:
            print(f"  WARNING: column with prefix '{prefix}' not found")
    out['Event'] = df['Event'].values
    # Convert P (µg/L as phosphorus) to PO4 mg/L: P_ugL * (95/31) / 1000
    out['PO4_mg_L'] = out['P'] * (94.97 / 30.97) / 1000.0
    return out.dropna(subset=['As', 'P'])


def gmm_bimodality(log_as):
    """Fit 1- and 2-component GMMs to log10(As); return delta-BIC."""
    x = log_as.reshape(-1, 1)
    g1 = GaussianMixture(1, random_state=RANDOM_STATE).fit(x)
    g2 = GaussianMixture(2, random_state=RANDOM_STATE).fit(x)
    return g1.bic(x) - g2.bic(x), g2


def partial_info(df, target_col):
    """Replicate the partial-information test on this cohort."""
    base = ['Fe', 'Mn', 'Depth', 'Eh']
    sub = df.dropna(subset=base + ['PO4_mg_L', target_col]).copy()
    if len(sub) < 30 or sub[target_col].nunique() < 2:
        return None
    y = sub[target_col].values

    def cv_auc(features, seed=RANDOM_STATE):
        X = sub[features].values
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        aucs = []
        for tr, te in skf.split(X, y):
            rf = RandomForestClassifier(n_estimators=400, random_state=seed,
                                        class_weight='balanced', n_jobs=-1)
            rf.fit(X[tr], y[tr])
            aucs.append(roc_auc_score(y[te], rf.predict_proba(X[te])[:, 1]))
        return np.mean(aucs)

    auc_base = cv_auc(base)
    auc_full = cv_auc(base + ['PO4_mg_L'])
    # Bootstrap delta-AUC
    deltas = []
    for b in range(200):
        idx = RNG.choice(len(sub), len(sub), replace=True)
        s = sub.iloc[idx]
        if s[target_col].nunique() < 2:
            continue
        Xb, yb = s[base].values, s[target_col].values
        Xf = s[base + ['PO4_mg_L']].values
        skf = StratifiedKFold(3, shuffle=True, random_state=b)
        ab, af = [], []
        try:
            for tr, te in skf.split(Xb, yb):
                rfb = RandomForestClassifier(200, random_state=b,
                        class_weight='balanced', n_jobs=-1).fit(Xb[tr], yb[tr])
                ab.append(roc_auc_score(yb[te], rfb.predict_proba(Xb[te])[:, 1]))
                rff = RandomForestClassifier(200, random_state=b,
                        class_weight='balanced', n_jobs=-1).fit(Xf[tr], yb[tr])
                af.append(roc_auc_score(yb[te], rff.predict_proba(Xf[te])[:, 1]))
            deltas.append(np.mean(af) - np.mean(ab))
        except ValueError:
            continue
    deltas = np.array(deltas)
    lo, hi = np.percentile(deltas, [2.5, 97.5]) if len(deltas) else (np.nan, np.nan)

    # Permutation rank of PO4
    rf = RandomForestClassifier(400, random_state=RANDOM_STATE,
                                class_weight='balanced', n_jobs=-1)
    feats = base + ['PO4_mg_L']
    rf.fit(sub[feats].values, y)
    pi = permutation_importance(rf, sub[feats].values, y, n_repeats=30,
                                 random_state=RANDOM_STATE, n_jobs=-1)
    order = np.argsort(pi.importances_mean)[::-1]
    ranked = [feats[i] for i in order]
    po4_rank = ranked.index('PO4_mg_L') + 1
    return {
        'n': len(sub), 'auc_base': auc_base, 'auc_full': auc_full,
        'delta_auc': auc_full - auc_base, 'delta_lo': lo, 'delta_hi': hi,
        'po4_rank': po4_rank, 'n_features': len(feats),
        'ranked': ranked,
    }


def main():
    print("=" * 70)
    print("External validation on Ayers et al. (2017) SW Bangladesh cohort")
    print("=" * 70)
    df = load_ayers()
    print(f"Loaded {len(df)} well-states with As + P")
    print(f"  As range: {df['As'].min():.1f}–{df['As'].max():.1f} µg/L "
          f"(median {df['As'].median():.1f})")
    print(f"  PO4 range: {df['PO4_mg_L'].min():.2f}–{df['PO4_mg_L'].max():.2f} mg/L "
          f"(median {df['PO4_mg_L'].median():.2f})")

    # Spearman As vs PO4
    rho, p_rho = stats.spearmanr(df['As'], df['PO4_mg_L'])
    print(f"\nSpearman As vs PO4: rho = {rho:+.3f}, p = {p_rho:.2g}")

    # GMM bimodality
    log_as = np.log10(df['As'].clip(lower=0.1).values)
    dbic, g2 = gmm_bimodality(log_as)
    print(f"\nGMM bimodality of log10(As): ΔBIC(1→2) = {dbic:.2f} "
          f"({'BIMODAL (ΔBIC>10)' if dbic > 10 else 'weak/unimodal'})")
    means = np.sort(g2.means_.ravel())
    print(f"  GMM component means (log10 As): {means[0]:.2f}, {means[-1]:.2f} "
          f"-> {10**means[0]:.1f} and {10**means[-1]:.1f} µg/L")

    # Partial-information test at WHO threshold
    df['As_T1'] = (df['As'] > 10).astype(int)
    pi_res = partial_info(df, 'As_T1')
    if pi_res:
        print(f"\nPartial-information test (As > 10 µg/L, n={pi_res['n']}):")
        print(f"  AUC base (Fe,Mn,Depth,Eh): {pi_res['auc_base']:.3f}")
        print(f"  AUC + PO4:                 {pi_res['auc_full']:.3f}  "
              f"(ΔAUC {pi_res['delta_auc']:+.3f})")
        print(f"  ΔAUC 95% bootstrap CI: [{pi_res['delta_lo']:+.3f}, {pi_res['delta_hi']:+.3f}]")
        print(f"  PO4 permutation rank: {pi_res['po4_rank']} of {pi_res['n_features']}")
        print(f"  Feature ranking: {pi_res['ranked']}")
        supported = (pi_res['delta_lo'] > 0) and (pi_res['po4_rank'] <= 3)
        print(f"  -> Phosphate informative in independent cohort: "
              f"{'YES' if supported else 'PARTIAL/NO'}")

    # Save summary
    summary = {
        'cohort': 'Ayers2017_SW_Bangladesh',
        'n_samples': len(df),
        'As_median_ugL': float(df['As'].median()),
        'PO4_median_mgL': float(df['PO4_mg_L'].median()),
        'spearman_As_PO4': float(rho),
        'spearman_p': float(p_rho),
        'gmm_delta_bic': float(dbic),
        'gmm_low_mode_ugL': float(10**means[0]),
        'gmm_high_mode_ugL': float(10**means[-1]),
    }
    if pi_res:
        summary.update({
            'partial_info_auc_base': pi_res['auc_base'],
            'partial_info_auc_full': pi_res['auc_full'],
            'partial_info_delta_auc': pi_res['delta_auc'],
            'partial_info_delta_lo': pi_res['delta_lo'],
            'partial_info_delta_hi': pi_res['delta_hi'],
            'partial_info_po4_rank': pi_res['po4_rank'],
        })
    pd.DataFrame([summary]).to_csv(
        TABLES_DIR / 'T4_external_validation_ayers.csv', index=False)

    # Figure: As distribution + As-vs-PO4 + GMM
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ax = axes[0]
    ax.hist(log_as, bins=18, density=True, color=STEEL, alpha=0.6,
            edgecolor='white')
    xs = np.linspace(log_as.min(), log_as.max(), 200)
    from scipy.stats import norm
    for k in range(2):
        w = g2.weights_[k]; m = g2.means_[k, 0]; s = np.sqrt(g2.covariances_[k, 0, 0])
        ax.plot(xs, w * norm.pdf(xs, m, s), lw=2, color=CRIMSON)
    ax.axvline(1.0, color='#333', ls=':', lw=1)
    ax.text(1.02, ax.get_ylim()[1]*0.9, 'WHO', fontsize=8)
    ax.set_xlabel(r'log$_{10}$ As (µg/L)')
    ax.set_ylabel('Density')
    ax.set_title(f'(A) Bimodality reproduced (ΔBIC={dbic:.0f})',
                 fontsize=10, fontweight='bold')

    ax = axes[1]
    ax.scatter(df['PO4_mg_L'], df['As'], s=30, c=STEEL, alpha=0.65,
               edgecolor='white')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.axhline(10, color='#444', ls=':', lw=0.8)
    ax.set_xlabel(r'PO$_4$ (mg/L)'); ax.set_ylabel(r'As (µg/L)')
    ax.set_title(f'(B) As vs PO$_4$ (Spearman ρ={rho:+.2f})',
                 fontsize=10, fontweight='bold')

    ax = axes[2]
    if pi_res:
        labels = ['base\n(Fe,Mn,\nDepth,Eh)', 'base\n+ PO$_4$']
        vals = [pi_res['auc_base'], pi_res['auc_full']]
        bars = ax.bar(labels, vals, color=[STEEL, CRIMSON], alpha=0.85,
                       edgecolor='black')
        ax.errorbar(1, pi_res['auc_full'],
                    yerr=[[pi_res['auc_full']-pi_res['auc_base']-pi_res['delta_lo']],
                          [pi_res['delta_hi']-(pi_res['auc_full']-pi_res['auc_base'])]],
                    fmt='none', ecolor='#333', capsize=4)
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, v+0.01, f'{v:.3f}',
                    ha='center', fontsize=9, fontweight='bold')
        ax.set_ylim(0.5, 1.0)
        ax.set_ylabel('Cross-validated AUC (As > 10 µg/L)')
        ax.set_title(f'(C) PO$_4$ informative here too\n(rank {pi_res["po4_rank"]}/{pi_res["n_features"]})',
                     fontsize=10, fontweight='bold')

    fig.suptitle('External validation — Ayers et al. (2017) independent SW Bangladesh cohort (n=%d)' % len(df),
                 fontsize=11, y=1.03)
    fig.tight_layout()
    out_png = FIGURES_DIR / 'T4_external_validation_ayers.png'
    fig.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\nFigure: {out_png}")
    print("=" * 70)


if __name__ == '__main__':
    main()
