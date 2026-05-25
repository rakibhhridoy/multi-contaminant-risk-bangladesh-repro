"""
T2: Stratified Multi-Contaminant Health Risk Assessment
Zone x Depth x Season x Age-group breakdown
+ Single-contaminant vs multi-contaminant comparison
+ Population-weighted exposure estimates
+ Bootstrap 95% CI on stratified medians
+ Probabilistic Monte Carlo HRA
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from config import (
    DATA_FILE, HEALTH_CONTAMINANTS, REFERENCE_DOSES,
    CANCER_SLOPE_FACTORS, CR_VI_FRACTION, EXPOSURE_PARAMS,
    AT_CARCINOGENIC, DEPTH_ZONES, PHYSIOGRAPHIC_ZONES,
    TABLES_DIR, FIGURES_DIR, DPI, RANDOM_STATE,
    assign_zones
)

# ─── DATA LOADING ────────────────────────────────────────────────────────────

def load_and_prepare():
    df = pd.read_csv(DATA_FILE)
    df = assign_zones(df)
    return df


# ─── HEALTH RISK CALCULATIONS ────────────────────────────────────────────────

def calculate_cdi(conc_mg_L, ir, ef, ed, bw):
    """Chronic Daily Intake (mg/kg/day)."""
    return (conc_mg_L * ir * ef * ed) / (bw * ed * 365)


def calculate_all_risks(df, age_group='adult_male'):
    """Calculate HI, CR, and individual HQ/CR per contaminant."""
    params = EXPOSURE_PARAMS[age_group]
    ir, ef, ed, bw = params['ir_L_day'], params['ef_days'], params['ed_years'], params['bw_kg']

    # Non-carcinogenic: Hazard Quotient per contaminant
    hi_total = np.zeros(len(df))
    hi_as_only = np.zeros(len(df))

    for contaminant, rfd in REFERENCE_DOSES.items():
        if contaminant not in df.columns:
            continue
        conc = df[contaminant].values.copy()
        if contaminant == 'As':
            conc = conc / 1000.0  # ug/L -> mg/L
        cdi = calculate_cdi(conc, ir, ef, ed, bw)
        hq = cdi / rfd
        df[f'HQ_{contaminant}'] = hq
        hi_total += hq
        if contaminant == 'As':
            hi_as_only = hq.copy()

    df[f'HI_multi'] = hi_total
    df[f'HI_as_only'] = hi_as_only
    df[f'HI_ratio'] = hi_total / np.where(hi_as_only > 0, hi_as_only, np.nan)

    # Carcinogenic risk
    cr_total = np.zeros(len(df))
    cr_as_only = np.zeros(len(df))

    for contaminant, csf in CANCER_SLOPE_FACTORS.items():
        if contaminant not in df.columns:
            continue
        conc = df[contaminant].values.copy()
        if contaminant == 'As':
            conc = conc / 1000.0
        cdi_cancer = (conc * ir * ef * ed) / (bw * AT_CARCINOGENIC)
        cr = cdi_cancer * csf
        df[f'CR_{contaminant}'] = cr
        cr_total += cr
        if contaminant == 'As':
            cr_as_only = cr.copy()

    # Add Cr(VI) cancer risk
    if 'Cr3+' in df.columns:
        conc_cr6 = df['Cr3+'].values * CR_VI_FRACTION
        cdi_cr6 = (conc_cr6 * ir * ef * ed) / (bw * AT_CARCINOGENIC)
        cr_cr6 = cdi_cr6 * 0.5  # Cr(VI) oral CSF
        df['CR_CrVI'] = cr_cr6
        cr_total += cr_cr6

    df['CR_multi'] = cr_total
    df['CR_as_only'] = cr_as_only

    return df


# ─── BOOTSTRAP HELPER ────────────────────────────────────────────────────────

def bootstrap_median_ci(values, n_boot=1000, ci=95, rng=None):
    """Return (lo, hi) bootstrap percentile CI for the median."""
    if rng is None:
        rng = np.random.default_rng(RANDOM_STATE)
    vals = np.asarray(values)
    if len(vals) < 3:
        return (np.nan, np.nan)
    medians = np.empty(n_boot)
    for b in range(n_boot):
        sample = rng.choice(vals, size=len(vals), replace=True)
        medians[b] = np.median(sample)
    alpha = (100 - ci) / 2
    return (np.percentile(medians, alpha), np.percentile(medians, 100 - alpha))


# ─── STRATIFIED SUMMARY ─────────────────────────────────────────────────────

def stratified_summary(df):
    """Create zone x depth x season summary table with bootstrap 95% CI."""
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for pz in PHYSIOGRAPHIC_ZONES.keys():
        for dz in DEPTH_ZONES.keys():
            for season in ['Dry', 'Wet']:
                mask = (df['phys_zone'] == pz) & (df['depth_zone'] == dz) & (df['Season'] == season)
                cell = df[mask]
                if len(cell) < 3:
                    continue

                # Bootstrap 95% CI on HI_multi median and CR_multi median
                hi_lo, hi_hi = bootstrap_median_ci(cell['HI_multi'].values, n_boot=1000, rng=rng)
                cr_lo, cr_hi = bootstrap_median_ci(cell['CR_multi'].values, n_boot=1000, rng=rng)

                rows.append({
                    'phys_zone': pz, 'depth_zone': dz, 'season': season,
                    'n_samples': len(cell),
                    'As_median_ugL': cell['As'].median(),
                    'Mn_median_mgL': cell['Mn2+'].median(),
                    'Fe_median_mgL': cell['Fe2+'].median(),
                    'HI_multi_median': cell['HI_multi'].median(),
                    'HI_multi_CI_lo': hi_lo,
                    'HI_multi_CI_hi': hi_hi,
                    'HI_multi_mean': cell['HI_multi'].mean(),
                    'HI_as_only_median': cell['HI_as_only'].median(),
                    'HI_exceed_pct': (cell['HI_multi'] > 1).mean() * 100,
                    'HI_ratio_median': cell['HI_ratio'].median(),
                    'CR_multi_median': cell['CR_multi'].median(),
                    'CR_multi_CI_lo': cr_lo,
                    'CR_multi_CI_hi': cr_hi,
                    'CR_as_only_median': cell['CR_as_only'].median(),
                    'CR_exceed_pct': (cell['CR_multi'] > 1e-4).mean() * 100,
                    'CR_ratio_median': (cell['CR_multi'] / cell['CR_as_only'].replace(0, np.nan)).median(),
                })
    return pd.DataFrame(rows)


def contaminant_contribution(df):
    """Quantify each contaminant's contribution to total HI."""
    hq_cols = [c for c in df.columns if c.startswith('HQ_')]
    contributions = {}
    for col in hq_cols:
        contaminant = col.replace('HQ_', '')
        contributions[contaminant] = (df[col] / df['HI_multi'].replace(0, np.nan)).median() * 100
    return pd.Series(contributions).sort_values(ascending=False)


# ─── PROBABILISTIC MONTE CARLO HRA ──────────────────────────────────────────

def probabilistic_hra(df, n_iter=2000):
    """National-level probabilistic HRA via Monte Carlo.

    Samples exposure parameters (ir, bw) from normal distributions and
    contaminant concentrations from the empirical distribution (with replacement).
    Returns DataFrames of HI and CR distributions with summary percentiles.
    """
    rng = np.random.default_rng(RANDOM_STATE)

    # Exposure parameter distributions (adult male reference)
    ir_mean, ir_std = 3.5, 0.7     # L/day
    bw_mean, bw_std = 60.0, 6.0    # kg
    ef = EXPOSURE_PARAMS['adult_male']['ef_days']
    ed = EXPOSURE_PARAMS['adult_male']['ed_years']

    # Contaminant concentration arrays (empirical)
    conc_arrays = {}
    for contaminant in REFERENCE_DOSES.keys():
        if contaminant in df.columns:
            vals = df[contaminant].values.copy()
            if contaminant == 'As':
                vals = vals / 1000.0  # ug/L -> mg/L
            conc_arrays[contaminant] = vals

    n_samples = len(df)
    hi_iter = np.empty(n_iter)
    cr_iter = np.empty(n_iter)

    for i in range(n_iter):
        # Sample exposure parameters (truncate to positive)
        ir_i = max(0.5, rng.normal(ir_mean, ir_std))
        bw_i = max(20.0, rng.normal(bw_mean, bw_std))

        # Sample concentrations with replacement from empirical distribution
        idx = rng.choice(n_samples, size=n_samples, replace=True)

        hi_total = np.zeros(n_samples)
        cr_total = np.zeros(n_samples)

        for contaminant, rfd in REFERENCE_DOSES.items():
            if contaminant not in conc_arrays:
                continue
            conc_sample = conc_arrays[contaminant][idx]
            cdi = calculate_cdi(conc_sample, ir_i, ef, ed, bw_i)
            hi_total += cdi / rfd

        for contaminant, csf in CANCER_SLOPE_FACTORS.items():
            if contaminant not in conc_arrays:
                continue
            conc_sample = conc_arrays[contaminant][idx]
            cdi_cancer = (conc_sample * ir_i * ef * ed) / (bw_i * AT_CARCINOGENIC)
            cr_total += cdi_cancer * csf

        # Cr(VI) cancer contribution
        if 'Cr3+' in conc_arrays:
            conc_cr6 = conc_arrays['Cr3+'][idx] * CR_VI_FRACTION
            cdi_cr6 = (conc_cr6 * ir_i * ef * ed) / (bw_i * AT_CARCINOGENIC)
            cr_total += cdi_cr6 * 0.5

        # Store national median for this iteration
        hi_iter[i] = np.median(hi_total)
        cr_iter[i] = np.median(cr_total)

    return hi_iter, cr_iter


# ─── FIGURES ─────────────────────────────────────────────────────────────────

def plot_hi_heatmap(summary_df, output_dir):
    """Heatmap: HI by zone x depth for each season."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for i, season in enumerate(['Dry', 'Wet']):
        s = summary_df[summary_df['season'] == season]
        pivot = s.pivot_table(index='phys_zone', columns='depth_zone',
                              values='HI_multi_median', aggfunc='first')
        # Reorder
        zone_order = list(PHYSIOGRAPHIC_ZONES.keys())
        depth_order = list(DEPTH_ZONES.keys())
        pivot = pivot.reindex(index=zone_order, columns=depth_order)

        sns.heatmap(pivot, ax=axes[i], annot=True, fmt='.1f', cmap='RdYlGn_r',
                    vmin=0, vmax=10, cbar_kws={'label': 'Hazard Index (median)'},
                    linewidths=0.5)
        axes[i].set_title(f'{season} Season — Cumulative HI', fontsize=14)
        axes[i].set_xlabel('Depth Zone')
        axes[i].set_ylabel('Physiographic Zone')

    plt.tight_layout()
    plt.savefig(output_dir / 'T2_F01_HI_heatmap_season.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved HI heatmap")


def plot_single_vs_multi(df, output_dir):
    """Scatter: As-only HI vs multi-contaminant HI."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # HI comparison
    ax = axes[0]
    ax.scatter(df['HI_as_only'], df['HI_multi'], alpha=0.3, s=10, c='steelblue')
    lim = max(df['HI_multi'].quantile(0.99), df['HI_as_only'].quantile(0.99))
    ax.plot([0, lim], [0, lim], 'k--', lw=1, label='1:1 line')
    ax.set_xlabel('As-only Hazard Index')
    ax.set_ylabel('Multi-contaminant Hazard Index')
    ax.set_title('Non-Carcinogenic Risk')
    ax.axhline(1, color='red', ls=':', lw=1, label='HI = 1 threshold')
    ax.axvline(1, color='red', ls=':', lw=1)
    ax.legend()

    # CR comparison
    ax = axes[1]
    ax.scatter(df['CR_as_only'], df['CR_multi'], alpha=0.3, s=10, c='darkred')
    lim = max(df['CR_multi'].quantile(0.99), df['CR_as_only'].quantile(0.99))
    ax.plot([0, lim], [0, lim], 'k--', lw=1, label='1:1 line')
    ax.set_xlabel('As-only Cancer Risk')
    ax.set_ylabel('Multi-contaminant Cancer Risk')
    ax.set_title('Carcinogenic Risk')
    ax.axhline(1e-4, color='red', ls=':', lw=1, label='CR = 10^-4 threshold')
    ax.axvline(1e-4, color='red', ls=':', lw=1)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'T2_F02_single_vs_multi.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved single vs multi comparison")


def plot_contaminant_contributions(contributions, output_dir):
    """Bar chart: each contaminant's % contribution to total HI."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#d73027', '#fc8d59', '#fee08b', '#91bfdb', '#4575b4', '#313695']
    contributions.plot(kind='bar', ax=ax, color=colors[:len(contributions)])
    ax.set_ylabel('Median contribution to total HI (%)')
    ax.set_title('Contaminant Contributions to Cumulative Hazard Index')
    ax.set_xlabel('')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'T2_F03_contaminant_contributions.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved contaminant contributions")


def plot_age_group_comparison(df, output_dir):
    """Box plot: HI by age group."""
    fig, ax = plt.subplots(figsize=(10, 6))
    age_data = []
    for ag in EXPOSURE_PARAMS.keys():
        # Recalculate for each age group
        temp = df.copy()
        temp = calculate_all_risks(temp, ag)
        age_data.append(pd.DataFrame({
            'Age Group': ag.replace('_', ' ').title(),
            'HI': temp['HI_multi']
        }))
    age_df = pd.concat(age_data)
    sns.boxplot(data=age_df, x='Age Group', y='HI', ax=ax, showfliers=False)
    ax.axhline(1, color='red', ls='--', lw=1.5, label='HI = 1 threshold')
    ax.set_ylabel('Cumulative Hazard Index')
    ax.set_title('Health Risk by Age Group')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'T2_F04_age_group_comparison.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved age group comparison")


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("T2: STRATIFIED MULTI-CONTAMINANT HEALTH RISK")
    print("=" * 65)

    df = load_and_prepare()
    print(f"Loaded {len(df)} samples ({df['phys_zone'].nunique()} zones, "
          f"{df['depth_zone'].nunique()} depth zones)")

    # Calculate risks (adult male as primary)
    df = calculate_all_risks(df, 'adult_male')

    # ─── Summary statistics ──────────────────────────────────────────────
    print(f"\n--- OVERALL RESULTS (adult male) ---")
    print(f"Multi-contaminant HI > 1: {(df['HI_multi'] > 1).sum()} ({(df['HI_multi'] > 1).mean()*100:.1f}%)")
    print(f"As-only HI > 1:          {(df['HI_as_only'] > 1).sum()} ({(df['HI_as_only'] > 1).mean()*100:.1f}%)")
    print(f"HI underestimation ratio: {df['HI_ratio'].median():.2f}x (median)")
    print(f"\nMulti-contaminant CR > 1e-4: {(df['CR_multi'] > 1e-4).sum()} ({(df['CR_multi'] > 1e-4).mean()*100:.1f}%)")
    print(f"As-only CR > 1e-4:          {(df['CR_as_only'] > 1e-4).sum()} ({(df['CR_as_only'] > 1e-4).mean()*100:.1f}%)")

    # ─── Contaminant contributions ───────────────────────────────────────
    contributions = contaminant_contribution(df)
    print(f"\n--- CONTAMINANT CONTRIBUTIONS TO HI ---")
    for cont, pct in contributions.items():
        print(f"  {cont:6s}: {pct:.1f}%")

    # ─── Stratified summary (with bootstrap CI) ─────────────────────────
    summary = stratified_summary(df)
    summary.to_csv(TABLES_DIR / 'T2_stratified_summary.csv', index=False)
    print(f"\nStratified summary: {len(summary)} zone-depth-season cells")
    print(f"  (includes bootstrap 95% CI on HI_multi_median and CR_multi_median)")

    # ─── Season comparison ───────────────────────────────────────────────
    dry = df[df['Season'] == 'Dry']
    wet = df[df['Season'] == 'Wet']
    print(f"\n--- SEASONAL COMPARISON ---")
    print(f"Dry season HI median: {dry['HI_multi'].median():.2f}")
    print(f"Wet season HI median: {wet['HI_multi'].median():.2f}")
    print(f"Dry HI > 1: {(dry['HI_multi'] > 1).mean()*100:.1f}%")
    print(f"Wet HI > 1: {(wet['HI_multi'] > 1).mean()*100:.1f}%")

    # ─── Probabilistic Monte Carlo HRA ───────────────────────────────────
    print(f"\n--- PROBABILISTIC HRA (2000 Monte Carlo iterations) ---")
    print(f"  Sampling: ir ~ N(3.5, 0.7), bw ~ N(60, 6), conc ~ empirical")
    hi_mc, cr_mc = probabilistic_hra(df, n_iter=2000)

    pcts = [2.5, 25, 50, 75, 97.5]
    hi_pcts = np.percentile(hi_mc, pcts)
    cr_pcts = np.percentile(cr_mc, pcts)

    print(f"\n  HI (national median) distribution:")
    for p, v in zip(pcts, hi_pcts):
        print(f"    P{p:5.1f}: {v:.4f}")
    print(f"    Mean:  {np.mean(hi_mc):.4f}")

    print(f"\n  CR (national median) distribution:")
    for p, v in zip(pcts, cr_pcts):
        print(f"    P{p:5.1f}: {v:.2e}")
    print(f"    Mean:  {np.mean(cr_mc):.2e}")

    # Save Monte Carlo results
    mc_df = pd.DataFrame({'HI_median_mc': hi_mc, 'CR_median_mc': cr_mc})
    mc_df.to_csv(TABLES_DIR / 'T2_probabilistic_hra_mc.csv', index=False)
    print(f"  Saved Monte Carlo distributions to T2_probabilistic_hra_mc.csv")

    # ─── Figures ─────────────────────────────────────────────────────────
    fig_dir = FIGURES_DIR
    print(f"\n--- GENERATING FIGURES ---")
    plot_hi_heatmap(summary, fig_dir)
    plot_single_vs_multi(df, fig_dir)
    plot_contaminant_contributions(contributions, fig_dir)
    plot_age_group_comparison(df, fig_dir)

    # ─── Save full results ───────────────────────────────────────────────
    df.to_csv(TABLES_DIR / 'T2_health_risk_full.csv', index=False)
    contributions.to_csv(TABLES_DIR / 'T2_contaminant_contributions.csv')
    print(f"\nAll results saved to {TABLES_DIR}")

    print("\n" + "=" * 65)
    print("T2 COMPLETE")
    print("=" * 65)


if __name__ == '__main__':
    main()
