"""
T1: Climate-Groundwater Contaminant Mobilization
Builds transfer functions: seasonal contrast -> calibration -> CMIP6 projection.
Uses wet-dry seasonal difference as proxy for recharge-driven geochemical change.
Projects future contaminant concentrations under SSP2-4.5 and SSP5-8.5.

Integration outputs consumed by T2/T5/T6:
  - T1_seasonal_transfer.csv  (with bootstrap CIs)
  - T1_projected_concentrations_2050.csv  (zone x depth x contaminant baselines + projected)
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from config import (
    DATA_FILE, DEPTH_ZONES, PHYSIOGRAPHIC_ZONES,
    SSP_SCENARIOS, PROJECTION_YEARS, HEALTH_CONTAMINANTS,
    TABLES_DIR, FIGURES_DIR, RANDOM_STATE, EXTERNAL_DIR,
    assign_zones
)

np.random.seed(RANDOM_STATE)

# ─── CLIMATE PARAMETERS ───────────────────────────────────────────────────────
# Bangladesh monsoon: wet season ~30% more precipitation than dry
MONSOON_PRECIP_RATIO = 1.30
N_BOOTSTRAP_CI = 500   # bootstrap resamples for sensitivity CI

# Contaminants to project
PROJECTION_CONTAMINANTS = ['As', 'Mn2+', 'Fe2+', 'Cr3+', 'NO3-']

# WHO guidelines for threshold analysis
WHO_GUIDELINES = {
    'As':   10.0,    # µg/L
    'Mn2+': 0.4,     # mg/L
    'Fe2+': 0.3,     # mg/L
    'Cr3+': 0.05,    # mg/L
    'NO3-': 50.0,    # mg/L
}

# CMIP6 precipitation change (% relative to 1970-2000 baseline)
# From WorldClim downscaled EC-Earth3-Veg for Bangladesh region
# These are approximate values; refined below if tif files available
PRECIP_CHANGE_SSP = {
    'SSP2-4.5': {2030: 5.0, 2040: 8.0, 2050: 12.0, 2060: 15.0},
    'SSP5-8.5': {2030: 7.0, 2040: 12.0, 2050: 18.0, 2060: 25.0},
}

# Temperature change (°C) — affects reaction kinetics
TEMP_CHANGE_SSP = {
    'SSP2-4.5': {2030: 0.6, 2040: 0.9, 2050: 1.3, 2060: 1.6},
    'SSP5-8.5': {2030: 0.8, 2040: 1.4, 2050: 2.1, 2060: 2.9},
}

# Temperature sensitivity of As mobilization (literature: ~10% increase per °C)
TEMP_SENSITIVITY_AS = 0.10
TEMP_SENSITIVITY_MN = 0.08
TEMP_SENSITIVITY_FE = 0.06


# ─── DATA LOADING ──────────────────────────────────────────────────────────────

def load_data():
    """Load dataset and assign zones using shared config function."""
    df = pd.read_csv(DATA_FILE)
    df = assign_zones(df)
    return df


def extract_cmip6_precip():
    """Extract CMIP6 precipitation from WorldClim tif files for Bangladesh."""
    try:
        import rasterio
        from rasterio.windows import from_bounds

        cmip6_dir = EXTERNAL_DIR / 'cmip6'
        results = {}

        for ssp_label, ssp_file_tag in [('SSP2-4.5', 'ssp245'), ('SSP5-8.5', 'ssp585')]:
            fpath = cmip6_dir / f'wc2.1_2.5m_prec_EC-Earth3-Veg_{ssp_file_tag}_2041-2060.tif'
            if not fpath.exists():
                continue

            with rasterio.open(fpath) as src:
                # Bangladesh bounding box
                window = from_bounds(87.8, 20.5, 92.8, 26.8, src.transform)
                data = src.read(1, window=window)
                # Mean annual precipitation (mm) for Bangladesh
                valid = data[data != src.nodata] if src.nodata else data.flatten()
                results[ssp_label] = {
                    'mean_precip_mm': float(np.nanmean(valid)),
                    'std_precip_mm': float(np.nanstd(valid)),
                    'min_precip_mm': float(np.nanmin(valid)),
                    'max_precip_mm': float(np.nanmax(valid)),
                }
                print(f"  {ssp_label}: mean precip = {results[ssp_label]['mean_precip_mm']:.0f} mm")

        return results
    except Exception as e:
        print(f"  CMIP6 extraction skipped: {e}")
        return {}


# ─── BOOTSTRAP CI HELPER ──────────────────────────────────────────────────────

def _bootstrap_median_diff(dry_vals, wet_vals, n_boot=N_BOOTSTRAP_CI):
    """Bootstrap 95% CI on the wet-dry median difference and derived quantities.

    Returns dict with keys:
        delta_CI_lo, delta_CI_hi,
        sensitivity_CI_lo, sensitivity_CI_hi,
        delta_pct_CI_lo, delta_pct_CI_hi
    """
    rng = np.random.RandomState(RANDOM_STATE)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        d_boot = np.median(rng.choice(dry_vals, size=len(dry_vals), replace=True))
        w_boot = np.median(rng.choice(wet_vals, size=len(wet_vals), replace=True))
        deltas[b] = w_boot - d_boot

    lo, hi = np.percentile(deltas, [2.5, 97.5])

    # Sensitivity = delta / 30% precip change
    sens_lo, sens_hi = lo / 30.0, hi / 30.0

    # Percentage change relative to dry median
    dry_med = np.median(dry_vals)
    if dry_med != 0:
        pct_lo = lo / dry_med * 100
        pct_hi = hi / dry_med * 100
    else:
        pct_lo = pct_hi = np.nan

    return {
        'sensitivity_CI_lo': sens_lo,
        'sensitivity_CI_hi': sens_hi,
        'delta_pct_CI_lo': pct_lo,
        'delta_pct_CI_hi': pct_hi,
    }


# ─── SEASONAL TRANSFER FUNCTION ─────────────────────────────────────────────

def build_seasonal_transfer(df):
    """
    Use wet-dry seasonal difference as calibration proxy.
    For each zone x depth x contaminant: delta_contaminant = wet_median - dry_median
    This represents the geochemical response to ~30% precipitation increase.

    Includes bootstrap 95% CIs on sensitivity and delta_pct.
    """
    results = []

    for dz in DEPTH_ZONES.keys():
        for pz in PHYSIOGRAPHIC_ZONES.keys():
            mask = (df['depth_zone'] == dz) & (df['phys_zone'] == pz)
            cell = df[mask]
            if len(cell) < 10:
                continue

            dry = cell[cell['Season'] == 'Dry']
            wet = cell[cell['Season'] == 'Wet']

            for cont in PROJECTION_CONTAMINANTS + ['ORP']:
                if cont not in cell.columns or len(dry) < 5 or len(wet) < 5:
                    continue

                dry_vals = dry[cont].dropna().values
                wet_vals = wet[cont].dropna().values
                if len(dry_vals) < 5 or len(wet_vals) < 5:
                    continue

                dry_med = np.median(dry_vals)
                wet_med = np.median(wet_vals)
                delta = wet_med - dry_med
                delta_pct = (delta / dry_med * 100) if dry_med != 0 else np.nan

                # Mann-Whitney test
                try:
                    stat, pval = stats.mannwhitneyu(
                        dry_vals, wet_vals, alternative='two-sided')
                except Exception:
                    stat, pval = np.nan, np.nan

                # Effect size (rank-biserial correlation)
                n1, n2 = len(dry_vals), len(wet_vals)
                r_effect = 1 - 2 * stat / (n1 * n2) if (n1 > 0 and n2 > 0 and not np.isnan(stat)) else np.nan

                # Sensitivity: delta_contaminant per 1% precipitation change
                sensitivity = delta / 30.0

                # Bootstrap CI
                ci = _bootstrap_median_diff(dry_vals, wet_vals)

                results.append({
                    'depth_zone': dz, 'phys_zone': pz, 'contaminant': cont,
                    'dry_median': dry_med, 'wet_median': wet_med,
                    'delta_median': delta, 'delta_pct': delta_pct,
                    'sensitivity_per_pct': sensitivity,
                    'sensitivity_CI_lo': ci['sensitivity_CI_lo'],
                    'sensitivity_CI_hi': ci['sensitivity_CI_hi'],
                    'delta_pct_CI_lo': ci['delta_pct_CI_lo'],
                    'delta_pct_CI_hi': ci['delta_pct_CI_hi'],
                    'mann_whitney_p': pval, 'effect_size_r': r_effect,
                    'n_dry': len(dry), 'n_wet': len(wet),
                    'significant': pval < 0.05 if not np.isnan(pval) else False,
                })

    return pd.DataFrame(results)


# ─── FUTURE PROJECTIONS ─────────────────────────────────────────────────────

def project_future_concentrations(transfer_df):
    """
    Project future contaminant concentrations under SSP scenarios.
    Two mechanisms: (1) precipitation-driven recharge change, (2) temperature-driven kinetics.
    """
    temp_sensitivity = {
        'As': TEMP_SENSITIVITY_AS, 'Mn2+': TEMP_SENSITIVITY_MN,
        'Fe2+': TEMP_SENSITIVITY_FE, 'Cr3+': 0.03, 'NO3-': 0.05,
    }

    all_projections = []

    for ssp in SSP_SCENARIOS:
        for year in PROJECTION_YEARS:
            precip_pct = PRECIP_CHANGE_SSP[ssp][year]
            temp_c = TEMP_CHANGE_SSP[ssp][year]

            for _, row in transfer_df.iterrows():
                cont = row['contaminant']
                if cont == 'ORP':
                    continue

                baseline = row['dry_median']  # use dry season as baseline
                sensitivity = row['sensitivity_per_pct']

                # Precipitation effect
                precip_delta = sensitivity * precip_pct

                # Temperature effect
                t_sens = temp_sensitivity.get(cont, 0.05)
                temp_delta = baseline * t_sens * temp_c

                # Combined projection
                projected = baseline + precip_delta + temp_delta
                projected = max(0, projected)  # non-negative

                # WHO exceedance
                who = WHO_GUIDELINES.get(cont, np.nan)
                currently_exceeds = baseline > who if not np.isnan(who) else np.nan
                projected_exceeds = projected > who if not np.isnan(who) else np.nan

                all_projections.append({
                    'ssp': ssp, 'year': year,
                    'depth_zone': row['depth_zone'],
                    'phys_zone': row['phys_zone'],
                    'contaminant': cont,
                    'baseline': baseline,
                    'precip_delta': precip_delta,
                    'temp_delta': temp_delta,
                    'projected': projected,
                    'total_change_pct': ((projected - baseline) / baseline * 100)
                                        if baseline > 0 else np.nan,
                    'baseline_exceeds_who': currently_exceeds,
                    'projected_exceeds_who': projected_exceeds,
                    'who_guideline': who,
                })

    return pd.DataFrame(all_projections)


def build_projected_concentrations_2050(transfer_df):
    """Build the T1->T2/T6 integration table: projected median concentrations at 2050.

    For each zone x depth x contaminant, computes:
        baseline_median  (dry-season median from data)
        projected_median_ssp245  (baseline + precip_delta + temp_delta under SSP2-4.5 2050)
        projected_median_ssp585  (same under SSP5-8.5 2050)

    Saves T1_projected_concentrations_2050.csv.
    """
    temp_sensitivity = {
        'As': TEMP_SENSITIVITY_AS, 'Mn2+': TEMP_SENSITIVITY_MN,
        'Fe2+': TEMP_SENSITIVITY_FE, 'Cr3+': 0.03, 'NO3-': 0.05,
    }
    year = 2050
    rows = []

    # Only non-ORP contaminants
    transfer_conts = transfer_df[transfer_df['contaminant'] != 'ORP']

    for _, row in transfer_conts.iterrows():
        cont = row['contaminant']
        baseline = row['dry_median']
        sensitivity = row['sensitivity_per_pct']
        t_sens = temp_sensitivity.get(cont, 0.05)

        proj = {}
        for ssp, col_suffix in [('SSP2-4.5', 'ssp245'), ('SSP5-8.5', 'ssp585')]:
            precip_pct = PRECIP_CHANGE_SSP[ssp][year]
            temp_c = TEMP_CHANGE_SSP[ssp][year]
            precip_delta = sensitivity * precip_pct
            temp_delta = baseline * t_sens * temp_c
            proj[col_suffix] = max(0, baseline + precip_delta + temp_delta)

        rows.append({
            'phys_zone': row['phys_zone'],
            'depth_zone': row['depth_zone'],
            'contaminant': cont,
            'baseline_median': baseline,
            'projected_median_ssp245': proj['ssp245'],
            'projected_median_ssp585': proj['ssp585'],
        })

    out = pd.DataFrame(rows)
    out.to_csv(TABLES_DIR / 'T1_projected_concentrations_2050.csv', index=False)
    print(f"  Saved T1_projected_concentrations_2050.csv ({len(out)} rows)")
    return out


def threshold_crossing_analysis(proj_df):
    """Identify zone-depth cells that cross WHO thresholds under projections."""
    results = []

    for ssp in SSP_SCENARIOS:
        for year in PROJECTION_YEARS:
            subset = proj_df[(proj_df['ssp'] == ssp) & (proj_df['year'] == year)]

            for cont in PROJECTION_CONTAMINANTS:
                c_sub = subset[subset['contaminant'] == cont]
                if len(c_sub) == 0:
                    continue

                n_baseline_exceed = c_sub['baseline_exceeds_who'].sum()
                n_projected_exceed = c_sub['projected_exceeds_who'].sum()
                n_new_exceed = ((~c_sub['baseline_exceeds_who'].astype(bool)) &
                                (c_sub['projected_exceeds_who'].astype(bool))).sum()
                n_total = len(c_sub)

                results.append({
                    'ssp': ssp, 'year': year, 'contaminant': cont,
                    'n_cells': n_total,
                    'baseline_exceed': int(n_baseline_exceed),
                    'projected_exceed': int(n_projected_exceed),
                    'new_exceedances': int(n_new_exceed),
                    'pct_baseline_exceed': n_baseline_exceed / n_total * 100,
                    'pct_projected_exceed': n_projected_exceed / n_total * 100,
                    'pct_new_exceed': n_new_exceed / n_total * 100,
                })

    return pd.DataFrame(results)


# ─── FIGURES ────────────────────────────────────────────────────────────────────

def plot_seasonal_sensitivity(transfer_df, output_dir):
    """Heatmap: seasonal sensitivity by zone x contaminant."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for i, metric in enumerate(['delta_pct', 'significant']):
        for j, cont in enumerate(PROJECTION_CONTAMINANTS):
            pass  # Single heatmap approach below

    # Delta percentage heatmap for As
    as_transfer = transfer_df[transfer_df['contaminant'] == 'As']

    for i, cont in enumerate(['As', 'Mn2+']):
        c_data = transfer_df[transfer_df['contaminant'] == cont]
        pivot = c_data.pivot_table(index='phys_zone', columns='depth_zone',
                                    values='delta_pct', aggfunc='first')
        zone_order = [z for z in PHYSIOGRAPHIC_ZONES.keys() if z in pivot.index]
        depth_order = list(DEPTH_ZONES.keys())
        pivot = pivot.reindex(index=zone_order, columns=depth_order)

        sns.heatmap(pivot, annot=True, fmt='.0f', cmap='RdBu_r', center=0,
                    ax=axes[i], linewidths=0.5,
                    cbar_kws={'label': '% change (wet - dry)'})
        axes[i].set_title(f'{cont} Seasonal Change (%)\n(proxy for recharge sensitivity)')

    plt.tight_layout()
    plt.savefig(output_dir / 'T1_F01_seasonal_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved seasonal sensitivity heatmap")


def plot_projection_timeline(proj_df, output_dir):
    """Line plot: projected contaminant change over time by SSP."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, cont in enumerate(PROJECTION_CONTAMINANTS):
        ax = axes[i]
        c_proj = proj_df[proj_df['contaminant'] == cont]

        for ssp in SSP_SCENARIOS:
            ssp_data = c_proj[c_proj['ssp'] == ssp]
            means = ssp_data.groupby('year')['total_change_pct'].agg(['mean', 'std']).reset_index()
            color = '#d73027' if '8.5' in ssp else '#4575b4'
            ax.plot(means['year'], means['mean'], 'o-', color=color, label=ssp, lw=2)
            ax.fill_between(means['year'],
                             means['mean'] - means['std'],
                             means['mean'] + means['std'],
                             color=color, alpha=0.15)

        ax.axhline(0, color='gray', ls='--', lw=0.5)
        ax.set_xlabel('Year')
        ax.set_ylabel('Projected change (%)')
        ax.set_title(f'{cont} concentration change')
        ax.legend()

    # WHO threshold crossings
    ax = axes[-1]
    crossing = proj_df.groupby(['ssp', 'year']).apply(
        lambda x: ((~x['baseline_exceeds_who'].astype(bool)) &
                    (x['projected_exceeds_who'].astype(bool))).sum()
    ).reset_index(name='new_exceedances')

    for ssp in SSP_SCENARIOS:
        ssp_data = crossing[crossing['ssp'] == ssp]
        color = '#d73027' if '8.5' in ssp else '#4575b4'
        ax.plot(ssp_data['year'], ssp_data['new_exceedances'], 'o-', color=color, label=ssp, lw=2)

    ax.set_xlabel('Year')
    ax.set_ylabel('New WHO threshold crossings')
    ax.set_title('Zone-depth cells newly exceeding WHO')
    ax.legend()

    plt.suptitle('T1: Climate-Driven Contaminant Mobilization Projections', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / 'T1_F02_projection_timeline.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved projection timeline")


def plot_zone_vulnerability(proj_df, output_dir):
    """Heatmap: zone vulnerability (% change at 2050 SSP5-8.5)."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Select 2050 SSP5-8.5 as worst-case reference
    worst = proj_df[(proj_df['ssp'] == 'SSP5-8.5') & (proj_df['year'] == 2050)]

    for i, cont in enumerate(['As', 'Mn2+', 'Fe2+']):
        c_data = worst[worst['contaminant'] == cont]
        pivot = c_data.pivot_table(index='phys_zone', columns='depth_zone',
                                    values='total_change_pct', aggfunc='mean')
        zone_order = [z for z in PHYSIOGRAPHIC_ZONES.keys() if z in pivot.index]
        depth_order = list(DEPTH_ZONES.keys())
        pivot = pivot.reindex(index=zone_order, columns=depth_order)

        sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd',
                    ax=axes[i], linewidths=0.5, vmin=0,
                    cbar_kws={'label': '% increase from baseline'})
        axes[i].set_title(f'{cont} -- 2050 SSP5-8.5\nProjected % Increase')

    plt.suptitle('Zone Vulnerability: Climate-Driven Contaminant Increase', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'T1_F03_zone_vulnerability.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved zone vulnerability heatmap")


def plot_threshold_crossing_summary(crossing_df, output_dir):
    """Grouped bar: new WHO exceedances by contaminant and scenario."""
    fig, ax = plt.subplots(figsize=(14, 7))

    # 2050 comparison
    c2050 = crossing_df[crossing_df['year'] == 2050]

    x = np.arange(len(PROJECTION_CONTAMINANTS))
    width = 0.35

    for i, ssp in enumerate(SSP_SCENARIOS):
        ssp_data = c2050[c2050['ssp'] == ssp]
        vals = []
        for cont in PROJECTION_CONTAMINANTS:
            row = ssp_data[ssp_data['contaminant'] == cont]
            vals.append(row['pct_new_exceed'].values[0] if len(row) > 0 else 0)

        color = '#4575b4' if '4.5' in ssp else '#d73027'
        ax.bar(x + i * width - width/2, vals, width, label=ssp, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(PROJECTION_CONTAMINANTS)
    ax.set_ylabel('% of zone-depth cells newly exceeding WHO guideline')
    ax.set_title('New WHO Threshold Crossings by 2050')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'T1_F04_threshold_crossings_2050.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved threshold crossing summary")


# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("T1: CLIMATE-GROUNDWATER CONTAMINANT MOBILIZATION")
    print("=" * 65)

    df = load_data()
    print(f"Loaded {len(df)} samples ({df['phys_zone'].nunique()} zones, "
          f"{df['depth_zone'].nunique()} depth zones)")

    # ─── CMIP6 data extraction ─────────────────────────────────────────────
    print("\n--- CMIP6 DATA EXTRACTION ---")
    cmip6_info = extract_cmip6_precip()

    # ─── Seasonal transfer function ────────────────────────────────────────
    print("\n--- SEASONAL TRANSFER FUNCTION (with bootstrap CIs) ---")
    transfer_df = build_seasonal_transfer(df)
    transfer_df.to_csv(TABLES_DIR / 'T1_seasonal_transfer.csv', index=False)
    print(f"Built {len(transfer_df)} zone-depth-contaminant transfer cells")

    # Summary: significant seasonal changes
    sig = transfer_df[transfer_df['significant'] == True]
    print(f"  Significant seasonal changes: {len(sig)}/{len(transfer_df)} ({len(sig)/len(transfer_df)*100:.0f}%)")

    print(f"\n  Contaminant sensitivity (median % change wet vs dry):")
    for cont in PROJECTION_CONTAMINANTS + ['ORP']:
        c_data = transfer_df[transfer_df['contaminant'] == cont]
        if len(c_data) > 0:
            med_pct = c_data['delta_pct'].median()
            n_sig = (c_data['significant'] == True).sum()
            direction = "+" if med_pct > 0 else "-"
            med_ci_lo = c_data['delta_pct_CI_lo'].median()
            med_ci_hi = c_data['delta_pct_CI_hi'].median()
            print(f"    {cont:6s}: {direction}{abs(med_pct):>6.1f}%  "
                  f"[{med_ci_lo:+.1f}, {med_ci_hi:+.1f}] 95% CI  "
                  f"({n_sig}/{len(c_data)} cells significant)")

    # ─── Future projections ────────────────────────────────────────────────
    print("\n--- FUTURE PROJECTIONS ---")
    proj_df = project_future_concentrations(transfer_df)
    proj_df.to_csv(TABLES_DIR / 'T1_future_projections.csv', index=False)

    # Summary by SSP and year
    print(f"\n  Projected median % change from baseline:")
    for ssp in SSP_SCENARIOS:
        print(f"\n  {ssp}:")
        for year in PROJECTION_YEARS:
            subset = proj_df[(proj_df['ssp'] == ssp) & (proj_df['year'] == year)]
            for cont in PROJECTION_CONTAMINANTS:
                c_sub = subset[subset['contaminant'] == cont]
                if len(c_sub) > 0:
                    med_change = c_sub['total_change_pct'].median()
                    print(f"    {year} {cont:6s}: {med_change:>+6.1f}%", end="")
            print()

    # ─── Threshold crossing analysis ───────────────────────────────────────
    print("\n--- THRESHOLD CROSSING ANALYSIS ---")
    crossing_df = threshold_crossing_analysis(proj_df)
    crossing_df.to_csv(TABLES_DIR / 'T1_threshold_crossings.csv', index=False)

    # Key findings
    as_2050_585 = crossing_df[(crossing_df['ssp'] == 'SSP5-8.5') &
                               (crossing_df['year'] == 2050) &
                               (crossing_df['contaminant'] == 'As')]
    if len(as_2050_585) > 0:
        row = as_2050_585.iloc[0]
        print(f"\n  As threshold crossings (SSP5-8.5, 2050):")
        print(f"    Baseline WHO exceedance: {row['baseline_exceed']}/{row['n_cells']} cells ({row['pct_baseline_exceed']:.0f}%)")
        print(f"    Projected WHO exceedance: {row['projected_exceed']}/{row['n_cells']} cells ({row['pct_projected_exceed']:.0f}%)")
        print(f"    NEW exceedances: {row['new_exceedances']} cells ({row['pct_new_exceed']:.0f}%)")

    # Multi-contaminant threshold crossings
    print(f"\n  Total new WHO exceedances across all contaminants:")
    for ssp in SSP_SCENARIOS:
        for year in [2030, 2050]:
            sub = crossing_df[(crossing_df['ssp'] == ssp) & (crossing_df['year'] == year)]
            total_new = sub['new_exceedances'].sum()
            total_cells = sub['n_cells'].sum()
            print(f"    {ssp} {year}: {total_new} new exceedances across {total_cells} zone-contaminant cells")

    # ─── Climate x Health compound risk ────────────────────────────────────
    print("\n--- CLIMATE-HEALTH COMPOUND RISK ---")
    # How many cells go from safe -> unsafe for MULTIPLE contaminants?
    for ssp in SSP_SCENARIOS:
        for year in [2050]:
            sub = proj_df[(proj_df['ssp'] == ssp) & (proj_df['year'] == year)]
            # Group by zone-depth cell, count contaminants exceeding
            zone_cells = sub.groupby(['phys_zone', 'depth_zone']).apply(
                lambda g: pd.Series({
                    'n_baseline_exceed': g['baseline_exceeds_who'].sum(),
                    'n_projected_exceed': g['projected_exceeds_who'].sum(),
                })
            ).reset_index()

            multi_risk = (zone_cells['n_projected_exceed'] >= 2).sum()
            multi_risk_baseline = (zone_cells['n_baseline_exceed'] >= 2).sum()
            print(f"  {ssp} {year}: {multi_risk_baseline}->{multi_risk} cells with >=2 contaminants above WHO "
                  f"(+{multi_risk - multi_risk_baseline} new)")

    # ─── Future health-risk integration table (T1->T2/T6) ─────────────────
    print("\n--- PROJECTED CONCENTRATIONS 2050 (integration table for T2/T6) ---")
    proj_2050_df = build_projected_concentrations_2050(transfer_df)

    # Quick summary
    for cont in PROJECTION_CONTAMINANTS:
        c_sub = proj_2050_df[proj_2050_df['contaminant'] == cont]
        if len(c_sub) > 0:
            baseline_med = c_sub['baseline_median'].median()
            ssp245_med = c_sub['projected_median_ssp245'].median()
            ssp585_med = c_sub['projected_median_ssp585'].median()
            print(f"  {cont:6s}: baseline {baseline_med:.3f} -> "
                  f"SSP2-4.5 {ssp245_med:.3f} | SSP5-8.5 {ssp585_med:.3f}")

    # ─── Figures ───────────────────────────────────────────────────────────
    print("\n--- GENERATING FIGURES ---")
    plot_seasonal_sensitivity(transfer_df, FIGURES_DIR)
    plot_projection_timeline(proj_df, FIGURES_DIR)
    plot_zone_vulnerability(proj_df, FIGURES_DIR)
    plot_threshold_crossing_summary(crossing_df, FIGURES_DIR)

    print(f"\nAll saved to {TABLES_DIR} and {FIGURES_DIR}")
    print("=" * 65)
    print("T1 CLIMATE-MOBILIZATION COMPLETE")
    print("=" * 65)


if __name__ == '__main__':
    main()
