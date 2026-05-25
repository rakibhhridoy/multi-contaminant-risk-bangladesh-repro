"""
T2: DALY Estimation and Population-Weighted Health Burden
Integrates cancer risk with disability weights, life expectancy,
and WorldPop population grid for zone-level burden estimates.

Dose-response models:
  - Arsenicosis skin lesions: logistic (Smith et al., 2000)
  - Non-cancer chronic (HI): exponential saturation
Bootstrap 95% CI on DALY estimates.
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
    DATA_FILE, DEPTH_ZONES, PHYSIOGRAPHIC_ZONES,
    HEALTH_CONTAMINANTS, REFERENCE_DOSES, CANCER_SLOPE_FACTORS,
    EXPOSURE_PARAMS, AT_CARCINOGENIC, CR_VI_FRACTION,
    TABLES_DIR, FIGURES_DIR, DPI, RANDOM_STATE,
    assign_zones
)

# ─── DALY PARAMETERS ─────────────────────────────────────────────────────────
# WHO Global Burden of Disease 2019 parameters for Bangladesh
LIFE_EXPECTANCY_BIRTH = 72.6  # Bangladesh 2022 (World Bank)
DISCOUNT_RATE = 0.03  # 3% standard

# Disability weights (WHO GBD 2019)
DISABILITY_WEIGHTS = {
    'skin_lesions': 0.011,       # Arsenicosis skin manifestations
    'peripheral_neuropathy': 0.072,  # Arsenic neuropathy
    'lung_cancer': 0.288,        # Arsenic-induced
    'bladder_cancer': 0.274,     # Arsenic-induced
    'skin_cancer': 0.032,        # Arsenic-induced
    'kidney_disease': 0.104,     # Arsenic chronic kidney
    'cardiovascular': 0.080,     # Arsenic cardiovascular
    'neurodevelopmental': 0.361, # Manganese neurotoxicity in children
}

# Cancer incidence rates per unit cancer risk (literature-based)
# Fraction of theoretical CR that manifests as clinical cancer
CANCER_MANIFESTATION_RATE = 0.5  # Conservative: 50% of theoretical CR

# Average duration of disability before death (years)
AVG_DISABILITY_DURATION = {
    'cancer': 2.0,
    'chronic': 20.0,
    'neurodevelopmental': 15.0,
}

# Bangladesh population by physiographic zone (approximate, from WorldPop 2020)
# These are rough estimates for the zones our samples cover
ZONE_POPULATION = {
    'Barind_Tract':           8_500_000,
    'Northern_Terrace':       12_000_000,
    'Brahmaputra_Floodplain': 25_000_000,
    'Ganges_Floodplain':      18_000_000,
    'GBM_Delta':              22_000_000,
    'Meghna_Floodplain':      35_000_000,
    'Eastern_Hills':          15_000_000,
}

# Fraction using groundwater as primary drinking source (Bangladesh average ~97%)
GW_USAGE_FRACTION = 0.97

# Age distribution (Bangladesh DHS 2022 approximate)
AGE_DISTRIBUTION = {
    'child_0_5': 0.10,
    'child_6_17': 0.22,
    'adult_female': 0.34,
    'adult_male': 0.34,
}

# ─── DOSE-RESPONSE PARAMETERS ───────────────────────────────────────────────
# Smith et al. (2000) logistic dose-response for arsenicosis skin lesions
# P(skin_lesion) = 1 / (1 + exp(-(a + b * As_ugL)))
SKIN_LESION_LOGISTIC_A = -3.5   # intercept (approximate from Bangladesh studies)
SKIN_LESION_LOGISTIC_B = 0.01   # slope per ug/L As


# ─── DATA LOADING ────────────────────────────────────────────────────────────

def load_health_risk_data():
    """Load T2 health risk results.

    Tries the pre-computed CSV first; if unavailable, loads raw data,
    assigns zones via config.assign_zones, and returns it (without HI/CR
    columns -- caller should check).
    """
    csv_path = TABLES_DIR / 'T2_health_risk_full.csv'
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        # Ensure zone columns are present (they should be in the CSV)
        if 'phys_zone' not in df.columns or 'depth_zone' not in df.columns:
            df = assign_zones(df)
        return df
    else:
        # Fallback: load raw data and assign zones
        print("  T2_health_risk_full.csv not found; loading raw data + assign_zones")
        df = pd.read_csv(DATA_FILE)
        df = assign_zones(df)
        return df


# ─── DOSE-RESPONSE FUNCTIONS ────────────────────────────────────────────────

def p_skin_lesion(as_ugL):
    """Probability of arsenicosis skin lesions (Smith et al. 2000 logistic).

    P = 1 / (1 + exp(-(a + b * As_ugL)))
    """
    linear = SKIN_LESION_LOGISTIC_A + SKIN_LESION_LOGISTIC_B * as_ugL
    return 1.0 / (1.0 + np.exp(-linear))


def p_chronic_disease_hi(hi):
    """Probability of non-cancer chronic disease given HI.

    Exponential dose-response: P = 1 - exp(-0.1 * max(0, HI - 1))
    Gives ~0 at HI=1, saturates toward 1 for large HI.
    """
    excess = np.maximum(0.0, hi - 1.0)
    return 1.0 - np.exp(-0.1 * excess)


# ─── DALY CALCULATION ────────────────────────────────────────────────────────

def calculate_yll(cr, pop, age_group):
    """Years of Life Lost from cancer mortality."""
    # Expected cancer cases = CR x population x manifestation rate
    cancer_cases = cr * pop * CANCER_MANIFESTATION_RATE
    # Remaining life expectancy at cancer onset (assume onset at age 55 for adults, 40 for children)
    onset_ages = {'adult_male': 55, 'adult_female': 55, 'child_6_17': 40, 'child_0_5': 40}
    remaining_le = max(0, LIFE_EXPECTANCY_BIRTH - onset_ages.get(age_group, 55))
    # Discounted YLL
    if DISCOUNT_RATE > 0:
        yll = cancer_cases * (1 - np.exp(-DISCOUNT_RATE * remaining_le)) / DISCOUNT_RATE
    else:
        yll = cancer_cases * remaining_le
    return yll, cancer_cases


def calculate_yld(hi, as_ugL, pop, age_group):
    """Years Lived with Disability from non-cancer chronic effects.

    Uses literature-based dose-response:
      - Skin lesions: logistic (Smith et al. 2000) on As concentration
      - Other chronic: exponential saturation on HI
    """
    # Skin lesion probability from arsenic dose-response
    p_skin = p_skin_lesion(as_ugL)

    # Non-cancer chronic probability from HI (exponential form)
    p_chronic = p_chronic_disease_hi(hi)

    # Combined probability of any chronic effect (union, assuming independence)
    p_any = 1.0 - (1.0 - p_skin) * (1.0 - p_chronic)
    affected = p_any * pop

    # Weight by disability type
    if age_group.startswith('child'):
        # Children: neurodevelopmental (Mn) + skin lesions (As)
        dw = (DISABILITY_WEIGHTS['neurodevelopmental'] * 0.4 +
              DISABILITY_WEIGHTS['skin_lesions'] * 0.6)
        duration = AVG_DISABILITY_DURATION['neurodevelopmental']
    else:
        # Adults: skin + neuropathy + cardiovascular
        dw = (DISABILITY_WEIGHTS['skin_lesions'] * 0.4 +
              DISABILITY_WEIGHTS['peripheral_neuropathy'] * 0.3 +
              DISABILITY_WEIGHTS['cardiovascular'] * 0.3)
        duration = AVG_DISABILITY_DURATION['chronic']

    # Discounted YLD
    if DISCOUNT_RATE > 0:
        yld = affected * dw * (1 - np.exp(-DISCOUNT_RATE * duration)) / DISCOUNT_RATE
    else:
        yld = affected * dw * duration

    return yld, affected


def estimate_dalys_by_zone(df):
    """Estimate DALYs per physiographic zone."""
    results = []

    for pz in PHYSIOGRAPHIC_ZONES.keys():
        zone_pop = ZONE_POPULATION.get(pz, 0)
        gw_pop = zone_pop * GW_USAGE_FRACTION
        zone_data = df[df['phys_zone'] == pz]

        if len(zone_data) < 5:
            continue

        for season in ['Dry', 'Wet']:
            s_data = zone_data[zone_data['Season'] == season]
            if len(s_data) < 3:
                continue

            total_yll = 0
            total_yld = 0
            total_daly_multi = 0
            total_daly_as_only = 0
            total_cancer_cases = 0

            for age_group, age_frac in AGE_DISTRIBUTION.items():
                age_pop = gw_pop * age_frac

                # Multi-contaminant
                median_cr = s_data['CR_multi'].median()
                median_hi = s_data['HI_multi'].median()
                median_as = s_data['As'].median()  # ug/L for dose-response
                yll, cancer_cases = calculate_yll(median_cr, age_pop, age_group)
                yld, affected = calculate_yld(median_hi, median_as, age_pop, age_group)
                total_yll += yll
                total_yld += yld
                total_daly_multi += yll + yld
                total_cancer_cases += cancer_cases

                # As-only
                median_cr_as = s_data['CR_as_only'].median()
                median_hi_as = s_data['HI_as_only'].median()
                yll_as, _ = calculate_yll(median_cr_as, age_pop, age_group)
                yld_as, _ = calculate_yld(median_hi_as, median_as, age_pop, age_group)
                total_daly_as_only += yll_as + yld_as

            results.append({
                'phys_zone': pz,
                'season': season,
                'population': zone_pop,
                'gw_dependent_pop': gw_pop,
                'n_samples': len(s_data),
                'median_HI_multi': s_data['HI_multi'].median(),
                'median_CR_multi': s_data['CR_multi'].median(),
                'median_As_ugL': s_data['As'].median(),
                'DALY_multi': total_daly_multi,
                'DALY_as_only': total_daly_as_only,
                'DALY_ratio': total_daly_multi / total_daly_as_only if total_daly_as_only > 0 else np.nan,
                'YLL': total_yll,
                'YLD': total_yld,
                'est_cancer_cases': total_cancer_cases,
                'DALY_per_100k': total_daly_multi / (zone_pop / 100_000),
            })

    return pd.DataFrame(results)


# ─── BOOTSTRAP CI ON DALYs ──────────────────────────────────────────────────

def bootstrap_daly_ci(df, n_boot=500):
    """Bootstrap 95% CI on national DALY estimates.

    For each bootstrap iteration, resample median HI/CR/As within each
    zone-season cell and propagate through the DALY calculation.
    Returns DataFrame with point estimates and [2.5%, 97.5%] CI.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    boot_daly_multi = np.empty(n_boot)
    boot_daly_as_only = np.empty(n_boot)
    boot_cancer = np.empty(n_boot)

    for b in range(n_boot):
        total_multi = 0.0
        total_as = 0.0
        total_cc = 0.0

        for pz in PHYSIOGRAPHIC_ZONES.keys():
            zone_pop = ZONE_POPULATION.get(pz, 0)
            gw_pop = zone_pop * GW_USAGE_FRACTION
            zone_data = df[df['phys_zone'] == pz]
            if len(zone_data) < 5:
                continue

            for season in ['Dry', 'Wet']:
                s_data = zone_data[zone_data['Season'] == season]
                if len(s_data) < 3:
                    continue

                # Bootstrap resample within this cell
                idx = rng.choice(len(s_data), size=len(s_data), replace=True)
                s_boot = s_data.iloc[idx]

                boot_cr = s_boot['CR_multi'].median()
                boot_hi = s_boot['HI_multi'].median()
                boot_as_conc = s_boot['As'].median()
                boot_cr_as = s_boot['CR_as_only'].median()
                boot_hi_as = s_boot['HI_as_only'].median()

                for age_group, age_frac in AGE_DISTRIBUTION.items():
                    age_pop = gw_pop * age_frac
                    yll, cc = calculate_yll(boot_cr, age_pop, age_group)
                    yld, _ = calculate_yld(boot_hi, boot_as_conc, age_pop, age_group)
                    total_multi += yll + yld
                    total_cc += cc

                    yll_as, _ = calculate_yll(boot_cr_as, age_pop, age_group)
                    yld_as, _ = calculate_yld(boot_hi_as, boot_as_conc, age_pop, age_group)
                    total_as += yll_as + yld_as

        boot_daly_multi[b] = total_multi
        boot_daly_as_only[b] = total_as
        boot_cancer[b] = total_cc

    ci = {
        'DALY_multi_median': np.median(boot_daly_multi),
        'DALY_multi_CI_lo': np.percentile(boot_daly_multi, 2.5),
        'DALY_multi_CI_hi': np.percentile(boot_daly_multi, 97.5),
        'DALY_as_only_median': np.median(boot_daly_as_only),
        'DALY_as_only_CI_lo': np.percentile(boot_daly_as_only, 2.5),
        'DALY_as_only_CI_hi': np.percentile(boot_daly_as_only, 97.5),
        'cancer_cases_median': np.median(boot_cancer),
        'cancer_cases_CI_lo': np.percentile(boot_cancer, 2.5),
        'cancer_cases_CI_hi': np.percentile(boot_cancer, 97.5),
    }
    return ci, boot_daly_multi, boot_daly_as_only


# ─── FIGURES ─────────────────────────────────────────────────────────────────

def plot_daly_by_zone(daly_df, output_dir):
    """Bar chart: DALYs by zone and season."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Total DALYs
    pivot = daly_df.pivot_table(index='phys_zone', columns='season',
                                 values='DALY_multi', aggfunc='first')
    zone_order = [z for z in PHYSIOGRAPHIC_ZONES.keys() if z in pivot.index]
    pivot = pivot.reindex(zone_order)
    pivot.plot(kind='barh', ax=axes[0], color=['#d73027', '#4575b4'])
    axes[0].set_xlabel('Total DALYs (discounted)')
    axes[0].set_title('Multi-Contaminant DALYs by Zone')
    axes[0].legend(title='Season')

    # DALY per 100k
    pivot2 = daly_df.pivot_table(index='phys_zone', columns='season',
                                  values='DALY_per_100k', aggfunc='first')
    pivot2 = pivot2.reindex(zone_order)
    pivot2.plot(kind='barh', ax=axes[1], color=['#d73027', '#4575b4'])
    axes[1].set_xlabel('DALYs per 100,000 population')
    axes[1].set_title('DALY Rate by Zone')
    axes[1].legend(title='Season')

    plt.tight_layout()
    plt.savefig(output_dir / 'T2_F05_daly_by_zone.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved DALY by zone figure")


def plot_daly_multi_vs_single(daly_df, output_dir):
    """Grouped bar: multi vs As-only DALYs."""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Annual average (mean of dry + wet)
    annual = daly_df.groupby('phys_zone')[['DALY_multi', 'DALY_as_only']].mean()
    zone_order = [z for z in PHYSIOGRAPHIC_ZONES.keys() if z in annual.index]
    annual = annual.reindex(zone_order)

    x = np.arange(len(annual))
    width = 0.35
    ax.barh(x - width/2, annual['DALY_as_only'], width, label='As-only', color='#fc8d59')
    ax.barh(x + width/2, annual['DALY_multi'], width, label='Multi-contaminant', color='#d73027')
    ax.set_yticks(x)
    ax.set_yticklabels(annual.index)
    ax.set_xlabel('Annual DALYs (average of dry + wet)')
    ax.set_title('Health Burden: Single vs Multi-Contaminant Assessment')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'T2_F06_daly_multi_vs_single.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved multi vs single DALY figure")


def plot_daly_summary_table(daly_df, output_dir):
    """Heatmap: DALY per 100k by zone x season."""
    fig, ax = plt.subplots(figsize=(10, 7))
    pivot = daly_df.pivot_table(index='phys_zone', columns='season',
                                 values='DALY_per_100k', aggfunc='first')
    zone_order = [z for z in PHYSIOGRAPHIC_ZONES.keys() if z in pivot.index]
    pivot = pivot.reindex(zone_order)

    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax,
                linewidths=0.5, cbar_kws={'label': 'DALYs per 100,000'})
    ax.set_title('Multi-Contaminant DALY Rate by Zone and Season')
    plt.tight_layout()
    plt.savefig(output_dir / 'T2_F07_daly_heatmap.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved DALY heatmap")


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("T2: DALY ESTIMATION & POPULATION-WEIGHTED BURDEN")
    print("=" * 65)

    df = load_health_risk_data()
    print(f"Loaded {len(df)} samples with health risk data")

    # Estimate DALYs
    daly_df = estimate_dalys_by_zone(df)
    daly_df.to_csv(TABLES_DIR / 'T2_daly_by_zone.csv', index=False)

    # ─── Summary ─────────────────────────────────────────────────────────
    total_daly_multi = daly_df['DALY_multi'].sum()
    total_daly_as = daly_df['DALY_as_only'].sum()
    total_cancer = daly_df['est_cancer_cases'].sum()

    print(f"\n--- NATIONAL DALY ESTIMATES ---")
    print(f"Total multi-contaminant DALYs: {total_daly_multi:,.0f}")
    print(f"Total As-only DALYs:           {total_daly_as:,.0f}")
    print(f"Underestimation factor:        {total_daly_multi/total_daly_as:.2f}x")
    print(f"Estimated cancer cases:        {total_cancer:,.0f}")

    # ─── Bootstrap CI on national DALYs ──────────────────────────────────
    print(f"\n--- BOOTSTRAP 95% CI (500 resamples) ---")
    ci, boot_multi, boot_as = bootstrap_daly_ci(df, n_boot=500)
    print(f"  DALY multi:   {ci['DALY_multi_median']:>12,.0f}  "
          f"[{ci['DALY_multi_CI_lo']:,.0f} - {ci['DALY_multi_CI_hi']:,.0f}]")
    print(f"  DALY As-only: {ci['DALY_as_only_median']:>12,.0f}  "
          f"[{ci['DALY_as_only_CI_lo']:,.0f} - {ci['DALY_as_only_CI_hi']:,.0f}]")
    print(f"  Cancer cases: {ci['cancer_cases_median']:>12,.0f}  "
          f"[{ci['cancer_cases_CI_lo']:,.0f} - {ci['cancer_cases_CI_hi']:,.0f}]")

    # Save bootstrap CI
    ci_df = pd.DataFrame([ci])
    ci_df.to_csv(TABLES_DIR / 'T2_daly_bootstrap_ci.csv', index=False)

    # Seasonal
    dry_daly = daly_df[daly_df['season'] == 'Dry']['DALY_multi'].sum()
    wet_daly = daly_df[daly_df['season'] == 'Wet']['DALY_multi'].sum()
    print(f"\nDry season DALYs:  {dry_daly:,.0f}")
    print(f"Wet season DALYs:  {wet_daly:,.0f}")
    print(f"Seasonal increase: {(wet_daly-dry_daly)/dry_daly*100:.1f}%")

    # By zone
    print(f"\n--- DALYs BY ZONE (annual average) ---")
    annual = daly_df.groupby('phys_zone')['DALY_multi'].mean().sort_values(ascending=False)
    for zone, daly in annual.items():
        pop = ZONE_POPULATION.get(zone, 0)
        rate = daly / (pop / 100_000) if pop > 0 else 0
        print(f"  {zone:30s}: {daly:>10,.0f} DALYs  ({rate:.0f} per 100k)")

    # ─── Dose-response info ──────────────────────────────────────────────
    print(f"\n--- DOSE-RESPONSE MODELS ---")
    print(f"  Skin lesions: logistic P = 1/(1+exp(-({SKIN_LESION_LOGISTIC_A} + "
          f"{SKIN_LESION_LOGISTIC_B}*As_ugL)))  [Smith et al. 2000]")
    print(f"  Chronic (HI): P = 1 - exp(-0.1 * max(0, HI-1))  [exponential saturation]")
    # Example values
    for as_val in [5, 10, 50, 100, 200]:
        print(f"    As={as_val:>4d} ug/L -> P(skin_lesion) = {p_skin_lesion(as_val):.4f}")

    # ─── Figures ─────────────────────────────────────────────────────────
    print(f"\n--- GENERATING FIGURES ---")
    plot_daly_by_zone(daly_df, FIGURES_DIR)
    plot_daly_multi_vs_single(daly_df, FIGURES_DIR)
    plot_daly_summary_table(daly_df, FIGURES_DIR)

    print(f"\nAll saved to {TABLES_DIR}")
    print("=" * 65)
    print("T2 DALY ESTIMATION COMPLETE")
    print("=" * 65)


if __name__ == '__main__':
    main()
