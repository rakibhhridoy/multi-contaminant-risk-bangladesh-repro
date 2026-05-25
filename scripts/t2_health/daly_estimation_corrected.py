"""
T2: DALY Estimation — CORRECTED VERSION (2026-05-21)
=====================================================
This script replaces daly_estimation.py for the Nature Water resubmission.

THREE BUGS FIXED relative to the original daly_estimation.py:

  Bug 1: Seasonal double-counting.
    Original (line 411): total = daly_df['DALY_multi'].sum()
    This summed Dry + Wet rows per zone, counting each person twice.
    Fix: average across seasons within each zone before summing zones.

  Bug 2: Inflated water-intake parameters.
    Original config.EXPOSURE_PARAMS uses ir_L_day=3.5 (adult male) and 3.0
    (adult female). EPA exposure factors handbook uses 2.0 L/day; supplementary
    Table S1 reports 2.5 / 2.0. We use 2.5 L/day for adult male as a
    tropical-climate adjustment, with a sensitivity sweep showing the
    correction is not load-bearing in either direction.
    Fix: define local EXPOSURE_PARAMS_CORRECTED and recompute HI before DALY.

  Bug 3: Lifetime cumulative cases reported as if annual.
    Original cancer_cases = CR_lifetime × pop × 0.5 gives LIFETIME cases over a
    70-year averaging time, but the paper compares the resulting DALYs to GBD's
    ANNUAL burden. Same lifetime/annual mismatch for YLD (uses 20-yr disability
    duration multiplied by prevalence-derived "affected" instead of incidence).
    Fix: convert to annual incidence by dividing lifetime risk by AT in years,
    and treat the chronic-disease term as point prevalence × DW (annual rate),
    consistent with GBD methodology.

VALIDATION TARGET: GBD 2019 attributes ~30-80k DALY/yr to arsenic in Bangladesh
(Murray et al. 2020; IHME GBD Results Tool). After all corrections, our
multi-contaminant annual DALY should be in the same order of magnitude
(possibly 2-5x larger because we add Mn, Fe, Cu, NO3 to the GBD baseline).

OUTPUT: T2_daly_by_zone_CORRECTED.csv and T2_daly_diagnostic.csv (side-by-side
old vs new for transparency).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from config import (
    DATA_FILE, DEPTH_ZONES, PHYSIOGRAPHIC_ZONES,
    HEALTH_CONTAMINANTS, REFERENCE_DOSES, CANCER_SLOPE_FACTORS,
    CR_VI_FRACTION, AT_CARCINOGENIC,
    TABLES_DIR, RANDOM_STATE, assign_zones,
)


# ─── CORRECTED EXPOSURE PARAMETERS ───────────────────────────────────────────
# Match supplementary Table S1 and align with EPA/WHO standards. Tropical
# adjustment retained (2.5 L/day adult male, 2.0 L/day adult female).
EXPOSURE_PARAMS_CORRECTED = {
    'adult_male':   {'bw_kg': 60, 'ir_L_day': 2.5, 'ef_days': 365, 'ed_years': 30},
    'adult_female': {'bw_kg': 55, 'ir_L_day': 2.0, 'ef_days': 365, 'ed_years': 30},
    'child_6_17':   {'bw_kg': 35, 'ir_L_day': 1.5, 'ef_days': 365, 'ed_years': 12},
    'child_0_5':    {'bw_kg': 15, 'ir_L_day': 1.0, 'ef_days': 365, 'ed_years': 6},
}

# ─── BANGLADESH POPULATION / DEMOGRAPHY ──────────────────────────────────────
# DEPRECATED: hand-coded zone populations. Kept only as fallback for when
# WorldPop raster is unavailable. The authoritative zone populations and
# population-weighted HI/CR statistics come from spatial_aggregation.py,
# which integrates WorldPop 2020 (bgd_ppp_2020_1km.tif) over physiographic
# polygons.
ZONE_POPULATION_FALLBACK = {
    'Barind_Tract':           8_500_000,
    'Northern_Terrace':       12_000_000,
    'Brahmaputra_Floodplain': 25_000_000,
    'Ganges_Floodplain':      18_000_000,
    'GBM_Delta':              22_000_000,
    'Meghna_Floodplain':      35_000_000,
    'Eastern_Hills':          15_000_000,
}


def _load_gridded_zone_stats():
    """Try to load pre-computed gridded zone statistics; return None on failure.

    Caller can then fall back to ZONE_POPULATION_FALLBACK + sample-median HI.
    """
    cache = TABLES_DIR / 'T2_zone_population_weighted_stats.csv'
    if cache.exists():
        try:
            return pd.read_csv(cache).set_index('phys_zone')
        except Exception as e:
            print(f"  [grid] failed to load cached stats: {e}")
            return None
    return None


# Initialize ZONE_POPULATION from gridded stats if available; else fallback.
_GRIDDED = _load_gridded_zone_stats()
if _GRIDDED is not None:
    ZONE_POPULATION = {z: int(_GRIDDED.loc[z, 'population'])
                       for z in _GRIDDED.index}
    print(f"  [grid] using WorldPop-derived zone populations "
          f"(total {sum(ZONE_POPULATION.values()):,})")
else:
    ZONE_POPULATION = dict(ZONE_POPULATION_FALLBACK)
    print(f"  [grid] using hardcoded fallback populations "
          f"(total {sum(ZONE_POPULATION.values()):,})")
GW_USAGE_FRACTION = 0.97
AGE_DISTRIBUTION = {
    'child_0_5': 0.10,
    'child_6_17': 0.22,
    'adult_female': 0.34,
    'adult_male': 0.34,
}

# ─── DALY PARAMETERS ─────────────────────────────────────────────────────────
LIFE_EXPECTANCY_BIRTH = 72.6  # Bangladesh 2022 (World Bank)
DISCOUNT_RATE = 0.03

# Disability weights (GBD 2019)
DISABILITY_WEIGHTS = {
    'skin_lesions': 0.011,
    'peripheral_neuropathy': 0.072,
    'lung_cancer': 0.288,
    'bladder_cancer': 0.274,
    'kidney_disease': 0.104,
    'cardiovascular': 0.080,
    'neurodevelopmental': 0.361,
}
CANCER_MANIFESTATION_RATE = 0.5

# Bangladesh-calibrated chronic-disease dose-response (Hill function).
# Replaces an HI>1 step function with a smooth, monotonic curve anchored to:
#   - Argos et al. 2010 (Lancet, HEALS cohort, n=11,746): all-cause mortality
#     hazard ratios HR=1.34 at As 10-50 µg/L, 1.68 at 150-300, 1.69 at >300.
#   - Smith et al. 2000 (Bull WHO): skin lesion prevalence ~10% at 200-400 µg/L,
#     rising to ~30% at >800 µg/L in highest-exposed Bangladeshi populations.
#   - Wasserman et al. 2006/2011 (EHP): Mn-attributable neurodevelopmental
#     impairment at concentrations routinely found in Bangladeshi groundwater.
#
#     P(HI) = P_max * HI^n / (K^n + HI^n)
#
# Anchor points (P_max=0.18, K=4.0, n=1.5):
#   HI=1  -> 2.0% excess prevalence  (low but non-zero, matches Argos low-tier)
#   HI=3  -> 7.1%                    (consistent with cardiovascular HR ~1.35)
#   HI=5  -> 10.5%                   (consistent with mid-tier skin/CVD signal)
#   HI=10 -> 14.4%                   (matches Smith 2000 high-exposure prevalence)
#   HI=20 -> 16.5% (asymptote ~ P_max)
#
# Three smooth-function advantages over the step:
#   (i)   No discontinuity at HI=1: removes instability in bootstrap UI.
#   (ii)  Mechanistically defensible: chronic disease begins to accrue below
#         regulatory thresholds (Argos 2010 finds elevated HR even at 10-50 µg/L).
#   (iii) Dose-response parameter uncertainty (K, n, P_max) can be propagated
#         into the bootstrap, replacing the implicit infinite confidence in
#         the previous step.
HILL_PMAX = 0.18
HILL_K = 4.0
HILL_N = 1.5


def annual_chronic_prevalence(hi, p_max=HILL_PMAX, k=HILL_K, n=HILL_N):
    """Annual excess prevalence of chronic As/Mn-attributable disease.

    Hill function in HI space; defaults match the Bangladesh-anchored
    parameters above. Pass p_max, k, n explicitly to propagate parameter
    uncertainty through bootstrap.
    """
    hi = np.maximum(hi, 0.0)
    return p_max * hi**n / (k**n + hi**n)


# ─── HEALTH RISK RECOMPUTATION WITH CORRECTED IR ─────────────────────────────

def calculate_hi_cr(df, age_group, params):
    """Compute HI_multi, HI_as_only, CR_multi, CR_as_only with given params."""
    ir = params['ir_L_day']
    bw = params['bw_kg']
    # CDI = C * IR / BW for chronic (EF*ED/AT cancels for non-cancer)
    hi_total = np.zeros(len(df))
    hi_as_only = np.zeros(len(df))
    cr_total = np.zeros(len(df))
    cr_as_only = np.zeros(len(df))
    ed = params['ed_years']

    for contaminant, rfd in REFERENCE_DOSES.items():
        if contaminant not in df.columns:
            continue
        conc = df[contaminant].values.copy()
        if contaminant == 'As':
            conc = conc / 1000.0  # ug/L -> mg/L
        cdi = conc * ir / bw  # mg/kg/day
        hq = cdi / rfd
        hi_total = hi_total + hq
        if contaminant == 'As':
            hi_as_only = hq.copy()

    for contaminant, csf in CANCER_SLOPE_FACTORS.items():
        if contaminant not in df.columns:
            continue
        conc = df[contaminant].values.copy()
        if contaminant == 'As':
            conc = conc / 1000.0
        cdi_cancer = (conc * ir * 365 * ed) / (bw * AT_CARCINOGENIC)
        cr = cdi_cancer * csf
        cr_total = cr_total + cr
        if contaminant == 'As':
            cr_as_only = cr.copy()

    if 'Cr3+' in df.columns:
        conc_cr6 = df['Cr3+'].values * CR_VI_FRACTION
        cdi_cr6 = (conc_cr6 * ir * 365 * ed) / (bw * AT_CARCINOGENIC)
        cr_total = cr_total + cdi_cr6 * 0.5

    return hi_total, hi_as_only, cr_total, cr_as_only


# ─── CORRECTED DALY ──────────────────────────────────────────────────────────

def annual_yll_from_cancer(cr_lifetime, pop, age_group):
    """Annual YLL from arsenic/chromium cancer.

    cr_lifetime is the LIFETIME cancer risk per person from CSF × CDI with
    AT_carc=70 years. Annual incidence rate = cr_lifetime / 70.
    Annual cancer cases = pop × annual_incidence × manifestation_rate.
    YLL per case = remaining life expectancy at cancer onset, discounted.
    """
    annual_incidence = cr_lifetime / 70.0
    annual_cases = pop * annual_incidence * CANCER_MANIFESTATION_RATE
    onset_age = {'adult_male': 55, 'adult_female': 55,
                 'child_6_17': 40, 'child_0_5': 40}.get(age_group, 55)
    remaining_le = max(0, LIFE_EXPECTANCY_BIRTH - onset_age)
    # Discounted YLL per death (3% discount, no age weighting per GBD2019)
    if DISCOUNT_RATE > 0:
        yll_per_case = (1 - np.exp(-DISCOUNT_RATE * remaining_le)) / DISCOUNT_RATE
    else:
        yll_per_case = remaining_le
    return annual_cases * yll_per_case, annual_cases


def annual_yld_from_chronic(hi, pop, age_group, dr_params=None):
    """Annual YLD from non-cancer chronic disease.

    Uses point prevalence × DW × population — the standard GBD YLD formula
    when prevalence is given as fraction of population at a snapshot in time.

    dr_params : dict or None
        Optional override for Hill dose-response parameters {p_max, k, n}.
        Used by bootstrap to propagate dose-response uncertainty.
    """
    if dr_params is None:
        prevalence = annual_chronic_prevalence(hi)
    else:
        prevalence = annual_chronic_prevalence(
            hi, dr_params['p_max'], dr_params['k'], dr_params['n']
        )
    affected = prevalence * pop
    if age_group.startswith('child'):
        # Children: dominated by Mn neurodevelopmental + skin
        dw = (0.4 * DISABILITY_WEIGHTS['neurodevelopmental']
              + 0.6 * DISABILITY_WEIGHTS['skin_lesions'])
    else:
        dw = (0.4 * DISABILITY_WEIGHTS['skin_lesions']
              + 0.3 * DISABILITY_WEIGHTS['peripheral_neuropathy']
              + 0.3 * DISABILITY_WEIGHTS['cardiovascular'])
    return affected * dw, affected


def bootstrap_annual_daly_ci(df, n_boot=1000):
    """95% uncertainty interval on national annual DALYs.

    Two uncertainty sources are combined per iteration:
      (1) Sampling/spatial uncertainty — concentrations (via their HI/CR) are
          resampled with replacement WITHIN each zone (independent per zone).
      (2) Exposure-parameter uncertainty — water intake (IR) is drawn as a
          lognormal multiplicative factor (CV 0.30) and body weight (BW) as a
          normal factor (CV 0.15), applied SYSTEMATICALLY across all zones in
          an iteration (these are national exposure assumptions, hence
          correlated across zones). HI and CR both scale by IR/BW.

    Returns a dict of medians and [2.5, 97.5] percentiles for national
    multi-contaminant DALYs, As-only DALYs, and annual cancer cases.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    zones = [z for z in PHYSIOGRAPHIC_ZONES if (df['phys_zone'] == z).sum() >= 5]
    # Cache per-zone HI/CR arrays
    cache = {}
    for z in zones:
        zd = df[df['phys_zone'] == z]
        cache[z] = (zd['HI_multi'].values, zd['HI_as_only'].values,
                    zd['CR_multi'].values, zd['CR_as_only'].values)

    # Anchor: if gridded statistics are available, scale each bootstrap-resampled
    # sample-median up to the corresponding population-weighted gridded median.
    # The within-zone bootstrap then captures sampling uncertainty around this
    # anchored, population-weighted central value rather than the sample-median
    # (which underweights the populous high-HI floodplain cores).
    # Separate scaling factor per metric (HI_multi, HI_as_only, CR_multi, CR_as_only).
    metric_cols = ['HI_multi', 'HI_as_only', 'CR_multi', 'CR_as_only']
    grid_scale = {z: {m: 1.0 for m in metric_cols} for z in zones}
    if _GRIDDED is not None:
        for z in zones:
            if z not in _GRIDDED.index:
                continue
            hm, ha, cm, ca = cache[z]
            sample_med = {'HI_multi': np.median(hm), 'HI_as_only': np.median(ha),
                          'CR_multi': np.median(cm), 'CR_as_only': np.median(ca)}
            for m in metric_cols:
                if sample_med[m] > 0:
                    grid_scale[z][m] = float(_GRIDDED.loc[z, f'{m}_pop_p50']
                                              / sample_med[m])

    sigma_ir = np.sqrt(np.log(1 + 0.30**2))  # lognormal sigma for CV=0.30
    # Dose-response parameter uncertainty (Hill function):
    #   P_max: lognormal CV 0.25 (reflecting Smith 2000 high-tier 10-30% range)
    #   K:     lognormal CV 0.25 (half-maximum HI; Argos 2010 transition zone)
    #   n:     lognormal CV 0.15 (Hill coefficient steepness)
    sig_pmax = np.sqrt(np.log(1 + 0.25**2))
    sig_k = np.sqrt(np.log(1 + 0.25**2))
    sig_n = np.sqrt(np.log(1 + 0.15**2))

    boot_multi = np.empty(n_boot)
    boot_as = np.empty(n_boot)
    boot_cancer = np.empty(n_boot)

    for b in range(n_boot):
        ir_factor = rng.lognormal(0.0, sigma_ir)
        bw_factor = max(0.4, rng.normal(1.0, 0.15))
        scale = ir_factor / bw_factor  # HI and CR both scale by IR/BW
        dr_b = {
            'p_max': float(HILL_PMAX * rng.lognormal(0.0, sig_pmax)),
            'k': float(HILL_K * rng.lognormal(0.0, sig_k)),
            'n': float(HILL_N * rng.lognormal(0.0, sig_n)),
        }

        tot_multi = tot_as = tot_cancer = 0.0
        for z in zones:
            hm, ha, cm, ca = cache[z]
            idx = rng.choice(len(hm), len(hm), replace=True)
            gs = grid_scale[z]  # per-metric scale factor dict
            m_hi = np.median(hm[idx]) * scale * gs['HI_multi']
            m_hi_as = np.median(ha[idx]) * scale * gs['HI_as_only']
            m_cr = np.median(cm[idx]) * scale * gs['CR_multi']
            m_cr_as = np.median(ca[idx]) * scale * gs['CR_as_only']
            gw_pop = ZONE_POPULATION.get(z, 0) * GW_USAGE_FRACTION
            for ag, frac in AGE_DISTRIBUTION.items():
                ap = gw_pop * frac
                yll_m, cc = annual_yll_from_cancer(m_cr, ap, ag)
                yld_m, _ = annual_yld_from_chronic(m_hi, ap, ag, dr_b)
                tot_multi += yll_m + yld_m
                tot_cancer += cc
                yll_a, _ = annual_yll_from_cancer(m_cr_as, ap, ag)
                yld_a, _ = annual_yld_from_chronic(m_hi_as, ap, ag, dr_b)
                tot_as += yll_a + yld_a
        boot_multi[b] = tot_multi
        boot_as[b] = tot_as
        boot_cancer[b] = tot_cancer

    def pct(a):
        return (float(np.percentile(a, 2.5)), float(np.median(a)),
                float(np.percentile(a, 97.5)))
    lo_m, md_m, hi_m = pct(boot_multi)
    lo_a, md_a, hi_a = pct(boot_as)
    lo_c, md_c, hi_c = pct(boot_cancer)
    return {
        'multi': (lo_m, md_m, hi_m),
        'as_only': (lo_a, md_a, hi_a),
        'cancer': (lo_c, md_c, hi_c),
        'n_boot': n_boot,
    }


def estimate_annual_dalys(df):
    """Estimate ANNUAL DALYs per zone.

    Uses population-weighted HI/CR medians from the gridded WorldPop ×
    IDW(HI) aggregation when available (T2_zone_population_weighted_stats.csv);
    otherwise falls back to sample-median HI/CR within the zone.
    """
    rows = []
    for pz in PHYSIOGRAPHIC_ZONES.keys():
        zdata = df[df['phys_zone'] == pz]
        if len(zdata) < 5:
            continue
        zone_pop = ZONE_POPULATION.get(pz, 0)
        gw_pop = zone_pop * GW_USAGE_FRACTION

        # Prefer gridded population-weighted statistics if loaded.
        if _GRIDDED is not None and pz in _GRIDDED.index:
            g = _GRIDDED.loc[pz]
            median_hi_multi = float(g['HI_multi_pop_p50'])
            median_hi_as = float(g['HI_as_only_pop_p50'])
            median_cr_multi = float(g['CR_multi_pop_p50'])
            median_cr_as = float(g['CR_as_only_pop_p50'])
        else:
            median_hi_multi = zdata['HI_multi'].median()
            median_hi_as = zdata['HI_as_only'].median()
            median_cr_multi = zdata['CR_multi'].median()
            median_cr_as = zdata['CR_as_only'].median()

        total_daly_multi = 0.0
        total_daly_as = 0.0
        total_yll = 0.0
        total_yld = 0.0
        total_cancer = 0.0

        for age_group, frac in AGE_DISTRIBUTION.items():
            age_pop = gw_pop * frac
            yll_m, cc_m = annual_yll_from_cancer(median_cr_multi, age_pop, age_group)
            yld_m, _ = annual_yld_from_chronic(median_hi_multi, age_pop, age_group)
            total_daly_multi += yll_m + yld_m
            total_yll += yll_m
            total_yld += yld_m
            total_cancer += cc_m

            yll_a, _ = annual_yll_from_cancer(median_cr_as, age_pop, age_group)
            yld_a, _ = annual_yld_from_chronic(median_hi_as, age_pop, age_group)
            total_daly_as += yll_a + yld_a

        rows.append({
            'phys_zone': pz,
            'population': zone_pop,
            'gw_dependent_pop': gw_pop,
            'n_samples': len(zdata),
            'median_HI_multi': median_hi_multi,
            'median_HI_as_only': median_hi_as,
            'median_CR_multi': median_cr_multi,
            'median_CR_as_only': median_cr_as,
            'annual_DALY_multi': total_daly_multi,
            'annual_DALY_as_only': total_daly_as,
            'DALY_ratio': total_daly_multi / total_daly_as if total_daly_as > 0 else np.nan,
            'annual_YLL': total_yll,
            'annual_YLD': total_yld,
            'annual_cancer_cases': total_cancer,
            'annual_DALY_per_100k': total_daly_multi / (zone_pop / 100_000),
        })
    return pd.DataFrame(rows)


def main():
    print("=" * 70)
    print("T2 DALY — CORRECTED (annual, no seasonal double-count, IR=2.5)")
    print("=" * 70)

    df = pd.read_csv(DATA_FILE)
    df = assign_zones(df)
    print(f"Loaded {len(df)} samples, {df['phys_zone'].nunique()} zones")

    # Recompute HI/CR with corrected exposure parameters (use adult_male as ref;
    # this is the same convention as the original health_risk_stratified.py)
    params = EXPOSURE_PARAMS_CORRECTED['adult_male']
    hi_multi, hi_as, cr_multi, cr_as = calculate_hi_cr(df, 'adult_male', params)
    df['HI_multi'] = hi_multi
    df['HI_as_only'] = hi_as
    df['CR_multi'] = cr_multi
    df['CR_as_only'] = cr_as

    print(f"\nMedian HI_multi (recomputed with IR=2.5): {np.median(hi_multi):.3f}")
    print(f"Median HI_as_only:                         {np.median(hi_as):.3f}")
    print(f"Median CR_multi (lifetime):                {np.median(cr_multi):.2e}")

    daly_df = estimate_annual_dalys(df)
    daly_df.to_csv(TABLES_DIR / 'T2_daly_by_zone_CORRECTED.csv', index=False)

    total_multi = daly_df['annual_DALY_multi'].sum()
    total_as = daly_df['annual_DALY_as_only'].sum()
    total_cancer = daly_df['annual_cancer_cases'].sum()
    total_pop = daly_df['population'].sum()

    print(f"\n--- NATIONAL ANNUAL DALY ESTIMATES (CORRECTED) ---")
    print(f"  Total annual DALYs (multi-contaminant): {total_multi:>12,.0f}")
    print(f"  Total annual DALYs (As-only):           {total_as:>12,.0f}")
    print(f"  Multi/As ratio:                         {total_multi/total_as:>12.2f}x")
    print(f"  Annual cancer incidence:                {total_cancer:>12,.0f}")
    print(f"  Total population:                       {total_pop:>12,.0f}")
    print(f"  DALY rate per 100,000:                  {total_multi/(total_pop/1e5):>12.1f}")

    # ─── Bootstrap 95% uncertainty intervals ─────────────────────────────
    print(f"\n--- BOOTSTRAP 95% UNCERTAINTY INTERVALS (1000 iterations) ---")
    print("  (concentration resampling within zone + IR/BW exposure uncertainty)")
    ci = bootstrap_annual_daly_ci(df, n_boot=1000)
    lo_m, md_m, hi_m = ci['multi']
    lo_a, md_a, hi_a = ci['as_only']
    lo_c, md_c, hi_c = ci['cancer']
    print(f"  Multi-contaminant DALYs:  {md_m:>10,.0f}  "
          f"[{lo_m:,.0f} – {hi_m:,.0f}]")
    print(f"  As-only DALYs:            {md_a:>10,.0f}  "
          f"[{lo_a:,.0f} – {hi_a:,.0f}]")
    print(f"  Annual cancer cases:      {md_c:>10,.0f}  "
          f"[{lo_c:,.0f} – {hi_c:,.0f}]")
    pd.DataFrame([
        {'metric': 'annual_DALY_multi', 'median': md_m, 'ci_lo': lo_m, 'ci_hi': hi_m},
        {'metric': 'annual_DALY_as_only', 'median': md_a, 'ci_lo': lo_a, 'ci_hi': hi_a},
        {'metric': 'annual_cancer_cases', 'median': md_c, 'ci_lo': lo_c, 'ci_hi': hi_c},
    ]).to_csv(TABLES_DIR / 'T2_daly_bootstrap_ci_CORRECTED.csv', index=False)

    print(f"\n--- VALIDATION vs GBD 2019 (BANGLADESH) ---")
    print(f"  GBD 2019 As-attributable DALYs:        ~30,000-80,000 /yr")
    print(f"  Our As-only annual DALYs:              {total_as:>12,.0f} /yr")
    if 20_000 <= total_as <= 200_000:
        print("  -> Consistent with GBD order of magnitude ✓")
    else:
        print(f"  -> Off by {total_as/50_000:.1f}x from GBD midpoint ⚠")

    print(f"\n--- DALYs BY ZONE (annual) ---")
    for _, r in daly_df.sort_values('annual_DALY_multi', ascending=False).iterrows():
        print(f"  {r['phys_zone']:25s}: {r['annual_DALY_multi']:>10,.0f} DALYs "
              f"({r['annual_DALY_per_100k']:>6.1f} per 100k)")

    # Side-by-side diagnostic vs original
    print(f"\n--- ORIGINAL vs CORRECTED (national totals) ---")
    try:
        orig = pd.read_csv(TABLES_DIR / 'T2_daly_by_zone.csv')
        orig_total = orig['DALY_multi'].sum()
        orig_as = orig['DALY_as_only'].sum()
        diag = pd.DataFrame([
            {'metric': 'DALY_multi (national)', 'original': orig_total,
             'corrected': total_multi, 'ratio': orig_total / total_multi},
            {'metric': 'DALY_as_only (national)', 'original': orig_as,
             'corrected': total_as, 'ratio': orig_as / total_as},
            {'metric': 'cancer_cases (national)', 'original': orig['est_cancer_cases'].sum(),
             'corrected': total_cancer,
             'ratio': orig['est_cancer_cases'].sum() / total_cancer},
        ])
        diag.to_csv(TABLES_DIR / 'T2_daly_diagnostic.csv', index=False)
        print(diag.to_string(index=False))
    except FileNotFoundError:
        print("  (original T2_daly_by_zone.csv not found; diagnostic skipped)")

    print("=" * 70)


if __name__ == '__main__':
    main()
