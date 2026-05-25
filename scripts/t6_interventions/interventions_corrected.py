"""
T6: Intervention Cost-Effectiveness — CORRECTED (2026-05-21)
============================================================
Replaces interventions.py for the Nature Water resubmission.

Fixes carried over from daly_estimation_corrected.py:
  - IR = 2.5 L/day (was 3.5)
  - Annual DALY (not lifetime cumulative) via prevalence × DW and CR_lifetime / 70
  - No seasonal double-count (zone-level aggregation)

Cost-effectiveness reframing:
  - Capital costs amortized to annual equivalent over a 10-year intervention
    lifetime at 3% discount (annuity factor 8.530).
  - ICER reported as USD per annual DALY averted, comparable to WHO CHOICE
    cost-effectiveness thresholds (1x and 3x GDP per capita).
  - Bangladesh GDP per capita 2024 ≈ $2,500 (very cost-effective if ICER < GDPpc;
    cost-effective if ICER < 3 × GDPpc = $7,500).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from config import (
    DATA_FILE, REFERENCE_DOSES, CANCER_SLOPE_FACTORS,
    CR_VI_FRACTION, AT_CARCINOGENIC,
    PHYSIOGRAPHIC_ZONES, TABLES_DIR, RANDOM_STATE, assign_zones,
)

# Import corrected DALY machinery (sibling tier folder under scripts/)
sys.path.insert(0, str(Path(__file__).parent.parent / 't2_health'))
from daly_estimation_corrected import (
    EXPOSURE_PARAMS_CORRECTED, ZONE_POPULATION, GW_USAGE_FRACTION,
    AGE_DISTRIBUTION, annual_yll_from_cancer, annual_yld_from_chronic,
    calculate_hi_cr,
)

np.random.seed(RANDOM_STATE)

# ─── INTERVENTION SCENARIOS ─────────────────────────────────────────────────
# Cost data: capex per well + opex per well-year. Annualized assuming 10-year
# capital life at 3% discount: annuity factor a = (1 - (1.03)^-10) / 0.03 = 8.530.
ANNUITY_FACTOR_10YR = 8.530

INTERVENTION_SCENARIOS = {
    'S0_baseline': {
        'description': 'No intervention',
        'capex_per_well_usd': 0, 'opex_per_well_yr_usd': 0,
    },
    'S1_deepen_wells': {
        'description': 'Deepen shallow/intermediate wells to >150 m',
        'capex_per_well_usd': 2500, 'opex_per_well_yr_usd': 20,
        'target_depth_zone': 'Deep',
    },
    'S2_po4_removal': {
        'description': 'PO4 removal sorbent at wellhead',
        'capex_per_well_usd': 150, 'opex_per_well_yr_usd': 30,
        'as_reduction_pct': 60, 'mn_reduction_pct': 10,
    },
    'S3_multi_treatment': {
        'description': 'Multi-contaminant treatment (Fe oxidation + As/Mn removal)',
        'capex_per_well_usd': 800, 'opex_per_well_yr_usd': 100,
        'as_reduction_pct': 85, 'mn_reduction_pct': 70, 'fe_reduction_pct': 90,
    },
    'S4_seasonal_switch': {
        'description': 'Switch to deep wells in wet season only',
        'capex_per_well_usd': 100, 'opex_per_well_yr_usd': 15,
        'wet_season_reduction_pct': 80,
    },
    'S5_fertilizer_policy': {
        'description': 'Reduce PO4 fertilizer 30% in high-As zones',
        'capex_per_well_usd': 0, 'opex_per_well_yr_usd': 0,
        'total_annual_cost_usd': 25_000_000,
        'as_reduction_pct': 20,
    },
}

WELLS_PER_ZONE = {
    'Barind_Tract':           50_000,
    'Northern_Terrace':       80_000,
    'Brahmaputra_Floodplain': 200_000,
    'Ganges_Floodplain':      150_000,
    'GBM_Delta':              180_000,
    'Meghna_Floodplain':      250_000,
    'Eastern_Hills':          100_000,
}

CONTAMINANTS = ['As', 'Mn2+', 'Fe2+', 'Cr3+', 'NO3-']
BD_GDP_PC_USD = 2500


# ─── INTERVENTION APPLICATION ───────────────────────────────────────────────

def apply_intervention(df, scenario_name, scenario):
    mod = df.copy()
    if scenario_name == 'S0_baseline':
        return mod
    if scenario_name == 'S1_deepen_wells':
        deep = df[df['depth_zone'] == 'Deep']
        if len(deep) < 10:
            return mod
        for cont in CONTAMINANTS:
            if cont in mod.columns:
                deep_median = deep[cont].median()
                shallow_mask = mod['depth_zone'].isin(['Shallow', 'Intermediate'])
                mod.loc[shallow_mask, cont] = deep_median
        return mod
    if scenario_name == 'S2_po4_removal':
        as_red = scenario.get('as_reduction_pct', 60) / 100
        mn_red = scenario.get('mn_reduction_pct', 10) / 100
        mod['As'] = mod['As'] * (1 - as_red)
        if 'Mn2+' in mod.columns:
            mod['Mn2+'] = mod['Mn2+'] * (1 - mn_red)
        return mod
    if scenario_name == 'S3_multi_treatment':
        as_red = scenario.get('as_reduction_pct', 85) / 100
        mn_red = scenario.get('mn_reduction_pct', 70) / 100
        fe_red = scenario.get('fe_reduction_pct', 90) / 100
        mod['As'] = mod['As'] * (1 - as_red)
        if 'Mn2+' in mod.columns:
            mod['Mn2+'] = mod['Mn2+'] * (1 - mn_red)
        if 'Fe2+' in mod.columns:
            mod['Fe2+'] = mod['Fe2+'] * (1 - fe_red)
        return mod
    if scenario_name == 'S4_seasonal_switch':
        wet_red = scenario.get('wet_season_reduction_pct', 80) / 100
        wet_mask = mod['Season'] == 'Wet'
        for cont in CONTAMINANTS:
            if cont in mod.columns:
                mod.loc[wet_mask, cont] = mod.loc[wet_mask, cont] * (1 - wet_red)
        return mod
    if scenario_name == 'S5_fertilizer_policy':
        as_red = scenario.get('as_reduction_pct', 20) / 100
        mod['As'] = mod['As'] * (1 - as_red)
        return mod
    return mod


# ─── ANNUAL DALY OVER ALL ZONES ─────────────────────────────────────────────

# Cache baseline sample medians per (zone, metric) so counterfactual scenarios
# can scale them against the gridded population-weighted medians consistently.
_BASELINE_MED_CACHE = {}


def baseline_sample_median(zone, metric):
    """Return the BASELINE (un-intervened) sample median of `metric` in `zone`.

    Computed once on first call and cached. The intervention pipeline then
    scales these baseline medians up to the gridded population-weighted
    medians, so percent reductions in concentrations propagate cleanly.
    """
    key = (zone, metric)
    if key in _BASELINE_MED_CACHE:
        return _BASELINE_MED_CACHE[key]
    base_df = pd.read_csv(DATA_FILE)
    base_df = assign_zones(base_df)
    params_local = EXPOSURE_PARAMS_CORRECTED['adult_male']
    hi_m, hi_a, cr_m, cr_a = calculate_hi_cr(base_df, 'adult_male', params_local)
    base_df['HI_multi'] = hi_m
    base_df['HI_as_only'] = hi_a
    base_df['CR_multi'] = cr_m
    base_df['CR_as_only'] = cr_a
    for z in PHYSIOGRAPHIC_ZONES:
        zd = base_df[base_df['phys_zone'] == z]
        if len(zd) < 5:
            continue
        for m in ('HI_multi', 'HI_as_only', 'CR_multi', 'CR_as_only'):
            _BASELINE_MED_CACHE[(z, m)] = float(zd[m].median())
    return _BASELINE_MED_CACHE.get(key, 0.0)


def national_annual_daly(df, params):
    """Compute national annual DALYs from a dataset (single age-group ref male).

    Aggregates HI/CR by zone. Uses gridded WorldPop × IDW-interpolated HI/CR
    medians when the precomputed table is available; falls back to sample
    medians within zone otherwise.

    For counterfactual scenarios (mod_df with reduced concentrations), gridded
    aggregation is re-run on the modified samples by applying the same
    sample-to-grid scaling factor computed on the baseline. This is exact when
    the intervention is a uniform multiplicative reduction of all wells in the
    zone, and a good approximation otherwise.
    """
    from daly_estimation_corrected import _GRIDDED  # type: ignore
    hi_multi, hi_as, cr_multi, cr_as = calculate_hi_cr(df, 'adult_male', params)
    df = df.copy()
    df['HI_multi'] = hi_multi
    df['HI_as_only'] = hi_as
    df['CR_multi'] = cr_multi
    df['CR_as_only'] = cr_as

    total_daly_multi = 0.0
    total_daly_as = 0.0
    total_cancer = 0.0
    for pz in PHYSIOGRAPHIC_ZONES.keys():
        zdata = df[df['phys_zone'] == pz]
        if len(zdata) < 5:
            continue
        zone_pop = ZONE_POPULATION.get(pz, 0)
        gw_pop = zone_pop * GW_USAGE_FRACTION
        m_hi = zdata['HI_multi'].median()
        m_cr = zdata['CR_multi'].median()
        m_hi_as = zdata['HI_as_only'].median()
        m_cr_as = zdata['CR_as_only'].median()
        if _GRIDDED is not None and pz in _GRIDDED.index:
            # Scale sample-medians up to the gridded population-weighted medians.
            # For counterfactual scenarios this preserves the fractional change
            # induced by the intervention while keeping spatial weighting.
            g = _GRIDDED.loc[pz]
            ref_hi = baseline_sample_median(pz, 'HI_multi')
            ref_hi_as = baseline_sample_median(pz, 'HI_as_only')
            ref_cr = baseline_sample_median(pz, 'CR_multi')
            ref_cr_as = baseline_sample_median(pz, 'CR_as_only')
            if ref_hi > 0:
                m_hi *= float(g['HI_multi_pop_p50']) / ref_hi
            if ref_hi_as > 0:
                m_hi_as *= float(g['HI_as_only_pop_p50']) / ref_hi_as
            if ref_cr > 0:
                m_cr *= float(g['CR_multi_pop_p50']) / ref_cr
            if ref_cr_as > 0:
                m_cr_as *= float(g['CR_as_only_pop_p50']) / ref_cr_as
        for age_group, frac in AGE_DISTRIBUTION.items():
            age_pop = gw_pop * frac
            yll_m, cc_m = annual_yll_from_cancer(m_cr, age_pop, age_group)
            yld_m, _ = annual_yld_from_chronic(m_hi, age_pop, age_group)
            total_daly_multi += yll_m + yld_m
            total_cancer += cc_m
            yll_a, _ = annual_yll_from_cancer(m_cr_as, age_pop, age_group)
            yld_a, _ = annual_yld_from_chronic(m_hi_as, age_pop, age_group)
            total_daly_as += yll_a + yld_a
    return total_daly_multi, total_daly_as, total_cancer


def annualized_cost(scenario, n_wells):
    """Total annual cost in USD for a given scenario, given n_wells deployed."""
    if 'total_annual_cost_usd' in scenario:
        return scenario['total_annual_cost_usd']
    capex_total = scenario['capex_per_well_usd'] * n_wells
    opex_annual = scenario['opex_per_well_yr_usd'] * n_wells
    capex_annual = capex_total / ANNUITY_FACTOR_10YR
    return capex_annual + opex_annual


def main():
    print("=" * 70)
    print("T6 INTERVENTIONS — CORRECTED (annual DALYs, annualized costs)")
    print("=" * 70)

    df = pd.read_csv(DATA_FILE)
    df = assign_zones(df)
    print(f"Loaded {len(df)} samples")

    params = EXPOSURE_PARAMS_CORRECTED['adult_male']

    # Baseline
    daly_base, daly_as_base, cancer_base = national_annual_daly(df, params)
    print(f"\nBaseline annual DALYs (multi): {daly_base:>12,.0f}")
    print(f"Baseline annual DALYs (As-only): {daly_as_base:>12,.0f}")
    print(f"Baseline annual cancer cases:   {cancer_base:>12,.0f}")

    rows = []
    total_wells = sum(WELLS_PER_ZONE.values())  # ~1.01M nationally

    for name, scenario in INTERVENTION_SCENARIOS.items():
        mod_df = apply_intervention(df, name, scenario)
        daly_mod, daly_as_mod, cancer_mod = national_annual_daly(mod_df, params)
        dalys_averted = daly_base - daly_mod
        cancer_averted = cancer_base - cancer_mod
        cost_annual = annualized_cost(scenario, total_wells)
        icer = cost_annual / dalys_averted if dalys_averted > 0 else np.inf

        rows.append({
            'scenario': name,
            'description': scenario['description'],
            'annual_DALY': daly_mod,
            'annual_DALY_averted': dalys_averted,
            'pct_reduction': 100 * dalys_averted / daly_base if daly_base > 0 else 0,
            'annual_cancer_cases': cancer_mod,
            'annual_cost_usd': cost_annual,
            'ICER_usd_per_daly': icer,
            'very_cost_effective': icer < BD_GDP_PC_USD,
            'cost_effective': icer < 3 * BD_GDP_PC_USD,
        })

    result = pd.DataFrame(rows)
    result.to_csv(TABLES_DIR / 'T6_interventions_CORRECTED.csv', index=False)

    print(f"\n--- SCENARIO RANKING (by ICER, lower is better) ---")
    print(f"WHO-CHOICE thresholds: very cost-effective < ${BD_GDP_PC_USD:,} | "
          f"cost-effective < ${3*BD_GDP_PC_USD:,}\n")
    display_cols = ['scenario', 'annual_DALY_averted', 'pct_reduction',
                    'annual_cost_usd', 'ICER_usd_per_daly', 'very_cost_effective']
    show = result[display_cols].sort_values('ICER_usd_per_daly')
    print(show.to_string(index=False, float_format=lambda x: f'{x:>12,.1f}'))

    print("=" * 70)


if __name__ == '__main__':
    main()
