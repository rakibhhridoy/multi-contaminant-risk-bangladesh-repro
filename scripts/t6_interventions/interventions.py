"""
T6: Counterfactual Intervention Scenarios
Models DALYs averted and cost-effectiveness of 6 mitigation strategies.
Uses T2 health risk and T1 climate projections as baseline.

Integration:
  - Reads T1_projected_concentrations_2050.csv for climate-amplified analysis
  - Adds probabilistic sensitivity analysis (PSA) with Beta-distributed efficacy
  - Computes ICER on the cost-effectiveness frontier
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from config import (
    DATA_FILE, DEPTH_ZONES, PHYSIOGRAPHIC_ZONES,
    HEALTH_CONTAMINANTS, REFERENCE_DOSES, CANCER_SLOPE_FACTORS,
    EXPOSURE_PARAMS, AT_CARCINOGENIC, CR_VI_FRACTION,
    TABLES_DIR, FIGURES_DIR, RANDOM_STATE,
    assign_zones
)

np.random.seed(RANDOM_STATE)

# ─── INTERVENTION SCENARIOS ────────────────────────────────────────────────────

INTERVENTION_SCENARIOS = {
    'S0_baseline': {
        'description': 'No intervention (current state)',
        'cost_per_well_usd': 0,
    },
    'S1_deepen_wells': {
        'description': 'Deepen all shallow/intermediate wells to >150m',
        'cost_per_well_usd': 2500,
        'target_depth_zone': 'Deep',
    },
    'S2_po4_removal': {
        'description': 'PO4-removal sorbent at wellhead',
        'cost_per_well_usd': 150,
        'as_reduction_pct': 60,
        'mn_reduction_pct': 10,
    },
    'S3_multi_treatment': {
        'description': 'Community-scale multi-contaminant treatment (Fe oxidation + As/Mn removal)',
        'cost_per_well_usd': 800,
        'as_reduction_pct': 85,
        'mn_reduction_pct': 70,
        'fe_reduction_pct': 90,
    },
    'S4_seasonal_switch': {
        'description': 'Switch to deep wells during monsoon (wet season only)',
        'cost_per_well_usd': 100,
        'wet_season_reduction_pct': 80,
    },
    'S5_fertilizer_policy': {
        'description': 'Reduce PO4 fertilizer 30% in high-As zones (indirect As reduction)',
        'cost_per_well_usd': 0,
        'cost_total_usd': 50_000_000,  # policy + agricultural cost
        'as_reduction_pct': 20,
    },
}

# Zone populations (from T2)
ZONE_POPULATION = {
    'Barind_Tract':           8_500_000,
    'Northern_Terrace':       12_000_000,
    'Brahmaputra_Floodplain': 25_000_000,
    'Ganges_Floodplain':      18_000_000,
    'GBM_Delta':              22_000_000,
    'Meghna_Floodplain':      35_000_000,
    'Eastern_Hills':          15_000_000,
}

GW_USAGE_FRACTION = 0.97

# DALY parameters (from T2 daly_estimation.py)
LIFE_EXPECTANCY = 72.6
DISCOUNT_RATE = 0.03
CANCER_MANIFESTATION = 0.5

DISABILITY_WEIGHTS = {
    'cancer': 0.288,
    'chronic': 0.05,  # weighted average of skin + neuropathy + cardio
}

CONTAMINANTS = ['As', 'Mn2+', 'Fe2+', 'Cr3+', 'NO3-']
WHO_GUIDELINES = {'As': 10.0, 'Mn2+': 0.4, 'Fe2+': 0.3, 'Cr3+': 0.05, 'NO3-': 50.0}

# Wells per zone (approximate from Bangladesh DPHE data)
WELLS_PER_ZONE = {
    'Barind_Tract':           50_000,
    'Northern_Terrace':       80_000,
    'Brahmaputra_Floodplain': 200_000,
    'Ganges_Floodplain':      150_000,
    'GBM_Delta':              180_000,
    'Meghna_Floodplain':      250_000,
    'Eastern_Hills':          100_000,
}

# PSA parameters
N_PSA = 1000  # MC iterations for probabilistic sensitivity analysis


# ─── DATA LOADING ──────────────────────────────────────────────────────────────

def load_data():
    """Load dataset with zone assignments using shared config function."""
    df = pd.read_csv(DATA_FILE)
    df = assign_zones(df)
    return df


# ─── HEALTH RISK CALCULATION ──────────────────────────────────────────────────

def calculate_hi_cr(df, params=None):
    """Calculate HI and CR for given concentrations."""
    if params is None:
        params = EXPOSURE_PARAMS['adult_male']

    ir = params['ir_L_day']
    bw = params['bw_kg']

    hi = np.zeros(len(df))
    cr = np.zeros(len(df))

    for cont, rfd in REFERENCE_DOSES.items():
        if cont not in df.columns:
            continue
        conc = df[cont].values.copy()
        if cont == 'As':
            conc = conc / 1000.0
        cdi = (conc * ir * 365 * 30) / (bw * 30 * 365)
        hi += cdi / rfd

    for cont, csf in CANCER_SLOPE_FACTORS.items():
        if cont not in df.columns:
            continue
        conc = df[cont].values.copy()
        if cont == 'As':
            conc = conc / 1000.0
        cdi_cancer = (conc * ir * 365 * 30) / (bw * AT_CARCINOGENIC)
        cr += cdi_cancer * csf

    if 'Cr3+' in df.columns:
        conc_cr6 = df['Cr3+'].values * CR_VI_FRACTION
        cdi_cr6 = (conc_cr6 * ir * 365 * 30) / (bw * AT_CARCINOGENIC)
        cr += cdi_cr6 * 0.5

    return hi, cr


def estimate_dalys(hi_array, cr_array, population):
    """Simplified DALY estimation from HI/CR arrays and population."""
    # YLL from cancer
    cancer_cases = np.mean(cr_array) * population * CANCER_MANIFESTATION
    remaining_le = max(0, LIFE_EXPECTANCY - 55)
    if DISCOUNT_RATE > 0:
        yll = cancer_cases * (1 - np.exp(-DISCOUNT_RATE * remaining_le)) / DISCOUNT_RATE
    else:
        yll = cancer_cases * remaining_le

    # YLD from chronic effects
    p_disease = np.clip(np.mean(hi_array) / 10, 0, 1) * float(np.mean(hi_array) > 1)
    affected = p_disease * population
    duration = 20.0
    dw = DISABILITY_WEIGHTS['chronic']
    if DISCOUNT_RATE > 0:
        yld = affected * dw * (1 - np.exp(-DISCOUNT_RATE * duration)) / DISCOUNT_RATE
    else:
        yld = affected * dw * duration

    return yll + yld, yll, yld


# ─── INTERVENTION MODELING ─────────────────────────────────────────────────────

def apply_intervention(df, scenario_name, scenario, efficacy_overrides=None):
    """Apply intervention scenario to dataset, return modified copy.

    Parameters
    ----------
    efficacy_overrides : dict or None
        If provided, overrides scenario reduction percentages.
        Keys: 'as_reduction_pct', 'mn_reduction_pct', 'fe_reduction_pct', etc.
    """
    mod = df.copy()
    eff = efficacy_overrides or {}

    if scenario_name == 'S0_baseline':
        return mod

    elif scenario_name == 'S1_deepen_wells':
        # Replace shallow/intermediate concentrations with deep zone medians
        deep = df[df['depth_zone'] == 'Deep']
        if len(deep) < 10:
            return mod
        for cont in CONTAMINANTS:
            if cont in mod.columns:
                deep_median = deep[cont].median()
                shallow_mask = mod['depth_zone'].isin(['Shallow', 'Intermediate'])
                mod.loc[shallow_mask, cont] = deep_median
        return mod

    elif scenario_name == 'S2_po4_removal':
        as_red = eff.get('as_reduction_pct', scenario.get('as_reduction_pct', 60)) / 100
        mn_red = eff.get('mn_reduction_pct', scenario.get('mn_reduction_pct', 10)) / 100
        mod['As'] = mod['As'] * (1 - as_red)
        if 'Mn2+' in mod.columns:
            mod['Mn2+'] = mod['Mn2+'] * (1 - mn_red)
        return mod

    elif scenario_name == 'S3_multi_treatment':
        as_red = eff.get('as_reduction_pct', scenario.get('as_reduction_pct', 85)) / 100
        mn_red = eff.get('mn_reduction_pct', scenario.get('mn_reduction_pct', 70)) / 100
        fe_red = eff.get('fe_reduction_pct', scenario.get('fe_reduction_pct', 90)) / 100
        mod['As'] = mod['As'] * (1 - as_red)
        if 'Mn2+' in mod.columns:
            mod['Mn2+'] = mod['Mn2+'] * (1 - mn_red)
        if 'Fe2+' in mod.columns:
            mod['Fe2+'] = mod['Fe2+'] * (1 - fe_red)
        return mod

    elif scenario_name == 'S4_seasonal_switch':
        # Only reduce wet season; keep dry season unchanged
        wet_red = eff.get('wet_season_reduction_pct',
                          scenario.get('wet_season_reduction_pct', 80)) / 100
        wet_mask = mod['Season'] == 'Wet'
        for cont in CONTAMINANTS:
            if cont in mod.columns:
                mod.loc[wet_mask, cont] = mod.loc[wet_mask, cont] * (1 - wet_red)
        return mod

    elif scenario_name == 'S5_fertilizer_policy':
        # Indirect As reduction through PO4 reduction
        as_red = eff.get('as_reduction_pct', scenario.get('as_reduction_pct', 20)) / 100
        mod['As'] = mod['As'] * (1 - as_red)
        return mod

    return mod


def run_all_scenarios(df):
    """Run all intervention scenarios and compare."""
    results = []

    # Calculate baseline
    hi_base, cr_base = calculate_hi_cr(df)
    total_pop = sum(ZONE_POPULATION.get(z, 0) for z in df['phys_zone'].unique())
    gw_pop = total_pop * GW_USAGE_FRACTION
    daly_base, yll_base, yld_base = estimate_dalys(hi_base, cr_base, gw_pop)

    for name, scenario in INTERVENTION_SCENARIOS.items():
        mod_df = apply_intervention(df, name, scenario)
        hi_mod, cr_mod = calculate_hi_cr(mod_df)
        daly_mod, yll_mod, yld_mod = estimate_dalys(hi_mod, cr_mod, gw_pop)

        # DALYs averted
        dalys_averted = daly_base - daly_mod

        # Cost
        total_wells = sum(WELLS_PER_ZONE.get(z, 0) for z in df['phys_zone'].unique())
        cost_per_well = scenario.get('cost_per_well_usd', 0)
        total_cost = scenario.get('cost_total_usd', cost_per_well * total_wells)

        # Cost-effectiveness
        ce = total_cost / dalys_averted if dalys_averted > 0 else np.inf

        # WHO threshold: CE < 1x GDP per capita = very cost-effective
        # Bangladesh GDP per capita ~$2,700 (2023)
        who_ce_threshold = 2700

        results.append({
            'scenario': name,
            'description': scenario['description'],
            'HI_median': np.median(hi_mod),
            'HI_exceed_pct': (hi_mod > 1).mean() * 100,
            'CR_median': np.median(cr_mod),
            'CR_exceed_pct': (cr_mod > 1e-4).mean() * 100,
            'DALYs': daly_mod,
            'DALYs_averted': dalys_averted,
            'pct_reduction': dalys_averted / daly_base * 100 if daly_base > 0 else 0,
            'total_cost_usd': total_cost,
            'cost_per_daly_averted': ce,
            'very_cost_effective': ce < who_ce_threshold,
        })

    return pd.DataFrame(results)


def zone_equity_analysis(df):
    """Which zones benefit most from each intervention?"""
    results = []

    for pz in PHYSIOGRAPHIC_ZONES.keys():
        zone_df = df[df['phys_zone'] == pz]
        if len(zone_df) < 10:
            continue

        hi_base, cr_base = calculate_hi_cr(zone_df)
        zone_pop = ZONE_POPULATION.get(pz, 0) * GW_USAGE_FRACTION
        daly_base, _, _ = estimate_dalys(hi_base, cr_base, zone_pop)

        for name, scenario in INTERVENTION_SCENARIOS.items():
            if name == 'S0_baseline':
                continue
            mod_df = apply_intervention(zone_df, name, scenario)
            hi_mod, cr_mod = calculate_hi_cr(mod_df)
            daly_mod, _, _ = estimate_dalys(hi_mod, cr_mod, zone_pop)

            results.append({
                'phys_zone': pz,
                'scenario': name,
                'baseline_daly': daly_base,
                'intervention_daly': daly_mod,
                'dalys_averted': daly_base - daly_mod,
                'pct_reduction': (daly_base - daly_mod) / daly_base * 100 if daly_base > 0 else 0,
                'baseline_hi_exceed_pct': (hi_base > 1).mean() * 100,
                'intervention_hi_exceed_pct': (hi_mod > 1).mean() * 100,
            })

    return pd.DataFrame(results)


# ─── PROBABILISTIC SENSITIVITY ANALYSIS (PSA) ────────────────────────────────

def run_psa(df, n_iter=N_PSA):
    """Run probabilistic sensitivity analysis with Beta-distributed treatment efficacy.

    S2 As reduction ~ Beta(6, 4)*100   -> mean 60%, spread ~15%
    S3 As reduction ~ Beta(8.5, 1.5)*100 -> mean 85%, spread ~10%

    Returns DataFrame with per-iteration DALYs averted and cost per DALY,
    plus summary statistics with 95% CI.
    """
    rng = np.random.RandomState(RANDOM_STATE)

    hi_base, cr_base = calculate_hi_cr(df)
    total_pop = sum(ZONE_POPULATION.get(z, 0) for z in df['phys_zone'].unique())
    gw_pop = total_pop * GW_USAGE_FRACTION
    daly_base, _, _ = estimate_dalys(hi_base, cr_base, gw_pop)

    total_wells = sum(WELLS_PER_ZONE.get(z, 0) for z in df['phys_zone'].unique())

    # PSA distributions for treatment efficacy
    psa_distributions = {
        'S2_po4_removal': {
            'as_reduction_pct': lambda: rng.beta(6, 4) * 100,       # mean ~60%
            'mn_reduction_pct': lambda: rng.beta(2, 18) * 100,      # mean ~10%
        },
        'S3_multi_treatment': {
            'as_reduction_pct': lambda: rng.beta(8.5, 1.5) * 100,   # mean ~85%
            'mn_reduction_pct': lambda: rng.beta(7, 3) * 100,       # mean ~70%
            'fe_reduction_pct': lambda: rng.beta(9, 1) * 100,       # mean ~90%
        },
        'S5_fertilizer_policy': {
            'as_reduction_pct': lambda: rng.beta(4, 16) * 100,      # mean ~20%
        },
    }

    iteration_rows = []

    for i in range(n_iter):
        for name, scenario in INTERVENTION_SCENARIOS.items():
            if name == 'S0_baseline':
                continue

            # Sample efficacy from PSA distributions if available
            efficacy_overrides = {}
            if name in psa_distributions:
                for key, sampler in psa_distributions[name].items():
                    efficacy_overrides[key] = sampler()

            mod_df = apply_intervention(df, name, scenario, efficacy_overrides=efficacy_overrides)
            hi_mod, cr_mod = calculate_hi_cr(mod_df)
            daly_mod, _, _ = estimate_dalys(hi_mod, cr_mod, gw_pop)
            dalys_averted = daly_base - daly_mod

            cost_per_well = scenario.get('cost_per_well_usd', 0)
            total_cost = scenario.get('cost_total_usd', cost_per_well * total_wells)
            ce = total_cost / dalys_averted if dalys_averted > 0 else np.inf

            iteration_rows.append({
                'iteration': i,
                'scenario': name,
                'dalys_averted': dalys_averted,
                'total_cost_usd': total_cost,
                'cost_per_daly': ce,
                **{k: v for k, v in efficacy_overrides.items()},
            })

    iter_df = pd.DataFrame(iteration_rows)

    # Summary with 95% CI
    summary = []
    for name in INTERVENTION_SCENARIOS:
        if name == 'S0_baseline':
            continue
        sc = iter_df[iter_df['scenario'] == name]
        averted = sc['dalys_averted'].values
        ce_vals = sc['cost_per_daly'].replace([np.inf], np.nan).dropna().values

        summary.append({
            'scenario': name,
            'dalys_averted_mean': np.mean(averted),
            'dalys_averted_median': np.median(averted),
            'dalys_averted_p2.5': np.percentile(averted, 2.5),
            'dalys_averted_p97.5': np.percentile(averted, 97.5),
            'cost_per_daly_mean': np.mean(ce_vals) if len(ce_vals) > 0 else np.inf,
            'cost_per_daly_median': np.median(ce_vals) if len(ce_vals) > 0 else np.inf,
            'cost_per_daly_p2.5': np.percentile(ce_vals, 2.5) if len(ce_vals) > 0 else np.nan,
            'cost_per_daly_p97.5': np.percentile(ce_vals, 97.5) if len(ce_vals) > 0 else np.nan,
        })

    summary_df = pd.DataFrame(summary)
    return iter_df, summary_df


# ─── ICER CALCULATION ─────────────────────────────────────────────────────────

def compute_icer(scenario_df):
    """Compute Incremental Cost-Effectiveness Ratio on the efficiency frontier.

    Sorts scenarios by DALYs averted (ascending), then identifies the frontier
    by removing dominated strategies. Returns DataFrame with ICER column.
    """
    sdf = scenario_df[scenario_df['scenario'] != 'S0_baseline'].copy()
    sdf = sdf.sort_values('DALYs_averted').reset_index(drop=True)

    # Add origin (no intervention)
    icer_rows = []

    # Build efficiency frontier via extended dominance elimination
    frontier = sdf.copy()

    # Step 1: remove strongly dominated (higher cost, fewer DALYs)
    non_dominated = []
    for i, row in frontier.iterrows():
        dominated = False
        for j, other in frontier.iterrows():
            if i == j:
                continue
            if other['DALYs_averted'] >= row['DALYs_averted'] and other['total_cost_usd'] <= row['total_cost_usd']:
                if other['DALYs_averted'] > row['DALYs_averted'] or other['total_cost_usd'] < row['total_cost_usd']:
                    dominated = True
                    break
        if not dominated:
            non_dominated.append(row)

    frontier = pd.DataFrame(non_dominated).sort_values('DALYs_averted').reset_index(drop=True)

    # Step 2: compute ICER relative to next-cheaper on the frontier
    prev_cost = 0.0
    prev_daly = 0.0

    for i, row in frontier.iterrows():
        incr_cost = row['total_cost_usd'] - prev_cost
        incr_daly = row['DALYs_averted'] - prev_daly
        icer = incr_cost / incr_daly if incr_daly > 0 else np.inf

        icer_rows.append({
            'scenario': row['scenario'],
            'description': row['description'],
            'DALYs_averted': row['DALYs_averted'],
            'total_cost_usd': row['total_cost_usd'],
            'incremental_cost': incr_cost,
            'incremental_dalys': incr_daly,
            'ICER': icer,
            'on_frontier': True,
        })

        prev_cost = row['total_cost_usd']
        prev_daly = row['DALYs_averted']

    # Mark non-frontier scenarios
    frontier_names = set(r['scenario'] for r in icer_rows)
    for _, row in sdf.iterrows():
        if row['scenario'] not in frontier_names:
            icer_rows.append({
                'scenario': row['scenario'],
                'description': row['description'],
                'DALYs_averted': row['DALYs_averted'],
                'total_cost_usd': row['total_cost_usd'],
                'incremental_cost': np.nan,
                'incremental_dalys': np.nan,
                'ICER': np.nan,
                'on_frontier': False,
            })

    return pd.DataFrame(icer_rows)


# ─── CLIMATE-AMPLIFIED INTERVENTION ANALYSIS ─────────────────────────────────

def load_future_concentrations():
    """Load T1_projected_concentrations_2050.csv if available."""
    fpath = TABLES_DIR / 'T1_projected_concentrations_2050.csv'
    if fpath.exists():
        return pd.read_csv(fpath)
    print(f"  WARNING: {fpath} not found. Run T1 first for climate-amplified analysis.")
    return None


def build_future_baseline(df, proj_2050):
    """Replace contaminant concentrations with SSP5-8.5 2050 projections.

    For each sample, look up its zone x depth x contaminant projected median
    and replace the dry-season baseline accordingly.
    """
    mod = df.copy()

    for _, prow in proj_2050.iterrows():
        pz = prow['phys_zone']
        dz = prow['depth_zone']
        cont = prow['contaminant']
        projected = prow['projected_median_ssp585']

        if cont not in mod.columns:
            continue

        mask = (mod['phys_zone'] == pz) & (mod['depth_zone'] == dz)
        if mask.sum() == 0:
            continue

        # Scale: multiply by ratio of projected / current baseline
        baseline = prow['baseline_median']
        if baseline > 0:
            scale_factor = projected / baseline
            mod.loc[mask, cont] = mod.loc[mask, cont] * scale_factor
        else:
            # If baseline is zero, set to projected value directly
            mod.loc[mask, cont] = projected

    return mod


def run_climate_amplified_scenarios(df, proj_2050):
    """Run all interventions on future SSP5-8.5 2050 baseline."""
    future_df = build_future_baseline(df, proj_2050)

    hi_base, cr_base = calculate_hi_cr(future_df)
    total_pop = sum(ZONE_POPULATION.get(z, 0) for z in future_df['phys_zone'].unique())
    gw_pop = total_pop * GW_USAGE_FRACTION
    daly_base, _, _ = estimate_dalys(hi_base, cr_base, gw_pop)

    results = []
    for name, scenario in INTERVENTION_SCENARIOS.items():
        mod_df = apply_intervention(future_df, name, scenario)
        hi_mod, cr_mod = calculate_hi_cr(mod_df)
        daly_mod, _, _ = estimate_dalys(hi_mod, cr_mod, gw_pop)
        dalys_averted = daly_base - daly_mod

        total_wells = sum(WELLS_PER_ZONE.get(z, 0) for z in future_df['phys_zone'].unique())
        cost_per_well = scenario.get('cost_per_well_usd', 0)
        total_cost = scenario.get('cost_total_usd', cost_per_well * total_wells)
        ce = total_cost / dalys_averted if dalys_averted > 0 else np.inf

        results.append({
            'scenario': name,
            'description': scenario['description'],
            'climate_scenario': 'SSP5-8.5_2050',
            'HI_median': np.median(hi_mod),
            'HI_exceed_pct': (hi_mod > 1).mean() * 100,
            'DALYs': daly_mod,
            'DALYs_averted': dalys_averted,
            'pct_reduction': dalys_averted / daly_base * 100 if daly_base > 0 else 0,
            'total_cost_usd': total_cost,
            'cost_per_daly_averted': ce,
        })

    return pd.DataFrame(results), daly_base


# ─── FIGURES ────────────────────────────────────────────────────────────────────

def plot_scenario_comparison(scenario_df, output_dir):
    """Bar chart: DALYs averted and cost-effectiveness by scenario."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # Exclude baseline
    sdf = scenario_df[scenario_df['scenario'] != 'S0_baseline'].copy()
    sdf['short_name'] = sdf['scenario'].str.replace('S[0-9]_', '', regex=True)

    # DALYs averted
    ax = axes[0]
    colors = ['#4575b4', '#91bfdb', '#fc8d59', '#d73027', '#fee08b']
    ax.barh(sdf['short_name'], sdf['DALYs_averted'], color=colors[:len(sdf)])
    ax.set_xlabel('DALYs Averted')
    ax.set_title('Health Benefit (DALYs Averted)')

    # Cost
    ax = axes[1]
    ax.barh(sdf['short_name'], sdf['total_cost_usd'] / 1e6, color=colors[:len(sdf)])
    ax.set_xlabel('Total Cost (million USD)')
    ax.set_title('Implementation Cost')

    # Cost-effectiveness
    ax = axes[2]
    ce = sdf['cost_per_daly_averted'].replace([np.inf], np.nan).dropna()
    ce_sdf = sdf.loc[ce.index]
    ax.barh(ce_sdf['short_name'], ce_sdf['cost_per_daly_averted'],
            color=[c for j, c in enumerate(colors) if j in ce.index])
    ax.axvline(2700, color='green', ls='--', lw=2, label='GDP/cap ($2,700)\n= very cost-effective')
    ax.set_xlabel('USD per DALY Averted')
    ax.set_title('Cost-Effectiveness')
    ax.legend(fontsize=9)

    plt.suptitle('T6: Intervention Scenario Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'T6_F01_scenario_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved scenario comparison")


def plot_equity_heatmap(equity_df, output_dir):
    """Heatmap: % DALY reduction by zone x scenario."""
    fig, ax = plt.subplots(figsize=(14, 7))

    pivot = equity_df.pivot_table(index='phys_zone', columns='scenario',
                                   values='pct_reduction', aggfunc='first')
    zone_order = [z for z in PHYSIOGRAPHIC_ZONES.keys() if z in pivot.index]
    pivot = pivot.reindex(zone_order)
    # Rename columns for readability
    pivot.columns = [c.replace('S1_', '').replace('S2_', '').replace('S3_', '')
                      .replace('S4_', '').replace('S5_', '') for c in pivot.columns]

    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlGn', ax=ax,
                linewidths=0.5, vmin=0, vmax=100,
                cbar_kws={'label': '% DALY Reduction'})
    ax.set_title('Equity Analysis: Health Benefit by Zone and Intervention')

    plt.tight_layout()
    plt.savefig(output_dir / 'T6_F02_equity_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved equity heatmap")


def plot_hi_reduction_boxplot(df, output_dir):
    """Box plots: HI distribution under each scenario."""
    fig, ax = plt.subplots(figsize=(14, 7))

    data = []
    for name, scenario in INTERVENTION_SCENARIOS.items():
        mod_df = apply_intervention(df, name, scenario)
        hi, _ = calculate_hi_cr(mod_df)
        short = name.replace('S0_', '').replace('S1_', '').replace('S2_', '') \
                     .replace('S3_', '').replace('S4_', '').replace('S5_', '')
        for val in hi:
            data.append({'Scenario': short, 'HI': val})

    plot_df = pd.DataFrame(data)
    sns.boxplot(data=plot_df, x='Scenario', y='HI', ax=ax, showfliers=False,
                palette='Set2')
    ax.axhline(1, color='red', ls='--', lw=1.5, label='HI = 1 threshold')
    ax.set_ylabel('Hazard Index')
    ax.set_title('Health Risk Under Different Interventions')
    ax.legend()
    plt.xticks(rotation=30, ha='right')

    plt.tight_layout()
    plt.savefig(output_dir / 'T6_F03_hi_reduction_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved HI reduction boxplot")


def plot_psa_ceac(psa_iter_df, output_dir):
    """Cost-effectiveness acceptability curve from PSA iterations."""
    fig, ax = plt.subplots(figsize=(12, 7))

    wtp_range = np.linspace(0, 10000, 200)
    scenarios = [s for s in INTERVENTION_SCENARIOS if s != 'S0_baseline']
    colors = ['#4575b4', '#91bfdb', '#fc8d59', '#d73027', '#fee08b']

    for i, name in enumerate(scenarios):
        sc = psa_iter_df[psa_iter_df['scenario'] == name]
        ce_vals = sc['cost_per_daly'].replace([np.inf], 1e15).values
        prob_ce = [np.mean(ce_vals <= wtp) for wtp in wtp_range]
        short = name.replace('S0_', '').replace('S1_', '').replace('S2_', '') \
                     .replace('S3_', '').replace('S4_', '').replace('S5_', '')
        ax.plot(wtp_range, prob_ce, color=colors[i % len(colors)], lw=2, label=short)

    ax.axvline(2700, color='green', ls='--', lw=1.5, label='WTP = $2,700/DALY')
    ax.set_xlabel('Willingness to Pay (USD per DALY averted)')
    ax.set_ylabel('Probability Cost-Effective')
    ax.set_title('Cost-Effectiveness Acceptability Curves (PSA)')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_dir / 'T6_F04_psa_ceac.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved PSA cost-effectiveness acceptability curves")


def plot_icer_frontier(icer_df, output_dir):
    """Scatter plot: cost-effectiveness plane with ICER frontier."""
    fig, ax = plt.subplots(figsize=(12, 8))

    frontier = icer_df[icer_df['on_frontier'] == True]
    off = icer_df[icer_df['on_frontier'] == False]

    # Plot all scenarios
    if len(off) > 0:
        ax.scatter(off['DALYs_averted'], off['total_cost_usd'] / 1e6,
                   c='gray', s=100, marker='x', zorder=5, label='Dominated')

    ax.scatter(frontier['DALYs_averted'], frontier['total_cost_usd'] / 1e6,
               c='#d73027', s=150, zorder=10, label='Frontier')

    # Draw frontier line
    frontier_sorted = frontier.sort_values('DALYs_averted')
    # Include origin
    x_vals = np.concatenate([[0], frontier_sorted['DALYs_averted'].values])
    y_vals = np.concatenate([[0], frontier_sorted['total_cost_usd'].values / 1e6])
    ax.plot(x_vals, y_vals, '--', color='#d73027', lw=1.5, alpha=0.7)

    # Annotate
    for _, row in icer_df.iterrows():
        short = row['scenario'].split('_', 1)[1] if '_' in row['scenario'] else row['scenario']
        ax.annotate(short, (row['DALYs_averted'], row['total_cost_usd'] / 1e6),
                    textcoords='offset points', xytext=(8, 5), fontsize=8)

    ax.set_xlabel('DALYs Averted')
    ax.set_ylabel('Total Cost (million USD)')
    ax.set_title('Cost-Effectiveness Plane with ICER Frontier')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'T6_F05_icer_frontier.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved ICER frontier plot")


# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("T6: COUNTERFACTUAL INTERVENTION SCENARIOS")
    print("=" * 65)

    df = load_data()
    print(f"Loaded {len(df)} samples")

    # ─── Run all scenarios ─────────────────────────────────────────────────
    print("\n--- SCENARIO RESULTS ---")
    scenario_df = run_all_scenarios(df)
    scenario_df.to_csv(TABLES_DIR / 'T6_scenario_results.csv', index=False)

    print(f"\n  {'Scenario':25s} {'HI>1%':>7s} {'DALYs':>12s} {'Averted':>12s} {'%Red':>6s} "
          f"{'Cost($M)':>10s} {'$/DALY':>10s} {'CE?':>5s}")
    print("  " + "-" * 95)
    for _, row in scenario_df.iterrows():
        ce_str = "Y" if row['very_cost_effective'] else "N"
        cost_m = row['total_cost_usd'] / 1e6
        ce_val = row['cost_per_daly_averted']
        ce_display = f"{ce_val:>10,.0f}" if ce_val < 1e9 else "      N/A"
        print(f"  {row['scenario']:25s} {row['HI_exceed_pct']:6.1f}% {row['DALYs']:>12,.0f} "
              f"{row['DALYs_averted']:>12,.0f} {row['pct_reduction']:5.1f}% "
              f"{cost_m:>9.0f} {ce_display} {ce_str:>5s}")

    # ─── ICER analysis ────────────────────────────────────────────────────
    print("\n--- INCREMENTAL COST-EFFECTIVENESS RATIO (ICER) ---")
    icer_df = compute_icer(scenario_df)
    icer_df.to_csv(TABLES_DIR / 'T6_icer_analysis.csv', index=False)

    frontier = icer_df[icer_df['on_frontier'] == True]
    print(f"  Efficiency frontier ({len(frontier)} scenarios):")
    for _, row in frontier.iterrows():
        icer_str = f"${row['ICER']:,.0f}" if not np.isinf(row['ICER']) else "N/A"
        print(f"    {row['scenario']:25s}: ICER = {icer_str}/DALY  "
              f"(incr cost ${row['incremental_cost']:,.0f}, incr DALYs {row['incremental_dalys']:,.0f})")

    dominated = icer_df[icer_df['on_frontier'] == False]
    if len(dominated) > 0:
        print(f"  Dominated scenarios: {', '.join(dominated['scenario'].tolist())}")

    # ─── Zone equity analysis ──────────────────────────────────────────────
    print("\n--- ZONE EQUITY ANALYSIS ---")
    equity_df = zone_equity_analysis(df)
    equity_df.to_csv(TABLES_DIR / 'T6_zone_equity.csv', index=False)

    # Which zones benefit most from S3 (multi-treatment)?
    s3_equity = equity_df[equity_df['scenario'] == 'S3_multi_treatment']
    print(f"\n  S3 (multi-treatment) DALY reduction by zone:")
    for _, row in s3_equity.sort_values('pct_reduction', ascending=False).iterrows():
        print(f"    {row['phys_zone']:30s}: {row['pct_reduction']:5.1f}% "
              f"({row['dalys_averted']:>12,.0f} DALYs averted)")

    # ─── Probabilistic Sensitivity Analysis ────────────────────────────────
    print(f"\n--- PROBABILISTIC SENSITIVITY ANALYSIS ({N_PSA} iterations) ---")
    psa_iter_df, psa_summary_df = run_psa(df)
    psa_iter_df.to_csv(TABLES_DIR / 'T6_psa_iterations.csv', index=False)
    psa_summary_df.to_csv(TABLES_DIR / 'T6_psa_summary.csv', index=False)

    print(f"\n  {'Scenario':25s} {'DALYs averted':>14s} {'95% CI':>25s} {'$/DALY median':>14s} {'95% CI':>25s}")
    print("  " + "-" * 110)
    for _, row in psa_summary_df.iterrows():
        averted_ci = f"[{row['dalys_averted_p2.5']:,.0f}, {row['dalys_averted_p97.5']:,.0f}]"
        ce_ci = f"[{row['cost_per_daly_p2.5']:,.0f}, {row['cost_per_daly_p97.5']:,.0f}]" \
                if not np.isnan(row['cost_per_daly_p2.5']) else "N/A"
        print(f"  {row['scenario']:25s} {row['dalys_averted_median']:>14,.0f} {averted_ci:>25s} "
              f"{row['cost_per_daly_median']:>14,.0f} {ce_ci:>25s}")

    # ─── Climate-amplified intervention analysis ──────────────────────────
    print("\n--- CLIMATE-AMPLIFIED INTERVENTION ANALYSIS (SSP5-8.5, 2050) ---")
    proj_2050 = load_future_concentrations()

    if proj_2050 is not None:
        climate_df, future_daly_base = run_climate_amplified_scenarios(df, proj_2050)
        climate_df.to_csv(TABLES_DIR / 'T6_climate_amplified_scenarios.csv', index=False)

        # Compare current vs future baseline DALYs
        current_base = scenario_df[scenario_df['scenario'] == 'S0_baseline']['DALYs'].values[0]
        print(f"\n  Baseline DALYs: current = {current_base:,.0f} | SSP5-8.5 2050 = {future_daly_base:,.0f} "
              f"(+{(future_daly_base - current_base) / current_base * 100:.1f}%)")

        print(f"\n  {'Scenario':25s} {'Current DALYs averted':>22s} {'Future DALYs averted':>22s} {'Change':>10s}")
        print("  " + "-" * 85)
        for _, crow in climate_df.iterrows():
            if crow['scenario'] == 'S0_baseline':
                continue
            current_averted = scenario_df[scenario_df['scenario'] == crow['scenario']]['DALYs_averted'].values
            current_averted = current_averted[0] if len(current_averted) > 0 else 0
            future_averted = crow['DALYs_averted']
            change_pct = ((future_averted - current_averted) / current_averted * 100) \
                         if current_averted > 0 else np.nan
            change_str = f"{change_pct:+.0f}%" if not np.isnan(change_pct) else "N/A"
            print(f"  {crow['scenario']:25s} {current_averted:>22,.0f} {future_averted:>22,.0f} {change_str:>10s}")
    else:
        print("  Skipped (T1_projected_concentrations_2050.csv not found)")

    # ─── Key findings ──────────────────────────────────────────────────────
    print("\n--- KEY FINDINGS ---")
    best_ce = scenario_df[scenario_df['scenario'] != 'S0_baseline'].nsmallest(1, 'cost_per_daly_averted')
    if len(best_ce) > 0:
        b = best_ce.iloc[0]
        print(f"  Most cost-effective: {b['scenario']} (${b['cost_per_daly_averted']:,.0f}/DALY)")

    best_health = scenario_df[scenario_df['scenario'] != 'S0_baseline'].nlargest(1, 'DALYs_averted')
    if len(best_health) > 0:
        b = best_health.iloc[0]
        print(f"  Most health benefit: {b['scenario']} ({b['DALYs_averted']:,.0f} DALYs averted, "
              f"{b['pct_reduction']:.0f}% reduction)")

    # Multi-contaminant advantage
    s2_averted = scenario_df[scenario_df['scenario'] == 'S2_po4_removal']['DALYs_averted'].values[0]
    s3_averted = scenario_df[scenario_df['scenario'] == 'S3_multi_treatment']['DALYs_averted'].values[0]
    print(f"\n  Multi-contaminant treatment advantage:")
    print(f"    S2 (As-focused): {s2_averted:,.0f} DALYs averted")
    print(f"    S3 (multi-treatment): {s3_averted:,.0f} DALYs averted")
    if s2_averted > 0:
        print(f"    Gain from multi-contaminant approach: {(s3_averted-s2_averted)/s2_averted*100:.0f}%")

    # ─── Figures ───────────────────────────────────────────────────────────
    print("\n--- GENERATING FIGURES ---")
    plot_scenario_comparison(scenario_df, FIGURES_DIR)
    plot_equity_heatmap(equity_df, FIGURES_DIR)
    plot_hi_reduction_boxplot(df, FIGURES_DIR)
    plot_psa_ceac(psa_iter_df, FIGURES_DIR)
    plot_icer_frontier(icer_df, FIGURES_DIR)

    print(f"\nAll saved to {TABLES_DIR} and {FIGURES_DIR}")
    print("=" * 65)
    print("T6 COUNTERFACTUAL INTERVENTIONS COMPLETE")
    print("=" * 65)


if __name__ == '__main__':
    main()
