"""
T5: Monte Carlo End-to-End Uncertainty Propagation
Chains: Climate uncertainty -> Contaminant change -> Health risk uncertainty
Produces credible intervals for DALYs and HI under each SSP scenario.
Uses parametric bootstrap / Monte Carlo sampling (no PyMC dependency).

NOTE: The filename is kept as bayesian_propagation.py for backward compatibility,
but the method is pure Monte Carlo propagation, not Bayesian inference.
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
    HEALTH_CONTAMINANTS, REFERENCE_DOSES, CANCER_SLOPE_FACTORS,
    EXPOSURE_PARAMS, AT_CARCINOGENIC, CR_VI_FRACTION,
    SSP_SCENARIOS, PROJECTION_YEARS,
    TABLES_DIR, FIGURES_DIR, RANDOM_STATE,
    assign_zones
)

np.random.seed(RANDOM_STATE)

N_MC = 5000  # Monte Carlo iterations

# Override config.EXPOSURE_PARAMS at module load so T5 baseline HI is computed
# with the same ingestion-rate / body-weight assumptions as the corrected DALY
# pipeline (adult-male: IR=2.5 L/day, BW=60 kg). Without this override T5 would
# silently use the stale config defaults (IR=3.5 / BW=60) and inflate the
# baseline HI by 40% relative to the headline numbers reported in the paper.
EXPOSURE_PARAMS = dict(EXPOSURE_PARAMS)  # local copy, do not mutate import
EXPOSURE_PARAMS['adult_male'] = {
    **EXPOSURE_PARAMS['adult_male'],
    'ir_L_day': 2.5,
    'bw_kg': 60,
}
EXPOSURE_PARAMS['adult_female'] = {
    **EXPOSURE_PARAMS['adult_female'],
    'ir_L_day': 2.0,
    'bw_kg': 55,
}
EXPOSURE_PARAMS['child_6_17'] = {
    **EXPOSURE_PARAMS['child_6_17'],
    'ir_L_day': 1.5,
    'bw_kg': 35,
}
EXPOSURE_PARAMS['child_0_5'] = {
    **EXPOSURE_PARAMS['child_0_5'],
    'ir_L_day': 1.0,
    'bw_kg': 15,
}

# ─── UNCERTAINTY SOURCES ───────────────────────────────────────────────────────

# 1. Climate projection uncertainty (GCM spread)
#    Expressed as coefficient of variation of precipitation change.
#    These synthetic defaults are OVERRIDDEN at runtime by the empirical
#    inter-model CV computed from the CMIP6 ensemble (T1_ensemble_deltaP_cv.csv)
#    when available — see _load_empirical_climate_cv(). This replaces a guessed
#    CV with one measured from a 5-6 model CMIP6 ensemble for Bangladesh.
CLIMATE_CV = {
    'SSP2-4.5': {2030: 0.40, 2040: 0.35, 2050: 0.30, 2060: 0.30},
    'SSP5-8.5': {2030: 0.35, 2040: 0.30, 2050: 0.25, 2060: 0.25},
}


def _load_empirical_climate_cv():
    """Override CLIMATE_CV with empirical inter-model spread if the CMIP6
    ensemble output exists. Applies the single 2050 ensemble CV across all
    projection years (the ensemble was evaluated for the 2041-2060 window).
    """
    from config import TABLES_DIR
    cv_path = TABLES_DIR / 'T1_ensemble_deltaP_cv.csv'
    if not cv_path.exists():
        print("  [climate CV] ensemble file absent — using synthetic CLIMATE_CV")
        return
    cv_df = pd.read_csv(cv_path)
    ssp_map = {'ssp245': 'SSP2-4.5', 'ssp585': 'SSP5-8.5'}
    n_applied = 0
    for _, r in cv_df.iterrows():
        label = ssp_map.get(str(r['ssp']))
        if label is None or label not in CLIMATE_CV:
            continue
        emp_cv = float(r['cv'])
        if np.isfinite(emp_cv) and emp_cv > 0:
            for yr in CLIMATE_CV[label]:
                CLIMATE_CV[label][yr] = emp_cv
            n_applied += 1
            print(f"  [climate CV] {label}: empirical inter-model CV = {emp_cv:.3f} "
                  f"({int(r['n'])} models)")
    if n_applied == 0:
        print("  [climate CV] no usable empirical CV — kept synthetic defaults")

# 2. Transfer function uncertainty (from seasonal calibration)
TRANSFER_CV = 0.50  # 50% uncertainty in sensitivity parameter

# 3. Exposure parameter uncertainty
EXPOSURE_CV = {
    'ir': 0.20,   # water intake +/-20%
    'bw': 0.10,   # body weight +/-10%
    'ef': 0.05,   # exposure frequency +/-5%
}

# 4. RfD/CSF uncertainty (published safety factors)
TOXICITY_CV = {
    'rfd': 0.30,     # RfD uncertainty +/-30% (within safety factor range)
    'csf': 0.25,     # CSF uncertainty +/-25%
}

# Reference precipitation changes (median estimates from T1)
PRECIP_CHANGE = {
    'SSP2-4.5': {2030: 5.0, 2040: 8.0, 2050: 12.0, 2060: 15.0},
    'SSP5-8.5': {2030: 7.0, 2040: 12.0, 2050: 18.0, 2060: 25.0},
}

# Temperature changes
TEMP_CHANGE = {
    'SSP2-4.5': {2030: 0.6, 2040: 0.9, 2050: 1.3, 2060: 1.6},
    'SSP5-8.5': {2030: 0.8, 2040: 1.4, 2050: 2.1, 2060: 2.9},
}

# Seasonal sensitivity per 1% precipitation change (from T1 transfer)
SENSITIVITY_MEDIAN = {'As': 0.84, 'Mn2+': -0.007, 'Fe2+': 0.001, 'Cr3+': -0.0003, 'NO3-': -0.002}

# Temperature sensitivity (fraction per degC)
TEMP_SENSITIVITY = {'As': 0.10, 'Mn2+': 0.08, 'Fe2+': 0.06, 'Cr3+': 0.03, 'NO3-': 0.05}

WHO_GUIDELINES = {'As': 10.0, 'Mn2+': 0.4, 'Fe2+': 0.3, 'Cr3+': 0.05, 'NO3-': 50.0}

CONTAMINANTS = ['As', 'Mn2+', 'Fe2+', 'Cr3+', 'NO3-']


# ─── DATA LOADING ──────────────────────────────────────────────────────────────

def load_data():
    """Load dataset with zone assignments using shared config function."""
    df = pd.read_csv(DATA_FILE)
    df = assign_zones(df)
    return df


# ─── MONTE CARLO PROPAGATION ──────────────────────────────────────────────────

def sample_lognormal(median, cv, n):
    """Sample from lognormal given median and CV."""
    if median == 0 or cv == 0:
        return np.full(n, median)
    sigma = np.sqrt(np.log(1 + cv**2))
    mu = np.log(abs(median)) - sigma**2 / 2
    samples = np.random.lognormal(mu, sigma, n)
    if median < 0:
        samples = -samples
    return samples


def propagate_single_scenario(df, ssp, year, zone=None, n_mc=N_MC):
    """
    Monte Carlo propagation for one SSP x year combination, optionally for a
    single physiographic zone.

    Parameters
    ----------
    df : DataFrame
        Full dataset (already zone-assigned).
    ssp : str
        SSP scenario label.
    year : int
        Projection year.
    zone : str or None
        If provided, restrict baseline to this physiographic zone.
        If None, use national medians (backward-compatible).
    n_mc : int
        Number of Monte Carlo iterations.

    Returns
    -------
    dict with hi_dist, cr_dist, hi_exceed_pct, cr_exceed_pct arrays.
    """
    precip_median = PRECIP_CHANGE[ssp][year]
    precip_cv = CLIMATE_CV[ssp][year]
    temp_median = TEMP_CHANGE[ssp][year]

    # Sample climate uncertainty
    precip_samples = sample_lognormal(precip_median, precip_cv, n_mc)
    temp_samples = np.random.normal(temp_median, temp_median * 0.15, n_mc)  # 15% CV

    # Baseline: dry season, optionally filtered by zone
    dry = df[df['Season'] == 'Dry']
    if zone is not None:
        dry = dry[dry['phys_zone'] == zone]

    if len(dry) < 5:
        # Fallback to national if zone has too few samples
        dry = df[df['Season'] == 'Dry']

    params = EXPOSURE_PARAMS['adult_male']
    ir_base, bw_base = params['ir_L_day'], params['bw_kg']

    hi_distributions = np.empty(n_mc)
    cr_distributions = np.empty(n_mc)
    exceed_hi = np.empty(n_mc)
    exceed_cr = np.empty(n_mc)

    for i in range(n_mc):
        dp = precip_samples[i]
        dt = temp_samples[i]

        # Sample transfer function uncertainty
        sens_samples = {}
        for cont in CONTAMINANTS:
            sens_samples[cont] = np.random.normal(
                SENSITIVITY_MEDIAN[cont],
                abs(SENSITIVITY_MEDIAN[cont]) * TRANSFER_CV
            )

        # Sample exposure parameters
        ir = max(0.5, np.random.normal(ir_base, ir_base * EXPOSURE_CV['ir']))
        bw = max(30, np.random.normal(bw_base, bw_base * EXPOSURE_CV['bw']))

        # Sample toxicity parameters
        rfd_factor = np.random.lognormal(0, TOXICITY_CV['rfd'])
        csf_factor = np.random.lognormal(0, TOXICITY_CV['csf'])

        # Project concentrations
        hi_total = 0.0
        cr_total = 0.0

        for cont in CONTAMINANTS:
            baseline_med = dry[cont].median()

            # Project
            precip_delta = sens_samples[cont] * dp
            temp_delta = baseline_med * TEMP_SENSITIVITY.get(cont, 0.05) * dt
            projected = max(0, baseline_med + precip_delta + temp_delta)

            # Convert As from ug/L to mg/L for risk calculation
            conc_mg = projected / 1000.0 if cont == 'As' else projected

            # CDI
            cdi = (conc_mg * ir * 365 * 30) / (bw * 30 * 365)

            # HI contribution
            if cont in REFERENCE_DOSES:
                rfd = REFERENCE_DOSES[cont] * rfd_factor
                hi_total += cdi / rfd

            # CR contribution
            if cont in CANCER_SLOPE_FACTORS:
                cdi_cancer = (conc_mg * ir * 365 * 30) / (bw * AT_CARCINOGENIC)
                csf = CANCER_SLOPE_FACTORS[cont] * csf_factor
                cr_total += cdi_cancer * csf

        hi_distributions[i] = hi_total
        cr_distributions[i] = cr_total
        exceed_hi[i] = float(hi_total > 1)
        exceed_cr[i] = float(cr_total > 1e-4)

    return {
        'hi_dist': hi_distributions,
        'cr_dist': cr_distributions,
        'hi_exceed_pct': np.mean(exceed_hi) * 100,
        'cr_exceed_pct': np.mean(exceed_cr) * 100,
    }


def run_full_propagation(df):
    """Run MC propagation per zone for all SSP x year combinations, then aggregate."""
    zone_results = []
    national_results = []

    zones = sorted(df['phys_zone'].unique())

    for ssp in SSP_SCENARIOS:
        for year in PROJECTION_YEARS:
            print(f"  Running {ssp} {year} ({N_MC} MC iterations per zone)...")

            # Per-zone propagation
            all_hi = []
            all_cr = []
            for pz in zones:
                out = propagate_single_scenario(df, ssp, year, zone=pz, n_mc=N_MC)
                zone_results.append({
                    'ssp': ssp, 'year': year, 'phys_zone': pz,
                    'HI_median': np.median(out['hi_dist']),
                    'HI_p2.5': np.percentile(out['hi_dist'], 2.5),
                    'HI_p25': np.percentile(out['hi_dist'], 25),
                    'HI_p75': np.percentile(out['hi_dist'], 75),
                    'HI_p97.5': np.percentile(out['hi_dist'], 97.5),
                    'HI_exceed_pct': out['hi_exceed_pct'],
                    'CR_median': np.median(out['cr_dist']),
                    'CR_p2.5': np.percentile(out['cr_dist'], 2.5),
                    'CR_p97.5': np.percentile(out['cr_dist'], 97.5),
                    'CR_exceed_pct': out['cr_exceed_pct'],
                })
                all_hi.append(out['hi_dist'])
                all_cr.append(out['cr_dist'])

            # National aggregate: average across zones (equal weight per zone)
            hi_national = np.mean(all_hi, axis=0)
            cr_national = np.mean(all_cr, axis=0)

            national_results.append({
                'ssp': ssp, 'year': year,
                'HI_median': np.median(hi_national),
                'HI_p2.5': np.percentile(hi_national, 2.5),
                'HI_p25': np.percentile(hi_national, 25),
                'HI_p75': np.percentile(hi_national, 75),
                'HI_p97.5': np.percentile(hi_national, 97.5),
                'HI_exceed_pct': np.mean(hi_national > 1) * 100,
                'CR_median': np.median(cr_national),
                'CR_p2.5': np.percentile(cr_national, 2.5),
                'CR_p97.5': np.percentile(cr_national, 97.5),
                'CR_exceed_pct': np.mean(cr_national > 1e-4) * 100,
            })

    zone_df = pd.DataFrame(zone_results)
    national_df = pd.DataFrame(national_results)
    return national_df, zone_df


def sensitivity_analysis(df):
    """Variance decomposition: which uncertainty source dominates?

    Fixes each source in turn and measures how much total HI variance
    decreases. This is a one-at-a-time (OAT) variance decomposition.
    """
    ssp, year = 'SSP5-8.5', 2050  # reference scenario
    base_result = propagate_single_scenario(df, ssp, year, zone=None, n_mc=2000)
    base_var = np.var(base_result['hi_dist'])

    sources = {}

    # 1. Fix climate
    original_cv = CLIMATE_CV[ssp][year]
    CLIMATE_CV[ssp][year] = 0.01
    fixed_climate = propagate_single_scenario(df, ssp, year, zone=None, n_mc=2000)
    sources['Climate projection'] = 1 - np.var(fixed_climate['hi_dist']) / base_var
    CLIMATE_CV[ssp][year] = original_cv

    # 2. Fix transfer function
    original_tf = globals()['TRANSFER_CV']
    globals()['TRANSFER_CV'] = 0.01
    fixed_tf = propagate_single_scenario(df, ssp, year, zone=None, n_mc=2000)
    sources['Transfer function'] = 1 - np.var(fixed_tf['hi_dist']) / base_var
    globals()['TRANSFER_CV'] = original_tf

    # 3. Fix exposure
    original_exp = dict(EXPOSURE_CV)
    for k in EXPOSURE_CV:
        EXPOSURE_CV[k] = 0.01
    fixed_exp = propagate_single_scenario(df, ssp, year, zone=None, n_mc=2000)
    sources['Exposure parameters'] = 1 - np.var(fixed_exp['hi_dist']) / base_var
    for k in original_exp:
        EXPOSURE_CV[k] = original_exp[k]

    # 4. Fix toxicity
    original_tox = dict(TOXICITY_CV)
    for k in TOXICITY_CV:
        TOXICITY_CV[k] = 0.01
    fixed_tox = propagate_single_scenario(df, ssp, year, zone=None, n_mc=2000)
    sources['Toxicity (RfD/CSF)'] = 1 - np.var(fixed_tox['hi_dist']) / base_var
    for k in original_tox:
        TOXICITY_CV[k] = original_tox[k]

    return sources


def sobol_sensitivity(df):
    """Sobol first-order sensitivity indices via Saltelli sampling.

    Uses SALib if available; otherwise falls back to the OAT variance-
    decomposition in sensitivity_analysis() and prints a note.

    Returns dict  {source_name: S1_index}.
    """
    try:
        from SALib.sample import saltelli as saltelli_sampler
        from SALib.analyze import sobol as sobol_analyzer
    except ImportError:
        print("  SALib not available; falling back to OAT variance decomposition.")
        print("  (Install SALib for proper Sobol indices: pip install SALib)")
        return None  # caller will use OAT instead

    # Define the problem: 4 factors, each a multiplier on its CV
    problem = {
        'num_vars': 4,
        'names': ['climate_cv_mult', 'transfer_cv_mult', 'exposure_cv_mult', 'toxicity_cv_mult'],
        'bounds': [[0.0, 2.0]] * 4,  # multiplier on default CV (0 = fixed, 2 = doubled)
    }

    N_sobol = 256  # base sample size; total = N*(2D+2) = 2560
    param_values = saltelli_sampler.sample(problem, N_sobol, calc_second_order=False)

    # Evaluate model for each parameter set
    ssp, year = 'SSP5-8.5', 2050
    precip_median = PRECIP_CHANGE[ssp][year]
    temp_median = TEMP_CHANGE[ssp][year]
    dry = df[df['Season'] == 'Dry']
    params_exp = EXPOSURE_PARAMS['adult_male']
    ir_base, bw_base = params_exp['ir_L_day'], params_exp['bw_kg']

    Y = np.empty(len(param_values))

    for idx, pset in enumerate(param_values):
        clim_mult, tf_mult, exp_mult, tox_mult = pset

        # Run a small MC (200 iterations) per parameter set
        n_inner = 200
        clim_cv = CLIMATE_CV[ssp][year] * clim_mult
        tf_cv = TRANSFER_CV * tf_mult
        exp_ir_cv = EXPOSURE_CV['ir'] * exp_mult
        exp_bw_cv = EXPOSURE_CV['bw'] * exp_mult
        tox_rfd_cv = TOXICITY_CV['rfd'] * tox_mult
        tox_csf_cv = TOXICITY_CV['csf'] * tox_mult

        precip_samples = sample_lognormal(precip_median, max(clim_cv, 0.001), n_inner)
        temp_samples = np.random.normal(temp_median, temp_median * 0.15, n_inner)

        hi_vals = np.empty(n_inner)
        for i in range(n_inner):
            dp = precip_samples[i]
            dt = temp_samples[i]

            sens_samples = {}
            for cont in CONTAMINANTS:
                sens_samples[cont] = np.random.normal(
                    SENSITIVITY_MEDIAN[cont],
                    abs(SENSITIVITY_MEDIAN[cont]) * max(tf_cv, 0.001)
                )

            ir = max(0.5, np.random.normal(ir_base, ir_base * max(exp_ir_cv, 0.001)))
            bw = max(30, np.random.normal(bw_base, bw_base * max(exp_bw_cv, 0.001)))
            rfd_factor = np.random.lognormal(0, max(tox_rfd_cv, 0.001))

            hi_total = 0.0
            for cont in CONTAMINANTS:
                baseline_med = dry[cont].median()
                precip_delta = sens_samples[cont] * dp
                temp_delta = baseline_med * TEMP_SENSITIVITY.get(cont, 0.05) * dt
                projected = max(0, baseline_med + precip_delta + temp_delta)
                conc_mg = projected / 1000.0 if cont == 'As' else projected
                cdi = (conc_mg * ir * 365 * 30) / (bw * 30 * 365)
                if cont in REFERENCE_DOSES:
                    rfd = REFERENCE_DOSES[cont] * rfd_factor
                    hi_total += cdi / rfd

            hi_vals[i] = hi_total

        Y[idx] = np.mean(hi_vals)

    Si = sobol_analyzer.analyze(problem, Y, calc_second_order=False, print_to_console=False)

    sobol_indices = {}
    for name, s1 in zip(problem['names'], Si['S1']):
        label = name.replace('_cv_mult', '').replace('_', ' ').title()
        sobol_indices[label] = float(s1)

    return sobol_indices


# ─── FIGURES ────────────────────────────────────────────────────────────────────

def plot_uncertainty_fan(results_df, output_dir):
    """Fan chart: HI credible intervals over time by SSP."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for i, ssp in enumerate(SSP_SCENARIOS):
        ax = axes[i]
        ssp_data = results_df[results_df['ssp'] == ssp].sort_values('year')

        years = ssp_data['year'].values
        median = ssp_data['HI_median'].values
        p25 = ssp_data['HI_p25'].values
        p75 = ssp_data['HI_p75'].values
        p2_5 = ssp_data['HI_p2.5'].values
        p97_5 = ssp_data['HI_p97.5'].values

        color = '#4575b4' if '4.5' in ssp else '#d73027'

        ax.fill_between(years, p2_5, p97_5, alpha=0.15, color=color, label='95% CI')
        ax.fill_between(years, p25, p75, alpha=0.3, color=color, label='50% CI')
        ax.plot(years, median, 'o-', color=color, lw=2, label='Median')
        ax.axhline(1, color='red', ls='--', lw=1.5, label='HI = 1 threshold')
        ax.set_xlabel('Year')
        ax.set_ylabel('Projected Hazard Index')
        ax.set_title(f'{ssp} -- Projected HI with Uncertainty')
        ax.legend(loc='upper left')
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_dir / 'T5_F01_uncertainty_fan.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved uncertainty fan chart")


def plot_hi_posterior(results_df, output_dir):
    """Violin/density: HI distribution at 2050 for both SSPs."""
    fig, ax = plt.subplots(figsize=(10, 6))

    df = load_data()
    dists = {}
    for ssp in SSP_SCENARIOS:
        out = propagate_single_scenario(df, ssp, 2050, zone=None, n_mc=3000)
        dists[ssp] = out['hi_dist']

    data = []
    for ssp, dist in dists.items():
        for val in dist:
            data.append({'SSP': ssp, 'HI': val})
    plot_df = pd.DataFrame(data)

    sns.violinplot(data=plot_df, x='SSP', y='HI', ax=ax, inner='quartile',
                   palette=['#4575b4', '#d73027'], cut=0)
    ax.axhline(1, color='red', ls='--', lw=1.5, label='HI = 1 threshold')
    ax.set_ylabel('Projected Hazard Index (2050)')
    ax.set_title('Uncertainty in Projected Health Risk at 2050')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'T5_F02_hi_posterior_2050.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved HI distribution violin")


def plot_sensitivity_tornado(sources, sobol_indices, output_dir):
    """Tornado plot: variance contribution by uncertainty source.

    Shows OAT variance decomposition. If Sobol indices are available,
    annotates them alongside.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    sorted_sources = sorted(sources.items(), key=lambda x: abs(x[1]), reverse=True)
    names = [s[0] for s in sorted_sources]
    values = [max(0, s[1]) * 100 for s in sorted_sources]

    colors = ['#d73027', '#fc8d59', '#fee08b', '#91bfdb']
    bars = ax.barh(range(len(names)), values, color=colors[:len(names)])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel('Variance Reduction When Fixed (%)')
    ax.set_title('Sensitivity Analysis: Dominant Uncertainty Sources\n(SSP5-8.5, 2050)')
    ax.invert_yaxis()

    # Annotate Sobol indices if available
    if sobol_indices is not None:
        for i, name in enumerate(names):
            # Try to match Sobol key to OAT key
            for skey, sval in sobol_indices.items():
                if skey.lower() in name.lower() or name.lower() in skey.lower():
                    ax.annotate(f'S1={sval:.2f}', xy=(values[i] + 1, i),
                                fontsize=8, va='center', color='#333333')
                    break

    plt.tight_layout()
    plt.savefig(output_dir / 'T5_F03_sensitivity_tornado.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved sensitivity tornado")


def plot_zone_uncertainty(zone_df, output_dir):
    """Heatmap: zone-level HI at 2050 SSP5-8.5 with uncertainty."""
    fig, ax = plt.subplots(figsize=(12, 6))

    sub = zone_df[(zone_df['ssp'] == 'SSP5-8.5') & (zone_df['year'] == 2050)]
    if len(sub) == 0:
        return

    sub = sub.sort_values('HI_median', ascending=False)
    labels = [f"{r['phys_zone']}\n{r['HI_median']:.2f} [{r['HI_p2.5']:.2f}-{r['HI_p97.5']:.2f}]"
              for _, r in sub.iterrows()]

    colors = ['#d73027' if r['HI_median'] > 1 else '#4575b4' for _, r in sub.iterrows()]
    ax.barh(range(len(sub)), sub['HI_median'].values, xerr=[
        sub['HI_median'].values - sub['HI_p2.5'].values,
        sub['HI_p97.5'].values - sub['HI_median'].values
    ], color=colors, capsize=4)
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels([r['phys_zone'] for _, r in sub.iterrows()])
    ax.axvline(1, color='red', ls='--', lw=1.5, label='HI = 1')
    ax.set_xlabel('Hazard Index (median, 95% CI)')
    ax.set_title('Zone-Level Health Risk Uncertainty (SSP5-8.5, 2050)')
    ax.legend()
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_dir / 'T5_F04_zone_uncertainty.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved zone uncertainty plot")


# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("T5: MONTE CARLO END-TO-END UNCERTAINTY PROPAGATION")
    print("=" * 65)

    # Override synthetic climate CV with empirical CMIP6 inter-model spread
    _load_empirical_climate_cv()

    df = load_data()
    print(f"Loaded {len(df)} samples ({df['phys_zone'].nunique()} zones)")
    print(f"MC iterations per zone: {N_MC}")

    # ─── Baseline (current conditions) ─────────────────────────────────────
    print("\n--- BASELINE HEALTH RISK (current, no climate change) ---")
    dry = df[df['Season'] == 'Dry']
    params = EXPOSURE_PARAMS['adult_male']
    ir, bw = params['ir_L_day'], params['bw_kg']

    hi_baseline = 0
    for cont in CONTAMINANTS:
        baseline_med = dry[cont].median()
        conc_mg = baseline_med / 1000.0 if cont == 'As' else baseline_med
        cdi = (conc_mg * ir * 365 * 30) / (bw * 30 * 365)
        if cont in REFERENCE_DOSES:
            hi_baseline += cdi / REFERENCE_DOSES[cont]
    print(f"  Baseline HI (dry season median, adult male): {hi_baseline:.2f}")

    # ─── Full propagation (per-zone) ──────────────────────────────────────
    print("\n--- MONTE CARLO PROPAGATION (per physiographic zone) ---")
    national_df, zone_df = run_full_propagation(df)
    national_df.to_csv(TABLES_DIR / 'T5_mc_propagation_results.csv', index=False)
    zone_df.to_csv(TABLES_DIR / 'T5_mc_zone_results.csv', index=False)

    # Print national results
    print(f"\n  {'SSP':12s} {'Year':>5s} {'HI median':>10s} {'HI 95% CI':>20s} {'P(HI>1)':>8s} {'P(CR>1e-4)':>10s}")
    print("  " + "-" * 70)
    for _, row in national_df.iterrows():
        print(f"  {row['ssp']:12s} {row['year']:5.0f} {row['HI_median']:10.2f} "
              f"[{row['HI_p2.5']:.2f}, {row['HI_p97.5']:.2f}]{'':<4s} "
              f"{row['HI_exceed_pct']:7.1f}% {row['CR_exceed_pct']:9.1f}%")

    # Print zone-level summary for SSP5-8.5 2050
    print(f"\n  Zone-level HI at 2050 SSP5-8.5:")
    z_2050 = zone_df[(zone_df['ssp'] == 'SSP5-8.5') & (zone_df['year'] == 2050)]
    for _, row in z_2050.sort_values('HI_median', ascending=False).iterrows():
        print(f"    {row['phys_zone']:30s}: HI = {row['HI_median']:.2f} "
              f"[{row['HI_p2.5']:.2f}, {row['HI_p97.5']:.2f}]  P(HI>1) = {row['HI_exceed_pct']:.0f}%")

    # ─── OAT Sensitivity analysis ─────────────────────────────────────────
    print("\n--- SENSITIVITY ANALYSIS (OAT variance decomposition) ---")
    sources = sensitivity_analysis(df)
    for name, var_frac in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  {name:25s}: {max(0, var_frac)*100:.1f}% of total variance")

    # ─── Sobol sensitivity indices ────────────────────────────────────────
    print("\n--- SOBOL FIRST-ORDER SENSITIVITY INDICES ---")
    sobol_indices = sobol_sensitivity(df)
    if sobol_indices is not None:
        for name, s1 in sorted(sobol_indices.items(), key=lambda x: -x[1]):
            print(f"  {name:25s}: S1 = {s1:.3f}")
    else:
        print("  (Skipped -- SALib not installed; using OAT decomposition above)")

    # ─── Key findings ──────────────────────────────────────────────────────
    print("\n--- KEY FINDINGS ---")
    hi_2050_585 = national_df[(national_df['ssp'] == 'SSP5-8.5') & (national_df['year'] == 2050)]
    if len(hi_2050_585) > 0:
        row = hi_2050_585.iloc[0]
        print(f"  SSP5-8.5 2050: HI = {row['HI_median']:.2f} [{row['HI_p2.5']:.2f}, {row['HI_p97.5']:.2f}]")
        print(f"  P(HI > 1) = {row['HI_exceed_pct']:.0f}%")
        if row['HI_p2.5'] > 0:
            print(f"  Uncertainty range spans {row['HI_p97.5']/row['HI_p2.5']:.1f}x")

    dominant = max(sources, key=sources.get)
    print(f"  Dominant uncertainty source (OAT): {dominant} ({sources[dominant]*100:.0f}% of variance)")

    if sobol_indices is not None:
        dominant_sobol = max(sobol_indices, key=sobol_indices.get)
        print(f"  Dominant uncertainty source (Sobol S1): {dominant_sobol} (S1={sobol_indices[dominant_sobol]:.3f})")

    # ─── Figures ───────────────────────────────────────────────────────────
    print("\n--- GENERATING FIGURES ---")
    plot_uncertainty_fan(national_df, FIGURES_DIR)
    plot_hi_posterior(national_df, FIGURES_DIR)
    plot_sensitivity_tornado(sources, sobol_indices, FIGURES_DIR)
    plot_zone_uncertainty(zone_df, FIGURES_DIR)

    print(f"\nAll saved to {TABLES_DIR} and {FIGURES_DIR}")
    print("=" * 65)
    print("T5 MONTE CARLO UNCERTAINTY PROPAGATION COMPLETE")
    print("=" * 65)


if __name__ == '__main__':
    main()
