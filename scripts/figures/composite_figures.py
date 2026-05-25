"""
Create composite 2×3 (or 2×4 for T2) panel figures for each test.
Reads existing individual PNGs and arranges them into publication-quality composites.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import string

# ─── Directories ─────────────────────────────────────────────────────────────
FIG_DIR = Path('output/figures')
OUT_DIR = FIG_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300

# ─── Figure groupings per test ───────────────────────────────────────────────
TESTS = {
    'T1': {
        'title': 'Test 1: Climate-Driven Contaminant Mobilisation',
        'files': [
            ('T1_F01_seasonal_sensitivity.png', 'Seasonal sensitivity heatmap'),
            ('T1_F02_projection_timeline.png', 'Climate projection timeline'),
            ('T1_F03_zone_vulnerability.png', 'Zone vulnerability'),
            ('T1_F04_threshold_crossings_2050.png', 'Threshold crossings by 2050'),
            ('T1_F05_contaminant_change_bar.png', 'Projected contaminant change (2050)'),
            ('T1_F06_spatial_vulnerability.png', 'Spatial climate vulnerability'),
        ],
        'grid': (2, 3),
    },
    'T2': {
        'title': 'Test 2: Multi-Contaminant Health Risk',
        'files': [
            ('T2_F01_HI_heatmap_season.png', 'HI heatmap by season'),
            ('T2_F02_single_vs_multi.png', 'Single vs. multi-contaminant'),
            ('T2_F03_contaminant_contributions.png', 'Contaminant contributions'),
            ('T2_F04_age_group_comparison.png', 'Age-group comparison'),
            ('T2_F05F06_merged.png', 'DALYs by zone & multi vs. As-only'),
            ('T2_spatial_risk_map.png', 'Spatial risk distribution'),
        ],
        'grid': (2, 3),
    },
    'T3': {
        'title': 'Test 3: Copula Joint Exceedance Risk',
        'files': [
            ('T3_F01_copula_pairwise_scatter.png', 'Copula pairwise scatter'),
            ('T3_F02_joint_exceedance_heatmap.png', 'Joint exceedance heatmap'),
            ('T3_F03_copula_family_comparison.png', 'Copula family comparison'),
            ('T3_F04_exceedance_count_dist.png', 'Exceedance count distribution'),
            ('T3_F05_kendall_tau_matrix.png', 'Kendall τ matrix'),
            ('T3_F06_tail_dependence.png', 'Tail dependence'),
        ],
        'grid': (2, 3),
    },
    'T4': {
        'title': 'Test 4: Tipping Points & Cascade Dynamics',
        'files': [
            ('T4_F01_gmm_densities.png', 'GMM density plots'),
            ('T4_F02_phase_diagrams.png', 'Phase diagrams'),
            ('T4_F03_cascade_heatmap.png', 'Cascade heatmap'),
            ('T4_F04_cusp_surface.png', 'Cusp catastrophe surface'),
            ('T4_F05_seasonal_shift.png', 'Seasonal bistability shift'),
            ('T4_F06_spatial_tipping.png', 'Spatial tipping-point states'),
        ],
        'grid': (2, 3),
    },
    'T5': {
        'title': 'Test 5: Bayesian Uncertainty Propagation',
        'files': [
            ('T5_F01_uncertainty_fan.png', 'Uncertainty fan chart'),
            ('T5_F02_hi_posterior_2050.png', 'HI posterior at 2050'),
            ('T5_F03_sensitivity_tornado.png', 'Sensitivity tornado'),
            ('T5_F04_zone_uncertainty.png', 'Zone-level uncertainty'),
            ('T5_F05_ssp_overlay.png', 'SSP scenario comparison'),
            ('T5_F06_depth_uncertainty.png', 'Depth-stratified HI'),
        ],
        'grid': (2, 3),
    },
    'T6': {
        'title': 'Test 6: Counterfactual Interventions',
        'files': [
            ('T6_F01_scenario_comparison.png', 'Scenario comparison'),
            ('T6_F02_equity_heatmap.png', 'Equity heatmap'),
            ('T6_F03_hi_reduction_boxplot.png', 'HI reduction boxplots'),
            ('T6_F04_psa_ceac.png', 'PSA & CEAC'),
            ('T6_F05_icer_frontier.png', 'ICER frontier'),
            ('T6_F06_spatial_intervention.png', 'Spatial intervention benefit'),
        ],
        'grid': (2, 3),
    },
}


def create_composite(test_key, test_info):
    """Create a single composite panel figure for one test."""
    nrows, ncols = test_info['grid']
    files = test_info['files']
    n_panels = len(files)

    # Figure sizing: ~3.5 in per column, ~3 in per row
    fig_w = ncols * 3.5
    fig_h = nrows * 3.2 + 0.35  # extra for suptitle

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h),
                             constrained_layout=True)
    axes_flat = axes.flatten()

    for i, ax in enumerate(axes_flat):
        if i < n_panels:
            fname, subtitle = files[i]
            fpath = FIG_DIR / fname
            if not fpath.exists():
                ax.text(0.5, 0.5, f'MISSING:\n{fname}', ha='center', va='center',
                        fontsize=9, color='red', transform=ax.transAxes)
                ax.set_facecolor('#f8f8f8')
            else:
                img = mpimg.imread(str(fpath))
                ax.imshow(img, aspect='auto')

            # Panel label (a), (b), ...
            label = f'({string.ascii_lowercase[i]})'
            ax.set_title(f'{label} {subtitle}', fontsize=9, fontweight='bold',
                         loc='left', pad=4)
        else:
            ax.set_visible(False)

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    out_path = OUT_DIR / f'{test_key}_composite.png'
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  ✓ {out_path.name}  ({nrows}×{ncols}, {n_panels} panels)')
    return out_path


def main():
    print('=' * 60)
    print('CREATING COMPOSITE PANEL FIGURES')
    print('=' * 60)

    for key in ['T1', 'T2', 'T3', 'T4', 'T5', 'T6']:
        info = TESTS[key]
        print(f'\n{info["title"]}')
        create_composite(key, info)

    print('\n' + '=' * 60)
    print(f'ALL COMPOSITES SAVED → {OUT_DIR}')
    print('=' * 60)


if __name__ == '__main__':
    main()
