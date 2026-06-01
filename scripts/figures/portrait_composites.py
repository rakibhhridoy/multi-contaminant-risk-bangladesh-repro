"""
Build 3x2 PORTRAIT composites for both the main figures and the supplementary
figures (replaces the wide 1x3 main splits and the wide 2x3 SI composites).

Main figures: the three original 6-panel composites are re-merged into ONE
3x2 portrait figure each (tipping / health / climate), so Figs 2-7 collapse to
Figs 2,3,4. Panels A-F top-to-bottom, left-to-right. SI S1/S2/S3 likewise go
3x2 portrait, numbering unchanged.

Panels are letter-labelled (A)-(F); full descriptions live in the LaTeX
captions. Reads the (already big-font) per-panel PNGs from output/figures and
writes composites to the submission and source folders. Run from project root.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
import string

FIG_DIR = Path('output/figures')
SUB = Path('Draft/STOTENSubmission/submission')
SRC = Path('Draft/STOTENSubmission/source')
DPI = 300
PANEL_H = 1600        # px: every panel scaled to this common height (undistorted)
LABEL_FONTSIZE = 18   # slightly smaller than before; lowercase (a),(b),...

# out_name : (panels in a..f order, (nrows, ncols), [output dirs])
SPECS = {
    # ── MAIN (spatial map -> Fig 5; one ancillary panel each -> SI; now 2x2 a-d)
    'fig2_tipping.png': ([
        'T4_F01_gmm_densities.png',     # a
        'T4_F02_phase_diagrams.png',    # b
        'T4_F03_cascade_heatmap.png',   # c
        'T4_F04_cusp_surface.png',      # d  (T4_F05 seasonal shift -> SI S10)
    ], (2, 2), [SUB, SRC]),
    'fig3_health.png': ([
        'T2_F01_HI_heatmap_season.png',        # a
        'T2_F02_single_vs_multi.png',          # b
        'T2_F05F06_merged.png',                # c  DALY by zone (swapped with d)
        'T2_F03_contaminant_contributions.png',# d  contributions (T2_F04 age-sex -> SI S11)
    ], (2, 2), [SUB, SRC]),
    'fig4_climate.png': ([
        'T1_F01_seasonal_sensitivity.png',     # a
        'T1_F02_projection_timeline.png',      # b
        'T1_F04_threshold_crossings_2050.png', # c  WHO threshold crossings (swapped with d)
        'T1_F05_contaminant_change_bar.png',   # d  divergent responses (T1_F03 zone-vuln -> SI S12)
    ], (2, 2), [SUB, SRC]),
    # ── NEW: the three Bangladesh IDW-surface maps, one figure (Fig 5) ─────
    'fig5_spatial_maps.png': ([
        'T4_F06_spatial_tipping.png',       # a  high-As mode fraction
        'T2_spatial_risk_map.png',          # b  hazard index
        'T1_F06_spatial_vulnerability.png', # c  climate vulnerability
    ], (1, 3), [SUB, SRC]),
    # ── SUPPLEMENTARY (numbering unchanged; embedded via SI PDF) ───────────
    'figS1_T3.png': ([
        'T3_F01_copula_pairwise_scatter.png',
        'T3_F02_joint_exceedance_heatmap.png',
        'T3_F03_copula_family_comparison.png',
        'T3_F04_exceedance_count_dist.png',
        'T3_F05_kendall_tau_matrix.png',
        'T3_F06_tail_dependence.png',
    ], (3, 2), [SRC]),
    'figS2_T5.png': ([
        'T5_F01_uncertainty_fan.png',
        'T5_F02_hi_posterior_2050.png',
        'T5_F03_sensitivity_tornado.png',
        'T5_F04_zone_uncertainty.png',
        'T5_F05_ssp_overlay.png',
        'T5_F06_depth_uncertainty.png',
    ], (3, 2), [SRC]),
    'figS3_T6.png': ([
        'T6_F01_scenario_comparison.png',
        'T6_F02_equity_heatmap.png',
        'T6_F03_hi_reduction_boxplot.png',
        'T6_F04_psa_ceac.png',
        'T6_F05_icer_frontier.png',
        'T6_F06_spatial_intervention.png',
    ], (3, 2), [SRC]),
}


def _scaled(path):
    """Load a panel and scale to the common PANEL_H, preserving aspect."""
    im = Image.open(FIG_DIR / path).convert('RGB')
    w, h = im.size
    return im.resize((max(1, round(w * PANEL_H / h)), PANEL_H), Image.LANCZOS)


def _pad_right(im, target_w):
    """Left-align the panel on a white canvas of width target_w (no distortion)."""
    canvas = Image.new('RGB', (target_w, PANEL_H), 'white')
    canvas.paste(im, (0, 0))
    return np.asarray(canvas)


def build(out_name, panels, layout, out_dirs):
    nrows, ncols = layout
    scaled = [_scaled(p) for p in panels]
    # uniform height already; pad each column to its own max width so every
    # panel shares an identical canvas -> imshow boxes match -> (a)/(b) labels
    # align across the figure, with no stretching of the plots.
    col_w = [max([scaled[i].size[0] for i in range(c, len(scaled), ncols)] or [PANEL_H])
             for c in range(ncols)]
    imgs = [_pad_right(scaled[i], col_w[i % ncols]) for i in range(len(scaled))]

    total_w, total_h = sum(col_w), PANEL_H * nrows
    fig_w = 13.0
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(fig_w, fig_w * total_h / total_w),
                             gridspec_kw={'width_ratios': col_w},
                             constrained_layout=True, squeeze=False)
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i < len(imgs):
            ax.imshow(imgs[i])
            ax.set_title(f'({string.ascii_lowercase[i]})', fontsize=LABEL_FONTSIZE,
                         fontweight='bold', loc='left', pad=4)
        else:
            ax.set_visible(False)   # trailing empty cell (e.g. 5 panels in 3x2)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    for d in out_dirs:
        fig.savefig(d / out_name, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  OK  {out_name}  ({nrows}x{ncols}, {len(panels)}p)  -> {", ".join(d.name for d in out_dirs)}')


if __name__ == '__main__':
    print('Building portrait composites ...')
    for name, (panels, layout, dirs) in SPECS.items():
        build(name, panels, layout, dirs)
    print('done.')
