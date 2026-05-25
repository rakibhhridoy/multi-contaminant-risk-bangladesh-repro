"""
Graphical abstract for the STOTEN submission  [added 2026-05-24]
================================================================
Single landscape panel telling the paper's story in three stages:
  (1) a cheap phosphate test flags wells near the arsenic tipping threshold;
  (2) multi-contaminant assessment doubles the disease burden vs arsenic-only;
  (3) climate pushes new zones over the WHO limit, but a cheap intervention averts
      most of the burden.

Numbers are the paper's headline figures (DALYs from daly_estimation_corrected.py;
costs from interventions_corrected.py). Output sized for the Elsevier graphical-
abstract spec (>= 1328 px wide, landscape).

Outputs:
  Draft/STOTENSubmission/figures_png/graphical_abstract.png  (+ tiff)
  Draft/STOTENSubmission/Graphical_Abstract.png
"""

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import FIGURES_DIR

# palette
C_AS   = '#c0392b'   # arsenic / high risk (red)
C_MN   = '#e67e22'   # manganese (orange)
C_PO4  = '#2980b9'   # phosphate / proxy (blue)
C_SAFE = '#27ae60'   # safe / intervention (green)
INK    = '#222222'
GREY   = '#888888'


def box(ax, x, y, w, h, fc, ec=INK, lw=1.2, alpha=1.0, rounding=0.02):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.0,rounding_size={rounding}",
                       fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=2)
    ax.add_patch(p)


def arrow(ax, x0, y0, x1, y1, color=INK, lw=2.4, ms=14):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>',
                 mutation_scale=ms, lw=lw, color=color, zorder=4))


def main():
    fig = plt.figure(figsize=(13.3, 5.4))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 13.3); ax.set_ylim(0, 5.4); ax.axis('off')

    # ── header ──────────────────────────────────────────────────────────────
    ax.text(6.65, 5.12,
            'Cumulative multi-contaminant groundwater exposure in Bangladesh',
            ha='center', va='center', fontsize=15.5, fontweight='bold', color=INK)
    ax.text(6.65, 4.74,
            'A low-cost phosphate proxy flags a doubled, climate-amplified health burden',
            ha='center', va='center', fontsize=11.5, color=GREY, style='italic')

    yb, yt = 0.5, 4.25   # band

    # ════════ STAGE 1: cheap phosphate proxy -> bistable arsenic ════════
    ax.text(2.05, 4.34, '1  Cheap proxy', ha='center', fontsize=12.5,
            fontweight='bold', color=C_PO4)
    # tube well + PO4 test
    box(ax, 0.55, 2.95, 1.15, 0.55, '#eaf2fb', ec=C_PO4, lw=1.6)
    ax.text(1.125, 3.22, 'PO$_4$ test', ha='center', va='center',
            fontsize=10.5, color=C_PO4, fontweight='bold')
    ax.text(1.125, 2.62, r'$\approx$1/10 the cost', ha='center', va='center',
            fontsize=9, color=GREY)
    ax.text(1.125, 2.30, 'of As assay', ha='center', va='center',
            fontsize=9, color=GREY)
    # bistable double-well sketch
    import numpy as np
    wx = np.linspace(2.1, 3.9, 200)
    t = np.linspace(-1.6, 1.6, 200)
    wy = 1.55 + 0.62 * (t**4 - 1.7 * t**2)  # double well
    ax.plot(wx, wy, color=INK, lw=2.0, zorder=3)
    ax.text(2.3, 1.05, 'low-As', ha='center', fontsize=8.5, color=C_SAFE)
    ax.text(3.7, 1.05, 'high-As', ha='center', fontsize=8.5, color=C_AS)
    ax.text(3.0, 2.55, 'saddle =\ntipping point', ha='center', fontsize=8.2, color=GREY)
    ax.scatter([2.36, 3.64], [1.42, 1.42], s=70, c=[C_SAFE, C_AS], zorder=5, edgecolor='white')
    ax.text(3.0, 3.62, 'PO$_4$ ranks wells by\nproximity to As threshold',
            ha='center', fontsize=9.2, color=INK)
    arrow(ax, 1.72, 3.0, 2.25, 2.6, color=C_PO4, lw=2.2)

    arrow(ax, 4.05, 2.4, 4.62, 2.4, color=INK, lw=2.6, ms=18)

    # ════════ STAGE 2: burden doubles ════════
    ax.text(6.65, 4.34, '2  Doubled burden', ha='center', fontsize=12.5,
            fontweight='bold', color=C_AS)
    axb = fig.add_axes([0.405, 0.345, 0.185, 0.37]); axb.set_facecolor('none')
    axb.bar([0, 1], [462, 942], width=0.62,
            color=[GREY, C_AS], edgecolor='white', zorder=3)
    # stack Mn share on multi bar
    axb.bar([1], [942 * 0.321], width=0.62, bottom=[942 * (1 - 0.321)],
            color=C_MN, edgecolor='white', zorder=4)
    axb.set_xticks([0, 1]); axb.set_xticklabels(['As-only', 'Multi'], fontsize=9.5)
    axb.set_ylim(0, 1180)
    axb.set_yticks([])
    for s in ['top', 'right', 'left']:
        axb.spines[s].set_visible(False)
    axb.text(0, 462 + 55, '462k', ha='center', fontsize=9.5, color=GREY, fontweight='bold')
    axb.text(1, 942 + 55, '942k', ha='center', fontsize=10.5, color=C_AS, fontweight='bold')
    axb.text(1, 942 * (1 - 0.321/2), 'Mn\n32%', ha='center', va='center',
             fontsize=7.5, color='white', fontweight='bold')
    axb.set_title('annual DALYs', fontsize=8.5, color=GREY, pad=2)
    ax.text(6.65, 1.28, r'2.0$\times$ higher than As-only', ha='center',
            fontsize=9.8, color=INK, fontweight='bold')
    ax.text(6.65, 0.92, 'manganese rivals As, yet is unmonitored', ha='center',
            fontsize=9.0, color=C_MN)

    arrow(ax, 8.68, 2.4, 9.25, 2.4, color=INK, lw=2.6, ms=18)

    # ════════ STAGE 3: climate + cheap fix ════════
    ax.text(11.15, 4.34, '3  Climate + cheap fix', ha='center', fontsize=12.5,
            fontweight='bold', color=C_SAFE)
    # climate up-arrow
    arrow(ax, 9.7, 1.7, 9.7, 3.5, color=C_AS, lw=3.0, ms=16)
    ax.text(10.78, 3.35, '3 compliant zones cross', ha='left', fontsize=9.3, color=C_AS)
    ax.text(10.78, 3.03, 'the WHO As limit by 2050', ha='left', fontsize=9.3, color=C_AS)
    ax.text(10.78, 2.71, '(CMIP6, SSP5-8.5)', ha='left', fontsize=8.2, color=GREY)
    # intervention box
    box(ax, 9.75, 0.95, 3.05, 1.15, '#eafaf1', ec=C_SAFE, lw=1.8)
    ax.text(11.27, 1.74, 'Seasonal well-switching', ha='center', fontsize=10,
            fontweight='bold', color=C_SAFE)
    ax.text(11.27, 1.40, 'averts 446k DALYs/yr at \$61 each', ha='center',
            fontsize=9, color=INK)
    ax.text(11.27, 1.12, r'14$\times$ cheaper than well-deepening', ha='center',
            fontsize=8.6, color=GREY)

    # frame separators
    for xsep in [4.55, 8.75]:
        ax.plot([xsep, xsep], [0.55, 4.2], color='#dddddd', lw=1.0, zorder=1)

    png = FIGURES_DIR / 'graphical_abstract.png'
    tiff = FIGURES_DIR / 'graphical_abstract.tiff'
    fig.savefig(png, dpi=200, bbox_inches='tight', facecolor='white')
    fig.savefig(FIGURES_DIR / 'Graphical_Abstract.png', dpi=200, bbox_inches='tight', facecolor='white')
    fig.savefig(tiff, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    from PIL import Image
    w, h = Image.open(png).size
    print(f"graphical_abstract.png written ({w}x{h} px)")


if __name__ == '__main__':
    main()
