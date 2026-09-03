"""Generate the conceptual Figure 1 for Nature Water submission.

Replaces the previous methods flowchart. Three panels:
  (A) Schematic of the bistable potential V(log As) with the two attractors
      and saddle, drawn at three phosphate regimes (low / mid / high)
      showing the wells separating as phosphate rises.
  (B) The mechanism cartoon: phosphate competitively desorbs arsenic from
      Fe-oxyhydroxide surfaces in reducing aquifers.
  (C) The policy cartoon: arsenic test strip (expensive, slow) vs phosphate
      colorimetric kit (cheap, fast) -> WASH network coverage map.

Output: figure1_conceptual.png and .tiff at 300 dpi.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib.lines import Line2D

from config import FIGURES_DIR
OUT_DIR = FIGURES_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

CRIMSON = '#c0392b'
STEEL = '#2980b9'
AMBER = '#e67e22'
SLATE = '#7f8c8d'
LEAF = '#27ae60'
DARK = '#212121'


def panel_A(ax):
    """Bistable effective potential at three PO4 levels."""
    x = np.linspace(0, 3, 400)
    # Construct a double-well V(x) = a*(x-x1)^2 * (x-x2)^2 / scale,
    # then add a Gaussian tilt that grows with phosphate to deepen the
    # high-As well. This keeps the curves bounded and visually comparable.
    def V(x, tilt):
        x1, x2 = 0.5, 2.1
        v = 1.4 * ((x - x1)**2) * ((x - x2)**2)
        # Tilt: bigger tilt -> deeper second well, lower saddle
        v = v - tilt * np.exp(-((x - x2)/0.45)**2)
        # Floor at min so curves sit on a common baseline
        return v - v.min()

    regimes = [
        ('Low PO$_4$ (< 1 mg/L)',         0.0, STEEL),
        ('Mid PO$_4$ (saddle, ~1.5--2)',  0.5, AMBER),
        ('High PO$_4$ (> 2 mg/L)',        1.1, CRIMSON),
    ]
    # Normalise so each curve has max=1 and a common visible range
    ymax = 1.0
    for label, tilt, color in regimes:
        v = V(x, tilt)
        v = v / v.max()  # rescale each to [0, 1]
        ax.plot(x, v, color=color, lw=2.5, label=label)

    # Mode + saddle annotations placed above any curve at the top of the plot
    ax.text(0.5, 0.04, 'low-As\nmode',
            ha='center', va='bottom', fontsize=8, color=DARK, style='italic')
    ax.text(2.1, 0.04, 'high-As\nmode',
            ha='center', va='bottom', fontsize=8, color=DARK, style='italic')
    ax.text(1.3, 0.78, 'saddle',
            ha='center', fontsize=8, color=DARK, style='italic',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor='#888', alpha=0.9))

    # WHO guideline marker
    ax.axvline(1.0, color='#444', ls=':', lw=0.8, alpha=0.5)
    ax.text(1.0, 1.02, 'WHO\n10 µg/L', ha='center', va='bottom',
            fontsize=7, color=DARK, fontweight='bold')

    ax.set_xlabel(r'log$_{10}$ As (µg/L)', fontsize=9)
    ax.set_ylabel(r'Effective potential $V$', fontsize=9)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 1.15)
    ax.tick_params(labelsize=8)
    ax.legend(loc='lower center', fontsize=7, frameon=False,
              bbox_to_anchor=(0.5, 0.30))
    ax.text(0.02, 1.06, '(A) Bistable arsenic potential, phosphate-controlled',
            transform=ax.transAxes, ha='left', va='bottom',
            fontsize=10, fontweight='bold', color=DARK)


def panel_B(ax):
    """Mechanism schematic: phosphate displaces arsenate from an HFO grain.

    Layout uses a tall canvas with clear vertical zones:
      top: italic one-line description
      middle: Fe grain (left) | PO4 in (arrow) | As out (arrow)
      bottom: reaction equation in its own boxed row (no overlap)
    """
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title + one-line description (top zone)
    ax.text(0.0, 9.8, '(B) Competitive desorption mechanism',
            ha='left', va='top', fontsize=11, fontweight='bold', color=DARK)
    ax.text(5.0, 8.7, 'Phosphate competitively displaces arsenate\n'
                      'from Fe-oxyhydroxide surfaces',
            ha='center', va='top', fontsize=8.5, color=DARK, style='italic')

    # Fe-oxyhydroxide grain (middle zone, centred)
    gy = 5.0
    grain = Circle((5.0, gy), 1.5, facecolor='#a87f4d', edgecolor='#5d4322',
                   lw=1.5, alpha=0.95, zorder=2)
    ax.add_patch(grain)
    ax.text(5.0, gy, 'Fe-(oxy)\nhydroxide', ha='center', va='center',
            fontsize=8.5, fontweight='bold', color='white', zorder=3)

    # Phosphate arriving from the LEFT (blue) — single clear arrow + ion
    ax.add_patch(Circle((1.4, gy), 0.45, facecolor=STEEL, edgecolor=DARK,
                        lw=0.6, zorder=3))
    ax.text(1.4, gy, r'PO$_4$', fontsize=8, color='white',
            ha='center', va='center', zorder=4, fontweight='bold')
    ax.annotate('', xy=(3.3, gy), xytext=(2.0, gy),
                arrowprops=dict(arrowstyle='-|>', color=STEEL, lw=2.2))
    ax.text(2.65, gy + 0.5, 'binds', ha='center', fontsize=7.5,
            color=STEEL, style='italic')

    # Arsenate leaving to the RIGHT (red) — single clear arrow + ion
    ax.annotate('', xy=(8.0, gy), xytext=(6.7, gy),
                arrowprops=dict(arrowstyle='-|>', color=CRIMSON, lw=2.2))
    ax.add_patch(Circle((8.6, gy), 0.45, facecolor=CRIMSON, edgecolor=DARK,
                        lw=0.6, zorder=3))
    ax.text(8.6, gy, 'As', fontsize=8, color='white',
            ha='center', va='center', zorder=4, fontweight='bold')
    ax.text(7.35, gy + 0.5, 'released', ha='center', fontsize=7.5,
            color=CRIMSON, style='italic')

    # Reaction equation (bottom zone, own boxed row, generous gap above)
    ax.text(5.0, 1.7,
            r'$\equiv$Fe–OAs $+$ H$_2$PO$_4^-$ $\;\rightarrow\;$ '
            r'$\equiv$Fe–OPO$_3$H$^-$ $+$ As$_{(aq)}$',
            ha='center', va='center', fontsize=9, color=DARK,
            bbox=dict(boxstyle='round,pad=0.45', facecolor='#fafafa',
                      edgecolor='#bbb', lw=0.8))


def panel_C(ax):
    """Policy schematic: arsenic vs phosphate surveillance, clean two columns."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.text(0.0, 9.8, '(C) Phosphate-led surveillance reframing',
            ha='left', va='top', fontsize=11, fontweight='bold', color=DARK)

    # ── Left column: Arsenic surveillance ──
    ax.text(2.3, 8.7, 'Arsenic surveillance', ha='center', fontsize=9.5,
            fontweight='bold', color=CRIMSON)
    box_as = FancyBboxPatch((0.3, 3.3), 4.0, 5.0, boxstyle="round,pad=0.1",
                             facecolor='#fef2f1', edgecolor=CRIMSON, lw=1.5,
                             zorder=1)
    ax.add_patch(box_as)
    for i, line in enumerate(['Lab AAS / ICP-MS',
                              'Field kit \\$5–15/test',
                              'Sample transport',
                              'Days–weeks']):
        ax.text(0.6, 7.7 - i*0.65, '•', fontsize=9, color=CRIMSON, va='center')
        ax.text(0.9, 7.7 - i*0.65, line, fontsize=7.5, va='center')
    ax.text(2.3, 4.55, r'$\sim$\$5–15 / test', ha='center', fontsize=10,
            fontweight='bold', color=CRIMSON)
    ax.text(2.3, 3.85, 'low rural coverage', ha='center', fontsize=7.5,
            style='italic', color='#666')

    # ── proxy arrow between columns (clear of both boxes) ──
    ax.annotate('', xy=(5.65, 5.8), xytext=(4.55, 5.8),
                arrowprops=dict(arrowstyle='-|>', color=DARK, lw=1.8))
    ax.text(5.1, 6.25, 'proxy', ha='center', fontsize=8, style='italic',
            color=DARK)

    # ── Right column: Phosphate proxy ──
    ax.text(7.7, 8.7, 'Phosphate proxy', ha='center', fontsize=9.5,
            fontweight='bold', color=STEEL)
    box_po4 = FancyBboxPatch((5.7, 3.3), 4.0, 5.0, boxstyle="round,pad=0.1",
                              facecolor='#eef5fb', edgecolor=STEEL, lw=1.5,
                              zorder=1)
    ax.add_patch(box_po4)
    for i, line in enumerate(['Colorimetric strip',
                              'Field kit \\$0.30–1/test',
                              'In situ, no transport',
                              'Minutes']):
        ax.text(6.0, 7.7 - i*0.65, '•', fontsize=9, color=STEEL, va='center')
        ax.text(6.3, 7.7 - i*0.65, line, fontsize=7.5, va='center')
    ax.text(7.7, 4.55, r'$\sim$10$\times$ cheaper', ha='center', fontsize=10,
            fontweight='bold', color=STEEL)
    ax.text(7.7, 3.85, 'full WASH network', ha='center', fontsize=7.5,
            style='italic', color='#666')

    # ── Threshold callout (bottom, own row) ──
    ax.text(5.0, 1.6,
            r'PO$_4$ $>$ 1.5–2.0 mg/L  $\Rightarrow$  well likely in high-As mode',
            ha='center', fontsize=8.5, fontweight='bold', color=DARK,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff9e6',
                      edgecolor=AMBER, lw=1.2))


def main():
    fig = plt.figure(figsize=(15, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.05], wspace=0.30)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    panel_A(ax_a)
    panel_B(ax_b)
    panel_C(ax_c)
    out_png = OUT_DIR / 'figure1_conceptual.png'
    out_tiff = FIGURES_DIR / 'figure1_conceptual.tiff'
    out_tiff.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(out_tiff, dpi=300, bbox_inches='tight', facecolor='white',
                pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)
    print(f"Saved {out_png}")
    print(f"Saved {out_tiff}")


if __name__ == '__main__':
    main()
