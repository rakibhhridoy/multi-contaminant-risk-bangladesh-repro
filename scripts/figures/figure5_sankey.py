"""
Figure 6 (manuscript): Sankey attribution diagrams.
(A) Contaminant-to-health-burden attribution flow
(B) Uncertainty decomposition flow

NOTE: numeric LABELS are the reported results (HI exceedance, contaminant HI
shares, national/zone posterior HI, DALYs, variance decomposition). Flow WIDTHS
are illustrative/hand-set for legibility, not proportional to data -- this is an
attribution schematic, not a quantitative flow chart. Reproduces output/figures/
figure5_sankey.png.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.sankey import Sankey
import numpy as np
from pathlib import Path

# ─── Output ──────────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import FIGURES_DIR
OUT_DIR = FIGURES_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)
DPI = 600

# ─── Lancet colour palette ───────────────────────────────────────────────────
CRIMSON  = '#B71C1C'
TEAL     = '#00695C'
AMBER    = '#E65100'
STEEL    = '#1565C0'
SLATE    = '#546E7A'
LGREY    = '#ECEFF1'

# Lighter versions for flows
CRIMSON_L = '#EF9A9A'
TEAL_L    = '#80CBC4'
AMBER_L   = '#FFCC80'
STEEL_L   = '#90CAF9'
SLATE_L   = '#B0BEC5'


def set_lancet_style():
    """Set publication style matching Lancet Planetary Health."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        'axes.labelsize': 10,
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,
        'figure.dpi': 150,
    })


def draw_sankey_A(ax):
    """
    Panel (A): Contaminant-to-health-burden attribution.

    Flow: Geochemical Drivers → Contaminants → Health Endpoints → Zone Burden

    Uses manual rectangles + bezier curves for a clean Sankey look,
    since matplotlib's built-in Sankey is limited.
    """
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 8.5)
    ax.axis('off')
    ax.set_title('(A)  Contaminant sources to health burden attribution',
                 fontsize=10, fontweight='bold', loc='left', pad=10)

    # ── Column positions ─────────────────────────────────────────────────
    col_x = [0.5, 3.2, 6.0, 8.8]
    col_w = 1.2

    # ── Column 1: Geochemical Drivers ────────────────────────────────────
    drivers = [
        ('Reductive\ndissolution\nof Fe-oxides', 4.0, CRIMSON_L, 55),
        ('PO$_4$ competitive\ndesorption', 2.2, AMBER_L, 30),
        ('Anthropogenic\ninputs (NO$_3$)', 1.0, STEEL_L, 15),
    ]

    # ── Column 2: Contaminants (% of HI) ─────────────────────────────────
    contaminants = [
        ('As\n36·6%', 2.8, CRIMSON_L, 36.6),
        ('Mn\n32·1%', 2.4, AMBER_L, 32.1),
        ('Fe\n6·1%', 0.9, TEAL_L, 6.1),
        ('Others\n25·2%', 1.2, SLATE_L, 25.2),
    ]

    # ── Column 3: Health Endpoints ───────────────────────────────────────
    endpoints = [
        ('Non-cancer\nHI > 1\n76·0%', 3.5, CRIMSON_L, 60),
        ('Cancer risk\nCR > 10$^{-4}$\n65·5%', 2.0, AMBER_L, 25),
        ('DALYs\n942k/yr', 1.5, STEEL_L, 15),
    ]

    # ── Column 4: Zone Burden (annual DALYs/100k), ranked by per-capita rate ─
    zones = [
        ('Meghna FP\n839', 1.8, CRIMSON_L, 28),
        ('GBM Delta\n807', 1.6, '#EF9A9A', 25),
        ('Ganges FP\n771', 1.4, AMBER_L, 20),
        ('Barind Tract\n487', 1.0, STEEL_L, 12),
        ('Other zones', 0.8, SLATE_L, 15),
    ]

    def draw_column(items, col_idx, header):
        """Draw a column of rectangles and return their y-positions."""
        x = col_x[col_idx]
        # Stack items with small gaps
        total_h = sum(h for _, h, _, _ in items)
        gap = 0.15
        total_with_gaps = total_h + gap * (len(items) - 1)
        y_start = (8.0 - total_with_gaps) / 2 + total_with_gaps

        positions = []
        y = y_start
        for label, h, color, _ in items:
            y -= h
            rect = mpatches.FancyBboxPatch(
                (x, y), col_w, h,
                boxstyle="round,pad=0.05",
                facecolor=color, edgecolor='white', linewidth=1.5,
                alpha=0.85
            )
            ax.add_patch(rect)
            ax.text(x + col_w/2, y + h/2, label,
                    ha='center', va='center', fontsize=8,
                    fontweight='bold', color='#212121')
            positions.append((x, y, h))
            y -= gap

        # Column header
        ax.text(x + col_w/2, y_start + 0.15, header,
                ha='center', va='bottom', fontsize=9,
                fontweight='bold', color=SLATE)

        return positions

    pos_drv = draw_column(drivers, 0, 'Geochemical\nDrivers')
    pos_con = draw_column(contaminants, 1, 'Contaminants\n(% of HI)')
    pos_end = draw_column(endpoints, 2, 'Health\nEndpoints')
    pos_zon = draw_column(zones, 3, 'Zone Burden\n(DALYs/100k)')

    def draw_flows(pos_left, pos_right, connections, alpha=0.2):
        """Draw curved flows between two columns."""
        for (li, ri, width_frac, color) in connections:
            xl, yl, hl = pos_left[li]
            xr, yr, hr = pos_right[ri]

            # Start and end y-centers (approximate)
            y1 = yl + hl * 0.5
            y2 = yr + hr * 0.5
            flow_w = min(hl, hr) * width_frac

            # Bezier curve
            mid_x = (xl + col_w + xr) / 2
            xs = np.array([xl + col_w, mid_x, mid_x, xr])

            from matplotlib.path import Path as MPath
            from matplotlib.patches import PathPatch

            verts = [
                (xl + col_w, y1 + flow_w/2),
                (mid_x, y1 + flow_w/2),
                (mid_x, y2 + flow_w/2),
                (xr, y2 + flow_w/2),
                (xr, y2 - flow_w/2),
                (mid_x, y2 - flow_w/2),
                (mid_x, y1 - flow_w/2),
                (xl + col_w, y1 - flow_w/2),
                (xl + col_w, y1 + flow_w/2),
            ]
            codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
                     MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
                     MPath.CLOSEPOLY]
            path = MPath(verts, codes)
            patch = PathPatch(path, facecolor=color, edgecolor='none',
                              alpha=alpha)
            ax.add_patch(patch)

    # Drivers → Contaminants flows
    flows_dc = [
        (0, 0, 0.7, CRIMSON_L),   # Reductive dissolution → As
        (0, 1, 0.5, CRIMSON_L),   # Reductive dissolution → Mn
        (0, 2, 0.4, CRIMSON_L),   # Reductive dissolution → Fe
        (1, 0, 0.5, AMBER_L),     # PO4 desorption → As
        (1, 1, 0.3, AMBER_L),     # PO4 → Mn
        (2, 3, 0.6, STEEL_L),     # Anthropogenic → Others (NO3, etc.)
    ]
    draw_flows(pos_drv, pos_con, flows_dc, alpha=0.32)

    # Contaminants → Endpoints flows
    flows_ce = [
        (0, 0, 0.6, CRIMSON_L),   # As → Non-cancer HI
        (0, 1, 0.5, CRIMSON_L),   # As → Cancer CR
        (1, 0, 0.5, AMBER_L),     # Mn → Non-cancer HI
        (2, 0, 0.3, TEAL_L),      # Fe → Non-cancer HI
        (3, 0, 0.3, SLATE_L),     # Others → Non-cancer HI
        (0, 2, 0.3, CRIMSON_L),   # As → DALYs
        (1, 2, 0.3, AMBER_L),     # Mn → DALYs
    ]
    draw_flows(pos_con, pos_end, flows_ce, alpha=0.28)

    # Endpoints → Zones flows
    flows_ez = [
        (0, 0, 0.4, CRIMSON_L),   # HI → Ganges
        (0, 1, 0.35, CRIMSON_L),  # HI → GBM Delta
        (0, 2, 0.3, AMBER_L),     # HI → Meghna
        (2, 0, 0.3, STEEL_L),     # DALYs → Ganges
        (2, 1, 0.25, STEEL_L),    # DALYs → GBM Delta
        (1, 3, 0.3, AMBER_L),     # CR → Barind
        (0, 4, 0.25, SLATE_L),    # HI → Others
    ]
    draw_flows(pos_end, pos_zon, flows_ez, alpha=0.25)


def draw_sankey_B(ax):
    """
    Panel (B): Uncertainty decomposition.

    Flow: Uncertainty Sources → Variance Contribution → Zone-level Impact
    """
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-1.2, 9.0)
    ax.axis('off')
    ax.set_title('(B)  Uncertainty source decomposition for projected HI$_{2050}$',
                 fontsize=10, fontweight='bold', loc='left', pad=14)

    col_x = [0.5, 3.5, 7.0]
    col_w = 1.4

    # ── Column 1: Uncertainty Sources ────────────────────────────────────
    sources = [
        ('Transfer\nfunction\n41%', 3.0, CRIMSON_L, 41),
        ('Toxicity\nparameters\n39%', 2.85, AMBER_L, 39),
        ('Exposure\nvariability\n21%', 1.5, STEEL_L, 21),
        ('Climate\nscenario\n<1%', 0.3, TEAL_L, 1),
    ]

    # ── Column 2: Projected HI range (2050, national posterior) ──────────
    hi_range = [
        ('SSP5-8.5\nMedian 3·41\n[2·25–5·09]', 3.0, CRIMSON_L, 55),
        ('SSP2-4.5\nMedian 2·68\n[1·82–3·98]', 2.5, AMBER_L, 45),
    ]

    # ── Column 3: Zone-level Impact (2050 SSP5-8.5 median HI, descending) ─
    zone_impact = [
        ('Ganges FP\nHI 4·00', 1.6, CRIMSON_L, 25),
        ('GBM Delta\nHI 3·27', 1.4, '#EF9A9A', 22),
        ('Barind\nHI 3·23', 1.2, AMBER_L, 18),
        ('Meghna FP\nHI 3·14', 1.0, STEEL_L, 14),
        ('Eastern Hills\nHI 2·51', 0.8, TEAL_L, 11),
    ]

    def draw_column(items, col_idx, header):
        x = col_x[col_idx]
        total_h = sum(h for _, h, _, _ in items)
        gap = 0.15
        total_with_gaps = total_h + gap * (len(items) - 1)
        y_start = (8.0 - total_with_gaps) / 2 + total_with_gaps

        positions = []
        y = y_start
        for label, h, color, _ in items:
            y -= h
            rect = mpatches.FancyBboxPatch(
                (x, y), col_w, h,
                boxstyle="round,pad=0.05",
                facecolor=color, edgecolor='white', linewidth=1.5,
                alpha=0.85
            )
            ax.add_patch(rect)
            ax.text(x + col_w/2, y + h/2, label,
                    ha='center', va='center', fontsize=8,
                    fontweight='bold', color='#212121')
            positions.append((x, y, h))
            y -= gap

        ax.text(x + col_w/2, y_start + 0.15, header,
                ha='center', va='bottom', fontsize=9,
                fontweight='bold', color=SLATE)
        return positions

    pos_src = draw_column(sources, 0, 'Uncertainty\nSources')
    pos_hi  = draw_column(hi_range, 1, 'Projected HI\n(2050)')
    pos_zon = draw_column(zone_impact, 2, 'Zone-Level\nImpact')

    # Shift column headers down to avoid overlapping with panel title
    # (headers are drawn inside draw_column at y_start + 0.4)

    def draw_flows(pos_left, pos_right, connections, cw_l, cw_r, alpha=0.2):
        from matplotlib.path import Path as MPath
        from matplotlib.patches import PathPatch

        for (li, ri, width_frac, color) in connections:
            xl, yl, hl = pos_left[li]
            xr, yr, hr = pos_right[ri]

            y1 = yl + hl * 0.5
            y2 = yr + hr * 0.5
            flow_w = min(hl, hr) * width_frac

            mid_x = (xl + cw_l + xr) / 2

            verts = [
                (xl + cw_l, y1 + flow_w/2),
                (mid_x, y1 + flow_w/2),
                (mid_x, y2 + flow_w/2),
                (xr, y2 + flow_w/2),
                (xr, y2 - flow_w/2),
                (mid_x, y2 - flow_w/2),
                (mid_x, y1 - flow_w/2),
                (xl + cw_l, y1 - flow_w/2),
                (xl + cw_l, y1 + flow_w/2),
            ]
            codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
                     MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
                     MPath.CLOSEPOLY]
            path = MPath(verts, codes)
            patch = PathPatch(path, facecolor=color, edgecolor='none',
                              alpha=alpha)
            ax.add_patch(patch)

    # Sources → HI range
    flows_sh = [
        (0, 0, 0.5, CRIMSON_L),   # Transfer → SSP5-8.5
        (0, 1, 0.4, CRIMSON_L),   # Transfer → SSP2-4.5
        (1, 0, 0.4, AMBER_L),     # Toxicity → SSP5-8.5
        (1, 1, 0.35, AMBER_L),    # Toxicity → SSP2-4.5
        (2, 0, 0.3, STEEL_L),     # Exposure → SSP5-8.5
        (2, 1, 0.25, STEEL_L),    # Exposure → SSP2-4.5
        (3, 0, 0.4, TEAL_L),      # Climate → SSP5-8.5
        (3, 1, 0.3, TEAL_L),      # Climate → SSP2-4.5
    ]
    draw_flows(pos_src, pos_hi, flows_sh, col_w, col_w, alpha=0.30)

    # HI range → Zones
    flows_hz = [
        (0, 0, 0.35, CRIMSON_L),
        (0, 1, 0.3, CRIMSON_L),
        (0, 2, 0.25, AMBER_L),
        (1, 2, 0.2, AMBER_L),
        (1, 3, 0.25, STEEL_L),
        (1, 4, 0.3, TEAL_L),
    ]
    draw_flows(pos_hi, pos_zon, flows_hz, col_w, col_w, alpha=0.25)

    # Add annotation: P(HI > 1) = 100% — centered at bottom of panel
    ax.text(5.0, -0.6,
            'P(HI > 1) = 100% across all zones and both SSP scenarios',
            ha='center', va='center', fontsize=9, fontstyle='italic',
            color=CRIMSON,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE',
                      edgecolor=CRIMSON, linewidth=0.6))


def main():
    set_lancet_style()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 14))
    fig.subplots_adjust(hspace=0.15)

    draw_sankey_A(ax1)
    draw_sankey_B(ax2)

    out = OUT_DIR / 'figure5_sankey.png'
    fig.savefig(out, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
