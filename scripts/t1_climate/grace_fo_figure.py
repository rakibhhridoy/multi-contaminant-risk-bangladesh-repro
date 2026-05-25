"""Build the combined GRACE-FO supplementary figure (figS4_GRACE.png).

Three panels:
  (A) Per-zone GRACE-FO groundwater storage percentile time series, 2018-2024
  (B) Mean storage percentile vs zone median arsenic (positive correlation)
  (C) Storage trend vs zone median arsenic (uplifted Pleistocene zones deplete fastest)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

from config import TABLES_DIR, PHYSIOGRAPHIC_ZONES

from config import FIGURES_DIR
OUT = FIGURES_DIR / 'figS4_GRACE.png'
ZONE_LABELS = {
    'Ganges_Floodplain': 'Ganges FP',
    'GBM_Delta': 'GBM Delta',
    'Meghna_Floodplain': 'Meghna FP',
    'Brahmaputra_Floodplain': 'Brahmaputra FP',
    'Northern_Terrace': 'N. Terrace',
    'Barind_Tract': 'Barind Tract',
    'Eastern_Hills': 'E. Hills',
}
ZONE_COLORS = {
    'Ganges_Floodplain': '#c0392b',
    'GBM_Delta': '#e67e22',
    'Meghna_Floodplain': '#27ae60',
    'Brahmaputra_Floodplain': '#16a085',
    'Northern_Terrace': '#2980b9',
    'Barind_Tract': '#8e44ad',
    'Eastern_Hills': '#7f8c8d',
}


def main():
    ts = pd.read_csv(TABLES_DIR / 'T1_grace_fo_zone_timeseries.csv',
                     parse_dates=['date'])
    merged = pd.read_csv(TABLES_DIR / 'T1_grace_fo_arsenic_merge.csv')

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.32, wspace=0.25)

    # ─── A: time series ──────────────────────────────────────────────────────
    axA = fig.add_subplot(gs[0, :])
    for z in PHYSIOGRAPHIC_ZONES:
        g = ts[ts['phys_zone'] == z].sort_values('date')
        if g.empty:
            continue
        axA.plot(g['date'], g['gws_percentile_mean'],
                 color=ZONE_COLORS.get(z, '#888'),
                 lw=1.6, alpha=0.9, label=ZONE_LABELS.get(z, z))
    axA.axhline(50, color='#333', ls='--', lw=0.7, alpha=0.4)
    axA.set_xlabel('Date')
    axA.set_ylabel('GRACE-FO groundwater storage percentile (0=dry, 100=wet)')
    axA.set_ylim(0, 100)
    axA.legend(loc='upper right', ncol=4, fontsize=9, frameon=False)
    axA.text(0.005, 0.95, '(A) Per-zone time series (n=345 weekly granules)',
             transform=axA.transAxes, ha='left', va='top',
             fontsize=11, fontweight='bold')

    # ─── B: mean storage percentile vs As ─────────────────────────────────────
    axB = fig.add_subplot(gs[1, 0])
    axB.scatter(merged['mean_percentile'], merged['as_median'],
                s=120, c=[ZONE_COLORS.get(z, '#888') for z in merged['phys_zone']],
                alpha=0.9, edgecolor='white')
    for _, r in merged.iterrows():
        axB.annotate(ZONE_LABELS.get(r['phys_zone'], r['phys_zone']),
                     (r['mean_percentile'], r['as_median']),
                     xytext=(6, 5), textcoords='offset points', fontsize=9)
    slope, intercept, rv, pv, _ = stats.linregress(
        merged['mean_percentile'], merged['as_median'])
    x = np.linspace(merged['mean_percentile'].min(),
                    merged['mean_percentile'].max(), 50)
    axB.plot(x, slope * x + intercept, '--', color='#444', lw=1.2)
    axB.text(0.04, 0.96, f'r = {rv:+.2f}, p = {pv:.2f}',
             transform=axB.transAxes, ha='left', va='top', fontsize=10,
             style='italic')
    axB.set_xlabel('Mean storage percentile (2018–2024)')
    axB.set_ylabel('Zone median arsenic (µg/L)')
    axB.text(0.005, 1.02,
             '(B) Wetter aquifers → higher arsenic (redox-driven mobilisation)',
             transform=axB.transAxes, ha='left', va='bottom',
             fontsize=11, fontweight='bold')

    # ─── C: storage trend vs As ───────────────────────────────────────────────
    axC = fig.add_subplot(gs[1, 1])
    axC.scatter(merged['trend_pct_per_yr'], merged['as_median'],
                s=120, c=[ZONE_COLORS.get(z, '#888') for z in merged['phys_zone']],
                alpha=0.9, edgecolor='white')
    for _, r in merged.iterrows():
        axC.annotate(ZONE_LABELS.get(r['phys_zone'], r['phys_zone']),
                     (r['trend_pct_per_yr'], r['as_median']),
                     xytext=(6, 5), textcoords='offset points', fontsize=9)
    axC.axvline(0, color='#333', ls='--', lw=0.7, alpha=0.5)
    slope2, intercept2, rv2, pv2, _ = stats.linregress(
        merged['trend_pct_per_yr'], merged['as_median'])
    x2 = np.linspace(merged['trend_pct_per_yr'].min(),
                     merged['trend_pct_per_yr'].max(), 50)
    axC.plot(x2, slope2 * x2 + intercept2, '--', color='#444', lw=1.2)
    axC.text(0.04, 0.96, f'r = {rv2:+.2f}, p = {pv2:.2f}',
             transform=axC.transAxes, ha='left', va='top', fontsize=10,
             style='italic')
    axC.set_xlabel('Storage trend (percentile units / year, 2018–2024)')
    axC.set_ylabel('Zone median arsenic (µg/L)')
    axC.text(0.005, 1.02,
             '(C) Uplifted Pleistocene zones deplete fastest, not high-As floodplains',
             transform=axC.transAxes, ha='left', va='bottom',
             fontsize=11, fontweight='bold')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == '__main__':
    main()
