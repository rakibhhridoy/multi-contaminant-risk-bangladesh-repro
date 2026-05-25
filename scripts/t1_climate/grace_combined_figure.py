"""Combined GRACEDADM + TELLUS figure for Figure S4.

Four panels:
  (A) GRACEDADM 7-day groundwater storage percentile, 2018-2024
  (B) TELLUS 24-year LWE thickness, 2002-2026
  (C) TELLUS multi-decadal LWE trend (cm/yr) vs zone median arsenic
  (D) GRACEDADM mean storage percentile vs zone median arsenic
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
    grace = pd.read_csv(TABLES_DIR / 'T1_grace_fo_zone_timeseries.csv',
                        parse_dates=['date'])
    grace_merge = pd.read_csv(TABLES_DIR / 'T1_grace_fo_arsenic_merge.csv')
    tellus = pd.read_csv(TABLES_DIR / 'T1_tellus_zone_timeseries.csv',
                         parse_dates=['date'])
    tellus_merge = pd.read_csv(TABLES_DIR / 'T1_tellus_arsenic_merge.csv')

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.34, wspace=0.24)

    # ─── A: GRACE-FO 7-day storage percentile 2018-2024 ──────────────────────
    axA = fig.add_subplot(gs[0, 0])
    for z in PHYSIOGRAPHIC_ZONES:
        g = grace[grace['phys_zone'] == z].sort_values('date')
        if g.empty:
            continue
        axA.plot(g['date'], g['gws_percentile_mean'],
                 color=ZONE_COLORS.get(z, '#888'),
                 lw=1.4, alpha=0.9, label=ZONE_LABELS.get(z, z))
    axA.axhline(50, color='#333', ls='--', lw=0.7, alpha=0.4)
    axA.set_xlabel('Date')
    axA.set_ylabel('Storage percentile (0=dry, 100=wet)')
    axA.set_ylim(0, 100)
    axA.legend(loc='upper right', ncol=2, fontsize=8, frameon=False)
    axA.text(0.005, 0.97,
             '(A) GRACEDADM 7-day record, 2018–2024 (n=345 granules)',
             transform=axA.transAxes, ha='left', va='top',
             fontsize=10, fontweight='bold')

    # ─── B: TELLUS 24-year LWE ───────────────────────────────────────────────
    axB = fig.add_subplot(gs[0, 1])
    for z in PHYSIOGRAPHIC_ZONES:
        t = tellus[tellus['phys_zone'] == z].sort_values('date')
        if t.empty:
            continue
        axB.plot(t['date'], t['lwe_cm'],
                 color=ZONE_COLORS.get(z, '#888'),
                 lw=1.2, alpha=0.9, label=ZONE_LABELS.get(z, z))
    axB.axhline(0, color='#333', ls='--', lw=0.7, alpha=0.4)
    axB.set_xlabel('Date')
    axB.set_ylabel('LWE thickness anomaly (cm)')
    axB.legend(loc='lower left', ncol=2, fontsize=8, frameon=False)
    axB.text(0.005, 0.97,
             '(B) TELLUS GRACE/GRACE-FO mascon, 2002–2026 (n=255 months)',
             transform=axB.transAxes, ha='left', va='top',
             fontsize=10, fontweight='bold')

    # ─── C: TELLUS trend vs As ───────────────────────────────────────────────
    axC = fig.add_subplot(gs[1, 0])
    axC.scatter(tellus_merge['trend_cm_per_yr'], tellus_merge['as_median'],
                s=120,
                c=[ZONE_COLORS.get(z, '#888') for z in tellus_merge['phys_zone']],
                alpha=0.9, edgecolor='white')
    for _, r in tellus_merge.iterrows():
        axC.annotate(ZONE_LABELS.get(r['phys_zone'], r['phys_zone']),
                     (r['trend_cm_per_yr'], r['as_median']),
                     xytext=(6, 5), textcoords='offset points', fontsize=9)
    axC.axvline(0, color='#333', ls='--', lw=0.7, alpha=0.5)
    slope, intercept, rv, pv, _ = stats.linregress(
        tellus_merge['trend_cm_per_yr'], tellus_merge['as_median'])
    x = np.linspace(tellus_merge['trend_cm_per_yr'].min(),
                    tellus_merge['trend_cm_per_yr'].max(), 50)
    axC.plot(x, slope * x + intercept, '--', color='#444', lw=1.2)
    axC.text(0.04, 0.96, f'r = {rv:+.2f}, p = {pv:.2f}',
             transform=axC.transAxes, ha='left', va='top',
             fontsize=10, style='italic')
    axC.set_xlabel('24-year LWE trend (cm/yr)')
    axC.set_ylabel('Zone median arsenic (µg/L)')
    axC.text(0.005, 1.02,
             '(C) Multi-decadal TWS trend: depletion is in low-As Pleistocene zones',
             transform=axC.transAxes, ha='left', va='bottom',
             fontsize=10, fontweight='bold')

    # ─── D: GRACE-FO 7-day mean vs As ────────────────────────────────────────
    axD = fig.add_subplot(gs[1, 1])
    axD.scatter(grace_merge['mean_percentile'], grace_merge['as_median'],
                s=120,
                c=[ZONE_COLORS.get(z, '#888') for z in grace_merge['phys_zone']],
                alpha=0.9, edgecolor='white')
    for _, r in grace_merge.iterrows():
        axD.annotate(ZONE_LABELS.get(r['phys_zone'], r['phys_zone']),
                     (r['mean_percentile'], r['as_median']),
                     xytext=(6, 5), textcoords='offset points', fontsize=9)
    slope2, intercept2, rv2, pv2, _ = stats.linregress(
        grace_merge['mean_percentile'], grace_merge['as_median'])
    x2 = np.linspace(grace_merge['mean_percentile'].min(),
                     grace_merge['mean_percentile'].max(), 50)
    axD.plot(x2, slope2 * x2 + intercept2, '--', color='#444', lw=1.2)
    axD.text(0.04, 0.96, f'r = {rv2:+.2f}, p = {pv2:.2f}',
             transform=axD.transAxes, ha='left', va='top',
             fontsize=10, style='italic')
    axD.set_xlabel('Mean GRACEDADM storage percentile (2018–2024)')
    axD.set_ylabel('Zone median arsenic (µg/L)')
    axD.text(0.005, 1.02,
             '(D) Wetter aquifers $\\rightarrow$ higher arsenic',
             transform=axD.transAxes, ha='left', va='bottom',
             fontsize=10, fontweight='bold')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == '__main__':
    main()
