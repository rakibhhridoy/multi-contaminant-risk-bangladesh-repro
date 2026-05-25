"""
T2: DALY dose-response one-way sensitivity (tornado)  [added 2026-05-24]
=======================================================================
Reviewer concern addressed: "the 942,000 annual DALY estimate is
assumption-heavy (Hill P_max/K/n, ingestion rate)."

This script performs a transparent one-at-a-time (OAT) sensitivity sweep of
the national multi-contaminant annual DALY over the four parameters that most
directly drive the central estimate, holding all others at their default while
varying one across a plausible literature range. It reuses the EXACT DALY
machinery of daly_estimation_corrected.py (gridded WorldPop-weighted zone
medians, Hill dose-response, GBD YLL/YLD convention) so the baseline reproduces
the headline 941,884 yr^-1.

Ranges (plausible, not extreme):
  P_max  0.135 - 0.225   (default 0.18; +/-25%, Smith-2000 high-tier 10-30%)
  K      3.0   - 5.0     (default 4.0; +/-25%, Argos-2010 transition zone)
  n      1.275 - 1.725   (default 1.5; +/-15%, Hill steepness)
  IR     2.0   - 3.0     (default 2.5 L/day; EPA 2.0 to high-tropical 3.0)

Outputs:
  output/tables/T2_daly_tornado.csv
  Draft/STOTENSubmission/figures_png/figS8_daly_tornado.png  (+ tiff)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import PHYSIOGRAPHIC_ZONES, TABLES_DIR, FIGURES_DIR
from daly_estimation_corrected import (
    _GRIDDED, ZONE_POPULATION, GW_USAGE_FRACTION, AGE_DISTRIBUTION,
    annual_yll_from_cancer, annual_yld_from_chronic,
    HILL_PMAX, HILL_K, HILL_N,
)

IR_BASELINE = 2.5  # gridded medians were computed at adult-male IR = 2.5 L/day


def national_daly(p_max=HILL_PMAX, k=HILL_K, n=HILL_N, ir=IR_BASELINE):
    """National annual multi-contaminant DALY under given dose-response params.

    Reproduces estimate_annual_dalys() exactly at the defaults. Ingestion rate
    enters linearly: HI and CR both scale by ir / IR_BASELINE relative to the
    gridded population-weighted medians (which were computed at IR_BASELINE).
    """
    if _GRIDDED is None:
        raise RuntimeError("gridded zone stats unavailable; run spatial_aggregation.py first")
    scale = ir / IR_BASELINE
    dr = {'p_max': p_max, 'k': k, 'n': n}
    total = 0.0
    for pz in PHYSIOGRAPHIC_ZONES:
        if pz not in _GRIDDED.index:
            continue
        g = _GRIDDED.loc[pz]
        hi = float(g['HI_multi_pop_p50']) * scale
        cr = float(g['CR_multi_pop_p50']) * scale
        gw_pop = ZONE_POPULATION.get(pz, 0) * GW_USAGE_FRACTION
        for ag, frac in AGE_DISTRIBUTION.items():
            ap = gw_pop * frac
            yll, _ = annual_yll_from_cancer(cr, ap, ag)
            yld, _ = annual_yld_from_chronic(hi, ap, ag, dr)
            total += yll + yld
    return total


PARAMS = [
    # (label, kwarg, low, default, high)
    (r'Ingestion rate (L/day)',        'ir',    2.0,   IR_BASELINE, 3.0),
    (r'Hill $P_{\max}$',               'p_max', 0.135, HILL_PMAX,   0.225),
    (r'Hill $K$ (half-max HI)',        'k',     3.0,   HILL_K,      5.0),
    (r'Hill $n$ (steepness)',          'n',     1.275, HILL_N,      1.725),
]


def main():
    base = national_daly()
    print("=" * 68)
    print("DALY dose-response tornado (one-at-a-time sensitivity)")
    print("=" * 68)
    print(f"Baseline national multi-contaminant DALY: {base:,.0f} yr^-1")
    print(f"(reference headline = 941,884 yr^-1; match check)\n")

    rows = []
    for label, kw, lo, dflt, hi in PARAMS:
        d_lo = national_daly(**{kw: lo})
        d_hi = national_daly(**{kw: hi})
        swing = abs(d_hi - d_lo)
        rows.append({
            'parameter': label, 'kwarg': kw,
            'low_value': lo, 'default_value': dflt, 'high_value': hi,
            'DALY_at_low': d_lo, 'DALY_at_high': d_hi,
            'DALY_baseline': base,
            'pct_low': 100 * (d_lo - base) / base,
            'pct_high': 100 * (d_hi - base) / base,
            'swing_DALY': swing,
            'swing_pct': 100 * swing / base,
        })
        print(f"  {label:28s}: {d_lo:>9,.0f} ({100*(d_lo-base)/base:+5.1f}%) "
              f".. {d_hi:>9,.0f} ({100*(d_hi-base)/base:+5.1f}%)  "
              f"swing {100*swing/base:4.1f}%")

    df = pd.DataFrame(rows).sort_values('swing_DALY', ascending=True)
    df.to_csv(TABLES_DIR / 'T2_daly_tornado.csv', index=False)
    print(f"\nSaved {TABLES_DIR / 'T2_daly_tornado.csv'}")

    # ── Tornado plot ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    y = np.arange(len(df))
    for i, (_, r) in enumerate(df.iterrows()):
        x0 = min(r['DALY_at_low'], r['DALY_at_high']) / 1e3
        x1 = max(r['DALY_at_low'], r['DALY_at_high']) / 1e3
        ax.barh(i, x1 - x0, left=x0, height=0.6,
                color='#4c72b0', alpha=0.85, edgecolor='white')
        ax.text(x0 - 4, i, f"{r['low_value']:g}", va='center', ha='right', fontsize=8, color='#333')
        ax.text(x1 + 4, i, f"{r['high_value']:g}", va='center', ha='left', fontsize=8, color='#333')
    ax.axvline(base / 1e3, color='#c0392b', ls='--', lw=1.3,
               label=f'Baseline {base/1e3:,.0f}k')
    ax.set_yticks(y)
    ax.set_yticklabels(df['parameter'])
    ax.set_xlabel('National multi-contaminant annual DALYs (thousands)')
    ax.set_title('One-way sensitivity of the annual DALY burden to dose-response assumptions',
                 fontsize=10, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9, frameon=False)
    ax.margins(x=0.16)
    fig.tight_layout()

    fig.savefig(FIGURES_DIR / 'figS8_daly_tornado.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'figS8_daly_tornado.tiff', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figS8_daly_tornado.png/.tiff")

    rng_lo = df[['DALY_at_low', 'DALY_at_high']].min().min()
    rng_hi = df[['DALY_at_low', 'DALY_at_high']].max().max()
    print(f"\nFull OAT envelope across all four parameters: "
          f"{rng_lo:,.0f} - {rng_hi:,.0f} yr^-1")
    print(f"(baseline {base:,.0f}; multi/As ratio is preserved because IR scales "
          f"both arms equally)")
    print("=" * 68)


if __name__ == '__main__':
    main()
