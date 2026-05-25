"""
T2: As-Mn synergistic-toxicity sensitivity bound  [added 2026-05-24]
====================================================================
Reviewer concern addressed: "the additive hazard index assumes no synergistic
toxicity, yet As-Mn neurodevelopmental synergy is documented (Wasserman 2011)."

We quantify how much the additive HI UNDERSTATES the burden under a transparent
interaction model, demonstrating that the additive estimate is a CONSERVATIVE
LOWER BOUND rather than a best guess.

Interaction model (illustrative, monotone, dimensionally an HQ):

    HI_syn = sum_i HQ_i  +  gamma * sqrt(HQ_As * HQ_Mn)

The geometric-mean cross term is symmetric, vanishes if either As or Mn is
absent, and adds hazard only where BOTH are co-elevated -- the toxicological
meaning of synergy. gamma = 0 recovers the additive HI; we sweep gamma over
{0, 0.25, 0.5, 1.0} as an interaction-strength bound. We do NOT claim a precise
synergy magnitude; the result is a directional bound on the additive assumption.

Carcinogenic risk (YLL arm) is unchanged -- the As-Mn synergy evidence is
neurodevelopmental/chronic, not carcinogenic -- so only the YLD arm responds.

Method mirrors the gridded DALY pipeline exactly: per-sample HI and the cross
term are interpolated (IDW k=10, p=2) onto the WorldPop grid, population-weighted
zone medians are formed per gamma, and DALYs are computed with the same Hill
dose-response and GBD YLL/YLD convention. gamma=0 reproduces ~941,884 yr^-1.

Outputs:
  output/tables/T2_daly_synergy_bound.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd

from config import (
    DATA_FILE, PHYSIOGRAPHIC_ZONES, TABLES_DIR, REFERENCE_DOSES, assign_zones,
)
from daly_estimation_corrected import (
    EXPOSURE_PARAMS_CORRECTED, calculate_hi_cr,
    _GRIDDED, ZONE_POPULATION, GW_USAGE_FRACTION, AGE_DISTRIBUTION,
    annual_yll_from_cancer, annual_yld_from_chronic,
)
from spatial_aggregation import (
    load_grid, _zone_for_point, interpolate_idw, weighted_quantile,
)

GAMMAS = [0.0, 0.25, 0.5, 1.0]


def main():
    print("=" * 68)
    print("As-Mn synergistic-toxicity DALY bound (gridded)")
    print("=" * 68)

    params = EXPOSURE_PARAMS_CORRECTED['adult_male']
    ir, bw = params['ir_L_day'], params['bw_kg']

    df = pd.read_csv(DATA_FILE)
    df = assign_zones(df)
    hi_multi, hi_as, cr_multi, cr_as = calculate_hi_cr(df, 'adult_male', params)
    df['HI_multi'] = hi_multi
    df['CR_multi'] = cr_multi

    # Per-sample hazard quotients for the cross term
    hq_as = (df['As'].values / 1000.0) * ir / bw / REFERENCE_DOSES['As']
    hq_mn = (df['Mn2+'].values) * ir / bw / REFERENCE_DOSES['Mn2+']
    df['cross_AsMn'] = np.sqrt(np.maximum(hq_as, 0) * np.maximum(hq_mn, 0))
    print(f"Median HQ_As={np.median(hq_as):.3f}  HQ_Mn={np.median(hq_mn):.3f}  "
          f"cross={np.median(df['cross_AsMn']):.3f}")

    # ── Grid: IDW of HI_multi and the cross term onto WorldPop cells ─────────
    lat, lon, pop = load_grid()
    centroids = {z: ((b['lat'][0] + b['lat'][1]) / 2,
                     (b['lon'][0] + b['lon'][1]) / 2)
                 for z, b in PHYSIOGRAPHIC_ZONES.items()}
    zones = np.array([_zone_for_point(la, lo, centroids, PHYSIOGRAPHIC_ZONES)
                      for la, lo in zip(lat, lon)])
    s_lat, s_lon = df['Latitude'].values, df['Longitude'].values
    cell_hi = interpolate_idw(s_lat, s_lon, df['HI_multi'].values, lat, lon, k=10, p=2)
    cell_cross = interpolate_idw(s_lat, s_lon, df['cross_AsMn'].values, lat, lon, k=10, p=2)

    # CR (YLL arm) is synergy-invariant; reuse cached gridded CR medians.
    cr_med = {z: float(_GRIDDED.loc[z, 'CR_multi_pop_p50'])
              for z in _GRIDDED.index}

    rows = []
    nat = {}
    for gamma in GAMMAS:
        cell_hi_syn = cell_hi + gamma * cell_cross
        total = 0.0
        zrows = []
        for z in PHYSIOGRAPHIC_ZONES:
            mask = zones == z
            if not mask.any() or z not in _GRIDDED.index:
                continue
            hi_syn_med = weighted_quantile(cell_hi_syn[mask], pop[mask], 0.50)
            cr = cr_med[z]
            gw_pop = ZONE_POPULATION.get(z, 0) * GW_USAGE_FRACTION
            ztot = 0.0
            for ag, frac in AGE_DISTRIBUTION.items():
                ap = gw_pop * frac
                yll, _ = annual_yll_from_cancer(cr, ap, ag)
                yld, _ = annual_yld_from_chronic(hi_syn_med, ap, ag)
                ztot += yll + yld
            total += ztot
            zrows.append((z, hi_syn_med, ztot))
        nat[gamma] = total
        rows.append({'gamma': gamma, 'national_DALY_multi': total})
        print(f"  gamma={gamma:<4}  national DALY = {total:>10,.0f}")

    base = nat[0.0]
    out = pd.DataFrame(rows)
    out['pct_increase_vs_additive'] = 100 * (out['national_DALY_multi'] - base) / base
    out.to_csv(TABLES_DIR / 'T2_daly_synergy_bound.csv', index=False)

    print(f"\nAdditive (gamma=0) recomputed baseline: {base:,.0f} yr^-1 "
          f"(headline 941,884)")
    for _, r in out.iterrows():
        print(f"  gamma={r['gamma']:<4}: {r['national_DALY_multi']:>10,.0f}  "
              f"({r['pct_increase_vs_additive']:+5.1f}% vs additive)")
    print(f"\nSaved {TABLES_DIR / 'T2_daly_synergy_bound.csv'}")
    print("Interpretation: additive HI is a conservative lower bound; plausible")
    print("As-Mn synergy raises the burden, never lowers it.")
    print("=" * 68)


if __name__ == '__main__':
    main()
