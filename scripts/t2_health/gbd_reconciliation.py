"""SI Table S3: reconcile this pipeline with GBD 2019 under GBD's restricted
endpoint set (skin lesions + cancer only). Reuses the corrected DALY machinery;
the ONLY change is the non-cancer YLD disability weight, set to skin-lesions
(0.011) for every age group, with the cancer YLL arm unchanged. Reproduces the
full-endpoint headline (942k/462k) and the GBD-restricted figures."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import numpy as np, pandas as pd
from config import DATA_FILE, TABLES_DIR, assign_zones, PHYSIOGRAPHIC_ZONES
from daly_estimation_corrected import (
    EXPOSURE_PARAMS_CORRECTED, AGE_DISTRIBUTION, GW_USAGE_FRACTION,
    DISABILITY_WEIGHTS, annual_chronic_prevalence, annual_yll_from_cancer,
    calculate_hi_cr, _GRIDDED, ZONE_POPULATION,
)

SKIN_DW = DISABILITY_WEIGHTS['skin_lesions']  # 0.011


def national(df, restrict_gbd):
    """Return (DALY_multi, DALY_as, YLL_as, YLD_as) national totals.
    restrict_gbd=True -> non-cancer YLD uses skin-lesion DW only (GBD endpoint set)."""
    tot_m = tot_a = yll_a = yld_a = 0.0
    for pz in PHYSIOGRAPHIC_ZONES:
        zd = df[df['phys_zone'] == pz]
        if len(zd) < 5:
            continue
        if _GRIDDED is not None and pz in _GRIDDED.index:
            g = _GRIDDED.loc[pz]
            hi_m, hi_a = float(g['HI_multi_pop_p50']), float(g['HI_as_only_pop_p50'])
            cr_m, cr_a = float(g['CR_multi_pop_p50']), float(g['CR_as_only_pop_p50'])
        else:
            hi_m, hi_a = zd['HI_multi'].median(), zd['HI_as_only'].median()
            cr_m, cr_a = zd['CR_multi'].median(), zd['CR_as_only'].median()
        gw = ZONE_POPULATION.get(pz, 0) * GW_USAGE_FRACTION
        for ag, frac in AGE_DISTRIBUTION.items():
            ap = gw * frac
            if restrict_gbd:
                dw = SKIN_DW
            else:
                dw = ((0.4*DISABILITY_WEIGHTS['neurodevelopmental'] + 0.6*SKIN_DW)
                      if ag.startswith('child')
                      else (0.4*SKIN_DW + 0.3*DISABILITY_WEIGHTS['peripheral_neuropathy']
                            + 0.3*DISABILITY_WEIGHTS['cardiovascular']))
            ym = annual_chronic_prevalence(hi_m)*ap*dw
            ya = annual_chronic_prevalence(hi_a)*ap*dw
            lm = annual_yll_from_cancer(cr_m, ap, ag)[0]
            la = annual_yll_from_cancer(cr_a, ap, ag)[0]
            tot_m += ym+lm; tot_a += ya+la; yll_a += la; yld_a += ya
    return tot_m, tot_a, yll_a, yld_a


def main():
    df = assign_zones(pd.read_csv(DATA_FILE))
    p = EXPOSURE_PARAMS_CORRECTED['adult_male']
    hi_m, hi_a, cr_m, cr_a = calculate_hi_cr(df, 'adult_male', p)
    df['HI_multi'], df['HI_as_only'], df['CR_multi'], df['CR_as_only'] = hi_m, hi_a, cr_m, cr_a

    fm, fa, _, _ = national(df, restrict_gbd=False)
    gm, ga, gyll, gyld = national(df, restrict_gbd=True)
    out = pd.DataFrame([
        {'quantity': 'Full endpoint (main text)', 'multi': round(fm, -2), 'as_only': round(fa, -2)},
        {'quantity': 'GBD-restricted (skin+cancer)', 'multi': round(gm, -2), 'as_only': round(ga, -2)},
        {'quantity': '  YLL component (cancer)', 'multi': '', 'as_only': round(gyll, -2)},
        {'quantity': '  YLD component (skin)', 'multi': '', 'as_only': round(gyld, -2)},
    ])
    out.to_csv(TABLES_DIR / 'T2_gbd_reconciliation.csv', index=False)
    print(out.to_string(index=False))
    print(f"\nGBD-restricted As-only = {ga:,.0f} (SI Table S3: 66,600; GBD 2019 range 30k-80k)")
    print(f"Full As-only = {fa:,.0f} (462,000); Full multi = {fm:,.0f} (942,000)")


if __name__ == '__main__':
    main()
