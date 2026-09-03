#!/usr/bin/env python3
"""R2.1: the multi/arsenic-only DALY ratio as a continuous function of the
manganese disability weight.

R2.1 asks why an arsenic-anchored dose-response is applied to a summed
multi-contaminant hazard index. Part of the answer is that the fully
contaminant-specific alternative does not settle the question, because its answer
is almost entirely determined by one unmeasured choice: how manganese-attributable
neurodevelopmental impairment is valued. This sweeps that weight continuously so
the bracketing can be seen rather than asserted.

Reproduces daly_contaminant_specific.py exactly at DW_MN_NEURO = 0.043.
Run from the bundle root, or directly; paths resolve either way.
"""
import sys, pathlib, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')

ROOT = pathlib.Path(__file__).resolve().parents[2]   # bundle root (holds config.py)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))   # sibling: daly_estimation_corrected

from config import (DATA_FILE, TABLES_DIR, REFERENCE_DOSES, CANCER_SLOPE_FACTORS,
                    CR_VI_FRACTION, AT_CARCINOGENIC, assign_zones)
from daly_estimation_corrected import (EXPOSURE_PARAMS_CORRECTED, GW_USAGE_FRACTION,
                                       AGE_DISTRIBUTION, annual_yll_from_cancer,
                                       annual_chronic_prevalence, DISABILITY_WEIGHTS)

CONTAMS = ['As', 'Mn2+', 'Fe2+', 'Cr3+', 'NO3-', 'Cu2+', 'Al3+']
REF_AGE = 'adult_male'
PMAX_MN, KMN, NMN = 0.10, 1.5, 1.5
MN_ADULT_SUSCEPT = 0.2
DW_GENERIC = 0.06

# GBD 2019 anchors that bound the plausible range for this endpoint
ANCHORS = [
    (0.043, 'mild intellectual disability',      'used here'),
    (0.100, 'borderline intellectual functioning', None),
    (0.361, 'moderate/severe neurodevelopmental', 'upper anchor'),
]

def p_mn_neuro(hq):
    hq = np.maximum(hq, 0.0)
    return PMAX_MN * hq ** NMN / (KMN ** NMN + hq ** NMN)

def adult_as_dw():
    return (0.4 * DISABILITY_WEIGHTS['skin_lesions']
            + 0.3 * DISABILITY_WEIGHTS['peripheral_neuropathy']
            + 0.3 * DISABILITY_WEIGHTS['cardiovascular'])

def child_as_dw():
    return (0.4 * DISABILITY_WEIGHTS['neurodevelopmental']
            + 0.6 * DISABILITY_WEIGHTS['skin_lesions'])

def build():
    df = assign_zones(pd.read_csv(DATA_FILE))
    params = EXPOSURE_PARAMS_CORRECTED[REF_AGE]
    ir, bw = params['ir_L_day'], params['bw_kg']
    hq = {}
    for c in CONTAMS:
        if c in df.columns:
            conc = df[c].values.astype(float) / (1000.0 if c == 'As' else 1.0)
            hq[c] = conc * ir / bw / REFERENCE_DOSES[c]
    hq = pd.DataFrame(hq, index=df.index); hq['phys_zone'] = df['phys_zone']
    g = pd.read_csv(TABLES_DIR / 'T2_zone_population_weighted_stats.csv').set_index('phys_zone')
    return hq, g, params

def national(hq, g, params, dw_mn):
    """National DALYs under each aggregation rule, for one manganese weight."""
    tot = {'as_only': 0.0, 'dose_add': 0.0, 'contam_specific': 0.0}
    for z in g.index:
        zhq = hq[hq.phys_zone == z]
        if len(zhq) < 5:
            continue
        pop = g.loc[z, 'population'] * GW_USAGE_FRACTION
        cols = [c for c in CONTAMS if c in zhq.columns]
        med = {c: float(zhq[c].median()) for c in cols}
        tm = sum(med.values())
        hqm = {c: g.loc[z, 'HI_multi_pop_p50'] * (med[c] / tm) if tm > 0 else 0.0 for c in cols}
        hi_as = g.loc[z, 'HI_as_only_pop_p50']
        yll_m = yll_a = 0.0
        for ag, frac in AGE_DISTRIBUTION.items():
            ap = pop * frac
            yll_m += annual_yll_from_cancer(g.loc[z, 'CR_multi_pop_p50'], ap, ag)[0]
            yll_a += annual_yll_from_cancer(g.loc[z, 'CR_as_only_pop_p50'], ap, ag)[0]
        yld_a = yld_d = yld_c = 0.0
        for ag, frac in AGE_DISTRIBUTION.items():
            ap = pop * frac
            dw_as = child_as_dw() if ag.startswith('child') else adult_as_dw()
            yld_a += annual_chronic_prevalence(hi_as) * ap * dw_as
            yld_d += annual_chronic_prevalence(sum(hqm.values())) * ap * dw_as
            cs = annual_chronic_prevalence(hqm.get('As', 0)) * dw_as
            cs += p_mn_neuro(hqm.get('Mn2+', 0)) * (1.0 if ag.startswith('child') else MN_ADULT_SUSCEPT) * dw_mn
            for c in cols:
                if c not in ('As', 'Mn2+'):
                    cs += annual_chronic_prevalence(hqm[c]) * DW_GENERIC
            yld_c += cs * ap
        tot['as_only'] += yld_a + yll_a
        tot['dose_add'] += yld_d + yll_m
        tot['contam_specific'] += yld_c + yll_m
    return tot

if __name__ == '__main__':
    hq, g, params = build()
    base = national(hq, g, params, 0.043)
    print(f"reproduction check at DW_Mn = 0.043")
    print(f"  as-only          {base['as_only']:>12,.0f}   (table: 462,318)")
    print(f"  dose-addition    {base['dose_add']:>12,.0f}   (table: 941,884)")
    print(f"  contam-specific  {base['contam_specific']:>12,.0f}   (table: 605,443)")
    print(f"  ratio            {base['contam_specific']/base['as_only']:>12.2f}x  (table: 1.31x)\n")

    weights = np.round(np.concatenate([np.arange(0.00, 0.401, 0.005), [0.043, 0.361]]), 4)
    weights = np.unique(weights)
    rows = []
    for w in weights:
        t = national(hq, g, params, float(w))
        rows.append(dict(dw=float(w),
                         ratio=t['contam_specific'] / t['as_only'],
                         daly=t['contam_specific']))
    out = pd.DataFrame(rows)
    dest = TABLES_DIR / 'T2_mn_weight_sensitivity.csv'
    out.to_csv(dest, index=False)
    print(f"swept {len(out)} manganese disability weights -> {dest.name}")
    for w, lab, _ in ANCHORS:
        r = out.iloc[(out.dw - w).abs().argmin()]
        print(f"  DW {r.dw:.3f}  ratio {r.ratio:5.2f}x   {lab}")
    da = base['dose_add'] / base['as_only']
    cross = out[out.ratio >= da]
    print(f"\n  dose-addition ratio {da:.2f}x")
    if len(cross):
        print(f"  contaminant-specific overtakes it at DW = {cross.iloc[0].dw:.3f}")
