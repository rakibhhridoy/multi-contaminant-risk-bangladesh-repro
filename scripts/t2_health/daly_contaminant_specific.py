"""
F6 robustness: contaminant-specific DALY re-architecture (2026-06-02).
=====================================================================
Addresses the reviewer concern that the national multi/As DALY ratio (~2.04x)
is mechanical: it is P_Hill(HI_multi)/P_Hill(HI_as) for a SINGLE arsenic-
calibrated Hill curve applied to the SUMMED multi-contaminant hazard index
(dose-addition). Because the curve is convex, summing hazard quotients before
applying it, and reading manganese's hazard off arsenic's dose-response, could
in principle inflate the ratio.

We test this with two alternative aggregation rules, both using the gridded
WorldPop-weighted, zone-resolved exposure surface of the main analysis:

  (1) RESPONSE-ADDITION (shared As curve): apply the same Hill curve to EACH
      contaminant's own HQ, then sum prevalences across contaminants
      (sum_i P_Hill(HQ_i)) instead of P_Hill(sum_i HQ_i). This removes the
      convexity/summed-HI effect while holding the dose-response fixed.

  (2) FULL CONTAMINANT-SPECIFIC: arsenic keeps its Argos/Smith-anchored Hill
      curve and arsenic endpoints (skin/neuropathy/cardiovascular + cancer);
      manganese gets its OWN neurodevelopmental dose-response anchored to
      Wasserman et al. 2006/2011 (IQ deficits in children at well-water Mn
      routinely found in Bangladesh) with the GBD neurodevelopmental disability
      weight (0.361); iron, copper and nitrate keep the shared Hill on their own
      HQ with a generic chronic disability weight. Per-contaminant annual DALYs
      are summed. This is the toxicologically appropriate response-addition of
      DISTINCT endpoints rather than dose-addition of a single endpoint.

Both are reported against the SAME arsenic-only baseline as the main text.
Output: T2_daly_contaminant_specific.csv (national + per-zone, both models).

NB: model (2) is exploratory and is presented as a robustness bound, not the
headline. Every Mn parameter is stated explicitly below.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
from config import (
    DATA_FILE, REFERENCE_DOSES, CANCER_SLOPE_FACTORS, CR_VI_FRACTION,
    AT_CARCINOGENIC, TABLES_DIR, assign_zones, PHYSIOGRAPHIC_ZONES,
)
from daly_estimation_corrected import (
    EXPOSURE_PARAMS_CORRECTED, AGE_DISTRIBUTION, GW_USAGE_FRACTION,
    DISABILITY_WEIGHTS, LIFE_EXPECTANCY_BIRTH, DISCOUNT_RATE,
    CANCER_MANIFESTATION_RATE, HILL_PMAX, HILL_K, HILL_N,
    annual_chronic_prevalence, annual_yll_from_cancer,
)

CONTAMS = [c for c in REFERENCE_DOSES if c != 'Fe2+'] + ['Fe2+']  # all w/ RfD
REF_AGE = 'adult_male'

# ── Manganese-specific neurodevelopmental dose-response (model 2) ────────────
# Anchored to Wasserman et al. 2006 (EHP 114:124) and 2011: children drinking
# well water with Mn >~0.4 mg/L show measurable (~6-point) IQ deficits, i.e. a
# clinically-relevant neurodevelopmental impairment in a sizeable minority. We
# model the prevalence of such impairment as a Hill function of the Mn hazard
# quotient (HQ_Mn = C_Mn*IR/BW/RfD_Mn, RfD_Mn = 0.024 mg/kg/day):
#   P_Mn(HQ) = PMAX_MN * HQ^n / (KMN^n + HQ^n)
# A ~6-point downward IQ shift does not disable a quarter of children; it modestly
# raises the prevalence of MILD intellectual disability. We therefore (i) cap the
# impaired fraction conservatively (PMAX_MN=0.10) and (ii) weight it with the GBD
# disability weight for MILD idiopathic intellectual disability (0.043), NOT the
# moderate/severe neurodevelopmental weight (0.361, which would be inappropriate
# for a population-wide sub-clinical IQ shift). Mn neurodevelopmental effects are a
# primarily PAEDIATRIC endpoint; adults accrue a reduced (motor/cognitive) fraction.
PMAX_MN, KMN, NMN = 0.10, 1.5, 1.5
MN_ADULT_SUSCEPT = 0.2   # adults accrue 20% of the child neuro response
DW_MN_NEURO = 0.043      # GBD mild idiopathic intellectual disability

def p_mn_neuro(hq):
    hq = np.maximum(hq, 0.0)
    return PMAX_MN * hq**NMN / (KMN**NMN + hq**NMN)

# Generic chronic disability weight for Fe/Cu/NO3 minor contributors
DW_GENERIC = 0.06


def hq_by_contaminant(df, params):
    ir, bw = params['ir_L_day'], params['bw_kg']
    hq = {}
    for c in CONTAMS:
        if c not in df.columns:
            continue
        conc = df[c].values.astype(float).copy()
        if c == 'As':
            conc = conc / 1000.0
        hq[c] = conc * ir / bw / REFERENCE_DOSES[c]
    return pd.DataFrame(hq, index=df.index)


def adult_as_dw():
    return (0.4 * DISABILITY_WEIGHTS['skin_lesions']
            + 0.3 * DISABILITY_WEIGHTS['peripheral_neuropathy']
            + 0.3 * DISABILITY_WEIGHTS['cardiovascular'])

def child_as_dw():
    return (0.4 * DISABILITY_WEIGHTS['neurodevelopmental']
            + 0.6 * DISABILITY_WEIGHTS['skin_lesions'])


def main():
    print("=" * 70)
    print("F6: CONTAMINANT-SPECIFIC DALY ROBUSTNESS (response-addition + full)")
    print("=" * 70)

    df = pd.read_csv(DATA_FILE)
    df = assign_zones(df)
    params = EXPOSURE_PARAMS_CORRECTED[REF_AGE]
    hq = hq_by_contaminant(df, params)
    hq['phys_zone'] = df['phys_zone']

    # cancer arm (As + Cr(VI)) — identical to main pipeline, shared by all models
    def cancer_cr(d):
        ir, bw, ed = params['ir_L_day'], params['bw_kg'], params['ed_years']
        cr = (d['As'].values / 1000.0) * ir * 365 * ed / (bw * AT_CARCINOGENIC) * CANCER_SLOPE_FACTORS['As']
        if 'Cr3+' in d.columns:
            cr = cr + (d['Cr3+'].values * CR_VI_FRACTION) * ir * 365 * ed / (bw * AT_CARCINOGENIC) * 0.5
        return cr
    df['CR'] = cancer_cr(df)

    g = pd.read_csv(TABLES_DIR / 'T2_zone_population_weighted_stats.csv').set_index('phys_zone')

    rows = []
    nat = {'as_only': 0.0, 'dose_add': 0.0, 'resp_add': 0.0, 'contam_specific': 0.0}
    for z in g.index:
        zhq = hq[hq['phys_zone'] == z]
        if len(zhq) < 5:
            continue
        pop = g.loc[z, 'population'] * GW_USAGE_FRACTION
        # Allocate the gridded pop-weighted HI_multi across contaminants by their
        # within-zone median HQ SHARE, so the per-contaminant HQs sum EXACTLY to
        # the official HI_multi_pop_p50 (=> dose-addition reproduces the headline).
        him_grid = g.loc[z, 'HI_multi_pop_p50']
        cols = [c for c in CONTAMS if c in zhq.columns]
        med = {c: float(zhq[c].median()) for c in cols}
        tot_med = sum(med.values())
        hq_med = {c: him_grid * (med[c] / tot_med) if tot_med > 0 else 0.0 for c in cols}
        hi_as = g.loc[z, 'HI_as_only_pop_p50']
        cr_med = g.loc[z, 'CR_multi_pop_p50']
        cr_as = g.loc[z, 'CR_as_only_pop_p50']

        # cancer YLL per zone (age-summed) — shared
        yll_multi = yll_as = 0.0
        for ag, frac in AGE_DISTRIBUTION.items():
            ap = pop * frac
            yll_multi += annual_yll_from_cancer(cr_med, ap, ag)[0]
            yll_as += annual_yll_from_cancer(cr_as, ap, ag)[0]

        # YLD per age group
        yld_as = yld_da = yld_ra = yld_cs = 0.0
        for ag, frac in AGE_DISTRIBUTION.items():
            ap = pop * frac
            dw_as = child_as_dw() if ag.startswith('child') else adult_as_dw()
            # As-only
            yld_as += annual_chronic_prevalence(hi_as) * ap * dw_as
            # (1) dose-addition  P(sum HQ)
            yld_da += annual_chronic_prevalence(sum(hq_med.values())) * ap * dw_as
            # (2) response-addition  sum P(HQ_i), shared curve + As DW
            yld_ra += sum(annual_chronic_prevalence(hq_med[c]) for c in cols) * ap * dw_as
            # (3) full contaminant-specific
            cs = annual_chronic_prevalence(hq_med.get('As', 0)) * dw_as           # As endpoints
            mn_suscept = 1.0 if ag.startswith('child') else MN_ADULT_SUSCEPT
            cs += p_mn_neuro(hq_med.get('Mn2+', 0)) * mn_suscept * DW_MN_NEURO     # Mn neuro
            for c in cols:
                if c in ('As', 'Mn2+'):
                    continue
                cs += annual_chronic_prevalence(hq_med[c]) * DW_GENERIC            # Fe/Cu/NO3/Al/Cr
            yld_cs += cs * ap

        daly_as = yld_as + yll_as
        daly_da = yld_da + yll_multi
        daly_ra = yld_ra + yll_multi
        daly_cs = yld_cs + yll_multi
        nat['as_only'] += daly_as
        nat['dose_add'] += daly_da
        nat['resp_add'] += daly_ra
        nat['contam_specific'] += daly_cs
        rows.append({'phys_zone': z, 'DALY_as_only': daly_as,
                     'DALY_dose_add': daly_da, 'DALY_resp_add': daly_ra,
                     'DALY_contam_specific': daly_cs})

    out = pd.DataFrame(rows)
    out.loc[len(out)] = {'phys_zone': 'NATIONAL',
                         'DALY_as_only': nat['as_only'],
                         'DALY_dose_add': nat['dose_add'],
                         'DALY_resp_add': nat['resp_add'],
                         'DALY_contam_specific': nat['contam_specific']}
    out.to_csv(TABLES_DIR / 'T2_daly_contaminant_specific.csv', index=False)

    print(f"\n{'Model':28s} {'National DALY/yr':>18s} {'multi/As':>10s}")
    print("-" * 58)
    print(f"{'Arsenic-only (baseline)':28s} {nat['as_only']:>18,.0f} {'1.00':>10s}")
    print(f"{'Dose-addition [CURRENT]':28s} {nat['dose_add']:>18,.0f} {nat['dose_add']/nat['as_only']:>9.2f}x")
    print(f"{'Response-addition':28s} {nat['resp_add']:>18,.0f} {nat['resp_add']/nat['as_only']:>9.2f}x")
    print(f"{'Full contaminant-specific':28s} {nat['contam_specific']:>18,.0f} {nat['contam_specific']/nat['as_only']:>9.2f}x")
    print(f"\n  -> saved {TABLES_DIR / 'T2_daly_contaminant_specific.csv'}")


if __name__ == '__main__':
    main()
