"""
Longitudinal (paired, two-epoch) multi-contaminant hazard-index recomputation.

Reproduces the "burden already rising" result in the main text: in the subset of
wells resurveyed across two epochs (2012-2013 baseline and a 2020-2021 resurvey)
with arsenic, manganese, iron and nitrate measured in BOTH epochs, the cumulative
hazard index (HI) is recomputed with the same adult-male exposure model used for
the cross-sectional analysis. Restricting to the four jointly-measured contaminants
(As, Mn, Fe, NO3; together ~77% of cumulative HI) makes the recomputed HI a
conservative lower bound on the true change.

Data: data/longitudinal_matched_wells_2012-2021.csv (705 matched wells x 2 epochs;
215 with all four contaminants in both epochs). Deterministic. Run from repo root.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / 'data' / 'longitudinal_matched_wells_2012-2021.csv'
OUT = ROOT / 'output' / 'tables'
OUT.mkdir(parents=True, exist_ok=True)

# Adult-male exposure model (matches the cross-sectional HRA / SI Table S2)
IR, BW = 2.5, 60.0                      # L/day, kg
RfD = {'As': 0.0003, 'Mn': 0.024, 'Fe': 0.70, 'NO3': 1.6}   # mg/kg/day
CSF_As = 1.5                            # (mg/kg/day)^-1
ED, AT_C = 30.0, 70.0 * 365.0           # cancer averaging

def hq(conc, ion, ug_per_L=False):
    c = conc / 1000.0 if ug_per_L else conc      # As reported in ug/L -> mg/L
    return c * IR / BW / RfD[ion]                 # EF*ED/AT = 1 for non-cancer

def cr_as(as_ugL):
    cdi = (as_ugL / 1000.0) * IR * 365.0 * ED / (BW * AT_C)
    return cdi * CSF_As

def main():
    d = pd.read_csv(CSV)
    parts = {}
    for per, g in d.groupby('Period'):
        g = g.set_index('pair_key')
        parts[per] = g
    old, new = parts['2012-2013'], parts['2020-2021']
    keys = old.index.intersection(new.index)

    def complete(df, k):
        r = df.loc[k]
        return np.isfinite([r['As'], r['Mn'], r['Fe'], r['NO3']]).all()
    cc = [k for k in keys if complete(old, k) and complete(new, k)]
    o, n = old.loc[cc], new.loc[cc]

    def HI(df):
        return (hq(df['As'], 'As', True) + hq(df['Mn'], 'Mn')
                + hq(df['Fe'], 'Fe') + hq(df['NO3'], 'NO3'))

    def HI_noMn(df):
        return (hq(df['As'], 'As', True) + hq(df['Fe'], 'Fe')
                + hq(df['NO3'], 'NO3'))

    HIo, HIn = HI(o), HI(n)
    CRo, CRn = cr_as(o['As']), cr_as(n['As'])
    as_safe_then_exceed = ((o['As'] <= 10) & (n['As'] > 10)).sum()
    as_safe_old = (o['As'] <= 10).sum()

    rows = [
        ('n_paired_wells', len(cc), '', ''),
        ('median_HI', round(HIo.median(), 3), round(HIn.median(), 3),
         f'Wilcoxon p={wilcoxon(HIn.values, HIo.values).pvalue:.4f}'),
        ('pct_HI_gt1', round(100*(HIo > 1).mean(), 1), round(100*(HIn > 1).mean(), 1), ''),
        ('median_HQ_As', round(hq(o['As'], 'As', True).median(), 3),
         round(hq(n['As'], 'As', True).median(), 3), ''),
        ('median_HQ_Mn', round(hq(o['Mn'], 'Mn').median(), 3),
         round(hq(n['Mn'], 'Mn').median(), 3), ''),
        ('median_HQ_Fe', round(hq(o['Fe'], 'Fe').median(), 3),
         round(hq(n['Fe'], 'Fe').median(), 3), ''),
        ('median_HQ_NO3', round(hq(o['NO3'], 'NO3').median(), 3),
         round(hq(n['NO3'], 'NO3').median(), 3), ''),
        ('median_CR_As', f'{CRo.median():.2e}', f'{CRn.median():.2e}',
         f'Wilcoxon p={wilcoxon(CRn.values, CRo.values).pvalue:.3f}'),
        ('pct_newly_exceed_WHO_As', round(100*as_safe_then_exceed/as_safe_old, 1),
         f'{as_safe_then_exceed}/{as_safe_old} safe-in-2012', ''),
    ]
    res = pd.DataFrame(rows, columns=['metric', 'epoch_2012_2013', 'epoch_2020_2021', 'test'])
    res.to_csv(OUT / 'T2_longitudinal_paired_hi.csv', index=False)
    print(res.to_string(index=False))
    print(f'\nwrote {OUT / "T2_longitudinal_paired_hi.csv"}')

    # ── Robustness: charge-balance restriction + Mn-exclusion ────────────────
    # The two epochs differ markedly in analytical quality (charge-balance pass
    # rate 35% in 2012 vs 83% in 2020), and the redox-sensitive Mn measurement is
    # the analyte most vulnerable to inter-survey filtration/preservation changes.
    # These robustness cuts test whether the apparent HI rise is real geochemical
    # change or an artefact of the protocol/quality shift.
    def subset_stats(keys_sub, label):
        if len(keys_sub) < 5:
            return None
        os_, ns_ = old.loc[keys_sub], new.loc[keys_sub]
        hi_o, hi_n = HI(os_), HI(ns_)
        hinm_o, hinm_n = HI_noMn(os_), HI_noMn(ns_)
        return {
            'subset': label,
            'n': len(keys_sub),
            'cbe_pass_2012_pct': round(100*(os_['CBE_flag'] == 'PASS').mean(), 1),
            'cbe_pass_2020_pct': round(100*(ns_['CBE_flag'] == 'PASS').mean(), 1),
            'HI4_2012': round(hi_o.median(), 3),
            'HI4_2020': round(hi_n.median(), 3),
            'HI4_wilcoxon_p': round(wilcoxon(hi_n.values, hi_o.values).pvalue, 4),
            'HI_noMn_2012': round(hinm_o.median(), 3),
            'HI_noMn_2020': round(hinm_n.median(), 3),
            'HI_noMn_wilcoxon_p': round(wilcoxon(hinm_n.values, hinm_o.values).pvalue, 4),
            'As_ugL_2012': round(os_['As'].median(), 2),
            'As_ugL_2020': round(ns_['As'].median(), 2),
            'As_wilcoxon_p': round(wilcoxon(ns_['As'].values, os_['As'].values).pvalue, 4),
            'Mn_mgL_2012': round(os_['Mn'].median(), 4),
            'Mn_mgL_2020': round(ns_['Mn'].median(), 4),
        }

    cc_cbe = [k for k in cc if old.loc[k, 'CBE_flag'] == 'PASS'
              and new.loc[k, 'CBE_flag'] == 'PASS']
    rob = [r for r in [
        subset_stats(cc, 'all paired-complete'),
        subset_stats(cc_cbe, 'charge-balance PASS in both epochs'),
    ] if r is not None]
    rob_df = pd.DataFrame(rob)
    rob_df.to_csv(OUT / 'T2_longitudinal_robustness.csv', index=False)
    print('\n── Robustness (CBE restriction + Mn-exclusion) ──')
    print(rob_df.to_string(index=False))
    print(f'\nwrote {OUT / "T2_longitudinal_robustness.csv"}')

if __name__ == '__main__':
    main()
