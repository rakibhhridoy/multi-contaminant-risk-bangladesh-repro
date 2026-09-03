#!/usr/bin/env python3
"""Export data for the twelve supplementary figures, same contract as export_data.py:
Python owns the numbers, D3 only draws them. Run from the bundle root.
"""
import json, sys, pathlib, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')

ROOT = pathlib.Path(__file__).resolve().parents[1]   # bundle root (holds config.py)
sys.path.insert(0, str(ROOT))
from config import (DATA_FILE, TABLES_DIR as T, RANDOM_STATE, HEALTH_CONTAMINANTS,
                    REFERENCE_DOSES, assign_zones)

OUT = pathlib.Path(__file__).resolve().parent / 'data'
OUT.mkdir(exist_ok=True)
np.random.seed(RANDOM_STATE)

def clean(x):
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 6)
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.bool_, bool)): return bool(x)
    return x

def recs(df):
    return [{k: clean(v) for k, v in r.items()} for r in df.to_dict('records')]

def dump(name, obj):
    p = OUT / f'{name}.json'
    p.write_text(json.dumps(obj, indent=1, allow_nan=False))
    print(f'  wrote {p.name:26s} {p.stat().st_size/1024:7.1f} kB')

df = assign_zones(pd.read_csv(DATA_FILE))
CONTAM4 = ['As', 'Mn2+', 'Fe2+', 'Cr3+']

# S1 copula
def s1():
    d = df.dropna(subset=CONTAM4)
    TH = {'As': 10., 'Mn2+': .08, 'Fe2+': 2., 'Cr3+': .05}
    exc = pd.DataFrame({c: (d[c] > TH[c]).astype(int) for c in CONTAM4})
    k = exc.sum(axis=1)
    pairs = pd.read_csv(T/'T3_pairwise_copula_full.csv')
    sc = d.sample(min(600, len(d)), random_state=RANDOM_STATE)
    dump('s1', dict(
        pairs=recs(pairs[['pair', 'best_copula', 'kendall_tau', 'p_value']]),
        tail=recs(pd.read_csv(T/'T3_tail_dependence.csv')),
        counts=[dict(k=int(i), frac=clean(float((k == i).mean()))) for i in range(5)],
        ge2=clean(float((k >= 2).mean())),
        individual=[dict(c=c, frac=clean(float((d[c] > TH[c]).mean())), thr=TH[c]) for c in CONTAM4],
        scatter=[dict(x=clean(float(a)), y=clean(float(b)))
                 for a, b in zip(sc['As'].values, sc['Fe2+'].values)],
        zones=recs(pd.read_csv(T/'T3_zone_joint_exceedance.csv')),
    ))

# S2 Monte Carlo
def s2():
    st = pd.read_csv(T/'T2_stratified_summary.csv')
    depth = (st.groupby('depth_zone')
               .apply(lambda g: pd.Series(dict(
                   n=int(g.n_samples.sum()),
                   med=float(np.average(g.HI_multi_median, weights=g.n_samples)),
                   lo=float(np.average(g.HI_multi_CI_lo, weights=g.n_samples)),
                   hi=float(np.average(g.HI_multi_CI_hi, weights=g.n_samples)),
                   exceed=float(np.average(g.HI_exceed_pct, weights=g.n_samples)))))
               .reset_index())
    dump('s2', dict(national=recs(pd.read_csv(T/'T5_mc_propagation_results.csv')),
                    zones=recs(pd.read_csv(T/'T5_mc_zone_results.csv')),
                    depth=recs(depth),
                    variance=[dict(source='transfer function', pct=41),
                              dict(source='toxicity parameters', pct=39),
                              dict(source='exposure parameters', pct=21),
                              dict(source='climate scenario', pct=1, lt=True)]))

# S3 interventions
def s3():
    iv = pd.read_csv(T/'T6_interventions_CORRECTED.csv')
    iv = iv[iv.scenario != 'S0_baseline']
    # Cost-effectiveness acceptability: share of PSA draws below each willingness
    # to pay. Built here rather than in D3 so the figure cannot re-derive it.
    ps = pd.read_csv(T/'T6_psa_iterations.csv')
    wtp = np.linspace(0, 3000, 61)
    ceac = []
    for sc, g in ps.groupby('scenario'):
        cpd = g.cost_per_daly.replace([np.inf, -np.inf], np.nan).dropna()
        if not len(cpd): continue
        ceac.append(dict(scenario=sc,
                         curve=[dict(w=float(w), p=clean(float((cpd <= w).mean()))) for w in wtp]))
    dump('s3', dict(scenarios=recs(iv), equity=recs(pd.read_csv(T/'T6_zone_equity.csv')),
                    ceac=ceac, gdp=2500, gdp3=7500))

# S4 GRACE
def s4():
    dump('s4', dict(
        grace_zone=recs(pd.read_csv(T/'T1_grace_fo_zone_summary.csv')),
        grace_ts=recs(pd.read_csv(T/'T1_grace_fo_zone_timeseries.csv')),
        tellus_trend=recs(pd.read_csv(T/'T1_tellus_zone_trend.csv')),
        grid=recs(pd.read_csv(T/'T1_grace_gridlevel_correlation.csv')),
        tellus_ts=recs(pd.read_csv(T/'T1_tellus_zone_timeseries.csv')),
        tellus_as=recs(pd.read_csv(T/'T1_tellus_arsenic_merge.csv')),
    ))

# S5 partial information
def s5():
    pi = pd.read_csv(T/'T4_partial_info_PO4.csv')
    dump('s5', dict(table=recs(pi),
                    blocked=[dict(thr='WHO 10', base=0.65, full=0.73, d=0.081, lo=0.047, hi=0.126, pgt0=100),
                             dict(thr='saddle 22', base=0.68, full=0.74, d=0.062, lo=0.020, hi=0.096, pgt0=99)],
                    random=[dict(thr='WHO 10', base=0.69, full=0.77, d=0.086),
                            dict(thr='saddle 22', base=0.73, full=0.79, d=0.059)]))

# S6 surface complexation
def s6():
    dump('s6', dict(scm=recs(pd.read_csv(T/'T4_surface_complexation.csv')),
                    titration=recs(pd.read_csv(T/'T4_phreeqc_titration.csv')), saddle=[1.5, 2.0]))

# S7 external validation
def s7():
    d = df.dropna(subset=['As', 'PO43-'])
    d = d[(d['As'] > 0) & (d['PO43-'] > 0)]
    dump('s7', dict(cohorts=recs(pd.read_csv(T/'T4_external_validation_combined.csv')),
                    bangladesh=dict(n=int(len(d)),
                                    rho=clean(float(d['As'].corr(d['PO43-'], method='spearman'))))))

# S8 tornado
def s8():
    dump('s8', dict(rows=recs(pd.read_csv(T/'T2_daly_tornado.csv')),
                    baseline=clean(float(pd.read_csv(T/'T2_daly_tornado.csv').DALY_baseline.iloc[0]))))

# S9 IDW cross-validation
def s9():
    dump('s9', dict(zone=recs(pd.read_csv(T/'T2_idw_zone_recovery.csv')),
                    cv=recs(pd.read_csv(T/'T2_idw_spatial_cv.csv'))))

# S10 between-campaign mode membership
def s10():
    CT = ['As', 'Mn2+', 'Fe2+', 'Cr3+', 'PO43-', 'NO3-']
    bis = pd.read_csv(T/'T4_bistability_all_contaminants.csv')
    ov = bis[bis.depth_zone == 'Overall'].set_index('contaminant').saddle_original.to_dict()
    both = df.groupby('Sample ID').Season.nunique()
    paired = df[df['Sample ID'].isin(set(both[both == 2].index))].drop_duplicates(['Sample ID', 'Season'])
    rows = []
    for c in CT:
        thr = ov.get(c)
        if thr is None or not np.isfinite(thr): continue
        w = paired.pivot_table(index='Sample ID', columns='Season', values=c, aggfunc='first').dropna()
        if len(w) < 30: continue
        rows.append(dict(contaminant=c, thr=clean(float(thr)), n=int(len(w)),
                         dry=clean(float((w['Dry'] > thr).mean())),
                         wet=clean(float((w['Wet'] > thr).mean()))))
    dump('s10', dict(rows=rows, n_paired=int(paired['Sample ID'].nunique())))

# S11 age-sex stratified HI
def s11():
    from config import EXPOSURE_PARAMS
    groups = {'adult_male': 'adult male', 'adult_female': 'adult female',
              'child_6_17': 'child 6-17 y', 'child_0_5': 'child 0-5 y'}
    out = []
    for key, lab in groups.items():
        p = EXPOSURE_PARAMS[key]
        hq = {}
        for c, rfd in REFERENCE_DOSES.items():
            if c not in df.columns: continue
            unit = HEALTH_CONTAMINANTS.get(c, {}).get('unit', 'mg/L')
            conc = df[c] / 1000.0 if unit.startswith('µ') else df[c]
            hq[c] = conc * p['ir_L_day'] / p['bw_kg'] / rfd
        hi = pd.DataFrame(hq).sum(axis=1).dropna()
        out.append(dict(group=lab, ir=p['ir_L_day'], bw=p['bw_kg'], n=int(len(hi)),
                        q=[clean(float(hi.quantile(q))) for q in (.05, .25, .5, .75, .95)],
                        exceed=clean(float((hi > 1).mean()))))
    dump('s11', dict(groups=out))

# S13 manganese disability-weight sensitivity (R2.1)
def s13():
    w = pd.read_csv(T/'T2_mn_weight_sensitivity.csv')
    n = pd.read_csv(T/'T2_daly_contaminant_specific.csv').query("phys_zone=='NATIONAL'").iloc[0]
    dose_add = float(n.DALY_dose_add / n.DALY_as_only)
    resp_add = float(n.DALY_resp_add / n.DALY_as_only)
    cross = w[w.ratio >= dose_add]
    dump('s13', dict(
        sweep=[dict(dw=clean(r.dw), ratio=clean(r.ratio)) for r in w.itertuples()],
        anchors=[dict(dw=0.043, label='mild intellectual disability', note='used here'),
                 dict(dw=0.100, label='borderline intellectual functioning', note=None),
                 dict(dw=0.361, label='moderate/severe neurodevelopmental', note='upper anchor')],
        dose_add=clean(dose_add), resp_add=clean(resp_add),
        crossover=clean(float(cross.iloc[0].dw)) if len(cross) else None,
    ))

# S12 per-zone projected change
def s12():
    pr = pd.read_csv(T/'T1_ensemble_projections_2050.csv')
    rows = []
    for (z, c, ssp), g in pr.groupby(['phys_zone', 'contaminant', 'ssp']):
        rows.append(dict(zone=z, contaminant=str(c), ssp=ssp,
                         pct=clean(float(g.pct_change_median.median()))))
    dump('s12', dict(rows=rows))

if __name__ == '__main__':
    print('exporting supplementary figure data ...')
    for f in (s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13):
        try: f()
        except Exception as e:
            print(f'  !! {f.__name__} FAILED: {type(e).__name__}: {e}')
    print('done')
