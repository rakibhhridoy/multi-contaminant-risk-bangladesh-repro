#!/usr/bin/env python3
"""Reproduce every number quoted in AUDIT_FINDINGS.md.

Run from the Paper2 root:   python3 RewriteJHMAFinal/audit/rerun_audits.py
Deterministic: seeded at config.RANDOM_STATE throughout.
"""
import sys, pathlib, numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config import DATA_FILE, RANDOM_STATE, assign_zones
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

T = ROOT / "output" / "tables"
line = lambda c="-": print(c*74)


def r23_cmip6():
    print("\nR2.3  CMIP6 inter-model CV: which quantity is which"); line()
    s = pd.read_csv(T/'T1_cmip6_ensemble_summary.csv')
    c = pd.read_csv(T/'T1_ensemble_deltaP_cv.csv')
    print("  CV of ABSOLUTE mean annual precipitation across models")
    print("  (= what the SUPERSEDED wetness-ratio equation propagates as the dP CV):")
    for _, r in s.iterrows():
        print(f"    {r.ssp}: {r['std']/r['mean']:.4f}   n={r.n_models}   <- SI reports these as 0.043 / 0.038")
    print("\n  CV of the FRACTIONAL CHANGE dP across models (current method, in the code):")
    for _, r in c.iterrows():
        print(f"    {r.ssp}: {r.cv:.4f}   mean={r['mean']:.4f}%  n={r.n}   <- main text reports these as 0.40 / 0.25")
    m = pd.read_csv(T/'T1_cmip6_ensemble_models.csv')
    print(f"\n  models: {m.model.nunique()} unique; per-SSP n = "
          f"{dict(m.groupby('ssp').size())}  (GFDL-ESM4 has no ssp245 on WorldClim)")


def r26_grouped_cv():
    print("\n\nR2.6  Phosphate cross-validation: random vs spatially grouped folds"); line()
    df = assign_zones(pd.read_csv(DATA_FILE))
    need = ['As','Fe2+','Mn2+','PO43-','Depth','ORP','NO3-','SO42-','Latitude','Longitude','District']
    df = df.dropna(subset=need).copy()
    df['As_T1'] = (df.As > 10).astype(int)      # WHO threshold
    df['As_T2'] = (df.As > 22).astype(int)      # GMM saddle
    df['blk'] = (np.floor(df.Latitude/0.5).astype(int).astype(str) + '_' +
                 np.floor(df.Longitude/0.5).astype(int).astype(str))
    BASE, FULL = ['Fe2+','Mn2+','Depth','ORP'], ['Fe2+','Mn2+','Depth','ORP','PO43-']
    print(f"  n={len(df)}  districts={df.District.nunique()}  0.5deg blocks={df.blk.nunique()}")

    def auc(X, y, splits):
        a = [roc_auc_score(y[te], RandomForestClassifier(
                n_estimators=500, random_state=RANDOM_STATE, class_weight='balanced',
                n_jobs=-1).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1])
             for tr, te in splits
             if len(np.unique(y[tr])) > 1 and len(np.unique(y[te])) > 1]
        return float(np.mean(a))

    for tgt, lab in [('As_T1','T1  As>10 ug/L (WHO)'), ('As_T2','T2  As>22 ug/L (GMM saddle)')]:
        y = df[tgt].values
        print(f"\n  {lab}")
        for name, sp in [
            ('random 5-fold (AS PUBLISHED)', StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE).split(df[FULL].values, y)),
            ('district-grouped 5-fold',      StratifiedGroupKFold(5, shuffle=True, random_state=RANDOM_STATE).split(df[FULL].values, y, df.District.values)),
            ('0.5deg block-grouped 5-fold',  StratifiedGroupKFold(5, shuffle=True, random_state=RANDOM_STATE).split(df[FULL].values, y, df.blk.values))]:
            sp = list(sp)
            b, f = auc(df[BASE].values, y, sp), auc(df[FULL].values, y, sp)
            print(f"    {name:30s} base {b:.3f}  full {f:.3f}  dAUC {f-b:+.3f}")


def r27_iron_benchmark():
    print("\n\nR2.7  Joint exceedance under health-based vs aesthetic benchmarks"); line()
    d = assign_zones(pd.read_csv(DATA_FILE)).dropna(subset=['As','Mn2+','Fe2+','Cr3+'])
    print(f"  n={len(d)}")
    for name, TH in [
        ('AS PUBLISHED        (Fe 0.3 mg/L, aesthetic)', {'As':10.,'Mn2+':.4,'Fe2+':.3,'Cr3+':.05}),
        ('HEALTH-BASED ONLY   (Fe 2.0 mg/L, WHO prov.)', {'As':10.,'Mn2+':.4,'Fe2+':2.,'Cr3+':.05}),
        ('Fe DROPPED entirely                         ', {'As':10.,'Mn2+':.4,'Cr3+':.05})]:
        k = sum((d[c] > t).astype(int) for c, t in TH.items())
        print(f"\n  {name}")
        for c, t in TH.items():
            print(f"      {c:5s} > {t:<5} {100*(d[c]>t).mean():5.1f}%")
        print(f"      >=1 {100*(k>=1).mean():5.1f}%   >=2 {100*(k>=2).mean():5.1f}% <-- the 57.4% claim"
              f"   >=3 {100*(k>=3).mean():5.1f}%")


def r21_aggregation():
    print("\n\nR2.1  DALY aggregation rule: multi/As ratio"); line()
    n = pd.read_csv(T/'T2_daly_contaminant_specific.csv').query("phys_zone=='NATIONAL'").iloc[0]
    a = n.DALY_as_only
    print(f"    arsenic-only baseline                {a:>10,.0f}")
    for c, lab in [('DALY_dose_add','dose-addition  P(sum HQ)  HEADLINE'),
                   ('DALY_resp_add','response-addition  sum P(HQ)      '),
                   ('DALY_contam_specific','fully contaminant-specific        ')]:
        print(f"    {lab} {n[c]:>10,.0f}   ratio {n[c]/a:.2f}x")


def r14_manganese():
    print("\n\nR1.4  Which manganese threshold produced '43%'"); line()
    d = assign_zones(pd.read_csv(DATA_FILE)).dropna(subset=['Mn2+'])
    for t, lab in [(0.4,'WHO 2011 health-based (USED IN THE PAPER)'),
                   (0.1,'aesthetic / acceptability'),
                   (0.08,'WHO 2021 background-document health-based')]:
        print(f"    Mn > {t:<5} mg/L  {100*(d['Mn2+']>t).mean():5.1f}%   {lab}")


if __name__ == '__main__':
    r23_cmip6(); r26_grouped_cv(); r27_iron_benchmark(); r21_aggregation(); r14_manganese()
    print("\n" + "="*74 + "\ndone\n")
