#!/usr/bin/env python3
"""Export every quantity the D3 figures draw, as JSON.

Python stays the computation layer: it owns the data, the interpolation and the
statistics, exactly as before. The D3 layer only draws what is in these files, so
a figure can never disagree with the analysis, and the JSON is auditable on its
own. Run from the bundle root:

    python3 figures_d3/export_data.py

Deterministic; seeded at config.RANDOM_STATE where any sampling occurs.
"""
import json, sys, pathlib, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')

ROOT = pathlib.Path(__file__).resolve().parents[1]   # bundle root (holds config.py)
sys.path.insert(0, str(ROOT))
from config import (DATA_FILE, TABLES_DIR, RANDOM_STATE, BANGLADESH_BBOX,
                    HEALTH_CONTAMINANTS, REFERENCE_DOSES, assign_zones)

OUT = pathlib.Path(__file__).resolve().parent / 'data'
OUT.mkdir(exist_ok=True)
np.random.seed(RANDOM_STATE)

def dump(name, obj):
    p = OUT / f'{name}.json'
    p.write_text(json.dumps(obj, indent=1, allow_nan=False))
    print(f'  wrote {p.name:34s} {p.stat().st_size/1024:7.1f} kB')

def clean(x):
    """numpy -> json-safe, dropping non-finite."""
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 6)
    if isinstance(x, (np.integer,)): return int(x)
    return x

df = assign_zones(pd.read_csv(DATA_FILE))
T = TABLES_DIR

# ---------------------------------------------------------------- FIGURE 1
# Screening performance: how the phosphate rule behaves as the cut-off moves,
# and what it costs relative to an arsenic assay.
def fig1():
    d = df.dropna(subset=['As', 'PO43-'])
    hi = d['As'] > 10
    grid = np.round(np.arange(0.2, 5.01, 0.05), 2)
    sweep = []
    for t in grid:
        flag = d['PO43-'] > t
        tp = int((flag & hi).sum()); fp = int((flag & ~hi).sum())
        fn = int((~flag & hi).sum()); tn = int((~flag & ~hi).sum())
        if tp + fn == 0 or tn + fp == 0: continue
        sweep.append(dict(thr=float(t),
                          sens=clean(tp/(tp+fn)), spec=clean(tn/(tn+fp)),
                          ppv=clean(tp/(tp+fp)) if tp+fp else None,
                          npv=clean(tn/(tn+fn)) if tn+fn else None,
                          flagged=clean(int(flag.sum())/len(d)),
                          missed=clean(fn/(tp+fn))))
    ops = pd.read_csv(T/'T4_po4_screening_operating_characteristics.csv')
    dump('fig1', dict(
        n=int(len(d)),
        prevalence=clean(float(hi.mean())),
        sweep=sweep,
        marks=[dict(thr=1.5, label='1.5 mg/L'), dict(thr=2.0, label='2.0 mg/L')],
        table=[{k: clean(v) for k, v in r.items()} for r in ops.to_dict('records')],
        cost=dict(arsenic=[5, 15], phosphate=[0.30, 1.0], unit='USD per test'),
    ))

# ---------------------------------------------------------------- FIGURE 2
# Concentration regimes and phosphate control.
def fig2():
    CT = ['As', 'Mn2+', 'Fe2+', 'Cr3+', 'PO43-', 'NO3-']
    dens = {}
    for c in CT:
        v = df[c].dropna(); v = v[v > 0]
        if len(v) < 30: continue
        # Arsenic is recorded in ug/L and the rest in mg/L. Put every series on
        # mg/L before taking logs, so the shared axis means one thing.
        if HEALTH_CONTAMINANTS.get(c, {}).get('unit', 'mg/L').startswith('\u00b5'):
            v = v / 1000.0
        lv = np.log10(v)
        xs = np.linspace(lv.min(), lv.max(), 220)
        # Gaussian KDE, Scott's rule; deterministic
        bw = 1.06 * lv.std() * len(lv) ** (-1/5)
        ys = np.exp(-0.5*((xs[:,None]-lv.values[None,:])/bw)**2).sum(1)/(len(lv)*bw*np.sqrt(2*np.pi))
        dens[c] = dict(x=[clean(a) for a in xs], y=[clean(a) for a in ys],
                       n=int(len(v)), median=clean(float(v.median())))
    # phosphate-conditioned arsenic (the empirical Fig 2d relationship)
    d = df.dropna(subset=['As','PO43-'])
    lo, hi = d['PO43-'].quantile([.05,.95]); edges = np.linspace(lo,hi,9)
    bins=[]
    for i in range(8):
        m = (d['PO43-']>=edges[i]) & (d['PO43-']<edges[i+1])
        if m.sum() < 15: continue
        sub = d.loc[m,'As']
        bins.append(dict(x=clean(float((edges[i]+edges[i+1])/2)), n=int(m.sum()),
                         frac=clean(float((sub>10).mean())),
                         med=clean(float(sub.median())),
                         q1=clean(float(sub.quantile(.25))), q3=clean(float(sub.quantile(.75)))))
    # Conditional co-occurrence, taken from the analysis table rather than
    # recomputed here, so the heatmap shows the same P(As high | PO4 high) = 0.48
    # the main text quotes. Recomputing on quartiles gave different values.
    cas = pd.read_csv(T/'T4_cascade_conditional.csv')
    pairs = [dict(src=r.contaminant_i, dst=r.contaminant_j,
                  p=clean(float(r.P_j_high_given_i_high)),
                  sig=bool(pd.notna(r.chi2_pvalue) and r.chi2_pvalue < 0.05))
             for r in cas.itertuples()
             if r.contaminant_i != r.contaminant_j
             and r.contaminant_i in CT and r.contaminant_j in CT]

    # Per-depth GMM antimodes, which panel (b) reports and the text cites for the
    # deep zone (0.14-0.18 ug/L).
    bis = pd.read_csv(T/'T4_bistability_all_contaminants.csv')
    anti = [dict(contaminant=r.contaminant, depth=r.depth_zone,
                 saddle=clean(float(r.saddle_original)), n=int(r.n_samples),
                 bimodal=bool(r.bimodal))
            for r in bis.itertuples() if pd.notna(r.saddle_original)]
    depth_as = []
    for dz, g in df.dropna(subset=['As']).groupby('depth_zone'):
        v = g['As'][g['As'] > 0]
        if len(v) < 20: continue
        depth_as.append(dict(depth=dz, n=int(len(v)),
                             q=[clean(float(v.quantile(q))) for q in (.05,.25,.5,.75,.95)],
                             frac_over=clean(float((v > 10).mean()))))
    # As-PO4 phase scatter, sampled for legibility. itertuples() mangles the
    # 'PO43-' column name, so index the frame directly.
    sc = d[(d['As'] > 0) & (d['PO43-'] > 0)].sample(min(700, len(d)), random_state=RANDOM_STATE)
    scatter = [dict(p=clean(float(a)), a=clean(float(b)))
               for a, b in zip(sc['PO43-'].values, sc['As'].values)]

    dump('fig2', dict(density=dens, po4_bins=bins, saddle=[1.5,2.0], scatter=scatter,
                      antimodes=anti, depth_as=depth_as,
                      who=10, gmm_saddle=22, pairs=pairs, contaminants=CT))

# ---------------------------------------------------------------- FIGURE 3
def fig3():
    z = pd.read_csv(T/'T2_daly_by_zone_CORRECTED.csv')
    contrib = pd.read_csv(T/'T2_contaminant_contributions.csv')
    contrib.columns = ['contaminant','pct']
    agg = pd.read_csv(T/'T2_daly_contaminant_specific.csv')
    # per-sample multi vs arsenic-only HI
    d = df.copy()
    hq = {}
    for c, rfd in REFERENCE_DOSES.items():
        if c in d.columns:
            unit = HEALTH_CONTAMINANTS.get(c, {}).get('unit', 'mg/L')
            conc = d[c] / 1000.0 if unit.startswith('µg') else d[c]
            hq[c] = conc * 2.5 / 60.0 / rfd
    H = pd.DataFrame(hq)
    d['HI_multi'] = H.sum(axis=1)
    d['HI_as'] = H['As'] if 'As' in H else np.nan
    s = d.dropna(subset=['HI_multi','HI_as']).sample(min(900, len(d)), random_state=RANDOM_STATE)
    dump('fig3', dict(
        zones=[{k: clean(v) for k, v in r.items()} for r in z.to_dict('records')],
        contributions=[{'contaminant': r.contaminant, 'pct': clean(r.pct)}
                       for r in contrib.itertuples()],
        aggregation=[{k: clean(v) for k, v in r.items()} for r in agg.to_dict('records')],
        scatter=[dict(a=clean(float(r.HI_as)), m=clean(float(r.HI_multi)))
                 for r in s.itertuples() if np.isfinite(r.HI_as) and np.isfinite(r.HI_multi)],
    ))

# ---------------------------------------------------------------- FIGURE 4
def fig4():
    tf = pd.read_csv(T/'T1_seasonal_transfer.csv')
    pr = pd.read_csv(T/'T1_ensemble_projections_2050.csv')
    cv = pd.read_csv(T/'T1_ensemble_deltaP_cv.csv')
    a = (pr[(pr.contaminant=='As') & (pr.ssp=='ssp585')]
         .groupby(['phys_zone','depth_zone'])
         .agg(base=('baseline_dry','first'), med=('ensemble_median','median'),
              lo=('ensemble_median','min'), hi=('ensemble_median','max')).reset_index())
    # National change must use the SAME statistic the manuscript quotes: the
    # median across zone-depth cells of the per-cell percentage change, with the
    # p5/p95 medians as the inter-model range. An aggregate sum-ratio over the
    # same table gives +52% where the text says +74%, because low-baseline cells
    # carry large percentage changes but little absolute mass.
    nat = []
    for c in ['As','Mn2+','Fe2+','Cr3+','NO3-']:
        for ssp in ['ssp245','ssp585']:
            x = pr[(pr.contaminant.astype(str) == c) & (pr.ssp == ssp)]
            if not len(x): continue
            nat.append(dict(contaminant=c, ssp=ssp,
                            pct=clean(float(x.pct_change_median.median())),
                            lo=clean(float(x.pct_change_p5.median())),
                            hi=clean(float(x.pct_change_p95.median()))))
    dump('fig4', dict(
        sensitivity=[{k: clean(v) for k, v in r.items()}
                     for r in tf[['depth_zone','phys_zone','contaminant',
                                  'sensitivity_per_pct','significant','n_pairs']].to_dict('records')],
        crossings=[{k: clean(v) for k, v in r.items()} for r in a.to_dict('records')],
        national=nat, who=10,
        cv=[{k: clean(v) for k, v in r.items()} for r in cv.to_dict('records')],
    ))

# ---------------------------------------------------------------- FIGURE 5
# IDW surfaces + the grid the contours are traced from.
def fig5():
    import os; os.environ['SHAPE_RESTORE_SHX'] = 'YES'
    import geopandas as gpd
    from scipy.spatial import cKDTree
    from shapely.geometry import Point
    bb = BANGLADESH_BBOX
    bd = gpd.read_file(ROOT/'data/gis/bgd_admbnda_adm0_bbs_20201113.shp')
    geom = bd.unary_union

    d = df.copy()
    hq = {}
    for c, rfd in REFERENCE_DOSES.items():
        if c in d.columns:
            unit = HEALTH_CONTAMINANTS.get(c, {}).get('unit','mg/L')
            conc = d[c]/1000.0 if unit.startswith('µg') else d[c]
            hq[c] = conc*2.5/60.0/rfd
    d['HI'] = pd.DataFrame(hq).sum(axis=1)
    d['ASX'] = (d['As'] > 10).astype(float)

    tf = pd.read_csv(T/'T1_seasonal_transfer.csv')
    sens = (tf[tf.contaminant=='As'].groupby('phys_zone').sensitivity_per_pct.mean())
    smax = sens.abs().max() or 1.0
    d['VULN'] = d['HI'] * d['phys_zone'].map(sens).abs().fillna(0) / smax

    N = 200
    gx = np.linspace(bb['lon_min'], bb['lon_max'], N)
    gy = np.linspace(bb['lat_min'], bb['lat_max'], N)
    GX, GY = np.meshgrid(gx, gy)
    pts = np.c_[GX.ravel(), GY.ravel()]
    inside = np.array([geom.contains(Point(p)) for p in pts])

    panels = {}
    for key, col, lab in [('asx','ASX','fraction above WHO As guideline'),
                          ('hi','HI','cumulative hazard index'),
                          ('vuln','VULN','climate-vulnerability index')]:
        s = d.dropna(subset=[col,'Latitude','Longitude'])
        tree = cKDTree(np.c_[s.Longitude, s.Latitude])
        dist, idx = tree.query(pts, k=min(10, len(s)))
        dist = np.maximum(dist, 1e-12); w = 1/dist**2
        Z = (w*s[col].values[idx]).sum(1)/w.sum(1)
        Z[~inside] = np.nan
        finite = Z[np.isfinite(Z)]
        panels[key] = dict(label=lab,
                           values=[None if not np.isfinite(v) else round(float(v),5) for v in Z],
                           vmin=clean(float(np.percentile(finite,2))),
                           vmax=clean(float(np.percentile(finite,98))))
    # The BBS boundary carries ~38,000 vertices in its main ring. Each panel draws
    # it twice (clip path and visible outline), so at full resolution it dominated
    # the SVG and produced a 72 MB PDF. Simplify to a tolerance that is well below
    # one printed pixel at the 168 pt panel width: 0.004 deg is roughly 400 m,
    # against about 3 km per point on the page.
    outline = []
    polys = list(geom.geoms) if geom.geom_type=='MultiPolygon' else [geom]
    for poly in sorted(polys, key=lambda g: -g.area)[:6]:
        simp = poly.simplify(0.004, preserve_topology=True)
        outline.append([[round(x,4), round(y,4)] for x, y in simp.exterior.coords])
    dump('fig5', dict(nx=N, ny=N, bbox=bb, panels=panels, outline=outline,
                      wells=[dict(x=round(float(r.Longitude),4), y=round(float(r.Latitude),4))
                             for r in d.dropna(subset=['Latitude','Longitude'])
                                       .sample(min(420,len(d)), random_state=RANDOM_STATE).itertuples()]))

# ---------------------------------------------------------------- FIGURE 6
def fig6():
    z = pd.read_csv(T/'T2_daly_by_zone_CORRECTED.csv')
    contrib = pd.read_csv(T/'T2_contaminant_contributions.csv'); contrib.columns=['contaminant','pct']
    dump('fig6', dict(
        contributions=[{'contaminant': r.contaminant, 'pct': clean(r.pct)}
                       for r in contrib.itertuples() if r.pct > 0.5],
        zones=[dict(zone=r.phys_zone, daly=clean(float(r.annual_DALY_multi)),
                    daly_as=clean(float(r.annual_DALY_as_only)))
               for r in z.itertuples()],
        total=clean(float(z.annual_DALY_multi.sum())),
        # Variance decomposition for projected HI at 2050 (Supplementary Table S5).
        # One-at-a-time: each source fixed at its median in turn, the reduction in
        # projected-HI variance recorded. Shares do not sum to 100 by construction.
        variance=[dict(source='transfer function', pct=41),
                  dict(source='toxicity parameters', pct=39),
                  dict(source='exposure parameters', pct=21),
                  dict(source='climate scenario', pct=1, lt=True)],
    ))

if __name__ == '__main__':
    print('exporting figure data ...')
    for f in (fig1, fig2, fig3, fig4, fig5, fig6):
        try: f()
        except Exception as e:
            print(f'  !! {f.__name__} FAILED: {type(e).__name__}: {e}')
    print('done')
