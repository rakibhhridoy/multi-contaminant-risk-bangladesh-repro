"""
T2: Spatial cross-validation of the IDW exposure surface  [added 2026-05-24]
============================================================================
Reviewer concern addressed: "the gridded DALY rests on inverse-distance
interpolation of HI from 988 locations to every WorldPop cell, but the
interpolation error is never quantified."

We validate the SAME IDW operator used by spatial_aggregation.py (k=10, p=2)
against held-out observations, on the sample-level multi-contaminant HI:

  (1) Random 10-fold CV   -- standard predictive skill.
  (2) Spatial-block CV    -- leave-one-0.5-deg-block-out: training set excludes
      ALL samples in the held-out block, so co-located (multi-season) samples
      cannot leak. This is the honest test of spatial extrapolation and is the
      headline metric.

Metrics on natural HI and on log10(HI): R^2, RMSE, MAE, mean bias, Spearman rho.

Outputs:
  output/tables/T2_idw_spatial_cv.csv
  Draft/STOTENSubmission/figures_png/figS9_idw_cv.png  (+ tiff)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy import stats

from config import DATA_FILE, RANDOM_STATE, assign_zones, FIGURES_DIR
from daly_estimation_corrected import EXPOSURE_PARAMS_CORRECTED, calculate_hi_cr

K, P = 10, 2


def idw_predict(train_xy, train_val, query_xy, k=K, p=P):
    """IDW prediction at query points from a training set (k nn, weight 1/d^p)."""
    k_eff = min(k, len(train_val))
    tree = cKDTree(train_xy)
    d, idx = tree.query(query_xy, k=k_eff)
    if k_eff == 1:
        d = d[:, None]; idx = idx[:, None]
    d = np.where(d < 1e-9, 1e-9, d)
    w = 1.0 / d ** p
    return np.sum(w * train_val[idx], axis=1) / np.sum(w, axis=1)


def metrics(obs, pred):
    resid = pred - obs
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((obs - obs.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    rmse = np.sqrt(np.mean(resid ** 2))
    mae = np.mean(np.abs(resid))
    bias = np.mean(resid)
    rho, _ = stats.spearmanr(obs, pred)
    return dict(R2=r2, RMSE=rmse, MAE=mae, bias=bias, spearman=rho)


def main():
    print("=" * 68)
    print("Spatial cross-validation of the IDW(HI) exposure surface")
    print("=" * 68)

    df = pd.read_csv(DATA_FILE)
    df = assign_zones(df)
    params = EXPOSURE_PARAMS_CORRECTED['adult_male']
    hi_multi, _, _, _ = calculate_hi_cr(df, 'adult_male', params)
    df = df.reset_index(drop=True)
    xy = df[['Latitude', 'Longitude']].values
    hi = np.asarray(hi_multi)
    loghi = np.log10(hi)
    n = len(df)
    print(f"n = {n} retained samples; {df.groupby(['Latitude','Longitude']).ngroups} unique locations")

    rng = np.random.default_rng(RANDOM_STATE)

    # ── (1) Random 10-fold CV ───────────────────────────────────────────────
    folds = rng.integers(0, 10, size=n)
    pred_rand = np.empty(n)
    for f in range(10):
        te = folds == f
        tr = ~te
        pred_rand[te] = idw_predict(xy[tr], hi[tr], xy[te])
    m_rand = metrics(hi, pred_rand)
    m_rand_log = metrics(loghi, np.log10(np.maximum(pred_rand, 1e-6)))

    # ── (2) Spatial-block CV (leave-one-0.5deg-block-out) ───────────────────
    blk = (np.floor(df['Latitude'].values / 0.5).astype(int) * 1000
           + np.floor(df['Longitude'].values / 0.5).astype(int))
    ublk = np.unique(blk)
    pred_blk = np.empty(n)
    for b in ublk:
        te = blk == b
        tr = ~te
        pred_blk[te] = idw_predict(xy[tr], hi[tr], xy[te])
    m_blk = metrics(hi, pred_blk)
    m_blk_log = metrics(loghi, np.log10(np.maximum(pred_blk, 1e-6)))
    print(f"Spatial blocks (0.5 deg): {len(ublk)}")

    # ── (3) Zone-level recovery: the quantity that actually enters the DALY ──
    # The DALY uses the per-zone (population-weighted) MEDIAN HI, not individual
    # wells. Test whether block-CV-predicted held-out wells recover each zone's
    # observed median HI. This is the decision-relevant validation.
    zdf = pd.DataFrame({'zone': df['phys_zone'].values, 'obs': hi, 'pred': pred_blk})
    zg = zdf.groupby('zone').agg(obs_med=('obs', 'median'),
                                 pred_med=('pred', 'median'),
                                 n=('obs', 'size')).reset_index()
    zg['abs_pct_err'] = 100 * np.abs(zg['pred_med'] - zg['obs_med']) / zg['obs_med']
    r_zone, p_zone = stats.pearsonr(zg['obs_med'], zg['pred_med'])
    mape_zone = zg['abs_pct_err'].mean()
    bias_zone = 100 * (zg['pred_med'] - zg['obs_med']).sum() / zg['obs_med'].sum()
    print(f"\n--- Zone-median recovery under spatial-block CV (decision-relevant) ---")
    print(zg.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    print(f"  zone-median obs-vs-pred  r={r_zone:+.3f} (p={p_zone:.3g}, n={len(zg)});  "
          f"mean |%err|={mape_zone:.1f}%;  net bias={bias_zone:+.1f}%")

    # ── (4) Block jackknife: stability of national median HI ────────────────
    jack = []
    for b in ublk:
        keep = blk != b
        jack.append(np.median(hi[keep]))
    jack = np.array(jack)
    print(f"  national median HI block-jackknife: {np.median(jack):.3f} "
          f"(min {jack.min():.3f}, max {jack.max():.3f}, "
          f"CV {100*jack.std()/jack.mean():.1f}%)")

    rows = [
        {'scheme': 'random_10fold', 'space': 'HI', **m_rand},
        {'scheme': 'random_10fold', 'space': 'log10_HI', **m_rand_log},
        {'scheme': 'spatial_block_0.5deg', 'space': 'HI', **m_blk},
        {'scheme': 'spatial_block_0.5deg', 'space': 'log10_HI', **m_blk_log},
        {'scheme': 'zone_median_recovery_blockCV', 'space': 'HI',
         'R2': np.nan, 'RMSE': np.nan, 'MAE': mape_zone, 'bias': bias_zone,
         'spearman': r_zone},
    ]
    out = pd.DataFrame(rows)
    from config import TABLES_DIR
    out.to_csv(TABLES_DIR / 'T2_idw_spatial_cv.csv', index=False)
    zg.to_csv(TABLES_DIR / 'T2_idw_zone_recovery.csv', index=False)

    def show(tag, m):
        print(f"  {tag:32s} R2={m['R2']:+.3f}  RMSE={m['RMSE']:.3f}  "
              f"MAE={m['MAE']:.3f}  bias={m['bias']:+.3f}  rho={m['spearman']:+.3f}")
    print("\n--- Predictive skill of IDW(HI), k=10 p=2 ---")
    show("random 10-fold (HI)", m_rand)
    show("random 10-fold (log10 HI)", m_rand_log)
    show("spatial-block 0.5deg (HI)", m_blk)
    show("spatial-block 0.5deg (log10 HI)", m_blk_log)

    # ── Figure: decision-relevant validation ───────────────────────────────
    # Panel A: across-zone median recovery (the quantity that enters the DALY).
    # Panel B: block-jackknife stability of the national median HI.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    ax.scatter(zg['obs_med'], zg['pred_med'], s=70, c='#c0392b',
               alpha=0.85, edgecolor='white', zorder=3)
    for _, r in zg.iterrows():
        ax.annotate(r['zone'].replace('_', ' '), (r['obs_med'], r['pred_med']),
                    xytext=(4, 3), textcoords='offset points', fontsize=7.5)
    lim = [0, max(zg['obs_med'].max(), zg['pred_med'].max()) * 1.15]
    ax.plot(lim, lim, 'k--', lw=1, alpha=0.6)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('Observed zone median HI')
    ax.set_ylabel('Block-CV predicted zone median HI')
    ax.set_title(f"Zone-median recovery (leave-block-out)\n"
                 f"r = {r_zone:.2f}, p = {p_zone:.2g}, n = {len(zg)}",
                 fontsize=10, fontweight='bold')

    ax = axes[1]
    ax.hist(jack, bins=18, color='#4c72b0', alpha=0.85, edgecolor='white')
    ax.axvline(np.median(jack), color='#c0392b', ls='--', lw=1.3,
               label=f'median {np.median(jack):.2f} (CV {100*jack.std()/jack.mean():.1f}%)')
    ax.set_xlabel('National median HI (leave-one-block-out)')
    ax.set_ylabel('Number of jackknife replicates')
    ax.set_title('Block-jackknife stability of the national burden',
                 fontsize=10, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, frameon=False)
    fig.tight_layout()

    fig.savefig(FIGURES_DIR / 'figS9_idw_cv.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'figS9_idw_cv.tiff', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved T2_idw_spatial_cv.csv and figS9_idw_cv.png/.tiff")
    print("=" * 68)


if __name__ == '__main__':
    main()
