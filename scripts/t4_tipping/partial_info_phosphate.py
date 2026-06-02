"""
Partial-information test: does dissolved phosphate carry information about the
arsenic mode that other redox proxies (Fe, Mn, depth, ORP) do not?

Motivation: a skeptical reviewer can argue that PO4 and As both originate from
organic-rich reducing sediments, so their cross-sectional correlation is just
co-occurrence — phosphate isn't "predicting" arsenic; both are downstream of
the same redox state. The phosphate-as-EWS claim requires phosphate to add
information *beyond* what other measurable redox proxies already provide.

Methodology
-----------
Target: arsenic mode (high vs low) defined two ways for robustness:
  (T1) As > 10 µg/L (WHO guideline)
  (T2) As > 22 µg/L (GMM-derived saddle in shallow zone)

Predictors:
  Base    = {Fe2+, Mn2+, Depth, ORP}             # redox proxies WITHOUT PO4
  Full    = Base + {PO4}                         # add PO4
  Anchor  = {Fe2+, Mn2+, Depth, ORP, NO3-, SO4}  # broader redox context

Models: RandomForestClassifier (n=500, default), 5-fold stratified CV
Metrics:
  - ROC-AUC for each feature set
  - ΔAUC = AUC(Full) - AUC(Base) ; bootstrap 95% CI
  - Permutation feature importance (drop-column variant)
  - Conditional mutual information I(As_mode ; PO4 | Base) by binning

Decision rule: phosphate-as-EWS claim is supported if BOTH:
  (a) ΔAUC > 0 with bootstrap CI lower bound > 0
  (b) PO4 ranks in top-3 by permutation importance when in Full
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_classif
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import DATA_FILE, TABLES_DIR, FIGURES_DIR, RANDOM_STATE, assign_zones

RNG = np.random.default_rng(RANDOM_STATE)
N_BOOT = 500


def load_data():
    df = pd.read_csv(DATA_FILE)
    df = assign_zones(df)
    # Required columns
    needed = ['As', 'Fe2+', 'Mn2+', 'PO43-', 'Depth', 'ORP', 'NO3-', 'SO42-']
    df = df.dropna(subset=needed).copy()
    df['As_T1'] = (df['As'] > 10).astype(int)   # WHO
    df['As_T2'] = (df['As'] > 22).astype(int)   # saddle
    return df


def cv_auc(df, features, target, n_splits=5, seed=RANDOM_STATE):
    X = df[features].values
    y = df[target].values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    for train, test in skf.split(X, y):
        rf = RandomForestClassifier(n_estimators=500, random_state=seed,
                                    class_weight='balanced', n_jobs=-1)
        rf.fit(X[train], y[train])
        p = rf.predict_proba(X[test])[:, 1]
        aucs.append(roc_auc_score(y[test], p))
    return np.mean(aucs), np.array(aucs)


def bootstrap_delta_auc(df, base_feats, full_feats, target, n_boot=N_BOOT):
    """Bootstrap CI on ΔAUC = AUC(Full) - AUC(Base), out-of-bag (no leakage).

    Earlier this ran 5-fold CV INSIDE each bootstrap resample, so duplicated rows
    could land in both train and test folds and optimistically bias ΔAUC. Here we
    fit on the in-bag rows and evaluate AUC on the out-of-bag rows (.632-style),
    which keeps train and test disjoint within every replicate.
    """
    deltas = np.empty(n_boot)
    n = len(df)
    X_base = df[base_feats].values
    X_full = df[full_feats].values
    y = df[target].values
    for b in range(n_boot):
        idx = RNG.choice(n, n, replace=True)
        oob = np.setdiff1d(np.arange(n), np.unique(idx))
        if len(oob) < 20 or len(np.unique(y[idx])) < 2 or len(np.unique(y[oob])) < 2:
            deltas[b] = np.nan
            continue
        rf_b = RandomForestClassifier(n_estimators=300, random_state=b,
                                      class_weight='balanced', n_jobs=-1)
        rf_f = RandomForestClassifier(n_estimators=300, random_state=b,
                                      class_weight='balanced', n_jobs=-1)
        rf_b.fit(X_base[idx], y[idx])
        rf_f.fit(X_full[idx], y[idx])
        a_base = roc_auc_score(y[oob], rf_b.predict_proba(X_base[oob])[:, 1])
        a_full = roc_auc_score(y[oob], rf_f.predict_proba(X_full[oob])[:, 1])
        deltas[b] = a_full - a_base
    return deltas


def operating_characteristics(df):
    """Sensitivity/specificity/PPV/NPV of a dissolved-PO4 threshold rule for
    flagging high-arsenic wells -- the actual screening claim. Reported at the
    proposed field thresholds (1.5 and 2.0 mg/L) against both arsenic-mode
    definitions (WHO 10 ug/L and the GMM saddle 22 ug/L)."""
    rows = []
    po4 = df['PO43-'].values
    for thr in (1.5, 2.0):
        for as_thr, lbl in ((10, 'WHO 10'), (22, 'saddle 22')):
            pred = po4 >= thr
            pos = df['As'].values > as_thr
            tp = int((pred & pos).sum()); fp = int((pred & ~pos).sum())
            fn = int((~pred & pos).sum()); tn = int((~pred & ~pos).sum())
            sens = tp / (tp + fn) if (tp + fn) else np.nan
            spec = tn / (tn + fp) if (tn + fp) else np.nan
            ppv = tp / (tp + fp) if (tp + fp) else np.nan
            npv = tn / (tn + fn) if (tn + fn) else np.nan
            rows.append({'PO4_threshold_mgL': thr, 'As_mode': lbl,
                         'prevalence': round(pos.mean(), 3),
                         'flagged_frac': round(pred.mean(), 3),
                         'sensitivity': round(sens, 3), 'specificity': round(spec, 3),
                         'PPV': round(ppv, 3), 'NPV': round(npv, 3)})
    return pd.DataFrame(rows)


def conditional_mi(df, target, base_feats, extra_feat, n_bins=5):
    """Approximate I(target ; extra_feat | base_feats) by binning base_feats.

    Computes MI(target ; extra_feat) within each stratum of binned base_feats,
    then averages weighted by stratum size.
    """
    # Stratify base features by quantile bins (1D index)
    strata = np.zeros(len(df), dtype=int)
    multiplier = 1
    for f in base_feats:
        b = pd.qcut(df[f].rank(method='first'), n_bins, labels=False)
        strata = strata + b.values * multiplier
        multiplier *= n_bins
    # Inside each stratum, compute MI(target ; extra_feat)
    total_mi = 0.0
    total_n = 0
    for s in np.unique(strata):
        mask = strata == s
        if mask.sum() < 30 or df.loc[mask, target].nunique() < 2:
            continue
        mi = mutual_info_classif(
            df.loc[mask, [extra_feat]].values,
            df.loc[mask, target].values,
            random_state=RANDOM_STATE,
        )[0]
        total_mi += mi * mask.sum()
        total_n += mask.sum()
    return total_mi / total_n if total_n > 0 else np.nan


def run_test(df, target_label, target_col):
    BASE = ['Fe2+', 'Mn2+', 'Depth', 'ORP']
    FULL = BASE + ['PO43-']
    ANCHOR = BASE + ['NO3-', 'SO42-']

    print(f"\n=== Target: {target_label} ({target_col}, prevalence "
          f"{df[target_col].mean():.2%}) ===")

    auc_base, _ = cv_auc(df, BASE, target_col)
    auc_full, _ = cv_auc(df, FULL, target_col)
    auc_anch, _ = cv_auc(df, ANCHOR, target_col)
    auc_anch_full, _ = cv_auc(df, ANCHOR + ['PO43-'], target_col)
    print(f"  AUC(Fe,Mn,Depth,ORP)            = {auc_base:.4f}")
    print(f"  AUC(Fe,Mn,Depth,ORP,PO4)        = {auc_full:.4f}  Δ={auc_full-auc_base:+.4f}")
    print(f"  AUC(Fe,Mn,Depth,ORP,NO3,SO4)    = {auc_anch:.4f}")
    print(f"  AUC(Fe,Mn,Depth,ORP,NO3,SO4,PO4)= {auc_anch_full:.4f}  Δ={auc_anch_full-auc_anch:+.4f}")

    print("  Bootstrapping ΔAUC (200 iters)...")
    deltas = bootstrap_delta_auc(df, BASE, FULL, target_col, n_boot=200)
    deltas = deltas[np.isfinite(deltas)]
    lo, med, hi = np.percentile(deltas, [2.5, 50, 97.5])
    print(f"  ΔAUC bootstrap: median={med:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    sig = "YES" if lo > 0 else ("MARGINAL" if med > 0 else "NO")
    print(f"  PO4 adds information beyond base: {sig}")

    # Permutation importance with PO4 in
    rf = RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE,
                                class_weight='balanced', n_jobs=-1)
    rf.fit(df[FULL].values, df[target_col].values)
    pi = permutation_importance(rf, df[FULL].values, df[target_col].values,
                                 n_repeats=30, random_state=RANDOM_STATE,
                                 n_jobs=-1)
    perm_table = pd.DataFrame({
        'feature': FULL,
        'importance_mean': pi.importances_mean,
        'importance_std': pi.importances_std,
    }).sort_values('importance_mean', ascending=False)
    print(f"  Permutation importance (Full model):")
    for _, r in perm_table.iterrows():
        flag = " <-- PO4" if r['feature'] == 'PO43-' else ""
        print(f"    {r['feature']:8s}: {r['importance_mean']:+.4f} ± "
              f"{r['importance_std']:.4f}{flag}")
    po4_rank = (perm_table['feature'] == 'PO43-').values.argmax() + 1
    po4_top3 = po4_rank <= 3
    print(f"  PO4 permutation rank: {po4_rank} of {len(FULL)} "
          f"({'top-3 ✓' if po4_top3 else 'not top-3 ✗'})")

    # Conditional MI: I(target ; PO4 | Fe,Mn,Depth,ORP)
    cmi = conditional_mi(df, target_col, BASE, 'PO43-', n_bins=3)
    print(f"  Conditional MI I({target_col} ; PO4 | base) ≈ {cmi:.4f} nats")

    decision = (lo > 0) and po4_top3
    print(f"\n  DECISION: phosphate adds information beyond redox proxies = "
          f"{'YES — claim supported' if decision else 'NO — claim NOT supported'}")

    return {
        'target': target_label,
        'auc_base': auc_base, 'auc_full': auc_full,
        'auc_anchor': auc_anch, 'auc_anchor_full': auc_anch_full,
        'delta_auc_median': med, 'delta_auc_lo': lo, 'delta_auc_hi': hi,
        'po4_perm_rank': int(po4_rank),
        'po4_perm_importance': float(perm_table.loc[
            perm_table['feature'] == 'PO43-', 'importance_mean'].iloc[0]),
        'conditional_mi': cmi,
        'decision_supported': bool(decision),
        'perm_table': perm_table,
    }


def main():
    print("=" * 75)
    print("Partial-information test for phosphate-as-EWS claim")
    print("=" * 75)
    df = load_data()
    print(f"Loaded {len(df)} samples with complete redox + PO4 measurements")

    r1 = run_test(df, "As > 10 µg/L (WHO)", "As_T1")
    r2 = run_test(df, "As > 22 µg/L (saddle)", "As_T2")

    summary = pd.DataFrame([
        {'target': r['target'],
         'AUC_base': r['auc_base'], 'AUC_full': r['auc_full'],
         'delta_AUC': r['auc_full'] - r['auc_base'],
         'delta_AUC_CI_lo': r['delta_auc_lo'],
         'delta_AUC_CI_hi': r['delta_auc_hi'],
         'PO4_perm_rank': r['po4_perm_rank'],
         'conditional_MI': r['conditional_mi'],
         'PO4_informative': r['decision_supported']}
        for r in [r1, r2]
    ])
    summary.to_csv(TABLES_DIR / 'T4_partial_info_PO4.csv', index=False)
    print("\n" + "=" * 75)
    print("SUMMARY")
    print("=" * 75)
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    # Operating characteristics of the field PO4-threshold screening rule
    oc = operating_characteristics(df)
    oc.to_csv(TABLES_DIR / 'T4_po4_screening_operating_characteristics.csv', index=False)
    print("\n--- PO4-threshold screening operating characteristics ---")
    print(oc.to_string(index=False))

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, r in zip(axes, [r1, r2]):
        pt = r['perm_table']
        colors = ['#c0392b' if f == 'PO43-' else '#7f8c8d' for f in pt['feature']]
        ax.barh(pt['feature'].iloc[::-1],
                pt['importance_mean'].iloc[::-1],
                xerr=pt['importance_std'].iloc[::-1],
                color=colors[::-1])
        ax.set_xlabel('Permutation importance (Δ AUC)')
        ax.axvline(0, color='#333', lw=0.6, ls='--')
        ax.set_title(r['target'], fontsize=11, fontweight='bold')
    fig.suptitle(
        f"Phosphate vs other redox proxies: does PO$_4$ add information about As mode?\n"
        f"ΔAUC(+PO$_4$) = {r1['auc_full']-r1['auc_base']:+.3f} (T1), "
        f"{r2['auc_full']-r2['auc_base']:+.3f} (T2)",
        fontsize=10, y=1.02)
    fig.tight_layout()
    out = FIGURES_DIR / 'T4_partial_info_PO4.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved: {out}")
    print("=" * 75)


if __name__ == '__main__':
    main()
