#!/usr/bin/env python3
"""R2.3 / climate-limb repair: recompute the seasonal transfer coefficients on
wells sampled in BOTH campaigns, using within-well paired differences.

The published T1_seasonal_transfer.csv contrasts unpaired dry and wet MEDIANS per
depth x zone x contaminant cell, so the contrast confounds seasonal change with
which wells happened to be visited in each campaign. This recomputes the same
coefficients as paired within-well differences over the 810 wells present in both
campaigns, and reports a Wilcoxon signed-rank test in place of Mann-Whitney.

sensitivity_per_pct keeps the published definition: delta / 30, where 30 is the
assumed dry-to-wet precipitation contrast in percent.

Writes: T1_seasonal_transfer_PAIRED.csv  (next to the published table)
Run from the Paper 2 root.
"""
import sys, pathlib, numpy as np, pandas as pd
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config import DATA_FILE, assign_zones, TABLES_DIR

CONTAM = ['As', 'Mn2+', 'Fe2+', 'Cr3+', 'NO3-']
PCT_CONTRAST = 30.0        # same denominator as the published table
N_BOOT = 2000
RNG = np.random.default_rng(42)

d = assign_zones(pd.read_csv(DATA_FILE))
both = d.groupby('Sample ID').Season.nunique()
paired_ids = set(both[both == 2].index)
d = d[d['Sample ID'].isin(paired_ids)].drop_duplicates(['Sample ID', 'Season'])
print(f"paired wells: {d['Sample ID'].nunique()}   rows: {len(d)}")

rows = []
for (dz, pz), g in d.groupby(['depth_zone', 'phys_zone']):
    w = g.pivot_table(index='Sample ID', columns='Season', values=CONTAM, aggfunc='first')
    for c in CONTAM:
        if ('Dry' not in w[c].columns) or ('Wet' not in w[c].columns):
            continue
        pair = w[c][['Dry', 'Wet']].dropna()
        if len(pair) < 8:
            continue
        diff = (pair['Wet'] - pair['Dry']).values
        med = float(np.median(diff))
        boot = np.array([np.median(RNG.choice(diff, len(diff), replace=True))
                         for _ in range(N_BOOT)])
        try:
            p = stats.wilcoxon(pair['Wet'], pair['Dry']).pvalue
        except ValueError:
            p = np.nan
        rows.append(dict(
            depth_zone=dz, phys_zone=pz, contaminant=c, n_pairs=len(pair),
            dry_median=float(pair['Dry'].median()), wet_median=float(pair['Wet'].median()),
            paired_delta_median=med,
            sensitivity_per_pct=med / PCT_CONTRAST,
            sensitivity_CI_lo=float(np.percentile(boot, 2.5)) / PCT_CONTRAST,
            sensitivity_CI_hi=float(np.percentile(boot, 97.5)) / PCT_CONTRAST,
            wilcoxon_p=p, significant=bool(p < 0.05) if p == p else False))

out = pd.DataFrame(rows)
out.to_csv(TABLES_DIR / 'T1_seasonal_transfer_PAIRED.csv', index=False)

# Promote to the canonical name that climate_transfer_ensemble.py consumes, after
# archiving the unpaired table once. Downstream consumers read
# T1_seasonal_transfer.csv and need the published column schema.
unpaired = TABLES_DIR / 'T1_seasonal_transfer_UNPAIRED.csv'
canon = TABLES_DIR / 'T1_seasonal_transfer.csv'
if not unpaired.exists() and canon.exists():
    import shutil; shutil.copy(canon, unpaired)
    print(f"archived unpaired table -> {unpaired.name}")
pub = pd.read_csv(unpaired) if unpaired.exists() else pd.read_csv(canon)

canon_df = out.rename(columns={'paired_delta_median': 'delta_median',
                               'wilcoxon_p': 'mann_whitney_p'})
canon_df['delta_pct'] = 100 * canon_df.delta_median / canon_df.dry_median.replace(0, np.nan)
canon_df['delta_pct_CI_lo'] = (100 * canon_df.sensitivity_CI_lo * PCT_CONTRAST
                               / canon_df.dry_median.replace(0, np.nan))
canon_df['delta_pct_CI_hi'] = (100 * canon_df.sensitivity_CI_hi * PCT_CONTRAST
                               / canon_df.dry_median.replace(0, np.nan))
canon_df['effect_size_r'] = np.nan
canon_df['n_dry'] = canon_df.n_pairs
canon_df['n_wet'] = canon_df.n_pairs
cols = [c for c in pub.columns if c in canon_df.columns] + ['n_pairs']
canon_df[cols].to_csv(canon, index=False)
print(f"wrote {canon.name} (paired, canonical) and "
      f"T1_seasonal_transfer_PAIRED.csv  ({len(out)} cells)\n")
m = out.merge(pub, on=['depth_zone', 'phys_zone', 'contaminant'],
              suffixes=('_paired', '_pub'))
print(f"{'contaminant':12s} {'cells':>6s} {'sig pub':>8s} {'sig paired':>11s} "
      f"{'sign flips':>11s}  median |sens| pub -> paired")
for c in CONTAM:
    s = m[m.contaminant == c]
    if s.empty:
        continue
    flip = ((np.sign(s.sensitivity_per_pct_paired) != np.sign(s.sensitivity_per_pct_pub))
            & (s.sensitivity_per_pct_pub != 0)).sum()
    print(f"  {c:10s} {len(s):6d} {int(s.significant_pub.sum()):8d} "
          f"{int(s.significant_paired.sum()):11d} {flip:11d}   "
          f"{s.sensitivity_per_pct_pub.abs().median():.5f} -> "
          f"{s.sensitivity_per_pct_paired.abs().median():.5f}")

print("\nDirection of the pooled paired contrast (all cells, weighted by n_pairs):")
for c in CONTAM:
    s = out[out.contaminant == c]
    if s.empty:
        continue
    w = np.average(s.paired_delta_median, weights=s.n_pairs)
    print(f"  {c:10s} pooled wet-minus-dry median delta = {w:+.4f}"
          f"   ({int(s.significant.sum())}/{len(s)} cells significant)")
