"""
GRACE-FO groundwater storage analysis for Bangladesh (2018-2024)
================================================================

Independent climate-coupling line of evidence: do high-arsenic-burden zones
co-locate with regions of declining groundwater storage and amplified seasonal
variance, as predicted by the climate-amplification thesis?

Data: GRACEDADM_CLSM025GL_7D (NASA GSFC HSL via GES DISC), 0.25 deg, 7-day
groundwater storage percentile (0 = very dry / depleted; 100 = very wet / full).
60 weekly granules covering 2018-06 through 2019-08 retrieved with earthaccess.

Outputs:
  - T1_grace_fo_zone_timeseries.csv  (zone, date, gws_percentile)
  - T1_grace_fo_zone_summary.csv     (zone-level mean, trend, seasonal amplitude)
  - figures/T1_grace_fo_timeseries.png
  - figures/T1_grace_fo_arsenic_correlation.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import re
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

from config import (
    EXTERNAL_DIR, PHYSIOGRAPHIC_ZONES, TABLES_DIR, FIGURES_DIR,
    DATA_FILE, assign_zones,
)

GRACE_DIR = EXTERNAL_DIR / 'grace_fo' / 'GRACEDADM_CLSM025GL_7D'
DATE_PATTERN = re.compile(r'\.A(\d{8})\.')
BD_BBOX = dict(lat=slice(20.5, 26.8), lon=slice(87.8, 92.8))


def _parse_date(name):
    m = DATE_PATTERN.search(name)
    return pd.to_datetime(m.group(1)) if m else None


def build_zone_time_series():
    """For each granule, slice to Bangladesh and compute per-zone mean gws."""
    rows = []
    files = sorted(GRACE_DIR.glob('GRACEDADM_CLSM025GL_7D.*.nc4'))
    print(f"Found {len(files)} granules")

    # Build a single (lat, lon) grid → zone-assignment lookup once
    # by opening the first granule's slice.
    ds0 = xr.open_dataset(files[0]).sel(**BD_BBOX)
    lats = ds0.lat.values
    lons = ds0.lon.values
    centroids = {
        z: ((b['lat'][0] + b['lat'][1]) / 2,
            (b['lon'][0] + b['lon'][1]) / 2)
        for z, b in PHYSIOGRAPHIC_ZONES.items()
    }

    def zone_of(la, lo):
        cands = [z for z, b in PHYSIOGRAPHIC_ZONES.items()
                 if b['lat'][0] <= la <= b['lat'][1]
                 and b['lon'][0] <= lo <= b['lon'][1]]
        if cands:
            return min(cands,
                       key=lambda z: (centroids[z][0] - la) ** 2
                                     + (centroids[z][1] - lo) ** 2)
        return min(centroids,
                   key=lambda z: (centroids[z][0] - la) ** 2
                                 + (centroids[z][1] - lo) ** 2)

    LA, LO = np.meshgrid(lats, lons, indexing='ij')
    zone_grid = np.array([
        [zone_of(la, lo) for la, lo in zip(la_row, lo_row)]
        for la_row, lo_row in zip(LA, LO)
    ])

    for f in files:
        date = _parse_date(f.name)
        try:
            ds = xr.open_dataset(f).sel(**BD_BBOX)
        except Exception as e:
            print(f"  SKIP {f.name}: {e}")
            continue
        gws = ds['gws_inst'].values.squeeze()
        for z in PHYSIOGRAPHIC_ZONES:
            mask = (zone_grid == z) & np.isfinite(gws)
            if mask.sum() == 0:
                continue
            rows.append({
                'date': date,
                'phys_zone': z,
                'gws_percentile_mean': float(gws[mask].mean()),
                'gws_percentile_std': float(gws[mask].std()),
                'n_cells': int(mask.sum()),
            })

    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / 'T1_grace_fo_zone_timeseries.csv', index=False)
    return df


def zone_summary(ts):
    """Per-zone trend slope + seasonal amplitude + mean."""
    rows = []
    for z, g in ts.groupby('phys_zone'):
        g = g.sort_values('date')
        years = (g['date'] - g['date'].min()).dt.days / 365.25
        slope, intercept, r, p, _ = stats.linregress(years, g['gws_percentile_mean'])
        season = g['gws_percentile_mean'].max() - g['gws_percentile_mean'].min()
        rows.append({
            'phys_zone': z,
            'mean_percentile': float(g['gws_percentile_mean'].mean()),
            'trend_pct_per_yr': float(slope),
            'trend_p_value': float(p),
            'seasonal_amplitude': float(season),
            'n_observations': len(g),
        })
    return pd.DataFrame(rows).sort_values('trend_pct_per_yr')


def correlate_with_arsenic(summary):
    """Merge GRACE-FO zone summary with sample-derived arsenic exposure."""
    df = pd.read_csv(DATA_FILE)
    df = assign_zones(df)
    arsenic_by_zone = df.groupby('phys_zone')['As'].agg(['median', 'mean', 'std']).reset_index()
    arsenic_by_zone.columns = ['phys_zone', 'as_median', 'as_mean', 'as_std']
    return summary.merge(arsenic_by_zone, on='phys_zone', how='left')


def plot_zone_timeseries(ts):
    fig, ax = plt.subplots(figsize=(10, 6))
    zones_order = list(PHYSIOGRAPHIC_ZONES.keys())
    cmap = plt.get_cmap('viridis')
    for i, z in enumerate(zones_order):
        g = ts[ts['phys_zone'] == z].sort_values('date')
        if len(g) == 0:
            continue
        ax.plot(g['date'], g['gws_percentile_mean'],
                color=cmap(i / len(zones_order)),
                lw=1.5, alpha=0.85, label=z.replace('_', ' '))
    ax.axhline(50, color='#333', ls='--', lw=0.7, alpha=0.5)
    ax.set_xlabel('Date')
    ax.set_ylabel('Groundwater storage percentile')
    ax.set_title('GRACE-FO groundwater storage by physiographic zone, 2018–2024',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='lower left', fontsize=8, ncol=2, frameon=False)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'T1_grace_fo_timeseries.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_arsenic_correlation(merged):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # Panel A: arsenic median vs mean GWS percentile
    ax = axes[0]
    ax.scatter(merged['mean_percentile'], merged['as_median'],
                s=90, c='#c0392b', alpha=0.85, edgecolor='white')
    for _, r in merged.iterrows():
        ax.annotate(r['phys_zone'].replace('_', ' '),
                    (r['mean_percentile'], r['as_median']),
                    xytext=(5, 4), textcoords='offset points', fontsize=8)
    if len(merged) >= 3:
        slope, intercept, rval, pval, _ = stats.linregress(
            merged['mean_percentile'], merged['as_median'])
        x = np.linspace(merged['mean_percentile'].min(), merged['mean_percentile'].max(), 50)
        ax.plot(x, slope * x + intercept, '--', color='#555', lw=1.2,
                label=f'r = {rval:+.2f}, p = {pval:.2g}')
        ax.legend(loc='upper right', fontsize=10, frameon=False)
    ax.set_xlabel('Mean GRACE-FO groundwater percentile (2018-2024)')
    ax.set_ylabel('Zone median arsenic (µg/L)')
    ax.set_title('Wetter aquifers $\\rightarrow$ higher arsenic',
                 fontsize=11, fontweight='bold')

    # Panel B: trend vs arsenic median
    ax = axes[1]
    ax.scatter(merged['trend_pct_per_yr'], merged['as_median'],
                s=90, c='#2980b9', alpha=0.85, edgecolor='white')
    for _, r in merged.iterrows():
        ax.annotate(r['phys_zone'].replace('_', ' '),
                    (r['trend_pct_per_yr'], r['as_median']),
                    xytext=(5, 4), textcoords='offset points', fontsize=8)
    ax.axvline(0, color='#333', ls='--', lw=0.7, alpha=0.5)
    ax.set_xlabel('GRACE-FO groundwater trend (percentile units / year)')
    ax.set_ylabel('Zone median arsenic (µg/L)')
    ax.set_title('Storage trend vs arsenic exposure',
                 fontsize=11, fontweight='bold')

    fig.tight_layout(w_pad=2.5)
    fig.savefig(FIGURES_DIR / 'T1_grace_fo_arsenic_correlation.png',
                dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    print("=" * 70)
    print("GRACE-FO groundwater storage × physiographic-zone arsenic")
    print("=" * 70)

    ts = build_zone_time_series()
    print(f"\nTime-series rows: {len(ts):,}")
    print(f"Date span: {ts['date'].min().date()} -> {ts['date'].max().date()}")
    print(f"Zones covered: {ts['phys_zone'].nunique()}")

    summary = zone_summary(ts)
    summary.to_csv(TABLES_DIR / 'T1_grace_fo_zone_summary.csv', index=False)
    print("\n--- Zone-level GRACE-FO summary ---")
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.2f}'))

    merged = correlate_with_arsenic(summary)
    merged.to_csv(TABLES_DIR / 'T1_grace_fo_arsenic_merge.csv', index=False)
    print("\n--- Merge with arsenic by zone ---")
    print(merged[['phys_zone', 'mean_percentile', 'trend_pct_per_yr',
                  'seasonal_amplitude', 'as_median']].to_string(
        index=False, float_format=lambda x: f'{x:.2f}'))

    if len(merged) >= 3:
        slope, intercept, r, p, _ = stats.linregress(
            merged['mean_percentile'], merged['as_median'])
        print(f"\nCorrelation: zone mean GRACE percentile vs zone median As")
        print(f"  r = {r:+.3f}, p = {p:.3g}")
        slope2, _, r2, p2, _ = stats.linregress(
            merged['trend_pct_per_yr'], merged['as_median'])
        print(f"Correlation: zone GRACE trend vs zone median As")
        print(f"  r = {r2:+.3f}, p = {p2:.3g}")

    plot_zone_timeseries(ts)
    plot_arsenic_correlation(merged)
    print("\nFigures saved to", FIGURES_DIR)
    print("=" * 70)


if __name__ == '__main__':
    main()
