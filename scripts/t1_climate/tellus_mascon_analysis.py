"""
TELLUS GRACE/GRACE-FO mascon long-term trend (2002-2026)
=========================================================

Complements the 7-day groundwater storage percentile analysis with the
24-year liquid-water-equivalent (LWE) thickness record from the JPL
RL06.3 V4 CRI mascon product. Provides the multi-decadal trend in
total terrestrial water storage that the 6.5-year GRACEDADM record
cannot resolve.

Hypothesis tested: do high-arsenic floodplain zones show DIFFERENT
multi-decadal TWS trends than low-arsenic uplifted zones?

Output:
  output/tables/T1_tellus_zone_trend.csv  (zone-level trend cm/yr)
  output/figures/T1_tellus_zone_timeseries.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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

TELLUS_NC = (EXTERNAL_DIR / 'grace_fo' /
             'TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.3_V4' /
             'GRCTellus.JPL.200204_202603.GLO.RL06.3M.MSCNv04CRI.nc')


def load_bangladesh_slice():
    """Open dataset, slice to Bangladesh, return DataArray of LWE (cm)."""
    ds = xr.open_dataset(TELLUS_NC)
    # Lons 0-360; Bangladesh 88-93 fits directly. Lats decreasing or increasing?
    bd = ds['lwe_thickness'].sel(lat=slice(20.5, 26.8), lon=slice(87.8, 92.8))
    return bd, ds.scale_factor.sel(lat=slice(20.5, 26.8),
                                    lon=slice(87.8, 92.8))


def zone_assign_grid(lats, lons):
    centroids = {z: ((b['lat'][0]+b['lat'][1])/2,
                     (b['lon'][0]+b['lon'][1])/2)
                 for z, b in PHYSIOGRAPHIC_ZONES.items()}
    LA, LO = np.meshgrid(lats, lons, indexing='ij')
    zg = np.empty(LA.shape, dtype=object)
    for i in range(LA.shape[0]):
        for j in range(LA.shape[1]):
            la, lo = LA[i, j], LO[i, j]
            cands = [z for z, b in PHYSIOGRAPHIC_ZONES.items()
                     if b['lat'][0] <= la <= b['lat'][1]
                     and b['lon'][0] <= lo <= b['lon'][1]]
            if cands:
                zg[i, j] = min(cands,
                               key=lambda z: (centroids[z][0]-la)**2
                                             +(centroids[z][1]-lo)**2)
            else:
                zg[i, j] = min(centroids,
                               key=lambda z: (centroids[z][0]-la)**2
                                             +(centroids[z][1]-lo)**2)
    return zg


def per_zone_trends(lwe, scale):
    """Compute zone-level LWE monthly time series and linear trend (cm/yr)."""
    zg = zone_assign_grid(lwe.lat.values, lwe.lon.values)
    # Apply scale factor (CRI mascon best practice)
    sf = scale.values
    sf = np.where(np.isfinite(sf), sf, 1.0)
    arr = lwe.values * sf  # (time, lat, lon)

    times = pd.to_datetime(lwe.time.values)
    rows = []
    trend_rows = []
    for z in PHYSIOGRAPHIC_ZONES:
        mask = (zg == z)
        if mask.sum() == 0:
            continue
        ts = np.array([
            np.nanmean(arr[t][mask]) for t in range(arr.shape[0])
        ])
        valid = np.isfinite(ts)
        if valid.sum() < 24:
            continue
        years = (times - times[0]).total_seconds() / (365.25 * 86400)
        slope, intercept, r, p, se = stats.linregress(
            years[valid], ts[valid])
        for t, v in zip(times, ts):
            rows.append({'phys_zone': z, 'date': t,
                         'lwe_cm': float(v) if np.isfinite(v) else None})
        trend_rows.append({
            'phys_zone': z,
            'n_obs': int(valid.sum()),
            'mean_lwe_cm': float(np.nanmean(ts)),
            'trend_cm_per_yr': float(slope),
            'trend_p': float(p),
            'amplitude_cm': float(np.nanmax(ts) - np.nanmin(ts)),
        })
    return pd.DataFrame(rows), pd.DataFrame(trend_rows)


def main():
    print("=" * 70)
    print("TELLUS mascon LWE trend 2002-2026 over Bangladesh")
    print("=" * 70)
    lwe, scale = load_bangladesh_slice()
    print(f"BD slice: {dict(lwe.sizes)}")
    ts, trends = per_zone_trends(lwe, scale)
    ts.to_csv(TABLES_DIR / 'T1_tellus_zone_timeseries.csv', index=False)
    trends = trends.sort_values('trend_cm_per_yr')
    trends.to_csv(TABLES_DIR / 'T1_tellus_zone_trend.csv', index=False)
    print("\nZone-level LWE trend:")
    print(trends.to_string(index=False, float_format=lambda x: f'{x:.3f}'))

    # Merge with arsenic
    df = pd.read_csv(DATA_FILE)
    df = assign_zones(df)
    as_zone = df.groupby('phys_zone')['As'].median().reset_index()
    as_zone.columns = ['phys_zone', 'as_median']
    merged = trends.merge(as_zone, on='phys_zone', how='left')
    merged.to_csv(TABLES_DIR / 'T1_tellus_arsenic_merge.csv', index=False)
    print("\nMerged with zone median arsenic:")
    print(merged[['phys_zone', 'trend_cm_per_yr', 'mean_lwe_cm',
                  'amplitude_cm', 'as_median']].to_string(
        index=False, float_format=lambda x: f'{x:.3f}'))

    if len(merged) >= 3:
        slope, _, r, p, _ = stats.linregress(
            merged['trend_cm_per_yr'], merged['as_median'])
        print(f"\nLWE trend vs zone median As: r = {r:+.3f}, p = {p:.3f}")
        slope2, _, r2, p2, _ = stats.linregress(
            merged['amplitude_cm'], merged['as_median'])
        print(f"LWE seasonal amplitude vs As: r = {r2:+.3f}, p = {p2:.3f}")

    # Plot
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ZONE_COLORS = {
        'Ganges_Floodplain': '#c0392b', 'GBM_Delta': '#e67e22',
        'Meghna_Floodplain': '#27ae60', 'Brahmaputra_Floodplain': '#16a085',
        'Northern_Terrace': '#2980b9', 'Barind_Tract': '#8e44ad',
        'Eastern_Hills': '#7f8c8d',
    }
    ts['date'] = pd.to_datetime(ts['date'])
    for z, sub in ts.groupby('phys_zone'):
        sub = sub.sort_values('date')
        ax.plot(sub['date'], sub['lwe_cm'],
                color=ZONE_COLORS.get(z, '#888'),
                lw=1.4, alpha=0.85,
                label=z.replace('_', ' '))
    ax.axhline(0, color='#333', ls='--', lw=0.7, alpha=0.4)
    ax.set_xlabel('Date')
    ax.set_ylabel('Liquid water equivalent thickness (cm)')
    ax.set_title('TELLUS GRACE/GRACE-FO mascon LWE by physiographic zone, 2002–2026',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='lower left', ncol=4, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'T1_tellus_zone_timeseries.png',
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure: {FIGURES_DIR / 'T1_tellus_zone_timeseries.png'}")
    print("=" * 70)


if __name__ == '__main__':
    main()
