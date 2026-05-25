"""
Gridded WorldPop × interpolated HI aggregation (2026-05-22)
============================================================
Replaces hardcoded ZONE_POPULATION dict in daly_estimation_corrected.py with
a defensible spatial integration:

  For each populated 1-km WorldPop cell in Bangladesh:
    1. Assign to a physiographic zone via centroid-nearest logic
       (same convention as config.assign_zones).
    2. Interpolate HI_multi, HI_as_only, CR_multi, CR_as_only from the
       988 sample locations using inverse-distance-weighted (IDW) k-nearest
       neighbours (k=10, p=2). IDW is intentionally conservative — it does
       not extrapolate beyond the observed range, unlike kriging or RF.
    3. Aggregate cell × population to give a true population-weighted
       distribution of HI/CR within each zone — replacing the previous
       zone-median statistic that ignored where people actually live.

Output: a per-zone DataFrame with:
  - population (sum of WorldPop cells assigned to zone)
  - n_cells
  - HI_multi_pop_mean, HI_multi_pop_p50 (weighted median), HI_multi_pop_p95
  - CR_multi_pop_mean, CR_multi_pop_p50, CR_multi_pop_p95
  - HI_as_only_pop_mean, …

This is the source-of-truth zone-level statistic for the corrected DALY
pipeline; daly_estimation_corrected.py imports `populated_zone_stats` to use
gridded values instead of hardcoded ones when this module is available.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import rasterio
from scipy.spatial import cKDTree

from config import (
    DATA_FILE, EXTERNAL_DIR, PHYSIOGRAPHIC_ZONES, TABLES_DIR, assign_zones,
)

WORLDPOP_TIF = EXTERNAL_DIR / 'worldpop' / 'bgd_ppp_2020_1km.tif'


def _zone_for_point(lat, lon, centroids, bboxes):
    """Return the zone whose centroid is nearest to (lat, lon) among those
    whose bounding box contains the point; fall back to globally-nearest
    centroid if no bbox match."""
    matches = []
    for zone, bbox in bboxes.items():
        if (bbox['lat'][0] <= lat <= bbox['lat'][1]
                and bbox['lon'][0] <= lon <= bbox['lon'][1]):
            matches.append(zone)
    if matches:
        best = min(matches, key=lambda z: (centroids[z][0] - lat) ** 2
                                          + (centroids[z][1] - lon) ** 2)
        return best
    # Fallback: globally nearest centroid
    return min(centroids, key=lambda z: (centroids[z][0] - lat) ** 2
                                        + (centroids[z][1] - lon) ** 2)


def load_grid():
    """Load WorldPop raster and return (lat, lon, pop) for populated cells."""
    with rasterio.open(WORLDPOP_TIF) as src:
        arr = src.read(1).astype('float64')
        if src.nodata is not None:
            arr = np.where(arr == src.nodata, np.nan, arr)
        nrows, ncols = src.shape
        # Cell-centre lat/lon via transform
        cols = np.arange(ncols)
        rows = np.arange(nrows)
        col_grid, row_grid = np.meshgrid(cols, rows)
        xs, ys = rasterio.transform.xy(
            src.transform, row_grid.ravel(), col_grid.ravel()
        )
        lon = np.array(xs).reshape(arr.shape)
        lat = np.array(ys).reshape(arr.shape)

    # Keep populated cells
    pop_flat = arr.ravel()
    lat_flat = lat.ravel()
    lon_flat = lon.ravel()
    mask = np.isfinite(pop_flat) & (pop_flat > 0)
    return lat_flat[mask], lon_flat[mask], pop_flat[mask]


def interpolate_idw(sample_lat, sample_lon, sample_val,
                    target_lat, target_lon, k=10, p=2):
    """IDW interpolation: target = sum(w_i * val_i)/sum(w_i), w_i = 1/d^p."""
    tree = cKDTree(np.column_stack([sample_lat, sample_lon]))
    # Query in batches to limit memory
    out = np.empty(len(target_lat))
    batch = 50_000
    for i in range(0, len(target_lat), batch):
        end = min(i + batch, len(target_lat))
        q = np.column_stack([target_lat[i:end], target_lon[i:end]])
        d, idx = tree.query(q, k=k)
        # Avoid divide-by-zero for coincident points
        d = np.where(d < 1e-9, 1e-9, d)
        w = 1.0 / (d ** p)
        v = sample_val[idx]
        out[i:end] = np.sum(w * v, axis=1) / np.sum(w, axis=1)
    return out


def weighted_quantile(values, weights, q):
    """Weighted quantile (linear interpolation between order statistics)."""
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cw = np.cumsum(w) / np.sum(w)
    return np.interp(q, cw, v)


def populated_zone_stats(df_with_risk):
    """Return DataFrame of zone-level population-weighted HI/CR stats.

    Parameters
    ----------
    df_with_risk : DataFrame
        Sample-level dataset with columns Latitude, Longitude,
        HI_multi, HI_as_only, CR_multi, CR_as_only already computed.

    Returns
    -------
    DataFrame indexed by phys_zone with columns:
      population, n_cells, HI_multi_pop_mean, HI_multi_pop_p50,
      HI_multi_pop_p95, HI_as_only_pop_mean, HI_as_only_pop_p50,
      CR_multi_pop_mean, CR_multi_pop_p50, CR_as_only_pop_mean,
      CR_as_only_pop_p50
    """
    lat, lon, pop = load_grid()

    # Centroids and bboxes once
    centroids = {
        z: ((b['lat'][0] + b['lat'][1]) / 2,
            (b['lon'][0] + b['lon'][1]) / 2)
        for z, b in PHYSIOGRAPHIC_ZONES.items()
    }
    zones = np.array([
        _zone_for_point(la, lo, centroids, PHYSIOGRAPHIC_ZONES)
        for la, lo in zip(lat, lon)
    ])

    # IDW each risk metric
    s_lat = df_with_risk['Latitude'].values
    s_lon = df_with_risk['Longitude'].values
    risk_cols = ['HI_multi', 'HI_as_only', 'CR_multi', 'CR_as_only']
    cell_risk = {}
    for col in risk_cols:
        cell_risk[col] = interpolate_idw(
            s_lat, s_lon, df_with_risk[col].values, lat, lon, k=10, p=2,
        )

    # Per-zone aggregation
    rows = []
    for z in PHYSIOGRAPHIC_ZONES:
        mask = zones == z
        if not mask.any():
            continue
        wp = pop[mask]
        row = {'phys_zone': z, 'population': float(wp.sum()),
               'n_cells': int(mask.sum())}
        for col in risk_cols:
            v = cell_risk[col][mask]
            row[f'{col}_pop_mean'] = float(np.average(v, weights=wp))
            row[f'{col}_pop_p50'] = float(weighted_quantile(v, wp, 0.50))
            row[f'{col}_pop_p95'] = float(weighted_quantile(v, wp, 0.95))
        rows.append(row)
    return pd.DataFrame(rows).set_index('phys_zone')


def main():
    print("=" * 70)
    print("Gridded spatial aggregation — WorldPop × IDW(HI)")
    print("=" * 70)

    # Recompute HI/CR with corrected exposure params on the sample data
    from daly_estimation_corrected import (
        EXPOSURE_PARAMS_CORRECTED, calculate_hi_cr,
    )

    df = pd.read_csv(DATA_FILE)
    df = assign_zones(df)
    params = EXPOSURE_PARAMS_CORRECTED['adult_male']
    hi_multi, hi_as, cr_multi, cr_as = calculate_hi_cr(df, 'adult_male', params)
    df['HI_multi'] = hi_multi
    df['HI_as_only'] = hi_as
    df['CR_multi'] = cr_multi
    df['CR_as_only'] = cr_as

    stats = populated_zone_stats(df)
    stats.to_csv(TABLES_DIR / 'T2_zone_population_weighted_stats.csv')

    print(f"\nZone-level population-weighted statistics:")
    show = stats[['population', 'n_cells', 'HI_multi_pop_mean',
                  'HI_multi_pop_p50', 'HI_as_only_pop_mean',
                  'CR_multi_pop_mean']].copy()
    show['population'] = show['population'].astype(int)
    print(show.to_string(float_format=lambda x: f'{x:.4f}'))

    total_pop = int(stats['population'].sum())
    print(f"\nTotal Bangladesh population (WorldPop): {total_pop:,}")
    print(f"(Hardcoded ZONE_POPULATION sum was 135,500,000)")
    print("=" * 70)


if __name__ == '__main__':
    main()
