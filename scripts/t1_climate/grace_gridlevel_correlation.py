"""
T1: Grid-level GRACE recharge-arsenic correlation  [added 2026-05-24]
=====================================================================
Reviewer concern addressed: "the storage/trend-arsenic correlations are computed
across only seven physiographic zones (n=7), which is underpowered."

We escape the n=7 zone aggregation by correlating at the native resolution of
each gravimetry product:

  GRACEDADM (0.25 deg land-surface assimilation, 2018-2024): build a per-cell
    time-mean storage field and per-cell linear trend, then aggregate wells to
    GRACE cells (median As per cell) -> n = number of distinct 0.25-deg cells
    containing wells. Well-level (pseudo-replicated) correlation reported too.

  TELLUS JPL RL06.3 mascon (2002-2026): per-cell LWE trend (cm/yr), aggregated
    by mascon_ID -- the HONEST degrees of freedom, since the CRI product has
    ~3-deg native mascon resolution despite a 0.5-deg grid.

Spearman rank correlation throughout (As is heavy-tailed). This neither inflates
nor manufactures significance: results are reported as-is, and if the grid-level
test is weak we recommend the zone-level result be read as directional.

Outputs:
  output/tables/T1_grace_gridlevel_correlation.csv
"""

import sys, re, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from scipy.spatial import cKDTree

from config import EXTERNAL_DIR, DATA_FILE, BANGLADESH_BBOX

GRACE_DIR = EXTERNAL_DIR / 'grace_fo' / 'GRACEDADM_CLSM025GL_7D'
TELLUS_NC = (EXTERNAL_DIR / 'grace_fo' /
             'TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.3_V4' /
             'GRCTellus.JPL.200204_202603.GLO.RL06.3M.MSCNv04CRI.nc')
LAT = slice(20.5, 26.8)
LON = slice(87.8, 92.8)
DATE_RE = re.compile(r'\.A(\d{8})\.')


def _trend_per_cell(arr, years):
    """Vectorised per-cell OLS slope of arr (T, ny, nx) vs years (T,)."""
    T = arr.shape[0]
    y = years - years.mean()
    denom = np.sum(y ** 2)
    a = arr - np.nanmean(arr, axis=0, keepdims=True)
    slope = np.nansum(a * y[:, None, None], axis=0) / denom
    return slope


def gracedadm_fields():
    files = sorted(glob.glob(str(GRACE_DIR / 'GRACEDADM_CLSM025GL_7D.*.nc4')))
    dates, stack = [], []
    lats = lons = None
    for f in files:
        m = DATE_RE.search(Path(f).name)
        if not m:
            continue
        ds = xr.open_dataset(f).sel(lat=LAT, lon=LON)
        v = ds['gws_inst'].values.squeeze()
        stack.append(v)
        dates.append(pd.to_datetime(m.group(1)))
        if lats is None:
            lats, lons = ds.lat.values, ds.lon.values
    arr = np.array(stack)
    dates = pd.to_datetime(dates)
    years = (dates - dates.min()).days.values / 365.25
    mean_field = np.nanmean(arr, axis=0)
    trend_field = _trend_per_cell(arr, years)
    print(f"  GRACEDADM: {len(stack)} granules, grid {mean_field.shape}, "
          f"{dates.min().date()}..{dates.max().date()}")
    return lats, lons, mean_field, trend_field


def tellus_trend():
    ds = xr.open_dataset(TELLUS_NC)
    lwe = ds['lwe_thickness'].sel(lat=LAT, lon=LON)
    sf = ds['scale_factor'].sel(lat=LAT, lon=LON).values
    sf = np.where(np.isfinite(sf), sf, 1.0)
    mid = ds['mascon_ID'].sel(lat=LAT, lon=LON).values if 'mascon_ID' in ds else None
    arr = lwe.values * sf
    times = pd.to_datetime(lwe.time.values)
    years = (times - times.min()).days.values / 365.25
    trend = _trend_per_cell(arr, years)  # cm/yr
    print(f"  TELLUS: {arr.shape[0]} months, grid {trend.shape}, "
          f"{times.min().date()}..{times.max().date()}")
    return lwe.lat.values, lwe.lon.values, trend, mid


def nearest_cell_values(well_lat, well_lon, lats, lons, *fields):
    LA, LO = np.meshgrid(lats, lons, indexing='ij')
    flat_la, flat_lo = LA.ravel(), LO.ravel()
    finite = np.isfinite(fields[0].ravel())
    tree = cKDTree(np.column_stack([flat_la[finite], flat_lo[finite]]))
    _, idx = tree.query(np.column_stack([well_lat, well_lon]))
    cell_id = np.where(finite)[0][idx]
    out = [f.ravel()[cell_id] for f in fields]
    return cell_id, out


def report(tag, x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    rho, p = stats.spearmanr(x[ok], y[ok])
    print(f"  {tag:42s} rho={rho:+.3f}  p={p:.2e}  n={int(ok.sum())}")
    return {'test': tag, 'spearman_rho': float(rho), 'p_value': float(p),
            'n': int(ok.sum())}


def main():
    print("=" * 70)
    print("Grid-level GRACE recharge-arsenic correlation (escaping n=7)")
    print("=" * 70)

    df = pd.read_csv(DATA_FILE)
    df = df[np.isfinite(df['As']) & np.isfinite(df['Latitude'])
            & np.isfinite(df['Longitude'])].copy()
    wlat, wlon, As = df['Latitude'].values, df['Longitude'].values, df['As'].values
    print(f"Wells with As + coordinates: {len(df)}")

    rows = []

    # ── GRACEDADM storage mean + trend ──────────────────────────────────────
    glat, glon, gmean, gtrend = gracedadm_fields()
    gcell, (w_mean, w_trend) = nearest_cell_values(wlat, wlon, glat, glon, gmean, gtrend)

    print("\n[GRACEDADM] WELL-LEVEL (pseudo-replicated; n inflated by cell sharing)")
    rows.append({'product': 'GRACEDADM', 'level': 'well', **report(
        'As vs storage mean', w_mean, As)})
    rows.append({'product': 'GRACEDADM', 'level': 'well', **report(
        'As vs storage trend', w_trend, As)})

    print("\n[GRACEDADM] CELL-LEVEL (median As per 0.25-deg cell; honest n)")
    cdf = pd.DataFrame({'cell': gcell, 'As': As, 'smean': w_mean, 'strend': w_trend})
    cg = cdf.groupby('cell').agg(As_med=('As', 'median'), smean=('smean', 'first'),
                                 strend=('strend', 'first'), n=('As', 'size')).reset_index()
    print(f"  distinct GRACE cells with wells: {len(cg)} "
          f"(median {cg['n'].median():.0f} wells/cell)")
    rows.append({'product': 'GRACEDADM', 'level': 'cell', **report(
        'cell-median As vs storage mean', cg['smean'].values, cg['As_med'].values)})
    rows.append({'product': 'GRACEDADM', 'level': 'cell', **report(
        'cell-median As vs storage trend', cg['strend'].values, cg['As_med'].values)})

    # ── TELLUS LWE trend ────────────────────────────────────────────────────
    tlat, tlon, ttrend, mid = tellus_trend()
    tcell, (w_ttrend,) = nearest_cell_values(wlat, wlon, tlat, tlon, ttrend)
    print("\n[TELLUS] WELL-LEVEL (pseudo-replicated)")
    rows.append({'product': 'TELLUS', 'level': 'well', **report(
        'As vs LWE trend', w_ttrend, As)})

    # Honest DOF: aggregate by mascon_ID
    if mid is not None:
        _, (w_mid,) = nearest_cell_values(wlat, wlon, tlat, tlon,
                                          np.where(np.isfinite(ttrend), mid, np.nan))
        mdf = pd.DataFrame({'mid': w_mid, 'As': As, 'ttrend': w_ttrend})
        mg = mdf.groupby('mid').agg(As_med=('As', 'median'),
                                    ttrend=('ttrend', 'first'),
                                    n=('As', 'size')).reset_index()
        print(f"\n[TELLUS] MASCON-LEVEL (honest DOF; ~3-deg native resolution)")
        print(f"  distinct mascons with wells: {len(mg)}")
        rows.append({'product': 'TELLUS', 'level': 'mascon', **report(
            'mascon-median As vs LWE trend', mg['ttrend'].values, mg['As_med'].values)})

    out = pd.DataFrame(rows)
    from config import TABLES_DIR
    out.to_csv(TABLES_DIR / 'T1_grace_gridlevel_correlation.csv', index=False)
    print(f"\nSaved {TABLES_DIR / 'T1_grace_gridlevel_correlation.csv'}")
    print("=" * 70)


if __name__ == '__main__':
    main()
