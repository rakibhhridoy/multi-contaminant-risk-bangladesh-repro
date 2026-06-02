"""
T1 ENSEMBLE: CMIP6 multi-model precipitation transfer (2026-05-21)
==================================================================
Upgrades climate_transfer.py from a single-GCM analysis (EC-Earth3-Veg) to a
multi-model ensemble. Eliminates the most-cited reviewer objection to the
EHP submission: that climate projections rested on a single Earth-system model.

Strategy:
  1. Scan data/external/cmip6/ for all WorldClim 2.1 CMIP6 precipitation
     rasters following the convention wc2.1_2.5m_prec_<MODEL>_<SSP>_*.tif.
  2. For each (model, ssp), compute Bangladesh-mean projected mean-annual
     precipitation. Compare to a WorldClim 2.1 historical baseline if
     available; otherwise treat the inter-model spread of projected absolute
     precipitation as the uncertainty signal.
  3. Aggregate over models to produce:
       - ensemble median precipitation change (% relative to ensemble mean)
       - 5th / 95th percentile inter-model spread
  4. Propagate through the same seasonal sensitivity transfer functions used
     in climate_transfer.py, yielding zone-level contaminant projections
     with ensemble-derived uncertainty bands (replaces synthetic CLIMATE_CV
     in T5).

Outputs:
  - output/tables/T1_cmip6_ensemble_summary.csv  (model, ssp, BD-mean precip)
  - output/tables/T1_ensemble_projections_2050.csv  (zone × contam × ensemble stats)
"""

import sys
import re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import rasterio
from rasterio.windows import from_bounds

from config import EXTERNAL_DIR, TABLES_DIR

CMIP6_DIR = EXTERNAL_DIR / 'cmip6'

# Bangladesh bounding box
BD_BBOX = (87.8, 20.5, 92.8, 26.8)  # (W, S, E, N)

# Filename pattern: wc2.1_2.5m_prec_<MODEL>_<SSP>_<PERIOD>.tif
PATTERN = re.compile(
    r'^wc2\.1_2\.5m_prec_(?P<model>[A-Za-z0-9_\-]+)_'
    r'(?P<ssp>ssp\d{3})_(?P<period>\d{4}-\d{4})\.tif$'
)


def _bd_mean_precip(tif_path):
    """Return Bangladesh-domain mean of mean-annual precipitation (mm/yr).

    WorldClim 2.1 'prec' files are stacks of 12 monthly mean precipitation
    bands (mm/month). Sum across months for mean annual; then take spatial
    mean within Bangladesh bbox.
    """
    with rasterio.open(tif_path) as src:
        window = from_bounds(*BD_BBOX, src.transform)
        # WorldClim CMIP6 prec is 12 bands (Jan..Dec, mm/month)
        if src.count == 12:
            arr = src.read(window=window).astype('float64')
            # Mask nodata
            nodata = src.nodata
            if nodata is not None:
                arr = np.where(arr == nodata, np.nan, arr)
            ann_mm = np.nansum(arr, axis=0)  # 2D field of annual mm
            return float(np.nanmean(ann_mm))
        else:
            arr = src.read(1, window=window).astype('float64')
            nodata = src.nodata
            if nodata is not None:
                arr = np.where(arr == nodata, np.nan, arr)
            return float(np.nanmean(arr))


def scan_ensemble():
    """Inventory all CMIP6 model/SSP files present in data/external/cmip6/."""
    rows = []
    for p in sorted(CMIP6_DIR.glob('wc2.1_2.5m_prec_*.tif')):
        m = PATTERN.match(p.name)
        if not m:
            continue
        rows.append({
            'model': m.group('model'),
            'ssp': m.group('ssp'),
            'period': m.group('period'),
            'path': str(p),
        })
    return pd.DataFrame(rows)


def compute_ensemble_summary():
    """Compute Bangladesh-mean precip for every available model/ssp."""
    inv = scan_ensemble()
    if inv.empty:
        print("No CMIP6 rasters found")
        return None

    print(f"Found {len(inv)} CMIP6 rasters across "
          f"{inv['model'].nunique()} models, {inv['ssp'].nunique()} SSPs")

    results = []
    for _, row in inv.iterrows():
        try:
            precip = _bd_mean_precip(row['path'])
        except Exception as e:
            print(f"  SKIP {row['model']} {row['ssp']}: unreadable ({e})")
            continue
        if not np.isfinite(precip) or precip <= 0:
            print(f"  SKIP {row['model']} {row['ssp']}: invalid precip {precip}")
            continue
        results.append({
            'model': row['model'],
            'ssp': row['ssp'],
            'period': row['period'],
            'bd_mean_annual_precip_mm': precip,
        })
        print(f"  {row['model']:25s} {row['ssp']}: {precip:7.1f} mm/yr")

    df = pd.DataFrame(results)

    # Compute per-SSP ensemble statistics
    summary = df.groupby('ssp')['bd_mean_annual_precip_mm'].agg(
        median='median', mean='mean',
        p5=lambda x: np.percentile(x, 5),
        p95=lambda x: np.percentile(x, 95),
        std='std', n_models='count',
    ).reset_index()
    print("\nEnsemble summary by SSP:")
    print(summary.to_string(index=False))

    # Save
    df.to_csv(TABLES_DIR / 'T1_cmip6_ensemble_models.csv', index=False)
    summary.to_csv(TABLES_DIR / 'T1_cmip6_ensemble_summary.csv', index=False)
    return df, summary


# Literature-anchored central precipitation change for Bangladesh by 2050
# (IPCC AR6 WG1 Atlas; Alamgir et al. 2019). Used as the ensemble-median ΔP;
# the INTER-MODEL spread (computed below) supplies the uncertainty band that
# replaces the synthetic CLIMATE_CV used in the single-model T5 run.
CENTRAL_DELTA_P = {'ssp245': 12.0, 'ssp585': 18.0}  # % precip increase by 2050


def ensemble_precipitation_delta(model_df):
    """Derive a per-model precipitation-change signal ΔP for each SSP.

    All WorldClim 2.1 CMIP6 files are future projections (2041-2060) that have
    been delta-downscaled onto a COMMON observed 1970-2000 baseline, so the
    inter-model differences in the absolute future field are, to first order,
    the inter-model differences in the projected CHANGE (the shared baseline
    cancels). We exploit this: we anchor the ensemble-MEAN ΔP to the
    well-established IPCC AR6 / Alamgir 2019 central value, which implies a
    common baseline B = mean(precip)/(1 + ΔP_central), and then read each
    model's GENUINE fractional change directly off that baseline:

        B_ssp     = mean_m(precip_m) / (1 + ΔP_central(ssp)/100)
        ΔP_model  = 100 × (precip_model − B_ssp) / B_ssp

    Unlike the earlier wetness-ratio scaling (ΔP = ΔP_central × precip/mean),
    which mechanically compressed the spread to the CV of the absolute field
    (~4%), this recovers the true inter-model spread of the fractional change
    (CV ~0.2-0.4). The ensemble mean still equals the literature ΔP_central by
    construction, but the spread is no longer artificially narrow. NB: even
    this is a lower bound, because delta-downscaling smooths the raw GCM
    monsoon-change disagreement.
    """
    model_df = model_df.copy()
    deltas = []
    for ssp, grp in model_df.groupby('ssp'):
        ens_mean = grp['bd_mean_annual_precip_mm'].mean()
        central = CENTRAL_DELTA_P.get(ssp, 12.0)
        baseline = ens_mean / (1.0 + central / 100.0)  # implied common baseline
        for idx, row in grp.iterrows():
            dp = 100.0 * (row['bd_mean_annual_precip_mm'] - baseline) / baseline
            deltas.append((idx, dp))
    for idx, dp in deltas:
        model_df.loc[idx, 'delta_P_pct'] = dp
    # Empirical inter-model CV of ΔP per SSP (for T5)
    cv_table = model_df.groupby('ssp')['delta_P_pct'].agg(
        mean='mean', std='std',
        cv=lambda x: x.std() / x.mean() if x.mean() else np.nan,
        n='count',
    ).reset_index()
    cv_table.to_csv(TABLES_DIR / 'T1_ensemble_deltaP_cv.csv', index=False)
    print("\nInter-model ΔP spread (replaces synthetic CLIMATE_CV in T5):")
    print(cv_table.to_string(index=False))
    return model_df, cv_table


# Central temperature change by 2050 and per-contaminant temperature
# sensitivity — identical to climate_transfer.py so the ensemble reproduces
# the single-model central projection (and only adds inter-model ΔP spread).
TEMP_CHANGE_2050 = {'ssp245': 1.3, 'ssp585': 2.1}  # °C
TEMP_SENSITIVITY = {'As': 0.10, 'Mn2+': 0.08, 'Fe2+': 0.06, 'Cr3+': 0.03, 'NO3-': 0.05}


def propagate_to_contaminants(model_df):
    """Multi-model ensemble of contaminant projections.

    Uses the EXACT transfer function from climate_transfer.py:

        precip_delta = sensitivity_per_pct × ΔP          (absolute units)
        temp_delta   = baseline × temp_sens × ΔT
        C_2050       = baseline + precip_delta + temp_delta

    where sensitivity_per_pct = (C_wet − C_dry)/30 is in absolute concentration
    units per 1% precipitation change. ΔP varies across the CMIP6 ensemble;
    ΔT is held at its central value (temperature ensemble is a follow-up), so
    the reported spread isolates the precipitation-driven inter-model
    uncertainty — the dominant climate-uncertainty term for monsoon-driven
    arsenic mobilisation.
    """
    transfer_file = TABLES_DIR / 'T1_seasonal_transfer.csv'
    if not transfer_file.exists():
        print("T1_seasonal_transfer.csv missing; cannot propagate.")
        return None

    transfer = pd.read_csv(transfer_file)
    print(f"\nPropagating ensemble through {len(transfer)} "
          "depth×zone×contaminant sensitivity cells (additive transfer)…")

    out_rows = []
    for _, row in transfer.iterrows():
        cont = row.get('contaminant')
        if cont == 'ORP':
            continue
        sens = row.get('sensitivity_per_pct', np.nan)
        baseline = row.get('dry_median', np.nan)
        if pd.isna(sens) or pd.isna(baseline) or baseline <= 0:
            continue
        t_sens = TEMP_SENSITIVITY.get(cont, 0.05)
        for ssp, sub in model_df.groupby('ssp'):
            dT = TEMP_CHANGE_2050.get(ssp, 1.3)
            temp_delta = baseline * t_sens * dT
            projections = np.array([
                max(0.0, baseline + sens * m['delta_P_pct'] + temp_delta)
                for _, m in sub.iterrows()
            ])
            if len(projections) == 0:
                continue
            out_rows.append({
                'depth_zone': row.get('depth_zone'),
                'phys_zone': row.get('phys_zone'),
                'contaminant': cont,
                'ssp': ssp,
                'n_models': len(projections),
                'baseline_dry': baseline,
                'ensemble_median': float(np.median(projections)),
                'ensemble_p5': float(np.percentile(projections, 5)),
                'ensemble_p95': float(np.percentile(projections, 95)),
                'ensemble_std': float(np.std(projections)),
                'pct_change_median': 100 * (np.median(projections) - baseline) / baseline,
                'pct_change_p5': 100 * (np.percentile(projections, 5) - baseline) / baseline,
                'pct_change_p95': 100 * (np.percentile(projections, 95) - baseline) / baseline,
            })

    out = pd.DataFrame(out_rows)
    out.to_csv(TABLES_DIR / 'T1_ensemble_projections_2050.csv', index=False)
    print(f"  wrote {len(out)} ensemble projection rows")
    return out


def main():
    print("=" * 70)
    print("T1 CMIP6 ENSEMBLE — multi-model precipitation transfer")
    print("=" * 70)

    res = compute_ensemble_summary()
    if res is None:
        print("Aborting: no CMIP6 files available")
        return
    model_df, summary = res

    model_df, cv_table = ensemble_precipitation_delta(model_df)
    model_df.to_csv(TABLES_DIR / 'T1_cmip6_ensemble_models.csv', index=False)

    out = propagate_to_contaminants(model_df)
    if out is not None and len(out) > 0:
        print("\n--- Arsenic projections by zone (SSP5-8.5), shallow ---")
        as_585 = out[(out['contaminant'].astype(str).str.contains('As'))
                     & (out['ssp'] == 'ssp585')
                     & (out['depth_zone'] == 'Shallow')].sort_values(
            'pct_change_median', ascending=False)
        print(as_585[['phys_zone', 'baseline_dry', 'ensemble_median',
                      'pct_change_median', 'pct_change_p5',
                      'pct_change_p95', 'n_models']].to_string(index=False))
        # National arsenic ensemble change (median over zones/depths)
        as_all = out[out['contaminant'].astype(str).str.contains('As')]
        for ssp in ['ssp245', 'ssp585']:
            s = as_all[as_all['ssp'] == ssp]
            if len(s):
                print(f"\n  National As change {ssp}: "
                      f"median {s['pct_change_median'].median():+.1f}% "
                      f"[{s['pct_change_p5'].median():+.1f}%, "
                      f"{s['pct_change_p95'].median():+.1f}%] "
                      f"across {s['n_models'].iloc[0]} models")

    print("=" * 70)


if __name__ == '__main__':
    main()
