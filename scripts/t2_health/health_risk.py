"""
T2: Multi-Contaminant Cumulative Health Risk Assessment
Calculates HI, CR, and DALYs for As + Mn + Cr + Fe + NO3 + Al
Stratified by zone × depth × season × age-sex group
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from config import (
    DATA_FILE, HEALTH_CONTAMINANTS, REFERENCE_DOSES,
    CANCER_SLOPE_FACTORS, CR_VI_FRACTION, EXPOSURE_PARAMS,
    AT_CARCINOGENIC, DEPTH_ZONES, PHYSIOGRAPHIC_ZONES,
    TABLES_DIR, FIGURES_DIR, RANDOM_STATE
)


def load_data():
    """Load and prepare dataset."""
    df = pd.read_csv(DATA_FILE)
    # Assign depth zones
    for zone, (lo, hi) in DEPTH_ZONES.items():
        df.loc[(df['Depth'] >= lo) & (df['Depth'] < hi), 'depth_zone'] = zone
    # Assign physiographic zones
    for zone, bbox in PHYSIOGRAPHIC_ZONES.items():
        mask = ((df['Latitude'] >= bbox['lat'][0]) & (df['Latitude'] <= bbox['lat'][1]) &
                (df['Longitude'] >= bbox['lon'][0]) & (df['Longitude'] <= bbox['lon'][1]))
        df.loc[mask, 'phys_zone'] = zone
    return df


def calculate_cdi(concentration_mg_L, params):
    """Calculate Chronic Daily Intake (mg/kg/day)."""
    return (concentration_mg_L * params['ir_L_day'] * params['ef_days'] * params['ed_years']) / \
           (params['bw_kg'] * params['ed_years'] * 365)


def calculate_hi(df, age_group='adult_male'):
    """Calculate non-carcinogenic Hazard Index for each sample."""
    params = EXPOSURE_PARAMS[age_group]
    hi_total = np.zeros(len(df))

    for contaminant, rfd in REFERENCE_DOSES.items():
        col = contaminant
        if col in df.columns:
            # Convert As from µg/L to mg/L
            conc = df[col].values.copy()
            if contaminant == 'As':
                conc = conc / 1000.0
            cdi = calculate_cdi(conc, params)
            hq = cdi / rfd
            hi_total += hq
            df[f'HQ_{contaminant}_{age_group}'] = hq

    df[f'HI_{age_group}'] = hi_total
    return df


def calculate_cr(df, age_group='adult_male'):
    """Calculate carcinogenic risk."""
    params = EXPOSURE_PARAMS[age_group]
    cr_total = np.zeros(len(df))

    for contaminant, csf in CANCER_SLOPE_FACTORS.items():
        col = contaminant
        if col in df.columns:
            conc = df[col].values.copy()
            if contaminant == 'As':
                conc = conc / 1000.0
            cdi = (conc * params['ir_L_day'] * params['ef_days'] * params['ed_years']) / \
                  (params['bw_kg'] * AT_CARCINOGENIC)
            cr = cdi * csf
            cr_total += cr
            df[f'CR_{contaminant}_{age_group}'] = cr

    df[f'CR_total_{age_group}'] = cr_total
    return df


def main():
    print("=" * 65)
    print("T2: MULTI-CONTAMINANT HEALTH RISK ASSESSMENT")
    print("=" * 65)

    df = load_data()
    print(f"Loaded {len(df)} samples")

    # Calculate for all age groups
    for age_group in EXPOSURE_PARAMS.keys():
        df = calculate_hi(df, age_group)
        df = calculate_cr(df, age_group)
        print(f"\n{age_group}:")
        print(f"  HI > 1 (non-carcinogenic risk): {(df[f'HI_{age_group}'] > 1).sum()} "
              f"({(df[f'HI_{age_group}'] > 1).mean()*100:.1f}%)")
        print(f"  CR > 1e-4 (unacceptable cancer risk): {(df[f'CR_total_{age_group}'] > 1e-4).sum()} "
              f"({(df[f'CR_total_{age_group}'] > 1e-4).mean()*100:.1f}%)")

    # Save results
    df.to_csv(TABLES_DIR / 'T2_health_risk_all_samples.csv', index=False)
    print(f"\nSaved to {TABLES_DIR / 'T2_health_risk_all_samples.csv'}")


if __name__ == '__main__':
    main()
