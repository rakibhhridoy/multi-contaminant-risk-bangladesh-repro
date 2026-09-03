"""
Configuration for Paper2: Climate Change & Multi-Contaminant Health Risk
Target: Environment International (~8,000 words) / Lancet Planetary Health (compressed ~5,000)
"""

from pathlib import Path

# ─── PATHS ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR     = PROJECT_ROOT / 'data'
EXTERNAL_DIR = DATA_DIR / 'external'
PROCESSED_DIR = DATA_DIR / 'processed'
OUTPUT_DIR   = PROJECT_ROOT / 'output'
FIGURES_DIR  = OUTPUT_DIR / 'figures'
TABLES_DIR   = OUTPUT_DIR / 'tables'
MAPS_DIR     = OUTPUT_DIR / 'maps'

for d in [OUTPUT_DIR, FIGURES_DIR, TABLES_DIR, MAPS_DIR, EXTERNAL_DIR, PROCESSED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── PRIMARY DATASET ─────────────────────────────────────────────────────────
# 1,807 samples × 45 features from Paper1 (Zenodo DOI: 10.5281/zenodo.19148957)
DATA_FILE = DATA_DIR / 'bangladesh_groundwater_arsenic_1807samples.csv'

# ─── COLUMN DEFINITIONS ─────────────────────────────────────────────────────
TARGET = 'As'  # µg/L
COL_DEPTH   = 'Depth'
COL_SEASON  = 'Season'
COL_LAT     = 'Latitude'
COL_LON     = 'Longitude'
COL_DISTRICT = 'District'

# Health-relevant contaminants (measured in dataset)
HEALTH_CONTAMINANTS = {
    'As':   {'unit': 'µg/L', 'who_guideline': 10.0,   'epa_mcl': 10.0},
    # Mn: WHO 2021 provisional guideline value (background document
    # WHO/HEP/ECH/WSH/2021.5). The 2011 4th-edition health-based value of 0.4 mg/L
    # is superseded; 0.1 mg/L is an acceptability (staining/taste) value.
    'Mn2+': {'unit': 'mg/L', 'who_guideline': 0.08,   'epa_mcl': 0.05},
    # Fe: WHO provisional HEALTH-BASED value. 0.3 mg/L is the EPA secondary MCL,
    # an aesthetic (staining) threshold, and is not health-based.
    'Fe2+': {'unit': 'mg/L', 'who_guideline': 2.0,    'epa_mcl': 0.3},
    'Cr3+': {'unit': 'mg/L', 'who_guideline': 0.05,   'epa_mcl': 0.1},
    'Al3+': {'unit': 'mg/L', 'who_guideline': 0.2,    'epa_mcl': 0.2},
    'Cu2+': {'unit': 'mg/L', 'who_guideline': 2.0,    'epa_mcl': 1.3},
    'NO3-': {'unit': 'mg/L', 'who_guideline': 50.0,   'epa_mcl': 10.0},
}

# EPA IRIS reference doses (mg/kg/day) for non-carcinogenic risk
REFERENCE_DOSES = {
    'As':   3.0e-4,    # EPA IRIS oral RfD
    'Mn2+': 2.4e-2,    # EPA IRIS oral RfD
    'Cr3+': 1.5e+0,    # EPA IRIS oral RfD (Cr III)
    'Al3+': 1.0e+0,    # EPA PPRTV
    'Cu2+': 4.0e-2,    # EPA IRIS oral RfD
    'NO3-': 1.6e+0,    # EPA IRIS oral RfD (as NO3)
}

# Cancer slope factors (per mg/kg/day) — carcinogenic contaminants only
CANCER_SLOPE_FACTORS = {
    'As': 1.5,         # EPA IRIS oral CSF (Group A carcinogen)
    # Cr(VI) CSF = 0.5, but we measure Cr(III) — apply Cr(VI) fraction from literature
}

# Cr(VI) fraction of total Cr (literature estimate for reducing groundwater)
CR_VI_FRACTION = 0.1   # Conservative; Bangladesh GW is mostly reducing → low Cr(VI)

# ─── EXPOSURE PARAMETERS ─────────────────────────────────────────────────────
# WHO/ICMR defaults for Bangladesh tropical climate
EXPOSURE_PARAMS = {
    'adult_male':   {'bw_kg': 60, 'ir_L_day': 3.5, 'ef_days': 365, 'ed_years': 30},
    'adult_female': {'bw_kg': 50, 'ir_L_day': 3.0, 'ef_days': 365, 'ed_years': 30},
    'child_6_17':   {'bw_kg': 25, 'ir_L_day': 1.5, 'ef_days': 365, 'ed_years': 12},
    'child_0_5':    {'bw_kg': 10, 'ir_L_day': 0.7, 'ef_days': 365, 'ed_years': 6},
}
AT_CARCINOGENIC = 70 * 365  # Averaging time for cancer risk (lifetime, days)

# ─── DEPTH ZONES ─────────────────────────────────────────────────────────────
DEPTH_ZONES = {
    'Shallow':      (0,   30),
    'Intermediate': (30,  80),
    'Medium_Deep':  (80,  150),
    'Deep':         (150, 500),
}

# ─── PHYSIOGRAPHIC ZONES ────────────────────────────────────────────────────
PHYSIOGRAPHIC_ZONES = {
    'Barind_Tract':           {'lat': (23.8, 25.6), 'lon': (88.0, 89.2)},
    'Northern_Terrace':       {'lat': (25.2, 26.7), 'lon': (88.0, 89.8)},
    'Brahmaputra_Floodplain': {'lat': (23.5, 25.5), 'lon': (89.5, 91.2)},
    'Ganges_Floodplain':      {'lat': (23.0, 24.5), 'lon': (88.5, 90.2)},
    'GBM_Delta':              {'lat': (21.5, 23.5), 'lon': (89.0, 90.5)},
    'Meghna_Floodplain':      {'lat': (22.5, 24.5), 'lon': (90.5, 91.5)},
    'Eastern_Hills':          {'lat': (21.5, 24.8), 'lon': (91.5, 92.7)},
}

# ─── CLIMATE SCENARIOS ───────────────────────────────────────────────────────
SSP_SCENARIOS = ['SSP2-4.5', 'SSP5-8.5']
PROJECTION_YEARS = [2030, 2040, 2050, 2060]
CMIP6_MODELS = ['EC-Earth3', 'GFDL-ESM4', 'MPI-ESM1-2-HR']  # ensemble

# ─── STATISTICAL PARAMETERS ─────────────────────────────────────────────────
ALPHA        = 0.05
RANDOM_STATE = 42
N_BOOTSTRAP  = 1000
N_JOBS       = 2   # Mac safe

# Bayesian (T5)
BAYES_DRAWS  = 4000
BAYES_TUNE   = 2000
BAYES_CHAINS = 4

# ─── SPATIAL ─────────────────────────────────────────────────────────────────
BANGLADESH_BBOX = {'lat_min': 20.5, 'lat_max': 26.8,
                   'lon_min': 87.8, 'lon_max': 92.8}

# ─── Fe2+ PROVISIONAL REFERENCE DOSE ───────────────────────────────────────
# WHO provisional guideline 2 mg/L for health; EPA secondary MCL 0.3 mg/L (aesthetic)
# Using ATSDR intermediate oral MRL as proxy RfD
REFERENCE_DOSES['Fe2+'] = 0.7  # mg/kg/day — ATSDR MRL (2024), conservative

# ─── PLOTTING ────────────────────────────────────────────────────────────────
DPI         = 300
FIGURE_SIZE = (14, 9)
COLORMAP    = 'RdYlBu_r'


# ─── SHARED ZONE ASSIGNMENT ─────────────────────────────────────────────────
import numpy as _np

def assign_zones(df):
    """Assign depth and physiographic zones with nearest-centroid tiebreaking.

    For samples matching multiple physiographic zone bounding boxes,
    assigns to the zone whose centroid is closest (Euclidean distance).
    Returns df with 'depth_zone' and 'phys_zone' columns added.
    Drops rows that match no zone.
    """
    # Depth zones
    for zone, (lo, hi) in DEPTH_ZONES.items():
        df.loc[(df['Depth'] >= lo) & (df['Depth'] < hi), 'depth_zone'] = zone

    # Physiographic zones — nearest centroid for overlaps
    centroids = {}
    for zone, bbox in PHYSIOGRAPHIC_ZONES.items():
        centroids[zone] = (
            (bbox['lat'][0] + bbox['lat'][1]) / 2,
            (bbox['lon'][0] + bbox['lon'][1]) / 2,
        )

    phys_zone = [None] * len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        lat, lon = row['Latitude'], row['Longitude']
        matches = []
        for zone, bbox in PHYSIOGRAPHIC_ZONES.items():
            if (bbox['lat'][0] <= lat <= bbox['lat'][1] and
                bbox['lon'][0] <= lon <= bbox['lon'][1]):
                matches.append(zone)

        if len(matches) == 1:
            phys_zone[i] = matches[0]
        elif len(matches) > 1:
            # Nearest centroid
            best_zone = min(matches, key=lambda z: (
                (lat - centroids[z][0])**2 + (lon - centroids[z][1])**2
            ))
            phys_zone[i] = best_zone

    df['phys_zone'] = phys_zone
    df = df.dropna(subset=['depth_zone', 'phys_zone'])
    return df
