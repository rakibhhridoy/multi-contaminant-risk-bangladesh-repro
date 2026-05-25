"""
Create two new T2 panels:
1. Merged F05+F06 (DALYs by zone + Multi vs As-only DALY) side-by-side
2. Spatial risk map of HI across Bangladesh
"""
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from pathlib import Path
from config import (
    DATA_FILE, HEALTH_CONTAMINANTS, REFERENCE_DOSES,
    EXPOSURE_PARAMS, DEPTH_ZONES, PHYSIOGRAPHIC_ZONES,
    FIGURES_DIR, assign_zones, COL_LAT, COL_LON,
    BANGLADESH_BBOX
)

DPI = 300
from config import FIGURES_DIR
OUT_DIR = FIGURES_DIR

ZONE_COLORS = {
    'Barind_Tract': '#7B1FA2', 'Northern_Terrace': '#1976D2',
    'Brahmaputra_Floodplain': '#00838F', 'Ganges_Floodplain': '#C62828',
    'GBM_Delta': '#E65100', 'Meghna_Floodplain': '#2E7D32',
    'Eastern_Hills': '#5D4037'
}

ZONE_LABELS = {
    'Barind_Tract': 'Barind Tract', 'Northern_Terrace': 'Northern Terrace',
    'Brahmaputra_Floodplain': 'Brahmaputra FP',
    'Ganges_Floodplain': 'Ganges FP',
    'GBM_Delta': 'GBM Delta', 'Meghna_Floodplain': 'Meghna FP',
    'Eastern_Hills': 'Eastern Hills'
}


def set_pub_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        'axes.labelsize': 11,
        'axes.labelweight': 'bold',
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,
        'legend.framealpha': 0.9,
        'figure.dpi': 150,
    })


def load_data():
    df = pd.read_csv(DATA_FILE)
    df = assign_zones(df)
    df = df.dropna(subset=['phys_zone'])
    return df


def compute_hi(df):
    """Compute cumulative Hazard Index for each sample."""
    adult = EXPOSURE_PARAMS['adult_male']
    IR = adult['ir_L_day']
    BW = adult['bw_kg']
    EF = adult['ef_days']
    ED = adult['ed_years']
    AT = ED * 365

    hi = np.zeros(len(df))
    for contam, info in HEALTH_CONTAMINANTS.items():
        if contam in df.columns:
            conc = df[contam].fillna(0).values
            # Convert µg/L to mg/L for As
            if info.get('unit') == 'µg/L':
                conc = conc / 1000.0
            cdi = (conc * IR * EF * ED) / (BW * AT)
            rfd = REFERENCE_DOSES.get(contam, 1e10)
            hi += cdi / rfd
    return hi


def create_merged_daly(df):
    """Create merged F05+F06: DALYs by zone (left) + Multi vs As-only (right)."""
    set_pub_style()

    daly_zone = pd.read_csv('output/tables/T2_daly_by_zone.csv')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    zones_order = ['Ganges_Floodplain', 'GBM_Delta', 'Meghna_Floodplain',
                   'Barind_Tract', 'Northern_Terrace', 'Brahmaputra_Floodplain',
                   'Eastern_Hills']

    # --- Left: DALY per 100k by zone (dry vs wet) ---
    # Aggregate to zone-level (average across seasons for the bar, show both)
    for i, z in enumerate(zones_order):
        zdata = daly_zone[daly_zone['phys_zone'] == z]
        dry = zdata[zdata['season'] == 'Dry']['DALY_per_100k'].values
        wet = zdata[zdata['season'] == 'Wet']['DALY_per_100k'].values
        c = ZONE_COLORS.get(z, '#888')
        if len(dry) > 0:
            ax1.barh(i - 0.18, dry[0], height=0.32, color=c, alpha=0.6,
                     label='Dry' if i == 0 else '')
        if len(wet) > 0:
            ax1.barh(i + 0.18, wet[0], height=0.32, color=c, alpha=1.0,
                     label='Wet' if i == 0 else '')

    labels = [ZONE_LABELS.get(z, z) for z in zones_order]
    ax1.set_yticks(range(len(zones_order)))
    ax1.set_yticklabels(labels)
    ax1.set_xlabel('DALYs per 100,000')
    ax1.set_title('DALYs by Zone & Season')
    ax1.invert_yaxis()
    ax1.legend(loc='lower right', fontsize=9)

    # --- Right: Multi vs As-only DALY by zone ---
    for i, z in enumerate(zones_order):
        zdata = daly_zone[daly_zone['phys_zone'] == z]
        multi = zdata['DALY_multi'].sum()
        as_only = zdata['DALY_as_only'].sum()
        pop = zdata['population'].iloc[0] if len(zdata) > 0 else 1
        multi_rate = multi / pop * 100000
        as_rate = as_only / pop * 100000

        ax2.barh(i - 0.18, multi_rate, height=0.32, color='#C62828', alpha=0.9,
                 label='Multi-contaminant' if i == 0 else '')
        ax2.barh(i + 0.18, as_rate, height=0.32, color='#1565C0', alpha=0.9,
                 label='As-only' if i == 0 else '')

    ax2.set_yticks(range(len(zones_order)))
    ax2.set_yticklabels(labels)
    ax2.set_xlabel('DALYs per 100,000')
    ax2.set_title('Multi vs. As-only DALYs')
    ax2.invert_yaxis()
    ax2.legend(loc='lower right', fontsize=9)

    fig.tight_layout()
    out = OUT_DIR / 'T2_F05F06_merged.png'
    fig.savefig(out, dpi=DPI, bbox_inches='tight', facecolor='white')
    fig.savefig(FIGURES_DIR / 'T2_F05F06_merged.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  ✓ T2_F05F06_merged.png')


def create_spatial_risk_map(df):
    """Create spatial map of HI across Bangladesh."""
    set_pub_style()

    hi = compute_hi(df)
    df = df.copy()
    df['HI'] = hi

    # Filter to Bangladesh bbox
    mask = ((df['Latitude'] >= BANGLADESH_BBOX['lat_min']) &
            (df['Latitude'] <= BANGLADESH_BBOX['lat_max']) &
            (df['Longitude'] >= BANGLADESH_BBOX['lon_min']) &
            (df['Longitude'] <= BANGLADESH_BBOX['lon_max']))
    df = df[mask].copy()

    fig, ax = plt.subplots(1, 1, figsize=(7, 8))

    # Try to load Bangladesh boundary shapefile
    bd_boundary = None
    try:
        import geopandas as gpd
        # Try naturalearth
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres')
                              if hasattr(gpd.datasets, 'get_path')
                              else gpd.datasets.get_path('naturalearth_lowres'))
        bd = world[world['name'] == 'Bangladesh']
        if len(bd) > 0:
            bd_boundary = bd
            bd.boundary.plot(ax=ax, color='black', linewidth=1.5)
    except Exception:
        pass

    # Draw zone boundaries as rectangles
    for zone, bbox in PHYSIOGRAPHIC_ZONES.items():
        from matplotlib.patches import Rectangle
        rect = Rectangle((bbox['lon'][0], bbox['lat'][0]),
                          bbox['lon'][1] - bbox['lon'][0],
                          bbox['lat'][1] - bbox['lat'][0],
                          linewidth=0.8, edgecolor=ZONE_COLORS.get(zone, '#888'),
                          facecolor='none', linestyle='--', alpha=0.5)
        ax.add_patch(rect)

    # Classify HI
    hi_cats = pd.cut(df['HI'], bins=[0, 1, 2, 5, 10, np.inf],
                     labels=['<1 (Safe)', '1–2 (Low)', '2–5 (Moderate)',
                             '5–10 (High)', '>10 (Very High)'])
    cat_colors = {
        '<1 (Safe)': '#2E7D32',
        '1–2 (Low)': '#F9A825',
        '2–5 (Moderate)': '#E65100',
        '5–10 (High)': '#C62828',
        '>10 (Very High)': '#4A148C'
    }

    for cat in ['<1 (Safe)', '1–2 (Low)', '2–5 (Moderate)', '5–10 (High)', '>10 (Very High)']:
        mask = hi_cats == cat
        if mask.sum() > 0:
            ax.scatter(df.loc[mask, 'Longitude'], df.loc[mask, 'Latitude'],
                       c=cat_colors[cat], s=15, alpha=0.7, edgecolors='none',
                       label=f'{cat} (n={mask.sum()})', zorder=3)

    ax.set_xlim(BANGLADESH_BBOX['lon_min'], BANGLADESH_BBOX['lon_max'])
    ax.set_ylim(BANGLADESH_BBOX['lat_min'], BANGLADESH_BBOX['lat_max'])
    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')
    ax.set_title('Spatial Distribution of\nCumulative Hazard Index')
    ax.legend(title='HI Category', loc='upper right', fontsize=8, title_fontsize=9,
              framealpha=0.95, markerscale=1.5)
    ax.set_aspect('equal')

    # Add zone labels
    for zone, bbox in PHYSIOGRAPHIC_ZONES.items():
        cx = (bbox['lon'][0] + bbox['lon'][1]) / 2
        cy = (bbox['lat'][0] + bbox['lat'][1]) / 2
        ax.text(cx, cy, ZONE_LABELS.get(zone, zone), fontsize=6,
                ha='center', va='center', alpha=0.4, fontweight='bold',
                color=ZONE_COLORS.get(zone, '#888'))

    fig.tight_layout()
    out = OUT_DIR / 'T2_spatial_risk_map.png'
    fig.savefig(out, dpi=DPI, bbox_inches='tight', facecolor='white')
    fig.savefig(FIGURES_DIR / 'T2_spatial_risk_map.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  ✓ T2_spatial_risk_map.png')


if __name__ == '__main__':
    print('Loading data...')
    df = load_data()
    print(f'  {len(df)} samples with zone assignment\n')

    print('Creating merged DALY panel (F05+F06)...')
    create_merged_daly(df)

    print('Creating spatial risk map...')
    create_spatial_risk_map(df)

    print('\nDone!')
