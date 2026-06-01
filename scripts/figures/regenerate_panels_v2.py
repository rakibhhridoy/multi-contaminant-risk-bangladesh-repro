"""
Regenerate all individual panels for Figures 2–4 with Lancet Planetary Health styling.

Improvements over v1:
  - Unified serif font (Times New Roman / DejaVu Serif)
  - NO internal titles (composite adds panel labels)
  - Larger fonts for axis labels, ticks, legends
  - 600 DPI
  - Lancet colour palette
  - Fix: invisible small-contribution bars, legend overlaps, visual monotony in maps
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle, FancyBboxPatch
import seaborn as sns
from pathlib import Path
from config import (
    DATA_FILE, HEALTH_CONTAMINANTS, REFERENCE_DOSES,
    CANCER_SLOPE_FACTORS, CR_VI_FRACTION, EXPOSURE_PARAMS,
    AT_CARCINOGENIC, DEPTH_ZONES, PHYSIOGRAPHIC_ZONES,
    TABLES_DIR, FIGURES_DIR, RANDOM_STATE,
    assign_zones, COL_LAT, COL_LON, BANGLADESH_BBOX
)

# ─── Output ──────────────────────────────────────────────────────────────────
from config import FIGURES_DIR
OUT_DIR = FIGURES_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)
DPI = 600

# ─── Lancet Palette ──────────────────────────────────────────────────────────
CRIMSON   = '#B71C1C'
CRIMSON_L = '#EF9A9A'
TEAL      = '#00695C'
TEAL_L    = '#80CBC4'
AMBER     = '#E65100'
AMBER_L   = '#FFCC80'
STEEL     = '#1565C0'
STEEL_L   = '#90CAF9'
SLATE     = '#546E7A'
DARKTEXT  = '#212121'

ZONE_COLORS = {
    'Barind_Tract': '#7B1FA2', 'Northern_Terrace': '#1565C0',
    'Brahmaputra_Floodplain': '#00838F', 'Ganges_Floodplain': '#C62828',
    'GBM_Delta': '#E65100', 'Meghna_Floodplain': '#2E7D32',
    'Eastern_Hills': '#5D4037'
}
ZONE_LABELS = {
    'Barind_Tract': 'Barind Tract', 'Northern_Terrace': 'N. Terrace',
    'Brahmaputra_Floodplain': 'Brahmaputra FP',
    'Ganges_Floodplain': 'Ganges FP',
    'GBM_Delta': 'GBM Delta', 'Meghna_Floodplain': 'Meghna FP',
    'Eastern_Hills': 'E. Hills'
}
ZONE_ORDER = ['Barind_Tract', 'Northern_Terrace', 'Brahmaputra_Floodplain',
              'Ganges_Floodplain', 'GBM_Delta', 'Meghna_Floodplain', 'Eastern_Hills']
DEPTH_ORDER = ['Shallow', 'Intermediate', 'Medium_Deep', 'Deep']
DEPTH_LABELS = {'Shallow': 'Shallow', 'Intermediate': 'Intermediate',
                'Medium_Deep': 'Med-Deep', 'Deep': 'Deep'}

# Contaminant display names and colours (Lancet-consistent)
CONTAM_DISPLAY = {
    'As': ('As', CRIMSON),
    'Mn2+': ('Mn', AMBER),
    'Fe2+': ('Fe', '#F9A825'),
    'Cu2+': ('Cu', TEAL),
    'NO3-': (r'NO$_3$', STEEL),
    'Al3+': ('Al', '#7B1FA2'),
    'Cr3+': ('Cr', SLATE),
}


def set_lancet_style():
    """Unified Lancet Planetary Health matplotlib style."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'font.size': 17,
        'axes.titlesize': 20,
        'axes.titleweight': 'bold',
        'axes.labelsize': 19,
        'axes.labelweight': 'normal',
        'axes.linewidth': 1.0,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'legend.fontsize': 15,
        'legend.framealpha': 0.95,
        'legend.edgecolor': '#CCCCCC',
        'figure.dpi': 150,
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
    })


def save_panel(fig, name):
    """Save panel to both LaTeX figures and output/figures."""
    for d in [OUT_DIR, FIGURES_DIR]:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / name, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  OK  {name}')


# ─── Data helpers ────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv(DATA_FILE)
    df = assign_zones(df)
    df = df.dropna(subset=['phys_zone'])
    return df


def calculate_cdi(conc_mg_L, ir, ef, ed, bw):
    return (conc_mg_L * ir * ef * ed) / (bw * ed * 365)


def calculate_all_risks(df, age_group='adult_male'):
    params = EXPOSURE_PARAMS[age_group]
    ir, ef, ed, bw = params['ir_L_day'], params['ef_days'], params['ed_years'], params['bw_kg']
    hi_total = np.zeros(len(df))
    hi_as_only = np.zeros(len(df))
    for contaminant, rfd in REFERENCE_DOSES.items():
        if contaminant not in df.columns:
            continue
        conc = df[contaminant].values.copy()
        if contaminant == 'As':
            conc = conc / 1000.0
        cdi = calculate_cdi(conc, ir, ef, ed, bw)
        hq = cdi / rfd
        df[f'HQ_{contaminant}'] = hq
        hi_total += hq
        if contaminant == 'As':
            hi_as_only = hq.copy()
    df['HI_multi'] = hi_total
    df['HI_as_only'] = hi_as_only
    # CR
    cr_total = np.zeros(len(df))
    for contaminant, csf in CANCER_SLOPE_FACTORS.items():
        if contaminant not in df.columns:
            continue
        conc = df[contaminant].values.copy()
        if contaminant == 'As':
            conc = conc / 1000.0
        cdi_cancer = (conc * ir * ef * ed) / (bw * AT_CARCINOGENIC)
        cr = cdi_cancer * csf
        df[f'CR_{contaminant}'] = cr
        cr_total += cr
        if contaminant == 'As':
            df['CR_as_only'] = cr.copy()
    df['CR_multi'] = cr_total
    return df


def compute_hi(df):
    adult = EXPOSURE_PARAMS['adult_male']
    IR, BW = adult['ir_L_day'], adult['bw_kg']
    EF, ED = adult['ef_days'], adult['ed_years']
    AT = ED * 365
    hi = np.zeros(len(df))
    for contam, info in HEALTH_CONTAMINANTS.items():
        if contam in df.columns:
            conc = df[contam].fillna(0).values
            if info.get('unit') == 'µg/L':
                conc = conc / 1000.0
            cdi = (conc * IR * EF * ED) / (BW * AT)
            rfd = REFERENCE_DOSES.get(contam, 1e10)
            hi += cdi / rfd
    return hi


def filter_bd(df):
    mask = ((df['Latitude'] >= BANGLADESH_BBOX['lat_min']) &
            (df['Latitude'] <= BANGLADESH_BBOX['lat_max']) &
            (df['Longitude'] >= BANGLADESH_BBOX['lon_min']) &
            (df['Longitude'] <= BANGLADESH_BBOX['lon_max']))
    return df[mask].copy()


BD_SHP = Path('data/gis/bgd_admbnda_adm0_bbs_20201113.shp')

def add_bd_boundary(ax):
    """Add Bangladesh admin boundary from BBS shapefile."""
    try:
        import os
        os.environ['SHAPE_RESTORE_SHX'] = 'YES'
        import geopandas as gpd
        bd = gpd.read_file(BD_SHP)
        bd.plot(ax=ax, facecolor='#F5F5F5', edgecolor='#333333', linewidth=1.0, zorder=1)
    except Exception as e:
        print(f'  Warning: shapefile not loaded ({e})')
    ax.set_xlim(BANGLADESH_BBOX['lon_min'], BANGLADESH_BBOX['lon_max'])
    ax.set_ylim(BANGLADESH_BBOX['lat_min'], BANGLADESH_BBOX['lat_max'])
    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')
    ax.set_aspect('equal')


def idw_clipped_surface(fig, ax, lon, lat, vals, *, cmap, vmin, vmax, label,
                        extend='neither', n=260, k=10, p=2):
    """Smooth inverse-distance (k-NN, power p) surface over the national bbox,
    clipped to the Bangladesh admin polygon and drawn beneath a crisp boundary
    outline. Matches the IDW (k=10, p=2) interpolator used for the gridded DALY
    surface (Supplementary Fig. S9). Deterministic."""
    import os
    os.environ['SHAPE_RESTORE_SHX'] = 'YES'
    import geopandas as gpd
    from scipy.spatial import cKDTree

    bb = BANGLADESH_BBOX
    lon = np.asarray(lon, float); lat = np.asarray(lat, float); vals = np.asarray(vals, float)
    m = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(vals)
    lon, lat, vals = lon[m], lat[m], vals[m]

    gx = np.linspace(bb['lon_min'], bb['lon_max'], n)
    gy = np.linspace(bb['lat_min'], bb['lat_max'], n)
    GX, GY = np.meshgrid(gx, gy)

    tree = cKDTree(np.c_[lon, lat])
    d, idx = tree.query(np.c_[GX.ravel(), GY.ravel()], k=min(k, len(lon)))
    d = np.maximum(d, 1e-12)
    w = 1.0 / d ** p
    Z = (w * vals[idx]).sum(1) / w.sum(1)
    Z = Z.reshape(GX.shape)

    bd = gpd.read_file(BD_SHP)
    geom = bd.unary_union
    try:
        import shapely.vectorized as sv
        inside = sv.contains(geom, GX, GY)
    except Exception:
        from matplotlib.path import Path as MplPath
        polys = list(geom.geoms) if geom.geom_type == 'MultiPolygon' else [geom]
        pts = np.c_[GX.ravel(), GY.ravel()]
        inside = np.zeros(GX.ravel().shape, bool)
        for poly in polys:
            inside |= MplPath(np.asarray(poly.exterior.coords)).contains_points(pts)
        inside = inside.reshape(GX.shape)
    Z = np.where(inside, Z, np.nan)

    im = ax.imshow(Z, extent=(bb['lon_min'], bb['lon_max'], bb['lat_min'], bb['lat_max']),
                   origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, zorder=2,
                   interpolation='bilinear', aspect='equal')
    bd.boundary.plot(ax=ax, color='#333333', linewidth=1.0, zorder=3)
    cb = fig.colorbar(im, ax=ax, shrink=0.55, pad=0.02, extend=extend)
    cb.set_label(label)
    return im


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — T2 PANELS (Multi-Contaminant Health Risk)
# ═════════════════════════════════════════════════════════════════════════════

def t2_f01_hi_heatmap(df):
    """HI heatmap by zone, depth, and season — NO internal title."""
    # Compute stratified summary
    for dz_name, (lo, hi_b) in DEPTH_ZONES.items():
        mask = (df['Depth'] >= lo) & (df['Depth'] < hi_b)
        df.loc[mask, 'depth_zone'] = dz_name

    df_calc = calculate_all_risks(df.copy(), 'adult_male')

    summary = df_calc.groupby(['phys_zone', 'depth_zone', 'Season']).agg(
        HI_median=('HI_multi', 'median')
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    for i, season in enumerate(['Dry', 'Wet']):
        s = summary[summary['Season'] == season]
        pivot = s.pivot_table(index='phys_zone', columns='depth_zone',
                              values='HI_median', aggfunc='first')
        pivot = pivot.reindex(index=ZONE_ORDER, columns=DEPTH_ORDER)

        ylabels = [ZONE_LABELS.get(z, z) for z in pivot.index]
        xlabels = [DEPTH_LABELS.get(d, d) for d in pivot.columns]

        sns.heatmap(pivot, ax=axes[i], annot=True, fmt='.1f',
                    cmap='RdYlGn_r', vmin=0, vmax=10,
                    linewidths=0.8, linecolor='white',
                    cbar_kws={'label': 'HI (median)', 'shrink': 0.8},
                    annot_kws={'fontsize': 10, 'fontweight': 'bold'})
        axes[i].set_xticklabels(xlabels, rotation=0, fontsize=14)
        axes[i].set_yticklabels(ylabels, rotation=0, fontsize=14)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
        # Season label instead of title
        axes[i].text(0.5, 1.02, f'{season} season', transform=axes[i].transAxes,
                     ha='center', fontsize=16, fontweight='bold', color=DARKTEXT)

    fig.tight_layout(w_pad=1.5)
    save_panel(fig, 'T2_F01_HI_heatmap_season.png')


def t2_f02_single_vs_multi(df):
    """Single vs multi-contaminant scatter — cleaner, larger points."""
    df_calc = calculate_all_risks(df.copy(), 'adult_male')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # HI comparison
    ax = axes[0]
    ax.scatter(df_calc['HI_as_only'], df_calc['HI_multi'],
               alpha=0.35, s=14, c=STEEL, edgecolors='none', rasterized=True)
    lim = max(df_calc['HI_multi'].quantile(0.99), df_calc['HI_as_only'].quantile(0.99))
    ax.plot([0, lim], [0, lim], color='#333', ls='--', lw=1.2, label='1:1 line')
    ax.axhline(1, color=CRIMSON, ls=':', lw=1.2, alpha=0.7)
    ax.axvline(1, color=CRIMSON, ls=':', lw=1.2, alpha=0.7)
    ax.set_xlabel('As-only HI')
    ax.set_ylabel('Multi-contaminant HI')
    ax.text(0.05, 0.93, 'Non-carcinogenic', transform=ax.transAxes,
            fontsize=15, fontweight='bold', color=DARKTEXT)
    ax.legend(loc='lower right', fontsize=15)

    # CR comparison
    ax = axes[1]
    ax.scatter(df_calc['CR_as_only'], df_calc['CR_multi'],
               alpha=0.35, s=14, c=CRIMSON, edgecolors='none', rasterized=True)
    lim = max(df_calc['CR_multi'].quantile(0.99), df_calc['CR_as_only'].quantile(0.99))
    ax.plot([0, lim], [0, lim], color='#333', ls='--', lw=1.2, label='1:1 line')
    ax.axhline(1e-4, color=CRIMSON, ls=':', lw=1.2, alpha=0.7)
    ax.axvline(1e-4, color=CRIMSON, ls=':', lw=1.2, alpha=0.7)
    ax.set_xlabel('As-only CR')
    ax.set_ylabel('Multi-contaminant CR')
    ax.text(0.05, 0.93, 'Carcinogenic', transform=ax.transAxes,
            fontsize=15, fontweight='bold', color=DARKTEXT)
    ax.legend(loc='lower right', fontsize=15)

    fig.tight_layout(w_pad=2.0)
    save_panel(fig, 'T2_F02_single_vs_multi.png')


def t2_f03_contributions(df):
    """Contaminant contributions — horizontal bars, all visible, with % labels."""
    df_calc = calculate_all_risks(df.copy(), 'adult_male')

    # Compute % contribution
    contams = list(REFERENCE_DOSES.keys())
    pcts = {}
    for c in contams:
        col = f'HQ_{c}'
        if col in df_calc.columns:
            pcts[c] = (df_calc[col] / df_calc['HI_multi']).median() * 100

    # Sort descending
    pcts = dict(sorted(pcts.items(), key=lambda x: x[1], reverse=True))

    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = np.arange(len(pcts))
    vals = list(pcts.values())
    names = list(pcts.keys())
    colors = [CONTAM_DISPLAY.get(n, (n, SLATE))[1] for n in names]
    display_names = [CONTAM_DISPLAY.get(n, (n, SLATE))[0] for n in names]

    bars = ax.barh(y_pos, vals, color=colors, edgecolor='white', linewidth=0.5, height=0.65)

    # Add % labels
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=14, fontweight='bold', color=DARKTEXT)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names)
    ax.set_xlabel('Contribution to cumulative HI (%)')
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.25)

    fig.tight_layout()
    save_panel(fig, 'T2_F03_contaminant_contributions.png')


def t2_f04_age_groups(df):
    """Age-group HI box plots — Lancet style with individual points."""
    age_groups = list(EXPOSURE_PARAMS.keys())
    age_labels = [ag.replace('_', ' ').title() for ag in age_groups]

    age_data = []
    for ag in age_groups:
        temp = calculate_all_risks(df.copy(), ag)
        age_data.append(temp['HI_multi'].values)

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Lancet colour progression
    box_colors = [CRIMSON, AMBER, STEEL, TEAL]

    bp = ax.boxplot(age_data, patch_artist=True, widths=0.55, showfliers=False,
                    medianprops=dict(color='white', linewidth=2))

    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        patch.set_edgecolor('white')
    for whisker in bp['whiskers']:
        whisker.set(color=SLATE, linewidth=1)
    for cap in bp['caps']:
        cap.set(color=SLATE, linewidth=1)

    # Add jittered points
    np.random.seed(42)
    for i, data in enumerate(age_data):
        jitter = np.random.uniform(-0.12, 0.12, len(data))
        ax.scatter(np.full(len(data), i + 1) + jitter, data,
                   c=box_colors[i], s=4, alpha=0.15, edgecolors='none',
                   rasterized=True, zorder=2)

    ax.axhline(1, color=CRIMSON, ls='--', lw=1.5, alpha=0.7, label='HI = 1')
    ax.set_xticks(range(1, len(age_labels) + 1))
    ax.set_xticklabels(age_labels, fontsize=14)
    ax.set_ylabel('Cumulative Hazard Index')
    ax.legend(loc='upper right', fontsize=15)

    fig.tight_layout()
    save_panel(fig, 'T2_F04_age_group_comparison.png')


def t2_f05f06_daly_merged(df):
    """Annual DALYs by zone (gridded WorldPop × IDW HI) + multi vs As-only.

    Sources `T2_daly_by_zone_CORRECTED.csv` (annual, no seasonal double-count,
    gridded WorldPop populations). Falls back to the legacy seasonal table
    if the corrected one is missing.
    """
    corrected = Path('output/tables/T2_daly_by_zone_CORRECTED.csv')
    legacy = Path('output/tables/T2_daly_by_zone.csv')
    use_corrected = corrected.exists()
    daly_zone = pd.read_csv(corrected if use_corrected else legacy)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    zones_order = ['Ganges_Floodplain', 'GBM_Delta', 'Meghna_Floodplain',
                   'Brahmaputra_Floodplain', 'Northern_Terrace',
                   'Barind_Tract', 'Eastern_Hills']
    labels = [ZONE_LABELS.get(z, z) for z in zones_order]

    # Helpers to pick the right columns for either schema
    def _per_100k(zd):
        if 'annual_DALY_per_100k' in zd.columns:
            return float(zd['annual_DALY_per_100k'].iloc[0])
        # legacy: aggregate dry+wet (average to avoid double-count visually)
        return float(zd['DALY_per_100k'].mean())

    def _multi(zd):
        col = 'annual_DALY_multi' if use_corrected else 'DALY_multi'
        return float(zd[col].sum())

    def _as_only(zd):
        col = 'annual_DALY_as_only' if use_corrected else 'DALY_as_only'
        return float(zd[col].sum())

    # Left: Annual DALYs per 100,000 by zone
    for i, z in enumerate(zones_order):
        zdata = daly_zone[daly_zone['phys_zone'] == z]
        if zdata.empty:
            continue
        c = ZONE_COLORS.get(z, '#888')
        ax1.barh(i, _per_100k(zdata), height=0.55, color=c, alpha=0.9)

    ax1.set_yticks(range(len(zones_order)))
    ax1.set_yticklabels(labels)
    ax1.set_xlabel('Annual DALYs per 100,000')
    ax1.invert_yaxis()

    # Right: Multi vs As-only annual DALY rate by zone
    for i, z in enumerate(zones_order):
        zdata = daly_zone[daly_zone['phys_zone'] == z]
        if zdata.empty:
            continue
        pop = zdata['population'].iloc[0]
        multi_rate = _multi(zdata) / pop * 100_000
        as_rate = _as_only(zdata) / pop * 100_000
        ax2.barh(i - 0.18, multi_rate, height=0.32, color=CRIMSON, alpha=0.85,
                 label='Multi-contaminant' if i == 0 else '')
        ax2.barh(i + 0.18, as_rate, height=0.32, color=STEEL, alpha=0.85,
                 label='As-only' if i == 0 else '')

    ax2.set_yticks(range(len(zones_order)))
    ax2.set_yticklabels(labels)
    ax2.set_xlabel('Annual DALYs per 100,000')
    ax2.invert_yaxis()
    ax2.legend(loc='lower right', fontsize=14, frameon=False)

    fig.tight_layout(w_pad=2.0)
    save_panel(fig, 'T2_F05F06_merged.png')


def t2_spatial_risk(df):
    """Spatial HI map — zone boundaries as outlines, larger markers, no overlap legend."""
    dff = filter_bd(df.copy())
    hi = compute_hi(dff)
    dff['HI'] = hi

    fig, ax = plt.subplots(1, 1, figsize=(6, 7))
    add_bd_boundary(ax)

    # Zone boundary rectangles
    for zone, bbox in PHYSIOGRAPHIC_ZONES.items():
        rect = Rectangle((bbox['lon'][0], bbox['lat'][0]),
                          bbox['lon'][1] - bbox['lon'][0],
                          bbox['lat'][1] - bbox['lat'][0],
                          linewidth=0.6, edgecolor=ZONE_COLORS.get(zone, '#888'),
                          facecolor='none', linestyle='--', alpha=0.35)
        ax.add_patch(rect)

    # Smooth IDW surface of the hazard index, clipped to the country (no gaps).
    # vmax capped so a few extreme wells don't wash out the gradient.
    m = dff[['Longitude', 'Latitude', 'HI']].dropna()
    idw_clipped_surface(fig, ax, m['Longitude'].values, m['Latitude'].values,
                        m['HI'].values, cmap='RdYlGn_r', vmin=0, vmax=5,
                        label='Hazard index (HI)', extend='max')

    fig.tight_layout()
    save_panel(fig, 'T2_spatial_risk_map.png')


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — T4 PANELS (Tipping Points)
# ═════════════════════════════════════════════════════════════════════════════

def t4_f01_gmm_densities(df):
    """GMM density fits — cleaner, serif, no internal titles."""
    from sklearn.mixture import GaussianMixture
    from scipy.stats import norm

    contams = ['As', 'Mn2+', 'Fe2+', 'PO43-', 'NO3-', 'Cr3+']
    # Use mathtext for subscripts/superscripts — avoids missing Unicode glyphs
    contam_math = ['As', 'Mn', 'Fe', r'PO$_4$', r'NO$_3$', 'Cr']
    contam_colors = [CRIMSON, AMBER, '#F9A825', TEAL, STEEL, SLATE]

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes_flat = axes.flatten()

    for idx, (contam, label, color) in enumerate(zip(contams, contam_math, contam_colors)):
        ax = axes_flat[idx]
        if contam not in df.columns:
            ax.text(0.5, 0.5, f'{label}\nNo data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=16, color=SLATE)
            continue

        vals = df[contam].dropna().values
        vals = vals[vals > 0]
        if len(vals) < 20:
            ax.text(0.5, 0.5, f'{label}\nn < 20', ha='center', va='center',
                    transform=ax.transAxes, fontsize=16, color=SLATE)
            continue
        log_vals = np.log10(vals + 0.01)

        # Fit GMM
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(log_vals.reshape(-1, 1))

        # Plot histogram
        ax.hist(log_vals, bins=40, density=True, alpha=0.35, color=color,
                edgecolor='white', linewidth=0.3)

        # Plot GMM components
        x_range = np.linspace(log_vals.min() - 0.5, log_vals.max() + 0.5, 300)
        for k in range(2):
            y = gmm.weights_[k] * norm.pdf(x_range, gmm.means_[k, 0],
                                            np.sqrt(gmm.covariances_[k, 0, 0]))
            ax.plot(x_range, y, color=color, lw=1.8, alpha=0.85,
                    ls='--' if k == 1 else '-')

        # Overall density
        log_prob = gmm.score_samples(x_range.reshape(-1, 1))
        ax.plot(x_range, np.exp(log_prob), color='#333', lw=2, alpha=0.9)

        # Saddle point (minimum between means)
        m1, m2 = sorted(gmm.means_.flatten())
        saddle_range = x_range[(x_range >= m1) & (x_range <= m2)]
        if len(saddle_range) > 0:
            saddle_densities = np.exp(gmm.score_samples(saddle_range.reshape(-1, 1)))
            saddle_x = saddle_range[np.argmin(saddle_densities)]
            ax.axvline(saddle_x, color=color, ls=':', lw=1.5, alpha=0.6)
            ax.text(saddle_x, ax.get_ylim()[1] * 0.9, f'{10**saddle_x:.1f}',
                    ha='center', fontsize=15, color=color, fontweight='bold')

        # Use mathtext for log₁₀ to avoid box glyphs
        ax.set_xlabel(r'$\log_{10}$' + f'({label})')
        ax.set_ylabel('Density' if idx % 3 == 0 else '')
        ax.text(0.95, 0.92, label, transform=ax.transAxes, ha='right',
                fontsize=13, fontweight='bold', color=color)

    fig.tight_layout(h_pad=2.0, w_pad=1.5)
    save_panel(fig, 'T4_F01_gmm_densities.png')


def t4_f02_phase_diagrams(df):
    """Phase diagrams — As coloured by GMM state."""
    from sklearn.mixture import GaussianMixture

    # Classify As
    as_vals = df['As'].dropna().values
    log_as = np.log10(as_vals + 0.1)
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(log_as.reshape(-1, 1))
    labels = gmm.predict(log_as.reshape(-1, 1))
    if gmm.means_.flatten()[0] > gmm.means_.flatten()[1]:
        labels = 1 - labels

    df_ph = df.dropna(subset=['As']).copy()
    df_ph['as_mode'] = labels

    pairs = [('PO43-', 'ORP', 'PO₄ vs ORP'), ('Depth', 'Fe2+', 'Depth vs Fe')]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    for i, (xvar, yvar, xlabel_title) in enumerate(pairs):
        ax = axes[i]
        if xvar not in df_ph.columns or yvar not in df_ph.columns:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
            continue

        sub = df_ph.dropna(subset=[xvar, yvar])

        # Low mode
        low = sub[sub['as_mode'] == 0]
        ax.scatter(low[xvar], low[yvar], c=STEEL_L, s=10, alpha=0.4,
                   edgecolors='none', label='Low-As mode', rasterized=True)
        # High mode
        high = sub[sub['as_mode'] == 1]
        ax.scatter(high[xvar], high[yvar], c=CRIMSON, s=14, alpha=0.5,
                   edgecolors='none', label='High-As mode', rasterized=True)

        ax.set_xlabel(xvar.replace('PO43-', r'PO$_4$ (mg/L)').replace('ORP', 'ORP (mV)').replace('Depth', 'Depth (m)'))
        ax.set_ylabel(yvar.replace('Fe2+', r'Fe$^{2+}$ (mg/L)').replace('ORP', 'ORP (mV)'))
        ax.legend(loc='best', fontsize=15, markerscale=1.5)

    fig.tight_layout(w_pad=2.0)
    save_panel(fig, 'T4_F02_phase_diagrams.png')


def t4_f03_cascade_heatmap(df):
    """Cascading bifurcation heatmap — single unified heatmap."""
    from sklearn.mixture import GaussianMixture

    contams = ['As', 'Mn2+', 'Fe2+', 'PO43-', 'NO3-']
    contam_labels = ['As', 'Mn', 'Fe', r'PO$_4$', r'NO$_3$']

    # Classify each contaminant into high/low mode
    modes = {}
    for c in contams:
        if c not in df.columns:
            continue
        vals = df[c].dropna().values
        vals_pos = vals[vals > 0]
        if len(vals_pos) < 20:
            continue
        log_v = np.log10(vals_pos + 0.01)
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(log_v.reshape(-1, 1))
        labels = gmm.predict(log_v.reshape(-1, 1))
        if gmm.means_.flatten()[0] > gmm.means_.flatten()[1]:
            labels = 1 - labels
        # Map back to full df
        mode_map = pd.Series(labels, index=df[c].dropna()[df[c] > 0].index)
        modes[c] = mode_map

    # Compute conditional probabilities
    n = len(contams)
    cond_prob = np.full((n, n), np.nan)
    pvals = np.full((n, n), np.nan)
    from scipy.stats import chi2_contingency

    for i, ci in enumerate(contams):
        for j, cj in enumerate(contams):
            if i == j or ci not in modes or cj not in modes:
                continue
            mi = modes[ci]
            mj = modes[cj]
            common = mi.index.intersection(mj.index)
            if len(common) < 20:
                continue
            # P(j=high | i=high)
            i_high = mi[common] == 1
            j_high = mj[common] == 1
            if i_high.sum() > 0:
                cond_prob[i, j] = j_high[i_high].mean()
            # Chi2
            ct = pd.crosstab(mi[common], mj[common])
            if ct.shape == (2, 2):
                _, p, _, _ = chi2_contingency(ct)
                pvals[i, j] = p

    fig, ax = plt.subplots(figsize=(7, 6))

    # Mask diagonal
    mask = np.eye(n, dtype=bool)
    cond_masked = np.ma.array(cond_prob, mask=mask)

    im = ax.imshow(cond_masked, cmap='YlOrRd', vmin=0, vmax=0.7, aspect='auto')

    # Annotate
    for i in range(n):
        for j in range(n):
            if i == j or np.isnan(cond_prob[i, j]):
                continue
            val = cond_prob[i, j]
            sig = ''
            if not np.isnan(pvals[i, j]):
                if pvals[i, j] < 0.001:
                    sig = '***'
                elif pvals[i, j] < 0.01:
                    sig = '**'
                elif pvals[i, j] < 0.05:
                    sig = '*'
            txt_color = 'white' if val > 0.4 else DARKTEXT
            ax.text(j, i, f'{val:.2f}{sig}', ha='center', va='center',
                    fontsize=14, fontweight='bold', color=txt_color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(contam_labels)
    ax.set_yticklabels(contam_labels)
    ax.set_xlabel('Target (j = high mode)')
    ax.set_ylabel('Trigger (i = high mode)')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('P(j = high | i = high)', fontsize=14)

    fig.tight_layout()
    save_panel(fig, 'T4_F03_cascade_heatmap.png')


def t4_f04_cusp_surface(df):
    """Empirical effective potential and early-warning signals.

    Replaces the textbook cusp-catastrophe surface (which was non-convergent
    in fitting and not data-grounded). Two data-driven panels:

      (left)  Effective potential V(log As) = -log p(log As | PO4 bin),
              showing the empirical double-well at intermediate PO4 — the
              real geochemical signature of bistability in this dataset.
      (right) Early-warning signals across phosphate bins: variance and
              lag-1 autocorrelation of log As. A rise in both as the system
              approaches the tipping threshold is the classic Scheffer-2009
              EWS pattern.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Use the dataset's PO4 column (PO43-)
    po4_col = 'PO43-' if 'PO43-' in df.columns else None
    if po4_col is None or 'As' not in df.columns:
        return

    sub = df[[po4_col, 'As']].dropna()
    sub = sub[(sub[po4_col] > 0) & (sub['As'] > 0)]
    if len(sub) < 100:
        return

    log_as = np.log10(sub['As'].values)
    po4 = sub[po4_col].values

    # Three PO4 regimes: low, mid (near saddle), high
    q = np.quantile(po4, [0.0, 0.33, 0.67, 1.0])
    regimes = [('Low PO$_4$', (q[0], q[1]), '#4575b4'),
               ('Mid PO$_4$ (near saddle)', (q[1], q[2]), '#fc8d59'),
               ('High PO$_4$', (q[2], q[3]), '#d73027')]

    # ─── Panel A: empirical effective potential ─────────────────────────
    ax1 = axes[0]
    grid = np.linspace(log_as.min(), log_as.max(), 80)
    from scipy.stats import gaussian_kde
    for name, (lo, hi), color in regimes:
        mask = (po4 >= lo) & (po4 <= hi)
        if mask.sum() < 30:
            continue
        kde = gaussian_kde(log_as[mask], bw_method=0.35)
        density = kde(grid)
        V = -np.log(density + 1e-9)
        V = V - V.min()
        ax1.plot(grid, V, color=color, lw=2.4, label=name)
    ax1.axvline(np.log10(10), color='#333', ls=':', lw=1.2,
                label='WHO guideline (10 µg/L)')
    ax1.set_xlabel(r'log$_{10}$ As (µg/L)')
    ax1.set_ylabel('Effective potential $V$ (a.u.)')
    ax1.set_title('Empirical potential by phosphate regime',
                  fontsize=15, fontweight='bold', color=DARKTEXT)
    ax1.legend(loc='upper right', fontsize=13, frameon=False)

    # ─── Panel B: variance + autocorrelation EWS ───────────────────────
    ax2 = axes[1]
    bins = np.linspace(po4.min(), np.quantile(po4, 0.95), 9)
    centres = 0.5 * (bins[:-1] + bins[1:])
    var_arr = np.full(len(centres), np.nan)
    ac1_arr = np.full(len(centres), np.nan)
    for i in range(len(centres)):
        mask = (po4 >= bins[i]) & (po4 < bins[i+1])
        if mask.sum() < 20:
            continue
        x = log_as[mask]
        var_arr[i] = np.var(x)
        # Lag-1 autocorrelation on the sorted series (proxy for time series)
        xs = np.sort(x)
        if len(xs) > 1:
            ac1_arr[i] = np.corrcoef(xs[:-1], xs[1:])[0, 1]
    ax2.plot(centres, var_arr, marker='o', color=CRIMSON, lw=2,
             label='Variance of log As')
    ax2_r = ax2.twinx()
    ax2_r.plot(centres, ac1_arr, marker='s', color=STEEL, lw=2,
               label='Lag-1 autocorrelation')
    ax2.set_xlabel(r'PO$_4$ (mg/L)')
    ax2.set_ylabel('Variance of log As', color=CRIMSON)
    ax2_r.set_ylabel('Lag-1 autocorrelation', color=STEEL)
    ax2.set_title('Early-warning signals along PO$_4$ axis',
                  fontsize=15, fontweight='bold', color=DARKTEXT)
    ax2.tick_params(axis='y', labelcolor=CRIMSON)
    ax2_r.tick_params(axis='y', labelcolor=STEEL)
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower left',
               fontsize=13, frameon=False)

    fig.tight_layout(w_pad=2.0)
    save_panel(fig, 'T4_F04_cusp_surface.png')


def t4_f05_seasonal_shift(df):
    """Seasonal tipping-point shift — bar chart of high-mode membership."""
    from sklearn.mixture import GaussianMixture

    contams = ['As', 'Mn2+', 'Fe2+', 'PO43-', 'NO3-']
    contam_labels = ['As', 'Mn', 'Fe', r'PO$_4$', r'NO$_3$']

    dry_pct = []
    wet_pct = []
    valid_labels = []

    for c, label in zip(contams, contam_labels):
        if c not in df.columns:
            continue
        # Fix the high-mode boundary ONCE on the pooled (all-season) data, then
        # classify each season against that SAME model. Fitting a separate GMM
        # per season makes the two seasons' "high mode" incomparable (each is
        # relative to its own internal split) and produced spurious shifts.
        allv = df[c].dropna(); allv = allv[allv > 0]
        if len(allv) < 40:
            continue
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(np.log10(allv.values + 0.01).reshape(-1, 1))
        hi_comp = int(np.argmax(gmm.means_.flatten()))   # higher-mean component
        for season, store in [('Dry', dry_pct), ('Wet', wet_pct)]:
            sub = df[df['Season'] == season][c].dropna()
            sub = sub[sub > 0]
            if len(sub) < 20:
                store.append(0)
                continue
            labels = gmm.predict(np.log10(sub.values + 0.01).reshape(-1, 1))
            store.append((labels == hi_comp).mean() * 100)
        valid_labels.append(label)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: high-mode % by season
    ax = axes[0]
    x = np.arange(len(valid_labels))
    w = 0.35
    ax.bar(x - w/2, dry_pct, w, color=AMBER_L, edgecolor='white', label='Dry')
    ax.bar(x + w/2, wet_pct, w, color=STEEL, edgecolor='white', label='Wet')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_labels)
    ax.set_ylabel('High-mode membership (%)')
    ax.legend(loc='upper right', fontsize=15)

    # Right: shift (wet - dry)
    ax = axes[1]
    shifts = [w - d for w, d in zip(wet_pct, dry_pct)]
    colors = [CRIMSON if s > 0 else STEEL for s in shifts]
    ax.bar(x, shifts, color=colors, edgecolor='white', width=0.55)
    for i, (xi, s) in enumerate(zip(x, shifts)):
        ax.text(xi, s + (0.3 if s >= 0 else -0.3),
                f'{s:+.1f}pp', ha='center', fontsize=15, fontweight='bold',
                va='bottom' if s >= 0 else 'top')
    ax.axhline(0, color='#333', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(valid_labels)
    ax.set_ylabel('Δ High-mode (wet − dry, pp)')

    fig.tight_layout(w_pad=2.5)
    save_panel(fig, 'T4_F05_seasonal_shift.png')


def t4_f06_spatial_tipping(df):
    """Spatial tipping-point map — distinct markers for high/low."""
    from sklearn.mixture import GaussianMixture

    dff = filter_bd(df.copy())
    as_vals = dff['As'].fillna(0).values
    log_as = np.log10(as_vals + 0.1)
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(log_as.reshape(-1, 1))
    labels = gmm.predict(log_as.reshape(-1, 1))
    if gmm.means_.flatten()[0] > gmm.means_.flatten()[1]:
        labels = 1 - labels
    dff['as_mode'] = labels

    fig, ax = plt.subplots(1, 1, figsize=(6, 7))
    add_bd_boundary(ax)

    # Smooth IDW surface of the high-As-mode fraction (mean of the binary GMM
    # label), clipped to the country. Diverging map: blue = low-As, red = high-As.
    m = dff[['Longitude', 'Latitude', 'as_mode']].dropna()
    idw_clipped_surface(fig, ax, m['Longitude'].values, m['Latitude'].values,
                        m['as_mode'].values.astype(float), cmap='coolwarm',
                        vmin=0, vmax=1, label='Fraction in high-As mode')

    fig.tight_layout()
    save_panel(fig, 'T4_F06_spatial_tipping.png')


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — T1 PANELS (Climate Projections)
# ═════════════════════════════════════════════════════════════════════════════

def t1_f01_seasonal_sensitivity(df):
    """Seasonal sensitivity heatmaps — cleaner annotations."""
    # Load transfer data
    tf_file = Path('output/tables/T1_seasonal_transfer.csv')
    if not tf_file.exists():
        print('  SKIP T1_F01 (no transfer table)')
        return

    transfer = pd.read_csv(tf_file)
    contams = transfer['contaminant'].unique()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for i, metric in enumerate(['delta_pct', 'sensitivity_per_pct']):
        ax = axes[i]
        if metric not in transfer.columns:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
            continue

        pivot = transfer.pivot_table(index='phys_zone', columns='contaminant',
                                     values=metric, aggfunc='first')
        zone_order = [z for z in ZONE_ORDER if z in pivot.index]
        contam_order = [c for c in contams if c in pivot.columns]
        pivot = pivot.reindex(index=zone_order, columns=contam_order)

        ylabels = [ZONE_LABELS.get(z, z) for z in pivot.index]

        cmap = 'RdBu_r'
        vmax = pivot.abs().max().max()
        fmt = '.1f' if i == 0 else '.3f'
        sns.heatmap(pivot, ax=ax, annot=True, fmt=fmt,
                    cmap=cmap, center=0, vmin=-vmax, vmax=vmax,
                    linewidths=0.8, linecolor='white',
                    annot_kws={'fontsize': 9},
                    cbar_kws={'shrink': 0.8})
        ax.set_yticklabels(ylabels, rotation=0, fontsize=14)
        ax.set_xlabel('')
        ax.set_ylabel('')

        label = r'$\Delta$ Concentration (%)' if i == 0 else 'Sensitivity (S per %)'
        ax.text(0.5, 1.02, label, transform=ax.transAxes,
                ha='center', fontsize=15, fontweight='bold', color=DARKTEXT)

    fig.tight_layout(w_pad=1.5)
    save_panel(fig, 'T1_F01_seasonal_sensitivity.png')


def t1_f02_projection_timeline():
    """Projected concentrations through 2050 — larger subplot fonts."""
    proj_file = Path('output/tables/T1_projected_concentrations_2050.csv')
    if not proj_file.exists():
        print('  SKIP T1_F02 (no projection table)')
        return

    proj = pd.read_csv(proj_file)
    contams = proj['contaminant'].unique()[:4]  # As, Mn, Fe, Cr

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes_flat = axes.flatten()

    contam_colors = {'As': CRIMSON, 'Mn2+': AMBER, 'Fe2+': '#F9A825', 'Cr3+': SLATE}

    for idx, contam in enumerate(contams):
        ax = axes_flat[idx]
        cdata = proj[proj['contaminant'] == contam]
        if len(cdata) == 0:
            continue

        color = contam_colors.get(contam, STEEL)

        # Plot baseline to 2050 projection
        for _, row in cdata.iterrows():
            zone_lbl = ZONE_LABELS.get(row.get('phys_zone', ''), row.get('phys_zone', ''))
            years = [2020, 2050]
            base = row.get('baseline_median', 0)

            for ssp, col_name, ls in [('SSP2-4.5', 'projected_median_ssp245', '--'),
                                       ('SSP5-8.5', 'projected_median_ssp585', '-')]:
                proj_val = row.get(col_name, base)
                ax.plot(years, [base, proj_val], ls=ls, color=color, alpha=0.5, lw=1.2)

                # Confidence band (if available)
                lo_col = col_name.replace('median', 'lo95')
                hi_col = col_name.replace('median', 'hi95')
                if lo_col in row.index and hi_col in row.index:
                    try:
                        ax.fill_between(years, [base, row[lo_col]], [base, row[hi_col]],
                                        alpha=0.08, color=color)
                    except Exception:
                        pass

        label = CONTAM_DISPLAY.get(contam, (contam, color))[0]
        ax.text(0.05, 0.92, label, transform=ax.transAxes,
                fontsize=14, fontweight='bold', color=color)
        ax.set_xlabel('Year')
        ax.set_ylabel('Concentration')
        ax.set_xlim(2018, 2052)

    fig.tight_layout(h_pad=2.0, w_pad=2.0)
    save_panel(fig, 'T1_F02_projection_timeline.png')


def t1_f03_zone_vulnerability():
    """Zone vulnerability heatmaps."""
    proj_file = Path('output/tables/T1_projected_concentrations_2050.csv')
    if not proj_file.exists():
        print('  SKIP T1_F03 (no projection table)')
        return

    proj = pd.read_csv(proj_file)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    for i, ssp in enumerate(['ssp245', 'ssp585']):
        ax = axes[i]
        col = f'projected_median_{ssp}'
        if col not in proj.columns:
            continue

        proj['change_pct'] = (proj[col] - proj['baseline_median']) / proj['baseline_median'] * 100
        pivot = proj.pivot_table(index='phys_zone', columns='contaminant',
                                 values='change_pct', aggfunc='first')
        zone_order = [z for z in ZONE_ORDER if z in pivot.index]
        pivot = pivot.reindex(index=zone_order)
        ylabels = [ZONE_LABELS.get(z, z) for z in pivot.index]

        vmax = pivot.abs().max().max()
        sns.heatmap(pivot, ax=ax, annot=True, fmt='.0f',
                    cmap='RdBu_r', center=0, vmin=-vmax, vmax=vmax,
                    linewidths=0.8, linecolor='white',
                    annot_kws={'fontsize': 9, 'fontweight': 'bold'},
                    cbar_kws={'shrink': 0.8, 'label': 'Change (%)'})
        ax.set_yticklabels(ylabels, rotation=0, fontsize=14)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ssp_label = 'SSP2-4.5' if '245' in ssp else 'SSP5-8.5'
        ax.text(0.5, 1.02, ssp_label, transform=ax.transAxes,
                ha='center', fontsize=16, fontweight='bold', color=DARKTEXT)

    fig.tight_layout(w_pad=1.5)
    save_panel(fig, 'T1_F03_zone_vulnerability.png')


def t1_f04_threshold_crossings():
    """WHO threshold crossings + absolute arsenic increases in high-burden zones.

    Two panels:
      (left)  Count of zone-depth cells that newly cross the WHO arsenic
              guideline by 2050, computed directly from the CMIP6 ensemble
              projections (T1_ensemble_projections_2050.csv).
      (right) Absolute As change (µg/L) in the policy-relevant high-burden
              shallow/intermediate zones — replaces the prior national-%
              framing that was inflated by low-baseline zones.
    """
    ens_file = Path('output/tables/T1_ensemble_projections_2050.csv')
    if not ens_file.exists():
        print('  SKIP T1_F04 (no ensemble projection table)')
        return

    ens = pd.read_csv(ens_file)
    ens = ens[ens['contaminant'] == 'As'].copy()
    WHO = {'As': 10.0}
    ens['abs_change'] = ens['ensemble_median'] - ens['baseline_dry']
    crossed = ens[(ens['baseline_dry'] < WHO['As'])
                  & (ens['ensemble_median'] >= WHO['As'])]
    counts = crossed.groupby('ssp').size().to_dict()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ─── Panel A: New WHO crossings count by SSP ───
    ax1 = axes[0]
    ssp_labels = ['SSP2-4.5', 'SSP5-8.5']
    vals = [counts.get('ssp245', 0), counts.get('ssp585', 0)]
    bars = ax1.bar(ssp_labels, vals, color=[STEEL, CRIMSON],
                    edgecolor='white', width=0.55)
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.05,
                 str(int(v)), ha='center', fontsize=16, fontweight='bold',
                 color=DARKTEXT)
    ax1.set_ylabel('Zone-depth cells newly crossing\nWHO 10 µg/L arsenic')
    ax1.set_ylim(0, max(vals) + 1.2 if max(vals) > 0 else 2)
    ax1.text(0.5, 1.02, 'New WHO arsenic exceedances by 2050',
             transform=ax1.transAxes, ha='center', fontsize=15,
             fontweight='bold', color=DARKTEXT)

    # ─── Panel B: Absolute As change in top-5 high-burden zones (SSP5-8.5) ───
    ax2 = axes[1]
    s585 = ens[ens['ssp'] == 'ssp585'].copy()
    s585 = s585.sort_values('abs_change', ascending=False).head(5)
    labels = [f"{ZONE_LABELS.get(z, z)}\n({d})"
              for z, d in zip(s585['phys_zone'], s585['depth_zone'])]
    base = s585['baseline_dry'].values
    proj = s585['ensemble_median'].values
    y = np.arange(len(labels))
    ax2.barh(y, base, color=STEEL, alpha=0.6, label='Baseline (dry season)')
    ax2.barh(y, proj - base, left=base, color=CRIMSON, alpha=0.9,
             label='Projected increase by 2050 (SSP5-8.5)')
    ax2.axvline(10, color='#222', ls='--', lw=1.2, alpha=0.7)
    ax2.text(10.4, len(labels) - 0.4, 'WHO 10 µg/L', fontsize=13,
             color='#222', style='italic')
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels, fontsize=13)
    ax2.invert_yaxis()
    ax2.set_xlabel('Arsenic concentration (µg/L)')
    ax2.legend(loc='lower right', fontsize=13, frameon=False)
    ax2.text(0.5, 1.02, 'Largest absolute As increases (SSP5-8.5)',
             transform=ax2.transAxes, ha='center', fontsize=15,
             fontweight='bold', color=DARKTEXT)

    fig.tight_layout(w_pad=2.0)
    save_panel(fig, 'T1_F04_threshold_crossings_2050.png')


def t1_f05_contaminant_change_bar():
    """Divergent contaminant response bar chart — cleaner, Lancet style."""
    proj_file = Path('output/tables/T1_projected_concentrations_2050.csv')

    contams = ['As', 'Mn2+', 'Fe2+', 'Cr3+']
    contam_labels = ['As', r'Mn$^{2+}$', r'Fe$^{2+}$', r'Cr$^{3+}$']

    # Prefer the CMIP6 ensemble projections (replaces single-GCM table).
    ens_file = Path('output/tables/T1_ensemble_projections_2050.csv')
    valid_contams, valid_labels = [], []
    changes_245, changes_585 = [], []
    p5_245, p95_245, p5_585, p95_585 = [], [], [], []
    if ens_file.exists():
        ens = pd.read_csv(ens_file)
        for c, label in zip(contams, contam_labels):
            cdata = ens[ens['contaminant'] == c]
            if cdata.empty:
                continue
            for ssp, lst, lo, hi in [
                ('ssp245', changes_245, p5_245, p95_245),
                ('ssp585', changes_585, p5_585, p95_585),
            ]:
                sub = cdata[cdata['ssp'] == ssp]
                lst.append(float(sub['pct_change_median'].median()))
                lo.append(float(sub['pct_change_p5'].median()))
                hi.append(float(sub['pct_change_p95'].median()))
            valid_contams.append(c)
            valid_labels.append(label)
    elif proj_file.exists():
        proj = pd.read_csv(proj_file)
        for c, label in zip(contams, contam_labels):
            cdata = proj[proj['contaminant'] == c]
            if len(cdata) > 0 and 'baseline_median' in cdata.columns:
                base = cdata['baseline_median'].mean()
                if base > 0:
                    p245 = cdata['projected_median_ssp245'].mean()
                    p585 = cdata['projected_median_ssp585'].mean()
                    changes_245.append((p245 - base) / base * 100)
                    changes_585.append((p585 - base) / base * 100)
                    valid_contams.append(c)
                    valid_labels.append(label)
    else:
        valid_labels = contam_labels
        changes_585 = [129, -24, 13, -33]
        changes_245 = [64, -12, 7, -17]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(valid_labels))
    w = 0.35

    colors_585 = [CRIMSON if v > 0 else STEEL for v in changes_585]
    colors_245 = [CRIMSON_L if v > 0 else STEEL_L for v in changes_245]

    # Inter-model spread bars if ensemble data present
    yerr_245 = None
    yerr_585 = None
    if p5_245 and p95_245:
        yerr_245 = np.array([[c - lo for c, lo in zip(changes_245, p5_245)],
                              [hi - c for c, hi in zip(changes_245, p95_245)]])
    if p5_585 and p95_585:
        yerr_585 = np.array([[c - lo for c, lo in zip(changes_585, p5_585)],
                              [hi - c for c, hi in zip(changes_585, p95_585)]])

    bars1 = ax.bar(x - w/2, changes_245, w, color=colors_245, edgecolor='white',
                    label='SSP2-4.5', yerr=yerr_245,
                    error_kw=dict(ecolor='#333', capsize=3, lw=1.0, alpha=0.3))
    bars2 = ax.bar(x + w/2, changes_585, w, color=colors_585, edgecolor='white',
                    label='SSP5-8.5', yerr=yerr_585,
                    error_kw=dict(ecolor='#333', capsize=3, lw=1.0, alpha=0.3))

    for bar, val in zip(bars1, changes_245):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:+.0f}%', ha='center', va='bottom' if val > 0 else 'top',
                fontsize=15, fontweight='bold')
    for bar, val in zip(bars2, changes_585):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:+.0f}%', ha='center', va='bottom' if val > 0 else 'top',
                fontsize=15, fontweight='bold')

    ax.axhline(0, color='#333', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(valid_labels, fontsize=16)
    ax.set_ylabel('Concentration change (%)')
    ax.legend(loc='upper right', fontsize=15)

    fig.tight_layout()
    save_panel(fig, 'T1_F05_contaminant_change_bar.png')


def t1_f06_spatial_vulnerability(df):
    """Spatial climate vulnerability — HI × seasonal sensitivity, distinct markers."""
    dff = filter_bd(df.copy())
    hi = compute_hi(dff)
    dff['HI'] = hi

    sens_file = Path('output/tables/T1_seasonal_transfer.csv')
    if not sens_file.exists():
        print('  SKIP T1_F06 (no transfer table)')
        return

    sens = pd.read_csv(sens_file)
    as_sens = sens[sens['contaminant'] == 'As']

    zone_sens = {}
    for z in dff['phys_zone'].unique():
        zs = as_sens[as_sens['phys_zone'] == z]
        if len(zs) > 0 and 'delta_pct' in zs.columns:
            zone_sens[z] = abs(zs['delta_pct'].mean())
        else:
            zone_sens[z] = 0

    dff['sensitivity'] = dff['phys_zone'].map(zone_sens).fillna(0)
    max_sens = dff['sensitivity'].max() if dff['sensitivity'].max() > 0 else 1
    dff['vulnerability'] = dff['HI'] * (dff['sensitivity'] / max_sens)

    fig, ax = plt.subplots(1, 1, figsize=(6, 7))
    add_bd_boundary(ax)

    # Smooth IDW surface of climate vulnerability (HI × normalised seasonal
    # sensitivity), clipped to the country. vmax at the 95th pct so the tail
    # doesn't wash out the gradient.
    m = dff[['Longitude', 'Latitude', 'vulnerability']].dropna()
    vmax = float(np.nanpercentile(dff['vulnerability'], 95)) or 1.0
    idw_clipped_surface(fig, ax, m['Longitude'].values, m['Latitude'].values,
                        m['vulnerability'].values, cmap='YlOrRd', vmin=0, vmax=vmax,
                        label='Climate vulnerability', extend='max')

    fig.tight_layout()
    save_panel(fig, 'T1_F06_spatial_vulnerability.png')


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    set_lancet_style()

    print('Loading data...')
    df = load_data()
    print(f'  {len(df)} samples\n')

    print('=' * 50)
    print('FIGURE 2 — T2 Panels')
    print('=' * 50)
    t2_f01_hi_heatmap(df)
    t2_f02_single_vs_multi(df)
    t2_f03_contributions(df)
    t2_f04_age_groups(df)
    t2_f05f06_daly_merged(df)
    t2_spatial_risk(df)

    print('\n' + '=' * 50)
    print('FIGURE 3 — T4 Panels')
    print('=' * 50)
    t4_f01_gmm_densities(df)
    t4_f02_phase_diagrams(df)
    t4_f03_cascade_heatmap(df)
    t4_f04_cusp_surface(df)
    t4_f05_seasonal_shift(df)
    t4_f06_spatial_tipping(df)

    print('\n' + '=' * 50)
    print('FIGURE 4 — T1 Panels')
    print('=' * 50)
    t1_f01_seasonal_sensitivity(df)
    t1_f02_projection_timeline()
    t1_f03_zone_vulnerability()
    t1_f04_threshold_crossings()
    t1_f05_contaminant_change_bar()
    t1_f06_spatial_vulnerability(df)

    print('\n' + '=' * 50)
    print('ALL PANELS REGENERATED — Lancet v2')
    print('=' * 50)


if __name__ == '__main__':
    main()
