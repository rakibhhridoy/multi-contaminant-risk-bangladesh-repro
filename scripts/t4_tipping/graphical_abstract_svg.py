"""
Graphical abstract (SVG, hand-editable) for the STOTEN submission.
=================================================================
Polished three-stage triptych:
  (1) Cheap proxy   - phosphate test strip + bistable double-well curve
  (2) Doubled burden- DALY bar chart (462k As-only vs 942k multi, Mn stacked)
  (3) Climate + fix - Bangladesh well-risk map + climate callout + intervention

Real geometry is computed here (boundary path from the BBS adm0 shapefile,
1,807 well dots coloured by arsenic class, the bistable curve), then written
into a clean, grouped, commented SVG so the file can be edited later by hand.

Master output : Draft/STOTENSubmission/Graphical_Abstract.svg
Raster (300+dpi for Elsevier): rasterised separately with rsvg-convert.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import MultiPolygon

ROOT = Path(__file__).resolve().parents[2]   # repo root (portable)
SHP = ROOT / 'data/gis/bgd_admbnda_adm0_bbs_20201113.shp'
CSV = ROOT / 'data/bangladesh_groundwater_arsenic_1807samples.csv'
OUT_SVG = ROOT / 'output' / 'figures' / 'Graphical_Abstract.svg'

# ---- palette ---------------------------------------------------------------
C_AS   = '#c0392b'   # arsenic / high risk
C_MN   = '#e67e22'   # manganese
C_PO4  = '#2980b9'   # phosphate / proxy
C_SAFE = '#27ae60'   # safe / intervention
INK    = '#1f2933'
GREY   = '#8a94a6'
PAPER  = '#ffffff'
PANEL  = '#f6f8fb'

# ---- canvas ----------------------------------------------------------------
W, H = 1328, 600

# ---- column geometry -------------------------------------------------------
COL_W = 390
COL_X = [40, 472, 904]            # left edge of each column
CX = [x + COL_W / 2 for x in COL_X]  # column centres
DIV_X = [451, 883]                # subtle vertical dividers


# ===========================================================================
#  GEOMETRY HELPERS
# ===========================================================================
def map_transform(bounds, box):
    """Equirectangular lon/lat -> SVG (x,y), aspect-corrected, y flipped."""
    lon0, lat0, lon1, lat1 = bounds
    bx, by, bw, bh = box
    coslat = np.cos(np.radians((lat0 + lat1) / 2))
    gw = (lon1 - lon0) * coslat
    gh = (lat1 - lat0)
    s = min(bw / gw, bh / gh)               # uniform scale, preserve shape
    ox = bx + (bw - gw * s) / 2             # centre inside box
    oy = by + (bh - gh * s) / 2

    def tf(lon, lat):
        x = ox + (lon - lon0) * coslat * s
        y = oy + (lat1 - lat) * s           # flip: north = up
        return x, y
    return tf


def boundary_paths(tf, simplify_tol=0.006, min_area=0.02):
    """Return list of SVG 'd' strings for the country outline polygons."""
    g = gpd.read_file(SHP).to_crs(4326)
    geom = g.geometry.union_all().simplify(simplify_tol)
    polys = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]
    polys = [p for p in polys if p.area >= min_area]
    ds = []
    for p in polys:
        xs, ys = p.exterior.coords.xy
        pts = [tf(x, y) for x, y in zip(xs, ys)]
        d = 'M' + ' L'.join(f'{x:.1f},{y:.1f}' for x, y in pts) + ' Z'
        ds.append(d)
    return ds


def well_dots(tf, bounds):
    """Return three lists of (x,y) for safe / moderate / high arsenic wells."""
    lon0, lat0, lon1, lat1 = bounds
    df = pd.read_csv(CSV)
    As = pd.to_numeric(df['As'], errors='coerce')
    lon = pd.to_numeric(df['Longitude'], errors='coerce')
    lat = pd.to_numeric(df['Latitude'], errors='coerce')
    m = (lon.between(lon0, lon1) & lat.between(lat0, lat1) & As.notna())
    df = df[m]; As = As[m]; lon = lon[m]; lat = lat[m]
    safe, mod, high = [], [], []
    for a, lo, la in zip(As, lon, lat):
        x, y = tf(lo, la)
        (safe if a <= 10 else mod if a <= 50 else high).append((x, y))
    return safe, mod, high


def idw_surface_b64(bounds, px_w=320, px_h=470, k=10, p=2, vmax=50.0):
    """Base64 PNG of an inverse-distance (k-NN, power p) arsenic surface over
    `bounds`, north-up, coloured green(safe)->red(high) with RdYlGn_r. Matches
    the IDW (k=10, p=2) interpolator used for the manuscript maps; the SVG #bd
    path clips it to the country, so out-of-country extrapolation is hidden."""
    import io, base64
    from scipy.spatial import cKDTree
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.image as mpimg
    lon0, lat0, lon1, lat1 = bounds
    df = pd.read_csv(CSV)
    As = pd.to_numeric(df['As'], errors='coerce')
    lon = pd.to_numeric(df['Longitude'], errors='coerce')
    lat = pd.to_numeric(df['Latitude'], errors='coerce')
    m = (lon.between(lon0, lon1) & lat.between(lat0, lat1) & As.notna() & (As >= 0))
    As, lon, lat = As[m].values, lon[m].values, lat[m].values
    gx = np.linspace(lon0, lon1, px_w)
    gy = np.linspace(lat1, lat0, px_h)          # row 0 = north (top)
    GX, GY = np.meshgrid(gx, gy)
    tree = cKDTree(np.c_[lon, lat])
    d, idx = tree.query(np.c_[GX.ravel(), GY.ravel()], k=min(k, len(As)))
    d = np.maximum(d, 1e-9)
    w = 1.0 / d ** p
    Z = ((w * As[idx]).sum(1) / w.sum(1)).reshape(GX.shape)
    rgba = plt.get_cmap('RdYlGn_r')(mcolors.Normalize(0, vmax, clip=True)(Z))
    buf = io.BytesIO()
    mpimg.imsave(buf, rgba, format='png')
    return base64.b64encode(buf.getvalue()).decode('ascii')


def bistable_path(box, n=240):
    """Double-well V(t)=t^4-1.7 t^2 mapped into box (bx,by,bw,bh)."""
    bx, by, bw, bh = box
    t = np.linspace(-1.7, 1.7, n)
    v = t**4 - 1.7 * t**2
    x = bx + (t - t.min()) / (t.max() - t.min()) * bw
    y = by + bh - (v - v.min()) / (v.max() - v.min()) * bh
    d = 'M' + ' L'.join(f'{xi:.1f},{yi:.1f}' for xi, yi in zip(x, y))
    # well + saddle positions (data t -> svg)
    def tx(tt): return bx + (tt - t.min()) / (t.max() - t.min()) * bw
    def ty(vv): return by + bh - (vv - v.min()) / (v.max() - v.min()) * bh
    tw = np.sqrt(1.7 / 2)
    low = (tx(-tw), ty((-tw)**4 - 1.7 * (-tw)**2))
    high = (tx(tw), ty((tw)**4 - 1.7 * (tw)**2))
    saddle = (tx(0), ty(0))
    return d, low, high, saddle


# ===========================================================================
#  SVG ASSEMBLY
# ===========================================================================
def esc(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def T(x, y, s, size, color=INK, weight='normal', anchor='middle',
      style='normal', spacing=None, family='Helvetica, Arial, sans-serif'):
    extra = f' letter-spacing="{spacing}"' if spacing else ''
    return (f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
            f'fill="{color}" font-weight="{weight}" font-style="{style}" '
            f'text-anchor="{anchor}"{extra}>{esc(s)}</text>')


def main():
    parts = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif">')
    parts.append(f'<rect width="{W}" height="{H}" fill="{PAPER}"/>')

    # ---- header ------------------------------------------------------------
    parts.append(T(W/2, 46,
        'Cumulative multi-contaminant groundwater exposure in Bangladesh',
        29, INK, 'bold'))
    parts.append(T(W/2, 78,
        'A low-cost phosphate proxy flags a doubled, climate-amplified health burden',
        17.5, GREY, style='italic'))
    parts.append(f'<line x1="40" y1="98" x2="{W-40}" y2="98" '
                 f'stroke="#e3e8f0" stroke-width="1.5"/>')

    # ---- column dividers (split to leave room for the flow arrow) ----------
    for dx in DIV_X:
        parts.append(f'<line x1="{dx}" y1="120" x2="{dx}" y2="302" '
                     f'stroke="#e8edf4" stroke-width="1.4"/>')
        parts.append(f'<line x1="{dx}" y1="338" x2="{dx}" y2="{H-30}" '
                     f'stroke="#e8edf4" stroke-width="1.4"/>')

    # ---- flow arrows between stages (shaft + proportional head) ------------
    for dx in DIV_X:
        parts.append(f'<line x1="{dx-18}" y1="320" x2="{dx+4}" y2="320" '
                     f'stroke="{INK}" stroke-width="3.2" stroke-linecap="round"/>')
        parts.append(f'<path d="M{dx+2} 311 L{dx+20} 320 L{dx+2} 329 Z" '
                     f'fill="{INK}"/>')

    # =======================================================================
    #  STAGE 1 - CHEAP PROXY
    # =======================================================================
    parts.append('<!-- ============ STAGE 1: cheap proxy ============ -->')
    parts.append(f'<circle cx="{COL_X[0]+18}" cy="135" r="13" fill="{C_PO4}"/>')
    parts.append(T(COL_X[0]+18, 140, '1', 15, '#fff', 'bold'))
    parts.append(T(COL_X[0]+40, 140, 'Cheap proxy', 17, C_PO4, 'bold', 'start'))

    # phosphate test strip icon ---------------------------------------------
    sx, sy = COL_X[0]+128, 168
    parts.append(f'<g id="test-strip">'
                 f'<rect x="{sx}" y="{sy}" width="22" height="74" rx="4" '
                 f'fill="#eaf2fb" stroke="{C_PO4}" stroke-width="2"/>')
    for i, c in enumerate(['#cfe3f6', '#7fb6e0', C_PO4, '#1c5e92']):
        parts.append(f'<rect x="{sx+3}" y="{sy+6+i*15}" width="16" height="12" '
                     f'rx="2" fill="{c}"/>')
    parts.append('</g>')
    parts.append(T(sx+11, sy+90, 'PO₄ strip', 11.5, C_PO4, 'bold'))

    # tube well icon (hand pump) --------------------------------------------
    wx, wy = COL_X[0]+240, 176
    parts.append(
        f'<g id="tubewell" stroke="{INK}" stroke-width="2.6" fill="none" '
        f'stroke-linecap="round" stroke-linejoin="round">'
        f'<line x1="{wx}" y1="{wy+10}" x2="{wx}" y2="{wy+56}"/>'        # body
        f'<line x1="{wx}" y1="{wy+16}" x2="{wx-24}" y2="{wy+6}"/>'      # handle lever
        f'<path d="M{wx} {wy+10} h15 a7 7 0 0 1 7 7 v9" />'            # spout
        f'</g>')
    parts.append(f'<circle cx="{wx-24}" cy="{wy+6}" r="3.2" fill="{INK}"/>')   # grip
    parts.append(f'<rect x="{wx-16}" y="{wy+56}" width="46" height="7" rx="2" '
                 f'fill="{GREY}"/>')                                          # base
    parts.append(T(wx+3, wy+82, 'tube well', 11, GREY))

    # bistable double-well curve --------------------------------------------
    cbox = (COL_X[0]+40, 300, COL_W-90, 120)
    d, low, high, saddle = bistable_path(cbox)
    parts.append(f'<g id="bistable">')
    parts.append(f'<path d="{d}" fill="none" stroke="{INK}" stroke-width="3" '
                 f'stroke-linejoin="round"/>')
    parts.append(f'<circle cx="{low[0]:.1f}" cy="{low[1]:.1f}" r="7" '
                 f'fill="{C_SAFE}" stroke="#fff" stroke-width="2"/>')
    parts.append(f'<circle cx="{high[0]:.1f}" cy="{high[1]:.1f}" r="7" '
                 f'fill="{C_AS}" stroke="#fff" stroke-width="2"/>')
    parts.append(f'<circle cx="{saddle[0]:.1f}" cy="{saddle[1]:.1f}" r="3.5" '
                 f'fill="none" stroke="{GREY}" stroke-width="2"/>')
    parts.append(T(low[0], low[1]+22, 'low-As', 10.5, C_SAFE, 'bold'))
    parts.append(T(high[0], high[1]+22, 'high-As', 10.5, C_AS, 'bold'))
    parts.append(T(saddle[0], saddle[1]-12, 'tipping point', 10, GREY,
                  style='italic'))
    parts.append('</g>')
    parts.append(T(CX[0], 460,
                  'PO₄ ranks wells by proximity', 12.5, INK))
    parts.append(T(CX[0], 478, 'to the arsenic tipping point', 12.5, INK))
    # cost pill
    parts.append(f'<rect x="{CX[0]-95}" y="498" width="190" height="30" rx="15" '
                 f'fill="#eaf2fb"/>')
    parts.append(T(CX[0], 518, '≈ 1/10 the cost of an As assay',
                  11.5, C_PO4, 'bold'))

    # =======================================================================
    #  STAGE 2 - DOUBLED BURDEN
    # =======================================================================
    parts.append('<!-- ============ STAGE 2: doubled burden ============ -->')
    parts.append(f'<circle cx="{COL_X[1]+18}" cy="135" r="13" fill="{C_AS}"/>')
    parts.append(T(COL_X[1]+18, 140, '2', 15, '#fff', 'bold'))
    parts.append(T(COL_X[1]+40, 140, 'Doubled burden', 17, C_AS, 'bold', 'start'))

    # bar chart -------------------------------------------------------------
    base_y = 430                      # baseline
    top_y = 200                       # plot top
    scale = (base_y - top_y) / 1100.0  # DALYs -> px (max ~1100k headroom)
    bw_bar = 78
    bx1 = CX[1] - 100                 # As-only bar centre area
    bx2 = CX[1] + 22                  # multi bar
    parts.append('<g id="dalybars">')
    parts.append(f'<line x1="{bx1-30}" y1="{base_y}" x2="{bx2+bw_bar+30}" '
                 f'y2="{base_y}" stroke="{GREY}" stroke-width="1.5"/>')
    # As-only
    h1 = 462 * scale
    parts.append(f'<rect x="{bx1}" y="{base_y-h1:.1f}" width="{bw_bar}" '
                 f'height="{h1:.1f}" rx="3" fill="{GREY}"/>')
    parts.append(T(bx1+bw_bar/2, base_y-h1-10, '462k', 13, GREY, 'bold'))
    parts.append(T(bx1+bw_bar/2, base_y+20, 'As-only', 12.5, INK))
    # multi: arsenic base + Mn stacked on top
    h2 = 942 * scale
    h2mn = 942 * 0.321 * scale
    parts.append(f'<rect x="{bx2}" y="{base_y-(h2-h2mn):.1f}" width="{bw_bar}" '
                 f'height="{(h2-h2mn):.1f}" fill="{C_AS}"/>')
    parts.append(f'<rect x="{bx2}" y="{base_y-h2:.1f}" width="{bw_bar}" '
                 f'height="{h2mn:.1f}" rx="3" fill="{C_MN}"/>')
    parts.append(T(bx2+bw_bar/2, base_y-h2-10, '942k', 15, C_AS, 'bold'))
    parts.append(T(bx2+bw_bar/2, base_y-(h2-h2mn)+h2mn/2+4, 'Mn 32%', 9.5,
                  '#fff', 'bold'))
    parts.append(T(bx2+bw_bar/2, base_y+20, 'Multi', 12.5, INK))
    parts.append(T(CX[1], top_y-18, 'annual DALYs', 11, GREY, style='italic'))
    parts.append('</g>')
    parts.append(T(CX[1], 470, '2.0× higher than arsenic-only',
                  13.5, INK, 'bold'))
    parts.append(f'<rect x="{CX[1]-130}" y="498" width="260" height="30" rx="15" '
                 f'fill="#fdeee6"/>')
    parts.append(T(CX[1], 518, 'manganese rivals As, yet unmonitored',
                  11.5, C_MN, 'bold'))

    # =======================================================================
    #  STAGE 3 - CLIMATE + CHEAP FIX
    # =======================================================================
    parts.append('<!-- ============ STAGE 3: climate + cheap fix ============ -->')
    parts.append(f'<circle cx="{COL_X[2]+18}" cy="135" r="13" fill="{C_SAFE}"/>')
    parts.append(T(COL_X[2]+18, 140, '3', 15, '#fff', 'bold'))
    parts.append(T(COL_X[2]+40, 140, 'Climate + cheap fix', 17, C_SAFE, 'bold',
                  'start'))

    # Bangladesh map ---------------------------------------------------------
    bounds_g = (88.009, 20.591, 92.68, 26.635)   # lon0,lat0,lon1,lat1
    mbox = (COL_X[2]+6, 160, 140, 215)           # bx,by,bw,bh
    tf = map_transform(bounds_g, mbox)
    paths = boundary_paths(tf)
    safe, mod, high = well_dots(tf, bounds_g)   # kept for the printed count only
    # projected sub-rectangle the country occupies inside mbox (matches tf)
    lon0, lat0, lon1, lat1 = bounds_g
    bx, by, bw, bh = mbox
    coslat = np.cos(np.radians((lat0 + lat1) / 2))
    gw = (lon1 - lon0) * coslat; gh = (lat1 - lat0)
    s = min(bw / gw, bh / gh)
    ox = bx + (bw - gw * s) / 2; oy = by + (bh - gh * s) / 2
    b64 = idw_surface_b64(bounds_g)
    parts.append('<g id="map">')
    parts.append('<clipPath id="bd">')          # clip the surface to land
    for d in paths:
        parts.append(f'<path d="{d}"/>')
    parts.append('</clipPath>')
    # IDW arsenic surface, clipped to the country, with a crisp outline on top
    parts.append(f'<image x="{ox:.1f}" y="{oy:.1f}" width="{gw*s:.1f}" '
                 f'height="{gh*s:.1f}" preserveAspectRatio="none" '
                 f'clip-path="url(#bd)" xlink:href="data:image/png;base64,{b64}"/>')
    for d in paths:
        parts.append(f'<path d="{d}" fill="none" stroke="{GREY}" '
                     f'stroke-width="1.2"/>')
    parts.append('</g>')   # /map
    # map legend: continuous green->red As gradient
    lx, ly = COL_X[2]+6, 392
    parts.append('<defs><linearGradient id="asgrad" x1="0" y1="0" x2="1" y2="0">'
                 f'<stop offset="0" stop-color="{C_SAFE}"/>'
                 f'<stop offset="0.5" stop-color="{C_MN}"/>'
                 f'<stop offset="1" stop-color="{C_AS}"/></linearGradient></defs>')
    parts.append(f'<rect x="{lx}" y="{ly-6}" width="118" height="9" rx="2" '
                 f'fill="url(#asgrad)" stroke="{GREY}" stroke-width="0.5"/>')
    parts.append(T(lx, ly+16, '≤10', 9, INK, anchor='start'))
    parts.append(T(lx+118, ly+16, '>50', 9, INK, anchor='end'))
    parts.append(T(lx+59, ly+16, 'As (µg/L), IDW', 8.5, GREY))

    # climate callout: thermometer + text, stacked in the right half ---------
    rcx = COL_X[2] + 270                      # right-half centre x (~1174)
    parts.append(f'<g id="thermo">'
                 f'<rect x="{rcx-5}" y="170" width="10" height="44" rx="5" '
                 f'fill="#fff" stroke="{C_AS}" stroke-width="2"/>'
                 f'<rect x="{rcx-2}" y="192" width="4" height="22" fill="{C_AS}"/>'
                 f'<circle cx="{rcx}" cy="220" r="9" fill="{C_AS}"/>'
                 f'<path d="M{rcx+13} 196 l7 -7 l7 7 M{rcx+20} 189 v22" '
                 f'stroke="{C_AS}" stroke-width="2.4" fill="none" '
                 f'stroke-linecap="round" stroke-linejoin="round"/>'
                 f'</g>')
    parts.append(T(rcx, 251, '3 zones cross the', 13, C_AS, 'bold'))
    parts.append(T(rcx, 270, 'WHO As limit by 2050', 13, C_AS))
    parts.append(T(rcx, 288, '(CMIP6, SSP5-8.5)', 9.5, GREY))

    # intervention box (full column width) ----------------------------------
    ibx, iby, ibw, ibh = COL_X[2]+2, 470, COL_W-4, 96
    parts.append(f'<rect x="{ibx}" y="{iby}" width="{ibw}" height="{ibh}" rx="12" '
                 f'fill="#eafaf1" stroke="{C_SAFE}" stroke-width="2"/>')
    # check icon
    parts.append(f'<circle cx="{ibx+24}" cy="{iby+26}" r="11" fill="{C_SAFE}"/>')
    parts.append(f'<path d="M{ibx+18} {iby+26} l4 4 l8 -9" stroke="#fff" '
                 f'stroke-width="2.6" fill="none" stroke-linecap="round" '
                 f'stroke-linejoin="round"/>')
    parts.append(T(ibx+42, iby+30, 'Seasonal well-switching', 14, C_SAFE,
                  'bold', 'start'))
    parts.append(T(ibx+ibw/2, iby+56, 'averts 446k DALYs/yr at $61 each', 12.5,
                  INK))
    parts.append(T(ibx+ibw/2, iby+78, '14× cheaper than well-deepening',
                  11.5, GREY))

    parts.append('</svg>')

    OUT_SVG.write_text('\n'.join(parts), encoding='utf-8')
    print(f'wrote {OUT_SVG}  ({OUT_SVG.stat().st_size/1024:.1f} KB)')
    print(f'wells plotted: safe={len(safe)} mod={len(mod)} high={len(high)}')

    # ---- rasterise: SVG is the editable master; Elsevier needs PNG/TIFF -----
    import subprocess, shutil
    pkg = ROOT / 'output' / 'figures'
    pkg.mkdir(parents=True, exist_ok=True)
    png = pkg / 'Graphical_Abstract.png'
    rc = shutil.which('rsvg-convert')
    if not rc:
        print('rsvg-convert not found - SVG written only'); return
    subprocess.run([rc, '-w', '2656', str(OUT_SVG), '-o', str(png)], check=True)
    try:
        from PIL import Image
        im = Image.open(png).convert('RGB')
        im.save(pkg / 'graphical_abstract.tiff',
                compression='tiff_lzw', dpi=(300, 300))
        print(f'rendered PNG + TIFF at {im.size[0]}x{im.size[1]} px')
    except Exception as e:
        print('tiff skipped:', e)


if __name__ == '__main__':
    main()
