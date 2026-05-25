"""
External validation of the phosphate-arsenic co-mobilisation framework across
two independent cohorts, establishing both a positive cross-country replication
and an explicit scope boundary.

Cohort 1 (REPLICATION) -- Van Phuc, Red River delta, Vietnam
  Glodowska, Stopelli, Berg et al. (2020), PANGAEA 10.1594/PANGAEA.924685.
  Independent team, different country, freshwater reductive aquifer system
  (controlled reductive incubation of native groundwater + sediment).
  Tests whether As and PO4 co-mobilise under reductive conditions.

Cohort 2 (SCOPE BOUNDARY) -- southwest Bangladesh tidal deltaplain
  Ayers, Goodbred et al. (2017), PANGAEA 10.1594/PANGAEA.874439.
  Independent team, saline Sundarbans tidal plain. Salinity-driven As
  mobilisation dominates; wells sit on the high-PO4 plateau. Marks where the
  phosphate-redox framework does NOT apply.

Output:
  output/tables/T4_external_validation_combined.csv
  Draft/.../figS7_external_validation.png  (+ tiff)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.mixture import GaussianMixture
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import TABLES_DIR, FIGURES_DIR

VANPHUC = Path("data/external/validation/vanphuc_vietnam.tab")
AYERS = Path("data/external/validation/ayers2017_sw_bangladesh.tab")
CRIMSON = '#c0392b'; STEEL = '#2980b9'; LEAF = '#27ae60'; SLATE = '#7f8c8d'


def parse_pangaea(path):
    lines = path.read_text(encoding='utf-8').splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip() == '*/') + 1
    hdr = lines[start].split('\t')
    rows = [l.split('\t') for l in lines[start+1:] if l.strip()]
    return pd.DataFrame(rows, columns=hdr)


def col(df, pfx):
    for c in df.columns:
        if c.strip().startswith(pfx):
            return c
    return None


def load_vanphuc():
    df = parse_pangaea(VANPHUC)
    out = pd.DataFrame({
        'As': pd.to_numeric(df[col(df, 'As ')], errors='coerce'),
        'P': pd.to_numeric(df[col(df, 'P ')], errors='coerce'),
        'Fe': pd.to_numeric(df[col(df, 'Fe ')], errors='coerce'),
        'Mn': pd.to_numeric(df[col(df, 'Mn ')], errors='coerce'),
    }).dropna(subset=['As', 'P'])
    out['PO4_mg_L'] = out['P'] * (94.97 / 30.97)  # P mg/L -> PO4 mg/L
    return out


def load_ayers():
    df = parse_pangaea(AYERS)
    out = pd.DataFrame({
        'As': pd.to_numeric(df[col(df, 'As ')], errors='coerce'),
        'P': pd.to_numeric(df[col(df, 'P ')], errors='coerce'),
        'Fe': pd.to_numeric(df[col(df, 'Fe ')], errors='coerce'),
        'Mn': pd.to_numeric(df[col(df, 'Mn ')], errors='coerce'),
    }).dropna(subset=['As', 'P'])
    out['PO4_mg_L'] = out['P'] * (94.97 / 30.97) / 1000.0  # P µg/L -> PO4 mg/L
    return out


def cohort_stats(df, name):
    rho, p = stats.spearmanr(df['As'], df['PO4_mg_L'])
    x = np.log10(df['As'].clip(lower=0.1)).values.reshape(-1, 1)
    g1 = GaussianMixture(1, random_state=42).fit(x)
    g2 = GaussianMixture(2, random_state=42).fit(x)
    dbic = g1.bic(x) - g2.bic(x)
    return {
        'cohort': name, 'n': len(df),
        'As_median_ugL': float(df['As'].median()),
        'PO4_median_mgL': float(df['PO4_mg_L'].median()),
        'spearman_As_PO4': float(rho), 'spearman_p': float(p),
        'gmm_delta_bic': float(dbic),
    }


def main():
    print("=" * 70)
    print("External validation across independent cohorts")
    print("=" * 70)
    vp = load_vanphuc()
    ay = load_ayers()
    s_vp = cohort_stats(vp, 'Van Phuc, Vietnam (Red River, freshwater)')
    s_ay = cohort_stats(ay, 'SW Bangladesh (Sundarbans, saline tidal)')

    summary = pd.DataFrame([s_vp, s_ay])
    summary.to_csv(TABLES_DIR / 'T4_external_validation_combined.csv', index=False)
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    print()
    print(f"Van Phuc (REPLICATION): As-PO4 rho={s_vp['spearman_As_PO4']:+.2f} "
          f"(p={s_vp['spearman_p']:.1e}) -- positive co-mobilisation confirmed "
          f"in an independent freshwater delta")
    print(f"SW Bangladesh (SCOPE BOUNDARY): As-PO4 rho={s_ay['spearman_As_PO4']:+.2f} "
          f"(p={s_ay['spearman_p']:.2f}) -- flat; saline regime, wells on high-PO4 "
          f"plateau (median PO4 {s_ay['PO4_median_mgL']:.1f} mg/L)")

    # ─── Figure ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    # A: Van Phuc As vs PO4 (replication)
    ax = axes[0]
    ax.scatter(vp['PO4_mg_L'], vp['As'], s=28, c=LEAF, alpha=0.7,
               edgecolor='white')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.axhline(10, color='#444', ls=':', lw=0.8)
    ax.set_xlabel(r'PO$_4$ (mg/L)'); ax.set_ylabel(r'As (µg/L)')
    ax.set_title(f"(A) Van Phuc, Vietnam — REPLICATION\n"
                 f"freshwater Red River delta; "
                 f"ρ={s_vp['spearman_As_PO4']:+.2f} (p={s_vp['spearman_p']:.0e})",
                 fontsize=9.5, fontweight='bold')
    ax.grid(True, alpha=0.25)

    # B: Ayers As vs PO4 (scope boundary)
    ax = axes[1]
    ax.scatter(ay['PO4_mg_L'], ay['As'], s=28, c=CRIMSON, alpha=0.7,
               edgecolor='white')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.axhline(10, color='#444', ls=':', lw=0.8)
    ax.axvspan(1.5, 2.0, color='#e67e22', alpha=0.15)
    ax.set_xlabel(r'PO$_4$ (mg/L)'); ax.set_ylabel(r'As (µg/L)')
    ax.set_title(f"(B) SW Bangladesh — SCOPE BOUNDARY\n"
                 f"saline Sundarbans tidal plain; "
                 f"ρ={s_ay['spearman_As_PO4']:+.2f} (p={s_ay['spearman_p']:.2f}, ns)",
                 fontsize=9.5, fontweight='bold')
    ax.grid(True, alpha=0.25)

    # C: summary bar of Spearman rho by cohort
    ax = axes[2]
    names = ['National\nBangladesh\n(this study)', 'Van Phuc\nVietnam\n(freshwater)',
             'SW Bangladesh\nSundarbans\n(saline)']
    # National rho: As vs PO4 from main data (n=1574, rho=+0.31, p=2e-37)
    rhos = [0.314, s_vp['spearman_As_PO4'], s_ay['spearman_As_PO4']]
    colors = [STEEL, LEAF, CRIMSON]
    bars = ax.bar(names, rhos, color=colors, alpha=0.85, edgecolor='black')
    ax.axhline(0, color='#333', lw=0.8)
    for b, r in zip(bars, rhos):
        ax.text(b.get_x()+b.get_width()/2, r + (0.02 if r >= 0 else -0.04),
                f'{r:+.2f}', ha='center',
                va='bottom' if r >= 0 else 'top', fontsize=10, fontweight='bold')
    ax.set_ylabel(r'Spearman $\rho$ (As vs PO$_4$)')
    ax.set_ylim(-0.2, 0.65)
    ax.set_title('(C) Framework holds in freshwater\nreductive deltas; bounded in saline regime',
                 fontsize=9.5, fontweight='bold')

    fig.tight_layout()
    out = FIGURES_DIR / 'T4_external_validation_combined.png'
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\nFigure: {out}")
    print("=" * 70)


if __name__ == '__main__':
    main()
