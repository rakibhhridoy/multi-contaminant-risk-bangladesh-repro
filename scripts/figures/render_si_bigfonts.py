"""
Re-render the T3/T5/T6 supplementary panels with enlarged, legible fonts —
STOTEN editor item 3, extended to the Supplementary figures (S1/S2/S3).

Strategy: the three tier scripts (copula_analysis / bayesian_propagation /
interventions) compute AND plot in one seeded pass (np.random.seed=42). Their
COMPUTED OUTPUTS are byte-identical run-to-run (verified: every T3/T5/T6 CSV
diffs clean). So we may freely re-run them for nicer figures without changing a
single number.

This wrapper does NOT edit the tier scripts. It monkeypatches matplotlib BEFORE
running each one, to (a) raise all default font sizes, (b) enforce a floor on
any small hard-coded fontsize=... passed to titles/labels/legends/annotations,
and (c) shrink oversized figesizes a little so fonts read larger once the panel
is tiled into the 2x3 composite. Plotted data is untouched — layout/typography
only. Run from the project root:  python3 render_si_bigfonts.py
"""
import sys, runpy
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

ROOT = Path(__file__).parent

# ── tunables ────────────────────────────────────────────────────────────────
FIG_SCALE   = 1.0    # keep original (generous) figsize so tight_layout has room
FONT_FLOOR  = 16     # min pt for axis labels / titles / legend / text
TICK_SIZE   = 14     # tick label size

plt.rcParams.update({
    'font.size':        FONT_FLOOR,
    'axes.titlesize':   FONT_FLOOR + 2,
    'axes.labelsize':   FONT_FLOOR,
    'xtick.labelsize':  TICK_SIZE,
    'ytick.labelsize':  TICK_SIZE,
    'legend.fontsize':  TICK_SIZE,
    'figure.titlesize': FONT_FLOOR + 6,
    'axes.titleweight': 'bold',
})

# ── figsize shrink ────────────────────────────────────────────────────────────
_orig_subplots = plt.subplots
def _subplots(*a, **k):
    fs = k.get('figsize')
    if fs:
        k['figsize'] = (fs[0] * FIG_SCALE, fs[1] * FIG_SCALE)
    return _orig_subplots(*a, **k)
plt.subplots = _subplots

_orig_figure = plt.figure
def _figure(*a, **k):
    fs = k.get('figsize')
    if fs:
        k['figsize'] = (fs[0] * FIG_SCALE, fs[1] * FIG_SCALE)
    return _orig_figure(*a, **k)
plt.figure = _figure

# ── font-floor on explicit fontsize= kwargs ──────────────────────────────────
def _floor(k):
    fs = k.get('fontsize')
    if fs is not None and isinstance(fs, (int, float)) and fs < FONT_FLOOR:
        k['fontsize'] = FONT_FLOOR
    return k

def _wrap(cls, name):
    orig = getattr(cls, name)
    def wrapped(self, *a, **k):
        return orig(self, *a, **_floor(k))
    setattr(cls, name, wrapped)

for _m in ('set_title', 'set_xlabel', 'set_ylabel', 'text', 'annotate', 'legend'):
    _wrap(Axes, _m)
_wrap(Figure, 'suptitle')
_wrap(Figure, 'text')

# pyplot-level helpers that delegate to the current axes/figure
_orig_pl_suptitle = plt.suptitle
plt.suptitle = lambda *a, **k: _orig_pl_suptitle(*a, **_floor(k))
_orig_pl_title = plt.title
plt.title = lambda *a, **k: _orig_pl_title(*a, **_floor(k))

# ── run each tier script in-process (patches persist; seed inside each) ───────
SCRIPTS = [
    ROOT / 't3_copula_joint_risk/scripts/copula_analysis.py',
    ROOT / 't5_bayesian_uncertainty/scripts/bayesian_propagation.py',
    ROOT / 't6_counterfactual_interventions/scripts/interventions.py',
]

if __name__ == '__main__':
    for s in SCRIPTS:
        print(f'\n=== rendering {s.name} (big fonts) ===')
        runpy.run_path(str(s), run_name='__main__')
    print('\nAll T3/T5/T6 panels re-rendered with enlarged fonts.')
