# D3 figure pipeline — Paper 2 (HAZADV-D-26-01282)

Replaces the matplotlib figure scripts for the six main figures, built 2026-09-02.

## Running it

    ./render.sh          # export data, render SVG, convert to PDF and PNG
                         # builds all 20 figures: 6 main + 13 supplementary + graphical abstract

Requires Node 22+ (`npm install` once) and Inkscape. Run from this directory.

## How it is split

**Python owns the data.** `export_data.py` (main figures) and `export_si.py`
(supplementary) read the analysis output tables and write one JSON file per figure
into `data/`. Every number a figure draws comes
from those files, so a figure cannot silently disagree with the analysis, and the
JSON is auditable on its own.

**D3 only draws.** `fig1.mjs` … `fig6.mjs` and `s1.mjs` … `s12.mjs` build
standalone SVG through jsdom; nothing is interactive. `svgkit.mjs` holds the shared
canvas, axes and colourbar helpers, `sikit.mjs` the 3×2 supplementary grid, and
`palette.mjs` the colours.

**Inkscape converts.** Not `rsvg-convert`: it emits uncompressed PDF content
streams and turned the contour map into an 8.6 MB file where Inkscape produces
456 kB from the identical SVG.

## Palette

Warm ramp and contour blue are taken verbatim from Paper 1's house style
(`Paper1/JHMRevision/figstyle.py`) so the two manuscripts read as one body of
work. Two-colour rule: **warm** for anything carrying risk, hazard, burden or
exceedance; **teal `#2A9D8F`** for everything else, replacing the green and blue
used previously. `#104281` is reserved for contour lines, the one cool mark that
stays legible over the warm ramp.

## Things that bit, and are guarded against

- **Panel width must be derived from the canvas width**, not chosen by hand. Two
  figures had their right column running off the page before this was fixed.
- **`0.565 * 100` is `56.49999999999999`**, which rounds to 56 and printed "44%
  missed" where the manuscript says 43%. Percentages are nudged by `1e-9` before
  rounding.
- **The national arsenic change has two definitions.** The manuscript quotes the
  median of per-cell percentage changes (+74%); an aggregate sum-ratio over the
  same table gives +52%. `export_data.py` uses the former, and says so.
- **Conditional probabilities come from `T4_cascade_conditional.csv`**, not
  recomputed on quartiles, so Figure 2c shows the 0.48 the text cites.
- **The BBS boundary carries ~38,000 vertices** and is drawn six times. It is
  simplified at 0.004 deg on export, well below one printed pixel at panel size.
- **IDW puts a bullseye around every well.** The field is Gaussian-blurred before
  contouring, and the colour scale is computed from the smoothed field so the
  bar describes what is actually drawn.
- **Contour stroke-width must be divided by the same factor the group is scaled
  by**, or the isolines land at twice their intended weight.
- **Contour labels need two guards, not one.** A minimum separation between
  labels, and a point-in-polygon test of the whole label box against the drawn
  coastline. Testing the interpolation mask is not sufficient: the outline drawn
  on the page is a simplified version of it, and labels still straddled the coast.
- **Anything drawn beside a Sankey hub lands on the ribbons**, because the hub
  sits inside the flow. Hub labels belong in the bottom margin.
- **A reference line outside the y-domain prints its label into the panel above.**
  If a threshold matters, put it inside the domain.
- **Check whether R² is negative before plotting it.** The IDW surface has none at
  the point level, and a bar chart of it would read as a broken axis (S9).
- **Check a table is not degenerate before plotting it.** The PHREEQC titration
  has non-zero sorbed mass at one of forty rows (S6).
- **Never hard-code the width of a box that contains text.** jsdom has no layout
  engine, so a box cannot measure its own text; a fixed width is correct only
  until the wording changes. Use `textWidth()` and `pill()` from `svgkit.mjs`,
  which size to the content. The same class of defect put figure columns
  off-canvas three times before panel widths were derived from `W2`.

## Figure inventory

| file | figure | panels |
|---|---|---|
| `fig1.mjs` | 1 | (a) screening performance (b) cost comparison |
| `fig2.mjs` | 2 | (a) densities (b) arsenic by depth (c) conditional co-occurrence (d) phosphate-conditioned arsenic |
| `fig3.mjs` | 3 | (a) median HI by zone (b) multi vs arsenic-only (c) annual burden (d) contributions |
| `fig4.mjs` | 4 | (a) paired sensitivity (b) crossings (c) national change (d) ΔP spread |
| `fig5.mjs` | 5 | (a) above WHO (b) hazard index (c) climate vulnerability — filled contour bands + isolines |
| `fig6.mjs` | 6 | (A) Sankey attribution (B) variance decomposition |
| `s1.mjs` | S1 | copula: scatter, exceedance, family, counts, τ matrix, tail |
| `s2.mjs` | S2 | Monte Carlo: fan, 2050 distribution, variance, zones, pathways, depth |
| `s3.mjs` | S3 | interventions: averted, equity, spread, CEAC, ICER frontier, S3 by zone |
| `s4.mjs` | S4 | GRACE: GRACEDADM series, TELLUS series, trend against arsenic |
| `s5.mjs` | S5 | partial information: AUC by scheme, block-bootstrap ΔAUC |
| `s6.mjs` | S6 | surface complexation: desorbed fraction, aqueous enrichment |
| `s7.mjs` | S7 | external validation across cohorts |
| `s8.mjs` | S8 | tornado sensitivity of the national burden |
| `s9.mjs` | S9 | IDW cross-validation: zone recovery, skill by scheme |
| `s10.mjs` | S10 | between-campaign mode membership, paired wells |
| `s11.mjs` | S11 | age–sex stratified hazard index |
| `s12.mjs` | S12 | projected change by zone and contaminant |
