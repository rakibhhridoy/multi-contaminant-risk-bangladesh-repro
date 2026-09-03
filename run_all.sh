#!/usr/bin/env bash
# =============================================================================
# Reproduce the full analysis for Hasan et al., "Cumulative multi-contaminant
# groundwater exposure in Bangladesh" (J. Hazard. Mater. Adv., HAZADV-D-26-01282).
#
# Run from the bundle root:
#   bash run_all.sh
#
# Order matches the manuscript pipeline and respects data dependencies:
#   - spatial_aggregation must run before any DALY script (writes the gridded
#     WorldPop x IDW(HI) cache that the DALY routine consumes).
#   - daly_estimation_corrected must run before sensitivity / synergy / IDW-CV
#     and before interventions_corrected (which imports it).
#
# Each step writes CSV tables to output/tables/. Bracket prints make it easy to
# grep the log for the headline numbers.
#
# FIGURES ARE NOT BUILT HERE. Since the JHMA revision all 20 figures (6 main,
# 13 supplementary, graphical abstract) are drawn by the D3 pipeline in
# figures_d3/, which needs Node 22+ and Inkscape. Run this script first, so the
# tables it consumes exist, then:
#
#     cd figures_d3 && npm install && ./render.sh
#
# See figures_d3/README.md and Section 4 of the bundle README.
# =============================================================================
set -eu

cd "$(dirname "$0")"
mkdir -p output/tables output/figures

run() {
    echo
    echo "============================================================"
    echo " [run_all] $1"
    echo "============================================================"
    python3 "$1"
}

# ── T2: gridded exposure surface + national DALY ─────────────────────────────
run scripts/t2_health/spatial_aggregation.py
run scripts/t2_health/daly_estimation_corrected.py

# ── T2: robustness analyses for the DALY estimate ────────────────────────────
run scripts/t2_health/daly_sensitivity_tornado.py   # Fig S8
run scripts/t2_health/daly_synergy_bound.py         # Table S7
run scripts/t2_health/idw_spatial_cv.py             # Fig S9
run scripts/t2_health/gbd_reconciliation.py         # Table S3
run scripts/t2_health/mn_weight_sensitivity.py      # Fig S13 (R2.1 weight sweep)

# ── T3: copula joint exceedance ──────────────────────────────────────────────
run scripts/t3_copula/copula_analysis.py

# ── T4: tipping points, partial information, mechanism, external validation ──
run scripts/t4_tipping/tipping_points.py
run scripts/t4_tipping/partial_info_phosphate.py    # Fig S5, Table S4
run scripts/t4_tipping/surface_complexation_simulation.py  # Fig S6
run scripts/t4_tipping/external_validation_combined.py     # Fig S7

# ── T1: climate projections + GRACE/TELLUS validation ────────────────────────
# The seasonal transfer coefficients MUST be built first: climate_transfer_ensemble
# consumes output/tables/T1_seasonal_transfer.csv. As of the 2026-09-01 JHMA
# revision these are WITHIN-WELL PAIRED differences over the 810 wells sampled in
# both campaigns, replacing the earlier unpaired campaign-median contrast.
run scripts/t1_climate/paired_seasonal_transfer.py
run scripts/t1_climate/climate_transfer_ensemble.py
run scripts/t1_climate/grace_fo_analysis.py
run scripts/t1_climate/tellus_mascon_analysis.py
run scripts/t1_climate/grace_gridlevel_correlation.py     # Table S8
run scripts/t1_climate/grace_combined_figure.py            # Fig S4

# ── T5: Bayesian / Monte-Carlo uncertainty propagation ───────────────────────
run scripts/t5_bayesian/bayesian_propagation.py

# ── T6: counterfactual interventions ─────────────────────────────────────────
run scripts/t6_interventions/interventions_corrected.py

# Figure 1 and the graphical abstract were previously built here by
# scripts/t4_tipping/{figure1_conceptual,graphical_abstract}.py. Both are
# superseded by figures_d3/ (fig1.mjs and ga.mjs) and are kept in the bundle
# only as a record of the earlier matplotlib pipeline. They are NOT run.

echo
echo "============================================================"
echo " [run_all] DONE. Tables in output/tables/."
echo " [run_all] To build the figures:  cd figures_d3 && ./render.sh"
echo "============================================================"
