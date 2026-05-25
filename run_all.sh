#!/usr/bin/env bash
# =============================================================================
# Reproduce the full analysis for Hasan et al., "Cumulative multi-contaminant
# groundwater exposure in Bangladesh" (Sci. Total Environ., submitted).
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
# Each step writes CSV tables to output/tables/ and figures to output/figures/.
# Bracket prints make it easy to grep the log for the headline numbers.
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

# ── T3: copula joint exceedance ──────────────────────────────────────────────
run scripts/t3_copula/copula_analysis.py

# ── T4: tipping points, partial information, mechanism, external validation ──
run scripts/t4_tipping/tipping_points.py
run scripts/t4_tipping/partial_info_phosphate.py    # Fig S5, Table S4
run scripts/t4_tipping/surface_complexation_simulation.py  # Fig S6
run scripts/t4_tipping/external_validation_combined.py     # Fig S7

# ── T1: climate projections + GRACE/TELLUS validation ────────────────────────
run scripts/t1_climate/climate_transfer_ensemble.py
run scripts/t1_climate/grace_fo_analysis.py
run scripts/t1_climate/tellus_mascon_analysis.py
run scripts/t1_climate/grace_gridlevel_correlation.py     # Table S8
run scripts/t1_climate/grace_combined_figure.py            # Fig S4

# ── T5: Bayesian / Monte-Carlo uncertainty propagation ───────────────────────
run scripts/t5_bayesian/bayesian_propagation.py

# ── T6: counterfactual interventions ─────────────────────────────────────────
run scripts/t6_interventions/interventions_corrected.py

# ── Conceptual + graphical-abstract figures ──────────────────────────────────
run scripts/t4_tipping/figure1_conceptual.py        # Figure 1
run scripts/t4_tipping/graphical_abstract.py        # Graphical Abstract

echo
echo "============================================================"
echo " [run_all] DONE. Outputs in output/tables/ and output/figures/"
echo "============================================================"
