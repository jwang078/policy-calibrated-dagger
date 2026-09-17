#!/bin/bash
# Regenerates every figure of the Policy-Calibrated DAgger paper from the
# snapshots in paper_plots/data (no dataset cache, checkpoints or sim needed;
# see README.md for what each folder holds and how the snapshots are made).
#   bash make_figures.sh            # all
#   bash make_figures.sh teaser     # one of: teaser overview partial wdial noise calib
set -u
P=$(cd "$(dirname "$0")" && pwd); PY=${PY:-$HOME/miniforge3/envs/splatsim/bin/python}
export LEROBOT_CACHE_DIR=${LEROBOT_CACHE_DIR:-/nonexistent} DELTAS_DIR=${DELTAS_DIR:-/nonexistent}   # force the snapshots
export FROM_CACHE=1
WHAT=${1:-"teaser overview partial wdial noise calib"}
log() { echo "[figures $(date +%H:%M:%S)] $*"; }
cd "$P"
for W in $WHAT; do case $W in
  teaser)   # Fig. 1: lever teaser (geometry + splat background are cached in the folder by render.py)
    log "Fig. 1  fig1_teaser_lever/plot.py"; (cd fig1_teaser_lever && $PY plot.py dag1_ep9_t4 > /dev/null) || exit 1
    log "        fig1_teaser/plot.py (planar teaser)"; (cd fig1_teaser && $PY plot.py > /dev/null) || exit 1 ;;
  overview) # Fig. 2: calibration panels (lever round 1, episode 14) embedded in the TikZ method overview
    log "Fig. 2  calibration panels -> fig_method_overview/calibration_panels.pdf"
    (cd fig_calibration_steps && FONT="Liberation Serif" MATHFONT=stix TASK=lever_r84 SUPTITLE= OUT=../fig_method_overview/calibration_panels.pdf \
      CONFIG_JSON='{"xlabel":"PCA dim 1","ylabel":"PCA dim 2","size_label":12,"size_title":15.5,"size_tick":10.5,"size_legend":11.5,"size_formula":12.5,"legend_xy":[0.99,0.98],"legend_align":"upper right","w_mark_frac":0.3,"titles":["(a) Open-loop sampling\n$\\hat\\Sigma_t$","(b) Blend-guided rollouts\nsettle at $W$","(c) Per-anchor\n$\\Sigma^{\\alpha}_t=(W^2/\\mathrm{tr}\\,\\hat\\Sigma_t)\\,\\hat\\Sigma_t$","(d) Pooled $\\bar\\Sigma^{\\alpha}$ and\nsampled correction labels"]}' \
      $PY plot.py > /dev/null) || exit 1
    log "        pdflatex fig_method_overview/method_overview_merged.tex"
    (cd fig_method_overview && pdflatex -interaction=nonstopmode -halt-on-error method_overview_merged.tex > build_m.log 2>&1 && grep -q "Output written" build_m.log) || { echo "pdflatex failed, see fig_method_overview/build_m.log"; exit 1; }
    (cd fig_method_overview && pdftoppm -png -r 130 -singlefile method_overview_merged.pdf method_overview_merged 2>/dev/null) ;;
  partial)  # Fig. 3: partial denoising at one observation (policy samples snapshotted; FROM_CACHE=1 replays them)
    log "Fig. 3  fig_partial_denoise/plot.py"; (cd fig_partial_denoise && $PY plot.py > /dev/null) || exit 1 ;;
  wdial)    # Fig. 4: blend dial / W fit (cells in w_dial_data.json)
    log "Fig. 4  fig_w_dial/plot.py --cached"; (cd fig_w_dial && $PY plot.py --cached > /dev/null && $PY plot_panels.py > /dev/null) || exit 1 ;;
  noise)    # per-joint sigma across rounds (not in the submitted paper)
    log "extra   fig_noise_levels/plot.py"; (cd fig_noise_levels && $PY plot.py > /dev/null) || exit 1 ;;
  calib)    # standalone 4-panel calibration figure, planar round 3 episode 32 (not in the submitted paper)
    log "extra   fig_calibration_steps/plot.py (planar)"; (cd fig_calibration_steps && $PY plot.py > /dev/null) || exit 1
    log "extra   fig_calibration_steps_lever_r84/plot.py"; (cd fig_calibration_steps_lever_r84 && $PY plot.py > /dev/null) || exit 1 ;;
  *) echo "unknown target $W"; exit 1 ;;
esac; done
log "FIGURES_DONE"
