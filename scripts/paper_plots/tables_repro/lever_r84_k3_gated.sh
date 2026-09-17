#!/bin/bash
# Lever round-3 seeds, gated behind planar: starts after the planar core K=6 arms and the closed-loop arm have trained.
S=/home/jennyw2/code/lerobot/my_scripts/paper_plots/tables_repro
log() { echo "[k3gate $(date +%H:%M:%S)] $*"; }
until grep -q "core arms done" $S/planar_k6.out 2>/dev/null; do sleep 120; done
until grep -q "arm rc=" $S/planar_cl_k6.out 2>/dev/null || ! pgrep -f "planar_cl_k[6]" >/dev/null; do sleep 120; done
log "planar core + closed-loop trained -> lever round 3"; bash $S/lever_r84_k3.sh
