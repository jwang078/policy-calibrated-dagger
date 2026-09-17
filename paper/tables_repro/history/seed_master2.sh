#!/bin/bash
# Seeds 4-5: waits for the current queue to drain (s3 grid + sub-dn2 arms),
# then 08dag lineage -> grid s4 -> 09dag lineage -> grid s5 -> 300-ep re-evals.
S=/home/jennyw2/code/lerobot/my_scripts/paper_plots/tables_repro
LR=/home/jennyw2/code/lerobot
log() { echo "[seeds45 $(date +%H:%M:%S)] $*"; }
# gate 1: s3 grid finished (log sentinel)
until grep -q "SEEDS_DONE" $S/seed_master.log 2>/dev/null; do sleep 600; done
log "s3 grid done"
# gate 2: no trainer running (bracket-trick pattern per pgrep rule; covers sub-dn2 arms)
while pgrep -f "bin/lerobot-trai[n]" >/dev/null; do sleep 600; done
until [ -f /home/jennyw2/code/lerobot/my_scripts/paper_plots/tables_repro/aniso_done ]; do sleep 300; done
while pgrep -f "bin/lerobot-trai[n]" >/dev/null; do sleep 600; done
log "GPU idle — starting 08dag lineage (seed 4)"
bash $LR/outputs/dagger/08dag_cmd.sh > $LR/outputs/dagger/08dag_run.log 2>&1
log "lineage 08dag rc=$? — grid s4"
bash $S/seed_grid.sh 08dag s4
log "lineage 09dag (seed 5)"
bash $LR/outputs/dagger/09dag_cmd.sh > $LR/outputs/dagger/09dag_run.log 2>&1
log "lineage 09dag rc=$? — grid s5"
bash $S/seed_grid.sh 09dag s5
log "grids done — 300-ep re-evals for s4/s5"
bash $S/reeval300_s45.sh
log "SEEDS45_DONE"
