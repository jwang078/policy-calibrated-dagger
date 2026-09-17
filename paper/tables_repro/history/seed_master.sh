#!/bin/bash
S=/home/jennyw2/code/lerobot/my_scripts/paper_plots/tables_repro
LR=/home/jennyw2/code/lerobot
log() { echo "[seeds $(date +%H:%M:%S)] $*"; }
log "lineage 06dag (seed B)"
bash $LR/outputs/dagger/06dag_cmd.sh > $LR/outputs/dagger/06dag_run.log 2>&1
log "lineage 06dag rc=$? — grid s2"
bash $S/seed_grid.sh 06dag s2
log "lineage 07dag (seed C)"
bash $LR/outputs/dagger/07dag_cmd.sh > $LR/outputs/dagger/07dag_run.log 2>&1
log "lineage 07dag rc=$? — grid s3"
bash $S/seed_grid.sh 07dag s3
log "SEEDS_DONE"
