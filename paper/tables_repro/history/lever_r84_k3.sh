#!/bin/bash
# Lever 84px round 3: training seeds 1,2 (HG + calibrated from base, +20k, 100-ep eval with matching seed),
# then three lanes fill the remaining eval seeds so every round-3 checkpoint (seeds 0,1,2) has 300 episodes.
S=/home/jennyw2/code/lerobot/my_scripts/paper_plots/tables_repro
log() { echo "[r84 k3 $(date +%H:%M:%S)] $*"; }
rm -f $S/LEVER_R84_SEEDS_DONE
SEEDS="1 2" ROUNDS="3" bash $S/lever_r84_seeds.sh > $S/lever_r84_seeds_k3.out 2>&1; log "seeds rc=$?"
for L in 0 1 2; do P=$((6042+L)); [ $L = 2 ] && P=6047; KS=3 bash $S/lever_r84_lane.sh $L 3 $P > $S/lever_r84_k3_lane$L.out 2>&1 & done; wait
log "LEVER_R84_K3_DONE"; touch $S/LEVER_R84_K3_DONE
