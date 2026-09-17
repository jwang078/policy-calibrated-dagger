#!/bin/bash
# Planar K=6 driver v7: after both core lanes finish -> closed-loop K=6 arm (from base, cl1_K6 schedule) + its 300-ep eval
# -> fixed-sigma phases (two lanes, each start gated on >= 12 GB free RAM).
S=/home/jennyw2/code/lerobot/my_scripts/paper_plots/tables_repro
log() { echo "[k6h $(date +%H:%M:%S)] $*"; }
avail() { awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo; }
until grep -q "LANE_DONE" $S/planar_k6_laneB.out 2>/dev/null; do sleep 120; done; log "lane B core done -> closed-loop arm"
bash $S/cl_arms_k6.sh > $S/cl_arms_k6.out 2>&1; log "closed-loop arm rc=$?"
bash $S/planar_cl_eval.sh 0 1 6030 > $S/planar_cl_eval.out 2>&1 & CLE=$!
for STD in 4 2 8 12 16; do
  until [ "$(avail)" -ge 12 ]; do sleep 60; done
  FIXED=1 STDS=$STD bash $S/planar_k6_train2.sh "s1 s3 s5" > $S/planar_k6_fixed${STD}_A.out 2>&1 & A=$!
  sleep 180; until [ "$(avail)" -ge 12 ]; do sleep 60; done
  FIXED=1 STDS=$STD bash $S/planar_k6_train2.sh "s2 s4" > $S/planar_k6_fixed${STD}_B.out 2>&1 & B=$!
  wait $A $B; log "fixed sigma=$STD done"
done
wait $CLE; log "PLANAR_K6_DONE"; touch $S/PLANAR_K6_DONE
