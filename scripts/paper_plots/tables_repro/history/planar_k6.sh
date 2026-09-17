#!/bin/bash
# Planar K=6 driver: extend the 5 lineages to round 6 (sequential, sim port 6001), compute dag6 stats,
# then two training lanes (core arms), then fixed-sigma arms (sigma=4 first across lineages, then 2, 8, 12, 16).
S=/home/jennyw2/code/lerobot/my_scripts/paper_plots/tables_repro; LR=/home/jennyw2/code/lerobot
log() { echo "[k6 $(date +%H:%M:%S)] $*"; }
cd "$LR"
for LIN in 05 06 07 08 09; do
  if [ -f "$LR/outputs/dataset_stats/planar_12_${LIN}dag_diff_r_dag6/stats_rel64.json" ]; then log "${LIN}dag round 6 present"; continue; fi
  log "lineage ${LIN}dag -> round 6"; bash outputs/dagger/${LIN}dag_cmd.sh --num_rounds=6 > outputs/dagger/${LIN}dag_run6.log 2>&1; log "${LIN}dag rc=$?"
  [ -d ~/.cache/huggingface/lerobot/JennyWWW/planar_12_${LIN}dag_diff_r_dag6/data ] || { log "ABORT: no dag6 for ${LIN}dag"; continue; }
  bash my_scripts/compute_relative_stats.sh --dataset_repo=JennyWWW/planar_12_${LIN}dag_diff_r_dag6 --chunk_sizes=64 > $S/k6_stats_${LIN}.log 2>&1; log "${LIN}dag stats rc=$?"
done
bash $S/planar_k6_train.sh "s1 s3 s5" > $S/planar_k6_laneA.out 2>&1 & A=$!
bash $S/planar_k6_train.sh "s2 s4" > $S/planar_k6_laneB.out 2>&1 & B=$!
wait $A $B; log "core arms done"
for STD in 4 2 8 12 16; do
  FIXED=1 STDS=$STD bash $S/planar_k6_train.sh "s1 s3 s5" > $S/planar_k6_fixed${STD}_A.out 2>&1 & A=$!
  FIXED=1 STDS=$STD bash $S/planar_k6_train.sh "s2 s4" > $S/planar_k6_fixed${STD}_B.out 2>&1 & B=$!
  wait $A $B; log "fixed sigma=$STD done"
done
log "PLANAR_K6_DONE"; touch $S/PLANAR_K6_DONE
