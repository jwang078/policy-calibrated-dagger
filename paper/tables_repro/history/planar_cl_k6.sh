#!/bin/bash
# Closed-loop pooled row, K=6 (priority #2): waits for the 5 lineage collections (sim port 6001) and the lever trainers,
# then: 10cl round 6 (collect with cl ft_dag5, measure sigma, cl1_K6 schedule, lineage finetune) -> from-base arm q6_dnpoolcl -> 300-ep eval.
S=/home/jennyw2/code/lerobot/my_scripts/paper_plots/tables_repro
log() { echo "[cl-k6 $(date +%H:%M:%S)] $*"; }
until grep -qE "09dag (stats rc=|round 6 present)" $S/planar_k6.out 2>/dev/null; do sleep 60; done; log "lineage collections done (port 6001 free)"
until grep -q "cal3_s2 rc=" $S/lever_r84_seeds_k3.out 2>/dev/null || ! pgrep -f "lever_r84_seed[s]" >/dev/null; do sleep 60; done; log "lever trainers done"
bash $S/closed_loop_10cl_k6.sh > $S/closed_loop_10cl_k6.out 2>&1; log "closed-loop round 6 rc=$?"
bash $S/cl_arms_k6.sh > $S/cl_arms_k6.out 2>&1; log "arm rc=$?"
bash $S/planar_cl_eval.sh 0 1 6030 > $S/planar_cl_eval.out 2>&1; log "CL_K6_DONE"
