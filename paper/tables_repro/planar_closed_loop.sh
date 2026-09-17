#!/bin/bash
# Table I closed-loop row: a DAgger lineage (tag 10cl) whose rounds are
# collected by the calibrated policy itself. Round 1 reuses lineage s1's
# round-1 interventions (the BC collects round 1 in both designs), so K=1 pairs
# exactly with s1. Each round K >= 2:
#   a. the orchestrator collects round K with the previous closed-loop fine-tune ft_dag{K-1}
#      (its own fine-tune fails on the missing round-K schedule entry — harmless);
#   b. rounds 1..K are measured on ft_dag{K-1} and the pooled schedule cl1_K<K> is built;
#   c. the orchestrator is re-invoked: collection is skipped, the fine-tune runs with that schedule.
# Afterwards the table arms q{K}_dnpoolcl are trained from base on the 10cl datasets
# (`bash planar_arms.sh cl` with ARMS=pool) and evaluated (`bash planar_eval.sh <port> cl`).
#
#   bash planar_closed_loop.sh [rounds]      default 5; the orchestrator uses sim port 6001
S=$(cd "$(dirname "$0")" && pwd); source "$S/lib_repro.sh"; LOG_TAG=closed-loop
NR=${1:-5}; FT_PREFIX=${PLANAR_BASE}_10cl_ft_dag
cd "$LR"
# ---- round 1 data, stats and K1 schedule come from lineage s1
if [ ! -d "$HF/planar_12_10cl_diff_r_dag1/meta" ]; then
  log "copying 05dag round-1 interventions -> 10cl"
  cp -r "$HF/planar_12_05dag_diff_r_dag1" "$HF/planar_12_10cl_diff_r_dag1"; rm -f "$HF"/planar_12_10cl_diff_r_dag1/dart_label_cache_*.npz
  sed -i 's/planar_12_05dag_diff_r_dag1/planar_12_10cl_diff_r_dag1/g' "$HF/planar_12_10cl_diff_r_dag1/meta/info.json"
fi
mkdir -p "$LR/outputs/dataset_stats/planar_12_10cl_diff_r_dag1"
[ -f "$LR/outputs/dataset_stats/planar_12_10cl_diff_r_dag1/stats_rel64.json" ] || cp "$LR/outputs/dataset_stats/planar_12_05dag_diff_r_dag1/stats_rel64.json" "$LR/outputs/dataset_stats/planar_12_10cl_diff_r_dag1/"
[ -f "$A/noise_schedule_pooled_cl1_K1.json" ] || $PY - "$A" <<'EOF'
import json, sys
A = sys.argv[1]; d = json.load(open(f"{A}/noise_schedule_pooled_s1_K1.json"))
json.dump({k.replace("05dag", "10cl"): v for k, v in d.items()}, open(f"{A}/noise_schedule_pooled_cl1_K1.json", "w")); print("cl1_K1 schedule seeded from s1_K1")
EOF
# ---- the per-round orchestrator command: dagger_cmds/05dag_cmd.sh with the 10cl tag and a DART-wrapped fine-tune
make_cmd() {  # NUM_ROUNDS SCHEDULE -> writes $OUT_TRAIN/10cl_cmd_k<N>.sh
  $PY - "$S/dagger_cmds/05dag_cmd.sh" "$1" "$2" "$OUT_TRAIN/10cl_cmd_k$1.sh" <<'EOF'
import re, sys
src, k, sched, out = open(sys.argv[1]).read(), sys.argv[2], sys.argv[3], sys.argv[4]
src = src.replace("--run_tag=05dag", "--run_tag=10cl").replace("--rerun_blends_from=05dag", "--rerun_blends_from=10cl")
src = re.sub(r"--num_rounds=\d+", f"--num_rounds={k}", src)
ft = ("--optimizer.lr=1e-6 --dataset.multi_source_feature_intersection=true "
      "--dataset.dart_relabel=true --dataset.dart_self_relabel_pattern=_r_dag\\\\d+$ --dataset.dart_state_noise_std=0 "
      f"--dataset.dart_state_noise_schedule={sched} --dataset.dart_state_noise_scale=1.0 --dataset.dart_vel_noise_std=0.3 "
      "--dataset.dart_mask_hold_tail=true --dataset.dart_selective_mask=true --policy.do_mask_loss_for_padding=true")
src = re.sub(r"--finetune_extra_args='[^']*'", f"--finetune_extra_args='{ft}'", src)
open(out, "w").write(src)
EOF
  echo "$OUT_TRAIN/10cl_cmd_k$1.sh"
}
have_ft() { [ -n "$(ls ${FT_PREFIX}$1/checkpoints 2>/dev/null)" ]; }
have_ds() { [ -d "$HF/planar_12_10cl_diff_r_dag$1/meta" ]; }
for K in $(seq 1 $NR); do
  have_ft $K && { log "round $K fine-tune exists, skip"; continue; }
  P=$((K-1))
  if ! have_ds $K; then
    CMD=$(make_cmd $K "$A/noise_schedule_pooled_cl1_K$([ $K = 1 ] && echo 1 || echo $P).json")
    log "round $K: collecting with ft_dag$P"; bash "$CMD" > "$OUT_TRAIN/10cl_round${K}.log" 2>&1; log "orchestrator rc=$?"
    have_ds $K || { log "round $K produced no dataset"; exit 1; }
  fi
  if ! have_ft $K; then
    MCK=${FT_PREFIX}$P/checkpoints/last/pretrained_model; [ $K = 1 ] && MCK=$PLANAR_BASE/checkpoints/last/pretrained_model
    [ -f "$A/noise_schedule_pooled_cl1_K$K.json" ] || POLICY=$MCK bash $S/planar_calibrate.sh cl $K || exit 1
    CMD=$(make_cmd $K "$A/noise_schedule_pooled_cl1_K$K.json")
    log "round $K: fine-tuning with the cl1_K$K schedule"; bash "$CMD" > "$OUT_TRAIN/10cl_round${K}b.log" 2>&1; log "orchestrator rc=$?"
    have_ft $K || { log "round $K: still no fine-tune"; exit 1; }
  fi
done
# the last round's schedule for the from-base table arm (measured on the final closed-loop fine-tune)
[ -f "$A/noise_schedule_pooled_cl1_K$NR.json" ] || POLICY=${FT_PREFIX}$NR/checkpoints/last/pretrained_model bash $S/planar_calibrate.sh cl $NR
log "CLOSED_LOOP_DONE (rounds 1..$NR); next: ARMS=pool bash planar_arms.sh cl && bash planar_eval.sh 6027 cl"
