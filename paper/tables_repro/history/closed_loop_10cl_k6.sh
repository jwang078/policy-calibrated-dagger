#!/bin/bash
# CLOSED-LOOP pooled-DART lineage (tag 10cl), Jenny 2026-09-13 queue:
# collect -> calibrate -> finetune-with-noise -> collect WITH the noise-trained
# policy -> ... Winner of the 5-seed head-to-head: pooled alpha*Sigma
# (+2.68/cell p=.0008 vs per-anchor +2.09 p=.0045; head-to-head ns).
#
# Design:
#  - Round 1 data REUSED from 05dag (base policy collects round 1 in both
#    designs) -> exact K=1 pairing with the s1 plain lineage.
#  - Per-round finetunes carry the pooled schedule (04dagsw precedent, but
#    schedule instead of fixed dn8; no raw_mix, matching the table recipe).
#  - Schedule for round k's finetune: rounds 1..k-1 measured on the previous
#    CLOSED-LOOP checkpoint (ft_dag{k-1}); round 1 uses the existing s1-K1
#    pooled schedule (same data, base checkpoint). This one-round lag is
#    DART's own protocol (psi_{k+1} estimated from round k).
#  - Orchestrator invoked with --num_rounds=k --resume per round so the
#    schedule can be rebuilt between rounds. All steps skip-if-done.
set -u
S=/home/jennyw2/code/lerobot/my_scripts/paper_plots/tables_repro
LR=/home/jennyw2/code/lerobot
PY=/home/jennyw2/miniforge3/envs/splatsim/bin/python
HF=$HOME/.cache/huggingface/lerobot/JennyWWW
FT_PREFIX=$LR/outputs/training/diffusion_planar_3joint_12_delta_stateng_10cl_ft_dag
log() { echo "[10cl $(date +%H:%M:%S)] $*"; }
cd "$LR"

# ---- pre: seed round-1 dataset + stats + K1 schedule from 05dag/s1 ----
if [ ! -d "$HF/planar_12_10cl_diff_r_dag1/meta" ]; then
  log "copying 05dag r_dag1 dataset -> 10cl"
  rm -rf "$HF/planar_12_10cl_diff_r_dag1"
  cp -r "$HF/planar_12_05dag_diff_r_dag1" "$HF/planar_12_10cl_diff_r_dag1"
  rm -f "$HF"/planar_12_10cl_diff_r_dag1/dart_label_cache_*.npz
  sed -i 's/planar_12_05dag_diff_r_dag1/planar_12_10cl_diff_r_dag1/g' \
    "$HF/planar_12_10cl_diff_r_dag1/meta/info.json"
fi
if [ ! -f "$LR/outputs/dataset_stats/planar_12_10cl_diff_r_dag1/stats_rel64.json" ]; then
  mkdir -p "$LR/outputs/dataset_stats/planar_12_10cl_diff_r_dag1"
  cp "$LR/outputs/dataset_stats/planar_12_05dag_diff_r_dag1/stats_rel64.json" \
     "$LR/outputs/dataset_stats/planar_12_10cl_diff_r_dag1/"
fi
if [ ! -f "$S/analysis/noise_schedule_pooled_cl1_K1.json" ]; then
  $PY - <<'EOF'
import json
S="/home/jennyw2/code/lerobot/my_scripts/paper_plots/tables_repro"
d=json.load(open(f"{S}/analysis/noise_schedule_pooled_s1_K1.json"))
out={k.replace("05dag","10cl"): v for k,v in d.items()}
json.dump(out, open(f"{S}/analysis/noise_schedule_pooled_cl1_K1.json","w"))
print("cl1_K1 schedule seeded from s1_K1:", list(out))
EOF
fi

# ---- per-round orchestrator invocation with the right schedule ----
make_cmd() {  # $1 = num_rounds, $2 = schedule path -> writes $S/10cl_cmd_k$1.sh
  $PY - "$1" "$2" <<'EOF'
import re, sys
k, sched = sys.argv[1], sys.argv[2]
src = open("/home/jennyw2/code/lerobot/outputs/dagger/05dag_cmd.sh").read()
src = src.replace("--run_tag=05dag", "--run_tag=10cl")
src = src.replace("--rerun_blends_from=05dag", "--rerun_blends_from=10cl")
src = re.sub(r"--num_rounds=\d+", f"--num_rounds={k}", src)
ft = ("--optimizer.lr=1e-6 --dataset.multi_source_feature_intersection=true "
      "--dataset.dart_relabel=true --dataset.dart_self_relabel_pattern=_r_dag\\\\d+$ "
      "--dataset.dart_state_noise_std=0 "
      f"--dataset.dart_state_noise_schedule={sched} "
      "--dataset.dart_state_noise_scale=1.0 --dataset.dart_vel_noise_std=0.3 "
      "--dataset.dart_mask_hold_tail=true --dataset.dart_selective_mask=true "
      "--policy.do_mask_loss_for_padding=true")
src = re.sub(r"--finetune_extra_args='[^']*'", f"--finetune_extra_args='{ft}'", src)
out = f"/home/jennyw2/code/lerobot/my_scripts/paper_plots/tables_repro/10cl_cmd_k{k}.sh"
open(out, "w").write(src)
print(out)
EOF
}

# Per round K>=2: the dart wrapper needs a schedule entry for EVERY r_dag
# dataset in the finetune, including the round just collected — so the flow is
# (a) orchestrator invocation collects round K (its finetune fails on the
# missing r_dagK entry, harmless: no partial ckpt); (b) measure rounds 1..K on
# the COLLECTING checkpoint ft_dag{K-1} and build the full same-round schedule
# (the table arms' protocol, with the noise-trained policy as measurement
# policy); (c) re-invoke — resume skips collection, finetune runs.
have_ft() { [ -d "${FT_PREFIX}$1/checkpoints" ] && [ -n "$(ls ${FT_PREFIX}$1/checkpoints 2>/dev/null)" ]; }
have_ds() { [ -d "$HF/planar_12_10cl_diff_r_dag$1/meta" ]; }

for K in 6; do
  if have_ft $K; then log "round $K ft exists, skip"; continue; fi
  PREV=$((K-1))
  # (a) collect round K if its dataset is not on disk yet
  if ! have_ds $K; then
    SCHED=$S/analysis/noise_schedule_pooled_cl1_K${PREV}.json
    [ $K -eq 1 ] && SCHED=$S/analysis/noise_schedule_pooled_cl1_K1.json
    make_cmd $K "$SCHED"
    log "round $K: orchestrator attempt 1 (collection; finetune may fail on missing r_dag$K schedule entry)"
    bash $S/10cl_cmd_k${K}.sh > $S/10cl_round${K}.log 2>&1
    log "round $K attempt 1 rc=$?"
    have_ds $K || { log "round $K collection produced no dataset — abort"; exit 1; }
  fi
  if have_ft $K; then log "round $K complete after attempt 1"; else
    # (b) full same-round schedule measured on the checkpoint that collected
    MCK=${FT_PREFIX}${PREV}/checkpoints/last/pretrained_model
    [ $K -eq 1 ] && MCK=$LR/outputs/training/diffusion_planar_3joint_12_delta_stateng/checkpoints/last/pretrained_model
    if [ ! -f "$S/analysis/noise_schedule_pooled_cl1_K${K}.json" ]; then
      log "measuring rounds 1..$K on $(basename $(dirname $(dirname $MCK)))"
      $PY $S/analysis/measure_sigma_multi.py cl1q${K} "$MCK" 10cl "$(seq -s, 1 $K)" || exit 1
      $PY $S/analysis/build_sigma_schedule_k.py cl1 10cl $K || exit 1
    fi
    # (c) retry: finetune with the complete schedule
    make_cmd $K "$S/analysis/noise_schedule_pooled_cl1_K${K}.json"
    log "round $K: orchestrator attempt 2 (finetune with cl1_K$K schedule)"
    bash $S/10cl_cmd_k${K}.sh > $S/10cl_round${K}b.log 2>&1
    log "round $K attempt 2 rc=$?"
    have_ft $K || { log "round $K still no ft ckpt — abort"; exit 1; }
  fi
done
# final schedule (rounds 1..5 on ft_dag5) for the from-base table arms
if [ ! -f "$S/analysis/noise_schedule_pooled_cl1_K5.json" ]; then
  $PY $S/analysis/measure_sigma_multi.py cl1q5 \
    ${FT_PREFIX}5/checkpoints/last/pretrained_model 10cl "1,2,3,4,5" || true
  $PY $S/analysis/build_sigma_schedule_k.py cl1 10cl 5 || true
fi
log "CL_DONE — closed-loop lineage complete (collector-chain protocol)."
log "Next: from-base 175k table arms on 10cl datasets for row-comparable numbers."
