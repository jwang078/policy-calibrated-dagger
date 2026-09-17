#!/bin/bash
# REVISED 2026-09-13 (Jenny): drop the dn2-floor calibrated variants entirely
# (iso dn2 stays). GPU goes straight to extra seeds s3/s4/s5 for the two
# calibrated arms: pooled DART (dnpool) and Sigma^alpha sched (dnsig),
# alternating pooled-first per K. All steps skip-if-done.
set -u
S=/home/jennyw2/code/lerobot/my_scripts/paper_plots/tables_repro
LR=/home/jennyw2/code/lerobot
PY=/home/jennyw2/miniforge3/envs/splatsim/bin/python
BASE_CKPT=$LR/outputs/training/diffusion_planar_3joint_12_delta_stateng/checkpoints/last/pretrained_model
SW='--dataset.dart_mask_hold_tail=true --dataset.dart_selective_mask=true --policy.do_mask_loss_for_padding=true'
log() { echo "[s5c $(date +%H:%M:%S)] $*"; }
cd "$LR"
gd() { case $1 in s1) echo scarcity_study;; s2) echo scarcity_study_s2;; s3) echo scarcity_study_s3;; s4) echo scarcity_study_s4;; s5) echo scarcity_study_s5;; esac; }
lin() { case $1 in s1) echo 05dag;; s2) echo 06dag;; s3) echo 07dag;; s4) echo 08dag;; s5) echo 09dag;; esac; }
run_arm() {
  local SFX=$1 K=$2 KIND=$3
  local GD=$(gd $SFX); local TAG=$(lin $SFX)
  local NAME=q${K}_${KIND}
  local DIR="$LR/outputs/training/$GD/$NAME"
  [ -d "$DIR/checkpoints" ] && [ -n "$(ls "$DIR/checkpoints" 2>/dev/null)" ] && { log "$SFX/$NAME exists, skip"; return; }
  rm -rf "$DIR"  # clear any partial
  local SCHED
  case $KIND in
    dnpool)   SCHED=$S/analysis/noise_schedule_pooled_${SFX}_K${K}.json;;
    dnsig)    SCHED=$S/analysis/noise_schedule_sigma_alpha_${SFX}_K${K}.json;;
  esac
  [ -f "$SCHED" ] || { log "MISSING SCHEDULE $SCHED — skip $SFX/$NAME"; return; }
  local REPOS STATS WTS
  REPOS=$($PY -c "
import json
print(json.dumps(['JennyWWW/planar_3joint_12']+[f'JennyWWW/planar_12_${TAG}_diff_r_dag{i}' for i in range(1,$K+1)]))")
  STATS=$($PY -c "
import json
print(json.dumps(['outputs/dataset_stats/planar_3joint_12/stats_rel64.json']+[f'outputs/dataset_stats/planar_12_${TAG}_diff_r_dag{i}/stats_rel64.json' for i in range(1,$K+1)]))")
  WTS=$($PY -c "
import json
print(json.dumps([0.7]+[round(0.3/$K,6)]*$K))")
  log "training $SFX/$NAME"
  bash my_scripts/resume_training.sh "$BASE_CKPT" \
    --optimizer.lr=1e-6 \
    --dataset.multi_source_feature_intersection=true \
    --dataset.repo_id= \
    --dataset.repo_ids="$REPOS" \
    --dataset.sample_weights="$WTS" --dataset.stats_paths="$STATS" \
    --dataset.norm_mode=aggregated --dataset.stats_path= \
    --dataset.use_weighted_sampling=true \
    --dataset.dart_relabel=true \
    '--dataset.dart_self_relabel_pattern=_r_dag\d+$' \
    --dataset.dart_state_noise_std=0 \
    --dataset.dart_state_noise_schedule=$SCHED \
    --dataset.dart_state_noise_scale=1.0 \
    --dataset.dart_vel_noise_std=0.3 \
    $SW \
    --steps=175000 --eval_freq=175000 --save_freq=175000 \
    --output_dir="$DIR" --job_name="s5c_${SFX}_$NAME" \
    --policy.repo_id="s5c_${SFX}_$NAME" --policy.push_to_hub=false \
    --eval.n_episodes=100 \
    > "$DIR.log" 2>&1
  log "$SFX/$NAME rc=$?"
}
prep_lineage() {
  local SFX=$1; local GD=$(gd $SFX); local TAG=$(lin $SFX)
  for K in 1 2 3 4 5; do
    $PY $S/analysis/measure_sigma_multi.py ${SFX}q${K} $LR/outputs/training/$GD/q$K/checkpoints/last/pretrained_model $TAG "$(seq -s, 1 $K)" || return 1
    $PY $S/analysis/build_sigma_schedule_k.py $SFX $TAG $K || return 1
  done
}
# pooled + sig to 5 lineages, alternating pooled-first
for SFX in s3 s4 s5; do
  prep_lineage $SFX || exit 1
  for K in 1 2 3 4 5; do
    run_arm $SFX $K dnpool
    run_arm $SFX $K dnsig
  done
done
log "SAMPLE5C_DONE"
