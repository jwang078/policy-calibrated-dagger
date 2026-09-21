#!/bin/bash
# Priority: proposed-method arms at K=1,3,4,5 on s1+s2 — ALL pooled-DART first,
# then all sigma-alpha. Afterwards restores the s5 grid + s4/s5 re-evals.
set -u
S=/home/jennyw2/code/lerobot/my_scripts/paper_plots/tables_repro
LR=/home/jennyw2/code/lerobot
PY=/home/jennyw2/miniforge3/envs/splatsim/bin/python
BASE_CKPT=$LR/outputs/training/diffusion_planar_3joint_12_delta_stateng/checkpoints/last/pretrained_model
SW='--dataset.dart_mask_hold_tail=true --dataset.dart_selective_mask=true --policy.do_mask_loss_for_padding=true'
log() { echo "[calibK $(date +%H:%M:%S)] $*"; }
cd "$LR"
# --- measurements (skip-if-done inside) ---
for pair in "s1 05dag scarcity_study" "s2 06dag scarcity_study_s2"; do
  set -- $pair; SFX=$1; LIN=$2; GD=$3
  for K in 1 3 4 5; do
    RDS=$(seq -s, 1 $K)
    $PY $S/analysis/measure_sigma_multi.py ${SFX}q${K} $LR/outputs/training/$GD/q$K/checkpoints/last/pretrained_model $LIN "$RDS" || exit 1
  done
done
# --- schedules ---
for pair in "s1 05dag" "s2 06dag"; do
  set -- $pair
  for K in 1 3 4 5; do $PY $S/analysis/build_sigma_schedule_k.py $1 $2 $K || exit 1; done
done
log "measurements + schedules done"
run_arm() {
  local SFX=$1 TAG=$2 K=$3 NAME=$4 SCHED=$5
  local GD=scarcity_study; [ "$SFX" = "s2" ] && GD=scarcity_study_s2
  local DIR="$LR/outputs/training/$GD/$NAME"
  [ -d "$DIR/checkpoints" ] && { log "$SFX/$NAME exists, skip"; return; }
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
    --output_dir="$DIR" --job_name="calibK_${SFX}_$NAME" \
    --policy.repo_id="calibK_${SFX}_$NAME" --policy.push_to_hub=false \
    --eval.n_episodes=100 \
    > "$DIR.log" 2>&1
  log "$SFX/$NAME rc=$?"
}
# --- ALL pooled first ---
for K in 1 3 4 5; do
  run_arm s1 05dag $K q${K}_dnpool $S/analysis/noise_schedule_pooled_s1_K${K}.json
  run_arm s2 06dag $K q${K}_dnpool $S/analysis/noise_schedule_pooled_s2_K${K}.json
done
log "pooled arms done — sigma-alpha arms"
for K in 1 3 4 5; do
  run_arm s1 05dag $K q${K}_dnsig $S/analysis/noise_schedule_sigma_alpha_s1_K${K}.json
  run_arm s2 06dag $K q${K}_dnsig $S/analysis/noise_schedule_sigma_alpha_s2_K${K}.json
done
log "calibrated arms complete — restoring s5 grid + re-evals"
bash $S/seed_grid.sh 09dag s5
bash $S/reeval300_s45.sh
log "CALIBK_DONE"
