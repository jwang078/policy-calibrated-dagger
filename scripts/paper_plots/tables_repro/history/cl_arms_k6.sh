#!/bin/bash
# From-base 175k table arms on the CLOSED-LOOP (10cl) datasets: q{K}_dnpoolcl,
# K=1..5, into scarcity_study_cl — identical recipe to the retrospective
# dnpool arms (sample5c run_arm), datasets/schedules swapped to 10cl/cl1.
set -u
S=/home/jennyw2/code/lerobot/my_scripts/paper_plots/tables_repro
LR=/home/jennyw2/code/lerobot
PY=/home/jennyw2/miniforge3/envs/splatsim/bin/python
BASE_CKPT=$LR/outputs/training/diffusion_planar_3joint_12_delta_stateng/checkpoints/last/pretrained_model
SW='--dataset.dart_mask_hold_tail=true --dataset.dart_selective_mask=true --policy.do_mask_loss_for_padding=true'
log() { echo "[clarms $(date +%H:%M:%S)] $*"; }
cd "$LR"
until [ "$(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo)" -ge 12 ]; do sleep 60; done; log "RAM gate passed"
GD=scarcity_study_cl
mkdir -p "$LR/outputs/training/$GD"
for K in 6; do
  NAME=q${K}_dnpoolcl
  DIR="$LR/outputs/training/$GD/$NAME"
  [ -d "$DIR/checkpoints" ] && [ -n "$(ls "$DIR/checkpoints" 2>/dev/null)" ] && { log "$NAME exists, skip"; continue; }
  rm -rf "$DIR"
  SCHED=$S/analysis/noise_schedule_pooled_cl1_K${K}.json
  [ -f "$SCHED" ] || { log "MISSING $SCHED — skip"; continue; }
  REPOS=$($PY -c "
import json
print(json.dumps(['JennyWWW/planar_3joint_12']+[f'JennyWWW/planar_12_10cl_diff_r_dag{i}' for i in range(1,$K+1)]))")
  STATS=$($PY -c "
import json
print(json.dumps(['outputs/dataset_stats/planar_3joint_12/stats_rel64.json']+[f'outputs/dataset_stats/planar_12_10cl_diff_r_dag{i}/stats_rel64.json' for i in range(1,$K+1)]))")
  WTS=$($PY -c "
import json
print(json.dumps([0.7]+[round(0.3/$K,6)]*$K))")
  log "training $NAME"
  bash my_scripts/resume_training.sh "$BASE_CKPT" \
    --optimizer.lr=${LR_FT:-1e-5} --num_workers=3 \
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
    --output_dir="$DIR" --job_name="clarms_$NAME" \
    --policy.repo_id="clarms_$NAME" --policy.push_to_hub=false \
    --eval.n_episodes=100 \
    > "$DIR.log" 2>&1
  log "$NAME rc=$?"
done
log "CLARMS_DONE"
