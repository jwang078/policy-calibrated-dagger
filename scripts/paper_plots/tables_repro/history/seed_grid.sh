#!/bin/bash
# SEED-REPLICA GRID: baselines + iso dn ladder on lineage $TAG, outputs to
# scarcity_study_$SFX. Usage: seed_grid.sh <TAG e.g. 06dag> <SFX e.g. s2>
set -u
TAG=$1; SFX=$2
S=/home/jennyw2/code/lerobot/my_scripts/paper_plots/tables_repro
LR=/home/jennyw2/code/lerobot
BASE_CKPT=$LR/outputs/training/diffusion_planar_3joint_12_delta_stateng/checkpoints/last/pretrained_model
SW='--dataset.dart_mask_hold_tail=true --dataset.dart_selective_mask=true --policy.do_mask_loss_for_padding=true'
log() { echo "[grid-$SFX $(date +%H:%M:%S)] $*"; }
cd "$LR"
for i in 1 2 3 4 5; do
  [ -f "$LR/outputs/dataset_stats/planar_12_${TAG}_diff_r_dag$i/stats_rel64.json" ] || \
    bash my_scripts/compute_relative_stats.sh --dataset_repo=JennyWWW/planar_12_${TAG}_diff_r_dag$i --chunk_sizes=64
done
run_arm() {
  local K=$1 NAME=$2; shift 2
  local DIR="$LR/outputs/training/scarcity_study_$SFX/$NAME"
  [ -d "$DIR/checkpoints" ] && { log "$NAME exists, skip"; return; }
  local REPOS STATS WTS
  REPOS=$(python3 -c "
import json
print(json.dumps(['JennyWWW/planar_3joint_12']+[f'JennyWWW/planar_12_${TAG}_diff_r_dag{i}' for i in range(1,$K+1)]))")
  STATS=$(python3 -c "
import json
print(json.dumps(['outputs/dataset_stats/planar_3joint_12/stats_rel64.json']+[f'outputs/dataset_stats/planar_12_${TAG}_diff_r_dag{i}/stats_rel64.json' for i in range(1,$K+1)]))")
  WTS=$(python3 -c "
import json
print(json.dumps([0.7]+[round(0.3/$K,6)]*$K))")
  log "training $NAME"
  bash my_scripts/resume_training.sh "$BASE_CKPT" \
    --optimizer.lr=1e-6 \
    --dataset.multi_source_feature_intersection=true \
    --dataset.repo_id= \
    --dataset.repo_ids="$REPOS" \
    --dataset.sample_weights="$WTS" --dataset.stats_paths="$STATS" \
    --dataset.norm_mode=aggregated --dataset.stats_path= \
    --dataset.use_weighted_sampling=true \
    "$@" \
    --steps=175000 --eval_freq=175000 --save_freq=175000 \
    --output_dir="$DIR" --job_name="scarcity_${SFX}_$NAME" \
    --policy.repo_id="scarcity_${SFX}_$NAME" --policy.push_to_hub=false \
    --eval.n_episodes=100 \
    > "$DIR.log" 2>&1
  log "$NAME rc=$?"
}
mkdir -p $LR/outputs/training/scarcity_study_$SFX
for K in 1 2 3 4 5; do
  run_arm $K "q${K}" --dataset.dart_relabel=false
  for STD in 2 4 8 12 16; do
    run_arm $K "q${K}_dn${STD}swv" \
      --dataset.dart_relabel=true \
      '--dataset.dart_self_relabel_pattern=_r_dag\d+$' \
      --dataset.dart_state_noise_std=$STD \
      --dataset.dart_vel_noise_std=0.3 \
      $SW
  done
done
log "GRID_${SFX}_DONE"
