#!/bin/bash
# Calibrated Sigma^alpha arm (DART-faithful anisotropic prediction), K=2, s1+s2.
set -u
S=/home/jennyw2/code/lerobot/my_scripts/paper_plots/tables_repro
LR=/home/jennyw2/code/lerobot
PY=/home/jennyw2/miniforge3/envs/splatsim/bin/python
BASE_CKPT=$LR/outputs/training/diffusion_planar_3joint_12_delta_stateng/checkpoints/last/pretrained_model
SW='--dataset.dart_mask_hold_tail=true --dataset.dart_selective_mask=true --policy.do_mask_loss_for_padding=true'
log() { echo "[poolarm $(date +%H:%M:%S)] $*"; }
while pgrep -f "bin/lerobot-trai[n]" >/dev/null; do sleep 300; done
log "training sigma-alpha arms"
cd "$LR"
run() {
  local SFX=$1 TAG=$2
  local DIR="$LR/outputs/training/scarcity_study/q2_dnpool"
  [ "$SFX" = "s2" ] && DIR="$LR/outputs/training/scarcity_study_s2/q2_dnpool"
  [ -d "$DIR/checkpoints" ] && { log "$SFX exists, skip"; return; }
  bash my_scripts/resume_training.sh "$BASE_CKPT" \
    --optimizer.lr=1e-6 \
    --dataset.multi_source_feature_intersection=true \
    --dataset.repo_id= \
    --dataset.repo_ids="[\"JennyWWW/planar_3joint_12\", \"JennyWWW/planar_12_${TAG}_diff_r_dag1\", \"JennyWWW/planar_12_${TAG}_diff_r_dag2\"]" \
    --dataset.sample_weights="[0.7, 0.15, 0.15]" \
    --dataset.stats_paths="[\"outputs/dataset_stats/planar_3joint_12/stats_rel64.json\", \"outputs/dataset_stats/planar_12_${TAG}_diff_r_dag1/stats_rel64.json\", \"outputs/dataset_stats/planar_12_${TAG}_diff_r_dag2/stats_rel64.json\"]" \
    --dataset.norm_mode=aggregated --dataset.stats_path= \
    --dataset.use_weighted_sampling=true \
    --dataset.dart_relabel=true \
    '--dataset.dart_self_relabel_pattern=_r_dag\d+$' \
    --dataset.dart_state_noise_std=0 \
    --dataset.dart_state_noise_schedule=$S/analysis/noise_schedule_pooled_${SFX}_K2.json \
    --dataset.dart_state_noise_scale=1.0 \
    --dataset.dart_vel_noise_std=0.3 \
    $SW \
    --steps=175000 --eval_freq=175000 --save_freq=175000 \
    --output_dir="$DIR" --job_name="pool_${SFX}_q2_dnpool" \
    --policy.repo_id="pool_${SFX}_q2_dnpool" --policy.push_to_hub=false \
    --eval.n_episodes=100 \
    > "$DIR.log" 2>&1
  log "$SFX rc=$?"
}
run s1 05dag
run s2 06dag
log "POOLARM_DONE"
