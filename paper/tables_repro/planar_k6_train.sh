#!/bin/bash
# Planar K=6 training lane. usage: planar_k6_train.sh "<lineages e.g. s1 s3 s5>" ; per lineage: q6 (HG from base) ->
# sigma measurement (q6 policy on dag1..6) -> K=6 schedules -> q6_dnpool -> q6_dnsig. Fixed-sigma arms run in a second phase (FIXED=1).
set -u
LINS=$1; FIXED=${FIXED:-0}
S=/home/jennyw2/code/lerobot/my_scripts/paper_plots/tables_repro
LR=/home/jennyw2/code/lerobot; PY=/home/jennyw2/miniforge3/envs/splatsim/bin/python
BASE_CKPT=$LR/outputs/training/diffusion_planar_3joint_12_delta_stateng/checkpoints/last/pretrained_model
SW='--dataset.dart_mask_hold_tail=true --dataset.dart_selective_mask=true --policy.do_mask_loss_for_padding=true'
log() { echo "[k6train $(date +%H:%M:%S)] $*"; }
lin() { case $1 in s1) echo "05dag scarcity_study";; s2) echo "06dag scarcity_study_s2";; s3) echo "07dag scarcity_study_s3";; s4) echo "08dag scarcity_study_s4";; s5) echo "09dag scarcity_study_s5";; esac; }
cd "$LR"; K=6
run_arm() { local SFX=$1 TAG=$2 GD=$3 NAME=$4; shift 4
  local DIR="$LR/outputs/training/$GD/$NAME"; [ -d "$DIR/checkpoints" ] && { log "$SFX/$NAME exists, skip"; return; }
  local REPOS STATS WTS
  REPOS=$($PY -c "import json;print(json.dumps(['JennyWWW/planar_3joint_12']+[f'JennyWWW/planar_12_${TAG}_diff_r_dag{i}' for i in range(1,$K+1)]))")
  STATS=$($PY -c "import json;print(json.dumps(['outputs/dataset_stats/planar_3joint_12/stats_rel64.json']+[f'outputs/dataset_stats/planar_12_${TAG}_diff_r_dag{i}/stats_rel64.json' for i in range(1,$K+1)]))")
  WTS=$($PY -c "import json;print(json.dumps([0.7]+[round(0.3/$K,6)]*$K))")
  log "training $SFX/$NAME"; rm -rf "$DIR"
  bash my_scripts/resume_training.sh "$BASE_CKPT" --optimizer.lr=1e-6 --num_workers=6 --dataset.multi_source_feature_intersection=true --dataset.repo_id= \
    --dataset.repo_ids="$REPOS" --dataset.sample_weights="$WTS" --dataset.stats_paths="$STATS" --dataset.norm_mode=aggregated --dataset.stats_path= --dataset.use_weighted_sampling=true \
    "$@" --steps=175000 --eval_freq=175000 --save_freq=175000 --output_dir="$DIR" --job_name="k6_${SFX}_$NAME" --policy.repo_id="k6_${SFX}_$NAME" --policy.push_to_hub=false --eval.n_episodes=100 > "$DIR.log" 2>&1
  log "$SFX/$NAME rc=$?"; }
for SFX in $LINS; do set -- $(lin $SFX); TAG=$1; GD=$2
  until [ -f "$LR/outputs/dataset_stats/planar_12_${TAG}_diff_r_dag6/stats_rel64.json" ]; do sleep 60; done
  if [ "$FIXED" = 0 ]; then
    run_arm $SFX $TAG $GD q6 --dataset.dart_relabel=false
    if [ ! -f $S/analysis/noise_schedule_pooled_${SFX}_K6.json ]; then
      $PY $S/analysis/measure_sigma_multi.py ${SFX}q6 $LR/outputs/training/$GD/q6/checkpoints/last/pretrained_model $TAG "1,2,3,4,5,6" > $S/k6_sigma_$SFX.log 2>&1; log "sigma $SFX rc=$?"
      $PY $S/analysis/build_sigma_schedule_k.py $SFX $TAG 6 > $S/k6_sched_$SFX.log 2>&1; log "schedule $SFX rc=$?"; fi
    [ -f $S/analysis/noise_schedule_pooled_${SFX}_K6.json ] || { log "ABORT $SFX: no K6 schedule"; continue; }
    for V in pool sig; do SCH=$S/analysis/noise_schedule_$([ $V = pool ] && echo pooled || echo sigma_alpha)_${SFX}_K6.json
      run_arm $SFX $TAG $GD q6_dn$V --dataset.dart_relabel=true '--dataset.dart_self_relabel_pattern=_r_dag\d+$' --dataset.dart_state_noise_std=0 --dataset.dart_state_noise_schedule=$SCH --dataset.dart_state_noise_scale=1.0 --dataset.dart_vel_noise_std=0.3 $SW; done
  else
    for STD in ${STDS:-4 2 8 12 16}; do run_arm $SFX $TAG $GD q6_dn${STD}swv --dataset.dart_relabel=true '--dataset.dart_self_relabel_pattern=_r_dag\d+$' --dataset.dart_state_noise_std=$STD --dataset.dart_vel_noise_std=0.3 $SW; done
  fi
done
log "LANE_DONE ($LINS FIXED=$FIXED)"
