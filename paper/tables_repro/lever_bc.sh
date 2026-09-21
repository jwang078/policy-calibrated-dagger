#!/bin/bash
# Table II step 1: the 84 px lever datasets and the BC policy.
#   1. 84x84 copy of the base demos (splatsim_approach_lever_13_smooth -> _r84), letterbox image columns dropped
#   2. BC from scratch, 75k steps (diffusion policy, base + wrist camera, 84 px inputs, no inline evals)
#   3. 100-scenario eval of the BC (outputs/eval300/lever_cam/bc_r84); the orchestrator's skip-succeeded
#      logic reads a copy of it from the training dir
# The splat node always renders 224 px; the policy resizes internally.
#   bash lever_bc.sh                      PORT=6043 by default
S=$(cd "$(dirname "$0")" && pwd); source "$S/lib_repro.sh"; LOG_TAG=lever-bc
PORT=${PORT:-6043}; BC_STEPS=${BC_STEPS:-75000}
cd "$LR"
if [ ! -d "$HF/splatsim_approach_lever_13_smooth_r${RES}/meta" ]; then
  log "shrinking base demos to ${RES}px"; $PY $S/analysis/shrink_lever_dataset.py || exit 1
  log "dropping letterbox image columns"; $PY $S/analysis/strip_letterbox_lever_r${RES}.py || exit 1
fi
[ -f outputs/dataset_stats/approach_lever_13_smooth_r${RES}/stats_rel64.json ] || \
  bash my_scripts/compute_relative_stats.sh --dataset_repo=JennyWWW/splatsim_approach_lever_13_smooth_r${RES} --chunk_sizes=64
if [ ! -d "$LEVER_BASE/checkpoints/last" ]; then
  log "BC $LEVER_BASE_NAME: $BC_STEPS steps"
  bash my_scripts/train_sweep.sh --model=diffusion --env_profile=small_engine \
    --dataset_repo=JennyWWW/splatsim_approach_lever_13_smooth_r${RES} --cameras=basewrist \
    --run_name=$LEVER_BASE_NAME --num_workers=$LEVER_WORKERS \
    --extra_args="--policy.input_features={\"observation.images.base_rgb\":{\"type\":\"VISUAL\",\"shape\":[3,$RES,$RES]},\"observation.images.wrist_rgb\":{\"type\":\"VISUAL\",\"shape\":[3,$RES,$RES]},\"observation.state\":{\"type\":\"STATE\",\"shape\":[7]}} --policy.resize_shape=[$RES,$RES] --policy.crop_ratio=1.0 --policy.down_dims=[128,256,512] --dataset.observation_noise_std={\"observation.state\":0.01,\"observation.environment_state\":0.005} --seed=0 --policy.normalize_env_state=true --steps=$BC_STEPS --save_freq=25000 --env_eval_freq=0 --wandb.enable=false" \
    > "$LEVER_BASE.log" 2>&1; log "BC rc=$?"
  [ -d "$LEVER_BASE/checkpoints/last" ] || { log "no BC checkpoint — see $LEVER_BASE.log"; exit 1; }
fi
mkdir -p "$OUT_EVAL"; SIM=$(start_lever_node $PORT); trap 'stop_node $SIM' EXIT; wait_port $PORT || exit 1
eval_lever bc_r${RES} "$LEVER_BASE/checkpoints/last/pretrained_model" $PORT 0
mkdir -p "$LEVER_BASE/eval"; cp "$OUT_EVAL/$LEVER_EVAL_GROUP/bc_r${RES}/eval_info.json" "$LEVER_BASE/eval/eval_info_step_$(printf %06d $BC_STEPS).json"
log "LEVER_BC_DONE"
