#!/bin/bash
# Closed-loop pooled row, extra eval seeds: 300-ep evals of q{K}_dnpoolcl (K=1..5) with --seed 1 and 2 -> SE over 3 eval seeds.
set -u
LANE=$1; NL=$2; PORT=$3
S=/home/jennyw2/code/lerobot/my_scripts/paper_plots/tables_repro; LR=/home/jennyw2/code/lerobot; EV=$HOME/miniforge3/envs/splatsim/bin/lerobot-eval
log() { echo "[clseed$LANE $(date +%H:%M:%S)] $*"; }
SUBSET100=$(python3 -c "print('['+','.join(str(i) for i in range(100))+']')")
cd /home/jennyw2/code/SplatSim
python -u scripts/launch_nodes.py --robot sim_pybullet_planar_interactive --robot_port $PORT --hostname 127.0.0.1 --eval_benchmark_repo_id JennyWWW/eval_planar_3joint_benchmark --robot_name planar_3joint --headless --control_gui > "$S/clseed${LANE}_sim.log" 2>&1 & SIM=$!; trap 'kill ${SIM:-} 2>/dev/null' EXIT
for i in $(seq 1 120); do grep -q "Serving\|serve loop\|Listening\|Robot server" "$S/clseed${LANE}_sim.log" 2>/dev/null && break; sleep 2; done; log "sim up on $PORT"; cd "$LR"
eval_one() { local GD=$1 NAME=$2 SD=$3; local CK="$LR/outputs/training/$GD/$NAME/checkpoints/last/pretrained_model"; local D="$LR/outputs/eval300/$GD/${NAME}_e$SD"
  [ -f "$D/eval_info.json" ] && { log "$GD/$NAME done"; return; }
  [ -d "$CK" ] || { log "$GD/$NAME: no ckpt"; return; }
  rm -rf "$D"; mkdir -p "$D"; log "eval300 $GD/$NAME seed $SD"
  $EV --env.type=splatsim --env.task=planar_3joint --env.camera_names='["base_rgb"]' --env.image_resize_modes='["letterbox"]' --env.fps=30 --env.eval_benchmark_repo_id=JennyWWW/eval_planar_3joint_benchmark \
    --policy.path="$CK" --eval.n_episodes=300 --output_dir="$D" --eval.batch_size=1 --eval.use_async_envs=false --seed=$SD --rename_map='{}' --env.episode_length=1000 --env.eval_benchmark_subset="$SUBSET100" \
    --env.external_port=$PORT --env.max_parallel_tasks=1 --env.robot_name=planar_3joint --env.cam_i=3 --env.use_gripper=true --env.debug_mode=off --env.headless=true --env.control_gui=true --env.splat_shadows=false \
    --env.include_oracle_info=false --env.terminate_on_collision=true --env.wrist_cam_ver=2 --env.teleop_pad_short_episodes=true --env.teleop_state_jump_split_threshold_rad=0.15 --env.num_dofs=3 --env.state_dim=4 --env.action_dim=4 --env.env_state_dim=8 > "$D/log.txt" 2>&1
  local RC=$?; local SUCC=$(python3 -c "import json;print(json.load(open('$D/eval_info.json'))['overall']['pc_success'])" 2>/dev/null); log "$GD/${NAME}_e$SD rc=$RC succ=${SUCC:-NA}"; }
JOBS=(); for SD in 1 2; do for K in 1 2 3 4 5; do JOBS+=("scarcity_study_cl q${K}_dnpoolcl $SD"); done; done
n=0; for J in "${JOBS[@]}"; do [ $((n % NL)) = $LANE ] && eval_one $J; n=$((n+1)); done
log "CLSEED_LANE${LANE}_DONE"
