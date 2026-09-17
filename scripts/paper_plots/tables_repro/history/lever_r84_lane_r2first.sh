#!/bin/bash
# Eval lane: usage lever_r84_lane.sh LANE NLANES PORT.  Evaluates every (round, training seed, arm, eval seed) cell of the
# 300-episode table for rounds 1-2 (eval seeds != training seed), taking items LANE, LANE+NLANES, ... from the list.
set -u
LANE=$1; NL=$2; PORT=$3
S=/home/jennyw2/code/lerobot/my_scripts/paper_plots/tables_repro
LR=/home/jennyw2/code/lerobot; TR=$LR/outputs/training; PY=/home/jennyw2/miniforge3/envs/splatsim/bin/python; EV=$HOME/miniforge3/envs/splatsim/bin/lerobot-eval
BASE=diffusion_approach_lever_13_smooth_r84_delta_basewrist
log() { echo "[lane$LANE $(date +%H:%M:%S)] $*"; }
ckpt() { local K=$1 TS=$2 ARM=$3   # checkpoint dir for round K, training seed TS, arm hg|cal (+20k = step 95000)
  if [ "$ARM" = cal ]; then echo "$TR/lever_r84_calib/q${K}_dnpool$([ $TS = 0 ] || echo _s$TS)/checkpoints/095000/pretrained_model"
  elif [ "$TS" = 0 ] && [ "$K" = 1 ]; then echo "$TR/${BASE}_d100_03dagcap_r84_ft_dag1/checkpoints/095000/pretrained_model"
  else echo "$TR/lever_r84_fb/hg${K}$([ $TS = 0 ] || echo _s$TS)/checkpoints/095000/pretrained_model"; fi; }
cd /home/jennyw2/code/SplatSim; python -u scripts/launch_nodes.py --robot sim_ur_pybullet_small_engine_new_interactive --robot_port $PORT --hostname 127.0.0.1 \
  --eval_benchmark_repo_id JennyWWW/eval_splatsim_approach_lever_13_benchmark --headless --control_gui --render_mode splat > "$S/r84_lane${LANE}_sim.log" 2>&1 & SIM=$!; trap 'kill $SIM 2>/dev/null' EXIT
for i in $(seq 1 150); do (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null && break; sleep 2; done; sleep 5; cd "$LR"
SUBSET100=$($PY -c "print('['+','.join(str(i) for i in range(100))+']')")
eval_one() { local NAME=$1 CK=$2 SD=$3; local D="$LR/outputs/eval300/lever_cam/$NAME"
  [ -f "$D/eval_info.json" ] && { log "$NAME done"; return; }
  for i in $(seq 1 60); do [ -d "$CK" ] && break; sleep 30; done; [ -d "$CK" ] || { log "$NAME: no ckpt $CK"; return; }
  rm -rf "$D"; mkdir -p "$D"; log "eval100 $NAME (seed $SD)"
  $EV --env.type=splatsim --env.task=upright_small_engine_new --env.camera_names='["base_rgb","wrist_rgb"]' --env.image_resize_modes='["stretch"]' \
    --env.fps=30 --env.eval_benchmark_repo_id=JennyWWW/eval_splatsim_approach_lever_13_benchmark --policy.path="$CK" --eval.n_episodes=100 --output_dir="$D" \
    --eval.batch_size=1 --eval.use_async_envs=false --seed=$SD --rename_map='{}' --env.episode_length=1000 --env.eval_benchmark_subset="$SUBSET100" \
    --env.external_port=$PORT --env.max_parallel_tasks=1 --env.robot_name=robot_iphone_w_engine_curtain --env.cam_i=3 --env.use_gripper=true \
    --env.debug_mode=off --env.headless=true --env.control_gui=true --env.include_oracle_info=false --env.terminate_on_collision=true --env.wrist_cam_ver=2 \
    --env.teleop_pad_short_episodes=true --env.teleop_state_jump_split_threshold_rad=0.15 --env.num_dofs=6 --env.state_dim=7 --env.action_dim=7 --env.env_state_dim=0 > "$D/log.txt" 2>&1
  local RC=$?; local SUCC=$($PY -c "import json;print(json.load(open('$D/eval_info.json'))['overall']['pc_success'])" 2>/dev/null); log "$NAME rc=$RC succ=${SUCC:-NA}"; }
# job list: round 1 first; within a round, first extra eval seed for every checkpoint, then the second
JOBS=()
for K in 2 1; do for PASS in 1 2; do for TS in 0 2 1; do for ARM in cal hg; do
  SDS=$(for s in 0 1 2; do [ $s != $TS ] && echo $s; done | sed -n ${PASS}p)
  JOBS+=("r84_${ARM}${K}_20k$([ $TS = 0 ] || echo _s$TS)_e$SDS|$(ckpt $K $TS $ARM)|$SDS")
done; done; done; done
n=0; for J in "${JOBS[@]}"; do if [ $((n % NL)) = $LANE ]; then IFS='|' read -r NAME CK SD <<< "$J"; eval_one "$NAME" "$CK" "$SD"; fi; n=$((n+1)); done
log "LANE${LANE}_DONE"
