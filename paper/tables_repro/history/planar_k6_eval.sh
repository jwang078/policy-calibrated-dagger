#!/bin/bash
# Planar 300-ep eval lane for K=6 arms. usage: planar_k6_eval.sh LANE NLANES PORT ; polls for checkpoints.
set -u
LANE=$1; NL=$2; PORT=$3
S=/home/jennyw2/code/lerobot/my_scripts/paper_plots/tables_repro; LR=/home/jennyw2/code/lerobot; EV=$HOME/miniforge3/envs/splatsim/bin/lerobot-eval
log() { echo "[k6eval$LANE $(date +%H:%M:%S)] $*"; }
SUBSET100=$(python3 -c "print('['+','.join(str(i) for i in range(100))+']')")
cd /home/jennyw2/code/SplatSim
python -u scripts/launch_nodes.py --robot sim_pybullet_planar_interactive --robot_port $PORT --hostname 127.0.0.1 --eval_benchmark_repo_id JennyWWW/eval_planar_3joint_benchmark --robot_name planar_3joint --headless --control_gui > "$S/k6eval${LANE}_sim.log" 2>&1 & SIM=$!; trap 'kill ${SIM:-} 2>/dev/null' EXIT
for i in $(seq 1 120); do grep -q "Serving\|serve loop\|Listening\|Robot server" "$S/k6eval${LANE}_sim.log" 2>/dev/null && break; sleep 2; done; log "sim up on $PORT"; cd "$LR"
eval_one() { local GD=$1 NAME=$2; local CK="$LR/outputs/training/$GD/$NAME/checkpoints/last/pretrained_model"; local D="$LR/outputs/eval300/$GD/$NAME"
  [ -f "$D/eval_info.json" ] && { log "$GD/$NAME done"; return; }
  until [ -d "$CK" ] || [ -f $S/PLANAR_K6_DONE ]; do sleep 60; done; [ -d "$CK" ] || { log "$GD/$NAME: no ckpt"; return; }
  rm -rf "$D"; mkdir -p "$D"; log "eval300 $GD/$NAME"
  $EV --env.type=splatsim --env.task=planar_3joint --env.camera_names='["base_rgb"]' --env.image_resize_modes='["letterbox"]' --env.fps=30 --env.eval_benchmark_repo_id=JennyWWW/eval_planar_3joint_benchmark \
    --policy.path="$CK" --eval.n_episodes=300 --output_dir="$D" --eval.batch_size=1 --eval.use_async_envs=false --seed=0 --rename_map='{}' --env.episode_length=1000 --env.eval_benchmark_subset="$SUBSET100" \
    --env.external_port=$PORT --env.max_parallel_tasks=1 --env.robot_name=planar_3joint --env.cam_i=3 --env.use_gripper=true --env.debug_mode=off --env.headless=true --env.control_gui=true --env.splat_shadows=false \
    --env.include_oracle_info=false --env.terminate_on_collision=true --env.wrist_cam_ver=2 --env.teleop_pad_short_episodes=true --env.teleop_state_jump_split_threshold_rad=0.15 --env.num_dofs=3 --env.state_dim=4 --env.action_dim=4 --env.env_state_dim=8 > "$D/log.txt" 2>&1
  local RC=$?; local SUCC=$(python3 -c "import json;print(json.load(open('$D/eval_info.json'))['overall']['pc_success'])" 2>/dev/null); log "$GD/$NAME rc=$RC succ=${SUCC:-NA}"; }
GDS="scarcity_study scarcity_study_s2 scarcity_study_s3 scarcity_study_s4 scarcity_study_s5"
JOBS=(); for NAME in q6 q6_dnpool q6_dnsig q6_dn4swv q6_dn2swv q6_dn8swv q6_dn12swv q6_dn16swv; do for GD in $GDS; do JOBS+=("$GD $NAME"); done; done
n=0; for J in "${JOBS[@]}"; do [ $((n % NL)) = $LANE ] && eval_one $J; n=$((n+1)); done
log "EVAL_LANE${LANE}_DONE"
