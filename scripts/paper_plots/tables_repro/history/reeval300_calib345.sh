#!/bin/bash
# 300-episode re-evals for the NEW calibrated arms (dnpool/dnsig, seeds 3-5),
# alternating pooled/sig per K per seed so partial results stay balanced.
# Own sim lane: port 6031. All skip-if-done.
set -u
S=/home/jennyw2/code/lerobot/my_scripts/paper_plots/tables_repro
LR=/home/jennyw2/code/lerobot
EV=$HOME/miniforge3/envs/splatsim/bin/lerobot-eval
log() { echo "[recal345 $(date +%H:%M:%S)] $*"; }
SUBSET100=$(python3 -c "print('['+','.join(str(i) for i in range(100))+']')")

cd /home/jennyw2/code/SplatSim
python -u scripts/launch_nodes.py --robot sim_pybullet_planar_interactive \
  --robot_port 6031 --hostname 127.0.0.1 \
  --eval_benchmark_repo_id JennyWWW/eval_planar_3joint_benchmark \
  --robot_name planar_3joint --headless --control_gui \
  > "$S/recal345_sim.log" 2>&1 &
SIM=$!
trap 'kill ${SIM:-} 2>/dev/null' EXIT
for i in $(seq 1 120); do grep -q "Serving\|serve loop\|Listening\|Robot server" "$S/recal345_sim.log" 2>/dev/null && break; sleep 2; done
log "sim node up on 6031"
cd "$LR"

eval_one() {
  local SD=$1 NAME=$2
  local CK="$LR/outputs/training/$SD/$NAME/checkpoints/last/pretrained_model"
  local D="$LR/outputs/eval300/$SD/$NAME"
  [ -f "$D/eval_info.json" ] && return 0
  [ -d "$CK" ] || return 1
  mkdir -p "$D"
  log "eval300 $SD/$NAME"
  $EV \
    --env.type=splatsim --env.task=planar_3joint \
    --env.camera_names='["base_rgb"]' --env.image_resize_modes='["letterbox"]' \
    --env.fps=30 --env.eval_benchmark_repo_id=JennyWWW/eval_planar_3joint_benchmark \
    --policy.path="$CK" \
    --eval.n_episodes=300 --output_dir="$D" \
    --eval.batch_size=1 --eval.use_async_envs=false \
    --seed=0 --rename_map='{}' --env.episode_length=1000 \
    --env.eval_benchmark_subset="$SUBSET100" \
    --env.external_port=6031 --env.max_parallel_tasks=1 \
    --env.robot_name=planar_3joint --env.cam_i=3 --env.use_gripper=true \
    --env.debug_mode=off --env.headless=true --env.control_gui=true \
    --env.splat_shadows=false --env.include_oracle_info=false \
    --env.terminate_on_collision=true --env.wrist_cam_ver=2 \
    --env.teleop_pad_short_episodes=true \
    --env.teleop_state_jump_split_threshold_rad=0.15 \
    --env.num_dofs=3 --env.state_dim=4 --env.action_dim=4 --env.env_state_dim=8 \
    > "$D/log.txt" 2>&1
  log "$SD/$NAME rc=$? succ=$(python3 -c "
import json
try: print(json.load(open('$D/eval_info.json'))['overall']['pc_success'])
except Exception: print('n/a')")"
}

for K in 1 2 3 4 5; do
  for SD in scarcity_study_s3 scarcity_study_s4 scarcity_study_s5; do
    eval_one $SD q${K}_dnpool || true
    eval_one $SD q${K}_dnsig || true
  done
done
log "RECAL345_DONE"
