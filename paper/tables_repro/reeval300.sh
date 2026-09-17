#!/bin/bash
# 300-episode re-evals (3 passes over the 100-scenario benchmark) for every
# arm in the calibrated-noise grids, all three seeds. Own sim lane: port 6023.
set -u
S=/home/jennyw2/code/lerobot/my_scripts/paper_plots/tables_repro
LR=/home/jennyw2/code/lerobot
EV=$HOME/miniforge3/envs/splatsim/bin/lerobot-eval
log() { echo "[re300 $(date +%H:%M:%S)] $*"; }
SUBSET100=$(python3 -c "print('['+','.join(str(i) for i in range(100))+']')")

cd /home/jennyw2/code/SplatSim
python -u scripts/launch_nodes.py --robot sim_pybullet_planar_interactive \
  --robot_port 6023 --hostname 127.0.0.1 \
  --eval_benchmark_repo_id JennyWWW/eval_planar_3joint_benchmark \
  --robot_name planar_3joint --headless --control_gui \
  > "$S/re300_sim.log" 2>&1 &
SIM=$!
trap 'kill ${SIM:-} 2>/dev/null' EXIT
for i in $(seq 1 120); do grep -q "Serving\|serve loop\|Listening\|Robot server" "$S/re300_sim.log" 2>/dev/null && break; sleep 2; done
log "sim node up on 6023"
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
    --env.external_port=6023 --env.max_parallel_tasks=1 \
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

ISO=""
for K in 1 2 3 4 5; do
  ISO="$ISO q$K"
  for STD in 2 4 8 12 16; do ISO="$ISO q${K}_dn${STD}swv"; done
done

# priority 0: BASELINES first, all seeds — every paired-gain cell divides
# through them, so their precision matters most
for K in 1 2 3 4 5; do
  for SD in scarcity_study scarcity_study_s2 scarcity_study_s3; do
    eval_one $SD q$K || true
  done
done

# priority 1: the seeded iso grids, seed 1 then seed 2
for SD in scarcity_study scarcity_study_s2; do
  for A in $ISO; do eval_one $SD $A; done
done

# priority 2: seed-1 extras (dn6, lateral, ellipsoid rows)
for K in 1 2 3 4 5; do
  for SFX in dn6swv dnellswv dnell8swv; do
    eval_one scarcity_study "q${K}_$SFX"
  done
done

# priority 3: seed 3 — poll as the running grid produces checkpoints
for TRY in $(seq 1 400); do
  MISS=0
  for A in $ISO; do
    eval_one scarcity_study_s3 $A || MISS=$((MISS+1))
  done
  [ $MISS -eq 0 ] && break
  grep -q "GRID_s3_DONE\|SEEDS_DONE" $S/seed_master.log 2>/dev/null && [ $MISS -gt 0 ] && SGD=1 || SGD=0
  [ "$SGD" = 1 ] && { log "s3 grid done, $MISS arms missing ckpts — stopping"; break; }
  log "s3: $MISS arms not ready, waiting 10 min"
  sleep 600
done
log "RE300_DONE"
