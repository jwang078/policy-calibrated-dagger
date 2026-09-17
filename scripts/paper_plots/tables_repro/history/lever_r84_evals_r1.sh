#!/bin/bash
# standalone 100-ep evals of the r84 round-1 arms on two nodes (6045/6046); gated on LEVER_R84_BLENDS2_DONE (RAM).
set -u
S=/home/jennyw2/code/lerobot/my_scripts/paper_plots/tables_repro
LR=/home/jennyw2/code/lerobot; TR=$LR/outputs/training; PY=/home/jennyw2/miniforge3/envs/splatsim/bin/python
EV=$HOME/miniforge3/envs/splatsim/bin/lerobot-eval; RES=84; PORT=6043
BASE=diffusion_approach_lever_13_smooth_r${RES}_delta_basewrist; BC=$TR/$BASE/checkpoints/last/pretrained_model
SW='--dataset.dart_mask_hold_tail=true --dataset.dart_selective_mask=true --policy.do_mask_loss_for_padding=true'
log() { echo "[r84cal $(date +%H:%M:%S)] $*"; }
DIR=$TR/lever_r${RES}_calib/q1_dnpool
# gate removed 13:27: no trainer running, RAM allows the two eval nodes alongside the blend lanes
# ── standalone 100-ep evals on TWO nodes (6045: cal1 +40k then cal1 +20k; 6046: hg1 +40k) ──
start_node() {  # PORT -> pid
  cd /home/jennyw2/code/SplatSim
  python -u scripts/launch_nodes.py --robot sim_ur_pybullet_small_engine_new_interactive --robot_port $1 --hostname 127.0.0.1 \
    --eval_benchmark_repo_id JennyWWW/eval_splatsim_approach_lever_13_benchmark --headless --control_gui --render_mode splat > "$S/r${RES}_caleval_sim_$1.log" 2>&1 &
  echo $!
}
SIM1=$(start_node 6045); SIM2=$(start_node 6046); trap 'kill ${SIM1:-} ${SIM2:-} 2>/dev/null' EXIT
for P in 6045 6046; do for i in $(seq 1 150); do (exec 3<>/dev/tcp/127.0.0.1/$P) 2>/dev/null && break; sleep 2; done; done; sleep 5
cd "$LR"; SUBSET100=$(python3 -c "print('['+','.join(str(i) for i in range(100))+']')")
eval_one() { local NAME=$1 CK=$2 PORT=$3; local D="$LR/outputs/eval300/lever_cam/$NAME"
  [ -f "$D/eval_info.json" ] && { log "$NAME done"; return; }; [ -d "$CK" ] || { log "$NAME: no ckpt $CK"; return; }; mkdir -p "$D"; log "eval100 $NAME on $PORT"
  $EV --env.type=splatsim --env.task=upright_small_engine_new --env.camera_names='["base_rgb","wrist_rgb"]' --env.image_resize_modes='["stretch"]' \
    --env.fps=30 --env.eval_benchmark_repo_id=JennyWWW/eval_splatsim_approach_lever_13_benchmark \
    --policy.path="$CK" --eval.n_episodes=100 --output_dir="$D" --eval.batch_size=1 --eval.use_async_envs=false --seed=0 --rename_map='{}' --env.episode_length=1000 \
    --env.eval_benchmark_subset="$SUBSET100" --env.external_port=$PORT --env.max_parallel_tasks=1 --env.robot_name=robot_iphone_w_engine_curtain --env.cam_i=3 --env.use_gripper=true \
    --env.debug_mode=off --env.headless=true --env.control_gui=true --env.include_oracle_info=false --env.terminate_on_collision=true --env.wrist_cam_ver=2 \
    --env.teleop_pad_short_episodes=true --env.teleop_state_jump_split_threshold_rad=0.15 --env.num_dofs=6 --env.state_dim=7 --env.action_dim=7 --env.env_state_dim=0 > "$D/log.txt" 2>&1
  log "$NAME rc=$? succ=$(python3 -c "
import json
try: print(json.load(open('$D/eval_info.json'))['overall']['pc_success'])
except Exception: print('n/a')")"; }
( eval_one r${RES}_cal1 "$DIR/checkpoints/last/pretrained_model" 6045; eval_one r${RES}_cal1_20k "$DIR/checkpoints/095000/pretrained_model" 6045 ) & PA=$!
( eval_one r${RES}_hg1 "$TR/${BASE}_d100_03dagcap_r${RES}_ft_dag1/checkpoints/last/pretrained_model" 6046 ) & PB=$!
wait $PA $PB
touch "$S/LEVER_R84_EVALS1_DONE"; log "LEVER_R84_EVALS1_DONE"
