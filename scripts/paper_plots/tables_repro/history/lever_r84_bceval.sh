#!/bin/bash
# Small-vision lever pipeline (prepared 2026-09-15 07:30; NOT started).
# Stage 0: BC from scratch on the PRE-SHRUNK 84x84 copy of the base dataset
#          (splatsim_approach_lever_13_smooth_r84, made by analysis/shrink_lever_dataset.py),
#          8 dataloader workers, policy input features 84x84, env renders at 84x84
#          for eval; --policy.resize_shape=[84,84] kept as a no-op safety. Same
#          recipe as the 224 base otherwise (train_sweep diffusion + the orchestrator's
#          round-0 extras: down_dims, obs noise, seed 0, lr 1e-5 cosine, batch 32).
#          Checkpoints at 25k/50k/75k, NO env evals during training (--env_eval_freq=0).
# Stage 1: 100-ep eval of the BC on node 6041 (needs the ext eval lane's node
#          to be gone or a free port; uses PORT below).
# Stage 2 (manual, printed at the end): DAgger round-1 re-collection with the
#          orchestrator, --initial_policy_path = the new BC.
# Gate: waits for $S/LEVER_R84_GO (touch it to start) so it can be armed early.
set -u
S=/home/jennyw2/code/lerobot/my_scripts/paper_plots/tables_repro
LR=/home/jennyw2/code/lerobot; EV=$HOME/miniforge3/envs/splatsim/bin/lerobot-eval
RES=${RES:-84}; STEPS=${STEPS:-75000}; PORT=${PORT:-6043}
RUN=diffusion_approach_lever_13_smooth_r${RES}_delta_basewrist
log() { echo "[r84 $(date +%H:%M:%S)] $*"; }

cd "$LR"
if [ ! -d outputs/training/$RUN/checkpoints/last ]; then
  log "BC $RUN: $STEPS steps at ${RES}px"
  bash my_scripts/train_sweep.sh --model=diffusion --env_profile=small_engine \
    --dataset_repo=JennyWWW/splatsim_approach_lever_13_smooth_r${RES} --cameras=basewrist \
    --run_name=$RUN --num_workers=8 \
    --extra_args="--policy.input_features={\"observation.images.base_rgb\":{\"type\":\"VISUAL\",\"shape\":[3,$RES,$RES]},\"observation.images.wrist_rgb\":{\"type\":\"VISUAL\",\"shape\":[3,$RES,$RES]},\"observation.state\":{\"type\":\"STATE\",\"shape\":[7]}} --policy.resize_shape=[$RES,$RES] --policy.crop_ratio=1.0 --policy.down_dims=[128,256,512] --dataset.observation_noise_std={\"observation.state\":0.01,\"observation.environment_state\":0.005} --seed=0 --policy.normalize_env_state=true --steps=$STEPS --save_freq=25000 --env_eval_freq=0 --wandb.enable=false" \
    > "$S/r${RES}_bc_train.log" 2>&1
  log "BC rc=$?"
fi
CK=$LR/outputs/training/$RUN/checkpoints/last/pretrained_model
[ -d "$CK" ] || { log "no BC checkpoint — abort"; exit 1; }
# ── eval 100 eps (own node) ──
cd /home/jennyw2/code/SplatSim
python -u scripts/launch_nodes.py --robot sim_ur_pybullet_small_engine_new_interactive \
  --robot_port $PORT --hostname 127.0.0.1 \
  --eval_benchmark_repo_id JennyWWW/eval_splatsim_approach_lever_13_benchmark \
  --headless --control_gui --render_mode splat > "$S/r${RES}_sim.log" 2>&1 &
SIM=$!; trap 'kill ${SIM:-} 2>/dev/null' EXIT
for i in $(seq 1 150); do (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null && break; sleep 2; done; sleep 5
cd "$LR"; D="$LR/outputs/eval300/lever_cam/bc_r${RES}"; mkdir -p "$D"
SUBSET100=$(python3 -c "print('['+','.join(str(i) for i in range(100))+']')")
log "eval100 bc_r${RES}"
$EV --env.type=splatsim --env.task=upright_small_engine_new \
  --env.camera_names='["base_rgb","wrist_rgb"]' --env.image_resize_modes='["stretch"]' \
  \
  --env.fps=30 --env.eval_benchmark_repo_id=JennyWWW/eval_splatsim_approach_lever_13_benchmark \
  --policy.path="$CK" --eval.n_episodes=100 --output_dir="$D" \
  --eval.batch_size=1 --eval.use_async_envs=false --seed=0 --rename_map='{}' --env.episode_length=1000 \
  --env.eval_benchmark_subset="$SUBSET100" --env.external_port=$PORT --env.max_parallel_tasks=1 \
  --env.robot_name=robot_iphone_w_engine_curtain --env.cam_i=3 --env.use_gripper=true \
  --env.debug_mode=off --env.headless=true --env.control_gui=true --env.include_oracle_info=false \
  --env.terminate_on_collision=true --env.wrist_cam_ver=2 --env.teleop_pad_short_episodes=true \
  --env.teleop_state_jump_split_threshold_rad=0.15 \
  --env.num_dofs=6 --env.state_dim=7 --env.action_dim=7 --env.env_state_dim=0 > "$D/log.txt" 2>&1
log "bc_r${RES} rc=$? succ=$(python3 -c "
import json
try: print(json.load(open('$D/eval_info.json'))['overall']['pc_success'])
except Exception: print('n/a')")"
touch "$S/LEVER_R84_BC_DONE"; log "LEVER_R84_BC_DONE"
cat <<MSG
# ── Stage 2 (manual): DAgger round 1 with the new BC. Needs a SHARED splat node
#    on port 6001 launched as the orchestrator header describes, then:
cd $LR && bash my_scripts/dagger_orchestrate.sh \$(grep -v '^--resume$' $S/lever_orchestrator_argv.txt | grep -v '^--num_rounds' | tr '\n' ' ') \\
  --num_rounds=1 --run_tag=d100_03dagcap_r${RES} --initial_policy_path=$CK \\
  --finetune_extra_args="--optimizer.lr=1e-6 --dataset.multi_source_feature_intersection=true --scheduler.name=constant --num_workers=4"
MSG
