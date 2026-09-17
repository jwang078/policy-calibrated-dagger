#!/bin/bash
# Seed replication of lever 84px rounds 1 and 2 (from-base protocol, +20k, same datasets & schedules as seed 0).
# Each seed gets its own copy of the BC 75k checkpoint whose training_state/rng_state.safetensors is re-seeded,
# because lerobot resume restores the checkpoint RNG *after* set_seed (so --seed alone would replicate seed 0).
set -u
S=/home/jennyw2/code/lerobot/my_scripts/paper_plots/tables_repro
LR=/home/jennyw2/code/lerobot; TR=$LR/outputs/training; PY=/home/jennyw2/miniforge3/envs/splatsim/bin/python
EV=$HOME/miniforge3/envs/splatsim/bin/lerobot-eval; RES=84
BASE=diffusion_approach_lever_13_smooth_r${RES}_delta_basewrist; PFX=lever_d100_03dagcap_r${RES}_diff
SEEDS=${SEEDS:-"1 2"}; ROUNDS=${ROUNDS:-"1 2"}; CKSTEP=95000
SW='--dataset.dart_mask_hold_tail=true --dataset.dart_selective_mask=true --policy.do_mask_loss_for_padding=true'
log() { echo "[r84 seeds $(date +%H:%M:%S)] $*"; }
cd "$LR"
weights() { $PY -c "
import json; C='/home/jennyw2/.cache/huggingface/lerobot/JennyWWW'; K=$1
f=[json.load(open(f'{C}/${PFX}_r_dag{i}/meta/info.json'))['total_frames'] for i in range(1,K+1)]; s=sum(f)
print('[0.7, '+', '.join(str(round(0.3*x/s,9)) for x in f)+']')"; }
repos()   { $PY -c "import json;print(json.dumps(['JennyWWW/splatsim_approach_lever_13_smooth_r${RES}']+['JennyWWW/${PFX}_r_dag%d'%i for i in range(1,$1+1)]))"; }
stats()   { $PY -c "import json;print(json.dumps(['$LR/outputs/dataset_stats/approach_lever_13_smooth_r${RES}/stats_rel64.json']+['$LR/outputs/dataset_stats/${PFX}_r_dag%d/stats_rel64.json'%i for i in range(1,$1+1)]))"; }
make_seed_base() { local SD=$1 D=$TR/${BASE}_seed$SD/checkpoints/075000
  [ -f "$D/training_state/rng_state.safetensors" ] && { echo "$D/pretrained_model"; return; }
  mkdir -p "$D"; cp -al "$TR/$BASE/checkpoints/075000/pretrained_model" "$D/pretrained_model"
  mkdir -p "$D/training_state"; for f in optimizer_param_groups.json optimizer_state.safetensors scheduler_state.json training_step.json; do ln "$TR/$BASE/checkpoints/075000/training_state/$f" "$D/training_state/$f"; done
  $PY - <<PYEOF
import sys; sys.path.insert(0, "$LR/src")
import random, numpy as np, torch
from safetensors.torch import save_file
from lerobot.utils.random_utils import serialize_rng_state
random.seed($SD); np.random.seed($SD); torch.manual_seed($SD); torch.cuda.manual_seed_all($SD)
save_file(serialize_rng_state(), "$D/training_state/rng_state.safetensors"); print("rng reseeded", $SD)
PYEOF
  echo "$D/pretrained_model"; }
start_node() { cd /home/jennyw2/code/SplatSim; python -u scripts/launch_nodes.py --robot sim_ur_pybullet_small_engine_new_interactive --robot_port $1 --hostname 127.0.0.1 \
    --eval_benchmark_repo_id JennyWWW/eval_splatsim_approach_lever_13_benchmark --headless --control_gui --render_mode splat > "$S/r${RES}_seeds_sim_$1.log" 2>&1 & echo $!; }
SUBSET100=$($PY -c "print('['+','.join(str(i) for i in range(100))+']')")
eval_one() { local NAME=$1 CK=$2 PORT=$3 SD=$4; local D="$LR/outputs/eval300/lever_cam/$NAME"
  [ -f "$D/eval_info.json" ] && { log "$NAME done"; return; }; [ -d "$CK" ] || { log "$NAME: no ckpt $CK"; return; }; mkdir -p "$D"; log "eval100 $NAME on $PORT"
  cd "$LR"; $EV --env.type=splatsim --env.task=upright_small_engine_new --env.camera_names='["base_rgb","wrist_rgb"]' --env.image_resize_modes='["stretch"]' \
    --env.fps=30 --env.eval_benchmark_repo_id=JennyWWW/eval_splatsim_approach_lever_13_benchmark --policy.path="$CK" --eval.n_episodes=100 --output_dir="$D" \
    --eval.batch_size=1 --eval.use_async_envs=false --seed=$SD --rename_map='{}' --env.episode_length=1000 --env.eval_benchmark_subset="$SUBSET100" \
    --env.external_port=$PORT --env.max_parallel_tasks=1 --env.robot_name=robot_iphone_w_engine_curtain --env.cam_i=3 --env.use_gripper=true \
    --env.debug_mode=off --env.headless=true --env.control_gui=true --env.include_oracle_info=false --env.terminate_on_collision=true --env.wrist_cam_ver=2 \
    --env.teleop_pad_short_episodes=true --env.teleop_state_jump_split_threshold_rad=0.15 --env.num_dofs=6 --env.state_dim=7 --env.action_dim=7 --env.env_state_dim=0 > "$D/log.txt" 2>&1
  local RC=$?; local SUCC=$($PY -c "import json;print(json.load(open('$D/eval_info.json'))['overall']['pc_success'])" 2>/dev/null); log "$NAME rc=$RC succ=${SUCC:-NA}"; }
SIM1=$(start_node 6045); SIM2=$(start_node 6046); trap 'kill ${SIM1:-} ${SIM2:-} 2>/dev/null' EXIT
for PT in 6045 6046; do for i in $(seq 1 150); do (exec 3<>/dev/tcp/127.0.0.1/$PT) 2>/dev/null && break; sleep 2; done; done; sleep 5; cd "$LR"
EP1=""; EP2=""
for SD in $SEEDS; do
  BC=$(make_seed_base $SD | tail -1); log "seed $SD base ckpt $BC"
  for K in $ROUNDS; do
    REPOS=$(repos $K); WTS=$(weights $K); STATS=$(stats $K); SCHED=$S/analysis/noise_schedule_pooled_lever_r${RES}_K${K}.json
    DIRH=$TR/lever_r${RES}_fb/hg${K}_s$SD
    if [ ! -d "$DIRH/checkpoints/0$CKSTEP" ]; then rm -rf "$DIRH"; log "training HG r$K seed $SD"
      bash my_scripts/resume_training.sh "$BC" --optimizer.lr=1e-6 --scheduler.name=constant --num_workers=8 --seed=$SD \
        --dataset.multi_source_feature_intersection=true --dataset.repo_id= --dataset.repo_ids="$REPOS" --dataset.sample_weights="$WTS" --dataset.stats_paths="$STATS" \
        --dataset.norm_mode=aggregated --dataset.stats_path= --dataset.use_weighted_sampling=true --steps=$CKSTEP --eval_freq=0 --env_eval_freq=0 --save_freq=20000 \
        --output_dir="$DIRH" --job_name=lever_r${RES}_hg${K}_s$SD --policy.repo_id=lever_r${RES}_hg${K}_s$SD --policy.push_to_hub=false --wandb.enable=false > "$DIRH.log" 2>&1; log "hg${K}_s$SD rc=$?"
    fi
    DIRC=$TR/lever_r${RES}_calib/q${K}_dnpool_s$SD
    if [ ! -d "$DIRC/checkpoints/0$CKSTEP" ]; then rm -rf "$DIRC"; log "training calibrated r$K seed $SD"
      bash my_scripts/resume_training.sh "$BC" --optimizer.lr=1e-6 --scheduler.name=constant --num_workers=8 --seed=$SD \
        --dataset.multi_source_feature_intersection=true --dataset.repo_id= --dataset.repo_ids="$REPOS" --dataset.sample_weights="$WTS" --dataset.stats_paths="$STATS" \
        --dataset.norm_mode=aggregated --dataset.stats_path= --dataset.use_weighted_sampling=true \
        --dataset.dart_relabel=true '--dataset.dart_self_relabel_pattern=_r_dag\d+$' --dataset.dart_state_noise_std=0 --dataset.dart_state_noise_schedule=$SCHED --dataset.dart_state_noise_scale=1.0 --dataset.dart_vel_noise_std=0.3 $SW \
        --steps=$CKSTEP --eval_freq=0 --env_eval_freq=0 --save_freq=20000 \
        --output_dir="$DIRC" --job_name=lever_r${RES}_cal${K}_s$SD --policy.repo_id=lever_r${RES}_cal${K}_s$SD --policy.push_to_hub=false --wandb.enable=false > "$DIRC.log" 2>&1; log "cal${K}_s$SD rc=$?"
    fi
    [ -n "$EP1" ] && wait $EP1 $EP2   # previous pair's evals must be done before reusing the nodes
    eval_one r${RES}_hg${K}_20k_s$SD "$DIRH/checkpoints/0$CKSTEP/pretrained_model" 6045 $SD & EP1=$!
    eval_one r${RES}_cal${K}_20k_s$SD "$DIRC/checkpoints/0$CKSTEP/pretrained_model" 6046 $SD & EP2=$!
  done
done
wait $EP1 $EP2; log "LEVER_R84_SEEDS_DONE"; touch $S/LEVER_R84_SEEDS_DONE
