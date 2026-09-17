#!/bin/bash
# r84 lineage: W from the round-1 blends -> Sigma-hat (BC on the round-1
# interventions) -> pooled K=1 schedule -> calibrated r1 arm (+40k from the BC, ckpt at +20k too,
# 0.7 base / 0.3 interventions, DART-wrapped, constant lr 1e-6) -> 100-ep evals
# of BOTH r1 arms under the standalone protocol (node 6043, 84px render).
set -u
S=/home/jennyw2/code/lerobot/my_scripts/paper_plots/tables_repro
LR=/home/jennyw2/code/lerobot; TR=$LR/outputs/training; PY=/home/jennyw2/miniforge3/envs/splatsim/bin/python
EV=$HOME/miniforge3/envs/splatsim/bin/lerobot-eval; RES=84; PORT=6043
BASE=diffusion_approach_lever_13_smooth_r${RES}_delta_basewrist; BC=$TR/$BASE/checkpoints/last/pretrained_model
SW='--dataset.dart_mask_hold_tail=true --dataset.dart_selective_mask=true --policy.do_mask_loss_for_padding=true'
log() { echo "[r84cal $(date +%H:%M:%S)] $*"; }
until [ -f "$S/LEVER_R84_BLENDS_DONE" ]; do sleep 60; done
SHORT=$(cat $S/LEVER_R84_INT_SHORT); PFX=${SHORT%_r_dag1}
cd "$LR"
INT_PREFIX=$PFX $PY $S/analysis/measure_w_lever.py 1 > $S/analysis/w_lever_r${RES}_rounds1.txt 2>&1
W=$(grep -oE "W = [0-9.]+" $S/analysis/w_lever_r${RES}_rounds1.txt | head -1 | awk '{print $3}')
log "W(round 1) = '$W'"; grep -E "dag|W =|CI" $S/analysis/w_lever_r${RES}_rounds1.txt
[ "$($PY -c "w=float('${W:-nan}'); print('yes' if 3<=w<=40 else 'no')")" = yes ] || { log "ABORT: W out of range"; exit 1; }
$PY $S/analysis/measure_sigma_lever.py r${RES}_q1_dag1 "$BC" "$SHORT" > $S/analysis/sigma_r${RES}_q1_dag1.log 2>&1; log "sigma rc=$?"
INT_PREFIX=$PFX TAGPFX=r${RES}_ $PY $S/analysis/build_pooled_schedule_lever.py 1 $W | tee -a $S/analysis/sigma_r${RES}_q1_dag1.log
SCHED=$S/analysis/noise_schedule_pooled_lever_r${RES}_K1.json; [ -f "$SCHED" ] || { log "ABORT: no schedule"; exit 1; }
until [ -f "$S/LEVER_R84_HG1_DONE" ]; do sleep 60; done     # GPU: wait for the HG finetune
DIR=$TR/lever_r${RES}_calib/q1_dnpool; mkdir -p $TR/lever_r${RES}_calib
if [ ! -d "$DIR/checkpoints/last" ]; then
  rm -rf "$DIR"; log "training calibrated r1 -> $DIR"
  bash my_scripts/resume_training.sh "$BC" --optimizer.lr=1e-6 --scheduler.name=constant --num_workers=8 \
    --dataset.multi_source_feature_intersection=true --dataset.repo_id= \
    --dataset.repo_ids="[\"JennyWWW/splatsim_approach_lever_13_smooth_r${RES}\", \"JennyWWW/$SHORT\"]" --dataset.sample_weights="[0.7, 0.3]" \
    --dataset.stats_paths="[\"$LR/outputs/dataset_stats/approach_lever_13_smooth_r${RES}/stats_rel64.json\", \"$LR/outputs/dataset_stats/$SHORT/stats_rel64.json\"]" \
    --dataset.norm_mode=aggregated --dataset.stats_path= --dataset.use_weighted_sampling=true \
    --dataset.dart_relabel=true '--dataset.dart_self_relabel_pattern=_r_dag\d+$' --dataset.dart_state_noise_std=0 \
    --dataset.dart_state_noise_schedule=$SCHED --dataset.dart_state_noise_scale=1.0 --dataset.dart_vel_noise_std=0.3 $SW \
    --steps=115000 --eval_freq=0 --env_eval_freq=0 --save_freq=20000 \
    --output_dir="$DIR" --job_name=lever_r${RES}_cal1 --policy.repo_id=lever_r${RES}_cal1 --policy.push_to_hub=false --wandb.enable=false > "$DIR.log" 2>&1
  log "cal1 rc=$?"
fi
# ── standalone 100-ep evals on TWO nodes (6043: cal1 +40k then cal1 +20k; 6044: hg1 +40k) ──
start_node() {  # PORT -> pid
  cd /home/jennyw2/code/SplatSim
  python -u scripts/launch_nodes.py --robot sim_ur_pybullet_small_engine_new_interactive --robot_port $1 --hostname 127.0.0.1 \
    --eval_benchmark_repo_id JennyWWW/eval_splatsim_approach_lever_13_benchmark --headless --control_gui --render_mode splat > "$S/r${RES}_caleval_sim_$1.log" 2>&1 &
  echo $!
}
SIM1=$(start_node 6043); SIM2=$(start_node 6044); trap 'kill ${SIM1:-} ${SIM2:-} 2>/dev/null' EXIT
for P in 6043 6044; do for i in $(seq 1 150); do (exec 3<>/dev/tcp/127.0.0.1/$P) 2>/dev/null && break; sleep 2; done; done; sleep 5
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
( eval_one r${RES}_cal1 "$DIR/checkpoints/last/pretrained_model" 6043; eval_one r${RES}_cal1_20k "$DIR/checkpoints/095000/pretrained_model" 6043 ) & PA=$!
( eval_one r${RES}_hg1 "$TR/${BASE}_d100_03dagcap_r${RES}_ft_dag1/checkpoints/last/pretrained_model" 6044 ) & PB=$!
wait $PA $PB
touch "$S/LEVER_R84_CAL1_DONE"; log "LEVER_R84_CAL1_DONE"
