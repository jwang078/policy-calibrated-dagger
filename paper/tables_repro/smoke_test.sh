#!/bin/bash
# Smoke test for the reproduction recipes: runs every command SHAPE the two
# tables depend on, but short (+20 training steps, 2 eval episodes), into
# lerobot/outputs/smoke/ and a scratch copy of this directory. Needs the GPU,
# the datasets in ~/.cache/huggingface/lerobot/JennyWWW and the base
# checkpoints under lerobot/outputs/training; nothing under this directory
# is modified. ~10 minutes. Usage:
#
#   bash smoke_test.sh            # everything
#   PARTS="train eval" bash smoke_test.sh
#
# Parts: train (planar q / dn4 / pooled / per-step arms, lever HG / calibrated
# arms), eval (planar + lever lerobot-eval against a fresh sim node), analysis
# (planar sigma measurement + schedule builders, lever W fit + schedule
# builder; rebuilt schedules are compared with the shipped ones), table (both
# table generators). Last run 2026-09-17: all parts passed, see README.
set -u
S=$(cd "$(dirname "$0")" && pwd); LR=/home/jennyw2/code/lerobot; PY=$HOME/miniforge3/envs/splatsim/bin/python; EV=$HOME/miniforge3/envs/splatsim/bin/lerobot-eval
PARTS=${PARTS:-"train eval analysis table"}
OUT=$LR/outputs/smoke; SCRATCH=$OUT/tables_repro_scratch; mkdir -p "$OUT" "$SCRATCH/analysis"
SW='--dataset.dart_mask_hold_tail=true --dataset.dart_selective_mask=true --policy.do_mask_loss_for_padding=true'
log() { echo "[smoke $(date +%H:%M:%S)] $*"; }
FAIL=0; check() { if [ "$1" = 0 ]; then log "PASS $2"; else log "FAIL $2 (rc=$1)"; FAIL=1; fi; }
cd "$LR"

# ---------------------------------------------------------------- train ----
train_one() {  # NAME BASE_CKPT extra-args...
  local NAME=$1 BASE=$2; shift 2; local DIR=$OUT/$NAME; rm -rf "$DIR"
  bash my_scripts/resume_training.sh "$BASE" "$@" --steps=75020 --eval_freq=0 --env_eval_freq=0 --save_freq=75020 \
    --output_dir="$DIR" --job_name=smoke_$NAME --policy.repo_id=smoke_$NAME --policy.push_to_hub=false --wandb.enable=false > "$DIR.log" 2>&1
  local RC=$?; [ -d "$DIR/checkpoints/075020/pretrained_model" ] || RC=$((RC+100)); check $RC "train $NAME"; }
if [[ " $PARTS " == *" train "* ]]; then
  PB=$LR/outputs/training/diffusion_planar_3joint_12_delta_stateng/checkpoints/last/pretrained_model; K=2; TAG=05dag
  REPOS=$($PY -c "import json;print(json.dumps(['JennyWWW/planar_3joint_12']+[f'JennyWWW/planar_12_${TAG}_diff_r_dag{i}' for i in range(1,$K+1)]))")
  STATS=$($PY -c "import json;print(json.dumps(['outputs/dataset_stats/planar_3joint_12/stats_rel64.json']+[f'outputs/dataset_stats/planar_12_${TAG}_diff_r_dag{i}/stats_rel64.json' for i in range(1,$K+1)]))")
  WTS=$($PY -c "import json;print(json.dumps([0.7]+[round(0.3/$K,6)]*$K))")
  COMMON=(--optimizer.lr=1e-5 --num_workers=4 --dataset.multi_source_feature_intersection=true --dataset.repo_id= --dataset.repo_ids="$REPOS" --dataset.sample_weights="$WTS" --dataset.stats_paths="$STATS" --dataset.norm_mode=aggregated --dataset.stats_path= --dataset.use_weighted_sampling=true --eval.n_episodes=100)
  DART=(--dataset.dart_relabel=true '--dataset.dart_self_relabel_pattern=_r_dag\d+$' --dataset.dart_vel_noise_std=0.3 $SW)
  train_one planar_q2         "$PB" "${COMMON[@]}" --dataset.dart_relabel=false
  train_one planar_q2_dn4swv  "$PB" "${COMMON[@]}" "${DART[@]}" --dataset.dart_state_noise_std=4
  train_one planar_q2_dnpool  "$PB" "${COMMON[@]}" "${DART[@]}" --dataset.dart_state_noise_std=0 --dataset.dart_state_noise_schedule=$S/analysis/noise_schedule_pooled_s1_K2.json --dataset.dart_state_noise_scale=1.0
  train_one planar_q2_dnsig   "$PB" "${COMMON[@]}" "${DART[@]}" --dataset.dart_state_noise_std=0 --dataset.dart_state_noise_schedule=$S/analysis/noise_schedule_sigma_alpha_s1_K2.json --dataset.dart_state_noise_scale=1.0
  RES=84; LB=$LR/outputs/training/diffusion_approach_lever_13_smooth_r${RES}_delta_basewrist/checkpoints/last/pretrained_model; PFX=lever_d100_03dagcap_r${RES}_diff; C=$HOME/.cache/huggingface/lerobot/JennyWWW
  WTS=$($PY -c "
import json; f=[json.load(open(f'$C/${PFX}_r_dag{i}/meta/info.json'))['total_frames'] for i in range(1,$K+1)]; s=sum(f)
print('[0.7, '+', '.join(str(round(0.3*x/s,9)) for x in f)+']')")
  REPOS=$($PY -c "import json;print(json.dumps(['JennyWWW/splatsim_approach_lever_13_smooth_r${RES}']+['JennyWWW/${PFX}_r_dag%d'%i for i in range(1,$K+1)]))")
  STATS=$($PY -c "import json;print(json.dumps(['$LR/outputs/dataset_stats/approach_lever_13_smooth_r${RES}/stats_rel64.json']+['$LR/outputs/dataset_stats/${PFX}_r_dag%d/stats_rel64.json'%i for i in range(1,$K+1)]))")
  LCOMMON=(--optimizer.lr=1e-6 --scheduler.name=constant --num_workers=8 --dataset.multi_source_feature_intersection=true --dataset.repo_id= --dataset.repo_ids="$REPOS" --dataset.sample_weights="$WTS" --dataset.stats_paths="$STATS" --dataset.norm_mode=aggregated --dataset.stats_path= --dataset.use_weighted_sampling=true)
  train_one lever_hg2  "$LB" "${LCOMMON[@]}"
  train_one lever_cal2 "$LB" "${LCOMMON[@]}" "${DART[@]}" --dataset.dart_state_noise_std=0 --dataset.dart_state_noise_schedule=$S/analysis/noise_schedule_pooled_lever_r${RES}_K${K}.json --dataset.dart_state_noise_scale=1.0
fi

# ----------------------------------------------------------------- eval ----
start_node() {  # ROBOT PORT BENCH extra-args... ; echoes pid
  cd /home/jennyw2/code/SplatSim; python -u scripts/launch_nodes.py --robot "$1" --robot_port "$2" --hostname 127.0.0.1 --eval_benchmark_repo_id "$3" --headless --control_gui "${@:4}" > "$OUT/node_$2.log" 2>&1 & echo $!; cd "$LR"; }
wait_port() { for i in $(seq 1 150); do (exec 3<>/dev/tcp/127.0.0.1/$1) 2>/dev/null && { sleep 5; return 0; }; sleep 2; done; return 1; }
if [[ " $PARTS " == *" eval "* ]]; then
  # planar: the shipped q5 checkpoint of lineage s1 (falls back to the smoke arm)
  CK=$LR/outputs/training/scarcity_study/q5/checkpoints/last/pretrained_model; [ -d "$CK" ] || CK=$OUT/planar_q2/checkpoints/075020/pretrained_model
  SIM=$(start_node sim_pybullet_planar_interactive 6023 JennyWWW/eval_planar_3joint_benchmark --robot_name planar_3joint); wait_port 6023
  D=$OUT/eval_planar; rm -rf "$D"; mkdir -p "$D"
  $EV --env.type=splatsim --env.task=planar_3joint --env.camera_names='["base_rgb"]' --env.image_resize_modes='["letterbox"]' \
    --env.fps=30 --env.eval_benchmark_repo_id=JennyWWW/eval_planar_3joint_benchmark --policy.path="$CK" --eval.n_episodes=2 --output_dir="$D" \
    --eval.batch_size=1 --eval.use_async_envs=false --seed=0 --rename_map='{}' --env.episode_length=1000 --env.eval_benchmark_subset="[0,1]" \
    --env.external_port=6023 --env.max_parallel_tasks=1 --env.robot_name=planar_3joint --env.cam_i=3 --env.use_gripper=true \
    --env.debug_mode=off --env.headless=true --env.control_gui=true --env.splat_shadows=false --env.include_oracle_info=false \
    --env.terminate_on_collision=true --env.wrist_cam_ver=2 --env.teleop_pad_short_episodes=true --env.teleop_state_jump_split_threshold_rad=0.15 \
    --env.num_dofs=3 --env.state_dim=4 --env.action_dim=4 --env.env_state_dim=8 > "$D/log.txt" 2>&1
  RC=$?; [ -f "$D/eval_info.json" ] || RC=$((RC+100)); check $RC "eval planar (2 episodes)"; kill $SIM 2>/dev/null; wait $SIM 2>/dev/null
  # lever: the shipped calibrated K=1 checkpoint (falls back to the smoke arm)
  CK=$LR/outputs/training/lever_r84_calib/q1_dnpool/checkpoints/095000/pretrained_model; [ -d "$CK" ] || CK=$OUT/lever_cal2/checkpoints/075020/pretrained_model
  SIM=$(start_node sim_ur_pybullet_small_engine_new_interactive 6045 JennyWWW/eval_splatsim_approach_lever_13_benchmark --render_mode splat); wait_port 6045
  D=$OUT/eval_lever; rm -rf "$D"; mkdir -p "$D"
  $EV --env.type=splatsim --env.task=upright_small_engine_new --env.camera_names='["base_rgb","wrist_rgb"]' --env.image_resize_modes='["stretch"]' \
    --env.fps=30 --env.eval_benchmark_repo_id=JennyWWW/eval_splatsim_approach_lever_13_benchmark --policy.path="$CK" --eval.n_episodes=2 --output_dir="$D" \
    --eval.batch_size=1 --eval.use_async_envs=false --seed=0 --rename_map='{}' --env.episode_length=1000 --env.eval_benchmark_subset="[0,1]" \
    --env.external_port=6045 --env.max_parallel_tasks=1 --env.robot_name=robot_iphone_w_engine_curtain --env.cam_i=3 --env.use_gripper=true \
    --env.debug_mode=off --env.headless=true --env.control_gui=true --env.include_oracle_info=false --env.terminate_on_collision=true --env.wrist_cam_ver=2 \
    --env.teleop_pad_short_episodes=true --env.teleop_state_jump_split_threshold_rad=0.15 --env.num_dofs=6 --env.state_dim=7 --env.action_dim=7 --env.env_state_dim=0 > "$D/log.txt" 2>&1
  RC=$?; [ -f "$D/eval_info.json" ] || RC=$((RC+100)); check $RC "eval lever (2 episodes)"; kill $SIM 2>/dev/null; wait $SIM 2>/dev/null
fi

# ------------------------------------------------------------- analysis ----
# The analysis scripts read/write $TABLES_REPRO_DIR/analysis; point them at a
# scratch copy (delta caches linked in) so the shipped schedules stay intact.
sched_diff() {  # NEW SHIPPED -> prints max |diff| over all rows
  $PY -c "
import json,numpy as np,sys
a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2])); md=0.0
for r in a:
    for e in a[r]:
        x=np.array(a[r][e]); y=np.array(b[r][e]); md=max(md, float(np.max(np.abs(x-y))) if x.shape==y.shape else float('inf'))
print(f'{md:.3g}')" "$1" "$2"; }
if [[ " $PARTS " == *" analysis "* ]]; then
  export TABLES_REPRO_DIR=$SCRATCH; ln -sf "$S"/analysis/*.npz "$SCRATCH/analysis/"
  # planar: rebuild the K=2 schedules of s1 from the cached deltas (deterministic -> identical)
  $PY $S/analysis/build_sigma_schedule_k.py s1 05dag 2 > /dev/null; RC=$?
  D=$(sched_diff $SCRATCH/analysis/noise_schedule_pooled_s1_K2.json $S/analysis/noise_schedule_pooled_s1_K2.json); log "pooled s1 K2 rebuilt, max|diff| vs shipped = $D"
  check $RC "analysis planar build_sigma_schedule_k (K=2)"
  # planar: re-measure round 1 with the q1 policy (stochastic policy samples -> pooled covariance within a few %)
  rm -f $SCRATCH/analysis/sigma_deltas_s1q1_dag1.npz
  $PY $S/analysis/measure_sigma_multi.py s1q1 $LR/outputs/training/scarcity_study/q1/checkpoints/last/pretrained_model 05dag "1" > $SCRATCH/measure_sigma_s1q1.log 2>&1; RC=$?
  [ -f $SCRATCH/analysis/sigma_deltas_s1q1_dag1.npz ] || RC=$((RC+100)); check $RC "analysis planar measure_sigma_multi (s1 K=1)"
  $PY $S/analysis/build_sigma_schedule_k.py s1 05dag 1 > /dev/null && log "pooled s1 K1 re-measured, max|diff| vs shipped = $(sched_diff $SCRATCH/analysis/noise_schedule_pooled_s1_K1.json $S/analysis/noise_schedule_pooled_s1_K1.json) (shipped rows are O(10-30))"
  # lever: W fit from the blend datasets and the K=2 pooled schedule from cached deltas (both deterministic)
  PFX=lever_d100_03dagcap_r84_diff
  INT_PREFIX=$PFX $PY $S/analysis/measure_w_lever.py 1,2 > "$SCRATCH/w_lever_r84_rounds1,2.txt" 2>&1; RC=$?
  W=$(grep -oE "W = [0-9.]+" "$SCRATCH/w_lever_r84_rounds1,2.txt" | head -1 | awk '{print $3}'); WS=$(grep -oE "W = [0-9.]+" "$S/analysis/w_lever_r84_rounds1,2.txt" | head -1 | awk '{print $3}')
  [ "$W" = "$WS" ] || RC=$((RC+100)); check $RC "analysis lever measure_w_lever (W=$W, shipped $WS)"
  INT_PREFIX=$PFX TAGPFX=r84_ $PY $S/analysis/build_pooled_schedule_lever.py 2 $W > /dev/null; RC=$?
  D=$(sched_diff $SCRATCH/analysis/noise_schedule_pooled_lever_r84_K2.json $S/analysis/noise_schedule_pooled_lever_r84_K2.json); [ "$D" = 0 ] || RC=$((RC+100)); check $RC "analysis lever build_pooled_schedule_lever (K=2, max|diff| $D)"
  # lever: re-measure round 1 with the BC policy (stochastic; needs the GPU)
  rm -f $SCRATCH/analysis/sigma_deltas_lever_r84_q1_dag1.npz
  $PY $S/analysis/measure_sigma_lever.py r84_q1_dag1 $LR/outputs/training/diffusion_approach_lever_13_smooth_r84_delta_basewrist/checkpoints/last/pretrained_model ${PFX}_r_dag1 > $SCRATCH/measure_sigma_lever.log 2>&1; RC=$?
  [ -f $SCRATCH/analysis/sigma_deltas_lever_r84_q1_dag1.npz ] || RC=$((RC+100)); check $RC "analysis lever measure_sigma_lever (K=1)"
fi

# ---------------------------------------------------------------- table ----
if [[ " $PARTS " == *" table "* ]]; then
  $PY $S/gen_table.py > /dev/null; RC=$?
  diff -q <(grep -v '^%' $S/table_full_benchmark.tex | sed -n '/begin{tabular}/,/end{tabular}/p') <(grep -v '^%' $S/table1_tabular.tex | sed -n '/begin{tabular}/,/end{tabular}/p') > /dev/null || RC=$((RC+100))
  check $RC "table I gen_table.py (tabular identical to table1_tabular.tex)"
  $PY $S/analysis/lever_table.py > /dev/null; check $? "table II analysis/lever_table.py"
fi
[ $FAIL = 0 ] && log "SMOKE OK" || log "SMOKE FAILED"; exit $FAIL
