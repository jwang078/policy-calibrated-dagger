#!/bin/bash
# Shared pieces of the Table I / Table II reproduction scripts. Source it:
#   S=$(cd "$(dirname "$0")" && pwd); source "$S/lib_repro.sh"
#
# Everything configurable is an environment variable with the value the paper
# used as default. smoke_test.sh overrides a few of them (OUT_TRAIN, OUT_EVAL,
# TABLES_REPRO_DIR, PLANAR_STEPS, LEVER_STEPS, N_EPISODES) to run each recipe
# in miniature; DRY=1 prints the training commands instead of running them.
set -u
S=${S:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
LR=${LR:-/home/jennyw2/code/lerobot}
SPLATSIM=${SPLATSIM:-/home/jennyw2/code/SplatSim}
PY=${PY:-$HOME/miniforge3/envs/splatsim/bin/python}
EV=${EV:-$HOME/miniforge3/envs/splatsim/bin/lerobot-eval}
HF=${HF:-$HOME/.cache/huggingface/lerobot/JennyWWW}
OUT_TRAIN=${OUT_TRAIN:-$LR/outputs/training}      # training dirs (scarcity_study*, lever_r84_*)
OUT_EVAL=${OUT_EVAL:-$LR/outputs/eval300}         # eval dirs (<group>/<arm>[_e<seed>]/eval_info.json)
export TABLES_REPRO_DIR=${TABLES_REPRO_DIR:-$S}   # where analysis/*.py read and write deltas + schedules
A=$TABLES_REPRO_DIR/analysis
DRY=${DRY:-0}
LOG_TAG=${LOG_TAG:-repro}
log() { echo "[$LOG_TAG $(date +%H:%M:%S)] $*"; }

# ---- planar (Table I) --------------------------------------------------------
PLANAR_BASE=${PLANAR_BASE:-$OUT_TRAIN/diffusion_planar_3joint_12_delta_stateng}   # BC, 75k steps on 500 demos
PLANAR_STEPS=${PLANAR_STEPS:-175000}     # every table arm: resume the BC checkpoint to this step count
# Learning rate of the from-base fine-tunes. The scripts that produced Table I
# passed 1e-6, but lerobot's resume discarded the flag until the 2026-09-14
# patch, so the arms actually trained on the BC checkpoint's own cosine
# schedule (1e-5 peak). With current code 1e-5 reproduces that; see README.
PLANAR_LR=${PLANAR_LR:-1e-5}
PLANAR_WORKERS=${PLANAR_WORKERS:-4}
PLANAR_PORT_LINEAGE=6001                 # the DAgger orchestrator's shared sim node
# lineage suffix -> orchestrator run tag and training group
lineage_tag() { case $1 in s1) echo 05dag;; s2) echo 06dag;; s3) echo 07dag;; s4) echo 08dag;; s5) echo 09dag;; cl) echo 10cl;; *) echo "unknown lineage $1" >&2; return 1;; esac; }
lineage_group() { case $1 in s1) echo scarcity_study;; cl) echo scarcity_study_cl;; *) echo scarcity_study_$1;; esac; }
schedule_tag() { case $1 in cl) echo cl1;; *) echo $1;; esac; }   # analysis/noise_schedule_*_<tag>_K<K>.json
# --dataset.repo_ids / stats_paths / sample_weights for base + rounds 1..K of a lineage.
# Interventions get an EXACT 0.3 share of every batch, split equally over the rounds.
planar_multi_args() {  # LINEAGE_TAG K
  local TAG=$1 K=$2
  $PY - "$TAG" "$K" <<'EOF'
import json, sys
tag, K = sys.argv[1], int(sys.argv[2])
repos = ["JennyWWW/planar_3joint_12"] + [f"JennyWWW/planar_12_{tag}_diff_r_dag{i}" for i in range(1, K + 1)]
stats = ["outputs/dataset_stats/planar_3joint_12/stats_rel64.json"] + [f"outputs/dataset_stats/planar_12_{tag}_diff_r_dag{i}/stats_rel64.json" for i in range(1, K + 1)]
wts = [0.7] + [round(0.3 / K, 6)] * K
j = lambda x: json.dumps(x, separators=(",", ":"))  # no spaces: one shell word per flag
print(f"--dataset.repo_ids={j(repos)}")
print(f"--dataset.stats_paths={j(stats)}")
print(f"--dataset.sample_weights={j(wts)}")
EOF
}
# per-round relative-action stats sidecars (the orchestrator writes them; recompute if missing)
planar_ensure_stats() {  # LINEAGE_TAG K
  local TAG=$1 K=$2 i
  for i in $(seq 1 $K); do
    [ -f "$LR/outputs/dataset_stats/planar_12_${TAG}_diff_r_dag$i/stats_rel64.json" ] || \
      (cd "$LR" && bash my_scripts/compute_relative_stats.sh --dataset_repo=JennyWWW/planar_12_${TAG}_diff_r_dag$i --chunk_sizes=64)
  done
}

# ---- lever (Table II) --------------------------------------------------------
RES=${RES:-84}
LEVER_BASE_NAME=diffusion_approach_lever_13_smooth_r${RES}_delta_basewrist
LEVER_BASE=${LEVER_BASE:-$OUT_TRAIN/$LEVER_BASE_NAME}                    # BC, 75k steps, 84 px
LEVER_RUN_TAG=d100_03dagcap_r${RES}                                       # orchestrator run tag
LEVER_PFX=lever_${LEVER_RUN_TAG}_diff                                     # intervention repos: ${LEVER_PFX}_r_dag{K}
LEVER_FT=${LEVER_FT:-20000}                                               # fine-tune length (+20k = step 95000)
LEVER_STEPS=${LEVER_STEPS:-$((75000 + LEVER_FT))}
LEVER_WORKERS=${LEVER_WORKERS:-8}
LEVER_BENCH=JennyWWW/eval_splatsim_approach_lever_13_benchmark
LEVER_HG_DIR=$OUT_TRAIN/lever_r${RES}_fb          # HG-DAgger arms hg{K}[_s{seed}]
LEVER_CAL_DIR=$OUT_TRAIN/lever_r${RES}_calib      # calibrated arms q{K}_dnpool[_s{seed}]
LEVER_EVAL_GROUP=lever_cam                        # $OUT_EVAL/lever_cam/<name>
LEVER_BLEND_RATIOS=${LEVER_BLEND_RATIOS:-"0.1 0.2 0.3 0.5 0.75 0.9"}
# base + rounds 1..K; the 0.3 intervention share is split by frame count
lever_multi_args() {  # K
  $PY - "$1" "$LEVER_PFX" "$RES" "$HF" "$LR" <<'EOF'
import json, sys
K, pfx, res, hf, lr = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
frames = [json.load(open(f"{hf}/{pfx}_r_dag{i}/meta/info.json"))["total_frames"] for i in range(1, K + 1)]
repos = [f"JennyWWW/splatsim_approach_lever_13_smooth_r{res}"] + [f"JennyWWW/{pfx}_r_dag{i}" for i in range(1, K + 1)]
stats = [f"{lr}/outputs/dataset_stats/approach_lever_13_smooth_r{res}/stats_rel64.json"] + [f"{lr}/outputs/dataset_stats/{pfx}_r_dag{i}/stats_rel64.json" for i in range(1, K + 1)]
wts = [0.7] + [round(0.3 * f / sum(frames), 9) for f in frames]
j = lambda x: json.dumps(x, separators=(",", ":"))  # no spaces: one shell word per flag
print(f"--dataset.repo_ids={j(repos)}")
print(f"--dataset.stats_paths={j(stats)}")
print(f"--dataset.sample_weights={j(wts)}")
EOF
}

# ---- DART flags --------------------------------------------------------------
# Common to every noised arm: self-relabel the intervention repos (each episode
# is its own demo), phase-space velocity noise 0.3, hold-tail masking.
DART_COMMON=(--dataset.dart_relabel=true '--dataset.dart_self_relabel_pattern=_r_dag\d+$' --dataset.dart_vel_noise_std=0.3
  --dataset.dart_mask_hold_tail=true --dataset.dart_selective_mask=true --policy.do_mask_loss_for_padding=true)
# fixed isotropic sigma (med-step units):  "${DART_COMMON[@]}" --dataset.dart_state_noise_std=SIGMA
# calibrated schedule:                       "${DART_COMMON[@]}" "${DART_SCHED[@]}" --dataset.dart_state_noise_schedule=FILE
DART_SCHED=(--dataset.dart_state_noise_std=0 --dataset.dart_state_noise_scale=1.0)

# ---- training ----------------------------------------------------------------
MULTI_COMMON=(--dataset.multi_source_feature_intersection=true --dataset.repo_id= --dataset.norm_mode=aggregated --dataset.stats_path= --dataset.use_weighted_sampling=true)
# resume the BC checkpoint into $DIR; skipped when the final checkpoint exists.
# Usage: train_from_base DIR JOB BASE_CKPT_DIR STEPS -- lerobot-train overrides...
train_from_base() {
  local DIR=$1 JOB=$2 BASE=$3 STEPS=$4; shift 4; [ "$1" = -- ] && shift
  local CK; CK=$(printf %06d "$STEPS")
  if [ -d "$DIR/checkpoints/$CK/pretrained_model" ]; then log "$JOB exists ($CK), skip"; return 0; fi
  rm -rf "$DIR"; mkdir -p "$(dirname "$DIR")"; log "training $JOB -> $DIR"
  local DRYARG=(); [ "$DRY" = 1 ] && DRYARG=(--dry-run)
  (cd "$LR" && bash my_scripts/resume_training.sh "$BASE/checkpoints/last/pretrained_model" "$@" --steps="$STEPS" \
     --output_dir="$DIR" --job_name="$JOB" --policy.repo_id="$JOB" --policy.push_to_hub=false "${DRYARG[@]}") > "$DIR.log" 2>&1
  local RC=$?; [ "$DRY" = 1 ] && { grep -A1 '^Command:' "$DIR.log" | tail -1; return $RC; }
  [ -d "$DIR/checkpoints/$CK/pretrained_model" ] || { log "$JOB FAILED (rc=$RC, no checkpoint $CK) — see $DIR.log"; return 1; }
  log "$JOB done"
}
# planar arm: from the BC checkpoint to PLANAR_STEPS, inline 100-episode eval at the end
planar_train() {  # DIR JOB LINEAGE_TAG K -- extra...
  local DIR=$1 JOB=$2 TAG=$3 K=$4; shift 4; [ "${1:-}" = -- ] && shift
  local EF=$PLANAR_STEPS; [ "${PLANAR_STEPS}" != 175000 ] && EF=0    # smoke: no inline eval
  local MA; mapfile -t MA < <(planar_multi_args "$TAG" "$K")
  train_from_base "$DIR" "$JOB" "$PLANAR_BASE" "$PLANAR_STEPS" -- --optimizer.lr=$PLANAR_LR --num_workers=$PLANAR_WORKERS \
    "${MULTI_COMMON[@]}" "${MA[@]}" --eval_freq=$EF --save_freq=$PLANAR_STEPS --eval.n_episodes=100 --wandb.enable=false "$@"
}
# lever arm: from the BC checkpoint, constant lr 1e-6, no inline eval, checkpoint every 20k
lever_train() {  # DIR JOB K -- extra...
  local DIR=$1 JOB=$2 K=$3; shift 3; [ "${1:-}" = -- ] && shift
  local MA; mapfile -t MA < <(lever_multi_args "$K")
  train_from_base "$DIR" "$JOB" "${LEVER_TRAIN_BASE:-$LEVER_BASE}" "$LEVER_STEPS" -- --optimizer.lr=1e-6 --scheduler.name=constant --num_workers=$LEVER_WORKERS \
    "${MULTI_COMMON[@]}" "${MA[@]}" --eval_freq=0 --env_eval_freq=0 --save_freq=20000 --wandb.enable=false "$@"
}

# ---- sim nodes ---------------------------------------------------------------
# Every eval / collection lane needs its own node on its own port. Ports used:
# planar lineages 6001, planar evals 6023-6031, lever blends 6042/6043, lever evals 6045-6049.
wait_port() { local i; for i in $(seq 1 150); do (exec 3<>/dev/tcp/127.0.0.1/$1) 2>/dev/null && { sleep 5; return 0; }; sleep 2; done; log "port $1 never came up"; return 1; }
# (the python must be backgrounded as a bare simple command, not inside a `cd && …` list: a backgrounded
# list runs in a wrapper subshell that keeps the $(...) pipe open and the caller never returns)
start_planar_node() {  # PORT -> pid on stdout
  ( cd "$SPLATSIM"; python -u scripts/launch_nodes.py --robot sim_pybullet_planar_interactive --robot_port "$1" --hostname 127.0.0.1 \
      --eval_benchmark_repo_id JennyWWW/eval_planar_3joint_benchmark --robot_name planar_3joint --headless --control_gui > "$OUT_EVAL/node_$1.log" 2>&1 &
    echo $! )
}
start_lever_node() {  # PORT [extra launch_nodes args, e.g. --strict_goal_tolerances for blends] -> pid
  ( cd "$SPLATSIM"; python -u scripts/launch_nodes.py --robot sim_ur_pybullet_small_engine_new_interactive --robot_port "$1" --hostname 127.0.0.1 \
      --eval_benchmark_repo_id $LEVER_BENCH --headless --control_gui --render_mode splat "${@:2}" > "$OUT_EVAL/node_$1.log" 2>&1 &
    echo $! )
}
stop_node() { [ -n "${1:-}" ] && kill "$1" 2>/dev/null; wait "$1" 2>/dev/null; return 0; }

# ---- evaluation --------------------------------------------------------------
N_EPISODES=${N_EPISODES:-300}                     # planar table cells: 300 = 3 passes over the 100 fixed scenarios
SUBSET100=$($PY -c "print('['+','.join(str(i) for i in range(100))+']')")
succ_of() { $PY -c "import json;print(json.load(open('$1/eval_info.json'))['overall']['pc_success'])" 2>/dev/null || echo n/a; }
# eval_planar GROUP NAME CKPT_DIR PORT [SEED] -> $OUT_EVAL/GROUP/NAME[_e<seed>]/eval_info.json
eval_planar() {
  local GROUP=$1 NAME=$2 CK=$3 PORT=$4 SEED=${5:-0}; local D=$OUT_EVAL/$GROUP/$NAME; [ "$SEED" != 0 ] && D=${D}_e$SEED
  [ -f "$D/eval_info.json" ] && { log "$GROUP/$(basename $D) done"; return 0; }
  [ -d "$CK" ] || { log "$GROUP/$NAME: no checkpoint $CK"; return 1; }
  rm -rf "$D"; mkdir -p "$D"; log "eval $N_EPISODES ep $GROUP/$(basename $D) on $PORT"
  (cd "$LR" && $EV --env.type=splatsim --env.task=planar_3joint --env.camera_names='["base_rgb"]' --env.image_resize_modes='["letterbox"]' \
    --env.fps=30 --env.eval_benchmark_repo_id=JennyWWW/eval_planar_3joint_benchmark --policy.path="$CK" --eval.n_episodes=$N_EPISODES --output_dir="$D" \
    --eval.batch_size=1 --eval.use_async_envs=false --seed=$SEED --rename_map='{}' --env.episode_length=1000 --env.eval_benchmark_subset="$SUBSET100" \
    --env.external_port=$PORT --env.max_parallel_tasks=1 --env.robot_name=planar_3joint --env.cam_i=3 --env.use_gripper=true \
    --env.debug_mode=off --env.headless=true --env.control_gui=true --env.splat_shadows=false --env.include_oracle_info=false \
    --env.terminate_on_collision=true --env.wrist_cam_ver=2 --env.teleop_pad_short_episodes=true --env.teleop_state_jump_split_threshold_rad=0.15 \
    --env.num_dofs=3 --env.state_dim=4 --env.action_dim=4 --env.env_state_dim=8) > "$D/log.txt" 2>&1
  log "$GROUP/$(basename $D) rc=$? success=$(succ_of "$D")"
}
# eval_lever NAME CKPT_DIR PORT [SEED] -> $OUT_EVAL/lever_cam/NAME/eval_info.json (NAME carries the seed); LEVER_N episodes (100)
LEVER_N=${LEVER_N:-100}
eval_lever() {
  local NAME=$1 CK=$2 PORT=$3 SEED=${4:-0} N=$LEVER_N; local D=$OUT_EVAL/$LEVER_EVAL_GROUP/$NAME
  [ -f "$D/eval_info.json" ] && { log "$NAME done"; return 0; }
  [ -d "$CK" ] || { log "$NAME: no checkpoint $CK"; return 1; }
  rm -rf "$D"; mkdir -p "$D"; log "eval $N ep $NAME (seed $SEED) on $PORT"
  (cd "$LR" && $EV --env.type=splatsim --env.task=upright_small_engine_new --env.camera_names='["base_rgb","wrist_rgb"]' --env.image_resize_modes='["stretch"]' \
    --env.fps=30 --env.eval_benchmark_repo_id=$LEVER_BENCH --policy.path="$CK" --eval.n_episodes=$N --output_dir="$D" \
    --eval.batch_size=1 --eval.use_async_envs=false --seed=$SEED --rename_map='{}' --env.episode_length=1000 --env.eval_benchmark_subset="$SUBSET100" \
    --env.external_port=$PORT --env.max_parallel_tasks=1 --env.robot_name=robot_iphone_w_engine_curtain --env.cam_i=3 --env.use_gripper=true \
    --env.debug_mode=off --env.headless=true --env.control_gui=true --env.include_oracle_info=false --env.terminate_on_collision=true --env.wrist_cam_ver=2 \
    --env.teleop_pad_short_episodes=true --env.teleop_state_jump_split_threshold_rad=0.15 --env.num_dofs=6 --env.state_dim=7 --env.action_dim=7 --env.env_state_dim=0) > "$D/log.txt" 2>&1
  log "$NAME rc=$? success=$(succ_of "$D")"
}
