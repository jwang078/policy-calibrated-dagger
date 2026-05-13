#!/usr/bin/env bash
set -euo pipefail
# train_sweep.sh
#
# Runs one or more lerobot-train jobs against a dataset.  Optionally loops
# over cumulative augmented-ratio subsets (RATIO_SWEEP) created by
# augment_ratios_sweep.sh.
#
# Usage:
#   bash my_scripts/train_sweep.sh [OPTIONS]
#
# All options have defaults (see USER CONFIG below).
#
# Options:
#   --dataset_short=STR     Short dataset name (without the "splatsim_" prefix)
#   --ratio_sweep           Enable the augmented-ratio sweep
#   --ratios="N N N"        Space-separated ratio list (used when --ratio_sweep)
#   --no_relative           Disable relative-action training (default: enabled)
#   --dry-run               Print commands without executing
#
# Example:
#   bash my_scripts/train_sweep.sh \
#       --dataset_short=approach_lever_11_50failsrrtpi05 \
#       --ratio_sweep \
#       --ratios="0.2 0.4 0.6 0.8 1.0"

# ── USER CONFIG (defaults) ────────────────────────────────────────────────────
DATASET_SHORT="approach_lever_11_50failsrrtpi05"
USE_RELATIVE_ACTIONS=true
RATIO_SWEEP=false
RATIOS=(0.2 0.4 0.6 0.8 1.0)
DRY_RUN=false
# ─────────────────────────────────────────────────────────────────────────────

for arg in "$@"; do
    case "$arg" in
        --dry-run)          DRY_RUN=true ;;
        --ratio_sweep)      RATIO_SWEEP=true ;;
        --no_relative)      USE_RELATIVE_ACTIONS=false ;;
        --dataset_short=*)  DATASET_SHORT="${arg#*=}" ;;
        --ratios=*)         IFS=' ' read -ra RATIOS <<< "${arg#*=}" ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

DATASET_REPO="JennyWWW/splatsim_${DATASET_SHORT}"

# Paths written by compute_relative_stats.sh
STATS_DIR=~/code/lerobot/outputs/dataset_stats/${DATASET_SHORT}
PI05_STATS_PATH="${STATS_DIR}/stats_pi05_rel50.json"
DIFFUSION_STATS_PATH="${STATS_DIR}/stats_diffusion_rel8.json"

# Validate that stats files exist when USE_RELATIVE_ACTIONS=true
if [[ "$USE_RELATIVE_ACTIONS" == true ]]; then
    for f in "$PI05_STATS_PATH" "$DIFFUSION_STATS_PATH"; do
        if [[ ! -f "$f" ]]; then
            echo "ERROR: USE_RELATIVE_ACTIONS=true but stats file not found: $f" >&2
            echo "Run my_scripts/compute_relative_stats.sh first." >&2
            exit 1
        fi
    done
fi

# ── Validate names before doing anything ─────────────────────
validate_names() {
    local errors=0
    local dataset_name="${DATASET_REPO#*/}"   # everything after the first /

    # --- DATASET_SHORT rules ---
    # Allowed characters: alphanumeric, _, -, .
    if [[ "$DATASET_SHORT" =~ [^a-zA-Z0-9_.-] ]]; then
        echo "ERROR: DATASET_SHORT contains invalid characters (only a-z, A-Z, 0-9, _, -, . allowed): '$DATASET_SHORT'" >&2
        errors=1
    fi
    # Cannot start or end with - or .
    if [[ "$DATASET_SHORT" =~ ^[.-] || "$DATASET_SHORT" =~ [.-]$ ]]; then
        echo "ERROR: DATASET_SHORT cannot start or end with '-' or '.': '$DATASET_SHORT'" >&2
        errors=1
    fi
    # Forbidden substrings
    if [[ "$DATASET_SHORT" == *"--"* || "$DATASET_SHORT" == *".."* ]]; then
        echo "ERROR: DATASET_SHORT cannot contain '--' or '..': '$DATASET_SHORT'" >&2
        errors=1
    fi

    # --- dataset_name rules (the part after JennyWWW/) ---
    # Allowed characters: alphanumeric, _, -, .
    if [[ "$dataset_name" =~ [^a-zA-Z0-9_.-] ]]; then
        echo "ERROR: dataset name contains invalid characters (only a-z, A-Z, 0-9, _, -, . allowed): '$dataset_name'" >&2
        errors=1
    fi
    # Cannot start or end with - or .
    if [[ "$dataset_name" =~ ^[.-] || "$dataset_name" =~ [.-]$ ]]; then
        echo "ERROR: dataset name cannot start or end with '-' or '.': '$dataset_name'" >&2
        errors=1
    fi
    # Forbidden substrings: -- and ..
    if [[ "$dataset_name" == *"--"* || "$dataset_name" == *".."* ]]; then
        echo "ERROR: dataset name cannot contain '--' or '..': '$dataset_name'" >&2
        errors=1
    fi
    # Cannot end with .git or .ipynb
    if [[ "$dataset_name" == *.git || "$dataset_name" == *.ipynb ]]; then
        echo "ERROR: dataset name cannot end with '.git' or '.ipynb': '$dataset_name'" >&2
        errors=1
    fi
    # Max length 56 (not including the dataset: prefix that sends it up to 64)
    if (( ${#DATASET_REPO} > 56 )); then
        echo "ERROR: dataset name exceeds 56 chars (${#DATASET_REPO}): '$DATASET_REPO'" >&2
        errors=1
    fi

    if (( errors > 0 )); then
        exit 1
    fi

    echo "Validation passed: DATASET_REPO='$DATASET_REPO' (dataset name: ${#DATASET_REPO}/56 chars)"
}
validate_names

TRAIN_SCRIPT="lerobot-train"  # make sure this is in your PATH (e.g. via lerobot's install.sh)

# ── Shared env/eval args (same for every run) ────────────────
SHARED_ARGS=(
    --wandb.enable=true
    --policy.device=cuda
    --env.type=splatsim
    --env.task=upright_small_engine_new
    --env.fps=30
    --env.eval_benchmark_repo_id=JennyWWW/eval_splatsim_approach_lever_benchmark_1000
    --eval.n_episodes=5
    --eval.batch_size=1
    --eval.use_async_envs=false
    --dataset.image_transforms.enable=true
)

# ── Policy-specific args ─────────────────────────────────────

DIFFUSION_ARGS=(
    --policy.type=diffusion
    --steps=75000
    --batch_size=32
    --eval_freq=25000
    --save_freq=25000
    --policy.vision_backbone=resnet18
    --policy.pretrained_backbone_weights=null
    --policy.use_group_norm=true
    "--policy.crop_shape=[224, 224]"
    --policy.crop_is_random=false
    --policy.optimizer_lr=1e-5
    --policy.use_separate_rgb_encoder_per_camera=true
)
DIFFUSION_RESIZE_MODE="stretch"

PI05_ARGS=(
    --policy.type=pi05
    --steps=6000
    # --steps=3000
    --batch_size=16
    --eval_freq=2000
    # --eval_freq=1000
    --save_freq=2000
    # --save_freq=1000
    --policy.scheduler_decay_steps=6000
    # --policy.scheduler_decay_steps=3000
    --policy.pretrained_path=lerobot/pi05_base
    --policy.compile_model=false
    --policy.gradient_checkpointing=true
    --policy.dtype=bfloat16
    --policy.train_expert_only=true
    --policy.use_amp=true
)
PI05_RESIZE_MODE="letterbox"

# ACT: trains from scratch on top of a pretrained ResNet18 backbone, with
# absolute actions + temporal ensembling (the canonical ACT setup, designed
# to handle chunk-boundary smoothing without needing relative actions).
# n_action_steps=1 is required when temporal_ensemble_coeff is set: the
# policy is queried every step and ensembled predictions are averaged.
ACT_ARGS=(
    --policy.type=act
    --steps=50000
    --batch_size=8
    --eval_freq=10000
    --save_freq=10000
    --policy.vision_backbone=resnet18
    --policy.chunk_size=50
    --policy.n_action_steps=1
    --policy.temporal_ensemble_coeff=0.01
    --policy.optimizer_lr=1e-5
    --policy.optimizer_lr_backbone=1e-5
    # kl_weight default in lerobot is 10, which causes the CVAE to mode-collapse
    # on small/simple datasets (policy outputs the dataset mean for everything).
    # Lowering to 1.0 lets the L1 reconstruction signal dominate.
    --policy.kl_weight=1.0
)
ACT_RESIZE_MODE="letterbox"

# ── Camera-specific args ─────────────────────────────────────
# Sets CAMERA_ARGS array. Call as: set_camera_args <resize_mode> <camera_suffix>
# camera_suffix: "basewrist" | "base" | "wrist"
set_camera_args() {
    local resize_mode=$1
    local camera_suffix=$2

    case "$camera_suffix" in
        basewrist)
            CAMERA_ARGS=(
                "--env.camera_names=[\"base_rgb\", \"wrist_rgb\"]"
                "--env.image_resize_modes=[\"${resize_mode}\"]"
                "--policy.input_features={\"observation.images.base_rgb\": {\"type\": \"VISUAL\", \"shape\": [3, 224, 224]}, \"observation.images.wrist_rgb\": {\"type\": \"VISUAL\", \"shape\": [3, 224, 224]}, \"observation.state\": {\"type\": \"STATE\", \"shape\": [7]}}"
                "--rename_map={\"observation.images.base_rgb_${resize_mode}\": \"observation.images.base_rgb\", \"observation.images.wrist_rgb_${resize_mode}\": \"observation.images.wrist_rgb\"}"
            )
            ;;
        base)
            CAMERA_ARGS=(
                "--env.camera_names=[\"base_rgb\"]"
                "--env.image_resize_modes=[\"${resize_mode}\"]"
                "--policy.input_features={\"observation.images.base_rgb\": {\"type\": \"VISUAL\", \"shape\": [3, 224, 224]}, \"observation.state\": {\"type\": \"STATE\", \"shape\": [7]}}"
                "--rename_map={\"observation.images.base_rgb_${resize_mode}\": \"observation.images.base_rgb\"}"
            )
            ;;
        wrist)
            CAMERA_ARGS=(
                "--env.camera_names=[\"wrist_rgb\"]"
                "--env.image_resize_modes=[\"${resize_mode}\"]"
                "--policy.input_features={\"observation.images.wrist_rgb\": {\"type\": \"VISUAL\", \"shape\": [3, 224, 224]}, \"observation.state\": {\"type\": \"STATE\", \"shape\": [7]}}"
                "--rename_map={\"observation.images.wrist_rgb_${resize_mode}\": \"observation.images.wrist_rgb\"}"
            )
            ;;
    esac
}

# ── Helper to run one training job ───────────────────────────
# run_job <policy_prefix> <camera_suffix> <policy_args_array_name> <resize_mode> [env_prefix] [extra_args_array_name]
run_job() {
    local policy_prefix=$1      # e.g. "diffusion" or "pi05"
    local camera_suffix=$2      # e.g. "basewrist", "base", "wrist"
    local -n policy_args=$3     # nameref to array
    local resize_mode=$4
    local env_prefix="${5:-}"        # optional env var prefix (e.g. PYTORCH_CUDA_ALLOC_CONF=...)
    local extra_args_ref="${6:-}"    # optional nameref to array of extra CLI args (e.g. --batch_size=8)

    local action_suffix
    action_suffix=$([[ "$USE_RELATIVE_ACTIONS" == true ]] && echo "delta" || echo "abs")
    local run_name="${policy_prefix}_${DATASET_SHORT}_${action_suffix}_${camera_suffix}"

    set_camera_args "$resize_mode" "$camera_suffix"

    local full_cmd=(
        $TRAIN_SCRIPT
        --dataset.repo_id="$DATASET_REPO"
        --output_dir="./outputs/training/${run_name}"
        --job_name="${run_name}"
        --policy.repo_id="${run_name}"
        "${SHARED_ARGS[@]}"
        "${policy_args[@]}"
        "${CAMERA_ARGS[@]}"
    )

    # Append relative-action flags if enabled, using the per-policy stats file.
    if [[ "$USE_RELATIVE_ACTIONS" == true ]]; then
        full_cmd+=(--policy.use_relative_actions=true)
        full_cmd+=(--policy.relative_exclude_joints='["gripper"]')
        case "$policy_prefix" in
            diffusion*) full_cmd+=(--dataset.stats_path="$DIFFUSION_STATS_PATH") ;;
            pi05*|pi0*) full_cmd+=(--dataset.stats_path="$PI05_STATS_PATH") ;;
        esac
    fi

    # Append per-job extra args last so they override any earlier defaults (e.g. batch_size)
    if [[ -n "$extra_args_ref" ]]; then
        local -n extra_args="$extra_args_ref"
        full_cmd+=("${extra_args[@]}")
    fi

    echo "Running: ${env_prefix:+$env_prefix }${full_cmd[*]}"
    if [[ "$DRY_RUN" == false ]]; then
        ${env_prefix:+env $env_prefix} "${full_cmd[@]}"
    fi
    echo ""
    echo "============================================================"
}

# ── Per-job overrides ─────────────────────────────────────────
# Edit here to set env vars or extra CLI args for specific jobs.
# Extra args are appended last and override matching args in the policy arg arrays.

DIFFUSION_BASEWRIST_ENV=""
DIFFUSION_BASEWRIST_EXTRA=()

DIFFUSION_BASE_ENV=""
DIFFUSION_BASE_EXTRA=()

DIFFUSION_WRIST_ENV=""
DIFFUSION_WRIST_EXTRA=()

# pi05 basewrist: needs extra VRAM setting + smaller batch size
PI05_BASEWRIST_ENV="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
PI05_BASEWRIST_EXTRA=(--batch_size=8)

PI05_BASE_ENV=""
PI05_BASE_EXTRA=(--batch_size=8)

PI05_WRIST_ENV=""
PI05_WRIST_EXTRA=(--batch_size=8)

ACT_BASEWRIST_ENV=""
ACT_BASEWRIST_EXTRA=()

ACT_BASE_ENV=""
ACT_BASE_EXTRA=()

ACT_WRIST_ENV=""
ACT_WRIST_EXTRA=()

# ── Run jobs ──────────────────────────────────────────────────

maybe_sleep() { [[ "$DRY_RUN" == false ]] && sleep 10; }

# All training jobs live here.  Wrapped in a function so the ratio sweep loop
# can call it once per merged dataset, then clean up before the next iteration.
_run_all_jobs() {
    run_job "pi05" "basewrist" PI05_ARGS "$PI05_RESIZE_MODE" "$PI05_BASEWRIST_ENV" PI05_BASEWRIST_EXTRA
    maybe_sleep

    # run_job "act" "basewrist" ACT_ARGS "$ACT_RESIZE_MODE" "$ACT_BASEWRIST_ENV" ACT_BASEWRIST_EXTRA
    # maybe_sleep

    # run_job "diffusion" "basewrist" DIFFUSION_ARGS "$DIFFUSION_RESIZE_MODE" "$DIFFUSION_BASEWRIST_ENV" DIFFUSION_BASEWRIST_EXTRA
    # maybe_sleep

    # run_job "pi05" "base"  PI05_ARGS "$PI05_RESIZE_MODE" "$PI05_BASE_ENV"  PI05_BASE_EXTRA
    # maybe_sleep

    # run_job "pi05" "wrist" PI05_ARGS "$PI05_RESIZE_MODE" "$PI05_WRIST_ENV" PI05_WRIST_EXTRA

    # run_job "diffusion" "base"  DIFFUSION_ARGS "$DIFFUSION_RESIZE_MODE" "$DIFFUSION_BASE_ENV"  DIFFUSION_BASE_EXTRA
    # maybe_sleep

    # run_job "diffusion" "wrist" DIFFUSION_ARGS "$DIFFUSION_RESIZE_MODE" "$DIFFUSION_WRIST_ENV" DIFFUSION_WRIST_EXTRA
    # maybe_sleep
}

# ── Plain run or ratio sweep ───────────────────────────────────

if [[ "$RATIO_SWEEP" == false ]]; then
    _run_all_jobs
else
    # Snapshot base dataset vars so each sweep iteration can restore stats paths.
    _BASE_DATASET_REPO="$DATASET_REPO"
    _BASE_DATASET_SHORT="$DATASET_SHORT"
    _BASE_PI05_STATS="$PI05_STATS_PATH"
    _BASE_DIFFUSION_STATS="$DIFFUSION_STATS_PATH"
    _HF_LEROBOT_HOME="$(python3 -c "
import os; from pathlib import Path
print(Path(os.environ.get('HF_LEROBOT_HOME', Path.home()/'.cache/huggingface/lerobot')))")"

    _ratio_to_tag() { python3 -c "import sys; r=float(sys.argv[1]); print(f'{int(round(r*10)):02d}')" "$1"; }

    _CUMULATIVE_RATIOS=()

    for _RATIO in "${RATIOS[@]}"; do
        _CUMULATIVE_RATIOS+=("$_RATIO")

        # Build merged dataset name from all cumulative tags joined by _
        _ALL_TAGS=""
        for _r in "${_CUMULATIVE_RATIOS[@]}"; do
            _t=$(_ratio_to_tag "$_r")
            _ALL_TAGS="${_ALL_TAGS:+${_ALL_TAGS}_}${_t}"
        done
        _MERGED_NAME="${_BASE_DATASET_SHORT}_base_piabsden${_ALL_TAGS}"
        _MERGED_REPO="JennyWWW/splatsim_${_MERGED_NAME}"
        _MERGED_ROOT="${_HF_LEROBOT_HOME}/JennyWWW/${_MERGED_NAME#splatsim_}"

        echo "============================================================"
        echo "RATIO SWEEP: cumulative ratios up to ${_RATIO} → ${_MERGED_REPO}"
        echo "============================================================"

        # Step 1: create the merged dataset
        if [[ "$DRY_RUN" == false ]]; then
            python my_scripts/merge_augmented_datasets_for_training.py \
                --base "$_BASE_DATASET_REPO" \
                --ratios "${_CUMULATIVE_RATIOS[@]}"
        else
            echo "[DRY-RUN] python my_scripts/merge_augmented_datasets_for_training.py \\"
            echo "    --base $_BASE_DATASET_REPO --ratios ${_CUMULATIVE_RATIOS[*]}"
        fi

        # Step 2: point training at merged dataset; reuse base stats
        DATASET_REPO="$_MERGED_REPO"
        DATASET_SHORT="$_MERGED_NAME"
        PI05_STATS_PATH="$_BASE_PI05_STATS"
        DIFFUSION_STATS_PATH="$_BASE_DIFFUSION_STATS"

        _run_all_jobs

        # Step 3: delete merged dataset to reclaim disk before next iteration
        if [[ "$DRY_RUN" == false && -d "$_MERGED_ROOT" ]]; then
            echo "Removing merged dataset to free disk: $_MERGED_ROOT"
            rm -rf "$_MERGED_ROOT"
        else
            echo "[DRY-RUN] rm -rf $_MERGED_ROOT"
        fi
        echo ""
    done

    # Restore base vars
    DATASET_REPO="$_BASE_DATASET_REPO"
    DATASET_SHORT="$_BASE_DATASET_SHORT"
    PI05_STATS_PATH="$_BASE_PI05_STATS"
    DIFFUSION_STATS_PATH="$_BASE_DIFFUSION_STATS"
fi
