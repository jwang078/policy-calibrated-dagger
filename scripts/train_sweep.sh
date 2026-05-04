#!/usr/bin/env bash

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

# ============================================================
# USER CONFIG — edit this section to change the experiment
# ============================================================
DATASET_SHORT="approach_lever_11_50failsrrtpi05"
# DATASET_SHORT="approach_lever_11_biasend_5path"
# DATASET_SHORT="approach_lever_10_rectify_5path"
# DATASET_SHORT="approach_lever_9_rectify_5path"
# DATASET_SHORT="approach_lever_7_lowres_5path_10fails"
# DATASET_SHORT="approach_lever_7_lowres_5path"
# DATASET_SHORT="approach_lever_6_noteleport_5path"

# Set to true to train with relative actions. Requires running
# compute_relative_stats.sh first to generate the stats files.
USE_RELATIVE_ACTIONS=true
# ============================================================

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

    # Append relative-action flags if enabled, using the per-policy stats file
    if [[ "$USE_RELATIVE_ACTIONS" == true ]]; then
        full_cmd+=(--policy.use_relative_actions=true)
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
PI05_BASE_EXTRA=()

PI05_WRIST_ENV=""
PI05_WRIST_EXTRA=()

# ── Run jobs ──────────────────────────────────────────────────

maybe_sleep() { [[ "$DRY_RUN" == false ]] && sleep 10; }

run_job "pi05" "basewrist" PI05_ARGS "$PI05_RESIZE_MODE" "$PI05_BASEWRIST_ENV" PI05_BASEWRIST_EXTRA
maybe_sleep

run_job "diffusion" "basewrist" DIFFUSION_ARGS "$DIFFUSION_RESIZE_MODE" "$DIFFUSION_BASEWRIST_ENV" DIFFUSION_BASEWRIST_EXTRA
maybe_sleep


# run_job "pi05" "base"  PI05_ARGS "$PI05_RESIZE_MODE" "$PI05_BASE_ENV"  PI05_BASE_EXTRA
# maybe_sleep

# run_job "pi05" "wrist" PI05_ARGS "$PI05_RESIZE_MODE" "$PI05_WRIST_ENV" PI05_WRIST_EXTRA



# run_job "diffusion" "base"  DIFFUSION_ARGS "$DIFFUSION_RESIZE_MODE" "$DIFFUSION_BASE_ENV"  DIFFUSION_BASE_EXTRA
# maybe_sleep

# run_job "diffusion" "wrist" DIFFUSION_ARGS "$DIFFUSION_RESIZE_MODE" "$DIFFUSION_WRIST_ENV" DIFFUSION_WRIST_EXTRA
# maybe_sleep
