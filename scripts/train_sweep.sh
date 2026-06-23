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
#   --dataset_repo=ID       Full dataset repo id, e.g.
#                           "JennyWWW/splatsim_approach_lever_11_50failsrrtpi05".
#                           DATASET_SHORT (used for the stats sidecar dir and
#                           the auto-derived run_name) is inferred by stripping
#                           "JennyWWW/" and an optional "splatsim_" prefix.
#   --ratio_sweep           Enable the augmented-ratio sweep
#   --ratios="N N N"        Space-separated ratio list (used when --ratio_sweep)
#   --no_relative           Disable relative-action training (default: enabled)
#   --model=NAME            Which policy to train: "pi05" | "diffusion" | "act".
#                           Default: pi05. Selects which run_job(...) invocation
#                           runs inside _run_all_jobs, which policy-args block
#                           is used, and which chunk size the relative-action
#                           stats sidecar is keyed off (pi05/pi0 → 50, diffusion
#                           → 8, act → none — uses absolute actions).
#   --env_external_port=N   Connect lerobot-train's inline eval to an external
#                           SplatSim ZMQ server at this port (e.g. 6001) instead
#                           of spawning a new one. Required when training has to
#                           share a GPU with other SplatSim consumers (e.g. the
#                           dagger_orchestrate.sh pipeline). User must launch
#                           SplatSim on that port BEFORE invoking this script.
#   --dry-run               Print commands without executing
#
# Example:
#   bash my_scripts/train_sweep.sh \
#       --dataset_repo=JennyWWW/splatsim_approach_lever_11_50failsrrtpi05 \
#       --ratio_sweep \
#       --ratios="0.2 0.4 0.6 0.8 1.0"

# ── USER CONFIG (defaults) ────────────────────────────────────────────────────
DATASET_REPO="JennyWWW/splatsim_approach_lever_11_50failsrrtpi05"
USE_RELATIVE_ACTIONS=true
RATIO_SWEEP=false
# Multi-dataset weighted-sampling mode passthroughs (all-or-nothing).
# When the orchestrator's --use_weighted_sampling + --final_mode=scratch
# need to train from scratch on the union of {base + every round's
# intervention + every round's blends} — the same per-source mix that
# the per-round step-6 finetune trains on — it forwards these four args
# verbatim to lerobot-train as --dataset.repo_ids / --dataset.sample_weights
# / --dataset.stats_paths / --dataset.norm_mode. The wrapper
# (MultiSourceNormalizingDataset) then handles per-source loading +
# stats aggregation; no merged dataset on disk is needed.
#
# Set ALL FOUR or NONE — half-set is rejected at validation. When set,
# --dataset_repo is cleared (multi mode is mutually exclusive with the
# single-dataset path via TrainPipelineConfig.validate), the
# stats_rel{N}.json sidecar lookup is skipped (each sub-dataset's
# sidecar is supplied directly in --multi_dataset_stats_paths), and
# --run_name= must be passed explicitly (DATASET_SHORT-derived naming
# has nothing meaningful to derive from).
#
# Format: JSON strings, same shape lerobot-train accepts:
#   --multi_dataset_repo_ids='["JennyWWW/foo","JennyWWW/bar"]'
#   --multi_dataset_sample_weights='[0.7,0.3]'
#   --multi_dataset_stats_paths='["/abs/path/foo.json","/abs/path/bar.json"]'
#   --multi_dataset_norm_mode='aggregated'  # or 'base_only'
MULTI_DATASET_REPO_IDS=""
MULTI_DATASET_SAMPLE_WEIGHTS=""
MULTI_DATASET_STATS_PATHS=""
MULTI_DATASET_NORM_MODE=""
# --headless: route both --env.headless=true (in-process PybulletRobotServerBase
# in p.DIRECT mode) and --policy.shared_autonomy_config.show_slider=false
# (defensive; gates the Tkinter slider + SA wrapper's pybullet GUI client if
# the policy carries SA config) into SHARED_ARGS. Default false → unchanged.
# Forwarded from dagger_orchestrate.sh --headless via HEADLESS_TRAIN_SCRATCH_ARGS.
HEADLESS=false
RATIOS=(0.2 0.4 0.6 0.8 1.0)
ENV_EXTERNAL_PORT=""
POLICY_PUSH_TO_HUB=""   # empty = use whatever the policy config default is
RUN_NAME_OVERRIDE=""    # set to override the auto-derived run_name (training dir basename)
MODEL="pi05"            # which policy to train: pi05 | diffusion | act
DRY_RUN=false
# Eval-scope passthroughs from the DAgger orchestrator (or other callers
# that want to control inline eval scope). Both empty by default so
# standalone train_sweep.sh uses its built-in SHARED_ARGS values:
# `--eval.n_episodes=5` and NO --env.eval_benchmark_subset (uses full
# benchmark). When the orchestrator passes them, they appear AFTER
# SHARED_ARGS in the final command line and draccus's
# last-occurrence-wins rule means the override applies.
EVAL_N_EPISODES=""
EVAL_BENCHMARK_SUBSET=""
# ─────────────────────────────────────────────────────────────────────────────

for arg in "$@"; do
    case "$arg" in
        --dry-run)              DRY_RUN=true ;;
        --ratio_sweep)          RATIO_SWEEP=true ;;
        --no_relative)          USE_RELATIVE_ACTIONS=false ;;
        --headless)             HEADLESS=true ;;
        --dataset_repo=*)       DATASET_REPO="${arg#*=}" ;;
        --ratios=*)             IFS=' ' read -ra RATIOS <<< "${arg#*=}" ;;
        --env_external_port=*)  ENV_EXTERNAL_PORT="${arg#*=}" ;;
        --policy.push_to_hub=*) POLICY_PUSH_TO_HUB="${arg#*=}" ;;
        --run_name=*)           RUN_NAME_OVERRIDE="${arg#*=}" ;;
        --model=*)              MODEL="${arg#*=}" ;;
        --eval_n_episodes=*)    EVAL_N_EPISODES="${arg#*=}" ;;
        --eval_benchmark_subset=*) EVAL_BENCHMARK_SUBSET="${arg#*=}" ;;
        --multi_dataset_repo_ids=*)      MULTI_DATASET_REPO_IDS="${arg#*=}" ;;
        --multi_dataset_sample_weights=*) MULTI_DATASET_SAMPLE_WEIGHTS="${arg#*=}" ;;
        --multi_dataset_stats_paths=*)   MULTI_DATASET_STATS_PATHS="${arg#*=}" ;;
        --multi_dataset_norm_mode=*)     MULTI_DATASET_NORM_MODE="${arg#*=}" ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

# All-or-nothing validation for the multi-dataset passthroughs. Half-set
# is almost certainly a caller bug (e.g. a missing JSON-encode in the
# orchestrator) — fail loudly here rather than silently emitting a broken
# lerobot-train command.
_multi_count=0
[[ -n "$MULTI_DATASET_REPO_IDS" ]]      && _multi_count=$((_multi_count + 1))
[[ -n "$MULTI_DATASET_SAMPLE_WEIGHTS" ]] && _multi_count=$((_multi_count + 1))
[[ -n "$MULTI_DATASET_STATS_PATHS" ]]   && _multi_count=$((_multi_count + 1))
[[ -n "$MULTI_DATASET_NORM_MODE" ]]     && _multi_count=$((_multi_count + 1))
if (( _multi_count > 0 && _multi_count < 4 )); then
    echo "ERROR: --multi_dataset_* args are all-or-nothing. Set all four or none. Currently: $_multi_count/4." >&2
    echo "  Got:" >&2
    echo "    --multi_dataset_repo_ids='$MULTI_DATASET_REPO_IDS'" >&2
    echo "    --multi_dataset_sample_weights='$MULTI_DATASET_SAMPLE_WEIGHTS'" >&2
    echo "    --multi_dataset_stats_paths='$MULTI_DATASET_STATS_PATHS'" >&2
    echo "    --multi_dataset_norm_mode='$MULTI_DATASET_NORM_MODE'" >&2
    exit 1
fi
MULTI_DATASET_MODE=false
if (( _multi_count == 4 )); then
    MULTI_DATASET_MODE=true
    if [[ -z "$RUN_NAME_OVERRIDE" ]]; then
        echo "ERROR: --multi_dataset_* mode requires --run_name=... (no DATASET_SHORT to auto-derive from)." >&2
        exit 1
    fi
fi

case "$MODEL" in
    pi05|diffusion|act) ;;
    *) echo "ERROR: --model must be one of pi05/diffusion/act (got '$MODEL')" >&2; exit 1 ;;
esac

# DATASET_SHORT is derived from DATASET_REPO by stripping "JennyWWW/" and an
# optional "splatsim_" prefix. It's used to construct the stats sidecar dir
# and the auto-derived run_name. Keeping a single source of truth (DATASET_REPO)
# avoids the bug where --dataset_short=foo passes a short that doesn't match
# the actual repo on disk (e.g. dag-merged datasets that omit the splatsim_
# prefix).
DATASET_SHORT="${DATASET_REPO#*/}"
DATASET_SHORT="${DATASET_SHORT#splatsim_}"

# Paths written by compute_relative_stats.sh. The sidecar files are named by
# the chunk size they were computed against (stats_rel{N}.json) — the policy
# type doesn't matter, only the chunk over which action deltas are computed.
# run_job picks the right one based on each policy's chunk_size.
STATS_DIR=~/code/lerobot/outputs/dataset_stats/${DATASET_SHORT}

# Resolve the chunk size (== n_action_steps for policies that consume their
# full chunk) used to construct the relative-action stats sidecar path
# (stats_rel${chunk}.json). Source of truth, in priority order:
#   1. Explicit --policy.n_action_steps=N in the policy args array.
#   2. Explicit --policy.chunk_size=N in the policy args array.
#   3. The policy class's default (per-prefix fallback): pi05/pi0 → 50,
#      diffusion → 8.
# Empty for policies that don't use the relative-action pipeline (e.g. act with
# temporal ensembling — n_action_steps=1 but chunk_size=50, and the policy uses
# absolute actions anyway).
#
# Note: we can't read this from a train_config.json the way dagger_orchestrate
# does for finetune, because run_job trains from scratch — no config exists yet.
# Args: $1 = policy_prefix, $2 = name of policy_args array (nameref).
_chunk_size_for_job() {
    local prefix="$1"
    local -n args_ref="$2"
    local override_nsteps="" override_chunk=""
    for a in "${args_ref[@]}"; do
        case "$a" in
            --policy.n_action_steps=*) override_nsteps="${a#*=}" ;;
            --policy.chunk_size=*)     override_chunk="${a#*=}" ;;
        esac
    done
    if [[ -n "$override_nsteps" ]]; then echo "$override_nsteps"; return; fi
    if [[ -n "$override_chunk"  ]]; then echo "$override_chunk";  return; fi
    case "$prefix" in
        diffusion*) echo 8 ;;
        pi05*|pi0*) echo 50 ;;
        *)          echo "" ;;
    esac
}

# Bare per-prefix fallback used only by the early validation block below (which
# doesn't have access to the per-job policy_args arrays). Keep these in sync
# with the per-prefix defaults in _chunk_size_for_job.
_chunk_size_for_prefix() {
    case "$1" in
        diffusion*) echo 8 ;;
        pi05*|pi0*) echo 50 ;;
        *)          echo "" ;;
    esac
}

# Validate that the stats file exists for the SELECTED model's chunk size when
# USE_RELATIVE_ACTIONS=true. Missing → the user forgot to run
# compute_relative_stats.sh. act uses absolute actions so no sidecar applies.
if [[ "$USE_RELATIVE_ACTIONS" == true ]]; then
    chunk="$(_chunk_size_for_prefix "$MODEL")"
    if [[ -n "$chunk" ]]; then
        f="${STATS_DIR}/stats_rel${chunk}.json"
        if [[ ! -f "$f" ]]; then
            echo "ERROR: USE_RELATIVE_ACTIONS=true but stats file not found: $f" >&2
            echo "Run my_scripts/compute_relative_stats.sh first." >&2
            exit 1
        fi
    fi
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
if [[ "$MULTI_DATASET_MODE" != true ]]; then
    validate_names
else
    echo "Multi-dataset mode: skipping single-dataset name validation (DATASET_REPO unused)."
fi

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

# When --env_external_port is set, route lerobot-train's inline eval to that
# port so it shares a single SplatSim ZMQ server with the rest of the pipeline.
# Otherwise lerobot-train spawns its own (which conflicts with another running
# SplatSim on the same GPU). User must launch SplatSim on this port externally.
if [[ -n "$ENV_EXTERNAL_PORT" ]]; then
    SHARED_ARGS+=( "--env.external_port=$ENV_EXTERNAL_PORT" )
fi
if [[ -n "$POLICY_PUSH_TO_HUB" ]]; then
    SHARED_ARGS+=( "--policy.push_to_hub=$POLICY_PUSH_TO_HUB" )
fi
# --headless propagation. Same surfaces gated as on the orchestrator's
# finetune path (HEADLESS_TRAIN_ARGS): env-side flag for the in-process sim,
# wrapper-side flag for the SA GUI client. Skip --env.headless when
# --env_external_port is set, since that path uses ZMQSplatSimGymEnv (no
# local pybullet client) and the external sim's GUI mode is the user's
# concern.
if [[ "$HEADLESS" == true ]]; then
    SHARED_ARGS+=( "--policy.shared_autonomy_config.show_slider=false" )
    if [[ -z "$ENV_EXTERNAL_PORT" ]]; then
        SHARED_ARGS+=( "--env.headless=true" )
    fi
fi

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
    # RUN_NAME_OVERRIDE (top-level, set via --run_name flag) overrides the
    # default naming derived from policy_prefix/dataset_short/action/camera.
    # Used by dagger_orchestrate.sh so scratch-mode rounds land at the same
    # path as finetune-mode rounds (${BASE_POLICY_NAME}_dag${r}).
    local run_name
    if [[ -n "${RUN_NAME_OVERRIDE:-}" ]]; then
        run_name="$RUN_NAME_OVERRIDE"
    else
        run_name="${policy_prefix}_${DATASET_SHORT}_${action_suffix}_${camera_suffix}"
    fi

    set_camera_args "$resize_mode" "$camera_suffix"

    local full_cmd
    if [[ "$MULTI_DATASET_MODE" == true ]]; then
        # Multi-dataset path: --dataset.repo_id MUST be empty (mutually
        # exclusive with --dataset.repo_ids per TrainPipelineConfig.validate).
        # --dataset.stats_path is also cleared so the wrapper's mode-
        # appropriate stats stand (mirrors the orchestrator's per-round
        # step-6 finetune logic — see the long comment around
        # `--dataset.stats_path=` in dagger_orchestrate.sh).
        full_cmd=(
            $TRAIN_SCRIPT
            --dataset.repo_id=
            --dataset.repo_ids="$MULTI_DATASET_REPO_IDS"
            --dataset.sample_weights="$MULTI_DATASET_SAMPLE_WEIGHTS"
            --dataset.stats_paths="$MULTI_DATASET_STATS_PATHS"
            --dataset.norm_mode="$MULTI_DATASET_NORM_MODE"
            --dataset.stats_path=
            --output_dir="./outputs/training/${run_name}"
            --job_name="${run_name}"
            --policy.repo_id="${run_name}"
            "${SHARED_ARGS[@]}"
            "${policy_args[@]}"
            "${CAMERA_ARGS[@]}"
        )
        # In multi mode, relative-action flag is forwarded but the
        # per-policy stats_rel{N}.json sidecar lookup is SKIPPED — each
        # sub-dataset's sidecar comes through --dataset.stats_paths above,
        # and the wrapper aggregates / picks-base depending on norm_mode.
        if [[ "$USE_RELATIVE_ACTIONS" == true ]]; then
            full_cmd+=(--policy.use_relative_actions=true)
            full_cmd+=(--policy.relative_exclude_joints='["gripper"]')
        fi
    else
        full_cmd=(
            $TRAIN_SCRIPT
            --dataset.repo_id="$DATASET_REPO"
            --output_dir="./outputs/training/${run_name}"
            --job_name="${run_name}"
            --policy.repo_id="${run_name}"
            "${SHARED_ARGS[@]}"
            "${policy_args[@]}"
            "${CAMERA_ARGS[@]}"
        )

        # Append relative-action flags if enabled, picking the stats sidecar by
        # the policy's chunk size (stats_rel{N}.json).
        if [[ "$USE_RELATIVE_ACTIONS" == true ]]; then
            full_cmd+=(--policy.use_relative_actions=true)
            full_cmd+=(--policy.relative_exclude_joints='["gripper"]')
            local chunk_size
            chunk_size="$(_chunk_size_for_job "$policy_prefix" "$3")"
            if [[ -n "$chunk_size" ]]; then
                full_cmd+=(--dataset.stats_path="${STATS_DIR}/stats_rel${chunk_size}.json")
            fi
        fi
    fi

    # Append per-job extra args last so they override any earlier defaults (e.g. batch_size)
    if [[ -n "$extra_args_ref" ]]; then
        local -n extra_args="$extra_args_ref"
        full_cmd+=("${extra_args[@]}")
    fi

    # Eval-scope overrides from --eval_n_episodes / --eval_benchmark_subset
    # appended LAST so they win over SHARED_ARGS's --eval.n_episodes=5 default
    # via draccus's last-occurrence-wins rule. Drives the inline eval scope
    # for the final-scratch step to match the orchestrator's intervention
    # subset (so per-round + final-scratch evals are directly comparable in
    # dagger_progress).
    if [[ -n "$EVAL_N_EPISODES" ]]; then
        full_cmd+=( --eval.n_episodes="$EVAL_N_EPISODES" )
    fi
    if [[ -n "$EVAL_BENCHMARK_SUBSET" ]]; then
        full_cmd+=( --env.eval_benchmark_subset="$EVAL_BENCHMARK_SUBSET" )
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
# Dispatches by $MODEL — only the selected model's basewrist job runs. Other
# camera variants (base/wrist-only) remain commented out — uncomment to enable.
_run_all_jobs() {
    case "$MODEL" in
        pi05)
            run_job "pi05" "basewrist" PI05_ARGS "$PI05_RESIZE_MODE" "$PI05_BASEWRIST_ENV" PI05_BASEWRIST_EXTRA
            maybe_sleep
            # run_job "pi05" "base"  PI05_ARGS "$PI05_RESIZE_MODE" "$PI05_BASE_ENV"  PI05_BASE_EXTRA
            # maybe_sleep
            # run_job "pi05" "wrist" PI05_ARGS "$PI05_RESIZE_MODE" "$PI05_WRIST_ENV" PI05_WRIST_EXTRA
            ;;
        diffusion)
            run_job "diffusion" "basewrist" DIFFUSION_ARGS "$DIFFUSION_RESIZE_MODE" "$DIFFUSION_BASEWRIST_ENV" DIFFUSION_BASEWRIST_EXTRA
            maybe_sleep
            # run_job "diffusion" "base"  DIFFUSION_ARGS "$DIFFUSION_RESIZE_MODE" "$DIFFUSION_BASE_ENV"  DIFFUSION_BASE_EXTRA
            # maybe_sleep
            # run_job "diffusion" "wrist" DIFFUSION_ARGS "$DIFFUSION_RESIZE_MODE" "$DIFFUSION_WRIST_ENV" DIFFUSION_WRIST_EXTRA
            ;;
        act)
            run_job "act" "basewrist" ACT_ARGS "$ACT_RESIZE_MODE" "$ACT_BASEWRIST_ENV" ACT_BASEWRIST_EXTRA
            maybe_sleep
            # run_job "act" "base"  ACT_ARGS "$ACT_RESIZE_MODE" "$ACT_BASE_ENV"  ACT_BASE_EXTRA
            # maybe_sleep
            # run_job "act" "wrist" ACT_ARGS "$ACT_RESIZE_MODE" "$ACT_WRIST_ENV" ACT_WRIST_EXTRA
            ;;
    esac
}

# ── Plain run or ratio sweep ───────────────────────────────────

if [[ "$RATIO_SWEEP" == false ]]; then
    _run_all_jobs
else
    # Snapshot base dataset vars so each sweep iteration can restore them.
    # STATS_DIR is derived from DATASET_SHORT and rewritten per iteration; the
    # per-chunk file paths are built on demand inside run_job, so no separate
    # PI05/DIFFUSION variables need snapshotting.
    _BASE_DATASET_REPO="$DATASET_REPO"
    _BASE_DATASET_SHORT="$DATASET_SHORT"
    _BASE_STATS_DIR="$STATS_DIR"
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

        # Step 2: point training at merged dataset; reuse base stats dir.
        DATASET_REPO="$_MERGED_REPO"
        DATASET_SHORT="$_MERGED_NAME"
        STATS_DIR="$_BASE_STATS_DIR"

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
    STATS_DIR="$_BASE_STATS_DIR"
fi
