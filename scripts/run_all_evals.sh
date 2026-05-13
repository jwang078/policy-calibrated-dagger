#!/bin/bash

# Script to run lerobot-eval on all policy checkpoints, in eval-benchmark
# mode against a fixed pre-recorded scenario set so different runs and
# checkpoints are directly comparable.
#
# Each lerobot-eval invocation launches its own splatsim process in
# EVAL_BENCHMARK mode bound to $EVAL_BENCHMARK_REPO_ID, with --seed=0 so
# the seed-pinned reset path picks scenario 0, 1, 2, ... in order. Setting
# --n-episodes=N therefore evaluates the FIRST N scenarios from the
# benchmark dataset.
#
# Handles diffusion_approach_lever_*, pi05_approach_lever_*, and act_approach_lever_* folders.
#
# Usage: ./run_all_evals.sh [--dry-run] [--list] [--first-only] [--n-episodes int] [--episode-length int] [--temporal-ensemble] [--temporal-ensemble-coeff float] [--temporal-ensemble-n-action-steps int]
#   --dry-run:                          Show commands without executing them
#   --list:                             Only list experiments and checkpoints to evaluate, then exit
#   --first-only:                       Only evaluate the first checkpoint per experiment (for debugging)
#   --n-episodes:                       Number of first-N benchmark scenarios to evaluate (default 5)
#   --episode-length:                   Max steps per episode (default: use env default)
#   --temporal-ensemble:                Enable TemporalEnsemblePolicyWrapper at eval time. Adds
#                                       --policy.temporal_ensemble_config.* flags and sets n_action_steps
#                                       (default 1 = max smoothing; override via --temporal-ensemble-n-action-steps).
#                                       When enabled, the output dir is tagged with _te_K{N}_c{COEFF} so
#                                       results don't collide with the un-ensembled baseline.
#                                       ACT checkpoints with legacy temporal_ensemble_coeff stay on their
#                                       inline path (the wrapper is skipped via lerobot's deprecation
#                                       check), so this is safe to enable across mixed checkpoint sets.
#   --temporal-ensemble-coeff:          Temporal-ensemble coefficient (default 0.01). Larger positive
#                                       values weight older predictions more heavily.
#   --temporal-ensemble-n-action-steps: Cadence at which the model is queried under temporal ensembling
#                                       (default 1 = max smoothing). K must be < chunk_size or the factory
#                                       raises (no-op smoothing). K=chunk_size-1 (e.g. 49 for pi05) gives
#                                       the minimum-smoothing setting that still mitigates chunk-boundary
#                                       jerk — good for multimodal tasks.
#   --force-act-to-wrapper-mode:        ACT-only opt-in. By default, ACT checkpoints trained with the
#                                       legacy temporal_ensemble_coeff use their inline ensembler and
#                                       skip the new wrapper. Pass this flag to force the wrapper path
#                                       on ACT — useful for A/B-testing wrapper equivalence vs the
#                                       inline ensembler. No effect on pi05/diffusion. Output dir is
#                                       additionally tagged with _forcenoactte when set.
#   --last-mile-debug:                  DEBUG/diagnostic. Wraps the policy with LastMileDebugWrapper which,
#                                       when the robot's end-effector is within --last-mile-debug-threshold
#                                       (meters, L2 position) of oracle_env_config.task.target_ee_pos, blends
#                                       the commanded joint targets toward q_goal_bias by
#                                       --last-mile-debug-alpha (1.0 = full override). Trigger metric matches
#                                       the simulator's success criterion (EE-space distance) — robust to
#                                       the kinematic-redundancy issue where the policy reaches a similar EE
#                                       pose via a totally different joint config than q_goal_bias. Requires
#                                       splatsim to expose current_ee_pos in get_env_config (recently added).
#                                       Output dir tagged _lastmile{threshold}a{alpha}.
#   --last-mile-debug-threshold:        EE position distance threshold (meters) below which the override fires.
#                                       Default 0.05 (5cm). The simulator's typical success tolerance is ~3cm
#                                       so 5cm fires shortly before the success region.
#   --last-mile-debug-alpha:            Blend weight 0..1. 0=pure policy (sanity check), 1=full goal override.
#                                       Default 1.0.
#   --temporal-ensemble-pin-noise:      Stochastic-policy diagnostic. pi05 (and diffusion) sample fresh
#                                       Gaussian noise every predict_action_chunk; without pinning,
#                                       each chunk fed to the ensembler is an independent draw from
#                                       p(actions|obs) and on multimodal tasks the average collapses
#                                       across modes (slow / wrong-direction progress). With this flag,
#                                       the wrapper samples noise once per episode and reuses it so
#                                       successive chunks differ only by observation conditioning.
#                                       No effect on deterministic ACT. Output dir tagged _pinnoise.

# Parse arguments
DRY_RUN=false
LIST_ONLY=false
FIRST_ONLY=false
LAST_ONLY=false
N_EPISODES=5  # Default value
EPISODE_LENGTH=""  # Default: unset (use env default)
TEMPORAL_ENSEMBLE=false
TEMPORAL_ENSEMBLE_COEFF=0.01
TEMPORAL_ENSEMBLE_N_ACTION_STEPS=1
FORCE_ACT_TO_WRAPPER_MODE=false
TEMPORAL_ENSEMBLE_PIN_NOISE=false
LAST_MILE_DEBUG=false
LAST_MILE_DEBUG_THRESHOLD=0.05
LAST_MILE_DEBUG_ALPHA=1.0

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --list)
            LIST_ONLY=true
            shift
            ;;
        --first-only)
            FIRST_ONLY=true
            shift
            ;;
        --last-only)
            LAST_ONLY=true
            shift
            ;;
        --n-episodes)
            N_EPISODES="$2"
            shift 2
            ;;
        --episode-length)
            EPISODE_LENGTH="$2"
            shift 2
            ;;
        --temporal-ensemble)
            TEMPORAL_ENSEMBLE=true
            shift
            ;;
        --temporal-ensemble-coeff)
            TEMPORAL_ENSEMBLE_COEFF="$2"
            shift 2
            ;;
        --temporal-ensemble-n-action-steps)
            TEMPORAL_ENSEMBLE_N_ACTION_STEPS="$2"
            shift 2
            ;;
        --force-act-to-wrapper-mode)
            FORCE_ACT_TO_WRAPPER_MODE=true
            shift
            ;;
        --temporal-ensemble-pin-noise)
            TEMPORAL_ENSEMBLE_PIN_NOISE=true
            shift
            ;;
        --last-mile-debug)
            LAST_MILE_DEBUG=true
            shift
            ;;
        --last-mile-debug-threshold)
            LAST_MILE_DEBUG_THRESHOLD="$2"
            shift 2
            ;;
        --last-mile-debug-alpha)
            LAST_MILE_DEBUG_ALPHA="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "See the header comment for the supported flags." >&2
            exit 1
            ;;
    esac
done

if [ "$DRY_RUN" = true ]; then
    echo "=== DRY RUN MODE - Commands will not be executed ==="
    echo ""
fi

OUTPUTS_DIR="/home/jennyw2/code/lerobot/outputs"
TIMESTAMP=$(date +"%Y-%m-%d-%H%M%S")
EVAL_OUTPUT_DIR="$OUTPUTS_DIR/eval_output/$TIMESTAMP"
EXP_PATTERNS=("$OUTPUTS_DIR"/training/diffusion_approach_lever_* "$OUTPUTS_DIR"/training/pi05_approach_lever_* "$OUTPUTS_DIR"/training/act_approach_lever_*)
# Fixed benchmark dataset that splatsim's EVAL_BENCHMARK mode replays. With
# --seed=0 in lerobot-eval, the seed-pinned reset path picks scenarios
# 0..n_episodes-1 in order, so all checkpoints see the same scenarios.
EVAL_BENCHMARK_REPO_ID="JennyWWW/eval_splatsim_approach_lever_benchmark_1000"

# Collect all experiment folders that will be processed
echo "========================================"
echo "EVALUATION PLAN"
echo "========================================"
echo ""
echo "Experiment folders to process:"
echo ""

eval_count=0
for exp_dir in "${EXP_PATTERNS[@]}"; do
    [ -d "$exp_dir" ] || continue
    exp_name=$(basename "$exp_dir")
    checkpoints_dir="$exp_dir/checkpoints"
    [ -d "$checkpoints_dir" ] || continue

    # Count checkpoints (excluding 'last' symlink which duplicates the last numerical)
    checkpoint_count=0
    for checkpoint_dir in "$checkpoints_dir"/*; do
        [ -d "$checkpoint_dir" ] || continue
        checkpoint_name=$(basename "$checkpoint_dir")
        [ "$checkpoint_name" == "last" ] && continue
        [ -d "$checkpoint_dir/pretrained_model" ] || continue
        ((checkpoint_count++))
    done

    echo "  - $exp_name ($checkpoint_count checkpoints)"

    # If last only, don't count all checkpoints, just 1
    if [ "$LAST_ONLY" = true ]; then
        checkpoint_count=1
        echo "    (will evaluate only the last checkpoint due to --last-only flag)"
    fi
    # If first only, also just count 1
    if [ "$FIRST_ONLY" = true ]; then
        checkpoint_count=1
        echo "    (will evaluate only the first checkpoint due to --first-only flag)"
    fi
    # If both first and last only, then throw an error because that can't be true
    if [ "$FIRST_ONLY" = true ] && [ "$LAST_ONLY" = true ]; then
        echo "Error: --first-only and --last-only flags cannot both be true"
        exit 1
    fi
    ((eval_count += checkpoint_count))
done

echo ""
echo "Total evaluations to run: $eval_count"
echo "========================================"

# Exit early if --list flag was provided
if [ "$LIST_ONLY" = true ]; then
    exit 0
fi

echo ""

# Function to get camera names from any checkpoint's train_config.json in the experiment
get_camera_names_from_experiment() {
    local checkpoints_dir="$1"

    # Find the first checkpoint with a train_config.json
    for checkpoint_dir in "$checkpoints_dir"/*; do
        [ -d "$checkpoint_dir" ] || continue
        local config_file="$checkpoint_dir/pretrained_model/train_config.json"
        if [ -f "$config_file" ]; then
            # Extract camera_names from config using Python
            # First try env.camera_names, then fall back to extracting from policy.input_features
            local result
            result=$(python3 -c "
import json
import re

config = json.load(open('$config_file'))

# Try env.camera_names first
env = config.get('env')
if env and env.get('camera_names'):
    print(json.dumps(env['camera_names']))
else:
    # Fall back to extracting from policy.input_features
    # Keys look like 'observation.images.base_rgb' -> extract 'base_rgb'
    input_features = config.get('policy', {}).get('input_features', {})
    camera_names = []
    for key in input_features:
        match = re.match(r'observation\.images\.(\w+)', key)
        if match:
            camera_names.append(match.group(1))
    if camera_names:
        print(json.dumps(sorted(camera_names)))
    else:
        print('[\"base_rgb\"]')
" 2>/dev/null)
            echo "$result"
            return
        fi
    done

    # Fallback if no config file found
    echo '["base_rgb"]'
}

# Function to get image resize modes (as JSON list string) from any checkpoint's train_config.json
get_image_resize_modes_from_experiment() {
    local checkpoints_dir="$1"

    for checkpoint_dir in "$checkpoints_dir"/*; do
        [ -d "$checkpoint_dir" ] || continue
        local config_file="$checkpoint_dir/pretrained_model/train_config.json"
        if [ -f "$config_file" ]; then
            local result
            result=$(python3 -c "
import json

config = json.load(open('$config_file'))

env = config.get('env', {})
# New field name (list)
if env.get('image_resize_modes'):
    print(json.dumps(env['image_resize_modes']))
# Old field name (single string), wrap in list
elif env.get('image_resize_mode'):
    print(json.dumps([env['image_resize_mode']]))
else:
    print('[\"letterbox\"]')
" 2>/dev/null)
            echo "$result"
            return
        fi
    done

    echo '["letterbox"]'
}

# Function to get rename_map (as JSON string) from any checkpoint's train_config.json
get_rename_map_from_experiment() {
    local checkpoints_dir="$1"

    for checkpoint_dir in "$checkpoints_dir"/*; do
        [ -d "$checkpoint_dir" ] || continue
        local config_file="$checkpoint_dir/pretrained_model/train_config.json"
        if [ -f "$config_file" ]; then
            local result
            result=$(python3 -c "
import json

config = json.load(open('$config_file'))
rename_map = config.get('rename_map', {})
print(json.dumps(rename_map))
" 2>/dev/null)
            echo "$result"
            return
        fi
    done

    echo '{}'
}

# Find the last numerical checkpoint (to skip it since 'last' symlink points to it)
get_last_numerical_checkpoint() {
    local checkpoints_dir="$1"
    # Get all numerical checkpoint directories, sort them, and get the last one
    ls -1 "$checkpoints_dir" | grep -E '^[0-9]+$' | sort -n | tail -1
}

# Process each matching experiment folder
for exp_dir in "${EXP_PATTERNS[@]}"; do
    # Skip if not a directory
    [ -d "$exp_dir" ] || continue

    exp_name=$(basename "$exp_dir")
    checkpoints_dir="$exp_dir/checkpoints"

    # Skip if no checkpoints directory
    if [ ! -d "$checkpoints_dir" ]; then
        echo "Skipping $exp_name: no checkpoints directory"
        continue
    fi

    # Get camera names from the experiment's config file
    camera_names=$(get_camera_names_from_experiment "$checkpoints_dir")

    # Get image resize modes and rename_map from the experiment's config file
    image_resize_modes=$(get_image_resize_modes_from_experiment "$checkpoints_dir")
    rename_map=$(get_rename_map_from_experiment "$checkpoints_dir")

    # Find the last numerical checkpoint to skip
    last_numerical=$(get_last_numerical_checkpoint "$checkpoints_dir")

    echo "========================================"
    echo "Processing experiment: $exp_name"
    echo "Camera names: $camera_names"
    echo "Image resize modes: $image_resize_modes"
    echo "Rename map: $rename_map"
    echo "Last numerical checkpoint: $last_numerical (will skip 'last' symlink)"
    echo "========================================"

    # Process each checkpoint
    first_done=false
    for checkpoint_dir in "$checkpoints_dir"/*; do
        [ -d "$checkpoint_dir" ] || continue

        checkpoint_name=$(basename "$checkpoint_dir")

        # Skip 'last' symlink (the last numerical checkpoint is the same)
        if [ "$checkpoint_name" == "last" ]; then
            echo "Skipping 'last' (same as $last_numerical)"
            continue
        fi

        # In --first-only mode, skip all but the first real checkpoint per experiment
        if [ "$FIRST_ONLY" = true ] && [ "$first_done" = true ]; then
            echo "Skipping $checkpoint_name (--first-only mode)"
            continue
        fi

        # In --last-only mode, skip all but the last real checkpoint per experiment
        if [ "$LAST_ONLY" = true ] && [ "$checkpoint_name" != "$last_numerical" ]; then
            echo "Skipping $checkpoint_name (--last-only mode)"
            continue
        fi

        policy_path="$checkpoint_dir/pretrained_model"

        # Check if pretrained_model exists
        if [ ! -d "$policy_path" ]; then
            echo "Skipping $checkpoint_name: no pretrained_model directory"
            continue
        fi

        # Create unique output directory for this evaluation. Tag the dir with
        # the TE config when enabled so a TE sweep doesn't overwrite the
        # un-ensembled baseline. Naming: `_te{coeff}` always; `_numactsteps{K}`
        # only when n_action_steps != 1 (the default max-smoothing case);
        # `_forcenoactte` when ACT's inline ensembler is being overridden by
        # the wrapper (i.e. ACT's built-in temporal ensembling is disabled in
        # favor of the policy-agnostic wrapper).
        eval_subdir="$EVAL_OUTPUT_DIR/${exp_name}_${checkpoint_name}"
        if [ "$TEMPORAL_ENSEMBLE" = true ]; then
            eval_subdir="${eval_subdir}_te${TEMPORAL_ENSEMBLE_COEFF}"
            if [ "$TEMPORAL_ENSEMBLE_N_ACTION_STEPS" != "1" ]; then
                eval_subdir="${eval_subdir}_numactsteps${TEMPORAL_ENSEMBLE_N_ACTION_STEPS}"
            fi
            if [ "$FORCE_ACT_TO_WRAPPER_MODE" = true ]; then
                eval_subdir="${eval_subdir}_forcenoactte"
            fi
            if [ "$TEMPORAL_ENSEMBLE_PIN_NOISE" = true ]; then
                eval_subdir="${eval_subdir}_pinnoise"
            fi
        fi
        if [ "$LAST_MILE_DEBUG" = true ]; then
            eval_subdir="${eval_subdir}_lastmile${LAST_MILE_DEBUG_THRESHOLD}a${LAST_MILE_DEBUG_ALPHA}"
        fi
        if [ "$DRY_RUN" = false ]; then
            mkdir -p "$eval_subdir"
        fi

        # Log file for this evaluation
        log_file="$eval_subdir/eval_log.txt"

        echo "----------------------------------------"
        echo "Running eval for: $exp_name / $checkpoint_name"
        echo "Policy path: $policy_path"
        echo "Output dir: $eval_subdir"
        echo "Log file: $log_file"
        echo "----------------------------------------"

        eval_cmd="lerobot-eval \\
            --env.type=splatsim \\
            --env.task=upright_small_engine_new \\
            --env.camera_names='$camera_names' \\
            --env.image_resize_modes='$image_resize_modes' \\
            --env.fps=30 \\
            --env.eval_benchmark_repo_id=$EVAL_BENCHMARK_REPO_ID \\
            --policy.path=$policy_path \\
            --eval.n_episodes=$N_EPISODES \\
            --output_dir=$eval_subdir \\
            --eval.batch_size=1 \\
            --eval.use_async_envs=false \\
            --seed=0 \\
            --rename_map='$rename_map'"

        if [ -n "$EPISODE_LENGTH" ]; then
            eval_cmd="$eval_cmd \\
            --env.episode_length=$EPISODE_LENGTH"
        fi

        # Dataset #7 (approach_lever_7_lowres_5path) was generated before the
        # fisheye wrist-camera change, so eval must also render the wrist cam
        # as pinhole to match training distribution.
        if [[ "$exp_name" == *"approach_lever_7_lowres_5path"* ]]; then
            eval_cmd="$eval_cmd \\
            --env.use_fisheye_wrist_camera=false"
        fi

        # Temporal ensembling: applies the TemporalEnsemblePolicyWrapper at eval
        # time for smoother chunk boundaries. Default n_action_steps=1 (model
        # queried every step, maximum smoothing); override via
        # --temporal-ensemble-n-action-steps for a lighter touch (e.g. K=49 on
        # pi05's chunk_size=50 smooths only the boundary). For ACT checkpoints
        # whose train config already set the legacy temporal_ensemble_coeff,
        # the inline path takes priority and the wrapper is skipped (see
        # ACTConfig.__post_init__).
        if [ "$TEMPORAL_ENSEMBLE" = true ]; then
            eval_cmd="$eval_cmd \\
            --policy.n_action_steps=$TEMPORAL_ENSEMBLE_N_ACTION_STEPS \\
            --policy.temporal_ensemble_config.enabled=true \\
            --policy.temporal_ensemble_config.coeff=$TEMPORAL_ENSEMBLE_COEFF"
            if [ "$FORCE_ACT_TO_WRAPPER_MODE" = true ]; then
                eval_cmd="$eval_cmd \\
            --policy.temporal_ensemble_config.force_act_to_wrapper_mode=true"
            fi
            if [ "$TEMPORAL_ENSEMBLE_PIN_NOISE" = true ]; then
                eval_cmd="$eval_cmd \\
            --policy.temporal_ensemble_config.pin_noise=true"
            fi
        fi

        # DEBUG: oracle last-mile override. Pulls commanded joint targets
        # toward oracle's q_goal_bias when within threshold. Diagnostic only;
        # the wrapper, factory call, and these CLI flags can all be removed
        # together once the precision hypothesis test is done.
        if [ "$LAST_MILE_DEBUG" = true ]; then
            eval_cmd="$eval_cmd \\
            --policy.last_mile_debug_config.enabled=true \\
            --policy.last_mile_debug_config.ee_distance_threshold=$LAST_MILE_DEBUG_THRESHOLD \\
            --policy.last_mile_debug_config.blend_alpha=$LAST_MILE_DEBUG_ALPHA"
        fi

        if [ "$DRY_RUN" = true ]; then
            echo "Command:"
            echo "$eval_cmd"
            echo ""
        else
            # Run eval and capture output to both terminal and log file
            # Filter out progress bar lines from the log file
            {
                echo "========================================"
                echo "Evaluation Log"
                echo "Experiment: $exp_name"
                echo "Checkpoint: $checkpoint_name"
                echo "Policy path: $policy_path"
                echo "Camera names: $camera_names"
                echo "Image resize modes: $image_resize_modes"
                echo "Rename map: $rename_map"
                echo "Start time: $(date)"
                echo "========================================"
                echo ""

                echo "Command:"
                echo "$eval_cmd"
                echo ""

                eval "$eval_cmd"

                echo ""
                echo "========================================"
                echo "End time: $(date)"
                echo "========================================"
            } 2>&1 | tee >(grep -v "Running rollout with at most 200 steps:" > "$log_file")
        fi

        echo "Completed: $exp_name / $checkpoint_name"
        echo ""
        first_done=true

        # Pause between evaluations to let the system rest
        if [ "$DRY_RUN" = false ]; then
            echo "GPU memory usage after eval:"
            nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
            # Kill any orphaned pybullet/python processes that might hold GPU memory
            echo "Pausing for 5 seconds..."
            sleep 5
            echo "GPU memory usage after pause:"
            nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
        fi
    done
done

echo "========================================"
echo "All evaluations complete!"
echo "========================================"

# Generate summary charts and tables
if [ "$DRY_RUN" = false ]; then
    echo ""
    echo "Generating evaluation summary..."
    python3 /home/jennyw2/code/lerobot/my_scripts/summarize_evals.py "$EVAL_OUTPUT_DIR"
fi
