#!/bin/bash

# Script to run lerobot-eval on all policy checkpoints
# Handles diffusion_approach_lever_* and pi05_training_approach_lever_* folders
#
# Usage: ./run_all_evals.sh [--dry-run] [--list]
#   --dry-run: Show commands without executing them
#   --list:    Only list experiments and checkpoints to evaluate, then exit

# Parse arguments
DRY_RUN=false
LIST_ONLY=false
for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --list)
            LIST_ONLY=true
            shift
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
EXP_PATTERNS=("$OUTPUTS_DIR"/training/diffusion_approach_lever_* "$OUTPUTS_DIR"/training/pi05_approach_lever_*)

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

    # Count checkpoints (excluding last numerical which equals 'last')
    last_numerical=$(ls -1 "$checkpoints_dir" | grep -E '^[0-9]+$' | sort -n | tail -1)
    checkpoint_count=0
    for checkpoint_dir in "$checkpoints_dir"/*; do
        [ -d "$checkpoint_dir" ] || continue
        checkpoint_name=$(basename "$checkpoint_dir")
        [ "$checkpoint_name" == "$last_numerical" ] && continue
        [ -d "$checkpoint_dir/pretrained_model" ] || continue
        ((checkpoint_count++))
    done

    echo "  - $exp_name ($checkpoint_count checkpoints)"
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

# Function to get image resize mode from any checkpoint's train_config.json in the experiment
get_image_resize_mode_from_experiment() {
    local checkpoints_dir="$1"

    # Find the first checkpoint with a train_config.json
    for checkpoint_dir in "$checkpoints_dir"/*; do
        [ -d "$checkpoint_dir" ] || continue
        local config_file="$checkpoint_dir/pretrained_model/train_config.json"
        if [ -f "$config_file" ]; then
            local result
            result=$(python3 -c "
import json

config = json.load(open('$config_file'))

env = config.get('env')
if env and env.get('image_resize_mode'):
    print(env['image_resize_mode'])
else:
    # Default to letterbox if not specified
    print('letterbox')
" 2>/dev/null)
            echo "$result"
            return
        fi
    done

    # Fallback if no config file found
    echo 'letterbox'
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

    # Get image resize mode from the experiment's config file
    image_resize_mode=$(get_image_resize_mode_from_experiment "$checkpoints_dir")

    # Find the last numerical checkpoint to skip
    last_numerical=$(get_last_numerical_checkpoint "$checkpoints_dir")

    echo "========================================"
    echo "Processing experiment: $exp_name"
    echo "Camera names: $camera_names"
    echo "Image resize mode: $image_resize_mode"
    echo "Last numerical checkpoint (will skip): $last_numerical"
    echo "========================================"

    # Process each checkpoint
    index=0
    for checkpoint_dir in "$checkpoints_dir"/*; do
        [ -d "$checkpoint_dir" ] || continue

        # Skip first checkpoint for debugging
        if [ $index -eq 0 ]; then
            ((index++))
            continue
        fi
        ((index++))

        checkpoint_name=$(basename "$checkpoint_dir")

        # Skip the last numerical checkpoint (it's the same as 'last')
        if [ "$checkpoint_name" == "$last_numerical" ]; then
            echo "Skipping $checkpoint_name (same as 'last')"
            continue
        fi

        policy_path="$checkpoint_dir/pretrained_model"

        # Check if pretrained_model exists
        if [ ! -d "$policy_path" ]; then
            echo "Skipping $checkpoint_name: no pretrained_model directory"
            continue
        fi

        # Create unique output directory for this evaluation
        eval_subdir="$EVAL_OUTPUT_DIR/${exp_name}_${checkpoint_name}"
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
            --env.image_resize_mode=$image_resize_mode \\
            --env.fps=30 \\
            --policy.path=$policy_path \\
            --eval.n_episodes=5 \\
            --output_dir=$eval_subdir \\
            --eval.batch_size=1 \\
            --eval.use_async_envs=false"

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
                echo "Image resize mode: $image_resize_mode"
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

        # Pause between evaluations to let the system rest
        echo "Pausing for 15 seconds..."
        sleep 15
    done
done

echo "========================================"
echo "All evaluations complete!"
echo "========================================"

# Generate summary charts and tables
echo ""
echo "Generating evaluation summary..."
python3 /home/jennyw2/code/lerobot/summarize_evals.py "$EVAL_OUTPUT_DIR"
