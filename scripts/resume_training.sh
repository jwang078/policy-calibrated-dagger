#!/usr/bin/env bash
set -euo pipefail
# resume_training.sh
#
# Resume a lerobot-train run from a saved checkpoint, optionally extending the
# total step count. Loads the full original training config from the
# checkpoint's train_config.json so you only specify what you want to OVERRIDE.
#
# Usage:
#   bash my_scripts/resume_training.sh <checkpoint_path> [OPTIONS]
#
# checkpoint_path is auto-resolved from any of:
#   - A train_config.json file directly
#   - A pretrained_model/ dir containing one
#   - A checkpoint dir (e.g. .../checkpoints/050000)
#   - An experiment dir (e.g. .../training/pi05_xyz) — picks
#     checkpoints/last/pretrained_model/train_config.json
#
# Options:
#   --steps=N                  Override total training steps. Also bumps
#                              policy.scheduler_decay_steps to match unless
#                              --scheduler_decay_steps is set explicitly.
#   --eval_freq=N              Override eval frequency.
#   --save_freq=N              Override save frequency.
#   --scheduler_decay_steps=N  Explicit override (defaults to --steps when set).
#   --dry-run                  Print the command without executing.
#
# Example:
#   bash my_scripts/resume_training.sh \
#       outputs/training/pi05_approach_lever_11_biasend_5path_grip0_abs_basewrist \
#       --steps=50000 --eval_freq=2000 --save_freq=2000

# ── parse positional ─────────────────────────────────────────────────────────
if [[ $# -lt 1 || "$1" == --* ]]; then
    echo "Usage: $0 <checkpoint_path> [OPTIONS]" >&2
    echo "  See the header of this script for details." >&2
    exit 1
fi
CKPT_INPUT="$1"
shift

# Resolve to a train_config.json path. Accept anything in the experiment tree.
resolve_config() {
    local input="$1"
    if [[ -f "$input" ]]; then
        echo "$input"
        return
    fi
    if [[ -d "$input" ]]; then
        local candidate
        for candidate in \
            "$input/train_config.json" \
            "$input/pretrained_model/train_config.json" \
            "$input/checkpoints/last/pretrained_model/train_config.json"
        do
            if [[ -f "$candidate" ]]; then
                echo "$candidate"
                return
            fi
        done
    fi
    echo ""
}

CONFIG_PATH="$(resolve_config "$CKPT_INPUT")"
if [[ -z "$CONFIG_PATH" ]]; then
    echo "Error: could not find train_config.json under '$CKPT_INPUT'." >&2
    echo "Tried:" >&2
    echo "  - $CKPT_INPUT (as a file)" >&2
    echo "  - $CKPT_INPUT/train_config.json" >&2
    echo "  - $CKPT_INPUT/pretrained_model/train_config.json" >&2
    echo "  - $CKPT_INPUT/checkpoints/last/pretrained_model/train_config.json" >&2
    exit 1
fi
# Make absolute so the resume command works no matter where it's run from.
CONFIG_PATH="$(readlink -f "$CONFIG_PATH")"

# ── parse options ────────────────────────────────────────────────────────────
STEPS=""
EVAL_FREQ=""
SAVE_FREQ=""
SCHEDULER_DECAY_STEPS=""
DRY_RUN=false

for arg in "$@"; do
    case "$arg" in
        --steps=*)                  STEPS="${arg#*=}" ;;
        --eval_freq=*)              EVAL_FREQ="${arg#*=}" ;;
        --save_freq=*)              SAVE_FREQ="${arg#*=}" ;;
        --scheduler_decay_steps=*)  SCHEDULER_DECAY_STEPS="${arg#*=}" ;;
        --dry-run)                  DRY_RUN=true ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

# Auto-follow: when --steps is set but --scheduler_decay_steps isn't, match it.
# Reason: training schedule (cosine decay etc.) is parameterized by
# scheduler_decay_steps; if you extend training steps without bumping the
# scheduler, the LR decays before training ends and you waste compute.
if [[ -n "$STEPS" && -z "$SCHEDULER_DECAY_STEPS" ]]; then
    SCHEDULER_DECAY_STEPS="$STEPS"
fi

# ── build command ────────────────────────────────────────────────────────────
CMD="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True lerobot-train"
CMD="$CMD --resume=true"
CMD="$CMD --config_path=$CONFIG_PATH"
[[ -n "$STEPS" ]]                  && CMD="$CMD --steps=$STEPS"
[[ -n "$EVAL_FREQ" ]]              && CMD="$CMD --eval_freq=$EVAL_FREQ"
[[ -n "$SAVE_FREQ" ]]              && CMD="$CMD --save_freq=$SAVE_FREQ"
[[ -n "$SCHEDULER_DECAY_STEPS" ]]  && CMD="$CMD --policy.scheduler_decay_steps=$SCHEDULER_DECAY_STEPS"

# ── print summary ────────────────────────────────────────────────────────────
echo "================================================================"
echo "Resume Training"
echo "================================================================"
echo "Config:        $CONFIG_PATH"
if [[ -n "$STEPS" ]]; then
    echo "Override:      --steps=$STEPS  (scheduler_decay_steps=$SCHEDULER_DECAY_STEPS)"
elif [[ -n "$SCHEDULER_DECAY_STEPS" ]]; then
    echo "Override:      --policy.scheduler_decay_steps=$SCHEDULER_DECAY_STEPS"
fi
[[ -n "$EVAL_FREQ" ]] && echo "Override:      --eval_freq=$EVAL_FREQ"
[[ -n "$SAVE_FREQ" ]] && echo "Override:      --save_freq=$SAVE_FREQ"
echo "================================================================"
echo
echo "Command:"
echo "$CMD"
echo

if [[ "$DRY_RUN" == true ]]; then
    echo "=== DRY RUN — not executing ==="
    exit 0
fi

# eval is safe here because every interpolated value either comes from the
# resolved config path or from numeric option values (no shell metachars).
eval "$CMD"
