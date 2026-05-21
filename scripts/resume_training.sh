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
#   --scheduler.name=NAME      Override the top-level cfg.scheduler.name (only
#                              applies to schedulers that expose a `name` field,
#                              i.e. the `diffuser` HF-diffusers wrapper). Use
#                              `constant` to flatten the LR at peak for a
#                              diffusion finetune — the HF cosine scheduler
#                              decays to 0 by end-of-training and is useless
#                              for resumes past the original step count.
#   --dataset.repo_id=ID       Override the dataset to finetune on. Usually paired
#                              with --dataset.stats_path (the inherited stats path
#                              from train_config.json is from the prior dataset
#                              and almost certainly wrong for a new one).
#   --dataset.stats_path=PATH  Override the sidecar relative-action stats path.
#                              Required when changing --dataset.repo_id under a
#                              policy that uses relative-action normalization.
#   --env.external_port=N      Connect lerobot-train's inline eval to an external
#                              SplatSim ZMQ server at this port (e.g. 6001) instead
#                              of spawning a new one. Required when running this
#                              alongside other GPU-hungry processes — see the
#                              dagger_orchestrate.sh shared-sim setup.
#   --env.eval_benchmark_repo_id=ID
#                              Override the inline eval's benchmark dataset.
#                              Resumed configs sometimes have this unset (e.g.
#                              checkpoints trained before benchmark eval was
#                              wired up), causing inline eval to fall back to
#                              random scenarios. Pass this to lock eval to the
#                              benchmark set.
#   --env.eval_benchmark_subset=JSON
#                              Restrict inline eval to a specific subset of
#                              benchmark episodes, e.g. "[0,1,2,3,4]". Pair
#                              with --env.eval_benchmark_repo_id for fixed
#                              round-over-round eval scenarios.
#   --policy.repo_id=ID        Rename the resumed run's policy.repo_id. Useful for
#                              tagging a finetune as a NEW training run (e.g. an
#                              _ft suffix) so it doesn't write back into the
#                              original training dir.
#   --output_dir=PATH          Redirect the resumed run's output to a new dir.
#                              Usually paired with --policy.repo_id and --job_name
#                              when creating a finetune-distinguished training dir.
#   --job_name=STR             Rename the wandb job. Pair with --output_dir.
#   --dry-run                  Print the command without executing.
#
# Example (basic, extend an existing run):
#   bash my_scripts/resume_training.sh \
#       outputs/training/pi05_approach_lever_11_biasend_5path_grip0_abs_basewrist \
#       --steps=50000 --eval_freq=2000 --save_freq=2000
#
# Example (finetune on a new merged DAgger dataset into a NEW _ft training dir):
#   bash my_scripts/resume_training.sh \
#       outputs/training/pi05_xyz \
#       --dataset.repo_id=JennyWWW/splatsim_xyz_dag1_merged \
#       --dataset.stats_path=~/code/lerobot/outputs/dataset_stats/xyz_dag1_merged/stats_rel50.json \
#       --policy.repo_id=pi05_xyz_dag1_merged_ft_delta_basewrist \
#       --output_dir=outputs/training/pi05_xyz_dag1_merged_ft_delta_basewrist \
#       --job_name=pi05_xyz_dag1_merged_ft_delta_basewrist \
#       --env.external_port=6001 \
#       --steps=4000 --eval_freq=2000 --save_freq=2000

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

# Migrate legacy SplatSimEnv field `use_fisheye_wrist_camera` (bool) →
# `wrist_cam_ver` (int) so configs saved before the refactor still load.
# Idempotent and a no-op if the field is absent. See
# splatsim/robots/sim_robot_pybullet_base.py:WRIST_CAM_FISHEYE_CALIBRATIONS.
python3 - "$CONFIG_PATH" <<'PY'
import json, sys
path = sys.argv[1]
with open(path) as f:
    cfg = json.load(f)
env = cfg.get("env") or {}
if "use_fisheye_wrist_camera" in env:
    old = env.pop("use_fisheye_wrist_camera")
    env.setdefault("wrist_cam_ver", 1 if old else 0)
    cfg["env"] = env
    with open(path, "w") as f:
        json.dump(cfg, f, indent=4)
    print(f"  Migrated env.use_fisheye_wrist_camera={old} → env.wrist_cam_ver={env['wrist_cam_ver']} in {path}")
PY

# ── parse options ────────────────────────────────────────────────────────────
STEPS=""
EVAL_FREQ=""
SAVE_FREQ=""
SCHEDULER_DECAY_STEPS=""
# decay_lr is the FLOOR of the cosine decay (the LR the scheduler holds
# forever after num_decay_steps). When resuming past the decay end, runtime LR
# is parked at this floor. Set this equal to the optimizer's peak LR to force
# constant-peak-LR behavior throughout the finetune, which is what you want
# when 200-2000 finetune steps at the (much smaller) cosine floor would
# barely move the model.
SCHEDULER_DECAY_LR=""
SCHEDULER_NAME=""
DATASET_REPO_ID=""
DATASET_STATS_PATH=""
ENV_EXTERNAL_PORT=""
ENV_EVAL_BENCHMARK_REPO_ID=""
ENV_EVAL_BENCHMARK_SUBSET=""
POLICY_REPO_ID=""
POLICY_PUSH_TO_HUB=""   # empty = inherit from train_config.json
OUTPUT_DIR=""
JOB_NAME=""
BATCH_SIZE=""           # empty = inherit from train_config.json
DRY_RUN=false

for arg in "$@"; do
    case "$arg" in
        --steps=*)                  STEPS="${arg#*=}" ;;
        --eval_freq=*)              EVAL_FREQ="${arg#*=}" ;;
        --save_freq=*)              SAVE_FREQ="${arg#*=}" ;;
        --scheduler_decay_steps=*)  SCHEDULER_DECAY_STEPS="${arg#*=}" ;;
        --scheduler_decay_lr=*)     SCHEDULER_DECAY_LR="${arg#*=}" ;;
        --scheduler.name=*)         SCHEDULER_NAME="${arg#*=}" ;;
        --dataset.repo_id=*)        DATASET_REPO_ID="${arg#*=}" ;;
        --dataset.stats_path=*)     DATASET_STATS_PATH="${arg#*=}" ;;
        --env.external_port=*)      ENV_EXTERNAL_PORT="${arg#*=}" ;;
        --env.eval_benchmark_repo_id=*) ENV_EVAL_BENCHMARK_REPO_ID="${arg#*=}" ;;
        --env.eval_benchmark_subset=*) ENV_EVAL_BENCHMARK_SUBSET="${arg#*=}" ;;
        --policy.repo_id=*)         POLICY_REPO_ID="${arg#*=}" ;;
        --policy.push_to_hub=*)     POLICY_PUSH_TO_HUB="${arg#*=}" ;;
        --output_dir=*)             OUTPUT_DIR="${arg#*=}" ;;
        --job_name=*)               JOB_NAME="${arg#*=}" ;;
        --batch_size=*)             BATCH_SIZE="${arg#*=}" ;;
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

# Defensive check: switching --dataset.repo_id without also overriding
# --dataset.stats_path inherits the prior dataset's stats path from
# train_config.json, which yields wrong relative-action normalization on the
# new dataset (silent degradation, not a crash). Warn loudly. Don't fail —
# user might do this intentionally for absolute-action policies (ACT) where
# the sidecar stats aren't used.
if [[ -n "$DATASET_REPO_ID" && -z "$DATASET_STATS_PATH" ]]; then
    INHERITED_STATS_PATH="$(python3 -c "
import json, sys
try:
    cfg = json.load(open(sys.argv[1]))
    print(cfg.get('dataset', {}).get('stats_path') or '')
except Exception:
    print('')
" "$CONFIG_PATH" 2>/dev/null)"
    if [[ -n "$INHERITED_STATS_PATH" ]]; then
        echo "" >&2
        echo "⚠  WARNING: --dataset.repo_id is set but --dataset.stats_path is not." >&2
        echo "   The inherited dataset.stats_path from train_config.json is:" >&2
        echo "     $INHERITED_STATS_PATH" >&2
        echo "   That sidecar was computed for the PRIOR dataset and is almost" >&2
        echo "   certainly wrong for $DATASET_REPO_ID. Pass --dataset.stats_path" >&2
        echo "   explicitly to override (e.g. point at the new dataset's sidecar)." >&2
        echo "   Continuing anyway — only ignore this if you're finetuning an" >&2
        echo "   absolute-action policy (ACT) that doesn't use the sidecar." >&2
        echo "" >&2
    fi
fi

# ── build command ────────────────────────────────────────────────────────────
# scheduler_decay_steps / scheduler_decay_lr are pi05/pi0-specific fields.
# Diffusion uses HF diffusers' cosine scheduler (no decay_lr concept); ACT
# uses a flat LR. For those, the --policy.scheduler_decay_* overrides would
# raise a draccus DecodingError ("fields not valid for DiffusionConfig").
# Probe the resumed config to determine which fields are supported and emit
# overrides accordingly.
POLICY_SUPPORTS_DECAY_STEPS="$(python3 -c "
import json, sys
try:
    c = json.load(open(sys.argv[1]))
    print('true' if 'scheduler_decay_steps' in c.get('policy', {}) else 'false')
except Exception:
    print('false')
" "$CONFIG_PATH" 2>/dev/null)"
POLICY_SUPPORTS_DECAY_LR="$(python3 -c "
import json, sys
try:
    c = json.load(open(sys.argv[1]))
    print('true' if 'scheduler_decay_lr' in c.get('policy', {}) else 'false')
except Exception:
    print('false')
" "$CONFIG_PATH" 2>/dev/null)"

CMD="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True lerobot-train"
CMD="$CMD --resume=true"
CMD="$CMD --config_path=$CONFIG_PATH"
[[ -n "$STEPS" ]]                  && CMD="$CMD --steps=$STEPS"
[[ -n "$EVAL_FREQ" ]]              && CMD="$CMD --eval_freq=$EVAL_FREQ"
[[ -n "$SAVE_FREQ" ]]              && CMD="$CMD --save_freq=$SAVE_FREQ"
if [[ -n "$SCHEDULER_DECAY_STEPS" ]]; then
    if [[ "$POLICY_SUPPORTS_DECAY_STEPS" == "true" ]]; then
        CMD="$CMD --policy.scheduler_decay_steps=$SCHEDULER_DECAY_STEPS"
    else
        echo "  Skipping --policy.scheduler_decay_steps=$SCHEDULER_DECAY_STEPS (policy has no scheduler_decay_steps field)."
    fi
fi
# On resume, cfg.scheduler is already populated from the loaded train_config.json
# and the `use_policy_training_preset` rebuild-from-policy branch does NOT fire,
# so overriding --policy.scheduler_decay_lr saves the new value into the policy
# config but the *runtime* scheduler keeps the original decay_lr. Override BOTH
# (policy field for consistency on the next save; top-level cfg.scheduler.decay_lr
# is the one that actually drives the runtime LR schedule).
if [[ -n "$SCHEDULER_DECAY_LR" ]]; then
    if [[ "$POLICY_SUPPORTS_DECAY_LR" == "true" ]]; then
        CMD="$CMD --policy.scheduler_decay_lr=$SCHEDULER_DECAY_LR --scheduler.decay_lr=$SCHEDULER_DECAY_LR"
    else
        echo "  Skipping --policy.scheduler_decay_lr=$SCHEDULER_DECAY_LR (policy has no scheduler_decay_lr field; diffusion/act use a different scheduler API)."
    fi
fi
[[ -n "$SCHEDULER_NAME" ]] && CMD="$CMD --scheduler.name=$SCHEDULER_NAME"
[[ -n "$DATASET_REPO_ID" ]]        && CMD="$CMD --dataset.repo_id=$DATASET_REPO_ID"
[[ -n "$DATASET_STATS_PATH" ]]     && CMD="$CMD --dataset.stats_path=$DATASET_STATS_PATH"
[[ -n "$ENV_EXTERNAL_PORT" ]]      && CMD="$CMD --env.external_port=$ENV_EXTERNAL_PORT"
[[ -n "$ENV_EVAL_BENCHMARK_REPO_ID" ]] && CMD="$CMD --env.eval_benchmark_repo_id=$ENV_EVAL_BENCHMARK_REPO_ID"
[[ -n "$ENV_EVAL_BENCHMARK_SUBSET" ]] && CMD="$CMD --env.eval_benchmark_subset=$ENV_EVAL_BENCHMARK_SUBSET"
[[ -n "$POLICY_REPO_ID" ]]         && CMD="$CMD --policy.repo_id=$POLICY_REPO_ID"
[[ -n "$POLICY_PUSH_TO_HUB" ]]     && CMD="$CMD --policy.push_to_hub=$POLICY_PUSH_TO_HUB"
[[ -n "$OUTPUT_DIR" ]]             && CMD="$CMD --output_dir=$OUTPUT_DIR"
[[ -n "$JOB_NAME" ]]               && CMD="$CMD --job_name=$JOB_NAME"
[[ -n "$BATCH_SIZE" ]]             && CMD="$CMD --batch_size=$BATCH_SIZE"

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
[[ -n "$SCHEDULER_DECAY_LR" ]] && echo "Override:      --policy.scheduler_decay_lr=$SCHEDULER_DECAY_LR --scheduler.decay_lr=$SCHEDULER_DECAY_LR"
[[ -n "$SCHEDULER_NAME" ]]     && echo "Override:      --scheduler.name=$SCHEDULER_NAME"
[[ -n "$EVAL_FREQ" ]]          && echo "Override:      --eval_freq=$EVAL_FREQ"
[[ -n "$SAVE_FREQ" ]]          && echo "Override:      --save_freq=$SAVE_FREQ"
[[ -n "$DATASET_REPO_ID" ]]    && echo "Override:      --dataset.repo_id=$DATASET_REPO_ID"
[[ -n "$DATASET_STATS_PATH" ]] && echo "Override:      --dataset.stats_path=$DATASET_STATS_PATH"
[[ -n "$ENV_EXTERNAL_PORT" ]]  && echo "Override:      --env.external_port=$ENV_EXTERNAL_PORT"
[[ -n "$ENV_EVAL_BENCHMARK_REPO_ID" ]] && echo "Override:      --env.eval_benchmark_repo_id=$ENV_EVAL_BENCHMARK_REPO_ID"
[[ -n "$ENV_EVAL_BENCHMARK_SUBSET" ]] && echo "Override:      --env.eval_benchmark_subset=$ENV_EVAL_BENCHMARK_SUBSET"
[[ -n "$POLICY_REPO_ID" ]]     && echo "Override:      --policy.repo_id=$POLICY_REPO_ID"
[[ -n "$POLICY_PUSH_TO_HUB" ]] && echo "Override:      --policy.push_to_hub=$POLICY_PUSH_TO_HUB"
[[ -n "$OUTPUT_DIR" ]]         && echo "Override:      --output_dir=$OUTPUT_DIR"
[[ -n "$JOB_NAME" ]]           && echo "Override:      --job_name=$JOB_NAME"
[[ -n "$BATCH_SIZE" ]]         && echo "Override:      --batch_size=$BATCH_SIZE"
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
