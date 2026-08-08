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
#   --num_workers=N            DataLoader worker processes. Default 16 (lerobot's
#                              own default of 4 starves the GPU on a many-core
#                              box). Empty (--num_workers=) inherits the value
#                              stored in the resumed train_config.json.
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
        # Order matters: lerobot-train derives checkpoint_path / pretrained_path
        # from the config file's LOCATION, so we must resolve to the config that
        # sits inside the checkpoint tree. The experiment root also holds a
        # train_config.json (pre-training snapshot written by lerobot_train.py
        # at run start) — it is NOT resumable (no model.safetensors next to it),
        # so it's the LAST fallback, only hit when the input dir has no
        # checkpoint-shaped layout.
        for candidate in \
            "$input/checkpoints/last/pretrained_model/train_config.json" \
            "$input/pretrained_model/train_config.json" \
            "$input/train_config.json"
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
# Tri-state: empty string + "not explicit" → omit the flag entirely (resumed
# train_config.json's repo_id wins). Empty string + "explicit" → forward as
# `--dataset.repo_id=` so it overrides the loaded value with empty, which is
# exactly what weighted-multi-dataset mode needs (the loaded repo_id would
# otherwise collide with --dataset.repo_ids via the mutual-exclusion check
# in TrainPipelineConfig.validate). Non-empty → forward normally.
DATASET_REPO_ID_EXPLICIT=false
DATASET_STATS_PATH=""
# Same tri-state as DATASET_REPO_ID above (see comment there). Needed so
# weighted-sampling-mode finetune can pass `--dataset.stats_path=` (empty)
# to CLEAR the value baked into the resumed train_config.json. Without this,
# the inherited stats_path triggers `lerobot_train.py`'s
# `Overriding dataset stats from ...` clobber AFTER the wrapper's
# normalization mode has already exposed the right stats — silently
# turning every "norm_mode=aggregated" run into effective "base_only".
# Empty + explicit → forward as `--dataset.stats_path=`, which the
# truthy check at lerobot_train.py:254 now treats as "don't override".
DATASET_STATS_PATH_EXPLICIT=false
ENV_EXTERNAL_PORT=""
ENV_EVAL_BENCHMARK_REPO_ID=""
ENV_EVAL_BENCHMARK_SUBSET=""
POLICY_REPO_ID=""
POLICY_PUSH_TO_HUB=""   # empty = inherit from train_config.json
OUTPUT_DIR=""
JOB_NAME=""
BATCH_SIZE=""           # empty = inherit from train_config.json
# DataLoader workers (lerobot-train --num_workers). lerobot's own default is 4,
# which underutilizes a many-core CPU and starves the GPU during batch assembly.
# Default 16 here (measured optimum on a 24-core box; 20+ regresses); it is emitted BEFORE the unknown-arg
# passthrough, so an explicit --num_workers inside a passthrough string still
# wins (draccus last-wins). Set empty (--num_workers=) to inherit whatever the
# resumed train_config.json holds.
NUM_WORKERS=16
# Video decoder (lerobot-train --dataset.video_backend). A train_config.json
# saved while torchcodec was unloadable records `video_backend: pyav`, and a
# resume inherits that pin forever — so default it explicitly here. torchcodec
# decodes ~4.8x faster than pyav on these AV1 datasets (bit-identical output)
# and caches decoders instead of reopening the container per call. Emitted
# BEFORE the unknown-arg passthrough, so an explicit --dataset.video_backend
# in a passthrough string still wins (draccus last-wins). Set empty
# (--video_backend=) to inherit whatever the resumed train_config.json holds.
VIDEO_BACKEND=torchcodec
DRY_RUN=false
# Unknown --key=value args are passed through to lerobot-train verbatim. Lets
# callers (e.g. dagger_orchestrate.sh's --finetune_extra_args) override any
# lerobot-train flag without this script having to enumerate them. Bare
# positional args or unknown --flag (without =) still error so typos are caught.
EXTRA_LEROBOT_TRAIN_ARGS=()
# --exclude_gripper_from_state: mirror of train_sweep.sh's flag. Resolved
# below (after we've inspected the checkpoint's train_config.json to know
# env.num_dofs / env.state_dim) into a concrete
# --observation_dim_slice='{"observation.state": [0,...,STATE_DIM-1] minus [NUM_DOFS]}'
# forwarded to lerobot-train. Applied whether or not the checkpoint's
# saved pipeline already carries a SelectObservationDimsProcessorStep
# (factory's post-hoc block inserts/replaces uniformly).
EXCLUDE_GRIPPER_FROM_STATE=false

for arg in "$@"; do
    case "$arg" in
        --steps=*)                  STEPS="${arg#*=}" ;;
        --eval_freq=*)              EVAL_FREQ="${arg#*=}" ;;
        --save_freq=*)              SAVE_FREQ="${arg#*=}" ;;
        --scheduler_decay_steps=*)  SCHEDULER_DECAY_STEPS="${arg#*=}" ;;
        --scheduler_decay_lr=*)     SCHEDULER_DECAY_LR="${arg#*=}" ;;
        --scheduler.name=*)         SCHEDULER_NAME="${arg#*=}" ;;
        --dataset.repo_id=*)        DATASET_REPO_ID="${arg#*=}"; DATASET_REPO_ID_EXPLICIT=true ;;
        --dataset.stats_path=*)     DATASET_STATS_PATH="${arg#*=}"; DATASET_STATS_PATH_EXPLICIT=true ;;
        --env.external_port=*)      ENV_EXTERNAL_PORT="${arg#*=}" ;;
        --env.eval_benchmark_repo_id=*) ENV_EVAL_BENCHMARK_REPO_ID="${arg#*=}" ;;
        --env.eval_benchmark_subset=*) ENV_EVAL_BENCHMARK_SUBSET="${arg#*=}" ;;
        --policy.repo_id=*)         POLICY_REPO_ID="${arg#*=}" ;;
        --policy.push_to_hub=*)     POLICY_PUSH_TO_HUB="${arg#*=}" ;;
        --output_dir=*)             OUTPUT_DIR="${arg#*=}" ;;
        --job_name=*)               JOB_NAME="${arg#*=}" ;;
        --batch_size=*)             BATCH_SIZE="${arg#*=}" ;;
        --num_workers=*)            NUM_WORKERS="${arg#*=}" ;;
        --video_backend=*)          VIDEO_BACKEND="${arg#*=}" ;;
        --dry-run)                  DRY_RUN=true ;;
        --exclude_gripper_from_state)         EXCLUDE_GRIPPER_FROM_STATE=true ;;
        --exclude_gripper_from_state=*)       EXCLUDE_GRIPPER_FROM_STATE="${arg#*=}" ;;
        --*=*)                      EXTRA_LEROBOT_TRAIN_ARGS+=( "$arg" ) ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

# Resolve --exclude_gripper_from_state → --observation_dim_slice=... by
# reading env.num_dofs + env.state_dim from the checkpoint's train_config
# (so we know where the gripper sits in observation.state and how wide the
# state vector is). Applied uniformly regardless of whether the resumed
# checkpoint already trained with the flag — factory.py's post-hoc block
# in make_pre_post_processors handles the insert/replace.
case "$EXCLUDE_GRIPPER_FROM_STATE" in
    true|false) ;;
    *) echo "ERROR: --exclude_gripper_from_state must be true or false (got '$EXCLUDE_GRIPPER_FROM_STATE')." >&2; exit 1 ;;
esac
if [[ "$EXCLUDE_GRIPPER_FROM_STATE" == "true" ]]; then
    _slice_json="$(python3 -c "
import json, sys
c = json.load(open('$CONFIG_PATH'))
env = c.get('env') or {}
num_dofs = env.get('num_dofs')
state_dim = env.get('state_dim')
if num_dofs is None or state_dim is None:
    sys.exit(f'ERROR: env.num_dofs / env.state_dim missing from {sys.argv[0]}\\'s train_config; cannot resolve slice.')
if state_dim <= num_dofs:
    sys.exit(f'ERROR: --exclude_gripper_from_state=true requires state_dim ({state_dim}) > num_dofs ({num_dofs}); no gripper dim to drop.')
keep = [i for i in range(state_dim) if i != num_dofs]
print(json.dumps({'observation.state': keep}))
")" || { echo "$_slice_json" >&2; exit 1; }
    EXTRA_LEROBOT_TRAIN_ARGS+=( --observation_dim_slice="$_slice_json" )
fi

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

# Build the lerobot-train invocation as a bash array (NOT a flat string).
# This preserves quoting on values that contain shell metacharacters — e.g.
# the bracketed JSON lists passed for weighted-multi-dataset training:
#   --dataset.repo_ids=["A","B"]
# The earlier flat-string + `eval "$CMD"` form word-split on the spaces inside
# such values and stripped the inner double-quotes, which broke draccus's
# JSON-mode list parser (the `--config_path=...json` path activates
# `draccus.config_type("json")` which requires `["A","B"]` not `[A,B]`).
# Array form sidesteps both problems: argv elements pass through exec
# unmolested, no eval, no word-splitting.
#
# PYTORCH_CUDA_ALLOC_CONF is exported so it sticks for the lerobot-train
# child without needing a separate `env` wrapper around the array exec.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CMD_ARGS=( lerobot-train --resume=true --config_path="$CONFIG_PATH" )
[[ -n "$STEPS" ]]                  && CMD_ARGS+=( --steps="$STEPS" )
# Upstream renamed --eval_freq → --env_eval_freq (env-eval cadence) and added
# a separate --eval_steps for dataset-eval. We only care about env-eval.
[[ -n "$EVAL_FREQ" ]]              && CMD_ARGS+=( --env_eval_freq="$EVAL_FREQ" )
[[ -n "$SAVE_FREQ" ]]              && CMD_ARGS+=( --save_freq="$SAVE_FREQ" )
if [[ -n "$SCHEDULER_DECAY_STEPS" ]]; then
    if [[ "$POLICY_SUPPORTS_DECAY_STEPS" == "true" ]]; then
        CMD_ARGS+=( --policy.scheduler_decay_steps="$SCHEDULER_DECAY_STEPS" )
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
        CMD_ARGS+=( --policy.scheduler_decay_lr="$SCHEDULER_DECAY_LR" --scheduler.decay_lr="$SCHEDULER_DECAY_LR" )
    else
        echo "  Skipping --policy.scheduler_decay_lr=$SCHEDULER_DECAY_LR (policy has no scheduler_decay_lr field; diffusion/act use a different scheduler API)."
    fi
fi
[[ -n "$SCHEDULER_NAME" ]]             && CMD_ARGS+=( --scheduler.name="$SCHEDULER_NAME" )
[[ "$DATASET_REPO_ID_EXPLICIT" == true ]] && CMD_ARGS+=( --dataset.repo_id="$DATASET_REPO_ID" )
[[ "$DATASET_STATS_PATH_EXPLICIT" == true ]] && CMD_ARGS+=( --dataset.stats_path="$DATASET_STATS_PATH" )
[[ -n "$ENV_EXTERNAL_PORT" ]]          && CMD_ARGS+=( --env.external_port="$ENV_EXTERNAL_PORT" )
[[ -n "$ENV_EVAL_BENCHMARK_REPO_ID" ]] && CMD_ARGS+=( --env.eval_benchmark_repo_id="$ENV_EVAL_BENCHMARK_REPO_ID" )
[[ -n "$ENV_EVAL_BENCHMARK_SUBSET" ]]  && CMD_ARGS+=( --env.eval_benchmark_subset="$ENV_EVAL_BENCHMARK_SUBSET" )
[[ -n "$POLICY_REPO_ID" ]]             && CMD_ARGS+=( --policy.repo_id="$POLICY_REPO_ID" )
[[ -n "$POLICY_PUSH_TO_HUB" ]]         && CMD_ARGS+=( --policy.push_to_hub="$POLICY_PUSH_TO_HUB" )
[[ -n "$OUTPUT_DIR" ]]                 && CMD_ARGS+=( --output_dir="$OUTPUT_DIR" )
[[ -n "$JOB_NAME" ]]                   && CMD_ARGS+=( --job_name="$JOB_NAME" )
[[ -n "$BATCH_SIZE" ]]                 && CMD_ARGS+=( --batch_size="$BATCH_SIZE" )
[[ -n "$NUM_WORKERS" ]]                && CMD_ARGS+=( --num_workers="$NUM_WORKERS" )
[[ -n "$VIDEO_BACKEND" ]]              && CMD_ARGS+=( --dataset.video_backend="$VIDEO_BACKEND" )
# Forward any unknown --key=value args verbatim. Note: if an override here
# duplicates one of the explicit flags above, lerobot-train uses the LAST
# value seen, so passthrough takes precedence — caller can override e.g.
# --eval_freq via the passthrough path if they really want to.
CMD_ARGS+=( "${EXTRA_LEROBOT_TRAIN_ARGS[@]}" )

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
[[ "$DATASET_REPO_ID_EXPLICIT" == true ]] && echo "Override:      --dataset.repo_id='$DATASET_REPO_ID'"
[[ "$DATASET_STATS_PATH_EXPLICIT" == true ]] && echo "Override:      --dataset.stats_path='$DATASET_STATS_PATH'"
[[ -n "$ENV_EXTERNAL_PORT" ]]  && echo "Override:      --env.external_port=$ENV_EXTERNAL_PORT"
[[ -n "$ENV_EVAL_BENCHMARK_REPO_ID" ]] && echo "Override:      --env.eval_benchmark_repo_id=$ENV_EVAL_BENCHMARK_REPO_ID"
[[ -n "$ENV_EVAL_BENCHMARK_SUBSET" ]] && echo "Override:      --env.eval_benchmark_subset=$ENV_EVAL_BENCHMARK_SUBSET"
[[ -n "$POLICY_REPO_ID" ]]     && echo "Override:      --policy.repo_id=$POLICY_REPO_ID"
[[ -n "$POLICY_PUSH_TO_HUB" ]] && echo "Override:      --policy.push_to_hub=$POLICY_PUSH_TO_HUB"
[[ -n "$OUTPUT_DIR" ]]         && echo "Override:      --output_dir=$OUTPUT_DIR"
[[ -n "$JOB_NAME" ]]           && echo "Override:      --job_name=$JOB_NAME"
[[ -n "$BATCH_SIZE" ]]         && echo "Override:      --batch_size=$BATCH_SIZE"
[[ -n "$NUM_WORKERS" ]]        && echo "Override:      --num_workers=$NUM_WORKERS"
[[ -n "$VIDEO_BACKEND" ]]      && echo "Override:      --dataset.video_backend=$VIDEO_BACKEND"
echo "================================================================"
echo
echo "Command:"
# Print the array with single-quoted args so the printed command is also
# the LITERAL command (no quote-stripping surprise for the user copying it
# into a shell). printf '%q' renders each token in shell-quoted form;
# stitching them together yields an exact, copy-pasteable command line.
printf 'PYTORCH_CUDA_ALLOC_CONF=%q' "$PYTORCH_CUDA_ALLOC_CONF"
for _a in "${CMD_ARGS[@]}"; do printf ' %q' "$_a"; done
echo
echo

if [[ "$DRY_RUN" == true ]]; then
    echo "=== DRY RUN — not executing ==="
    exit 0
fi

# Direct array exec — no eval, no flat string. Each CMD_ARGS element passes
# through as a single argv token, so quoted values like
# `--dataset.repo_ids=["A","B"]` reach lerobot-train byte-for-byte.
"${CMD_ARGS[@]}"
