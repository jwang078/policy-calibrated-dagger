#!/usr/bin/env bash
set -euo pipefail
# baseline_continue_training.sh
#
# Matched-compute baseline: take an existing policy checkpoint and continue
# training it on its ORIGINAL dataset for N more steps, saving + evaluating
# at regular intervals. The output is a sibling "_baseline_<N>k" training
# directory holding checkpoints at every save_freq step.
#
# This is the no-DAgger control for "is DAgger doing anything?" comparisons:
# DAgger lineage's _ft_dag<R> at step (base_steps + R*per_round_steps) gets
# compared against this baseline's checkpoint at the same total step.
#
# Why not just use dagger_orchestrate.sh --blends=1.0 (PURE_POLICY_MODE)?
# That runs the full orchestrator — reuses source intervention data
# read-only, replays the policy at ratio=1.0 to produce _blend100 rollout
# datasets for PCA viz, and only THEN does base-only finetuning. If you
# don't need the _blend100 rollouts and don't want the rerun-mode source-
# existence checks, this script is the strict minimum: one resume_training
# invocation with no DAgger plumbing.
#
# Usage:
#   bash my_scripts/baseline_continue_training.sh <base_policy_dir_or_ckpt> \
#       --extra_steps=N \
#       [--save_freq=N] [--eval_freq=N] [--scheduler.name=constant] \
#       [--env.eval_benchmark_repo_id=...] [--env.eval_benchmark_subset='[...]'] \
#       [--env.episode_length=N] [--env.terminate_on_collision=true/false] \
#       [--eval.n_episodes=N] [--seed=N] \
#       [--output_tag=SHORT_SUFFIX]    # default auto-derived from --extra_steps
#       [--no_manage_splatsim]         # default OFF — wrapper auto-launches
#                                      # SplatSim on --env.external_port. Pass
#                                      # this when you already have one up.
#       [--headless]                   # forward to launch_nodes.py (pybullet
#                                      # in DIRECT mode; ~50% less GPU memory)
#       [--dry-run]
#
# The base policy's dataset.repo_id and dataset.stats_path are inherited from
# its train_config.json — there's nothing to override for the data side.
# Everything you DO pass gets forwarded verbatim to resume_training.sh.
#
# Examples:
#   # 10K more steps on the base — matches 10 DAgger rounds of 1K each.
#   bash my_scripts/baseline_continue_training.sh \
#       outputs/training/diffusion_approach_lever_11_biasend_5path_grip0_delta_basewrist \
#       --extra_steps=10000 \
#       --save_freq=1000 --eval_freq=1000 \
#       --eval.n_episodes=30 --seed=0 \
#       --env.terminate_on_collision=true --env.episode_length=1000 \
#       --env.eval_benchmark_repo_id=JennyWWW/eval_splatsim_approach_lever_benchmark_1000

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_splatsim_manage.sh
source "$SCRIPT_DIR/lib_splatsim_manage.sh"

if (( $# < 1 )); then
    sed -n '1,/^SCRIPT_DIR=/p' "$0" | grep '^#' | sed 's/^# \?//'
    exit 1
fi
BASE_POLICY="$1"; shift

EXTRA_STEPS=""
OUTPUT_TAG=""
DRY_RUN=false
# Sim management. Default ON because lerobot-train's inline eval (fired at
# every --eval_freq) needs a SplatSim ZMQ server on --env.external_port —
# without one, eval hangs at 0/N batches indefinitely. dagger_orchestrate.sh
# auto-launches its own; we mirror that. Pass --no_manage_splatsim when you
# already have a sim running (e.g. you launched launch_nodes.py in another
# shell, or this is the second baseline against a shared sim).
MANAGE_SPLATSIM=true
HEADLESS_LOCAL=false
# Pass-through args go straight to resume_training.sh.
PASSTHROUGH=()
while (( $# > 0 )); do
    case "$1" in
        --extra_steps=*)        EXTRA_STEPS="${1#*=}" ;;
        --extra_steps)          shift; EXTRA_STEPS="$1" ;;
        --output_tag=*)         OUTPUT_TAG="${1#*=}" ;;
        --output_tag)           shift; OUTPUT_TAG="$1" ;;
        --no_manage_splatsim)   MANAGE_SPLATSIM=false ;;
        --manage_splatsim)      MANAGE_SPLATSIM=true ;;
        --headless)             HEADLESS_LOCAL=true ;;
        --dry-run)              DRY_RUN=true ;;
        -h|--help)
            sed -n '1,/^SCRIPT_DIR=/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        *)                      PASSTHROUGH+=( "$1" ) ;;
    esac
    shift
done

if [[ -z "$EXTRA_STEPS" ]]; then
    echo "ERROR: --extra_steps=N is required." >&2
    exit 1
fi
if ! [[ "$EXTRA_STEPS" =~ ^[0-9]+$ ]] || (( EXTRA_STEPS <= 0 )); then
    echo "ERROR: --extra_steps must be a positive integer (got '$EXTRA_STEPS')." >&2
    exit 1
fi

# Resolve BASE_POLICY to an experiment dir + a train_config.json.
# Accept any of: experiment dir, checkpoints/last, checkpoints/NNNNNN,
# checkpoints/last/pretrained_model. Mirror resume_training.sh's tolerant
# auto-resolve so users can paste the same path either script accepts.
_train_cfg=""
if [[ -d "$BASE_POLICY/checkpoints/last/pretrained_model" ]]; then
    EXP_DIR="$BASE_POLICY"
    _train_cfg="$BASE_POLICY/checkpoints/last/pretrained_model/train_config.json"
elif [[ -f "$BASE_POLICY/train_config.json" ]]; then
    _train_cfg="$BASE_POLICY/train_config.json"
    EXP_DIR=$(dirname "$(dirname "$(dirname "$BASE_POLICY")")")
elif [[ -d "$BASE_POLICY/pretrained_model" ]]; then
    _train_cfg="$BASE_POLICY/pretrained_model/train_config.json"
    EXP_DIR=$(dirname "$(dirname "$BASE_POLICY")")
else
    echo "ERROR: can't resolve a train_config.json from '$BASE_POLICY'." >&2
    echo "  Tried: \$P/checkpoints/last/pretrained_model, \$P/train_config.json, \$P/pretrained_model" >&2
    exit 1
fi
if [[ ! -f "$_train_cfg" ]]; then
    echo "ERROR: train_config.json not found at '$_train_cfg'." >&2
    exit 1
fi

# Pull the source step count + dataset bits out of the base config. No
# user-facing overrides for these — the whole point is "same dataset, more
# training". If you want a different dataset, you want lerobot-train from
# scratch, not this script.
_meta=$(python3 -c "
import json
d = json.load(open('$_train_cfg'))
ds = d.get('dataset') or {}
print(d.get('steps') or 0)
print(ds.get('repo_id') or '')
print(ds.get('stats_path') or '')
")
BASE_STEPS=$(printf '%s\n' "$_meta" | sed -n '1p')
BASE_DS_REPO=$(printf '%s\n' "$_meta" | sed -n '2p')
BASE_DS_STATS=$(printf '%s\n' "$_meta" | sed -n '3p')

if (( BASE_STEPS <= 0 )); then
    echo "ERROR: base train_config has steps=$BASE_STEPS (expected >0)." >&2
    exit 1
fi
TOTAL_STEPS=$((BASE_STEPS + EXTRA_STEPS))

# Output dir: <exp_basename>_baseline_<extra>k. Lives as a SIBLING of the
# base experiment dir so dagger_progress.sh's --filter won't accidentally
# scoop it into a lineage (since it has no _dag<N> suffix).
EXP_NAME=$(basename "$EXP_DIR")
TRAINING_ROOT=$(dirname "$EXP_DIR")
if [[ -z "$OUTPUT_TAG" ]]; then
    # Format extra_steps as "Nk" when round, else "Nsteps".
    if (( EXTRA_STEPS % 1000 == 0 )); then
        OUTPUT_TAG="baseline_$((EXTRA_STEPS / 1000))k"
    else
        OUTPUT_TAG="baseline_${EXTRA_STEPS}steps"
    fi
fi
OUTPUT_DIR="$TRAINING_ROOT/${EXP_NAME}_${OUTPUT_TAG}"

# Scan PASSTHROUGH for the two flags lib_splatsim_manage.sh needs to launch
# a sim: --env.external_port and --env.eval_benchmark_repo_id. They're how
# lerobot-train's inline eval connects to the simulator; we need them to
# match the sim we're about to launch. Falls back to lib defaults if absent.
# We DON'T modify PASSTHROUGH — it still gets forwarded verbatim. We just
# read the values for sim-launch.
SIM_PORT=""
SIM_BENCHMARK=""
for _p in "${PASSTHROUGH[@]}"; do
    case "$_p" in
        --env.external_port=*)         SIM_PORT="${_p#*=}" ;;
        --env.eval_benchmark_repo_id=*) SIM_BENCHMARK="${_p#*=}" ;;
    esac
done

# Diagnostics block — surface everything that will get sent to
# resume_training.sh so the user can sanity-check before the long-running
# train kicks off.
echo "════════════════════════════════════════════════════════════════"
echo "Matched-compute baseline (no DAgger):"
echo "════════════════════════════════════════════════════════════════"
echo "  Base experiment dir:  $EXP_DIR"
echo "  Base policy ckpt:     $EXP_DIR/checkpoints/last/pretrained_model"
echo "  Base trained to step: $BASE_STEPS"
echo "  + extra steps:        $EXTRA_STEPS"
echo "  → total target step:  $TOTAL_STEPS"
echo ""
echo "  Dataset (inherited):  $BASE_DS_REPO"
echo "  Stats path:           $BASE_DS_STATS"
echo ""
echo "  Output dir:           $OUTPUT_DIR"
echo "  Output tag:           $OUTPUT_TAG"
echo "  job_name:             ${EXP_NAME}_${OUTPUT_TAG}"
echo ""
echo "  Pass-through to resume_training.sh:"
for p in "${PASSTHROUGH[@]}"; do echo "    $p"; done
echo "════════════════════════════════════════════════════════════════"

# Build the resume_training.sh command. We always force:
#   --steps=<total>     so it knows when to stop
#   --policy.repo_id    matches the new output dir name
#   --output_dir / --job_name   sibling of EXP_DIR, not in-place
#   --policy.push_to_hub=false  matched-compute baseline isn't a hub artifact
#                               (user can flip later if they want to publish)
# Everything else is either inherited from train_config.json (by
# resume_training.sh) or whatever the user passed in PASSTHROUGH.
CMD=(
    bash "$SCRIPT_DIR/resume_training.sh"
    "$EXP_DIR/checkpoints/last/pretrained_model"
    "--steps=$TOTAL_STEPS"
    "--policy.repo_id=${EXP_NAME}_${OUTPUT_TAG}"
    "--output_dir=$OUTPUT_DIR"
    "--job_name=${EXP_NAME}_${OUTPUT_TAG}"
    "--policy.push_to_hub=false"
    "${PASSTHROUGH[@]}"
)

if [[ "$DRY_RUN" == true ]]; then
    if [[ "$MANAGE_SPLATSIM" == true ]]; then
        echo "[DRY-RUN] Would start SplatSim on port ${SIM_PORT:-6001} (benchmark=${SIM_BENCHMARK:-<lib default>})"
        echo "[DRY-RUN] Then invoke:"
    else
        echo "[DRY-RUN] (--no_manage_splatsim — assumes sim already on port ${SIM_PORT:-6001})"
        echo "[DRY-RUN] Would invoke:"
    fi
    printf '  %s \\\n' "${CMD[@]}" | sed '$ s/ \\$//'
    exit 0
fi

# Output dir clash guard. resume_training.sh would happily overwrite — but
# for a baseline run this almost certainly means the user is re-running the
# same command and would lose their prior baseline checkpoints. Bail with
# instructions rather than silently clobbering.
if [[ -d "$OUTPUT_DIR" ]]; then
    echo "WARN: output dir already exists: $OUTPUT_DIR" >&2
    echo "      Move/delete it, OR pass --output_tag=<different> to make a new one." >&2
    echo "      (resume_training.sh would otherwise overwrite checkpoints there.)" >&2
    exit 1
fi

# Launch SplatSim before the train run if managed. The script can't `exec`
# resume_training.sh anymore (we lose the EXIT trap if we do — exec replaces
# the process and post-exec cleanup never runs). So invoke as a child + wait,
# and let the EXIT trap handle sim teardown on either success, failure, or
# Ctrl-C.
if [[ "$MANAGE_SPLATSIM" == true ]]; then
    if [[ -n "$SIM_PORT" ]]; then
        ENV_EXTERNAL_PORT="$SIM_PORT"
    fi
    if [[ -n "$SIM_BENCHMARK" ]]; then
        EVAL_BENCHMARK_REPO_ID="$SIM_BENCHMARK"
    fi
    if [[ -z "${EVAL_BENCHMARK_REPO_ID:-}" ]]; then
        echo "ERROR: --env.eval_benchmark_repo_id is required when --manage_splatsim is on." >&2
        echo "  Pass it in the pass-through args, or use --no_manage_splatsim if you'll launch the sim yourself." >&2
        exit 1
    fi
    HEADLESS="$HEADLESS_LOCAL"
    # Co-locate the sim log with this baseline's output dir so it gets
    # wiped together with the rest of the run's artifacts on cleanup.
    MANAGED_SIM_LOG_DIR="$OUTPUT_DIR/dagger"
    export ENV_EXTERNAL_PORT EVAL_BENCHMARK_REPO_ID HEADLESS MANAGED_SIM_LOG_DIR
    trap 'splat_stop_sim || true' EXIT
    splat_start_sim
fi

"${CMD[@]}"
