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
#   --cameras=NAME          Which IMAGE inputs to train against:
#                           "basewrist" (default) | "base" | "wrist" | "state".
#                           basewrist → both base_rgb + wrist_rgb; base / wrist →
#                             that one camera; state → NO image (oracle/state-only).
#                           observation.state (joints+gripper) is ALWAYS included.
#                           Appears as the `_<cameras>` suffix on the auto-derived
#                           run_name (with "os" appended when env_state is also
#                           consumed, e.g. _baseos), so swapping it changes the
#                           output path. Downstream callers that construct expected
#                           names (dagger_orchestrate.sh's round-0 check) mirror
#                           this derivation.
#   --include_env_state_obs=BOOL
#                           Whether the policy consumes the ORACLE input
#                           observation.environment_state (object coords). Its
#                           WIDTH is a dataset property from the env profile
#                           (ENV_STATE_DIM); this only toggles consumption.
#                           Default: true for --cameras=state (needs it), else
#                           false. Combine with --cameras for the feature modes:
#                             oracle-only   = --cameras=state
#                             vision-only   = --cameras=base
#                             vision+oracle = --cameras=base --include_env_state_obs=true
#                           One dual-purpose dataset serves all of them.
#   --env_external_port=N   Connect lerobot-train's inline eval to an external
#                           SplatSim ZMQ server at this port (e.g. 6001) instead
#                           of spawning a new one. Required when training has to
#                           share a GPU with other SplatSim consumers (e.g. the
#                           dagger_orchestrate.sh pipeline). User must launch
#                           SplatSim on that port BEFORE invoking this script.
#   --num_workers=N         DataLoader worker processes for lerobot-train.
#                           Default 16 (lerobot's own default of 4 leaves the
#                           CPU underutilized and starves the GPU). Pass an
#                           empty value (--num_workers=) to omit the flag.
#   --dry-run               Print commands without executing
#
# Example:
#   bash my_scripts/train_sweep.sh \
#       --dataset_repo=JennyWWW/splatsim_approach_lever_11_50failsrrtpi05 \
#       --ratio_sweep \
#       --ratios="0.2 0.4 0.6 0.8 1.0"

# ── USER CONFIG (defaults) ────────────────────────────────────────────────────
# Env profile: `--env_profile=NAME` sources my_scripts/env_profiles/NAME.sh,
# which sets the env-specific values below (ENV_TASK, ROBOT_NAME, NUM_DOFS,
# CAMERAS, DATASET_REPO, EVAL_BENCHMARK_REPO_ID) in ONE place so swapping
# environments (small_engine <-> planar <-> ...) is a single flag. Precedence:
# these built-in defaults < profile < explicit CLI flags. No profile → these
# UR5 defaults reproduce the historical hardcoded behavior.
PROFILE_NAME=""
# Env-specific (overridable via profile or explicit flags):
ENV_TASK="upright_small_engine_new"   # lerobot --env.task + splatsim register_env key
ROBOT_NAME=""                          # "" = lerobot SplatsimEnv default; profile sets for non-UR5 arms
NUM_DOFS=6                             # arm joints; state/action dim = NUM_DOFS + 1 (gripper)
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
# Modifiers for --headless (both no-ops without it), forwarded from
# dagger_orchestrate.sh's flags of the same name:
#   --control_gui → --env.control_gui=true: the in-process sim keeps SplatSim's
#     Tk control panel over its p.DIRECT pybullet client.
#   --keep_sa_gui → skip the show_slider=false injection: the SA wrapper keeps
#     its ratio slider + its own pybullet GUI window.
CONTROL_GUI=false
KEEP_SA_GUI=false
# --splat_shadows: route --env.splat_shadows=true into SHARED_ARGS so the
# in-process inline-eval sim composites PyBullet shadows onto its splat
# renders. INDEPENDENT of --headless (shadows are a rendering choice, not a
# GUI one). Skipped when --env_external_port is set — that path renders on
# the external sim, which owns its own --splat_shadows at launch. Forwarded
# from dagger_orchestrate.sh --splat_shadows via SPLAT_SHADOW_TRAIN_ARGS so
# eval imagery matches the shadow setting the datasets were recorded with.
SPLAT_SHADOWS=false
RATIOS=(0.2 0.4 0.6 0.8 1.0)
ENV_EXTERNAL_PORT=""
POLICY_PUSH_TO_HUB=""   # empty = use whatever the policy config default is
RUN_NAME_OVERRIDE=""    # set to override the auto-derived run_name (training dir basename)
MODEL="pi05"            # which policy to train: pi05 | diffusion | act
# Which camera set to train against. Drives BOTH the run's naming
# (`_${CAMERAS}` suffix on the run_name) and the actual
# `--policy.input_features` map (`set_camera_args` at the top of the
# file). Values:
#   basewrist (default) — both base_rgb + wrist_rgb + observation.state
#   base                — base_rgb + observation.state only
#   wrist               — wrist_rgb + observation.state only
# Historically hardcoded in _run_all_jobs; exposed here as a CLI knob
# so callers (e.g. dagger_orchestrate.sh) can pick base-only / wrist-only
# training without editing this file. Downstream users of the
# orchestrator-derived name (like dagger_orchestrate.sh's round-0 safety
# check) must mirror this default or thread the value through.
CAMERAS="basewrist"
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
# Eval benchmark dataset used by the inline eval phase. Empty = keep the
# built-in default baked into SHARED_ARGS below. Any non-empty value is
# forwarded as `--env.eval_benchmark_repo_id=` at the END of the command
# line so draccus's last-occurrence-wins rule silently replaces the
# SHARED_ARGS default. Wired up so the DAgger orchestrator can force
# every downstream training round (including round 0) to eval against
# the same benchmark the intervention + finetune eval steps use.
EVAL_BENCHMARK_REPO_ID=""
# Raw passthrough: whatever the caller wants appended VERBATIM to the
# lerobot-train command line. Word-split, so multi-flag strings work:
# --extra_args='--eval.n_episodes=30 --seed=0 --env.terminate_on_collision=true'.
# Appended AFTER SHARED_ARGS + per-job EXTRA arrays, so draccus's
# last-wins rule means any conflicting flag here overrides the earlier
# defaults. Matches the DAgger orchestrator's --finetune_extra_args
# passthrough (which now feeds this arg for round-0 / per-round-scratch /
# final-scratch training so the user defines eval knobs ONCE at the sweep
# level, not once per training path).
EXTRA_ARGS_STR=""
# --num_workers=N: DataLoader worker processes (lerobot-train --num_workers).
# lerobot's built-in default is 4, which leaves the CPU badly underutilized on
# a many-core box — the GPU stalls waiting on batch assembly. Default here is
# 16 (measured optimum on a 24-core machine; 20+ regresses). Emitted in SHARED_ARGS, so --extra_args
# (and the DAgger orchestrator's --round0_extra_args) can still override it
# via draccus last-wins. Set to empty (--num_workers=) to leave lerobot's
# own default in place.
NUM_WORKERS=16
# --exclude_gripper_from_state: drop the gripper dim (last dim of
# observation.state) from what the policy consumes. The dataset still
# records the full [NUM_DOFS + 1] state — only the runtime policy input
# view is sliced. Backed by TrainPipelineConfig.observation_dim_slice
# (see src/lerobot/configs/train.py). Reason: constant/unused dims have
# min == max in dataset stats, poisoning MIN_MAX normalization AND
# wasting an input slot; e.g. the always-0 gripper in the planar env.
# Only active when observation.state actually contains a trailing gripper
# dim (STATE_DIM > NUM_DOFS_ARM).
EXCLUDE_GRIPPER_FROM_STATE=false
# ─────────────────────────────────────────────────────────────────────────────

# Pre-scan for --env_profile BEFORE the main arg loop so the profile can set
# env-specific defaults that explicit CLI flags (parsed below) then override.
# Precedence: built-in defaults (above) < profile < explicit flags.
for arg in "$@"; do
    case "$arg" in
        --env_profile=*) PROFILE_NAME="${arg#*=}" ;;
    esac
done
if [[ -n "$PROFILE_NAME" ]]; then
    _PROFILE_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env_profiles/${PROFILE_NAME}.sh"
    if [[ ! -f "$_PROFILE_FILE" ]]; then
        echo "ERROR: --env_profile='$PROFILE_NAME' not found: $_PROFILE_FILE" >&2
        echo "Available profiles:" >&2
        ls "$(dirname "$_PROFILE_FILE")" 2>/dev/null | sed 's/\.sh$//;s/^/  /' >&2
        exit 1
    fi
    # shellcheck disable=SC1090
    source "$_PROFILE_FILE"
    echo "Loaded env profile '$PROFILE_NAME' (task=$ENV_TASK, num_dofs=$NUM_DOFS, cameras=$CAMERAS)."
fi

for arg in "$@"; do
    case "$arg" in
        --dry-run)              DRY_RUN=true ;;
        --env_profile=*)        ;;  # consumed in the pre-scan above; no-op here
        --ratio_sweep)          RATIO_SWEEP=true ;;
        --no_relative)          USE_RELATIVE_ACTIONS=false ;;
        --headless)             HEADLESS=true ;;
        --control_gui)          CONTROL_GUI=true ;;
        --splat_shadows)        SPLAT_SHADOWS=true ;;
        --splat_shadows=*)      SPLAT_SHADOWS="${arg#*=}" ;;
        --keep_sa_gui)          KEEP_SA_GUI=true ;;
        --dataset_repo=*)       DATASET_REPO="${arg#*=}" ;;
        --ratios=*)             IFS=' ' read -ra RATIOS <<< "${arg#*=}" ;;
        --env_external_port=*)  ENV_EXTERNAL_PORT="${arg#*=}" ;;
        --policy.push_to_hub=*) POLICY_PUSH_TO_HUB="${arg#*=}" ;;
        --run_name=*)           RUN_NAME_OVERRIDE="${arg#*=}" ;;
        --model=*)              MODEL="${arg#*=}" ;;
        --cameras=*)                CAMERAS="${arg#*=}" ;;
        --include_env_state_obs=*)  INCLUDE_ENV_STATE_OBS="${arg#*=}" ;;
        --eval_n_episodes=*)    EVAL_N_EPISODES="${arg#*=}" ;;
        --eval_benchmark_subset=*) EVAL_BENCHMARK_SUBSET="${arg#*=}" ;;
        --eval_benchmark_repo_id=*) EVAL_BENCHMARK_REPO_ID="${arg#*=}" ;;
        --eval_benchmark=*)         EVAL_BENCHMARK_REPO_ID="${arg#*=}" ;;  # short alias
        --extra_args=*)             EXTRA_ARGS_STR="${arg#*=}" ;;
        --num_workers=*)            NUM_WORKERS="${arg#*=}" ;;
        --exclude_gripper_from_state)         EXCLUDE_GRIPPER_FROM_STATE=true ;;
        --exclude_gripper_from_state=*)       EXCLUDE_GRIPPER_FROM_STATE="${arg#*=}" ;;
        --multi_dataset_repo_ids=*)      MULTI_DATASET_REPO_IDS="${arg#*=}" ;;
        --multi_dataset_sample_weights=*) MULTI_DATASET_SAMPLE_WEIGHTS="${arg#*=}" ;;
        --multi_dataset_stats_paths=*)   MULTI_DATASET_STATS_PATHS="${arg#*=}" ;;
        --multi_dataset_norm_mode=*)     MULTI_DATASET_NORM_MODE="${arg#*=}" ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

# ACTION dim = arm joints + 1 gripper (always). STATE dim defaults to the same,
# but an oracle-info profile overrides STATE_DIM (state = joints+gripper + object
# coords, wider than the action). Both flow to --policy.input_features (state
# shape) + --env.{state,action}_dim so env/policy/dataset agree on widths.
ACTION_DIM=$((NUM_DOFS + 1))
STATE_DIM="${STATE_DIM:-$ACTION_DIM}"
# Oracle env_state (observation.environment_state, FeatureType.ENV): privileged
# world state (object coords). The WIDTH is a fixed property of the dataset/env —
# the env profile provides it via ENV_STATE_DIM (2 planar-reach, 6 planar-2obst,
# 15 small-engine, …) — NOT something the user sets per run. The user only toggles
# whether the POLICY consumes it, via --include_env_state_obs=true|false.
ENV_STATE_DIM="${ENV_STATE_DIM:-0}"
# Default: consume env_state for a state-only run (no image → it needs it),
# otherwise off (pure vision). Profile or --include_env_state_obs overrides.
if [ -z "${INCLUDE_ENV_STATE_OBS:-}" ]; then
    if [ "$CAMERAS" = "state" ]; then INCLUDE_ENV_STATE_OBS=true; else INCLUDE_ENV_STATE_OBS=false; fi
fi
case "$INCLUDE_ENV_STATE_OBS" in
    true|false) ;;
    *) echo "ERROR: --include_env_state_obs must be true or false (got '$INCLUDE_ENV_STATE_OBS')." >&2; exit 1 ;;
esac
if [ "$INCLUDE_ENV_STATE_OBS" = "true" ] && [ "${ENV_STATE_DIM}" -le 0 ]; then
    echo "ERROR: --include_env_state_obs=true but the env profile records no oracle" >&2
    echo "       env_state (ENV_STATE_DIM=0). Use a profile whose dataset stores object" >&2
    echo "       coords, or set --include_env_state_obs=false." >&2
    exit 1
fi
# EFFECTIVE width fed to the policy/env: the recorded width when consumed, else 0.
if [ "$INCLUDE_ENV_STATE_OBS" = "true" ]; then EFF_ENV_STATE_DIM="$ENV_STATE_DIM"; else EFF_ENV_STATE_DIM=0; fi

# --exclude_gripper_from_state validation + derivation. When true, drop the
# gripper dim from observation.state as the policy input (dataset unchanged).
# The gripper joint always sits at index NUM_DOFS in observation.state (arm
# joints occupy [0..NUM_DOFS-1]; gripper at NUM_DOFS; any oracle coords a
# legacy profile pushed into observation.state at NUM_DOFS+1..STATE_DIM-1).
# So we build a "keep all indices except NUM_DOFS" list and reduce the
# declared state shape by 1.
case "$EXCLUDE_GRIPPER_FROM_STATE" in
    true|false) ;;
    *) echo "ERROR: --exclude_gripper_from_state must be true or false (got '$EXCLUDE_GRIPPER_FROM_STATE')." >&2; exit 1 ;;
esac
OBSERVATION_DIM_SLICE_JSON=""
EFF_POLICY_STATE_DIM="$STATE_DIM"
if [ "$EXCLUDE_GRIPPER_FROM_STATE" = "true" ]; then
    if [ "$STATE_DIM" -le "$NUM_DOFS" ]; then
        echo "ERROR: --exclude_gripper_from_state=true requires STATE_DIM (=$STATE_DIM) > NUM_DOFS (=$NUM_DOFS)," >&2
        echo "       i.e. observation.state must actually contain a gripper dim to drop." >&2
        exit 1
    fi
    # Emit [0,1,...,NUM_DOFS-1, NUM_DOFS+1, ..., STATE_DIM-1] — every index
    # except NUM_DOFS (the gripper joint slot).
    _keep_indices=""
    _first=1
    for _i in $(seq 0 $((STATE_DIM - 1))); do
        [ "$_i" = "$NUM_DOFS" ] && continue
        if [ "$_first" = "1" ]; then _keep_indices="$_i"; _first=0
        else _keep_indices="$_keep_indices,$_i"; fi
    done
    OBSERVATION_DIM_SLICE_JSON="{\"observation.state\": [${_keep_indices}]}"
    EFF_POLICY_STATE_DIM=$((STATE_DIM - 1))
fi

# Reusable --policy.input_features fragment: the proprioceptive state, plus the
# environment_state input when consumed. The diffusion policy requires an image OR
# environment_state, so a state-only run must consume it (guarded below).
# NOTE: use EFF_POLICY_STATE_DIM (not STATE_DIM) so the declared shape matches
# what the slice step will actually emit — matches TrainPipelineConfig.validate()'s
# post-slice shrink so we don't fight it via `last-wins` draccus semantics.
STATE_FEATURE_JSON="\"observation.state\": {\"type\": \"STATE\", \"shape\": [${EFF_POLICY_STATE_DIM}]}"
if [ "${EFF_ENV_STATE_DIM}" -gt 0 ]; then
    STATE_FEATURE_JSON="${STATE_FEATURE_JSON}, \"observation.environment_state\": {\"type\": \"ENV\", \"shape\": [${EFF_ENV_STATE_DIM}]}"
fi

# Naming tag = the `_<cameras>` run_name suffix. Delegated to dagger_naming.py
# (see camera_name_tag()) so this string is derived in ONE place and the
# orchestrator's pre-flight consistency check (parse_camera_name_tag) always
# agrees with what train_sweep.sh actually emits. The rules encoded there:
#   * base tag       = cameras verbatim (state / base / wrist / basewrist).
#   * +os suffix     iff cameras != "state" AND env_state is consumed.
#   * +ng suffix     iff --exclude_gripper_from_state was set.
_TRAIN_SWEEP_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_INCLUDE_ENV_STATE_BOOL=false
[ "${EFF_ENV_STATE_DIM}" -gt 0 ] && _INCLUDE_ENV_STATE_BOOL=true
CAMERA_NAME_TAG="$(python3 "$_TRAIN_SWEEP_SH_DIR/dagger_naming.py" camera_name_tag \
    --cameras="$CAMERAS" \
    --include_env_state="$_INCLUDE_ENV_STATE_BOOL" \
    --exclude_gripper="$EXCLUDE_GRIPPER_FROM_STATE")"

# --cameras validation. Only three variants are wired through
# `set_camera_args`; a bad value would silently produce an empty
# CAMERA_ARGS and lerobot-train would then complain about missing input
# features far downstream. Fail loudly here.
case "$CAMERAS" in
    basewrist|base|wrist|state) ;;
    *)
        echo "ERROR: --cameras='$CAMERAS' is not valid. Expected one of: basewrist, base, wrist, state." >&2
        exit 1
        ;;
esac

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

# Resolve the chunk size used to construct the relative-action stats sidecar
# path (stats_rel${chunk}.json). This MUST equal the policy's TRAINING horizon
# (the length of the rel-action sequence the model is trained on), NOT its
# inference-time n_action_steps. For pi0-family policies horizon == n_action_steps
# so the two are equivalent, but diffusion typically has horizon=64 while
# n_action_steps=32: only the first 32 predicted actions get executed per call,
# but the model still trains on all 64 positions — and rel-actions at positions
# 30-63 can be ~10× larger than at positions 0-7, so sizing stats to the
# smaller window undershoots the true target range → up to ~half the training
# targets normalize outside [-1,+1] and get clipped by clip_sample=True →
# policy learns to output near-zero actions and barely moves at inference.
#
# Source of truth, in priority order:
#   1. Explicit --policy.horizon=N in the policy args array.
#   2. Explicit --policy.chunk_size=N in the policy args array.
#   3. Explicit --policy.n_action_steps=N in the policy args array (used only
#      as a final CLI override when horizon isn't set — assumes the user knows
#      they're aligning to the same value).
#   4. The policy class's default (per-prefix fallback): pi05/pi0 → 50,
#      diffusion → 64.
# Empty for policies that don't use the relative-action pipeline (e.g. act with
# temporal ensembling — the policy uses absolute actions anyway).
#
# Note: we can't read this from a train_config.json the way dagger_orchestrate
# does for finetune, because run_job trains from scratch — no config exists yet.
# Args: $1 = policy_prefix, $2 = name of policy_args array (nameref).
_chunk_size_for_job() {
    local prefix="$1"
    local -n args_ref="$2"
    local override_horizon="" override_chunk="" override_nsteps=""
    for a in "${args_ref[@]}"; do
        case "$a" in
            --policy.horizon=*)        override_horizon="${a#*=}" ;;
            --policy.chunk_size=*)     override_chunk="${a#*=}" ;;
            --policy.n_action_steps=*) override_nsteps="${a#*=}" ;;
        esac
    done
    if [[ -n "$override_horizon" ]]; then echo "$override_horizon"; return; fi
    if [[ -n "$override_chunk"   ]]; then echo "$override_chunk";   return; fi
    if [[ -n "$override_nsteps"  ]]; then echo "$override_nsteps";  return; fi
    case "$prefix" in
        diffusion*) echo 64 ;;
        pi05*|pi0*) echo 50 ;;
        *)          echo "" ;;
    esac
}

# Bare per-prefix fallback used only by the early validation block below (which
# doesn't have access to the per-job policy_args arrays). Keep these in sync
# with the per-prefix defaults in _chunk_size_for_job.
_chunk_size_for_prefix() {
    case "$1" in
        diffusion*) echo 64 ;;
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
# Resolve eval-related overrides UP FRONT so SHARED_ARGS carries the final
# value directly. Previously these were hardcoded here (--eval.n_episodes=5,
# a fixed benchmark repo) and the CLI overrides appended at the END of the
# command, producing duplicate flags and relying on draccus's last-wins
# rule. That worked but was confusing to read in the emitted `Running: ...`
# line ("wait, I see =5 first — is that what's being used?"). Now each flag
# appears exactly once, at the resolved value.
_EVAL_N_EPISODES_ARG="${EVAL_N_EPISODES:-5}"
_EVAL_BENCHMARK_REPO_ID_ARG="${EVAL_BENCHMARK_REPO_ID:-JennyWWW/eval_splatsim_approach_lever_benchmark_1000}"
SHARED_ARGS=(
    --wandb.enable=true
    --policy.device=cuda
    --env.type=splatsim
    --env.task="$ENV_TASK"
    --env.fps=30
    # Arm-size passthrough so the eval env, policy, and dataset agree on vector
    # widths (planar arm != UR5). Derived from NUM_DOFS via the env profile.
    --env.num_dofs="$NUM_DOFS"
    --env.state_dim="$STATE_DIM"
    --env.action_dim="$ACTION_DIM"
    --env.env_state_dim="$EFF_ENV_STATE_DIM"
    --env.eval_benchmark_repo_id="$_EVAL_BENCHMARK_REPO_ID_ARG"
    --eval.n_episodes="$_EVAL_N_EPISODES_ARG"
    --eval.batch_size=1
    --eval.use_async_envs=false
    --dataset.image_transforms.enable=true
)
# DataLoader workers. Placed right after SHARED_ARGS so it sits BEFORE the
# per-job extras / --extra_args passthrough in the emitted command line
# (draccus last-wins ⇒ caller can still override).
if [[ -n "$NUM_WORKERS" ]]; then
    SHARED_ARGS+=( --num_workers="$NUM_WORKERS" )
fi
# Only pass --env.robot_name when the profile set it (empty = lerobot
# SplatsimEnv's own default, preserving historical UR5 behavior).
if [[ -n "${ROBOT_NAME:-}" ]]; then
    SHARED_ARGS+=( --env.robot_name="$ROBOT_NAME" )
fi
# Conditional: eval_benchmark_subset is only meaningful when the caller
# passes it. Adding an empty string would confuse draccus, so keep it out
# of SHARED_ARGS when unset.
if [[ -n "$EVAL_BENCHMARK_SUBSET" ]]; then
    SHARED_ARGS+=( --env.eval_benchmark_subset="$EVAL_BENCHMARK_SUBSET" )
fi

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
    # --keep_sa_gui: the SA wrapper's slider + pybullet GUI window live in the
    # lerobot process (not the sim), so they can stay up over a headless sim.
    # Explicit both ways so the emitted command documents the decision.
    if [[ "$KEEP_SA_GUI" == true ]]; then
        SHARED_ARGS+=( "--policy.shared_autonomy_config.show_slider=true" )
    else
        SHARED_ARGS+=( "--policy.shared_autonomy_config.show_slider=false" )
    fi
    if [[ -z "$ENV_EXTERNAL_PORT" ]]; then
        SHARED_ARGS+=( "--env.headless=true" )
        # --control_gui: in-process sim keeps the Tk control panel over its
        # p.DIRECT pybullet client (SplatSimEnv.control_gui → show_control_gui).
        [[ "$CONTROL_GUI" == true ]] && SHARED_ARGS+=( "--env.control_gui=true" )
    fi
fi

# --splat_shadows: rendering choice, so gated independently of --headless.
# Same external-port exemption as above (the external sim renders, and it
# owns its own --splat_shadows).
if [[ "$SPLAT_SHADOWS" == true && -z "$ENV_EXTERNAL_PORT" ]]; then
    SHARED_ARGS+=( "--env.splat_shadows=true" )
fi

# ── Policy-specific args ─────────────────────────────────────

DIFFUSION_ARGS=(
    --policy.type=diffusion
    --steps=75000
    --batch_size=32
    --env_eval_freq=25000
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
    --env_eval_freq=2000
    # --env_eval_freq=1000
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
    --env_eval_freq=10000
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
                "--policy.input_features={\"observation.images.base_rgb\": {\"type\": \"VISUAL\", \"shape\": [3, 224, 224]}, \"observation.images.wrist_rgb\": {\"type\": \"VISUAL\", \"shape\": [3, 224, 224]}, ${STATE_FEATURE_JSON}}"
                "--rename_map={\"observation.images.base_rgb_${resize_mode}\": \"observation.images.base_rgb\", \"observation.images.wrist_rgb_${resize_mode}\": \"observation.images.wrist_rgb\"}"
            )
            ;;
        base)
            CAMERA_ARGS=(
                "--env.camera_names=[\"base_rgb\"]"
                "--env.image_resize_modes=[\"${resize_mode}\"]"
                "--policy.input_features={\"observation.images.base_rgb\": {\"type\": \"VISUAL\", \"shape\": [3, 224, 224]}, ${STATE_FEATURE_JSON}}"
                "--rename_map={\"observation.images.base_rgb_${resize_mode}\": \"observation.images.base_rgb\"}"
            )
            ;;
        wrist)
            CAMERA_ARGS=(
                "--env.camera_names=[\"wrist_rgb\"]"
                "--env.image_resize_modes=[\"${resize_mode}\"]"
                "--policy.input_features={\"observation.images.wrist_rgb\": {\"type\": \"VISUAL\", \"shape\": [3, 224, 224]}, ${STATE_FEATURE_JSON}}"
                "--rename_map={\"observation.images.wrist_rgb_${resize_mode}\": \"observation.images.wrist_rgb\"}"
            )
            ;;
        state)
            # State-only (no image): observation.state (proprioception) PLUS a
            # separate observation.environment_state (object coords) — both come
            # from ${STATE_FEATURE_JSON}. No camera_names / rename_map. For the
            # planar oracle env (--env_profile=planar_oracle*). The diffusion
            # policy requires an image OR environment_state, so env_state MUST be
            # consumed here — i.e. --include_env_state_obs=true (the default for
            # --cameras=state) and a profile that records it (ENV_STATE_DIM > 0).
            if [ "${EFF_ENV_STATE_DIM}" -le 0 ]; then
                echo "ERROR: --cameras=state has no image, so it needs" >&2
                echo "       observation.environment_state — but env_state isn't being consumed." >&2
                echo "       Use --include_env_state_obs=true with an oracle-recording profile" >&2
                echo "       (ENV_STATE_DIM > 0), or add images via --cameras=base/wrist/basewrist." >&2
                exit 1
            fi
            CAMERA_ARGS=(
                "--env.camera_names=[]"
                "--policy.input_features={${STATE_FEATURE_JSON}}"
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
        run_name="${policy_prefix}_${DATASET_SHORT}_${action_suffix}_${CAMERA_NAME_TAG}"
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
            --dataset.use_weighted_sampling=true
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

    # --exclude_gripper_from_state → --observation_dim_slice=... on the
    # lerobot-train side. Emitted BEFORE per-job extra_args and
    # EXTRA_ARGS_STR so those can still override (draccus last-wins).
    # Empty when the flag is off — no-op passthrough.
    if [[ -n "$OBSERVATION_DIM_SLICE_JSON" ]]; then
        full_cmd+=(--observation_dim_slice="$OBSERVATION_DIM_SLICE_JSON")
    fi

    # Append per-job extra args last so they override any earlier defaults (e.g. batch_size)
    if [[ -n "$extra_args_ref" ]]; then
        local -n extra_args="$extra_args_ref"
        full_cmd+=("${extra_args[@]}")
    fi

    # Append caller-supplied raw passthrough (--extra_args=STR). Word-split
    # into individual argv tokens so multi-flag strings behave correctly.
    # Draccus last-wins → any flag here overrides same-key flags earlier
    # in the command (e.g. --eval.n_episodes=30 from finetune_extra_args
    # overrides SHARED_ARGS's default 5).
    if [[ -n "$EXTRA_ARGS_STR" ]]; then
        # shellcheck disable=SC2206  # intentional word-split of user-supplied string
        local -a _extra_split=( $EXTRA_ARGS_STR )
        full_cmd+=("${_extra_split[@]}")
    fi

    # Eval-scope overrides (--eval_n_episodes / --eval_benchmark_subset /
    # --eval_benchmark_repo_id) are handled UP FRONT in SHARED_ARGS above,
    # so each flag appears exactly once in the emitted command line at its
    # resolved value. No need to re-append here.

    # Print with per-arg single-quoting so args containing spaces,
    # brackets, or JSON (--policy.input_features={...},
    # --env.camera_names=["base_rgb", "wrist_rgb"], etc.) survive a
    # copy-paste back into a shell. Previously used ${full_cmd[*]} which
    # joins with bare spaces — copy-pasting re-split at every internal
    # space and truncated the JSON blobs mid-value.
    #
    # Per-arg rule:
    #   * no whitespace / metachar → print as-is (readable)
    #   * anything else → wrap in single quotes and escape any embedded
    #     single quote as '\'' (standard bash quote-escape idiom)
    _shell_quote_one() {
        local s="$1"
        if [[ "$s" =~ [[:space:]\{\}\[\]\|\;\&\<\>\(\)\$\`\"\\] ]]; then
            printf "'%s'" "${s//\'/\'\\\'\'}"
        else
            printf '%s' "$s"
        fi
    }
    printf 'Running:'
    [[ -n "$env_prefix" ]] && printf ' %s' "$env_prefix"
    for _arg in "${full_cmd[@]}"; do
        printf ' '
        _shell_quote_one "$_arg"
    done
    printf '\n'
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

# state-only (oracle) — no image; the resnet backbone still loads but sees no
# image features, so this is cheap. Bigger batch is fine (tiny per-sample cost).
DIFFUSION_STATE_ENV=""
DIFFUSION_STATE_EXTRA=()

# pi05 basewrist: needs extra VRAM setting + smaller batch size
PI05_BASEWRIST_ENV="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
PI05_BASEWRIST_EXTRA=(--batch_size=8)

PI05_BASE_ENV=""
PI05_BASE_EXTRA=(--batch_size=8)

PI05_WRIST_ENV=""
PI05_WRIST_EXTRA=(--batch_size=8)

PI05_STATE_ENV=""
PI05_STATE_EXTRA=(--batch_size=8)

ACT_BASEWRIST_ENV=""
ACT_BASEWRIST_EXTRA=()

ACT_BASE_ENV=""
ACT_BASE_EXTRA=()

ACT_WRIST_ENV=""
ACT_WRIST_EXTRA=()

ACT_STATE_ENV=""
ACT_STATE_EXTRA=()

# ── Run jobs ──────────────────────────────────────────────────

maybe_sleep() { [[ "$DRY_RUN" == false ]] && sleep 10; }

# All training jobs live here.  Wrapped in a function so the ratio sweep loop
# can call it once per merged dataset, then clean up before the next iteration.
# Dispatches by $MODEL + $CAMERAS. Only the selected model's selected-camera
# job runs. Per-model env/extra arrays are name-suffixed by camera setup
# (e.g. `PI05_BASEWRIST_ENV`, `PI05_BASE_ENV`, `PI05_WRIST_ENV`), so we
# uppercase $CAMERAS and use it to index the right pair via bash namerefs.
_run_all_jobs() {
    # Uppercase camera key for env/extra variable-name lookup.
    local cam_upper
    cam_upper="$(echo "$CAMERAS" | tr '[:lower:]' '[:upper:]')"
    case "$MODEL" in
        pi05)
            local -n _pi_env="PI05_${cam_upper}_ENV"
            local -n _pi_extra="PI05_${cam_upper}_EXTRA"
            run_job "pi05" "$CAMERAS" PI05_ARGS "$PI05_RESIZE_MODE" "$_pi_env" _pi_extra
            maybe_sleep
            ;;
        diffusion)
            local -n _df_env="DIFFUSION_${cam_upper}_ENV"
            local -n _df_extra="DIFFUSION_${cam_upper}_EXTRA"
            run_job "diffusion" "$CAMERAS" DIFFUSION_ARGS "$DIFFUSION_RESIZE_MODE" "$_df_env" _df_extra
            maybe_sleep
            ;;
        act)
            local -n _act_env="ACT_${cam_upper}_ENV"
            local -n _act_extra="ACT_${cam_upper}_EXTRA"
            run_job "act" "$CAMERAS" ACT_ARGS "$ACT_RESIZE_MODE" "$_act_env" _act_extra
            maybe_sleep
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
