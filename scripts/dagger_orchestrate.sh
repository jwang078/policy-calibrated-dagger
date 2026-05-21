#!/usr/bin/env bash
set -euo pipefail
# dagger_orchestrate.sh
#
# Automated multi-round DAgger pipeline. Each round does:
#   1. Record interventions on benchmark scenarios with the current policy
#   2. Compute sidecar stats for the per-round intervention dataset
#   3. Run augment_ratios_sweep.sh with --ratios "0.0" (hardlink-alias no-op)
#   4. Merge cumulatively (base + dag1 + ... + dagN) into a new training dataset
#   5. Compute sidecar stats for the merged dataset
#   6. Train: finetune (rounds 1..N-1 by default) or train-from-scratch (round N)
#   7. Cleanup: rm -rf round (N-1)'s merged dataset to bound disk use
#
# Hard constraints:
#   * Requires a SHARED EXTERNAL SplatSim ZMQ server running at --env_external_port
#     BEFORE this script is invoked. GPU memory can't host multiple SplatSim
#     instances plus training simultaneously. Launch with e.g.:
#       cd ~/code/SplatSim && python scripts/launch_nodes.py \
#         --robot sim_ur_pybullet_small_engine_new_interactive \
#         --robot_port 6001 --robot_name robot_iphone_w_engine_new \
#         --eval_benchmark_repo_id JennyWWW/eval_splatsim_approach_lever_benchmark_1000
#     Keep that running for the whole orchestration.
#
# Usage:
#   bash my_scripts/dagger_orchestrate.sh --base_short=STR --num_rounds=N [OPTIONS]
#
# Required:
#   --base_short=STR              Base dataset short name. The full repo id is
#                                 derived as ${HF_USER}/${BASE_SHORT} and the
#                                 per-round artifacts as ${HF_USER}/${BASE_SHORT}_dag{N}*.
#                                 The orchestrator no longer prepends "splatsim_"
#                                 to derived names — that prefix used to eat 9
#                                 chars off the 56-char repo limit and made many-
#                                 round runs impossible. If your existing base
#                                 dataset is named "splatsim_X" on disk/hub,
#                                 either rename it or symlink "X" → "splatsim_X"
#                                 in ~/.cache/huggingface/lerobot/${HF_USER}/ so
#                                 the orchestrator can find it under the prefix-less
#                                 name.
#   --num_rounds=N                Number of DAgger rounds (>=1).
#   --no_strip_splatsim_prefix    Keep the "splatsim_" prefix on derived dag/merged
#                                 dataset names (matches the base dataset exactly,
#                                 useful if the rest of your tooling expects the
#                                 prefix). Default behavior strips it to save 9
#                                 chars under the 56-char HF repo limit. Pass
#                                 --strip_splatsim_prefix to force the default
#                                 explicitly.
#
# Starting point (optional):
#   --initial_policy_path=PATH    Skip round-0 base training; use this checkpoint
#                                 as the round-1 input policy.
#   (unset)                       Run train_sweep.sh on the base dataset first.
#
# Shared external SplatSim:
#   --env_external_port=N         (default 6001)
#   --env_external_host=STR       (default 127.0.0.1)
#
# Per-round training mode:
#   --intermediate_mode=MODE      "finetune" or "scratch" used for EVERY dag
#                                 round 1..N (default "finetune"). Naming kept
#                                 for back-compat; effectively "--mode".
#   --final_mode=MODE             "scratch" (default) or "finetune". Controls
#                                 an OPTIONAL extra training step that runs
#                                 AFTER all N dag rounds, using round N's
#                                 final merged dataset:
#                                   scratch  → do an extra from-scratch train.
#                                              The plot-friendly mode: all N
#                                              dag rounds are uniform finetune
#                                              steps (the round-over-round
#                                              curve is monotonic), and the
#                                              extra scratch step gives you
#                                              a deployable policy trained on
#                                              the same final data.
#                                   finetune → skip the extra step; the last
#                                              dag round's finetune is the
#                                              deployable policy.
#
# Intervention recording:
#   --intervention_n_episodes=N   Forwarded to lerobot-eval as --eval.n_episodes
#                                 (default 100 — first 100 benchmark scenarios in order).
#   --intervention_sample_from_first=Y
#                                 Randomly sample --intervention_n_episodes scenarios
#                                 from the first Y indices of the benchmark dataset.
#                                 Unset (default): use first N in order.
#                                 Example: --intervention_n_episodes=20 --intervention_sample_from_first=100
#                                 → pick 20 random scenarios from indices [0..99].
#   --intervention_sample_seed=S  RNG seed for reproducible random sampling
#                                 (default 0). Same seed = same subset across reruns,
#                                 which is what you want for DAgger optimizing the
#                                 same scenarios round over round.
#   --intervention_oversample=N   Repeat each per-round intervention dataset N
#                                 times in the cumulative merge (default 1 = no
#                                 oversample). Useful when intervention frames
#                                 are a small fraction of the merged dataset
#                                 (e.g. 5 intervention episodes vs 300 base eps
#                                 = 1.6% of frames) and the policy under-weights
#                                 them during random batch sampling. With N=3,
#                                 the intervention frames make up ~5% of the
#                                 merged dataset → ~3× more likely to be drawn
#                                 in a random batch. Each round's merge becomes
#                                 N× larger on disk; pair with a reasonable
#                                 N value (2-5) and watch disk usage.
#   --intervention_max_episode_length=N
#                                 Forwarded to lerobot-eval as
#                                 --env.episode_length (default 5000). This is the
#                                 MAX step cap per episode, NOT the length of any
#                                 single human intervention within the episode. The
#                                 SplatSimEnv config default of 400 is too short for
#                                 RRT-to-goal recovery to complete on hard scenarios.
#   --eval_benchmark_repo_id=ID   (default JennyWWW/eval_splatsim_approach_lever_benchmark_1000)
#   --intervention_extra_args=STR Raw passthrough to lerobot-eval
#                                 (e.g. --policy.shared_autonomy_config.enabled=true).
#   --intervention_method=METHOD  "rrt" (default) or "oracle_goal".
#                                 "rrt" uses the SA wrapper's RRT-to-goal planner
#                                 (recorded as FrameSource.RRT).
#                                 "oracle_goal" uses OracleGoalGuidanceSource —
#                                 a straight-line joint-space interpolation
#                                 from q_start → oracle q_goal_bias, no planner.
#                                 Recorded as FrameSource.BLEND_INTERVENTION_100
#                                 (verbatim playback). Use for DAgger label
#                                 generation when you want a deterministic
#                                 correction signal rather than RRT's variability.
#                                 Forwarded as --intervention.method=METHOD.
#   --intervention_oracle_goal_chunk_steps=N
#                                 (default 80) Number of waypoints in the
#                                 q_start → q_goal_bias linear interpolation chunk.
#                                 Only used iff --intervention_method=oracle_goal.
#                                 The intervention controller's rrt_steps_min/max
#                                 still picks a target_steps within this chunk
#                                 (typically less than N → partial playback).
#                                 Forwarded as
#                                 --intervention.oracle_goal_chunk_steps=N.
#
# Model / action format (controls naming, must match train_sweep.sh's choices):
#   --model=MODEL                 "pi" / "diff" / "act"  (default "pi")
#   --action_format=FMT           "abs" / "rel". When --initial_policy_path is
#                                 set and this flag is omitted, the orchestrator
#                                 auto-detects it from the policy's train_config.json
#                                 (use_relative_actions: true→rel, false→abs). Pass
#                                 it explicitly to override the auto-detection.
#                                 Default when no initial policy: "rel".
#
# Finetune overrides (forwarded to resume_training.sh):
#   --finetune_steps=N            (default 4000)
#   --finetune_eval_freq=N        (default 2000)
#   --finetune_save_freq=N        (default 2000)
#   --finetune_batch_size=N       (default empty = inherit from base train_config.json)
#                                 Lower this if you hit CUDA OOM — the shared external
#                                 SplatSim costs ~2.75 GiB the original from-scratch
#                                 training had to itself, so the finetune step is
#                                 tighter on memory than the round-0 training was.
#   --finetune_decay_lr=FLOAT     Override the cosine scheduler's floor LR
#                                 (--policy.scheduler_decay_lr). Default empty
#                                 = AUTO-SET to the policy's peak optimizer_lr
#                                 (read from train_config.json — 2.5e-5 for
#                                 pi05, 1e-5 for diffusion). This forces a
#                                 CONSTANT peak LR through the finetune, which
#                                 is what you want for DAgger resumes past the
#                                 cosine decay end (otherwise runtime LR is
#                                 parked at the original ~1e-6 floor — too
#                                 small to actually move the policy in
#                                 200-2000 steps even with large gradients).
#                                 Pass an explicit value to override the
#                                 auto-set (e.g. 5e-5 for an aggressive run).
#
# Resume/restart:
#   --start_round=N               Explicitly start at round N (default: auto-detect).
#   --force_restart               Skip resume-prompt; restart from round 1, deleting
#                                 any existing dag1..dag{num_rounds} artifacts. Asks
#                                 for confirmation token.
#   --retrain_round=N             Re-run ONLY round N's training step (step 6), writing
#                                 to a suffixed output dir so the original training dir
#                                 isn't overwritten. Use this to iterate on finetune
#                                 hyperparameters (LR, steps, oversample) without
#                                 redoing intervention recording or merging.
#                                 Auto-detects what data exists: if round N's merged
#                                 dataset is on disk, runs step 6 only; otherwise
#                                 re-runs steps 4-6 (re-merge from existing
#                                 intervention datasets, recompute stats, train).
#                                 Skips the post-loop final-scratch phase (the chain
#                                 stops at the retrained round). Mutually exclusive
#                                 with --start_round/--force_restart.
#   --retrain_suffix=NAME         Suffix appended to the retrained round's training
#                                 output dir + policy run name (default "v2"). Must
#                                 be alphanumeric/underscore/hyphen. Example: dag6
#                                 → "_v2" → outputs/training/..._ft_dag6_v2/
#
# Merge routing:
#   --skip_alias_step             Skip step 3 (the augment_ratios_sweep.sh ratio=0
#                                 hardlink-alias). Pass the per-round intervention
#                                 datasets directly into merge step 4 by their raw
#                                 names ({base}_dag{N}). Use this when your base
#                                 dataset name plus "_dag{N}_pirel00" suffix would
#                                 exceed the 56-char HuggingFace repo limit. Default
#                                 OFF — the alias step is on by default so future
#                                 ratio>0 augmentations slot into the same pipeline.
#
# Offline mode:
#   --push_to_hub                 When set, push artifacts to HuggingFace Hub:
#                                 (a) the per-round intervention dataset (after
#                                     step 1), and
#                                 (b) the trained policy checkpoint (after step 6).
#                                 Default OFF — DAgger generates 3+ datasets and
#                                 1+ multi-GB policy per round; keeping them
#                                 local-only avoids minutes-to-hours of unwanted
#                                 hub uploads per round.
#
# SplatSim lifecycle:
#   --manage_splatsim             (default ON) The orchestrator launches its own
#                                 SplatSim ZMQ server on --env_external_port at
#                                 startup and shuts it down before each training
#                                 step, then re-launches it after training. This
#                                 lets lerobot-train spawn its own in-process sim
#                                 for inline eval (same CUDA context as training
#                                 → memory poolable → no OOM), while keeping the
#                                 external sim available for intervention recording
#                                 and the next round's interventions. The sim is
#                                 also killed on orchestrator exit.
#   --no_manage_splatsim          Opt out: the user is responsible for launching
#                                 and managing SplatSim manually on
#                                 --env_external_port (legacy behavior). The
#                                 orchestrator will neither kill nor relaunch it,
#                                 and training will keep --env.external_port set.
#                                 Risk: external sim's 2.7 GiB CUDA context is
#                                 hard-partitioned during training → potential OOM.
#   --splatsim_root=PATH          Root of the SplatSim repo (default $HOME/code/SplatSim).
#                                 launch_nodes.py is run from this dir.
#   --splatsim_robot=NAME         --robot arg for launch_nodes.py
#                                 (default sim_ur_pybullet_small_engine_new_interactive).
#   --splatsim_robot_name=NAME    --robot_name arg for launch_nodes.py
#                                 (default robot_iphone_w_engine_new).
#
# Misc:
#   --dry-run                     Print every command instead of executing.
#
# Example:
#   bash my_scripts/dagger_orchestrate.sh \
#       --base_short=approach_lever_7_lowres_5path \
#       --num_rounds=3 \
#       --initial_policy_path=outputs/training/pi05_approach_lever_7_lowres_5path_delta_basewrist \
#       --env_external_port=6001 --intervention_n_episodes=100 \
#       --intermediate_mode=finetune --final_mode=scratch

# ── USER CONFIG (defaults) ────────────────────────────────────────────────────
HF_USER="JennyWWW"
BASE_SHORT=""
BASE_REPO_OVERRIDE=""
STRIP_SPLATSIM_PREFIX=true   # strip splatsim_ from the dag-name prefix; flip with --no_strip_splatsim_prefix
# Optional tag inserted into derived dag dataset names AND derived policy
# training-dir names so a new dagger run doesn't overwrite a previous run
# that used the same base. Empty (default) = original naming. Set to
# e.g. "d30" for a diverse-30-scenario run. The tag costs N+1 chars in the
# dataset name; ensure num_rounds + base_short + tag fits HF's 56-char repo-
# name limit (pre-flight will fail otherwise with a clear message).
RUN_TAG=""
# Optional override of the dag dataset-name root, post-derivation. Lets you
# use a SHORT name for dag artifacts even when BASE_REPO has a long name
# (e.g. BASE_REPO=JennyWWW/splatsim_approach_lever_11_biasend_5path_grip0
# → DAG_SHORT_OVERRIDE=lever_grip0 → dag artifacts become
# lever_grip0_r_dag1, lever_grip0_r_dag1_m, etc.). The actual BASE_REPO
# data source is untouched; only dag artifact names change.
DAG_SHORT_OVERRIDE=""
NUM_ROUNDS=""
INITIAL_POLICY_PATH=""
ENV_EXTERNAL_PORT="6001"
ENV_EXTERNAL_HOST="127.0.0.1"
INTERMEDIATE_MODE="finetune"
FINAL_MODE="scratch"
INTERVENTION_N_EPISODES="100"
INTERVENTION_OVERSAMPLE="1"
INTERVENTION_MAX_EPISODE_LENGTH="5000"
INTERVENTION_SAMPLE_FROM_FIRST=""   # empty → run first N in order; set → random subset
INTERVENTION_SAMPLE_SEED="0"
EVAL_BENCHMARK_REPO_ID="JennyWWW/eval_splatsim_approach_lever_benchmark_1000"
INTERVENTION_EXTRA_ARGS=""
# Intervention method selector. "rrt" uses the SA wrapper's RRT-to-goal
# planner (existing behavior); "oracle_goal" uses a straight-line joint-space
# interpolation from q_start to the oracle's q_goal_bias. Recorded frames are
# committed to the dataset in both cases (tagged FrameSource.RRT for rrt,
# FrameSource.BLEND_INTERVENTION_100 for oracle_goal VERBATIM playback).
INTERVENTION_METHOD="rrt"
INTERVENTION_ORACLE_GOAL_CHUNK_STEPS="80"
MODEL="pi"
ACTION_FORMAT=""   # empty → auto-detect from --initial_policy_path; fall back to "rel"
FINETUNE_STEPS="4000"
FINETUNE_EVAL_FREQ="2000"
FINETUNE_SAVE_FREQ="2000"
FINETUNE_BATCH_SIZE=""
FINETUNE_DECAY_LR=""
START_ROUND=""    # empty → auto-detect
FORCE_RESTART=false
RETRAIN_ROUND=""    # empty → not in retrain mode; set → only re-train this round to a suffixed dir
RETRAIN_SUFFIX="v2"
SKIP_ALIAS_STEP=false
PUSH_TO_HUB=false   # offline by default — see header for rationale
MANAGE_SPLATSIM=true
SPLATSIM_ROOT="$HOME/code/SplatSim"
SPLATSIM_ROBOT="sim_ur_pybullet_small_engine_new_interactive"
SPLATSIM_ROBOT_NAME="robot_iphone_w_engine_new"
DRY_RUN=false
# ─────────────────────────────────────────────────────────────────────────────

# Path constants.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LEROBOT_CACHE="$HOME/.cache/huggingface/lerobot"
STATS_BASE="$LEROBOT_ROOT/outputs/dataset_stats"

# Computed at the end of arg parsing (see "intervention scenario subset" block):
# JSON list of episode indices to pass as --env.eval_benchmark_subset, or empty
# when using default first-N-in-order behavior.
INTERVENTION_SUBSET_JSON=""

# ── parse args ────────────────────────────────────────────────────────────────
for arg in "$@"; do
    case "$arg" in
        --base_short=*)               BASE_SHORT="${arg#*=}" ;;
        --base_repo_id=*)             BASE_REPO_OVERRIDE="${arg#*=}" ;;
        --no_strip_splatsim_prefix)   STRIP_SPLATSIM_PREFIX=false ;;
        --strip_splatsim_prefix)      STRIP_SPLATSIM_PREFIX=true ;;
        --run_tag=*)                  RUN_TAG="${arg#*=}" ;;
        --dag_short_override=*)       DAG_SHORT_OVERRIDE="${arg#*=}" ;;
        --num_rounds=*)               NUM_ROUNDS="${arg#*=}" ;;
        --initial_policy_path=*)      INITIAL_POLICY_PATH="${arg#*=}" ;;
        --env_external_port=*)        ENV_EXTERNAL_PORT="${arg#*=}" ;;
        --env_external_host=*)        ENV_EXTERNAL_HOST="${arg#*=}" ;;
        --intermediate_mode=*)        INTERMEDIATE_MODE="${arg#*=}" ;;
        --final_mode=*)               FINAL_MODE="${arg#*=}" ;;
        --intervention_n_episodes=*)         INTERVENTION_N_EPISODES="${arg#*=}" ;;
        --intervention_oversample=*)         INTERVENTION_OVERSAMPLE="${arg#*=}" ;;
        --intervention_max_episode_length=*) INTERVENTION_MAX_EPISODE_LENGTH="${arg#*=}" ;;
        --intervention_sample_from_first=*)  INTERVENTION_SAMPLE_FROM_FIRST="${arg#*=}" ;;
        --intervention_sample_seed=*)        INTERVENTION_SAMPLE_SEED="${arg#*=}" ;;
        --eval_benchmark_repo_id=*)          EVAL_BENCHMARK_REPO_ID="${arg#*=}" ;;
        --intervention_extra_args=*)         INTERVENTION_EXTRA_ARGS="${arg#*=}" ;;
        --intervention_method=*)             INTERVENTION_METHOD="${arg#*=}" ;;
        --intervention_oracle_goal_chunk_steps=*) INTERVENTION_ORACLE_GOAL_CHUNK_STEPS="${arg#*=}" ;;
        --model=*)                    MODEL="${arg#*=}" ;;
        --action_format=*)            ACTION_FORMAT="${arg#*=}" ;;
        --finetune_steps=*)           FINETUNE_STEPS="${arg#*=}" ;;
        --finetune_eval_freq=*)       FINETUNE_EVAL_FREQ="${arg#*=}" ;;
        --finetune_save_freq=*)       FINETUNE_SAVE_FREQ="${arg#*=}" ;;
        --finetune_batch_size=*)      FINETUNE_BATCH_SIZE="${arg#*=}" ;;
        --finetune_decay_lr=*)        FINETUNE_DECAY_LR="${arg#*=}" ;;
        --start_round=*)              START_ROUND="${arg#*=}" ;;
        --force_restart)              FORCE_RESTART=true ;;
        --retrain_round=*)            RETRAIN_ROUND="${arg#*=}" ;;
        --retrain_suffix=*)           RETRAIN_SUFFIX="${arg#*=}" ;;
        --skip_alias_step)            SKIP_ALIAS_STEP=true ;;
        --push_to_hub)                PUSH_TO_HUB=true ;;
        --manage_splatsim)            MANAGE_SPLATSIM=true ;;
        --no_manage_splatsim)         MANAGE_SPLATSIM=false ;;
        --splatsim_root=*)            SPLATSIM_ROOT="${arg#*=}" ;;
        --splatsim_robot=*)           SPLATSIM_ROBOT="${arg#*=}" ;;
        --splatsim_robot_name=*)      SPLATSIM_ROBOT_NAME="${arg#*=}" ;;
        --dry-run)                    DRY_RUN=true ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

if [[ -z "$BASE_SHORT" ]]; then
    echo "ERROR: --base_short is required" >&2; exit 1
fi
if [[ -z "$NUM_ROUNDS" || ! "$NUM_ROUNDS" =~ ^[0-9]+$ ]] || (( NUM_ROUNDS < 1 )); then
    echo "ERROR: --num_rounds=N (positive integer) is required" >&2; exit 1
fi
# Validate intervention sampling: when --intervention_sample_from_first=Y is set,
# Y must be a positive integer and >= --intervention_n_episodes (can't sample
# 100 items from a pool of 50).
if [[ -n "$INTERVENTION_SAMPLE_FROM_FIRST" ]]; then
    if ! [[ "$INTERVENTION_SAMPLE_FROM_FIRST" =~ ^[0-9]+$ ]] || (( INTERVENTION_SAMPLE_FROM_FIRST < 1 )); then
        echo "ERROR: --intervention_sample_from_first=$INTERVENTION_SAMPLE_FROM_FIRST must be a positive integer" >&2; exit 1
    fi
    if (( INTERVENTION_SAMPLE_FROM_FIRST < INTERVENTION_N_EPISODES )); then
        echo "ERROR: --intervention_sample_from_first ($INTERVENTION_SAMPLE_FROM_FIRST) must be >=" \
             "--intervention_n_episodes ($INTERVENTION_N_EPISODES) — can't sample more than the pool size" >&2; exit 1
    fi
fi
if ! [[ "$INTERVENTION_OVERSAMPLE" =~ ^[0-9]+$ ]] || (( INTERVENTION_OVERSAMPLE < 1 )); then
    echo "ERROR: --intervention_oversample=$INTERVENTION_OVERSAMPLE must be a positive integer (1 = no oversampling)" >&2; exit 1
fi

# Retrain-mode validation. --retrain_round=N is mutually exclusive with
# --start_round/--force_restart since it pins the start at round N step 6 and
# uses a different code path that skips the resume-prompt entirely.
if [[ -n "$RETRAIN_ROUND" ]]; then
    if ! [[ "$RETRAIN_ROUND" =~ ^[0-9]+$ ]] || (( RETRAIN_ROUND < 1 || RETRAIN_ROUND > NUM_ROUNDS )); then
        echo "ERROR: --retrain_round=$RETRAIN_ROUND must be between 1 and $NUM_ROUNDS" >&2; exit 1
    fi
    if [[ -z "$RETRAIN_SUFFIX" ]]; then
        echo "ERROR: --retrain_round requires --retrain_suffix=NAME (e.g. v2); cannot be empty" >&2; exit 1
    fi
    if ! [[ "$RETRAIN_SUFFIX" =~ ^[a-zA-Z0-9_-]+$ ]]; then
        echo "ERROR: --retrain_suffix='$RETRAIN_SUFFIX' must be alphanumeric/underscore/hyphen only" >&2; exit 1
    fi
    if [[ -n "$START_ROUND" || "$FORCE_RESTART" == true ]]; then
        echo "ERROR: --retrain_round is mutually exclusive with --start_round / --force_restart" >&2; exit 1
    fi
fi

case "$INTERMEDIATE_MODE" in finetune|scratch) ;; *) echo "ERROR: --intermediate_mode must be 'finetune' or 'scratch'" >&2; exit 1;; esac
case "$FINAL_MODE"        in finetune|scratch) ;; *) echo "ERROR: --final_mode must be 'finetune' or 'scratch'" >&2; exit 1;; esac
case "$MODEL"             in pi|diff|act)      ;; *) echo "ERROR: --model must be one of pi/diff/act" >&2; exit 1;; esac
case "$INTERVENTION_METHOD" in rrt|oracle_goal) ;; *) echo "ERROR: --intervention_method must be 'rrt' or 'oracle_goal'" >&2; exit 1;; esac

# Auto-detect ACTION_FORMAT from --initial_policy_path's train_config.json when
# the user didn't pass --action_format explicitly. Reading use_relative_actions
# off the actual policy avoids the subtle stats-mismatch class of bugs (rel
# stats sidecar applied to an absolute-action policy → silent corruption).
if [[ -z "$ACTION_FORMAT" && -n "$INITIAL_POLICY_PATH" ]]; then
    DETECT_CFG=""
    for candidate in \
        "$INITIAL_POLICY_PATH/train_config.json" \
        "$INITIAL_POLICY_PATH/pretrained_model/train_config.json" \
        "$INITIAL_POLICY_PATH/checkpoints/last/pretrained_model/train_config.json"
    do
        if [[ -f "$candidate" ]]; then
            DETECT_CFG="$candidate"; break
        fi
    done
    if [[ -n "$DETECT_CFG" ]]; then
        DETECTED_REL="$(python3 -c "
import json,sys
c = json.load(open(sys.argv[1]))
print('true' if c.get('policy',{}).get('use_relative_actions') else 'false')
" "$DETECT_CFG")"
        if [[ "$DETECTED_REL" == "true" ]]; then
            ACTION_FORMAT="rel"
        else
            ACTION_FORMAT="abs"
        fi
        echo "Auto-detected --action_format=$ACTION_FORMAT from $DETECT_CFG (use_relative_actions=$DETECTED_REL)."
    else
        echo "WARNING: --action_format not specified and could not find train_config.json under $INITIAL_POLICY_PATH" >&2
        echo "  Falling back to --action_format=rel." >&2
        ACTION_FORMAT="rel"
    fi
fi
# Fallback when no initial policy was provided (from-scratch training).
[[ -z "$ACTION_FORMAT" ]] && ACTION_FORMAT="rel"

case "$ACTION_FORMAT"     in abs|rel)          ;; *) echo "ERROR: --action_format must be 'abs' or 'rel'" >&2; exit 1;; esac

# Single-letter action-format infix used in dag dataset names — see
# int_short_for_round / merged_short_for_round below for rationale (HF 56-char
# repo-name limit leaves only ~10 chars of suffix budget for long base names).
case "$ACTION_FORMAT" in
    rel) ACTION_INFIX="r" ;;
    abs) ACTION_INFIX="a" ;;
esac

# Base repo id — used by both the Round-0 block (to train from scratch) and
# the per-round merge step (which always merges from originals: base + all
# dag{1..r}).
#
# Auto-detect from the resumed policy's train_config.json (the
# dataset.repo_id field) when --initial_policy_path is set. This avoids a
# previously-silent bug where the orchestrator derived
# BASE_REPO=${HF_USER}/${BASE_SHORT} but the actual policy was trained on a
# differently-named variant (e.g. ${BASE_SHORT}_grip0). Mixing the wrong base
# into the merge silently corrupts training because the base's action
# distribution conflicts with what the policy already learned (and what the
# RRT intervention data continues to provide). Pass --base_repo_id to override.
BASE_REPO=""
if [[ -n "${BASE_REPO_OVERRIDE:-}" ]]; then
    BASE_REPO="$BASE_REPO_OVERRIDE"
    echo "Base repo (from --base_repo_id):    $BASE_REPO"
elif [[ -n "$INITIAL_POLICY_PATH" ]]; then
    # Reuse DETECT_CFG from the action-format auto-detect block above.
    if [[ -n "${DETECT_CFG:-}" && -f "$DETECT_CFG" ]]; then
        DETECTED_BASE="$(python3 -c "
import json, sys
c = json.load(open(sys.argv[1]))
print(c.get('dataset', {}).get('repo_id', ''))
" "$DETECT_CFG")"
        if [[ -n "$DETECTED_BASE" ]]; then
            BASE_REPO="$DETECTED_BASE"
            echo "Auto-detected base repo from policy's train_config.json: $BASE_REPO"
        fi
    fi
fi
if [[ -z "$BASE_REPO" ]]; then
    BASE_REPO="${HF_USER}/${BASE_SHORT}"
    echo "Base repo (derived from --base_short): $BASE_REPO"
fi
# Sanity check: the merge base should exist on disk. If not, fall back to the
# splatsim_-prefixed variant (legacy naming) and warn. This catches the case
# where the auto-detected repo_id is a stale path from before a rename.
if [[ ! -d "$LEROBOT_CACHE/$BASE_REPO" ]]; then
    LEGACY_BASE="${HF_USER}/splatsim_${BASE_REPO#${HF_USER}/}"
    if [[ -d "$LEROBOT_CACHE/$LEGACY_BASE" ]]; then
        echo "WARNING: $BASE_REPO not found on disk; falling back to legacy $LEGACY_BASE." >&2
        BASE_REPO="$LEGACY_BASE"
    else
        echo "WARNING: merge base $BASE_REPO not found on disk." >&2
        echo "  Expected: $LEROBOT_CACHE/$BASE_REPO" >&2
        echo "  Pass --base_repo_id=ID explicitly, or symlink the dataset under that name." >&2
    fi
fi

# BASE_DATASET_SHORT is the short name used to derive every per-round dataset
# name: intervention dataset, alias dataset, and merged training dataset. It
# matches the base dataset's name (so dag artifacts inherit identifying
# suffixes like _grip0, _lowres, etc.) but strips the HF user prefix and the
# optional "splatsim_" prefix to keep names short within the 56-char hub
# limit. Computed AFTER the BASE_REPO legacy-fallback above so the strip
# handles both `JennyWWW/foo` and `JennyWWW/splatsim_foo` uniformly.
#
# Example:
#   BASE_REPO          = JennyWWW/splatsim_approach_lever_11_biasend_5path_grip0
#   BASE_DATASET_SHORT = approach_lever_11_biasend_5path_grip0
#   dag1 dataset       = JennyWWW/approach_lever_11_biasend_5path_grip0_dag1
#   dag3_m merged      = JennyWWW/approach_lever_11_biasend_5path_grip0_dag3_m
BASE_DATASET_SHORT="${BASE_REPO#${HF_USER}/}"
if [[ "$STRIP_SPLATSIM_PREFIX" == true ]]; then
    BASE_DATASET_SHORT="${BASE_DATASET_SHORT#splatsim_}"
fi
# --dag_short_override lets the user completely replace BASE_DATASET_SHORT
# with a shorter name. Useful when the auto-derived BASE_DATASET_SHORT is
# at the HF 56-char limit and would overflow after adding _dag${N}_m or
# adding a --run_tag. Doesn't affect BASE_REPO (the actual data source);
# only changes what the dag artifacts are named on disk and on the hub.
if [[ -n "$DAG_SHORT_OVERRIDE" ]]; then
    BASE_DATASET_SHORT="$DAG_SHORT_OVERRIDE"
fi
# Optional --run_tag suffix to differentiate from prior dag runs on the same
# base. Inserted into both BASE_DATASET_SHORT and BASE_POLICY_NAME so dataset
# names and training dirs both get tagged. Applied AFTER the override so the
# user can combine them, e.g. --dag_short_override=lever_grip0 --run_tag=d30.
if [[ -n "$RUN_TAG" ]]; then
    BASE_DATASET_SHORT="${BASE_DATASET_SHORT}_${RUN_TAG}"
fi
# Model-type tag. "pi" (pi05) gets no tag — preserves back-compat with every
# existing pi05 DAgger lineage on disk (dag artifact names were originally
# model-agnostic). "diff" / "act" get an explicit tag so pi05 and diffusion/act
# runs on the same base + run_tag + method don't collide on the same dataset
# names. Training dirs are already disambiguated by the model prefix
# (pi05_/diffusion_/act_), but dag artifact names were not.
MODEL_TAG=""
case "$MODEL" in
    pi)   MODEL_TAG="" ;;
    diff) MODEL_TAG="diff" ;;
    act)  MODEL_TAG="act" ;;
    *) echo "ERROR: no MODEL_TAG mapping for model='$MODEL'" >&2; exit 1 ;;
esac
if [[ -n "$MODEL_TAG" ]]; then
    BASE_DATASET_SHORT="${BASE_DATASET_SHORT}_${MODEL_TAG}"
fi
# Intervention-method tag. "rrt" gets no tag (preserves existing artifact
# names on disk and on the hub — back-compat with all DAgger runs to date).
# Other methods get a 2-char tag so the user can tell runs apart at a glance,
# and dagger_progress.sh / dagger_plot.py auto-discover them as separate
# lineages (each method has its own table + plot).
METHOD_TAG=""
case "$INTERVENTION_METHOD" in
    rrt)         METHOD_TAG="" ;;
    oracle_goal) METHOD_TAG="og" ;;
    *) echo "ERROR: no METHOD_TAG mapping for intervention_method='$INTERVENTION_METHOD'" >&2; exit 1 ;;
esac
if [[ -n "$METHOD_TAG" ]]; then
    BASE_DATASET_SHORT="${BASE_DATASET_SHORT}_${METHOD_TAG}"
fi
echo "Dag dataset short prefix (dag artifacts named ${BASE_DATASET_SHORT}_${ACTION_INFIX}_dag{N}{,_m,_${MODEL}${ACTION_FORMAT}00}): $BASE_DATASET_SHORT"

# ── intervention scenario subset (computed once, reused every round) ──────────
# When --intervention_sample_from_first=Y is set, pick INTERVENTION_N_EPISODES
# scenarios at random from [0..Y-1] using the given seed, and pass them as
# --env.eval_benchmark_subset=[...] to every per-round lerobot-eval invocation.
# Computed once and reused so every DAgger round operates on the SAME scenarios
# — that's the standard DAgger setup ("keep improving on the same hard set").
if [[ -n "$INTERVENTION_SAMPLE_FROM_FIRST" ]]; then
    INTERVENTION_SUBSET_JSON=$(python3 -c "
import random, sys
seed = int(sys.argv[1])
n = int(sys.argv[2])
pool = int(sys.argv[3])
random.seed(seed)
indices = sorted(random.sample(range(pool), n))
print('[' + ','.join(str(i) for i in indices) + ']')
" "$INTERVENTION_SAMPLE_SEED" "$INTERVENTION_N_EPISODES" "$INTERVENTION_SAMPLE_FROM_FIRST")
    echo "Intervention subset: $INTERVENTION_N_EPISODES scenarios sampled from [0..$((INTERVENTION_SAMPLE_FROM_FIRST-1))] (seed=$INTERVENTION_SAMPLE_SEED)"
    echo "  → $INTERVENTION_SUBSET_JSON"
    echo
fi

# Per-round training-output naming uses 'delta'/'rel' for relative and 'abs'/'abs' for absolute.
# train_sweep.sh writes "delta_basewrist" / "abs_basewrist" depending on USE_RELATIVE_ACTIONS.
if [[ "$ACTION_FORMAT" == "rel" ]]; then
    TRAIN_OUTPUT_ACTION_TAG="delta"
else
    TRAIN_OUTPUT_ACTION_TAG="abs"
fi
# Full model prefix for training output dirs (train_sweep.sh writes "pi05_", "diffusion_", "act_").
case "$MODEL" in
    pi)   TRAIN_OUTPUT_MODEL_PREFIX="pi05" ;;
    diff) TRAIN_OUTPUT_MODEL_PREFIX="diffusion" ;;
    act)  TRAIN_OUTPUT_MODEL_PREFIX="act" ;;
esac

# Base policy basename — every DAgger round writes to
# ${LEROBOT_ROOT}/outputs/training/${BASE_POLICY_NAME}_dag${N}/. Tying the dag
# output dirs to the actual initial policy name means it's always obvious where
# a given checkpoint came from (e.g. dag2 of which base), and the `_grip0` /
# `_grip2` / camera-suffix tags that distinguish base variants carry through
# into the dag dirs automatically.
#
# When --initial_policy_path is set we use its basename (walking up if the path
# points at /checkpoints/.../pretrained_model). When not set, fall back to the
# name train_sweep.sh would produce for a from-scratch round-0 base training.
BASE_POLICY_NAME=""
if [[ -n "$INITIAL_POLICY_PATH" ]]; then
    # Normalize the path: strip a trailing /pretrained_model and any
    # /checkpoints/<step|last>/ segment so basename returns the run dir name.
    _stripped="${INITIAL_POLICY_PATH%/}"
    _stripped="${_stripped%/pretrained_model}"
    _stripped="${_stripped%/checkpoints/*}"
    BASE_POLICY_NAME="$(basename "$_stripped")"
else
    # No initial policy: round 0 will train from scratch using train_sweep.sh,
    # which produces this name pattern. Round 1+ dag dirs hang off of it.
    BASE_POLICY_NAME="${TRAIN_OUTPUT_MODEL_PREFIX}_${BASE_SHORT}_${TRAIN_OUTPUT_ACTION_TAG}_basewrist"
fi
# Optional --run_tag suffix on the policy name (mirrors the dataset-name
# tagging above). Training dirs have no length limit (local FS), so we apply
# the tag unconditionally when set.
if [[ -n "$RUN_TAG" ]]; then
    BASE_POLICY_NAME="${BASE_POLICY_NAME}_${RUN_TAG}"
fi
# Intervention-method tag on the policy name (mirrors the dataset-name tagging
# above). "rrt" gets no tag — back-compat with every existing DAgger lineage.
if [[ -n "$METHOD_TAG" ]]; then
    BASE_POLICY_NAME="${BASE_POLICY_NAME}_${METHOD_TAG}"
fi
echo "Base policy name (dag dirs will be named ${BASE_POLICY_NAME}_dag{N}):  $BASE_POLICY_NAME"

# Derive the lineage tag dagger_progress.sh uses to filter training dirs. It
# expects `${BASE_SHORT}_${ACTION_TAG}_basewrist` as a SUFFIX of training dirs.
# When --initial_policy_path is set, our BASE_POLICY_NAME comes from its
# basename and may include infixes (e.g. _grip0) that the user's --base_short
# arg doesn't — so we extract the lineage out of BASE_POLICY_NAME instead of
# the raw --base_short.
PROGRESS_BASE_SHORT="${BASE_POLICY_NAME#${TRAIN_OUTPUT_MODEL_PREFIX}_}"
PROGRESS_BASE_SHORT="${PROGRESS_BASE_SHORT%_${TRAIN_OUTPUT_ACTION_TAG}_basewrist}"

# ── helpers ───────────────────────────────────────────────────────────────────

# Validate dataset names (copied from train_sweep.sh:73-126).
# Accepts a repo_id like "JennyWWW/splatsim_xxx" and aborts on failure.
validate_repo_name() {
    local repo="$1"
    local dataset_name="${repo#*/}"
    local errors=0
    if [[ "$dataset_name" =~ [^a-zA-Z0-9_.-] ]]; then
        echo "ERROR: dataset name has invalid chars (a-zA-Z0-9_.- only): '$dataset_name'" >&2; errors=1
    fi
    if [[ "$dataset_name" =~ ^[.-] || "$dataset_name" =~ [.-]$ ]]; then
        echo "ERROR: dataset name cannot start/end with - or .: '$dataset_name'" >&2; errors=1
    fi
    if [[ "$dataset_name" == *"--"* || "$dataset_name" == *".."* ]]; then
        echo "ERROR: dataset name cannot contain -- or ..: '$dataset_name'" >&2; errors=1
    fi
    if [[ "$dataset_name" == *.git || "$dataset_name" == *.ipynb ]]; then
        echo "ERROR: dataset name cannot end with .git or .ipynb: '$dataset_name'" >&2; errors=1
    fi
    if (( ${#repo} > 56 )); then
        echo "ERROR: dataset name exceeds 56 chars (${#repo}): '$repo'" >&2; errors=1
    fi
    if (( errors > 0 )); then return 1; fi
    return 0
}

# Per-round name derivations. All dag artifact names hang off BASE_DATASET_SHORT
# (derived from BASE_REPO above, splatsim_ prefix stripped), so the dag names
# inherit every identifying suffix from the base dataset (_grip0, _lowres, etc.)
# while staying short enough to fit under HuggingFace's 56-char repo-name limit.
#
# Action-format infix (a single letter, "r" or "a") keeps abs and rel runs from
# colliding on the same dataset names — both run flavors can co-exist on disk
# under the same base dataset. Single-letter (not "rel"/"abs") because long base
# names like `approach_lever_11_biasend_5path_grip0` (37 chars) only leave a
# ~10-char suffix budget once `JennyWWW/` (9 chars) is accounted for. The infix
# sits between BASE_DATASET_SHORT and _dag${N}, so the dataset short for round 1
# of a rel run on the grip0 base is e.g.
# "approach_lever_11_biasend_5path_grip0_r_dag1".
int_short_for_round()    { echo "${BASE_DATASET_SHORT}_${ACTION_INFIX}_dag$1"; }
int_repo_for_round()     { echo "${HF_USER}/$(int_short_for_round "$1")"; }
alias_short_for_round()  { echo "$(int_short_for_round "$1")_${MODEL}${ACTION_FORMAT}00"; }
alias_repo_for_round()   { echo "${HF_USER}/$(alias_short_for_round "$1")"; }
# Merged dataset uses a compact "_m" suffix (not "_merged") to fit under the
# 56-char HuggingFace repo-name limit. The merged dataset is a derived artifact
# that gets deleted between rounds anyway; the cryptic suffix is acceptable.
merged_short_for_round() { echo "${BASE_DATASET_SHORT}_${ACTION_INFIX}_dag$1_m"; }
merged_repo_for_round()  { echo "${HF_USER}/$(merged_short_for_round "$1")"; }

# Training mode for a given round number. As of the refactor away from
# "last round special-cases to FINAL_MODE", every dag round uses the same
# mode. FINAL_MODE now controls an optional post-loop phase, not a round.
mode_for_round() {
    echo "$INTERMEDIATE_MODE"
}

# Whether the post-loop "final from-scratch train on round N's data" phase
# runs. Disabled iff --intermediate_mode=scratch (round N already trained
# from scratch on the same data — the extra step would be a duplicate run).
do_final_scratch() {
    [[ "$FINAL_MODE" == "scratch" && "$INTERMEDIATE_MODE" != "scratch" ]]
}

# Output dir for the post-loop final from-scratch step. Naming pattern:
# ${BASE_POLICY_NAME}_dag${NUM_ROUNDS} (NO "_ft_" infix, since this step
# is from scratch, not a finetune). The finetune dag rounds live at
# ${BASE_POLICY_NAME}_ft_dag${r}, so the two are disambiguated by the
# presence/absence of "_ft_" — easy to distinguish in plots.
train_output_dir_final_scratch() {
    echo "$LEROBOT_ROOT/outputs/training/${BASE_POLICY_NAME}_dag${NUM_ROUNDS}"
}

# Training output dir naming. Mode-aware so finetune rounds get an "_ft"
# infix between the base policy name and the dag round suffix; scratch rounds
# don't (no finetuning was done). Examples for BASE_POLICY_NAME=foo:
#   foo_ft_dag3  ← finetune round 3
#   foo_dag10    ← scratch round 10 (typical for final_mode=scratch)
# Since mode is deterministic from round number, every callsite can hit the
# right path without passing extra state.
train_output_dir_for_round() {
    local round_n="$1"
    local mode
    mode="$(mode_for_round "$round_n")"
    local infix=""
    [[ "$mode" == "finetune" ]] && infix="_ft"
    # In retrain mode, the retrained round gets a suffix so we don't clobber
    # the original training dir. Other rounds (e.g. PREV_R when starting at
    # round N) resolve to their ORIGINAL non-suffixed dirs so we can pull the
    # input checkpoint from there.
    local suffix=""
    if [[ -n "$RETRAIN_ROUND" && "$round_n" == "$RETRAIN_ROUND" ]]; then
        suffix="_${RETRAIN_SUFFIX}"
    fi
    echo "$LEROBOT_ROOT/outputs/training/${BASE_POLICY_NAME}${infix}_dag${round_n}${suffix}"
}
# Back-compat aliases. They both call through to the mode-aware function
# so external scripts that referenced the old function names keep working.
train_output_dir_scratch()  { train_output_dir_for_round "$1"; }
train_output_dir_finetune() { train_output_dir_for_round "$1"; }

# Disk existence checks for resume detection. We try hard to detect crashed/
# partial states so resume doesn't skip past an incomplete step. These checks
# are necessary-but-not-sufficient (see notes below each function).
dataset_exists() {
    local d="$LEROBOT_CACHE/$1"
    # 1. Descriptor must exist.
    [[ -f "$d/meta/info.json" ]] || return 1
    # 2. Reject when total_episodes == 0 — this is the most common partial-
    #    crash signature (env init wrote info.json but the script errored
    #    before any rollout). Falls back to "no episodes" if info.json is
    #    unparsable.
    local n_eps
    n_eps=$(python3 -c "
import json, sys
try:
    info = json.load(open(sys.argv[1]))
    print(int(info.get('total_episodes', 0)))
except Exception:
    print(0)
" "$d/meta/info.json" 2>/dev/null)
    [[ -n "$n_eps" && "$n_eps" -gt 0 ]] || return 1
    # 3. At least one data parquet must be on disk. info.json could claim
    #    episodes that didn't actually finish writing.
    compgen -G "$d/data/chunk-*/*.parquet" >/dev/null || return 1
    return 0
    # NOTE: this does NOT verify that all `total_episodes` parquets are
    # present and well-formed. A recording that crashed at episode 50 of 100
    # would still report total_episodes=50 (the recorder commits per-success)
    # and pass this check — which is fine, the partial dataset is still
    # usable. What this DOES catch: total_episodes=0 (the empty-info case),
    # and data/ being empty.
}
stats_exists() {
    local short="$1"
    # When --action_format=abs, steps 2/5 are skipped entirely and these
    # relative-action sidecars are never produced. Treat that case as
    # "completed" for resume-detection purposes.
    if [[ "$ACTION_FORMAT" != "rel" ]]; then
        return 0
    fi
    # Files are named by chunk size (stats_rel{N}.json), produced by
    # compute_relative_stats.sh — currently chunks 50 (pi05) and 8 (diffusion).
    # We check whichever sidecar corresponds to the active model's chunk size;
    # the other one isn't needed but compute_relative_stats.sh writes both
    # anyway, so a successful run yields both and either model is satisfied.
    local needed_chunk
    case "$TRAIN_OUTPUT_MODEL_PREFIX" in
        diffusion*) needed_chunk=8 ;;
        pi05*|pi0*) needed_chunk=50 ;;
        *)          needed_chunk="" ;;
    esac
    if [[ -z "$needed_chunk" ]]; then
        # No chunk applies (e.g. act → absolute actions only). Treat as present.
        return 0
    fi
    [[ -f "$STATS_BASE/$short/stats_rel${needed_chunk}.json" ]]
}

# Read the policy's n_action_steps (== chunk size for the relative-action stats
# sidecar) from a resumed checkpoint's train_config.json. Echoes empty if the
# config doesn't exist or doesn't carry the field. Pass either the
# pretrained_model dir or a path that contains one.
n_action_steps_from_policy_path() {
    local policy_path="$1"
    local resolved cfg
    resolved="$(readlink -f "$policy_path" 2>/dev/null || echo "$policy_path")"
    for cfg in \
        "$resolved/train_config.json" \
        "$resolved/pretrained_model/train_config.json" \
        "$resolved/checkpoints/last/pretrained_model/train_config.json"
    do
        if [[ -f "$cfg" ]]; then
            python3 -c "
import json, sys
c = json.load(open(sys.argv[1]))
n = c.get('policy', {}).get('n_action_steps')
print(n if n is not None else '')
" "$cfg"
            return 0
        fi
    done
    echo ""
}
training_exists() {
    local exp_dir="$1"
    local last="$exp_dir/checkpoints/last/pretrained_model"
    # 1. Both descriptor AND weights file must exist (descriptor alone could
    #    be from a save that crashed mid-write).
    [[ -f "$last/train_config.json" && -f "$last/model.safetensors" ]] || return 1
    # 2. The last checkpoint's step number must match the planned total
    #    steps from train_config.json. Otherwise training was interrupted
    #    (crash, OOM, ctrl-C) and we should re-run from where it left off.
    #    lerobot-train's save_freq writes checkpoints/{step}/ on every save,
    #    and 'last' symlinks to the most recent. If steps=4000 in
    #    train_config.json but `last` points at 002000, training was cut off.
    local target_steps actual_steps
    target_steps=$(python3 -c "
import json, sys
try:
    cfg = json.load(open(sys.argv[1]))
    print(int(cfg.get('steps', 0)))
except Exception:
    print(0)
" "$last/train_config.json" 2>/dev/null)
    [[ -n "$target_steps" && "$target_steps" -gt 0 ]] || return 1
    # 'last' is a symlink to e.g. checkpoints/004000. Resolve and grab the
    # numeric step from the path.
    actual_steps=$(readlink -f "$exp_dir/checkpoints/last" 2>/dev/null \
        | xargs -r basename \
        | grep -oE '^[0-9]+$' || echo "")
    [[ -n "$actual_steps" ]] || return 1
    # Force base-10 arithmetic; otherwise "004000" gets parsed as octal and
    # silently mismatches 4000.
    (( 10#$actual_steps == target_steps )) || return 1
    return 0
    # NOTE: this does not verify that model.safetensors is non-corrupt (only
    # that the file exists with the right step count). A torch.load on a
    # corrupt safetensors would crash downstream — which is the right
    # behavior (better to fail loud at training-resume time than to silently
    # restart). It also assumes lerobot-train always saves at the final step,
    # which it does (the train loop calls save() once at step==total_steps).
}

# Resolve the latest checkpoint dir from a training experiment dir.
# Mirrors resume_training.sh's resolve_config. Tries (in order):
#   1. {exp}/checkpoints/last/pretrained_model
#   2. {exp}/checkpoints/<highest-numeric>/pretrained_model (when 'last' symlink missing)
#   3. {exp}/pretrained_model
#   4. {exp} itself (if it already contains train_config.json)
resolve_latest_checkpoint() {
    local exp_dir="$1"
    local candidate
    # 1. checkpoints/last
    candidate="$exp_dir/checkpoints/last/pretrained_model"
    [[ -d "$candidate" && -f "$candidate/train_config.json" ]] && { echo "$candidate"; return 0; }
    # 2. highest-numbered checkpoint under checkpoints/
    if [[ -d "$exp_dir/checkpoints" ]]; then
        local highest
        highest=$(ls -1 "$exp_dir/checkpoints" 2>/dev/null \
            | grep -E '^[0-9]+$' | sort -n | tail -1)
        if [[ -n "$highest" ]]; then
            candidate="$exp_dir/checkpoints/$highest/pretrained_model"
            [[ -d "$candidate" && -f "$candidate/train_config.json" ]] && { echo "$candidate"; return 0; }
        fi
    fi
    # 3. directly-nested pretrained_model
    candidate="$exp_dir/pretrained_model"
    [[ -d "$candidate" && -f "$candidate/train_config.json" ]] && { echo "$candidate"; return 0; }
    # 4. the exp_dir itself
    [[ -d "$exp_dir" && -f "$exp_dir/train_config.json" ]] && { echo "$exp_dir"; return 0; }
    echo "ERROR: cannot resolve a checkpoint dir from '$exp_dir'" >&2
    return 1
}

# Dry-run helper.
run_or_echo() {
    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY-RUN] $*"
    else
        echo "+ $*"
        "$@"
    fi
}

# ── SplatSim lifecycle helpers (used when --manage_splatsim is on) ────────────
# We track the child pid in MANAGED_SIM_PID. start_sim / stop_sim are idempotent
# so they can be called freely around training. The EXIT trap (installed once,
# right after these definitions) guarantees the sim is cleaned up even if the
# orchestrator dies mid-run.
MANAGED_SIM_PID=""
MANAGED_SIM_LOG=""

port_in_use() {
    # echoes pid bound to TCP port $1 (first one found), or empty.
    # Be defensive: lsof returns nonzero when there are no matches, and the
    # `| head` pipeline interacts badly with `set -o pipefail` — without the
    # `|| true` here the command substitution returns nonzero, which set -e
    # propagates and exits the script silently.
    local out
    out="$(lsof -ti "tcp:$1" -sTCP:LISTEN 2>/dev/null || true)"
    printf '%s\n' "${out%%$'\n'*}"
}

wait_for_port() {
    # Args: $1=port, $2=max_wait_seconds.
    # Returns 0 once a TCP connect succeeds; 1 on timeout.
    local port="$1" max_wait="$2"
    local i
    for ((i=1; i<=max_wait; i++)); do
        if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
            exec 3<&-; exec 3>&-
            return 0
        fi
        sleep 1
    done
    return 1
}

start_sim() {
    echo "[start_sim] entering (managed=$MANAGE_SPLATSIM, dry-run=$DRY_RUN, pid=${MANAGED_SIM_PID:-<unset>})"
    [[ "$MANAGE_SPLATSIM" == true ]] || return 0
    if [[ "$DRY_RUN" == true ]]; then
        # Idempotent in dry-run too — only print "would start" on transitions.
        [[ "$MANAGED_SIM_PID" == "DRYRUN" ]] && return 0
        echo "[DRY-RUN] would start SplatSim on port $ENV_EXTERNAL_PORT:"
        echo "[DRY-RUN]   cwd: $SPLATSIM_ROOT"
        echo "[DRY-RUN]   cmd: python scripts/launch_nodes.py --robot $SPLATSIM_ROBOT --robot_port $ENV_EXTERNAL_PORT --hostname $ENV_EXTERNAL_HOST --robot_name $SPLATSIM_ROBOT_NAME --eval_benchmark_repo_id $EVAL_BENCHMARK_REPO_ID"
        MANAGED_SIM_PID="DRYRUN"
        return 0
    fi
    # Already running under our management?
    if [[ -n "$MANAGED_SIM_PID" ]] && kill -0 "$MANAGED_SIM_PID" 2>/dev/null; then
        return 0
    fi
    # Port already in use by someone else?
    local existing
    existing="$(port_in_use "$ENV_EXTERNAL_PORT")"
    if [[ -n "$existing" ]]; then
        echo "ERROR: port $ENV_EXTERNAL_PORT already in use by pid $existing." >&2
        echo "  Either kill it, change --env_external_port, or pass --no_manage_splatsim." >&2
        exit 1
    fi
    mkdir -p "$LEROBOT_ROOT/outputs/dagger"
    MANAGED_SIM_LOG="$LEROBOT_ROOT/outputs/dagger/splatsim_$(date +%Y%m%d_%H%M%S).log"
    local launch_cmd=(
        python scripts/launch_nodes.py
        --robot              "$SPLATSIM_ROBOT"
        --robot_port         "$ENV_EXTERNAL_PORT"
        --hostname           "$ENV_EXTERNAL_HOST"
        --robot_name         "$SPLATSIM_ROBOT_NAME"
        --eval_benchmark_repo_id "$EVAL_BENCHMARK_REPO_ID"
    )
    echo "Starting SplatSim:"
    echo "  cwd:     $SPLATSIM_ROOT"
    echo "  cmd:     ${launch_cmd[*]}"
    echo "  log:     $MANAGED_SIM_LOG"
    # Launch sim in background. setsid puts it in its own session so it survives
    # SIGINT to our shell (we kill it explicitly in stop_sim / trap).
    (
        cd "$SPLATSIM_ROOT" || exit 1
        setsid "${launch_cmd[@]}" </dev/null >"$MANAGED_SIM_LOG" 2>&1
    ) &
    SIM_LAUNCH_BG_PID=$!
    echo "  launcher subshell pid: $SIM_LAUNCH_BG_PID (sim will fork under setsid)"
    # Capture the actual sim pid (the setsid grandchild) by polling for the
    # port to come up, then querying lsof.
    echo -n "  waiting for port $ENV_EXTERNAL_PORT to come up "
    if wait_for_port "$ENV_EXTERNAL_PORT" 300; then
        echo "ready."
    else
        echo
        echo "ERROR: SplatSim did not come up within 300s. Last 30 log lines:" >&2
        tail -30 "$MANAGED_SIM_LOG" >&2 || true
        exit 1
    fi
    MANAGED_SIM_PID="$(port_in_use "$ENV_EXTERNAL_PORT")"
    if [[ -z "$MANAGED_SIM_PID" ]]; then
        echo "WARNING: port is up but pid lookup via lsof returned empty." >&2
    else
        echo "  pid:     $MANAGED_SIM_PID"
    fi
    # Give the sim a beat to finish internal init beyond just port-bind.
    sleep 3
}

stop_sim() {
    [[ "$MANAGE_SPLATSIM" == true ]] || return 0
    if [[ "$MANAGED_SIM_PID" == "DRYRUN" ]]; then
        echo "[DRY-RUN] would stop SplatSim on port $ENV_EXTERNAL_PORT"
        MANAGED_SIM_PID=""
        return 0
    fi
    [[ -z "$MANAGED_SIM_PID" ]] && return 0
    if ! kill -0 "$MANAGED_SIM_PID" 2>/dev/null; then
        MANAGED_SIM_PID=""
        return 0
    fi
    echo "Stopping SplatSim (pid=$MANAGED_SIM_PID)..."
    # SIGTERM the whole process group so any pybullet/gaussian-splat subprocs die too.
    local pgid
    pgid="$(ps -o pgid= -p "$MANAGED_SIM_PID" 2>/dev/null | tr -d ' ')"
    if [[ -n "$pgid" ]]; then
        kill -TERM -"$pgid" 2>/dev/null || true
    else
        kill -TERM "$MANAGED_SIM_PID" 2>/dev/null || true
    fi
    # Wait up to 30s for graceful exit.
    local i
    for ((i=1; i<=30; i++)); do
        kill -0 "$MANAGED_SIM_PID" 2>/dev/null || break
        sleep 1
    done
    if kill -0 "$MANAGED_SIM_PID" 2>/dev/null; then
        echo "  sim did not exit on SIGTERM; sending SIGKILL."
        [[ -n "$pgid" ]] && kill -KILL -"$pgid" 2>/dev/null || kill -KILL "$MANAGED_SIM_PID" 2>/dev/null || true
    fi
    # Wait for the port to actually free (kernel may keep it in TIME_WAIT briefly).
    for ((i=1; i<=15; i++)); do
        [[ -z "$(port_in_use "$ENV_EXTERNAL_PORT")" ]] && break
        sleep 1
    done
    MANAGED_SIM_PID=""
    echo "  stopped."
}

# Cleanup on exit (success or crash). Idempotent.
trap 'stop_sim' EXIT

# ── pre-flight: validate every round's derived names ──────────────────────────
echo "Pre-flight: validating derived repo names for $NUM_ROUNDS rounds..."
ANY_NAME_TOO_LONG=false
ALIAS_NAMES_OVERFLOW=false
for r in $(seq 1 "$NUM_ROUNDS"); do
    # Always validate the raw intervention and merged-output names.
    for name in "$(int_repo_for_round "$r")" "$(merged_repo_for_round "$r")"; do
        if ! validate_repo_name "$name"; then
            ANY_NAME_TOO_LONG=true
        fi
    done
    # Only validate the alias name when the alias step is enabled.
    if [[ "$SKIP_ALIAS_STEP" == false ]]; then
        if ! validate_repo_name "$(alias_repo_for_round "$r")"; then
            ANY_NAME_TOO_LONG=true
            ALIAS_NAMES_OVERFLOW=true
        fi
    fi
done
if [[ "$ANY_NAME_TOO_LONG" == true ]]; then
    echo "ERROR: one or more derived repo names violate the 56-char limit or other rules." >&2
    if [[ "$ALIAS_NAMES_OVERFLOW" == true ]]; then
        echo "  Note: the alias suffix '_${MODEL}${ACTION_FORMAT}00' adds 8 chars on top of '_dag{N}'." >&2
        echo "  Either:" >&2
        echo "    (a) pass --skip_alias_step to bypass augment_ratios_sweep.sh and merge" >&2
        echo "        intervention datasets directly (loses forward-compat with" >&2
        echo "        future ratio>0 blending augmentations on these rounds), OR" >&2
        echo "    (b) use a shorter base dataset name (current dag-name prefix '$BASE_DATASET_SHORT'," >&2
        echo "        length ${#BASE_DATASET_SHORT}; was derived from BASE_REPO=$BASE_REPO)." >&2
    else
        echo "  Use a shorter base dataset name (current dag-name prefix '$BASE_DATASET_SHORT'," >&2
        echo "  length ${#BASE_DATASET_SHORT}; was derived from BASE_REPO=$BASE_REPO) and retry." >&2
    fi
    exit 1
fi
echo "  ✓ all derived names valid (alias step: $([ "$SKIP_ALIAS_STEP" == true ] && echo SKIPPED || echo enabled))."
echo

# ── resume detection ──────────────────────────────────────────────────────────
# For each round r, count completed steps (1..7). Step 7 (cleanup) is treated
# as auto-complete once step 6 succeeds, so we only probe steps 1..6.
declare -a ROUND_COMPLETED_STEPS  # parallel array indexed 1..N
ROUND_COMPLETED_STEPS[0]=0  # unused

for r in $(seq 1 "$NUM_ROUNDS"); do
    int_short="$(int_short_for_round "$r")"
    int_repo="$(int_repo_for_round "$r")"
    alias_repo="$(alias_repo_for_round "$r")"
    merged_short="$(merged_short_for_round "$r")"
    merged_repo="$(merged_repo_for_round "$r")"
    ft_dir="$(train_output_dir_for_round "$r")"
    scratch_dir="$ft_dir"  # both modes write to the same path under the new naming

    step=0
    # Check training output FIRST. If the trained policy is on disk, the round
    # is complete regardless of intermediate artifacts (the merged dataset and
    # its stats sidecar may have been legitimately deleted by step 7 cleanup
    # in the next round). Without this short-circuit, a sequential counter
    # would mark a fully-completed round as PARTIAL whenever cleanup ran.
    if dataset_exists "$int_repo" \
       && { training_exists "$ft_dir" || training_exists "$scratch_dir"; }; then
        step=6
    else
        dataset_exists "$int_repo"             && step=1
        (( step == 1 )) && stats_exists "$int_short"      && step=2
        if [[ "$SKIP_ALIAS_STEP" == true ]]; then
            # Step 3 is a no-op when skipped; auto-advance.
            (( step == 2 )) && step=3
        else
            (( step == 2 )) && dataset_exists "$alias_repo"   && step=3
        fi
        (( step == 3 )) && dataset_exists "$merged_repo"  && step=4
        (( step == 4 )) && stats_exists "$merged_short"   && step=5
    fi
    ROUND_COMPLETED_STEPS[$r]=$step
done

# Find the furthest round/step combo.
FURTHEST_ROUND=0
FURTHEST_STEP=0
for r in $(seq 1 "$NUM_ROUNDS"); do
    s=${ROUND_COMPLETED_STEPS[$r]}
    if (( s > 0 )); then
        FURTHEST_ROUND=$r
        FURTHEST_STEP=$s
    fi
    # Stop scanning once we hit an incomplete round (later rounds depend on this one).
    (( s < 6 )) && break
done

# Probe the optional post-loop final-scratch step.
FINAL_SCRATCH_DONE=false
if do_final_scratch; then
    if training_exists "$(train_output_dir_final_scratch)"; then
        FINAL_SCRATCH_DONE=true
    fi
fi

# Print the step legend so the user understands what "5/6 steps complete"
# refers to in the detection table below.
echo "Steps per round:"
echo "  1. Record interventions on benchmark scenarios (lerobot-eval --intervention.method=...)"
echo "  2. Compute sidecar relative-action stats for the intervention dataset"
if [[ "$SKIP_ALIAS_STEP" == true ]]; then
    echo "  3. (SKIPPED — --skip_alias_step) Hardlink-alias under _${MODEL}${ACTION_FORMAT}00 naming"
else
    echo "  3. Hardlink-alias intervention dataset under _${MODEL}${ACTION_FORMAT}00 naming (augment_ratios_sweep.sh ratio=0)"
fi
echo "  4. Cumulative merge: base + dag1[..dag${NUM_ROUNDS}] → training dataset"
echo "  5. Compute sidecar relative-action stats for the merged dataset"
echo "  6. Train policy ($INTERMEDIATE_MODE) on the merged dataset"
if do_final_scratch; then
    echo "  + (post-loop) Extra from-scratch train on round $NUM_ROUNDS's merged data"
    echo "                → $(train_output_dir_final_scratch)"
fi
echo

echo "Resume detection:"
for r in $(seq 1 "$NUM_ROUNDS"); do
    s=${ROUND_COMPLETED_STEPS[$r]}
    if (( s == 6 )); then
        verdict="COMPLETE (dataset ✓, policy ✓)"
    elif (( s >= 1 )); then
        verdict="PARTIAL — will redo from step $((s + 1))"
    else
        verdict="not started"
    fi
    echo "  Round $r: $s/6 steps complete — $verdict"
done
if do_final_scratch; then
    if [[ "$FINAL_SCRATCH_DONE" == true ]]; then
        echo "  Final scratch: COMPLETE (policy ✓)"
    else
        echo "  Final scratch: not done"
    fi
fi
echo
echo "Note: a round is only considered fully complete when BOTH its intervention"
echo "      dataset (step 1, non-empty) AND its trained policy checkpoint (step 6,"
echo "      with model.safetensors) are present on disk."
echo

# Restart-from-scratch handler. Prompts the user to confirm before deleting
# every dag artifact on disk. Sets EFFECTIVE_START_ROUND=1, EFFECTIVE_START_STEP=1
# on success. Exits the script on abort.
restart_from_scratch() {
    RESTART_PATHS=()
    for r in $(seq 1 "$NUM_ROUNDS"); do
        RESTART_PATHS+=( "$LEROBOT_CACHE/$(int_repo_for_round "$r")" )
        RESTART_PATHS+=( "$LEROBOT_CACHE/$(alias_repo_for_round "$r")" )
        RESTART_PATHS+=( "$LEROBOT_CACHE/$(merged_repo_for_round "$r")" )
        RESTART_PATHS+=( "$STATS_BASE/$(int_short_for_round "$r")" )
        RESTART_PATHS+=( "$STATS_BASE/$(merged_short_for_round "$r")" )
        RESTART_PATHS+=( "$(train_output_dir_for_round "$r")" )
        # Intervention output dir lives INSIDE the training dir
        # (<train>/dagger/interventions/), so the rm above already takes it
        # out. Legacy paths from older orchestrator layouts kept here so
        # --force_restart still cleans pre-migration runs:
        #   outputs/dagger/<train_basename>/   (mirror layout, briefly used)
        #   outputs/dagger/round_${r}/          (original layout)
        RESTART_PATHS+=( "$LEROBOT_ROOT/outputs/dagger/$(basename "$(train_output_dir_for_round "$r")")" )
        RESTART_PATHS+=( "$LEROBOT_ROOT/outputs/dagger/round_${r}" )
    done
    do_final_scratch && RESTART_PATHS+=( "$(train_output_dir_final_scratch)" )
    EXISTING_PATHS=()
    for p in "${RESTART_PATHS[@]}"; do
        [[ -e "$p" ]] && EXISTING_PATHS+=( "$p" )
    done
    if (( ${#EXISTING_PATHS[@]} == 0 )); then
        echo "No dag artifacts to delete; nothing to do."
    else
        echo "The following dag artifacts will be deleted (rm -rf):"
        for p in "${EXISTING_PATHS[@]}"; do
            echo "  $p"
        done
        echo
    fi
    echo -n "Type 'restart' to confirm: "
    read -r CONFIRM
    [[ "$CONFIRM" == "restart" ]] || { echo "Aborted."; exit 1; }
    echo "Clearing prior dag artifacts..."
    for p in "${EXISTING_PATHS[@]}"; do
        run_or_echo rm -rf "$p"
    done
    EFFECTIVE_START_ROUND=1
    EFFECTIVE_START_STEP=1
}

# Decide start_round. EFFECTIVE_START_ROUND > NUM_ROUNDS means "skip the main
# dag-round loop entirely and only run the post-loop final-scratch step."
ALL_ROUNDS_DONE=false
(( FURTHEST_ROUND == NUM_ROUNDS && FURTHEST_STEP == 6 )) && ALL_ROUNDS_DONE=true

if [[ -n "$RETRAIN_ROUND" ]]; then
    # Retrain mode: jump straight to round $RETRAIN_ROUND's training step,
    # skipping all earlier rounds and the data steps (1-5) for that round.
    # Auto-pick the entry step based on what data is on disk:
    #   - if the merged dataset (+ stats sidecar in rel mode) exists, start step 6.
    #   - else if all per-round intervention datasets 1..N exist, start step 4
    #     (re-merge from existing intervention datasets, recompute stats, train).
    #   - else error out — we can't reconstitute the data.
    EFFECTIVE_START_ROUND="$RETRAIN_ROUND"
    rt_merged_repo="$(merged_repo_for_round "$RETRAIN_ROUND")"
    rt_merged_short="$(merged_short_for_round "$RETRAIN_ROUND")"
    rt_have_merged=true
    if ! dataset_exists "$rt_merged_repo"; then
        rt_have_merged=false
    elif [[ "$ACTION_FORMAT" == "rel" ]] && ! stats_exists "$rt_merged_short"; then
        rt_have_merged=false
    fi
    if [[ "$rt_have_merged" == true ]]; then
        EFFECTIVE_START_STEP=6
        echo "Retrain mode: round $RETRAIN_ROUND, merged dataset present → running step 6 only."
    else
        # Need to re-merge. step 4 reads from either the alias datasets (default)
        # or the raw intervention datasets (--skip_alias_step). All rounds
        # 1..$RETRAIN_ROUND must have those sources available, otherwise the
        # merge will fail mid-flight. Validate up front.
        for prev in $(seq 1 "$RETRAIN_ROUND"); do
            if [[ "$SKIP_ALIAS_STEP" == true ]]; then
                rt_need_repo="$(int_repo_for_round "$prev")"
            else
                rt_need_repo="$(alias_repo_for_round "$prev")"
            fi
            if ! dataset_exists "$rt_need_repo"; then
                echo "ERROR: --retrain_round=$RETRAIN_ROUND with missing merged dataset needs" >&2
                echo "  $rt_need_repo (round $prev's merge source) on disk, but it is missing." >&2
                echo "  Re-run the orchestrator normally (without --retrain_round) to" >&2
                echo "  regenerate the data pipeline before retrying." >&2
                exit 1
            fi
        done
        EFFECTIVE_START_STEP=4
        echo "Retrain mode: round $RETRAIN_ROUND, merged dataset missing → running steps 4-6."
    fi
    echo "  Retrain output dir: $(train_output_dir_for_round "$RETRAIN_ROUND")"
    echo "  (post-loop final-scratch phase skipped in retrain mode)"
elif [[ -n "$START_ROUND" ]]; then
    # Explicit override.
    if ! [[ "$START_ROUND" =~ ^[0-9]+$ ]] || (( START_ROUND < 1 || START_ROUND > NUM_ROUNDS )); then
        echo "ERROR: --start_round=$START_ROUND must be between 1 and $NUM_ROUNDS" >&2; exit 1
    fi
    EFFECTIVE_START_ROUND=$START_ROUND
    EFFECTIVE_START_STEP=1
    echo "Using explicit --start_round=$START_ROUND (begins at step 1)."
elif [[ "$FORCE_RESTART" == true ]]; then
    EFFECTIVE_START_ROUND=1
    EFFECTIVE_START_STEP=1
    if [[ "$DRY_RUN" != true ]]; then
        echo "--force_restart: this will rm -rf all dag1..dag${NUM_ROUNDS} datasets,"
        echo "  their stats sidecars, and per-round training output dirs."
        echo -n "Type 'restart' to confirm: "
        read -r CONFIRM
        [[ "$CONFIRM" == "restart" ]] || { echo "Aborted."; exit 1; }
    fi
    echo "--force_restart: clearing prior dag artifacts..."
    for r in $(seq 1 "$NUM_ROUNDS"); do
        run_or_echo rm -rf "$LEROBOT_CACHE/$(int_repo_for_round "$r")"
        run_or_echo rm -rf "$LEROBOT_CACHE/$(alias_repo_for_round "$r")"
        run_or_echo rm -rf "$LEROBOT_CACHE/$(merged_repo_for_round "$r")"
        run_or_echo rm -rf "$STATS_BASE/$(int_short_for_round "$r")"
        run_or_echo rm -rf "$STATS_BASE/$(merged_short_for_round "$r")"
        # New layout nests interventions under the training dir
        # (<train>/dagger/...), so this rm already takes them out.
        run_or_echo rm -rf "$(train_output_dir_for_round "$r")"
        # Legacy intervention layouts kept here for back-compat:
        run_or_echo rm -rf "$LEROBOT_ROOT/outputs/dagger/$(basename "$(train_output_dir_for_round "$r")")"
        run_or_echo rm -rf "$LEROBOT_ROOT/outputs/dagger/round_${r}"
    done
    do_final_scratch && run_or_echo rm -rf "$(train_output_dir_final_scratch)"
elif (( FURTHEST_ROUND == 0 )); then
    # No prior work detected.
    EFFECTIVE_START_ROUND=1
    EFFECTIVE_START_STEP=1
    echo "No prior DAgger artifacts found; starting from round 1, step 1."
elif [[ "$ALL_ROUNDS_DONE" == true ]] && { ! do_final_scratch || [[ "$FINAL_SCRATCH_DONE" == true ]]; }; then
    # Everything is done — either no final-scratch was requested, or it
    # already exists. Offer restart-from-scratch as the only do-something path.
    if do_final_scratch; then
        MSG="all $NUM_ROUNDS dag rounds AND the post-loop final-scratch step are complete"
    else
        MSG="all $NUM_ROUNDS dag rounds complete (no final-scratch requested with --final_mode=$FINAL_MODE)"
    fi
    echo "Pipeline detected: $MSG."
    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY-RUN] Would exit without running anything."
        exit 0
    fi
    echo -n "Re-run anyway? [n/restart-from-scratch] "
    read -r RESP
    case "$RESP" in
        restart-from-scratch|restart) restart_from_scratch ;;
        *) echo "Aborted."; exit 0 ;;
    esac
else
    # Auto-detected partial progress. Compute (next_r, next_s, msg) describing
    # the immediate next step. Cases:
    #   (a) all rounds done, final-scratch pending → NEXT_R=NUM_ROUNDS+1 (skip loop)
    #   (b) furthest round fully trained → start next round at step 1
    #   (c) furthest round mid-flight → resume at step (furthest_step+1)
    if [[ "$ALL_ROUNDS_DONE" == true ]]; then
        # do_final_scratch && !FINAL_SCRATCH_DONE (the other branches above caught the rest).
        NEXT_R=$((NUM_ROUNDS + 1))
        NEXT_S=1
        MSG="all $NUM_ROUNDS dag rounds complete; next is the post-loop final from-scratch train"
    elif (( FURTHEST_STEP == 6 )); then
        NEXT_R=$((FURTHEST_ROUND + 1))
        NEXT_S=1
        MSG="round $FURTHEST_ROUND fully complete; next is round $NEXT_R, step 1"
    else
        NEXT_R=$FURTHEST_ROUND
        NEXT_S=$((FURTHEST_STEP + 1))
        MSG="round $FURTHEST_ROUND has $FURTHEST_STEP/6 steps complete; next is round $FURTHEST_ROUND, step $NEXT_S"
    fi

    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY-RUN] Detected: $MSG. Would resume there."
        EFFECTIVE_START_ROUND=$NEXT_R
        EFFECTIVE_START_STEP=$NEXT_S
    else
        echo "Pipeline detected: $MSG"
        echo -n "Resume from there? [Y/n/restart-from-scratch] "
        read -r RESP
        case "$RESP" in
            ""|y|Y|yes)
                EFFECTIVE_START_ROUND=$NEXT_R
                EFFECTIVE_START_STEP=$NEXT_S
                ;;
            n|N|no)               echo "Aborted."; exit 0 ;;
            restart-from-scratch|restart) restart_from_scratch ;;
            *) echo "Unrecognized response. Aborting." >&2; exit 1 ;;
        esac
    fi
fi

# Stale-merged-dataset sweep. The orchestrator's step 7 cleans up round (r-1)'s
# merged dataset after round r's training, leaving at most one merged dataset on
# disk at a time. But if a prior run crashed (e.g. SIGABRT from PyBullet/Tcl
# teardown, or the user Ctrl-C'd between steps 6 and 7), step 7 may never have
# run and an older merged_m dir is now an orphan. Sweep them on every resume.
#
# Rules:
#   - Round R is "the round we're entering". Its merged dataset (dag{R}_m) is
#     potentially needed when starting at step >= 5; we KEEP it.
#   - When EFFECTIVE_START_ROUND > NUM_ROUNDS (i.e. entering the post-loop
#     final-scratch step), we KEEP round NUM_ROUNDS's merged dataset since the
#     final-scratch step trains on it.
#   - Everything else is stale and removed.
KEEP_MERGED_ROUND="$EFFECTIVE_START_ROUND"
(( KEEP_MERGED_ROUND > NUM_ROUNDS )) && KEEP_MERGED_ROUND="$NUM_ROUNDS"
echo "Stale-merged-dataset sweep (keeping at most round ${KEEP_MERGED_ROUND}'s merged dataset)..."
stale_found=0
for stale_r in $(seq 1 "$NUM_ROUNDS"); do
    if (( stale_r == KEEP_MERGED_ROUND )); then continue; fi
    stale_merged_repo="$(merged_repo_for_round "$stale_r")"
    stale_merged_short="$(merged_short_for_round "$stale_r")"
    stale_dataset_path="$LEROBOT_CACHE/$stale_merged_repo"
    stale_stats_path="$STATS_BASE/$stale_merged_short"
    found_this_round=0
    if [[ -e "$stale_dataset_path" ]]; then
        echo "  Removing stale merged dataset: $stale_dataset_path"
        run_or_echo rm -rf "$stale_dataset_path"
        found_this_round=1
    fi
    if [[ -e "$stale_stats_path" ]]; then
        echo "  Removing stale merged stats:   $stale_stats_path"
        run_or_echo rm -rf "$stale_stats_path"
        found_this_round=1
    fi
    (( found_this_round )) && stale_found=1
done
(( stale_found )) || echo "  (none found)"

# If we're starting mid-pipeline at a dag round > 1 OR entering only the
# post-loop final-scratch phase (EFFECTIVE_START_ROUND > NUM_ROUNDS), the
# previous round's training output must exist so we can pull current_policy
# from it. Round 0 is optional (run train_sweep.sh) only when
# EFFECTIVE_START_ROUND == 1.
#
# CURRENT_POLICY isn't consumed by the final-scratch step itself (it trains
# from scratch), but it's reported in the final summary as "last finetune-round
# policy" — resolve it from round NUM_ROUNDS in that case so the summary is
# accurate when the orchestrator re-enters only to do the final-scratch step.
if (( EFFECTIVE_START_ROUND > 1 )); then
    PREV_R=$((EFFECTIVE_START_ROUND - 1))
    (( PREV_R > NUM_ROUNDS )) && PREV_R="$NUM_ROUNDS"
    PREV_FT="$(train_output_dir_for_round "$PREV_R")"
    PREV_SCRATCH="$PREV_FT"
    if training_exists "$PREV_FT"; then
        CURRENT_POLICY="$(resolve_latest_checkpoint "$PREV_FT")"
    else
        echo "ERROR: starting at round $EFFECTIVE_START_ROUND requires a trained policy from round $PREV_R," >&2
        echo "  but $PREV_FT doesn't exist." >&2
        exit 1
    fi
    if (( EFFECTIVE_START_ROUND > NUM_ROUNDS )); then
        echo "Post-loop final-scratch phase; last finetune-round policy: $CURRENT_POLICY"
    else
        echo "Round $EFFECTIVE_START_ROUND will resume from policy: $CURRENT_POLICY"
    fi
fi

# ── shared training helpers (used by Round 0, the per-round loop, and the
# post-loop final-scratch phase) ──────────────────────────────────────────────
# When the sim is orchestrator-managed, omit --env.external_port from training
# so lerobot-train falls through to its own in-process env factory. In
# --no_manage_splatsim mode, forward the port so it shares the user's sim.
TRAIN_EXT_PORT_SWEEP=()
TRAIN_EXT_PORT_RESUME=()
if [[ "$MANAGE_SPLATSIM" == false ]]; then
    TRAIN_EXT_PORT_SWEEP=( --env_external_port="$ENV_EXTERNAL_PORT" )
    TRAIN_EXT_PORT_RESUME=( --env.external_port="$ENV_EXTERNAL_PORT" )
fi
# Offline-mode flag: when --push_to_hub is NOT set on the orchestrator (the
# default), disable the trained policy's hub push.
OFFLINE_POLICY_ARG=()
if [[ "$PUSH_TO_HUB" == false ]]; then
    OFFLINE_POLICY_ARG+=( "--policy.push_to_hub=false" )
fi
# Runner that tolerates the well-known PyBullet/Tcl teardown abort.
# lerobot-train completes training and eval, saves the checkpoint, then
# SIGABRTs in env.close() from "Tcl_AsyncDelete: async handler deleted by
# the wrong thread". Exit code is 134 (SIGABRT) but the checkpoint is
# already on disk and complete. We verify training_exists() (which checks
# model.safetensors AND that the last-checkpoint step matches the planned
# target) and treat the run as successful when that's the case. Reads
# TRAIN_OUTPUT_DIR from the caller's scope.
run_training_step() {
    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY-RUN] $*"
        return 0
    fi
    echo "+ $*"
    local rc=0
    "$@" || rc=$?
    if (( rc == 0 )); then
        return 0
    fi
    if training_exists "$TRAIN_OUTPUT_DIR"; then
        echo "WARNING: training subprocess exited with status $rc, but" >&2
        echo "  $TRAIN_OUTPUT_DIR/checkpoints/last/pretrained_model" >&2
        echo "  is present with the expected step count — treating as success" >&2
        echo "  (likely a benign PyBullet/Tcl teardown abort)." >&2
        return 0
    fi
    echo "ERROR: training failed (exit $rc) and expected checkpoint at" >&2
    echo "  $TRAIN_OUTPUT_DIR/checkpoints/last/pretrained_model" >&2
    echo "  is missing or incomplete." >&2
    return "$rc"
}
# Print GPU state before training. Useful when finetune OOMs and you want
# to identify the actual memory consumers.
print_gpu_state() {
    if [[ "$DRY_RUN" != true ]] && command -v nvidia-smi >/dev/null 2>&1; then
        echo "GPU state before training:"
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1 \
            | sed 's/^/  /'
        nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader 2>&1 \
            | sed 's/^/  total: /'
        echo
    fi
}

# ── Round 0: train base if needed; always resolve CURRENT_POLICY for round 1 ──
# Split into two responsibilities:
#   (a) resolve CURRENT_POLICY for round 1's input (always needed when round 1
#       is in the loop, even if resuming mid-round means step 1 won't actually
#       run — the round header still prints it).
#   (b) actually invoke train_sweep.sh on the base dataset (only when starting
#       from step 1 fresh AND no prior base training exists AND no
#       --initial_policy_path was given).
if (( EFFECTIVE_START_ROUND == 1 )); then
    BASE_TRAINING_DIR="$LEROBOT_ROOT/outputs/training/${TRAIN_OUTPUT_MODEL_PREFIX}_${BASE_SHORT}_${TRAIN_OUTPUT_ACTION_TAG}_basewrist"

    if [[ -n "$INITIAL_POLICY_PATH" ]]; then
        # Resolve the user-supplied path to a pretrained_model dir.
        if [[ -d "$INITIAL_POLICY_PATH/checkpoints/last/pretrained_model" ]]; then
            CURRENT_POLICY="$INITIAL_POLICY_PATH/checkpoints/last/pretrained_model"
        elif [[ -d "$INITIAL_POLICY_PATH/pretrained_model" ]]; then
            CURRENT_POLICY="$INITIAL_POLICY_PATH/pretrained_model"
        elif [[ -d "$INITIAL_POLICY_PATH" && -f "$INITIAL_POLICY_PATH/train_config.json" ]]; then
            CURRENT_POLICY="$INITIAL_POLICY_PATH"
        else
            CURRENT_POLICY="$(resolve_latest_checkpoint "$INITIAL_POLICY_PATH")"
        fi
        echo "Round 0: skipped (using --initial_policy_path=$CURRENT_POLICY)."
    elif training_exists "$BASE_TRAINING_DIR"; then
        CURRENT_POLICY="$(resolve_latest_checkpoint "$BASE_TRAINING_DIR")"
        echo "Round 0: skipped (found existing base training at $BASE_TRAINING_DIR)."
    elif (( EFFECTIVE_START_STEP == 1 )); then
        # Starting fresh and no prior base — train it now. The orchestrator
        # has not started its managed sim yet (start_sim happens inside the
        # per-round loop), so in managed mode we omit --env_external_port and
        # let lerobot-train spawn its own in-process sim.
        echo "Round 0: training base policy on $BASE_REPO..."
        # train_sweep.sh pre-flights stats_rel{8,50}.json when training a
        # relative-action policy. Generate them up-front if missing so the
        # base-train step is self-contained. Skip in abs mode (--no_relative
        # below disables the pre-flight).
        if [[ "$ACTION_FORMAT" == "rel" ]] && ! stats_exists "$BASE_DATASET_SHORT"; then
            echo "Round 0: computing sidecar rel-action stats for base ($BASE_DATASET_SHORT)..."
            run_or_echo bash "$SCRIPT_DIR/compute_relative_stats.sh" --dataset_repo="$BASE_REPO"
        fi
        ROUND0_ABS_ACTION_ARG=()
        if [[ "$ACTION_FORMAT" == "abs" ]]; then
            ROUND0_ABS_ACTION_ARG=( --no_relative )
        fi
        # Use run_training_step (not run_or_echo) so the benign PyBullet/Tcl
        # teardown SIGABRT at the end of lerobot-train is tolerated the same
        # way it is in the per-round and post-loop training steps. Reads
        # TRAIN_OUTPUT_DIR from this scope.
        TRAIN_OUTPUT_DIR="$BASE_TRAINING_DIR"
        run_training_step bash "$SCRIPT_DIR/train_sweep.sh" \
            --dataset_repo="$BASE_REPO" \
            --model="$TRAIN_OUTPUT_MODEL_PREFIX" \
            "${ROUND0_ABS_ACTION_ARG[@]}" \
            "${TRAIN_EXT_PORT_SWEEP[@]}" \
            "${OFFLINE_POLICY_ARG[@]}"
        if [[ "$DRY_RUN" != true ]]; then
            CURRENT_POLICY="$(resolve_latest_checkpoint "$BASE_TRAINING_DIR")"
        else
            CURRENT_POLICY="$BASE_TRAINING_DIR/checkpoints/last/pretrained_model"
        fi
    else
        # Resuming mid-round-1 with no initial_policy_path AND no base training
        # dir. We have no way to source a round-1 input policy. Abort.
        echo "ERROR: resuming at round 1, step $EFFECTIVE_START_STEP, but cannot resolve" >&2
        echo "  a round-1 input policy. Either:" >&2
        echo "    - pass --initial_policy_path=<existing checkpoint>, or" >&2
        echo "    - run from step 1 so train_sweep.sh trains the base policy." >&2
        exit 1
    fi
fi

# ── per-round loop ────────────────────────────────────────────────────────────
# (training helpers — TRAIN_EXT_PORT_*, OFFLINE_POLICY_ARG, run_training_step,
#  print_gpu_state — were hoisted above Round 0 so they're available to all
#  three training phases.)
#
# In retrain mode the loop runs ONLY round $RETRAIN_ROUND (chain breaks at the
# suffixed output dir — subsequent rounds would be ambiguous about which
# checkpoint to source as input). Otherwise it runs through round $NUM_ROUNDS.
EFFECTIVE_END_ROUND="$NUM_ROUNDS"
[[ -n "$RETRAIN_ROUND" ]] && EFFECTIVE_END_ROUND="$RETRAIN_ROUND"
for r in $(seq "$EFFECTIVE_START_ROUND" "$EFFECTIVE_END_ROUND"); do
    INT_SHORT="$(int_short_for_round "$r")"
    INT_REPO="$(int_repo_for_round "$r")"
    ALIAS_REPO="$(alias_repo_for_round "$r")"
    MERGED_SHORT="$(merged_short_for_round "$r")"
    MERGED_REPO="$(merged_repo_for_round "$r")"
    MERGED_STATS_DIR="$STATS_BASE/$MERGED_SHORT"

    # Every dag round uses INTERMEDIATE_MODE. FINAL_MODE controls a separate
    # post-loop phase below — not a round.
    MODE="$INTERMEDIATE_MODE"
    TRAIN_OUTPUT_DIR="$(train_output_dir_for_round "$r")"

    # The starting step within this round.
    if (( r == EFFECTIVE_START_ROUND )); then STEP=$EFFECTIVE_START_STEP; else STEP=1; fi

    echo
    echo "════════════════════════════════════════════════════════════════"
    echo "ROUND $r / $NUM_ROUNDS   (mode=$MODE, starting at step $STEP)"
    echo "  Intervention dataset: $INT_REPO"
    echo "  Alias dataset:        $ALIAS_REPO"
    echo "  Merged dataset:       $MERGED_REPO"
    echo "  Training output:      $TRAIN_OUTPUT_DIR"
    echo "  Current policy:       $CURRENT_POLICY"
    echo "════════════════════════════════════════════════════════════════"

    # Ensure the external SplatSim is running before step 1 (intervention
    # recording) — the only sim-using step. Steps 2-5 are pure data ops
    # (stats, alias, merge, stats) that don't touch the sim, and step 6
    # explicitly stops the sim before training. Idempotent: no-op if we
    # already launched it or if --no_manage_splatsim.
    if (( STEP <= 1 )); then
        start_sim
    fi

    # Step 1: Record interventions.
    if (( STEP <= 1 )); then
        echo "--- Round $r, Step 1: record interventions ---"
        # Optional --env.eval_benchmark_subset for random-sample mode. When
        # empty, lerobot-eval defaults to the first N scenarios.
        SUBSET_ARG=()
        if [[ -n "$INTERVENTION_SUBSET_JSON" ]]; then
            SUBSET_ARG+=( "--env.eval_benchmark_subset=$INTERVENTION_SUBSET_JSON" )
        fi
        # Offline-mode flag: skip the TeleopRecording dataset push when
        # --push_to_hub is NOT set on the orchestrator (the default). When
        # the user opts in with --push_to_hub, omit this flag so the
        # wrapper's default (push enabled) wins.
        OFFLINE_DATASET_ARG=()
        if [[ "$PUSH_TO_HUB" == false ]]; then
            OFFLINE_DATASET_ARG+=( "--env.teleop_push_to_hub=false" )
        fi
        # Intervention recording is driven by `lerobot-eval` with the
        # `--intervention.*` config field (see AGENTS.md / CLAUDE.md).
        #
        # Three hard requirements that lerobot-eval validates at startup
        # when --intervention.method is set:
        #   1. --policy.shared_autonomy_config.enabled=true
        #      → both 'rrt' and 'oracle_goal' guidance sources live on the SA
        #         wrapper; without it there's no intervention path.
        #   2. --env.include_oracle_info=true
        #      → RRT/oracle_goal need target_ee_pos / q_goal_bias from
        #         oracle_env_config. Without it the planner refuses to plan.
        #   3. --seed=0
        #      → EvalPipelineConfig.seed defaults to 1000 (configs/eval.py:39).
        #         splatsim's seed-pinned reset uses this to pick scenario 0,
        #         1, 2, ... in order. With the default seed, the recorder
        #         starts at a random scenario in the benchmark and counts
        #         forward from there — which breaks DAgger's "keep training
        #         on the same hard scenarios round over round" semantics.
        # Other SA settings (forward_flow_ratio, blend_strategy, etc.) can
        # be added via --intervention_extra_args.
        # shellcheck disable=SC2086  # INTERVENTION_EXTRA_ARGS may contain multiple flags
        run_or_echo lerobot-eval \
            --policy.path="$CURRENT_POLICY" \
            --policy.shared_autonomy_config.enabled=true \
            --env.type=splatsim \
            --env.task=upright_small_engine_new \
            --env.fps=30 \
            --env.external_port="$ENV_EXTERNAL_PORT" \
            --env.external_host="$ENV_EXTERNAL_HOST" \
            --env.include_oracle_info=true \
            --env.episode_length="$INTERVENTION_MAX_EPISODE_LENGTH" \
            --env.eval_benchmark_repo_id="$EVAL_BENCHMARK_REPO_ID" \
            --env.teleop_dataset_repo_id="$INT_REPO" \
            --eval.n_episodes="$INTERVENTION_N_EPISODES" \
            --eval.batch_size=1 \
            --eval.use_async_envs=false \
            --seed=0 \
            --output_dir="$TRAIN_OUTPUT_DIR/dagger/interventions" \
            --intervention.method="$INTERVENTION_METHOD" \
            --intervention.oracle_goal_chunk_steps="$INTERVENTION_ORACLE_GOAL_CHUNK_STEPS" \
            "${SUBSET_ARG[@]}" \
            "${OFFLINE_DATASET_ARG[@]}" \
            $INTERVENTION_EXTRA_ARGS
    fi

    # Step 2: Stats on intervention dataset (relative-action sidecar).
    # Only meaningful for use_relative_actions=True policies; for absolute
    # policies, lerobot-train uses the dataset's own meta/stats.json directly
    # and this sidecar would be incorrect to apply via --dataset.stats_path.
    if (( STEP <= 2 )); then
        if [[ "$ACTION_FORMAT" == "rel" ]]; then
            echo "--- Round $r, Step 2: compute sidecar stats for $INT_SHORT ---"
            run_or_echo bash "$SCRIPT_DIR/compute_relative_stats.sh" --dataset_repo="$INT_REPO"
        else
            echo "--- Round $r, Step 2: SKIPPED (--action_format=abs; rel-stats sidecar not used) ---"
        fi
    fi

    # Step 3: Hardlink-alias via augment_ratios_sweep.sh ratio=0 branch.
    # Skipped when --skip_alias_step is set; merge consumes raw intervention
    # datasets directly instead.
    if (( STEP <= 3 )) && [[ "$SKIP_ALIAS_STEP" == false ]]; then
        echo "--- Round $r, Step 3: hardlink-alias intervention dataset under $ALIAS_REPO ---"
        run_or_echo bash "$SCRIPT_DIR/augment_ratios_sweep.sh" \
            --dataset_short="$INT_SHORT" \
            --ratios="0.0" \
            --model="$MODEL" \
            --action_format="$ACTION_FORMAT"
    elif (( STEP <= 3 )); then
        echo "--- Round $r, Step 3: SKIPPED (--skip_alias_step) ---"
    fi

    # Step 4: Merge cumulatively (base + dag1 + ... + dag{r}, using either
    # the _pirel00 aliases or the raw dag{N} datasets depending on whether
    # the alias step is enabled).
    if (( STEP <= 4 )); then
        echo "--- Round $r, Step 4: cumulative merge → $MERGED_REPO ---"
        # The merge tool refuses to overwrite an existing target dir
        # (LeRobotDatasetMetadata.create uses mkdir(exist_ok=False)). If a
        # previous run created this round's merged dataset but then crashed
        # before stats were computed (step 5), resume detection brings us
        # back here — and the bare merge would fail with FileExistsError.
        # Nuke any pre-existing target so we start clean. Safe because the
        # only way we reach step 4 (instead of 5+) is if stats DON'T exist
        # for this merged dataset, which means it's stale/partial anyway.
        STALE_MERGED_DIR="$LEROBOT_CACHE/$MERGED_REPO"
        if [[ -e "$STALE_MERGED_DIR" ]]; then
            echo "  Removing stale merged dataset dir before re-merge: $STALE_MERGED_DIR"
            run_or_echo rm -rf "$STALE_MERGED_DIR"
        fi
        # Build the list of intervention sources, each repeated
        # INTERVENTION_OVERSAMPLE times. aggregate_datasets just iterates the
        # repo_ids list without dedup, so duplicate entries concatenate the
        # same data multiple times into the merged dataset. Effect: the
        # intervention frames make up oversample× their natural share of the
        # merged dataset, so a random batch is N× more likely to draw them.
        # Useful when the intervention is a small fraction of total data and
        # the policy under-weights it during random sampling.
        EXTRA_SOURCES=()
        for prev in $(seq 1 "$r"); do
            if [[ "$SKIP_ALIAS_STEP" == true ]]; then
                SRC="$(int_repo_for_round "$prev")"
            else
                SRC="$(alias_repo_for_round "$prev")"
            fi
            for ((dup=0; dup < INTERVENTION_OVERSAMPLE; dup++)); do
                EXTRA_SOURCES+=( "$SRC" )
            done
        done
        if (( INTERVENTION_OVERSAMPLE > 1 )); then
            echo "  intervention oversample: ${INTERVENTION_OVERSAMPLE}× → each dag{1..$r} included ${INTERVENTION_OVERSAMPLE} times in merge"
        fi
        run_or_echo python "$SCRIPT_DIR/merge_augmented_datasets_for_training.py" \
            --base "$BASE_REPO" \
            --extra_sources "${EXTRA_SOURCES[@]}" \
            --output_repo_id="$MERGED_REPO"
    fi

    # Step 5: Stats on merged dataset (relative-action sidecar; only used when
    # ACTION_FORMAT=rel — see Step 2 comment).
    if (( STEP <= 5 )); then
        if [[ "$ACTION_FORMAT" == "rel" ]]; then
            echo "--- Round $r, Step 5: compute sidecar stats for $MERGED_SHORT ---"
            run_or_echo bash "$SCRIPT_DIR/compute_relative_stats.sh" --dataset_repo="$MERGED_REPO"
        else
            echo "--- Round $r, Step 5: SKIPPED (--action_format=abs; rel-stats sidecar not used) ---"
        fi
    fi

    # Cleanup of round (r-1)'s merged dataset BEFORE this round's training
    # starts (used to be Step 7 after training). Both dag{r-1}_m and dag{r}_m
    # exist on disk briefly between the merge step and this cleanup; doing
    # the cleanup before training means at training time we only hold dag{r}_m.
    # Saves ~15 GB of disk during the longest-running step (training itself).
    # Safe because dag{r-1}_m was only needed for round (r-1)'s training,
    # which has already produced its checkpoint by the time we get here.
    # Originals (base + per-round dag/alias) stay on disk for future mix-and-match.
    if (( r > 1 )) && (( STEP <= 6 )); then
        echo "--- Round $r, Pre-train cleanup: remove round $((r-1))'s merged dataset ---"
        PREV_MERGED_REPO="$(merged_repo_for_round $((r-1)))"
        PREV_MERGED_SHORT="$(merged_short_for_round $((r-1)))"
        run_or_echo rm -rf "$LEROBOT_CACHE/$PREV_MERGED_REPO"
        run_or_echo rm -rf "$STATS_BASE/$PREV_MERGED_SHORT"
    fi

    # Step 6: Train.
    if (( STEP <= 6 )); then
        echo "--- Round $r, Step 6: train ($MODE) ---"
        # Stop the orchestrator-managed external SplatSim before training so
        # lerobot-train can spawn its OWN in-process sim for inline eval. Both
        # sims share GPU memory cost but the in-process one pools its CUDA
        # context with training (lower fragmentation, lower OOM risk).
        # In --no_manage_splatsim mode this is a no-op and the external sim
        # stays up; the user takes responsibility for the memory tradeoff.
        stop_sim
        print_gpu_state
        if [[ "$MODE" == "scratch" ]]; then
            # Pass --run_name so train_sweep.sh writes to ${BASE_POLICY_NAME}_dag${r}
            # instead of its default ${MODEL}_<dataset>_<action>_basewrist naming.
            # This keeps scratch-mode and finetune-mode rounds at the same path.
            # In retrain mode the same suffix used for TRAIN_OUTPUT_DIR is also
            # applied to the run_name so wandb/job_name don't collide.
            SCRATCH_RUN_NAME="${BASE_POLICY_NAME}_dag${r}"
            if [[ -n "$RETRAIN_ROUND" && "$r" == "$RETRAIN_ROUND" ]]; then
                SCRATCH_RUN_NAME="${SCRATCH_RUN_NAME}_${RETRAIN_SUFFIX}"
            fi
            # Forward the action-format choice. train_sweep.sh defaults to
            # USE_RELATIVE_ACTIONS=true and pre-flights stats_rel{N}.json — when
            # we're training an abs-action policy those sidecars don't exist
            # (steps 2/5 skipped them on purpose), so omit them with --no_relative.
            ABS_ACTION_ARG=()
            if [[ "$ACTION_FORMAT" == "abs" ]]; then
                ABS_ACTION_ARG=( --no_relative )
            fi
            run_training_step bash "$SCRIPT_DIR/train_sweep.sh" \
                --dataset_repo="$MERGED_REPO" \
                --run_name="$SCRATCH_RUN_NAME" \
                --model="$TRAIN_OUTPUT_MODEL_PREFIX" \
                "${ABS_ACTION_ARG[@]}" \
                "${TRAIN_EXT_PORT_SWEEP[@]}" \
                "${OFFLINE_POLICY_ARG[@]}"
        else
            # Finetune mode. Two important details:
            #
            # 1. lerobot-train's `steps` is the TOTAL target step count, not
            #    additional. When resuming from step 26000 and user asks for
            #    --finetune_steps=200, we must pass --steps=26200 so training
            #    runs for 200 more steps from where it left off. Without this,
            #    `cfg.steps <= current_step` and the loop exits at step 0.
            #
            # 2. The new `_ft`-suffixed training dir is created by overriding
            #    --policy.repo_id / --output_dir / --job_name on the resumed
            #    run; otherwise lerobot-train writes back into the original
            #    training dir from train_config.json.
            FT_RUN_NAME="${BASE_POLICY_NAME}_ft_dag${r}"
            if [[ -n "$RETRAIN_ROUND" && "$r" == "$RETRAIN_ROUND" ]]; then
                FT_RUN_NAME="${FT_RUN_NAME}_${RETRAIN_SUFFIX}"
            fi
            # Determine the current checkpoint step. In real runs this comes
            # from the resolved path (.../checkpoints/{step}/pretrained_model/).
            # In dry-run, paths from previous loop iterations are synthetic and
            # contain "checkpoints/last/..." rather than a numeric step, so we
            # fall back to a cached running counter that's bumped each round.
            CURRENT_STEP=""
            RESOLVED_POLICY="$(readlink -f "$CURRENT_POLICY" 2>/dev/null || echo "$CURRENT_POLICY")"
            CURRENT_STEP="$(echo "$RESOLVED_POLICY" | grep -oE 'checkpoints/[0-9]+/' | head -1 | grep -oE '[0-9]+' || true)"
            if [[ -z "$CURRENT_STEP" ]]; then
                if [[ "$DRY_RUN" == true && -n "${DRYRUN_NEXT_STEP:-}" ]]; then
                    CURRENT_STEP="$DRYRUN_NEXT_STEP"
                else
                    echo "ERROR: could not extract current step from policy path: $CURRENT_POLICY" >&2
                    echo "  (resolved as: $RESOLVED_POLICY)" >&2
                    echo "  Expected path of the form .../checkpoints/{step}/pretrained_model/" >&2
                    exit 1
                fi
            fi
            # Force base-10 to avoid octal parsing of '026000'.
            TARGET_STEPS=$((10#${CURRENT_STEP} + FINETUNE_STEPS))
            # Cache the predicted next-round step for dry-run continuation.
            DRYRUN_NEXT_STEP="$TARGET_STEPS"
            echo "Finetune: resuming from step ${CURRENT_STEP} → target step ${TARGET_STEPS} (+${FINETUNE_STEPS})"
            # Sanity check: the orchestrator's --action_format must match the
            # resumed policy's use_relative_actions. Mismatch → wrong stats
            # normalization → silent policy corruption. Fail loudly here so the
            # user catches it before training kicks off rather than discovering
            # the policy got worse after the fact.
            #
            # Gate on the config file existing rather than DRY_RUN so dry-run
            # of round 1 (real --initial_policy_path) catches mismatches just
            # like a real run would; later-round dry-runs with synthetic paths
            # silently skip.
            RESUME_CFG_PATH="$(readlink -f "$CURRENT_POLICY")/train_config.json"
            if [[ -f "$RESUME_CFG_PATH" ]]; then
                POLICY_REL="$(python3 -c "
import json,sys
c = json.load(open(sys.argv[1]))
print('true' if c.get('policy',{}).get('use_relative_actions') else 'false')
" "$RESUME_CFG_PATH")"
                EXPECTED_REL="false"
                [[ "$ACTION_FORMAT" == "rel" ]] && EXPECTED_REL="true"
                if [[ "$POLICY_REL" != "$EXPECTED_REL" ]]; then
                    echo "ERROR: --action_format=$ACTION_FORMAT but resumed policy has use_relative_actions=$POLICY_REL." >&2
                    echo "  Resumed policy:   $CURRENT_POLICY" >&2
                    echo "  Set --action_format=$([[ "$POLICY_REL" == "true" ]] && echo rel || echo abs) on the orchestrator." >&2
                    echo "  (Mismatch would re-normalize actions with the wrong sidecar and silently corrupt the policy.)" >&2
                    exit 1
                fi
            fi
            FT_BATCH_SIZE_ARG=()
            [[ -n "$FINETUNE_BATCH_SIZE" ]] && FT_BATCH_SIZE_ARG=(--batch_size="$FINETUNE_BATCH_SIZE")
            # Force a CONSTANT peak LR for the finetune. Mechanism depends on
            # the resumed policy's scheduler config:
            #
            # - cosine_decay_with_warmup (pi05/pi0): set decay_lr = peak_lr.
            #   The lambda blends toward decay_lr at end-of-decay; equating
            #   the two gives a flat peak. Default to auto-detected
            #   optimizer_lr from train_config.json; override with
            #   --finetune_decay_lr=<value>.
            # - diffuser/cosine (diffusion): the HF cosine scheduler decays
            #   to 0 by num_training_steps and is useless past the original
            #   end. Switch to scheduler.name=constant which returns a
            #   constant lambda=1, giving lr = base_lr (peak from the saved
            #   optimizer's initial_lr) every step.
            #
            # Both knobs default to "flatten at peak"; an explicit
            # --finetune_decay_lr=<value> still wins for cosine_decay_with_warmup.
            EFFECTIVE_DECAY_LR="$FINETUNE_DECAY_LR"
            FT_SCHEDULER_NAME_ARG=()
            # Gate detection on the train_config.json existing rather than on
            # DRY_RUN. This way the round-1 dry-run (which has a real
            # --initial_policy_path on disk) shows the auto-set scheduler
            # flag; later-round dry-runs with synthetic paths fall through
            # silently because the file isn't there yet.
            # Always recompute from the current CURRENT_POLICY — bash variables
            # persist across loop iterations and we'd otherwise reuse round
            # (r-1)'s config path in round r.
            RESUME_CFG_PATH="$(readlink -f "$CURRENT_POLICY")/train_config.json"
            if [[ -f "$RESUME_CFG_PATH" ]]; then
                POLICY_SUPPORTS_DECAY_LR="$(python3 -c "
import json,sys
c = json.load(open(sys.argv[1]))
print('true' if 'scheduler_decay_lr' in c.get('policy',{}) else 'false')
" "$RESUME_CFG_PATH")"
                SCHEDULER_TYPE="$(python3 -c "
import json,sys
c = json.load(open(sys.argv[1]))
print((c.get('scheduler') or {}).get('type') or '')
" "$RESUME_CFG_PATH")"
                if [[ "$POLICY_SUPPORTS_DECAY_LR" != "true" ]]; then
                    if [[ -n "$EFFECTIVE_DECAY_LR" ]]; then
                        echo "WARNING: --finetune_decay_lr=$EFFECTIVE_DECAY_LR ignored — policy at $CURRENT_POLICY has no scheduler_decay_lr field (likely diffusion). Skipping override." >&2
                    fi
                    EFFECTIVE_DECAY_LR=""
                    # Diffuser scheduler (HF cosine that decays to 0 by end-of-training):
                    # switch to constant so the finetune runs at peak LR.
                    if [[ "$SCHEDULER_TYPE" == "diffuser" ]]; then
                        FT_SCHEDULER_NAME_ARG=(--scheduler.name=constant)
                        echo "Finetune: auto-set --scheduler.name=constant (diffuser scheduler decays to 0 at end-of-training; flat LR keeps the finetune effective)."
                    fi
                elif [[ -z "$EFFECTIVE_DECAY_LR" ]]; then
                    PEAK_LR="$(python3 -c "
import json,sys
c = json.load(open(sys.argv[1]))
print(c.get('policy',{}).get('optimizer_lr') or '')
" "$RESUME_CFG_PATH")"
                    if [[ -n "$PEAK_LR" ]]; then
                        EFFECTIVE_DECAY_LR="$PEAK_LR"
                        echo "Finetune: auto-set --scheduler_decay_lr=$EFFECTIVE_DECAY_LR (peak optimizer_lr from train_config.json — override with --finetune_decay_lr=<value>)."
                    fi
                fi
            elif [[ "$DRY_RUN" != true ]]; then
                # A real run with no train_config.json under CURRENT_POLICY is
                # a real problem — the resume itself will fail downstream. Warn
                # so the user notices it here rather than after the sim spins up.
                echo "WARNING: train_config.json not found at $RESUME_CFG_PATH; skipping scheduler/decay-lr auto-detection." >&2
            fi
            FT_DECAY_LR_ARG=()
            [[ -n "$EFFECTIVE_DECAY_LR" ]] && FT_DECAY_LR_ARG=(--scheduler_decay_lr="$EFFECTIVE_DECAY_LR")
            # The relative-action stats sidecar (stats_rel{N}.json, where N is
            # the policy's chunk size) is only correct for policies trained with
            # use_relative_actions=True. For absolute-action policies, overriding
            # stats_path to a relative sidecar re-normalizes absolute action
            # values against delta stats, which DESTROYS the action distribution
            # and corrupts the policy over even a few hundred finetune steps.
            # When ACTION_FORMAT=abs, omit the override so lerobot-train uses the
            # merged dataset's own meta/stats.json (computed at merge time with
            # absolute actions).
            FT_STATS_PATH_ARG=()
            if [[ "$ACTION_FORMAT" == "rel" ]]; then
                # Read the chunk size straight off the resumed policy's
                # train_config.json (`policy.n_action_steps`) — that's the
                # authoritative source of truth for which sidecar to use, no
                # per-model hardcoding required.
                FT_CHUNK="$(n_action_steps_from_policy_path "$CURRENT_POLICY")"
                if [[ -z "$FT_CHUNK" ]]; then
                    if [[ "$DRY_RUN" == true ]]; then
                        # Dry-run for round >1 walks through synthetic policy
                        # paths whose train_config.json doesn't exist yet
                        # (the prior round wasn't actually trained). Use a
                        # placeholder so the dry-run can continue and the user
                        # sees the rest of the planned commands. The placeholder
                        # is wrapped in <> to make it obvious it's not a real
                        # path that would resolve.
                        FT_CHUNK="<unknown-in-dry-run>"
                        echo "  [dry-run] could not read policy.n_action_steps from $CURRENT_POLICY/train_config.json (round >1 dry-run); using <unknown-in-dry-run> placeholder in --dataset.stats_path."
                    else
                        echo "ERROR: could not read policy.n_action_steps from $CURRENT_POLICY/train_config.json" >&2
                        echo "  Needed to pick the right stats_rel{N}.json sidecar for finetune." >&2
                        exit 1
                    fi
                fi
                FT_STATS_PATH_ARG=( --dataset.stats_path="$MERGED_STATS_DIR/stats_rel${FT_CHUNK}.json" )
            fi
            # Force inline eval onto the benchmark dataset + the same subset
            # the intervention recording uses, regardless of what the resumed
            # train_config.json had baked in. Some checkpoints (especially
            # older ones, or diffusion runs that predate the benchmark wiring
            # in train_sweep.sh) save these as None and would otherwise have
            # their inline eval fall back to random scenarios — which breaks
            # round-over-round DAgger progress comparison.
            FT_EVAL_BENCHMARK_ARG=( --env.eval_benchmark_repo_id="$EVAL_BENCHMARK_REPO_ID" )
            if [[ -n "$INTERVENTION_SUBSET_JSON" ]]; then
                FT_EVAL_BENCHMARK_ARG+=( --env.eval_benchmark_subset="$INTERVENTION_SUBSET_JSON" )
            fi
            run_training_step bash "$SCRIPT_DIR/resume_training.sh" "$CURRENT_POLICY" \
                --dataset.repo_id="$MERGED_REPO" \
                "${FT_STATS_PATH_ARG[@]}" \
                --policy.repo_id="$FT_RUN_NAME" \
                --output_dir="$TRAIN_OUTPUT_DIR" \
                --job_name="$FT_RUN_NAME" \
                --steps="$TARGET_STEPS" \
                --eval_freq="$FINETUNE_EVAL_FREQ" \
                --save_freq="$FINETUNE_SAVE_FREQ" \
                "${TRAIN_EXT_PORT_RESUME[@]}" \
                "${FT_BATCH_SIZE_ARG[@]}" \
                "${FT_DECAY_LR_ARG[@]}" \
                "${FT_SCHEDULER_NAME_ARG[@]}" \
                "${FT_EVAL_BENCHMARK_ARG[@]}" \
                "${OFFLINE_POLICY_ARG[@]}"
        fi
        # Relaunch the external sim now that training has freed the GPU,
        # so the next round's intervention recording can use it. Gated on
        # EFFECTIVE_END_ROUND (not NUM_ROUNDS) so retrain mode — where the
        # loop ends early at RETRAIN_ROUND — doesn't waste a sim start
        # after the final training in the loop.
        if (( r < EFFECTIVE_END_ROUND )); then
            start_sim
        fi
    fi

    # Roll the policy forward for the next round.
    if [[ "$DRY_RUN" == true ]]; then
        CURRENT_POLICY="$TRAIN_OUTPUT_DIR/checkpoints/last/pretrained_model"
    else
        CURRENT_POLICY="$(resolve_latest_checkpoint "$TRAIN_OUTPUT_DIR")"
    fi

    # wandb artifact cache grows unboundedly across runs (each finetune logs
    # a fresh policy checkpoint artifact). Cap it at 5GB at the end of every
    # round so a long DAgger run doesn't slowly fill the disk. Best-effort:
    # silently skipped if wandb CLI isn't installed.
    if [[ "$DRY_RUN" != true ]] && command -v wandb >/dev/null 2>&1; then
        echo "--- Round $r, Step 7: wandb artifact cache cleanup (cap 5GB) ---"
        wandb artifact cache cleanup 5GB 2>&1 | sed 's/^/  /' || true
    fi

    # HuggingFace datasets-library parquet build cache: lerobot-edit-dataset's
    # merge operation calls datasets.load_dataset under the hood, which caches
    # each merge's parquet view under ~/.cache/huggingface/datasets/parquet/
    # in a hash-keyed directory. Every round's merge creates a fresh ~30 GB
    # entry, and these are NEVER cleaned up by the merge tooling itself.
    # Across N rounds that's ~30N GB of pure waste in the global HF cache,
    # independent of the per-round merged-dataset rm we do under
    # ~/.cache/huggingface/lerobot/. Nuke them at the end of each round.
    # Safe — this is a build cache; rebuilt on demand by the next merge.
    HF_PARQUET_CACHE="$HOME/.cache/huggingface/datasets/parquet"
    if [[ "$DRY_RUN" != true && -d "$HF_PARQUET_CACHE" ]]; then
        BEFORE_SIZE="$(du -sh "$HF_PARQUET_CACHE" 2>/dev/null | awk '{print $1}')"
        echo "--- Round $r, Step 7b: clean HF datasets parquet cache (was $BEFORE_SIZE) ---"
        rm -rf "$HF_PARQUET_CACHE"/*
    fi

    # Refresh the round-over-round progress table + plot. Best-effort: never
    # fails the orchestrator. Restricted to this run's lineage so multi-lineage
    # machines don't slow the call down scanning unrelated dirs.
    if [[ "$DRY_RUN" != true ]]; then
        echo "--- Round $r, Step 7c: refresh dagger_progress table + plot ---"
        bash "$SCRIPT_DIR/dagger_progress.sh" \
            --base_short="$PROGRESS_BASE_SHORT" \
            --action="$TRAIN_OUTPUT_ACTION_TAG" \
            --model="$TRAIN_OUTPUT_MODEL_PREFIX" 2>&1 | sed 's/^/  /' || true
    fi
done

# ── post-loop: optional final from-scratch train on round N's merged data ─────
# Motivation: when --intermediate_mode=finetune the N dag rounds give you a
# monotonic round-over-round curve (each step finetunes from the previous),
# but the final deployable policy is a finetune over a long sequence of
# weight updates. Training one extra model from scratch on the same final
# merged dataset gives a clean "deployable policy on full data" reference
# that's directly comparable to the finetune curve.
FINAL_POLICY_PATH="$CURRENT_POLICY"  # default: last finetune round
# In retrain mode skip the post-loop final-scratch phase. It would either
# re-train the existing final-scratch from scratch (clobbering it) or no-op
# (already complete) — neither matches user intent, which is "retune round N".
if [[ -n "$RETRAIN_ROUND" ]] && do_final_scratch; then
    echo
    echo "Post-loop final-scratch phase: SKIPPED (in --retrain_round mode)."
fi
if [[ -z "$RETRAIN_ROUND" ]] && do_final_scratch; then
    FINAL_SCRATCH_DIR="$(train_output_dir_final_scratch)"
    FINAL_SCRATCH_RUN_NAME="$(basename "$FINAL_SCRATCH_DIR")"
    LAST_MERGED_REPO="$(merged_repo_for_round "$NUM_ROUNDS")"

    if training_exists "$FINAL_SCRATCH_DIR"; then
        echo
        echo "════════════════════════════════════════════════════════════════"
        echo "POST-LOOP FINAL SCRATCH: already complete at $FINAL_SCRATCH_DIR"
        echo "════════════════════════════════════════════════════════════════"
        if [[ "$DRY_RUN" == true ]]; then
            FINAL_POLICY_PATH="$FINAL_SCRATCH_DIR/checkpoints/last/pretrained_model"
        else
            FINAL_POLICY_PATH="$(resolve_latest_checkpoint "$FINAL_SCRATCH_DIR")"
        fi
    else
        echo
        echo "════════════════════════════════════════════════════════════════"
        echo "POST-LOOP FINAL SCRATCH: train from scratch on round $NUM_ROUNDS's merged data"
        echo "  Dataset:         $LAST_MERGED_REPO"
        echo "  Training output: $FINAL_SCRATCH_DIR"
        echo "════════════════════════════════════════════════════════════════"

        # train_sweep.sh manages its own sim if --no_manage_splatsim is set;
        # otherwise stop the orchestrator-managed sim so lerobot-train can
        # spawn its own in-process eval sim.
        stop_sim
        print_gpu_state

        # Sanity: the merged dataset must exist. The stale-merged sweep above
        # keeps round NUM_ROUNDS's merged when EFFECTIVE_START_ROUND > NUM_ROUNDS,
        # but if the user manually deleted it the final-scratch step would silently
        # operate on a missing dataset — bail with a clear message instead.
        if [[ "$DRY_RUN" != true && ! -f "$LEROBOT_CACHE/$LAST_MERGED_REPO/meta/info.json" ]]; then
            echo "ERROR: final-scratch step needs $LAST_MERGED_REPO but it's not on disk." >&2
            echo "  Path checked: $LEROBOT_CACHE/$LAST_MERGED_REPO" >&2
            echo "  Re-run round $NUM_ROUNDS's step 4 (merge) to regenerate it." >&2
            exit 1
        fi

        ABS_ACTION_ARG=()
        if [[ "$ACTION_FORMAT" == "abs" ]]; then
            ABS_ACTION_ARG=( --no_relative )
        fi

        TRAIN_OUTPUT_DIR="$FINAL_SCRATCH_DIR"   # consumed by run_training_step
        run_training_step bash "$SCRIPT_DIR/train_sweep.sh" \
            --dataset_repo="$LAST_MERGED_REPO" \
            --run_name="$FINAL_SCRATCH_RUN_NAME" \
            --model="$TRAIN_OUTPUT_MODEL_PREFIX" \
            "${ABS_ACTION_ARG[@]}" \
            "${TRAIN_EXT_PORT_SWEEP[@]}" \
            "${OFFLINE_POLICY_ARG[@]}"

        if [[ "$DRY_RUN" == true ]]; then
            FINAL_POLICY_PATH="$FINAL_SCRATCH_DIR/checkpoints/last/pretrained_model"
        else
            FINAL_POLICY_PATH="$(resolve_latest_checkpoint "$FINAL_SCRATCH_DIR")"
        fi

        # Post-final-scratch cleanup: round N's merged dataset is no longer
        # needed (final-scratch consumed it, no further training will). Mirrors
        # the per-round pre-train cleanup that deletes dag{r-1}_m.
        echo "--- Post-loop cleanup: remove round $NUM_ROUNDS's merged dataset ---"
        run_or_echo rm -rf "$LEROBOT_CACHE/$LAST_MERGED_REPO"
        run_or_echo rm -rf "$STATS_BASE/$(merged_short_for_round "$NUM_ROUNDS")"

        if [[ "$DRY_RUN" != true ]] && command -v wandb >/dev/null 2>&1; then
            echo "--- Post-loop: wandb artifact cache cleanup (cap 5GB) ---"
            wandb artifact cache cleanup 5GB 2>&1 | sed 's/^/  /' || true
        fi

        if [[ "$DRY_RUN" != true ]]; then
            echo "--- Post-loop: refresh dagger_progress table + plot ---"
            bash "$SCRIPT_DIR/dagger_progress.sh" \
                --base_short="$BASE_SHORT" \
                --action="$TRAIN_OUTPUT_ACTION_TAG" \
                --model="$TRAIN_OUTPUT_MODEL_PREFIX" 2>&1 | sed 's/^/  /' || true
        fi
    fi
fi

# ── final summary ─────────────────────────────────────────────────────────────
echo
echo "════════════════════════════════════════════════════════════════"
echo "DAgger orchestration complete!"
echo "════════════════════════════════════════════════════════════════"
echo "Per-round intervention datasets:"
for r in $(seq 1 "$NUM_ROUNDS"); do
    echo "  Round $r: $(int_repo_for_round "$r")"
done
echo
echo "Final merged training dataset:  $(merged_repo_for_round "$NUM_ROUNDS")"
echo "Last finetune-round policy:     $CURRENT_POLICY"
if do_final_scratch && [[ -z "$RETRAIN_ROUND" ]]; then
    echo "Final-scratch policy:           $FINAL_POLICY_PATH"
fi
echo
echo "All per-round intervention datasets remain on disk for future"
echo "mix-and-match (their hardlink aliases under _${MODEL}${ACTION_FORMAT}00 also persist)."
