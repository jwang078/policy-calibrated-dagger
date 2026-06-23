#!/bin/bash
#
# Re-evaluate a DAgger lineage's per-round checkpoints with a fresh
# `lerobot-eval` invocation — typically to override a setting (most often
# --env.episode_length) that the training-time inline eval used.
#
# Motivation: when a DAgger-trained policy is slower than the base policy
# (e.g. intervention episodes always ramp up from zero velocity), the
# training-time inline eval can hit env.episode_length too often and bias
# the success% downward via truncation rather than genuine failure. This
# script lets you re-evaluate every round at a more generous cap so the
# resulting numbers reflect policy capability, not the truncation horizon.
#
# Per-round output landing pattern:
#   <train_dir>/reevals/<tag>/eval_info.json
#   <train_dir>/reevals/<tag>/videos/splatsim_0/eval_episode_*.mp4
#   <train_dir>/reevals/<tag>/eval_log.txt
# where tag = "eplen<N>_n<K>_seed<S>" by default (includes everything that
# changes results so different sweeps don't clobber each other).
#
# Resume semantics: if a round's <reeval_dir>/eval_info.json already exists,
# the round is skipped (idempotent on re-runs). Pass --force_rerun to
# overwrite.
#
# Companion: `dagger_progress.sh --prefer_reeval` cascades to these files
# when present, falling back to the training wandb log if not.
#
# Usage:
#   bash my_scripts/dagger_reeval_lineage.sh \
#       --filter d5jvm_g0_03dag \
#       --episode_length=800 \
#       [--n_episodes=5] [--seed=0] \
#       [--env_external_port=6001] [--headless] [--no_manage_splatsim] \
#       [--model=diffusion] \
#       [--dry-run] [--force_rerun]
#
# Options:
#   --filter SUBSTR [SUBSTR ...]  Restrict to lineages whose name contains ANY
#                                 of the given substrings (OR semantics —
#                                 matches dagger_progress.sh). Required.
#   --exclude_filter SUBSTR [...] Additive negative filter on lineage names.
#                                 A lineage matching ANY entry here is dropped
#                                 even if it matched --filter. Compose to
#                                 scope precisely:
#                                   "source only, drop reruns":
#                                     --filter d30_coll2_03dag --exclude_filter _rr_
#                                   "only b010, drop b050/b090":
#                                     --filter d30_coll2_03dag_rr_b010
#   --rounds N1,N2,...            Whitelist of round numbers (e.g. --rounds=2,
#                                 --rounds=2,5,9). Empty = all rounds the
#                                 lineage has on disk. Matches against the
#                                 `_dag<N>` segment in the round dir name.
#   --skip_nc                     Omit `_ft_dag<N>_nc` round dirs (step-6b
#                                 collision-filtered sibling policies).
#                                 Use when you only want raw rounds.
#   --nc_only                     Inverse of --skip_nc — only re-eval the
#                                 `_nc` sibling policies. Mutually exclusive
#                                 with --skip_nc.
#   --no_inherit_train_env        Disable the default behavior of forwarding
#                                 every passive env knob from each round's
#                                 train_config.json (terminate_on_collision,
#                                 max_parallel_tasks, headless, use_gripper,
#                                 include_oracle_info, wrist_cam_ver, ...) to
#                                 the reeval. Structural fields (features,
#                                 state/action dim, image shape) and fields
#                                 we already pass explicitly (task, fps,
#                                 episode_length, ...) are filtered out
#                                 regardless. Without this flag, the reeval
#                                 contract matches training-time inline eval.
#   --terminate_on_collision={auto,true,false}
#                                 One-off override on top of env inheritance.
#                                 `auto` defers to train_config (default).
#                                 Pass true/false to force regardless.
#   --episode_length=N            REQUIRED. New env.episode_length cap.
#   --n_episodes=K                Number of eval episodes. DEFAULT: auto-resolve
#                                 from each round's sidecar (matches what
#                                 training-time inline eval actually used —
#                                 either --eval.n_episodes inside
#                                 --finetune_extra_args, or
#                                 --intervention_n_episodes). So a 30-scenario
#                                 lineage gets a 30-episode reeval, a 5-scenario
#                                 lineage gets 5. Pass --n_episodes=N to force a
#                                 fixed value across all lineages. Fallback when
#                                 the sidecar has neither field: 5.
#   --seed=S                      Eval RNG seed (default 0).
#   --env_external_port=P         Splatsim port (default 6001). When the script
#                                 auto-launches its own sim (default), this is
#                                 the port the sim binds and lerobot-eval
#                                 connects to.
#   --no_manage_splatsim          Skip the auto-launched sim. Use when there's
#                                 already a SplatSim process bound to
#                                 --env_external_port (e.g. you launched it
#                                 manually in another shell).
#   --headless                    When auto-launching the sim, pass --headless
#                                 to launch_nodes.py (pybullet runs in DIRECT
#                                 mode, no GUI, ~50% less GPU memory).
#   --model=PREFIX                Filter by model prefix (diffusion / pi05 / act);
#                                 default scans all three.
#   --eval_benchmark_repo_id=REPO Override the eval benchmark dataset (default:
#                                 reads from each round's sidecar; falls back
#                                 to JennyWWW/eval_splatsim_approach_lever_benchmark_1000).
#                                 When auto-launching the sim, this is also
#                                 what the sim binds to.
#   --task=TASK                   Override env.task (default reads from sidecar
#                                 or falls back to upright_small_engine_new).
#   --dry-run                     Print commands without executing.
#   --force_rerun                 Re-evaluate even if reeval eval_info.json
#                                 already exists. (Alias: --override.)
#   --override                    Alias for --force_rerun.
#   --no_continue_on_error        Abort the whole loop on the FIRST per-round
#                                 failure (legacy fail-fast behavior). Default
#                                 is ON: failed rounds are logged + listed in
#                                 the summary footer, and the loop continues
#                                 to the next round. Re-running the same
#                                 command picks up failed rounds on retry
#                                 (the resume guard only skips successfully-
#                                 written eval_info.json files).
#
# Cascade priority for dagger_progress.sh:
#   1. Latest <train_dir>/reevals/*/eval_info.json (by mtime). This is the
#      DEFAULT behavior — opt out with --no_prefer_reeval to force the
#      legacy wandb-log scrape.
#   2. Wandb output.log's "Suite overall aggregated" line (fallback).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_lerobot_eval.sh
source "$SCRIPT_DIR/lib_lerobot_eval.sh"
# shellcheck source=lib_splatsim_manage.sh
source "$SCRIPT_DIR/lib_splatsim_manage.sh"

# Defaults
FILTERS=()
# --exclude_filter SUBSTR ... — additive negative filter on lineage names.
# A lineage matching ANY of these is dropped even if it matched --filter.
# Lets you compose "all 03dag lineages EXCEPT the rerun blends" with
# `--filter d30_coll2_03dag --exclude_filter _rr_`, etc.
EXCLUDE_FILTERS=()
# --rounds N1,N2,...  — restrict to specific round numbers (e.g. --rounds=2
# or --rounds=2,5,9). Empty = all rounds the lineage has on disk.
ROUNDS_FILTER=()
# --skip_nc / --nc_only — restrict to raw rounds or `_nc` (step-6b sibling)
# rounds only. Default off = include both. Mutually exclusive.
SKIP_NC=false
NC_ONLY=false
# --terminate_on_collision={auto,true,false} — one-off override on top of
# the env-inheritance pass (see INHERIT_TRAIN_ENV below). `auto` defers to
# whatever train_config.json had. Pass true/false to force regardless.
TERMINATE_ON_COLLISION="auto"
# --no_inherit_train_env — by DEFAULT every passive env knob from each
# round's train_config.json's `env` block is forwarded as --env.<k>=<v> to
# the reeval (terminate_on_collision, max_parallel_tasks, use_gripper,
# include_oracle_info, wrist_cam_ver, debug_mode, headless, ...). Auto-
# derived structural fields (features, state/action dim, image shape) and
# fields we already pass explicitly (task, fps, episode_length, ...) are
# filtered out. CLI overrides (--task, --episode_length, --terminate_on_collision)
# always win. Pass --no_inherit_train_env to disable inheritance entirely
# (the legacy behavior — reeval gets only the lle_build_eval_cmd defaults).
INHERIT_TRAIN_ENV=true
EPISODE_LENGTH=""
N_EPISODES=""           # empty → auto-resolve per-round from sidecar (see below).
N_EPISODES_OVERRIDDEN=false  # set true when --n_episodes is passed on the CLI
N_EPISODES_FALLBACK=5   # last-resort default when neither CLI nor sidecar resolves
SEED=0
ENV_EXTERNAL_PORT=6001
MODEL_PREFIX=""  # empty → scan all known prefixes
TRAINING_ROOT="${HOME}/code/lerobot/outputs/training"
DRY_RUN=false
FORCE_RERUN=false
EVAL_BENCHMARK_REPO_ID_OVERRIDE=""
TASK_OVERRIDE=""
KNOWN_MODEL_PREFIXES=(diffusion pi05 act)
DEFAULT_EVAL_BENCHMARK_REPO_ID="JennyWWW/eval_splatsim_approach_lever_benchmark_1000"
DEFAULT_TASK="upright_small_engine_new"
# Sim management. By default we launch our own SplatSim on ENV_EXTERNAL_PORT
# (mirroring dagger_orchestrate.sh's behavior so users don't need to keep a
# separate `launch_nodes.py` shell open). Pass --no_manage_splatsim to opt out
# when there's already one running on the same port.
MANAGE_SPLATSIM=true
HEADLESS=false
# Per-round failure handling. Default: ON — a single round's failure (eval
# crash, set -e trip, sim hiccup) logs the failure and continues to the next
# round instead of tearing down the entire multi-lineage run. Pass
# --no_continue_on_error to revert to fail-fast (the legacy behavior).
CONTINUE_ON_ERROR=true

# Parse args. `--filter A B C` (variadic) supported alongside `--filter=A,B`
# for parity with dagger_progress.sh.
args=( "$@" )
i=0
while (( i < ${#args[@]} )); do
    arg="${args[$i]}"
    case "$arg" in
        --episode_length=*)            EPISODE_LENGTH="${arg#*=}" ;;
        --n_episodes=*)                N_EPISODES="${arg#*=}"; N_EPISODES_OVERRIDDEN=true ;;
        --seed=*)                      SEED="${arg#*=}" ;;
        --env_external_port=*)         ENV_EXTERNAL_PORT="${arg#*=}" ;;
        --model=*)                     MODEL_PREFIX="${arg#*=}" ;;
        --eval_benchmark_repo_id=*)    EVAL_BENCHMARK_REPO_ID_OVERRIDE="${arg#*=}" ;;
        --task=*)                      TASK_OVERRIDE="${arg#*=}" ;;
        --training_root=*)             TRAINING_ROOT="${arg#*=}" ;;
        --dry-run)                     DRY_RUN=true ;;
        --force_rerun)                 FORCE_RERUN=true ;;
        --override)                    FORCE_RERUN=true ;;  # alias
        --no_manage_splatsim)          MANAGE_SPLATSIM=false ;;
        --headless)                    HEADLESS=true ;;
        --no_continue_on_error)        CONTINUE_ON_ERROR=false ;;
        --filter=*)
            IFS=, read -ra _vals <<< "${arg#*=}"
            for _f in "${_vals[@]}"; do
                [[ -n "$_f" ]] && FILTERS+=( "$_f" )
            done
            ;;
        --filter)
            i=$((i + 1))
            while (( i < ${#args[@]} )) && [[ "${args[$i]}" != --* ]]; do
                FILTERS+=( "${args[$i]}" )
                i=$((i + 1))
            done
            i=$((i - 1))
            ;;
        --exclude_filter=*)
            IFS=, read -ra _vals <<< "${arg#*=}"
            for _f in "${_vals[@]}"; do
                [[ -n "$_f" ]] && EXCLUDE_FILTERS+=( "$_f" )
            done
            ;;
        --exclude_filter)
            i=$((i + 1))
            while (( i < ${#args[@]} )) && [[ "${args[$i]}" != --* ]]; do
                EXCLUDE_FILTERS+=( "${args[$i]}" )
                i=$((i + 1))
            done
            i=$((i - 1))
            ;;
        --rounds=*)
            IFS=, read -ra _vals <<< "${arg#*=}"
            for _r in "${_vals[@]}"; do
                [[ -n "$_r" ]] && ROUNDS_FILTER+=( "$_r" )
            done
            ;;
        --rounds)
            i=$((i + 1))
            while (( i < ${#args[@]} )) && [[ "${args[$i]}" != --* ]]; do
                ROUNDS_FILTER+=( "${args[$i]}" )
                i=$((i + 1))
            done
            i=$((i - 1))
            ;;
        --skip_nc)                     SKIP_NC=true ;;
        --nc_only)                     NC_ONLY=true ;;
        --terminate_on_collision=*)    TERMINATE_ON_COLLISION="${arg#*=}" ;;
        --terminate_on_collision)      TERMINATE_ON_COLLISION="true" ;;
        --no_terminate_on_collision)   TERMINATE_ON_COLLISION="false" ;;
        --no_inherit_train_env)        INHERIT_TRAIN_ENV=false ;;
        --inherit_train_env)           INHERIT_TRAIN_ENV=true ;;
        -h|--help)
            sed -n '1,/^set -euo pipefail/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
    i=$((i + 1))
done

if [[ -z "$EPISODE_LENGTH" ]]; then
    echo "ERROR: --episode_length=N is required (the whole point of this script)" >&2
    exit 1
fi
if (( ${#FILTERS[@]} == 0 )); then
    echo "ERROR: --filter SUBSTR is required (otherwise this would re-eval every lineage on disk)" >&2
    exit 1
fi
if [[ "$SKIP_NC" == true && "$NC_ONLY" == true ]]; then
    echo "ERROR: --skip_nc and --nc_only are mutually exclusive." >&2
    exit 1
fi
# Validate ROUNDS_FILTER entries are integers — typos like --rounds=2,foo
# should fail fast rather than silently match nothing.
for _r in "${ROUNDS_FILTER[@]}"; do
    if ! [[ "$_r" =~ ^[0-9]+$ ]]; then
        echo "ERROR: --rounds entry '$_r' is not an integer." >&2
        exit 1
    fi
done
case "$TERMINATE_ON_COLLISION" in
    auto|true|false) ;;
    *)
        echo "ERROR: --terminate_on_collision must be auto, true, or false (got '$TERMINATE_ON_COLLISION')." >&2
        exit 1
        ;;
esac

# Build the model-prefix scan list.
if [[ -n "$MODEL_PREFIX" ]]; then
    MODEL_PREFIXES=( "$MODEL_PREFIX" )
else
    MODEL_PREFIXES=( "${KNOWN_MODEL_PREFIXES[@]}" )
fi

# NOTE: REEVAL_TAG is computed PER-ROUND inside the loop below now (not
# once here at startup), because different lineages may have used different
# n_episodes during training-time eval, and the auto-resolved per-round value
# can differ. Lineages get tag dirs like `reevals/eplen800_n5_seed0/` or
# `reevals/eplen800_n30_seed0/` accordingly, and they coexist on disk.

echo "════════════════════════════════════════════════════════════════"
if [[ "$N_EPISODES_OVERRIDDEN" == "true" ]]; then
    echo "DAgger re-eval: filter='${FILTERS[*]}'  episode_length=$EPISODE_LENGTH  n_episodes=$N_EPISODES (CLI override)  seed=$SEED"
else
    echo "DAgger re-eval: filter='${FILTERS[*]}'  episode_length=$EPISODE_LENGTH  n_episodes=auto (per-round from sidecar; fallback=$N_EPISODES_FALLBACK)  seed=$SEED"
fi
if (( ${#EXCLUDE_FILTERS[@]} > 0 )); then
    echo "  exclude_filter (lineages dropped if any match): '${EXCLUDE_FILTERS[*]}'"
fi
if (( ${#ROUNDS_FILTER[@]} > 0 )); then
    echo "  rounds (round number whitelist): ${ROUNDS_FILTER[*]}"
fi
if [[ "$SKIP_NC" == true ]]; then
    echo "  variant filter: --skip_nc → raw rounds only (omit _nc step-6b siblings)"
elif [[ "$NC_ONLY" == true ]]; then
    echo "  variant filter: --nc_only → _nc rounds only (omit raw)"
fi
if [[ "$INHERIT_TRAIN_ENV" == true ]]; then
    if [[ "$TERMINATE_ON_COLLISION" == "auto" ]]; then
        echo "  env inheritance: ON (forward train_config.env passive knobs per round)"
    else
        echo "  env inheritance: ON, terminate_on_collision forced to $TERMINATE_ON_COLLISION (CLI override)"
    fi
else
    echo "  env inheritance: OFF (--no_inherit_train_env). terminate_on_collision=$TERMINATE_ON_COLLISION"
fi
echo "Output tag (per-round): reevals/eplen${EPISODE_LENGTH}_n<auto>_seed${SEED}/"
[[ "$DRY_RUN" == "true" ]] && echo "DRY-RUN: commands will be printed, not executed."
[[ "$FORCE_RERUN" == "true" ]] && echo "FORCE_RERUN: existing eval_info.json files will be overwritten."
echo "════════════════════════════════════════════════════════════════"
echo

# ── _base_lineage_dir: find the base-policy training dir for a lineage
# (the policy BEFORE any DAgger rounds — what dagger_progress.sh prints as
# `dag0`). Mirrors the resolution logic in dagger_progress.sh's print loop
# including the `_basewrist_` fallback (lineages named
# `<...>_basewrist_<tag>` were finetuned from a base policy at
# `<...>_basewrist`, NOT at the full lineage name — the lineage name only
# exists as `_ft_dag<N>` siblings). Echoes the dir path, or empty if not
# on disk.
_base_lineage_dir() {
    local prefix="$1" lineage="$2"
    local base_dir="$TRAINING_ROOT/${prefix}_${lineage}"
    if [[ ! -d "$base_dir" && "$lineage" == *_basewrist_* ]]; then
        base_dir="$TRAINING_ROOT/${prefix}_${lineage%_basewrist_*}_basewrist"
    fi
    [[ -d "$base_dir" ]] && echo "$base_dir" || echo ""
}

# ── _round_dir_matches_filters: should this round dir be processed?
# Args: $1 = round basename like `ft_dag2` / `dag5_s` / `ft_dag3_nc` /
# `dag0` (synthetic round-0 from the base lineage dir).
# Honors --rounds, --skip_nc, --nc_only. Used inside the per-lineage loop.
_round_dir_matches_filters() {
    local round_name="$1"
    # _nc / raw split.
    local is_nc=false
    [[ "$round_name" == *_nc ]] && is_nc=true
    if [[ "$SKIP_NC" == true && "$is_nc" == true ]]; then
        return 1
    fi
    if [[ "$NC_ONLY" == true && "$is_nc" == false ]]; then
        return 1
    fi
    # Round-number whitelist. Regex matches both `ft_dag<N>` / `dag<N>_s`
    # (underscore-prefixed) AND bare `dag<N>` (the synthetic base-policy
    # round-0 name we inject from _base_lineage_dir).
    if (( ${#ROUNDS_FILTER[@]} > 0 )); then
        local round_num
        round_num=$(printf '%s' "$round_name" | sed -E 's/^(.*_)?dag([0-9]+).*/\2/')
        local matched=false
        for want in "${ROUNDS_FILTER[@]}"; do
            if [[ "$round_num" == "$want" ]]; then
                matched=true; break
            fi
        done
        [[ "$matched" == true ]] || return 1
    fi
    return 0
}

# ── lineage_filter_matches: does this lineage name pass the --filter /
# --exclude_filter combo? Substring-based (positive list ANY-match, then
# negative list ANY-match drops).
_lineage_filter_matches() {
    local lineage="$1"
    # Negative list takes precedence.
    for ef in "${EXCLUDE_FILTERS[@]}"; do
        [[ "$lineage" == *"$ef"* ]] && return 1
    done
    for f in "${FILTERS[@]}"; do
        [[ "$lineage" == *"$f"* ]] && return 0
    done
    return 1
}

# ── _resolve_sidecar_field: read a field from <train_dir>/dagger/config.json
# by argv key. Defaults to empty if missing.
_resolve_sidecar_field() {
    local train_dir="$1" key="$2"
    local sidecar="$train_dir/dagger/config.json"
    [[ -f "$sidecar" ]] || { echo ""; return; }
    python3 -c "
import json, sys
sc = json.load(open('$sidecar'))
argv = (sc.get('orchestrator_invocation') or {}).get('argv') or []
for a in argv:
    if a.startswith('--$key='):
        print(a.split('=', 1)[1]); break
" 2>/dev/null
}

# ── _train_config_path: find this round's train_config.json. Prefer
# checkpoints/last/, fall back to the highest-numbered checkpoint. Echoes
# empty if none on disk.
_train_config_path() {
    local train_dir="$1"
    local cfg="$train_dir/checkpoints/last/pretrained_model/train_config.json"
    if [[ -f "$cfg" ]]; then echo "$cfg"; return; fi
    ls "$train_dir/checkpoints/"*"/pretrained_model/train_config.json" 2>/dev/null | head -1
}

# ── _resolve_env_inherit_args: read train_config.json's `env` block and
# emit a space-separated string of `--env.<k>=<v>` flags suitable for
# appending to lerobot-eval's argv. Filters out:
#   • DERIVED   — fields lerobot-eval auto-populates from policy/env at
#                 startup (features, state/action dim, image shape, type).
#                 Round-tripping them is at best redundant, at worst stale.
#   • EXPLICIT  — fields we already pass via lle_build_eval_cmd or per-CLI
#                 overrides (task, fps, episode_length, ...). Duplicate
#                 --env.X= flags would let draccus pick whichever it sees
#                 last and obscure the override.
#   • TELEOP    — reeval is always passive (no teleop dataset writes); the
#                 teleop_* knobs only fire in intervention mode.
#   • None      — would emit `--env.k=None` and trip draccus type checks.
# Args:
#   $1 = train_dir
#   $2 = tc_override ("auto" | "true" | "false") — when not "auto",
#         forcibly replaces the inherited terminate_on_collision.
# Echoes flags on stdout (empty if no train_config.json found / inheritance
# disabled / no fields survived the filter).
_resolve_env_inherit_args() {
    local train_dir="$1" tc_override="$2"
    local cfg
    cfg=$(_train_config_path "$train_dir")
    [[ -z "$cfg" || ! -f "$cfg" ]] && { echo ""; return; }
    python3 - "$cfg" "$tc_override" <<'PYEOF'
import json, sys, shlex
cfg_path, tc_override = sys.argv[1], sys.argv[2]
try:
    cfg = json.load(open(cfg_path))
    env = cfg.get('env') or {}
except Exception:
    print('')
    sys.exit(0)
DERIVED = {
    'type', 'features', 'features_map',
    'state_dim', 'action_dim',
    'observation_height', 'observation_width',
    'port', 'render_mode',
}
EXPLICIT = {
    'task', 'task_description',
    'fps', 'camera_names', 'image_resize_modes',
    'eval_benchmark_repo_id', 'eval_benchmark_subset',
    'episode_length', 'external_port', 'external_host',
    'disable_env_checker',  # set by EnvConfig defaults; not worth round-tripping
}
TELEOP = {
    'teleop_dataset_repo_id', 'teleop_min_episode_length', 'teleop_push_to_hub',
}
def fmt(v):
    if isinstance(v, bool): return 'true' if v else 'false'
    if isinstance(v, (list, dict)): return shlex.quote(json.dumps(v))
    return shlex.quote(str(v))
out = []
for k, v in env.items():
    if k in DERIVED or k in EXPLICIT or k in TELEOP: continue
    if v is None: continue
    out.append(f'--env.{k}={fmt(v)}')
if tc_override in ('true', 'false'):
    out = [a for a in out if not a.startswith('--env.terminate_on_collision=')]
    out.append(f'--env.terminate_on_collision={tc_override}')
print(' '.join(out))
PYEOF
}

# ── _resolve_sidecar_n_episodes: figure out the n_episodes value training-
# time inline eval ACTUALLY used for this round, so the re-eval samples the
# same number of episodes. Otherwise (e.g. with the script's old hardcoded
# default of 5) a 30-scenario run gets re-evaluated on 5 episodes and the
# resulting succ% has way more variance than what's in the chart — defeating
# the entire purpose of the reeval-replaces-wandb-source cascade.
#
# Priority (highest first):
#   1. --eval.n_episodes=N nested inside --finetune_extra_args=... — this is
#      what lerobot-train's inline eval actually consumed.
#   2. --intervention_n_episodes=N — the orchestrator's per-round eval scope.
#      Usually identical to (1), but (1) is more precise when they differ.
# Returns the resolved integer on stdout, or empty if neither is present.
_resolve_sidecar_n_episodes() {
    local train_dir="$1"
    local sidecar="$train_dir/dagger/config.json"
    [[ -f "$sidecar" ]] || { echo ""; return; }
    python3 -c "
import json, re, sys
sc = json.load(open('$sidecar'))
argv = (sc.get('orchestrator_invocation') or {}).get('argv') or []
# Priority 1: parse --eval.n_episodes=N out of --finetune_extra_args=...
for a in argv:
    if a.startswith('--finetune_extra_args='):
        m = re.search(r'--eval\.n_episodes=(\d+)', a)
        if m:
            print(m.group(1)); sys.exit()
# Priority 2: --intervention_n_episodes=N
for a in argv:
    if a.startswith('--intervention_n_episodes='):
        print(a.split('=', 1)[1]); sys.exit()
" 2>/dev/null
}

# ── _round_table_row: parse a reeval eval_info.json into one summary row.
# Prints the row to stdout (tab-separated) or empty if parse failed.
_round_table_row() {
    local round_label="$1" eval_info_json="$2"
    [[ -f "$eval_info_json" ]] || { echo ""; return; }
    python3 -c "
import json, sys
try:
    d = json.load(open('$eval_info_json'))
except Exception:
    sys.exit()
# Prefer the aggregated forms if present (older + post-eval-complete files
# have these); fall back to computing from per_task[].metrics.
o = d.get('overall') or {}
g = (d.get('per_group') or {}).get('splatsim') or {}
src = o or g
if src.get('pc_success') is None:
    # Aggregate from per_task. Sum across per_task[].metrics arrays.
    succ_total, succ_n = 0, 0
    pos, ori, coll, trunc = [], [], [], []
    for t in d.get('per_task') or []:
        m = t.get('metrics') or {}
        successes = m.get('successes') or []
        succ_total += sum(1 for s in successes if s)
        succ_n += len(successes)
        info = m.get('info_metrics') or {}
        pos.extend(info.get('final_position_error_m') or [])
        ori.extend(info.get('final_orientation_error_deg') or [])
        coll.extend(info.get('in_collision') or [])
        trunc.extend(info.get('truncated') or [])
    if succ_n == 0:
        sys.exit()
    succ = 100.0 * succ_total / succ_n
    avg = lambda xs: (sum(xs) / len(xs)) if xs else float('nan')
    print(f'$round_label\\t{succ:.0f}\\t{avg(pos):.3f}\\t{avg(ori):.1f}\\t{avg(coll):.1f}\\t{avg(trunc):.1f}')
else:
    print(f'$round_label\\t{src[\"pc_success\"]:.0f}\\t{src[\"avg_final_position_error_m\"]:.3f}\\t{src[\"avg_final_orientation_error_deg\"]:.1f}\\t{src.get(\"avg_in_collision\", float(\"nan\")):.1f}\\t{src.get(\"avg_truncated\", float(\"nan\")):.1f}')
"
}

# ── Launch our own SplatSim (unless --no_manage_splatsim). All re-evals in
# this run share one sim process, so we pay the startup cost (~30s) once and
# then re-eval every checkpoint in series against the same backend. Same
# lifecycle as dagger_orchestrate.sh's auto-managed sim, including the EXIT
# trap that guarantees teardown even if the script crashes or the user
# Ctrl+C's mid-run. EVAL_BENCHMARK_REPO_ID must be set before splat_start_sim
# — use the override if provided, else the default (matches what most lineages
# evaluated against during training).
if [[ "$MANAGE_SPLATSIM" == "true" ]]; then
    if [[ -n "$EVAL_BENCHMARK_REPO_ID_OVERRIDE" ]]; then
        EVAL_BENCHMARK_REPO_ID="$EVAL_BENCHMARK_REPO_ID_OVERRIDE"
    else
        EVAL_BENCHMARK_REPO_ID="$DEFAULT_EVAL_BENCHMARK_REPO_ID"
    fi
    trap 'splat_stop_sim' EXIT
    splat_start_sim
    echo
fi

# ── _resolve_round_reeval_paths: compute per-round n_episodes + reeval_dir
# + eval_info_json. Sets four script-globals (no `local` because the main
# loop and the pre-scan both consume them at top-level scope, matching the
# existing convention noted above for `_samp_seed` etc.):
#   _round_n_eps, _round_n_eps_src, _round_reeval_tag, reeval_dir, eval_info_json
# Pre-scan + main loop call this so the planned skip-list and the actual
# skip check stay in lockstep — no chance of divergence.
_resolve_round_reeval_paths() {
    local train_dir="$1"
    if [[ "$N_EPISODES_OVERRIDDEN" == "true" ]]; then
        _round_n_eps="$N_EPISODES"
        _round_n_eps_src="CLI override"
    else
        _round_n_eps="$(_resolve_sidecar_n_episodes "$train_dir")"
        if [[ -n "$_round_n_eps" ]]; then
            _round_n_eps_src="sidecar"
        else
            _round_n_eps="$N_EPISODES_FALLBACK"
            _round_n_eps_src="script fallback"
        fi
    fi
    _round_reeval_tag="eplen${EPISODE_LENGTH}_n${_round_n_eps}_seed${SEED}"
    reeval_dir="$train_dir/reevals/$_round_reeval_tag"
    eval_info_json="$reeval_dir/eval_info.json"
}

# ── Pre-scan: walk the same lineage/round/filter logic the main loop will,
# inventory which rounds would be SKIPPED (eval_info.json already on disk)
# vs RUN. Lets the user see the plan up-front and bail if they expected
# more (or fewer) runs to happen. Quiet on stdout — we just populate two
# arrays + print a summary block + (optionally) prompt y/n. Skipped on
# --override (everything runs regardless) and --dry-run (no real work
# anyway). Identical short-circuit / globbing logic to the main loop —
# any divergence would be a bug since the user's confirmation would no
# longer match what runs.
PLAN_RUN=()   # "$prefix_$lineage/$round_name → $reeval_dir"
PLAN_SKIP=()  # "$prefix_$lineage/$round_name (existing $eval_info_json)"
for prefix in "${MODEL_PREFIXES[@]}"; do
    mapfile -t _ps_lineages < <(
        ls -d "$TRAINING_ROOT/${prefix}_"*_dag[0-9]* 2>/dev/null \
            | while read -r d; do basename "$d"; done \
            | sed -E "s/^${prefix}_//; s/(_ft)?_dag[0-9]+(_[^/]*)?$//" \
            | sort -u
    )
    for _ps_lineage in "${_ps_lineages[@]}"; do
        [[ -n "$_ps_lineage" ]] || continue
        _lineage_filter_matches "$_ps_lineage" || continue
        mapfile -t _ps_round_dirs < <(
            { ls -d "$TRAINING_ROOT/${prefix}_${_ps_lineage}_dag"[0-9]*    2>/dev/null || true
              ls -d "$TRAINING_ROOT/${prefix}_${_ps_lineage}_ft_dag"[0-9]* 2>/dev/null || true
            } | sort -V
        )
        # Prepend the base-policy dir as the synthetic dag0 round so users
        # can scope to it via --rounds=0. dagger_progress.sh treats this dir
        # the same way (its "dag0" row), so the chart and the reeval set
        # agree on which checkpoints exist.
        _ps_base_dir=$(_base_lineage_dir "$prefix" "$_ps_lineage")
        if [[ -n "$_ps_base_dir" ]]; then
            _ps_round_dirs=( "$_ps_base_dir" "${_ps_round_dirs[@]}" )
        fi
        for _ps_train_dir in "${_ps_round_dirs[@]}"; do
            [[ -d "$_ps_train_dir" ]] || continue
            # Round name: bare `dag0` for the base-policy dir (it has no
            # `_dag<N>` suffix in its basename to strip), else the usual
            # `<basename> minus <prefix>_<lineage>_` prefix.
            if [[ "$_ps_train_dir" == "$_ps_base_dir" ]]; then
                _ps_round_name="dag0"
            else
                _ps_round_name=$(basename "$_ps_train_dir" | sed -E "s/^${prefix}_${_ps_lineage}_//")
            fi
            _round_dir_matches_filters "$_ps_round_name" || continue
            _resolve_round_reeval_paths "$_ps_train_dir"
            local_id="${prefix}_${_ps_lineage}/${_ps_round_name}"
            if [[ -f "$eval_info_json" ]]; then
                PLAN_SKIP+=( "$local_id (existing: $eval_info_json)" )
            else
                PLAN_RUN+=( "$local_id → $reeval_dir" )
            fi
        done
    done
done

echo "────────────────── Pre-scan ──────────────────"
echo "Planned to RUN  (${#PLAN_RUN[@]}):"
if (( ${#PLAN_RUN[@]} == 0 )); then
    echo "  (none)"
else
    for r in "${PLAN_RUN[@]}"; do echo "  • $r"; done
fi
echo
echo "Would SKIP (already-complete reeval; pass --override / --force_rerun to redo) (${#PLAN_SKIP[@]}):"
if (( ${#PLAN_SKIP[@]} == 0 )); then
    echo "  (none)"
else
    for s in "${PLAN_SKIP[@]}"; do echo "  • $s"; done
fi
echo "──────────────────────────────────────────────"

if (( ${#PLAN_RUN[@]} == 0 && ${#PLAN_SKIP[@]} == 0 )); then
    echo "Nothing matched the filter set. Exiting."
    exit 0
fi
# Interactive confirmation when there are skips AND the user hasn't already
# said "redo everything" (--override) AND we're not just dry-running. Stdin
# might not be a tty in some pipelines (CI, nohup, sweep wrappers) — fall
# back to "proceed without asking" with a notice in that case so we don't
# block indefinitely on /dev/null.
if (( ${#PLAN_SKIP[@]} > 0 )) && [[ "$FORCE_RERUN" != "true" && "$DRY_RUN" != "true" ]]; then
    if [[ -t 0 ]]; then
        printf "Proceed? Skip the %d already-complete run(s) and run the remaining %d? [Y/n] " "${#PLAN_SKIP[@]}" "${#PLAN_RUN[@]}"
        read -r _confirm
        case "${_confirm:-y}" in
            y|Y|yes|YES) ;;
            *) echo "Aborted by user."; exit 1 ;;
        esac
    else
        echo "(stdin is not a tty; proceeding without prompt — pass --override to redo skipped runs)"
    fi
fi
echo

# ── For each model prefix, discover lineages, then per-lineage per-round eval.
n_lineages_evaluated=0
SUMMARY_ROWS=()    # accumulated per-round summary rows for final table
FAILED_ROUNDS=()   # accumulated "$prefix_$lineage/$round_name" strings for rounds that errored
SKIPPED_RUNS=()    # accumulated "$prefix_$lineage/$round_name → $eval_info_json" entries
                   # for rounds short-circuited by the already-complete guard (no --override).

for prefix in "${MODEL_PREFIXES[@]}"; do
    # Lineage discovery mirrors dagger_progress.sh:122-126 (strip model
    # prefix + the _dag<N> / _ft_dag<N> trailing suffix).
    mapfile -t lineages < <(
        ls -d "$TRAINING_ROOT/${prefix}_"*_dag[0-9]* 2>/dev/null \
            | while read -r d; do basename "$d"; done \
            | sed -E "s/^${prefix}_//; s/(_ft)?_dag[0-9]+(_[^/]*)?$//" \
            | sort -u
    )
    for lineage in "${lineages[@]}"; do
        [[ -n "$lineage" ]] || continue
        _lineage_filter_matches "$lineage" || continue
        echo "── lineage: ${prefix}_${lineage}"
        n_lineages_evaluated=$((n_lineages_evaluated + 1))

        # Discover this lineage's per-round dirs. Use `|| true` on each `ls`
        # so a missing variant (e.g. lineages with only finetune rounds and
        # no scratch rounds, or vice versa) doesn't trip `set -e` inside
        # the process substitution and silently zero out round_dirs.
        mapfile -t round_dirs < <(
            { ls -d "$TRAINING_ROOT/${prefix}_${lineage}_dag"[0-9]*    2>/dev/null || true
              ls -d "$TRAINING_ROOT/${prefix}_${lineage}_ft_dag"[0-9]* 2>/dev/null || true
            } | sort -V
        )
        # Prepend the base-policy dir as synthetic dag0 — same logic as the
        # pre-scan, kept in lockstep so the planned and actual work agree.
        base_lineage_dir=$(_base_lineage_dir "$prefix" "$lineage")
        if [[ -n "$base_lineage_dir" ]]; then
            round_dirs=( "$base_lineage_dir" "${round_dirs[@]}" )
        fi
        if (( ${#round_dirs[@]} == 0 )); then
            echo "  (no _dag<N> dirs found, skipping)"
            continue
        fi

        for train_dir in "${round_dirs[@]}"; do
            [[ -d "$train_dir" ]] || continue
            if [[ "$train_dir" == "$base_lineage_dir" ]]; then
                round_name="dag0"
            else
                round_name=$(basename "$train_dir" | sed -E "s/^${prefix}_${lineage}_//")
            fi
            # Per-round filter: --rounds (round-number whitelist) +
            # --skip_nc / --nc_only (variant filter). When the round is
            # filtered out, log a quiet line so the user can see what was
            # skipped (helpful for verifying their filter actually scoped
            # to what they meant). The reeval/sim setup costs above are
            # cheap relative to a 30-episode eval, so the savings here are
            # real.
            if ! _round_dir_matches_filters "$round_name"; then
                echo "  $round_name: skipped (per --rounds/--skip_nc/--nc_only filter)"
                continue
            fi
            # Find checkpoint. Prefer checkpoints/last/pretrained_model.
            ckpt="$train_dir/checkpoints/last/pretrained_model"
            if [[ ! -d "$ckpt" ]]; then
                # Fall back to highest numbered checkpoint.
                # Trailing `|| true` is LOAD-BEARING. When the round's
                # `checkpoints/` dir doesn't exist (partial training that
                # died before writing any checkpoint), `ls` exits 2; `grep`
                # then exits 1 (no input → no match). `pipefail` surfaces
                # that non-zero exit, and `set -e` kills the script on this
                # simple `var=$(...)` assignment in bash 5.x+. The `|| true`
                # swallows the pipeline exit so we just get an empty
                # ckpt_num and fall through to the "no checkpoint found"
                # branch below.
                ckpt_num=$( { ls "$train_dir/checkpoints" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1; } || true )
                if [[ -n "$ckpt_num" && -d "$train_dir/checkpoints/$ckpt_num/pretrained_model" ]]; then
                    ckpt="$train_dir/checkpoints/$ckpt_num/pretrained_model"
                else
                    echo "  $round_name: no checkpoint found, skipping"
                    continue
                fi
            fi

            # Per-round n_episodes + reeval paths. See _resolve_round_reeval_paths
            # docstring — keeps the pre-scan and the main loop's skip check
            # in lockstep (they call the same function).
            _resolve_round_reeval_paths "$train_dir"

            if [[ -f "$eval_info_json" && "$FORCE_RERUN" != "true" ]]; then
                echo "  $round_name: SKIP (exists at $eval_info_json — pass --override / --force_rerun to redo)"
                row=$(_round_table_row "$round_name" "$eval_info_json")
                [[ -n "$row" ]] && SUMMARY_ROWS+=( "$row" )
                SKIPPED_RUNS+=( "${prefix}_${lineage}/${round_name} (existing: $eval_info_json)" )
                continue
            fi

            # Resolve eval benchmark + subset for THIS round from sidecar.
            bench_repo=""
            if [[ -n "$EVAL_BENCHMARK_REPO_ID_OVERRIDE" ]]; then
                bench_repo="$EVAL_BENCHMARK_REPO_ID_OVERRIDE"
            else
                bench_repo=$(_resolve_sidecar_field "$train_dir" "env.eval_benchmark_repo_id")
                [[ -z "$bench_repo" ]] && bench_repo=$(_resolve_sidecar_field "$train_dir" "eval_benchmark_repo_id")
                [[ -z "$bench_repo" ]] && bench_repo="$DEFAULT_EVAL_BENCHMARK_REPO_ID"
            fi
            task="$TASK_OVERRIDE"
            [[ -z "$task" ]] && task=$(_resolve_sidecar_field "$train_dir" "env.task")
            [[ -z "$task" ]] && task="$DEFAULT_TASK"
            # Subset JSON: the orchestrator computes the scenario subset at
            # startup from --intervention_sample_seed + --intervention_sample_from_first
            # + --intervention_n_episodes (see dagger_orchestrate.sh:1129-1138)
            # and passes it inline as --env.eval_benchmark_subset to every
            # per-round eval. That subset is what the chart's existing eval
            # numbers were measured on, so we re-derive it from the same
            # sidecar argv to keep the re-eval scenario set identical.
            # NOTE: `local` would error here ("can only be used in a function")
            # since we're at top-level script scope inside the main for-loop.
            # Underscored names keep the namespace tidy without `local`.
            subset_json=""
            _samp_seed=$(_resolve_sidecar_field "$train_dir" "intervention_sample_seed")
            _samp_from=$(_resolve_sidecar_field "$train_dir" "intervention_sample_from_first")
            _n_eps=$(_resolve_sidecar_field "$train_dir" "intervention_n_episodes")
            if [[ -n "$_samp_from" ]]; then
                # Same sampling code as dagger_orchestrate.sh:1130-1138 — use
                # the configured n_episodes for the SUBSET (the SCENARIOS to
                # eval on), defaulting to the per-round resolved n_episodes
                # (which matches training-time eval) if the orchestrator
                # didn't pin one explicitly.
                [[ -z "$_samp_seed" ]] && _samp_seed=0
                [[ -z "$_n_eps" ]] && _n_eps="$_round_n_eps"
                subset_json=$(python3 -c "
import random, sys
random.seed(int(sys.argv[1]))
print('[' + ','.join(str(i) for i in sorted(random.sample(range(int(sys.argv[3])), int(sys.argv[2])))) + ']')
" "$_samp_seed" "$_n_eps" "$_samp_from")
            fi

            # Inherit train_config.json's env block by default so the reeval
            # matches the training-time inline eval contract (matters most
            # for terminate_on_collision — a lineage trained with that flag
            # ON gets misleadingly inflated success% if reeval'd OFF, since
            # post-collision recoveries the original eval would have
            # terminated keep running). Also forwards passive knobs like
            # max_parallel_tasks, headless, include_oracle_info, etc.
            env_inherit_args=""
            if [[ "$INHERIT_TRAIN_ENV" == true ]]; then
                env_inherit_args=$(_resolve_env_inherit_args "$train_dir" "$TERMINATE_ON_COLLISION")
            elif [[ "$TERMINATE_ON_COLLISION" != "auto" ]]; then
                # Inheritance off but CLI explicitly forced the value — still honor it.
                env_inherit_args="--env.terminate_on_collision=$TERMINATE_ON_COLLISION"
            fi

            mkdir -p "$reeval_dir"
            log_file="$reeval_dir/eval_log.txt"

            echo "  $round_name → $reeval_dir/"
            echo "    ckpt: $ckpt"
            echo "    benchmark: $bench_repo  task=$task"
            echo "    n_episodes: $_round_n_eps  [$_round_n_eps_src]"
            if [[ -n "$env_inherit_args" ]]; then
                echo "    inherited env: $env_inherit_args"
            else
                echo "    inherited env: (none — no train_config.json or inheritance disabled)"
            fi
            [[ -n "$subset_json" ]] && echo "    subset: $subset_json"

            # extra_args composition: external_port (script-managed) +
            # inherited env flags. Both feed into a single shell-substituted
            # string downstream, so a single space-separated concatenation
            # is fine (each flag is already shell-quoted by the inheriter).
            extra_args="--env.external_port=$ENV_EXTERNAL_PORT"
            if [[ -n "$env_inherit_args" ]]; then
                extra_args="$extra_args $env_inherit_args"
            fi
            cmd=$(lle_build_eval_cmd \
                --policy_path="$ckpt" \
                --output_dir="$reeval_dir" \
                --eval_benchmark_repo_id="$bench_repo" \
                --task="$task" \
                --n_episodes="$_round_n_eps" \
                --seed="$SEED" \
                --episode_length="$EPISODE_LENGTH" \
                --eval_benchmark_subset="$subset_json" \
                --extra_args="$extra_args")

            # Write the reeval_config.json sidecar BEFORE running the eval, so
            # it's persisted even if the eval crashes partway. dagger_progress.sh
            # reads this to annotate rows with what the reeval changed vs the
            # original training-time eval; users read it to reproduce the reeval
            # exactly (`bash my_scripts/dagger_reeval_lineage.sh <argv...>`).
            REEVAL_CONFIG_JSON="$reeval_dir/reeval_config.json"
            if [[ "$DRY_RUN" != "true" ]]; then
                python3 - "$REEVAL_CONFIG_JSON" "$_round_reeval_tag" \
                    "$ckpt" "$bench_repo" "$task" "$subset_json" \
                    "$EPISODE_LENGTH" "$_round_n_eps" "$SEED" "$ENV_EXTERNAL_PORT" \
                    "$train_dir" "$round_name" "$0" "${args[@]}" <<'PYEOF'
import json, os, sys, time
out_path, tag, ckpt, bench, task, subset, eplen, neps, seed, port, train_dir, round_label, script, *invocation_argv = sys.argv[1:]
payload = {
    "reeval_tag": tag,
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "invocation": {
        "script": script,
        # Sort argv for diff-friendliness — same rationale as
        # dagger_orchestrate.sh's sidecar argv. Consumers (if any) look up by
        # --<key>= prefix, not by position.
        "argv": sorted(invocation_argv),
    },
    "resolved": {
        "policy_path": ckpt,
        "eval_benchmark_repo_id": bench,
        "task": task,
        "eval_benchmark_subset": subset or None,
        "episode_length": int(eplen),
        "n_episodes": int(neps),
        "seed": int(seed),
        "env_external_port": int(port),
    },
    "source": {
        "training_dir": train_dir,
        "round_label": round_label,
        "orchestrator_sidecar": os.path.join(train_dir, "dagger", "config.json"),
    },
    "note": (
        f"Re-evaluation overrides for the chart: env.episode_length={eplen}, "
        f"eval.n_episodes={neps}, seed={seed}. Compare against the original "
        f"training-time inline eval (read from the wandb output.log in "
        f"{train_dir}/wandb/) by passing --no_prefer_reeval to dagger_progress.sh."
    ),
}
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(payload, f, indent=2)
PYEOF
                echo "    config: $REEVAL_CONFIG_JSON"
            fi

            if [[ "$DRY_RUN" == "true" ]]; then
                echo "    [DRY-RUN] (would write $REEVAL_CONFIG_JSON)"
                echo "    [DRY-RUN] $cmd"
            else
                # Per-round eval wrapped so a single failure (eval crash, set -e
                # trip on a subtle non-zero pipeline exit, sim hiccup) doesn't
                # tear down the entire multi-lineage loop. The `(set +e; ... ;
                # exit $?)` subshell pattern temporarily disables set -e for
                # just this round, captures the final exit code, then lets the
                # outer script decide whether to continue or abort based on
                # --no_continue_on_error. The `tee >(grep ... > "$log_file")`
                # process substitution stays inside the subshell so its
                # broken-pipe / async-exit quirks don't affect the parent.
                set +e
                (
                    set +e  # ensure even nested subshells don't fast-fail
                    {
                        echo "════════════════════════════════════════"
                        echo "Reeval: ${prefix}_${lineage} / $round_name"
                        echo "Tag: $_round_reeval_tag"
                        echo "Start time: $(date)"
                        echo "Command:"
                        echo "$cmd"
                        echo "════════════════════════════════════════"
                        eval "$cmd"
                        echo "End time: $(date)"
                    } 2>&1 | tee >(grep -v "Running rollout" > "$log_file")
                )
                _eval_rc=$?
                set -e
                if (( _eval_rc != 0 )); then
                    echo "  ⚠ $round_name: eval failed (rc=$_eval_rc). See $log_file for the tail." >&2
                    FAILED_ROUNDS+=( "${prefix}_${lineage}/${round_name}" )
                    if [[ "$CONTINUE_ON_ERROR" != "true" ]]; then
                        echo "  Aborting (--no_continue_on_error). Re-run the same command to resume; failed rounds will be retried." >&2
                        exit "$_eval_rc"
                    fi
                    # On failure, no eval_info.json was written (or it's partial).
                    # Skip the summary row append so the failed round doesn't get
                    # silently treated as "successful with zero metrics".
                    continue
                fi
                row=$(_round_table_row "$round_name" "$eval_info_json")
                [[ -n "$row" ]] && SUMMARY_ROWS+=( "$row" )
            fi
        done
        echo
    done
done

if (( n_lineages_evaluated == 0 )); then
    echo "No lineages matched filter='${FILTERS[*]}'. Nothing to do."
    exit 0
fi

# ── Summary table at the end.
echo "════════════════════════════════════════════════════════════════"
echo "Re-eval summary (eplen=${EPISODE_LENGTH}, n_episodes=per-round-from-sidecar, seed=${SEED}):"
echo "════════════════════════════════════════════════════════════════"
printf "%-12s  %6s  %8s  %8s  %8s  %8s\n" "Round" "succ%" "pos_err" "ori_err" "in_coll" "trunc"
printf "%-12s  %6s  %8s  %8s  %8s  %8s\n" "-----" "----" "-------" "-------" "-------" "-----"
for row in "${SUMMARY_ROWS[@]}"; do
    IFS=$'\t' read -r round succ pos ori coll trunc <<< "$row"
    printf "%-12s  %6s  %8s  %8s  %8s  %8s\n" "$round" "$succ" "$pos" "$ori" "$coll" "$trunc"
done
echo

# Failure summary. Always print if any rounds errored — even in
# --no_continue_on_error mode (the script will have already exited by then,
# but if it gets here with failures, it ran with --continue_on_error).
if (( ${#FAILED_ROUNDS[@]} > 0 )); then
    echo "════════════════════════════════════════════════════════════════"
    echo "⚠ ${#FAILED_ROUNDS[@]} round(s) FAILED during eval — re-run the same command to retry:"
    for fr in "${FAILED_ROUNDS[@]}"; do
        echo "  - $fr"
    done
    echo "════════════════════════════════════════════════════════════════"
    echo
fi

# Skipped summary. Same idea — confirm at the end which already-complete
# runs were short-circuited so the user knows what to add --override for
# if they actually wanted to redo them.
if (( ${#SKIPPED_RUNS[@]} > 0 )); then
    echo "════════════════════════════════════════════════════════════════"
    echo "ℹ ${#SKIPPED_RUNS[@]} round(s) SKIPPED (already had eval_info.json — pass --override to redo):"
    for sr in "${SKIPPED_RUNS[@]}"; do
        echo "  - $sr"
    done
    echo "════════════════════════════════════════════════════════════════"
    echo
fi

echo "Run \`bash my_scripts/dagger_progress.sh --filter ${FILTERS[*]}\` — the chart prefers reeval results by default; rows sourced from a reeval are tagged with \`*<reeval_tag>\` in the eval_step column. Pass --no_prefer_reeval to force the legacy wandb-log scrape."
