#!/usr/bin/env bash

# Summarize DAgger progress: scans `outputs/training/${MODEL_PREFIX}_*[_ft]_dag*/`
# and prints, FOR EACH discovered DAgger lineage, a one-row-per-round table
# with final train metrics (loss, grad-norm, runtime LR) and final eval metrics
# (success rate, position error, collision count, truncation rate). Designed to
# be safe to run WHILE dagger_orchestrate.sh is running — it only reads the
# wandb run's output.log files.
#
# A "lineage" is the dir-name part between ${MODEL_PREFIX}_ and the trailing
# [_ft]_dag${N} suffix. Both finetune (`..._ft_dag{N}`) and scratch
# (`..._dag{N}`) round dirs from the same lineage are folded into one table.
#
# Usage:
#   bash my_scripts/dagger_progress.sh [OPTIONS]
#
# Options:
#   --base_short=STR    Lineage filter: restrict to a single lineage built from
#                       ${BASE_SHORT}_${ACTION_TAG}_basewrist[_${RUN_TAG}]. Omit
#                       to print every lineage found under TRAINING_ROOT.
#   --action=TAG        Action format tag (abs|delta). Default: abs. Only used
#                       when --base_short is set.
#   --run_tag=TAG       Optional run tag appended to the lineage (e.g. "d30"),
#                       matching --run_tag in dagger_orchestrate.sh. Only used
#                       when --base_short is set.
#   --model=PREFIX      Policy prefix in dir name. Default: scan all known
#                       prefixes (pi05, diffusion, act) and print one block
#                       per prefix that has at least one matching lineage on
#                       disk. Pass an explicit value to filter to one prefix.
#   --watch=SECONDS     Re-print the table every SECONDS seconds. Ctrl-C to stop.
#                       Default: print once and exit.
#   --filter SUBSTR [SUBSTR ...]
#                       Substring filter on lineage names. Each value is
#                       wrapped as `*SUBSTR*` and matched (case-sensitive)
#                       with OR semantics — a lineage shows up if ANY filter
#                       value appears in its name. Multiple values can be
#                       passed three equivalent ways:
#                           --filter grip0 g0           (space-separated)
#                           --filter=grip0,g0           (comma-separated, =)
#                           --filter grip0 --filter g0  (repeated flag)
#                       Both the per-lineage tables and the dagger_plot.py
#                       outputs are filtered to the matching lineages.
#   --no_prefer_reeval  Opt OUT of the default reeval cascade. By default,
#                       for each round the chart looks for a re-eval
#                       eval_info.json under <train_dir>/reevals/*/ (produced
#                       by dagger_reeval_lineage.sh) and uses those metrics
#                       when present, falling back to the training wandb log's
#                       "Suite overall aggregated" line when no reeval exists.
#                       The `eval_step` column is prefixed with `*<tag>` when
#                       a row was sourced from a reeval, so provenance is
#                       obvious. Pass --no_prefer_reeval to force-read from
#                       the wandb log even when reeval files exist.
#   --reeval_tag=TAG    Pin to a specific reeval tag (e.g. "eplen800_n5_seed0").
#                       Default: use the most-recently-modified reeval subdir
#                       per round. Implicitly enables the reeval cascade —
#                       a no-op if --no_prefer_reeval is also set.

set -euo pipefail

BASE_SHORT=""
ACTION_TAG="abs"
RUN_TAG=""
MODEL_PREFIX=""   # empty = scan all known prefixes; set via --model=PREFIX to filter
WATCH_SEC=""
FILTERS=()
PREFER_REEVAL=true     # default ON: reeval > wandb-log when reeval exists. Disable with --no_prefer_reeval.
REEVAL_TAG=""          # empty = use most-recent reeval subdir; set via --reeval_tag to pin
# --reeval_seeds=S1,S2,S3 — when set, aggregates per-round reevals across the
# listed seed values (matching reeval tag pattern `eplen*_n*_seed<S>`) and
# displays each metric cell as `mean±std`. Useful for multi-seed eval
# experiments where you want a single row per round with the cross-seed
# variance baked in. Mutually exclusive with --reeval_tag (which pins a
# single tag). Empty = single-reeval behavior (unchanged).
#
# Fallback: if a requested seed has no reeval on disk but matches the seed
# lerobot-train was launched with (per the round's train_config.json), the
# training-time inline eval at `<round>/eval/eval_info_step_*.json` is
# used as that seed's contribution. Lets you pass --reeval_seeds=0,1,2,3
# and pick up the "free" seed-0 sample from training without re-running.
# The row's tag column gets a `+trainN` suffix (e.g. `seeds:4of4(...)+train0`)
# so you can tell when the training-time eval contributed.
REEVAL_SEEDS=()
TRAINING_ROOT="${HOME}/code/lerobot/outputs/training"

# Known model prefixes — must match train_sweep.sh's run_job() prefix arg.
KNOWN_MODEL_PREFIXES=(pi05 diffusion act)

# Manual arg loop so `--filter SUBSTR` (space, no `=`) works alongside
# the existing `--name=value` forms. The previous for-loop didn't support
# the bare-arg variant.
args=( "$@" )
i=0
while (( i < ${#args[@]} )); do
    arg="${args[$i]}"
    case "$arg" in
        --base_short=*)   BASE_SHORT="${arg#*=}" ;;
        --action=*)       ACTION_TAG="${arg#*=}" ;;
        --run_tag=*)      RUN_TAG="${arg#*=}" ;;
        --model=*)        MODEL_PREFIX="${arg#*=}" ;;
        --watch=*)        WATCH_SEC="${arg#*=}" ;;
        --prefer_reeval)     PREFER_REEVAL=true ;;
        --no_prefer_reeval)  PREFER_REEVAL=false ;;
        --reeval_tag=*)      REEVAL_TAG="${arg#*=}"; PREFER_REEVAL=true ;;
        --reeval_seeds=*)
            IFS=',' read -ra _sparts <<< "${arg#*=}"
            for _s in "${_sparts[@]}"; do
                [[ -n "$_s" ]] && REEVAL_SEEDS+=( "$_s" )
            done
            PREFER_REEVAL=true
            ;;
        --filter=*)
            # Single-value form. Comma-separated values are split so
            # `--filter=A,B` is equivalent to `--filter A B`.
            IFS=',' read -ra _fparts <<< "${arg#*=}"
            for _f in "${_fparts[@]}"; do
                [[ -n "$_f" ]] && FILTERS+=( "$_f" )
            done
            ;;
        --filter)
            # Space-separated multi-value form: consume every following arg
            # until we hit another `--*` flag (or run out). A lineage matches
            # if ANY of the collected substrings appears in its name (OR
            # semantics), so `--filter grip0 g0` shows lineages containing
            # either `grip0` OR `g0`. Requires at least one value.
            i=$((i + 1))
            if (( i >= ${#args[@]} )) || [[ "${args[$i]}" == --* ]]; then
                echo "ERROR: --filter requires at least one value" >&2; exit 1
            fi
            while (( i < ${#args[@]} )) && [[ "${args[$i]}" != --* ]]; do
                FILTERS+=( "${args[$i]}" )
                i=$((i + 1))
            done
            # The outer `i+=1` below will over-increment by one — back off
            # so the next iteration processes the flag we stopped at (or
            # exits cleanly if we ran off the end).
            i=$((i - 1))
            ;;
        -h|--help)
            sed -n '1,/^set -euo pipefail/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
    i=$((i + 1))
done

# discover_lineages: print every unique lineage key for DAgger training dirs.
# A lineage key is the dir-name part between ${MODEL_PREFIX}_ and the trailing
# [_ft]_dag${N} suffix. Examples:
#   pi05_foo_abs_basewrist_ft_dag5  →  foo_abs_basewrist
#   pi05_foo_abs_basewrist_dag10    →  foo_abs_basewrist  (same lineage, scratch round)
discover_lineages() {
    # A lineage is identified by having at least one _dag${N} or _ft_dag${N}
    # dir under it. The base policy dir alone (no dag rounds) doesn't count —
    # it'd surface every standalone training dir under TRAINING_ROOT and clutter
    # the output. The dag0 row inside each lineage's table only appears when
    # that lineage has at least one dag round (per the check in print_table).
    #
    # The strip regex also handles --retrain_suffix runs (orchestrator writes
    # to `..._ft_dag${N}_${suffix}`): the optional `(_[^/]*)?` consumes any
    # trailing suffix so a retrain dir folds into the same lineage as its
    # canonical round.
    ls -d "$TRAINING_ROOT/${MODEL_PREFIX}_"*_dag[0-9]* 2>/dev/null \
        | while read -r d; do basename "$d"; done \
        | sed -E "s/^${MODEL_PREFIX}_//; s/(_ft)?_dag[0-9]+(_[^/]*)?$//" \
        | sort -u
}

# Detect whether a lineage was trained with --env.terminate_on_collision=true.
# When true, in_coll loses information (an episode ends on first collision,
# so avg_in_collision is roughly the rate of collision-terminated episodes —
# information already implied by the (1 - success_rate) of the row). We
# suppress the in_coll column from the table for such lineages.
# Args: $1=lineage name. Echoes "true" or "false". Default "false" when no
# train_config.json is found or the flag isn't set.
_lineage_terminate_on_collision() {
    local lineage="$1"
    local cfg
    # First available train_config.json across the lineage's training dirs.
    # All rounds in a lineage are expected to share this setting.
    cfg=$(
        { ls "$TRAINING_ROOT/${MODEL_PREFIX}_${lineage}_dag"[0-9]*/checkpoints/*/pretrained_model/train_config.json    2>/dev/null
          ls "$TRAINING_ROOT/${MODEL_PREFIX}_${lineage}_ft_dag"[0-9]*/checkpoints/*/pretrained_model/train_config.json 2>/dev/null
          ls "$TRAINING_ROOT/${MODEL_PREFIX}_${lineage}/checkpoints/"*"/pretrained_model/train_config.json"           2>/dev/null
        } | head -1
    )
    if [[ -z "$cfg" || ! -f "$cfg" ]]; then
        echo "false"; return
    fi
    python3 -c "
import json, sys
try:
    d = json.load(open('$cfg'))
    print('true' if (d.get('env') or {}).get('terminate_on_collision') else 'false')
except Exception:
    print('false')
" 2>/dev/null || echo "false"
}

# Read the seed lerobot-train was launched with (and therefore the seed the
# training-time inline eval — eval/eval_info_step_*.json — was run at). Used
# by the multi-seed aggregation path so that when `--reeval_seeds=0,1,2,3`
# is asked for and seed 0 has no reeval on disk, we can fall back to the
# training-time eval as that seed's data point (instead of leaving the seed
# slot empty). Echoes the seed integer as a string, or "" if no train_config.
# Args: $1=round dir.
_train_config_seed() {
    local round_dir="$1"
    local cfg="$round_dir/checkpoints/last/pretrained_model/train_config.json"
    if [[ ! -f "$cfg" ]]; then
        cfg=$(ls "$round_dir/checkpoints/"*"/pretrained_model/train_config.json" 2>/dev/null | head -1)
    fi
    [[ -z "$cfg" || ! -f "$cfg" ]] && { echo ""; return; }
    python3 -c "
import json
try:
    d = json.load(open('$cfg'))
    s = d.get('seed')
    print(s if s is not None else '')
except Exception:
    print('')
" 2>/dev/null
}

# Extract train/eval metrics from a single training dir's most-recent wandb
# log and print one table row. Args: $1=round_label (e.g. "dag0", "dag5"),
# $2=training_dir. Returns 0 if a row was printed (even an "(no log yet)"
# placeholder), so callers can use `&& found=1`.
print_row() {
    local round_label="$1"
    local DIR="$2"
    local log
    log=$(ls -t "$DIR"/wandb/run-*/files/output.log 2>/dev/null | head -1)
    if [[ -z "$log" || ! -f "$log" ]]; then
        printf "%-6s %s\n" "$round_label" "(no log yet)"
        return 0
    fi
    # Train line: last `step:NNk smpl:...` entry. Empty if no progress yet.
    # Wrap each extraction in `|| true` because `set -e + pipefail` will
    # otherwise blow up on the grep-with-no-match common during in-flight runs.
    local train_line=""
    train_line=$({ grep -E "step:[0-9]+K?\s+smpl" "$log" 2>/dev/null | tail -1; } || true)
    local loss="" grad="" lr=""
    loss=$({ echo "$train_line" | grep -oE "loss:[0-9.]+"  | cut -d: -f2; } || true)
    grad=$({ echo "$train_line" | grep -oE "grdn:[0-9.]+"  | cut -d: -f2; } || true)
    lr=$(  { echo "$train_line" | grep -oE "lr:[0-9.e+-]+" | cut -d: -f2; } || true)

    # ── EVAL METRICS CASCADE ──
    # Priority 1 (default; opt out via --no_prefer_reeval): use the most-recent
    # re-eval eval_info.json under <DIR>/reevals/*/ (or a specific tag if
    # pinned via --reeval_tag). This is what dagger_reeval_lineage.sh writes
    # and is preferred because it reflects the user's explicit re-evaluation
    # choices (e.g. longer --env.episode_length) over the training-time inline
    # eval that ran at whatever cap the orchestrator used.
    # Priority 2: training-time eval_info.json under <DIR>/eval/eval_info_step_*.json.
    # Written by lerobot_train.py at every eval step (added with
    # lerobot's per-step eval_info dump). Same dict shape as the standalone
    # eval and reeval files — so we get per-task `successes` +
    # `info_metrics.episode_length` lists, which lets us compute
    # `succ_ep_len`. Picks the most-recent file by mtime.
    # Priority 3 (fallback): scrape the wandb log's last "Suite overall
    # aggregated" line — the legacy chart behavior. Used when neither
    # reeval nor training-time eval_info.json exists (e.g. lineages
    # trained before that dump was added).
    local succ="" pos_err="" ori_err="" in_coll="" trunc="" ep_len="" succ_ep_len=""
    local fail_pos_err="" fail_ori_err="" fail_ep_len="" eval_step=""
    local reeval_json="" reeval_used_tag=""
    # Multi-seed aggregation branch: when --reeval_seeds=S1,S2,S3 is set, find
    # the per-seed reeval JSON for each S (latest matching tag per seed) and
    # aggregate cross-seed stats. Sets a sentinel `reeval_json="MULTI_SEED"`
    # that the parser below treats specially.
    local _multi_seed_json_list=""
    local _multi_seed_count=0
    if [[ "$PREFER_REEVAL" == "true" && "${#REEVAL_SEEDS[@]}" -gt 0 ]]; then
        # Resolve once per round: the seed lerobot-train was launched with.
        # If it matches one of the requested seeds AND that seed has no
        # reeval on disk, we'll plug in eval/eval_info_step_*.json as that
        # seed's contribution (gives "free" cross-seed sample n+1 without
        # re-running the same seed).
        local _train_seed
        _train_seed=$(_train_config_seed "$DIR")
        local _train_fallback_used=false
        for _s in "${REEVAL_SEEDS[@]}"; do
            # Match any eplen / n_episodes for this seed; tie-break by mtime
            # so the most-recent reeval per seed wins.
            local _per_seed_json
            _per_seed_json=$(ls -t "$DIR"/reevals/*_seed${_s}/eval_info.json 2>/dev/null | head -1 || true)
            if [[ -z "$_per_seed_json" && -n "$_train_seed" && "$_train_seed" == "$_s" ]]; then
                # Fall back to the training-time inline eval for this seed.
                _per_seed_json=$(ls -t "$DIR"/eval/eval_info_step_*.json 2>/dev/null | head -1 || true)
                [[ -n "$_per_seed_json" ]] && _train_fallback_used=true
            fi
            if [[ -n "$_per_seed_json" ]]; then
                _multi_seed_json_list+="${_per_seed_json}"$'\n'
                _multi_seed_count=$((_multi_seed_count + 1))
            fi
        done
        if (( _multi_seed_count > 0 )); then
            reeval_json="MULTI_SEED"
            reeval_used_tag="seeds:${_multi_seed_count}of${#REEVAL_SEEDS[@]}(${REEVAL_SEEDS[*]// /,})"
            # Mark when one of the seed slots was filled by the training-
            # time inline eval (eval/eval_info_step_*.json) instead of a
            # reeval, so the user can tell which rows had a "free" extra
            # sample without re-running.
            [[ "$_train_fallback_used" == "true" ]] && reeval_used_tag="${reeval_used_tag}+train${_train_seed}"
        fi
    elif [[ "$PREFER_REEVAL" == "true" ]]; then
        if [[ -n "$REEVAL_TAG" ]]; then
            local pinned="$DIR/reevals/$REEVAL_TAG/eval_info.json"
            [[ -f "$pinned" ]] && reeval_json="$pinned" && reeval_used_tag="$REEVAL_TAG"
        else
            # Pick the most-recently-modified reeval eval_info.json.
            reeval_json=$(ls -t "$DIR"/reevals/*/eval_info.json 2>/dev/null | head -1)
            if [[ -n "$reeval_json" ]]; then
                reeval_used_tag="$(basename "$(dirname "$reeval_json")")"
            fi
        fi
    fi
    # Tier 2: training-time eval_info.json. Tried if no reeval was picked
    # above (either no reeval on disk, or --no_prefer_reeval).
    if [[ -z "$reeval_json" ]]; then
        local train_eval_json=""
        train_eval_json=$(ls -t "$DIR"/eval/eval_info_step_*.json 2>/dev/null | head -1)
        if [[ -n "$train_eval_json" && -f "$train_eval_json" ]]; then
            reeval_json="$train_eval_json"
            # Mark provenance so the row's eval_step column shows where
            # the data came from (e.g. "*train-step76000"). Distinct
            # prefix from reeval tags so users can tell them apart.
            local _step_str
            _step_str=$(basename "$train_eval_json" .json | sed 's/^eval_info_step_//')
            reeval_used_tag="train-step${_step_str}"
        fi
    fi
    if [[ "$reeval_json" == "MULTI_SEED" ]]; then
        # Multi-seed branch: read all the per-seed JSONs and report
        # `mean±std` across seeds for each metric. Each per-seed eval is
        # independently summarized (succ%, pos_err, ori_err, in_coll, trunc,
        # ep_len, succ_eplen, fail_pos_err, fail_ori_err) using the same
        # logic as the single-seed parser, then the per-seed values get
        # aggregated cross-seed via mean ± sample-std.
        local eval_summary
        eval_summary=$(printf '%s' "$_multi_seed_json_list" | python3 -c "
import json, math, statistics, sys
paths = [ln for ln in sys.stdin.read().splitlines() if ln]
def summarize(d):
    o = d.get('overall') or {}
    g = (d.get('per_group') or {}).get('splatsim') or {}
    src = o or g
    succ_eplen_vals, fail_pos_err_vals, fail_ori_err_vals, fail_eplen_vals = [], [], [], []
    for t in d.get('per_task') or []:
        m = t.get('metrics') or {}
        successes = m.get('successes') or []
        info = m.get('info_metrics') or {}
        eplen_task = info.get('episode_length') or []
        pos_task   = info.get('final_position_error_m') or []
        ori_task   = info.get('final_orientation_error_deg') or []
        n = min(len(successes), len(eplen_task), len(pos_task), len(ori_task))
        for s, L, P, O in zip(successes[:n], eplen_task[:n], pos_task[:n], ori_task[:n]):
            if s:
                succ_eplen_vals.append(L)
            else:
                fail_pos_err_vals.append(P); fail_ori_err_vals.append(O); fail_eplen_vals.append(L)
    succ_eplen   = (sum(succ_eplen_vals)   / len(succ_eplen_vals))   if succ_eplen_vals   else float('nan')
    fail_pos_err = (sum(fail_pos_err_vals) / len(fail_pos_err_vals)) if fail_pos_err_vals else float('nan')
    fail_ori_err = (sum(fail_ori_err_vals) / len(fail_ori_err_vals)) if fail_ori_err_vals else float('nan')
    fail_eplen   = (sum(fail_eplen_vals)   / len(fail_eplen_vals))   if fail_eplen_vals   else float('nan')
    if src.get('pc_success') is None:
        succ_total, succ_n = 0, 0
        pos, ori, coll, trunc, eplen = [], [], [], [], []
        for t in d.get('per_task') or []:
            m = t.get('metrics') or {}
            successes = m.get('successes') or []
            succ_total += sum(1 for s in successes if s); succ_n += len(successes)
            info = m.get('info_metrics') or {}
            pos.extend(info.get('final_position_error_m') or [])
            ori.extend(info.get('final_orientation_error_deg') or [])
            coll.extend(info.get('in_collision') or [])
            trunc.extend(info.get('truncated') or [])
            eplen.extend(info.get('episode_length') or [])
        if succ_n == 0: return None
        avg = lambda xs: (sum(xs) / len(xs)) if xs else float('nan')
        return [100.0*succ_total/succ_n, avg(pos), avg(ori), avg(coll), avg(trunc), avg(eplen), succ_eplen, fail_pos_err, fail_ori_err, fail_eplen]
    else:
        return [src['pc_success'], src['avg_final_position_error_m'], src['avg_final_orientation_error_deg'],
                src.get('avg_in_collision', float('nan')), src.get('avg_truncated', float('nan')),
                src.get('avg_episode_length', float('nan')), succ_eplen, fail_pos_err, fail_ori_err, fail_eplen]
per_seed = []
for p in paths:
    try:
        s = summarize(json.load(open(p)))
        if s is not None: per_seed.append(s)
    except Exception:
        pass
if not per_seed: sys.exit()
# Column-wise mean ± sample stdev (NaN-safe — drop NaNs per column). n is
# the per-column sample count post-NaN-drop so the formatter can suppress
# the std portion when n<=1 (a single-sample stdev is undefined; printing
# 'plus-or-minus zero' is misleading because it implies low variance when
# it's really just unmeasured). NB: no backticks in this comment — the
# whole python -c "..." block is in DOUBLE quotes, so bash would interpret
# backticks as command substitution and crash with 'command not found'.
def mstd(vs):
    vs = [v for v in vs if v == v]  # filter NaN
    if not vs: return (float('nan'), 0.0, 0)
    if len(vs) == 1: return (vs[0], 0.0, 1)
    return (sum(vs)/len(vs), statistics.stdev(vs), len(vs))
cols = list(zip(*per_seed))
agg = [mstd(list(c)) for c in cols]
# Format each cell. With n>=2 seeds we get 'mean±std'; with n<=1 we drop
# the std and pad to the same width so columns still line up.
mean_fmts = ['{:.0f}', '{:.3f}', '{:.1f}', '{:.1f}', '{:.1f}',
             '{:.0f}', '{:.0f}', '{:.3f}', '{:.1f}', '{:.0f}']
std_fmts  = ['±{:.0f}', '±{:.3f}', '±{:.1f}', '±{:.1f}', '±{:.1f}',
             '±{:.0f}', '±{:.0f}', '±{:.3f}', '±{:.1f}', '±{:.0f}']
parts = []
for mfmt, sfmt, (m, s, n) in zip(mean_fmts, std_fmts, agg):
    if n <= 1:
        parts.append(mfmt.format(m))
    else:
        parts.append(mfmt.format(m) + sfmt.format(s))
print(' '.join(parts))
" 2>/dev/null || echo "")
        if [[ -n "$eval_summary" ]]; then
            read -r succ pos_err ori_err in_coll trunc ep_len succ_ep_len fail_pos_err fail_ori_err fail_ep_len <<< "$eval_summary"
            eval_step="*${reeval_used_tag}"
        fi
    elif [[ -n "$reeval_json" && -f "$reeval_json" ]]; then
        # Parse the eval_info.json. Prefer the aggregated `overall` /
        # `per_group.splatsim` blocks; fall back to averaging
        # `per_task[].metrics.{successes, info_metrics.{truncated, ...}}`
        # for partial / older files where the aggregates weren't written.
        local eval_summary
        eval_summary=$(python3 -c "
import json, sys
try:
    d = json.load(open('$reeval_json'))
except Exception:
    sys.exit()
o = d.get('overall') or {}
g = (d.get('per_group') or {}).get('splatsim') or {}
src = o or g
# Walk per_task for success/failure-conditioned metrics — the overall
# aggregates don't preserve these. For each (success, metric_value) pair,
# bucket by success status and average each bucket separately.
#   succ_eplen     = mean episode_length when success
#   fail_pos_err   = mean final_position_error_m when NOT success
#   fail_ori_err   = mean final_orientation_error_deg when NOT success
#   fail_eplen     = mean episode_length when NOT success (= how long the
#                    policy thrashed before giving up; large value with
#                    high fail_pos_err = policy fought the scenario for
#                    most of the cap and still didn't reach the goal)
succ_eplen_vals = []
fail_pos_err_vals = []
fail_ori_err_vals = []
fail_eplen_vals = []
for t in d.get('per_task') or []:
    m = t.get('metrics') or {}
    successes = m.get('successes') or []
    info = m.get('info_metrics') or {}
    eplen_task = info.get('episode_length') or []
    pos_task   = info.get('final_position_error_m') or []
    ori_task   = info.get('final_orientation_error_deg') or []
    n = min(len(successes), len(eplen_task), len(pos_task), len(ori_task))
    for s, L, P, O in zip(successes[:n], eplen_task[:n], pos_task[:n], ori_task[:n]):
        if s:
            succ_eplen_vals.append(L)
        else:
            fail_pos_err_vals.append(P)
            fail_ori_err_vals.append(O)
            fail_eplen_vals.append(L)
succ_eplen   = (sum(succ_eplen_vals)   / len(succ_eplen_vals))   if succ_eplen_vals   else float('nan')
fail_pos_err = (sum(fail_pos_err_vals) / len(fail_pos_err_vals)) if fail_pos_err_vals else float('nan')
fail_ori_err = (sum(fail_ori_err_vals) / len(fail_ori_err_vals)) if fail_ori_err_vals else float('nan')
fail_eplen   = (sum(fail_eplen_vals)   / len(fail_eplen_vals))   if fail_eplen_vals   else float('nan')

if src.get('pc_success') is None:
    succ_total, succ_n = 0, 0
    pos, ori, coll, trunc, eplen = [], [], [], [], []
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
        eplen.extend(info.get('episode_length') or [])
    if succ_n == 0:
        sys.exit()
    succ = 100.0 * succ_total / succ_n
    avg = lambda xs: (sum(xs) / len(xs)) if xs else float('nan')
    print(f'{succ:.0f} {avg(pos):.3f} {avg(ori):.1f} {avg(coll):.1f} {avg(trunc):.1f} {avg(eplen):.0f} {succ_eplen:.0f} {fail_pos_err:.3f} {fail_ori_err:.1f} {fail_eplen:.0f}')
else:
    print(f'{src[\"pc_success\"]:.0f} {src[\"avg_final_position_error_m\"]:.3f} {src[\"avg_final_orientation_error_deg\"]:.1f} {src.get(\"avg_in_collision\", float(\"nan\")):.1f} {src.get(\"avg_truncated\", float(\"nan\")):.1f} {src.get(\"avg_episode_length\", float(\"nan\")):.0f} {succ_eplen:.0f} {fail_pos_err:.3f} {fail_ori_err:.1f} {fail_eplen:.0f}')
" 2>/dev/null || echo "")
        if [[ -n "$eval_summary" ]]; then
            read -r succ pos_err ori_err in_coll trunc ep_len succ_ep_len fail_pos_err fail_ori_err fail_ep_len <<< "$eval_summary"
            # Mark the row's provenance in the `eval_step` column: prefix with
            # an `*` so callers can ctrl-F for re-evaluated rounds.
            eval_step="*${reeval_used_tag}"
        fi
    fi

    # Fallback: scrape the training wandb log's last "Suite overall aggregated"
    # entry. Used when no reeval is present OR --no_prefer_reeval was passed.
    if [[ -z "$succ" ]]; then
        local eval_line=""
        eval_line=$({ grep "Suite overall aggregated" "$log" 2>/dev/null | tail -1; } || true)
        eval_step=$({ grep -oE "Eval policy at step [0-9]+" "$log" 2>/dev/null | tail -1 | grep -oE '[0-9]+$'; } || true)
        local eval_summary=""
        if [[ -n "$eval_line" ]]; then
            eval_summary=$(echo "$eval_line" | python3 -c "
import sys, ast, re
line = sys.stdin.read()
m = re.search(r'(\{.*\})', line)
if m:
    d = ast.literal_eval(m.group(1))
    succ = d.get('pc_success', float('nan'))
    pos  = d.get('avg_final_position_error_m', float('nan'))
    ori  = d.get('avg_final_orientation_error_deg', float('nan'))
    coll = d.get('avg_in_collision', float('nan'))
    trunc= d.get('avg_truncated', float('nan'))
    eplen= d.get('avg_episode_length', float('nan'))
    print(f'{succ:.0f} {pos:.3f} {ori:.1f} {coll:.1f} {trunc:.1f} {eplen:.0f}')
" 2>/dev/null || echo "")
        fi
        if [[ -n "$eval_summary" ]]; then
            read -r succ pos_err ori_err in_coll trunc ep_len <<< "$eval_summary"
        else
            succ="-"; pos_err="-"; ori_err="-"; in_coll="-"; trunc="-"; ep_len="-"
        fi
        # wandb-log aggregate has unconditional means only — no success/fail-
        # conditional versions are logged, so leave them dashed in the
        # fallback path. The user can opt into reeval (default) or train
        # a fresh round (now writes eval_info_step_*.json) to surface them.
        succ_ep_len="-"
        fail_pos_err="-"
        fail_ori_err="-"
        fail_ep_len="-"
    fi
    [[ -z "$loss" ]]  && loss="-"
    [[ -z "$grad" ]]  && grad="-"
    [[ -z "$lr"   ]]  && lr="-"
    [[ -z "$ep_len" ]] && ep_len="-"
    [[ -z "$succ_ep_len" ]] && succ_ep_len="-"
    [[ -z "$fail_pos_err" ]] && fail_pos_err="-"
    [[ -z "$fail_ori_err" ]] && fail_ori_err="-"
    [[ -z "$fail_ep_len" ]] && fail_ep_len="-"
    [[ -z "$eval_step" ]] && eval_step="-"
    # ``hide_in_coll`` is set as a `local` in the calling print_table — bash's
    # dynamic scoping makes it visible here without an explicit arg pass.
    # In multi-seed mode the eval cells are formatted as `mean±std` (wider
    # than single values), so widen every eval column to fit cleanly.
    local _multi_seed=false
    [[ "${#REEVAL_SEEDS[@]}" -gt 0 ]] && _multi_seed=true
    if [[ "$_multi_seed" == "true" ]]; then
        if [[ "${hide_in_coll:-false}" == "true" ]]; then
            printf "%-7s %-9s %-9s %-9s %-9s %-11s %-11s %-9s %-9s %-13s %-13s %-13s %-11s %s\n" \
                "$round_label" "$loss" "$grad" "$lr" "$succ" "$pos_err" "$ori_err" "$trunc" "$ep_len" "$succ_ep_len" "$fail_pos_err" "$fail_ori_err" "$fail_ep_len" "$eval_step"
        else
            printf "%-7s %-9s %-9s %-9s %-9s %-11s %-11s %-9s %-9s %-9s %-13s %-13s %-13s %-11s %s\n" \
                "$round_label" "$loss" "$grad" "$lr" "$succ" "$pos_err" "$ori_err" "$in_coll" "$trunc" "$ep_len" "$succ_ep_len" "$fail_pos_err" "$fail_ori_err" "$fail_ep_len" "$eval_step"
        fi
    elif [[ "${hide_in_coll:-false}" == "true" ]]; then
        printf "%-6s %-9s %-9s %-9s %-6s %-9s %-9s %-6s %-7s %-11s %-13s %-13s %-11s %s\n" \
            "$round_label" "$loss" "$grad" "$lr" "$succ" "$pos_err" "$ori_err" "$trunc" "$ep_len" "$succ_ep_len" "$fail_pos_err" "$fail_ori_err" "$fail_ep_len" "$eval_step"
    else
        printf "%-6s %-9s %-9s %-9s %-6s %-9s %-9s %-8s %-6s %-7s %-11s %-13s %-13s %-11s %s\n" \
            "$round_label" "$loss" "$grad" "$lr" "$succ" "$pos_err" "$ori_err" "$in_coll" "$trunc" "$ep_len" "$succ_ep_len" "$fail_pos_err" "$fail_ori_err" "$fail_ep_len" "$eval_step"
    fi
}

_print_table_header() {
    # Args: $1 = hide_in_coll ("true" / "false"). Renders the column header
    # and the separator rule. Factored so the raw + nc tables for a single
    # lineage can share the exact same column layout without drift.
    local hide_in_coll="$1"
    local _multi_seed=false
    [[ "${#REEVAL_SEEDS[@]}" -gt 0 ]] && _multi_seed=true
    if [[ "$_multi_seed" == "true" ]]; then
        if [[ "$hide_in_coll" == "true" ]]; then
            printf "%-7s %-9s %-9s %-9s %-9s %-11s %-11s %-9s %-9s %-13s %-13s %-13s %-11s %s\n" \
                "Round" "loss" "grad" "lr" "succ%" "pos_err" "ori_err" "trunc" "ep_len" "succ_eplen" "fail_pos_err" "fail_ori_err" "fail_eplen" "eval_step"
            printf "%-7s %-9s %-9s %-9s %-9s %-11s %-11s %-9s %-9s %-13s %-13s %-13s %-11s %s\n" \
                "-----" "----" "----" "--" "-----" "-------" "-------" "-----" "------" "----------" "------------" "------------" "----------" "---------"
        else
            printf "%-7s %-9s %-9s %-9s %-9s %-11s %-11s %-9s %-9s %-9s %-13s %-13s %-13s %-11s %s\n" \
                "Round" "loss" "grad" "lr" "succ%" "pos_err" "ori_err" "in_coll" "trunc" "ep_len" "succ_eplen" "fail_pos_err" "fail_ori_err" "fail_eplen" "eval_step"
            printf "%-7s %-9s %-9s %-9s %-9s %-11s %-11s %-9s %-9s %-9s %-13s %-13s %-13s %-11s %s\n" \
                "-----" "----" "----" "--" "-----" "-------" "-------" "-------" "-----" "------" "----------" "------------" "------------" "----------" "---------"
        fi
    elif [[ "$hide_in_coll" == "true" ]]; then
        printf "%-6s %-9s %-9s %-9s %-6s %-9s %-9s %-6s %-7s %-11s %-13s %-13s %-11s %s\n" \
            "Round" "loss" "grad" "lr" "succ%" "pos_err" "ori_err" "trunc" "ep_len" "succ_eplen" "fail_pos_err" "fail_ori_err" "fail_eplen" "eval_step"
        printf "%-6s %-9s %-9s %-9s %-6s %-9s %-9s %-6s %-7s %-11s %-13s %-13s %-11s %s\n" \
            "-----" "----" "----" "--" "-----" "-------" "-------" "-----" "------" "----------" "------------" "------------" "----------" "---------"
    else
        printf "%-6s %-9s %-9s %-9s %-6s %-9s %-9s %-8s %-6s %-7s %-11s %-13s %-13s %-11s %s\n" \
            "Round" "loss" "grad" "lr" "succ%" "pos_err" "ori_err" "in_coll" "trunc" "ep_len" "succ_eplen" "fail_pos_err" "fail_ori_err" "fail_eplen" "eval_step"
        printf "%-6s %-9s %-9s %-9s %-6s %-9s %-9s %-8s %-6s %-7s %-11s %-13s %-13s %-11s %s\n" \
            "-----" "----" "----" "--" "-----" "-------" "-------" "-------" "-----" "------" "----------" "------------" "------------" "----------" "---------"
    fi
}

print_table() {
    local lineage="$1"
    echo "DAgger progress for: ${MODEL_PREFIX}_${lineage}{,[_ft]_dag*}"
    # Surface the plot path (whether it currently exists or not, so users can
    # ctrl-click as soon as `python my_scripts/dagger_plot.py` is run).
    local plot_path="${HOME}/code/lerobot/outputs/dagger/dagger_progress_${MODEL_PREFIX}_${lineage}.png"
    if [[ -f "$plot_path" ]]; then
        echo "Plot: $plot_path"
    else
        echo "Plot: $plot_path  (not generated yet — run: python my_scripts/dagger_plot.py)"
    fi
    # Read once per lineage; print_row picks it up via dynamic scope.
    local hide_in_coll
    hide_in_coll=$(_lineage_terminate_on_collision "$lineage")
    _print_table_header "$hide_in_coll"

    local found=0

    # Gather dag rounds by globbing both finetune and scratch round dirs for
    # this lineage. The trailing `_dag[0-9]*` glob matches `_dag5`, `_dag10`,
    # etc.
    local dirs
    dirs=$( { ls -d "$TRAINING_ROOT/${MODEL_PREFIX}_${lineage}_dag"[0-9]*    2>/dev/null; \
              ls -d "$TRAINING_ROOT/${MODEL_PREFIX}_${lineage}_ft_dag"[0-9]* 2>/dev/null; \
            } | awk -F'_dag' '{print $NF"\t"$0}' | sort -n | cut -f2- )

    # Round 0: the base policy training dir for this lineage. Same naming
    # scheme as the dag rounds but without any _dag${N} suffix. Only show it
    # when at least one dag round exists, since otherwise every standalone
    # base training dir would show up as "dag0" in its own one-row table.
    #
    # Lineage may include a run tag (e.g. `..._basewrist_d30`) that's only
    # present on the dag artifacts — the actual base policy dir is untagged
    # (`..._basewrist`). Try the tagged path first, then strip everything
    # after `_basewrist` and retry.
    local base_dir="$TRAINING_ROOT/${MODEL_PREFIX}_${lineage}"
    if [[ -n "$dirs" && ! -d "$base_dir" && "$lineage" == *_basewrist_* ]]; then
        base_dir="$TRAINING_ROOT/${MODEL_PREFIX}_${lineage%_basewrist_*}_basewrist"
    fi
    if [[ -n "$dirs" && -d "$base_dir" ]]; then
        print_row "dag0" "$base_dir" && found=1
    fi

    # Bucket round dirs into "raw" (no `_nc` suffix) and "nc" (step-6b
    # collision-filtered sibling policies). The two get their own tables so
    # they don't visually intermix — they're trained on different data and
    # share only their starting checkpoint per round, so reading them as one
    # series is misleading.
    local -a _raw_dirs=() _nc_dirs=()
    for DIR in $dirs; do
        if [[ "$(basename "$DIR")" == *_nc ]]; then
            _nc_dirs+=( "$DIR" )
        else
            _raw_dirs+=( "$DIR" )
        fi
    done

    for DIR in "${_raw_dirs[@]}"; do
        local round_label round_num variant retrain_suffix
        # Distinguish three variants:
        #   _ft_dag10               → "dag10"             (canonical finetune)
        #   _dag10                  → "dag10_s"           (post-loop scratch)
        #   _ft_dag1_${suffix}      → "dag1_${suffix}"    (--retrain_round=1)
        #   _dag10_${suffix}        → "dag10_s_${suffix}" (scratch retrain, rare)
        local _basename
        _basename="$(basename "$DIR")"
        if [[ "$_basename" == *_ft_dag[0-9]* ]]; then
            variant=ft
        else
            variant=scratch
        fi
        round_num=$(echo "$_basename"   | sed -E 's/.*_dag([0-9]+)(_.*)?$/\1/')
        retrain_suffix=$(echo "$_basename" | sed -E 's/.*_dag[0-9]+(_.*)?$/\1/')
        if [[ "$variant" == "scratch" ]]; then
            round_label="dag${round_num}_s${retrain_suffix}"
        else
            round_label="dag${round_num}${retrain_suffix}"
        fi
        print_row "$round_label" "$DIR" && found=1
    done

    if (( found == 0 )); then
        echo "(no training dirs found for lineage=$lineage)"
        return
    fi

    # Second table: step-6b collision-filtered sibling policies (`_nc`).
    # Each row uses the same eval schema as the raw table, so the header /
    # column layout is identical. Round 0 (base policy) is repeated at the
    # top as a fixed anchor — the nc policies all branch off it round-by-
    # round, so showing it in both tables makes "did filtering move the
    # needle off baseline" readable from one table alone.
    if (( ${#_nc_dirs[@]} > 0 )); then
        echo
        echo "Collision-filtered sibling policies (step 6b, --filter_blend_collisions):"
        _print_table_header "$hide_in_coll"
        if [[ -d "$base_dir" ]]; then
            print_row "dag0" "$base_dir"
        fi
        for DIR in "${_nc_dirs[@]}"; do
            local round_label round_num
            local _basename="$(basename "$DIR")"
            round_num=$(echo "$_basename" | sed -E 's/.*_dag([0-9]+)(_.*)?$/\1/')
            # Always _ft mode in current orchestrator (step 6b only wires the
            # finetune path); label as "dag${N}_nc" for consistency with the
            # comparison-plot legend.
            round_label="dag${round_num}_nc"
            print_row "$round_label" "$DIR"
        done
    fi

    # Reeval footer: when --prefer_reeval is on (default), surface the human-
    # readable note from any reeval_config.json under this lineage's rounds so
    # the user can see WHAT changed between the training-time inline eval and
    # the values shown in the table. Rows sourced from a reeval are already
    # tagged in the eval_step column with `*<reeval_tag>`; this footer maps
    # those tags to their resolved overrides + a reproducibility hint.
    if [[ "$PREFER_REEVAL" == "true" ]]; then
        local -a _reeval_configs
        mapfile -t _reeval_configs < <(
            { ls "$TRAINING_ROOT/${MODEL_PREFIX}_${lineage}_dag"[0-9]*/reevals/*/reeval_config.json    2>/dev/null || true
              ls "$TRAINING_ROOT/${MODEL_PREFIX}_${lineage}_ft_dag"[0-9]*/reevals/*/reeval_config.json 2>/dev/null || true
            } | sort -u
        )
        if (( ${#_reeval_configs[@]} > 0 )); then
            # Group by reeval_tag (dirname's basename) so we print one footer
            # line per unique tag. Different rounds may have different reeval
            # tags if the user re-ran with varied --episode_length, etc.
            declare -A _seen_tags=()
            for cfg in "${_reeval_configs[@]}"; do
                tag="$(basename "$(dirname "$cfg")")"
                [[ -n "${_seen_tags[$tag]:-}" ]] && continue
                _seen_tags[$tag]=1
                python3 - "$cfg" "$tag" <<'PYEOF'
import json, sys
cfg_path, tag = sys.argv[1], sys.argv[2]
try:
    cfg = json.load(open(cfg_path))
except Exception:
    sys.exit()
r = cfg.get("resolved") or {}
note = cfg.get("note") or ""
overrides = [
    f"env.episode_length={r.get('episode_length')}",
    f"eval.n_episodes={r.get('n_episodes')}",
    f"seed={r.get('seed')}",
]
print(f"  note: rows tagged *{tag} = re-evaluation ({', '.join(overrides)}). config: {cfg_path}")
PYEOF
            done
        fi
    fi

    # Show what's currently happening for the latest in-flight round of THIS
    # lineage. Best-effort; never fail the table on probe errors.
    local newest_log=""
    newest_log=$( { ls -t "$TRAINING_ROOT/${MODEL_PREFIX}_${lineage}_dag"[0-9]*/wandb/run-*/files/output.log    2>/dev/null; \
                    ls -t "$TRAINING_ROOT/${MODEL_PREFIX}_${lineage}_ft_dag"[0-9]*/wandb/run-*/files/output.log 2>/dev/null; \
                  } | xargs -r ls -t 2>/dev/null | head -1 || true)
    if [[ -n "$newest_log" && -f "$newest_log" ]]; then
        local last_step=""
        last_step=$( { grep -E "step:[0-9]+K?\s+smpl" "$newest_log" 2>/dev/null | tail -1 | grep -oE "step:[0-9]+K?" | head -1; } || true )
        if [[ -n "$last_step" ]]; then
            # Four dirnames to climb from .../wandb/run-XXX/files/output.log up to the training dir.
            # Print the absolute path so it's ctrl+clickable in VSCode/iTerm.
            local train_dir
            train_dir="$(dirname "$(dirname "$(dirname "$(dirname "$newest_log")")")")"
            echo "Latest train log:  ($last_step)  $train_dir"
        fi
    fi
    return 0
}

# Per-prefix block: tables + plot. Caller must set MODEL_PREFIX before
# invoking. Returns 0 if any lineages were found and printed; 1 if none.
print_all_for_prefix() {
    local lineages
    if [[ -n "$BASE_SHORT" ]]; then
        lineages="${BASE_SHORT}_${ACTION_TAG}_basewrist"
        [[ -n "$RUN_TAG" ]] && lineages="${lineages}_${RUN_TAG}"
    else
        lineages=$(discover_lineages)
        if [[ -z "$lineages" ]]; then
            return 1
        fi
    fi

    # Apply the --filter substrings (each treated as *FILTER*) to the
    # discovered lineage list before we walk it. OR semantics: a lineage is
    # kept if ANY filter substring appears in it. Returning 1 here propagates
    # "nothing matched in this prefix" upward so multi-prefix mode silently
    # skips.
    if (( ${#FILTERS[@]} > 0 )); then
        local filtered=""
        for lin in $lineages; do
            for _f in "${FILTERS[@]}"; do
                if [[ "$lin" == *"$_f"* ]]; then
                    filtered+="${lin}"$'\n'
                    break
                fi
            done
        done
        lineages="${filtered%$'\n'}"
        if [[ -z "$lineages" ]]; then
            return 1
        fi
    fi

    local first=1
    for lin in $lineages; do
        if (( first == 0 )); then
            echo
            echo "----------------------------------------------------------------------"
            echo
        fi
        print_table "$lin"
        first=0
    done

    # Regenerate the PNG plots via dagger_plot.py. Best-effort, never fails
    # the table. We always run dagger_plot.py in auto-discover mode (no
    # --base_short) because its --base_short filter doesn't yet understand
    # run_tag-suffixed lineages; auto-discover finds every lineage
    # including the tagged ones. Output prefixed with "[plot] " so it's
    # clearly attributed.
    #
    # When --filter is set, forward it to dagger_plot.py so the plot script
    # itself skips work for non-matching lineages (no PNG regeneration, no
    # comparison plots for filtered-out source lineages, no metric scans).
    local plot_script="$(dirname "$0")/dagger_plot.py"
    if [[ -f "$plot_script" ]]; then
        echo
        local plot_args=( --model="$MODEL_PREFIX" )
        # Forward EVERY filter substring to dagger_plot.py via its nargs='+'
        # --filter (OR-matched there too). When no filters set, omit the
        # flag so the plot script auto-discovers all lineages.
        if (( ${#FILTERS[@]} > 0 )); then
            plot_args+=( --filter "${FILTERS[@]}" )
        fi
        python3 "$plot_script" "${plot_args[@]}" 2>&1 | sed 's/^/  [plot] /' || true
    fi
    return 0
}

print_all() {
    echo "Scanned at: $(date '+%Y-%m-%d %H:%M:%S')"

    # Determine which prefixes to scan: explicit --model=X → just that one;
    # empty → all known prefixes that have at least one matching lineage.
    local prefixes_to_scan=()
    if [[ -n "$MODEL_PREFIX" ]]; then
        prefixes_to_scan=("$MODEL_PREFIX")
    else
        prefixes_to_scan=("${KNOWN_MODEL_PREFIXES[@]}")
    fi

    local n_blocks=0
    local first_block=1
    for pfx in "${prefixes_to_scan[@]}"; do
        MODEL_PREFIX="$pfx"
        # Pre-check: does this prefix have any matching dag dir? If not, skip
        # silently in multi-prefix mode (so the output isn't littered with
        # "no DAgger training dirs found" errors for prefixes the user
        # doesn't use). In single-prefix mode (--model=X) we DO want the
        # error, so handle that at the end.
        if [[ -z "$BASE_SHORT" ]] && [[ -z "$(discover_lineages)" ]]; then
            continue
        fi

        if (( first_block == 0 )); then
            echo
            echo "======================================================================"
            echo "== MODEL: $pfx"
            echo "======================================================================"
            echo
        elif (( ${#prefixes_to_scan[@]} > 1 )); then
            echo
            echo "======================================================================"
            echo "== MODEL: $pfx"
            echo "======================================================================"
            echo
        fi
        if print_all_for_prefix; then
            n_blocks=$((n_blocks + 1))
        fi
        first_block=0
    done

    if (( n_blocks == 0 )); then
        if (( ${#FILTERS[@]} > 0 )); then
            local _flist
            _flist=$(printf "'%s' " "${FILTERS[@]}")
            echo "ERROR: --filter ${_flist}matched no DAgger lineages (OR semantics)." >&2
            echo "  Tried prefix(es): ${prefixes_to_scan[*]}" >&2
            echo "  Drop --filter to see every lineage, or adjust the substrings." >&2
        else
            echo "ERROR: no DAgger training dirs found under $TRAINING_ROOT" >&2
            if [[ -n "$MODEL_PREFIX" ]]; then
                echo "  expected pattern: ${MODEL_PREFIX}_<lineage>[_ft]_dag<N>" >&2
            else
                echo "  scanned prefixes: ${KNOWN_MODEL_PREFIXES[*]}" >&2
                echo "  expected pattern: <prefix>_<lineage>[_ft]_dag<N>" >&2
            fi
        fi
        return 1
    fi

    # Show the orchestrator's intervention output dir for the in-flight round,
    # if any. Shared across lineages and prefixes; print once at the bottom.
    # Three layouts to handle (newest first):
    #   1. outputs/training/<policy>_dag<N>/dagger/interventions  (current)
    #   2. outputs/dagger/<policy>_dag<N>/interventions           (legacy mirror)
    #   3. outputs/dagger/round_<N>/interventions                 (original)
    local current_round_dir=""
    current_round_dir=$( { ls -td "${TRAINING_ROOT}"/*/dagger/interventions 2>/dev/null; \
                          ls -td "${HOME}"/code/lerobot/outputs/dagger/*/interventions 2>/dev/null; \
                          ls -td "${HOME}"/code/lerobot/outputs/dagger/round_*       2>/dev/null; \
                        } | head -1 || true)
    if [[ -n "$current_round_dir" ]]; then
        echo
        echo "Latest intervention dir: $current_round_dir"
    fi
}

if [[ -n "$WATCH_SEC" ]]; then
    while true; do
        clear
        print_all || true
        echo
        echo "Refreshing every ${WATCH_SEC}s. Ctrl-C to stop."
        sleep "$WATCH_SEC"
    done
else
    print_all
fi
