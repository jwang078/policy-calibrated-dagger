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

set -euo pipefail

BASE_SHORT=""
ACTION_TAG="abs"
RUN_TAG=""
MODEL_PREFIX=""   # empty = scan all known prefixes; set via --model=PREFIX to filter
WATCH_SEC=""
FILTERS=()
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
        --base_short=*)  BASE_SHORT="${arg#*=}" ;;
        --action=*)      ACTION_TAG="${arg#*=}" ;;
        --run_tag=*)     RUN_TAG="${arg#*=}" ;;
        --model=*)       MODEL_PREFIX="${arg#*=}" ;;
        --watch=*)       WATCH_SEC="${arg#*=}" ;;
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
    # Eval line: last "Suite overall aggregated" entry.
    local eval_line=""
    eval_line=$({ grep "Suite overall aggregated" "$log" 2>/dev/null | tail -1; } || true)
    local eval_step=""
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
    print(f'{succ:.0f} {pos:.3f} {ori:.1f} {coll:.1f} {trunc:.1f}')
" 2>/dev/null || echo "")
    fi
    local succ="" pos_err="" ori_err="" in_coll="" trunc=""
    if [[ -n "$eval_summary" ]]; then
        read -r succ pos_err ori_err in_coll trunc <<< "$eval_summary"
    else
        succ="-"; pos_err="-"; ori_err="-"; in_coll="-"; trunc="-"
    fi
    [[ -z "$loss" ]]  && loss="-"
    [[ -z "$grad" ]]  && grad="-"
    [[ -z "$lr"   ]]  && lr="-"
    [[ -z "$eval_step" ]] && eval_step="-"
    printf "%-6s %-9s %-9s %-9s %-6s %-9s %-9s %-8s %-6s %s\n" \
        "$round_label" "$loss" "$grad" "$lr" "$succ" "$pos_err" "$ori_err" "$in_coll" "$trunc" "$eval_step"
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
    printf "%-6s %-9s %-9s %-9s %-6s %-9s %-9s %-8s %-6s %s\n" \
        "Round" "loss" "grad" "lr" "succ%" "pos_err" "ori_err" "in_coll" "trunc" "eval_step"
    printf "%-6s %-9s %-9s %-9s %-6s %-9s %-9s %-8s %-6s %s\n" \
        "-----" "----" "----" "--" "-----" "-------" "-------" "-------" "-----" "---------"

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

    for DIR in $dirs; do
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
            echo "Latest train log: $(basename "$(dirname "$(dirname "$(dirname "$(dirname "$newest_log")")")")")  ($last_step)"
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
