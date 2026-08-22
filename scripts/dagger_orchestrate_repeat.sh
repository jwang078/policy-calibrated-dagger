#!/usr/bin/env bash

# Repeat wrapper around dagger_orchestrate_sweep.sh: run the SAME sweep N
# times with identical hyperparameters, so per-round success rate can be
# reported as mean ± std across repetitions. Plot the result with
#   python my_scripts/dagger_plot_repeats.py --rep_tag=<TAG> --model=<MODEL>
#
# Usage:
#   bash my_scripts/dagger_orchestrate_repeat.sh --repeats=10 <every sweep flag>
#
# Flags consumed HERE (everything else forwards verbatim to the sweep):
#   --repeats=N                  Number of repetitions (required, >= 1).
#   --repeat_from=K              First repetition to run (default 1). Use to
#                                extend a finished study (--repeats=15
#                                --repeat_from=11) or to hop past reps you're
#                                running elsewhere. Every rep is individually
#                                resumable through the sweep's own --resume.
#   --repeat_continue_on_error   Keep going past a failed repetition (default:
#                                abort on first failure). Distinct from the
#                                sweep-level --continue_on_error, which governs
#                                blend-iterations WITHIN a rep and forwards
#                                through untouched.
#   --no_repeat_vary_seeds       Don't inject the per-rep seed variation
#                                described below (reps then differ only through
#                                GPU nondeterminism — probably not what you
#                                want for a mean±std study).
#   --repeat_dry_run          Print each repetition's full sweep command and
#                                exit without running anything.
#
# NAMING (kept to ONE extra character per rep for HF's 56-char repo limit):
#   Repetition 1 uses the tags exactly as given, so an existing lineage counts
#   as rep 1 and resumes instead of redoing work. Repetition k >= 2 rewrites
#   --run_tag=TAG and --rerun_blends_from=TAG to TAG<k> (03dag -> 03dag2).
#   The sweep's own pre-flight length check re-validates every rep's names.
#
# WHAT VARIES BETWEEN REPETITIONS (unless --no_repeat_vary_seeds):
#   * Intervention-recording rollout randomness: rep k >= 2 appends
#     `--seed=<k-1>` to --intervention_extra_args, which wins over the
#     orchestrator's emitted --seed=0 (draccus last-occurrence-wins). The
#     seed feeds set_seed (torch/np — diffusion sampling noise + RRT planner
#     draws); scenario coverage is identical across reps regardless, because
#     splatsim's EVAL_BENCHMARK scenario selection is positional
#     (benchmark_start_index = absolute episode index), not seed-driven.
#     (Without this, the whole recording is deterministically seeded from
#     --seed=0 and reps differ only through GPU nondeterminism.)
#   * Blend rollout noise: rep k >= 2 shifts --sample_seed by k-1 inside
#     --blend_extra_args (rewriting an existing non-negative value, or
#     appending 42+k-1 to mirror the blend script's default of 42). An
#     explicit negative sample_seed (= unseeded) is left alone.
#
# WHAT DELIBERATELY DOES NOT VARY:
#   * Training seeds. Kept identical so every rep trains with the exact same
#     hyperparameters; randomness enters each rep's training through its
#     independently-recorded data, which is the point. (Eval scenario coverage
#     would be unaffected either way — scenario selection is positional, not
#     seed-driven.)
#   * --intervention_sample_seed (which scenarios get recorded): all reps
#     attempt the same scenario subset, apples-to-apples.
#
# ROUND-0 SHARING: the round-0 base training dir carries no run tag
# (outputs/training/<model>_<base_short>_<action>_<camtag>), so rep 1 trains
# it once and every later rep skip-detects the existing checkpoint. That is
# also why reps run SEQUENTIALLY here — a parallel rep 2 would race rep 1's
# round-0 training.

set -euo pipefail

REPEATS=""
REPEAT_FROM=1
REPEAT_CONTINUE_ON_ERROR=false
REPEAT_VARY_SEEDS=true
REPEAT_DRY_RUN=false
SWEEP_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --repeats=*)                REPEATS="${arg#*=}" ;;
        --repeat_from=*)            REPEAT_FROM="${arg#*=}" ;;
        --repeat_continue_on_error) REPEAT_CONTINUE_ON_ERROR=true ;;
        --no_repeat_vary_seeds)     REPEAT_VARY_SEEDS=false ;;
        --repeat_dry_run)           REPEAT_DRY_RUN=true ;;
        -h|--help)
            sed -n '1,/^set -euo pipefail/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        *) SWEEP_ARGS+=( "$arg" ) ;;
    esac
done

if ! [[ "$REPEATS" =~ ^[0-9]+$ ]] || (( REPEATS < 1 )); then
    echo "ERROR: --repeats=N is required (positive integer). Got: '${REPEATS:-<unset>}'" >&2
    exit 1
fi
if ! [[ "$REPEAT_FROM" =~ ^[0-9]+$ ]] || (( REPEAT_FROM < 1 || REPEAT_FROM > REPEATS )); then
    echo "ERROR: --repeat_from must be in [1, --repeats]. Got: '$REPEAT_FROM' (repeats=$REPEATS)" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP="$SCRIPT_DIR/dagger_orchestrate_sweep.sh"
if [[ ! -f "$SWEEP" ]]; then
    echo "ERROR: dagger_orchestrate_sweep.sh not found at $SWEEP" >&2
    exit 1
fi

# The tag we suffix per rep. --rerun_blends_from's tag part is authoritative
# (that's the source lineage each rep re-records); fall back to --run_tag for
# non-rerun invocations.
BASE_TAG=""
RESUME_SET=false
for a in "${SWEEP_ARGS[@]}"; do
    case "$a" in
        --rerun_blends_from=*) v="${a#*=}"; BASE_TAG="${v%%:*}" ;;
        --run_tag=*)           [[ -z "$BASE_TAG" ]] && BASE_TAG="${a#*=}" ;;
        --resume)              RESUME_SET=true ;;
    esac
done
if [[ -z "$BASE_TAG" ]]; then
    echo "ERROR: need --rerun_blends_from=TAG or --run_tag=TAG in the sweep args" >&2
    echo "  (the repetition suffix is applied to that tag)." >&2
    exit 1
fi
if [[ "$RESUME_SET" != "true" ]]; then
    echo "WARNING: --resume is not among the sweep args; every orchestrator invocation in"
    echo "  every repetition will pause for interactive confirmation. Strongly consider --resume."
fi

# Build one repetition's sweep argv: suffix the tags, inject the per-rep seeds.
# Rep 1 passes through byte-identical (an existing lineage IS rep 1).
build_rep_args() {
    local k="$1"
    local suf=""
    (( k >= 2 )) && suf="$k"
    local seed_off=$(( k - 1 ))
    local saw_int_extra=false
    REP_ARGS=()
    local a v
    for a in "${SWEEP_ARGS[@]}"; do
        case "$a" in
            --run_tag=*)
                REP_ARGS+=( "--run_tag=${a#*=}$suf" ) ;;
            --rerun_blends_from=*)
                v="${a#*=}"
                if [[ "$v" == *:* ]]; then
                    REP_ARGS+=( "--rerun_blends_from=${v%%:*}$suf:${v#*:}" )
                else
                    REP_ARGS+=( "--rerun_blends_from=$v$suf" )
                fi
                ;;
            --intervention_extra_args=*)
                saw_int_extra=true
                v="${a#*=}"
                if [[ "$REPEAT_VARY_SEEDS" == "true" && $k -ge 2 ]]; then
                    # Appended last → draccus last-occurrence-wins over the
                    # orchestrator's emitted --seed=0 AND any --seed already
                    # in the user string.
                    v="$v --seed=$seed_off"
                fi
                REP_ARGS+=( "--intervention_extra_args=$v" )
                ;;
            --blend_extra_args=*)
                v="${a#*=}"
                if [[ "$REPEAT_VARY_SEEDS" == "true" && $k -ge 2 ]]; then
                    if [[ "$v" =~ --sample_seed[=\ ]([0-9]+) ]]; then
                        local base_seed="${BASH_REMATCH[1]}"
                        v="${v//--sample_seed=$base_seed/--sample_seed=$(( base_seed + seed_off ))}"
                        v="${v//--sample_seed $base_seed/--sample_seed $(( base_seed + seed_off ))}"
                    elif [[ ! "$v" =~ --sample_seed ]]; then
                        # Blend script default is 42; mirror it + offset.
                        v="$v --sample_seed=$(( 42 + seed_off ))"
                    fi
                    # explicit negative sample_seed (unseeded) → leave alone
                fi
                REP_ARGS+=( "--blend_extra_args=$v" )
                ;;
            *) REP_ARGS+=( "$a" ) ;;
        esac
    done
    if [[ "$REPEAT_VARY_SEEDS" == "true" && $k -ge 2 && "$saw_int_extra" != "true" ]]; then
        REP_ARGS+=( "--intervention_extra_args=--seed=$seed_off" )
    fi
}

echo "Repeat study: tag '$BASE_TAG', repetitions $REPEAT_FROM..$REPEATS (rep 1 = bare tag, rep k = '${BASE_TAG}<k>')"
echo "  Per-rep seed variation: $REPEAT_VARY_SEEDS"
echo

n_succ=0
n_fail=0
failures=()
study_start=$(date +%s)
for (( k = REPEAT_FROM; k <= REPEATS; k++ )); do
    build_rep_args "$k"
    rep_tag="$BASE_TAG"; (( k >= 2 )) && rep_tag="$BASE_TAG$k"
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo "REPETITION $k / $REPEATS (tag: $rep_tag)"
    echo "  bash $SWEEP ${REP_ARGS[*]}"
    echo "════════════════════════════════════════════════════════════════════════════════"
    if [[ "$REPEAT_DRY_RUN" == "true" ]]; then
        continue
    fi
    rep_start=$(date +%s)
    if bash "$SWEEP" "${REP_ARGS[@]}"; then
        n_succ=$((n_succ + 1))
        echo "Repetition $k SUCCEEDED ($(( $(date +%s) - rep_start ))s)."
    else
        n_fail=$((n_fail + 1))
        failures+=( "$rep_tag" )
        echo "Repetition $k FAILED ($(( $(date +%s) - rep_start ))s)."
        if [[ "$REPEAT_CONTINUE_ON_ERROR" != "true" ]]; then
            echo "Aborting. Pass --repeat_continue_on_error to keep going past failed reps."
            break
        fi
    fi
    echo
done

if [[ "$REPEAT_DRY_RUN" == "true" ]]; then
    echo "(--repeat_dry_run: nothing was run)"
    exit 0
fi

echo "════════════════════════════════════════════════════════════════════════════════"
echo "Repeat study complete: $n_succ succeeded, $n_fail failed (total: $(( $(date +%s) - study_start ))s)."
echo "Plot mean ± std across reps with:"
echo "  python my_scripts/dagger_plot_repeats.py --rep_tag=$BASE_TAG"
if (( n_fail > 0 )); then
    echo "Failures at reps: ${failures[*]}"
    exit 1
fi
