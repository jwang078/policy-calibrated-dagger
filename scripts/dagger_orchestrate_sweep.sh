#!/usr/bin/env bash

# Sweep wrapper for dagger_orchestrate.sh. Two modes:
#
#   SINGLE-RATIO MODE  (--sweep_blends="..."):
#       Invoke the orchestrator once per ratio listed in --sweep_blends, with
#       each iteration's --blends set to that single ratio.
#
#   COMBINATION MODE   (--combination_pool="..." --sweep_combinations_of=K):
#       Enumerate every K-combination from --combination_pool and invoke the
#       orchestrator once per combination with --blends set to the full
#       K-tuple. K=1 is equivalent to SINGLE-RATIO mode over the pool.
#
# Common to both: the orchestrator's BLENDS_TAG derivation (b<NNN>_<NNN>_…,
# sorted descending) auto-disambiguates lineage names across iterations,
# so --run_tag stays constant across the sweep. Each iteration is
# independent and resumable. Cached per-ratio blend datasets from previous
# sweep iterations (or single-ratio reruns) are reused automatically by the
# orchestrator's step-2 resume detection.
#
# Usage:
#   bash my_scripts/dagger_orchestrate_sweep.sh <mode flags> <orchestrator flags>
#
# Mode flags (pick exactly one mode):
#   --sweep_blends=LIST
#       SINGLE-RATIO. Space-separated single ratios; one orchestrator
#       invocation per ratio. Quote to keep it one shell word.
#         e.g. --sweep_blends="0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1"
#
#   --combination_pool=LIST --sweep_combinations_of=K[,K2,...]
#       COMBINATION. Enumerate every K-combination from the pool; one
#       orchestrator invocation per combination.
#         e.g. --combination_pool="0.1 0.3 0.5 0.7 0.9" --sweep_combinations_of=2
#       produces C(5,2)=10 invocations: pairs [0.1,0.3], [0.1,0.5], … [0.7,0.9].
#       K=1 → behaves like --sweep_blends over the pool. K must satisfy 1≤K≤N.
#
#       MULTI-K LISTS: --sweep_combinations_of accepts a comma- OR space-
#       separated list of K values (e.g. =1,2 or "1 2 3"), in which case
#       the wrapper validates each K independently (length checks per K),
#       then concatenates all C(N,Kᵢ) iterations into one sweep — same
#       --run_tag, blends_tag auto-disambiguates the lineages
#       (rerun_v1_b010 vs rerun_v1_b090_050 don't collide).
#         e.g. --sweep_combinations_of=1,2 over pool of 5 →
#              C(5,1)+C(5,2) = 5+10 = 15 iterations in one invocation.
#
# Common options:
#   --continue_on_error     Keep sweeping past failed iterations (default:
#                           abort on first failure).
#   --dry-run               Pass --dry-run to each orchestrator invocation
#                           (the wrapper still iterates over all combinations).
#   --auto_create_source    Before iterating, run dagger_orchestrate.sh once
#                           in NON-rerun mode to create the source lineage
#                           that --rerun_blends_from points at. Strips
#                           --rerun_blends_from and replaces --run_tag with
#                           the source's tag; everything else (--num_rounds,
#                           --intervention_n_episodes, --intervention_extra_args,
#                           --finetune_*, etc.) is forwarded. Always passes
#                           --resume to the create step so a fully-complete
#                           source exits 0 quickly; partial source resumes;
#                           missing source is created from scratch. Requires
#                           --rerun_blends_from=TAG (no `:BLENDS_TAG` form —
#                           sources with their own blends would need a
#                           separate spec we don't yet support).
#
#                           CONVENIENCE: when this flag is on, the wrapper
#                           auto-suffixes the rerun's --run_tag with the
#                           string set by --auto_rerun_tag_suffix (default
#                           "_rr") IF either (a) --run_tag is omitted, OR
#                           (b) --run_tag equals --rerun_blends_from. So:
#                             --rerun_blends_from=30ep   (--run_tag omitted)
#                                 → source's run_tag=30ep, rerun's=30ep_rr
#                             --run_tag=30ep --rerun_blends_from=30ep
#                                 → same as above (collision auto-resolved)
#                             --run_tag=my_v2 --rerun_blends_from=30ep
#                                 → explicit; no auto-suffix
#   --auto_rerun_tag_suffix=NAME
#                           Suffix used by the auto-disambiguation above.
#                           Default "_rr". Has no effect when --run_tag is
#                           explicitly set to something different from the
#                           source's tag.
#
# Pre-flight name length validation (COMBINATION mode only):
#   The wrapper predicts the merged-dataset name for every combination
#   (`<HF_USER>/<BASE_DATASET_SHORT>_<a|r>_dag<N>_m` and, if --skip_alias_step
#   is NOT set, the alias dataset too). If ANY combination's name would
#   exceed HuggingFace's 56-char repo-name limit, the wrapper PRINTS the
#   full per-combination table and EXITS BEFORE any orchestrator runs.
#   Shorten --run_tag or --dag_short_override and retry.
#
# Example (rerun-blends with combinations of 2):
#   bash my_scripts/dagger_orchestrate_sweep.sh \
#       --combination_pool="0.1 0.3 0.5 0.7 0.9" --sweep_combinations_of=2 \
#       --base_short=approach_lever_11_biasend_5path_grip0 \
#       --initial_policy_path=outputs/training/diffusion_approach_lever_11_biasend_5path_delta_basewrist \
#       --model=diff --action_format=rel \
#       --intermediate_mode=finetune --final_mode=scratch \
#       --target_intervention_volume=3 \
#       --finetune_steps=1000 --finetune_eval_freq=1000 --finetune_save_freq=1000 \
#       --env_external_port=6001 --skip_alias_step \
#       --dag_short_override=lever_grip0 --run_tag=rerun_v1 \
#       --rerun_blends_from=d5jvm \
#       --blend_extra_args='--blend_mode=every_step' \
#       --resume
#
# Resulting lineages for that example (10 finetune rounds + 1 from-scratch each):
#   diffusion_..._rerun_v1_b030_010   (pair [0.1, 0.3])
#   diffusion_..._rerun_v1_b050_010   (pair [0.1, 0.5])
#   diffusion_..._rerun_v1_b050_030   (pair [0.3, 0.5])
#   … etc, 10 total.

set -euo pipefail

SWEEP_BLENDS=""
COMBINATION_POOL=""
SWEEP_COMBINATIONS_OF=""
CONTINUE_ON_ERROR=false
AUTO_CREATE_SOURCE=false
AUTO_RERUN_TAG_SUFFIX="_rr"   # appended to source's run_tag when sweep tag is missing or collides
ORCHESTRATOR_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --sweep_blends=*)            SWEEP_BLENDS="${arg#*=}" ;;
        --combination_pool=*)        COMBINATION_POOL="${arg#*=}" ;;
        --sweep_combinations_of=*)   SWEEP_COMBINATIONS_OF="${arg#*=}" ;;
        --continue_on_error)         CONTINUE_ON_ERROR=true ;;
        --auto_create_source)        AUTO_CREATE_SOURCE=true ;;
        --auto_rerun_tag_suffix=*)   AUTO_RERUN_TAG_SUFFIX="${arg#*=}" ;;
        --blends=*)
            echo "ERROR: don't pass --blends to the sweep wrapper; use --sweep_blends or --combination_pool instead." >&2
            echo "  The wrapper splits the sweep spec into individual --blends invocations." >&2
            exit 1
            ;;
        -h|--help)
            sed -n '1,/^set -euo pipefail/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        *) ORCHESTRATOR_ARGS+=( "$arg" ) ;;
    esac
done

# Mode selection + mutual-exclusion validation.
if [[ -n "$COMBINATION_POOL" || -n "$SWEEP_COMBINATIONS_OF" ]]; then
    if [[ -z "$COMBINATION_POOL" || -z "$SWEEP_COMBINATIONS_OF" ]]; then
        echo "ERROR: --combination_pool and --sweep_combinations_of must be set TOGETHER." >&2
        echo "  Got --combination_pool='$COMBINATION_POOL', --sweep_combinations_of='$SWEEP_COMBINATIONS_OF'." >&2
        exit 1
    fi
    if [[ -n "$SWEEP_BLENDS" ]]; then
        echo "ERROR: --sweep_blends is mutually exclusive with --combination_pool / --sweep_combinations_of." >&2
        echo "  Pick one mode per invocation." >&2
        exit 1
    fi
fi

if [[ -z "$SWEEP_BLENDS" && -z "$COMBINATION_POOL" ]]; then
    echo "ERROR: must specify EITHER --sweep_blends='RATIO_LIST' OR --combination_pool + --sweep_combinations_of." >&2
    echo "  Examples:" >&2
    echo "    --sweep_blends='0.9 0.8 0.7'" >&2
    echo "    --combination_pool='0.1 0.3 0.5 0.7 0.9' --sweep_combinations_of=2" >&2
    exit 1
fi

# Auto-disambiguate rerun's --run_tag from source's tag. The orchestrator
# requires `--run_tag != --rerun_blends_from` (to keep the new lineage from
# clobbering the source on disk). For the convenience case where the user
# either (a) omits --run_tag entirely or (b) reuses the same value for both,
# auto-suffix the rerun's run_tag with $AUTO_RERUN_TAG_SUFFIX so the user
# doesn't have to think about "they must differ" — one tag input, two
# distinct lineage families on disk.
# When user provides a distinct --run_tag, we leave it alone (explicit > magic).
# Runs before the length check (Python helper) so dataset-name predictions
# reflect the final tag the orchestrator will actually receive.
if [[ "$AUTO_CREATE_SOURCE" == "true" ]]; then
    _CUR_RUN_TAG=""
    _CUR_RERUN_FROM=""
    for a in "${ORCHESTRATOR_ARGS[@]}"; do
        case "$a" in
            --run_tag=*)            _CUR_RUN_TAG="${a#*=}" ;;
            --rerun_blends_from=*)  _CUR_RERUN_FROM="${a#*=}" ;;
        esac
    done
    if [[ -z "$_CUR_RERUN_FROM" ]]; then
        echo "ERROR: --auto_create_source requires --rerun_blends_from=TAG to also be set." >&2
        exit 1
    fi
    # Parse source's run_tag out of --rerun_blends_from=TAG[:BLENDS_TAG].
    _SRC_RUN_TAG="${_CUR_RERUN_FROM%%:*}"
    if [[ -z "$_CUR_RUN_TAG" || "$_CUR_RUN_TAG" == "$_SRC_RUN_TAG" ]]; then
        _NEW_RUN_TAG="${_SRC_RUN_TAG}${AUTO_RERUN_TAG_SUFFIX}"
        if [[ "$_NEW_RUN_TAG" == "$_SRC_RUN_TAG" ]]; then
            echo "ERROR: --auto_rerun_tag_suffix is empty; auto-disambiguation would still collide." >&2
            exit 1
        fi
        _filtered=()
        for a in "${ORCHESTRATOR_ARGS[@]}"; do
            case "$a" in
                --run_tag=*) continue ;;
                *) _filtered+=( "$a" ) ;;
            esac
        done
        ORCHESTRATOR_ARGS=( "${_filtered[@]}" "--run_tag=$_NEW_RUN_TAG" )
        if [[ -z "$_CUR_RUN_TAG" ]]; then
            echo "[auto_create_source] --run_tag not provided; using rerun_tag='$_NEW_RUN_TAG' (= source_tag + '$AUTO_RERUN_TAG_SUFFIX')."
        else
            echo "[auto_create_source] --run_tag collided with --rerun_blends_from='$_SRC_RUN_TAG'; using rerun_tag='$_NEW_RUN_TAG' (= source_tag + '$AUTO_RERUN_TAG_SUFFIX')."
        fi
    fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORCH="$SCRIPT_DIR/dagger_orchestrate.sh"
if [[ ! -f "$ORCH" ]]; then
    echo "ERROR: dagger_orchestrate.sh not found at $ORCH" >&2
    exit 1
fi

# Build the iteration list. Each entry is a single --blends argument value
# (one ratio for single-ratio mode, a space-separated K-tuple for combo mode).
RATIO_LISTS_ARR=()

if [[ -n "$COMBINATION_POOL" ]]; then
    # COMBINATION mode. Hand off to a Python helper that (per K value):
    #   (1) enumerates C(N, K) combinations from the pool,
    #   (2) replicates dagger_orchestrate.sh's BASE_DATASET_SHORT derivation
    #       (stem + run_tag + model_tag + method_tag + blends_tag) for each
    #       combination,
    #   (3) checks every predicted longest derived dataset name against the
    #       56-char HF limit (merged and, when --skip_alias_step is NOT set,
    #       alias too — alias is longer for diffusion lineages),
    #   (4) prints the per-combination table to stderr, and
    #   (5) on stdout, emits one combination per line if and only if ALL
    #       combinations fit. If ANY would overflow, exits non-zero before
    #       any orchestrator runs.
    # The helper inspects ORCHESTRATOR_ARGS by name (--base_short, --run_tag,
    # --model, etc.) to keep its derivation in lock-step with the orchestrator
    # without re-parsing every flag.
    #
    # --sweep_combinations_of accepts a single K (e.g. "2") OR a comma /
    # space-separated list of K values (e.g. "1,2" or "1 2 3"). For each K
    # we run the helper independently; if ANY K has any overflow, we abort
    # before any orchestrator runs. Combos across all K values are concatenated
    # into a single RATIO_LISTS_ARR for the iteration loop below.
    HF_USER="${HF_USER:-JennyWWW}"
    IFS=', ' read -ra SWEEP_COMBO_K_LIST <<< "$SWEEP_COMBINATIONS_OF"
    _k_filtered=()
    for k in "${SWEEP_COMBO_K_LIST[@]}"; do
        [[ -n "$k" ]] && _k_filtered+=( "$k" )
    done
    SWEEP_COMBO_K_LIST=( "${_k_filtered[@]}" )
    if (( ${#SWEEP_COMBO_K_LIST[@]} == 0 )); then
        echo "ERROR: --sweep_combinations_of parsed to an empty list. Got: '$SWEEP_COMBINATIONS_OF'" >&2
        exit 1
    fi
    for k in "${SWEEP_COMBO_K_LIST[@]}"; do
        if ! [[ "$k" =~ ^[0-9]+$ ]] || (( k < 1 )); then
            echo "ERROR: --sweep_combinations_of value '$k' must be a positive integer." >&2
            echo "  Use a single value (e.g. =2) or a list (e.g. =1,2 or '1 2 3')." >&2
            exit 1
        fi
    done

    # PYTHONPATH=$SCRIPT_DIR so the inline helper can `import dagger_naming`
    # (the canonical naming module — keeps this length-check in lock-step
    # with the orchestrator's actual name derivation).
    for SWEEP_K_VAL in "${SWEEP_COMBO_K_LIST[@]}"; do
    COMBO_OUTPUT=$(PYTHONPATH="$SCRIPT_DIR" python3 - "$COMBINATION_POOL" "$SWEEP_K_VAL" "$HF_USER" "${ORCHESTRATOR_ARGS[@]}" <<'PY' || exit $?
import sys
from itertools import combinations

from dagger_naming import (
    derive_base_dataset_short,
    format_blends_tag,
    merged_repo,
    alias_repo,
    nocoll_short,
)

pool_str = sys.argv[1]
k = int(sys.argv[2])
hf_user = sys.argv[3]
orch_args = sys.argv[4:]

def get_flag(name, default=None):
    pfx = f"--{name}="
    for a in orch_args:
        if a.startswith(pfx):
            return a[len(pfx):]
    return default

def has_bool_flag(name):
    return f"--{name}" in orch_args

# Pull every orchestrator flag that affects BASE_DATASET_SHORT.
base_short = get_flag("base_short") or ""
base_repo_id = get_flag("base_repo_id") or ""
dag_short_override = get_flag("dag_short_override") or ""
run_tag = get_flag("run_tag") or ""
model = get_flag("model") or "pi"
intervention_method = get_flag("intervention_method") or "rrt"
action_format = get_flag("action_format") or "rel"
num_rounds_str = get_flag("num_rounds") or ""
skip_alias = has_bool_flag("skip_alias_step")
strip_splatsim = not has_bool_flag("no_strip_splatsim_prefix")   # default true
filter_collisions = has_bool_flag("filter_blend_collisions")
# In rerun mode the blend (and therefore the `_nocoll` sibling) name uses
# the SOURCE lineage's prefix, not the current sweep iteration's base_short.
# Source datasets are read-only and were length-validated when they were
# originally created, so skip the nocoll length probe to avoid false-
# positive overflows.
rerun_mode = bool(get_flag("rerun_blends_from")) or bool(get_flag("reuse_intervention_from"))

if not (base_short or base_repo_id):
    print("ERROR: --base_short or --base_repo_id required for combination length validation.", file=sys.stderr)
    sys.exit(1)

# Mirror dagger_orchestrate.sh's BASE_DATASET_STEM derivation.
if base_repo_id:
    stem = base_repo_id.split("/", 1)[1] if "/" in base_repo_id else base_repo_id
else:
    stem = base_short
if strip_splatsim and stem.startswith("splatsim_"):
    stem = stem[len("splatsim_"):]
if dag_short_override:
    stem = dag_short_override

MODEL_TAGS  = {"pi": "", "diff": "diff", "act": "act"}
METHOD_TAGS = {"rrt": "", "oracle_goal": "og"}
if model not in MODEL_TAGS:
    print(f"ERROR: unknown --model='{model}' (expected pi/diff/act).", file=sys.stderr); sys.exit(1)
if intervention_method not in METHOD_TAGS:
    print(f"ERROR: unknown --intervention_method='{intervention_method}'.", file=sys.stderr); sys.exit(1)
if action_format == "rel":
    action_infix = "r"
elif action_format == "abs":
    action_infix = "a"
else:
    print(f"ERROR: unknown --action_format='{action_format}'.", file=sys.stderr); sys.exit(1)
model_tag  = MODEL_TAGS[model]
method_tag = METHOD_TAGS[intervention_method]

# Conservative default for the length check. The orchestrator's own pre-flight
# does the final authoritative check at the actual NUM_ROUNDS; this is just an
# early-warning gate. dag<N> width is fixed at 2 chars for N=10..99, so any
# NUM_ROUNDS in that range gives identical predicted lengths.
num_rounds = int(num_rounds_str) if num_rounds_str else 10

# Parse the pool. Mirror orchestrator's 1.0-drop and 0.0-reject behavior so
# the user sees consistent semantics.
pool = []
for r_str in pool_str.replace(",", " ").replace("[", " ").replace("]", " ").split():
    r = float(r_str)
    if r == 0.0:
        print("ERROR: --combination_pool contains 0.0 (orchestrator rejects 0.0).", file=sys.stderr); sys.exit(1)
    if r == 1.0:
        # silently filter; matches the orchestrator's "= raw intervention" skip
        continue
    pool.append(r)

if k <= 0:
    print(f"ERROR: --sweep_combinations_of={k} must be >= 1.", file=sys.stderr); sys.exit(1)
if k > len(pool):
    print(f"ERROR: --sweep_combinations_of={k} exceeds pool size {len(pool)} (after filtering 1.0).", file=sys.stderr); sys.exit(1)

combos = list(combinations(pool, k))

def predicted_longest(combo):
    """Predict longest derived dataset name for a given combination.

    Uses dagger_naming's canonical helpers so this stays in lock-step with
    the orchestrator's actual derivation."""
    blends_tag = format_blends_tag(list(combo))
    base_dataset_short = derive_base_dataset_short(
        stem, run_tag=run_tag, model_tag=model_tag, method_tag=method_tag, blends_tag=blends_tag
    )
    candidates = [merged_repo(hf_user, base_dataset_short, action_infix, num_rounds)]
    if not skip_alias:
        # Note: in non-rerun mode SOURCE_INT_SHORT_PREFIX == BASE_DATASET_SHORT,
        # so the alias name is derived from base_dataset_short. In rerun mode
        # the alias step is typically skipped (--skip_alias_step) so this
        # branch is only exercised for fresh-recording sweeps anyway.
        candidates.append(
            alias_repo(hf_user, base_dataset_short, action_infix, num_rounds, model, action_format)
        )
    if filter_collisions and not rerun_mode:
        # Fresh-recording sweep: the `_nocoll` blend uses the sweep iteration's
        # BASE_DATASET_SHORT as the source prefix, so it's worth probing for
        # overflows that the bare blend name wouldn't trigger. Any ratio in the
        # combo works — all blend ratios produce the same name length.
        # In rerun mode this name uses the SOURCE lineage's prefix instead,
        # which was validated at source-creation time → skip.
        any_ratio = combo[0]
        nc_short = nocoll_short(base_dataset_short, action_infix, num_rounds, any_ratio)
        candidates.append(f"{hf_user}/{nc_short}")
    return max(candidates, key=len)

# Per-combination table + overflow detection.
print(f"Combination-mode sweep: C({len(pool)}, {k}) = {len(combos)} combination(s)", file=sys.stderr)
print(f"  Pool (after filtering 1.0): {pool}", file=sys.stderr)
print(f"  K (combination size): {k}", file=sys.stderr)
print(f"  Length check uses NUM_ROUNDS=dag{num_rounds} (orchestrator does authoritative check at pre-flight)", file=sys.stderr)
print(file=sys.stderr)
print(f"  {'combination':<28}  {'predicted longest derived dataset':<60}  {'len':>4}  status", file=sys.stderr)
print(f"  {'-'*28}  {'-'*60}  ----  ------", file=sys.stderr)

overflows = []
combo_lines = []
for combo in combos:
    longest = predicted_longest(combo)
    overflows_flag = len(longest) > 56
    combo_str = " ".join(f"{r:g}" for r in combo)
    if overflows_flag:
        overflows.append(combo_str)
    print(f"  [{combo_str}]".ljust(30) + f"  {longest:<60}  {len(longest):>4}  " + ("OVERFLOW" if overflows_flag else "ok"), file=sys.stderr)
    combo_lines.append(combo_str)

if overflows:
    print(file=sys.stderr)
    print(f"ERROR: {len(overflows)} of {len(combos)} combinations would exceed HuggingFace's 56-char repo-name limit.", file=sys.stderr)
    print("  No orchestrator runs were started. To proceed, shorten one of:", file=sys.stderr)
    print(f"    --run_tag (currently '{run_tag}', {len(run_tag)} chars)", file=sys.stderr)
    if dag_short_override:
        print(f"    --dag_short_override (currently '{dag_short_override}', {len(dag_short_override)} chars)", file=sys.stderr)
    else:
        print(f"    add --dag_short_override=<SHORTER> (current stem '{stem}', {len(stem)} chars)", file=sys.stderr)
    print(f"  Or reduce --sweep_combinations_of from {k} (fewer ratios per combination → shorter blends_tag).", file=sys.stderr)
    sys.exit(2)

# stdout: one combination per line, space-separated ratios. Consumed by the
# bash caller into RATIO_LISTS_ARR.
for line in combo_lines:
    print(line)
PY
)
    helper_rc=$?
    if (( helper_rc != 0 )); then
        # The Python helper already streamed the diagnostic table + error reason to stderr.
        exit "$helper_rc"
    fi
    # Append this K's combos to the cumulative iteration list.
    while IFS= read -r _line; do
        [[ -n "$_line" ]] && RATIO_LISTS_ARR+=( "$_line" )
    done <<< "$COMBO_OUTPUT"
    done   # for SWEEP_K_VAL

else
    # SINGLE-RATIO mode. Parse the list (allow brackets/commas for ergonomics).
    _clean=$(echo "$SWEEP_BLENDS" | tr ',[]' '   ')
    # shellcheck disable=SC2206  # intentional word-split
    RATIO_LISTS_ARR=( $_clean )
    if (( ${#RATIO_LISTS_ARR[@]} == 0 )); then
        echo "ERROR: --sweep_blends parsed to an empty list. Got: '$SWEEP_BLENDS'" >&2
        exit 1
    fi
fi

total=${#RATIO_LISTS_ARR[@]}
if [[ -n "$COMBINATION_POOL" ]]; then
    echo
    echo "All $total combination(s) fit within the 56-char limit. Starting sweep."
else
    echo "Sweep over $total ratio(s): ${RATIO_LISTS_ARR[*]}"
fi

# Optional source-create step. Always invoked with --resume so the orchestrator
# is responsible for "is it already done?" detection; if complete, it exits 0
# in seconds. The sweep iterations below assume the source exists; this step
# guarantees that.
if [[ "$AUTO_CREATE_SOURCE" == "true" ]]; then
    REVERSE_FROM=""
    for a in "${ORCHESTRATOR_ARGS[@]}"; do
        case "$a" in
            --rerun_blends_from=*) REVERSE_FROM="${a#*=}" ;;
        esac
    done
    if [[ -z "$REVERSE_FROM" ]]; then
        echo "ERROR: --auto_create_source requires --rerun_blends_from=TAG to also be set." >&2
        echo "  We need the source's run_tag to know which lineage to create." >&2
        exit 1
    fi
    if [[ "$REVERSE_FROM" == *:* ]]; then
        echo "ERROR: --auto_create_source doesn't support --rerun_blends_from=TAG:BLENDS_TAG yet." >&2
        echo "  Sources with their own blends would need a separate spec we don't have." >&2
        echo "  Got --rerun_blends_from='$REVERSE_FROM'." >&2
        exit 1
    fi
    SOURCE_RUN_TAG="$REVERSE_FROM"
    # Build CREATE_ARGS = ORCHESTRATOR_ARGS minus --rerun_blends_from and minus
    # --run_tag (we replace it with source's tag), then append the source tag
    # and --resume. Any --blends caller had in ORCHESTRATOR_ARGS will also be
    # stripped (defensive — caller shouldn't pass --blends to the wrapper, but
    # belt-and-suspenders since the rerun's iteration tag has its own --blends).
    CREATE_ARGS=()
    for a in "${ORCHESTRATOR_ARGS[@]}"; do
        case "$a" in
            --rerun_blends_from=*|--run_tag=*|--blends=*) continue ;;
            *) CREATE_ARGS+=( "$a" ) ;;
        esac
    done
    CREATE_ARGS+=( "--run_tag=$SOURCE_RUN_TAG" --resume )

    echo
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo "Auto-create source lineage (--auto_create_source): run_tag='$SOURCE_RUN_TAG'"
    echo "  Invoking:"
    echo "    bash $ORCH ${CREATE_ARGS[*]}"
    echo "  (Idempotent: if the source is already fully complete, this exits 0 in seconds.)"
    echo "════════════════════════════════════════════════════════════════════════════════"
    if ! bash "$ORCH" "${CREATE_ARGS[@]}"; then
        echo "Auto-create source step FAILED. Aborting sweep before any blend iteration." >&2
        exit 1
    fi
    echo "Auto-create source step complete; proceeding with sweep."
    echo
fi
echo "Each iteration invokes:"
echo "  bash $ORCH --blends=<combo> ${ORCHESTRATOR_ARGS[*]}"
echo

n_succ=0
n_fail=0
failures=()
sweep_start=$(date +%s)
for i in "${!RATIO_LISTS_ARR[@]}"; do
    combo="${RATIO_LISTS_ARR[$i]}"
    # Infer K from the combo (number of space-separated ratios). Lets a
    # multi-K sweep label each iteration with its blend count without
    # needing a parallel K-array.
    iter_k=$(echo "$combo" | wc -w | tr -d ' ')
    iter_start=$(date +%s)
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo "Sweep iteration $((i + 1)) / $total: K=$iter_k --blends=\"$combo\""
    echo "  (elapsed sweep time so far: $(( iter_start - sweep_start ))s)"
    echo "════════════════════════════════════════════════════════════════════════════════"
    if bash "$ORCH" --blends="$combo" "${ORCHESTRATOR_ARGS[@]}"; then
        n_succ=$((n_succ + 1))
        echo "Sweep iteration --blends=\"$combo\" SUCCEEDED ($(( $(date +%s) - iter_start ))s)."
    else
        n_fail=$((n_fail + 1))
        failures+=( "$combo" )
        echo "Sweep iteration --blends=\"$combo\" FAILED ($(( $(date +%s) - iter_start ))s)."
        if [[ "$CONTINUE_ON_ERROR" != "true" ]]; then
            echo "Aborting sweep. Pass --continue_on_error to keep going past failures."
            break
        fi
    fi
    echo
done

echo "════════════════════════════════════════════════════════════════════════════════"
echo "Sweep complete: $n_succ succeeded, $n_fail failed (total wall time: $(( $(date +%s) - sweep_start ))s)."
if (( n_fail > 0 )); then
    echo "Failures at: ${failures[*]}"
    exit 1
fi
