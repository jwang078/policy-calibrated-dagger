#!/usr/bin/env bash

# Clean up a DAgger lineage's artifacts (training dirs, datasets, alias
# datasets, merged datasets, stats sidecars, blend datasets) given any one
# of its per-round training dirs. The base policy training dir and the
# plots/comparison_* dirs under outputs/dagger/ are preserved.
#
# Implementation: thin wrapper that derives the orchestrator argv for the
# target lineage and invokes:
#     bash dagger_orchestrate.sh <derived argv> --force_restart --cleanup_only
# so the deletion logic lives entirely in dagger_orchestrate.sh.
#
# Two argv-resolution paths, tried in order:
#   1. Sidecar:  read <train_dir>/dagger/config.json's recorded argv
#                (written by dagger_orchestrate.sh every round, since the
#                config-sidecar change). Exact recovery — no guessing.
#   2. Auto-detect:  derive flags from the training-dir name + disk scan
#                for older lineages that predate the sidecar.
#                Recovers --base_short, --model, --action_format, --num_rounds,
#                --initial_policy_path, --run_tag, --dag_short_override (when
#                used), --intermediate_mode by parsing the basename and
#                globbing the HF cache. Falls back to placeholder values for
#                flags the orchestrator requires but cleanup doesn't care
#                about (e.g. --finetune_steps).
#
# Usage:
#   bash my_scripts/dagger_cleanup_lineage.sh <training_dir_path> \
#       [--dry-run] [-y|--yes] [--detect_siblings]
#
# Options:
#   --dry-run   Pass --dry-run through to dagger_orchestrate.sh (lists what
#               would be deleted, doesn't rm).
#   -y, --yes   Pipe "restart" into the orchestrator's confirmation prompt so
#               the deletion runs unattended. Without this, the orchestrator
#               will prompt for confirmation interactively.
#   --also_delete_blends
#               Forwarded to dagger_orchestrate.sh. In rerun-blends mode,
#               blend datasets are by default PRESERVED on cleanup since
#               they're cross-rerun-cacheable (see orchestrator's
#               --also_delete_blends docstring). Set this flag to delete
#               them too. No effect outside rerun mode.
#   --filter_blend_collisions
#               Forwarded to dagger_orchestrate.sh for cosmetic symmetry.
#               The orchestrator's cleanup ALWAYS attempts to rm -rf any
#               `_nocoll` siblings whether this flag is set or not (rm is
#               idempotent on missing paths), so this flag mainly serves
#               to be recorded in the cleanup invocation's audit trail.
#   --keep_round_1_intervention
#               PRESERVE round 1's raw intervention dataset + alias +
#               int-stats sidecar. Round 1's merged dataset + training dir,
#               all blends, and every round 2..N artifact are still wiped.
#               Useful for "restart all the finetuning from scratch but keep
#               the expensive round-1 human recording" workflows. Without
#               this flag (and without -y), the script prompts y/n
#               interactively. With -y but without this flag, defaults to
#               DELETE round 1 (the legacy behavior). No-op in rerun mode.
#   --detect_siblings
#               Auto-detect "sibling" lineages on disk that share the same
#               prefix (everything up through the run_tag) AND the same
#               number of blend ratios K as the target, then run cleanup on
#               all of them. Useful for cleaning up an entire K-fold sweep
#               family in one command. Example: given a single 1-blend
#               lineage `rerun_v1_b010`, detects all of
#               {b010, b030, b050, b070, b090} (any rerun_v1_b<NNN> with
#               exactly one blend) and cleans them up. K=0 (no blends) and
#               K=2 (pair-blends) lineages are NOT included unless the
#               target itself is K=0 / K=2. Confirmation prompt asks once
#               for the whole batch (or -y to skip); each per-lineage
#               cleanup runs non-interactively after that single confirm.
#
# Examples:
#   bash my_scripts/dagger_cleanup_lineage.sh \
#       outputs/training/diffusion_..._rerun_v1_b090_ft_dag4
#
#   # Wipe an entire K=1 sweep family in one go:
#   bash my_scripts/dagger_cleanup_lineage.sh \
#       outputs/training/diffusion_..._rerun_v1_b010_ft_dag1 \
#       --detect_siblings

set -euo pipefail

TRAIN_DIR=""
DRY_RUN_FLAG=()
AUTO_CONFIRM=false
DETECT_SIBLINGS=false
ALSO_DELETE_BLENDS_FLAG=()
# --keep_round_1_intervention: tri-state. "explicit_yes" → user passed the
# flag, skip the prompt. "explicit_no" → reserved for future; not currently
# distinguishable from default. "unset" → prompt the user interactively
# (unless -y, in which case treat as no/delete-everything). Set by either the
# explicit flag below or the y/n prompt later in the script.
KEEP_ROUND_1=unset
# The orchestrator's cleanup unconditionally rm -rf's `_nocoll` siblings
# alongside their `_blend<NNN>` parents (rm is idempotent on missing paths),
# so this passthrough is mostly cosmetic — keeps the flag surface symmetric
# across the three scripts. Forwarded to the orchestrator so it ends up in
# the cleanup invocation's recorded argv for auditing.
FILTER_BLEND_COLLISIONS_FLAG=()

for arg in "$@"; do
    case "$arg" in
        --dry-run)   DRY_RUN_FLAG=( --dry-run ) ;;
        -y|--yes)    AUTO_CONFIRM=true ;;
        --detect_siblings) DETECT_SIBLINGS=true ;;
        --also_delete_blends) ALSO_DELETE_BLENDS_FLAG=( --also_delete_blends ) ;;
        --filter_blend_collisions) FILTER_BLEND_COLLISIONS_FLAG=( --filter_blend_collisions ) ;;
        --keep_round_1_intervention) KEEP_ROUND_1=explicit_yes ;;
        -h|--help)
            sed -n '1,/^set -euo pipefail/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        --*) echo "ERROR: unknown flag: $arg" >&2; exit 1 ;;
        *)
            if [[ -n "$TRAIN_DIR" ]]; then
                echo "ERROR: only one positional arg (training dir path) allowed" >&2; exit 1
            fi
            TRAIN_DIR="$arg"
            ;;
    esac
done

if [[ -z "$TRAIN_DIR" ]]; then
    echo "ERROR: training dir path required (run with --help for usage)" >&2; exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Normalize path: accept absolute, relative-to-cwd, or basename-only.
TRAIN_DIR="${TRAIN_DIR%/}"
if [[ ! "$TRAIN_DIR" =~ ^/ ]]; then
    if [[ -d "$TRAIN_DIR" ]]; then
        TRAIN_DIR="$(cd "$TRAIN_DIR" && pwd)"
    elif [[ -d "$LEROBOT_ROOT/outputs/training/$TRAIN_DIR" ]]; then
        TRAIN_DIR="$LEROBOT_ROOT/outputs/training/$TRAIN_DIR"
    fi
fi
if [[ ! -d "$TRAIN_DIR" ]]; then
    echo "ERROR: training dir not found: $TRAIN_DIR" >&2; exit 1
fi

# Prompt the user (once, up-front) whether to preserve round 1's intervention
# recording. Common workflow: "restart all the finetuning from scratch but
# don't waste the expensive human-recorded round-1 intervention". Resolution:
#   --keep_round_1_intervention explicitly set → skip prompt, preserve.
#   -y/--yes set                                → skip prompt, DEFAULT to delete
#       (caller opted into headless; the legacy behavior is delete-all).
#   neither set                                 → prompt y/N (default no, i.e.
#       delete-all, matching the prior behavior of this script).
# The choice propagates to --detect_siblings recursion below.
PRESERVE_R1_FLAG=()
if [[ "$KEEP_ROUND_1" == "explicit_yes" ]]; then
    PRESERVE_R1_FLAG=( --preserve_round_1_intervention )
    echo "[cleanup] --keep_round_1_intervention set: round 1's raw intervention will be PRESERVED."
elif [[ "$AUTO_CONFIRM" == true ]]; then
    : # headless; legacy delete-all behavior
else
    echo
    echo "Round 1's raw intervention recording is the expensive human-in-the-loop data."
    echo "If you preserve it, the lineage restarts from 'round 1 step 4 (merge)' instead"
    echo "of re-recording from scratch. Round 1's merged dataset + training dir + blends"
    echo "and all of rounds 2..N are still wiped either way."
    echo -n "Keep round 1's raw intervention dataset? [y/N]: "
    read -r KEEP_R1_REPLY
    if [[ "$KEEP_R1_REPLY" =~ ^[Yy]$ ]]; then
        PRESERVE_R1_FLAG=( --preserve_round_1_intervention )
        KEEP_ROUND_1=explicit_yes  # propagate to --detect_siblings recursion
        echo "[cleanup] round 1's raw intervention will be PRESERVED."
    else
        echo "[cleanup] round 1's raw intervention WILL be deleted (legacy behavior)."
    fi
fi

# Forward the keep-decision to recursive sibling cleanup invocations so all
# siblings honor the same choice without re-prompting.
KEEP_R1_FORWARD=()
[[ "$KEEP_ROUND_1" == "explicit_yes" ]] && KEEP_R1_FORWARD=( --keep_round_1_intervention )

# Sibling detection: find K-matched sibling lineages on disk that share the
# target's prefix-up-through-run_tag and the same number of blend ratios K.
# Echoes one canonical training-dir path per sibling lineage to stdout.
# Used both by the explicit --detect_siblings mode and by the interactive
# opt-in prompt below (when --detect_siblings was NOT passed but siblings
# exist on disk anyway, so the user can opt in rather than discover them
# the hard way after the fact).
detect_sibling_paths() {
    local training_root="$(dirname "$TRAIN_DIR")"
    local target_basename="$(basename "$TRAIN_DIR")"
    python3 - "$training_root" "$target_basename" <<'PY'
import os, re, sys
from collections import defaultdict

training_root, target_basename = sys.argv[1], sys.argv[2]

# Strip _ft_dag<N> / _dag<N> → lineage name.
m = re.match(r"^(.+?)(?:_ft)?_dag\d+$", target_basename)
if not m:
    sys.exit(f"ERROR: '{target_basename}' is not a recognizable round training dir name")
lineage_name = m.group(1)

# Pull out trailing blends_tag (_b<NNN>(_<NNN>)*) if present.
mb = re.match(r"^(.+?)_b(\d{3}(?:_\d{3})*)$", lineage_name)
if mb:
    prefix = mb.group(1)
    k = mb.group(2).count("_") + 1
else:
    prefix = lineage_name
    k = 0

# Sibling pattern: same prefix, same K, any blend digits.
if k > 0:
    pat = re.compile(
        rf"^{re.escape(prefix)}_b\d{{3}}(?:_\d{{3}}){{{k - 1}}}((?:_ft)?_dag\d+)$"
    )
else:
    # K=0: lineages with no _b suffix and same prefix.
    pat = re.compile(rf"^{re.escape(prefix)}((?:_ft)?_dag\d+)$")

# Group matching round-dir names by lineage; one cleanup invocation per
# lineage is enough (the per-lineage cleanup wipes ALL rounds of that
# lineage, so we just need any one round-dir path to invoke it).
by_lineage = defaultdict(list)
for entry in os.listdir(training_root):
    if not os.path.isdir(os.path.join(training_root, entry)):
        continue
    sm = pat.match(entry)
    if not sm:
        continue
    lin = entry[: -len(sm.group(1))]
    by_lineage[lin].append(entry)

for lin in sorted(by_lineage.keys()):
    rounds = sorted(
        by_lineage[lin],
        key=lambda r: (0 if r.endswith("_ft_dag1") else
                       1 if r.endswith("_dag1")    else
                       2,
                       r),
    )
    print(os.path.join(training_root, rounds[0]))
PY
}

# Interactive sibling opt-in: when --detect_siblings was NOT explicitly
# passed and the script is running interactively (not -y), run the detection
# preemptively. If there are other lineages on disk that match the target's
# prefix + K pattern (so the user is likely operating on one of a sweep
# family), show them and offer to clean them up too. Default is N so the
# script behaves identically to before for users who say no (or who don't
# notice the prompt — though Enter prints "Aborted" so it's hard to miss).
if [[ "$DETECT_SIBLINGS" != true && "$AUTO_CONFIRM" != true ]]; then
    _AUTO_SIBLINGS=()
    while IFS= read -r line; do
        _AUTO_SIBLINGS+=( "$line" )
    done < <(detect_sibling_paths)
    # The detection always includes the target itself (the regex pattern
    # matches it by construction). 1 sibling = just the target = nothing
    # extra to offer. > 1 means there are real siblings worth flagging.
    if (( ${#_AUTO_SIBLINGS[@]} > 1 )); then
        _OTHER_SIBLINGS=()
        _TARGET_BASE="$(basename "$TRAIN_DIR" | sed -E 's/(_ft)?_dag[0-9]+$//')"
        for p in "${_AUTO_SIBLINGS[@]}"; do
            _LIN="$(basename "$p" | sed -E 's/(_ft)?_dag[0-9]+$//')"
            [[ "$_LIN" != "$_TARGET_BASE" ]] && _OTHER_SIBLINGS+=( "$_LIN" )
        done
        echo
        echo "Heads up — found ${#_OTHER_SIBLINGS[@]} other lineage(s) on disk sharing the"
        echo "target's prefix + blend-count (K=$(( ${#_OTHER_SIBLINGS[@]} > 0 ? 1 : 0 )) family):"
        echo "  TARGET: $_TARGET_BASE"
        for s in "${_OTHER_SIBLINGS[@]}"; do
            echo "  SIBLING: $s"
        done
        echo
        echo "By default this script cleans up ONLY the target. If you're cleaning up a"
        echo "sweep family, you can opt in to cleaning the siblings too (equivalent to"
        echo "re-running with --detect_siblings)."
        echo -n "Also clean up the ${#_OTHER_SIBLINGS[@]} sibling(s) above? [y/N]: "
        read -r _SIBLING_REPLY
        if [[ "$_SIBLING_REPLY" =~ ^[Yy]$ ]]; then
            DETECT_SIBLINGS=true
            echo "[cleanup] sibling cleanup ENABLED via interactive opt-in."
        else
            echo "[cleanup] proceeding with target-only cleanup."
        fi
    fi
fi

# Sibling-detection mode: find K-matched sibling lineages on disk (same
# prefix up through run_tag, same number of blend ratios), confirm once,
# then re-invoke this script per-lineage with -y to do the actual cleanup.
# Re-invocation rather than a refactored function keeps the existing
# single-target cleanup path 100% unchanged.
if [[ "$DETECT_SIBLINGS" == true ]]; then
    SIBLING_PATHS=()
    while IFS= read -r line; do
        SIBLING_PATHS+=( "$line" )
    done < <(detect_sibling_paths)

    if (( ${#SIBLING_PATHS[@]} == 0 )); then
        echo "ERROR: --detect_siblings: no sibling lineages found." >&2
        echo "  Expected to find at least the target itself; nothing matched the" >&2
        echo "  same-prefix + same-K pattern derived from:" >&2
        echo "    $TARGET_BASENAME" >&2
        exit 1
    fi

    echo "[detect_siblings] Found ${#SIBLING_PATHS[@]} lineage(s) sharing target's prefix + blend-count K:"
    for p in "${SIBLING_PATHS[@]}"; do
        echo "  $(basename "$p" | sed -E 's/(_ft)?_dag[0-9]+$//')"
    done
    echo

    if (( ${#DRY_RUN_FLAG[@]} == 0 )) && [[ "$AUTO_CONFIRM" != true ]]; then
        echo -n "Type 'delete-all' to confirm cleanup of all ${#SIBLING_PATHS[@]} lineages above: "
        read -r CONFIRM
        [[ "$CONFIRM" == "delete-all" ]] || { echo "Aborted."; exit 1; }
    fi

    # Recurse: per-lineage cleanup. -y at the inner level suppresses each
    # sibling's own confirmation prompt since the user already confirmed
    # the whole batch above. Don't let `set -e` abort the loop on a single
    # failure — keep going so a downstream sibling that still has artifacts
    # gets a chance to be cleaned up too.
    overall_rc=0
    for p in "${SIBLING_PATHS[@]}"; do
        echo
        echo "=== [detect_siblings] cleaning up: $(basename "$p" | sed -E 's/(_ft)?_dag[0-9]+$//') ==="
        bash "$0" "$p" -y "${DRY_RUN_FLAG[@]}" "${ALSO_DELETE_BLENDS_FLAG[@]}" "${FILTER_BLEND_COLLISIONS_FLAG[@]}" "${KEEP_R1_FORWARD[@]}" && rc=0 || rc=$?
        if (( rc != 0 )); then
            overall_rc="$rc"
            echo "[detect_siblings] WARN: cleanup failed for $p (rc=$rc); continuing." >&2
        fi
    done
    exit "$overall_rc"
fi

ORIG_ARGV=()
CFG="$TRAIN_DIR/dagger/config.json"
if [[ -f "$CFG" ]]; then
    # Resolution path 1: sidecar exists → use recorded argv (exact).
    mapfile -t ORIG_ARGV < <(python3 -c "
import json, sys
cfg = json.load(open(sys.argv[1]))
for a in cfg.get('orchestrator_invocation', {}).get('argv', []):
    print(a)
" "$CFG")
    if (( ${#ORIG_ARGV[@]} == 0 )); then
        echo "[cleanup] $CFG exists but has no orchestrator_invocation.argv; falling back to auto-detect." >&2
    else
        echo "[cleanup] reusing original argv from sidecar: $CFG"
    fi
fi

if (( ${#ORIG_ARGV[@]} == 0 )); then
    # Resolution path 2: auto-detect from training-dir basename + disk scan.
    # This handles lineages that predate the sidecar-writing change.
    echo "[cleanup] no sidecar argv available; auto-detecting from training dir name + disk..."
    TRAIN_BASENAME="$(basename "$TRAIN_DIR")"

    # Retrain-variant fast path: training dirs from `--retrain_round=N
    # --retrain_suffix=SUFFIX` are named `<lineage>{_ft,}_dag<N>_<SUFFIX>` and
    # share their dataset/stats artifacts with the canonical round's training
    # dir. The variant's ONLY unique artifact is the training dir itself, so
    # the cleanup is just `rm -rf <train_dir>` (no orchestrator invocation,
    # which would also try to nuke the shared datasets).
    if [[ "$TRAIN_BASENAME" =~ _dag[0-9]+_(.+)$ ]]; then
        RETRAIN_SUFFIX="${BASH_REMATCH[1]}"
        echo "[cleanup] detected --retrain_round variant (suffix '$RETRAIN_SUFFIX')."
        echo "  Variants share datasets/stats with the canonical round, so cleanup"
        echo "  is just rm -rf on the variant's training dir."
        echo
        echo "Will DELETE:"
        echo "  $TRAIN_DIR"
        echo
        if (( ${#DRY_RUN_FLAG[@]} > 0 )); then
            echo "[--dry-run] would rm -rf the path above."
            exit 0
        fi
        if [[ "$AUTO_CONFIRM" != true ]]; then
            echo -n "Type 'delete' to confirm: "
            read -r CONFIRM
            [[ "$CONFIRM" == "delete" ]] || { echo "Aborted."; exit 1; }
        fi
        rm -rf "$TRAIN_DIR"
        echo "Deleted."
        exit 0
    fi

    # Strip _ft_dag<N> / _dag<N> → BASE_POLICY_NAME for the normal-lineage path.
    BASE_POLICY_NAME=$(echo "$TRAIN_BASENAME" | sed -E 's/(_ft)?_dag[0-9]+$//')
    if [[ "$BASE_POLICY_NAME" == "$TRAIN_BASENAME" ]]; then
        echo "ERROR: '$TRAIN_BASENAME' doesn't look like a DAgger round training dir." >&2
        echo "  Expected basename ending in '_ft_dag<N>' or '_dag<N>'." >&2
        exit 1
    fi

    # Model prefix → --model flag.
    DET_MODEL=""
    for pfx_model in "pi05:pi" "diffusion:diff" "act:act"; do
        pfx="${pfx_model%:*}"; mdl="${pfx_model#*:}"
        if [[ "$BASE_POLICY_NAME" == "${pfx}_"* ]]; then
            MODEL_PREFIX="$pfx"; DET_MODEL="$mdl"; break
        fi
    done
    if [[ -z "$DET_MODEL" ]]; then
        echo "ERROR: could not detect model prefix (pi05/diffusion/act) in $BASE_POLICY_NAME." >&2; exit 1
    fi

    # Action format from _delta_basewrist / _abs_basewrist.
    if [[ "$BASE_POLICY_NAME" == *"_delta_basewrist"* ]]; then
        DET_ACTION_FORMAT="rel"; ACTION_INFIX="r"; ACTION_TAG="delta"
    elif [[ "$BASE_POLICY_NAME" == *"_abs_basewrist"* ]]; then
        DET_ACTION_FORMAT="abs"; ACTION_INFIX="a"; ACTION_TAG="abs"
    else
        echo "ERROR: BASE_POLICY_NAME='$BASE_POLICY_NAME' has no _delta_basewrist or _abs_basewrist segment." >&2; exit 1
    fi

    # Split BASE_POLICY_NAME → BASE_POLICY_STEM + LINEAGE_TAGS.
    # BASE_POLICY_STEM = `<model>_<base_short>_<action>_basewrist` (the round-0
    # base policy dir's name); LINEAGE_TAGS = everything after `_basewrist_`,
    # or empty if the lineage was trained without --run_tag (the round-N
    # training dirs are then named `<stem>_ft_dag<N>` with no tag in between).
    if [[ "$BASE_POLICY_NAME" =~ ^(${MODEL_PREFIX}_.+_${ACTION_TAG}_basewrist)_(.+)$ ]]; then
        BASE_POLICY_STEM="${BASH_REMATCH[1]}"
        LINEAGE_TAGS="${BASH_REMATCH[2]}"
    elif [[ "$BASE_POLICY_NAME" =~ ^${MODEL_PREFIX}_.+_${ACTION_TAG}_basewrist$ ]]; then
        BASE_POLICY_STEM="$BASE_POLICY_NAME"
        LINEAGE_TAGS=""
    else
        echo "ERROR: could not parse BASE_POLICY_NAME='$BASE_POLICY_NAME'." >&2
        echo "  Expected name like '<model_prefix>_<base_short>_<delta|abs>_basewrist[_<tags>]'." >&2
        exit 1
    fi
    DET_BASE_SHORT=$(echo "$BASE_POLICY_STEM" | sed -E "s/^${MODEL_PREFIX}_//; s/_${ACTION_TAG}_basewrist$//")

    # --initial_policy_path: just point at the base policy training dir. The
    # orchestrator only uses this to derive BASE_POLICY_STEM (which we
    # already know matches); it doesn't actually need to exist for cleanup.
    DET_INITIAL_POLICY="$LEROBOT_ROOT/outputs/training/$BASE_POLICY_STEM"

    # --intermediate_mode: finetune if _ft_dag* dirs exist, else scratch.
    DET_INTERMEDIATE_MODE="finetune"
    if ! ls -d "$LEROBOT_ROOT/outputs/training/${BASE_POLICY_NAME}"_ft_dag[0-9]* >/dev/null 2>&1; then
        DET_INTERMEDIATE_MODE="scratch"
    fi

    # --num_rounds: max <N> across all matching training dirs.
    DET_NUM_ROUNDS=0
    for d in "$LEROBOT_ROOT/outputs/training/${BASE_POLICY_NAME}"_dag[0-9]* \
             "$LEROBOT_ROOT/outputs/training/${BASE_POLICY_NAME}"_ft_dag[0-9]*; do
        [[ -d "$d" ]] || continue
        n=$(basename "$d" | grep -oE 'dag[0-9]+' | head -1 | grep -oE '[0-9]+')
        (( n > DET_NUM_ROUNDS )) && DET_NUM_ROUNDS=$n
    done
    if (( DET_NUM_ROUNDS == 0 )); then
        echo "ERROR: no per-round training dirs found matching ${BASE_POLICY_NAME}{_ft,}_dag*." >&2; exit 1
    fi

    # --dag_short_override: derive from disk if BASE_DATASET_SHORT (as used
    # by the dataset names) differs from the default DET_BASE_SHORT + tags.
    # Scan HF cache for any path matching `*_<LINEAGE_TAGS>_<a|r>_dag*` AND
    # `*_<LINEAGE_TAGS>_<MODEL_TAG>_<a|r>_dag*` (for diffusion's _diff
    # insertion). Take the longest common prefix that ends with LINEAGE_TAGS.
    LEROBOT_CACHE="${LEROBOT_CACHE:-$HOME/.cache/huggingface/lerobot}"
    HF_USER="${HF_USER:-JennyWWW}"
    case "$DET_MODEL" in
        pi)   MODEL_TAG="" ;;
        diff) MODEL_TAG="diff" ;;
        act)  MODEL_TAG="act" ;;
    esac
    EXPECTED_DATASET_TAIL=""
    [[ -n "$LINEAGE_TAGS" ]] && EXPECTED_DATASET_TAIL="_${LINEAGE_TAGS}"
    [[ -n "$MODEL_TAG" ]] && EXPECTED_DATASET_TAIL="${EXPECTED_DATASET_TAIL}_${MODEL_TAG}"
    EXPECTED_DATASET_TAIL="${EXPECTED_DATASET_TAIL}_${ACTION_INFIX}_dag"
    STATS_BASE="$LEROBOT_ROOT/outputs/dataset_stats"
    sample_match=""
    # First pass: try the exact "no --dag_short_override" pattern. This is
    # both fast and unambiguous — for untagged lineages (LINEAGE_TAGS="")
    # it's the only sound scan since a loose `*_r_dag*` glob would match
    # every rel-mode lineage on disk.
    tag_suffix="${EXPECTED_DATASET_TAIL%_${ACTION_INFIX}_dag}"   # e.g. "_d30jvm" / "" / "_d5jvm_diff"
    exact_no_override="${DET_BASE_SHORT}${tag_suffix}_${ACTION_INFIX}_dag"
    for d in "$LEROBOT_CACHE/$HF_USER"/${exact_no_override}[0-9]* \
             "$STATS_BASE"/${exact_no_override}[0-9]*; do
        [[ -d "$d" ]] || continue
        sample_match="$d"; break
    done
    # Second pass: only safe when LINEAGE_TAGS is non-empty (so the loose
    # `*_<LINEAGE_TAGS>_<a|r>_dag*` glob uniquely identifies the lineage).
    # This catches --dag_short_override cases.
    if [[ -z "$sample_match" && -n "$LINEAGE_TAGS" ]]; then
        for d in "$LEROBOT_CACHE/$HF_USER"/*"$EXPECTED_DATASET_TAIL"[0-9]* \
                 "$STATS_BASE"/*"$EXPECTED_DATASET_TAIL"[0-9]*; do
            [[ -d "$d" ]] || continue
            sample_match="$d"; break
        done
    fi

    DET_DAG_SHORT_OVERRIDE=""
    if [[ -n "$sample_match" ]]; then
        sample_basename=$(basename "$sample_match")
        # Recover BASE_DATASET_SHORT by stripping the `_<a|r>_dag<N>...` tail.
        det_base_dataset_short=$(echo "$sample_basename" | sed -E "s/_${ACTION_INFIX}_dag[0-9]+.*\$//")
        # Expected dataset short WITHOUT --dag_short_override:
        # `<BASE_SHORT>_<run_tag>[_<model_tag>]` (= BASE_SHORT + tag-tail).
        # Tag-tail = part of EXPECTED_DATASET_TAIL before `_<a|r>_dag`.
        tag_suffix="${EXPECTED_DATASET_TAIL%_${ACTION_INFIX}_dag}"   # e.g. "_d30jvm" or "_d5jvm_diff"
        expected_no_override="${DET_BASE_SHORT}${tag_suffix}"
        if [[ "$det_base_dataset_short" != "$expected_no_override" ]]; then
            # User used --dag_short_override. Recover the override value by
            # stripping the tag-tail from the detected dataset short.
            DET_DAG_SHORT_OVERRIDE="${det_base_dataset_short%${tag_suffix}}"
        fi
    fi

    # --run_tag: this part of LINEAGE_TAGS we want to forward. For non-blend,
    # non-method-tagged lineages, run_tag == LINEAGE_TAGS. For blend lineages,
    # LINEAGE_TAGS ends with `_b<NNN>[_<NNN>...]` which the orchestrator
    # rebuilds from --blends. Detect blend suffix and strip.
    DET_RUN_TAG="$LINEAGE_TAGS"
    DET_BLENDS=""
    if [[ "$LINEAGE_TAGS" =~ ^(.+)_b([0-9]{3}(_[0-9]{3})*)$ ]]; then
        DET_RUN_TAG="${BASH_REMATCH[1]}"
        # Convert tag like "090_080" → blends "0.9 0.8". Delegate ratio
        # parsing to dagger_naming.ratio_for_blend_tag so the bash cleanup
        # script + Python naming module agree byte-for-byte on the
        # tag↔ratio round-trip.
        IFS='_' read -ra _bparts <<< "${BASH_REMATCH[2]}"
        _blends_arr=()
        for p in "${_bparts[@]}"; do
            _blends_arr+=( "$(python3 "$SCRIPT_DIR/dagger_naming.py" blend_ratio --tag="$p")" )
        done
        DET_BLENDS="${_blends_arr[*]}"
    fi

    # Assemble argv. Use minimal placeholders for flags the orchestrator
    # requires but cleanup doesn't care about (finetune_steps, env port).
    ORIG_ARGV=(
        --base_short="$DET_BASE_SHORT"
        --num_rounds="$DET_NUM_ROUNDS"
        --initial_policy_path="$DET_INITIAL_POLICY"
        --model="$DET_MODEL"
        --action_format="$DET_ACTION_FORMAT"
        --intermediate_mode="$DET_INTERMEDIATE_MODE"
        --target_intervention_volume=10
        --finetune_steps=1000
        --env_external_port=6001
        --skip_alias_step
    )
    [[ -n "$DET_DAG_SHORT_OVERRIDE" ]] && ORIG_ARGV+=( --dag_short_override="$DET_DAG_SHORT_OVERRIDE" )
    [[ -n "$DET_RUN_TAG" ]]            && ORIG_ARGV+=( --run_tag="$DET_RUN_TAG" )
    [[ -n "$DET_BLENDS" ]]             && ORIG_ARGV+=( --blends="$DET_BLENDS" )

    echo "[cleanup] auto-detected argv:"
    printf '  %s\n' "${ORIG_ARGV[@]}"
fi

# Strip flags that shouldn't be forwarded for a cleanup invocation:
#   --force_restart  → we're adding it; don't double up
#   --cleanup_only   → same
#   --resume         → cleanup doesn't resume anything
#   --dry-run        → controlled by THIS wrapper's --dry-run
#   --intervention_oversample → migrated to --target_intervention_volume.
#     Older sidecars (lineages trained before the flag rename) carry the
#     old name in their recorded argv. Cleanup doesn't care about the
#     actual value — just needs the orchestrator's startup validation to
#     pass — so drop the old flag and inject a placeholder N=10 below.
FILTERED_ARGV=()
HAS_TARGET_VOLUME=false
for a in "${ORIG_ARGV[@]}"; do
    case "$a" in
        --force_restart|--cleanup_only|--resume|--dry-run) continue ;;
        --intervention_oversample=*) continue ;;
        --target_intervention_volume=*) HAS_TARGET_VOLUME=true; FILTERED_ARGV+=( "$a" ) ;;
        *) FILTERED_ARGV+=( "$a" ) ;;
    esac
done
if [[ "$HAS_TARGET_VOLUME" != true ]]; then
    FILTERED_ARGV+=( --target_intervention_volume=10 )
fi

ORCH="$SCRIPT_DIR/dagger_orchestrate.sh"
echo "[cleanup] invoking:"
echo "  bash $ORCH ${FILTERED_ARGV[*]} --force_restart --cleanup_only ${ALSO_DELETE_BLENDS_FLAG[*]} ${FILTER_BLEND_COLLISIONS_FLAG[*]} ${PRESERVE_R1_FLAG[*]} ${DRY_RUN_FLAG[*]}"
echo

if [[ "$AUTO_CONFIRM" == true ]]; then
    # Pipe the confirmation token. dagger_orchestrate.sh's --force_restart
    # block reads exactly one line, so a single "restart\n" is enough.
    printf 'restart\n' | bash "$ORCH" "${FILTERED_ARGV[@]}" --force_restart --cleanup_only "${ALSO_DELETE_BLENDS_FLAG[@]}" "${FILTER_BLEND_COLLISIONS_FLAG[@]}" "${PRESERVE_R1_FLAG[@]}" "${DRY_RUN_FLAG[@]}"
else
    bash "$ORCH" "${FILTERED_ARGV[@]}" --force_restart --cleanup_only "${ALSO_DELETE_BLENDS_FLAG[@]}" "${FILTER_BLEND_COLLISIONS_FLAG[@]}" "${PRESERVE_R1_FLAG[@]}" "${DRY_RUN_FLAG[@]}"
fi
