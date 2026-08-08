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
#       [--dry-run] [-y|--yes] [--detect_siblings] \
#       [--delete_episodes='[N1,N2,...]'] [--skip_dataset_edit]
#
# Options:
#   --dry-run   Pass --dry-run through to dagger_orchestrate.sh (lists what
#               would be deleted, doesn't rm).
#   -y, --yes   Pipe "restart" into the orchestrator's confirmation prompt so
#               the deletion runs unattended. Without this, the orchestrator
#               will prompt for confirmation interactively.
#   --delete_episodes='[N1,N2,...]'
#               SURGICAL episode-level cleanup. Removes the listed episode
#               indices from round R's intervention dataset (R = the round
#               number parsed from the training dir name), refreshes the
#               round's rel-action stats sidecar, then rm -rf's training
#               dirs and blend datasets for rounds R..NUM_ROUNDS so they
#               retrain on the cleaned data. The intervention dataset
#               itself is PRESERVED — only the listed episodes are removed
#               in place. After running, re-invoke dagger_orchestrate.sh
#               (or dagger_orchestrate_sweep.sh) with --resume to retrain
#               rounds R..NUM_ROUNDS against the cleaned dataset.
#               Indices are LeRobot dataset episode_index values
#               (0-indexed). Example after dagger_detect_dataset_anomalies
#               flagged bad episodes in round 2:
#                 --delete_episodes='[0, 5, 9, 10, 13, 17, 22, 27, 29, 30, 34, 36, 38]'
#               Bypasses the orchestrator entirely (deletions run inline).
#               Skips the keep-round-1-intervention prompt (round-1
#               intervention is touched only if the target round IS 1).
#               Composes with --detect_siblings: the sibling recursion
#               auto-adds --skip_dataset_edit so the dataset edit runs
#               exactly once (the first sibling) but every sibling's
#               downstream training dirs / blends still get nuked.
#   --skip_dataset_edit
#               In --delete_episodes mode, SKIP the lerobot-edit-dataset
#               call and the stats-sidecar refresh — only nuke training
#               dirs and blend datasets for rounds R..NUM_ROUNDS. Used
#               internally by --detect_siblings recursion when multiple
#               rerun-sibling lineages share the same source intervention
#               dataset (the edit must run once, not per-sibling). Safe
#               to pass manually too if the dataset has already been
#               cleaned by another invocation.
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
#   --blends_only
#               SURGICAL blend cleanup: delete ONLY the blend-derived
#               artifacts of this lineage, preserving the intervention
#               recordings (and, in rerun mode, everything source-owned).
#               Specifically deletes:
#                 - `_blend<NNN>` raw blend datasets + stats sidecars
#                 - `_blend<NNN>_nocoll` siblings + stats sidecars
#                 - `dagger/blend_collision_filter/` audit subdirs
#                 - the lineage's merged datasets (`*_m`) + sidecars
#                   (stale once blends change; absent in weighted mode)
#                 - the lineage's per-round training dirs (`_dag<N>`,
#                   `_ft_dag<N>`, `_ft_dag<N>_nc`, ...) + their
#                   outputs/dagger eval dirs (trained on the blends →
#                   stale once blends are regenerated)
#               PRESERVED: intervention datasets + alias datasets +
#               int-stats sidecars, the round-0 base policy, and (rerun
#               mode) the entire source lineage. Re-running the
#               orchestrator/sweep with --resume then re-blends (step 2)
#               and re-trains (step 6) only.
#               In rerun mode, blend datasets are SHARED across sibling
#               reruns requesting the same ratio at the same source round
#               — deleting them here means the next sibling sweep run
#               re-blends once, then re-uses the result.
#               Incompatible with --nc_only / --delete_episodes /
#               --from_round / --also_delete_blends. Skips the
#               keep-round-1-intervention prompt (interventions untouched).
#   --nc_only
#               SURGICAL collision-filter cleanup: delete ONLY the
#               artifacts produced by --filter_blend_collisions (step 2's
#               filter sub-step and step 6b). Specifically:
#                 - `_ft_dag<N>_nc` training dirs
#                 - `dagger/blend_collision_filter/` audit subdirs under
#                   each raw round's training dir
#                 - `_blend<NNN>_nocoll` blend datasets in the HF cache
#                 - `_blend<NNN>_nocoll` stats sidecars
#               Raw policies, raw blends, intervention datasets, alias /
#               merged datasets, and round-0 base policy are PRESERVED.
#               Re-running dagger_orchestrate_sweep.sh with
#               --filter_blend_collisions then triggers step 2's filter
#               sub-step (re-creates `_nocoll` datasets) and step 6b
#               (re-trains `_nc` policies).
#               In rerun mode, `_nocoll` blend datasets are SHARED across
#               reruns of the same source — deleting them for one rerun
#               cascades to all sibling reruns' next sweep run (each
#               re-filters once, then re-uses the result).
#               Skips the keep-round-1-intervention prompt (R1 isn't
#               touched). Skips the orchestrator delegation entirely
#               (deletions run inline). Composes with --detect_siblings.
#   --keep_round_1_intervention
#               PRESERVE the first-cleaned round's raw intervention dataset +
#               alias + int-stats sidecar. The first cleaned round is round 1
#               by default, or --from_round=N when that's set (so this then
#               preserves round N's intervention). That round's merged dataset
#               + training dir, all blends, and every later-round artifact are
#               still wiped. Useful for "restart the finetuning from scratch
#               but keep the expensive human recording" workflows. Without this
#               flag (and without -y), the script prompts y/n interactively.
#               With -y but without this flag, defaults to DELETE the round's
#               intervention (the legacy behavior). No-op in rerun mode.
#   --from_round=N
#               Restrict cleanup to rounds N..NUM_ROUNDS, leaving rounds
#               1..(N-1) intact. Use when a data anomaly is detected in
#               round N (e.g. via dagger_detect_dataset_anomalies.py
#               flagging a polluted intervention dataset) and you want to
#               re-record from round N onwards without losing the clean
#               1..(N-1) artifacts. Round (N-1)'s trained policy survives
#               and becomes the branching point for the next round-N
#               recording. Default N=1 (= wipe all rounds — the historical
#               behavior). Errors if N > NUM_ROUNDS. Forwarded as-is to
#               dagger_orchestrate.sh's identically-named flag, which
#               narrows the for-loops in both restart_from_scratch() and
#               the --force_restart path. Composes with --detect_siblings
#               (every sibling lineage's cleanup honors the same N).
#   --detect_siblings
#               Auto-detect related lineages on disk and clean them up too.
#               Two flavors are detected and unioned:
#                 - K-siblings: lineages sharing the target's prefix (through
#                   the run_tag) AND the target's blend-count K. Useful for
#                   cleaning up an entire K-fold sweep family in one command.
#                   Example: given a single 1-blend lineage `rerun_v1_b010`,
#                   detects all of {b010, b030, b050, b070, b090} (any
#                   rerun_v1_b<NNN> with exactly one blend) and cleans them
#                   up. K=0 (no blends) and K=2 (pair-blends) lineages are
#                   NOT included unless the target itself is K=0 / K=2.
#                 - Rerun children: lineages whose dagger/config.json sidecar
#                   `rerun_mode.source_run_tag` (+ `source_blends_tag`)
#                   matches the TARGET's run_tag (+ blends_tag). Catches the
#                   case where the target is a BASE lineage and rerun
#                   lineages were spawned from its intervention data —
#                   they have a different prefix from the target so the
#                   K-sibling regex doesn't catch them, but they're orphaned
#                   by the target's deletion (no source interventions to
#                   blend against) and worth cleaning up alongside.
#               Confirmation prompt asks once for the whole batch (or -y to
#               skip); each per-lineage cleanup runs non-interactively after
#               that single confirm.
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
# --nc_only: surgical mode — wipe only step-6b/filter artifacts (nc training
# dirs, nocoll blend datasets + stats, blend_collision_filter audit subdirs).
# Raw policies and raw blends are preserved. See the docstring above.
NC_ONLY=false
# --blends_only: surgical mode — wipe blend-derived artifacts (raw blends,
# nocoll siblings, both sidecars, audit dirs, merged datasets, per-round
# training dirs) while preserving interventions. See the docstring above.
BLENDS_ONLY=false
# --from_round=N: forwarded to dagger_orchestrate.sh's --from_round (the
# orchestrator owns the per-round cleanup loop, so it owns the bounds).
# Validated downstream; here we just pass through. Default empty = no
# restriction (orchestrator treats it as N=1, the historical behavior).
FROM_ROUND=""
# --delete_episodes='[N1,N2,...]': surgical episode-level cleanup. Triggers
# the inline (orchestrator-bypassing) delete-episodes mode below. Empty =
# regular whole-round cleanup. Format: JSON-style list of ints, e.g.
# '[0, 5, 9, 13]'. Validated as JSON before use.
DELETE_EPISODES=""
SKIP_DATASET_EDIT=false

for arg in "$@"; do
    case "$arg" in
        --dry-run)   DRY_RUN_FLAG=( --dry-run ) ;;
        -y|--yes)    AUTO_CONFIRM=true ;;
        --detect_siblings) DETECT_SIBLINGS=true ;;
        --also_delete_blends) ALSO_DELETE_BLENDS_FLAG=( --also_delete_blends ) ;;
        --filter_blend_collisions) FILTER_BLEND_COLLISIONS_FLAG=( --filter_blend_collisions ) ;;
        --nc_only)   NC_ONLY=true ;;
        --blends_only) BLENDS_ONLY=true ;;
        --from_round=*) FROM_ROUND="${arg#*=}" ;;
        --delete_episodes=*) DELETE_EPISODES="${arg#*=}" ;;
        --skip_dataset_edit) SKIP_DATASET_EDIT=true ;;
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

# --blends_only incompatibilities. It is its own surgical scope; combining it
# with the other scoping flags would be ambiguous about what gets deleted.
if [[ "$BLENDS_ONLY" == true ]]; then
    if [[ "$NC_ONLY" == true ]]; then
        echo "ERROR: --blends_only and --nc_only are mutually exclusive (blends_only is a superset of nc_only's scope)." >&2; exit 1
    fi
    if [[ -n "$DELETE_EPISODES" ]]; then
        echo "ERROR: --delete_episodes is incompatible with --blends_only (they target different artifact categories)." >&2; exit 1
    fi
    if [[ -n "$FROM_ROUND" ]]; then
        echo "ERROR: --from_round is incompatible with --blends_only (blends-only mode wipes all rounds' blend artifacts unconditionally)." >&2; exit 1
    fi
    if (( ${#ALSO_DELETE_BLENDS_FLAG[@]} > 0 )); then
        echo "ERROR: --also_delete_blends is redundant with --blends_only (blends are the whole scope); drop it." >&2; exit 1
    fi
fi

# --from_round validation. Lower-bound only; the orchestrator does the
# NUM_ROUNDS upper-bound check after lineage auto-detection.
if [[ -n "$FROM_ROUND" ]]; then
    if ! [[ "$FROM_ROUND" =~ ^[0-9]+$ ]] || (( FROM_ROUND < 1 )); then
        echo "ERROR: --from_round=$FROM_ROUND must be a positive integer >= 1" >&2; exit 1
    fi
    if [[ "$NC_ONLY" == true ]]; then
        echo "ERROR: --from_round is incompatible with --nc_only (nc-only mode wipes all rounds' _nc artifacts unconditionally)." >&2; exit 1
    fi
fi
FROM_ROUND_FLAG=()
[[ -n "$FROM_ROUND" ]] && FROM_ROUND_FLAG=( "--from_round=$FROM_ROUND" )

# --delete_episodes validation. Must be parseable as a JSON array of
# non-negative integers. Done up-front (before any cleanup runs) so a
# typo fails immediately instead of partway through deletion.
if [[ -n "$DELETE_EPISODES" ]]; then
    if [[ "$NC_ONLY" == true ]]; then
        echo "ERROR: --delete_episodes is incompatible with --nc_only (they target different artifact categories)." >&2; exit 1
    fi
    if [[ -n "$FROM_ROUND" ]]; then
        echo "ERROR: --delete_episodes is incompatible with --from_round (the round number is inferred from the training dir name)." >&2; exit 1
    fi
    # Normalize + validate (allow spaces, trailing commas, etc.). Echoes
    # the canonicalized "[N1, N2, ...]" form on success; non-zero exit on
    # parse failure or non-int / negative elements.
    DELETE_EPISODES_CANON="$(python3 -c "
import json, sys
raw = sys.argv[1]
try:
    eps = json.loads(raw)
except json.JSONDecodeError as e:
    sys.exit(f'ERROR: --delete_episodes is not valid JSON: {e}')
if not isinstance(eps, list):
    sys.exit('ERROR: --delete_episodes must be a JSON list (e.g. [0, 5, 9]).')
if not eps:
    sys.exit('ERROR: --delete_episodes must contain at least one episode index.')
clean = []
for x in eps:
    if not isinstance(x, int) or isinstance(x, bool) or x < 0:
        sys.exit(f'ERROR: each --delete_episodes element must be a non-negative int; got {x!r}.')
    clean.append(x)
clean = sorted(set(clean))
print(json.dumps(clean))
" "$DELETE_EPISODES")" || exit 1
    DELETE_EPISODES="$DELETE_EPISODES_CANON"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=lib_dagger_lineage.sh
source "$SCRIPT_DIR/lib_dagger_lineage.sh"

# Normalize path: accept absolute, relative-to-cwd, "outputs/training/<basename>"
# relative to the repo, or a bare basename (looked up under outputs/training/).
# In --delete_episodes mode the target round's training dir may not exist yet
# (intervention recorded, training not started) — DGR_ALLOW_MISSING makes the
# helper return a plausibly-absolute path anyway so dirname/basename below
# stay stable.
if [[ -n "$DELETE_EPISODES" ]]; then
    TRAIN_DIR="$(DGR_ALLOW_MISSING=true dgr_normalize_train_dir "$TRAIN_DIR")" || exit 1
    if [[ ! -d "$TRAIN_DIR" ]]; then
        echo "[cleanup] note: training dir does not exist yet (round not trained):"
        echo "  $TRAIN_DIR"
        echo "[cleanup] --delete_episodes continues against the round's intervention dataset."
    fi
else
    TRAIN_DIR="$(dgr_normalize_train_dir "$TRAIN_DIR")" || exit 1
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
#
# The round being asked about is the FIRST round cleanup will touch: round 1
# by default, or --from_round=N when that's set (cleanup spans rounds N..end,
# so round N's intervention is the first one at risk — rounds 1..(N-1) are
# untouched regardless). Preserving means that round restarts from its merge
# step instead of re-recording; deleting means re-record it from scratch.
KEEP_ROUND_NUM="${FROM_ROUND:-1}"
PRESERVE_R1_FLAG=()
# Rerun lineages never own intervention data (it's source-owned and the
# orchestrator's rerun-mode cleanup never touches it), so the
# keep-intervention prompt below would be a no-op question — skip it.
_target_is_rerun() {
    local _cfg="$TRAIN_DIR/dagger/config.json"
    [[ -f "$_cfg" ]] || return 1
    python3 -c "
import json, sys
cfg = json.load(open(sys.argv[1]))
sys.exit(0 if cfg.get('rerun_mode') else 1)
" "$_cfg" 2>/dev/null
}
if [[ "$NC_ONLY" == true ]]; then
    : # nc-only mode doesn't touch intervention data; the prompt is irrelevant.
elif [[ "$BLENDS_ONLY" == true ]]; then
    : # blends-only mode doesn't touch intervention data; the prompt is irrelevant.
elif _target_is_rerun; then
    echo "[cleanup] rerun lineage (sidecar has rerun_mode): intervention data is source-owned"
    echo "[cleanup] and never deleted by cleanup — skipping the keep-intervention prompt."
elif [[ -n "$DELETE_EPISODES" ]]; then
    : # surgical delete_episodes mode operates on a specific round; the
      # keep-intervention prompt is only meaningful for whole-lineage cleanups.
elif [[ "$KEEP_ROUND_1" == "explicit_yes" ]]; then
    PRESERVE_R1_FLAG=( --preserve_round_1_intervention )
    echo "[cleanup] --keep_round_1_intervention set: round $KEEP_ROUND_NUM's raw intervention will be PRESERVED."
elif [[ "$AUTO_CONFIRM" == true ]]; then
    : # headless; legacy delete-all behavior
else
    echo
    echo "Round $KEEP_ROUND_NUM's raw intervention recording is the expensive human-in-the-loop data."
    echo "If you preserve it, the lineage restarts from 'round $KEEP_ROUND_NUM step 4 (merge)' instead"
    echo "of re-recording from scratch. Round $KEEP_ROUND_NUM's merged dataset + training dir + blends"
    echo "and all later rounds are still wiped either way."
    echo -n "Keep round $KEEP_ROUND_NUM's raw intervention dataset? [y/N]: "
    read -r KEEP_R1_REPLY
    if [[ "$KEEP_R1_REPLY" =~ ^[Yy]$ ]]; then
        PRESERVE_R1_FLAG=( --preserve_round_1_intervention )
        KEEP_ROUND_1=explicit_yes  # propagate to --detect_siblings recursion
        echo "[cleanup] round $KEEP_ROUND_NUM's raw intervention will be PRESERVED."
    else
        echo "[cleanup] round $KEEP_ROUND_NUM's raw intervention WILL be deleted (legacy behavior)."
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

# Strip _ft_dag<N> / _dag<N> (with an optional trailing retrain-suffix like
# `_nc` from step 6b or `_<custom>` from --retrain_suffix) → lineage name.
m = re.match(r"^(.+?)(?:_ft)?_dag\d+(?:_[A-Za-z][A-Za-z0-9_]*)?$", target_basename)
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

# Sibling pattern: same prefix, same K, any blend digits. Trailing
# `_<suffix>` (e.g. `_nc` from step 6b, `_<retrain_suffix>`) is optional.
# Required to detect siblings when the user is targeting an `_nc` round
# AND the raw (non-_nc) sibling dirs aren't on disk (e.g. lineages built
# with `--filter_blend_collisions` where only _nc training dirs persist).
_SUFFIX = r"(?:_[A-Za-z][A-Za-z0-9_]*)?"
if k > 0:
    pat = re.compile(
        rf"^{re.escape(prefix)}_b\d{{3}}(?:_\d{{3}}){{{k - 1}}}((?:_ft)?_dag\d+{_SUFFIX})$"
    )
else:
    # K=0: lineages with no _b suffix and same prefix.
    pat = re.compile(rf"^{re.escape(prefix)}((?:_ft)?_dag\d+{_SUFFIX})$")

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

# Read the target's own run_tag + blends_tag from its sidecar (if present).
# Used by detect_rerun_child_paths to match other lineages whose sidecars
# point at THIS target as their rerun source. No sidecar → can't match,
# rerun-child detection is silently skipped (pre-sidecar lineages).
_TARGET_RUN_TAG=""
_TARGET_BLENDS_TAG=""
if [[ -f "$TRAIN_DIR/dagger/config.json" ]]; then
    _tmp_naming="$(python3 -c "
import json, sys
try:
    cfg = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)
naming = cfg.get('naming') or {}
print(naming.get('run_tag') or '')
print(naming.get('blends_tag') or '')
" "$TRAIN_DIR/dagger/config.json" 2>/dev/null || true)"
    _TARGET_RUN_TAG="$(printf '%s\n' "$_tmp_naming" | sed -n 1p)"
    _TARGET_BLENDS_TAG="$(printf '%s\n' "$_tmp_naming" | sed -n 2p)"
fi

# Rerun-child detection: find lineages on disk whose sidecar
# `rerun_mode.source_run_tag` (+ `source_blends_tag`) points at THIS target.
# Catches the case where the target is a BASE lineage and one or more rerun
# lineages (e.g. `_rr_b010 / _rr_b050 / _rr_b090`) were spawned from its
# intervention data — those rerun lineages have a different prefix from the
# target so K-sibling detection doesn't catch them, but they're orphaned by
# the target's deletion (no source interventions to blend against) so they're
# worth offering for cleanup. Echoes one canonical training-dir path per
# rerun-child lineage to stdout. Empty output when target has no sidecar.
detect_rerun_child_paths() {
    [[ -z "$_TARGET_RUN_TAG" ]] && return 0
    local training_root="$(dirname "$TRAIN_DIR")"
    local target_basename="$(basename "$TRAIN_DIR")"
    python3 - "$training_root" "$target_basename" "$_TARGET_RUN_TAG" "$_TARGET_BLENDS_TAG" <<'PY'
import json, os, re, sys
from collections import defaultdict

training_root, target_basename, target_run_tag, target_blends_tag = sys.argv[1:5]
target_blends_tag = target_blends_tag or ""

# Target lineage name = basename minus the round suffix. Compared against
# each candidate sidecar's `source_policy_basename` — run_tag alone is
# AMBIGUOUS across sweep families (e.g. planar_3joint_3 and planar_3joint_4
# lineages can share run_tag `d100_05dag`), and offering the wrong family's
# reruns for cleanup is a data-loss hazard.
target_lineage = re.sub(r"(?:_ft)?_dag\d+(?:_[A-Za-z][A-Za-z0-9_]*)?$", "", target_basename)

by_lineage = defaultdict(list)
for entry in sorted(os.listdir(training_root)):
    full = os.path.join(training_root, entry)
    if not os.path.isdir(full):
        continue
    # One sidecar per lineage is enough — pick round-1 dirs as canonical.
    # Trailing `_<suffix>` (e.g. `_nc` from step 6b) is allowed so we still
    # pick up rerun children whose only on-disk dirs are _nc variants.
    if not re.search(r"(?:_ft)?_dag1(?:_[A-Za-z][A-Za-z0-9_]*)?$", entry):
        continue
    cfg_path = os.path.join(full, "dagger", "config.json")
    if not os.path.isfile(cfg_path):
        continue
    try:
        cfg = json.load(open(cfg_path))
    except Exception:
        continue
    rr = cfg.get("rerun_mode") or {}
    src_run = rr.get("source_run_tag") or ""
    src_blends = rr.get("source_blends_tag") or ""
    if src_run != target_run_tag:
        continue
    if src_blends != target_blends_tag:
        continue
    # Family check: the sidecar's source_policy_basename must BE the target
    # lineage. Pre-source_policy_basename sidecars (older reruns, before
    # dagger_retrofit_rerun_sidecar.sh) lack the field — fall back to the
    # tag-only match above for those.
    src_policy = rr.get("source_policy_basename") or ""
    if src_policy and src_policy != target_lineage:
        continue
    m = re.match(r"^(.+?)(?:_ft)?_dag\d+(?:_[A-Za-z][A-Za-z0-9_]*)?$", entry)
    if not m:
        continue
    lineage = m.group(1)
    # Don't include the target itself (shouldn't happen — target's own
    # rerun_mode is null when it IS the source — but defensive).
    if entry == target_basename:
        continue
    by_lineage[lineage].append(entry)

for lineage in sorted(by_lineage.keys()):
    rounds = sorted(by_lineage[lineage])
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
    _AUTO_RERUN_CHILDREN=()
    while IFS= read -r line; do
        _AUTO_RERUN_CHILDREN+=( "$line" )
    done < <(detect_rerun_child_paths)
    # K-sibling detection includes the target itself (the regex matches it
    # by construction); strip it from the offer list. Rerun-child detection
    # explicitly excludes the target so its output is already clean.
    _OTHER_SIBLINGS=()
    _TARGET_BASE="$(basename "$TRAIN_DIR" | sed -E 's/(_ft)?_dag[0-9]+(_.*)?$//')"
    for p in "${_AUTO_SIBLINGS[@]}"; do
        _LIN="$(basename "$p" | sed -E 's/(_ft)?_dag[0-9]+(_.*)?$//')"
        [[ "$_LIN" != "$_TARGET_BASE" ]] && _OTHER_SIBLINGS+=( "$_LIN" )
    done
    _OTHER_RERUN_CHILDREN=()
    for p in "${_AUTO_RERUN_CHILDREN[@]}"; do
        _LIN="$(basename "$p" | sed -E 's/(_ft)?_dag[0-9]+(_.*)?$//')"
        _OTHER_RERUN_CHILDREN+=( "$_LIN" )
    done
    _TOTAL_OFFERED=$(( ${#_OTHER_SIBLINGS[@]} + ${#_OTHER_RERUN_CHILDREN[@]} ))
    if (( _TOTAL_OFFERED > 0 )); then
        # On-disk dag round numbers for a lineage (both `_dagN` scratch and
        # `_ft_dagN` finetune dirs, incl. `_nc`-style suffixed variants),
        # echoed as a comma list ("1,2"). Empty output = no round dirs.
        _rounds_on_disk() {
            local _tr="$1" _lin="$2" _d _n _list=()
            for _d in "$_tr/${_lin}"_dag[0-9]* "$_tr/${_lin}"_ft_dag[0-9]*; do
                [[ -d "$_d" ]] || continue
                _n="${_d##*_dag}"; _n="${_n%%_*}"
                [[ "$_n" =~ ^[0-9]+$ ]] && _list+=( "$_n" )
            done
            (( ${#_list[@]} > 0 )) || return 0
            printf '%s\n' "${_list[@]}" | sort -un | paste -sd,
        }
        _TRAINING_ROOT_FOR_LIST="$(dirname "$TRAIN_DIR")"
        _fmt_lineage_line() {
            local _label="$1" _lin="$2" _r
            _r="$(_rounds_on_disk "$_TRAINING_ROOT_FOR_LIST" "$_lin")"
            echo "    ${_label}: ${_lin}  (dag rounds on disk: ${_r:-none})"
        }
        echo
        echo "Heads up — found $_TOTAL_OFFERED other lineage(s) on disk related to the target:"
        echo "  TARGET: $_TARGET_BASE"
        if (( ${#_OTHER_SIBLINGS[@]} > 0 )); then
            echo "  [K-siblings: same prefix + blend-count]"
            for s in "${_OTHER_SIBLINGS[@]}"; do
                _fmt_lineage_line "SIBLING" "$s"
            done
        fi
        if (( ${#_OTHER_RERUN_CHILDREN[@]} > 0 )); then
            echo "  [rerun children: sidecar's rerun_mode source points at target]"
            for s in "${_OTHER_RERUN_CHILDREN[@]}"; do
                _fmt_lineage_line "RERUN_CHILD" "$s"
            done
        fi
        echo
        echo "By default this script cleans up ONLY the target. If you're cleaning up a"
        echo "sweep family or want to nuke the target's downstream reruns too, you can"
        echo "opt in to cleaning the related lineages above (equivalent to re-running"
        echo "with --detect_siblings)."
        if [[ -n "$FROM_ROUND" ]]; then
            echo "NOTE: --from_round=$FROM_ROUND applies to the related lineages too — only their"
            echo "  rounds ${FROM_ROUND}.. (and, with --also_delete_blends, the blend datasets named"
            echo "  after source rounds ${FROM_ROUND}..) are wiped; earlier rounds are preserved."
        fi
        echo -n "Also clean up the $_TOTAL_OFFERED related lineage(s) above? [y/N]: "
        read -r _SIBLING_REPLY
        if [[ "$_SIBLING_REPLY" =~ ^[Yy]$ ]]; then
            DETECT_SIBLINGS=true
            echo "[cleanup] related-lineage cleanup ENABLED via interactive opt-in."
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
    # Append rerun children. detect_rerun_child_paths excludes the target
    # itself so there's no de-dup needed; detect_sibling_paths INCLUDES the
    # target, so the union is just concatenation. Each rerun child lives
    # under a different prefix, so it can't collide with a K-sibling entry.
    while IFS= read -r line; do
        SIBLING_PATHS+=( "$line" )
    done < <(detect_rerun_child_paths)

    if (( ${#SIBLING_PATHS[@]} == 0 )); then
        echo "ERROR: --detect_siblings: no sibling lineages found." >&2
        echo "  Expected to find at least the target itself; nothing matched the" >&2
        echo "  same-prefix + same-K pattern (nor rerun-child sidecar pointer)" >&2
        echo "  derived from:" >&2
        echo "    $(basename "$TRAIN_DIR")" >&2
        exit 1
    fi

    echo "[detect_siblings] Found ${#SIBLING_PATHS[@]} lineage(s) related to target (K-siblings + rerun children):"
    for p in "${SIBLING_PATHS[@]}"; do
        _lineage_name="$(basename "$p" | sed -E 's/(_ft)?_dag[0-9]+(_.*)?$//')"
        _training_root="$(dirname "$p")"
        # Enumerate every round training dir under this lineage — both
        # `_dag<N>` (scratch mode) and `_ft_dag<N>` (finetune mode), with
        # optional trailing `_<suffix>` variants (e.g. `_nc` from step 6b).
        # Group into three buckets so the user sees exactly which rounds
        # will be wiped instead of just a count:
        #   raw_rounds  = `_ft_dag<N>` (canonical finetune) OR `_dag<N>`
        #                 (scratch). Numeric list, sorted, deduped.
        #   nc_rounds   = `_ft_dag<N>_nc` (step-6b nocoll retrain).
        #   other_rounds = `_ft_dag<N>_<suffix>` with any other suffix
        #                  (--retrain_suffix variants, etc). Rare; shown
        #                  with the suffix name so the user notices them.
        # bash-y set: use `printf | sort -un` to dedup + sort numeric.
        declare -a _raw_list=() _nc_list=() _other_list=()
        for d in "$_training_root/${_lineage_name}"_dag[0-9]* \
                 "$_training_root/${_lineage_name}"_ft_dag[0-9]*; do
            [[ -d "$d" ]] || continue
            _bn="$(basename "$d")"
            _rn=$(echo "$_bn" | grep -oE 'dag[0-9]+' | head -1 | grep -oE '[0-9]+')
            [[ -z "$_rn" ]] && continue
            # Suffix after `_dag<N>` — empty for canonical, non-empty for
            # variants like `_nc`. Extracted by stripping the shared prefix.
            _suffix=$(echo "$_bn" | sed -E "s/^${_lineage_name}(_ft)?_dag${_rn}//; s/^_//")
            if [[ -z "$_suffix" ]]; then
                _raw_list+=( "$_rn" )
            elif [[ "$_suffix" == "nc" ]]; then
                _nc_list+=( "$_rn" )
            else
                _other_list+=( "${_rn}_${_suffix}" )
            fi
        done
        # Dedup + sort each bucket. Filter out empty strings from
        # `${_arr[@]:-}` expansion — under bash's `set -u`, an empty
        # array `_arr=()` interpolates as one empty string via `:-`,
        # making `$#` read as 1 instead of 0. Skip empties explicitly
        # so we don't render `"1 round(s) []"` for empty buckets.
        _fmt_rounds() {
            local _name="$1"; shift
            local _nonempty=() _v
            for _v in "$@"; do
                [[ -n "$_v" ]] && _nonempty+=( "$_v" )
            done
            (( ${#_nonempty[@]} == 0 )) && return
            local _sorted _count
            _sorted=$(printf '%s\n' "${_nonempty[@]}" | sort -un | paste -sd,)
            _count=$(printf '%s\n' "${_nonempty[@]}" | sort -un | wc -l)
            _count="${_count// /}"
            echo "    → ${_name}: ${_count} round(s) [${_sorted}]"
        }
        echo "  ${_lineage_name}"
        _fmt_rounds "raw"   "${_raw_list[@]:-}"
        _fmt_rounds "_nc"   "${_nc_list[@]:-}"
        _fmt_rounds "other" "${_other_list[@]:-}"
        if (( ${#_raw_list[@]} == 0 && ${#_nc_list[@]} == 0 && ${#_other_list[@]} == 0 )); then
            echo "    → (no on-disk round dirs found for this lineage)"
        fi
    done
    echo

    # Use a mode-specific confirmation token + message so the user knows
    # exactly what scope this batch will wipe. nc_only is a SURGICAL mode
    # (only step-6b/filter artifacts), so reusing the "delete-all" token
    # from the legacy full-cleanup path would be misleading.
    if (( ${#DRY_RUN_FLAG[@]} == 0 )) && [[ "$AUTO_CONFIRM" != true ]]; then
        if [[ "$NC_ONLY" == true ]]; then
            echo "MODE: --nc_only — only step-6b / filter_blend_collisions artifacts will be deleted:"
            echo "  * _ft_dag<N>_nc training dirs"
            echo "  * dagger/blend_collision_filter/ audit subdirs"
            echo "  * _blend<NNN>_nocoll datasets + stats sidecars"
            echo "  Raw policies, raw blends, intervention/alias/merged datasets are PRESERVED."
            echo -n "Type 'delete-nc-all' to confirm nc-only cleanup of all ${#SIBLING_PATHS[@]} lineages above: "
            read -r CONFIRM
            [[ "$CONFIRM" == "delete-nc-all" ]] || { echo "Aborted."; exit 1; }
        elif [[ "$BLENDS_ONLY" == true ]]; then
            echo "MODE: --blends_only — only blend-derived artifacts will be deleted:"
            echo "  * _blend<NNN> datasets (+ _nocoll siblings) + stats sidecars"
            echo "  * dagger/blend_collision_filter/ audit subdirs"
            echo "  * merged (*_m) datasets + sidecars"
            echo "  * per-round training dirs (incl. _nc) + outputs/dagger eval dirs"
            echo "  Intervention/alias datasets + int-stats and the round-0 base policy are PRESERVED."
            echo -n "Type 'delete-blends-all' to confirm blends-only cleanup of all ${#SIBLING_PATHS[@]} lineages above: "
            read -r CONFIRM
            [[ "$CONFIRM" == "delete-blends-all" ]] || { echo "Aborted."; exit 1; }
        else
            echo -n "Type 'delete-all' to confirm cleanup of all ${#SIBLING_PATHS[@]} lineages above: "
            read -r CONFIRM
            [[ "$CONFIRM" == "delete-all" ]] || { echo "Aborted."; exit 1; }
        fi
    fi

    # Recurse: per-lineage cleanup. -y at the inner level suppresses each
    # sibling's own confirmation prompt since the user already confirmed
    # the whole batch above. Don't let `set -e` abort the loop on a single
    # failure — keep going so a downstream sibling that still has artifacts
    # gets a chance to be cleaned up too.
    NC_ONLY_FORWARD=()
    [[ "$NC_ONLY" == true ]] && NC_ONLY_FORWARD=( --nc_only )
    BLENDS_ONLY_FORWARD=()
    [[ "$BLENDS_ONLY" == true ]] && BLENDS_ONLY_FORWARD=( --blends_only )
    # --delete_episodes recursion: forward the episode list to every
    # sibling, but the lerobot-edit-dataset call must run exactly ONCE
    # (the source intervention dataset is shared across rerun siblings).
    # First sibling runs the edit; the rest get --skip_dataset_edit so
    # they only nuke their own downstream training dirs and blends.
    DELETE_EPISODES_FORWARD=()
    [[ -n "$DELETE_EPISODES" ]] && DELETE_EPISODES_FORWARD=( "--delete_episodes=$DELETE_EPISODES" )
    overall_rc=0
    _is_first_sibling=true
    for p in "${SIBLING_PATHS[@]}"; do
        echo
        echo "=== [detect_siblings] cleaning up: $(basename "$p" | sed -E 's/(_ft)?_dag[0-9]+(_.*)?$//') ==="
        SKIP_EDIT_FORWARD=()
        if [[ -n "$DELETE_EPISODES" && "$_is_first_sibling" != true ]]; then
            SKIP_EDIT_FORWARD=( --skip_dataset_edit )
        fi
        bash "$0" "$p" -y "${DRY_RUN_FLAG[@]}" "${ALSO_DELETE_BLENDS_FLAG[@]}" "${FILTER_BLEND_COLLISIONS_FLAG[@]}" "${KEEP_R1_FORWARD[@]}" "${NC_ONLY_FORWARD[@]}" "${BLENDS_ONLY_FORWARD[@]}" "${FROM_ROUND_FLAG[@]}" "${DELETE_EPISODES_FORWARD[@]}" "${SKIP_EDIT_FORWARD[@]}" && rc=0 || rc=$?
        if (( rc != 0 )); then
            overall_rc="$rc"
            echo "[detect_siblings] WARN: cleanup failed for $p (rc=$rc); continuing." >&2
        fi
        _is_first_sibling=false
    done
    exit "$overall_rc"
fi

# ── --nc_only: surgical collision-filter cleanup ──────────────────────────────
# Bypass the orchestrator delegate entirely. We only need to find and rm:
#   1. `_ft_dag<N>_nc` training dirs under the lineage's TRAINING_ROOT.
#   2. `dagger/blend_collision_filter/` audit subdirs under each raw round's
#      training dir (sibling of the `_nc` dir).
#   3. `_blend<NNN>_nocoll` datasets in the HF cache.
#   4. `_blend<NNN>_nocoll` stats sidecars.
# The lineage's blend ratios + source intervention prefix come from the
# round-1 sidecar (every round writes the same one).
if [[ "$NC_ONLY" == true ]]; then
    LEROBOT_CACHE="${LEROBOT_CACHE:-$HOME/.cache/huggingface/lerobot}"
    HF_USER="${HF_USER:-JennyWWW}"
    STATS_BASE="$LEROBOT_ROOT/outputs/dataset_stats"
    TRAINING_ROOT="$(dirname "$TRAIN_DIR")"
    LINEAGE_BASE="$(basename "$TRAIN_DIR" | sed -E 's/(_ft)?_dag[0-9]+(_.*)?$//')"

    # Derive blend ratios + source intervention-prefix from the sidecar so we
    # know which `_nocoll` dataset names to glob in the cache / stats dirs.
    CFG="$TRAIN_DIR/dagger/config.json"
    if [[ ! -f "$CFG" ]]; then
        echo "ERROR: --nc_only needs the round's dagger/config.json sidecar; not found at:" >&2
        echo "  $CFG" >&2
        echo "  (Pre-sidecar lineages aren't supported by --nc_only. Use the full delete-all" >&2
        echo "   cleanup, or rm the _nc / _nocoll artifacts by hand.)" >&2
        exit 1
    fi
    NC_META="$(python3 -c "
import json, sys
cfg = json.load(open(sys.argv[1]))
ratios = (cfg.get('config') or {}).get('blends') or []
fmt = ((cfg.get('config') or {}).get('action_format') or 'rel').lower()
infix = 'r' if fmt == 'rel' else 'a'
rerun = cfg.get('rerun_mode') or {}
# In rerun mode the nocoll dataset is named after the SOURCE lineage's
# intervention prefix; in non-rerun mode it's named after the lineage's
# own base_dataset_short. Both schemas yield <prefix>_<infix>_dag<N>_blend<NNN>.
prefix = (rerun.get('source_int_short_prefix') or '').strip() \
         or (cfg.get('naming') or {}).get('base_dataset_short') or ''
print(infix)
print(prefix)
for r in ratios:
    print(f'{int(round(float(r) * 100)):03d}')
" "$CFG")"
    ACTION_INFIX="$(printf '%s\n' "$NC_META" | sed -n 1p)"
    SOURCE_INT_PREFIX="$(printf '%s\n' "$NC_META" | sed -n 2p)"
    mapfile -t BLEND_TAGS < <(printf '%s\n' "$NC_META" | sed -n '3,$p')
    if [[ -z "$SOURCE_INT_PREFIX" || "${#BLEND_TAGS[@]}" -eq 0 ]]; then
        echo "ERROR: sidecar at $CFG didn't yield a usable source intervention prefix or blend list." >&2
        echo "  Got: prefix='$SOURCE_INT_PREFIX', blend_tags=(${BLEND_TAGS[*]})" >&2
        exit 1
    fi

    echo "[nc_only] target lineage: $LINEAGE_BASE"
    echo "[nc_only] nocoll dataset prefix: $SOURCE_INT_PREFIX (action_infix=$ACTION_INFIX)"
    echo "[nc_only] blend tags: ${BLEND_TAGS[*]}"

    # 1. _nc training dirs (with optional _<suffix> for retrain variants).
    declare -a NC_DIRS=()
    while IFS= read -r p; do
        [[ -z "$p" ]] && continue
        NC_DIRS+=( "$p" )
    done < <(ls -d "$TRAINING_ROOT/${LINEAGE_BASE}"_ft_dag[0-9]*_nc 2>/dev/null || true)

    # 2. blend_collision_filter audit subdirs under each RAW round's dir
    #    (skip the _nc round dirs from this list to avoid double-rm; the _nc
    #    dirs themselves are wiped wholesale in step 1).
    declare -a AUDIT_DIRS=()
    while IFS= read -r p; do
        [[ -z "$p" ]] && continue
        AUDIT_DIRS+=( "$p" )
    done < <(
        for d in "$TRAINING_ROOT/${LINEAGE_BASE}"_ft_dag[0-9]*; do
            [[ -d "$d" ]] || continue
            # Skip the _nc dirs — those are dropped wholesale in step 1.
            [[ "$d" == *_nc ]] && continue
            audit="$d/dagger/blend_collision_filter"
            [[ -d "$audit" ]] && echo "$audit"
        done
    )

    # 3 + 4. nocoll blend datasets in the HF cache + stats sidecars. One pair
    # per (round, ratio) combination; glob the round-number wildcard.
    declare -a NOCOLL_DATASETS=()
    declare -a NOCOLL_STATS=()
    for tag in "${BLEND_TAGS[@]}"; do
        pattern="${SOURCE_INT_PREFIX}_${ACTION_INFIX}_dag*_blend${tag}_nocoll"
        while IFS= read -r p; do
            [[ -z "$p" ]] && continue
            NOCOLL_DATASETS+=( "$p" )
        done < <(ls -d "$LEROBOT_CACHE/$HF_USER/"$pattern 2>/dev/null || true)
        while IFS= read -r p; do
            [[ -z "$p" ]] && continue
            NOCOLL_STATS+=( "$p" )
        done < <(ls -d "$STATS_BASE/"$pattern 2>/dev/null || true)
    done

    TOTAL=$(( ${#NC_DIRS[@]} + ${#AUDIT_DIRS[@]} + ${#NOCOLL_DATASETS[@]} + ${#NOCOLL_STATS[@]} ))
    if (( TOTAL == 0 )); then
        echo "[nc_only] nothing to delete — no _nc / _nocoll / audit artifacts found for this lineage."
        exit 0
    fi

    echo
    echo "Will DELETE (${TOTAL} item(s)):"
    if (( ${#NC_DIRS[@]} > 0 )); then
        echo "  [_nc training dirs] (${#NC_DIRS[@]}):"
        printf '    %s\n' "${NC_DIRS[@]}"
    fi
    if (( ${#AUDIT_DIRS[@]} > 0 )); then
        echo "  [blend_collision_filter audit subdirs] (${#AUDIT_DIRS[@]}):"
        printf '    %s\n' "${AUDIT_DIRS[@]}"
    fi
    if (( ${#NOCOLL_DATASETS[@]} > 0 )); then
        echo "  [_nocoll blend datasets] (${#NOCOLL_DATASETS[@]}):"
        printf '    %s\n' "${NOCOLL_DATASETS[@]}"
    fi
    if (( ${#NOCOLL_STATS[@]} > 0 )); then
        echo "  [_nocoll stats sidecars] (${#NOCOLL_STATS[@]}):"
        printf '    %s\n' "${NOCOLL_STATS[@]}"
    fi
    echo
    echo "PRESERVED: raw policies (_ft_dag<N>), raw blends (_blend<NNN>),"
    echo "           intervention datasets, alias/merged datasets, R0 base policy."

    if (( ${#DRY_RUN_FLAG[@]} > 0 )); then
        echo
        echo "[--dry-run] would rm -rf the items above."
        exit 0
    fi

    if [[ "$AUTO_CONFIRM" != true ]]; then
        echo
        echo -n "Type 'delete-nc' to confirm: "
        read -r CONFIRM
        [[ "$CONFIRM" == "delete-nc" ]] || { echo "Aborted."; exit 1; }
    fi

    for p in "${NC_DIRS[@]}" "${AUDIT_DIRS[@]}" "${NOCOLL_DATASETS[@]}" "${NOCOLL_STATS[@]}"; do
        rm -rf "$p"
    done
    echo "[nc_only] Deleted ${TOTAL} item(s)."
    echo "[nc_only] Re-run dagger_orchestrate_sweep.sh with --filter_blend_collisions"
    echo "[nc_only] to re-produce _nocoll datasets (step 2's filter sub-step) and"
    echo "[nc_only] re-train _nc policies (step 6b)."
    exit 0
fi

# ── --blends_only: surgical blend-artifact cleanup ────────────────────────────
# Bypasses the orchestrator delegate. Deletes everything DERIVED from blends
# while preserving the intervention recordings:
#   1. `_blend<NNN>` raw blend datasets in the HF cache + stats sidecars.
#   2. `_blend<NNN>_nocoll` siblings + stats sidecars.
#   3. The lineage's merged datasets (`*_m`) + sidecars (contain blend frames
#      in merge mode; glob simply misses in weighted mode).
#   4. The lineage's per-round training dirs (`_dag<N>` / `_ft_dag<N>` incl.
#      `_nc` and other suffix variants) + their outputs/dagger eval dirs —
#      they were trained against the blends, so they're stale once the blends
#      are regenerated. (This also removes the blend_collision_filter audit
#      subdirs, which live inside the training dirs.)
# PRESERVED: intervention datasets + alias datasets + int-stats sidecars, the
# round-0 base policy, and (rerun mode) the entire source lineage.
if [[ "$BLENDS_ONLY" == true ]]; then
    LEROBOT_CACHE="${LEROBOT_CACHE:-$HOME/.cache/huggingface/lerobot}"
    HF_USER="${HF_USER:-JennyWWW}"
    STATS_BASE="$LEROBOT_ROOT/outputs/dataset_stats"
    TRAINING_ROOT="$(dirname "$TRAIN_DIR")"
    LINEAGE_BASE="$(basename "$TRAIN_DIR" | sed -E 's/(_ft)?_dag[0-9]+(_.*)?$//')"

    CFG="$TRAIN_DIR/dagger/config.json"
    if [[ ! -f "$CFG" ]]; then
        echo "ERROR: --blends_only needs the round's dagger/config.json sidecar; not found at:" >&2
        echo "  $CFG" >&2
        echo "  (Pre-sidecar lineages aren't supported by --blends_only. Use the full cleanup," >&2
        echo "   or rm the _blend<NNN> artifacts by hand.)" >&2
        exit 1
    fi
    BL_META="$(python3 -c "
import json, sys
cfg = json.load(open(sys.argv[1]))
ratios = (cfg.get('config') or {}).get('blends') or []
fmt = ((cfg.get('config') or {}).get('action_format') or 'rel').lower()
infix = 'r' if fmt == 'rel' else 'a'
rerun = cfg.get('rerun_mode') or {}
# Blend datasets are named after the source intervention prefix in rerun
# mode, or the lineage's own dag prefix otherwise (same rule as --nc_only).
blend_prefix = (rerun.get('source_int_short_prefix') or '').strip() \
    or (cfg.get('naming') or {}).get('base_dataset_short') or ''
# Merged datasets are always named after the lineage's OWN dag prefix.
lineage_prefix = (cfg.get('naming') or {}).get('base_dataset_short') or ''
print(infix)
print(blend_prefix)
print(lineage_prefix)
for r in ratios:
    print(f'{int(round(float(r) * 100)):03d}')
" "$CFG")"
    ACTION_INFIX="$(printf '%s\n' "$BL_META" | sed -n 1p)"
    BLEND_PREFIX="$(printf '%s\n' "$BL_META" | sed -n 2p)"
    LINEAGE_PREFIX="$(printf '%s\n' "$BL_META" | sed -n 3p)"
    mapfile -t BLEND_TAGS < <(printf '%s\n' "$BL_META" | sed -n '4,$p')
    if [[ -z "$BLEND_PREFIX" || "${#BLEND_TAGS[@]}" -eq 0 ]]; then
        echo "ERROR: sidecar at $CFG didn't yield a usable blend prefix or blend list." >&2
        echo "  Got: blend_prefix='$BLEND_PREFIX', blend_tags=(${BLEND_TAGS[*]:-})" >&2
        exit 1
    fi

    echo "[blends_only] target lineage: $LINEAGE_BASE"
    echo "[blends_only] blend dataset prefix: $BLEND_PREFIX (action_infix=$ACTION_INFIX)"
    echo "[blends_only] blend tags: ${BLEND_TAGS[*]}"

    # 1 + 2. Blend datasets (+ nocoll siblings) + stats sidecars. The bare
    # `_blend<NNN>` pattern has no trailing wildcard, so it can't swallow the
    # `_nocoll` sibling — that's collected by its own explicit pattern.
    declare -a BLEND_PATHS=()
    for tag in "${BLEND_TAGS[@]}"; do
        for pattern in \
            "${BLEND_PREFIX}_${ACTION_INFIX}_dag[0-9]*_blend${tag}" \
            "${BLEND_PREFIX}_${ACTION_INFIX}_dag[0-9]*_blend${tag}_nocoll"; do
            while IFS= read -r p; do
                [[ -z "$p" ]] && continue
                BLEND_PATHS+=( "$p" )
            done < <(ls -d "$LEROBOT_CACHE/$HF_USER/"$pattern "$STATS_BASE/"$pattern 2>/dev/null || true)
        done
    done

    # 3. Merged datasets + sidecars (merge-mode lineages only; empty glob
    # in weighted mode).
    declare -a MERGED_PATHS=()
    if [[ -n "$LINEAGE_PREFIX" ]]; then
        while IFS= read -r p; do
            [[ -z "$p" ]] && continue
            MERGED_PATHS+=( "$p" )
        done < <(ls -d "$LEROBOT_CACHE/$HF_USER/${LINEAGE_PREFIX}_${ACTION_INFIX}"_dag[0-9]*_m \
                       "$STATS_BASE/${LINEAGE_PREFIX}_${ACTION_INFIX}"_dag[0-9]*_m 2>/dev/null || true)
    fi

    # 4. Per-round training dirs (all suffix variants) + outputs/dagger eval dirs.
    declare -a TRAIN_PATHS=()
    while IFS= read -r p; do
        [[ -z "$p" ]] && continue
        TRAIN_PATHS+=( "$p" )
    done < <(ls -d "$TRAINING_ROOT/${LINEAGE_BASE}"_dag[0-9]* \
                   "$TRAINING_ROOT/${LINEAGE_BASE}"_ft_dag[0-9]* \
                   "$LEROBOT_ROOT/outputs/dagger/${LINEAGE_BASE}"_dag[0-9]* \
                   "$LEROBOT_ROOT/outputs/dagger/${LINEAGE_BASE}"_ft_dag[0-9]* 2>/dev/null || true)

    TOTAL=$(( ${#BLEND_PATHS[@]} + ${#MERGED_PATHS[@]} + ${#TRAIN_PATHS[@]} ))
    if (( TOTAL == 0 )); then
        echo "[blends_only] nothing to delete — no blend / merged / round-training artifacts found for this lineage."
        exit 0
    fi

    echo
    echo "Will DELETE (${TOTAL} item(s)):"
    if (( ${#BLEND_PATHS[@]} > 0 )); then
        echo "  [blend datasets + nocoll siblings + stats sidecars] (${#BLEND_PATHS[@]}):"
        printf '    %s\n' "${BLEND_PATHS[@]}"
    fi
    if (( ${#MERGED_PATHS[@]} > 0 )); then
        echo "  [merged datasets + sidecars] (${#MERGED_PATHS[@]}):"
        printf '    %s\n' "${MERGED_PATHS[@]}"
    fi
    if (( ${#TRAIN_PATHS[@]} > 0 )); then
        echo "  [per-round training dirs + eval dirs] (${#TRAIN_PATHS[@]}):"
        printf '    %s\n' "${TRAIN_PATHS[@]}"
    fi
    echo
    echo "PRESERVED: intervention datasets + alias datasets + int-stats sidecars,"
    echo "           the round-0 base policy, and (rerun mode) the source lineage."

    if (( ${#DRY_RUN_FLAG[@]} > 0 )); then
        echo
        echo "[--dry-run] would rm -rf the items above."
        exit 0
    fi

    if [[ "$AUTO_CONFIRM" != true ]]; then
        echo
        echo -n "Type 'delete-blends' to confirm: "
        read -r CONFIRM
        [[ "$CONFIRM" == "delete-blends" ]] || { echo "Aborted."; exit 1; }
    fi

    for p in "${BLEND_PATHS[@]:-}" "${MERGED_PATHS[@]:-}" "${TRAIN_PATHS[@]:-}"; do
        [[ -n "$p" ]] && rm -rf "$p"
    done
    echo "[blends_only] Deleted ${TOTAL} item(s)."
    echo "[blends_only] Re-run dagger_orchestrate.sh / dagger_orchestrate_sweep.sh with"
    echo "[blends_only] --resume to re-blend (step 2) and re-train (step 6) against the"
    echo "[blends_only] preserved intervention data."
    exit 0
fi

# ── --delete_episodes: surgical episode-level cleanup ─────────────────────────
# Workflow (bypasses orchestrator):
#   1. Parse round number R from training dir name.
#   2. Read sidecar's `config.weighted_repo_ids[R]` to find round R's
#      intervention dataset repo id (fall back to dagger_naming derivation
#      if the sidecar predates weighted_sampling — rare; warns then asks
#      user to supply --skip_dataset_edit if they want to proceed).
#   3. lerobot-edit-dataset --operation.type delete_episodes (skipped
#      when --skip_dataset_edit set).
#   4. compute_relative_stats.sh to refresh rel-action stats sidecar
#      (skipped when --skip_dataset_edit set; the sidecar would be stale
#      after the dataset edit and downstream training would normalize
#      against the wrong range).
#   5. rm -rf training dirs for rounds R..NUM_ROUNDS (so they retrain on
#      cleaned data).
#   6. rm -rf blend datasets + sidecars for rounds R..NUM_ROUNDS (blends
#      derive from round-R intervention; stale after the edit).
# Intervention datasets for rounds 1..(R-1) and the cleaned round-R
# intervention itself are PRESERVED.
if [[ -n "$DELETE_EPISODES" ]]; then
    LEROBOT_CACHE="${LEROBOT_CACHE:-$HOME/.cache/huggingface/lerobot}"
    STATS_BASE="$LEROBOT_ROOT/outputs/dataset_stats"
    TRAINING_ROOT="$(dirname "$TRAIN_DIR")"
    TRAIN_BASENAME="$(basename "$TRAIN_DIR")"

    # Parse round number from `<...>{_ft,}_dag<N>{_<suffix>}` basename.
    if [[ "$TRAIN_BASENAME" =~ _dag([0-9]+)(_.*)?$ ]]; then
        TARGET_ROUND="${BASH_REMATCH[1]}"
    else
        echo "ERROR: training dir name doesn't end in _dag<N>: $TRAIN_BASENAME" >&2; exit 1
    fi
    # Lineage base for globbing sibling round dirs / blend datasets.
    LINEAGE_BASE="$(basename "$TRAIN_DIR" | sed -E 's/(_ft)?_dag[0-9]+(_.*)?$//')"

    # Sidecar provides the exact intervention repo id for round R AND the
    # final NUM_ROUNDS for the lineage. The target round's own sidecar is
    # preferred, but it may not exist yet (intervention recorded, training
    # not started). In that case fall back to the HIGHEST existing round's
    # sidecar in the same lineage and derive round R's intervention repo by
    # round-number substitution (the canonical name pattern is
    # `<user>/<prefix>_dag<N>`). Pre-sidecar lineages remain unsupported.
    CFG="$TRAIN_DIR/dagger/config.json"
    if [[ ! -f "$CFG" ]]; then
        CFG=""
        _best_round=-1
        for d in "$TRAINING_ROOT/${LINEAGE_BASE}"_ft_dag[0-9]* \
                 "$TRAINING_ROOT/${LINEAGE_BASE}"_dag[0-9]*; do
            [[ -f "$d/dagger/config.json" ]] || continue
            n=$(basename "$d" | grep -oE 'dag[0-9]+' | head -1 | grep -oE '[0-9]+')
            if (( n > _best_round )); then _best_round=$n; CFG="$d/dagger/config.json"; fi
        done
        if [[ -z "$CFG" ]]; then
            echo "ERROR: --delete_episodes needs a dagger/config.json sidecar to resolve the" >&2
            echo "  intervention dataset path, and none was found — neither in the target round's" >&2
            echo "  training dir nor any earlier round's dir for lineage '$LINEAGE_BASE'." >&2
            echo "  Pre-sidecar lineages aren't supported by this surgical mode. For those, use" >&2
            echo "  the legacy whole-round path:  --from_round=$TARGET_ROUND  (re-records the round)." >&2
            exit 1
        fi
        echo "[delete_episodes] round $TARGET_ROUND not trained yet; deriving its intervention"
        echo "[delete_episodes] dataset name from round $_best_round's sidecar: $CFG"
    fi
    META="$(python3 -c "
import json, re, sys
cfg = json.load(open(sys.argv[1]))
cf = cfg.get('config') or {}
target_round = int(sys.argv[2])
# weighted_repo_ids[0] = base; [1..N] = round 1..N intervention repos.
wri = cf.get('weighted_repo_ids') or []
wsp = cf.get('weighted_stats_paths') or []
if not wri:
    sys.exit('ERROR: sidecar has no weighted_repo_ids.')
if len(wri) > target_round:
    repo = wri[target_round]
    stats = wsp[target_round] if target_round < len(wsp) else ''
else:
    # Target round not represented in this (earlier) sidecar — derive its
    # name by substituting the round number into the highest known
    # intervention repo (entries [1..] are round 1..k; [0] is base).
    src = wri[-1]
    repo = re.sub(r'_dag[0-9]+$', f'_dag{target_round}', src)
    if repo == src:
        sys.exit(f'ERROR: could not derive round-{target_round} intervention name from {src!r} '
                 '(no _dag<N> suffix to substitute).')
    stats = ''  # refreshed downstream by compute_relative_stats.sh
print(repo)
print(stats)
print((cf.get('action_format') or 'rel').lower())
" "$CFG" "$TARGET_ROUND")" || exit 1
    INT_REPO_ID="$(printf '%s\n' "$META" | sed -n 1p)"
    STATS_PATH="$(printf '%s\n' "$META" | sed -n 2p)"
    ACTION_FORMAT="$(printf '%s\n' "$META" | sed -n 3p)"
    INT_DATASET_PATH="$LEROBOT_CACHE/$INT_REPO_ID"
    INT_DATASET_SHORT="${INT_REPO_ID#*/}"
    INT_REPO_USER="${INT_REPO_ID%/*}"
    ACTION_INFIX="r"; [[ "$ACTION_FORMAT" == "abs" ]] && ACTION_INFIX="a"

    # Derive NUM_ROUNDS from a DISK scan, not the sidecar — the sidecar
    # only reflects rounds trained UP TO that point. If the target's
    # sidecar is from dag<TARGET> but rounds dag<TARGET+1>..dag<N> have
    # been trained since (their training used dag<TARGET>'s polluted
    # data and also needs to be re-run), they must be nuked too.
    NUM_ROUNDS=0
    for d in "$TRAINING_ROOT/${LINEAGE_BASE}"_ft_dag[0-9]* \
             "$TRAINING_ROOT/${LINEAGE_BASE}"_dag[0-9]*; do
        [[ -d "$d" ]] || continue
        n=$(basename "$d" | grep -oE 'dag[0-9]+' | head -1 | grep -oE '[0-9]+')
        (( n > NUM_ROUNDS )) && NUM_ROUNDS=$n
    done
    # The target round's own training dir may not exist yet (intervention
    # recorded, training not started), so it won't appear in the disk scan.
    # Clamp up to TARGET_ROUND so the round-range seq below includes it (the
    # rm globs are all `[[ -d ]]`-guarded, so missing downstream dirs no-op).
    (( TARGET_ROUND > NUM_ROUNDS )) && NUM_ROUNDS=$TARGET_ROUND

    if [[ "$SKIP_DATASET_EDIT" != true && ! -d "$INT_DATASET_PATH" ]]; then
        echo "ERROR: intervention dataset not found on disk: $INT_DATASET_PATH" >&2
        echo "  (parsed from sidecar weighted_repo_ids[$TARGET_ROUND] = $INT_REPO_ID)" >&2; exit 1
    fi

    # Build the cleanup list: training dirs + blend datasets + blend
    # sidecars for rounds R..NUM_ROUNDS.
    # Blend dataset naming: <round-N int-dataset-short>_blend<NNN>[_nocoll].
    # The round number lives in the int_short itself (e.g.
    # `lever_g0_d30_fast_03dag_diff_r_dag4`), so we need to derive each
    # round's int_short to glob its blends — not just a `*_dagN_blend*`
    # glob which would match other lineages' blends. Reuse the sidecar's
    # weighted_repo_ids[r] for each round.
    declare -a NUKE_TRAIN_DIRS=() NUKE_BLENDS=() NUKE_BLEND_STATS=()
    for r in $(seq "$TARGET_ROUND" "$NUM_ROUNDS"); do
        # Training dirs: include both `_ft_dag<N>` and `_dag<N>` (scratch
        # mode) and any `_<suffix>` variants (e.g. `_nc` from step 6b).
        for d in "$TRAINING_ROOT/${LINEAGE_BASE}"_ft_dag${r} \
                 "$TRAINING_ROOT/${LINEAGE_BASE}"_ft_dag${r}_* \
                 "$TRAINING_ROOT/${LINEAGE_BASE}"_dag${r} \
                 "$TRAINING_ROOT/${LINEAGE_BASE}"_dag${r}_*; do
            [[ -d "$d" ]] && NUKE_TRAIN_DIRS+=( "$d" )
        done
        # Per-round int_short for blend globbing. The sidecar's
        # weighted_repo_ids only goes up to the target round's knowledge,
        # so for rounds > TARGET_ROUND we derive the name by substituting
        # the round number in the target's int_repo_id (canonical pattern:
        # `<user>/<prefix>_dag<N>`). Same for the stats sidecar dir name.
        ROUND_R_INT_SHORT="$(echo "$INT_DATASET_SHORT" | sed -E "s/_dag[0-9]+$/_dag${r}/")"
        ROUND_R_INT_USER="$INT_REPO_USER"
        # Blend datasets: <round-r int_short>_blend<NNN>[_nocoll].
        for bd in "$LEROBOT_CACHE/$ROUND_R_INT_USER/${ROUND_R_INT_SHORT}_blend"[0-9]*; do
            [[ -d "$bd" ]] && NUKE_BLENDS+=( "$bd" )
        done
        # Blend stats sidecars under outputs/dataset_stats.
        for bs in "$STATS_BASE/${ROUND_R_INT_SHORT}_blend"[0-9]*; do
            [[ -d "$bs" ]] && NUKE_BLEND_STATS+=( "$bs" )
        done
    done

    # Sanity: dedup (the glob expansion can produce duplicates for the
    # `_*` variant matching the same dir).
    declare -A _seen=()
    declare -a NUKE_TRAIN_DIRS_DEDUP=()
    for d in "${NUKE_TRAIN_DIRS[@]:-}"; do
        [[ -z "$d" ]] && continue
        [[ -n "${_seen[$d]:-}" ]] && continue
        _seen[$d]=1; NUKE_TRAIN_DIRS_DEDUP+=( "$d" )
    done
    # Reassign without the `[@]:-` idiom — on an empty array that injects a
    # spurious empty element, which would inflate the "training dirs" count to
    # 1 (and print a blank line) when the target round isn't trained yet.
    NUKE_TRAIN_DIRS=()
    (( ${#NUKE_TRAIN_DIRS_DEDUP[@]} )) && NUKE_TRAIN_DIRS=( "${NUKE_TRAIN_DIRS_DEDUP[@]}" )

    echo
    echo "[delete_episodes] target lineage: $LINEAGE_BASE"
    echo "[delete_episodes] target round:   $TARGET_ROUND (of $NUM_ROUNDS)"
    echo "[delete_episodes] intervention:   $INT_REPO_ID"
    echo "[delete_episodes] stats sidecar:  ${STATS_PATH:-<derived>}"
    echo "[delete_episodes] episode indices to remove: $DELETE_EPISODES"
    echo
    if [[ "$SKIP_DATASET_EDIT" == true ]]; then
        echo "  EDIT STEPS:    SKIPPED (--skip_dataset_edit set)"
    else
        echo "  EDIT STEPS:"
        echo "    1. lerobot-edit-dataset --repo_id $INT_REPO_ID \\"
        echo "         --operation.type delete_episodes \\"
        echo "         --operation.episode_indices '$DELETE_EPISODES'"
        echo "    2. bash my_scripts/compute_relative_stats.sh --dataset_repo=$INT_REPO_ID"
    fi
    n_train="${#NUKE_TRAIN_DIRS[@]}"
    n_blends="${#NUKE_BLENDS[@]}"
    n_blend_stats="${#NUKE_BLEND_STATS[@]}"
    echo "  RM -RF (rounds ${TARGET_ROUND}..${NUM_ROUNDS}):"
    echo "    training dirs    : $n_train"
    [[ $n_train -gt 0 ]] && printf "      %s\n" "${NUKE_TRAIN_DIRS[@]}"
    echo "    blend datasets   : $n_blends"
    [[ $n_blends -gt 0 ]] && printf "      %s\n" "${NUKE_BLENDS[@]}"
    echo "    blend sidecars   : $n_blend_stats"
    [[ $n_blend_stats -gt 0 ]] && printf "      %s\n" "${NUKE_BLEND_STATS[@]}"
    echo
    echo "  PRESERVED: round 1..$((TARGET_ROUND - 1)) artifacts, cleaned round-${TARGET_ROUND}"
    echo "             intervention dataset, base policy."

    if (( ${#DRY_RUN_FLAG[@]} > 0 )); then
        echo
        echo "[--dry-run] would run the edit steps + rm -rf the items above."
        exit 0
    fi

    if [[ "$AUTO_CONFIRM" != true ]]; then
        echo
        echo -n "Type 'delete-episodes' to confirm: "
        read -r CONFIRM
        [[ "$CONFIRM" == "delete-episodes" ]] || { echo "Aborted."; exit 1; }
    fi

    if [[ "$SKIP_DATASET_EDIT" != true ]]; then
        echo
        echo "[delete_episodes] Step 1/2: removing $(python3 -c "import json,sys;print(len(json.loads(sys.argv[1])))" "$DELETE_EPISODES") episode(s) via lerobot-edit-dataset..."
        # Call lerobot-edit-dataset directly (it's installed as a CLI entry
        # point at python -m lerobot.scripts.lerobot_edit_dataset:main).
        # `--repo_id` is REQUIRED for delete_episodes (the script validates
        # this — `--root` alone isn't enough since it needs the canonical
        # repo identifier for metadata updates). We omit `--root` and let
        # the script resolve `~/.cache/huggingface/lerobot/<repo_id>`
        # automatically; that's where the dataset actually lives.
        # Don't use `uv run` — `lerobot-edit-dataset` is installed as a
        # CLI in the active conda/pip env, and uv isn't always in PATH.
        lerobot-edit-dataset \
            --repo_id "$INT_REPO_ID" \
            --operation.type delete_episodes \
            --operation.episode_indices "$DELETE_EPISODES"
        echo
        echo "[delete_episodes] Step 2/2: refreshing rel-action stats sidecar..."
        bash "$SCRIPT_DIR/compute_relative_stats.sh" --dataset_repo="$INT_REPO_ID"
    fi

    echo
    echo "[delete_episodes] rm -rf downstream artifacts..."
    total_rm=$(( n_train + n_blends + n_blend_stats ))
    for p in "${NUKE_TRAIN_DIRS[@]:-}" "${NUKE_BLENDS[@]:-}" "${NUKE_BLEND_STATS[@]:-}"; do
        [[ -z "$p" ]] && continue
        rm -rf "$p"
    done
    echo "[delete_episodes] Deleted $total_rm downstream item(s)."
    echo
    echo "[delete_episodes] DONE. Next steps:"
    echo "  1. Verify with: python3 my_scripts/dagger_detect_dataset_anomalies.py \\"
    echo "       --dataset_root $INT_DATASET_PATH --no_expand_to_lineage"
    echo "  2. Re-run the orchestrator (or sweep wrapper) with --resume to retrain"
    echo "     rounds ${TARGET_ROUND}..${NUM_ROUNDS} against the cleaned dataset."
    exit 0
fi

ORIG_ARGV=()
CFG="$TRAIN_DIR/dagger/config.json"
if [[ -f "$CFG" ]]; then
    # Resolution path 1: sidecar exists → use recorded argv (exact).
    # NUL-delimited reading: a sidecar argv value can legitimately contain
    # embedded newlines (e.g. when the user's original shell command had a
    # line-break inside a quoted `--intervention_extra_args='...'` value;
    # bash preserves it, the orchestrator records it verbatim, and the
    # JSON encoder round-trips it as `\n`). Reading line-by-line via
    # `mapfile -t` would split that one argv element into multiple bash
    # array entries, then re-passing them to the orchestrator surfaces as
    # `Unknown argument: ...` on whatever fragment the split produced.
    # `\0` is illegal in argv strings (Linux exec contract), so it's a
    # safe delimiter that survives any in-value whitespace.
    mapfile -t -d '' ORIG_ARGV < <(python3 -c "
import json, sys
cfg = json.load(open(sys.argv[1]))
for a in cfg.get('orchestrator_invocation', {}).get('argv', []):
    sys.stdout.write(a)
    sys.stdout.write('\0')
" "$CFG")
    if (( ${#ORIG_ARGV[@]} == 0 )); then
        echo "[cleanup] $CFG exists but has no orchestrator_invocation.argv; falling back to auto-detect." >&2
    else
        echo "[cleanup] reusing original argv from sidecar: $CFG"
        # The sidecar's --num_rounds reflects the round count WHEN THIS SIDECAR
        # WAS WRITTEN, which can be SMALLER than the number of round training
        # dirs now on disk (the lineage was extended afterwards, or this sidecar
        # is from an earlier round). Cleaning with the stale value leaves the
        # higher rounds' training dirs (and datasets) behind — e.g. sidecar
        # says 10 but dag11..dag17 exist. Disk-scan the lineage for the true max
        # round and bump --num_rounds up to it so all rounds get cleaned. The
        # orchestrator's cleanup rm globs are `[[ -d ]]`-guarded, so
        # over-counting is a safe no-op. Mirrors the --delete_episodes path.
        _clean_lineage_base="$(basename "$TRAIN_DIR" | sed -E 's/(_ft)?_dag[0-9]+(_.*)?$//')"
        _clean_training_root="$(dirname "$TRAIN_DIR")"
        _disk_max_round=0
        for _d in "$_clean_training_root/${_clean_lineage_base}"_ft_dag[0-9]* \
                  "$_clean_training_root/${_clean_lineage_base}"_dag[0-9]*; do
            [[ -d "$_d" ]] || continue
            _rn=$(basename "$_d" | grep -oE 'dag[0-9]+' | head -1 | grep -oE '[0-9]+')
            [[ -n "$_rn" ]] && (( _rn > _disk_max_round )) && _disk_max_round=$_rn
        done
        if (( _disk_max_round > 0 )); then
            _sidecar_nr=""
            for _a in "${ORIG_ARGV[@]}"; do
                [[ "$_a" == --num_rounds=* ]] && _sidecar_nr="${_a#*=}"
            done
            if [[ -z "$_sidecar_nr" ]] || (( _disk_max_round > _sidecar_nr )); then
                echo "[cleanup] disk has round dirs up to dag${_disk_max_round}; overriding" \
                     "--num_rounds (sidecar said '${_sidecar_nr:-unset}') so ALL rounds are cleaned."
                _new_argv=()
                for _a in "${ORIG_ARGV[@]}"; do
                    [[ "$_a" == --num_rounds=* ]] && continue
                    _new_argv+=( "$_a" )
                done
                _new_argv+=( "--num_rounds=$_disk_max_round" )
                ORIG_ARGV=( "${_new_argv[@]}" )
            fi
        fi
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
    BASE_POLICY_NAME=$(echo "$TRAIN_BASENAME" | sed -E 's/(_ft)?_dag[0-9]+(_.*)?$//')
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
echo "  bash $ORCH ${FILTERED_ARGV[*]} --force_restart --cleanup_only ${ALSO_DELETE_BLENDS_FLAG[*]} ${FILTER_BLEND_COLLISIONS_FLAG[*]} ${PRESERVE_R1_FLAG[*]} ${FROM_ROUND_FLAG[*]} ${DRY_RUN_FLAG[*]}"
echo

if [[ "$AUTO_CONFIRM" == true ]]; then
    # Pipe the confirmation token. dagger_orchestrate.sh's --force_restart
    # block reads exactly one line, so a single "restart\n" is enough.
    printf 'restart\n' | bash "$ORCH" "${FILTERED_ARGV[@]}" --force_restart --cleanup_only "${ALSO_DELETE_BLENDS_FLAG[@]}" "${FILTER_BLEND_COLLISIONS_FLAG[@]}" "${PRESERVE_R1_FLAG[@]}" "${FROM_ROUND_FLAG[@]}" "${DRY_RUN_FLAG[@]}"
else
    bash "$ORCH" "${FILTERED_ARGV[@]}" --force_restart --cleanup_only "${ALSO_DELETE_BLENDS_FLAG[@]}" "${FILTER_BLEND_COLLISIONS_FLAG[@]}" "${PRESERVE_R1_FLAG[@]}" "${FROM_ROUND_FLAG[@]}" "${DRY_RUN_FLAG[@]}"
fi
