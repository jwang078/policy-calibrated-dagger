#!/usr/bin/env bash

# Retroactively write a minimal dagger/config.json sidecar into each existing
# training dir of a rerun-blends lineage that predates the orchestrator's
# auto-sidecar support. This is what dagger_plot.py reads to auto-pair the
# rerun lineage with its source for the per-metric overlay comparison plots.
#
# Usage:
#   bash my_scripts/dagger_retrofit_rerun_sidecar.sh \
#       --rerun_policy_basename=BASENAME \
#       --source_policy_basename=BASENAME \
#       --source_run_tag=TAG \
#       [--source_blends_tag=TAG] \
#       [--source_int_short_prefix=PREFIX] \
#       [--dry-run]
#
# Flags:
#   --rerun_policy_basename     The rerun lineage's BASE_POLICY_NAME
#                               (everything before `_ft_dag<N>` / `_dag<N>`).
#                               Example:
#                                 diffusion_approach_lever_11_biasend_5path_delta_basewrist_rerun_v1_b090_050
#   --source_policy_basename    The source lineage's BASE_POLICY_NAME.
#                               Example:
#                                 diffusion_approach_lever_11_biasend_5path_delta_basewrist_d5jvm
#   --source_run_tag            The source's --run_tag (e.g. d5jvm).
#   --source_blends_tag         Optional: the source's blends_tag (e.g. b090_080).
#                               Empty for a no-blends source.
#   --source_int_short_prefix   Optional: the source intervention prefix
#                               (e.g. lever_grip0_d5jvm_diff). Recorded for
#                               provenance; not used by dagger_plot.py.
#   --dry-run                   Print what would be written, don't write.
#
# Idempotency: if a sidecar with `rerun_mode` already exists, it's left alone.

set -euo pipefail

RERUN_POLICY_BASENAME=""
SOURCE_POLICY_BASENAME=""
SOURCE_RUN_TAG=""
SOURCE_BLENDS_TAG=""
SOURCE_INT_SHORT_PREFIX=""
DRY_RUN=false

for arg in "$@"; do
    case "$arg" in
        --rerun_policy_basename=*)    RERUN_POLICY_BASENAME="${arg#*=}" ;;
        --source_policy_basename=*)   SOURCE_POLICY_BASENAME="${arg#*=}" ;;
        --source_run_tag=*)           SOURCE_RUN_TAG="${arg#*=}" ;;
        --source_blends_tag=*)        SOURCE_BLENDS_TAG="${arg#*=}" ;;
        --source_int_short_prefix=*)  SOURCE_INT_SHORT_PREFIX="${arg#*=}" ;;
        --dry-run)                    DRY_RUN=true ;;
        -h|--help)
            sed -n '1,/^set -euo pipefail/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

if [[ -z "$RERUN_POLICY_BASENAME" || -z "$SOURCE_POLICY_BASENAME" || -z "$SOURCE_RUN_TAG" ]]; then
    echo "ERROR: required flags: --rerun_policy_basename, --source_policy_basename, --source_run_tag" >&2
    exit 1
fi

LEROBOT_ROOT="${LEROBOT_ROOT:-$HOME/code/lerobot}"

dirs=$( { ls -d "$LEROBOT_ROOT/outputs/training/${RERUN_POLICY_BASENAME}"_dag[0-9]*    2>/dev/null; \
          ls -d "$LEROBOT_ROOT/outputs/training/${RERUN_POLICY_BASENAME}"_ft_dag[0-9]* 2>/dev/null; \
        } | sort -u)

if [[ -z "$dirs" ]]; then
    echo "ERROR: no training dirs found matching ${RERUN_POLICY_BASENAME}{_ft,}_dag*" >&2
    echo "  Looked in: $LEROBOT_ROOT/outputs/training/" >&2
    exit 1
fi

n_dirs=$(echo "$dirs" | wc -l)
echo "Found $n_dirs training dir(s) for $RERUN_POLICY_BASENAME"
echo "Source pointer:  $SOURCE_POLICY_BASENAME (run_tag=$SOURCE_RUN_TAG, blends_tag='$SOURCE_BLENDS_TAG')"
echo

n_wrote=0
n_skipped=0
for d in $dirs; do
    sidecar="$d/dagger/config.json"
    if [[ -f "$sidecar" ]]; then
        if python3 -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get('rerun_mode') else 1)" "$sidecar" 2>/dev/null; then
            echo "  $sidecar: rerun_mode already set; skipping"
            n_skipped=$((n_skipped + 1))
            continue
        fi
    fi
    if [[ "$DRY_RUN" == true ]]; then
        echo "  [DRY-RUN] would write: $sidecar"
        n_wrote=$((n_wrote + 1))
        continue
    fi
    mkdir -p "$(dirname "$sidecar")"
    RPB="$RERUN_POLICY_BASENAME" \
    SPB="$SOURCE_POLICY_BASENAME" \
    SRT="$SOURCE_RUN_TAG" \
    SBT="$SOURCE_BLENDS_TAG" \
    SIP="$SOURCE_INT_SHORT_PREFIX" \
    OUT="$sidecar" \
    python3 - <<'PY'
import json, os, datetime
out = {
    "schema_version": 1,
    "retrofit": True,
    "retrofit_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "rerun_mode": {
        "source_run_tag":          os.environ["SRT"],
        "source_blends_tag":       os.environ["SBT"],
        "source_int_short_prefix": os.environ["SIP"],
        "source_policy_basename":  os.environ["SPB"],
    },
    "naming": {
        "base_policy_name": os.environ["RPB"],
    },
}
with open(os.environ["OUT"], "w") as f:
    json.dump(out, f, indent=2)
PY
    echo "  wrote: $sidecar"
    n_wrote=$((n_wrote + 1))
done

echo
echo "Summary: wrote=$n_wrote, skipped=$n_skipped, total=$n_dirs"
