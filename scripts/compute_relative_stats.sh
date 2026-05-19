#!/usr/bin/env bash
set -euo pipefail

# Computes relative-action normalization stats for each policy's chunk size and saves
# them to ~/lerobot_stats/<dataset>/ so that train_sweep.sh can reference them via
# --dataset.stats_path without needing to store the dataset twice.
#
# Run this once per dataset before training with USE_RELATIVE_ACTIONS=true in train_sweep.sh.
#
# Usage:
#   bash my_scripts/compute_relative_stats.sh [OPTIONS]
#
# Options:
#   --dataset_repo=ID      Full dataset repo id, e.g.
#                          "JennyWWW/splatsim_approach_lever_11_50failsrrtpi05".
#                          DATASET_SHORT (used to derive the stats sidecar dir)
#                          is inferred by stripping "JennyWWW/" and an optional
#                          "splatsim_" prefix. Mirrors train_sweep.sh's flag.
#   --dry-run              Print commands without executing

# ============================================================
# USER CONFIG (defaults — override via --dataset_repo=...)
# ============================================================
DATASET_REPO="JennyWWW/splatsim_approach_lever_11_50failsrrtpi05"
EXCLUDE_JOINTS="['gripper']"
# ============================================================

DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --dry-run)          DRY_RUN=true ;;
        --dataset_repo=*)   DATASET_REPO="${arg#*=}" ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

# Derive DATASET_SHORT from the repo for the stats sidecar dir naming. Strip
# the "JennyWWW/" prefix and an optional "splatsim_" prefix so dag datasets
# (named JennyWWW/foo_dag1_m without the splatsim_ prefix) land in a sidecar
# dir of just "foo_dag1_m".
DATASET_SHORT="${DATASET_REPO#*/}"
DATASET_SHORT="${DATASET_SHORT#splatsim_}"
DATASET_CACHE=~/.cache/huggingface/lerobot/${DATASET_REPO}
STATS_JSON="${DATASET_CACHE}/meta/stats.json"
STATS_DIR=~/code/lerobot/outputs/dataset_stats/${DATASET_SHORT}

echo "Dataset : $DATASET_REPO"
echo "Stats dir: $STATS_DIR"
echo ""

if [[ "$DRY_RUN" == false ]]; then
    mkdir -p "$STATS_DIR"
fi

# Relative-action stats files are named by their chunk size (which is what they
# actually depend on — the policy type doesn't matter, only the chunk over which
# the action deltas are computed). Consumers look up the correct file using
# their policy's chunk_size / n_action_steps.
for CHUNK in 50 8; do
    OUT="${STATS_DIR}/stats_rel${CHUNK}.json"
    echo "Computing relative-action stats (chunk_size=${CHUNK})..."
    if [[ "$DRY_RUN" == false ]]; then
        lerobot-edit-dataset \
            --repo_id "$DATASET_REPO" \
            --operation.type recompute_stats \
            --operation.relative_action true \
            --operation.chunk_size "$CHUNK" \
            --operation.relative_exclude_joints "${EXCLUDE_JOINTS}"
        cp "$STATS_JSON" "$OUT"
        echo "Saved → $OUT"
    else
        echo "[dry-run] would save → $OUT"
    fi
    echo ""
done

echo "============================================================"
echo "Done. To use these stats, set USE_RELATIVE_ACTIONS=true in train_sweep.sh."
