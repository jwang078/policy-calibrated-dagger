#!/usr/bin/env bash

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
#   --dataset_short=STR    Short dataset name (without the "splatsim_" prefix)
#   --dry-run              Print commands without executing

# ============================================================
# USER CONFIG (defaults — override via --dataset_short=...)
# ============================================================
DATASET_SHORT="approach_lever_11_50failsrrtpi05"
EXCLUDE_JOINTS="['gripper']"
# ============================================================

DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --dry-run)          DRY_RUN=true ;;
        --dataset_short=*)  DATASET_SHORT="${arg#*=}" ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

DATASET_REPO="JennyWWW/splatsim_${DATASET_SHORT}"
DATASET_CACHE=~/.cache/huggingface/lerobot/${DATASET_REPO}
STATS_JSON="${DATASET_CACHE}/meta/stats.json"
STATS_DIR=~/code/lerobot/outputs/dataset_stats/${DATASET_SHORT}

echo "Dataset : $DATASET_REPO"
echo "Stats dir: $STATS_DIR"
echo ""

if [[ "$DRY_RUN" == false ]]; then
    mkdir -p "$STATS_DIR"
fi

# ── pi05 stats (chunk_size=50) ────────────────────────────────
echo "Computing pi05 relative-action stats (chunk_size=50)..."
if [[ "$DRY_RUN" == false ]]; then
    lerobot-edit-dataset \
        --repo_id "$DATASET_REPO" \
        --operation.type recompute_stats \
        --operation.relative_action true \
        --operation.chunk_size 50 \
        --operation.relative_exclude_joints "${EXCLUDE_JOINTS}"
    cp "$STATS_JSON" "${STATS_DIR}/stats_pi05_rel50.json"
    echo "Saved → ${STATS_DIR}/stats_pi05_rel50.json"
else
    echo "[dry-run] would save → ${STATS_DIR}/stats_pi05_rel50.json"
fi
echo ""

# ── diffusion stats (chunk_size=8) ───────────────────────────
echo "Computing diffusion relative-action stats (chunk_size=8)..."
if [[ "$DRY_RUN" == false ]]; then
    lerobot-edit-dataset \
        --repo_id "$DATASET_REPO" \
        --operation.type recompute_stats \
        --operation.relative_action true \
        --operation.chunk_size 8 \
        --operation.relative_exclude_joints "${EXCLUDE_JOINTS}"
    cp "$STATS_JSON" "${STATS_DIR}/stats_diffusion_rel8.json"
    echo "Saved → ${STATS_DIR}/stats_diffusion_rel8.json"
else
    echo "[dry-run] would save → ${STATS_DIR}/stats_diffusion_rel8.json"
fi
echo ""

echo "============================================================"
echo "Done. To use these stats, set USE_RELATIVE_ACTIONS=true in train_sweep.sh."
