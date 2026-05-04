#!/usr/bin/env bash

# Computes relative-action normalization stats for each policy's chunk size and saves
# them to ~/lerobot_stats/<dataset>/ so that train_sweep.sh can reference them via
# --dataset.stats_path without needing to store the dataset twice.
#
# Run this once per dataset before training with USE_RELATIVE_ACTIONS=true in train_sweep.sh.
# Usage: ./compute_relative_stats.sh [--dry-run]

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

# ============================================================
# USER CONFIG — keep in sync with train_sweep.sh
# ============================================================
DATASET_SHORT="approach_lever_11_50failsrrtpi05"
# DATASET_SHORT="approach_lever_11_biasend_5path"
# DATASET_SHORT="approach_lever_10_rectify_5path"
# DATASET_SHORT="approach_lever_9_rectify_5path"
# DATASET_SHORT="approach_lever_8_fisheye_trim_5path"
# DATASET_SHORT="approach_lever_7_lowres_5path_10fails"
# DATASET_SHORT="approach_lever_7_lowres_5path"
# DATASET_SHORT="approach_lever_6_noteleport_5path"
EXCLUDE_JOINTS="['gripper']"
# ============================================================

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
