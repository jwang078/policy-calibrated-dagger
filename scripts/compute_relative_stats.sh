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
#   --chunk_sizes=N,M,...  Comma-separated chunk sizes to compute. Each N must
#                          match the target policy's TRAINING horizon (the length
#                          of the rel-action sequence the model is trained on),
#                          NOT its inference-time n_action_steps. E.g. a diffusion
#                          policy with horizon=64 / n_action_steps=32 needs
#                          --chunk_sizes=64: rel deltas at horizon positions
#                          30-63 are much larger than at positions 0-31, and
#                          sizing stats to the shorter window undershoots the
#                          real target range → up to ~half of training targets
#                          normalize outside [-1,+1] and get clipped when the
#                          diffusion policy has clip_sample=True → policy learns
#                          to produce tiny actions and barely moves at inference.
#                          For pi0 (horizon == n_action_steps) either value works.
#                          Multiple sizes computed sequentially. dagger_orchestrate.sh
#                          reads policy.horizon and passes the exact size needed
#                          on demand.
#                          Default: "50,64" — pi0 horizon + diffusion horizon.
#   --dry-run              Print commands without executing

# ============================================================
# USER CONFIG (defaults — override via --dataset_repo=...)
# ============================================================
DATASET_REPO="JennyWWW/splatsim_approach_lever_11_50failsrrtpi05"
EXCLUDE_JOINTS="['gripper']"
CHUNK_SIZES="50,64"  # pi0 horizon (=50) + diffusion horizon (=64)
# ============================================================

DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --dry-run)          DRY_RUN=true ;;
        --dataset_repo=*)   DATASET_REPO="${arg#*=}" ;;
        --chunk_sizes=*)    CHUNK_SIZES="${arg#*=}" ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

# Validate + split the comma-separated chunk sizes. Refuse empty / non-integer
# so a typo can't silently produce zero sidecars.
IFS=',' read -ra _CHUNKS_ARR <<< "$CHUNK_SIZES"
if [[ ${#_CHUNKS_ARR[@]} -eq 0 ]]; then
    echo "ERROR: --chunk_sizes must contain at least one integer (got '$CHUNK_SIZES')." >&2
    exit 1
fi
for _c in "${_CHUNKS_ARR[@]}"; do
    if ! [[ "$_c" =~ ^[0-9]+$ ]] || [[ "$_c" -le 0 ]]; then
        echo "ERROR: --chunk_sizes entries must be positive integers (got '$_c' in '$CHUNK_SIZES')." >&2
        exit 1
    fi
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
for CHUNK in "${_CHUNKS_ARR[@]}"; do
    OUT="${STATS_DIR}/stats_rel${CHUNK}.json"
    echo "Computing relative-action stats (chunk_size=${CHUNK})..."
    if [[ "$DRY_RUN" == false ]]; then
        # IMPORTANT: --new_repo_id must MATCH --repo_id and --operation.overwrite
        # must be true, otherwise lerobot-edit-dataset writes the rel-computed
        # stats to <repo>_recomputed_stats/meta/stats.json (a full COPY of the
        # dataset) and leaves the original meta/stats.json unchanged. The `cp`
        # below reads from the ORIGINAL, so without in-place mode the copied
        # sidecar would silently be the abs stats — the bug this fixes.
        # In-place still creates a backup (LeRobotDataset copies aside before
        # rewriting), so no data loss risk.
        lerobot-edit-dataset \
            --repo_id "$DATASET_REPO" \
            --new_repo_id "$DATASET_REPO" \
            --operation.type recompute_stats \
            --operation.relative_action true \
            --operation.chunk_size "$CHUNK" \
            --operation.relative_exclude_joints "${EXCLUDE_JOINTS}" \
            --operation.overwrite true
        cp "$STATS_JSON" "$OUT"
        echo "Saved → $OUT"
    else
        echo "[dry-run] would save → $OUT"
    fi
    echo ""
done

echo "============================================================"
echo "Done. To use these stats, set USE_RELATIVE_ACTIONS=true in train_sweep.sh."
