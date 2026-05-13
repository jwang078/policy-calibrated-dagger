#!/usr/bin/env bash
# augment_ratios_sweep.sh
#
# Generates one augmented dataset per forward_flow_ratio on top of a source
# intervention dataset.  Each run augments the same source episodes and writes
# the blended rollouts to a separate HuggingFace repo so they can later be
# merged (via merge_augmented_datasets_for_training.py) without disk duplication.
#
# Usage:
#   ./my_scripts/augment_ratios_sweep.sh [OPTIONS]
#
# All options have defaults (see USER CONFIG below).  Pass --flag=value to
# override any of them without editing the file.
#
# Options:
#   --dataset_short=STR     Short dataset name (default: approach_lever_11_50failsrrtpi05)
#   --policy_path=PATH      Policy checkpoint path
#   --episode_range=RANGE   Episode range, e.g. "301-379" or "all"
#   --ratios="N N N"        Space-separated ratio list, e.g. "0.2 0.4 0.6"
#   --env_task=STR          Splatsim task name
#   --env_port=INT          Splatsim ZMQ port
#   --dry-run               Print commands without executing
#   --push                  Push output datasets to HuggingFace Hub
#
# Example:
#   bash my_scripts/augment_ratios_sweep.sh \
#       --dataset_short=approach_lever_11_50failsrrtpi05 \
#       --policy_path=outputs/training/pi05_.../checkpoints/006000/pretrained_model \
#       --ratios="0.2 0.4 0.6 0.8 1.0" \
#       --episode_range=301-379 \
#       --push
#
# Outputs:
#   JennyWWW/${SOURCE_SHORT}_piabsden02   (ratio 0.2)
#   JennyWWW/${SOURCE_SHORT}_piabsden04   (ratio 0.4)
#   ...

set -euo pipefail

# ── USER CONFIG (defaults) ────────────────────────────────────────────────────
HF_USER="JennyWWW"
SOURCE_SHORT="splatsim_approach_lever_11_50failsrrtpi05"
EPISODE_RANGE="301-379"
POLICY_PATH="outputs/training/pi05_approach_lever_11_biasend_5path_delta_basewrist/checkpoints/006000/pretrained_model"
BLEND_STRATEGY="denoise"
GUIDANCE_REPR="absolute_pos"
BLEND_MODE="once_per_chunk"
ENV_TASK="upright_small_engine_new"
ENV_EXTERNAL_PORT=6001
RATIOS=(0.2 0.4 0.6 0.8 1.0)
DRY_RUN=false
PUSH_TO_HUB=false
# ─────────────────────────────────────────────────────────────────────────────

for arg in "$@"; do
    case "$arg" in
        --dry-run)            DRY_RUN=true ;;
        --push)               PUSH_TO_HUB=true ;;
        --dataset_short=*)    SOURCE_SHORT="splatsim_${arg#*=}" ;;
        --policy_path=*)      POLICY_PATH="${arg#*=}" ;;
        --episode_range=*)    EPISODE_RANGE="${arg#*=}" ;;
        --ratios=*)           IFS=' ' read -ra RATIOS <<< "${arg#*=}" ;;
        --env_task=*)         ENV_TASK="${arg#*=}" ;;
        --env_port=*)         ENV_EXTERNAL_PORT="${arg#*=}" ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

SOURCE_DATASET="${HF_USER}/${SOURCE_SHORT}"

ratio_to_tag() {
    # Convert float → 2-digit string: 0.2 → "02", 1.0 → "10"
    python3 -c "import sys; r=float(sys.argv[1]); print(f'{int(r*10):02d}')" "$1"
}

run() {
    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY-RUN] $*"
    else
        echo "+ $*"
        eval "$@"
    fi
}

echo "=== augment_ratios_sweep.sh ==="
echo "Source:  ${SOURCE_DATASET}"
echo "Policy:  ${POLICY_PATH}"
echo "Episodes: ${EPISODE_RANGE}"
echo "Ratios:  ${RATIOS[*]}"
echo "Push:    ${PUSH_TO_HUB}"
echo

for RATIO in "${RATIOS[@]}"; do
    TAG=$(ratio_to_tag "$RATIO")
    TARGET_SHORT="${SOURCE_SHORT}_piabsden${TAG}"
    TARGET_DATASET="${HF_USER}/${TARGET_SHORT}"

    echo "──────────────────────────────────────────────"
    echo "ratio=${RATIO}  →  ${TARGET_DATASET}"
    echo "──────────────────────────────────────────────"

    PUSH_FLAG=""
    [[ "$PUSH_TO_HUB" == true ]] && PUSH_FLAG="--push_to_hub"

    run python my_scripts/augment_dataset_with_blending.py \
        --policy_path="${POLICY_PATH}" \
        --dataset_repo_id="${SOURCE_DATASET}" \
        --target_dataset_repo_id="${TARGET_DATASET}" \
        --episode_index="${EPISODE_RANGE}" \
        --forward_flow_ratios="[${RATIO}]" \
        --blend_strategy="${BLEND_STRATEGY}" \
        --guidance_repr="${GUIDANCE_REPR}" \
        --blend_mode="${BLEND_MODE}" \
        --drain_chunk \
        --env_task="${ENV_TASK}" \
        --env_external_port="${ENV_EXTERNAL_PORT}" \
        ${PUSH_FLAG}

    echo
done

echo "=== Done.  Per-ratio datasets ==="
for RATIO in "${RATIOS[@]}"; do
    TAG=$(ratio_to_tag "$RATIO")
    echo "  ${HF_USER}/${SOURCE_SHORT}_piabsden${TAG}  (ratio=${RATIO})"
done
echo
echo "Next step: run merge_datasets_for_training.py to create local merged"
echo "training datasets (zero-copy via hardlinks)."
