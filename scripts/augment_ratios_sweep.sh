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
#   --policy_path=PATH      Policy checkpoint path (required when any ratio > 0)
#   --episode_range=RANGE   Episode range, e.g. "301-379" or "all"
#   --ratios="N N N"        Space-separated ratio list, e.g. "0.2 0.4 0.6"
#                           Special value 0.0 = NO-OP: skip SplatSim replay and
#                           just register the source dataset under the canonical
#                           suffix via a zero-copy hardlink alias. Useful for
#                           DAgger intervention datasets (which already pair
#                           obs with human/RRT actions — no replay needed).
#   --model=STR             Suffix tag for model class: "pi" / "diff" / "act"
#                           (default: pi). Replaces the old hardcoded "pi" in
#                           the _piabsden suffix.
#   --action_format=STR     Suffix tag for action format: "abs" or "rel"
#                           (default: rel — reflects use_relative_actions=true).
#                           Replaces the old hardcoded "abs" in the suffix.
#   --env_task=STR          Splatsim task name (ratio>0 only)
#   --env_port=INT          Splatsim ZMQ port (ratio>0 only)
#   --dry-run               Print commands without executing
#   --push                  Push output datasets to HuggingFace Hub
#
# Example (DAgger no-op merge):
#   bash my_scripts/augment_ratios_sweep.sh \
#       --dataset_short=approach_lever_7_lowres_5path_dag1 \
#       --ratios="0.0"
#   →  creates JennyWWW/splatsim_approach_lever_7_lowres_5path_dag1_pirel00 as
#      a hardlink alias of the source dataset, no SplatSim required.
#
# Example (full blending sweep):
#   bash my_scripts/augment_ratios_sweep.sh \
#       --dataset_short=approach_lever_11_50failsrrtpi05 \
#       --policy_path=outputs/training/pi05_.../checkpoints/006000/pretrained_model \
#       --ratios="0.2 0.4 0.6 0.8 1.0" \
#       --episode_range=301-379 \
#       --push
#
# Outputs:
#   ratio=0.0 (no-op):  JennyWWW/${SOURCE_SHORT}_${MODEL}${ACTION_FORMAT}00
#   ratio>0:            JennyWWW/${SOURCE_SHORT}_${MODEL}${ACTION_FORMAT}${BLEND_TAG}${NN}
#                       where BLEND_TAG is "den" (denoise) or "lerp" (interpolate)

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
MODEL="pi"
ACTION_FORMAT="rel"
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
        --model=*)            MODEL="${arg#*=}" ;;
        --action_format=*)    ACTION_FORMAT="${arg#*=}" ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

# Map blend_strategy to a short tag for the dataset suffix.
# Only used for ratio>0; ratio=0 omits the blend tag entirely.
case "$BLEND_STRATEGY" in
    denoise)     BLEND_TAG="den"  ;;
    interpolate) BLEND_TAG="lerp" ;;
    *)           BLEND_TAG="$BLEND_STRATEGY" ;;
esac

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

# Compute the per-ratio target name. Ratio=0 omits the blend tag entirely
# because no blending happens — the source dataset is used as-is via hardlink.
target_name_for_ratio() {
    local r="$1"
    local tag
    tag=$(ratio_to_tag "$r")
    if [[ "$tag" == "00" ]]; then
        echo "${SOURCE_SHORT}_${MODEL}${ACTION_FORMAT}${tag}"
    else
        echo "${SOURCE_SHORT}_${MODEL}${ACTION_FORMAT}${BLEND_TAG}${tag}"
    fi
}

echo "=== augment_ratios_sweep.sh ==="
echo "Source:        ${SOURCE_DATASET}"
echo "Policy:        ${POLICY_PATH}"
echo "Episodes:      ${EPISODE_RANGE}"
echo "Ratios:        ${RATIOS[*]}"
echo "Model tag:     ${MODEL}"
echo "Action format: ${ACTION_FORMAT}"
echo "Blend tag:     ${BLEND_TAG}  (used only for ratio>0)"
echo "Push:          ${PUSH_TO_HUB}"
echo

LEROBOT_CACHE="$HOME/.cache/huggingface/lerobot"

for RATIO in "${RATIOS[@]}"; do
    TAG=$(ratio_to_tag "$RATIO")
    TARGET_SHORT=$(target_name_for_ratio "$RATIO")
    TARGET_DATASET="${HF_USER}/${TARGET_SHORT}"

    echo "──────────────────────────────────────────────"
    echo "ratio=${RATIO}  →  ${TARGET_DATASET}"
    echo "──────────────────────────────────────────────"

    # Ratio=0 ⇒ NO-OP: skip the SplatSim closed-loop replay entirely. The
    # source dataset already pairs the right obs with the right actions
    # (e.g. DAgger interventions), so we just register it under the canonical
    # suffix via a hardlink alias. cp -r --link uses hardlinks where possible;
    # falls back to copies otherwise.
    if [[ "$TAG" == "00" ]]; then
        SRC_DIR="${LEROBOT_CACHE}/${SOURCE_DATASET}"
        DST_DIR="${LEROBOT_CACHE}/${TARGET_DATASET}"
        if [[ ! -d "$SRC_DIR" ]]; then
            if [[ "$DRY_RUN" == true ]]; then
                echo "[DRY-RUN] [ratio=0 no-op] would hardlink ${SRC_DIR} → ${DST_DIR} (source not on disk yet — expected to be created upstream)"
                echo
                continue
            fi
            echo "[ratio=0 no-op] source dataset not on disk at $SRC_DIR — aborting." >&2
            echo "  Run the upstream step that creates ${SOURCE_DATASET} first" >&2
            echo "  (e.g. lerobot-eval --intervention.method=rrt for DAgger datasets)." >&2
            exit 1
        fi
        if [[ -e "$DST_DIR" ]]; then
            echo "[ratio=0 no-op] alias already exists at $DST_DIR — leaving in place."
        else
            echo "[ratio=0 no-op] hardlinking ${SRC_DIR} → ${DST_DIR}"
            run mkdir -p "$(dirname "$DST_DIR")"
            run cp -r --link --reflink=auto "$SRC_DIR" "$DST_DIR"
        fi
        echo
        continue
    fi

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
    echo "  ${HF_USER}/$(target_name_for_ratio "$RATIO")  (ratio=${RATIO})"
done
echo
echo "Next step: run merge_augmented_datasets_for_training.py to create the"
echo "merged training dataset (zero-copy via hardlinks)."
