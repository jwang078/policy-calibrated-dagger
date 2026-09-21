#!/bin/bash
#
# Shared helpers for DAgger training-dir path resolution.
#
# Sourced by dagger_reeval_lineage.sh and dagger_cleanup_lineage.sh so both
# accept the same "run folder path" shorthand — an absolute path, a repo-
# relative path ("outputs/training/<basename>"), or a bare basename (looked
# up under outputs/training/). Also owns the inverse mapping "training-dir
# path → (model_prefix, kind, lineage_key)" used by reeval to convert
# positional paths into lineage matchers, delegating the round-dir
# inversion to `dagger_naming.py lineage_of` so there's exactly one owner
# of the naming rules.
#
# Contract for callers:
#   * `DGR_KNOWN_MODEL_PREFIXES` (space-separated) — override before source
#     if you need a superset. Default: "diffusion pi05 act".
#   * `DGR_ALLOW_MISSING=true` — let `dgr_normalize_train_dir` return a
#     path even if the target doesn't exist yet (used by cleanup's
#     `--delete_episodes` mode where the round dir may be pre-training).

: "${DGR_KNOWN_MODEL_PREFIXES:=diffusion pi05 act}"

DGR_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DGR_LEROBOT_ROOT="$(cd "$DGR_LIB_DIR/.." && pwd)"
DGR_TRAINING_ROOT_DEFAULT="$DGR_LEROBOT_ROOT/outputs/training"

# dgr_normalize_train_dir <path> [training_root]
#   Resolve <path> to an absolute path. Accepts:
#     * absolute path (used verbatim after trailing-slash trim)
#     * relative-to-cwd (if it exists on disk)
#     * bare basename (looked up under $training_root)
#     * repo-relative (looked up under $DGR_LEROBOT_ROOT)
#   Echoes the resolved absolute path.
#
#   By default returns 1 (with error to stderr) when nothing exists at the
#   resolved location. Set `DGR_ALLOW_MISSING=true` to instead compose an
#   absolute path from repo root / training root and return 0 anyway — used
#   by cleanup's `--delete_episodes` mode where the round dir may not have
#   been created yet.
dgr_normalize_train_dir() {
    local path="$1"
    local training_root="${2:-$DGR_TRAINING_ROOT_DEFAULT}"
    path="${path%/}"
    if [[ -z "$path" ]]; then
        echo "ERROR: dgr_normalize_train_dir: empty path" >&2
        return 1
    fi
    local resolved=""
    if [[ "$path" =~ ^/ ]]; then
        resolved="$path"
    elif [[ -d "$path" ]]; then
        resolved="$(cd "$path" && pwd)"
    elif [[ -d "$training_root/$path" ]]; then
        resolved="$training_root/$path"
    elif [[ -d "$DGR_LEROBOT_ROOT/$path" ]]; then
        resolved="$DGR_LEROBOT_ROOT/$path"
    else
        if [[ "${DGR_ALLOW_MISSING:-false}" == "true" ]]; then
            # Path doesn't exist yet — compose the most plausible absolute
            # location so downstream dirname/basename operations are stable.
            if [[ "$path" == */* ]]; then
                resolved="$DGR_LEROBOT_ROOT/$path"
            else
                resolved="$training_root/$path"
            fi
        else
            echo "ERROR: cannot resolve to a directory (tried cwd, $training_root/, $DGR_LEROBOT_ROOT/): $path" >&2
            return 1
        fi
    fi
    if [[ ! -d "$resolved" && "${DGR_ALLOW_MISSING:-false}" != "true" ]]; then
        echo "ERROR: not a directory: $resolved" >&2
        return 1
    fi
    printf '%s\n' "$resolved"
    return 0
}

# dgr_model_prefix_of <basename>
#   Echo the model prefix (from $DGR_KNOWN_MODEL_PREFIXES) that <basename>
#   starts with (e.g. "diffusion" for "diffusion_planar_3joint_..."). Empty
#   line if no known prefix matches. Never returns non-zero — callers check
#   for empty stdout.
dgr_model_prefix_of() {
    local bn="$1"
    local mp
    for mp in $DGR_KNOWN_MODEL_PREFIXES; do
        if [[ "$bn" == "${mp}_"* ]]; then
            printf '%s\n' "$mp"
            return 0
        fi
    done
    printf '\n'
    return 0
}

# dgr_lineage_from_train_dir <path>
#   Given a training-dir path (already normalized), echo three lines:
#     1. model prefix ("diffusion" / "pi05" / "act")
#     2. kind:
#        - "round" — path IS a DAgger round dir (`_dag<N>` / `_ft_dag<N>`)
#        - "base"  — path is NOT a round dir; treat as a base-policy dir
#                    whose downstream lineages should be enumerated
#     3. lineage key:
#        - For "round": inverse via `dagger_naming.py lineage_of`, e.g.
#          "diffusion_lever_grip0_d5jvm_ft_dag5" → "lever_grip0_d5jvm".
#          Callers use this for an EXACT lineage match.
#        - For "base": "<basename> minus <prefix>_". Callers use this as an
#          exact-prefix filter — a downstream lineage "planar_3joint_dbase_d100"
#          matches base "planar_3joint_dbase" because the lineage starts
#          with "planar_3joint_dbase_".
#   Returns 1 (with error to stderr) if the basename doesn't start with a
#   known model prefix, or if `lineage_of` returned empty for a round-shaped
#   basename.
dgr_lineage_from_train_dir() {
    local path="$1"
    local bn
    bn="$(basename "$path")"
    local model
    model="$(dgr_model_prefix_of "$bn")"
    if [[ -z "$model" ]]; then
        echo "ERROR: basename doesn't start with a known model prefix ($DGR_KNOWN_MODEL_PREFIXES): $bn" >&2
        return 1
    fi
    local rest="${bn#${model}_}"
    if [[ "$rest" =~ (_ft)?_dag[0-9]+ ]]; then
        local lineage
        lineage="$(python3 "$DGR_LIB_DIR/dagger_naming.py" lineage_of --model="$model" "$bn" 2>/dev/null || true)"
        if [[ -z "$lineage" ]]; then
            echo "ERROR: dagger_naming.lineage_of returned empty for basename '$bn' (model=$model)" >&2
            return 1
        fi
        printf '%s\n%s\n%s\n' "$model" "round" "$lineage"
    else
        printf '%s\n%s\n%s\n' "$model" "base" "$rest"
    fi
    return 0
}
