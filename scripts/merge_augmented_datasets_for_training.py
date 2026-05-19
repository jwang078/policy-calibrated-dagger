#!/usr/bin/env python3
"""Build merged training datasets from base + per-ratio augmented datasets.

Part of the blending augmentation pipeline:

    augment_ratios_sweep.sh                     ← generates per-ratio augmented datasets
    merge_augmented_datasets_for_training.py    ← THIS SCRIPT (optional standalone use)
    train_sweep.sh                              ← trains on each merged dataset,
                                                   can call this script internally

Uses lerobot-edit-dataset's native merge operation under the hood so all
metadata (stats, tasks, video encoding) is handled correctly.

Naming convention (matches augment_ratios_sweep.sh's generalized suffixes):
    base:  {HF_USER}/{BASE_SHORT}
    aug (ratio>0): {HF_USER}/{BASE_SHORT}_{MODEL}{ACTION_FORMAT}{BLEND_TAG}{NN}
                   e.g. _pirelden02 for pi05 + relative + denoise + ratio 0.2
    aug (ratio=0): {HF_USER}/{BASE_SHORT}_{MODEL}{ACTION_FORMAT}00
                   (no blend tag — ratio=0 is a no-op alias of the source)
    merge: {HF_USER}/{BASE_SHORT}_base_{MODEL}{ACTION_FORMAT}{BLEND_TAG}{NN}[_{NN2}...]
           (cumulative-ratio sweep) OR a user-provided --output_repo_id.

Three modes:

(A) Default — create ONE merged dataset from all given ratios:

    python my_scripts/merge_augmented_datasets_for_training.py \\
        --base  JennyWWW/splatsim_approach_lever_11_50failsrrtpi05 \\
        --ratios 0.2 0.4

(B) Sweep mode — create all cumulative intermediate merged datasets:

    python my_scripts/merge_augmented_datasets_for_training.py \\
        --base  JennyWWW/splatsim_approach_lever_11_50failsrrtpi05 \\
        --ratios 0.2 0.4 0.6 \\
        --sweep

(C) Explicit-sources mode — used by dagger_orchestrate.sh to merge a base
    with an arbitrary list of per-round intervention aliases. Naming is
    explicit, no `_piabsden{NN}` derivation:

    python my_scripts/merge_augmented_datasets_for_training.py \\
        --base JennyWWW/splatsim_..._base \\
        --extra_sources JennyWWW/splatsim_..._dag1_pirel00 \\
                        JennyWWW/splatsim_..._dag2_pirel00 \\
        --output_repo_id JennyWWW/splatsim_..._dag2_merged

Train against one:
    lerobot-train --dataset.repo_id=JennyWWW/..._base_piabsden02 ...

Note: train_sweep.sh calls this script in default (non-sweep) mode for each
cumulative subset and deletes the merged dataset after training to save disk.
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def ratio_to_tag(ratio: float) -> str:
    """Convert ratio float to 2-digit tag: 0.2 → '02', 1.0 → '10'."""
    return f"{int(round(ratio * 10)):02d}"


_BLEND_STRATEGY_TO_TAG = {"denoise": "den", "interpolate": "lerp"}


def aug_suffix(ratio: float, model: str, action_format: str, blend_strategy: str) -> str:
    """Build the per-ratio aug-dataset suffix matching augment_ratios_sweep.sh.

    Ratio=0 omits the blend tag (no blending happens — alias of source dataset).
    Ratio>0 includes the blend tag (``den`` / ``lerp``).
    """
    tag = ratio_to_tag(ratio)
    if tag == "00":
        return f"{model}{action_format}{tag}"
    blend_tag = _BLEND_STRATEGY_TO_TAG.get(blend_strategy, blend_strategy)
    return f"{model}{action_format}{blend_tag}{tag}"


def build_merged_repo_id(
    base_repo_id: str,
    ratios: list[float],
    model: str,
    action_format: str,
    blend_strategy: str,
) -> str:
    """Return the canonical merged repo ID for the given cumulative ratio subset."""
    hf_user, base_name = base_repo_id.split("/", 1)
    suffixes = "_".join(aug_suffix(r, model, action_format, blend_strategy) for r in ratios)
    return f"{hf_user}/{base_name}_base_{suffixes}"


def run_merge(
    sources: list[str],
    new_repo_id: str,
    dry_run: bool = False,
) -> str:
    """Invoke `lerobot-edit-dataset --operation.type=merge` for a fixed set of sources."""
    repo_ids_arg = "[" + ",".join(f'"{s}"' for s in sources) + "]"
    cmd = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_edit_dataset",
        f"--new_repo_id={new_repo_id}",
        "--operation.type=merge",
        f"--operation.repo_ids={repo_ids_arg}",
    ]
    if dry_run:
        print(f"[DRY-RUN] {' '.join(cmd)}")
    else:
        print(f"+ {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    return new_repo_id


def merge_by_ratios(
    base_repo_id: str,
    ratios: list[float],
    model: str,
    action_format: str,
    blend_strategy: str,
    output_repo_id: str | None,
    dry_run: bool = False,
) -> str:
    """Mode A: derive aug-dataset names from --ratios and merge base + augs."""
    hf_user, base_name = base_repo_id.split("/", 1)
    sources = [base_repo_id] + [
        f"{hf_user}/{base_name}_{aug_suffix(r, model, action_format, blend_strategy)}" for r in ratios
    ]
    if output_repo_id is None:
        output_repo_id = build_merged_repo_id(base_repo_id, ratios, model, action_format, blend_strategy)
    return run_merge(sources, output_repo_id, dry_run=dry_run)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="Base dataset repo ID.")
    ap.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        default=None,
        help="Forward-flow ratios for ratio-derived aug dataset names. e.g. 0.2 0.4 0.6. "
        "Mutually exclusive with --extra_sources.",
    )
    ap.add_argument(
        "--extra_sources",
        nargs="+",
        default=None,
        help="Explicit list of additional source repo_ids to merge with --base. "
        "Use when sources don't follow the ratio-derived naming "
        "(e.g. DAgger intervention aliases). Mutually exclusive with --ratios.",
    )
    ap.add_argument(
        "--output_repo_id",
        default=None,
        help="Explicit name for the merged output repo. When unset, derived "
        "from --base and --ratios (the legacy behavior). Required when "
        "using --extra_sources because the naming can't be auto-derived.",
    )
    ap.add_argument(
        "--model",
        default="pi",
        help="Model tag for naming: pi / diff / act (default: pi). Matches "
        "augment_ratios_sweep.sh's --model.",
    )
    ap.add_argument(
        "--action_format",
        default="rel",
        choices=["abs", "rel"],
        help="Action format for naming: abs / rel (default: rel). Matches "
        "augment_ratios_sweep.sh's --action_format.",
    )
    ap.add_argument(
        "--blend_strategy",
        default="denoise",
        help="Blend strategy used to produce the aug datasets — affects naming "
        "only (denoise → den, interpolate → lerp). Default: denoise.",
    )
    ap.add_argument(
        "--sweep",
        action="store_true",
        help="Create ALL cumulative intermediate merged datasets (one per ratio), "
        "not just the final one. Only valid with --ratios.",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if (args.ratios is None) == (args.extra_sources is None):
        ap.error("Pass exactly one of --ratios or --extra_sources.")

    if args.extra_sources is not None:
        # Mode C: explicit sources. No naming derivation; --output_repo_id required.
        if args.sweep:
            ap.error("--sweep is incompatible with --extra_sources (cumulative naming requires --ratios).")
        if args.output_repo_id is None:
            ap.error("--output_repo_id is required when using --extra_sources (can't be auto-derived).")
        sources = [args.base, *args.extra_sources]
        new_repo = run_merge(sources, args.output_repo_id, dry_run=args.dry_run)
        print(f"\nCreated: {new_repo}")
        print("Sources merged:")
        for s in sources:
            print(f"  - {s}")
        return

    # --ratios path.
    if args.sweep:
        print(f"Sweep mode: {len(args.ratios)} merged datasets")
        for i, _ratio in enumerate(args.ratios):
            cumulative = args.ratios[: i + 1]
            new_repo = merge_by_ratios(
                args.base,
                cumulative,
                args.model,
                args.action_format,
                args.blend_strategy,
                output_repo_id=None,  # sweep always derives names
                dry_run=args.dry_run,
            )
            print(f"  → {new_repo}\n")
        print("Done.")
    else:
        new_repo = merge_by_ratios(
            args.base,
            args.ratios,
            args.model,
            args.action_format,
            args.blend_strategy,
            output_repo_id=args.output_repo_id,
            dry_run=args.dry_run,
        )
        print(f"\nCreated: {new_repo}")
        print("\nTrain with:")
        print(f"  lerobot-train --dataset.repo_id={new_repo} ...")


if __name__ == "__main__":
    main()
