#!/usr/bin/env python3
"""Build merged training datasets from base + per-ratio augmented datasets.

Part of the blending augmentation pipeline:

    augment_ratios_sweep.sh                     ← generates per-ratio augmented datasets
    merge_augmented_datasets_for_training.py    ← THIS SCRIPT (optional standalone use)
    train_sweep.sh                              ← trains on each merged dataset,
                                                   can call this script internally

Uses lerobot-edit-dataset's native merge operation under the hood so all
metadata (stats, tasks, video encoding) is handled correctly.

Naming convention (matches augment_ratios_sweep.sh):
    base:  {HF_USER}/{BASE_SHORT}
    aug:   {HF_USER}/{BASE_SHORT}_piabsden{NN}   e.g. piabsden02 for ratio 0.2
    merge: {HF_USER}/{BASE_SHORT}_base_piabsden{NN}[_{NN2}...]

Default mode — create ONE merged dataset from all given ratios:

    python my_scripts/merge_augmented_datasets_for_training.py \\
        --base  JennyWWW/splatsim_approach_lever_11_50failsrrtpi05 \\
        --ratios 0.2 0.4

    → creates: JennyWWW/splatsim_approach_lever_11_50failsrrtpi05_base_piabsden02_04

Sweep mode — create all cumulative intermediate merged datasets:

    python my_scripts/merge_augmented_datasets_for_training.py \\
        --base  JennyWWW/splatsim_approach_lever_11_50failsrrtpi05 \\
        --ratios 0.2 0.4 0.6 \\
        --sweep

    → creates three datasets:
        ...base_piabsden02
        ...base_piabsden02_04
        ...base_piabsden02_04_06

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


def build_merged_repo_id(base_repo_id: str, ratios: list[float]) -> str:
    """Return the canonical merged repo ID for the given cumulative ratio subset."""
    hf_user, base_name = base_repo_id.split("/", 1)
    tags = "_".join(ratio_to_tag(r) for r in ratios)
    return f"{hf_user}/{base_name}_base_piabsden{tags}"


def run_merge(base_repo_id: str, ratios: list[float], dry_run: bool = False) -> str:
    """Merge base + aug datasets for the given cumulative ratios into one new repo."""
    hf_user, base_name = base_repo_id.split("/", 1)
    sources = [base_repo_id] + [f"{hf_user}/{base_name}_piabsden{ratio_to_tag(r)}" for r in ratios]
    new_repo_id = build_merged_repo_id(base_repo_id, ratios)
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="Base dataset repo ID.")
    ap.add_argument(
        "--ratios", nargs="+", type=float, required=True, help="Forward-flow ratios. e.g. 0.2 0.4 0.6"
    )
    ap.add_argument(
        "--sweep",
        action="store_true",
        help="Create ALL cumulative intermediate merged datasets (one per ratio), not just the final one.",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.sweep:
        # Sweep mode: create base+0.2, base+0.2+0.4, base+0.2+0.4+0.6, …
        print(f"Sweep mode: {len(args.ratios)} merged datasets")
        for i, _ratio in enumerate(args.ratios):
            cumulative = args.ratios[: i + 1]
            new_repo = run_merge(args.base, cumulative, dry_run=args.dry_run)
            print(f"  → {new_repo}\n")
        print("Done.")
    else:
        # Default: create exactly ONE merged dataset from all provided ratios
        new_repo = run_merge(args.base, args.ratios, dry_run=args.dry_run)
        print(f"\nCreated: {new_repo}")
        print("\nTrain with:")
        print(f"  lerobot-train --dataset.repo_id={new_repo} ...")


if __name__ == "__main__":
    main()
