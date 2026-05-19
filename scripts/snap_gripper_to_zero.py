#!/usr/bin/env python3
"""Create a copy of a LeRobot dataset with the gripper dim of observation.state
and action snapped to exactly 0, then recompute stats.

Background: dataset 11 has gripper state drifting up to ~0.11 due to physics
jitter, while action remains 0. Some policies treat the noisy state input as a
real signal. This script produces a cleaned copy where both state[-1] and
action[-1] are exactly 0 in every frame.

Stats (meta/stats.json and sidecar relative-action stats files) are
recomputed automatically so they reflect the cleaned data.

Usage:
    python my_scripts/snap_gripper_to_zero.py \\
        --src ~/.cache/huggingface/lerobot/JennyWWW/splatsim_approach_lever_11_biasend_5path \\
        --dst-repo-id JennyWWW/splatsim_approach_lever_11_biasend_5path_gripzero
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

# Match compute_relative_stats.sh
_PI05_CHUNK_SIZE = 50
_DIFFUSION_CHUNK_SIZE = 8
_EXCLUDE_JOINTS = "['gripper']"
_LEROBOT_HOME_DEFAULT = os.path.expanduser("~/.cache/huggingface/lerobot")


def snap_data(src: str, dst: str) -> None:
    """Copy meta wholesale and rewrite each data parquet with gripper snapped to 0."""
    if not os.path.isdir(src):
        sys.exit(f"Source dataset not found: {src}")
    if os.path.exists(dst):
        sys.exit(f"Destination already exists: {dst}  (delete it or pick a different --dst-repo-id)")

    print(f"Copying metadata structure: {src} → {dst}")
    shutil.copytree(os.path.join(src, "meta"), os.path.join(dst, "meta"))
    os.makedirs(os.path.join(dst, "data"), exist_ok=True)
    for f in os.listdir(src):
        full = os.path.join(src, f)
        if os.path.isfile(full):
            shutil.copy2(full, os.path.join(dst, f))

    src_files = sorted(glob.glob(os.path.join(src, "data", "chunk-*", "*.parquet")))
    print(f"Processing {len(src_files)} parquet files...")

    total_frames = 0
    total_state_changed = 0
    total_action_changed = 0

    def snap_last_dim(v):
        arr = np.asarray(v, dtype=np.float32).copy()
        arr[-1] = 0.0
        return arr

    for src_file in src_files:
        rel = os.path.relpath(src_file, src)
        dst_file = os.path.join(dst, rel)
        os.makedirs(os.path.dirname(dst_file), exist_ok=True)

        df = pd.read_parquet(src_file)

        if "observation.state" in df.columns:
            before = np.stack(df["observation.state"].to_numpy())[:, -1]
            df["observation.state"] = df["observation.state"].apply(snap_last_dim)
            total_state_changed += int((np.abs(before) > 1e-9).sum())
        if "action" in df.columns:
            before = np.stack(df["action"].to_numpy())[:, -1]
            df["action"] = df["action"].apply(snap_last_dim)
            total_action_changed += int((np.abs(before) > 1e-9).sum())

        total_frames += len(df)
        df.to_parquet(dst_file, index=False)

    print(f"\nFrames processed: {total_frames}")
    print(
        f"  observation.state[gripper] != 0 originally: {total_state_changed}  ({100 * total_state_changed / max(total_frames, 1):.1f}%)"
    )
    print(
        f"  action[gripper] != 0 originally:            {total_action_changed}  ({100 * total_action_changed / max(total_frames, 1):.1f}%)"
    )


def run_recompute_stats(repo_id: str, root: str, *, relative: bool, chunk_size: int = 50) -> None:
    """Shell out to lerobot-edit-dataset to recompute meta/stats.json in place."""
    cmd = [
        "lerobot-edit-dataset",
        "--repo_id",
        repo_id,
        "--root",
        root,
        "--operation.type",
        "recompute_stats",
    ]
    if relative:
        cmd += [
            "--operation.relative_action",
            "true",
            "--operation.chunk_size",
            str(chunk_size),
            "--operation.relative_exclude_joints",
            _EXCLUDE_JOINTS,
        ]
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def recompute_all_stats(dst_root: str, dst_repo_id: str, dataset_short: str) -> None:
    """Recompute base (absolute) stats and the two relative-action sidecars.

    Mirrors my_scripts/compute_relative_stats.sh. The dataset's own
    meta/stats.json ends up with absolute stats; sidecars in
    outputs/dataset_stats/<short>/ hold the relative variants for training.
    """
    sidecar_dir = os.path.join(
        os.path.expanduser("~"), "code", "lerobot", "outputs", "dataset_stats", dataset_short
    )
    os.makedirs(sidecar_dir, exist_ok=True)
    stats_json = os.path.join(dst_root, "meta", "stats.json")

    # Relative-action stats files are named by their chunk size (which is what
    # they actually depend on — policy type doesn't matter, only the chunk over
    # which action deltas are computed). Consumers look up the file using their
    # policy's chunk_size / n_action_steps.
    for chunk in (_PI05_CHUNK_SIZE, _DIFFUSION_CHUNK_SIZE):
        run_recompute_stats(dst_repo_id, dst_root, relative=True, chunk_size=chunk)
        sidecar = os.path.join(sidecar_dir, f"stats_rel{chunk}.json")
        shutil.copy2(stats_json, sidecar)
        print(f"Saved → {sidecar}")

    # 3) Final pass: restore absolute stats in meta/stats.json so this dataset's
    #    own stats reflect the cleaned absolute-action distribution. ACT and any
    #    other absolute-action consumer rely on meta/stats.json.
    run_recompute_stats(dst_repo_id, dst_root, relative=False)
    print("\nmeta/stats.json now contains absolute-action stats.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--src",
        required=True,
        help="Source dataset root (containing data/ and meta/)",
    )
    p.add_argument(
        "--dst-repo-id",
        required=True,
        help="Destination repo id, e.g. 'JennyWWW/splatsim_approach_lever_11_biasend_5path_gripzero'",
    )
    p.add_argument(
        "--lerobot-home",
        default=_LEROBOT_HOME_DEFAULT,
        help=f"Base dir where the destination dataset lives on disk (default: {_LEROBOT_HOME_DEFAULT})",
    )
    p.add_argument(
        "--skip-stats",
        action="store_true",
        help="Skip stats recomputation (data is still snapped). Useful for dry runs.",
    )
    args = p.parse_args()

    dst_root = os.path.join(args.lerobot_home, args.dst_repo_id)
    dataset_short = args.dst_repo_id.split("/")[-1].removeprefix("splatsim_")

    snap_data(args.src, dst_root)

    if not args.skip_stats:
        recompute_all_stats(dst_root, args.dst_repo_id, dataset_short)
    else:
        print(
            "\n[--skip-stats] Stats not recomputed. Run my_scripts/compute_relative_stats.sh manually if needed."
        )


if __name__ == "__main__":
    main()
