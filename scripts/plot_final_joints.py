"""Plot the spread of final-frame joint positions across all episodes in a LeRobot dataset.

Usage:
    python plot_final_joints.py [--repo_id JennyWWW/splatsim_approach_lever_10_rectify_5path]
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_DATASET_DIR = (
    "/home/jennyw2/.cache/huggingface/lerobot/JennyWWW/splatsim_approach_lever_10_rectify_5path"
)


def load_final_frames(dataset_dir: str) -> pd.DataFrame:
    """Read all data parquets, return one row per episode (the last frame)."""
    parquets = sorted(glob.glob(os.path.join(dataset_dir, "data", "chunk-*", "file-*.parquet")))
    if not parquets:
        raise FileNotFoundError(f"No parquets under {dataset_dir}/data")
    print(f"Reading {len(parquets)} parquet file(s)...")

    cols = ["episode_index", "frame_index", "observation.state", "action"]
    dfs = [pd.read_parquet(p, columns=cols) for p in parquets]
    df = pd.concat(dfs, ignore_index=True)

    # Last frame per episode
    last_idx = df.groupby("episode_index")["frame_index"].idxmax()
    last = df.loc[last_idx].sort_values("episode_index").reset_index(drop=True)
    print(f"Loaded {len(last)} episodes.")
    return last


def to_array(col: pd.Series) -> np.ndarray:
    return np.stack([np.asarray(v, dtype=np.float64) for v in col.values])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", default=DEFAULT_DATASET_DIR)
    ap.add_argument("--out", default="/home/jennyw2/code/lerobot/my_scripts/plots/final_joint_spread.png")
    args = ap.parse_args()

    last = load_final_frames(args.dataset_dir)
    state = to_array(last["observation.state"])  # (N_eps, 7)
    action = to_array(last["action"])  # (N_eps, 7)

    joint_names = [f"joint_{i + 1}" for i in range(6)] + ["gripper"]

    # Print stats (joints in radians; convert to degrees for the 6 joints)
    print("\n=== observation.state at the LAST frame of each episode ===")
    print(f"{'dim':<10} {'min':>10} {'max':>10} {'spread':>10} {'std':>10} {'unit':>6}")
    for i, name in enumerate(joint_names):
        col = state[:, i]
        unit = "deg" if i < 6 else "frac"
        scale = 180.0 / np.pi if i < 6 else 1.0
        print(
            f"{name:<10} {col.min() * scale:>10.3f} {col.max() * scale:>10.3f} "
            f"{(col.max() - col.min()) * scale:>10.3f} {col.std() * scale:>10.4f} {unit:>6}"
        )

    print("\n=== action at the LAST frame of each episode ===")
    print(f"{'dim':<10} {'min':>10} {'max':>10} {'spread':>10} {'std':>10} {'unit':>6}")
    for i, name in enumerate(joint_names):
        col = action[:, i]
        unit = "deg" if i < 6 else "frac"
        scale = 180.0 / np.pi if i < 6 else 1.0
        print(
            f"{name:<10} {col.min() * scale:>10.3f} {col.max() * scale:>10.3f} "
            f"{(col.max() - col.min()) * scale:>10.3f} {col.std() * scale:>10.4f} {unit:>6}"
        )

    # state - action (the controller lag at the last frame)
    print("\n=== (state - action) at the LAST frame (controller lag) ===")
    diff = state - action
    print(f"{'dim':<10} {'mean':>10} {'std':>10} {'unit':>6}")
    for i, name in enumerate(joint_names):
        unit = "deg" if i < 6 else "frac"
        scale = 180.0 / np.pi if i < 6 else 1.0
        print(f"{name:<10} {diff[:, i].mean() * scale:>10.4f} {diff[:, i].std() * scale:>10.4f} {unit:>6}")

    # Plot histograms for the 6 joints (degrees) + gripper (fraction)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for i, name in enumerate(joint_names):
        ax = axes[i]
        unit = "deg" if i < 6 else "frac"
        scale = 180.0 / np.pi if i < 6 else 1.0
        ax.hist(state[:, i] * scale, bins=40, alpha=0.7, label="state", color="C0")
        ax.hist(action[:, i] * scale, bins=40, alpha=0.5, label="action", color="C1")
        spread = (state[:, i].max() - state[:, i].min()) * scale
        ax.set_title(f"{name} (state spread = {spread:.2f} {unit})")
        ax.set_xlabel(unit)
        ax.legend(fontsize=8)
    axes[-1].axis("off")
    fig.suptitle(
        f"Final-frame joint spread across {len(state)} episodes\n({os.path.basename(args.dataset_dir)})"
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"\nSaved histogram to {args.out}")


if __name__ == "__main__":
    main()
