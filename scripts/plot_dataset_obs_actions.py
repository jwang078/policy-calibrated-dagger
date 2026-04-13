#!/usr/bin/env python
"""Plot observation.state vs action from a dataset episode, one subplot per joint.

Example:
    python my_scripts/plot_dataset_obs_actions.py --episode_index 306 --frame_index 2 --n_frames 20
"""

from __future__ import annotations

import argparse
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_DATASET_DIR = (
    Path.home() / ".cache/huggingface/lerobot/JennyWWW/splatsim_approach_lever_7_lowres_5path_10fails/data"
)
JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]


def load_frames(dataset_dir: Path, episode_index: int, frame_index: int, n_frames: int) -> pd.DataFrame:
    files = sorted(dataset_dir.rglob("*.parquet"))
    dfs = []
    for f in files:
        df = pd.read_parquet(f, filters=[("episode_index", "==", episode_index)])
        if len(df) > 0:
            dfs.append(df)
    df = pd.concat(dfs).sort_values("frame_index").reset_index(drop=True)
    mask = df["frame_index"] >= frame_index
    return df[mask].head(n_frames).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--episode_index", type=int, default=306)
    parser.add_argument("--frame_index", type=int, default=0)
    parser.add_argument("--n_frames", type=int, default=30)
    parser.add_argument("--output_path", type=Path, default=None)
    parser.add_argument("--no_show", action="store_true")
    args = parser.parse_args()

    df = load_frames(args.dataset_dir, args.episode_index, args.frame_index, args.n_frames)
    print(
        f"Loaded {len(df)} frames from episode {args.episode_index} starting at frame_index {args.frame_index}"
    )

    states = np.stack(df["observation.state"].apply(np.array).values)  # [T, 7]
    actions = np.stack(df["action"].apply(np.array).values)  # [T, 7]
    timesteps = df["frame_index"].values

    n_dims = states.shape[1]
    n_cols = 4
    n_rows = ceil(n_dims / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows), sharex=True)
    axes = np.array(axes).reshape(-1)

    for dim in range(n_dims):
        ax = axes[dim]
        ax.plot(
            timesteps,
            states[:, dim],
            color="steelblue",
            label="state[t]",
            linewidth=1.8,
            marker="o",
            markersize=3,
        )
        ax.plot(
            timesteps,
            actions[:, dim],
            color="tomato",
            label="action[t]",
            linewidth=1.8,
            marker="s",
            markersize=3,
            linestyle="--",
        )
        # action[t] vs state[t+1]: shift state forward by one
        ax.plot(
            timesteps[:-1],
            states[1:, dim],
            color="green",
            label="state[t+1]",
            linewidth=1.2,
            marker="^",
            markersize=3,
            linestyle=":",
        )

        ax2 = ax.twinx()
        delta_same = actions[:, dim] - states[:, dim]
        delta_shift = actions[:-1, dim] - states[1:, dim]
        ax2.bar(timesteps, delta_same, alpha=0.15, color="purple", label="action[t]−state[t]", width=0.4)
        ax2.bar(
            timesteps[:-1], delta_shift, alpha=0.15, color="green", label="action[t]−state[t+1]", width=0.4
        )
        ax2.set_ylabel("Δ", fontsize=7, color="gray")
        ax2.tick_params(axis="y", labelsize=7)
        ax.set_title(JOINT_NAMES[dim] if dim < len(JOINT_NAMES) else f"dim_{dim}", fontsize=9)
        ax.set_ylabel("rad")
        ax.grid(True, alpha=0.3)

    for idx in range(n_dims, len(axes)):
        axes[idx].set_visible(False)

    # Shared legend
    handles = [
        plt.Line2D([0], [0], color="steelblue", marker="o", markersize=4, label="state[t]"),
        plt.Line2D([0], [0], color="tomato", marker="s", markersize=4, linestyle="--", label="action[t]"),
        plt.Line2D([0], [0], color="green", marker="^", markersize=4, linestyle=":", label="state[t+1]"),
        plt.Rectangle((0, 0), 1, 1, color="purple", alpha=0.3, label="action[t] − state[t]"),
        plt.Rectangle((0, 0), 1, 1, color="green", alpha=0.3, label="action[t] − state[t+1]"),
    ]
    fig.legend(handles=handles, loc="lower right", fontsize=9, ncol=3)
    fig.suptitle(
        f"Episode {args.episode_index}, frames {args.frame_index}–{args.frame_index + len(df) - 1}",
        fontsize=12,
    )
    plt.tight_layout()

    if args.output_path:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(args.output_path), bbox_inches="tight", dpi=150)
        print(f"Saved → {args.output_path}")
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
