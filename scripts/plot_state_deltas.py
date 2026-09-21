"""Plot histogram of frame-to-frame observation.state deltas for a LeRobot dataset.

Reads all parquet files directly, groups by episode_index, computes per-joint
differences between consecutive frames (same episode only), and writes a per-joint
histogram plus an overall norm histogram.

Usage:
    python my_scripts/plot_state_deltas.py \
        --dataset-root /home/jennyw2/.cache/huggingface/lerobot/JennyWWW/splatsim_approach_lever_7_lowres_5path \
        --out my_scripts/plots/state_deltas.png
"""

from __future__ import annotations

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


def load_state_episodes(dataset_root: str) -> tuple[np.ndarray, np.ndarray, list[str] | None]:
    files = sorted(glob.glob(os.path.join(dataset_root, "data", "chunk-*", "file-*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files under {dataset_root}/data")

    state_list: list[np.ndarray] = []
    ep_list: list[np.ndarray] = []
    for f in files:
        t = pq.read_table(f, columns=["observation.state", "episode_index", "frame_index"])
        df = t.to_pandas()
        df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
        state_list.append(np.stack(df["observation.state"].to_numpy()))
        ep_list.append(df["episode_index"].to_numpy())

    state = np.concatenate(state_list, axis=0).astype(np.float32)
    episode = np.concatenate(ep_list, axis=0).astype(np.int64)

    # joint names from info.json if present
    joint_names: list[str] | None = None
    import json

    info_path = os.path.join(dataset_root, "meta", "info.json")
    if os.path.exists(info_path):
        with open(info_path) as fp:
            info = json.load(fp)
        joint_names = info.get("features", {}).get("observation.state", {}).get("names")
    return state, episode, joint_names


def compute_deltas(state: np.ndarray, episode: np.ndarray) -> np.ndarray:
    """Return (N-K, D) deltas = state[t+1] - state[t] within each episode."""
    diff = np.diff(state, axis=0)
    same_ep = np.diff(episode) == 0
    return diff[same_ep]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--out", default="my_scripts/plots/state_deltas.png")
    p.add_argument("--bins", type=int, default=120)
    args = p.parse_args()

    state, episode, joint_names = load_state_episodes(args.dataset_root)
    deltas = compute_deltas(state, episode)
    norms = np.linalg.norm(deltas, axis=1)

    d = deltas.shape[1]
    names = joint_names if joint_names and len(joint_names) == d else [f"dim_{i}" for i in range(d)]

    ncols = 4
    nrows = int(np.ceil((d + 1) / ncols))  # +1 for the norm panel
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = axes.flatten()

    for i in range(d):
        ax = axes[i]
        x = deltas[:, i]
        ax.hist(x, bins=args.bins, color="steelblue", alpha=0.85)
        ax.set_yscale("log")
        ax.set_title(f"{names[i]} — Δ per step")
        zero_frac = float(np.mean(np.abs(x) < 1e-6))
        ax.axvline(0.0, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.text(
            0.02,
            0.95,
            f"zero frac: {zero_frac:.2%}\nstd: {x.std():.4f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )

    ax = axes[d]
    ax.hist(norms, bins=args.bins, color="darkorange", alpha=0.85)
    ax.set_yscale("log")
    ax.set_title("||Δstate|| per step (all dims)")
    zero_frac = float(np.mean(norms < 1e-6))
    ax.text(
        0.98,
        0.95,
        f"zero frac: {zero_frac:.2%}\nmean: {norms.mean():.4f}",
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )

    for j in range(d + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"observation.state frame-to-frame deltas — {args.dataset_root.split('/')[-1]}\n"
        f"{deltas.shape[0]:,} transitions across {np.unique(episode).size} episodes",
        fontsize=11,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
