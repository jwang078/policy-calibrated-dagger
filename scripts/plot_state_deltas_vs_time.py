"""Plot per-step delta magnitude of observation.state vs frames-from-end of episode.

Shows the median / IQR / 5-95% band of ||Δstate|| and per-joint |Δ| as a function
of how far a frame is from the last frame of its episode, to see whether the
"approach end" regime (small but non-zero deltas) is well represented.

Usage:
    python my_scripts/plot_state_deltas_vs_time.py \
        --dataset-root /home/jennyw2/.cache/huggingface/lerobot/JennyWWW/splatsim_approach_lever_7_lowres_5path \
        --out my_scripts/plots/state_deltas_vs_time.png
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


def load_sorted(dataset_root: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str] | None]:
    files = sorted(glob.glob(os.path.join(dataset_root, "data", "chunk-*", "file-*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files under {dataset_root}/data")

    state_list: list[np.ndarray] = []
    ep_list: list[np.ndarray] = []
    fr_list: list[np.ndarray] = []
    for f in files:
        t = pq.read_table(f, columns=["observation.state", "episode_index", "frame_index"])
        df = t.to_pandas().sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
        state_list.append(np.stack(df["observation.state"].to_numpy()))
        ep_list.append(df["episode_index"].to_numpy())
        fr_list.append(df["frame_index"].to_numpy())

    state = np.concatenate(state_list, axis=0).astype(np.float32)
    episode = np.concatenate(ep_list, axis=0).astype(np.int64)
    frame = np.concatenate(fr_list, axis=0).astype(np.int64)

    # global sort in case files are interleaved
    order = np.lexsort((frame, episode))
    state = state[order]
    episode = episode[order]
    frame = frame[order]

    joint_names: list[str] | None = None
    info_path = os.path.join(dataset_root, "meta", "info.json")
    if os.path.exists(info_path):
        with open(info_path) as fp:
            info = json.load(fp)
        joint_names = info.get("features", {}).get("observation.state", {}).get("names")
    return state, episode, frame, joint_names


def compute_per_step_deltas(state: np.ndarray, episode: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (deltas (M, D), frames_from_end (M,)) where M = total intra-episode transitions.

    frames_from_end is computed for the source frame t of delta state[t+1]-state[t].
    Last frame of each episode has no outgoing delta.
    """
    # per-episode last frame_index
    unique_eps, counts = np.unique(episode, return_counts=True)
    ep_len = dict(zip(unique_eps, counts, strict=False))

    diff = np.diff(state, axis=0)
    same_ep = np.diff(episode) == 0
    deltas = diff[same_ep]

    # position within episode of the source frame
    pos_in_ep = np.zeros_like(episode)
    idx = 0
    for _ep, c in zip(unique_eps, counts, strict=False):
        pos_in_ep[idx : idx + c] = np.arange(c)
        idx += c
    frames_from_end = np.array([ep_len[e] - 1 - p for e, p in zip(episode, pos_in_ep, strict=False)])
    frames_from_end_src = frames_from_end[:-1][same_ep]
    return deltas, frames_from_end_src


def bin_stats(
    x: np.ndarray, y: np.ndarray, max_x: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-integer-x median, 25/75 and 5/95 percentiles, and count."""
    xs = np.arange(0, max_x + 1)
    med = np.full_like(xs, np.nan, dtype=np.float64)
    p25 = np.full_like(xs, np.nan, dtype=np.float64)
    p75 = np.full_like(xs, np.nan, dtype=np.float64)
    p05 = np.full_like(xs, np.nan, dtype=np.float64)
    p95 = np.full_like(xs, np.nan, dtype=np.float64)
    cnt = np.zeros_like(xs, dtype=np.int64)
    for i, xv in enumerate(xs):
        sel = x == xv
        if sel.any():
            yy = y[sel]
            med[i] = np.median(yy)
            p25[i], p75[i] = np.percentile(yy, [25, 75])
            p05[i], p95[i] = np.percentile(yy, [5, 95])
            cnt[i] = yy.size
    return xs, med, p25, p75, p05, p95, cnt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--out", default="my_scripts/plots/state_deltas_vs_time.png")
    p.add_argument("--max-frames-from-end", type=int, default=120)
    args = p.parse_args()

    state, episode, frame, joint_names = load_sorted(args.dataset_root)
    deltas, fend = compute_per_step_deltas(state, episode)
    norms = np.linalg.norm(deltas, axis=1)

    max_x = min(args.max_frames_from_end, int(fend.max()))
    d = deltas.shape[1]
    names = joint_names if joint_names and len(joint_names) == d else [f"dim_{i}" for i in range(d)]

    ncols = 4
    nrows = int(np.ceil((d + 1) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows))
    axes = axes.flatten()

    # panel 0: overall norm
    ax = axes[0]
    xs, med, p25, p75, p05, p95, cnt = bin_stats(fend, norms, max_x)
    ax.fill_between(xs, p05, p95, color="orange", alpha=0.15, label="5–95%")
    ax.fill_between(xs, p25, p75, color="orange", alpha=0.35, label="25–75%")
    ax.plot(xs, med, color="darkorange", linewidth=1.5, label="median")
    ax.set_xlabel("frames from end of episode")
    ax.set_ylabel("||Δstate||")
    ax.set_title("||Δstate|| vs frames-from-end")
    ax.invert_xaxis()  # closer to end on the right
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    # per-joint |Δ|
    for i in range(d):
        ax = axes[i + 1]
        xs, med, p25, p75, p05, p95, cnt = bin_stats(fend, np.abs(deltas[:, i]), max_x)
        ax.fill_between(xs, p05, p95, color="steelblue", alpha=0.15)
        ax.fill_between(xs, p25, p75, color="steelblue", alpha=0.35)
        ax.plot(xs, med, color="navy", linewidth=1.5)
        ax.set_xlabel("frames from end")
        ax.set_ylabel(f"|Δ {names[i]}|")
        ax.set_title(f"{names[i]}")
        ax.invert_xaxis()
        ax.grid(alpha=0.3)

    for j in range(d + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"per-step delta magnitude vs frames-from-end — {args.dataset_root.split('/')[-1]}\n"
        f"{deltas.shape[0]:,} transitions, {np.unique(episode).size} episodes "
        f"(x-axis: distance to episode end, right = end)",
        fontsize=11,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
