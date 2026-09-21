"""Plot the first-N-frames joint-position deltas of each episode, overlayed
across one or more LeRobot datasets.

Motivation:
    When RRT interventions are timed via `ruckig_parametrize_path` with the
    default `start_vel=zeros`, every RRT-driven episode begins with a
    "dead stop" — the first few frames have near-zero ||q_t - q_{t-1}||
    while ruckig accelerates from rest. Base teleop episodes, in contrast,
    typically start with continuous motion if the demonstrator was already
    moving when recording began.

    Visualizing this onset-delta pattern reveals whether the RRT
    intervention dataset is teaching the policy a "freeze-then-redirect"
    motif that doesn't exist in the base teleop distribution. The
    diffusion policy's 2-frame observation history makes velocity
    observable, so a distribution mismatch here would translate to
    stuttering rollouts after deployment.

Usage:
    # Compare a base dataset against an intervention dataset
    python my_scripts/plot_episode_onset_deltas.py \\
        --dataset_roots \\
            ~/.cache/huggingface/lerobot/JennyWWW/approach_lever_11_biasend_5path_grip0 \\
            ~/.cache/huggingface/lerobot/JennyWWW/lever_g0_d30_coll_03dag_diff_r_dag7 \\
        --labels base intervention_dag7

    # Compare multiple intervention rounds against base
    python my_scripts/plot_episode_onset_deltas.py \\
        --dataset_roots \\
            ~/.cache/huggingface/lerobot/JennyWWW/approach_lever_11_biasend_5path_grip0 \\
            ~/.cache/huggingface/lerobot/JennyWWW/lever_g0_d30_coll_03dag_diff_r_dag1 \\
            ~/.cache/huggingface/lerobot/JennyWWW/lever_g0_d30_coll_03dag_diff_r_dag7

Output: outputs/dagger/episode_onset_deltas_<timestamp>.png
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Reuse the parquet loader from the existing state-deltas script.
sys.path.insert(0, str(Path(__file__).parent))
from plot_state_deltas import load_state_episodes  # type: ignore[import-not-found]


def per_episode_onset_deltas(
    state: np.ndarray,
    episode: np.ndarray,
    n_onset: int,
) -> tuple[np.ndarray, int]:
    """For each episode, compute ||q_t - q_{t-1}||_2 over the first ``n_onset``
    valid frames. Returns a ``[n_episodes_used, n_onset]`` matrix; episodes
    shorter than ``n_onset+1`` frames are dropped (we need at least one
    delta per requested onset frame).

    Returns (matrix, n_skipped) — n_skipped reports how many episodes were
    too short to include.
    """
    rows: list[np.ndarray] = []
    n_skipped = 0
    for ep_idx in np.unique(episode):
        ep_mask = episode == ep_idx
        ep_state = state[ep_mask]
        if ep_state.shape[0] < n_onset + 1:
            n_skipped += 1
            continue
        # frame-to-frame joint-space delta over the first n_onset transitions.
        # Each row is one episode's "delta at frame k = ||q_k - q_{k-1}||_2"
        # for k in 1..n_onset.
        deltas = ep_state[1 : n_onset + 1] - ep_state[:n_onset]
        rows.append(np.linalg.norm(deltas, axis=1))
    if not rows:
        return np.empty((0, n_onset)), n_skipped
    return np.stack(rows, axis=0), n_skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_roots",
        nargs="+",
        required=True,
        help="One or more LeRobotDataset root directories.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Per-dataset display labels. Defaults to basename of each path.",
    )
    parser.add_argument(
        "--n_onset_frames",
        type=int,
        default=30,
        help="Number of starting frames per episode to plot (default 30).",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=200,
        help="Cap on episodes per dataset (random sample if more). Default 200.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for the episode subsample.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output PNG path. Defaults to outputs/dagger/episode_onset_deltas_<timestamp>.png",
    )
    args = parser.parse_args()

    if args.labels is not None and len(args.labels) != len(args.dataset_roots):
        parser.error(
            f"--labels has {len(args.labels)} entries but --dataset_roots has {len(args.dataset_roots)}"
        )
    labels = args.labels or [Path(r).name for r in args.dataset_roots]
    rng = np.random.default_rng(args.seed)

    # Layout: 1 row, 2 columns — left = per-episode overlay, right = mean ± std.
    # Each dataset gets a distinct color used in both panels.
    fig, (ax_overlay, ax_summary) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(args.dataset_roots), 10)))

    for idx, (root, label) in enumerate(zip(args.dataset_roots, labels)):
        try:
            state, episode, _names = load_state_episodes(root)
        except Exception as e:
            print(f"[skip] {label}: failed to load — {type(e).__name__}: {e}")
            continue
        matrix, n_skipped = per_episode_onset_deltas(state, episode, args.n_onset_frames)
        if matrix.shape[0] == 0:
            print(f"[skip] {label}: no episodes long enough for n_onset_frames={args.n_onset_frames}")
            continue
        # Subsample episodes if too many — keeps the overlay panel readable.
        if matrix.shape[0] > args.max_episodes:
            sel = rng.choice(matrix.shape[0], size=args.max_episodes, replace=False)
            matrix_plot = matrix[sel]
        else:
            matrix_plot = matrix
        color = colors[idx]
        # Per-episode thin lines (left panel). Low alpha so overlapping episodes
        # show up as a density cloud rather than a wall of color.
        x = np.arange(1, args.n_onset_frames + 1)
        for ep_row in matrix_plot:
            ax_overlay.plot(x, ep_row, color=color, alpha=0.08, linewidth=0.8)
        # Mean ± std (right panel) using the FULL matrix (not the subsample) for
        # accurate statistics regardless of plotting limit.
        mean_curve = matrix.mean(axis=0)
        std_curve = matrix.std(axis=0)
        ax_summary.plot(
            x,
            mean_curve,
            color=color,
            linewidth=2.0,
            label=f"{label}  (n={matrix.shape[0]} ep, skipped {n_skipped})",
        )
        ax_summary.fill_between(
            x,
            mean_curve - std_curve,
            mean_curve + std_curve,
            color=color,
            alpha=0.18,
        )
        print(
            f"{label}: {matrix.shape[0]} episodes used, {n_skipped} skipped "
            f"(too short). Onset mean delta at frame 1: {mean_curve[0] * 1000:.3f} mrad/step; "
            f"at frame 5: {mean_curve[4] * 1000:.3f} mrad/step."
        )

    ax_overlay.set_title(f"Per-episode onset deltas (first {args.n_onset_frames} frames)")
    ax_overlay.set_xlabel("Frame index since episode start")
    ax_overlay.set_ylabel(r"$\|q_t - q_{t-1}\|_2$  (radians / step)")
    ax_overlay.grid(True, alpha=0.3)

    ax_summary.set_title("Mean ± std across episodes")
    ax_summary.set_xlabel("Frame index since episode start")
    ax_summary.grid(True, alpha=0.3)
    ax_summary.legend(loc="upper right", fontsize=9)

    fig.suptitle(
        "Joint-position delta at episode onset — base vs intervention\n"
        r"Look for a near-zero plateau in the first $\sim$5 frames of "
        'intervention datasets; that\'s the ruckig $v_0{=}0$ "dead stop" signature.',
    )
    fig.tight_layout()

    if args.out is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("outputs/dagger")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"episode_onset_deltas_{ts}.png"
    else:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
