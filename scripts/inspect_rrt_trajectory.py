#!/usr/bin/env python3
"""Inspect the joint position / velocity / acceleration profiles of one or
more episodes from a lerobot dataset. Designed to answer:

    "Did my ruckig-parametrization change actually speed up the recorded
    RRT trajectories, or eliminate the per-corner zero-velocity stops?"

The dataset's `action` column IS the post-ruckig commanded trajectory at
control_hz — same numbers SplatSim/the robot consumed. So velocity =
gradient(action, dt) and acceleration = gradient(velocity, dt). Plotting
position + velocity + acceleration per joint surfaces the relevant
diagnostics immediately:

    - per-corner zero-velocity stops → distinct valleys at v=0 between
      otherwise-positive velocity peaks (the bug we just fixed in
      splatsim/utils/rrt_path_utils.py:ruckig_parametrize_path).
    - peak velocity → height of the highest velocity peak.
    - episode duration → total samples / fps.

Two usage modes:

  1. Single episode — plot the position/velocity/acceleration of one
     intervention's trajectory.
        python my_scripts/inspect_rrt_trajectory.py \\
            --dataset_path ~/.cache/huggingface/lerobot/JennyWWW/<name> \\
            --episode_id 14

  2. Compare two datasets at the same episode_id — overlays the curves
     so you can directly see "old recording" vs "new recording (with the
     ruckig fix)". The two datasets need the same DOF; episode lengths
     can differ.
        python my_scripts/inspect_rrt_trajectory.py \\
            --dataset_path /path/to/old_recording \\
            --compare_dataset_path /path/to/new_recording \\
            --episode_id 14

By default writes to outputs/rrt_inspect/<timestamp>.png; override with --out.
"""

import argparse
import glob
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


def load_episode_action(
    dataset_path: Path,
    episode_id: int,
) -> tuple[np.ndarray, dict]:
    """Read the `action` column for one episode from a lerobot dataset.
    Returns (actions[N, DOF], metadata-dict).

    Reads `meta/info.json` for control_hz (defaults to 30 if missing) +
    action dim names. Parquet rows are sorted by frame_index so the returned
    array is time-ordered.
    """
    import json

    info_path = dataset_path / "meta" / "info.json"
    info: dict = {}
    if info_path.is_file():
        with open(info_path) as f:
            info = json.load(f)
    fps = float(info.get("fps", 30))
    action_names: list[str] | None = None
    try:
        action_names = list(info["features"]["action"]["names"])
    except (KeyError, TypeError):
        pass

    pf_paths = sorted(glob.glob(str(dataset_path / "data/chunk-*/file-*.parquet")))
    if not pf_paths:
        raise FileNotFoundError(f"No parquet files under {dataset_path}/data")
    rows: list[tuple[int, list[float]]] = []
    for pf in pf_paths:
        tbl = pq.read_table(pf, columns=["action", "episode_index", "frame_index"])
        ep = tbl["episode_index"].to_pylist()
        fr = tbl["frame_index"].to_pylist()
        act = tbl["action"].to_pylist()
        for e, f, a in zip(ep, fr, act, strict=True):
            if int(e) == episode_id and a is not None:
                rows.append((int(f), a))
    if not rows:
        raise ValueError(
            f"Episode {episode_id} not found in {dataset_path}. "
            f"Check the dataset's meta/episodes for valid IDs."
        )
    rows.sort(key=lambda x: x[0])
    actions = np.asarray([r[1] for r in rows], dtype=np.float64)
    meta = {
        "n_frames": actions.shape[0],
        "dof": actions.shape[1],
        "fps": fps,
        "duration_s": actions.shape[0] / fps,
        "action_names": action_names,
        "frame_range": (rows[0][0], rows[-1][0]),
    }
    return actions, meta


def compute_derivatives(positions: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Numerical velocity + acceleration via central difference (gradient).
    Matches what a downstream controller sees after integrating the
    commanded position stream."""
    dt = 1.0 / fps
    vel = np.gradient(positions, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    return vel, acc


def plot_one_or_more(
    curves: dict[str, tuple[np.ndarray, float]],
    out_path: Path,
    title_lines: list[str],
    dim_labels: list[str] | None,
) -> None:
    """For each entry in `curves` (name → (actions[N, DOF], fps)), overlay
    its position/velocity/acceleration on a shared DOF×3 subplot grid."""
    any_actions = next(iter(curves.values()))[0]
    dof = any_actions.shape[1]
    if dim_labels is None or len(dim_labels) != dof:
        dim_labels = [f"joint_{j + 1}" for j in range(dof)]
    cmap = plt.get_cmap("tab10")
    colors = {name: cmap(i % 10) for i, name in enumerate(curves.keys())}

    fig, axes = plt.subplots(
        dof,
        3,
        figsize=(13, 1.8 * dof + 1.5),
        sharex=False,
        squeeze=False,
    )
    deriv_titles = ["position [rad]", "velocity [rad/s]", "acceleration [rad/s²]"]

    # Stash global y-ranges per (dim, deriv) so the panels of the same dim
    # share scale across datasets — visual comparison is otherwise
    # apples-to-oranges.
    dim_min = [[np.inf, np.inf, np.inf] for _ in range(dof)]
    dim_max = [[-np.inf, -np.inf, -np.inf] for _ in range(dof)]

    for name, (positions, fps) in curves.items():
        vel, acc = compute_derivatives(positions, fps)
        ts = np.arange(len(positions)) / fps
        for j in range(dof):
            axes[j, 0].plot(ts, positions[:, j], color=colors[name], label=name, linewidth=1.4)
            axes[j, 1].plot(ts, vel[:, j], color=colors[name], label=name, linewidth=1.4)
            axes[j, 2].plot(ts, acc[:, j], color=colors[name], label=name, linewidth=1.4)
            axes[j, 1].axhline(0, color="grey", linestyle=":", linewidth=0.7, alpha=0.5)
            for k, arr in enumerate((positions[:, j], vel[:, j], acc[:, j])):
                dim_min[j][k] = min(dim_min[j][k], float(arr.min()))
                dim_max[j][k] = max(dim_max[j][k], float(arr.max()))

    for j in range(dof):
        axes[j, 0].set_ylabel(dim_labels[j], fontsize=10)
        for k in range(3):
            axes[j, k].grid(True, alpha=0.3)
            lo, hi = dim_min[j][k], dim_max[j][k]
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                margin = 0.05 * (hi - lo)
                axes[j, k].set_ylim(lo - margin, hi + margin)
    for k, t in enumerate(deriv_titles):
        axes[0, k].set_title(t, fontsize=11)
        axes[-1, k].set_xlabel("time [s]")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(len(labels), 3),
        bbox_to_anchor=(0.5, 0.995),
        fontsize=10,
        frameon=False,
    )
    fig.suptitle("\n".join(title_lines), fontsize=10, y=0.97)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> int:
    p = argparse.ArgumentParser(
        prog="inspect_rrt_trajectory",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dataset_path",
        type=Path,
        required=True,
        help="Path to a lerobot dataset root (the dir containing "
        "`meta/` and `data/`). Typically "
        "~/.cache/huggingface/lerobot/<user>/<name>.",
    )
    p.add_argument(
        "--episode_id",
        type=int,
        required=True,
        help="Episode index to plot. See dataset's meta/episodes for valid IDs.",
    )
    p.add_argument(
        "--compare_dataset_path",
        type=Path,
        default=None,
        help="Optional second dataset to overlay at the same "
        "episode_id — for 'before vs after the ruckig fix' "
        "comparisons. Datasets must have the same DOF.",
    )
    p.add_argument(
        "--compare_episode_id",
        type=int,
        default=None,
        help="Episode in --compare_dataset_path to overlay. Defaults to --episode_id.",
    )
    p.add_argument("--out", type=Path, default=None, help="Output PNG. Default outputs/rrt_inspect/<ts>.png")
    args = p.parse_args()

    actions, meta = load_episode_action(args.dataset_path, args.episode_id)
    print(
        f"[main] {args.dataset_path.name} ep{args.episode_id}: "
        f"n_frames={meta['n_frames']}, dof={meta['dof']}, "
        f"duration={meta['duration_s']:.2f}s @ {meta['fps']:.0f}fps"
    )
    curves = {f"{args.dataset_path.name}/ep{args.episode_id}": (actions, meta["fps"])}
    title_meta = f"{args.dataset_path.name} ep{args.episode_id}  n_frames={meta['n_frames']}  duration={meta['duration_s']:.2f}s"

    if args.compare_dataset_path is not None:
        cmp_ep = args.compare_episode_id if args.compare_episode_id is not None else args.episode_id
        cmp_actions, cmp_meta = load_episode_action(args.compare_dataset_path, cmp_ep)
        print(
            f"[compare] {args.compare_dataset_path.name} ep{cmp_ep}: "
            f"n_frames={cmp_meta['n_frames']}, dof={cmp_meta['dof']}, "
            f"duration={cmp_meta['duration_s']:.2f}s @ {cmp_meta['fps']:.0f}fps"
        )
        if cmp_meta["dof"] != meta["dof"]:
            raise ValueError(
                f"DOF mismatch: {meta['dof']} (primary) vs {cmp_meta['dof']} (compare). "
                f"Can't overlay on the same per-joint axes."
            )
        curves[f"{args.compare_dataset_path.name}/ep{cmp_ep}"] = (cmp_actions, cmp_meta["fps"])
        speedup = meta["duration_s"] / max(cmp_meta["duration_s"], 1e-6)
        print(f"[compare] duration ratio (primary / compare) = {speedup:.2f}x")
        title_meta += (
            f"\nvs  {args.compare_dataset_path.name} ep{cmp_ep}  "
            f"n_frames={cmp_meta['n_frames']}  duration={cmp_meta['duration_s']:.2f}s  "
            f"(primary {speedup:.2f}x longer than compare)"
        )

    out_path = args.out
    if out_path is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = Path("outputs/rrt_inspect") / f"trajectory_{ts}.png"

    title_lines = [
        "Recorded action column from lerobot dataset (= post-ruckig RRT trajectory) — "
        "per-joint position / velocity / acceleration",
        title_meta,
        "valleys at v=0 between positive peaks => per-corner forced stops "
        "(the splatsim ruckig_parametrize_path bug we fixed)",
    ]
    plot_one_or_more(curves, out_path, title_lines, meta["action_names"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
