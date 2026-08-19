#!/usr/bin/env python
"""Visualize DART relabeling in a blend dataset.

For each requested episode of a (relabeled) blend dataset, plots against its
source intervention episode (paired via ``source_episode_idx`` metadata):

  * per-joint time series — executed state, stored action (the label track),
    and the source demo;
  * joint-space phase planes (j1/j2, j2/j3) — demo path, executed path, and
    correction arrows from visited states to their labels (the DART
    geometry: arrows should point from the perturbed path back toward the
    demo corridor);
  * a summary strip — deviation from the demo corridor and label-correction
    magnitude |action - state| over time.

Works on executed-label blends too (arrows then show the executed action's
step instead of a correction — useful for comparing the two label modes).

Usage:
    python my_scripts/visualize_blend_relabel.py \
        --blend_repo_id JennyWWW/planar_12_..._blend010 \
        --source_repo_id JennyWWW/planar_12_03dag_diff_r_dag1 \
        --episode_index 0 [--out out.png]
"""

from __future__ import annotations

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_episode(repo_id: str, ep: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (states, actions) arm-dims for one episode."""
    root = os.path.expanduser(f"~/.cache/huggingface/lerobot/{repo_id}")
    files = sorted(glob.glob(root + "/data/**/*.parquet", recursive=True))
    df = pd.concat([pd.read_parquet(f) for f in files])
    g = df[df.episode_index == ep]
    if len(g) == 0:
        raise SystemExit(f"{repo_id}: episode {ep} not found")
    return (
        np.stack(g["observation.state"].to_numpy()),
        np.stack(g["action"].to_numpy()),
    )


def _source_episode_for(blend_repo: str, ep: int) -> int:
    root = os.path.expanduser(f"~/.cache/huggingface/lerobot/{blend_repo}")
    m = pd.read_parquet(root + "/meta/episodes/chunk-000/file-000.parquet")
    row = m[m.episode_index == ep]
    if len(row) and "source_episode_idx" in row.columns:
        return int(row.iloc[0]["source_episode_idx"])
    return ep


def _plot_relabel_view(S_bl, A_bl, S_src, n, arrow_every, title, out):
    """Render the DART view for (executed states, labels, demo states)."""
    dev = np.linalg.norm(S_bl[:, None, :] - S_src[None, :, :], axis=2).min(axis=1)
    corr = np.linalg.norm(A_bl - S_bl, axis=1)

    n_planes = max(1, n - 1)
    fig, axes = plt.subplots(2, max(n, n_planes + 1), figsize=(5.5 * max(n, n_planes + 1), 9))
    fig.suptitle(f"{title}\ngrey=demo  blue=executed states  red=label track  arrows: state → label")

    t_bl = np.arange(len(S_bl))
    t_src = np.arange(len(S_src))
    for j in range(n):
        ax = axes[0][j]
        ax.plot(t_src, S_src[:, j], color="0.6", lw=3, label="demo (source states)")
        ax.plot(t_bl, S_bl[:, j], color="tab:blue", lw=1.2, label="executed state")
        ax.plot(t_bl, A_bl[:, j], color="tab:red", lw=1.0, ls="--", label="stored action")
        ax.set_title(f"joint_{j + 1}")
        ax.set_xlabel("tick")
        if j == 0:
            ax.legend(fontsize=8)

    for k in range(n_planes):
        jx, jy = k, k + 1
        ax = axes[1][k]
        ax.plot(S_src[:, jx], S_src[:, jy], color="0.6", lw=3, label="demo path")
        ax.plot(S_bl[:, jx], S_bl[:, jy], color="tab:blue", lw=1.2, label="executed path")
        idx = np.arange(0, len(S_bl), max(1, arrow_every))
        ax.quiver(
            S_bl[idx, jx],
            S_bl[idx, jy],
            (A_bl - S_bl)[idx, jx],
            (A_bl - S_bl)[idx, jy],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color="tab:red",
            width=0.004,
            label="state → label",
        )
        ax.set_xlabel(f"joint_{jx + 1}")
        ax.set_ylabel(f"joint_{jy + 1}")
        ax.set_title(f"phase plane j{jx + 1}/j{jy + 1} with correction arrows")
        if k == 0:
            ax.legend(fontsize=8)

    ax = axes[1][n_planes]
    ax.plot(t_bl, dev, color="tab:blue", label="deviation from demo corridor")
    ax.plot(t_bl, corr, color="tab:red", label="|action − state| (label correction)")
    ax.set_xlabel("tick")
    ax.set_ylabel("rad")
    ax.set_title("deviation vs label-correction magnitude")
    ax.legend(fontsize=8)
    for col in range(n_planes + 1, axes.shape[1]):
        axes[1][col].axis("off")

    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"saved → {out}")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--blend_repo_id", default=None, help="dataset mode: relabeled blend repo")
    ap.add_argument(
        "--npz",
        default=None,
        help="npz mode: a visualize_shared_autonomy_sim rollout_data.npz — the "
        "chosen ratio's commanded track is relabeled ON THE FLY (DART labels "
        "computed here, commanded track as the state proxy)",
    )
    ap.add_argument("--ratio", default="0.50", help="npz mode: which ratio track (e.g. 0.50)")
    ap.add_argument("--source_repo_id", required=True)
    ap.add_argument("--episode_index", type=int, default=0)
    ap.add_argument("--source_episode_index", type=int, default=None, help="override the metadata pairing")
    ap.add_argument("--arrow_every", type=int, default=5)
    ap.add_argument("--num_arm_joints", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    n = args.num_arm_joints

    if args.npz:
        z = np.load(args.npz, allow_pickle=True)
        key = f"ratio_{float(args.ratio):.2f}"
        track = np.asarray(z[key], dtype=np.float64)
        track = track[~np.isnan(track).any(axis=1)]
        g_act = np.asarray(z["guidance_actions_raw"], dtype=np.float64)
        src_ep = args.source_episode_index if args.source_episode_index is not None else args.episode_index
        S_src_full, _ = _load_episode(args.source_repo_id, src_ep)
        # guidance rows are the source episode's tail (frames [n_obs:]) —
        # align the demo-state polyline to the guidance grid by length.
        off = max(0, len(S_src_full) - len(g_act))
        S_src = S_src_full[off : off + len(g_act)]
        from dart_labels import demo_geometry, per_frame_labels

        geom = demo_geometry(S_src, g_act, n_arm=n)
        _idxs, A_lb = per_frame_labels(track, geom, index_window=45)
        A_lb = A_lb.astype(np.float64)
        out = args.out or f"relabel_view_npz_{key}.png"
        _plot_relabel_view(
            track[:, :n],
            A_lb[:, :n],
            S_src[:, :n],
            n,
            args.arrow_every,
            f"Relabel view (on-the-fly) — {key} of {os.path.basename(os.path.dirname(args.npz))} "
            f"(source ep {src_ep}; commanded track as state proxy)",
            out,
        )
        return

    if not args.blend_repo_id:
        ap.error("pass either --blend_repo_id (dataset mode) or --npz (rollout mode)")
    src_ep = (
        args.source_episode_index
        if args.source_episode_index is not None
        else _source_episode_for(args.blend_repo_id, args.episode_index)
    )
    S_bl, A_bl = _load_episode(args.blend_repo_id, args.episode_index)
    S_src, _A_src = _load_episode(args.source_repo_id, src_ep)
    out = args.out or f"relabel_view_{args.blend_repo_id.split('/')[-1]}_ep{args.episode_index}.png"
    _plot_relabel_view(
        S_bl[:, :n],
        A_bl[:, :n],
        S_src[:, :n],
        n,
        args.arrow_every,
        f"Relabel view — blend ep {args.episode_index} (source ep {src_ep})",
        out,
    )


if __name__ == "__main__":
    main()
