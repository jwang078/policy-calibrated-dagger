#!/usr/bin/env python
"""Visualize train-time DART chunk synthesis (dart_labels.chunk_labels).

For a set of anchor (conditioning) frames along a blend rollout, synthesizes
the full H-step expert label chunk the training dataloader would serve —
demo clock advancing one index per tick, corridor offset decaying at
``rate x med_step`` per tick — and plots:

  * per-joint traces on the DEMO-INDEX axis (projection-aligned): demo,
    executed track (each state at its projected demo index), and each
    anchor's chunk at its own demo clock i0+k — correct labels land ON the
    grey demo curve. (On a wall-tick axis a lagging rollout's chunks
    converge to a lag-shifted copy of the demo — a parallel offset line —
    which reads as an error but is just the robot being behind schedule;)
  * a wall-clock vs demo-clock panel making that lag explicit;
  * joint-space phase planes: chunk curves fanning from anchor states into
    the demo corridor — the geometry the policy actually learns;
  * chunk corridor-deviation vs position k (quintic S-decay to ZERO — the
    within-chunk convergence per-frame labels could not express);
  * chunk step-speed profile vs the demo's own cruise band (smoothness at
    every position, including the state->label_0 first step, shown at k=0).

npz mode (fast iteration — projection computed on the fly):
    python my_scripts/visualize_dart_chunks.py \
        --npz <run>/rollout_data.npz --ratio 0.5 \
        --source_repo_id JennyWWW/... --episode_index 0

dataset mode (validates the recorded ``relabel_demo_index`` end to end):
    python my_scripts/visualize_dart_chunks.py \
        --blend_repo_id JennyWWW/..._blend --source_repo_id JennyWWW/... \
        --episode_index 0
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
from dart_labels import DemoGeometry, _interp_rows, chunk_labels, demo_geometry, project_states

# ── figure text (EDIT ME — mirrors paper_plots/dart_chunks_src47/plot.py) ──
TEXT = {
    "suptitle": "",  # empty = one line identifying the dataset/episode
    "joint_group_title": "Corrections from Policy-Augmented Trajectory to Expert Intervention",
    "joint_titles": ["Joint 1 Corrections", "Joint 2 Corrections", "Joint 3 Corrections"],
    "clock_title": "Task Progress vs. Time",
    "phase_title": "Joint-Space Corrections",
    "zoom_title": "Recovery Chunk (Detail)",
    "dev_title": "Within-Chunk Convergence to the Demo",
    "speed_title": "Commanded Speed Along the Chunk",
    "xlabel_time": "Timestep",
    "xlabel_progress": "Demo Index (Progress-Aligned)",
    "ylabel_joint": "Joint{j} Position",
    "legend_demo": "Expert Intervention Trajectory",
    "legend_state": "Policy-Augmented Trajectory",
    "legend_demo_path": "Expert Intervention Trajectory",
    "legend_state_path": "Policy-Augmented Trajectory",
    "legend_chunk": "Label chunk",
    "legend_projection": "Corridor projection",
    "legend_pace": "Intervention pace (y = x)",
    "legend_progress": "Projected intervention index",
    "legend_cruise": "Intervention cruise p5–p95",
}


def _load_episode(repo_id: str, ep: int, cols=("observation.state", "action")) -> dict[str, np.ndarray]:
    root = os.path.expanduser(f"~/.cache/huggingface/lerobot/{repo_id}")
    files = sorted(glob.glob(root + "/data/**/*.parquet", recursive=True))
    df = pd.concat([pd.read_parquet(f) for f in files])
    g = df[df.episode_index == ep]
    if len(g) == 0:
        raise SystemExit(f"{repo_id}: episode {ep} not found")
    return {c: np.stack(g[c].to_numpy()) for c in cols if c in g.columns}


def _source_episode_for(blend_repo: str, ep: int) -> int:
    root = os.path.expanduser(f"~/.cache/huggingface/lerobot/{blend_repo}")
    m = pd.read_parquet(root + "/meta/episodes/chunk-000/file-000.parquet")
    row = m[m.episode_index == ep]
    if len(row) and "source_episode_idx" in row.columns:
        return int(row.iloc[0]["source_episode_idx"])
    return ep


def _plot(
    S: np.ndarray,
    idxs: np.ndarray,
    geom: DemoGeometry,
    n: int,
    horizon: int,
    rate: float,
    ease_out: float,
    fps: float,
    anchor_every: int,
    title: str,
    out: str,
) -> None:
    anchors = list(range(0, len(S) - 1, max(1, anchor_every)))

    def _smooth_vel(t: int, w: int = 3) -> np.ndarray:
        """Windowed velocity — a 1-tick FD on a noisy blend path points anywhere."""
        lo, hi = max(0, t - w), min(len(S) - 1, t + w)
        return (S[hi] - S[lo]) / max(1, hi - lo)

    chunks, infos = {}, {}
    for t in anchors:
        inf: dict = {}
        chunks[t] = chunk_labels(
            S[t],
            float(idxs[t]),
            geom,
            horizon=horizon,
            rate=rate,
            ease_out=ease_out,
            prev_state=S[t - 1] if t > 0 else None,
            velocity=_smooth_vel(t),
            info=inf,
        )
        infos[t] = inf

    def _chunk_clock(t: int) -> np.ndarray:
        """X-coords for anchor t's chunk: each label at its OWN demo clock.

        The label function reports its structure (info): the brake holds the
        clock at the anchor's index, the merge advances to the rendezvous
        i0+di, the fill advances one index per tick (or holds after an
        at-rest landing). Plotting at nearest-demo-point instead detached
        far-off-corridor chunks from their own anchor dot — the anchor sits
        at the CURSOR's index while 'nearest' is a geometric accident.
        """
        inf = infos.get(t, {})
        i0, end = float(idxs[t]), float(len(geom.P) - 1)
        h = len(chunks[t])
        if "t_merge" not in inf:  # pursuit fallback: demo pace from i0
            return np.minimum(i0 + np.arange(h), end)
        tb, tm = int(inf.get("t_brake", 0)), max(1, int(inf["t_merge"]))
        i_r = min(end, i0 + float(inf.get("di", 0.0)))
        m = min(tm, h - tb)
        xs = np.empty(h)
        for k in range(h):
            if k < tb:
                xs[k] = i0
            elif k < tb + m:
                xs[k] = i0 + (i_r - i0) * (k - tb + 1) / tm
            elif inf.get("end_clamped"):
                xs[k] = i_r
            else:
                xs[k] = min(i_r + (k - tb - m + 1), end)
        return xs

    cmap = plt.get_cmap("plasma")
    colors = {t: cmap(i / max(1, len(anchors) - 1)) for i, t in enumerate(anchors)}

    def _arrows_along(ax, C, jx, jy, color, ks, lw=1.4):
        """Arrowheads along curve C at positions ks — direction of time."""
        for k in ks:
            if k + 1 >= len(C):
                break
            p0, p1 = C[k, [jx, jy]], C[k + 1, [jx, jy]]
            if np.linalg.norm(p1 - p0) < 1e-6:
                continue
            ax.annotate(
                "",
                xy=p1,
                xytext=p0,
                arrowprops={"arrowstyle": "-|>", "color": color, "lw": lw, "shrinkA": 0, "shrinkB": 0},
            )

    # LAYOUT: left 2x2 = the paper-facing figure (Joint 1/2/3 Corrections +
    # Joint-Space Corrections) with the group title and shared legend
    # centered above it, screenshot-able as a standalone figure; right 2x2 =
    # diagnostics (Recovery Chunk detail, Task Progress, Convergence, Speed).
    fig = plt.figure(figsize=(21, 9.6))
    gs_l = fig.add_gridspec(2, 2, left=0.045, right=0.475, top=0.83, bottom=0.07, hspace=0.34, wspace=0.26)
    gs_r = fig.add_gridspec(2, 2, left=0.555, right=0.985, top=0.90, bottom=0.07, hspace=0.34, wspace=0.28)
    ax_joints = [fig.add_subplot(gs_l[0, 0]), fig.add_subplot(gs_l[0, 1]), fig.add_subplot(gs_l[1, 0])]
    ax_phase = fig.add_subplot(gs_l[1, 1])
    ax_zoom = fig.add_subplot(gs_r[0, 0])
    ax_clock = fig.add_subplot(gs_r[0, 1])
    ax_dev = fig.add_subplot(gs_r[1, 0])
    ax_speed = fig.add_subplot(gs_r[1, 1])
    fig.suptitle(TEXT["suptitle"] or title, y=0.985, fontsize=11)

    for j in range(min(n, 3)):
        ax = ax_joints[j]
        # Layering: demo at the bottom, the blended-policy track above it
        # (equally thick — both are background context), DART chunks on top.
        ax.plot(np.arange(len(geom.P)), geom.P[:, j], color="0.75", lw=6, zorder=1, label=TEXT["legend_demo"])
        ax.plot(
            idxs,
            S[:, j],
            color="tab:blue",
            lw=6,
            alpha=0.45,
            zorder=2,
            solid_capstyle="round",
            label=TEXT["legend_state"],
        )
        for t in anchors:
            cx = _chunk_clock(t)
            ax.plot(cx, chunks[t][:, j], color=colors[t], lw=1.2, alpha=0.95, zorder=3)
            ax.plot([idxs[t]], [S[t, j]], marker="o", color=colors[t], ms=6, mec="k", mew=0.6, zorder=4)
            _arrows_along(ax, np.column_stack([cx, chunks[t][:, j]]), 0, 1, colors[t], ks=(8, 20), lw=1.1)
        ax.set_title(TEXT["joint_titles"][j] if j < len(TEXT["joint_titles"]) else f"Joint {j + 1}")
        ax.set_xlabel(TEXT["xlabel_progress"])
        ax.set_ylabel(TEXT["ylabel_joint"].format(j=j + 1))
    # wall clock vs demo clock — the lag the labels resume from.
    ax = ax_clock
    ax.plot([0, len(S)], [0, len(S)], color="0.6", ls="--", label=TEXT["legend_pace"])
    ax.plot(np.arange(len(S)), idxs, color="tab:blue", lw=1.4, label=TEXT["legend_progress"])
    for t in anchors:
        ax.plot([t], [idxs[t]], marker="o", color=colors[t], ms=6, mec="k", mew=0.6)
    ax.set_xlabel(TEXT["xlabel_time"])
    ax.set_ylabel("Demo Index")
    ax.set_title(TEXT["clock_title"])
    ax.legend(fontsize=9)

    # overview phase plane j1/j2.
    jx, jy = 0, 1
    ax = ax_phase
    ax.plot(geom.P[:, jx], geom.P[:, jy], color="0.75", lw=6, zorder=1, label=TEXT["legend_demo_path"])
    ax.plot(
        S[:, jx],
        S[:, jy],
        color="tab:blue",
        lw=6,
        alpha=0.45,
        zorder=2,
        solid_capstyle="round",
        label=TEXT["legend_state_path"],
    )
    for t in anchors:
        ax.plot(chunks[t][:, jx], chunks[t][:, jy], color=colors[t], lw=1.4, alpha=0.95, zorder=3)
        ax.plot([S[t, jx]], [S[t, jy]], marker="o", color=colors[t], ms=6, mec="k", mew=0.6, zorder=4)
        _arrows_along(ax, chunks[t], jx, jy, colors[t], ks=(0, 6, 14))
    ax.set_xlabel(TEXT["ylabel_joint"].format(j=jx + 1))
    ax.set_ylabel(TEXT["ylabel_joint"].format(j=jy + 1))
    ax.set_title(TEXT["phase_title"])
    ax.set_aspect(
        "equal", adjustable="datalim"
    )  # both axes are radians — unequal aspect steepens every angle
    # no per-axis legend: the shared legend above the left 2x2 covers it.

    # zoom on the worst anchor — the rejoin geometry at deviation scale.
    d0s = {t: float(np.linalg.norm(S[t, :n] - _interp_rows(geom.P, float(idxs[t])))) for t in anchors}
    t_star = max(d0s, key=d0s.get)
    C = chunks[t_star]
    proj0 = _interp_rows(geom.P, float(idxs[t_star]))
    ax = ax_zoom
    ax.plot(geom.P[:, jx], geom.P[:, jy], color="0.75", lw=10, label=TEXT["legend_demo_path"])
    ax.plot(S[:, jx], S[:, jy], color="tab:blue", lw=1.2, label=TEXT["legend_state_path"])
    ax.plot(C[:, jx], C[:, jy], color=colors[t_star], lw=2.2, label=TEXT["legend_chunk"])
    _arrows_along(ax, C, jx, jy, colors[t_star], ks=(0, 1, 2, 3, 5, 8, 12), lw=1.8)
    ax.plot([S[t_star, jx]], [S[t_star, jy]], marker="o", color=colors[t_star], ms=9, mec="k", mew=1.0)
    ax.plot([proj0[jx]], [proj0[jy]], marker="x", color="k", ms=9, mew=2, label=TEXT["legend_projection"])
    ax.annotate(
        "",
        xy=C[0, [jx, jy]],
        xytext=S[t_star, [jx, jy]],
        arrowprops={"arrowstyle": "-|>", "color": "k", "lw": 2.2, "shrinkA": 0, "shrinkB": 0},
    )
    pts = np.vstack([C[:14, [jx, jy]], S[t_star, [jx, jy]][None], proj0[[jx, jy]][None]])
    span = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), 4 * geom.med_step)
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    ax.set_xlim(cx - 0.75 * span, cx + 0.75 * span)
    ax.set_ylim(cy - 0.75 * span, cy + 0.75 * span)
    ax.set_xlabel(TEXT["ylabel_joint"].format(j=jx + 1))
    ax.set_ylabel(TEXT["ylabel_joint"].format(j=jy + 1))
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(TEXT["zoom_title"])
    ax.legend(fontsize=9, loc="best")

    # chunk corridor deviation vs position k — the within-chunk convergence.
    ax = ax_dev
    for t in anchors:
        d = np.linalg.norm(chunks[t][:, None, :n] - geom.P[None, :, :], axis=2).min(axis=1)
        ax.plot(np.arange(horizon), d, color=colors[t], lw=1.0, alpha=0.9)
    ax.set_xlabel("Chunk position k")
    ax.set_ylabel("Deviation (rad)")
    ax.set_title(TEXT["dev_title"])

    # step-speed profile: state->label_0 at k=0, then label deltas; cruise band.
    ax = ax_speed
    sp_demo = np.linalg.norm(np.diff(geom.P, axis=0), axis=1) * fps
    ax.axhspan(
        np.percentile(sp_demo, 5), np.percentile(sp_demo, 95), color="0.85", label=TEXT["legend_cruise"]
    )
    first_sps, max_sps = [], []
    for t in anchors:
        seq = np.vstack([S[t][None, :n], chunks[t][:, :n]])
        sp = np.linalg.norm(np.diff(seq, axis=0), axis=1) * fps
        ax.plot(np.arange(horizon), sp, color=colors[t], lw=1.0, alpha=0.9)
        first_sps.append(sp[0])
        max_sps.append(sp.max())
    ax.set_xlabel("Chunk position k")
    ax.set_ylabel("Speed (rad/s)")
    ax.set_title(TEXT["speed_title"])
    ax.legend(fontsize=9)
    # Group heading + ONE shared legend centered over the LEFT 2x2 block
    # (the standalone paper figure).
    cx_group = 0.5 * (0.045 + 0.475)
    fig.text(cx_group, 0.935, TEXT["joint_group_title"], ha="center", va="bottom", fontsize=14)
    handles, labels_ = ax_joints[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels_,
        loc="lower center",
        bbox_to_anchor=(cx_group, 0.885),
        ncol=2,
        fontsize=10,
        frameon=False,
    )
    fig.savefig(out, dpi=110)
    print(f"saved → {out}")

    dev0 = np.array([np.linalg.norm(S[t, :n] - chunks[t][0, :n]) for t in anchors])
    rejoin = [
        int(np.argmax(np.linalg.norm(chunks[t][:, None, :n] - geom.P[None, :, :], axis=2).min(axis=1) < 1e-3))
        for t in anchors
    ]
    print(
        f"{len(anchors)} anchors | first-step speed max {max(first_sps):.3f} rad/s "
        f"(demo cruise med {np.median(sp_demo):.3f}) | any-step max {max(max_sps):.3f} | "
        f"|label0-state| max {dev0.max():.4f} | ticks-to-corridor p50 {int(np.median(rejoin))} "
        f"of H={len(chunks[anchors[0]])}"
    )


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", default=None, help="rollout_data.npz (on-the-fly projection)")
    ap.add_argument("--ratio", default="0.50", help="npz mode: which ratio track")
    ap.add_argument("--blend_repo_id", default=None, help="dataset mode: blend repo with relabel_demo_index")
    ap.add_argument("--source_repo_id", required=True)
    ap.add_argument("--episode_index", type=int, default=0)
    ap.add_argument("--source_episode_index", type=int, default=None)
    ap.add_argument("--horizon", type=int, default=32)
    ap.add_argument("--rate", type=float, default=1.0)
    ap.add_argument(
        "--ease_out",
        type=float,
        default=0.3,
        help="proportional closure fraction near the corridor (C1 merge)",
    )
    ap.add_argument("--anchor_every", type=int, default=15)
    ap.add_argument("--index_window", type=int, default=45, help="npz mode: projection window (demo steps)")
    ap.add_argument("--num_arm_joints", type=int, default=3)
    ap.add_argument("--fps", type=float, default=30.0)
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
        S_src_full = _load_episode(args.source_repo_id, src_ep)["observation.state"]
        off = max(0, len(S_src_full) - len(g_act))  # guidance = source tail
        geom = demo_geometry(S_src_full[off : off + len(g_act)], g_act, n_arm=n)
        idxs = project_states(track, geom, index_window=args.index_window)
        title = f"DART chunks (on-the-fly) — {key} of {os.path.basename(os.path.dirname(args.npz))} (source ep {src_ep})"
        out = args.out or f"dart_chunks_npz_{key}.png"
        _plot(
            track[:, :n],
            idxs,
            geom,
            n,
            args.horizon,
            args.rate,
            args.ease_out,
            args.fps,
            args.anchor_every,
            title,
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
    bl = _load_episode(
        args.blend_repo_id, args.episode_index, cols=("observation.state", "relabel_demo_index")
    )
    if "relabel_demo_index" not in bl:
        raise SystemExit(
            f"{args.blend_repo_id}: no relabel_demo_index column — record with --relabel_actions=guidance"
        )
    src = _load_episode(args.source_repo_id, src_ep)
    geom = demo_geometry(src["observation.state"], src["action"], n_arm=n)
    idxs = np.asarray(bl["relabel_demo_index"], dtype=np.float64).reshape(-1)
    title = f"DART chunks — blend ep {args.episode_index} of {args.blend_repo_id} (source ep {src_ep})"
    out = args.out or f"dart_chunks_{args.blend_repo_id.split('/')[-1]}_ep{args.episode_index}.png"
    _plot(
        bl["observation.state"][:, :n],
        idxs,
        geom,
        n,
        args.horizon,
        args.rate,
        args.ease_out,
        args.fps,
        args.anchor_every,
        title,
        out,
    )


if __name__ == "__main__":
    main()
