#!/usr/bin/env python
"""Visualize train-time DART chunk synthesis (dart_labels.chunk_labels).

For a set of anchor (conditioning) frames along a blend rollout, synthesizes
the full H-step expert label chunk the training dataloader would serve —
demo clock advancing one index per tick, corridor offset decaying at
``rate x med_step`` per tick — and plots:

  * per-joint time series: demo, executed track, and each anchor's chunk
    overlaid at its wall ticks (chunks should peel off the executed track
    and land on the demo);
  * joint-space phase planes: chunk curves fanning from anchor states into
    the demo corridor — the geometry the policy actually learns;
  * chunk corridor-deviation vs position k (linear decay to ZERO — the
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

FPS = 30.0


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
    anchor_every: int,
    title: str,
    out: str,
) -> None:
    anchors = list(range(0, len(S) - 1, max(1, anchor_every)))
    chunks = {t: chunk_labels(S[t], float(idxs[t]), geom, horizon=horizon, rate=rate) for t in anchors}
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

    ncols = max(n, 4)
    fig, axes = plt.subplots(2, ncols, figsize=(5.2 * ncols, 9))
    fig.suptitle(
        f"{title}\nTHICK grey=demo  blue=executed  colored curves=synthesized H={horizon} label chunks "
        f"(land ON the demo, hiding it — see zoom)  ●=conditioning state  arrowheads=direction of time  "
        f"(rate={rate} x med_step={geom.med_step:.4f})"
    )

    t_ax = np.arange(len(S))
    for j in range(n):
        ax = axes[0][j]
        ax.plot(np.arange(len(geom.P)), geom.P[:, j], color="0.75", lw=6, label="demo (source states)")
        ax.plot(t_ax, S[:, j], color="tab:blue", lw=1.2, label="executed state")
        for t in anchors:
            ax.plot(np.arange(t, t + horizon), chunks[t][:, j], color=colors[t], lw=1.1, alpha=0.9)
            ax.plot([t], [S[t, j]], marker="o", color=colors[t], ms=6, mec="k", mew=0.6)
        ax.set_title(f"joint_{j + 1}")
        ax.set_xlabel("tick")
        if j == 0:
            ax.legend(fontsize=8)
    for col in range(n, ncols):
        axes[0][col].axis("off")

    # overview phase plane j1/j2.
    jx, jy = 0, 1
    ax = axes[1][0]
    ax.plot(geom.P[:, jx], geom.P[:, jy], color="0.75", lw=6, label="demo path (thick grey)")
    ax.plot(S[:, jx], S[:, jy], color="tab:blue", lw=1.2, label="executed path")
    for t in anchors:
        ax.plot(chunks[t][:, jx], chunks[t][:, jy], color=colors[t], lw=1.4, alpha=0.95)
        ax.plot([S[t, jx]], [S[t, jy]], marker="o", color=colors[t], ms=6, mec="k", mew=0.6)
        _arrows_along(ax, chunks[t], jx, jy, colors[t], ks=(0, 6, 14))
    ax.set_xlabel(f"joint_{jx + 1}")
    ax.set_ylabel(f"joint_{jy + 1}")
    ax.set_title("phase plane j1/j2 — chunks leave ● and merge onto the grey demo")
    ax.legend(fontsize=8)

    # zoom on the worst anchor — the rejoin geometry at deviation scale.
    d0s = {t: float(np.linalg.norm(S[t, :n] - _interp_rows(geom.P, float(idxs[t])))) for t in anchors}
    t_star = max(d0s, key=d0s.get)
    C = chunks[t_star]
    proj0 = _interp_rows(geom.P, float(idxs[t_star]))
    ax = axes[1][1]
    ax.plot(geom.P[:, jx], geom.P[:, jy], color="0.75", lw=10, label="demo path")
    ax.plot(S[:, jx], S[:, jy], color="tab:blue", lw=1.2, label="executed path")
    ax.plot(C[:, jx], C[:, jy], color=colors[t_star], lw=2.2, label="label chunk")
    _arrows_along(ax, C, jx, jy, colors[t_star], ks=(0, 1, 2, 3, 5, 8, 12), lw=1.8)
    ax.plot([S[t_star, jx]], [S[t_star, jy]], marker="o", color=colors[t_star], ms=9, mec="k", mew=1.0)
    ax.plot([proj0[jx]], [proj0[jy]], marker="x", color="k", ms=9, mew=2, label="corridor projection")
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
    ax.set_xlabel(f"joint_{jx + 1}")
    ax.set_ylabel(f"joint_{jy + 1}")
    ax.set_title(
        f"ZOOM: worst anchor t={t_star} (dev {d0s[t_star]:.3f} rad)\n"
        "black arrow = first commanded step (state → label_0)"
    )
    ax.legend(fontsize=8, loc="best")

    # chunk corridor deviation vs position k — the within-chunk convergence.
    ax = axes[1][2]
    for t in anchors:
        d = np.linalg.norm(chunks[t][:, None, :n] - geom.P[None, :, :], axis=2).min(axis=1)
        ax.plot(np.arange(horizon), d, color=colors[t], lw=1.0, alpha=0.9)
    ax.set_xlabel("chunk position k")
    ax.set_ylabel("rad")
    ax.set_title("label_k deviation from demo corridor\n(linear decay to 0 = converges IN-chunk)")

    # step-speed profile: state->label_0 at k=0, then label deltas; cruise band.
    ax = axes[1][3]
    sp_demo = np.linalg.norm(np.diff(geom.P, axis=0), axis=1) * FPS
    ax.axhspan(
        np.percentile(sp_demo, 5), np.percentile(sp_demo, 95), color="0.85", label="demo cruise p5-p95"
    )
    first_sps, max_sps = [], []
    for t in anchors:
        seq = np.vstack([S[t][None, :n], chunks[t][:, :n]])
        sp = np.linalg.norm(np.diff(seq, axis=0), axis=1) * FPS
        ax.plot(np.arange(horizon), sp, color=colors[t], lw=1.0, alpha=0.9)
        first_sps.append(sp[0])
        max_sps.append(sp.max())
    ax.set_xlabel("chunk position k  (k=0 is state → label_0)")
    ax.set_ylabel("rad/s")
    ax.set_title("commanded step speed along chunk")
    ax.legend(fontsize=8)
    for col in range(4, ncols):
        axes[1][col].axis("off")

    fig.tight_layout()
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
    ap.add_argument("--anchor_every", type=int, default=15)
    ap.add_argument("--index_window", type=int, default=45, help="npz mode: projection window (demo steps)")
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
        S_src_full = _load_episode(args.source_repo_id, src_ep)["observation.state"]
        off = max(0, len(S_src_full) - len(g_act))  # guidance = source tail
        geom = demo_geometry(S_src_full[off : off + len(g_act)], g_act, n_arm=n)
        idxs = project_states(track, geom, index_window=args.index_window)
        title = f"DART chunks (on-the-fly) — {key} of {os.path.basename(os.path.dirname(args.npz))} (source ep {src_ep})"
        out = args.out or f"dart_chunks_npz_{key}.png"
        _plot(track[:, :n], idxs, geom, n, args.horizon, args.rate, args.anchor_every, title, out)
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
        bl["observation.state"][:, :n], idxs, geom, n, args.horizon, args.rate, args.anchor_every, title, out
    )


if __name__ == "__main__":
    main()
