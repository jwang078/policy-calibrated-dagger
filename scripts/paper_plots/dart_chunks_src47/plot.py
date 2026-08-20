#!/usr/bin/env python
"""Paper figure: DART-relabeled blend rollout with synthesized label chunks (src47).

STANDALONE — needs only numpy + matplotlib and the .npz next to this script.
The npz snapshots everything (blend states, stored projection indices, source
demo states/actions), so the figure regenerates identically even after the
scratch datasets are gone. Data provenance is in the npz's ``meta_note``.

Regenerate:  python plot.py            (writes PNG next to this script)

All titles / axis labels / legends are in the TEXT block below — tune freely.
The label-chunk synthesis (demo clock + rate-limited offset closure with C1
ease-out merge) is inlined verbatim from lerobot.datasets.dart_relabel so
this file has no repo dependencies.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── figure text (EDIT ME) ─────────────────────────────────────────────────
TEXT = {
    "suptitle": "",  # e.g. "Chunk-Level DART Relabeling of a Policy-Augmented Rollout"
    "joint_titles": ["Joint 1 Trajectory", "Joint 2 Trajectory", "Joint 3 Trajectory"],
    "clock_title": "Task Progress vs. Time",
    "phase_title": "Joint-Space Paths with Recovery Chunks",
    "zoom_title": "Recovery Chunk (Detail)",
    "dev_title": "Within-Chunk Convergence to the Demo",
    "speed_title": "Commanded Speed Along the Chunk",
    "xlabel_time": "Timestep",
    "ylabel_joint": "Joint{j} Position",
    "legend_demo": "Demo",
    "legend_state": "Policy-Augmented State",
    "legend_demo_path": "Demo path",
    "legend_state_path": "Policy-Augmented path",
    "legend_chunk": "Label chunk",
    "legend_projection": "Corridor projection",
    "legend_pace": "Demo pace (y = x)",
    "legend_progress": "Projected demo index",
    "legend_cruise": "Demo cruise p5–p95",
}

# ── plot parameters ───────────────────────────────────────────────────────
H = 32  # synthesized chunk length = the policy's executed n_action_steps
FPS = 30.0
N = 3  # arm joints
ANCHOR_EVERY = 15
RATE, EASE_OUT = 1.0, 0.3  # label-synthesis knobs (match training defaults)
DPI = 110

HERE = Path(__file__).resolve().parent
DATA = HERE / "src47_after_taper_stallcut.npz"
OUT = HERE / "src47_after_taper_stallcut_paper.png"


# ── DART label synthesis (inlined from lerobot.datasets.dart_relabel) ─────
def interp_rows(mat: np.ndarray, i: float) -> np.ndarray:
    """Linear interpolation of row ``i`` (float, clamped) of matrix ``mat``."""
    idx = int(np.clip(np.floor(i), 0, len(mat) - 1))
    nxt = min(idx + 1, len(mat) - 1)
    f = float(np.clip(i - idx, 0.0, 1.0))
    return (1.0 - f) * mat[idx] + f * mat[nxt]


def chunk_labels(state, demo_index, P, A, med_step, horizon, rate=1.0, ease_out=0.3):
    """Expert response chunk from one visited state.

    Demo clock advances from the projected index; the corridor offset
    closes at min(rate*med_step, ease_out*d) per tick (C1 merge).
    """
    q = np.asarray(state, dtype=np.float64)[: P.shape[1]]
    proj0 = interp_rows(P, demo_index)
    offset = q - proj0
    d0 = float(np.linalg.norm(offset))
    u = offset / d0 if d0 > 1e-9 else np.zeros_like(offset)
    close = max(0.0, float(rate)) * med_step
    labels = np.empty((horizon, A.shape[1]), dtype=np.float64)
    end = float(len(A) - 1)
    d_k = d0
    for k in range(horizon):
        labels[k] = interp_rows(A, min(demo_index + k, end))
        d_k = max(0.0, d_k - min(close, ease_out * d_k) if ease_out > 0 else d_k - close)
        if d_k < 0.05 * med_step:
            d_k = 0.0
        if d_k > 0.0:
            labels[k, : P.shape[1]] += d_k * u
    return labels


# ── load snapshot ─────────────────────────────────────────────────────────
z = np.load(DATA, allow_pickle=True)
S = np.asarray(z["blend_states"], dtype=np.float64)[:, :N]
idxs = np.asarray(z["relabel_demo_index"], dtype=np.float64)
P = np.asarray(z["demo_states"], dtype=np.float64)[:, :N]
A = np.asarray(z["demo_actions"], dtype=np.float64)
seg_len = np.linalg.norm(np.diff(P, axis=0), axis=1)
med_step = float(np.median(seg_len[seg_len > 1e-9]))
demo_end = len(P) - 1

# anchors stop one action chunk before the demo end: no chunk extends past it.
anchors = [t for t in range(0, len(S) - 1, ANCHOR_EVERY) if idxs[t] + H <= demo_end]
chunks = {t: chunk_labels(S[t], float(idxs[t]), P, A, med_step, H, RATE, EASE_OUT) for t in anchors}
cmap = plt.get_cmap("plasma")
colors = {t: cmap(i / max(1, len(anchors) - 1)) for i, t in enumerate(anchors)}


def arrows(ax, C, jx, jy, color, ks, lw=1.4):
    """Arrowheads along curve C — direction of time."""
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


# ── figure ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(20.8, 9))
if TEXT["suptitle"]:
    fig.suptitle(TEXT["suptitle"])

for j in range(N):
    ax = axes[0][j]
    ax.plot(np.arange(len(P)), P[:, j], color="0.75", lw=6, label=TEXT["legend_demo"])
    ax.plot(idxs, S[:, j], color="tab:blue", lw=1.2, label=TEXT["legend_state"])
    for t in anchors:
        ax.plot(idxs[t] + np.arange(H), chunks[t][:, j], color=colors[t], lw=1.1, alpha=0.9)
        ax.plot([idxs[t]], [S[t, j]], marker="o", color=colors[t], ms=6, mec="k", mew=0.6)
    ax.set_xlabel(TEXT["xlabel_time"])
    ax.set_ylabel(TEXT["ylabel_joint"].format(j=j + 1))
    ax.set_title(TEXT["joint_titles"][j])
    if j == 0:
        ax.legend(fontsize=9)

ax = axes[0][3]
ax.plot([0, len(S)], [0, len(S)], color="0.6", ls="--", label=TEXT["legend_pace"])
ax.plot(np.arange(len(S)), idxs, color="tab:blue", lw=1.4, label=TEXT["legend_progress"])
for t in anchors:
    ax.plot([t], [idxs[t]], marker="o", color=colors[t], ms=6, mec="k", mew=0.6)
ax.set_xlabel(TEXT["xlabel_time"])
ax.set_ylabel("Demo index")
ax.set_title(TEXT["clock_title"])
ax.legend(fontsize=9)

jx, jy = 0, 1
ax = axes[1][0]
ax.plot(P[:, jx], P[:, jy], color="0.75", lw=6, label=TEXT["legend_demo_path"])
ax.plot(S[:, jx], S[:, jy], color="tab:blue", lw=1.2, label=TEXT["legend_state_path"])
for t in anchors:
    ax.plot(chunks[t][:, jx], chunks[t][:, jy], color=colors[t], lw=1.4, alpha=0.95)
    ax.plot([S[t, jx]], [S[t, jy]], marker="o", color=colors[t], ms=6, mec="k", mew=0.6)
    arrows(ax, chunks[t], jx, jy, colors[t], ks=(0, 6, 14))
ax.set_xlabel(TEXT["ylabel_joint"].format(j=1))
ax.set_ylabel(TEXT["ylabel_joint"].format(j=2))
ax.set_title(TEXT["phase_title"])
ax.legend(fontsize=9)

d0s = {t: float(np.linalg.norm(S[t] - interp_rows(P, float(idxs[t])))) for t in anchors}
t_star = max(d0s, key=d0s.get)
C = chunks[t_star]
proj0 = interp_rows(P, float(idxs[t_star]))
ax = axes[1][1]
ax.plot(P[:, jx], P[:, jy], color="0.75", lw=10, label=TEXT["legend_demo_path"])
ax.plot(S[:, jx], S[:, jy], color="tab:blue", lw=1.2, label=TEXT["legend_state_path"])
ax.plot(C[:, jx], C[:, jy], color=colors[t_star], lw=2.2, label=TEXT["legend_chunk"])
arrows(ax, C, jx, jy, colors[t_star], ks=(0, 1, 2, 3, 5, 8, 12), lw=1.8)
ax.plot([S[t_star, jx]], [S[t_star, jy]], marker="o", color=colors[t_star], ms=9, mec="k", mew=1.0)
ax.plot([proj0[jx]], [proj0[jy]], marker="x", color="k", ms=9, mew=2, label=TEXT["legend_projection"])
ax.annotate(
    "",
    xy=C[0, [jx, jy]],
    xytext=S[t_star, [jx, jy]],
    arrowprops={"arrowstyle": "-|>", "color": "k", "lw": 2.2, "shrinkA": 0, "shrinkB": 0},
)
pts = np.vstack([C[:14, [jx, jy]], S[t_star, [jx, jy]][None], proj0[[jx, jy]][None]])
span = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), 4 * med_step)
cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
ax.set_xlim(cx - 0.75 * span, cx + 0.75 * span)
ax.set_ylim(cy - 0.75 * span, cy + 0.75 * span)
ax.set_xlabel(TEXT["ylabel_joint"].format(j=1))
ax.set_ylabel(TEXT["ylabel_joint"].format(j=2))
ax.set_title(TEXT["zoom_title"])
ax.legend(fontsize=9, loc="best")

ax = axes[1][2]
for t in anchors:
    d = np.linalg.norm(chunks[t][:, None, :N] - P[None, :, :], axis=2).min(axis=1)
    ax.plot(np.arange(H), d, color=colors[t], lw=1.0, alpha=0.9)
ax.set_xlabel("Chunk position k")
ax.set_ylabel("Deviation (rad)")
ax.set_title(TEXT["dev_title"])

ax = axes[1][3]
sp_demo = seg_len * FPS
ax.axhspan(np.percentile(sp_demo, 5), np.percentile(sp_demo, 95), color="0.85", label=TEXT["legend_cruise"])
for t in anchors:
    seq = np.vstack([S[t][None], chunks[t][:, :N]])
    sp = np.linalg.norm(np.diff(seq, axis=0), axis=1) * FPS
    ax.plot(np.arange(H), sp, color=colors[t], lw=1.0, alpha=0.9)
ax.set_xlabel("Chunk position k")
ax.set_ylabel("Speed (rad/s)")
ax.set_title(TEXT["speed_title"])
ax.legend(fontsize=9)

fig.tight_layout()
fig.savefig(OUT, dpi=DPI)
print(f"saved -> {OUT}  ({len(anchors)} anchors, last at demo index {idxs[anchors[-1]]:.0f} of {demo_end})")
