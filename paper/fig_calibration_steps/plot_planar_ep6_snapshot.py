#!/usr/bin/env python
"""Calibration-pipeline figure (joint space, one intervention episode):
(a) partial-diffusion blend rollouts tracking the expert  -> closed-loop W
(b) open-loop sampling spread of the policy at recorded anchors
(c) calibrated injection band (alpha-rescaled to W^2) + servo recovery labels

All data is real: modern (non-anchored) blend datasets, the sigma_deltas
measurement files, the deployed pooled schedule, and dart_relabel's actual
label generator. Style knobs in CONFIG.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

# ── CONFIG ───────────────────────────────────────────────────────────────────
EPISODE = 6  # source intervention episode (r_dag3)
JOINT = 0  # 0 -> "Joint 1"
BLEND_TAGS = ("010", "020", "030")  # panel (a): all complete + graded on ep32
LABEL_ANCHORS = (25, 55, 85)  # panel (c): servo labels start at these frames (inside XLIM)
LABEL_SEED = 7

COL_EXPERT = "#9a9a9a"
LW_EXPERT = 3.0
COL_BLENDS = ("#7fb3e8", "#4a8fd6", "#2a5f9e")  # light -> dark with ratio
COL_OPEN_BAND = "#b08bc9"
COL_DEP_BAND = "#3b7dd1"
COL_LABELS = ("#f2a541", "#e0821f", "#c95d08")
XLIM = (0, 10**6)  # frame range shown bold; (0, 10**6) = whole intervention (W uses ticks 40:120)
FIGSIZE = (12.6, 3.6)
DPI = 150
# ─────────────────────────────────────────────────────────────────────────────

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fig_calibration_steps.png")
_ANALYSIS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tables_repro", "analysis")
SIGMA_NPZ = os.path.join(_ANALYSIS, "sigma_deltas_s1q3_dag3.npz")
SCHED_JSON = os.path.join(_ANALYSIS, "noise_schedule_pooled_s1_K3.json")
for _f in (SIGMA_NPZ, SCHED_JSON):  # local copies win if present
    _l = os.path.join(HERE, os.path.basename(_f))
    if os.path.exists(_l):
        pass
SIGMA_NPZ = (
    os.path.join(HERE, "sigma_deltas_s1q3_dag3.npz")
    if os.path.exists(os.path.join(HERE, "sigma_deltas_s1q3_dag3.npz"))
    else SIGMA_NPZ
)
SCHED_PA_JSON = os.path.join(HERE, "noise_schedule_sigma_alpha_v2_s1_K3.json")  # per-anchor alpha*Sigma_t
SCHED_JSON = (
    os.path.join(HERE, "noise_schedule_pooled_s1_K3.json")
    if os.path.exists(os.path.join(HERE, "noise_schedule_pooled_s1_K3.json"))
    else SCHED_JSON
)

sys.path.insert(0, os.path.expanduser("~/code/lerobot/src"))
import matplotlib  # noqa: E402

from lerobot.datasets.dart_relabel import chunk_labels, demo_geometry  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CACHE = os.path.expanduser("~/.cache/huggingface/lerobot/JennyWWW")


def joints_of(repo, episode):
    df = pd.read_parquet(f"{CACHE}/{repo}/data/chunk-000/file-000.parquet")
    g = df[df.episode_index == episode].sort_values("frame_index")
    return (
        np.stack([np.asarray(v, dtype=float) for v in g["observation.state"]]),
        np.stack([np.asarray(v, dtype=float) for v in g["action"]]),
    )


St, Ac = joints_of("planar_12_05dag_diff_r_dag3", EPISODE)
expert = St[:, JOINT]
T = np.arange(len(expert))

# blend episode index: source episodes were collected every-2nd -> ep//2
BLEND_EP = EPISODE // 2
sys.path.insert(0, os.path.expanduser("~/code/lerobot/my_scripts"))
from lib_sa_rollout import progress_guidance_index  # noqa: E402

_D3 = St[:, :3]
_dm = np.linalg.norm(np.diff(_D3, axis=0), axis=1)
_med = float(np.median(_dm[_dm > 1e-6]))
ESCAPE = 15.0
blends = {}
blends_full = {}
for tag in BLEND_TAGS:
    try:
        B, _ = joints_of(f"planar_12_05dag_diff_r_dag3_blend{tag}", BLEND_EP)
    except Exception:
        continue
    j = 0
    dv = []
    for t in range(len(B)):
        j = progress_guidance_index(_D3, B[t, :3], j, window=48)
        dv.append(np.linalg.norm(B[t, :3] - _D3[j]) / _med)
    assert not (np.array(dv) >= ESCAPE).any(), f"blend {tag} escapes on ep {EPISODE}"
    blends[tag] = B[:, JOINT]
    blends_full[tag] = B[:, :3]

# open-loop spread at anchors (per-anchor std of sampled-chunk deviations)
z = np.load(SIGMA_NPZ, allow_pickle=True)
d = z[f"ep{EPISODE}_d"]  # (anchors, draws, steps, 3)
ta = z[f"ep{EPISODE}_t"].astype(int)
order = np.argsort(ta)
ta = ta[order]
# RMS about zero = sqrt of the diagonal of the per-anchor uncentered second
# moment C[JOINT,JOINT] -- the exact quantity the calibration estimates
sig_open = np.array([np.sqrt((d[i][:, :, JOINT] ** 2).mean()) for i in order])

# deployed (alpha-rescaled) per-joint sigma from the pooled schedule
sched = json.load(open(SCHED_JSON))
row = np.asarray(sched["JennyWWW/planar_12_05dag_diff_r_dag3"][str(EPISODE)], dtype=float)[0]
dd = np.linalg.norm(np.diff(St[:, :3], axis=0), axis=1)
ms = float(np.median(dd[dd > 1e-6]))
Sg = np.array([[row[0], row[3], row[4]], [row[3], row[1], row[5]], [row[4], row[5], row[2]]]) * ms**2
sig_dep = float(np.sqrt(Sg[JOINT, JOINT]))
# per-anchor alpha*Sigma_t rows (med^2 units, same layout) -> rad^2
_pa = np.asarray(
    json.load(open(SCHED_PA_JSON))["JennyWWW/planar_12_05dag_diff_r_dag3"][str(EPISODE)], dtype=float
)
_al = json.load(
    open(os.path.join(HERE, "alpha_s1_K3.json"))
)  # from the schedule builder: W^2 / mean tr(Sigma_t)
W_MED, ALPHA_HAT, TR_MEAN = float(_al["W_med"]), float(_al["alpha"]), float(_al["tr_mean_med2"])


def Sg_at(t):
    r = _pa[min(int(t), len(_pa) - 1)]
    return np.array([[r[0], r[3], r[4]], [r[3], r[1], r[5]], [r[4], r[5], r[2]]]) * ms**2


# servo recovery labels from schedule samples at chosen anchors
geom = demo_geometry(St, Ac, 3)
L = np.linalg.cholesky(Sg + 1e-12 * np.eye(3))
rng = np.random.default_rng(LABEL_SEED)
label_curves = []
label_starts = {}
for t0 in LABEL_ANCHORS:
    qn = St[t0, :3] + L @ rng.standard_normal(3)
    vrec = (St[t0] - St[max(t0 - 1, 0)])[:3]
    lab = chunk_labels(qn, float(t0), geom, horizon=64, velocity=vrec)[:, :3]
    label_curves.append((t0, qn[JOINT], lab))
    label_starts[t0] = qn

# ── figure (joint space: x = joint 2, y = joint 1; time only sets the segment) ─
JX, JY = 1, 0  # x-axis joint, y-axis joint (0-based)
T0, T1 = XLIM[0], min(XLIM[1], len(St) - 1)
seg = slice(T0, min(T1 + 1, len(St)))
ELL_EVERY = 1  # ellipse at every N frames, filled as ONE union region
SHOW_PER_ANCHOR = False  # extra panel: per-anchor alpha*Sigma_t (= (a) scaled by sqrt(alpha)); on planar alpha=0.81 so it is ~(a)
N_SAMPLES = 40  # (c): sampled noised states at LABEL_ANCHORS[0]
plt.rcParams.update({"font.size": 9.5, "font.family": "DejaVu Sans"})
_np = 4 if SHOW_PER_ANCHOR else 3
fig, axes = plt.subplots(1, _np, figsize=(FIGSIZE[0] * _np / 3, FIGSIZE[1]), sharex=True, sharey=True)

_th = np.linspace(0, 2 * np.pi, 120)
_circ = np.stack([np.cos(_th), np.sin(_th)])


def ellipse_pts(center, S2):
    """1-sigma ellipse of a 2x2 covariance S2 (in [JX, JY] order), (N, 2)."""
    ev, evec = np.linalg.eigh(S2)
    pts = evec @ (np.sqrt(np.clip(ev, 0, None))[:, None] * _circ)
    return np.column_stack([center[0] + pts[0], center[1] + pts[1]])


from matplotlib.patches import PathPatch as _PathPatch
from matplotlib.path import Path as _Path


def union_region(ax, polys, color, alpha, zorder):
    """Fill the UNION of many polygons as one flat region: a single compound
    path filled with the nonzero winding rule (all rings share orientation),
    so overlaps do not stack alpha.
    """
    verts, codes = [], []
    for P in polys:
        verts.extend(P.tolist() + [P[0].tolist()])
        codes.extend([_Path.MOVETO] + [_Path.LINETO] * (len(P) - 1) + [_Path.CLOSEPOLY])
    ax.add_patch(
        _PathPatch(_Path(verts, codes), facecolor=color, edgecolor="none", alpha=alpha, zorder=zorder)
    )


def expert_path(ax):
    # whole intervention faintly (labels beyond the segment land on it), segment bold
    ax.plot(
        St[:, JX],
        St[:, JY],
        "-",
        lw=LW_EXPERT * 0.6,
        color=COL_EXPERT,
        solid_capstyle="round",
        zorder=2,
        alpha=0.35,
    )
    ax.plot(
        St[seg, JX],
        St[seg, JY],
        "-",
        lw=LW_EXPERT,
        color=COL_EXPERT,
        solid_capstyle="round",
        zorder=3,
        alpha=0.9,
    )
    ax.plot([St[T0, JX]], [St[T0, JY]], "o", ms=5, color=COL_EXPERT, zorder=3)


# (a) open-loop sampling spread: per-anchor 1-sigma ellipses (uncentered 2nd moment)
ax = axes[0]
expert_path(ax)
union_region(
    ax,
    [
        ellipse_pts(St[t, [JX, JY]], (Sg_at(t) / ALPHA_HAT)[np.ix_([JX, JY], [JX, JY])])
        for t in range(T0, min(T1 + 1, len(St)), ELL_EVERY)
    ],
    COL_OPEN_BAND,
    0.55,
    zorder=3.5,
)
ax.set_title(r"(a) open-loop sampling spread $\hat\Sigma_t$ (per anchor)", fontsize=9.5)

# (b) blended rollouts settle inside the W tube -> alpha-hat = W^2 / tr(Sigma-hat)
ax = axes[1]
expert_path(ax)
for tag, col in zip(BLEND_TAGS, COL_BLENDS):
    if tag in blends_full:
        B = blends_full[tag]  # whole blended rollout, no truncation
        r_lbl = "r=0." + tag.lstrip("0") if tag != "010" else "r=0.10"
        ax.plot(B[:, JX], B[:, JY], "-", lw=1.7, color=col, alpha=0.85, zorder=4, label=f"blend {r_lbl}")
union_region(
    ax,
    [
        ellipse_pts(St[t, [JX, JY]], np.eye(2) * (W_MED * ms) ** 2)
        for t in range(T0, min(T1 + 1, len(St)), ELL_EVERY)
    ],
    "#9a9a9a",
    0.18,
    zorder=2.2,
)
ax.set_title(r"(b) blended rollouts settle at $W$", fontsize=9.5)
ax.text(
    0.03,
    0.05,
    rf"$W$ = {W_MED:.0f} med-steps (tube)"
    + "\n"
    + rf"$\hat\alpha = W^2/\overline{{\mathrm{{tr}}\,\hat\Sigma_t}}$ = {W_MED:.0f}$^2$/{TR_MEAN:.0f} = {ALPHA_HAT:.2f}",
    transform=ax.transAxes,
    fontsize=8,
    va="bottom",
    ha="left",
    color="#333333",
)
ax.legend(fontsize=7.5, frameon=False, loc="upper right")

if SHOW_PER_ANCHOR:
    # (c) per-anchor calibrated noise: alpha-hat * Sigma-hat_t = panel (b) rescaled
    ax = axes[2]
    expert_path(ax)
    union_region(
        ax,
        [
            ellipse_pts(St[t, [JX, JY]], Sg_at(t)[np.ix_([JX, JY], [JX, JY])])
            for t in range(T0, min(T1 + 1, len(St)), ELL_EVERY)
        ],
        COL_DEP_BAND,
        0.35,
        zorder=3.5,
    )
    ax.set_title(
        rf"(c) per-anchor $\hat\alpha\,\hat\Sigma_t$  (= (a) $\times\sqrt{{\hat\alpha}}$ = {np.sqrt(ALPHA_HAT):.2f})",
        fontsize=9.5,
    )

# (d)/(c) pooled Sigma-bar^alpha (deployed) + servo labels
ax = axes[-1]
expert_path(ax)
S2 = Sg[np.ix_([JX, JY], [JX, JY])]
union_region(
    ax,
    [ellipse_pts(St[t, [JX, JY]], S2) for t in range(T0, min(T1 + 1, len(St)), ELL_EVERY)],
    COL_DEP_BAND,
    0.35,
    zorder=3.5,
)
rng_d = np.random.default_rng(LABEL_SEED + 1)
q_anchor = St[LABEL_ANCHORS[1], :3]
samples = q_anchor[None, :] + (L @ rng_d.standard_normal((3, N_SAMPLES))).T
ax.scatter(samples[:, JX], samples[:, JY], s=7, color=COL_DEP_BAND, alpha=0.5, edgecolors="none", zorder=5)
for (t0, q0, lab), col in zip(label_curves, COL_LABELS):
    qn = label_starts[t0]
    ax.plot(lab[:, JX], lab[:, JY], "-", lw=1.8, color=col, zorder=6)
    ax.plot([qn[JX]], [qn[JY]], "o", ms=4.5, color=col, zorder=7)
ax.set_title(
    ("(d) " if SHOW_PER_ANCHOR else "(c) ") + r"pooled $\bar\Sigma^{\alpha}$ (deployed) + servo labels",
    fontsize=9.5,
)

axes[0].set_ylabel(f"Joint {JY + 1} position [rad]")
for ax in axes:
    ax.set_xlabel(f"Joint {JX + 1} position [rad]")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#e4e4e4", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_aspect("equal")
# limits: the path segment plus the deployed 1-sigma ellipse, with padding
_r = 1.6 * float(np.sqrt(np.linalg.eigvalsh(S2).max()))
_allx = np.concatenate(
    [St[:, JX]] + [B[:, JX] for B in blends_full.values()] + [lab[:, JX] for _, _, lab in label_curves]
)
_ally = np.concatenate(
    [St[:, JY]] + [B[:, JY] for B in blends_full.values()] + [lab[:, JY] for _, _, lab in label_curves]
)
xlo, xhi = _allx.min() - _r, _allx.max() + _r
ylo, yhi = _ally.min() - _r, _ally.max() + _r
for ax in axes:
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
plt.tight_layout()
plt.savefig(OUT, dpi=DPI, bbox_inches="tight")
print(
    f"wrote {OUT} | pooled 1-sigma (J{JX + 1},J{JY + 1}) = {np.sqrt(np.diag(S2)).round(4).tolist()} rad, "
    f"open-loop sigma range {sig_open.min():.3f}-{sig_open.max():.3f} rad"
)
