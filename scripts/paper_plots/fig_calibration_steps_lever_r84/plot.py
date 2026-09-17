#!/usr/bin/env python
"""Calibration-pipeline figure for the LEVER image policy (6-joint UR arm), joint
space (x = joint JX, y = joint JY), one round-1 intervention episode:
(a) per-anchor open-loop spread Sigma-hat_t (union of 1-sigma ellipses)
(b) blended rollouts (r = 0.1/0.2/0.3) inside the W tube, with the DART scale s = W^2 / mean tr(Sigma_t)
(c) deployed pooled Sigma-bar^alpha tube + sampled noised states + servo labels
Optional (d): per-anchor s * Sigma-hat_t (= (a) scaled by sqrt(s)).
Data: round-1 intervention dataset, its blend datasets, the base policy's
open-loop deltas (bundled npz), the deployed pooled schedule (bundled json).
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

# ── CONFIG ───────────────────────────────────────────────────────────────────
EPISODE = int(os.environ.get("EPISODE", 14))  # source intervention episode (r_dag1, even index)
ROTATE_TO_PATH = True  # rotate the projection plane so the episode's net motion is horizontal
PROJ = "pca_dataset"  # "pca_dataset": PCA plane of ALL round-1 intervention states; "pca": this episode only; "joints": raw pair JX,JY
JX, JY = 2, 0  # only used when PROJ == "joints"
BLEND_TAGS = ("010", "020", "030")
LABEL_FRACS = (0.2, 0.5, 0.8)  # servo-label anchors as fractions of the episode
LABEL_SEED = 7
N_SAMPLES = 40
N_LABELS = 6  # (d): servo labels from noised states at anchors spread along the episode
LABEL_ANCHOR_POWER = 1.6  # anchor fractions u^p, u uniform: >1 biases anchors toward the start
LABEL_DRAW_SEED = 12  # seed for the noised label starts in (d) (change if labels overlap)
LABEL_KEEP = (2, 4, 5)  # (d): which of the N_LABELS anchors (0-based, in episode order) to draw; None = all
W_MARK_FRAC = 0.5  # (b): where along the episode the W radius marker is drawn
W_MARK_SIDE = 1  # (b): +1 / -1 = which side of the path the marker points to
LABEL_MIN_MAHAL = 0.8  # (d): ... and at least this far out (so the arc is visible)
LABEL_MAX_MAHAL = 1.8  # (d): redraw a label start until its projected offset is within this many 1-sigma radii (0 = no rejection)
LABEL_TRIM_MED = (
    1.0  # (d): draw each servo label only until it first comes within N med-steps of the expert path
)
ELL_EVERY = 1
SMOOTH_ANCHORS = (
    9  # (a)/(c): average Sigma_t over +-N anchors before drawing (smooth ribbon instead of slivers)
)
N_OUTLINES = 2  # 1-sigma ellipse outlines per panel (a, c, d), spread along the episode
OUTLINE_EVERY = 36  # draw 1-sigma ellipse OUTLINES on the tubes every N frames (0 = none)
SHOW_PER_ANCHOR = True
COL_EXPERT, LW_EXPERT = "#9a9a9a", 3.0
COL_BLENDS = ("#7fb3e8", "#4a8fd6", "#2a5f9e")
COL_OPEN_BAND, COL_DEP_BAND = "#b08bc9", "#3b7dd1"
COL_LABELS = ("#f2a541", "#e0821f", "#c95d08")
GRID_2x2 = False  # 2x2 panel grid (wide, equal-aspect panels stay readable) instead of 1x4
Y_STRETCH = 2.5  # display scale of the across-motion axis relative to along-motion (1 = equal aspect; ellipses are stretched by this factor vertically)
SUPTITLE = "Offline policy calibration of the injected noise (lever, 84 px, round 1)"  # "" = none
FIGSIZE = (12.6, 3.6)
DPI = 150
# ─────────────────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUT", os.path.join(HERE, "fig_calibration_steps_lever_r84.png"))
SRC_REPO = "lever_d100_03dagcap_r84_diff_r_dag1"
NARM = 6
sys.path.insert(0, os.path.expanduser("~/code/lerobot/src"))
sys.path.insert(0, os.path.expanduser("~/code/lerobot/my_scripts"))
from lerobot.datasets.dart_relabel import demo_geometry, chunk_labels  # noqa
from lib_sa_rollout import progress_guidance_index  # noqa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa
from matplotlib.path import Path as _Path
from matplotlib.patches import PathPatch as _PathPatch

CACHE = os.path.expanduser("~/.cache/huggingface/lerobot/JennyWWW")


def joints_of(repo, episode):
    df = pd.concat(
        [
            pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.state", "action"])
            for f in glob.glob(f"{CACHE}/{repo}/data/**/*.parquet", recursive=True)
        ]
    )
    g = df[df.episode_index == episode].sort_values("frame_index")
    return (
        np.stack([np.asarray(v, dtype=float) for v in g["observation.state"]]),
        np.stack([np.asarray(v, dtype=float) for v in g["action"]]),
    )


St, Ac = joints_of(SRC_REPO, EPISODE)
D = St[:, :NARM]
dm = np.linalg.norm(np.diff(D, axis=0), axis=1)
ms = float(np.median(dm[dm > 1e-6]))
# fixed 2-D projection used by EVERY panel: points -> (q - mu) @ P, covariances -> P^T S P
if PROJ in ("pca", "pca_dataset"):
    if PROJ == "pca_dataset":
        _all = pd.concat(
            [
                pd.read_parquet(f, columns=["observation.state"])
                for f in glob.glob(f"{CACHE}/{SRC_REPO}/data/**/*.parquet", recursive=True)
            ]
        )
        _fit = np.stack([np.asarray(v, dtype=float)[:NARM] for v in _all["observation.state"]])
    else:
        _fit = D
    _mu = _fit.mean(0)
    _u, _sv, _vt = np.linalg.svd(_fit - _mu, full_matrices=False)
    P2 = _vt[:2].T
    EXPL = _sv[:2] ** 2 / (_sv**2).sum()
    AXL = (
        f"joint-space PC1 [rad]  ({EXPL[0] * 100:.0f}% of {'dataset' if PROJ == 'pca_dataset' else 'path'} variance)",
        f"joint-space PC2 [rad]  ({EXPL[1] * 100:.0f}%)",
    )
else:
    _mu = np.zeros(NARM)
    P2 = np.zeros((NARM, 2))
    P2[JX, 0] = 1
    P2[JY, 1] = 1
    AXL = (f"Joint {JX + 1} position [rad]", f"Joint {JY + 1} position [rad]")
if ROTATE_TO_PATH and PROJ != "joints":
    # rotate the 2-D basis (same plane) so this episode's net motion points along +x:
    # axes read "along motion" / "across motion" and the panels stay wide.
    _d = (D[-1] - D[0]) @ P2
    _th = np.arctan2(_d[1], _d[0])
    _Rm = np.array([[np.cos(-_th), -np.sin(-_th)], [np.sin(-_th), np.cos(-_th)]])
    P2 = P2 @ _Rm.T
    AXL = (
        "joint-space PCA plane, along motion [rad]",
        "across motion [rad]" + (f"  (axis ×{Y_STRETCH:g})" if Y_STRETCH != 1 else ""),
    )


def proj(q):
    return (np.asarray(q) - _mu) @ P2


def cov2(S):
    return P2.T @ S @ P2


PD = proj(D)
# blends (every-2nd source episode -> blend episode EPISODE // 2), escape-checked
blends, settle = {}, {}
for tag in BLEND_TAGS:
    B, _ = joints_of(f"{SRC_REPO}_blend{tag}", EPISODE // 2)
    B = B[:, :NARM]
    j = 0
    dv = []
    for t in range(len(B)):
        j = progress_guidance_index(D, B[t], j, window=48)
        dv.append(np.linalg.norm(B[t] - D[j]) / ms)
    assert not (np.array(dv) >= 15).any(), f"blend {tag} escapes on ep {EPISODE}"
    blends[tag] = B
    dv = np.array(dv)
    settle[tag] = float(np.sqrt(np.mean(dv[40 : min(len(dv), 120)] ** 2)))  # RMS over ticks 40:120, med-steps
# open-loop deltas (base policy, K=4 draws, stride 2)
z = np.load(os.path.join(HERE, "sigma_deltas_lever_r84_q1_dag1.npz"))
d = z[f"ep{EPISODE}_d"]
ta = z[f"ep{EPISODE}_t"].astype(int)
order = np.argsort(ta)
ta = ta[order]


def C_at_anchor(k):  # uncentered 2nd moment (rad^2), 6x6, anchor index k in sorted order
    X = d[order[k]].reshape(-1, NARM).astype(np.float64)
    return X.T @ X / len(X)


C_anchor = [C_at_anchor(k) for k in range(len(ta))]


def C_at(t, smooth=None):  # nearest anchor, optionally averaged over +-smooth anchors
    k = int(np.argmin(np.abs(ta - t)))
    if smooth is None:
        smooth = SMOOTH_ANCHORS
    lo, hi = max(0, k - smooth), min(len(C_anchor), k + smooth + 1)
    return np.mean(C_anchor[lo:hi], axis=0)


# deployed pooled schedule (21 upper-triangle entries, med^2) -> rad^2
IU6 = [(i, j) for i in range(6) for j in range(i, 6)]
row = np.asarray(
    json.load(open(os.path.join(HERE, "noise_schedule_pooled_lever_r84_K1.json")))[f"JennyWWW/{SRC_REPO}"][
        str(EPISODE)
    ][0],
    dtype=float,
)
Sg = np.zeros((6, 6))
for (i, j), v in zip(IU6, row):
    Sg[i, j] = Sg[j, i] = v
Sg *= ms**2
_al = json.load(open(os.path.join(HERE, "dart_scale_lever_r84_K1.json")))
W_MED, DART_SCALE, TR_MEAN = _al["W_med"], _al["scale"], _al["tr_mean_med2"]
# servo labels from noised starts
geom = demo_geometry(St, Ac, NARM)
L = np.linalg.cholesky(Sg + 1e-12 * np.eye(6))
rng = np.random.default_rng(LABEL_SEED)
LABEL_ANCHORS = [int(f * (len(St) - 1)) for f in LABEL_FRACS]
label_curves = []
for t0 in LABEL_ANCHORS:
    qn = D[t0] + L @ rng.standard_normal(6)
    vrec = (St[t0] - St[max(t0 - 1, 0)])[:NARM]
    lab = chunk_labels(qn, float(t0), geom, horizon=64, velocity=vrec)[:, :NARM]
    label_curves.append((t0, qn, lab))

# ── figure ───────────────────────────────────────────────────────────────────
plt.rcParams.update({"font.size": 9.5, "font.family": "DejaVu Sans"})
_np_ = 4 if SHOW_PER_ANCHOR else 3
if GRID_2x2 and _np_ == 4:
    fig, _axs = plt.subplots(2, 2, figsize=(FIGSIZE[0] * 0.75, FIGSIZE[1] * 1.15), sharex=True, sharey=True)
    axes = list(_axs.ravel())
else:
    fig, axes = plt.subplots(1, _np_, figsize=(FIGSIZE[0] * _np_ / 3, FIGSIZE[1]), sharex=True, sharey=True)
_th = np.linspace(0, 2 * np.pi, 120)
_circ = np.stack([np.cos(_th), np.sin(_th)])


def ellipse_pts(center, S2):
    ev, evec = np.linalg.eigh(S2)
    pts = evec @ (np.sqrt(np.clip(ev, 0, None))[:, None] * _circ)
    return np.column_stack([center[0] + pts[0], center[1] + pts[1]])


def union_region(ax, polys, color, alpha, zorder):
    verts, codes = [], []
    for P in polys:
        verts.extend(P.tolist() + [P[0].tolist()])
        codes.extend([_Path.MOVETO] + [_Path.LINETO] * (len(P) - 1) + [_Path.CLOSEPOLY])
    ax.add_patch(
        _PathPatch(_Path(verts, codes), facecolor=color, edgecolor="none", alpha=alpha, zorder=zorder)
    )


def outlines(ax, cov_at, color, lw=1.0, alpha=0.9, every=None):
    if not N_OUTLINES:
        return
    for t in np.linspace(0, len(D) - 1, N_OUTLINES + 2)[1:-1].round().astype(int):
        P = ellipse_pts(PD[t], cov2(cov_at(t)))
        ax.plot(P[:, 0], P[:, 1], "-", lw=0.9, color=color, alpha=0.75, zorder=4.5)


def expert_path(ax):
    ax.plot(
        PD[:, 0], PD[:, 1], "-", lw=LW_EXPERT, color=COL_EXPERT, solid_capstyle="round", zorder=3, alpha=0.9
    )
    ax.plot([PD[0, 0]], [PD[0, 1]], "o", ms=5, color=COL_EXPERT, zorder=3)


frames = range(0, len(D), ELL_EVERY)

ax = axes[0]
expert_path(ax)
union_region(ax, [ellipse_pts(PD[t], cov2(C_at(t))) for t in frames], COL_OPEN_BAND, 0.45, zorder=3.5)
outlines(ax, C_at, "#7a4f96")
ax.set_title(r"(a) open-loop sampling spread $\hat\Sigma_t$ (per anchor)", fontsize=9.5)

ax = axes[1]
expert_path(ax)
union_region(
    ax, [ellipse_pts(PD[t], np.eye(2) * (W_MED * ms) ** 2) for t in frames], "#9a9a9a", 0.18, zorder=2.2
)
for tag, col in zip(BLEND_TAGS, COL_BLENDS):
    B = proj(blends[tag])
    r_lbl = "r=0." + tag.lstrip("0") if tag != "010" else "r=0.10"
    ax.plot(
        B[:, 0],
        B[:, 1],
        "-",
        lw=1.7,
        color=col,
        alpha=0.85,
        zorder=4,
        label=f"blend {r_lbl}:  $x$ = {settle[tag]:.1f}",
    )
ax.set_title(r"(b) blended rollouts settle at $W$", fontsize=9.5)
ax.legend(fontsize=7.5, frameon=False, loc="upper right")
# W as a drawn radius: perpendicular from the path to the tube edge at mid-episode
_tm = int(W_MARK_FRAC * (len(D) - 1))
_tan = PD[min(_tm + 3, len(D) - 1)] - PD[max(_tm - 3, 0)]
_tan /= np.linalg.norm(_tan) + 1e-12
_nrm = np.array([-_tan[1], _tan[0]]) * W_MARK_SIDE
_p0 = PD[_tm]
_p1 = _p0 + _nrm * W_MED * ms
ax.annotate(
    "",
    xy=_p1,
    xytext=_p0,
    arrowprops=dict(arrowstyle="|-|", color="#444444", lw=1.2, shrinkA=0, shrinkB=0, mutation_scale=4),
    zorder=7,
)
ax.text(
    *(_p0 + 0.5 * (_p1 - _p0) + 0.06 * W_MED * ms * _tan),
    r"$W$",
    fontsize=10,
    color="#333333",
    ha="left",
    va="center",
    zorder=8,
)
ax.text(
    0.03,
    0.05,
    rf"$W^2/\mathrm{{tr}}\,\hat\Sigma_t$ = {W_MED:.1f}$^2$/{TR_MEAN:.0f} = {DART_SCALE:.2f}",
    transform=ax.transAxes,
    fontsize=8,
    va="bottom",
    ha="left",
    color="#333333",
)

if SHOW_PER_ANCHOR:
    ax = axes[2]
    expert_path(ax)
    union_region(
        ax, [ellipse_pts(PD[t], DART_SCALE * cov2(C_at(t))) for t in frames], COL_DEP_BAND, 0.30, zorder=3.5
    )
    outlines(ax, lambda t: DART_SCALE * C_at(t), "#1f4f8f")
    ax.set_title(
        rf"(c) per-anchor $s\,\hat\Sigma_t$  (= (a) $\times\sqrt{{s}}$ = {np.sqrt(DART_SCALE):.2f})",
        fontsize=9.5,
    )

rng_d = np.random.default_rng(LABEL_SEED + 1)
q_anchor = D[LABEL_ANCHORS[1]]
samples = q_anchor[None, :] + (L @ rng_d.standard_normal((6, N_SAMPLES))).T
PS = proj(samples)

ax = axes[-1]
expert_path(ax)
union_region(ax, [ellipse_pts(PD[t], cov2(Sg)) for t in frames], COL_DEP_BAND, 0.18, zorder=2.5)
outlines(ax, lambda t: Sg, "#1f4f8f")
# labels start from the sampled states of (d): the N_LABELS farthest from the path
# anchors: N_LABELS evenly spaced quantiles u in (0,1), mapped through u**p (p>1 -> denser near the start),
# excluding the last ~horizon frames so every label has trajectory left to rejoin
_u = (np.arange(N_LABELS) + 0.5) / N_LABELS
_t_lab = np.unique(np.round(_u**LABEL_ANCHOR_POWER * max(len(D) - 25, 1)).astype(int))
_label_pts = []
_rng_l = np.random.default_rng(LABEL_DRAW_SEED)
_cols = plt.cm.YlOrBr(np.linspace(0.45, 0.9, len(_t_lab)))
for _i, (c_, t0) in enumerate(zip(_cols, _t_lab)):
    qn = D[t0] + L @ _rng_l.standard_normal(6)  # draw for every anchor so the kept ones don't change
    if LABEL_MAX_MAHAL > 0:  # keep starts inside/near the drawn 1-sigma tube (projected Mahalanobis)
        _S2i = np.linalg.inv(cov2(Sg) + 1e-12 * np.eye(2))
        for _try in range(200):
            _d2 = proj(qn) - PD[t0]
            _m2 = float(_d2 @ _S2i @ _d2)
            if LABEL_MIN_MAHAL**2 <= _m2 <= LABEL_MAX_MAHAL**2:
                break
            qn = D[t0] + L @ _rng_l.standard_normal(6)
    if LABEL_KEEP is not None and _i not in LABEL_KEEP:
        continue
    _v = (St[t0] - St[max(t0 - 1, 0)])[:NARM]
    lab = chunk_labels(qn, float(t0), geom, horizon=64, velocity=_v)[:, :NARM]
    # trim where the label first rejoins the path (afterwards it just tracks the expert)
    _dmin = np.array([np.linalg.norm(D - q, axis=1).min() for q in lab]) / ms
    _hit = int(np.argmax(_dmin < LABEL_TRIM_MED)) if (_dmin < LABEL_TRIM_MED).any() else len(lab) - 1
    lab = lab[: _hit + 1]
    PL = proj(lab)
    pq = proj(qn)
    _label_pts.append(PL)
    ax.plot(PL[:, 0], PL[:, 1], "-", lw=1.6, color=c_, zorder=6)
    ax.plot([pq[0]], [pq[1]], "o", ms=4.5, color=c_, zorder=7)
ax.set_title(
    ("(d) " if SHOW_PER_ANCHOR else "(c) ") + r"pooled $\bar\Sigma^{\alpha}$: noised states + servo labels",
    fontsize=9.5,
)

for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    if ax.get_subplotspec().is_first_col():
        ax.set_ylabel(AXL[1])
    if ax.get_subplotspec().is_last_row():
        ax.set_xlabel(AXL[0])
    ax.grid(color="#e4e4e4", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_aspect(Y_STRETCH)
_r = 1.6 * float(np.sqrt(np.linalg.eigvalsh(cov2(Sg)).max()))
_pts = np.concatenate([PD] + [proj(B) for B in blends.values()] + _label_pts)
_allx, _ally = _pts[:, 0], _pts[:, 1]
for ax in axes:
    ax.set_xlim(_allx.min() - _r, _allx.max() + _r)
    ax.set_ylim(_ally.min() - _r, _ally.max() + _r)
plt.tight_layout()
if SUPTITLE:
    _top = max(
        a.get_position().y1 for a in axes
    )  # just above the panel titles (equal-aspect panels are short)
    fig.suptitle(SUPTITLE, fontsize=11.5, y=_top + 0.11)
plt.savefig(OUT, dpi=DPI, bbox_inches="tight")
ev = np.sqrt(np.linalg.eigvalsh(Sg)) / ms
ev2 = np.sqrt(np.linalg.eigvalsh(cov2(Sg))) / ms
print(
    f"pooled ellipse in the plotted plane ({PROJ}): sigma {ev2.round(2).tolist()} med-steps, aspect {ev2[1] / ev2[0]:.2f}:1"
)
print(
    f"wrote {OUT} | ep {EPISODE} T={len(St)} | pooled sigma per joint (med) {np.round(np.sqrt(np.diag(Sg)) / ms, 2).tolist()} | principal {np.round(ev, 2).tolist()} | alpha {DART_SCALE:.3f}"
)
