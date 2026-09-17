#!/usr/bin/env python
"""Calibration-pipeline figure (planar 3-joint task by default; TASK="lever" for the UR arm), joint
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
TASK = os.environ.get(
    "TASK", "planar"
)  # "planar" (3 joints, round 3, W=8.0) or "lever" (6 joints, round 1, W=8.77)
EPISODE = int(
    os.environ.get("EPISODE", 14 if os.environ.get("TASK") == "lever_r84" else 32)
)  # source intervention episode (even index: blends exist for every 2nd episode)
ROTATE_TO_PATH = True  # rotate the projection plane so the episode's net motion is horizontal
PROJ = "pca_dataset"  # "pca_dataset": PCA plane of ALL round-1 intervention states; "pca": this episode only; "joints": raw pair JX,JY
JX, JY = 2, 0  # only used when PROJ == "joints"
BLEND_TAGS = ("010", "020", "030")
LABEL_FRACS = (0.2, 0.5, 0.8)  # servo-label anchors as fractions of the episode
LABEL_SEED = 7
N_SAMPLES = 40
N_LABELS = 6  # (d): servo labels from noised states at anchors spread along the episode
LABEL_ANCHOR_POWER = 1.6  # anchor fractions u^p, u uniform: >1 biases anchors toward the start
LABEL_DRAW_SEED = int(
    os.environ.get("LABEL_SEED", 12)
)  # seed for the noised label starts in (d) (change if labels overlap)
LABEL_KEEP = (2, 4, 5)  # (d): which of the N_LABELS anchors (0-based, in episode order) to draw; None = all
W_MARK_FRAC = 0.5  # (b): where along the episode the W radius marker is drawn
W_MARK_SIDE = 1  # (b): +1 / -1 = which side of the path the marker points to
LABEL_MIN_MAHAL = float(
    os.environ.get("LABEL_MIN_MAHAL", 0.45)
)  # (d): ... and at least this far out (so the arc is visible)
LABEL_MAX_MAHAL = float(
    os.environ.get("LABEL_MAX_MAHAL", 1.0)
)  # (d): redraw a label start until its projected offset is within this many 1-sigma radii (0 = no rejection)
LABEL_TRIM_MED = float(
    os.environ.get("LABEL_TRIM_MED", 0.25)
)  # (d): draw each servo label only until it first comes within N med-steps of the expert path
LABEL_SNAP = (
    os.environ.get("LABEL_SNAP", "1") == "1"
)  # (d): append the nearest expert point so each label visibly rejoins the path
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
SUPTITLE = ""  # e.g. "Offline policy calibration of the injected noise (planar, round 3)"; "" = none
FIGSIZE = (12.6, 3.6)
DPI = 150
_OUT_OVERRIDE = os.environ.get(
    "OUT"
)  # optional output path override (e.g. a .pdf for the merged method figure)
# ─────────────────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
# calibration inputs (sigma deltas, schedules) live in tables_repro/analysis; a copy next to this
# script takes precedence (e.g. a variant not in the tables).
_ANALYSIS = os.path.join(os.path.dirname(HERE), "tables_repro", "analysis")


def _find(name, *dirs):
    for d in list(dirs) + [_ANALYSIS]:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{name}: not in {list(dirs) + [_ANALYSIS]} (bash tables_repro/analysis/unpack_schedules.sh?)")


if TASK == "lever_r84":  # 84px lineage, round 1 (BC on dag1): s = 0.15 -> (c) is visibly narrower than (a)
    R84 = os.path.join(os.path.dirname(HERE), "fig_calibration_steps_lever_r84")
    OUT = os.path.join(R84, "fig_calibration_steps_lever_r84.png")
    SRC_REPO = "lever_d100_03dagcap_r84_diff_r_dag1"
    NARM = 6
    SIGMA_NPZ = _find("sigma_deltas_lever_r84_q1_dag1.npz", R84)
    SCHED_JSON = _find("noise_schedule_pooled_lever_r84_K1.json", R84)
    SCALE_JSON = os.path.join(R84, "dart_scale_lever_r84_K1.json")
elif TASK == "lever":
    OUT = os.path.join(HERE, "fig_calibration_steps_lever.png")
    SRC_REPO = "lever_d100_03dagcap_cam_diff_r_dag1"
    NARM = 6
    SIGMA_NPZ = _find("sigma_deltas_lever_q1_dag1.npz", HERE)  # 224 px lineage, not in the tables
    SCHED_JSON = _find("noise_schedule_pooled_lever_K1.json", HERE)
    SCALE_JSON = os.path.join(HERE, "dart_scale_lever_K1.json")
else:
    OUT = os.path.join(HERE, "fig_calibration_steps.png")
    SRC_REPO = "planar_12_05dag_diff_r_dag3"
    NARM = 3
    SIGMA_NPZ = _find("sigma_deltas_s1q3_dag3.npz", HERE)
    SCHED_JSON = _find("noise_schedule_pooled_s1_K3.json", HERE)
    SCALE_JSON = os.path.join(HERE, "dart_scale_s1_K3.json")
IU = [(i, j) for i in range(NARM) for j in range(i, NARM)]
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
z = np.load(SIGMA_NPZ)
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
row = np.asarray(json.load(open(SCHED_JSON))[f"JennyWWW/{SRC_REPO}"][str(EPISODE)][0], dtype=float)
Sg = np.zeros((NARM, NARM))
if NARM == 3:  # planar v2 rows: [s11, s22, s33, s12, s13, s23]
    Sg = np.array([[row[0], row[3], row[4]], [row[3], row[1], row[5]], [row[4], row[5], row[2]]])
else:  # 21 upper-triangle entries, IU order
    for (i, j), v in zip(IU, row):
        Sg[i, j] = Sg[j, i] = v
Sg *= ms**2
_al = json.load(open(SCALE_JSON))
W_MED, DART_SCALE, TR_MEAN = _al["W_med"], _al["scale"], _al["tr_mean_med2"]
# servo labels from noised starts
geom = demo_geometry(St, Ac, NARM)
L = np.linalg.cholesky(Sg + 1e-12 * np.eye(NARM))
rng = np.random.default_rng(LABEL_SEED)
LABEL_ANCHORS = [int(f * (len(St) - 1)) for f in LABEL_FRACS]
label_curves = []
for t0 in LABEL_ANCHORS:
    qn = D[t0] + L @ rng.standard_normal(NARM)
    vrec = (St[t0] - St[max(t0 - 1, 0)])[:NARM]
    lab = chunk_labels(qn, float(t0), geom, horizon=64, velocity=vrec)[:, :NARM]
    label_curves.append((t0, qn, lab))

# ── CONFIG (style; the GUI built by build_gui.py emits this block) ──────────
CONFIG = {
    "suptitle": "Offline policy calibration of the injected noise",
    "titles": [
        r"(a) Open-loop sampling $\hat\Sigma_t$",
        r"(b) Blend-guided rollouts settle at $W$",
        r"(c) Per-anchor $\Sigma^{\alpha}_t = (W^2/\mathrm{tr}\,\hat\Sigma_t)\,\hat\Sigma_t$",
        r"(d) Pooled $\bar\Sigma^{\alpha}$ and sampled correction trajectories",
    ],
    "xlabel": "PCA projection of joint space (dim 1)",
    "ylabel": "PCA projection of joint space (dim 2)",
    "blend_label": "blend r={r:.2f}:  $x$ = {x:.1f}",
    "legend_xy": [0.98, 0.02],
    "legend_align": "lower right",
    "formula_text": r"$W^2/\mathrm{tr}\,\hat\Sigma_t$ = {WVAL:.1f}$^2$/{TRMEAN:.0f} = {SVAL:.2f}",
    "formula_xy": [0.03, 0.05],
    "w_label": r"$W$",
    "w_mark_frac": 0.5,
    "w_mark_side": 1,
    "col_expert": "#9a9a9a",
    "col_blends": ["#7fb3e8", "#4a8fd6", "#2a5f9e"],
    "col_open": "#b08bc9",
    "col_open_outline": "#7a4f96",
    "col_dep": "#3b7dd1",
    "col_dep_outline": "#1f4f8f",
    "col_labels": ["#f2a541", "#e0821f", "#c95d08"],
    "col_wtube": "#9a9a9a",
    "col_wmark": "#444444",
    "alpha_open": 0.45,
    "alpha_dep": 0.3,
    "alpha_wtube": 0.18,
    "lw_expert": 3,
    "lw_blend": 1.7,
    "lw_label": 1.8,
    "size_title": 9.5,
    "size_label": 9.5,
    "size_tick": 9.5,
    "size_legend": 7.5,
    "size_formula": 8,
    "size_suptitle": 11.5,
}
if os.environ.get("CONFIG_JSON"):
    CONFIG.update(json.loads(os.environ["CONFIG_JSON"]))  # per-render overrides (merged figure)
# ── geometry (everything the panels draw, in plotted-plane coordinates) ─────
_fmt = dict(
    WVAL=W_MED, TRMEAN=TR_MEAN, SVAL=DART_SCALE, SQRT_S=float(np.sqrt(DART_SCALE)), YSTRETCH=Y_STRETCH
)  # UPPERCASE placeholders: never collide with LaTeX
import re as _re


def _sub(t, **kw):
    v = {**_fmt, **kw}
    return _re.sub(
        r"\{([A-Za-z_]\w*)(?::([^}]*))?\}",
        lambda m: format(v[m.group(1)], m.group(2) or "") if m.group(1) in v else m.group(0),
        t,
    )


_th = np.linspace(0, 2 * np.pi, 120)
_circ = np.stack([np.cos(_th), np.sin(_th)])


def ellipse_pts(center, S2):
    ev, evec = np.linalg.eigh(S2)
    pts = evec @ (np.sqrt(np.clip(ev, 0, None))[:, None] * _circ)
    return np.column_stack([center[0] + pts[0], center[1] + pts[1]])


frames = range(0, len(D), ELL_EVERY)


def _outline_frames():
    return np.linspace(0, len(D) - 1, N_OUTLINES + 2)[1:-1].round().astype(int) if N_OUTLINES else []


G = {
    "path": PD,
    "start": PD[0],
    "tube_open": [ellipse_pts(PD[t], cov2(C_at(t))) for t in frames],
    "outl_open": [ellipse_pts(PD[t], cov2(C_at(t))) for t in _outline_frames()],
    "tube_w": [ellipse_pts(PD[t], np.eye(2) * (W_MED * ms) ** 2) for t in frames],
    "blends": {tag: proj(blends[tag]) for tag in BLEND_TAGS},
    "settle": settle,
    "tube_pa": [ellipse_pts(PD[t], DART_SCALE * cov2(C_at(t))) for t in frames],
    "outl_pa": [ellipse_pts(PD[t], DART_SCALE * cov2(C_at(t))) for t in _outline_frames()],
    "tube_dep": [ellipse_pts(PD[t], cov2(Sg)) for t in frames],
    "outl_dep": [ellipse_pts(PD[t], cov2(Sg)) for t in _outline_frames()],
}


# W marker (perpendicular radius at a chosen fraction of the episode)
def w_marker(frac, side):
    tm = int(frac * (len(D) - 1))
    tan = PD[min(tm + 3, len(D) - 1)] - PD[max(tm - 3, 0)]
    tan /= np.linalg.norm(tan) + 1e-12
    nrm = np.array([-tan[1], tan[0]]) * side
    p0 = PD[tm]
    return p0, p0 + nrm * W_MED * ms, tan


# servo labels from noised starts (drawn for every anchor; LABEL_KEEP selects)
_u = (np.arange(N_LABELS) + 0.5) / N_LABELS
_t_lab = np.unique(np.round(_u**LABEL_ANCHOR_POWER * max(len(D) - 25, 1)).astype(int))
_rng_l = np.random.default_rng(LABEL_DRAW_SEED)
_S2i = np.linalg.inv(cov2(Sg) + 1e-12 * np.eye(2))
G["labels"] = []
for _i, t0 in enumerate(_t_lab):
    qn = D[t0] + L @ _rng_l.standard_normal(NARM)
    if LABEL_MAX_MAHAL > 0:
        for _try in range(200):
            _d2 = proj(qn) - PD[t0]
            _m2 = float(_d2 @ _S2i @ _d2)
            if LABEL_MIN_MAHAL**2 <= _m2 <= LABEL_MAX_MAHAL**2:
                break
            qn = D[t0] + L @ _rng_l.standard_normal(NARM)
    if LABEL_KEEP is not None and _i not in LABEL_KEEP:
        continue
    _v = (St[t0] - St[max(t0 - 1, 0)])[:NARM]
    lab = chunk_labels(qn, float(t0), geom, horizon=64, velocity=_v)[:, :NARM]
    _dmin = np.array([np.linalg.norm(D - q, axis=1).min() for q in lab]) / ms
    _hit = int(np.argmax(_dmin < LABEL_TRIM_MED)) if (_dmin < LABEL_TRIM_MED).any() else len(lab) - 1
    _lab = lab[: _hit + 1]
    if LABEL_SNAP:  # close the gap: end the drawn label on the nearest expert point
        _j = int(np.argmin(np.linalg.norm(D - _lab[-1], axis=1)))
        _lab = np.vstack([_lab, D[_j]])
    G["labels"].append({"start": proj(qn), "curve": proj(_lab)})

# ── figure ───────────────────────────────────────────────────────────────────

plt.rcParams.update(
    {
        "font.size": CONFIG["size_tick"],
        "font.family": os.environ.get("FONT", "DejaVu Sans"),
        "mathtext.fontset": os.environ.get("MATHFONT", "dejavusans"),
    }
)
_np_ = 4 if SHOW_PER_ANCHOR else 3
fig, axes = plt.subplots(1, _np_, figsize=(FIGSIZE[0] * _np_ / 3, FIGSIZE[1]), sharex=True, sharey=True)


def union_region(ax, polys, color, alpha, zorder):
    verts, codes = [], []
    for Pp in polys:
        verts.extend(Pp.tolist() + [Pp[0].tolist()])
        codes.extend([_Path.MOVETO] + [_Path.LINETO] * (len(Pp) - 1) + [_Path.CLOSEPOLY])
    ax.add_patch(
        _PathPatch(_Path(verts, codes), facecolor=color, edgecolor="none", alpha=alpha, zorder=zorder)
    )


def outlines(ax, polys, color):
    for Pp in polys:
        ax.plot(Pp[:, 0], Pp[:, 1], "-", lw=0.9, color=color, alpha=0.75, zorder=4.5)


def expert_path(ax):  # drawn above the tubes/outlines (zorder 5) so it keeps its own color
    ax.plot(
        PD[:, 0],
        PD[:, 1],
        "-",
        lw=CONFIG["lw_expert"],
        color=CONFIG["col_expert"],
        solid_capstyle="round",
        zorder=5,
        alpha=1.0,
    )
    ax.plot([PD[0, 0]], [PD[0, 1]], "o", ms=5, color=CONFIG["col_expert"], zorder=5)


titles = [_sub(t) for t in CONFIG["titles"]]
ax = axes[0]
expert_path(ax)
union_region(ax, G["tube_open"], CONFIG["col_open"], CONFIG["alpha_open"], 3.5)
outlines(ax, G["outl_open"], CONFIG["col_open_outline"])
ax.set_title(titles[0], fontsize=CONFIG["size_title"])
ax = axes[1]
expert_path(ax)
union_region(ax, G["tube_w"], CONFIG["col_wtube"], CONFIG["alpha_wtube"], 2.2)
for tag, col in zip(BLEND_TAGS, CONFIG["col_blends"]):
    B = G["blends"][tag]
    ax.plot(
        B[:, 0],
        B[:, 1],
        "-",
        lw=CONFIG["lw_blend"],
        color=col,
        alpha=0.85,
        zorder=6,
        label=_sub(CONFIG["blend_label"], r=int(tag) / 100, x=settle[tag]),
    )
_p0, _p1, _tan = w_marker(CONFIG["w_mark_frac"], CONFIG["w_mark_side"])
ax.annotate(
    "",
    xy=_p1,
    xytext=_p0,
    arrowprops=dict(
        arrowstyle="|-|", color=CONFIG["col_wmark"], lw=1.2, shrinkA=0, shrinkB=0, mutation_scale=4
    ),
    zorder=7,
)
ax.text(
    *(_p0 + 0.5 * (_p1 - _p0) + 0.06 * W_MED * ms * _tan),
    CONFIG["w_label"],
    fontsize=10,
    color=CONFIG["col_wmark"],
    ha="left",
    va="center",
    zorder=8,
)
ax.set_title(titles[1], fontsize=CONFIG["size_title"])
ax.legend(
    fontsize=CONFIG["size_legend"],
    frameon=False,
    loc=CONFIG["legend_align"],
    bbox_to_anchor=tuple(CONFIG["legend_xy"]),
)
ax.text(
    *CONFIG["formula_xy"],
    _sub(CONFIG["formula_text"]),
    transform=ax.transAxes,
    fontsize=CONFIG["size_formula"],
    va="bottom",
    ha="left",
    color="#333333",
)
if SHOW_PER_ANCHOR:
    ax = axes[2]
    expert_path(ax)
    union_region(ax, G["tube_pa"], CONFIG["col_dep"], CONFIG["alpha_dep"], 3.5)
    outlines(ax, G["outl_pa"], CONFIG["col_dep_outline"])
    ax.set_title(titles[2], fontsize=CONFIG["size_title"])
ax = axes[-1]
expert_path(ax)
union_region(ax, G["tube_dep"], CONFIG["col_dep"], CONFIG["alpha_dep"], 3.5)
outlines(ax, G["outl_dep"], CONFIG["col_dep_outline"])
for lab, col in zip(G["labels"], CONFIG["col_labels"]):
    ax.plot(lab["curve"][:, 0], lab["curve"][:, 1], "-", lw=CONFIG["lw_label"], color=col, zorder=6)
    ax.plot([lab["start"][0]], [lab["start"][1]], "o", ms=4.5, color=col, zorder=7)
ax.set_title(titles[3] if SHOW_PER_ANCHOR else titles[3], fontsize=CONFIG["size_title"])
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#e4e4e4", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_aspect(Y_STRETCH)
    ax.set_xlabel(_sub(CONFIG["xlabel"]), fontsize=CONFIG["size_label"])
    ax.tick_params(labelsize=CONFIG["size_tick"])
axes[0].set_ylabel(_sub(CONFIG["ylabel"]), fontsize=CONFIG["size_label"])
_r = 1.6 * float(np.sqrt(np.linalg.eigvalsh(cov2(Sg)).max()))
_pts = np.concatenate([PD] + list(G["blends"].values()) + [l["curve"] for l in G["labels"]])
_allx, _ally = _pts[:, 0], _pts[:, 1]
for ax in axes:
    ax.set_xlim(_allx.min() - _r, _allx.max() + _r)
    ax.set_ylim(_ally.min() - _r, _ally.max() + _r)
plt.tight_layout()
if _OUT_OVERRIDE:
    OUT = _OUT_OVERRIDE
if os.environ.get("SUPTITLE") is not None:
    CONFIG["suptitle"] = os.environ["SUPTITLE"]
if CONFIG["suptitle"]:
    _top = max(a.get_position().y1 for a in axes)
    fig.suptitle(_sub(CONFIG["suptitle"]), fontsize=CONFIG["size_suptitle"], y=_top + 0.11)
plt.savefig(OUT, dpi=DPI, bbox_inches="tight")
# ── data dump for the GUI ────────────────────────────────────────────────────
fig.canvas.draw()
_W, _H = fig.get_size_inches() * fig.dpi


def _box(a):
    b = a.get_position()
    return {
        "x0": b.x0 * _W,
        "x1": b.x1 * _W,
        "y0": (1 - b.y1) * _H,
        "y1": (1 - b.y0) * _H,
        "xlim": list(a.get_xlim()),
        "ylim": list(a.get_ylim()),
    }


_dump = {
    "fig_px": [_W, _H],
    "axes": [_box(a) for a in axes],
    "config": CONFIG,
    "fmt": _fmt,
    "blend_tags": list(BLEND_TAGS),
    "path": PD.tolist(),
    "tube_open": [q.tolist() for q in G["tube_open"]],
    "outl_open": [q.tolist() for q in G["outl_open"]],
    "tube_w": [q.tolist() for q in G["tube_w"]],
    "blends": {k: v.tolist() for k, v in G["blends"].items()},
    "settle": settle,
    "tube_pa": [q.tolist() for q in G["tube_pa"]],
    "outl_pa": [q.tolist() for q in G["outl_pa"]],
    "tube_dep": [q.tolist() for q in G["tube_dep"]],
    "outl_dep": [q.tolist() for q in G["outl_dep"]],
    "labels": [{"start": l["start"].tolist(), "curve": l["curve"].tolist()} for l in G["labels"]],
    "w_marker_fn": {"W_ms": W_MED * ms, "n": len(D)},
    "task": TASK,
    "episode": EPISODE,
}
json.dump(_dump, open(os.path.join(HERE, "fig_data.json"), "w"))
ev = np.sqrt(np.linalg.eigvalsh(Sg)) / ms
ev2 = np.sqrt(np.linalg.eigvalsh(cov2(Sg))) / ms
print(
    f"wrote {OUT} | {TASK} ep {EPISODE} T={len(St)} | pooled ellipse in plane: sigma {ev2.round(2).tolist()} med, aspect {ev2[1] / ev2[0]:.2f}:1 | s={DART_SCALE:.3f}"
)
