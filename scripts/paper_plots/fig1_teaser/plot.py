#!/usr/bin/env python
"""Figure 1 teaser: real-scene render of one intervention with the calibrated
noise tube and synthesized recovery chunks.

Everything styleable lives in the CONFIG block below. The geometry is exact:
the camera is the sim's own (verified sub-pixel), the EE convention is the
finger-pad midpoint calibrated to the rendered fingertips, the recovery arcs
are the actual dart_relabel servo labels (trimmed where they touch the path),
and the band width is the measured mean 1-sigma EE projection of the round's
pooled noise.

Run:  python my_scripts/paper_plots/fig1_teaser/plot.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

# ── CONFIG ───────────────────────────────────────────────────────────────────
EPISODE = 24  # source intervention episode (r_dag3)
FRAME = 61  # anchor frame (the rendered robot state)
NOISED_PICKS = (3, 5)  # which saved draws become the recovery arcs
RES = 1024  # render resolution
SCHEDULE = "noise_schedule_pooled_s1_K3.json"
SOURCE_REPO = "planar_12_05dag_diff_r_dag3"

COL_BG = (179, 180, 203)
COL_PATH = "#1a5fb4"
COL_ARCS = ("#ff7800", "#ff7800")
COL_BAND = (0.21, 0.52, 0.89)
BAND_STROKES = ((3.12, 0.10), (2.28, 0.16), (1.44, 0.24))
LW_PATH = 5.9
LW_ARC = 4.4
CROP_TOP = 0.33  # fraction of image height cropped off the top
FIG_DPI = 140

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig1_teaser.png")
SCRATCH = os.path.dirname(os.path.abspath(__file__))
SCHED_PATH = os.path.join(SCRATCH, SCHEDULE)
if not os.path.exists(SCHED_PATH):  # the schedules live in tables_repro/analysis (unpack_schedules.sh)
    SCHED_PATH = os.path.join(os.path.dirname(SCRATCH), "tables_repro", "analysis", SCHEDULE)
STATES_JSON = os.path.join(SCRATCH, "servo_states_ep24_f61.json")  # saved anchor + draws
# ─────────────────────────────────────────────────────────────────────────────

sys.path.append(os.path.expanduser("~/code/lerobot/my_scripts"))
sys.path.insert(0, os.path.expanduser("~/code/lerobot/src"))
import dart_sim_video as dsv  # noqa: E402
import matplotlib  # noqa: E402
import pybullet as pb  # noqa: E402

from lerobot.datasets.dart_relabel import chunk_labels, demo_geometry  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ── data ─────────────────────────────────────────────────────────────────────
CACHE = os.path.expanduser("~/.cache/huggingface/lerobot/JennyWWW")
df = pd.read_parquet(f"{CACHE}/{SOURCE_REPO}/data/chunk-000/file-000.parquet")
g = df[df.episode_index == EPISODE].sort_values("frame_index")
St = np.stack([np.asarray(v, dtype=float) for v in g["observation.state"]])
Ac = np.stack([np.asarray(v, dtype=float) for v in g["action"]])
env = np.stack([np.asarray(v, dtype=float) for v in g["observation.environment_state"]])
T = FRAME
states = json.load(open(STATES_JSON))
anchor = np.array(states["anchor_q"])
ghosts = [np.array(states["noised_q"][i]) for i in NOISED_PICKS]

geom = demo_geometry(St, Ac, 3)
vrec = (St[T] - St[T - 1])[:3]
labels = [chunk_labels(qn, float(T), geom, horizon=64, velocity=vrec)[:, :3] for qn in ghosts]

# ── FK: EE = finger-pad midpoint ─────────────────────────────────────────────
cid = pb.connect(pb.DIRECT)
body = pb.loadURDF(dsv.PLANAR_URDF, useFixedBase=True, physicsClientId=cid)
PAD_L, PAD_R = 8, 13


def fk_raw(q):
    dsv._pose(pb, cid, body, q)
    pl = np.array(pb.getLinkState(body, PAD_L, computeForwardKinematics=True, physicsClientId=cid)[0])
    pr = np.array(pb.getLinkState(body, PAD_R, computeForwardKinematics=True, physicsClientId=cid)[0])
    return ((pl + pr) / 2)[[0, 2]]


# ── background render (native colors, sim-matched scene + camera) ────────────
dsv._scene_objects(pb, cid, env[T])
dsv._pose(pb, cid, body, anchor)
view = pb.computeViewMatrix(list(dsv.CAMERA_EYE), list(dsv.CAMERA_TARGET), [0, 0, 1])
proj = pb.computeProjectionMatrixFOV(dsv.CAMERA_FOV, 1.0, 0.01, 20.0)
_, _, rgb, _, _ = pb.getCameraImage(
    RES, RES, view, proj, renderer=pb.ER_TINY_RENDERER, lightDirection=[0.4, -1.0, 0.8], physicsClientId=cid
)
img = np.reshape(rgb, (RES, RES, 4))[:, :, :3].astype(np.uint8)
img[(img == 255).all(axis=2)] = np.array(COL_BG, dtype=np.uint8)
H, W = img.shape[:2]

# exact camera projection: world (x, z at y=0) -> image pixels
Vm = np.array(view).reshape(4, 4, order="F")
Pm = np.array(proj).reshape(4, 4, order="F")


def to_px(w):
    c = Pm @ (Vm @ np.array([w[0], 0.0, w[1], 1.0]))
    n = c[:3] / c[3]
    return np.array([(n[0] + 1) / 2 * W, (1 - n[1]) / 2 * H])


# calibrate EE to the VISIBLE fingertip gap (pale finger pixels near estimate)
f = img.astype(float)
r_, g_, b_ = f[..., 0], f[..., 1], f[..., 2]
p0 = to_px(fk_raw(anchor))
yy, xx = np.mgrid[0:H, 0:W]
near = (xx - p0[0]) ** 2 + (yy - p0[1]) ** 2 <= (14 * W / 224) ** 2
pale = (r_ > 120) & (g_ > 120) & (b_ > 120) & (np.abs(r_ - b_) < 30) & near
dw = np.zeros(2)
if pale.sum() >= 6:
    pys, pxs = np.nonzero(pale)
    tip_px = np.array([pxs.mean(), pys.mean()])
    e = 0.01
    Jp = np.column_stack(
        [(to_px(fk_raw(anchor) + [e, 0]) - p0) / e, (to_px(fk_raw(anchor) + [0, e]) - p0) / e]
    )
    dw = np.linalg.solve(Jp, tip_px - p0)


def fk(q):
    return fk_raw(q) + dw


def P(w):
    return tuple(to_px(w))


# ── figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8.0, 8.0), dpi=FIG_DPI)
ax.imshow(img)
ax.set_axis_off()

# noise tube: constant width = mean 1-sigma EE projection of the pooled noise
sched = json.load(open(SCHED_PATH))
prof = np.asarray(sched[f"JennyWWW/{SOURCE_REPO}"][str(EPISODE)], dtype=float)
dd = np.linalg.norm(np.diff(St[:, :3], axis=0), axis=1)
ms = float(np.median(dd[dd > 1e-6]))
row = prof[0]
Sg = np.array([[row[0], row[3], row[4]], [row[3], row[1], row[5]], [row[4], row[5], row[2]]]) * ms**2
jac = dsv.make_planar_jacobian()
sigs = []
for t in range(T, len(St), 4):
    J = jac(St[t, :3])
    sigs.append(np.sqrt(0.5 * np.trace(J @ Sg @ J.T)))
sigma_ee = float(np.mean(sigs))
q0 = fk(St[T, :3])
sigma_px = float(np.linalg.norm(to_px(q0 + [sigma_ee, 0]) - to_px(q0)))
pt_per_px = 72.0 / FIG_DPI
ppb = np.array([P(fk(St[t, :3])) for t in range(T, len(St))])
for wmul, alpha in BAND_STROKES:
    ax.plot(
        ppb[:, 0],
        ppb[:, 1],
        "-",
        lw=2 * sigma_px * wmul * pt_per_px,
        color=COL_BAND,
        alpha=alpha,
        solid_capstyle="round",
        zorder=2.5,
    )
print(f"band: sigma_ee = {sigma_ee * 100:.1f} cm = {sigma_px:.1f} px")

# recovery arcs (real servo labels, trimmed where they first touch the path)
path_ee = np.array([fk(St[t, :3]) for t in range(len(St))])


def trim(lab):
    ees = np.array([fk(q) for q in lab])
    d = np.array([np.linalg.norm(path_ee - e, axis=1).min() for e in ees])
    hit = np.argmax(d < 0.008) if (d < 0.008).any() else len(d) - 1
    return ees[: hit + 1]


for lab, col in zip(labels, COL_ARCS):
    cp = np.array([P(e) for e in trim(lab)])
    ax.plot(cp[:, 0], cp[:, 1], "-", lw=LW_ARC, color=col, solid_capstyle="round", zorder=4)

# expert path, drawn on top
pp = np.array([P(fk(St[t, :3])) for t in range(T, len(St), 2)])
ax.plot(pp[:, 0], pp[:, 1], "-", lw=LW_PATH, color=COL_PATH, alpha=0.95, solid_capstyle="round", zorder=5)

ax.set_ylim(H, H * CROP_TOP)
plt.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
fig.set_size_inches(8.0, 8.0 * (1 - CROP_TOP))
plt.savefig(OUT, dpi=FIG_DPI)
pb.disconnect(cid)
print("wrote", OUT)
