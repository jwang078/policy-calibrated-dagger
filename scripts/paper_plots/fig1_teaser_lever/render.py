#!/usr/bin/env python
"""Figure-1 teaser (lever / SplatSim): geometry + photoreal background.

Runs the SplatSim small-engine server IN-PROCESS (no ZMQ client), restores the
benchmark scenario of one recorded intervention episode, poses the robot at the
anchor frame and renders the base camera with the Gaussian splat. Everything
drawn on top is computed with the SAME pybullet robot and the SAME camera:
  * expert intervention path   : finger-pad-midpoint FK of the recorded states
  * calibrated noise band      : samples of the round's pooled Sigma^alpha
                                 (med^2 -> rad^2 with this episode's median
                                 joint step) around every 2nd path point, FK'd
                                 and projected -> per-point 1-sigma pixel ellipses
  * recovery arcs              : the real dart_relabel servo labels for a few
                                 noised draws at the anchor (6-joint), FK'd
Outputs (next to this file): bg_<tag>.png, geom_<tag>.json, check_<tag>.png
(the check image marks the FK'd anchor EE on the render — it must sit on the
gripper pads).

usage (splatsim env):  python render.py [REPO_SHORT] [EPISODE] [ANCHOR_T] [SCHEDULE_JSON]
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.expanduser("~/code/SplatSim"))
sys.path.insert(0, os.path.expanduser("~/code/lerobot/src"))
HERE = os.path.dirname(os.path.abspath(__file__))
S = "/tmp/claude-1000/-home-jennyw2-code-SplatSim/74005341-2b66-4c2f-b778-056deb6aff45/scratchpad"
REPO = sys.argv[1] if len(sys.argv) > 1 else "lever_d100_03dagcap_r84_diff_r_dag3"
EPISODE = int(sys.argv[2]) if len(sys.argv) > 2 else 7
T0 = int(sys.argv[3]) if len(sys.argv) > 3 else 4
SCHED = (
    sys.argv[4] if len(sys.argv) > 4 else None
)  # default: newest noise_schedule_pooled_lever_r84_K*.json containing REPO
N_BAND = 250
N_ARCS = 3
ARC_SEED = 0
BAND_EVERY = 2
TAG = f"{REPO[-4:]}_ep{EPISODE}_t{T0}"
_B = os.path.expanduser(
    "~/code/lerobot/outputs/training/diffusion_approach_lever_13_smooth_r84_delta_basewrist_d100_03dagcap_r84_ft_dag"
)
CSV = {
    "dag1": f"{_B}1/dagger/interventions/intervention_per_scenario.csv",
    "dag3": f"{S}/r84_dag3_dagger_backup/interventions/intervention_per_scenario.csv",
}.get(REPO[-4:])
# ── episode data ─────────────────────────────────────────────────────────────
C = os.path.expanduser(f"~/.cache/huggingface/lerobot/JennyWWW/{REPO}")
df = pd.concat(
    [
        pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.state", "action"])
        for f in glob.glob(C + "/data/chunk-*/*.parquet")
    ]
)
g = df[df.episode_index == EPISODE].sort_values("frame_index")
St = np.stack([np.asarray(v, float) for v in g["observation.state"]])
Ac = np.stack([np.asarray(v, float) for v in g["action"]])
D = St[:, :6]
dm = np.linalg.norm(np.diff(D, axis=0), axis=1)
ms = float(np.median(dm[dm > 1e-6]))
# scenario of this episode: CSV row == dataset episode (verified by frame matching for dag3 rows 0-9)
# dataset episode -> benchmark scenario, verified by matching each episode's first frame to the recorded
# intervention video at trigger+1 (dag3: identity for rows 0-9; dag1: rows with several cycles yield several episodes)
EP2SCEN = {
    "dag3": {i: i for i in range(10)},
    "dag1": {0: 0, 2: 1, 3: 9, 4: 12, 9: 20, 10: 26, 11: 37, 12: 40},
}
scen_of_row = None
if CSV and os.path.exists(CSV):
    import csv

    scen_of_row = [int(r["scenario_idx"]) for r in csv.DictReader(open(CSV))]
m = EP2SCEN.get(REPO[-4:], {})
scenario = None
if EPISODE in m:
    scenario = scen_of_row[m[EPISODE]] if (REPO.endswith("dag3") and scen_of_row) else m[EPISODE]
if scenario is None:
    raise SystemExit(f"scenario unknown for {REPO} episode {EPISODE}; known: {sorted(m)}")
# ── pooled schedule -> rad^2 ─────────────────────────────────────────────────
if SCHED is None:
    cands = sorted(glob.glob(f"{S}/analysis/noise_schedule_pooled_lever_r84_K*.json"))
    cands = [c for c in cands if f"JennyWWW/{REPO}" in json.load(open(c))]
    if not cands:
        raise SystemExit("no schedule contains this repo yet")
    SCHED = cands[-1]
IU6 = [(i, j) for i in range(6) for j in range(i, 6)]
row = np.asarray(json.load(open(SCHED))[f"JennyWWW/{REPO}"][str(EPISODE)][0], float)
Sg = np.zeros((6, 6))
for v, (i, j) in zip(row, IU6):
    Sg[i, j] = Sg[j, i] = v
Sg = Sg * ms * ms  # med^2 -> rad^2 for this episode
L = np.linalg.cholesky(Sg + 1e-12 * np.eye(6))
print(
    f"episode {EPISODE} (scenario {scenario}) T={len(St)} ms={ms:.4f} rad/frame; schedule {os.path.basename(SCHED)}; per-joint sigma (med) {np.round(np.sqrt(np.diag(Sg)) / ms, 2).tolist()}"
)
# ── in-process splat server ──────────────────────────────────────────────────
from splatsim.configs.mode_config import RenderMode
from splatsim.robots.sim_robot_pybullet_small_engine import (
    UprightRobotSmallEngineNewPybulletRobotServer as Srv,
)

srv = Srv(
    port=6099,
    host="127.0.0.1",
    serve_mode=Srv.SERVE_MODES.EVAL_BENCHMARK,
    camera_names=["base_rgb"],
    robot_name="robot_iphone_w_engine_curtain",
    cam_i=3,
    use_gripper=True,
    image_resize_modes=["stretch"],
    eval_benchmark_repo_id="JennyWWW/eval_splatsim_approach_lever_13_benchmark",
    eval_benchmark_subset=[scenario],
    wrist_cam_ver=2,
    headless=True,
    show_control_gui=False,
    render_mode=RenderMode("splat"),
    debug_fast_control=True,
)
srv._init_lerobot_dataset("JennyWWW/eval_splatsim_approach_lever_13_benchmark", read_only=True)
srv.restore_episode_scenario(scenario)
if hasattr(srv, "_reset_episode_state"):
    srv._reset_episode_state()
pb = srv.pybullet_client
rid = srv.splatsim_robot.sim_id
names = {pb.getJointInfo(rid, j)[12].decode(): j for j in range(pb.getNumJoints(rid))}
PADS = [names[k] for k in ("left_inner_finger_pad", "right_inner_finger_pad")]
SIGNS = list(srv.splatsim_robot.config.articulation_config.joint_signs)[:6]


def pose(
    q6,
):  # deterministic kinematic snap of the 6 arm joints (same joint indexing/signs as command_joint_state)
    for i, qi in enumerate(np.asarray(q6, float)[:6]):
        pb.resetJointState(rid, i + 1, qi * SIGNS[i], targetVelocity=0.0)


def fk(q6):
    pose(q6)
    p = [np.array(pb.getLinkState(rid, l, computeForwardKinematics=True)[0]) for l in PADS]
    return (p[0] + p[1]) / 2


_q = D[0]
pose(_q)
_got = np.array([pb.getJointState(rid, i + 1)[0] * SIGNS[i] for i in range(6)])
print("joint snap check: max|q - read| =", float(np.abs(_got - _q).max()), "signs", SIGNS)
# ── camera ───────────────────────────────────────────────────────────────────
cam = srv.base_camera
c = cam.camera
V = np.asarray(c.world_view_transform.detach().cpu().numpy(), float).T  # world -> camera (4x4)
W, H = int(c.image_width), int(c.image_height)
if getattr(cam, "intrinsic_matrix", None) is not None:
    K = np.asarray(cam.intrinsic_matrix.detach().cpu().numpy(), float)
else:
    fx = (W / 2) / np.tan(float(c.FoVx) / 2)
    fy = (H / 2) / np.tan(float(c.FoVy) / 2)
    K = np.array([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]])
model = getattr(cam, "camera_model", "pinhole")
if model == "fisheye":  # the render is rectified to a pinhole with fx,fy * FISHEYE_RECTIFY_ZOOM
    from splatsim.robots.sim_robot_pybullet_base import FISHEYE_RECTIFY_ZOOM

    K = K.copy()
    K[0, 0] *= FISHEYE_RECTIFY_ZOOM
    K[1, 1] *= FISHEYE_RECTIFY_ZOOM


def project(P):  # (N,3) world -> (N,2) pixels
    P = np.atleast_2d(P)
    Pc = (V @ np.c_[P, np.ones(len(P))].T).T[:, :3]
    return np.c_[K[0, 0] * Pc[:, 0] / Pc[:, 2] + K[0, 2], K[1, 1] * Pc[:, 1] / Pc[:, 2] + K[1, 2]]


if os.environ.get(
    "SCAN"
):  # SCAN=1: FK-project every episode with a known scenario (both repos) and report frame coverage; no render
    for repo in ["lever_d100_03dagcap_r84_diff_r_dag1", "lever_d100_03dagcap_r84_diff_r_dag3"]:
        m2 = EP2SCEN[repo[-4:]]
        C2 = os.path.expanduser(f"~/.cache/huggingface/lerobot/JennyWWW/{repo}")
        df2 = pd.concat(
            [
                pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.state"])
                for f in glob.glob(C2 + "/data/chunk-*/*.parquet")
            ]
        )
        for e in sorted(m2):
            g2 = df2[df2.episode_index == e].sort_values("frame_index")
            D2 = np.stack([np.asarray(v, float) for v in g2["observation.state"]])[:, :6]
            px = project(np.array([fk(q) for q in D2]))
            inside = np.mean((px[:, 0] > 20) & (px[:, 0] < W - 20) & (px[:, 1] > 20) & (px[:, 1] < H - 20))
            print(
                f"SCAN {repo[-4:]} ep{e:2d} scen{m2[e]:3d} T={len(D2):3d} in-frame={inside * 100:3.0f}%  x[{px[:, 0].min():4.0f},{px[:, 0].max():4.0f}] y[{px[:, 1].min():4.0f},{px[:, 1].max():4.0f}]  start({px[0, 0]:.0f},{px[0, 1]:.0f}) end({px[-1, 0]:.0f},{px[-1, 1]:.0f}) span={np.hypot(*(px.max(0) - px.min(0))):.0f}px"
            )
    raise SystemExit(0)
# ── render anchor ────────────────────────────────────────────────────────────
anchor = D[T0]
pose(anchor)
srv.open_gripper()
cls, srv._tick_shadow_masks = (
    srv._capture_obs_snapshot_and_masks()
)  # same pre-render sequence as get_observations
obs = srv.get_observations(render_images=False)
srv.prep_image_rendering(data=obs, cached_link_states=cls)
img = srv.render_image("base_rgb", cached_link_states=cls)
img = np.clip(np.transpose(np.asarray(img), (1, 2, 0)), 0, 1)
from PIL import Image, ImageDraw

Image.fromarray((img * 255).astype(np.uint8)).save(f"{HERE}/bg_{TAG}.png")
print("render", img.shape, "camera", model, "K diag", np.round(np.diag(K), 1).tolist())
# ── geometry ─────────────────────────────────────────────────────────────────
ee_path = np.array([fk(q) for q in D])
px_path = project(ee_path)
px_anchor = project(fk(anchor))[0]
print(
    "EE 3D: path start",
    np.round(ee_path[0], 3).tolist(),
    "anchor",
    np.round(ee_path[T0], 3).tolist(),
    "path end",
    np.round(ee_path[-1], 3).tolist(),
    "| px start/end",
    np.round(px_path[0], 1).tolist(),
    np.round(px_path[-1], 1).tolist(),
)
rng = np.random.default_rng(1)
band = []
for t in range(0, len(D), BAND_EVERY):
    z = rng.standard_normal((N_BAND, 6))
    Q = D[t] + z @ L.T
    pts = project(np.array([fk(q) for q in Q]))
    mu = pts.mean(0)
    cov = np.cov(pts.T)
    band.append({"t": t, "mu": mu.tolist(), "cov": cov.tolist()})
from lerobot.datasets.dart_relabel import chunk_labels, demo_geometry

geom = demo_geometry(St, Ac, 6)
rr = np.random.default_rng(ARC_SEED)
arcs = []
ARC_TIMES = [int(x) for x in os.environ.get("ARC_TIMES", "").split(",") if x] or sorted(
    {T0, len(D) // 4, len(D) // 2, (3 * len(D)) // 4}
)
for ta in ARC_TIMES:  # 8 noised draws per anchor time; the GUI/plot pick which to show
    for i in range(8):
        qn = D[ta] + L @ rr.standard_normal(6)
        vrec = (St[ta] - St[max(ta - 1, 0)])[:6]
        lab = chunk_labels(qn, float(ta), geom, horizon=64, velocity=vrec)[:, :6]
        mah = float(np.sqrt((qn - D[ta]) @ np.linalg.solve(Sg, qn - D[ta])))
        arcs.append(
            {
                "t0": int(ta),
                "draw": i,
                "start_q": qn.tolist(),
                "mahal": mah,
                "px": project(np.array([fk(q) for q in lab])).tolist(),
                "px_start": project(fk(qn))[0].tolist(),
            }
        )
pose(anchor)
json.dump(
    {
        "repo": REPO,
        "episode": EPISODE,
        "scenario": scenario,
        "T0": T0,
        "ms": ms,
        "W": W,
        "H": H,
        "K": K.tolist(),
        "V": V.tolist(),
        "px_path": px_path.tolist(),
        "px_anchor": px_anchor.tolist(),
        "band": band,
        "arcs": arcs,
        "schedule": os.path.basename(SCHED),
        "sigma_med": (np.sqrt(np.diag(Sg)) / ms).tolist(),
    },
    open(f"{HERE}/geom_{TAG}.json", "w"),
)
# ── self-check overlay ───────────────────────────────────────────────────────
chk = Image.fromarray((img * 255).astype(np.uint8))
d = ImageDraw.Draw(chk)
d.line([tuple(p) for p in px_path], fill=(30, 90, 220), width=3)
for a in arcs[::8]:
    d.line([tuple(p) for p in a["px"]], fill=(255, 120, 0), width=2)
x, y = px_anchor
d.ellipse([x - 6, y - 6, x + 6, y + 6], outline=(255, 0, 0), width=3)
chk.save(f"{HERE}/check_{TAG}.png")
print("wrote", f"bg_{TAG}.png geom_{TAG}.json check_{TAG}.png  anchor px", np.round(px_anchor, 1).tolist())
