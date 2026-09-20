#!/usr/bin/env python
"""Render every frame of the teaser's intervention episode with the same splat
camera as render.py (the figure background), for the paper video.

usage:  python render_episode_video.py [REPO_SHORT] [EPISODE] [ANCHOR_T]
        (defaults = the Fig. 1 teaser: lever_d100_03dagcap_r84_diff_r_dag1, episode 9, anchor 4)

Writes video/<tag>/frames/f%04d.png (640x359 like bg_<tag>.png) and video/<tag>/joints.npy;
make_video_assets.sh turns them into the stills and clips.
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
REPO = sys.argv[1] if len(sys.argv) > 1 else "lever_d100_03dagcap_r84_diff_r_dag1"
EPISODE = int(sys.argv[2]) if len(sys.argv) > 2 else 9
T0 = int(sys.argv[3]) if len(sys.argv) > 3 else 4
TAG = f"{REPO[-4:]}_ep{EPISODE}_t{T0}"
OUT = os.path.join(HERE, "video", TAG)
os.makedirs(f"{OUT}/frames", exist_ok=True)
G = json.load(open(f"{HERE}/geom_{TAG}.json"))  # scenario + camera check
scenario = int(G["scenario"])

C = os.path.expanduser(f"~/.cache/huggingface/lerobot/JennyWWW/{REPO}")
df = pd.concat(
    pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.state"])
    for f in glob.glob(C + "/data/chunk-*/*.parquet")
)
g = df[df.episode_index == EPISODE].sort_values("frame_index")
D = np.stack([np.asarray(v, float)[:6] for v in g["observation.state"]])
np.save(f"{OUT}/joints.npy", D)
print(f"episode {EPISODE} (scenario {scenario}): {len(D)} frames, anchor {T0}")

from splatsim.configs.mode_config import RenderMode  # noqa: E402
from splatsim.robots.sim_robot_pybullet_small_engine import (  # noqa: E402
    UprightRobotSmallEngineNewPybulletRobotServer as Srv,
)

srv = Srv(
    port=6098,
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
SIGNS = list(srv.splatsim_robot.config.articulation_config.joint_signs)[:6]


def pose(q6):
    for i, qi in enumerate(np.asarray(q6, float)[:6]):
        pb.resetJointState(rid, i + 1, qi * SIGNS[i], targetVelocity=0.0)


from PIL import Image  # noqa: E402

for t, q in enumerate(D):
    pose(q)
    srv.open_gripper()
    cls, srv._tick_shadow_masks = srv._capture_obs_snapshot_and_masks()
    obs = srv.get_observations(render_images=False)
    srv.prep_image_rendering(data=obs, cached_link_states=cls)
    img = srv.render_image("base_rgb", cached_link_states=cls)
    img = np.clip(np.transpose(np.asarray(img), (1, 2, 0)), 0, 1)
    Image.fromarray((img * 255).astype(np.uint8)).save(f"{OUT}/frames/f{t:04d}.png")
    if t % 20 == 0:
        print(f"  frame {t}/{len(D)} {img.shape[1]}x{img.shape[0]}")
# the anchor frame must match the figure background
bg = np.asarray(Image.open(f"{HERE}/bg_{TAG}.png")).astype(int)
fr = np.asarray(Image.open(f"{OUT}/frames/f{T0:04d}.png")).astype(int)
print("anchor frame vs bg_<tag>.png: max |diff| =", int(np.abs(bg - fr).max()) if bg.shape == fr.shape else f"shape {bg.shape} vs {fr.shape}")
print("done", OUT)
