"""SplatSim on lerobot's EnvHub.

    from lerobot.envs import make_env
    envs = make_env("JennyWWW/splatsim-env", trust_remote_code=True)

Everything lives in the `lerobot_env_splatsim` plugin shipped with SplatSim (github.com/jwang078/SplatSim);
this file only forwards to it. Install SplatSim first (`install.sh`; see requirements.txt) and unpack a scene
from JennyWWW/splatsim-scenes into SplatSim/data/stages/. Once the plugin is installed, `lerobot-eval
--env.type=splatsim ...` works without this file at all; the hub entry exists so the environment is
discoverable the same way as the other EnvHub environments.
"""

from __future__ import annotations

import os

from lerobot_env_splatsim import SplatSimEnv

# Presets for the string API (`make_env("JennyWWW/splatsim-env")`): the two paper tasks, selected with
# SPLATSIM_TASK; SPLATSIM_PORT points at a running SplatSim node (scripts/launch_nodes.py), otherwise the
# simulator starts in-process. With an EnvConfig (`--env.type=splatsim` through the plugin) the config is
# used as is and these are ignored.
_PRESETS = {
    # planar 3-joint reaching with obstacles (state-only policies; no scan needed)
    "planar_3joint": dict(
        task="planar_3joint", robot_name="planar_3joint", camera_names=["base_rgb"],
        image_resize_modes=["letterbox"], num_dofs=3, state_dim=4, action_dim=4, env_state_dim=8,
        eval_benchmark_repo_id="JennyWWW/eval_planar_3joint_benchmark",
    ),
    # UR5 engine lever, rendered from the robot_iphone_w_engine_curtain scan (JennyWWW/splatsim-scenes)
    "upright_small_engine_new": dict(
        task="upright_small_engine_new", robot_name="robot_iphone_w_engine_curtain",
        camera_names=["base_rgb", "wrist_rgb"], image_resize_modes=["stretch"],
        num_dofs=6, state_dim=7, action_dim=7, env_state_dim=0,
        eval_benchmark_repo_id="JennyWWW/eval_splatsim_approach_lever_13_benchmark",
    ),
}


def _default_config() -> SplatSimEnv:
    task = os.environ.get("SPLATSIM_TASK", "planar_3joint")
    if task not in _PRESETS:
        raise ValueError(f"SPLATSIM_TASK={task!r}; known presets: {sorted(_PRESETS)}")
    port = os.environ.get("SPLATSIM_PORT")
    return SplatSimEnv(**_PRESETS[task], external_port=int(port) if port else None, headless=True,
                       use_gripper=True, episode_length=1000)


def make_env(n_envs: int = 1, use_async_envs: bool = False, cfg=None):
    """Return {suite: {task_id: VectorEnv}} for the SplatSim task (the shape lerobot's evaluator expects)."""
    if not isinstance(cfg, SplatSimEnv):
        cfg = _default_config()
    return cfg.create_envs(n_envs=n_envs, use_async_envs=use_async_envs)
