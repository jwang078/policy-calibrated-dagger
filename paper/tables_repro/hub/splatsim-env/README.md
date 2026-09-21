---
license: mit
tags: [lerobot, envhub, robotics, simulation, gaussian-splatting]
---
# SplatSim (EnvHub entry)

[SplatSim](https://github.com/jwang078/SplatSim) renders a PyBullet robot inside a Gaussian-splat scan of a real
scene, so image policies train and evaluate on photoreal observations with a physics engine underneath. It is the
simulator behind *Policy-Calibrated DAgger* (ICRA 2027): a planar 3-joint reaching task with obstacles and a UR5
engine-lever task rendered from a scan of the real cell.

## Setup

SplatSim needs a CUDA rasterizer and PyBullet, so it is installed from source, not from this repo:

```bash
git clone https://github.com/jwang078/SplatSim && cd SplatSim && ./install.sh   # also installs lerobot_env_splatsim
hf download JennyWWW/splatsim-scenes --repo-type dataset --local-dir /tmp/scenes
tar xzf /tmp/scenes/robot_iphone_w_engine_curtain.tar.gz -C data/stages           # the lever scene
```

## Use

```python
from lerobot.envs import make_env
envs = make_env("JennyWWW/splatsim-env", trust_remote_code=True)
# SPLATSIM_TASK=planar_3joint (default) | upright_small_engine_new ; SPLATSIM_PORT=<port of a running node>, else in-process
```

or, once the plugin is installed, straight from the CLI with the full config surface:

```bash
lerobot-eval --env.type=splatsim --env.task=planar_3joint --env.robot_name=planar_3joint \
             --env.eval_benchmark_repo_id=JennyWWW/eval_planar_3joint_benchmark --policy.path=... --eval.n_episodes=10
```

`env.py` is a short forwarder to `lerobot_env_splatsim.SplatSimEnv`; the observation/action spaces, tasks and
options are documented in the SplatSim README.
