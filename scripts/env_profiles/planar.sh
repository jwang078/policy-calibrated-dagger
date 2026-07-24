#!/usr/bin/env bash
# Env profile: 3-joint planar reacher "planar_3joint".
#
# Sourced by train_sweep.sh / dagger_orchestrate_sweep.sh via
# `--env_profile=planar`. See small_engine.sh for the field contract.
#
# The planar arm is a fast, splat-free debug env (see SplatSim
# sim_robot_pybullet_planar.py). It has 3 arm joints (+ gripper), a single
# base camera (no wrist), and its image obs come from PyBullet's getCameraImage.
#
# NOTE: for inline eval, either
#   * launch the planar server separately and pass --env_external_port=PORT
#     (e.g. `python scripts/launch_nodes.py --robot sim_pybullet_planar_interactive
#      --robot_port 6002 --robot_name planar_3joint --no_camera_rendering`), or
#   * let train_sweep spawn it in-process (task `planar_3joint` is registered in
#     splatsim/gym_env.py).
# Relative-action training also needs the stats sidecar: run
#   my_scripts/compute_relative_stats.sh for DATASET_REPO first.

ENV_TASK="planar_3joint"                      # lerobot --env.task + splatsim register_env key
ROBOT_VARIANT="sim_pybullet_planar_interactive"  # launch_nodes.py --robot (dagger server launch)
ROBOT_NAME="planar_3joint"                    # objects.yaml key / --env.robot_name / server --robot_name
NUM_DOFS=3                                     # state/action dim = 4
CAMERAS="base"                                # base_rgb only (no wrist camera)
# Recorded oracle env_state width: block + 2 obstacles × (x,z) = 6. Datasets
# store it, so image-only (default) OR image+oracle (--include_env_state_obs=true)
# OR oracle-only (--cameras=state) all train from the SAME dataset.
ENV_STATE_DIM=6
DATASET_REPO="JennyWWW/planar_3joint"
# Inline-eval benchmark. Defaults to the training dataset (eval on train
# scenarios) — swap for a dedicated held-out planar benchmark when you have one.
EVAL_BENCHMARK_REPO_ID="JennyWWW/eval_planar_3joint_benchmark"
