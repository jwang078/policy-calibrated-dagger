#!/usr/bin/env bash
# Env profile: 3-joint planar reacher, ORACLE STATE, ZERO obstacles.
#
# The simplest possible scene — pure reach, no avoidance:
#   observation.state             = [joint_1, joint_2, joint_3, gripper] = 4 dims
#   observation.environment_state = [block(x,z), ee(x,z)]                = 4 dims
# This is the GREEN BASELINE: a state-only diffusion policy should hit ~100%
# success in a few thousand steps. If even this fails, the bug is in the
# pipeline / action space / normalization — not the task.
#
# Record a fresh dataset with the simple oracle server:
#   python scripts/launch_nodes.py \
#     --robot sim_pybullet_planar_oracle_simple_interactive \
#     --robot_port 6002 --robot_name planar_3joint
# then GENERATE_TRAJECTORIES, and run compute_relative_stats.sh on it.

ENV_TASK="planar_3joint_oracle_simple"
ROBOT_VARIANT="sim_pybullet_planar_oracle_simple_interactive"
ROBOT_NAME="planar_3joint"
NUM_DOFS=3               # action = 4, state = 4 (joints + gripper)
ENV_STATE_DIM=4          # block (x,z) + gripper EE (x,z); pre-EE datasets are
                         # 2-wide — retrofit via append_ee_to_env_state.py
CAMERAS="state"
DATASET_REPO="JennyWWW/planar_3joint_oracle_simple"
EVAL_BENCHMARK_REPO_ID="JennyWWW/planar_3joint_oracle_simple"
# Image-observation source for the sim server, forwarded to launch_nodes.py
# as --render_mode. The planar env has no splat assets (RENDER_SPLATS=False)
# — the PyBullet camera is its only image source, pinned here explicitly so
# a launch-side default change can't silently alter recorded/eval imagery.
RENDER_MODE="pybullet"
