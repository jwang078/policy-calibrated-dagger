#!/usr/bin/env bash
# Env profile: 3-joint planar reacher, ORACLE STATE (no image).
#
# observation.state             = [joint_1, joint_2, joint_3, gripper]   = 4 dims
# observation.environment_state = [block(x,z), obstacle_1(x,z),
#                                  obstacle_2(x,z)]                       = 6 dims
# Exact object coords in, joint deltas out — a pure control problem (no
# perception), so a small state-only diffusion policy trains fast. Same
# 2-obstacle reach scene as the vision profile; use this to isolate control from
# vision. Trained with --cameras=state (no image features).
#
# NOTE: record a fresh dataset with the oracle server so the state carries the
# object coords:
#   python scripts/launch_nodes.py --robot sim_pybullet_planar_oracle_interactive \
#     --robot_port 6002 --robot_name planar_3joint
# then GENERATE_TRAJECTORIES in the GUI. The vision dataset (JennyWWW/planar_3joint)
# won't have the extra state dims. Also run compute_relative_stats.sh on it.

ENV_TASK="planar_3joint_oracle"
ROBOT_VARIANT="sim_pybullet_planar_oracle_interactive"
ROBOT_NAME="planar_3joint"
NUM_DOFS=3                # action = 4, state = 4 (joints + gripper)
ENV_STATE_DIM=6          # observation.environment_state = 3 objects x 2 coords (x,z)
CAMERAS="state"          # state-only: no image features
DATASET_REPO="JennyWWW/planar_3joint_oracle"
EVAL_BENCHMARK_REPO_ID="JennyWWW/planar_3joint_oracle"
