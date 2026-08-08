#!/usr/bin/env bash
# Env profile: UR5 "upright_small_engine_new" (the historical default).
#
# Sourced by train_sweep.sh / dagger_orchestrate_sweep.sh via
# `--env_profile=small_engine`. Defines every environment-specific value so
# swapping environments is a single flag. Sourcing this reproduces the scripts'
# previous hardcoded UR5 behavior exactly.
#
# Precedence: built-in script defaults < this profile < explicit CLI flags.

ENV_TASK="upright_small_engine_new"          # lerobot --env.task + splatsim register_env key
ROBOT_VARIANT="sim_ur_pybullet_small_engine_new_interactive"  # launch_nodes.py --robot (dagger server launch)
# Splat robot-object name (launch_nodes --robot_name / SA wrapper URDF).
# Pinned explicitly to the value launch_nodes would resolve anyway from
# UprightRobotSmallEngineNewPybulletRobotServer.DEFAULT_ROBOT_NAME, so the
# SA wrapper can load the URDF eagerly (blend step runs without oracle
# receipt) instead of relying on the ''-defers-to-oracle lazy path.
ROBOT_NAME="robot_iphone_w_engine_curtain"
NUM_DOFS=6                                    # arm joints; state/action dim = NUM_DOFS + 1 (gripper) = 7
CAMERAS="basewrist"                           # base_rgb + wrist_rgb
# Recorded oracle env_state width: [box1(x,y), box2(x,y), ee(x,y,z)] = 7.
# Only the RANDOMIZED boxes + the EE are recorded — engine/table/wall are
# pinned per-scenario, so their coords were constant dims in the historical
# 15-wide layout (5 objects × xyz, no EE). Changed 2026-08-04; datasets and
# checkpoints from the 15-wide era (e.g. approach_lever_13_smooth) are NOT
# width-compatible — re-record (or migrate: slice box x,y + FK-append EE).
# NEW recordings store it (image-based training ignores it by default); add
# --include_env_state_obs=true for image+oracle.
ENV_STATE_DIM=7
DATASET_REPO="JennyWWW/splatsim_approach_lever_13_smooth"
EVAL_BENCHMARK_REPO_ID="JennyWWW/eval_splatsim_approach_lever_13_benchmark"
# Image-observation source for the sim server, forwarded to launch_nodes.py
# as --render_mode. This env's datasets are recorded with the Gaussian-splat
# base camera, so eval/blend imagery MUST be splat too (a PyBullet-camera
# render is a big visual covariate shift vs the training data). Pinned here
# explicitly so launch-side defaults / GUI dropdown state can't silently
# flip it (observed: eval videos rendered by the PyBullet camera).
RENDER_MODE="splat"
