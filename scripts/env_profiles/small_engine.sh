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
ROBOT_NAME=""                                # "" = lerobot SplatsimEnv / launch_nodes default (robot_iphone_w_engine_curtain)
NUM_DOFS=6                                    # arm joints; state/action dim = NUM_DOFS + 1 (gripper) = 7
CAMERAS="basewrist"                           # base_rgb + wrist_rgb
# Recorded oracle env_state width: 5 scene objects × (x,y,z) = 15. NEW recordings
# store it (image-based training ignores it by default); add
# --include_env_state_obs=true for image+oracle. Datasets recorded BEFORE this
# lack the column — re-record (or migrate) them to use oracle mode.
ENV_STATE_DIM=15
DATASET_REPO="JennyWWW/splatsim_approach_lever_12_clean"
EVAL_BENCHMARK_REPO_ID="JennyWWW/eval_splatsim_approach_lever_benchmark_1000"
