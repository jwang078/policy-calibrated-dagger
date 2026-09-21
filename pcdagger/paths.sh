#!/bin/bash
# Shell twin of pcdagger/paths.py: source this to get the same locations with the same defaults.
#   source "$(dirname "$0")/../pcdagger/paths.sh"      (adjust the relative path from your script)
export LEROBOT_ROOT=${LEROBOT_ROOT:-$HOME/code/lerobot}
export SPLATSIM_ROOT=${SPLATSIM_ROOT:-$HOME/code/SplatSim}
export PCDAGGER_OUTPUTS=${PCDAGGER_OUTPUTS:-$LEROBOT_ROOT/outputs}
export LEROBOT_CACHE_DIR=${LEROBOT_CACHE_DIR:-$HOME/.cache/huggingface/lerobot/JennyWWW}
export PCDAGGER_ROOT=${PCDAGGER_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export PCDAGGER_PY=${PCDAGGER_PY:-$HOME/miniforge3/envs/splatsim/bin/python}
