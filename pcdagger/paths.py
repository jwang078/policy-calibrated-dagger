"""Where things live. Every location is an environment variable with the layout of Jenny's
machine as the default, so scripts and figure code never hard-code `~/code/...` again.

    LEROBOT_ROOT      the lerobot checkout (fork)                 default ~/code/lerobot
    SPLATSIM_ROOT     the SplatSim checkout                       default ~/code/SplatSim
    PCDAGGER_OUTPUTS  training dirs, evals, dataset stats         default $LEROBOT_ROOT/outputs
    LEROBOT_CACHE_DIR the LeRobot dataset cache for JennyWWW/*    default ~/.cache/huggingface/lerobot/JennyWWW

Shell scripts get the same values from `pcdagger/paths.sh` (source it).
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LEROBOT_ROOT = Path(os.environ.get("LEROBOT_ROOT", "~/code/lerobot")).expanduser()
SPLATSIM_ROOT = Path(os.environ.get("SPLATSIM_ROOT", "~/code/SplatSim")).expanduser()
OUTPUTS = Path(os.environ.get("PCDAGGER_OUTPUTS", LEROBOT_ROOT / "outputs")).expanduser()
TRAINING = OUTPUTS / "training"
EVAL300 = OUTPUTS / "eval300"
DATASET_STATS = OUTPUTS / "dataset_stats"
DATASET_CACHE = Path(
    os.environ.get("LEROBOT_CACHE_DIR", "~/.cache/huggingface/lerobot/JennyWWW")
).expanduser()
PAPER = REPO_ROOT / "paper"
