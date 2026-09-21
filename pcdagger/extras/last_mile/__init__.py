"""Last-mile help wrapper.

Eval-time policy wrapper that detects when the inner policy needs help and
applies one of several help mechanisms. Two pluggable halves:

* ``Detector`` — decides whether help is needed this step.
* ``Helper`` — produces the replacement / blended action.

See ``wrapper.py`` for the wrapper, ``detectors.py`` for Detector backends
(currently ``OracleEEDistanceDetector``), and ``helpers.py`` for Helper
backends (currently ``BlendToGoalBiasHelper``). Stall / RRT / alt-policy
backends are added in subsequent refactor steps.
"""

from lerobot.policies.last_mile.detectors import (
    Detector,
    DetectorState,
    DetectorVerdict,
    NoEEProgressDetector,
    OracleEEDistanceDetector,
    StallDetector,
    build_detector,
)
from lerobot.policies.last_mile.helpers import (
    BlendToGoalBiasHelper,
    Helper,
    HelperOutput,
    RRTToGoalHelper,
    SwapToAltPolicyHelper,
    build_helper,
)
from lerobot.policies.last_mile.joint_history import JointHistoryBuffer
from lerobot.policies.last_mile.wrapper import RAW_STATE_KEY, LastMileWrapper

__all__ = [
    "BlendToGoalBiasHelper",
    "Detector",
    "DetectorState",
    "DetectorVerdict",
    "Helper",
    "HelperOutput",
    "JointHistoryBuffer",
    "LastMileWrapper",
    "NoEEProgressDetector",
    "OracleEEDistanceDetector",
    "RAW_STATE_KEY",
    "RRTToGoalHelper",
    "StallDetector",
    "SwapToAltPolicyHelper",
    "build_detector",
    "build_helper",
]
