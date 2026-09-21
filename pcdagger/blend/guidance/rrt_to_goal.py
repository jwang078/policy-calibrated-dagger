"""Compatibility shim — the canonical RRT-to-goal planner moved to SplatSim.

The `RRTToGoalPlanner` and its helpers now live in `splatsim.utils.rrt_to_goal`
(single source of truth, shared with SplatSim's trajectory-generation demo
recorder so the demo planner and the DAgger/SA intervention planner are the
same code). SplatSim owns the low-level RRT primitives this planner wraps
(`splatsim.utils.rrt_path_utils`), so co-locating the planner there removes the
duplicate implementation that SplatSim's `TrajectoryGenerator` used to mirror.

This module re-exports the canonical planner LAZILY (PEP 562 `__getattr__`) and
keeps the lerobot-side `RRTMode` alias, so every existing caller keeps importing
from `lerobot.policies.rrt_to_goal` unchanged:
  * `guidance/rrt_source.py`      → `RRTToGoalPlanner`, `PathSelectionStrategy`,
                                     `RRTRuntimeState`, `RRTPlanningError`,
                                     `extract_task_goal`
  * `guidance/oracle_goal_source.py` → `extract_task_goal`
  * `shared_autonomy_wrapper.py`  → `RRTMode` (+ lazy `check_chunk_collision`)
  * `shared_autonomy_gui.py`      → `RRTMode`
  * `scripts/intervention_controller.py` → `RRTMode`

Why lazy: `import lerobot.policies.rrt_to_goal` and the `RRTMode` alias must keep
working WITHOUT splatsim installed (splatsim is not a lerobot dependency). Only
touching a planner symbol (e.g. `RRTToGoalPlanner`) pulls splatsim in — which
only happens in intervention/trajectory-gen contexts where splatsim is present.

`RRTMode` is defined here (not re-exported from splatsim) so it resolves without
splatsim AND is the exact same `GuidanceMode` object the canonical planner's
`RRTRuntimeState.mode` uses (splatsim imports the same `GuidanceMode`), keeping
`InterventionController`'s `state.mode == RRTMode.IDLE/PLANNING/EXECUTING`
comparisons valid across the repo boundary.
"""

from __future__ import annotations

from lerobot.policies.guidance.base import GuidanceMode

RRTMode = GuidanceMode


def __getattr__(name: str):
    """Lazily resolve any other public symbol from `splatsim.utils.rrt_to_goal`
    (RRTToGoalPlanner, PathSelectionStrategy, IkGoalSelectionStrategy,
    RRTRuntimeState, RRTPlanningError, extract_task_goal, check_chunk_collision,
    …). Deferred import keeps plain lerobot usage splatsim-free (PEP 562)."""
    # Don't trigger the splatsim import for dunder probes (pickle, help(),
    # importlib machinery, etc.) — those are never planner symbols.
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)
    from splatsim.utils import rrt_to_goal as _canonical

    try:
        return getattr(_canonical, name)
    except AttributeError as exc:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r} "
            f"(not found in canonical splatsim.utils.rrt_to_goal either)"
        ) from exc
