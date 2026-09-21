"""Pluggable guidance sources for the SA wrapper.

See `base.py` for the protocol contract. Concrete sources land in sibling
modules over the rollout (rrt_source, observation_teleop_source,
oracle_goal_source) and get added to re-exports as each one is wired up.
"""

"""Guidance package init.

Only base types are re-exported at the package level. Concrete sources
(`rrt_source`, `oracle_goal_source`, `observation_teleop_source`) and the
back-compat view are imported via their full submodule path. This avoids
a circular import: `rrt_to_goal.py` imports `GuidanceMode` from this
package, and concrete sources import back into `rrt_to_goal.py` — putting
both on the package's top-level import chain would deadlock loading.
"""

from lerobot.policies.guidance.base import (
    GuidanceCallCtx,
    GuidanceMode,
    GuidanceSource,
    GuidanceSourceState,
    GuidanceStepResult,
    IntegrationMode,
)

__all__ = [
    "GuidanceCallCtx",
    "GuidanceMode",
    "GuidanceSource",
    "GuidanceSourceState",
    "GuidanceStepResult",
    "IntegrationMode",
]
