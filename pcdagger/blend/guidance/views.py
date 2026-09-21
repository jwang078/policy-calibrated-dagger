"""Back-compat proxy views for the SA wrapper's old `_rrt` attribute.

External callers used to directly access `wrapper._rrt.mode`, write
`wrapper._rrt.target_steps`, etc. After the guidance-source refactor, that
state lives inside `RRTGuidanceSource.state` (an `RRTRuntimeState` dataclass
instance). The wrapper exposes `_rrt` as a property returning an
`_RRTBackCompatView` so external reads/writes still hit the right place.
Today `shared_autonomy_gui.py` still reads `wrapper._rrt.mode` through the
view; `InterventionController` (`lerobot.scripts.intervention_controller`)
has migrated to `wrapper._rrt_source` + `.state` directly, and
`last_mile/helpers.py` only touches the wrapper's `auto_pause_on_rrt_finish`
property — but the view remains the supported back-compat surface.

This is a one-class file because the abstraction boundary is single-purpose.
Other future sources may grow their own views; for now only RRT needs one
because only RRT had a publicly-accessed runtime-state attribute.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lerobot.policies.guidance.rrt_source import RRTGuidanceSource


class _RRTBackCompatView:
    """Transparent proxy for `wrapper._rrt`.

    Reads and writes go to the underlying `RRTGuidanceSource.state`
    (which is an `RRTRuntimeState` instance — same dataclass that used
    to live directly on the wrapper). `.planner` and `.oracle_env_config`
    are plain fields on that dataclass, forwarded like everything else.

    Implemented via __getattr__/__setattr__ rather than @property
    forwarding so we don't have to enumerate every field on the state
    dataclass — anything the state has, the view forwards.
    """

    __slots__ = ("_source",)

    def __init__(self, source: RRTGuidanceSource) -> None:
        # Bypass __setattr__ for our own slot.
        object.__setattr__(self, "_source", source)

    def __getattr__(self, name: str):
        # Called only for attrs not in __slots__ — i.e. anything that should
        # forward to the underlying source's runtime-state dataclass.
        return getattr(object.__getattribute__(self, "_source").state, name)

    def __setattr__(self, name: str, value) -> None:
        # External code does `wrapper._rrt.target_steps = X` —
        # route the write to the underlying state dataclass.
        if name == "_source":
            object.__setattr__(self, name, value)
            return
        setattr(object.__getattribute__(self, "_source").state, name, value)

    def __repr__(self) -> str:
        return f"_RRTBackCompatView({object.__getattribute__(self, '_source').state!r})"
