"""Core types for the `GuidanceSource` abstraction.

The SA wrapper hosts a list of pluggable `GuidanceSource` implementations. Each
source describes WHERE its guidance comes from (RRT plan, observation key,
oracle interpolation). The wrapper decides HOW to integrate the source's chunk
with the inner policy via the source's declared `integration_mode`:

  * VERBATIM — the source's chunk is the action; inner policy ignored.
  * BLENDED  — the source's chunk is fed through the wrapper's shared blend
               math (DENOISE / LINEAR_INTERPOLATION) against the inner
               policy's predicted chunk at the wrapper's `forward_flow_ratio`.

Method-triggered sources (RRT, OracleGoal) expose `trigger()` / `cancel()`.
Observation-driven sources (ObservationTeleop) auto-activate via `update()`
and raise on `trigger()`.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    import torch
    from torch import Tensor


class GuidanceMode(Enum):
    """Method-triggered lifecycle state of a source.

    For observation-driven sources, this stays at IDLE; activation is
    decided by `is_active()` based on observation content.
    """

    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"


class IntegrationMode(Enum):
    """How the wrapper integrates a source's chunk with the inner policy."""

    VERBATIM = "verbatim"
    BLENDED = "blended"


@dataclass
class GuidanceSourceState:
    """Per-source runtime state. The RRT source's instance of this is what
    `wrapper._rrt` proxies to via the back-compat view.
    """

    mode: GuidanceMode = GuidanceMode.IDLE
    chunk: np.ndarray | None = None  # [T, num_dofs+gripper] joint waypoints
    step: int = 0
    target_steps: int | None = None  # caller hint for "executing X/Y waypoints" log
    cancel_requested: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class GuidanceCallCtx:
    """Per-step inputs handed to each source's `update()` / `next_action()`.

    The wrapper owns the read-side: it samples the obs queue, runs the inner
    policy, and decodes the actual joint state — then bundles the results here
    so sources don't need wrapper-internal knowledge.

    Several fields are optional because `update()` runs BEFORE the wrapper has
    called the inner policy and decoded actual_q — at that point only `batch`
    is reliably populated. `next_action()` always gets a fully-populated ctx.
    """

    batch: dict[str, Tensor]  # mutable; sources may pop OBS_GUIDANCE_CHUNK
    desired_q: np.ndarray | None = None  # commanded joint state (post-sync, num_dofs+gripper)
    actual_q_history: deque | None = None  # ring buffer of recent actual_q observations
    latest_actual_q: np.ndarray | None = None
    inner_action: Tensor | None = None  # raw inner policy output (used as blend anchor)
    inner_dtype: torch.dtype | None = None
    inner_device: torch.device | None = None
    oracle_env_config: dict | None = None  # already popped from batch by wrapper


@dataclass
class GuidanceStepResult:
    """A source's contribution for one `select_action` tick."""

    action: Tensor  # normalized action to return from select_action
    raw7: np.ndarray | None = None  # raw [num_dofs+gripper] for _desired_q update
    advance_step: bool = True
    finished: bool = False  # chunk exhausted; triggers wrapper's _finish_source
    flush_inner_queue_after: bool = False
    frame_source: object | None = None  # FrameSource enum value; None = wrapper picks default


class GuidanceSource(Protocol):
    """Pluggable producer of action/guidance for the SA wrapper.

    Implementations describe WHERE guidance comes from. The `integration_mode`
    attribute (VERBATIM or BLENDED) determines HOW the wrapper integrates the
    source's chunk with the inner policy each step.
    """

    name: str
    state: GuidanceSourceState
    integration_mode: IntegrationMode

    def update(self, ctx: GuidanceCallCtx) -> None:
        """Per-step state refresh. Called BEFORE `is_active()` each tick.

        Observation-driven sources read `ctx.batch[OBS_GUIDANCE_CHUNK]` here
        and update their internal `has_guidance` flag. Method-triggered
        sources typically no-op.
        """
        ...

    def is_active(self) -> bool:
        """True iff this source wants to produce the next action.

        Method-triggered: `state.mode in (PLANNING, EXECUTING)`.
        Observation-driven: `has_guidance or draining_prior_chunk`.
        """
        ...

    def trigger(self, ctx: GuidanceCallCtx | None = None) -> None:
        """Activate this source (method-triggered sources only).

        Observation-driven sources MUST raise NotImplementedError — they
        auto-activate via observation content, not external triggers.
        """
        ...

    def cancel(self) -> None:
        """Hard-reset: chunk cleared, mode → IDLE, locks released."""
        ...

    def next_action(self, ctx: GuidanceCallCtx) -> GuidanceStepResult:
        """Produce the next action. Only called when `is_active()` is True."""
        ...

    def reset(self) -> None:
        """Episode-boundary reset; called from wrapper.reset()."""
        ...

    def update_oracle_config(self, cfg: dict) -> None:
        """Notify the source that a fresh oracle env config arrived.

        RRT uses this to load/refresh obstacle geometry; other sources
        typically just cache the config for later use in `trigger()`.
        Default: cache the cfg as `self._oracle_cfg`.
        """
        ...
