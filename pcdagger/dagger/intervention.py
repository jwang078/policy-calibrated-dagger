#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Intervention-driven policy supervision state machine.

`InterventionController` watches an SA-wrapped policy's progress through a
single scenario and triggers `RRTGuidanceSource` or `OracleGoalGuidanceSource`
on stall/collision. It owns the policy/intervention alternation contract:
stall threshold + collision-detected immediate trigger, plan-failure retry +
backoff (RRT only), controller-initiated cancel after a random waypoint
budget, and per-scenario max-cycles cap.

Used by `lerobot-eval`'s intervention path. When
`EvalPipelineConfig.intervention is not None`, the rollout loop instantiates
this controller and calls `tick(success, in_collision)` after each
`policy.select_action`.

The controller is source-agnostic: at `__init__` it picks one of
`wrapper._rrt_source` / `wrapper._oracle_goal_source` based on
`cfg.method`, then reads `self._source.state.mode` and calls
`self._source.trigger()` / `self._cancel()`. Plan-failure branches are
RRT-only (oracle_goal interpolation never fails — there's no planner).

Helpers in this module:
* `_extract_success`, `_extract_in_collision` — pull bools out of gym info dicts
  that may be from either the live or final_info path.
* `ScenarioResult` — the per-scenario record written to the CSV.
* `InterventionContext` — the glue object passed into `lerobot_eval.rollout()`
  to switch it from passive eval into intervention mode.
"""

from __future__ import annotations

import csv
import logging
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from lerobot.configs.intervention import InterventionConfig
from lerobot.policies.last_mile.detectors import EEDistanceProgressTracker
from lerobot.policies.rrt_to_goal import RRTMode

if TYPE_CHECKING:
    from lerobot.policies.shared_autonomy_wrapper import SharedAutonomyPolicyWrapper
    from lerobot.policies.teleop_recording import TeleopRecordingContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Info-dict extraction helpers
# ---------------------------------------------------------------------------


def _extract_info_bool(info: dict, key: str) -> bool:
    """Pull a single boolean metric out of either the live or final info dict.

    The simulator's ``check_metrics()`` is spread into ``info`` on every step
    (both local and ZMQ paths), so per-step env signals like ``is_success`` and
    ``in_collision`` are reachable here.
    """
    val = info["final_info"].get(key, False) if "final_info" in info else info.get(key, False)
    if hasattr(val, "tolist"):
        # Numpy array per-env; we run with num_envs=1, so just take the first.
        vals = val.tolist()
        return bool(vals[0]) if vals else False
    return bool(val)


def _extract_success(info: dict) -> bool:
    return _extract_info_bool(info, "is_success")


def _extract_in_collision(info: dict) -> bool:
    return _extract_info_bool(info, "in_collision")


def _extract_float_metric(info: dict, key: str) -> float | None:
    """Pull a scalar metric out of info under either the live or final_info
    path. Returns None if absent. Same final_info dispatch as
    ``_extract_info_bool``.
    """
    val = info["final_info"].get(key) if "final_info" in info else info.get(key)
    if val is None:
        return None
    if hasattr(val, "tolist"):
        # Numpy array per-env (num_envs=1 in intervention mode).
        vals = val.tolist()
        return float(vals[0]) if vals else None
    return float(val)


def _extract_position_error_m(info: dict) -> float | None:
    """Pull the env's per-step EE-to-goal distance out of info."""
    return _extract_float_metric(info, "position_error_m")


def _extract_orientation_error_deg(info: dict) -> float | None:
    """Pull the env's per-step EE-to-goal orientation error (degrees)."""
    return _extract_float_metric(info, "orientation_error_deg")


def _extract_collision_kind(info: dict) -> str | None:
    """Pull the env's collision kind out of info: ``"self"``, ``"obstacle"``,
    or None when not currently in collision (or the env doesn't surface a kind).

    Tries two key conventions in this order:
      1. ``collision_kind`` — direct string ("self" / "obstacle" / None).
      2. ``collision_kind_code`` — integer code (0=none, 1=obstacle, 2=self)
         used by the env's check_metrics. Mapped to the string form here so
         downstream consumers don't have to know the numeric mapping.
    Returns None for any other / missing combination.
    """
    payload = info.get("final_info") if "final_info" in info else info

    def _scalar(val):
        if val is None:
            return None
        if hasattr(val, "tolist"):
            arr = val.tolist()
            return arr[0] if arr else None
        return val

    raw = _scalar(payload.get("collision_kind"))
    if raw in ("self", "obstacle"):
        return str(raw)
    code = _scalar(payload.get("collision_kind_code"))
    if code is None:
        return None
    try:
        c = int(code)
    except (TypeError, ValueError):
        return None
    if c == 1:
        return "obstacle"
    if c == 2:
        return "self"
    return None


# ---------------------------------------------------------------------------
# Per-scenario result
# ---------------------------------------------------------------------------


@dataclass
class ScenarioResult:
    """Row written to `intervention_per_scenario.csv` after each scenario."""

    scenario_idx: int
    success: bool
    cycles_used: int
    status: str
    plan_failures: int
    method: str = ""
    # Comma-separated chronological list of trigger reasons for each cycle
    # that fired this scenario. Possible values: "time stall",
    # "self_collision", "obstacle_collision", "no_progress",
    # "no_progress_ori". The legacy "in_collision" label is still emitted
    # as a fallback when the env doesn't surface a collision-kind signal
    # (e.g., older envs that only publish `in_collision: bool`), so older
    # CSVs and grep patterns expecting the bare "in_collision" remain
    # interoperable. Empty string if no cycles fired
    # (e.g. instant success).
    triggers: str = ""
    # Comma-separated chronological list of scenario-relative step indices at
    # which each trigger fired. Parallel to `triggers` (same ordering and
    # length): trigger_steps.split(",")[i] is when triggers.split(",")[i]
    # fired. Step 0 = first tick of the scenario; counts every tick (policy
    # phase + RRT phase). Empty string when no cycles fired.
    trigger_steps: str = ""
    # Comma-separated count of plan steps the i-th triggered cycle actually
    # executed. Parallel to `triggers` / `trigger_steps`. Two completion
    # modes: controller-cancel (value == target_rrt_steps, sampled in
    # [rrt_steps_min, rrt_steps_max]) OR natural finish (value < target).
    # 0 indicates the trigger fired but the plan never reached EXECUTING
    # (planning failed). Use with `trigger_steps[i]` to derive video frame
    # ranges: intervention spans [trigger_steps[i], trigger_steps[i] +
    # rrt_steps_executed[i]).
    rrt_steps_executed: str = ""


# ---------------------------------------------------------------------------
# Intervention controller
# ---------------------------------------------------------------------------


class InterventionController:
    """State machine driving policy/intervention alternation across one scenario.

    The controller never touches the env directly — it only reads the
    wrapper's guidance source state and calls source.trigger() / cancel().
    """

    def __init__(
        self,
        wrapper: SharedAutonomyPolicyWrapper,
        cfg: InterventionConfig,
    ) -> None:
        self.wrapper = wrapper
        self.cfg = cfg
        # Pick the guidance source by intervention method. The controller's
        # state machine is source-agnostic from here on: it reads `self._source.state`
        # for mode/chunk, calls `self._source.trigger()` to start a cycle, and
        # calls `self._cancel()` to abort. Plan-failure / backoff logic only
        # runs when method == "rrt" (oracle_goal interpolation can never fail).
        if cfg.method == "rrt":
            self._source = wrapper._rrt_source
            self._cancel = wrapper._cancel_rrt
        elif cfg.method == "oracle_goal":
            self._source = wrapper._oracle_goal_source
            # Use chunk_steps from config so the controller's rrt_steps_min/max
            # picks a target_steps inside [0, chunk_steps]. (We re-use the same
            # rrt_steps_min/max fields for both methods to keep configs simple.)
            wrapper._oracle_goal_source.chunk_steps = int(cfg.oracle_goal_chunk_steps)
            self._cancel = wrapper._cancel_oracle_goal
        else:
            raise ValueError(f"InterventionConfig.method must be 'rrt' or 'oracle_goal', got {cfg.method!r}")
        # SA wrapper's collision-detection mode drives which controller-side
        # triggers use no-lookback:
        #   "pre_jump_lookback" → NEVER no-lookback (all triggers rewind).
        #   "future_chunk"      → ALWAYS no-lookback (no rewind for any trigger).
        #   "hybrid"            → no-lookback ONLY for collision-related
        #                         triggers (in_collision, self_collision,
        #                         obstacle_collision); stall/no-progress
        #                         triggers still rewind. This is the
        #                         "best of both" mode: shield catches
        #                         predicted collisions and they use no-
        #                         lookback (current state is valid), while
        #                         stall/no-progress triggers rewind to
        #                         before the dead-stop so RRT plans from
        #                         a moving state.
        self._collision_detection_mode = getattr(wrapper, "_collision_detection_mode", "pre_jump_lookback")
        # Reasons that are "collision-related" (the robot is in or about to
        # be in collision). Hybrid mode dispatches these to no-lookback
        # regardless of where they originated (controller-side or wrapper-
        # side shield). Keep in sync with the trigger labels written into
        # the per-scenario CSV.
        self._collision_trigger_reasons: set[str] = {
            "in_collision",  # legacy fallback when env doesn't surface a kind
            "self_collision",
            "obstacle_collision",
            "future_chunk_coll",
        }
        # Remembers the most recently-passed trigger reason so retry
        # invocations (which don't pass a fresh reason — they're retrying
        # the same trigger) can re-use it for the no-lookback dispatch.
        self._last_trigger_reason: str | None = None
        # Optional no-progress triggers. Both share the anchor-based algorithm
        # with last_mile's NoEEProgressDetector. None when disabled
        # (window_steps=0); otherwise update() is called each tick with the
        # env's position_error_m / orientation_error_deg, and a should_fire
        # verdict behaves like the step-count stall trigger. Both can be
        # enabled together — they fire independently.
        if cfg.no_progress_window_steps > 0:
            self._progress_tracker: EEDistanceProgressTracker | None = EEDistanceProgressTracker(
                no_progress_window=cfg.no_progress_window_steps,
                min_decrease=cfg.no_progress_min_decrease_m,
                min_warmup_steps=cfg.no_progress_warmup_steps,
                reposition_grace_steps=cfg.no_progress_reposition_grace_steps,
                reposition_turnaround=cfg.no_progress_reposition_turnaround_m,
            )
        else:
            self._progress_tracker = None
        if cfg.no_progress_orientation_window_steps > 0:
            self._orientation_tracker: EEDistanceProgressTracker | None = EEDistanceProgressTracker(
                no_progress_window=cfg.no_progress_orientation_window_steps,
                min_decrease=cfg.no_progress_orientation_min_decrease_deg,
                min_warmup_steps=cfg.no_progress_orientation_warmup_steps,
                reposition_grace_steps=cfg.no_progress_orientation_reposition_grace_steps,
                reposition_turnaround=cfg.no_progress_orientation_reposition_turnaround_deg,
            )
        else:
            self._orientation_tracker = None
        # Set once per scenario (first missing-metric tick) so the warning
        # doesn't spam every tick when the env doesn't surface the metric.
        self._missing_position_error_warned: bool = False
        self._missing_orientation_error_warned: bool = False

        # per-scenario state — set in ``reset_for_new_scenario``
        self.policy_step_count: int = 0
        self.rrt_step_count: int = 0
        self.target_rrt_steps: int = 0
        # Threshold of policy steps required before the next intervention
        # trigger. Starts at ``policy_steps_before_rrt`` for the first cycle;
        # after the first executed cycle, gets resampled from
        # [policy_steps_between_rrt_min, policy_steps_between_rrt_max] each
        # time it's reset, so post-intervention we check in more often.
        self.next_policy_threshold: int = cfg.policy_steps_before_rrt
        self.cycles_used: int = 0
        self.plan_failures: int = 0
        self.controller_initiated_cancel: bool = False
        self.prev_mode: RRTMode = RRTMode.IDLE
        # True from the tick we trigger an RRT plan until the source either
        # transitions into EXECUTING (planning succeeded) or is observed back
        # in IDLE without having executed (planning failed). Robust to fast
        # PLANNING→IDLE transitions that finish entirely between two ticks
        # (e.g. when start-in-collision rejects before any actual RRT runs).
        self.pending_rrt_trigger: bool = False
        self.unexpected_natural_finish: bool = False
        # Set after a backoff fires (max_plan_failures hit). While True, the
        # collision trigger is suppressed so we don't burst-retrigger on the
        # very next tick — the policy gets the full backoff window to do
        # something. Cleared once policy_step_count crosses
        # next_policy_threshold (or on scenario reset).
        self.in_backoff_cooldown: bool = False
        # Number of completed backoff rounds in this scenario. Reset on
        # scenario reset; advance the scenario when this hits the configured
        # cap (otherwise unbounded since cycles_used only counts executed cycles).
        self.backoff_rounds: int = 0
        # Stuck-detection state — used by the planner-side in_collision
        # override at the top of tick() to gate retries on actually being
        # stuck (not just close-to-obstacle). See InterventionConfig's
        # `stuck_threshold_rad_per_tick` / `stuck_consecutive_ticks`.
        # `_prev_actual_q` is the previous tick's wrapper._latest_actual_q
        # (None on the first tick of a scenario); `_consecutive_stuck_ticks`
        # counts ticks where |q_curr - q_prev| stayed below the threshold.
        # Both reset on scenario reset.
        self._prev_actual_q: np.ndarray | None = None
        self._consecutive_stuck_ticks: int = 0
        # Mode at the previous tick's stuck-eval — used to detect RRT
        # mode transitions (IDLE→EXECUTING or EXECUTING→IDLE) so we can
        # reset the stuck counter at each boundary. Policy-mode "stuck"
        # (slow approach) shouldn't carry into RRT-mode (where the
        # signal we want is "joint physically blocked despite being
        # commanded to move fast"). Initialized to IDLE to match
        # `prev_mode` at scenario start.
        self._prev_stuck_eval_mode: RRTMode = RRTMode.IDLE
        # One-shot latch for mid-RRT collisions. While True, further
        # in_collision observations during the same EXECUTING chunk are
        # ignored (we already requested a retry; the new path needs a few
        # ticks to take effect). Cleared the moment in_collision flips
        # back to False.
        self._in_collision_during_rrt: bool = False
        # Debug telemetry for the mid-RRT collision WARNING — populated at
        # each IDLE→EXECUTING transition and read at the WARNING site to
        # report how far the robot has moved since the chunk started + the
        # recent Δq pattern. Helps distinguish "real wedge" from "ruckig
        # ramp-up false positive" by exposing whether the robot is actually
        # tracking the commanded waypoints.
        self._chunk_start_actual_q: np.ndarray | None = None
        self._recent_dq: deque[float] = deque(maxlen=10)
        self.last_status: str = "running"
        # Chronological list of trigger reasons for each intervention cycle
        # that fired this scenario. Same vocabulary as the "Triggering %s
        # (%s)..." log line: "time stall", "self_collision",
        # "obstacle_collision", "no_progress", "no_progress_ori". Legacy
        # "in_collision" is still emitted as a fallback when the env
        # doesn't surface a collision-kind signal. Reset on scenario reset;
        # appended on every fire.
        self.trigger_reasons: list[str] = []
        # Parallel list of scenario-relative step indices at which each
        # trigger in ``trigger_reasons`` fired. Together these answer "what
        # caused intervention N, and when in the scenario did it happen".
        self.trigger_steps: list[int] = []
        # Parallel list of how many RRT/oracle plan steps actually executed
        # for the i-th triggered cycle. Two completion modes feed this:
        #   * controller-initiated cancel → final value == ``target_rrt_steps``
        #     (the random cap chosen at trigger time from
        #     [rrt_steps_min, rrt_steps_max]).
        #   * natural finish (plan exhausted before the cap) → final value <
        #     ``target_rrt_steps`` — the plan was shorter than the cap.
        # Appended as 0 at trigger fire (in case the trigger never reaches
        # EXECUTING — e.g. planning fails) and overwritten by ``rrt_step_count``
        # at cycle completion. Used by downstream tooling (and humans grepping
        # the per-scenario CSV) to map video timestamps back to "policy vs
        # intervention" segments: cycle i runs the env for
        # ``rrt_steps_executed[i]`` ticks starting at ``trigger_steps[i]``.
        self.rrt_steps_executed: list[int] = []
        # Total ticks (policy + RRT phases) since the last scenario reset.
        # Incremented at the top of every ``tick()`` call so it's monotonic
        # within a scenario regardless of which phase is active.
        self.total_step_count: int = 0

    def _trigger_source(self, reason: str | None = None, no_lookback_override: bool | None = None) -> None:
        """Trigger the active guidance source, dispatching no_lookback per
        the wrapper's collision-detection mode AND (for hybrid) the per-
        trigger reason.

        Args:
            reason: the trigger reason being fired (e.g., "in_collision",
                "time stall", "no_progress", "self_collision"). Optional —
                when omitted (retry path), the last recorded reason is
                reused so the retry uses the same dispatch policy as the
                original trigger. Pass an explicit reason on the first
                fire of every cycle to keep the mapping accurate.
            no_lookback_override: when not None, force the lookback choice
                instead of deriving it from the mode/reason. Used by the
                drift-stall re-plan, whose lookback policy is set explicitly
                by the `rrt_drift_trigger` config rather than the mode.
        """
        if reason is not None:
            self._last_trigger_reason = reason
        effective_reason = reason or self._last_trigger_reason

        use_no_lookback = False
        if no_lookback_override is not None:
            use_no_lookback = bool(no_lookback_override)
        elif self._collision_detection_mode == "future_chunk":
            use_no_lookback = True
        elif self._collision_detection_mode == "hybrid":
            # Only collision-related triggers go no-lookback; stall /
            # no-progress triggers still rewind.
            use_no_lookback = effective_reason in self._collision_trigger_reasons

        # Rest-start triggers ("time stall", "no_progress", "no_progress_ori")
        # will start RRT from a stopped (or near-stopped) state: the robot
        # either timed out or stopped making progress, then lookback
        # teleports back and ruckig defaults start_vel=0. Tell the teleop
        # recorder to drop the first n_obs_steps - 1 frames of this RRT
        # segment so the recorded dataset doesn't contain velocity-from-
        # rest artifacts that mismatch the policy's observation history at
        # training time.
        if (
            effective_reason in {"time stall", "no_progress", "no_progress_ori", "drift_stall"}
            and not use_no_lookback
        ):
            from lerobot.policies.teleop_recording import TeleopRecordingContext

            n_obs_steps = int(getattr(self.wrapper.inner_policy.config, "n_obs_steps", 1))
            TeleopRecordingContext.get_instance().rrt_extra_leading_trim = max(0, n_obs_steps - 1)

        # Only the RRT source accepts no_lookback; the oracle-goal source
        # has no rewind concept, so omit the kwarg for it.
        if self.cfg.method == "rrt" and use_no_lookback:
            self._source.trigger(no_lookback=True)
        else:
            self._source.trigger()

    def reset_for_new_scenario(self) -> None:
        self.policy_step_count = 0
        self.rrt_step_count = 0
        self.target_rrt_steps = 0
        self.next_policy_threshold = self.cfg.policy_steps_before_rrt
        self.cycles_used = 0
        self.plan_failures = 0
        self.controller_initiated_cancel = False
        self.prev_mode = RRTMode.IDLE
        self.pending_rrt_trigger = False
        self.unexpected_natural_finish = False
        self.in_backoff_cooldown = False
        self.backoff_rounds = 0
        self.last_status = "running"
        self.trigger_reasons = []
        self.trigger_steps = []
        self.rrt_steps_executed = []
        self.total_step_count = 0
        if self._progress_tracker is not None:
            self._progress_tracker.reset()
        if self._orientation_tracker is not None:
            self._orientation_tracker.reset()
        self._missing_position_error_warned = False
        self._missing_orientation_error_warned = False
        # Stuck-detection state: forget the prior scenario's last actual_q
        # and zero the consecutive-stuck counter so the next scenario's
        # first tick doesn't compute Δq against stale data.
        self._prev_actual_q = None
        self._consecutive_stuck_ticks = 0
        self._prev_stuck_eval_mode = RRTMode.IDLE
        # Debug-telemetry state for mid-RRT WARNING — clear chunk-start
        # baseline and recent-Δq buffer so a new scenario doesn't blend
        # the prior scenario's motion history.
        self._chunk_start_actual_q = None
        self._recent_dq.clear()

    def _check_no_progress(
        self,
        tracker: EEDistanceProgressTracker | None,
        metric_value: float | None,
        metric_name: str,
        window_attr: str,
        missing_flag: str,
    ) -> bool:
        """Feed one metric into its tracker, return whether it fired this tick.

        ``tracker`` is None when the trigger is disabled via window=0 — fast
        return. Otherwise:
        * If the env didn't surface the metric, warn once per scenario.
        * If the controller is in backoff cooldown, skip the update so the
          tracker doesn't accumulate stalled-progress credit while the
          policy is on a forced grace window.
        """
        if tracker is None:
            return False
        if metric_value is None:
            if not getattr(self, missing_flag):
                logger.warning(
                    "InterventionConfig.%s=%d but env info has no `%s` "
                    "field; this no-progress trigger will not fire.",
                    window_attr,
                    getattr(self.cfg, window_attr),
                    metric_name,
                )
                setattr(self, missing_flag, True)
            return False
        if self.in_backoff_cooldown:
            return False
        update = tracker.update(self.policy_step_count, float(metric_value))
        return update.should_fire

    def _resample_post_intervention_threshold(self) -> None:
        """Pick the next ``policy_step_count`` threshold to use AFTER an
        intervention cycle has executed. Random uniform draw from the
        configured between range so the controller checks in on the policy
        more often (and at slightly varied cadences) once it has demonstrated
        it's intervening.
        """
        lo = max(1, self.cfg.policy_steps_between_rrt_min)
        hi = max(lo, self.cfg.policy_steps_between_rrt_max)
        self.next_policy_threshold = random.randint(lo, hi)

    def _finalize_active_rrt_steps(self) -> None:
        """If a cycle was still mid-EXECUTING when the scenario ends, overwrite
        the placeholder 0 in ``rrt_steps_executed[-1]`` with the actual
        ``rrt_step_count``.

        Background: ``tick()`` appends 0 to ``rrt_steps_executed`` at trigger
        fire and overwrites it at cycle completion (controller-cancel /
        natural-finish). But if the scenario ends via ``return "advance"``
        BEFORE either completion path runs (e.g. env reports success while
        RRT is still actively executing), the placeholder stays 0 and the
        CSV misreports "0 steps executed" for the last cycle.

        Idempotent — only updates when the placeholder is still its initial
        0 AND a cycle is currently in flight (``rrt_step_count > 0``). Safe
        to call before every ``return "advance"`` exit path.
        """
        if self.rrt_step_count > 0 and self.rrt_steps_executed and self.rrt_steps_executed[-1] == 0:
            self.rrt_steps_executed[-1] = self.rrt_step_count

    def tick(
        self,
        success: bool,
        in_collision: bool = False,
        collision_kind: str | None = None,
        position_error_m: float | None = None,
        orientation_error_deg: float | None = None,
    ) -> str:
        """Advance one step. Returns ``"continue"`` or ``"advance"``.

        ``in_collision`` is the env's current collision state (read from
        ``info["in_collision"]``). When the policy is driving (mode == IDLE)
        and the robot is in collision, we trigger an intervention immediately
        rather than waiting for ``policy_step_count`` to reach the threshold —
        collisions mean the policy is already failing, so there's no reason
        to keep accumulating bad transitions.

        ``position_error_m`` is the env's per-step EE-to-goal distance (read
        from ``info["position_error_m"]``). When the no-progress tracker is
        enabled (``cfg.no_progress_window_steps > 0``), this value is fed to
        the tracker each tick; a no-progress verdict triggers an intervention
        the same way ``policy_step_count >= threshold`` does. Pass ``None``
        to skip the position no-progress trigger for this step (also
        auto-skipped when the tracker is disabled at construction).

        ``orientation_error_deg`` is the env's per-step EE-to-goal orientation
        error in degrees (read from ``info["orientation_error_deg"]``). Same
        wiring as ``position_error_m`` but on the orientation tracker — catches
        wrist-twist failure modes that the position tracker misses.

        Pure-teleop fast path: when the SA wrapper's ``forward_flow_ratio``
        is 0.0, the user is in full manual control and the automated
        intervention has no place stepping on their input. The controller
        just watches for the success signal and otherwise stays out of the
        way. (The wrapper's existing has-guidance-cancels-active-RRT path
        still applies if ratio is dropped to 0 mid-execution.)
        """
        mode: RRTMode = self._source.state.mode
        prev_mode = self.prev_mode
        # Capture mode for next tick BEFORE branches that might mutate it via
        # _cancel — we want the mode the source had when this tick started.
        self.prev_mode = mode
        # Scenario-relative step index. Incremented before any return so the
        # counter accurately reflects "ticks observed since scenario reset"
        # whether or not this tick ends up doing real work.
        self.total_step_count += 1

        # ── Drift-stall re-plan ───────────────────────────────────────────
        # The SA wrapper's drift-abort cancelled a wedged chunk and (when
        # `rrt_drift_trigger` is "lookback"/"no_lookback") requested a re-plan.
        # The cancel completes one tick later, so by now the source is IDLE.
        # Fire a fresh trigger through the SAME machinery the other triggers
        # use — recorded as reason "drift_stall", with the lookback choice
        # forced from the config. The episode was already discarded by the
        # wrapper, so no drift frames reach the dataset either way.
        _drift_no_lookback = getattr(self.wrapper, "_rrt_drift_replan_no_lookback", None)
        if self.cfg.method == "rrt" and _drift_no_lookback is not None and mode == RRTMode.IDLE:
            self.wrapper._rrt_drift_replan_no_lookback = None
            self.target_rrt_steps = random.randint(self.cfg.rrt_steps_min, self.cfg.rrt_steps_max)
            # Set pending so the "externally-triggered RRT" detector below
            # doesn't ALSO book this cycle when the source leaves IDLE.
            self.pending_rrt_trigger = True
            self.plan_failures = 0
            self.rrt_step_count = 0
            self.policy_step_count = 0
            self.trigger_reasons.append("drift_stall")
            self.trigger_steps.append(self.total_step_count)
            self.rrt_steps_executed.append(0)
            self._source.state.target_steps = self.target_rrt_steps
            logger.info(
                "Triggering RRT (drift_stall re-plan) at scenario step %d (no_lookback=%s).",
                self.total_step_count,
                bool(_drift_no_lookback),
            )
            self._trigger_source(reason="drift_stall", no_lookback_override=bool(_drift_no_lookback))
            return "continue"

        # Override the env-provided `in_collision` with a planner-matched
        # one when running RRT method. The env's `is_robot_in_collision`
        # uses 0.0 clearance by default (only flags actual geometric
        # penetration), which MISSES the "joint physically wedged against
        # geometry but not penetrating" case. PyBullet's position
        # controller can't move a joint past a contact even though the
        # contact normal stops further motion at ~0 penetration — joint
        # sits stuck, env reports `in_collision=False`, the mid-RRT retry
        # check at the EXECUTING branch below never fires, RRT runs to
        # plan-end while the robot can't track the commanded waypoint
        # (manifests in recorded data as a state "stuck-then-snap"
        # discontinuity flagged by detect_dataset_anomalies's TELEPORT
        # class).
        #
        # The planner-side check uses the SAME clearances the RRT planner
        # plans with (--policy.shared_autonomy_config.rrt_obstacle_clearance
        # / .rrt_self_collision_clearance), so the controller's collision
        # signal matches the planner's worldview: if the planner would
        # refuse to route THROUGH this config, treat the robot being AT
        # this config as retry-worthy.
        #
        # Gates:
        #   - method == "rrt" only (oracle_goal source has no planner).
        #   - wrapper._latest_actual_q must exist (refreshed every
        #     wrapper.select_action; only None before the very first
        #     observation arrives, which is also when no RRT can be
        #     running, so the controller's in_collision logic is a no-op
        #     anyway in that window).
        # Replaces the env value entirely (not OR'd) per design — planner
        # clearance is STRICTLY more conservative (>=) than env's 0.0
        # default, so any env-positive case is also planner-positive.
        if self.cfg.method == "rrt":
            actual_q = getattr(self.wrapper, "_latest_actual_q", None)
            if actual_q is not None:
                # Reset stuck-tracking on RRT mode transitions. Policy-mode
                # "stuck" (e.g., robot slow-approaching the goal at < 0.01
                # rad/tick for many consecutive ticks) is a DIFFERENT signal
                # from RRT-mode "stuck" (joint physically blocked despite
                # being commanded to move fast). Without this reset, a long
                # slow-policy-approach accumulates the counter past the
                # threshold; the moment RRT starts a fresh cycle, the
                # FIRST RRT tick inherits stuck=True even though RRT is
                # ramping up from rest normally — every chunk-step-0
                # planner-check then false-fires `obstacle_collision` and
                # triggers an immediate retry. Reset on EITHER direction
                # of mode transition: IDLE→non-IDLE (cycle starting) AND
                # non-IDLE→IDLE (cycle ending) so neither phase contaminates
                # the other's stuck signal.
                _src_mode = self._source.state.mode
                if _src_mode != self._prev_stuck_eval_mode:
                    self._consecutive_stuck_ticks = 0
                    self._prev_actual_q = None
                    # Reset chunk-start baseline + recent-Δq buffer on EVERY
                    # mode transition. We want the WARNING-site telemetry
                    # to describe the CURRENT EXECUTING phase only — prior
                    # chunks' motion history would be misleading. Re-captured
                    # below when the new mode is EXECUTING.
                    self._recent_dq.clear()
                    if _src_mode == RRTMode.EXECUTING:
                        self._chunk_start_actual_q = actual_q.copy()
                    else:
                        self._chunk_start_actual_q = None
                self._prev_stuck_eval_mode = _src_mode
                # Stuck-detection: track the consecutive-tick streak where
                # the robot's joint-L2 |Δstate| stays below the threshold.
                # Used below to GATE the planner-side in_collision override
                # — a true wedge has both "in_collision" and "can't move",
                # whereas a false-positive approach-near-obstacle has
                # "in_collision" but the robot is tracking commanded motion
                # normally.
                if self._prev_actual_q is not None:
                    dq = float(np.linalg.norm(actual_q - self._prev_actual_q))
                    self._recent_dq.append(dq)
                    if dq < self.cfg.stuck_threshold_rad_per_tick:
                        self._consecutive_stuck_ticks += 1
                    else:
                        self._consecutive_stuck_ticks = 0
                self._prev_actual_q = actual_q.copy()

                planner_in_collision, planner_kind = self._source.is_in_collision_at(actual_q)
                # Gate the override on stuck-detection. Without the gate,
                # the controller's per-tick check fires on every config
                # within the in-progress clearance — including legitimate
                # approach configs where the goal IS within clearance of
                # the target object. The gate requires the robot to ALSO
                # be stuck (Δq < threshold for N of last N*2 ticks) before
                # treating the proximity as a real wedge. `threshold == 0`
                # disables the gate (legacy "fire on every proximity" mode).
                #
                # Uses a "K of last 2K ticks" window (not "K consecutive")
                # because the position controller's residual motion when
                # the arm is sliding against an obstacle often bounces
                # ACROSS the threshold (Δq ~ 0.005-0.015 rad while the
                # threshold is 0.01), which would reset the consecutive
                # counter every few ticks and never converge. The window
                # test tolerates a couple of above-threshold outliers per
                # cycle while still requiring the arm to be mostly-stuck.
                _need = int(self.cfg.stuck_consecutive_ticks)
                _window_len = max(_need * 2, _need + 1)
                _recent = list(self._recent_dq)[-_window_len:]
                _stuck_in_window = sum(1 for _d in _recent if _d < self.cfg.stuck_threshold_rad_per_tick)
                stuck = self.cfg.stuck_threshold_rad_per_tick > 0.0 and _stuck_in_window >= _need
                if self.cfg.stuck_threshold_rad_per_tick <= 0.0:
                    # Gate disabled — preserve legacy behavior (fire on
                    # every planner-positive tick).
                    in_collision = planner_in_collision
                    collision_kind = planner_kind
                elif planner_in_collision and stuck:
                    # WEDGE confirmed (proximity + can't move). Fire retry.
                    in_collision = True
                    collision_kind = planner_kind
                else:
                    # Either no proximity OR moving normally → not a wedge.
                    # Suppress the trigger; the env's own in_collision
                    # (penetration-only) is also gated off here since the
                    # planner check is strictly more conservative.
                    in_collision = False
                    collision_kind = None

        # Detect externally-triggered RRT (e.g., the SA wrapper's future_chunk
        # predictive shield called ``rrt_source.trigger()`` directly inside
        # select_action). Without this branch, ``target_rrt_steps`` would
        # stay at its reset_for_new_scenario default of 0, and the auto-
        # cancel below would fire after a single chunk step — making
        # shield-driven episodes look like back-to-back 1-step plans.
        # Sample target_rrt_steps now and book-keep the cycle so the rest
        # of tick() (and the per-scenario CSV) treats this like any other
        # intervention.
        if (
            prev_mode == RRTMode.IDLE
            and mode in (RRTMode.PLANNING, RRTMode.EXECUTING)
            and not self.pending_rrt_trigger
        ):
            self.target_rrt_steps = random.randint(self.cfg.rrt_steps_min, self.cfg.rrt_steps_max)
            self.pending_rrt_trigger = True
            self.plan_failures = 0
            self.rrt_step_count = 0
            self.policy_step_count = 0
            # Reset the controller-cancel flag for this fresh cycle. Without
            # this, a previous cycle's auto-cancel (which sets the flag in
            # the EXECUTING branch below) can leak into THIS shield-triggered
            # cycle: the flag normally resets at the mode-IDLE branch
            # below, but a shield trigger fires INSIDE select_action() so
            # the mode transitions IDLE → EXECUTING within a single
            # controller.tick() — the IDLE branch never runs between the
            # two cycles, the flag stays True, and the auto-cancel guard
            # at the EXECUTING branch (`if not controller_initiated_cancel
            # and rrt_step_count >= target_rrt_steps`) gates off, causing
            # the new cycle to run to FULL chunk length instead of stopping
            # at the sampled `target_rrt_steps`. Symptom: log says
            # "executing N/M waypoints" but actual recording is M frames
            # because the cap was silently disabled.
            self.controller_initiated_cancel = False
            # Use a distinct trigger label so the per-scenario CSV makes
            # it easy to count shield-driven vs controller-driven cycles.
            self.trigger_reasons.append("future_chunk_coll")
            self.trigger_steps.append(self.total_step_count)
            self.rrt_steps_executed.append(0)
            # Seed _last_trigger_reason so a same-cycle retry (e.g., plan
            # failure → re-trigger) routes through the same no-lookback
            # dispatch the shield used. Without this, the retry's
            # implicit reason would be whatever the previous cycle had,
            # potentially flipping the no-lookback decision in hybrid mode.
            self._last_trigger_reason = "future_chunk_coll"
            # Advertise the cancel point to the source so its "executing
            # X / Y waypoints" log shows partial vs total.
            self._source.state.target_steps = self.target_rrt_steps
            logger.info(
                "External RRT trigger detected (future_chunk shield) at "
                "scenario step %d — sampled target_rrt_steps=%d (cycle %d/%d).",
                self.total_step_count,
                self.target_rrt_steps,
                self.cycles_used + 1,
                self.cfg.max_cycles_per_scenario,
            )

        if success:
            self.last_status = "success"
            self._finalize_active_rrt_steps()
            return "advance"

        # Mid-RRT-execution collision: the planned path collided when
        # actually executed in sim — typically because ruckig smoothing
        # curved the RRT-raw path through an obstacle the raw path
        # avoided. Ask the source to abort the current chunk, add the
        # offending IK goal to its exclusion list, and replan to a
        # different IK branch (with fresh ruckig). The source runs the
        # replan synchronously, so by the next tick the state will be
        # EXECUTING with a new chunk (or IDLE on planner failure, which
        # then flows through the existing plan-failed branch below).
        # Use a one-shot latch so we don't spam retries while the
        # collision persists across multiple ticks (the new path needs
        # a few ticks to actually move the robot out).
        #
        # NOTE: this branch USED TO `return "continue"` early after
        # firing the retry. That short-circuited the auto-cancel
        # cap-check in the EXECUTING branch below — for a cycle that
        # keeps the planner-check seeing in_collision=True every tick
        # (e.g., final approach where the goal is intentionally within
        # the planner's clearance of the target object), the cap NEVER
        # fired and the cycle ran until env success / chunk exhaustion.
        # Now we just fire the retry and FALL THROUGH so the EXECUTING
        # branch below still increments rrt_step_count and honors the
        # cap regardless of whether the per-tick collision-check is
        # firing.
        if mode == RRTMode.EXECUTING and in_collision:  # noqa: SIM102
            if not getattr(self, "_in_collision_during_rrt", False):
                self._in_collision_during_rrt = True
                # Debug telemetry: surface WHY the per-tick collision check
                # fired so we can distinguish a real wedge from a ruckig
                # ramp-up false positive without rerunning with verbose=True.
                #   pair  — closest violating link pair + actual distance
                #   stuck — consecutive-tick stuck counter state + recent Δq
                #   move  — |actual_q - chunk_start_actual_q| (how far the
                #           robot has actually moved since EXECUTING began;
                #           tiny = still ramping up, large = real progress)
                pair_info = self._source.describe_collision_at(actual_q)
                if pair_info is not None:
                    obs_pair = pair_info.get("closest_obstacle")
                    self_pair = pair_info.get("closest_self")

                    def _fmt_pair(d: dict | None) -> str:
                        if d is None:
                            return "n/a"
                        flag = "VIOLATION" if d["in_violation"] else "ok"
                        return (
                            f"{d['link_a_name']} ⟷ {d['link_b_name']}: "
                            f"dist={d['distance_m'] * 1000:.2f}mm, threshold={d['threshold_m'] * 1000:.2f}mm [{flag}]"
                        )

                    pair_str = f"obs={_fmt_pair(obs_pair)} | self={_fmt_pair(self_pair)}"
                else:
                    pair_str = "(planner uninitialized; describe returned None)"
                recent_dq_str = (
                    "[" + ", ".join(f"{x * 1000:.2f}" for x in list(self._recent_dq)[-5:]) + "] mrad/tick"
                    if self._recent_dq
                    else "(no Δq samples yet)"
                )
                if self._chunk_start_actual_q is not None:
                    move_norm = float(np.linalg.norm(actual_q - self._chunk_start_actual_q))
                    move_str = f"{move_norm * 1000:.1f}mrad joint-L2 since chunk start"
                else:
                    move_str = "(no chunk-start baseline)"
                logger.warning(
                    "Collision detected mid-RRT (scenario step %d, chunk step %d, planner_kind=%s) — "
                    "asking source to replan to a different IK branch. "
                    "[debug] %s | stuck_counter=%d/%d (last 5 |Δq|=%s) | %s",
                    self.total_step_count,
                    self.rrt_step_count,
                    collision_kind or "unknown",
                    pair_str,
                    self._consecutive_stuck_ticks,
                    self.cfg.stuck_consecutive_ticks,
                    recent_dq_str,
                    move_str,
                )
                self._source.request_retry_after_collision()
            # Intentional fall-through to the EXECUTING branch below.
        # Collision cleared (either we're no longer EXECUTING or
        # in_collision flipped back to False) — reset the latch so a
        # future collision can trigger another retry.
        if not in_collision:
            self._in_collision_during_rrt = False

        # Pure teleop priority — no automated triggers while ratio==0. We do
        # NOT increment policy_step_count here (no "policy stall" to count
        # when the policy isn't driving), and we skip the trigger logic
        # entirely. The wrapper auto-cancels any in-flight intervention on
        # has_guidance from the keyboard agent.
        if self.wrapper.forward_flow_ratio == 0.0:
            return "continue"

        # Natural intervention finish: was EXECUTING last tick, now IDLE,
        # controller didn't cancel. Wait one more step so the env has a chance
        # to register success on the goal pose; the next tick handles the verdict.
        if prev_mode == RRTMode.EXECUTING and mode == RRTMode.IDLE and not self.controller_initiated_cancel:
            logger.warning(
                "%s chunk exhausted on its own (natural finish). Waiting one step "
                "to see if the env reports success on the planned goal pose...",
                self.cfg.method.upper(),
            )
            self.unexpected_natural_finish = True
            self.cycles_used += 1
            # Record this cycle's executed step count BEFORE the reset so the
            # CSV shows how far the plan got before exhausting itself. Mirrors
            # the controller-cancel branch above.
            if self.rrt_steps_executed:
                self.rrt_steps_executed[-1] = self.rrt_step_count
            self.policy_step_count = 0
            self.rrt_step_count = 0
            self.backoff_rounds = 0
            self.in_backoff_cooldown = False
            # An intervention cycle ran to completion — shorten cadence for next check.
            self._resample_post_intervention_threshold()
            return "continue"

        if self.unexpected_natural_finish:
            # Env did not report success this step → goal-vs-success mismatch.
            logger.warning(
                "Natural %s finish did not produce env success. Possible "
                "mismatch between intervention goal pose and env success condition; "
                "marking scenario and advancing.",
                self.cfg.method.upper(),
            )
            self.last_status = "rrt_finished_no_success"
            self._finalize_active_rrt_steps()
            return "advance"

        # Plan failure detection — RRT-only. We use a "pending trigger" flag set
        # the moment we call source.trigger(); if the next observation of IDLE
        # arrives WITHOUT the source ever entering EXECUTING, planning failed.
        # Robust to the source completing PLANNING → IDLE entirely between two
        # ticks (e.g. start-in-collision rejects). OracleGoal interpolation
        # never fails — `state.mode` transitions PLANNING → EXECUTING instantly
        # in source.trigger() — so this branch is only meaningful for "rrt".
        if self.cfg.method == "rrt" and self.pending_rrt_trigger and mode == RRTMode.IDLE:
            self.pending_rrt_trigger = False
            self.plan_failures += 1
            logger.info(
                "RRT plan failed (attempt %d/%d).",
                self.plan_failures,
                self.cfg.max_plan_failures,
            )
            if self.plan_failures < self.cfg.max_plan_failures:
                logger.info("Retrying RRT plan...")
                self._trigger_source()
                self.pending_rrt_trigger = True
                return "continue"
            self.backoff_rounds += 1
            logger.warning(
                "RRT plan failed %d times in a row (backoff round %d/%d); "
                "letting the policy run for another %d steps before the next "
                "attempt. Collision-triggered RRT is suppressed during this window.",
                self.cfg.max_plan_failures,
                self.backoff_rounds,
                self.cfg.max_backoff_rounds_per_scenario,
                self.next_policy_threshold,
            )
            if self.backoff_rounds >= self.cfg.max_backoff_rounds_per_scenario:
                logger.warning(
                    "Hit max %d backoff round(s) for this scenario; advancing.",
                    self.cfg.max_backoff_rounds_per_scenario,
                )
                self.last_status = "max_backoff_rounds"
                self._finalize_active_rrt_steps()
                return "advance"
            self.plan_failures = 0
            self.policy_step_count = 0
            self.in_backoff_cooldown = True
            return "continue"

        if mode == RRTMode.PLANNING:
            return "continue"

        if mode == RRTMode.EXECUTING:
            # Planning succeeded — clear the pending flag so a future IDLE
            # transition is correctly treated as natural-finish (or our cancel),
            # not as a plan failure.
            self.pending_rrt_trigger = False
            self.rrt_step_count += 1
            if not self.controller_initiated_cancel and self.rrt_step_count >= self.target_rrt_steps:
                logger.info(
                    "Auto-cancelling %s after %d step(s) (random target=%d).",
                    self.cfg.method.upper(),
                    self.rrt_step_count,
                    self.target_rrt_steps,
                )
                self._cancel()
                self.controller_initiated_cancel = True
                self.cycles_used += 1
                # Record this cycle's executed step count BEFORE resetting
                # rrt_step_count. Overwrites the 0 placeholder appended at
                # trigger fire. Mirrored in the natural-finish branch below.
                if self.rrt_steps_executed:
                    self.rrt_steps_executed[-1] = self.rrt_step_count
                self.rrt_step_count = 0
                self.policy_step_count = 0
                # An intervention cycle just executed successfully — the planner
                # is working again, so clear backoff state.
                self.backoff_rounds = 0
                self.in_backoff_cooldown = False
                # Shorten the cadence for the next check-in (sampled fresh
                # each time for variation).
                self._resample_post_intervention_threshold()
                if self.cycles_used >= self.cfg.max_cycles_per_scenario:
                    logger.warning(
                        "Reached max %d intervention cycle(s) without success; advancing scenario.",
                        self.cfg.max_cycles_per_scenario,
                    )
                    self.last_status = "max_cycles_reached"
                    self._finalize_active_rrt_steps()
                    return "advance"
            return "continue"

        # mode == RRTMode.IDLE
        if self.cycles_used >= self.cfg.max_cycles_per_scenario:
            self.last_status = "max_cycles_reached"
            self._finalize_active_rrt_steps()
            return "advance"

        # Reset the controller-cancel flag now that the cancel has settled.
        self.controller_initiated_cancel = False
        self.policy_step_count += 1

        # No-progress triggers: feed per-step metrics into the (optional)
        # trackers. Each maintains anchor-based progress tracking and fires
        # when its metric hasn't improved for the configured window of
        # consecutive policy steps. Position and orientation trackers are
        # independent — either firing triggers an intervention.
        should_trigger_no_progress_pos = self._check_no_progress(
            self._progress_tracker,
            position_error_m,
            metric_name="position_error_m",
            window_attr="no_progress_window_steps",
            missing_flag="_missing_position_error_warned",
        )
        should_trigger_no_progress_ori = self._check_no_progress(
            self._orientation_tracker,
            orientation_error_deg,
            metric_name="orientation_error_deg",
            window_attr="no_progress_orientation_window_steps",
            missing_flag="_missing_orientation_error_warned",
        )

        # Triggers, all gated on mode == IDLE:
        #   * stall: policy_step_count >= threshold (lifts backoff cooldown).
        #     Gated on policy_steps_before_rrt / policy_steps_between_rrt.
        #   * collision: policy hit an obstacle. NOT gated on
        #     `policy_steps_before_rrt` NOR on `in_backoff_cooldown` — an
        #     arm actively wedged against geometry needs help immediately,
        #     regardless of whether earlier plans failed OR whether we're
        #     still in the "let policy try first" warmup window. The
        #     controller's `stuck`-gate on the planner-side in_collision
        #     override (`_consecutive_stuck_ticks >= stuck_consecutive_ticks`
        #     AND `planner_in_collision`) already prevents false positives
        #     from legitimate approach-near-obstacle configs.
        #   * no-progress: EE progress trackers (position + orientation).
        #     Suppressed during backoff cooldown (drift-based signals are
        #     ambiguous mid-recovery; wait for the stall gate to lift
        #     cooldown before re-arming them).
        should_trigger_stall = self.policy_step_count >= self.next_policy_threshold
        should_trigger_collision = in_collision
        if should_trigger_stall:
            self.in_backoff_cooldown = False
        if (
            should_trigger_stall
            or should_trigger_collision
            or should_trigger_no_progress_pos
            or should_trigger_no_progress_ori
        ):
            self.target_rrt_steps = random.randint(self.cfg.rrt_steps_min, self.cfg.rrt_steps_max)
            if should_trigger_collision:
                # Differentiate self-collision from obstacle-collision in the
                # per-scenario CSV so downstream analysis can correlate trigger
                # frequency with failure mode (e.g., is wrist-pretzel driving
                # interventions vs. arm-into-wall?). When the env doesn't
                # surface a kind (e.g., a legacy env that only publishes
                # `in_collision` bool), fall back to the generic label so old
                # CSVs stay parseable.
                if collision_kind == "self":
                    reason = "self_collision"
                elif collision_kind == "obstacle":
                    reason = "obstacle_collision"
                else:
                    reason = "in_collision"
            elif should_trigger_no_progress_pos:
                reason = "no_progress"
            elif should_trigger_no_progress_ori:
                reason = "no_progress_ori"
            else:
                reason = "time stall"
            self.trigger_reasons.append(reason)
            # Step index at which this trigger fired (since scenario reset).
            # Parallels trigger_reasons by position — written together into
            # intervention_per_scenario.csv's `triggers` + `trigger_steps`
            # columns.
            self.trigger_steps.append(self.total_step_count)
            # Placeholder; overwritten at cycle completion (controller-cancel
            # or natural-finish branches below). Stays 0 iff this trigger
            # never reaches EXECUTING — i.e. planning failed outright.
            self.rrt_steps_executed.append(0)
            logger.info(
                "Triggering %s (%s) at scenario step %d, after %d policy steps (cycle %d/%d, target=%d).",
                self.cfg.method.upper(),
                reason,
                self.total_step_count,
                self.policy_step_count,
                self.cycles_used + 1,
                self.cfg.max_cycles_per_scenario,
                self.target_rrt_steps,
            )
            self.plan_failures = 0
            self.rrt_step_count = 0
            # Reset the policy counter so a fast plan-fail on the next tick
            # can't burst-retrigger here on every step (the pending_rrt_trigger
            # branch above is the single source of truth for retries / backoff).
            self.policy_step_count = 0
            # Advertise our planned cancel point so the source's "executing
            # X / Y waypoints" log shows partial vs. total.
            self._source.state.target_steps = self.target_rrt_steps
            # Pass the trigger reason so hybrid mode can dispatch
            # no-lookback only for collision-related reasons.
            self._trigger_source(reason=reason)
            self.pending_rrt_trigger = True
        return "continue"


# ---------------------------------------------------------------------------
# Glue context passed into lerobot_eval.rollout()
# ---------------------------------------------------------------------------


@dataclass
class InterventionContext:
    """Per-run state for intervention-driven rollouts.

    Passed to `lerobot_eval.rollout()` as `intervention_ctx=`; when None the
    rollout falls back to passive vectorized eval. When set, the rollout
    switches to single-env per-scenario iteration with the controller
    ticking after each policy.select_action.

    Holds the controller plus the bookkeeping needed to write
    `intervention_per_scenario.csv` alongside the standard `eval_info.json`.
    """

    controller: InterventionController
    teleop_context: TeleopRecordingContext
    csv_path: Path
    _csv_file: object | None = field(default=None, repr=False, compare=False)
    _csv_writer: object | None = field(default=None, repr=False, compare=False)
    # Index of the scenario being processed by the current rollout() call.
    # Incremented by lerobot_eval.rollout() each invocation; pushed to
    # `TeleopRecordingContext.source_scenario_idx` so the recorded dataset
    # tags each saved episode with the scenario it came from. This is a
    # RUN-LOCAL rollout counter (0, 1, 2, ...), not the underlying eval
    # benchmark index — see `benchmark_subset` for the mapping.
    scenario_idx: int = 0
    n_committed_episodes: int = 0
    # The eval-benchmark subset the rollouts are drawing from, in the order
    # scenarios are visited. Used to translate the rollout-local
    # `scenario_idx` into the underlying benchmark episode index for the
    # per-scenario CSV, so downstream tooling can correlate rows to
    # scenarios without re-deriving the subset. When set, the CSV's
    # `scenario_idx` column reports `subset[rollout_idx % len(subset)]`;
    # when None (env doesn't expose a subset), the raw rollout counter is
    # reported unchanged. Populated from `cfg.env.eval_benchmark_subset` at
    # context construction in lerobot_eval.
    benchmark_subset: list[int] | None = None

    # `method`: per-run constant ("rrt" / "oracle_goal"), recorded on every
    # row so when CSVs from different runs are concatenated (or when grepping
    # one), each scenario carries its intervention method.
    # `triggers`: chronological comma-separated list of what fired each cycle
    # in the scenario ("time stall", "in_collision", "no_progress",
    # "no_progress_ori"). Empty when no cycles fired (instant success).
    # Useful for diagnosing whether interventions are triggering at the right
    # times.
    # `trigger_steps`: parallel to `triggers`, same comma-separated layout.
    # Each integer is the scenario-relative tick index at which the
    # corresponding trigger fired (ticks counted from 0 at scenario reset).
    # So `triggers="no_progress,time stall"` + `trigger_steps="450,1120"`
    # means the first cycle fired at step 450 for "no_progress" and the second
    # at step 1120 for "time stall". Empty string when no cycles fired.
    # `rrt_steps_executed`: parallel to `triggers` / `trigger_steps`. Each
    # integer is how many plan steps the i-th cycle actually ran (== target
    # when the controller cancelled the cycle at its random cap, < target
    # when the plan exhausted itself first, 0 when planning failed outright
    # before any EXECUTING steps). Combined with `trigger_steps[i]` it gives
    # the exact [start, end) tick range of intervention i — useful for
    # mapping back to video frames.
    CSV_COLUMNS = (
        "scenario_idx",
        "success",
        "cycles_used",
        "status",
        "plan_failures",
        "method",
        "triggers",
        "trigger_steps",
        "rrt_steps_executed",
    )

    def open_csv(self) -> None:
        """Open the per-scenario CSV file for writing. Header row is emitted
        immediately; rows are appended via `record_scenario_result`.
        """
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        # File lifetime spans the whole rollout (closed in close_csv); a
        # context manager would have to wrap the entire intervention loop.
        self._csv_file = open(self.csv_path, "w", newline="")  # noqa: SIM115
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(self.CSV_COLUMNS)
        self._csv_file.flush()

    def resolve_scenario_idx(self, rollout_idx: int) -> int:
        """Translate a rollout-local counter to the underlying eval benchmark
        index via `benchmark_subset`. Returns `rollout_idx` unchanged when
        no subset is configured or when it's empty."""
        subset = self.benchmark_subset
        if not subset:
            return int(rollout_idx)
        return int(subset[int(rollout_idx) % len(subset)])

    def record_scenario_result(self, scenario_idx: int, success: bool) -> None:
        """Append a row to the CSV for the just-finished scenario.

        Reads controller state directly so callers don't have to remember
        which fields belong on the row. The `scenario_idx` argument is the
        rollout-local counter; the CSV records the resolved benchmark index
        so downstream tooling can join against `eval_info.json`.
        """
        if self._csv_writer is None:
            raise RuntimeError("InterventionContext.record_scenario_result called before open_csv()")
        ctrl = self.controller
        row = (
            self.resolve_scenario_idx(scenario_idx),
            int(bool(success)),
            ctrl.cycles_used,
            ctrl.last_status,
            ctrl.plan_failures,
            ctrl.cfg.method,
            ",".join(ctrl.trigger_reasons),
            ",".join(str(s) for s in ctrl.trigger_steps),
            ",".join(str(s) for s in ctrl.rrt_steps_executed),
        )
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def close_csv(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None
