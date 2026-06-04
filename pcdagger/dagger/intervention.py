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
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

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
    # that fired this scenario. Possible values: "time stall", "in_collision",
    # "no_progress", "no_progress_ori". Empty string if no cycles fired
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
        self.last_status: str = "running"
        # Chronological list of trigger reasons for each intervention cycle
        # that fired this scenario. Same vocabulary as the "Triggering %s
        # (%s)..." log line: "time stall", "in_collision", "no_progress",
        # "no_progress_ori". Reset on scenario reset; appended on every fire.
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

        if success:
            self.last_status = "success"
            self._finalize_active_rrt_steps()
            return "advance"

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
                self._source.trigger()
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
        #   * stall: policy_step_count >= threshold (lifts backoff cooldown)
        #   * collision: policy hit an obstacle (suppressed during cooldown)
        #   * no-progress (position): EE position not improving (suppressed
        #     during cooldown)
        #   * no-progress (orientation): EE orientation not improving (also
        #     suppressed during cooldown)
        # Trigger fires whichever first. Backoff cooldown gives the policy
        # the full window after a planning backoff so we don't burst-retrigger
        # the moment it's handed back control.
        should_trigger_stall = self.policy_step_count >= self.next_policy_threshold
        should_trigger_collision = in_collision and not self.in_backoff_cooldown
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
            self._source.trigger()
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
    # tags each saved episode with the scenario it came from.
    scenario_idx: int = 0
    n_committed_episodes: int = 0

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

    def record_scenario_result(self, scenario_idx: int, success: bool) -> None:
        """Append a row to the CSV for the just-finished scenario.

        Reads controller state directly so callers don't have to remember
        which fields belong on the row.
        """
        if self._csv_writer is None:
            raise RuntimeError("InterventionContext.record_scenario_result called before open_csv()")
        ctrl = self.controller
        row = (
            scenario_idx,
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
