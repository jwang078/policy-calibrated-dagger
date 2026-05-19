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

    def tick(self, success: bool, in_collision: bool = False) -> str:
        """Advance one step. Returns ``"continue"`` or ``"advance"``.

        ``in_collision`` is the env's current collision state (read from
        ``info["in_collision"]``). When the policy is driving (mode == IDLE)
        and the robot is in collision, we trigger an intervention immediately
        rather than waiting for ``policy_step_count`` to reach the threshold —
        collisions mean the policy is already failing, so there's no reason
        to keep accumulating bad transitions.

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

        if success:
            self.last_status = "success"
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
                    return "advance"
            return "continue"

        # mode == RRTMode.IDLE
        if self.cycles_used >= self.cfg.max_cycles_per_scenario:
            self.last_status = "max_cycles_reached"
            return "advance"

        # Reset the controller-cancel flag now that the cancel has settled.
        self.controller_initiated_cancel = False
        self.policy_step_count += 1
        # Two triggers, both gated on mode == IDLE: stall (policy_step_count
        # >= threshold) and collision (the policy has driven the robot into
        # an obstacle, no point in waiting). Trigger fires whichever first.
        # During backoff cooldown the collision trigger is suppressed so we
        # don't burst-retrigger right after a backoff (the policy gets the
        # full window to make progress on its own); the stall trigger still
        # fires on the threshold and lifts the cooldown.
        should_trigger_stall = self.policy_step_count >= self.next_policy_threshold
        should_trigger_collision = in_collision and not self.in_backoff_cooldown
        if should_trigger_stall:
            self.in_backoff_cooldown = False
        if should_trigger_stall or should_trigger_collision:
            self.target_rrt_steps = random.randint(self.cfg.rrt_steps_min, self.cfg.rrt_steps_max)
            reason = "in_collision" if should_trigger_collision else "stall"
            logger.info(
                "Triggering %s (%s) after %d policy steps (cycle %d/%d, target=%d).",
                self.cfg.method.upper(),
                reason,
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

    CSV_COLUMNS = ("scenario_idx", "success", "cycles_used", "status", "plan_failures")

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
        )
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def close_csv(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None
