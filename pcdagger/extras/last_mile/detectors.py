#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Detector backends for the last-mile help wrapper.

A Detector answers: "does the inner policy need help right now?" given the
current step's state. It owns per-episode counters and logging flags; the
wrapper only calls ``detect()`` / ``reset()`` / ``episode_summary()``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from lerobot.configs.last_mile import DetectBackend, LastMileConfig
from lerobot.policies.last_mile.joint_history import JointHistoryBuffer

logger = logging.getLogger(__name__)


@dataclass
class DetectorState:
    """Inputs handed to ``Detector.detect`` each step."""

    step_index: int  # 1-based, incremented before detect()
    raw_obs_state: Tensor | None  # pre-normalization joints; injected by lerobot_eval
    oracle_env_config: dict | None  # batch["oracle_env_config"]; may be None
    inner_action_norm: Tensor  # the action the inner policy just produced
    is_success: bool  # True once the env reports success this episode


@dataclass
class DetectorVerdict:
    """Per-step decision from a Detector."""

    should_help: bool
    # Forwarded verbatim to the Helper. Concrete schema is backend-specific:
    # OracleEEDistanceDetector populates {"q_goal_bias": Tensor, "ee_dist": float};
    # StallDetector currently populates {"q_goal_bias": Tensor | None} (oracle
    # may or may not be available — the helper has to be tolerant).
    context: dict | None = None


class Detector(Protocol):
    def detect(self, state: DetectorState) -> DetectorVerdict: ...
    def reset(self) -> None: ...
    def episode_summary(self) -> str | None:
        """Return a one-line summary or None to suppress. Called from
        wrapper.reset() at end-of-episode."""
        ...


# ---------------------------------------------------------------------------
# Oracle EE-distance helpers (shared with wrapper's old behavior)
# ---------------------------------------------------------------------------


def _extract_q_goal_bias(oracle_cfg) -> Tensor | None:
    if not isinstance(oracle_cfg, dict):
        return None
    task = oracle_cfg.get("task")
    if not isinstance(task, dict):
        return None
    q = task.get("q_goal_bias")
    if q is None:
        return None
    return torch.as_tensor(q, dtype=torch.float32)


def _extract_target_ee_pos(oracle_cfg) -> Tensor | None:
    if not isinstance(oracle_cfg, dict):
        return None
    task = oracle_cfg.get("task")
    if not isinstance(task, dict):
        return None
    p = task.get("target_ee_pos")
    if p is None:
        return None
    return torch.as_tensor(p, dtype=torch.float32)


def _extract_current_ee_pos(oracle_cfg) -> Tensor | None:
    if not isinstance(oracle_cfg, dict):
        return None
    p = oracle_cfg.get("current_ee_pos")
    if p is None:
        return None
    return torch.as_tensor(p, dtype=torch.float32)


# ---------------------------------------------------------------------------
# OracleEEDistanceDetector
# ---------------------------------------------------------------------------


class OracleEEDistanceDetector:
    """Fire when ``||current_ee_pos − target_ee_pos|| < threshold`` (meters).

    Requires ``oracle_env_config`` with ``current_ee_pos`` (top-level) and
    ``task.target_ee_pos`` / ``task.q_goal_bias`` populated. When any of
    those are missing, never fires and logs a one-time warning.
    """

    def __init__(self, ee_distance_threshold: float) -> None:
        self.ee_distance_threshold = float(ee_distance_threshold)
        # Per-episode counters and one-shot logging flags.
        self._fire_count: int = 0
        self._min_ee_dist: float = float("inf")
        self._min_ee_dist_step: int = -1
        self._missing_oracle_warned: bool = False
        self._saw_oracle_logged: bool = False
        # Track whether detect() ever ran this episode. Lets episode_summary
        # distinguish "never invoked" from "invoked but never fired" (fixes
        # the misleading log in the pre-refactor code).
        self._ever_invoked: bool = False

    def detect(self, state: DetectorState) -> DetectorVerdict:
        self._ever_invoked = True
        oracle_cfg = state.oracle_env_config
        q_goal_bias = _extract_q_goal_bias(oracle_cfg)
        target_ee_pos = _extract_target_ee_pos(oracle_cfg)
        current_ee_pos = _extract_current_ee_pos(oracle_cfg)
        if q_goal_bias is None or target_ee_pos is None or current_ee_pos is None:
            if not self._missing_oracle_warned:
                missing = []
                if q_goal_bias is None:
                    missing.append("task.q_goal_bias")
                if target_ee_pos is None:
                    missing.append("task.target_ee_pos")
                if current_ee_pos is None:
                    missing.append("current_ee_pos (top-level)")
                logger.warning(
                    "OracleEEDistanceDetector: oracle missing fields %s. "
                    "Detector will not fire. Check env.get_env_config().",
                    missing,
                )
                self._missing_oracle_warned = True
            return DetectorVerdict(should_help=False)
        if not self._saw_oracle_logged:
            logger.info(
                "OracleEEDistanceDetector: ✓ oracle ready. q_goal_bias=%s, "
                "target_ee_pos=%s, initial current_ee_pos=%s.",
                q_goal_bias.detach().cpu().numpy().round(3).tolist(),
                target_ee_pos.detach().cpu().numpy().round(3).tolist(),
                current_ee_pos.detach().cpu().numpy().round(3).tolist(),
            )
            self._saw_oracle_logged = True

        ee_dist = float(
            torch.linalg.vector_norm(
                current_ee_pos.to(dtype=torch.float32) - target_ee_pos.to(dtype=torch.float32)
            ).item()
        )
        if ee_dist < self._min_ee_dist:
            self._min_ee_dist = ee_dist
            self._min_ee_dist_step = state.step_index

        if ee_dist >= self.ee_distance_threshold:
            return DetectorVerdict(should_help=False, context={"ee_dist": ee_dist})

        self._fire_count += 1
        return DetectorVerdict(
            should_help=True,
            context={"q_goal_bias": q_goal_bias, "ee_dist": ee_dist},
        )

    def reset(self) -> None:
        self._fire_count = 0
        self._min_ee_dist = float("inf")
        self._min_ee_dist_step = -1
        self._saw_oracle_logged = False
        self._ever_invoked = False

    def episode_summary(self) -> str | None:
        if not self._ever_invoked:
            return None
        if self._min_ee_dist == float("inf"):
            return (
                "OracleEEDistanceDetector: EE distance never computed (oracle missing — see warnings above)."
            )
        verdict = "FIRED" if self._fire_count > 0 else "NEVER FIRED"
        tail = (
            "→ try raising ee_distance_threshold above this min to see it fire."
            if self._fire_count == 0
            else ""
        )
        return (
            f"OracleEEDistanceDetector: {verdict} ({self._fire_count} times). "
            f"Closest EE approach: {self._min_ee_dist:.4f} m at step "
            f"{self._min_ee_dist_step} (threshold={self.ee_distance_threshold:.4f} m). {tail}"
        ).rstrip()


# ---------------------------------------------------------------------------
# StallDetector
# ---------------------------------------------------------------------------


class StallDetector:
    """Fire when joint angles haven't moved meaningfully in the recent window.

    Pushes ``raw_obs_state[:n_joints]`` to a rolling buffer of length
    ``window``; once the buffer is full and we're past warmup, fires if the
    per-joint range L2 over the window is below ``joint_l2_threshold``.

    **Success-region gate**: when ``oracle_env_config`` provides EE positions,
    we suppress firing inside the goal region (default 1.5× the typical
    success tolerance). This prevents false positives during natural goal
    settling, where joint motion legitimately drops to near zero. Without
    oracle info available, stall fires purely on motion — caller's
    responsibility to disable this detector or pick a tighter threshold.

    Optionally forwards ``q_goal_bias`` to the helper if the oracle exposes
    it; the blend helper needs it. RRT / alt-policy helpers don't.
    """

    def __init__(
        self,
        n_joints: int,
        window: int,
        joint_l2_threshold: float,
        min_warmup_steps: int,
        success_region_m: float = 0.03,
    ) -> None:
        self.n_joints = int(n_joints)
        self.window = int(window)
        self.joint_l2_threshold = float(joint_l2_threshold)
        self.min_warmup_steps = int(min_warmup_steps)
        self.success_region_m = float(success_region_m)

        self._history = JointHistoryBuffer(maxlen=self.window)
        # Per-episode counters / one-shot flags.
        self._fire_count: int = 0
        self._min_range_l2: float = float("inf")
        self._min_range_l2_step: int = -1
        self._missing_raw_state_warned: bool = False
        self._saw_raw_state_logged: bool = False
        self._ever_invoked: bool = False
        # Once an episode succeeds (oracle EE close to target), we latch and
        # never fire again — avoids late-episode flapping when the robot is
        # settling at goal.
        self._success_latched: bool = False

    def detect(self, state: DetectorState) -> DetectorVerdict:
        self._ever_invoked = True
        raw_state = state.raw_obs_state
        if raw_state is None:
            if not self._missing_raw_state_warned:
                logger.warning(
                    "StallDetector: no raw obs.state in batch. lerobot-eval "
                    "must inject it under RAW_STATE_KEY before the policy "
                    "preprocessor. Detector will not fire."
                )
                self._missing_raw_state_warned = True
            return DetectorVerdict(should_help=False)
        if not self._saw_raw_state_logged:
            logger.info(
                "StallDetector: ✓ raw obs.state received (shape=%s). "
                "Window=%d, joint_l2_threshold=%.4f rad, warmup=%d.",
                tuple(raw_state.shape),
                self.window,
                self.joint_l2_threshold,
                self.min_warmup_steps,
            )
            self._saw_raw_state_logged = True

        # Push the joint slice (excluding gripper).
        self._history.push(raw_state.reshape(-1)[: self.n_joints])

        # Success-region gate via oracle EE distance. Latches once we've
        # been inside the success region — even if the robot drifts back
        # out, stall already shouldn't fire late-episode.
        oracle_cfg = state.oracle_env_config
        current_ee = _extract_current_ee_pos(oracle_cfg)
        target_ee = _extract_target_ee_pos(oracle_cfg)
        if current_ee is not None and target_ee is not None:
            ee_dist = float(
                torch.linalg.vector_norm(
                    current_ee.to(dtype=torch.float32) - target_ee.to(dtype=torch.float32)
                ).item()
            )
            if ee_dist < self.success_region_m:
                self._success_latched = True
        if self._success_latched:
            return DetectorVerdict(should_help=False)

        # is_success flag from the env (if propagated) — defensive belt &
        # braces in case the success_region gate hasn't yet latched.
        if state.is_success:
            self._success_latched = True
            return DetectorVerdict(should_help=False)

        # Warmup gate.
        if state.step_index < self.min_warmup_steps:
            return DetectorVerdict(should_help=False)
        # Window-not-full gate.
        if len(self._history) < self.window:
            return DetectorVerdict(should_help=False)

        range_l2 = self._history.range_l2()
        if range_l2 < self._min_range_l2:
            self._min_range_l2 = range_l2
            self._min_range_l2_step = state.step_index

        if range_l2 >= self.joint_l2_threshold:
            return DetectorVerdict(should_help=False, context={"range_l2": range_l2})

        # Stalled. Forward q_goal_bias if oracle provides it (helps the
        # blend helper; RRT / swap don't read it).
        q_goal_bias = _extract_q_goal_bias(oracle_cfg)
        self._fire_count += 1
        return DetectorVerdict(
            should_help=True,
            context={"q_goal_bias": q_goal_bias, "range_l2": range_l2},
        )

    def reset(self) -> None:
        self._history.clear()
        self._fire_count = 0
        self._min_range_l2 = float("inf")
        self._min_range_l2_step = -1
        self._saw_raw_state_logged = False
        self._ever_invoked = False
        self._success_latched = False

    def episode_summary(self) -> str | None:
        if not self._ever_invoked:
            return None
        if self._min_range_l2 == float("inf"):
            return (
                "StallDetector: never accumulated a full window "
                f"(window={self.window}, warmup={self.min_warmup_steps}, "
                "raw obs.state may be missing — see warnings above)."
            )
        verdict = "FIRED" if self._fire_count > 0 else "NEVER FIRED"
        tail = (
            "→ try raising joint_l2_threshold above this min to see it fire." if self._fire_count == 0 else ""
        )
        return (
            f"StallDetector: {verdict} ({self._fire_count} times). "
            f"Min joint range L2 over window: {self._min_range_l2:.4f} rad at "
            f"step {self._min_range_l2_step} (threshold={self.joint_l2_threshold:.4f} rad)."
            f" {tail}"
        ).rstrip()


# ---------------------------------------------------------------------------
# NoEEProgressDetector
# ---------------------------------------------------------------------------


class NoEEProgressDetector:
    """Fire when the EE hasn't beaten its CURRENT ANCHOR for N steps.

    Catches the "policy is drifting / wandering" failure mode that
    ``StallDetector`` misses: the joints keep moving (so joint range stays
    large) but the EE never gets closer to the goal. Requires oracle EE
    positions (same source as ``OracleEEDistanceDetector``).

    Local-anchor semantics ("progress from A, then from B, then from C"):

    The detector tracks a single ``anchor_ee`` — the EE distance at the end
    of the most recent progress phase. The no-progress counter measures
    "steps since we last beat the anchor by ``min_decrease_m``". Two events
    can reset the anchor:

      (1) **Progress.** Current EE drops below ``anchor_ee - min_decrease_m``.
          New anchor = current EE. This is the normal "still converging"
          case.

      (2) **Repositioning detected.** EE has been ABOVE the anchor for
          ``reposition_grace_steps`` consecutive steps. We declare the
          robot has entered a new epoch and reset the anchor to the local
          MIN observed during that away-phase. The new anchor may be
          WORSE than the previous one (e.g., reach A=0.30, reposition up
          to 0.50, come back down to B=0.40 → new anchor 0.40, even
          though A=0.30 was a tighter global best). This implements the
          user's requested behavior: "measure progress from the end of A,
          then from the end of B, then from the end of C".

    Per-step bookkeeping:
      * If current EE < anchor - min_decrease_m → progress, anchor moves
        down with the EE, counter resets.
      * Else if current EE > anchor → away-phase: track running min of the
        away-phase, increment ``steps_above_anchor``. After ``reposition_
        grace_steps`` consecutive above-anchor steps, anchor moves to the
        away-phase local min, counter resets.
      * Else (EE near anchor but not enough below to be progress) → still
        in current epoch, increment counter, reset ``steps_above_anchor``.
      * Fire when ``steps_without_progress >= no_progress_window``.

    Picking ``reposition_grace_steps`` < ``no_progress_window`` means
    legitimate repositioning resets the anchor BEFORE the fire window
    elapses. The wrapper-level latch keeps the helper dispatched for the
    rest of the episode once fired.
    """

    def __init__(
        self,
        no_progress_window: int,
        min_decrease_m: float,
        min_warmup_steps: int,
        reposition_grace_steps: int = 30,
        reposition_turnaround_m: float = 0.01,
    ) -> None:
        self.no_progress_window = int(no_progress_window)
        self.min_decrease_m = float(min_decrease_m)
        self.min_warmup_steps = int(min_warmup_steps)
        self.reposition_grace_steps = int(reposition_grace_steps)
        self.reposition_turnaround_m = float(reposition_turnaround_m)

        # Local anchor and per-epoch tracking.
        self._anchor_ee: float = float("inf")
        self._anchor_set_step: int = -1
        # Running min/max observed during the current away-phase. The MAX
        # is used to detect turnaround: if the current EE is meaningfully
        # below the peak, the robot has come back down (repositioning ended).
        # If it's still equal to the max, the robot is still drifting away
        # (or stuck at peak) — no turnaround, no anchor-reset.
        self._away_phase_min: float = float("inf")
        self._away_phase_max: float = -float("inf")
        self._steps_above_anchor: int = 0
        self._steps_without_progress: int = 0

        # Diagnostics.
        self._min_ee_so_far: float = float("inf")
        self._min_ee_step: int = -1
        self._anchor_resets_via_progress: int = 0
        self._anchor_resets_via_repositioning: int = 0
        self._fire_count: int = 0
        self._missing_oracle_warned: bool = False
        self._saw_oracle_logged: bool = False
        self._ever_invoked: bool = False

    def detect(self, state: DetectorState) -> DetectorVerdict:
        self._ever_invoked = True
        oracle_cfg = state.oracle_env_config
        q_goal_bias = _extract_q_goal_bias(oracle_cfg)
        target_ee_pos = _extract_target_ee_pos(oracle_cfg)
        current_ee_pos = _extract_current_ee_pos(oracle_cfg)
        if target_ee_pos is None or current_ee_pos is None:
            if not self._missing_oracle_warned:
                logger.warning(
                    "NoEEProgressDetector: oracle missing target_ee_pos or "
                    "current_ee_pos. Detector will not fire."
                )
                self._missing_oracle_warned = True
            return DetectorVerdict(should_help=False)
        if not self._saw_oracle_logged:
            logger.info(
                "NoEEProgressDetector: ✓ oracle ready. target_ee_pos=%s, "
                "no_progress_window=%d, reposition_grace=%d, min_decrease=%.4f m, warmup=%d.",
                target_ee_pos.detach().cpu().numpy().round(3).tolist(),
                self.no_progress_window,
                self.reposition_grace_steps,
                self.min_decrease_m,
                self.min_warmup_steps,
            )
            self._saw_oracle_logged = True

        ee_dist = float(
            torch.linalg.vector_norm(
                current_ee_pos.to(dtype=torch.float32) - target_ee_pos.to(dtype=torch.float32)
            ).item()
        )

        # Diagnostic: global best across episode.
        if ee_dist < self._min_ee_so_far:
            self._min_ee_so_far = ee_dist
            self._min_ee_step = state.step_index

        # ----- update anchor / counters -----
        if ee_dist < self._anchor_ee - self.min_decrease_m:
            # (1) Progress past the current anchor → new anchor.
            self._anchor_ee = ee_dist
            self._anchor_set_step = state.step_index
            self._away_phase_min = float("inf")
            self._away_phase_max = -float("inf")
            self._steps_above_anchor = 0
            self._steps_without_progress = 0
            self._anchor_resets_via_progress += 1
        else:
            self._steps_without_progress += 1
            if ee_dist > self._anchor_ee:
                # Away-phase. Track local min/max and count consecutive
                # above-steps.
                if self._steps_above_anchor == 0:
                    # Entering away-phase.
                    self._away_phase_min = ee_dist
                    self._away_phase_max = ee_dist
                else:
                    self._away_phase_min = min(self._away_phase_min, ee_dist)
                    self._away_phase_max = max(self._away_phase_max, ee_dist)
                self._steps_above_anchor += 1

                # (2) Repositioning detected → reset anchor. Two conditions
                # must BOTH hold:
                #   (a) We've been above the anchor long enough that this
                #       counts as a genuine epoch (not a one-step blip).
                #   (b) The EE has come BACK DOWN from the away-phase peak
                #       by at least ``reposition_turnaround_m``. This is the
                #       turnaround signal — without it, slow monotonic drift
                #       would just keep resetting the anchor forever.
                if (
                    self._steps_above_anchor >= self.reposition_grace_steps
                    and ee_dist < self._away_phase_max - self.reposition_turnaround_m
                ):
                    self._anchor_ee = ee_dist
                    self._anchor_set_step = state.step_index
                    self._away_phase_min = float("inf")
                    self._away_phase_max = -float("inf")
                    self._steps_above_anchor = 0
                    self._steps_without_progress = 0
                    self._anchor_resets_via_repositioning += 1
            else:
                # ee_dist is at or just below anchor but not enough below to
                # count as progress. Reset away-phase tracking.
                self._steps_above_anchor = 0
                self._away_phase_min = float("inf")
                self._away_phase_max = -float("inf")

        # ----- decide whether to fire -----
        if state.step_index < self.min_warmup_steps:
            return DetectorVerdict(
                should_help=False,
                context={
                    "ee_dist": ee_dist,
                    "anchor_ee": self._anchor_ee,
                    "min_ee_so_far": self._min_ee_so_far,
                },
            )

        if self._steps_without_progress < self.no_progress_window:
            return DetectorVerdict(
                should_help=False,
                context={
                    "ee_dist": ee_dist,
                    "anchor_ee": self._anchor_ee,
                    "min_ee_so_far": self._min_ee_so_far,
                },
            )

        # Stuck — fire.
        self._fire_count += 1
        return DetectorVerdict(
            should_help=True,
            context={
                "q_goal_bias": q_goal_bias,
                "ee_dist": ee_dist,
                "anchor_ee": self._anchor_ee,
                "min_ee_so_far": self._min_ee_so_far,
            },
        )

    def reset(self) -> None:
        self._anchor_ee = float("inf")
        self._anchor_set_step = -1
        self._away_phase_min = float("inf")
        self._away_phase_max = -float("inf")
        self._steps_above_anchor = 0
        self._steps_without_progress = 0
        self._min_ee_so_far = float("inf")
        self._min_ee_step = -1
        self._anchor_resets_via_progress = 0
        self._anchor_resets_via_repositioning = 0
        self._fire_count = 0
        self._saw_oracle_logged = False
        self._ever_invoked = False

    def episode_summary(self) -> str | None:
        if not self._ever_invoked:
            return None
        if self._min_ee_so_far == float("inf"):
            return "NoEEProgressDetector: oracle never read; detector never measured."
        verdict = "FIRED" if self._fire_count > 0 else "NEVER FIRED"
        tail = (
            f"→ try lowering no_progress_window (currently {self.no_progress_window}) to see it fire."
            if self._fire_count == 0
            else ""
        )
        return (
            f"NoEEProgressDetector: {verdict} ({self._fire_count} times). "
            f"Best EE approach (global): {self._min_ee_so_far:.4f} m at step {self._min_ee_step}. "
            f"Anchor resets: {self._anchor_resets_via_progress} via progress, "
            f"{self._anchor_resets_via_repositioning} via repositioning. "
            f"Final no-progress streak: {self._steps_without_progress} steps "
            f"(window={self.no_progress_window}, min_decrease={self.min_decrease_m:.4f} m). "
            f"{tail}"
        ).rstrip()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_detector(cfg: LastMileConfig, n_joints: int = 6) -> Detector:
    backend: DetectBackend = cfg.detect_backend
    if backend == "oracle_ee_distance":
        return OracleEEDistanceDetector(
            ee_distance_threshold=cfg.oracle_ee_distance_params.ee_distance_threshold,
        )
    if backend == "stall":
        p = cfg.stall_params
        return StallDetector(
            n_joints=n_joints,
            window=p.window,
            joint_l2_threshold=p.joint_l2_threshold,
            min_warmup_steps=p.min_warmup_steps,
        )
    if backend == "no_ee_progress":
        p = cfg.no_ee_progress_params
        return NoEEProgressDetector(
            no_progress_window=p.no_progress_window,
            min_decrease_m=p.min_decrease_m,
            min_warmup_steps=p.min_warmup_steps,
            reposition_grace_steps=p.reposition_grace_steps,
            reposition_turnaround_m=p.reposition_turnaround_m,
        )
    raise ValueError(f"Unknown detect_backend: {backend!r}")
