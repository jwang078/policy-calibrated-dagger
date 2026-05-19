#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Last-mile help wrapper.

Detect-then-help shim around any chunk-predicting policy. The wrapper itself
is thin: it tracks raw obs.state per step, delegates detection to a
configurable ``Detector`` backend, and delegates help to a configurable
``Helper`` backend. See ``detectors.py`` and ``helpers.py``.

Lifecycle per step:

* ``select_action(batch)`` — calls ``inner.select_action`` (passthrough,
  keeps stateful policies' obs/action queues hot), then asks the detector
  whether to fire. If yes (and no multi-step helper is already active),
  ``helper.begin()`` latches and a context blob is forwarded to the helper.
* ``apply_help(action_raw, raw_obs_state)`` — called by ``lerobot_eval``
  AFTER the postprocessor has produced raw joint commands. Dispatches to
  the helper, which returns either a replacement action (``owns_action``),
  a blended action, or a no-op (true passthrough).
* ``reset()`` — end-of-episode summary from the detector, then both
  backends ``reset()``.

This file deliberately doesn't know about EE distance, q_goal_bias, RRT,
or alt policies. Those concerns live in the Detector and Helper backends.
"""

from __future__ import annotations

import logging

from torch import Tensor, nn

from lerobot.configs.last_mile import LastMileConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.last_mile.detectors import (
    Detector,
    DetectorState,
    DetectorVerdict,
    build_detector,
)
from lerobot.policies.last_mile.helpers import Helper, HelperOutput, build_helper
from lerobot.policies.pretrained import PreTrainedPolicy

logger = logging.getLogger(__name__)

# Key under which lerobot-eval stashes the raw (pre-normalization) joint state
# so this wrapper can read it without re-implementing inverse normalization.
RAW_STATE_KEY = "_raw_obs_state_for_lastmile"


class LastMileWrapper(PreTrainedPolicy):
    """Thin detect-then-help shim around the inner policy.

    See module docstring for the per-step lifecycle. Backends are built
    once at construction; their internal state is reset between episodes
    via ``reset()``.
    """

    config_class = PreTrainedConfig
    name = "last_mile_wrapper"

    def __init__(
        self,
        inner_policy: PreTrainedPolicy,
        cfg: LastMileConfig,
        detector: Detector | None = None,
        helper: Helper | None = None,
    ) -> None:
        # Bypass PreTrainedPolicy.__init__ — proxy the inner policy's config.
        # Same pattern as SharedAutonomyPolicyWrapper.__init__ (intentional).
        nn.Module.__init__(self)
        self.config: PreTrainedConfig = inner_policy.config
        self.inner_policy = inner_policy
        self.cfg = cfg
        self.detector: Detector = detector if detector is not None else build_detector(cfg)
        self.helper: Helper = helper if helper is not None else build_helper(cfg)

        # Per-step state: detector's verdict from select_action consumed by
        # apply_help. Always overwritten in select_action so a stale value
        # can't leak between steps.
        self._pending_verdict: DetectorVerdict = DetectorVerdict(should_help=False)
        # Episode-level latch: once the detector fires for the first time
        # this episode, keep dispatching to the helper every subsequent step
        # using ``_latched_context`` from the first fire. This prevents the
        # blend (or any single-step helper) from stuttering on/off when the
        # detector's trigger metric oscillates around its threshold near goal.
        # Cleared on ``reset()``.
        self._latched_active: bool = False
        self._latched_context: dict | None = None
        self._step_count: int = 0

        logger.info(
            "LastMileWrapper: enabled=%s, detect_backend=%s, help_backend=%s.",
            cfg.enabled,
            cfg.detect_backend,
            cfg.help_backend,
        )

    # ------------------------------------------------------------------
    # Helper-specific wiring
    # ------------------------------------------------------------------

    def register_sa_wrapper(self, sa_wrapper) -> None:
        """Register an outer SharedAutonomyPolicyWrapper.

        Called by the factory when ``help_backend == "rrt_to_goal"`` AND an SA
        wrapper is also in the stack. The RRT helper delegates to it via
        ``trigger_rrt_to_goal`` / ``is_rrt_active``. No-op for other helpers.
        """
        if hasattr(self.helper, "attach_sa_wrapper"):
            self.helper.attach_sa_wrapper(sa_wrapper)

    # ------------------------------------------------------------------
    # Inference path
    # ------------------------------------------------------------------

    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Passthrough that asks the detector whether help is needed.

        The detector's verdict is staged on ``self._pending_verdict`` for
        ``apply_help`` to consume.
        """
        # Capture oracle_env_config BEFORE delegating to the inner policy.
        # SharedAutonomyPolicyWrapper.select_action does ``batch.pop("oracle_env_config")``
        # (it consumes the key as it caches the obstacle config), so by the
        # time we'd try to read it post-inner-call, it'd be gone. Capture by
        # reference here — the same dict the inner sees is fine as long as
        # we grab the value before it's popped.
        oracle_env_config = batch.get("oracle_env_config")

        # If the helper has unloaded the inner policy (swap backend), it must
        # provide an alt action via ``select_action_for_swap``. The action
        # returned here is the alt's already-postprocessed RAW joint command;
        # the eval-loop's regular postprocessor will run on it (incorrectly
        # applying inverse normalization for delta policies), but apply_help
        # discards that result entirely.
        if self.helper.owns_inference():
            if not hasattr(self.helper, "select_action_for_swap"):
                raise RuntimeError(
                    "Helper claims owns_inference=True but lacks "
                    "select_action_for_swap(batch, wrapper). Implementation bug."
                )
            action_norm = self.helper.select_action_for_swap(batch, wrapper=self)
        else:
            action_norm = self.inner_policy.select_action(batch, **kwargs)

        # Reset pending verdict each step — a stale one from a previous
        # step would otherwise leak into the next apply_help call.
        self._pending_verdict = DetectorVerdict(should_help=False)

        if not self.cfg.enabled:
            return action_norm
        self._step_count += 1

        # Read the raw (pre-normalization) joint state injected by lerobot_eval.
        raw_state = batch.get(RAW_STATE_KEY)

        # Per-step success signal. Some envs report it via batch["success"];
        # currently the eval loop doesn't propagate it pre-action, so default
        # to False. Detectors that care (e.g. StallDetector) can also gate
        # against the success region via oracle distance.
        is_success = bool(batch.get("success", False))

        state = DetectorState(
            step_index=self._step_count,
            raw_obs_state=raw_state,
            oracle_env_config=oracle_env_config,
            inner_action_norm=action_norm,
            is_success=is_success,
        )
        verdict = self.detector.detect(state)
        if verdict.should_help:
            # First-fire latch (per episode). The helper.begin() inside is
            # gated separately by its own is_active() so multi-step helpers
            # (RRT, swap) don't re-trigger themselves.
            if not self._latched_active:
                self._latched_active = True
                if not self.helper.is_active():
                    self.helper.begin(verdict.context, wrapper=self)
                    # If begin() activated an owns_inference helper (e.g. swap),
                    # the inner_policy is now unloaded and the action_norm we
                    # computed above is stale. Recompute via the helper so the
                    # trigger step immediately uses the alt policy's action
                    # instead of waiting one step.
                    if self.helper.owns_inference() and hasattr(self.helper, "select_action_for_swap"):
                        action_norm = self.helper.select_action_for_swap(batch, wrapper=self)
            # Refresh latched context every step the detector fires; this
            # keeps q_goal_bias / target_ee_pos current if the oracle ever
            # changes mid-episode (it usually doesn't, but cheap insurance).
            self._latched_context = verdict.context
        self._pending_verdict = verdict
        return action_norm

    def apply_help(self, action_raw: Tensor, raw_obs_state: Tensor | None) -> Tensor:
        """Apply the staged help (if any) to a raw absolute-joint action.

        Called by lerobot_eval AFTER the postprocessor. Operates entirely in
        raw joint space, so it works identically for delta-action and
        absolute-action policies (no inverse normalization needed).

        Dispatch policy: once the detector has fired at any point in the
        episode (``_latched_active``), keep applying the help every step
        with the cached context. This eliminates "stuttering" where a
        single-step helper (e.g. blend) toggles on/off as the trigger
        metric oscillates near its threshold. Multi-step helpers (RRT,
        swap) also remain dispatched via ``helper.is_active()`` while
        they're mid-execution. Cleared on ``reset()``.
        """
        if not self._pending_verdict.should_help and not self.helper.is_active() and not self._latched_active:
            return action_raw

        # Prefer the verdict's fresh context this step; fall back to the
        # latched context from the first/most-recent fire. q_goal_bias is
        # typically a static oracle value within an episode, so this is safe.
        ctx = self._pending_verdict.context
        if ctx is None or ctx.get("q_goal_bias") is None:
            ctx = self._latched_context

        out: HelperOutput = self.helper.help(
            action_raw=action_raw,
            raw_obs_state=raw_obs_state,
            ctx=ctx,
            wrapper=self,
        )
        if out.owns_action:
            return out.action_raw if out.action_raw is not None else action_raw
        if out.action_raw is not None:
            return out.action_raw
        return action_raw

    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        # Help is applied post-postprocessor on the executed action only.
        # Chunk forecasts returned to outer consumers (e.g. SA) are untouched.
        return self.inner_policy.predict_action_chunk(batch, **kwargs)

    def reset(self) -> None:
        # Helper.reset() runs FIRST because swap-style helpers need to restore
        # the inner policy before we can call inner.reset(). For other helpers
        # this is a cheap no-op.
        if self.cfg.enabled and self._step_count > 0:
            summary = self.detector.episode_summary()
            if summary is not None:
                logger.info(
                    "LastMileWrapper episode summary (%d steps, helper=%s): %s",
                    self._step_count,
                    self.cfg.help_backend,
                    summary,
                )
        self.detector.reset()
        self.helper.reset()
        # Now inner_policy is guaranteed to be alive again (or was never
        # unloaded). Safe to forward the reset.
        if self.inner_policy is not None:
            self.inner_policy.reset()
        self._pending_verdict = DetectorVerdict(should_help=False)
        self._latched_active = False
        self._latched_context = None
        self._step_count = 0

    # ------------------------------------------------------------------
    # Delegation to inner policy
    # ------------------------------------------------------------------

    def forward(self, batch, **kwargs):
        return self.inner_policy.forward(batch, **kwargs)

    def get_optim_params(self):
        return self.inner_policy.get_optim_params()

    def train(self, mode: bool = True):
        self.inner_policy.train(mode)
        return self

    def eval(self):
        self.inner_policy.eval()
        return self

    def parameters(self, recurse: bool = True):
        return self.inner_policy.parameters(recurse)

    def to(self, *args, **kwargs):
        if self.inner_policy is not None:
            self.inner_policy.to(*args, **kwargs)
        # Drop cached detector context on device move — the context dicts
        # may hold tensors (e.g. q_goal_bias) that would be stranded on the
        # old device when the helper next consumes them.
        self._pending_verdict = DetectorVerdict(should_help=False)
        self._latched_context = None
        # Don't clear _latched_active — the latch is an episode-state flag,
        # not a device-tied resource.
        return self

    def use_original_modules(self):
        if hasattr(self.inner_policy, "use_original_modules"):
            self.inner_policy.use_original_modules()
