#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Helper backends for the last-mile help wrapper.

A Helper answers: "given the inner policy's raw joint command + the env
state, what action should actually be executed?" The wrapper calls
``help()`` for every step after detection has fired. Multi-step helpers
(RRT, swap) latch via ``begin()`` and report ``is_active()`` so the
wrapper knows whether to keep dispatching through them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from torch import Tensor

from lerobot.configs.last_mile import HelpBackend, LastMileConfig

logger = logging.getLogger(__name__)


@dataclass
class HelperOutput:
    """What ``Helper.help`` says should be executed this step.

    Three logical cases:

    * ``owns_action=True``: ignore the inner policy's action; use
      ``action_raw`` verbatim (alt-policy swap, RRT delegation).
    * ``owns_action=False`` and ``action_raw is not None``: blend/replace
      the inner action with ``action_raw`` (the blender's behavior).
    * Both fields default: true passthrough — the inner policy's action
      is returned unchanged.
    """

    action_raw: Tensor | None = None
    owns_action: bool = False
    # Currently unused — reserved for future helpers that should clear the
    # inner policy's chunk queue after running (e.g. RRT or swap).
    flush_inner_queue_after: bool = False


class Helper(Protocol):
    def help(
        self,
        action_raw: Tensor,
        raw_obs_state: Tensor | None,
        ctx: dict | None,
        wrapper,
    ) -> HelperOutput: ...
    def begin(self, ctx: dict | None, wrapper) -> None:
        """Multi-step helpers latch here. No-op for single-step helpers."""
        ...

    def reset(self) -> None: ...
    def is_active(self) -> bool:
        """True while a multi-step helper is mid-execution and should keep
        getting dispatched even when the detector no longer fires."""
        ...

    def owns_inference(self) -> bool:
        """True = skip ``inner.select_action`` entirely this step. Only
        meaningful for swap-style helpers that have unloaded the inner."""
        ...


# ---------------------------------------------------------------------------
# BlendToGoalBiasHelper
# ---------------------------------------------------------------------------


class BlendToGoalBiasHelper:
    """Blend the commanded joints toward ``q_goal_bias`` in raw joint space.

    Single-step: detection fires every step within range and we blend once.
    The pybullet safety gate (max joint jump) prevents kinematic-redundancy
    teleports that would crash the simulator with a C-level SIGABRT.
    """

    def __init__(self, blend_alpha: float, max_safe_joint_jump: float) -> None:
        self.blend_alpha = float(blend_alpha)
        self.max_safe_joint_jump = float(max_safe_joint_jump)
        # Mirrors the old `_fire_count` so logging cadence is unchanged.
        self._fire_count: int = 0

    def help(
        self,
        action_raw: Tensor,
        raw_obs_state: Tensor | None,
        ctx: dict | None,
        wrapper,
    ) -> HelperOutput:
        if raw_obs_state is None or ctx is None:
            return HelperOutput()
        q_goal_bias = ctx.get("q_goal_bias")
        if q_goal_bias is None:
            return HelperOutput()

        n_joints = int(q_goal_bias.shape[-1])
        alpha = self.blend_alpha
        q_goal_bias_t = q_goal_bias.to(action_raw.device, dtype=action_raw.dtype)

        # Flatten raw_obs_state and verify it has at least n_joints elements.
        # The previous code silently truncated unexpected shapes; now we fail
        # loudly so a wrongly-routed obs.state doesn't quietly produce garbage.
        flat = raw_obs_state.reshape(-1)
        if flat.numel() < n_joints:
            logger.warning(
                "BlendToGoalBiasHelper: raw_obs_state has %d elements but "
                "q_goal_bias needs %d joints. Passing through.",
                int(flat.numel()),
                n_joints,
            )
            return HelperOutput()
        raw_state_joints = flat[:n_joints].to(action_raw.device, dtype=action_raw.dtype)
        import torch

        # Diagnostic distance: EE distance from the oracle detector if present;
        # otherwise the stall detector's joint range L2. We log whichever is in
        # the context so the same helper plays cleanly with both detectors.
        ee_dist = ctx.get("ee_dist")
        range_l2 = ctx.get("range_l2")
        if ee_dist is not None:
            dist_label = f"ee_dist={float(ee_dist):.4f}m"
        elif range_l2 is not None:
            dist_label = f"range_l2={float(range_l2):.4f}rad"
        else:
            dist_label = "(no trigger metric in ctx)"

        # Compute the blended command in raw joint space.
        desired_joints = (1.0 - alpha) * action_raw[..., :n_joints] + alpha * q_goal_bias_t
        # Per-step joint delta from CURRENT state to the desired command.
        # When the policy and q_goal_bias agree, this is small. When they're
        # in different IK branches, individual joints can need multiple rad
        # of motion. We clamp EACH JOINT'S delta independently to
        # ±``max_safe_joint_jump`` — the override always fires, but big
        # per-joint jumps get spread across multiple steps. Per-joint clip
        # (rather than vector-L2) means a single joint never moves faster
        # than the cap, regardless of how many other joints are also moving.
        delta = desired_joints - raw_state_joints
        max_abs_before = float(delta.abs().max().item())
        delta = torch.clamp(delta, -self.max_safe_joint_jump, self.max_safe_joint_jump)
        max_abs_after = float(delta.abs().max().item())
        if max_abs_before > self.max_safe_joint_jump:
            clipped_label = f" clipped max-joint {max_abs_before:.3f}→{max_abs_after:.3f}rad"
        else:
            clipped_label = ""
        commanded_joints = raw_state_joints + delta

        if self._fire_count == 0 or self._fire_count % 25 == 0:
            step_index = getattr(wrapper, "_step_count", -1)
            print(
                f"BlendToGoalBiasHelper: override applied step={step_index} "
                f"{dist_label} alpha={alpha:.2f} max_joint_delta={max_abs_after:.3f}rad"
                f"{clipped_label}",
                flush=True,
            )

        blended = action_raw.clone()
        blended[..., :n_joints] = commanded_joints
        self._fire_count += 1
        return HelperOutput(action_raw=blended, owns_action=False)

    def begin(self, ctx: dict | None, wrapper) -> None:
        # Single-step helper — no latching.
        pass

    def reset(self) -> None:
        self._fire_count = 0

    def is_active(self) -> bool:
        return False

    def owns_inference(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# RRTToGoalHelper
# ---------------------------------------------------------------------------


class RRTToGoalHelper:
    """Delegate "help" to an outer SharedAutonomyPolicyWrapper's RRT.

    Requires the LastMileWrapper to be stacked OUTSIDE a SharedAutonomyPolicyWrapper:
        LastMile(SA(inner_policy)).
    The factory finds SA in the wrapper chain and passes the reference via
    ``wrapper.register_sa_wrapper(sa)``; the helper holds it on ``self._sa``.

    Semantics — ONE plan per episode, follow it to completion:
    * The first time the detector fires, ``begin()`` calls
      ``sa.trigger_rrt_to_goal()`` and SA plans a trajectory that reaches
      the goal. With ``rrt_blocking_plan=True`` (default), this call blocks
      until SA is in EXECUTING (or back to IDLE if planning failed).
    * Subsequent calls to ``help()`` are true passthrough — SA owns the
      action stream while EXECUTING and the inner policy drives once the
      chunk exhausts. The helper does NOT re-trigger RRT.
    * This matches the contract of the other help backends: blend keeps
      blending the same way every step, swap keeps using the same alt
      policy — and now RRT follows its one initial plan to completion
      without re-planning storms.
    * The wrapper-level latch in ``LastMileWrapper`` separately prevents
      ``begin()`` from being called more than once per episode.

    ``attach_sa_wrapper(sa)`` also disables SA's ``auto_pause_on_rrt_finish``
    so the eval loop doesn't block on ``_run_event.wait()`` after the chunk
    completes (that's only relevant in GUI sessions).
    """

    def __init__(self, disable_sa_recording: bool = True) -> None:
        self._sa = None  # set by wrapper.register_sa_wrapper()
        self._triggered_this_episode: bool = False
        self._original_auto_pause: bool | None = None
        self._disable_sa_recording = bool(disable_sa_recording)

    def attach_sa_wrapper(self, sa) -> None:
        """Called by LastMileWrapper.register_sa_wrapper().

        Configures SA for the chosen use case:
          * Disables ``auto_pause_on_rrt_finish`` so the eval loop doesn't
            hang on ``_run_event.wait()`` after the RRT chunk completes.
          * If ``disable_sa_recording`` was True at construction (the eval
            default), also disables SA's teleop-recording bookkeeping and
            the recording-cleanup teleport. When running under
            ``lerobot-eval --intervention.method=rrt`` the caller sets this
            to False so SA keeps recording the RRT-driven trajectories.
        """
        self._sa = sa
        if getattr(sa, "auto_pause_on_rrt_finish", False):
            self._original_auto_pause = sa.auto_pause_on_rrt_finish
            sa.auto_pause_on_rrt_finish = False
            logger.info(
                "RRTToGoalHelper: disabled SA.auto_pause_on_rrt_finish (was %s) "
                "so the eval loop doesn't block after RRT chunk completion.",
                self._original_auto_pause,
            )
        if self._disable_sa_recording and hasattr(sa, "disable_recording"):
            sa.disable_recording()
            logger.info(
                "RRTToGoalHelper: called SA.disable_recording() — no episodes "
                "will be recorded (set rrt_to_goal_params.disable_sa_recording=False "
                "to keep recording, e.g. for lerobot-eval --intervention.method=rrt)."
            )
        else:
            logger.info(
                "RRTToGoalHelper: leaving SA recording machinery intact "
                "(rrt_to_goal_params.disable_sa_recording=False)."
            )

    def begin(self, ctx: dict | None, wrapper) -> None:
        """Trigger RRT exactly once per episode."""
        if self._sa is None:
            logger.error(
                "RRTToGoalHelper: no SharedAutonomyPolicyWrapper registered. "
                "The factory must build SA first and pass it via "
                "wrapper.register_sa_wrapper(sa). Skipping trigger."
            )
            return
        if self._triggered_this_episode:
            # Per-helper latch (in addition to the wrapper-level latch).
            # Belt-and-suspenders against any caller invoking begin() twice.
            return
        if self._sa.is_rrt_active():
            # SA is already in PLANNING/EXECUTING (e.g. manually triggered
            # before us). Don't toggle it off.
            self._triggered_this_episode = True
            return
        logger.info("RRTToGoalHelper: triggering RRT-to-goal (one plan, this episode).")
        self._sa.trigger_rrt_to_goal()
        self._triggered_this_episode = True

    def help(
        self,
        action_raw: Tensor,
        raw_obs_state: Tensor | None,
        ctx: dict | None,
        wrapper,
    ) -> HelperOutput:
        # True passthrough. SA's select_action returns the RRT chunk action
        # while EXECUTING, and the inner policy resumes after chunk
        # exhaustion. We deliberately do NOT re-trigger — one plan per
        # episode, as configured.
        return HelperOutput()

    def reset(self) -> None:
        self._triggered_this_episode = False

    def is_active(self) -> bool:
        return self._sa is not None and self._sa.is_rrt_active()

    def owns_inference(self) -> bool:
        # SA already owns inference when RRT is active. From LastMile's
        # perspective, the inner policy call is still needed (SA's select_action
        # internally handles the RRT short-circuit).
        return False


# ---------------------------------------------------------------------------
# SwapToAltPolicyHelper
# ---------------------------------------------------------------------------


class SwapToAltPolicyHelper:
    """Swap the inner policy with an alt one for the rest of the episode.

    Designed for the case where the inner (e.g. delta-action) policy struggles
    with a sub-task (e.g. last-mile precision) that an alt policy (e.g.
    absolute-action) handles better. Single-GPU constraint: only one policy
    fits in VRAM at a time, so the swap is a hard ``del + load`` sequence.

    Lifecycle:

    1. ``begin()`` (first trigger only, guarded by ``_was_triggered_this_episode``):
       a. Cache the inner policy's state_dict to CPU (or skip if
          ``inner_unload_strategy="disk_reload"``).
       b. ``del wrapper.inner_policy``, ``gc.collect``, ``torch.cuda.empty_cache``.
       c. ``PreTrainedPolicy.from_pretrained(alt_policy_path).to(device).eval()``.
       d. Build the alt's postprocessor pipeline (raw stats from the alt
          checkpoint, so the post-policy → raw-joint conversion uses the
          right normalization).

    2. ``select_action_for_swap(batch, wrapper)`` (called from the wrapper
       when ``owns_inference`` is True):
       Runs ``alt.select_action(batch)`` + the alt postprocessor → raw
       absolute-joint command. Caches it on the helper.

    3. ``help()``: returns the cached raw action with ``owns_action=True``.
       The eval-loop's regular postprocessor still runs on whatever the
       wrapper returned from select_action, but apply_help discards that
       result entirely.

    4. ``reset()`` (episode end): **GUARDED on ``_was_triggered_this_episode``**.
       If False (helper never fired this episode), no-op — inner is still on
       GPU, alt was never loaded. If True: unload alt, reload inner from the
       cached state_dict (or re-from_pretrained for ``disk_reload`` strategy),
       reattach to ``wrapper.inner_policy``.

    Memory accounting: during the swap window (between ``del inner`` and
    ``alt.to(device)``), GPU holds zero policies. Peak host RAM during the
    swap (``cpu_cache`` strategy): inner state_dict (~6.8 GB for PI05) +
    transient buffers from from_pretrained.
    """

    def __init__(
        self,
        alt_policy_path: str,
        inner_unload_strategy: str = "cpu_cache",
    ) -> None:
        self._alt_policy_path = alt_policy_path
        self._inner_unload_strategy = inner_unload_strategy
        # Per-episode latch. Cleared in reset().
        self._was_triggered_this_episode: bool = False
        # Loaded only after begin() has fired.
        self._alt = None  # PreTrainedPolicy | None
        self._alt_postprocessor = None
        self._inner_state_dict_cpu: dict | None = None
        self._inner_config = None  # PreTrainedConfig of the inner policy
        # Cached raw action computed in select_action_for_swap, consumed by
        # help(). Overwritten every step.
        self._cached_alt_raw = None
        # Where to reattach the restored inner. Stashed during begin() so
        # reset() doesn't need wrapper passed in.
        self._wrapper_ref = None
        self._device = None  # str | torch.device

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def begin(self, ctx: dict | None, wrapper) -> None:
        if self._was_triggered_this_episode:
            return
        from lerobot.policies.last_mile.policy_swap import (
            cache_state_dict_to_cpu,
            load_policy_from_path,
            unload_policy_inplace,
        )

        self._wrapper_ref = wrapper
        inner = wrapper.inner_policy
        if inner is None:
            logger.error("SwapToAltPolicyHelper.begin: wrapper.inner_policy is None. Helper cannot proceed.")
            return

        # Resolve device from the inner policy's config (matches the device
        # the eval loop is operating on).
        device = getattr(inner.config, "device", "cuda")
        self._device = device

        # 1. Cache inner state_dict (cpu_cache strategy) so we can rebuild
        #    after the episode without going to disk.
        if self._inner_unload_strategy == "cpu_cache":
            logger.info("SwapToAltPolicyHelper: caching inner state_dict to CPU…")
            self._inner_state_dict_cpu = cache_state_dict_to_cpu(inner)
        else:
            # disk_reload: don't cache; reset() will re-from_pretrained.
            self._inner_state_dict_cpu = None
        self._inner_config = inner.config

        # 2. Drop inner from GPU.
        logger.info("SwapToAltPolicyHelper: unloading inner policy from GPU…")
        unload_policy_inplace(wrapper, "inner_policy")

        # 3. Load alt.
        self._alt = load_policy_from_path(self._alt_policy_path, device)

        # 4. Build alt's postprocessor. Uses the alt checkpoint's saved
        #    normalization stats so raw joint commands come out scaled
        #    correctly. Mirrors how _wrap_with_shared_autonomy builds its
        #    postprocessor.
        self._alt_postprocessor = self._build_alt_postprocessor(self._alt_policy_path)

        self._was_triggered_this_episode = True
        logger.info(
            "SwapToAltPolicyHelper: swap complete (alt=%s on %s).",
            self._alt_policy_path,
            device,
        )

    def _build_alt_postprocessor(self, path: str):
        from lerobot.policies.factory import (
            POLICY_POSTPROCESSOR_DEFAULT_NAME,
            policy_action_to_transition,
            transition_to_policy_action,
        )
        from lerobot.processor import PolicyProcessorPipeline

        return PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=path,
            config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )

    def select_action_for_swap(self, batch: dict, wrapper) -> Tensor:
        """Compute the alt policy's raw absolute-joint command for this step.

        Called by ``LastMileWrapper.select_action`` when ``owns_inference()`` is
        True. Caches the result so ``help()`` can return it post-postprocessor.
        Returns the SAME raw tensor (the eval loop's regular postprocessor
        will run on it, but ``apply_help`` overwrites the result).
        """
        import torch

        if self._alt is None:
            raise RuntimeError("SwapToAltPolicyHelper.select_action_for_swap called before begin().")
        with torch.inference_mode():
            alt_norm = self._alt.select_action(batch)
        alt_raw = self._alt_postprocessor(alt_norm)
        # Clone defensively — the eval-loop's postprocessor may modify in place
        # downstream, which would corrupt _cached_alt_raw before help() reads it.
        self._cached_alt_raw = alt_raw.clone() if isinstance(alt_raw, Tensor) else alt_raw
        return alt_raw

    def help(
        self,
        action_raw: Tensor,
        raw_obs_state: Tensor | None,
        ctx: dict | None,
        wrapper,
    ) -> HelperOutput:
        if self._cached_alt_raw is None:
            # Defensive — shouldn't happen if owns_inference path was followed.
            return HelperOutput()
        return HelperOutput(action_raw=self._cached_alt_raw, owns_action=True)

    def reset(self) -> None:
        """Episode end: GUARD on the trigger latch and only do work if needed."""
        if not self._was_triggered_this_episode:
            # Helper never fired — inner is still on GPU, nothing to restore.
            # Reset bookkeeping anyway.
            self._cached_alt_raw = None
            return

        from lerobot.policies.last_mile.policy_swap import (
            load_policy_from_path,
            restore_policy_from_state,
        )

        logger.info("SwapToAltPolicyHelper.reset: restoring inner policy…")
        # 1. Drop alt from GPU.
        if self._alt is not None:
            self._alt = None
            import gc

            import torch as _t

            gc.collect()
            if _t.cuda.is_available():
                _t.cuda.empty_cache()

        # 2. Restore inner.
        wrapper = self._wrapper_ref
        if wrapper is None:
            logger.error(
                "SwapToAltPolicyHelper.reset: no wrapper ref cached. "
                "Cannot restore inner — wrapper.inner_policy will stay None."
            )
        elif self._inner_unload_strategy == "cpu_cache":
            if self._inner_state_dict_cpu is None or self._inner_config is None:
                logger.error("SwapToAltPolicyHelper.reset: missing cached inner; cannot restore.")
            else:
                wrapper.inner_policy = restore_policy_from_state(
                    self._inner_config, self._inner_state_dict_cpu, self._device
                )
        else:  # disk_reload
            if self._inner_config is None or self._inner_config.pretrained_path is None:
                logger.error(
                    "SwapToAltPolicyHelper.reset: disk_reload strategy requires "
                    "inner.config.pretrained_path; got None. Cannot restore inner."
                )
            else:
                wrapper.inner_policy = load_policy_from_path(self._inner_config.pretrained_path, self._device)

        # 3. Clear all per-episode state.
        self._was_triggered_this_episode = False
        self._inner_state_dict_cpu = None
        self._inner_config = None
        self._alt_postprocessor = None
        self._cached_alt_raw = None
        self._wrapper_ref = None
        self._device = None
        unload_policy_inplace_owner = None  # noqa: F841
        logger.info("SwapToAltPolicyHelper.reset: inner restored.")

    def is_active(self) -> bool:
        return self._was_triggered_this_episode

    def owns_inference(self) -> bool:
        return self._was_triggered_this_episode


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_helper(cfg: LastMileConfig) -> Helper:
    backend: HelpBackend = cfg.help_backend
    if backend == "blend_to_goal_bias":
        params = cfg.blend_to_goal_bias_params
        return BlendToGoalBiasHelper(
            blend_alpha=params.blend_alpha,
            max_safe_joint_jump=params.max_safe_joint_jump,
        )
    if backend == "rrt_to_goal":
        params = cfg.rrt_to_goal_params
        return RRTToGoalHelper(disable_sa_recording=params.disable_sa_recording)
    if backend == "swap_to_alt_policy":
        params = cfg.swap_to_alt_policy_params
        if not params.alt_policy_path:
            raise ValueError(
                "help_backend='swap_to_alt_policy' requires "
                "swap_to_alt_policy_params.alt_policy_path to be set."
            )
        return SwapToAltPolicyHelper(
            alt_policy_path=params.alt_policy_path,
            inner_unload_strategy=params.inner_unload_strategy,
        )
    raise ValueError(f"Unknown help_backend: {backend!r}")
