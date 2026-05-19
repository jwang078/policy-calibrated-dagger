"""Observation-driven teleop / blend guidance source.

Consumes the `observation.policy_guidance_chunk` key from the env's batch on
every tick. When the chunk is non-NaN this source becomes active and produces
the next action by either:

  * VERBATIM (`forward_flow_ratio == 0.0`): pure FK+IK teleop or hold action.
  * BLENDED (`0 < forward_flow_ratio < 1`): mix the guidance into the inner
    policy's predicted chunk via DENOISE or LINEAR_INTERPOLATION; subsequent
    ticks drain the cached `_guided_chunk` until exhausted.

This is fundamentally different in lifecycle from method-triggered sources
(RRT, OracleGoal): it auto-activates from observation content rather than
external `trigger()` calls. `trigger()` raises NotImplementedError.

Source-owned state migrated out of the wrapper:
  * `_guided_chunk` — blended action chunk buffer
  * `_chunk_step` — cursor into `_guided_chunk`
  * `_had_guidance_last_step` — used to detect just-released guidance
  * `_last_decoded_guidance_chunk` — diagnostic decode of the most recent
    guidance chunk back to raw joints
  * `has_guidance` — set by `update()` per tick

Stays on the wrapper (used by multiple sources): `_normalize_policy_guidance_action`,
`get_hold_action`, `get_full_teleop_action`, `_project_delta_for_collision`,
`_sync_joints`, `_get_ee_pose`, `postprocessor`, `inner_policy`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch import Tensor

from lerobot.policies.guidance.base import (
    GuidanceCallCtx,
    GuidanceMode,
    GuidanceSourceState,
    GuidanceStepResult,
    IntegrationMode,
)

if TYPE_CHECKING:
    from lerobot.policies.shared_autonomy_wrapper import SharedAutonomyPolicyWrapper

logger = logging.getLogger(__name__)


# Module-level key (also re-exported from shared_autonomy_wrapper for back-compat).
OBS_GUIDANCE_CHUNK = "observation.policy_guidance_chunk"


class ObservationTeleopGuidanceSource:
    """Observation-driven guidance source.

    Lifecycle:
      * `update(ctx)` is called every tick. It pops `OBS_GUIDANCE_CHUNK` from
        the batch, sets `has_guidance` for the tick, and caches the raw chunk.
      * `is_active()` returns True iff there's active guidance OR the source
        is still draining a previously-built blended chunk.
      * `next_action(ctx)` produces the action via either pure-teleop or
        blend-and-drain logic.
      * `cancel()` / `reset()` clear the cached blended chunk.
      * `trigger()` raises — this source is observation-driven.
    """

    name = "obs_teleop"

    def __init__(self, wrapper: SharedAutonomyPolicyWrapper) -> None:
        self._wrapper = wrapper
        # The base Protocol expects a `state` attribute; obs-driven sources
        # don't use mode/chunk/target_steps but we keep an empty dataclass
        # so the dispatcher loop can iterate sources uniformly.
        self.state: GuidanceSourceState = GuidanceSourceState()
        # Source-owned state migrated from the wrapper.
        self._guided_chunk: Tensor | None = None
        self._chunk_step: int = 99_999_999_999
        self._had_guidance_last_step: bool = False
        self._last_decoded_guidance_chunk: np.ndarray | None = None
        # Per-tick state set by update().
        self._guidance_chunk_raw: Tensor | None = None
        self.has_guidance: bool = False

    # ── Protocol API ───────────────────────────────────────────────────── #

    @property
    def integration_mode(self) -> IntegrationMode:
        """Reports VERBATIM when ratio==0 (pure teleop), BLENDED otherwise.

        Computed off the wrapper's current `forward_flow_ratio` so external
        edits via the slider GUI are reflected without explicit notification.
        """
        return (
            IntegrationMode.VERBATIM if self._wrapper.forward_flow_ratio == 0.0 else IntegrationMode.BLENDED
        )

    def update(self, ctx: GuidanceCallCtx) -> None:
        """Pop `OBS_GUIDANCE_CHUNK` from the batch and update has_guidance.

        Must be called BEFORE `is_active()` each tick. Caches the raw chunk
        on `self._guidance_chunk_raw` for use in `next_action()`.
        """
        self._guidance_chunk_raw = ctx.batch.pop(OBS_GUIDANCE_CHUNK, None)
        self.has_guidance = (
            self._guidance_chunk_raw is not None and not torch.isnan(self._guidance_chunk_raw).all()
        )

    def is_active(self) -> bool:
        """True iff this source wants to handle the next action.

        At ratio==0 (pure teleop mode), this source is always active —
        either it emits a teleop action or it emits a hold action.
        At ratio>0, it's active only when guidance is present OR a prior
        blended chunk is still being drained.
        """
        ratio = self._wrapper.forward_flow_ratio
        if ratio == 0.0:
            return True
        draining = self._guided_chunk is not None and not self._chunk_exhausted()
        return self.has_guidance or draining

    def trigger(self, ctx: GuidanceCallCtx | None = None) -> None:
        raise NotImplementedError(
            "ObservationTeleopGuidanceSource is observation-driven, not method-triggered. "
            "Inject guidance into observation.policy_guidance_chunk to activate it."
        )

    def cancel(self) -> None:
        """Clear the cached blended chunk and reset cursors."""
        self._guided_chunk = None
        self._chunk_step = 99_999_999_999  # forces chunk_exhausted on next call
        self._had_guidance_last_step = False

    def reset(self) -> None:
        """Episode-boundary reset."""
        self._guided_chunk = None
        self._chunk_step = 0  # fresh-episode value (different from cancel's "force-exhausted")
        self._had_guidance_last_step = False
        self._last_decoded_guidance_chunk = None
        self._guidance_chunk_raw = None
        self.has_guidance = False

    def update_oracle_config(self, cfg: dict) -> None:
        del cfg  # Obs-driven source doesn't use oracle config.

    # ── Main dispatch ──────────────────────────────────────────────────── #

    def next_action(self, ctx: GuidanceCallCtx, base_noise: Tensor | None = None) -> GuidanceStepResult:
        """Produce the next action.

        Only called when `is_active()` is True. Dispatches across:
          (a) ratio == 0 + has_guidance → pure teleop FK+IK
          (b) ratio == 0 + no guidance → hold action
          (c) drain path: existing blended chunk + (no guidance OR ONCE_PER_CHUNK)
          (d) (re)build the blended chunk and emit the first step
        """
        from lerobot.policies.shared_autonomy_wrapper import (
            BlendMode,
            FrameSource,
            PolicyGuidanceRepresentation,
        )

        wrapper = self._wrapper
        guidance_chunk_raw = self._guidance_chunk_raw
        ratio = wrapper.forward_flow_ratio

        # Track for next tick — debug / future-use hook.
        self._had_guidance_last_step = self.has_guidance

        # --- (a) and (b): pure teleop / hold (ratio == 0) ---
        if ratio == 0.0:
            if self.has_guidance:
                if wrapper.policy_guidance_representation == PolicyGuidanceRepresentation.ABSOLUTE_POS:
                    action = wrapper._normalize_policy_guidance_action(guidance_chunk_raw[:, 0, :])
                else:
                    action = wrapper.get_full_teleop_action(guidance_chunk_raw[:, 0, :])
            else:
                action = wrapper.get_hold_action(ctx.inner_action)
            return GuidanceStepResult(
                action=action,
                frame_source=FrameSource.TELEOP if self.has_guidance else FrameSource.POLICY,
            )

        # --- 0 < ratio < 1 path ---
        chunk_exhausted = self._chunk_exhausted()

        # (c) Drain path: return next action from existing blended chunk without re-blending.
        if (
            self._guided_chunk is not None
            and not chunk_exhausted
            and (not self.has_guidance or wrapper.blend_mode == BlendMode.ONCE_PER_CHUNK)
        ):
            action = self._guided_chunk[:, self._chunk_step, :]
            self._chunk_step += 1
            # Tag as POLICY — the blend path's frames are not committed by the recorder
            # under the current FrameSource semantics. Step 4 may opt some of these
            # into BLEND_INTERVENTION_<XXX> tagging for the goal-bias-blend use case;
            # the existing obs-teleop blend keeps POLICY tagging for back-compat.
            return GuidanceStepResult(action=action, frame_source=FrameSource.POLICY)

        # (d) Rebuild blended chunk.
        return self._build_and_emit_blended(ctx, base_noise=base_noise)

    # ── Helpers ────────────────────────────────────────────────────────── #

    def _chunk_exhausted(self) -> bool:
        return self._guided_chunk is None or self._chunk_step >= self._wrapper.config.n_action_steps

    def _build_and_emit_blended(self, ctx: GuidanceCallCtx, base_noise: Tensor | None) -> GuidanceStepResult:
        """The blend-construct path: build / refresh `_guided_chunk`, emit step 0."""
        from lerobot.policies.shared_autonomy_wrapper import (
            FrameSource,
            GuidanceBlendStrategy,
            PolicyGuidanceRepresentation,
        )

        wrapper = self._wrapper
        guidance_chunk_raw = self._guidance_chunk_raw
        ratio = wrapper.forward_flow_ratio
        chunk_exhausted = self._chunk_exhausted()

        # max_action_dim: PI0.5 pads actions to this size; diffusion uses raw action_dim.
        max_action_dim = getattr(wrapper.config, "max_action_dim", None)
        batch_size = (
            guidance_chunk_raw.shape[0] if guidance_chunk_raw is not None else ctx.inner_action.shape[0]
        )

        # Determine anchor chunk for IK:
        # - If chunk exhausted or no buffer yet: get a fresh policy chunk via predict_action_chunk.
        #   inner_policy.select_action was already called above (obs queues updated for diffusion),
        #   so predict_action_chunk reads from up-to-date obs queues.
        # - Otherwise: reuse _guided_chunk so guidance accumulates on top of itself.
        if chunk_exhausted or self._guided_chunk is None:
            noise_kwargs = {"noise": base_noise} if base_noise is not None else {}
            anchor_chunk = wrapper.inner_policy.predict_action_chunk(ctx.batch, **noise_kwargs)
            self._chunk_step = 0
        else:
            anchor_chunk = self._guided_chunk

        device = anchor_chunk.device
        anchor_len = anchor_chunk.shape[1]
        action_dim = anchor_chunk.shape[2]

        # Build the normalized guidance chunk to use as noise anchor.
        # Clone anchor and zero-pad to max_action_dim if needed (required by PI0.5).
        guidance_chunk = anchor_chunk.clone()
        if max_action_dim is not None and action_dim < max_action_dim:
            pad = torch.zeros(
                batch_size,
                anchor_len,
                max_action_dim - action_dim,
                dtype=guidance_chunk.dtype,
                device=device,
            )
            guidance_chunk = torch.cat([guidance_chunk, pad], dim=2)
        else:
            max_action_dim = action_dim

        if (
            wrapper.policy_guidance_representation == PolicyGuidanceRepresentation.ABSOLUTE_POS
            and guidance_chunk_raw is not None
        ):
            self._fill_chunk_absolute(guidance_chunk, guidance_chunk_raw, anchor_len, action_dim)
        elif (
            wrapper.policy_guidance_representation == PolicyGuidanceRepresentation.DELTA
            and guidance_chunk_raw is not None
        ):
            self._fill_chunk_delta(
                guidance_chunk, guidance_chunk_raw, anchor_chunk, anchor_len, action_dim, batch_size, device
            )
        else:
            raise NotImplementedError(
                f"Unsupported policy_guidance_representation: {wrapper.policy_guidance_representation}"
            )

        # Diagnostic: decode the constructed guidance_chunk back to raw joints
        # so callers can compare what is being fed into the blend against the
        # demo trajectory.
        decoded_steps = [
            wrapper.postprocessor(guidance_chunk[:, t_abs, :action_dim]) for t_abs in range(anchor_len)
        ]
        self._last_decoded_guidance_chunk = torch.stack(decoded_steps, dim=1).detach().cpu().numpy()

        # Build the guidance noise regardless of blend strategy to consume the rng seed.
        # It's ok to do this duplicate work for INTERPOLATE because that's a debug mode.
        x_tsw = self._build_guidance_noise_from_chunk(guidance_chunk, ratio, base_noise=base_noise)

        if wrapper.guidance_blend_strategy == GuidanceBlendStrategy.INTERPOLATE:
            # Simple linear interpolation in clean action space — no denoising.
            blended = anchor_chunk.clone()
            g = guidance_chunk[:, :, :action_dim]
            blended[:, :, :action_dim] = ratio * anchor_chunk + (1.0 - ratio) * g
            # Snap first n_anchor_steps to guidance exactly (mirrors DENOISE inpainting).
            if wrapper.n_anchor_steps > 0:
                n_a = min(wrapper.n_anchor_steps, anchor_len - self._chunk_step)
                blended[:, self._chunk_step : self._chunk_step + n_a, :action_dim] = guidance_chunk[
                    :, self._chunk_step : self._chunk_step + n_a, :action_dim
                ]
            self._guided_chunk = blended
        elif wrapper.guidance_blend_strategy == GuidanceBlendStrategy.DENOISE:
            denoise_kwargs: dict = {"noise": x_tsw, "sa_noise_ratio": ratio}
            if wrapper.n_anchor_steps > 0:
                # Pass first n_anchor_steps of normalized guidance as anchor_action.
                # The denoising loop will re-anchor these positions at every step,
                # so the final chunk exactly matches guidance at steps 0..n_a-1
                # while letting the model generate a coherent continuation.
                n_a = min(wrapper.n_anchor_steps, anchor_len - self._chunk_step)
                denoise_kwargs["anchor_action"] = guidance_chunk[
                    :, self._chunk_step : self._chunk_step + n_a, :action_dim
                ]
            blended = wrapper.inner_policy.predict_action_chunk(ctx.batch, **denoise_kwargs)
            self._guided_chunk = blended
        else:
            raise NotImplementedError(
                f"Unsupported guidance_blend_strategy: {wrapper.guidance_blend_strategy}"
            )

        action = self._guided_chunk[:, self._chunk_step, :]
        self._chunk_step += 1
        # POLICY tag: the obs-driven blend frames are not committed by the
        # recorder under current FrameSource semantics.
        return GuidanceStepResult(action=action, frame_source=FrameSource.POLICY)

    def _fill_chunk_absolute(
        self,
        guidance_chunk: Tensor,
        guidance_chunk_raw: Tensor,
        anchor_len: int,
        action_dim: int,
    ) -> None:
        """ABSOLUTE_POS: full per-step guidance chunk; normalize each step and overwrite anchor."""
        wrapper = self._wrapper
        n_provided = guidance_chunk_raw.shape[1]
        n_remaining = anchor_len - self._chunk_step
        n_fill = min(n_provided, n_remaining)
        for t_rel in range(n_fill):
            step_raw = guidance_chunk_raw[:, t_rel, :]  # [B, action_dim]
            step_norm = wrapper._normalize_policy_guidance_action(step_raw)
            t_abs = self._chunk_step + t_rel
            guidance_chunk[:, t_abs, :action_dim] = step_norm
        # If guidance is shorter than remaining chunk, repeat last step.
        if n_fill < n_remaining:
            last_norm = guidance_chunk[:, self._chunk_step + n_fill - 1, :action_dim]
            for t_abs in range(self._chunk_step + n_fill, anchor_len):
                guidance_chunk[:, t_abs, :action_dim] = last_norm

    def _fill_chunk_delta(
        self,
        guidance_chunk: Tensor,
        guidance_chunk_raw: Tensor,
        anchor_chunk: Tensor,
        anchor_len: int,
        action_dim: int,
        batch_size: int,
        device: torch.device,
    ) -> None:
        """DELTA: integrate EE deltas step-by-step from `_desired_q` (default) or anchor-seeded (legacy)."""
        wrapper = self._wrapper
        # Hardcoded toggle. Default (False) = step-by-step integration seeded from
        # _desired_q, matching what ratio=0.0 / get_full_teleop_action does. The
        # legacy path (True) is anchor-seeded with an R_0-frame cumulative offset;
        # easy to flip back if the new path turns out to be wrong for some case.
        use_legacy_anchor_seeded_delta = False

        guidance_chunk_np = guidance_chunk_raw.cpu().numpy()
        n_provided = guidance_chunk_np.shape[1]
        n_remaining = anchor_len - self._chunk_step

        if not use_legacy_anchor_seeded_delta:
            # Step-by-step DELTA integration. Seed from _desired_q and apply each
            # delta in its own native local EE frame. Produces an absolute-joint
            # trace of the user's intended trajectory, so the blend pulls the
            # policy toward the demo rather than toward an anchor-offset that
            # drifts whenever the policy disagrees with the demo.
            assert wrapper._desired_q is not None, "_desired_q must be seeded before DELTA blend"
            for b in range(batch_size):
                q_seed = wrapper._desired_q.reshape(-1).copy()[: wrapper.num_dofs]
                last_delta = guidance_chunk_np[b, 0]
                for t_rel in range(n_remaining):
                    d = guidance_chunk_np[b, t_rel] if t_rel < n_provided else last_delta
                    if t_rel < n_provided:
                        last_delta = d
                    d_pos, d_rot, d_gripper = d[:3], d[3:6], d[6]
                    q_new = wrapper._project_delta_for_collision(
                        q_seed, d_pos, d_rot, skip_collision=wrapper.skip_collision
                    )
                    q_seed = q_new[: wrapper.num_dofs].copy()
                    t_abs = self._chunk_step + t_rel
                    raw_step = np.concatenate([q_new, [float(d_gripper)]])
                    step_t = torch.tensor(raw_step, dtype=anchor_chunk.dtype, device=device).unsqueeze(0)
                    step_norm = wrapper._normalize_policy_guidance_action(step_t)
                    guidance_chunk[:, t_abs, :action_dim] = step_norm
        else:
            # Legacy anchor-seeded DELTA: for each step t, take the anchor joint
            # position at t and shift it by the EE delta accumulated from
            # chunk_step to t, expressed in anchor[chunk_step]'s EE frame (R_0).
            # Designed to keep guidance[t] close to anchor[t] for live keyboard
            # teleop; produces wrong absolute trajectories when fed a full
            # pre-recorded chunk of demo deltas.
            assert wrapper._desired_q is not None, "_desired_q must be seeded before DELTA blend"
            for b in range(batch_size):
                q_seed = wrapper._desired_q.reshape(-1).copy()[: wrapper.num_dofs]
                # Compute R_0: EE orientation at anchor[chunk_step]; all subsequent
                # accumulated deltas are expressed in this frame.
                wrapper._sync_joints(q_seed)
                _, quat_0 = wrapper._get_ee_pose()
                rot_0 = Rotation.from_quat(quat_0)

                accumulated_pos = np.zeros(3)
                accumulated_rot = Rotation.identity()
                last_delta = guidance_chunk_np[b, 0]
                for t_rel in range(n_remaining):
                    d = guidance_chunk_np[b, t_rel] if t_rel < n_provided else last_delta
                    if t_rel < n_provided:
                        last_delta = d
                    accumulated_pos = accumulated_pos + d[:3]
                    accumulated_rot = accumulated_rot * Rotation.from_euler("XYZ", d[3:6])
                    d_gripper = d[6]

                    t_abs = self._chunk_step + t_rel
                    anchor_qt_raw = wrapper.postprocessor(anchor_chunk[[b], t_abs, :])
                    anchor_qt = anchor_qt_raw[0, : wrapper.num_dofs].cpu().numpy()

                    # Transform accumulated_pos from rot_0's frame to anchor[t]'s EE
                    # frame so that _compute_next_joints (which applies delta in the
                    # current EE frame) produces the intended world-frame displacement.
                    wrapper._sync_joints(anchor_qt)
                    _, quat_t = wrapper._get_ee_pose()
                    rot_t = Rotation.from_quat(quat_t)
                    delta_pos_in_t_frame = rot_t.inv().apply(rot_0.apply(accumulated_pos))
                    delta_rot_in_t_frame = accumulated_rot.as_euler("XYZ")

                    q_new = wrapper._project_delta_for_collision(
                        anchor_qt,
                        delta_pos_in_t_frame,
                        delta_rot_in_t_frame,
                        skip_collision=wrapper.skip_collision,
                    )
                    raw_step = np.concatenate([q_new, [float(d_gripper)]])
                    step_t = torch.tensor(raw_step, dtype=anchor_chunk.dtype, device=device).unsqueeze(0)
                    step_norm = wrapper._normalize_policy_guidance_action(step_t)
                    guidance_chunk[:, t_abs, :action_dim] = step_norm

    def _build_guidance_noise_from_chunk(
        self, guidance_chunk: Tensor, ratio: float, base_noise: Tensor | None = None
    ) -> Tensor:
        """Construct DENOISE noise from the (normalized) guidance chunk + base noise.

        Delegates to the wrapper method that used to live alongside this code.
        Kept as a thin shim so the source surfaces this entry point at the
        protocol level; the underlying math is wrapper-side because it can
        be shared with future sources that use the DENOISE path.
        """
        return self._wrapper._build_guidance_noise_from_chunk(guidance_chunk, ratio, base_noise=base_noise)

    # Suppress unused-state warnings from the Protocol type checker.
    _ = (GuidanceMode, GuidanceSourceState)
