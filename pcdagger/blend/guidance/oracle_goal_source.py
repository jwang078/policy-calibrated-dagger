"""Oracle-goal guidance source.

Method-triggered source (like RRT). When triggered, builds a joint-space
linear-interpolation chunk from a pre-jump `q_start` (the OLDEST entry in
the wrapper's `actual_q_history`; unlike RRT there is no per-trigger
lookback sampling and no env teleport back to `q_start`) to the oracle
env config's `q_goal_bias`. The chunk is then played
back verbatim, one waypoint per `next_action()` call, until exhausted.

Use case: DAgger intervention recording where the user wants a deterministic
"correction toward goal" signal as the intervention label, without the
RRT planner's planning cost or complexity. Each emitted frame is tagged
`FrameSource.BLEND_INTERVENTION_100` (verbatim ⇒ "100% guidance, 0%
policy"), which the recorder commits to the dataset.

Naming rule: the source describes *where the guidance comes from* (the
oracle's goal). "Blend" / "verbatim" is the integration concern, exposed
via `integration_mode`.

Currently implements `integration_mode = VERBATIM` only. `BLENDED` is a
planned extension (factor wrapper-side blend math + use it here) but
deferred until the simpler verbatim path is exercised end-to-end.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch

from lerobot.policies.guidance.base import (
    GuidanceCallCtx,
    GuidanceMode,
    GuidanceSourceState,
    GuidanceStepResult,
    IntegrationMode,
)
from lerobot.policies.rrt_to_goal import extract_task_goal

if TYPE_CHECKING:
    from lerobot.policies.shared_autonomy_wrapper import SharedAutonomyPolicyWrapper

logger = logging.getLogger(__name__)


class OracleGoalGuidanceSource:
    """Method-triggered oracle-goal source.

    Lifecycle:
      * `trigger(ctx)`: extract `q_goal_bias` from cached oracle config + the
        wrapper's recent `actual_q_history`, build an N-step linear
        interpolation in joint space, transition to EXECUTING.
      * `next_action(ctx)`: pop one waypoint from the chunk; emit as a
        normalized action tagged `FrameSource.BLEND_INTERVENTION_100`.
      * `cancel()`: hard-reset to IDLE.
    """

    name = "oracle_goal"

    def __init__(
        self,
        wrapper: SharedAutonomyPolicyWrapper,
        *,
        chunk_steps: int = 80,
        max_safe_joint_jump: float = 0.5,
        integration_mode: IntegrationMode = IntegrationMode.VERBATIM,
    ) -> None:
        self._wrapper = wrapper
        self.state: GuidanceSourceState = GuidanceSourceState()
        self.chunk_steps = int(chunk_steps)
        self.max_safe_joint_jump = float(max_safe_joint_jump)
        self.integration_mode = integration_mode
        # Cached oracle config — populated each step by `update_oracle_config`
        # (called from the wrapper's select_action when it pops the key from
        # the batch). The most-recent config is the one `trigger()` reads.
        self._oracle_cfg: dict | None = None

    # ── Protocol API ───────────────────────────────────────────────────── #

    def update(self, ctx: GuidanceCallCtx) -> None:
        del ctx  # No per-step state to refresh; everything is method-triggered.

    def is_active(self) -> bool:
        return self.state.mode in (GuidanceMode.PLANNING, GuidanceMode.EXECUTING)

    def trigger(self, ctx: GuidanceCallCtx | None = None) -> None:
        """Build the interpolation chunk and transition to EXECUTING.

        Toggle behavior (mirrors RRT): if already PLANNING/EXECUTING,
        request a cancel and return. Otherwise compute q_start from
        the wrapper's actual_q_history (oldest entry = pre-jump pose),
        extract q_goal_bias from the cached oracle config, build the
        chunk, transition to EXECUTING.
        """
        del ctx  # OracleGoal reads from the wrapper back-ref.
        st = self.state
        with st.lock:
            if st.mode in (GuidanceMode.PLANNING, GuidanceMode.EXECUTING):
                st.cancel_requested = True
                logger.info("OracleGoal cancellation requested (state=%s)", st.mode.value)
                return
            st.mode = GuidanceMode.PLANNING
            st.cancel_requested = False

        wrapper = self._wrapper

        if self.integration_mode != IntegrationMode.VERBATIM:
            # Defer BLENDED implementation: requires factoring the obs source's
            # blend math (DENOISE / LINEAR_INTERPOLATION) into a wrapper-level
            # `_apply_blend` so OracleGoal can call it without duplicating.
            # The simpler VERBATIM path covers the DAgger use case (clean
            # correction-toward-goal labels at frame tag BLEND_INTERVENTION_100).
            with st.lock:
                st.mode = GuidanceMode.IDLE
            raise NotImplementedError(
                "OracleGoalGuidanceSource.BLENDED integration_mode is not yet implemented. "
                "Use VERBATIM (default) for now. (Reminder on ratio semantics: "
                "forward_flow_ratio=0.0 is guidance-dominant — x_tsw = ratio*noise + "
                "(1-ratio)*guidance — and 1.0 is pure policy.)"
            )

        if self._oracle_cfg is None:
            logger.warning(
                "OracleGoal triggered but no oracle_env_config available. "
                "Set env.include_oracle_info=true to enable."
            )
            with st.lock:
                st.mode = GuidanceMode.IDLE
            return
        goal = extract_task_goal(self._oracle_cfg)
        if goal is None:
            logger.warning(
                "OracleGoal triggered but oracle_env_config has no task.target_ee_pos / target_ee_quat"
            )
            with st.lock:
                st.mode = GuidanceMode.IDLE
            return
        _, _, q_goal_bias = goal
        if q_goal_bias is None:
            logger.warning("OracleGoal triggered but oracle_env_config has no task.q_goal_bias")
            with st.lock:
                st.mode = GuidanceMode.IDLE
            return
        # Pull a pre-jump q_start: always the OLDEST entry in actual_q_history
        # (~N steps before the trigger — before the policy started commanding
        # bad actions). Simpler than RRT, which samples a per-trigger lookback
        # from [steps_min, steps_max] and teleports the env to q_start; here
        # there is no sampling and no teleport (the chunk's first waypoint just
        # commands the robot back toward the pre-jump pose). Falls back through
        # current actual_q → desired_q.
        if len(wrapper._actual_q_history) > 0:
            q_start_full = wrapper._actual_q_history[0].reshape(-1).copy()
        elif wrapper._latest_actual_q is not None:
            q_start_full = wrapper._latest_actual_q.reshape(-1).copy()
        else:
            # Guard lives here (not at the top of trigger) because _desired_q
            # is only needed for this last-resort fallback; next_action's
            # chunk-exhausted branch has its own assert.
            if wrapper._desired_q is None:
                logger.warning(
                    "OracleGoal triggered before any joint state was seeded "
                    "(no history, no actual_q, no desired_q); aborting"
                )
                with st.lock:
                    st.mode = GuidanceMode.IDLE
                return
            q_start_full = wrapper._desired_q.reshape(-1).copy()

        # Build the chunk as a linear interpolation in joint space, INCLUDING
        # the gripper (last dim). Shape: [chunk_steps, num_dofs + 1].
        num_dofs = wrapper.num_dofs
        q_start_arm = q_start_full[:num_dofs].astype(np.float64)
        q_goal_arm = np.asarray(q_goal_bias, dtype=np.float64).reshape(-1)[:num_dofs]
        # Preserve current gripper across the chunk — the oracle's q_goal_bias
        # specifies arm joints only, so the gripper column is held constant at
        # its pre-jump value for the whole chunk.
        gripper = float(q_start_full[num_dofs]) if q_start_full.shape[0] > num_dofs else 0.0
        arm_traj = np.linspace(q_start_arm, q_goal_arm, num=self.chunk_steps)
        # Per-row L_inf clamp: cap per-step joint motion at max_safe_joint_jump
        # to protect pybullet's integrator from teleport-like steps. (Same idea
        # as the BlendToGoalBiasHelper safety gate.) Cumulative effect: a path
        # that would have moved 2 rad in one step gets split across multiple.
        clamped = [arm_traj[0]]
        for i in range(1, len(arm_traj)):
            prev = clamped[-1]
            delta = arm_traj[i] - prev
            max_abs = float(np.max(np.abs(delta))) if delta.size else 0.0
            if max_abs > self.max_safe_joint_jump:
                delta = delta * (self.max_safe_joint_jump / max_abs)
            clamped.append(prev + delta)
        arm_traj = np.asarray(clamped)
        # Append the gripper column unchanged across all rows.
        chunk = np.concatenate([arm_traj, np.full((arm_traj.shape[0], 1), gripper)], axis=1)

        with st.lock:
            if st.cancel_requested:
                logger.info("OracleGoal plan ready but cancellation was requested; discarding")
                st.mode = GuidanceMode.IDLE
                st.cancel_requested = False
                return
            st.chunk = chunk
            st.step = 0
            st.mode = GuidanceMode.EXECUTING
            n_total = len(chunk)
            # target_steps is a caller hint used ONLY for this X/Y log line —
            # the source itself NEVER truncates: next_action plays the chunk
            # to len(st.chunk). Early cutoff is enforced externally by the
            # InterventionController.
            target = st.target_steps
            exec_str = f"{target}/{n_total}" if target is not None and target < n_total else f"{n_total}"
            logger.info(
                "OracleGoal executing %s waypoints (VERBATIM, q_start→q_goal_bias interpolation)",
                exec_str,
            )

    def cancel(self) -> None:
        st = self.state
        with st.lock:
            st.chunk = None
            st.step = 0
            st.mode = GuidanceMode.IDLE
            st.cancel_requested = False
            st.target_steps = None

    def next_action(self, ctx: GuidanceCallCtx) -> GuidanceStepResult:
        from lerobot.policies.teleop_recording import FrameSource

        st = self.state
        wrapper = self._wrapper

        if st.chunk is None or st.mode != GuidanceMode.EXECUTING:
            return GuidanceStepResult(action=wrapper.get_hold_action(ctx.inner_action))

        if st.cancel_requested:
            self.cancel()
            return GuidanceStepResult(
                action=wrapper.get_hold_action(ctx.inner_action),
                flush_inner_queue_after=True,
            )

        if st.step >= len(st.chunk):
            self.cancel()
            assert wrapper._desired_q is not None
            wrapper._last_raw_action = wrapper._desired_q.reshape(-1).copy()
            return GuidanceStepResult(
                action=wrapper.get_hold_action(ctx.inner_action),
                flush_inner_queue_after=True,
            )

        # Normal execution: pop next waypoint, build raw7 (joint + gripper).
        raw7 = st.chunk[st.step].astype(np.float64)
        st.step += 1
        wrapper._last_raw_action = raw7
        raw_t = torch.tensor(raw7, dtype=ctx.inner_dtype, device=ctx.inner_device).unsqueeze(0)
        action = wrapper._normalize_policy_guidance_action(raw_t)
        # VERBATIM ⇒ 100% intervention content, hence the _100 tag (the tag
        # family counts INTERVENTION share — the COMPLEMENT of
        # forward_flow_ratio, under which pure guidance is 0.0).
        return GuidanceStepResult(
            action=action,
            raw7=raw7,
            frame_source=FrameSource.BLEND_INTERVENTION_100,
        )

    def reset(self) -> None:
        st = self.state
        st.chunk = None
        st.step = 0
        st.mode = GuidanceMode.IDLE
        st.cancel_requested = False
        st.target_steps = None
        self._oracle_cfg = None

    def update_oracle_config(self, cfg: dict) -> None:
        """Cache the oracle config for the next trigger() call.

        Unlike RRT (which calls planner.load_obstacles here), OracleGoal
        doesn't need obstacles — straight-line interpolation has no
        collision check. Just stash the dict.
        """
        if cfg is None:
            return
        self._oracle_cfg = cfg
