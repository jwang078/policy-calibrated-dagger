"""RRT-to-goal guidance source.

Owns everything RRT-specific that used to live directly on
`SharedAutonomyPolicyWrapper`: the `RRTRuntimeState` instance, the
plan/cancel/finish methods, the pre-jump lookback config, the env-teleport
plumbing, and the obstacle adoption that used to happen in
`_maybe_update_oracle_obstacles`.

Wrapper-owned state (pybullet client, robot id, joint indices, num_dofs,
fps, joint limits, `_desired_q`, `_actual_q_history`, `_latest_actual_q`,
`_obstacle_ids`, `forward_flow_ratio`) is accessed via the back-reference
`self._wrapper`. This is a small encapsulation breach in exchange for not
having to refactor the wrapper's per-step state plumbing — the source needs
read access to a lot of wrapper-managed state to compute q_start, plan, and
log the trigger.

`integration_mode` is `VERBATIM`: the source's chunk is the action; the
inner policy's output is ignored while RRT is executing.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from typing import TYPE_CHECKING

import numpy as np
import pybullet as p
import torch

from lerobot.policies.guidance.base import (
    GuidanceCallCtx,
    GuidanceMode,
    GuidanceStepResult,
    IntegrationMode,
)
from lerobot.policies.rrt_to_goal import (
    PathSelectionStrategy,
    RRTPlanningError,
    RRTRuntimeState,
    RRTToGoalPlanner,
    extract_task_goal,
)

if TYPE_CHECKING:
    from lerobot.policies.shared_autonomy_wrapper import SharedAutonomyPolicyWrapper

logger = logging.getLogger(__name__)


class RRTGuidanceSource:
    """Method-triggered RRT-to-goal source.

    Lifecycle:
      * `trigger()` toggles: idle → request plan → (blocking or threaded)
                              `_do_rrt_plan()` → EXECUTING.
                              If already PLANNING/EXECUTING, sets
                              `state.cancel_requested = True` and returns.
      * `next_action(ctx)` pops one waypoint from `state.chunk` and emits it.
      * `cancel()` resets state to IDLE; the wrapper additionally clears its
                   obs-driven cache (the wrapper's `_cancel_rrt` does both).
    """

    name = "rrt"
    integration_mode = IntegrationMode.VERBATIM

    def __init__(
        self,
        wrapper: SharedAutonomyPolicyWrapper,
        *,
        pre_jump_lookback_steps: int = 5,
        teleport_to_q_start: bool = True,
        blocking_plan: bool = True,
        auto_pause_on_finish: bool = True,
        path_selection: PathSelectionStrategy | str | None = None,
    ) -> None:
        self._wrapper = wrapper
        # Same dataclass that used to live on the wrapper as `_rrt`. The
        # _RRTBackCompatView proxies `wrapper._rrt.X` to `self.state.X`.
        self.state: RRTRuntimeState = RRTRuntimeState()
        self.pre_jump_lookback_steps = int(pre_jump_lookback_steps)
        self.teleport_to_q_start = bool(teleport_to_q_start)
        self.blocking_plan = bool(blocking_plan)
        # Stays as a source-level attribute; wrapper exposes a property shim.
        # Disabled by lerobot-eval --intervention mode and last-mile RRTToGoalHelper
        # for their respective headless paths.
        self.auto_pause_on_finish = bool(auto_pause_on_finish)
        # Strategy used by the planner to score among IK-goal-candidate paths.
        # Default `EE_ARC_LENGTH` (today's behavior). Accepts the enum or its
        # string value ("ee_arc_length" / "joint_arc_length" /
        # "joint_velocity_match") for ergonomic config wiring.
        if path_selection is None:
            self.path_selection = PathSelectionStrategy.EE_ARC_LENGTH
        elif isinstance(path_selection, str):
            self.path_selection = PathSelectionStrategy(path_selection)
        else:
            self.path_selection = path_selection
        # Env handle for the pre-execution teleport. Set externally via the
        # wrapper's `set_env_for_teleport(env)`; None means teleport is a no-op.
        self._env_for_teleport: object | None = None
        # Latched on the FIRST oracle-obstacle load to tear down the wrapper's
        # hardcoded fallback obstacles (loaded in __init__ for no-oracle envs)
        # exactly once, when the first real oracle config arrives.
        self._oracle_replaced_static_obstacles = False

    # ── Public lifecycle API ────────────────────────────────────────────── #

    def update(self, ctx: GuidanceCallCtx) -> None:
        # RRT is method-triggered; nothing to do per-step here. The oracle
        # config that select_action popped from the batch is forwarded
        # through `update_oracle_config()` by the wrapper, not via the
        # per-step update path.
        del ctx

    def is_active(self) -> bool:
        """True while RRT is planning or executing.

        Reads `state.mode` under no lock — safe because the underlying enum
        write is atomic and a stale True/False is acceptable (the next tick
        re-checks).
        """
        return self.state.mode in (GuidanceMode.PLANNING, GuidanceMode.EXECUTING)

    def trigger(self, ctx: GuidanceCallCtx | None = None) -> None:
        """Toggle: start RRT-to-goal if idle, cancel if planning/executing.

        When `blocking_plan` is True (the default), this call blocks
        until planning + teleport finish and the source is in EXECUTING (or
        IDLE if planning failed). Use this in headless modes so the env is
        not stepped while the planner is working.

        When False (legacy / GUI mode), planning runs on a daemon worker
        thread and this method returns immediately; the source stays in
        PLANNING until the worker finishes.
        """
        del ctx  # RRT doesn't need ctx — it reads from the wrapper back-ref
        st = self.state
        with st.lock:
            if st.mode in (GuidanceMode.PLANNING, GuidanceMode.EXECUTING):
                st.cancel_requested = True
                logger.info("RRT cancellation requested (state=%s)", st.mode.value)
                return
            st.mode = GuidanceMode.PLANNING
            st.cancel_requested = False
        if self.blocking_plan:
            self._do_plan()
        else:
            threading.Thread(target=self._do_plan, daemon=True, name="rrt-plan").start()

    def cancel(self) -> None:
        """Reset source state to IDLE.

        Does NOT touch wrapper-side obs-driven cache (`_guided_chunk`,
        `_flush_inner_action_queue`). The wrapper's `_cancel_rrt` calls
        this then handles wrapper-side cleanup.
        """
        st = self.state
        st.chunk = None
        st.step = 0
        st.mode = GuidanceMode.IDLE
        st.cancel_requested = False
        # Clear the controller's advertised cancel hint so a later trigger
        # without one doesn't print a stale "X/Y" in the executing log.
        st.target_steps = None

    def next_action(self, ctx: GuidanceCallCtx) -> GuidanceStepResult:
        """Pop the next waypoint and return it as a normalized action.

        Only called when `is_active()` is True. Handles three sub-cases:
          1. cancel requested (or obs guidance arrived) → tell the wrapper
             to cancel and return a hold action.
          2. chunk exhausted → tell the wrapper to finish (auto-pause + cancel).
          3. normal step → emit next waypoint.
        """
        from lerobot.policies.teleop_recording import FrameSource

        st = self.state
        wrapper = self._wrapper
        # The wrapper's select_action already decided is_active() == True,
        # but it doesn't pre-check chunk-exhausted / cancel-requested. Do that here.
        if st.chunk is None or st.mode != GuidanceMode.EXECUTING:
            # Defensive: shouldn't happen given is_active() returned True,
            # but if it does, fall back to a hold to keep select_action's
            # contract (always returns an action).
            return GuidanceStepResult(action=wrapper.get_hold_action(ctx.inner_action))

        if st.cancel_requested:
            # Wrapper observes finished=True + cancel_requested and will
            # take the "cancel + hold" path. Source clears its own state
            # via cancel().
            self.cancel()
            return GuidanceStepResult(
                action=wrapper.get_hold_action(ctx.inner_action),
                finished=True,
                flush_inner_queue_after=True,
            )

        if st.step >= len(st.chunk):
            # Goal reached: source clears its own state; wrapper handles
            # auto-pause via the `auto_pause_on_finish` consult.
            self.cancel()
            assert wrapper._desired_q is not None  # seeded above
            wrapper._last_raw_action = wrapper._desired_q.reshape(-1).copy()
            return GuidanceStepResult(
                action=wrapper.get_hold_action(ctx.inner_action),
                finished=True,
                flush_inner_queue_after=True,
            )

        # Normal execution: pop waypoint, build raw7 (joint + gripper).
        wp = st.chunk[st.step][: wrapper.num_dofs]
        st.step += 1
        gripper = float(wrapper._desired_q[-1]) if wrapper._desired_q is not None else 0.0
        raw7 = np.concatenate([wp, [gripper]]).astype(np.float64)
        wrapper._last_raw_action = raw7
        raw_t = torch.tensor(raw7, dtype=ctx.inner_dtype, device=ctx.inner_device).unsqueeze(0)
        action = wrapper._normalize_policy_guidance_action(raw_t)
        return GuidanceStepResult(
            action=action,
            raw7=raw7,
            frame_source=FrameSource.RRT,
        )

    def reset(self) -> None:
        """Episode-boundary reset (called from wrapper.reset())."""
        st = self.state
        st.chunk = None
        st.step = 0
        st.mode = GuidanceMode.IDLE
        st.cancel_requested = False
        st.target_steps = None

    def update_oracle_config(self, cfg: dict) -> None:
        """Cache the oracle env config and (re)load obstacles when its hash changes.

        Loads obstacles into the wrapper's pybullet client and adopts them as
        the authoritative obstacle set for IK collision projection too. The
        hardcoded fallback bodies (loaded by `wrapper._load_static_obstacles`
        for no-oracle environments) are torn down on the first oracle load so
        the planner doesn't trip over duplicate table/wall geometry.
        """
        if cfg is None:
            return
        self.state.oracle_env_config = cfg
        try:
            planner = self._ensure_planner()
            oracle_ids = planner.load_obstacles(cfg)
        except Exception:
            logger.exception("Failed to load oracle obstacles into pybullet client")
            return
        # Replace the hardcoded fallback obstacles with the oracle set.
        # Idempotent — only runs once, the first time oracle info arrives.
        wrapper = self._wrapper
        if not self._oracle_replaced_static_obstacles:
            for body_id in wrapper._obstacle_ids:
                with contextlib.suppress(p.error):
                    p.removeBody(body_id, physicsClientId=wrapper._pb_client)
            self._oracle_replaced_static_obstacles = True
        wrapper._obstacle_ids = list(oracle_ids)

    # ── Wrapper-facing helpers ──────────────────────────────────────────── #

    def set_env_for_teleport(self, env: object) -> None:
        """Register the gym env handle used to teleport the sim's joint state
        before RRT execution begins.
        """
        self._env_for_teleport = env

    def set_teleport_enabled(self, enabled: bool) -> None:
        """Toggle the pre-execution teleport-to-q_start optimization.
        Called by `wrapper.disable_recording()` for non-recording eval paths
        that don't have a teleportable env handle.
        """
        self.teleport_to_q_start = bool(enabled)

    # ── Internal planning ───────────────────────────────────────────────── #

    def _ensure_planner(self) -> RRTToGoalPlanner:
        """Lazy-init the planner (so the import / setup happens only when used)."""
        wrapper = self._wrapper
        if self.state.planner is None:
            self.state.planner = RRTToGoalPlanner(
                pb_client=wrapper._pb_client,
                robot_id=wrapper._robot_id,
                joint_indices=list(range(1, 1 + wrapper.num_dofs)),
                ee_link_index=wrapper._ee_link,
                num_dofs=wrapper.num_dofs,
                fps=wrapper._fps,
                lower_limits=np.asarray(wrapper.lower_limits, dtype=np.float64),
                upper_limits=np.asarray(wrapper.upper_limits, dtype=np.float64),
                num_ik_candidates=16,
                path_selection=self.path_selection,
            )
        return self.state.planner

    def _do_plan(self) -> None:
        """Worker entry: plan a trajectory, then transition to EXECUTING.

        Same logic as the wrapper's old `_do_rrt_plan` but reads q_start
        sources, _desired_q, and forward_flow_ratio off `self._wrapper`.
        """
        st = self.state
        wrapper = self._wrapper
        try:
            if st.oracle_env_config is None:
                logger.warning(
                    "RRT triggered but no oracle_env_config available. "
                    "Set env.include_oracle_info=true to enable."
                )
                st.mode = GuidanceMode.IDLE
                return
            goal = extract_task_goal(st.oracle_env_config)
            if goal is None:
                logger.warning(
                    "RRT triggered but oracle_env_config has no task.target_ee_pos / target_ee_quat"
                )
                st.mode = GuidanceMode.IDLE
                return
            target_ee_pos, target_ee_quat, q_goal_bias = goal
            if wrapper._desired_q is None:
                logger.warning("RRT triggered before _desired_q seeded; aborting")
                st.mode = GuidanceMode.IDLE
                return

            planner = self._ensure_planner()
            planner.load_obstacles(st.oracle_env_config)
            # Prefer a pre-jump pose from the rolling history of actual joint
            # observations. The OLDEST entry in the buffer is approximately
            # `pre_jump_lookback_steps` steps before the trigger — usually
            # before the policy started commanding the bad chunk that led to
            # collision. Planning from this pose (rather than the current
            # post-collision actual_q) produces a clean trajectory, and the
            # teleport step below puts the sim there before execution so the
            # recording starts on a smooth, non-wedged segment. Fallbacks:
            #   1. oldest buffered actual_q (pre-jump)
            #   2. latest actual_q (current observation — used if buffer empty)
            #   3. _desired_q (commanded — used only if no obs ever processed)
            if len(wrapper._actual_q_history) > 0:
                q_start_full = wrapper._actual_q_history[0].reshape(-1).copy()
            elif wrapper._latest_actual_q is not None:
                q_start_full = wrapper._latest_actual_q.reshape(-1).copy()
            else:
                q_start_full = wrapper._desired_q.reshape(-1).copy()
            q_start = q_start_full[: wrapper.num_dofs].copy()

            # Compute the robot's recent joint velocity from the trailing
            # samples in `_actual_q_history`. Mean per-step delta over the
            # last few samples (matches the planner's leading-edge window
            # default). Used only by PathSelectionStrategy.JOINT_VELOCITY_MATCH;
            # other strategies ignore it. Pass `None` if the history is too
            # short to derive a velocity — the planner will raise if the
            # strategy needs it.
            recent_vel = self._compute_recent_joint_velocity(wrapper)

            chunk = planner.plan(
                q_start,
                target_ee_pos,
                target_ee_quat,
                q_goal_bias,
                recent_joint_velocity=recent_vel,
            )
            # Sim-only: teleport the env's robot to q_start so the first
            # commanded RRT action is followed by physics without a multi-frame
            # "catch up from wedged pose" interlude.
            if self.teleport_to_q_start:
                self._teleport_env_to_q_start(q_start_full)
        except RRTPlanningError as e:
            logger.warning("RRT planning failed: %s", e)
            st.mode = GuidanceMode.IDLE
            return
        except Exception:
            logger.exception("Unexpected error during RRT planning")
            st.mode = GuidanceMode.IDLE
            return

        with st.lock:
            if st.cancel_requested:
                logger.info("RRT plan ready but cancellation was requested; discarding")
                st.mode = GuidanceMode.IDLE
                st.cancel_requested = False
                return
            st.chunk = chunk
            st.step = 0
            st.mode = GuidanceMode.EXECUTING
            n_total = len(chunk)
            target = st.target_steps
            exec_str = f"{target}/{n_total}" if target is not None and target < n_total else f"{n_total}"
            logger.info(
                "RRT executing %s waypoints (current ratio=%.2f)", exec_str, wrapper.forward_flow_ratio
            )

    def _compute_recent_joint_velocity(self, wrapper: SharedAutonomyPolicyWrapper) -> np.ndarray | None:
        """Derive recent joint velocity from the wrapper's `_actual_q_history`.

        The history is a deque of recent actual_q observations sized to
        `pre_jump_lookback_steps + 1` entries. We compute the mean per-step
        joint delta over the trailing samples — matches the planner's
        leading-edge averaging window so the two velocities are directly
        comparable.

        Returns None when the history is too short to derive a velocity
        (need at least 2 entries). The planner only consumes this value
        when `PathSelectionStrategy.JOINT_VELOCITY_MATCH` is active.
        """
        history = wrapper._actual_q_history
        if len(history) < 2:
            return None
        # `history[0]` is the OLDEST entry; the last few are most recent.
        # Convert to a stacked array for easy diff.
        as_array = np.asarray(list(history), dtype=np.float64)  # [N, num_dofs+gripper]
        # Mean per-step delta across the (N-1) consecutive pairs.
        deltas = np.diff(as_array, axis=0)
        avg_delta = deltas.mean(axis=0)
        return avg_delta[: wrapper.num_dofs]

    def _teleport_env_to_q_start(self, q_start_full: np.ndarray) -> None:
        """Forward a joint-state teleport request to the gym env's robot
        server. Tolerant: silent no-op if no env handle has been provided or
        the env's robot_server doesn't expose teleport_joint_state (e.g. a
        backend that hasn't implemented it yet, or a real-robot stub).
        """
        env = self._env_for_teleport
        wrapper = self._wrapper
        if env is None:
            logger.warning(
                "Teleport-to-q_start requested but no env handle has been set on the wrapper. "
                "Call wrapper.set_env_for_teleport(env) once after env creation. "
                "Skipping teleport — the recorded intervention will start with catch-up frames."
            )
            return
        try:
            # Walk past any VectorEnv layers, then any gym.Wrapper layers, then
            # ask SplatSim's gym env for its inner robot_server. Tricky bit:
            # SplatSimGymEnv overrides .unwrapped to return self.robot_server
            # directly (non-standard), so after unwrap we may already be AT
            # the backend that has teleport_joint_state — no extra
            # .robot_server hop needed.
            base = env
            while hasattr(base, "envs"):  # VectorEnv
                base = base.envs[0]
            base = getattr(base, "unwrapped", base)  # gym.Wrapper chain
            target: object | None = None
            for candidate in (base, getattr(base, "robot_server", None)):
                if candidate is not None and hasattr(candidate, "teleport_joint_state"):
                    target = candidate
                    break
            if target is None:
                logger.warning(
                    "Teleport-to-q_start: could not find a teleport_joint_state method. "
                    "Walked env chain to: %s (type=%s). robot_server=%s. Skipping teleport.",
                    base,
                    type(base).__name__,
                    type(getattr(base, "robot_server", None)).__name__
                    if getattr(base, "robot_server", None) is not None
                    else "<missing>",
                )
                return
            splatsim_robot = getattr(target, "splatsim_robot", None)
            logger.info(
                "Teleporting sim to RRT q_start (lookback=%d): %s — via %s.teleport_joint_state",
                self.pre_jump_lookback_steps,
                np.array2string(q_start_full[: wrapper.num_dofs], precision=3),
                type(target).__name__,
            )
            target.teleport_joint_state(splatsim_robot, q_start_full.tolist())  # type: ignore[attr-defined]
            logger.info("Teleport call returned.")
        except Exception:
            logger.exception(
                "Teleport-to-q_start failed; continuing without rewind. "
                "RRT will still execute but the recorded intervention may "
                "start with a few catch-up frames."
            )
