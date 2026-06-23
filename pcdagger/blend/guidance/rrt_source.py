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
        # Collision-detection mode (see SharedAutonomyConfig docstring).
        # "pre_jump_lookback" → use the rewind/teleport flow, sample lookback
        # from [steps_min, steps_max].
        # "future_chunk" → no rewind. Trigger callers pass no_lookback=True
        # to trigger(); _do_plan() reads q_start = wrapper._latest_actual_q.
        collision_detection: str = "pre_jump_lookback",
        # Tunables for pre_jump_lookback mode. Ignored when
        # collision_detection == "future_chunk".
        pre_jump_lookback_steps_min: int = 5,
        pre_jump_lookback_steps_max: int | None = None,
        teleport_to_q_start: bool = True,
        blocking_plan: bool = True,
        auto_pause_on_finish: bool = True,
        path_selection: PathSelectionStrategy | str | None = None,
        segment_at_sharp_corners: bool = True,
        ik_goal_selection: str | None = None,
        num_path_candidates_per_ik: int = 1,
        max_path_attempts_per_ik: int = 5,
        path_perturbation_scale: float = 0.001,
        num_ik_candidates: int = 16,
        obstacle_clearance: float | None = None,
        self_collision_clearance: float | None = None,
        self_collision_skip_pairs: list[list[int]] | None = None,
        diagnostic_log_pairs: str = "off",
    ) -> None:
        self._wrapper = wrapper
        # Same dataclass that used to live on the wrapper as `_rrt`. The
        # _RRTBackCompatView proxies `wrapper._rrt.X` to `self.state.X`.
        self.state: RRTRuntimeState = RRTRuntimeState()
        if collision_detection not in ("pre_jump_lookback", "future_chunk", "hybrid"):
            raise ValueError(
                "collision_detection must be 'pre_jump_lookback', "
                f"'future_chunk', or 'hybrid', got {collision_detection!r}"
            )
        self.collision_detection = collision_detection
        # Lookback tunables — only consulted in pre_jump_lookback mode.
        # In future_chunk mode these are stored but unused (the no-lookback
        # path bypasses them).
        self.pre_jump_lookback_steps_min = int(pre_jump_lookback_steps_min)
        # Optional upper bound for per-trigger random lookback sampling. When
        # None, behavior is deterministic at `pre_jump_lookback_steps_min`.
        # When set, each `_do_plan` invocation samples a fresh lookback in
        # the closed interval [_min, _max].
        # The wrapper sizes its actual_q_history deque to fit the max so the
        # sampled value can always be honored.
        self.pre_jump_lookback_steps_min_max = (
            int(pre_jump_lookback_steps_max) if pre_jump_lookback_steps_max is not None else None
        )
        if self.collision_detection in ("pre_jump_lookback", "hybrid") and (
            self.pre_jump_lookback_steps_min_max is not None
            and self.pre_jump_lookback_steps_min_max < self.pre_jump_lookback_steps_min
        ):
            raise ValueError(
                f"pre_jump_lookback_steps_max ({self.pre_jump_lookback_steps_min_max}) "
                f"must be >= pre_jump_lookback_steps_min ({self.pre_jump_lookback_steps_min})"
            )
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
        # Flag forwarded to ruckig parametrization. When True (default),
        # ruckig splits the path at sharp-angle waypoints and forces zero
        # velocity at each boundary — historical stop-and-go mode. When
        # False, ruckig optimizes the full path in one call without forced
        # internal stops. Empirically the two modes produce indistinguishable
        # trajectories on typical manipulation RRT plans (which rarely have
        # sharp internal corners), so True is the conservative default. See
        # `splatsim.utils.rrt_path_utils.ruckig_parametrize_path`.
        self.segment_at_sharp_corners = bool(segment_at_sharp_corners)
        # Forwarded to the planner. When set, the planner picks IK
        # candidates by goal-state geometry alone (no path scoring) and
        # path_selection is unused. See IkGoalSelectionStrategy in
        # `lerobot.policies.rrt_to_goal`. None = no IK pre-selection.
        self.ik_goal_selection = ik_goal_selection
        # Per-IK multi-path-candidate scoring knobs. See planner ctor for
        # semantics. Default num=1 preserves single-attempt behavior.
        self.num_path_candidates_per_ik = int(num_path_candidates_per_ik)
        self.max_path_attempts_per_ik = int(max_path_attempts_per_ik)
        self.path_perturbation_scale = float(path_perturbation_scale)
        self.num_ik_candidates = int(num_ik_candidates)
        # RRT-planner collision clearances. None = use SplatSim's defaults
        # (_COLLISION_CLEARANCE = 0.01 m obstacle, self = 0.0 m). Override
        # via the SA config to give the policy drift margin along the
        # planned path — see SharedAutonomyConfig docstring for trade-offs.
        # The planner stores these and threads them through every
        # check_links_in_collision + get_path call.
        self.obstacle_clearance = float(obstacle_clearance) if obstacle_clearance is not None else None
        self.self_collision_clearance = (
            float(self_collision_clearance) if self_collision_clearance is not None else None
        )
        # Normalized to list[tuple[int, int]] for the planner. None / empty
        # → no skips. Order within each pair doesn't matter; the planner
        # normalizes (a,b) and (b,a) to the same frozenset for lookup.
        if self_collision_skip_pairs:
            self.self_collision_skip_pairs = [
                (int(pair[0]), int(pair[1])) for pair in self_collision_skip_pairs
            ]
        else:
            self.self_collision_skip_pairs = None
        # Validated upstream by SharedAutonomyConfig.__post_init__. Re-check
        # here in case the source is constructed standalone (e.g. tests).
        if diagnostic_log_pairs not in ("off", "first", "always"):
            raise ValueError(
                f"diagnostic_log_pairs must be one of 'off'/'first'/'always', got {diagnostic_log_pairs!r}"
            )
        self.diagnostic_log_pairs = diagnostic_log_pairs
        # Env handle for the pre-execution teleport. Set externally via the
        # wrapper's `set_env_for_teleport(env)`; None means teleport is a no-op.
        self._env_for_teleport: object | None = None
        # Latched on the FIRST oracle-obstacle load to tear down the wrapper's
        # hardcoded fallback obstacles (loaded in __init__ for no-oracle envs)
        # exactly once, when the first real oracle config arrives.
        self._oracle_replaced_static_obstacles = False
        # Cached full-DOF q_start (joints + gripper) from the most recent
        # plan(). Used by request_retry_after_collision() to teleport back
        # to the SAME start the original plan used, rather than the current
        # mid-execution pose where the collision happened.
        self._last_q_start_full: np.ndarray | None = None

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

    def trigger(self, ctx: GuidanceCallCtx | None = None, *, no_lookback: bool = False) -> None:
        """Toggle: start RRT-to-goal if idle, cancel if planning/executing.

        When `blocking_plan` is True (the default), this call blocks
        until planning + teleport finish and the source is in EXECUTING (or
        IDLE if planning failed). Use this in headless modes so the env is
        not stepped while the planner is working.

        When False (legacy / GUI mode), planning runs on a daemon worker
        thread and this method returns immediately; the source stays in
        PLANNING until the worker finishes.

        Args:
            ctx: unused for RRT (it reads from the wrapper back-ref).
            no_lookback: when True, _do_plan skips the lookback sampling
                and teleport entirely — q_start = wrapper._latest_actual_q
                (the robot's current config) and ruckig's start_vel matches
                the robot's recent joint velocity. Used by the wrapper's
                future-chunk predictive shield (and by the controller when
                operating in `collision_detection="future_chunk"` mode).
                When False (default), the historical lookback flow runs.
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
            # Stash the no_lookback flag on the runtime state so _do_plan
            # (potentially running on a worker thread) can read it without
            # touching the source object's ctor-time mode.
            st.no_lookback = bool(no_lookback)
            # Fresh user-initiated trigger → clear the collision-history
            # filter so a brand-new RRT cycle isn't blocked by previous
            # exclusions. Retries call _do_plan directly without going
            # through trigger(), so the filter survives across retries
            # within the same cycle.
            st.excluded_q_goals = []
        if self.blocking_plan:
            self._do_plan()
        else:
            threading.Thread(target=self._do_plan, daemon=True, name="rrt-plan").start()

    def request_retry_after_collision(self) -> bool:
        """Abort the current EXECUTING chunk and replan with the offending
        IK goal added to the exclusion list. Used when the controller
        observes `info["in_collision"]` mid-execution — typically because
        ruckig smoothing curved the RRT-raw path through an obstacle the
        raw path avoided.

        Sequence:
          1. Add the chunk's chosen q_goal to `excluded_q_goals` so the
             same IK branch won't be re-picked.
          2. Force state back to PLANNING.
          3. Call `_do_plan` synchronously. This re-runs the IK solver,
             filters out excluded goals, runs RRT against remaining
             candidates, and re-applies ruckig — same code path as the
             original trigger. Teleport-to-q_start fires again with the
             SAME pre-trigger pose (cached in `_last_q_start_full`).
          4. On success → EXECUTING with the new chunk.
             On failure (all IKs exhausted or all RRT attempts failed) →
             planner raises, _do_plan catches and goes IDLE; the
             controller's normal "plan failed" branch then handles the
             rest (backoff / retrigger / mark scenario).

        Returns True if a new plan was successfully started, False if
        not (no chosen_q_goal cached, or planner failed). The caller
        (controller) doesn't strictly need this — it'll see the source's
        mode transition on the next tick — but it's useful for logging.
        """
        st = self.state
        if st.mode != GuidanceMode.EXECUTING:
            logger.warning(
                "request_retry_after_collision called but source is %s, not EXECUTING; ignoring",
                st.mode.value,
            )
            return False
        if st.chosen_q_goal is None:
            logger.warning(
                "request_retry_after_collision: no cached chosen_q_goal; cannot exclude. Cancelling instead."
            )
            with st.lock:
                st.cancel_requested = True
            return False
        with st.lock:
            st.excluded_q_goals.append(np.asarray(st.chosen_q_goal, dtype=np.float64).copy())
            logger.warning(
                "Mid-execution collision: excluding IK goal %s and replanning (%d total excluded so far)",
                np.array2string(np.asarray(st.chosen_q_goal), precision=3),
                len(st.excluded_q_goals),
            )
            # Reset chunk state — _do_plan will overwrite if planning succeeds.
            st.chunk = None
            st.step = 0
            st.mode = GuidanceMode.PLANNING
            st.cancel_requested = False
        # blocking_plan or not — for collision retry we run synchronously
        # so the controller's next tick sees the new EXECUTING state.
        self._do_plan()
        return st.mode == GuidanceMode.EXECUTING

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
        # Pick up the env's URDF-known self-collision skip pairs (published
        # via SplatSim's `get_env_config()` → ZMQ → here) so the SA-config
        # override remains OPTIONAL: by default, the planner uses whatever
        # the env's URDF declares (e.g. `[(0, 2)]` for the UR robot's
        # base_link vs upper_arm_link). If the user explicitly set
        # `rrt_self_collision_skip_pairs` on the SA config, THAT wins —
        # treat the CLI as a per-run override (use case: add custom pairs
        # beyond what the env publishes).
        if self.self_collision_skip_pairs is None:
            _oracle_pairs = cfg.get("self_collision_skip_pairs") or []
            if _oracle_pairs:
                normalized = [(int(p[0]), int(p[1])) for p in _oracle_pairs]
                # Update both the source's view AND the planner's already-
                # built kwargs dict. `_collision_kwargs` is the one the
                # planner unpacks at every callsite, so this update is
                # what actually changes behavior.
                self.self_collision_skip_pairs = normalized
                planner._collision_kwargs["self_collision_skip_pairs"] = normalized
                logger.info(
                    "RRTGuidanceSource: picked up self_collision_skip_pairs=%s "
                    "from oracle env config (no SA-config override).",
                    normalized,
                )
        # One-shot human-readable summary of the robot's URDF link layout
        # AND the active self-collision skip set. Helps users sanity-check
        # the skip list against the actual link indices (which depend on
        # URDF order and can shift if the robot is swapped). Runs once per
        # planner — guarded by an instance attr on self.
        if not getattr(self, "_link_layout_logged", False):
            self._link_layout_logged = True
            wrapper = self._wrapper
            n_joints = p.getNumJoints(wrapper._robot_id, physicsClientId=wrapper._pb_client)
            jtype_str = {
                p.JOINT_REVOLUTE: "rev",
                p.JOINT_PRISMATIC: "pri",
                p.JOINT_FIXED: "fix",
                p.JOINT_SPHERICAL: "sph",
                p.JOINT_PLANAR: "pla",
            }
            link_names: list[str] = ["base (WORLD)"]
            logger.info("Robot URDF link layout (n_links=%d, incl. base at index -1):", n_joints + 1)
            logger.info(
                "  idx=%2d  name=%-28s  parent=%-3s  jtype=%s",
                -1,
                "base (WORLD)",
                "—",
                "—",
            )
            for j in range(n_joints):
                info = p.getJointInfo(wrapper._robot_id, j, physicsClientId=wrapper._pb_client)
                name = info[12].decode()
                parent = info[16]
                jt = jtype_str.get(info[2], "?")
                link_names.append(name)
                logger.info(
                    "  idx=%2d  name=%-28s  parent=%-3d  jtype=%s",
                    j,
                    name,
                    parent,
                    jt,
                )
            # Resolved skip pairs (may come from SA config override OR
            # the env's published list, whichever was set above).
            active_pairs = self.self_collision_skip_pairs or []
            if active_pairs:
                logger.info(
                    "Active SELF_COLLISION_SKIP_PAIRS (%d total):",
                    len(active_pairs),
                )
                # Pretty-print each pair with the resolved link names so
                # the user can audit which structural pair each idx tuple
                # actually refers to.
                for a, b in active_pairs:
                    na = link_names[a + 1] if 0 <= a + 1 < len(link_names) else f"?(idx={a})"
                    nb = link_names[b + 1] if 0 <= b + 1 < len(link_names) else f"?(idx={b})"
                    logger.info("  (%2d, %2d)  %s  ⟷  %s", a, b, na, nb)
            else:
                logger.info(
                    "Active SELF_COLLISION_SKIP_PAIRS: <none> — all non-adjacent "
                    "pairs will be checked for self-collision.",
                )
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
                num_ik_candidates=self.num_ik_candidates,
                path_selection=self.path_selection,
                segment_at_sharp_corners=self.segment_at_sharp_corners,
                ik_goal_selection=self.ik_goal_selection,
                num_path_candidates_per_ik=self.num_path_candidates_per_ik,
                max_path_attempts_per_ik=self.max_path_attempts_per_ik,
                path_perturbation_scale=self.path_perturbation_scale,
                obstacle_clearance=self.obstacle_clearance,
                self_collision_clearance=self.self_collision_clearance,
                self_collision_skip_pairs=self.self_collision_skip_pairs,
                diagnostic_log_pairs=self.diagnostic_log_pairs,
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
            # `pre_jump_lookback_steps_min` steps before the trigger — usually
            # before the policy started commanding the bad chunk that led to
            # collision. Planning from this pose (rather than the current
            # post-collision actual_q) produces a clean trajectory, and the
            # teleport step below puts the sim there before execution so the
            # recording starts on a smooth, non-wedged segment. Fallbacks:
            #   1. oldest buffered actual_q (pre-jump)
            #   2. latest actual_q (current observation — used if buffer empty)
            #   3. _desired_q (commanded — used only if no obs ever processed)
            # Branch on the per-trigger no_lookback flag (stashed by
            # trigger() onto st before _do_plan started). Lookback path
            # rewinds + teleports; no-lookback path plans from current q.
            no_lookback = bool(st.no_lookback)
            # Consume the flag so a subsequent retry inside the same cycle
            # (request_retry_after_collision → _do_plan directly) doesn't
            # carry over a stale True. Retries always use the same
            # _last_q_start_full anyway via the teleport callsite below.
            st.no_lookback = False
            if no_lookback:
                # Predictive-shield / future_chunk path: no rewind, no
                # teleport. q_start = robot's CURRENT joint state. We're
                # safe to plan from here by construction (the FK shield
                # checked the chunk BEFORE letting it execute).
                if wrapper._latest_actual_q is not None:
                    q_start_full = wrapper._latest_actual_q.reshape(-1).copy()
                else:
                    q_start_full = wrapper._desired_q.reshape(-1).copy()
                effective_lookback = 0
                logger.info(
                    "RRT plan (no-lookback): q_start = current robot state (skipping pre-jump teleport)."
                )
            else:
                # Sample the per-trigger effective lookback. When max is
                # None, this collapses to the fixed value (legacy
                # behavior). When set, samples uniformly from the closed
                # [min, max] interval — adds dataset diversity at
                # recording time. Sampling uses Python's global RNG so it
                # inherits whatever seed lerobot-eval set up.
                if self.pre_jump_lookback_steps_min_max is not None:
                    import random

                    effective_lookback = random.randint(
                        self.pre_jump_lookback_steps_min,
                        self.pre_jump_lookback_steps_min_max,
                    )
                    logger.info(
                        "Sampled lookback for this trigger: %d (range [%d, %d])",
                        effective_lookback,
                        self.pre_jump_lookback_steps_min,
                        self.pre_jump_lookback_steps_min_max,
                    )
                else:
                    effective_lookback = self.pre_jump_lookback_steps_min
                # Cap lookback at the wrapper's "frames since last RRT cycle
                # ended" counter so the rewind can't reach back into a prior
                # cycle's trajectory. Sampling from that region would
                # teleport the robot to a joint config that RRT itself put
                # the robot at (not a config the POLICY drove to), and the
                # env-teleport spike then leaks into the recorded teleop
                # dataset as a 1+ rad single-frame discontinuity — exactly
                # the bug we observed in lever_g0_d30_coll_03dag_diff_r_dag1
                # episode 18. Cap is the simplest fix that preserves the
                # "rewind to a moving state" benefit lookback was designed
                # for: rewind as far as possible into POLICY-driven history,
                # never into RRT-era frames.
                post_rrt_frames = getattr(wrapper, "_frames_since_last_rrt_end", None)
                if post_rrt_frames is not None and post_rrt_frames < effective_lookback:
                    logger.info(
                        "Capping lookback %d → %d (only %d policy-driven frames "
                        "since the last RRT cycle ended; further rewind would "
                        "land in a prior RRT chunk's trajectory).",
                        effective_lookback,
                        post_rrt_frames,
                        post_rrt_frames,
                    )
                    effective_lookback = post_rrt_frames
                # Walk back through `_actual_q_history` to find the joint
                # state `effective_lookback` steps before now. deque
                # indexing: [-1] = most recent (just pushed), [-(N+1)] =
                # N steps ago. Fall back to the OLDEST entry if the
                # buffer hasn't filled to the requested depth yet (e.g.
                # trigger fires within the first few frames of a scenario).
                _hist = wrapper._actual_q_history
                if effective_lookback <= 0:
                    # Cap collapsed to zero — typically because this trigger
                    # fired the same tick a prior RRT cycle ended. There's
                    # no policy-driven history to rewind into, so degenerate
                    # to "use current state" (= the no-lookback path's
                    # q_start choice). Subsequent escape-handling + ruckig
                    # start_vel still apply; we just skip the teleport.
                    if wrapper._latest_actual_q is not None:
                        q_start_full = wrapper._latest_actual_q.reshape(-1).copy()
                    else:
                        q_start_full = wrapper._desired_q.reshape(-1).copy()
                elif len(_hist) > effective_lookback:
                    q_start_full = _hist[-(effective_lookback + 1)].reshape(-1).copy()
                elif len(_hist) > 0:
                    q_start_full = _hist[0].reshape(-1).copy()
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

            # In no-lookback mode we want ruckig to begin at the robot's
            # ACTUAL recent velocity instead of v=0 — otherwise the chunk
            # produces a dead-stop onset that contradicts the continuous
            # motion the robot is in. recent_vel may be None when the
            # history hasn't accumulated enough samples; in that case we
            # silently fall back to v=0 (same as the lookback path).
            _ruckig_start_vel = recent_vel if no_lookback else None
            chunk, escape_end_q = planner.plan(
                q_start,
                target_ee_pos,
                target_ee_quat,
                q_goal_bias,
                recent_joint_velocity=recent_vel,
                exclude_q_goals=list(self.state.excluded_q_goals),
                ruckig_start_vel=_ruckig_start_vel,
            )
            # Capture the chosen IK goal so request_retry_after_collision()
            # can add it to the excluded list on retry. Set on planner state
            # right before its return; copy out so source state is independent.
            if planner._last_chosen_q_goal is not None:
                self.state.chosen_q_goal = planner._last_chosen_q_goal.copy()
            # Remember q_start_full for any retry — the retry must teleport
            # back to the same start (the policy's pre-trigger pose, NOT the
            # current pose mid-execution).
            self._last_q_start_full = q_start_full.copy()
            # Sim-only env teleport before chunk execution. Three cases:
            #
            #   (a) Escape happened (escape_end_q is not None): teleport
            #       directly to the post-escape config. This REPLACES any
            #       q_start_full teleport — teleporting to the wedged config
            #       first is pointless (the planner already moved past it)
            #       and would put the env robot in collision for one tick.
            #       Why teleport rather than execute the escape waypoints in
            #       the env: the escape segment used to be prepended to the
            #       chunk and stepped via env.step() so the PD controller
            #       could physically push the robot out of contact. That
            #       worked, but recorded ~3% of intervention episodes with
            #       10×-mean-delta outlier frames at onset (sim-PD artifact,
            #       not a transferable skill). Now planner.plan() returns
            #       chunk WITHOUT the escape, and we land the env robot at
            #       the planner's escape end-state via a single teleport.
            #       The recorded episode begins at the smooth ruckig start.
            #
            #   (b) No escape + lookback path (historical default): teleport
            #       to q_start_full (the lookback-sampled config). Unchanged.
            #
            #   (c) No escape + no-lookback path (collision shield / future-
            #       chunk): no teleport — the robot is already at q_start
            #       by construction.
            if self.teleport_to_q_start:
                if escape_end_q is not None:
                    # Build a full-state vector (joint + gripper if present)
                    # by reusing q_start_full as the template and overwriting
                    # the joint slots with escape_end_q.
                    teleport_target = q_start_full.copy()
                    teleport_target[: len(escape_end_q)] = escape_end_q
                    self._teleport_env_to_q_start(teleport_target, 0)
                elif not no_lookback:
                    self._teleport_env_to_q_start(q_start_full, effective_lookback)
        except RRTPlanningError as e:
            logger.warning("RRT planning failed: %s", e)
            st.mode = GuidanceMode.IDLE
            return
        except Exception as e:
            # Surface the exception type + message inline so it's visible
            # alongside the "ERROR ... Unexpected error" line, in addition to
            # the full traceback that logger.exception() writes via exc_info.
            logger.exception("Unexpected error during RRT planning: %s: %s", type(e).__name__, e)
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
            # NOTE: do NOT include `forward_flow_ratio` in this log — RRT
            # chunks play VERBATIM (see shared_autonomy_wrapper.select_action
            # line ~1105: `wp = rrt.chunk[rrt.step][: self.num_dofs]` with
            # no ratio math). Including the wrapper's `forward_flow_ratio`
            # here was previously misleading users into thinking RRT
            # recordings were being blended at that ratio when in fact
            # the ratio only governs obs-teleop blending (a different
            # guidance source that isn't active during RRT execution).
            logger.info(
                "RRT executing %s waypoints (verbatim playback — forward_flow_ratio not applied to RRT chunks)",
                exec_str,
            )

    def _compute_recent_joint_velocity(self, wrapper: SharedAutonomyPolicyWrapper) -> np.ndarray | None:
        """Derive recent joint velocity from the wrapper's `_actual_q_history`.

        The history is a deque of recent actual_q observations sized to
        `pre_jump_lookback_steps_min + 1` entries (or `_max + 1` when the
        random-sampling mode is configured). We compute the mean per-step
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

    def _teleport_env_to_q_start(self, q_start_full: np.ndarray, lookback_used: int) -> None:
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
                lookback_used,
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
