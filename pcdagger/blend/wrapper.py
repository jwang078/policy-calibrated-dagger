#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Policy wrapper for shared autonomy that works transparently with lerobot_eval.py.

Extracts policy_guidance_chunk (a 7-d delta vector [dx,dy,dz,droll,dpitch,dyaw,gripper])
from the observation dict, then applies FK→IK guidance to the full predicted action chunk
and re-runs partial diffusion/flow-matching denoising with the guided chunk as the noise
anchor. This means guidance is applied coherently across the entire action window.

Works with any noise/flow-based policy (PI0.5, Diffusion) without modifying lerobot_eval.py.
"""

from __future__ import annotations

import collections
import logging
import threading
from enum import Enum
from typing import TYPE_CHECKING, cast

import numpy as np
import pybullet as p
import torch
from scipy.spatial.transform import Rotation
from splatsim.configs.env_config import SplatObjectConfig
from splatsim.utils.paths import resolve_splatsim_path
from torch import Tensor, nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.shared_autonomy import FutureChunkConfig, PreJumpLookbackConfig
from lerobot.policies.guidance import GuidanceCallCtx
from lerobot.policies.guidance.observation_teleop_source import ObservationTeleopGuidanceSource
from lerobot.policies.guidance.oracle_goal_source import OracleGoalGuidanceSource
from lerobot.policies.guidance.rrt_source import RRTGuidanceSource
from lerobot.policies.guidance.views import _RRTBackCompatView
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rrt_to_goal import RRTMode
from lerobot.policies.teleop_recording import FrameSource
from lerobot.processor import AbsoluteActionsProcessorStep, PolicyProcessorPipeline, to_relative_actions
from lerobot.utils.constants import ACTION

if TYPE_CHECKING:
    from lerobot.policies.teleop_recording import TeleopRecordingContext

logger = logging.getLogger(__name__)

OBS_GUIDANCE_CHUNK = "observation.policy_guidance_chunk"
OBS_STATE = "observation.state"


class PolicyGuidanceRepresentation(Enum):
    """How the guidance action passed in observation.policy_guidance_chunk is interpreted.

    DELTA:        (default) 7-d EE delta [dx, dy, dz, droll, dpitch, dyaw, gripper].
                  FK→IK is applied to convert to absolute joint positions.
    ABSOLUTE_POS: 7-d absolute joint positions [j1, …, j6, gripper] (raw, unnormalized).
                  FK→IK is skipped; the guidance is used directly as the target joints.
    """

    DELTA = "delta"
    ABSOLUTE_POS = "absolute_pos"


class BlendMode(Enum):
    """How often guidance blending is applied within an action chunk.

    EVERY_STEP:     (default) Re-blend every select_action call that has guidance.
                    Each call runs a full denoising pass with fresh random noise.
                    Allows continuous steering but sacrifices temporal coherence.
    ONCE_PER_CHUNK: Blend only when a new anchor chunk is generated (chunk exhausted
                    or first guidance call). Subsequent calls with guidance drain the
                    blended chunk without re-blending. Produces temporally coherent
                    action chunks from a single denoising pass.
    """

    EVERY_STEP = "every_step"
    ONCE_PER_CHUNK = "once_per_chunk"


class GuidanceBlendStrategy(Enum):
    """How the guidance chunk is blended with the policy output.

    DENOISE:     (default) Build partially-noised guidance, then run the model's
                 denoising from t=ratio down to t=0. The model's visual conditioning
                 can override guidance if it has a strong prior.
    INTERPOLATE: Simple linear interpolation in clean action space:
                 blended = ratio * policy_output + (1-ratio) * guidance.
                 No denoising involved. Guarantees the guidance has proportional
                 influence, but the result is not "on-manifold".
    """

    DENOISE = "denoise"
    INTERPOLATE = "interpolate"


class SharedAutonomyPolicyWrapper(PreTrainedPolicy):
    """Wraps a policy to blend human EE-delta guidance with diffusion/flow policy output.

    The keyboard agent sends a 7-d delta [dx,dy,dz,droll,dpitch,dyaw,gripper] as
    observation.policy_guidance_chunk (or all-NaN when no key is held).

    At each select_action() call this wrapper:
    1. Always calls inner_policy.select_action(batch) to keep obs queues updated (needed
       for policies like diffusion that maintain n_obs_steps history).
    2. When guidance is active: applies FK→IK delta to all remaining steps in the current
       chunk, re-runs partial denoising (noise scheduling) with the guided chunk as anchor,
       and returns the next action from the blended chunk buffer.
    3. The blended chunk buffer (_guided_chunk) is refreshed every guidance step with the
       latest delta, and drains step-by-step between refreshes.
    4. On transition out of guidance: drains the remaining buffer before handing back to
       the inner policy.
    """

    config_class = PreTrainedConfig
    name = "shared_autonomy_wrapper"

    # Class-level default arm joint limits (matches SplatSim UR5's
    # PybulletRobotServerBase). Used as a fallback when the loaded URDF
    # doesn't publish limits. Instance-level `self.lower_limits` /
    # `self.upper_limits` (populated in `_load_urdf`) override these per-
    # robot from the URDF's own `getJointInfo` values, so no subclass edit
    # is needed for new robots.
    lower_limits = [-np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi]
    upper_limits = [np.pi, 0, np.pi, np.pi, np.pi, np.pi]

    def __init__(
        self,
        inner_policy: PreTrainedPolicy,
        inverse_postprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        inverse_preprocessor: PolicyProcessorPipeline,
        forward_flow_ratio: float,
        show_slider: bool = True,
        start_paused: bool = False,
        robot_name: str | None = None,
        max_joint_delta: float = 0.016,
        num_dofs: int | None = None,
        policy_guidance_representation: PolicyGuidanceRepresentation = PolicyGuidanceRepresentation.DELTA,
        blend_mode: BlendMode | str = BlendMode.EVERY_STEP,
        guidance_blend_strategy: GuidanceBlendStrategy | str = GuidanceBlendStrategy.DENOISE,
        n_anchor_steps: int = 0,
        fps: int = 30,
        rrt_collision_detection: str = "pre_jump_lookback",
        rrt_pre_jump_lookback: PreJumpLookbackConfig | None = None,
        rrt_future_chunk: FutureChunkConfig | None = None,
        rrt_teleport_to_q_start: bool = True,
        rrt_blocking_plan: bool = True,
        rrt_path_selection: str | None = None,
        rrt_segment_at_sharp_corners: bool = True,
        rrt_ik_goal_selection: str | None = None,
        rrt_num_path_candidates_per_ik: int = 1,
        rrt_max_path_attempts_per_ik: int = 5,
        rrt_path_perturbation_scale: float = 0.001,
        rrt_num_ik_candidates: int = 16,
        rrt_obstacle_clearance: float | None = None,
        rrt_self_collision_clearance: float | None = None,
        rrt_in_progress_obstacle_clearance: float | None = None,
        rrt_in_progress_self_collision_clearance: float | None = None,
        rrt_self_collision_skip_pairs: list[list[int]] | None = None,
        rrt_diagnostic_log_pairs: str = "off",
        rrt_ik_skip_gripper_obstacle_pairs: bool = False,
        rrt_escape_clearance_factor: float = 1.5,
        rrt_rewind_clearance_factor: float | None = None,
        rrt_final_approach_dist: float = 0.0,
        rrt_final_approach_vel_scale: float = 0.3,
        rrt_final_approach_acc_scale: float = 0.25,
        rrt_uniform_path_speed: bool = False,
        rrt_abort_on_drift_rad: float = 0.15,
        rrt_abort_on_drift_ticks: int = 8,
        rrt_drift_trigger: str = "lookback",
        shield_check_every_n_ticks: int = 1,
        debug_shield_force_trigger: bool = False,
        debug_shield_trace_anchor: bool = False,
        debug_rrt_drift_log: bool = False,
    ):
        # Bypass PreTrainedPolicy.__init__ — we proxy the inner policy's config
        nn.Module.__init__(self)
        self.config: PreTrainedConfig = inner_policy.config
        self.inner_policy = inner_policy
        self.inverse_postprocessor = inverse_postprocessor
        self.postprocessor = postprocessor  # normalized → raw joints
        self.inverse_preprocessor = inverse_preprocessor  # normalized obs.state → raw joints
        self.forward_flow_ratio = forward_flow_ratio
        self.blend_mode = BlendMode(blend_mode) if isinstance(blend_mode, str) else blend_mode
        self.guidance_blend_strategy = (
            GuidanceBlendStrategy(guidance_blend_strategy)
            if isinstance(guidance_blend_strategy, str)
            else guidance_blend_strategy
        )
        self._desired_q: np.ndarray | None = None  # raw joint-space IK seed [num_dofs]
        # Cached dummy-action template for `_lightweight_inner_call`'s
        # skip-forward-pass branch. Lazily populated on first use so we
        # only hit next(inner.parameters()) once (rather than every RRT
        # tick during long executions). Shape = (1, num_dofs+1),
        # dtype/device matches the inner policy's parameters.
        self._inner_action_template: Tensor | None = None
        # Most recent ACTUAL joint state, unnormalized from the latest observation.
        # Used as q_start for RRT planning so the plan starts where the robot is,
        # not where it was commanded to be (which can diverge when the policy
        # commands the robot into an obstacle — the env physics stops the real
        # robot at the surface while _desired_q keeps accumulating commanded poses,
        # producing demos that begin with a jarring "teleport to commanded pose"
        # before the recovery trajectory).
        self._latest_actual_q: np.ndarray | None = None
        # Ring buffer of the most recent ~N actual joint observations. When RRT
        # is triggered, q_start is taken from the oldest entry — that's the pose
        # the robot was at BEFORE the policy's current (presumably bad) action
        # chunk started commanding the robot toward a collision. Combined with
        # _maybe_teleport_to_q_start below, this makes the recorded RRT segment
        # begin at a clean pre-jump pose with no sim catch-up frames.
        # `_actual_q_history` is wrapper-owned (written every step from obs decode,
        # read by the RRT source via the wrapper back-ref to derive q_start).
        # Sized to fit the MAX of (min, max) lookback values so the source can
        # always reach as far back as the per-trigger random sample asks for.
        # When the lookback's steps_max is None, this reduces to the historical
        # behavior (sized for the single fixed lookback value).
        # In future_chunk mode no lookback rewind happens, but we still need a
        # tiny history (few samples) so _compute_recent_joint_velocity can
        # derive `start_vel` for the ruckig parametrization.
        if rrt_collision_detection == "future_chunk":
            # No lookback ever. Just enough history for a 2-3 sample
            # velocity estimate (for ruckig start_vel).
            _effective_max_lookback = 4
        else:
            # pre_jump_lookback OR hybrid: stall/no-progress triggers
            # still use lookback, so the deque must hold enough history
            # for the per-trigger random sample.
            _lb_cfg = rrt_pre_jump_lookback or PreJumpLookbackConfig()
            _effective_max_lookback = max(
                int(_lb_cfg.steps_min),
                int(_lb_cfg.steps_max) if _lb_cfg.steps_max is not None else 0,
            )
        self._actual_q_history: collections.deque[np.ndarray] = collections.deque(
            maxlen=max(1, _effective_max_lookback + 1)
        )
        # Diagnostic: set by the RRT source's _teleport_env_to_q_start to the
        # arm-joint pose the robot was teleported to (= the planned chunk
        # start). On the NEXT real obs decode we measure how far the robot
        # ACTUALLY landed from it and log the error, then clear. ~0 = clean
        # landing; large = the rewind pose was invalid (physics ejected the
        # robot) or the teleport didn't take over ZMQ — in which case the
        # open-loop chunk runs away from the real robot (the divergence we're
        # chasing). None = no teleport pending a landing check.
        self._pending_teleport_landing: np.ndarray | None = None
        # Frames-since-last-RRT-cycle-end counter. Used to cap the RRT
        # source's lookback so it never rewinds into a prior RRT cycle's
        # trajectory (which would teleport the env's robot to a config the
        # POLICY never actually drove through). Incremented in select_action
        # whenever RRT is IDLE (policy driving); reset to 0 the moment RRT
        # leaves IDLE (a new cycle started). Episode reset() also zeros it.
        # See rrt_source._do_plan() lookback path for the cap site.
        self._frames_since_last_rrt_end: int = 0
        # Cached previous-tick RRT mode for the off-by-one fix in select_action's
        # counter update (see comment there). Initialized to IDLE so the very
        # first scenario tick's "was_idle" check is True (no prior RRT cycle,
        # so we're trivially "still IDLE" from the implicit pre-scenario IDLE
        # state) and the counter increments from frame 1 onward.
        self._prev_rrt_mode: RRTMode = RRTMode.IDLE
        self._teleop_context: TeleopRecordingContext | None = None  # set by policy factory
        self._start_paused = start_paused
        self._run_event = threading.Event()
        if not start_paused:
            self._run_event.set()

        # The observation-driven path (pure teleop + DENOISE/INTERPOLATE blend)
        # is owned by ObservationTeleopGuidanceSource. The wrapper accesses
        # its state — `_guided_chunk`, `_chunk_step`, `_had_guidance_last_step`,
        # `_last_decoded_guidance_chunk` — via property shims further down.
        self._obs_teleop_source = ObservationTeleopGuidanceSource(self)
        # Method-triggered oracle-goal source for DAgger interventions. Builds
        # a linear-interpolation chunk from current q_start to the oracle's
        # q_goal_bias and plays it back verbatim. Triggered by external code
        # (lerobot-eval --intervention) via `self._oracle_goal_source.trigger()`.
        self._oracle_goal_source = OracleGoalGuidanceSource(self)

        # ── DOF resolution (num_dofs = arm joints, excluding gripper) ──
        # Priority: explicit > policy action_feature_names > action_dim - 1.
        # Auto-detection lets the same wrapper support planar_3joint (3 DOF),
        # UR5 small_engine (6 DOF), and future variants without per-robot
        # class subclasses. See `_resolve_num_dofs` docstring.
        resolved_num_dofs = self._resolve_num_dofs(num_dofs, inner_policy)
        self.num_dofs = resolved_num_dofs
        self._max_joint_delta = max_joint_delta
        self._prev_dq: np.ndarray | None = None  # previous joint velocity (raw, [num_dofs])
        self.skip_collision: bool = False  # set True for visualization (dataset guidance is known-safe)
        self.policy_guidance_representation = policy_guidance_representation
        self.n_anchor_steps = n_anchor_steps
        self._fps = fps

        # All RRT-mode state — planning lifecycle, chunk playback, plan thread —
        # is owned by the RRTGuidanceSource. The wrapper accesses RRT state via
        # this source; external callers (lerobot-eval --intervention, the GUI,
        # last_mile/helpers.py) access it via the back-compat `_rrt` property.
        # `auto_pause_on_rrt_finish` lives on the source; the wrapper exposes
        # it as a property shim further down.
        # Mode + per-mode nested config — used at runtime by the FK shield
        # (future_chunk mode) and threaded into the RRT source so it knows
        # which q_start policy to follow on each trigger.
        if rrt_collision_detection not in ("pre_jump_lookback", "future_chunk", "hybrid"):
            raise ValueError(
                "rrt_collision_detection must be 'pre_jump_lookback', "
                f"'future_chunk', or 'hybrid', got {rrt_collision_detection!r}"
            )
        self._collision_detection_mode = rrt_collision_detection
        # Drift-abort guard: cancel an RRT chunk when the robot stops tracking
        # it (drift > rad for N consecutive ticks). `_rrt_drift_streak` counts
        # consecutive high-drift playback ticks; reset on a fresh chunk and on
        # any below-threshold tick. rad <= 0 disables the guard.
        self._rrt_abort_on_drift_rad = float(rrt_abort_on_drift_rad)
        self._rrt_abort_on_drift_ticks = int(rrt_abort_on_drift_ticks)
        self._rrt_drift_streak = 0
        # What to do after a drift stall (see SharedAutonomyConfig.rrt_drift_trigger).
        self._rrt_drift_trigger = str(rrt_drift_trigger)
        # Set by the drift-abort to request a controller-side re-plan once the
        # cancelled chunk reaches IDLE. None = no re-plan ("discard" mode);
        # otherwise the no_lookback flag to pass to _trigger_source. The
        # InterventionController consumes + clears this in its tick().
        self._rrt_drift_replan_no_lookback: bool | None = None
        self._future_chunk_config = rrt_future_chunk or FutureChunkConfig()
        _lookback_cfg = rrt_pre_jump_lookback or PreJumpLookbackConfig()
        self._rrt_source = RRTGuidanceSource(
            self,
            collision_detection=rrt_collision_detection,
            pre_jump_lookback_steps_min=int(_lookback_cfg.steps_min),
            pre_jump_lookback_steps_max=(
                int(_lookback_cfg.steps_max) if _lookback_cfg.steps_max is not None else None
            ),
            teleport_to_q_start=bool(rrt_teleport_to_q_start),
            blocking_plan=bool(rrt_blocking_plan),
            auto_pause_on_finish=True,
            path_selection=rrt_path_selection,
            segment_at_sharp_corners=rrt_segment_at_sharp_corners,
            ik_goal_selection=rrt_ik_goal_selection,
            num_path_candidates_per_ik=rrt_num_path_candidates_per_ik,
            max_path_attempts_per_ik=rrt_max_path_attempts_per_ik,
            path_perturbation_scale=rrt_path_perturbation_scale,
            num_ik_candidates=rrt_num_ik_candidates,
            obstacle_clearance=rrt_obstacle_clearance,
            self_collision_clearance=rrt_self_collision_clearance,
            in_progress_obstacle_clearance=rrt_in_progress_obstacle_clearance,
            in_progress_self_collision_clearance=rrt_in_progress_self_collision_clearance,
            self_collision_skip_pairs=rrt_self_collision_skip_pairs,
            diagnostic_log_pairs=rrt_diagnostic_log_pairs,
            ik_skip_gripper_obstacle_pairs=rrt_ik_skip_gripper_obstacle_pairs,
            escape_clearance_factor=rrt_escape_clearance_factor,
            rewind_clearance_factor=rrt_rewind_clearance_factor,
            final_approach_dist=rrt_final_approach_dist,
            final_approach_vel_scale=rrt_final_approach_vel_scale,
            final_approach_acc_scale=rrt_final_approach_acc_scale,
            uniform_path_speed=rrt_uniform_path_speed,
        )

        # NOTE on `ratio` scope: forward_flow_ratio is applied ONLY to the
        # obs-teleop blending path (when `observation.policy_guidance_chunk`
        # arrives — typically from a keyboard teleop or live human source).
        # The RRT-EXECUTING path (`_rrt.chunk` playback during DAgger
        # intervention recording) and the oracle-goal path BYPASS this
        # ratio entirely and play the planned waypoints verbatim. So
        # `ratio=0.4` does NOT imply RRT recovery chunks are 40% policy /
        # 60% plan — those are 100% plan. Confusion this caused has
        # already burned us once.
        logger.info(
            f"SharedAutonomyPolicyWrapper: forward_flow_ratio={forward_flow_ratio} "
            f"(obs-teleop blending only — RRT chunks play verbatim), "
            f"robot={robot_name!r} ({'explicit' if robot_name else 'auto-detect from oracle'}), "
            f"num_dofs={self.num_dofs}, "
            f"rrt_collision_detection={self._collision_detection_mode}"
        )

        # Pybullet client is created up-front (needs a valid clientId for the
        # RRT source's constructor + property shims). The URDF itself is
        # loaded either now (explicit `robot_name`) or lazily on the first
        # `select_action` receipt of `oracle_env_config["robot"]` (when the
        # env is configured with `--env.include_oracle_info=true`). `_urdf_loaded`
        # gates the lazy path.
        self._pb_client = p.connect(p.GUI if show_slider else p.DIRECT)
        self._show_slider = show_slider  # for the launch_ratio_slider guard below
        self._robot_id: int | None = None
        self._ee_link: int | None = None
        self._num_pb_joints: int = 0
        self._num_movable_joints: int = 0
        self._loaded_robot_name: str | None = None
        self._urdf_loaded: bool = False
        self._obstacle_ids: list[int] = []
        # DEBUG-ONLY: shield force-trigger + anchor trace. See docstrings on
        # the corresponding SharedAutonomyConfig fields for semantics.
        self._debug_shield_force_trigger: bool = bool(debug_shield_force_trigger)
        self._debug_shield_trace_anchor: bool = bool(debug_shield_trace_anchor)
        self._debug_rrt_drift_log: bool = bool(debug_rrt_drift_log)
        # Shield rate-limiter: only run the expensive per-tick FK collision
        # sweep every N ticks. Counter increments each tick the shield gate
        # is otherwise open; check runs at counter % N == 0. Reset on
        # scenario reset.
        self._shield_check_every_n_ticks: int = max(1, int(shield_check_every_n_ticks))
        self._shield_check_tick_counter: int = 0
        # Shield-cooldown counter. Set to N by a failed shield trigger; ticks
        # down to 0 while the shield is suppressed. Prevents per-tick "shield
        # fires → RRT fails to plan (start in collision) → flush queue →
        # diffusion re-predicts fresh chunk with new noise → different
        # commanded position → visible shake" cascade. See select_action's
        # shield block.
        self._shield_cooldown_ticks: int = 0
        # How many ticks to suppress the shield after a failed plan. Short
        # enough (~0.5 s at 30 fps) that the arm gets fresh RRT attempts
        # quickly when stuck in collision (sliding on an obstacle) — the
        # planner has three escape methods and one of them may succeed on
        # a subsequent tick as the policy nudges the arm slightly. Long
        # enough that we don't spend ALL CPU on retrying escape chains
        # (each attempt is ~50-200 ms of pybullet iters). Was previously
        # 60 (2 s) which was too passive: episodes could burn hundreds of
        # collision ticks between attempts.
        self._SHIELD_COOLDOWN_ON_PLAN_FAIL: int = 15
        # Per-scenario latch: once we've logged "shield can't fire because
        # oracle has no task goal", stop firing the shield for the rest of
        # this scenario so we don't spam the log or spin uselessly. Reset
        # on new-oracle receipt (= new episode). See _shield_can_plan().
        self._shield_disabled_no_goal: bool = False
        # Identity of the oracle_env_config dict most recently applied.
        # Used to detect new episodes (new oracle received) and reset the
        # per-scenario shield-disabled latch.
        self._last_applied_oracle_id: int | None = None
        if robot_name is not None:
            self._load_urdf(robot_name)
            # Static-scene fallback (table + walls) is small_engine-specific
            # geometry the pre-oracle-era wrapper always loaded. When the
            # URDF is deferred to oracle we skip it — the oracle always
            # publishes the correct obstacles and would tear these down
            # immediately anyway, and they'd cause false collisions for
            # arbitrary non-small_engine robots. Explicit robot_name path
            # keeps the historical behavior for backwards compat.
            self._load_static_obstacles()

        if show_slider:
            from lerobot.policies.shared_autonomy_gui import launch_ratio_slider

            launch_ratio_slider(self)

    @staticmethod
    def _resolve_num_dofs(explicit: int | None, inner_policy: PreTrainedPolicy) -> int:
        """Resolve arm-joint DOF count (excludes gripper).

        Priority:
          1. `explicit` — the value passed to __init__ / read from
             SharedAutonomyConfig.num_dofs. Wins whenever non-None so users
             can pin an unusual layout.
          2. `inner_policy.config.action_feature_names` — the per-dim action
             names published by the policy config (e.g. ["joint_1",
             "joint_2", "joint_3", "gripper"]). We count non-gripper entries.
             This is the cleanest signal because it names each action dim.
          3. `action_dim - 1` — assumes exactly one gripper dim. Fallback
             for policies that don't publish action_feature_names.
        Raises when the policy provides no action shape at all (fatal —
        the wrapper can't proceed without knowing DOF count).
        """
        if explicit is not None:
            return int(explicit)
        pol_cfg = inner_policy.config
        names = getattr(pol_cfg, "action_feature_names", None)
        if names:
            n = sum(1 for name in names if "gripper" not in name.lower())
            if n > 0:
                logger.info(
                    "SA wrapper: auto-detected num_dofs=%d from "
                    "policy.action_feature_names=%s (excluded gripper dims).",
                    n,
                    names,
                )
                return n
        action_feat = (pol_cfg.output_features or {}).get(ACTION)
        if action_feat is not None and getattr(action_feat, "shape", None):
            action_dim = int(action_feat.shape[0])
            # Assume 1 gripper dim (matches every SplatSim robot to date).
            n = max(action_dim - 1, 1)
            logger.info(
                "SA wrapper: auto-detected num_dofs=%d from action_dim=%d "
                "(assumed 1 gripper dim; set --policy.shared_autonomy_config.num_dofs=N to override).",
                n,
                action_dim,
            )
            return n
        raise ValueError(
            "SA wrapper: could not resolve num_dofs — pass "
            "--policy.shared_autonomy_config.num_dofs=N or ensure the "
            "policy publishes output_features['action'].shape / "
            "action_feature_names."
        )

    def _load_urdf(self, robot_name: str) -> None:
        """Load the arm URDF into the wrapper's private pybullet client.
        Idempotent-ish: safe to call once at __init__ (explicit robot_name)
        or lazily on first oracle receipt. Populates:
          * self._robot_id, self._ee_link, self._num_pb_joints
          * self._num_movable_joints
          * self.lower_limits / self.upper_limits — auto-derived from URDF's
            getJointInfo when the URDF publishes limits; falls back to the
            class-level defaults otherwise. Overwriting the class attrs on
            self doesn't affect the class default (Python attribute lookup).
        Called a second time (different robot_name) logs a warning and no-ops
        — hot-reloading a different URDF into the same pybullet client is
        risky (obstacle IDs, joint index assumptions, RRT source state);
        the user should restart the wrapper if the robot changed mid-run.
        """
        if self._urdf_loaded:
            if robot_name != self._loaded_robot_name:
                logger.warning(
                    "SA wrapper: URDF already loaded as %r; ignoring lazy-load "
                    "request for %r (hot-reload not supported). Restart the "
                    "wrapper if the robot really changed.",
                    self._loaded_robot_name,
                    robot_name,
                )
            return
        robot_config = SplatObjectConfig(name="robot", splat_name=robot_name)
        urdf_path = resolve_splatsim_path(robot_config.urdf_path)
        ee_link_name = robot_config.wrist_camera_link_name
        # Match SplatSim's load_urdf flags for articulated objects so the
        # planner's collision shapes are byte-identical to the simulator's:
        #   - URDF_USE_IMPLICIT_CYLINDER: use analytical cylinders for any
        #     <geometry><cylinder/></geometry>. Without this pybullet falls
        #     back to a convex mesh approximation (a few mm smaller in
        #     radius), which is enough for the planner to declare a tight
        #     path collision-free that the simulator then registers as a graze.
        #   - URDF_USE_SELF_COLLISION: enables robot-vs-self getClosestPoints
        #     reports, which the planner's self-collision checks rely on.
        #   - URDF_USE_SELF_COLLISION_EXCLUDE_PARENT: ignore parent↔child
        #     joint pairs in those reports (otherwise every adjacent link
        #     would always look "in collision" because they touch at the
        #     joint).
        urdf_flags = (
            p.URDF_USE_IMPLICIT_CYLINDER
            | p.URDF_USE_SELF_COLLISION
            | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
        )
        # Base position comes from the same SplatObjectConfig that SplatSim
        # uses when it loads the robot on the sim side (see
        # PybulletRobotServerBase._load_urdf → splatsim_obj.config.base_position).
        # This MUST match the sim's value — otherwise the planner's private
        # pybullet client places the arm at a different world position than
        # the actual sim, so obstacles/goal-EE-poses (which are in world
        # frame) map to different arm configs. Result: planner reports
        # "collision-free" for a path the real robot crashes through (or
        # vice versa), and the planner's start_config check falsely fails.
        # Was previously hardcoded to [0, 0, -0.088] (small_engine's value);
        # planar_3joint uses [0, 0, 0] → 88 mm vertical drift.
        _base_pos = list(robot_config.base_position)
        logger.info(
            "SA wrapper: loading robot %r at base_position=%s (from SplatObjectConfig).",
            robot_name,
            _base_pos,
        )
        self._robot_id = p.loadURDF(
            urdf_path,
            useFixedBase=True,
            basePosition=_base_pos,
            flags=urdf_flags,
            physicsClientId=self._pb_client,
        )
        self._ee_link = self._find_ee_link(ee_link_name)
        self._num_pb_joints = p.getNumJoints(self._robot_id, physicsClientId=self._pb_client)
        self._num_movable_joints = sum(
            1
            for i in range(self._num_pb_joints)
            if p.getJointInfo(self._robot_id, i, physicsClientId=self._pb_client)[2] != p.JOINT_FIXED
        )
        self._derive_joint_limits_from_urdf()
        self._loaded_robot_name = robot_name
        self._urdf_loaded = True
        logger.info(
            "SA wrapper: loaded URDF %r (path=%s, ee_link=%s, pb_joints=%d, movable=%d, num_dofs=%d).",
            robot_name,
            urdf_path,
            ee_link_name,
            self._num_pb_joints,
            self._num_movable_joints,
            self.num_dofs,
        )
        # One-time AABB log per robot link at the rest pose. Matches the
        # diagnostic we use for obstacles in load_obstacles, so you can
        # eyeball that all gripper / arm links have non-degenerate
        # collision geometry after the URDF flag change.
        self._log_robot_link_aabbs()

    def _derive_joint_limits_from_urdf(self) -> None:
        """Populate instance-level `lower_limits` / `upper_limits` from the
        loaded URDF's per-joint info (fields [8]=lower, [9]=upper of
        getJointInfo). Only the first `num_dofs` MOVABLE (non-fixed) joints
        are consumed — matches _sync_joints' `1..1+num_dofs` convention.
        If any joint reports lower >= upper (unlimited in URDF), we fall
        back to the class-level default for the whole array — mixing per-
        joint URDF limits with class defaults would produce a nonsense
        combined range."""
        if self._robot_id is None:
            return
        limits: list[tuple[float, float]] = []
        movable_seen = 0
        for i in range(self._num_pb_joints):
            info = p.getJointInfo(self._robot_id, i, physicsClientId=self._pb_client)
            if info[2] == p.JOINT_FIXED:
                continue
            if movable_seen >= self.num_dofs:
                break
            lo, hi = float(info[8]), float(info[9])
            limits.append((lo, hi))
            movable_seen += 1
        if len(limits) == self.num_dofs and all(lo < hi for lo, hi in limits):
            self.lower_limits = [lo for lo, _ in limits]
            self.upper_limits = [hi for _, hi in limits]
            logger.info(
                "SA wrapper: derived joint limits from URDF: lower=%s, upper=%s",
                self.lower_limits,
                self.upper_limits,
            )
        else:
            logger.info(
                "SA wrapper: URDF joint limits missing/degenerate; keeping "
                "class-level defaults (lower=%s, upper=%s).",
                self.lower_limits,
                self.upper_limits,
            )

    # ---- pybullet FK + IK -------------------------------------------------- #

    def _find_ee_link(self, link_name: str) -> int:
        for i in range(p.getNumJoints(self._robot_id, physicsClientId=self._pb_client)):
            info = p.getJointInfo(self._robot_id, i, physicsClientId=self._pb_client)
            if info[12].decode("utf-8") == link_name:
                return i
        raise ValueError(f"Link '{link_name}' not found in URDF.")

    def _sync_joints(self, q: np.ndarray):
        for i in range(self.num_dofs):
            p.resetJointState(self._robot_id, i + 1, q[i], physicsClientId=self._pb_client)

    def _log_robot_link_aabbs(self) -> None:
        """Log every robot link's name and AABB at the rest pose.

        Mirrors the diagnostic we emit for obstacles. A degenerate AABB
        (zero-volume) on an arm or gripper link means that link has no
        collision geometry in the URDF and would be silently skipped by
        the planner's collision check — useful to eyeball after URDF
        changes.
        """
        try:
            n = p.getNumJoints(self._robot_id, physicsClientId=self._pb_client)
            base_aabb = p.getAABB(self._robot_id, linkIndex=-1, physicsClientId=self._pb_client)
            entries: list[str] = [f"base(-1): aabb={base_aabb}"]
            for link_i in range(n):
                info = p.getJointInfo(self._robot_id, link_i, physicsClientId=self._pb_client)
                link_name = info[12].decode("utf-8")
                aabb = p.getAABB(self._robot_id, linkIndex=link_i, physicsClientId=self._pb_client)
                entries.append(f"{link_name}({link_i}): aabb={aabb}")
            logger.info(
                "Robot link AABBs at rest pose (n_links=%d):\n  %s",
                n + 1,
                "\n  ".join(entries),
            )
        except p.error as e:
            logger.warning("Failed to log robot link AABBs: %s", e)

    def _get_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        state = p.getLinkState(
            self._robot_id,
            self._ee_link,
            computeForwardKinematics=True,
            physicsClientId=self._pb_client,
        )
        return np.array(state[4]), np.array(state[5])  # pos, quat (xyzw)

    def _compute_next_joints(self, q: np.ndarray, delta_pos: np.ndarray, delta_rot: np.ndarray) -> Tensor:
        q = q[: self.num_dofs]  # crop out the gripper

        self._sync_joints(q)

        pos, quat = self._get_ee_pose()
        r_current = Rotation.from_quat(quat)
        target_pos = pos + r_current.apply(delta_pos)
        r_delta = Rotation.from_euler("XYZ", delta_rot)
        target_quat = (r_current * r_delta).as_quat()

        rest = list(q)
        for i in range(self.num_dofs):
            if abs(q[i]) > 2.5:  # approaching ±π — bias IK away from singularity
                rest[i] = 0.0

        # Build null-space IK arrays. All must have length = num_movable_joints.
        # Arm DOFs use class-level limits; remaining movable joints (gripper) get
        # wide limits so they don't constrain the solution.
        n_movable = self._num_movable_joints
        n_extra = n_movable - self.num_dofs
        ll = self.lower_limits + [-np.pi] * n_extra
        ul = self.upper_limits + [np.pi] * n_extra
        jr = [u - lo for lo, u in zip(ll, ul, strict=True)]
        rp = rest + [0.0] * n_extra

        joint_poses = p.calculateInverseKinematics(
            self._robot_id,
            self._ee_link,
            target_pos,
            target_quat,
            lowerLimits=ll,
            upperLimits=ul,
            jointRanges=jr,
            restPoses=rp,
            jointDamping=[0.1] * n_movable,
            maxNumIterations=1000,
            residualThreshold=1e-6,
            physicsClientId=self._pb_client,
        )
        q_ik = np.array(joint_poses[: self.num_dofs])
        # if np.max(np.abs(q_ik - q)) > 0.15:
        #     return q  # reject singularity / far branch
        # delta_q = np.clip(q_ik - q, -self._max_joint_delta, self._max_joint_delta)
        delta_q = q_ik - q
        return q + delta_q

    def _load_static_obstacles(self) -> None:
        """Load hardcoded static scene geometry into the IK pybullet client.

        TODO(hardcoded): positions/sizes from UprightRobotSmallEngineNewPybulletRobotServer.
        Update here if the scene layout changes.
        """
        # Table: size=(1.5, 1.0, 0.05), center at (0, 0.3, -0.025)
        shape = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[0.75, 0.5, 0.025], physicsClientId=self._pb_client
        )
        self._obstacle_ids.append(
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=shape,
                basePosition=[0, 0.3, -0.025],
                physicsClientId=self._pb_client,
            )
        )
        # Wall: size=(3.0, 0.05, 1.5), center at (0, -0.225, 0.75)
        shape = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[1.5, 0.025, 0.75], physicsClientId=self._pb_client
        )
        self._obstacle_ids.append(
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=shape,
                basePosition=[0, -0.225, 0.75],
                physicsClientId=self._pb_client,
            )
        )

    # ---- RRT-to-Goal mode ------------------------------------------------- #

    # ── RRT lifecycle: thin shims over `self._rrt_source` ──────────────── #
    # All RRT planning/execution machinery lives on the RRTGuidanceSource.
    # These methods exist for back-compat with external callers that learned
    # the wrapper-level API before the source extraction (lerobot-eval --intervention,
    # last_mile/helpers.py, shared_autonomy_gui.py). New callers should prefer
    # `self._rrt_source.<...>` directly.

    def _shield_can_plan(self) -> bool:
        """Return True iff the RRT source has enough info to plan a recovery
        trajectory right now. Used to gate the future-chunk shield so it
        doesn't infinite-retrigger against a collision the planner can't
        resolve. Currently checks for a task goal
        (oracle_env_config.task.target_ee_pos/quat) — planar reacher envs
        that don't publish a task goal will return False, letting the
        wrapper log ONCE and disable the shield for the scenario. Extend
        this check if future planner failure modes need gating too."""
        from splatsim.utils.rrt_to_goal import extract_task_goal

        oracle = self._rrt_source.state.oracle_env_config
        if oracle is None:
            return False
        return extract_task_goal(oracle) is not None

    def _check_future_chunk_collision(self) -> tuple[bool, int | None, str | None, np.ndarray | None]:
        """Run the future-chunk predictive shield.

        Peeks at the inner policy's already-cached action chunk (no extra
        forward pass), denormalizes each step in policy-action space, and
        FK-checks the cumulative future joint trajectory against the
        wrapper's pybullet client (which holds the same obstacles RRT uses).

        Returns ``(any_collides, first_step_idx, kind)`` mirroring
        ``rrt_to_goal.check_chunk_collision``. When ``any_collides`` is
        True, the caller should preempt the policy and trigger RRT from
        the current state (no rewind / no teleport).

        Pre-conditions: ``_latest_actual_q`` has been refreshed THIS tick
        AND the pybullet client has been synced via ``_sync_joints``.
        Reads action-format intent from the inner policy's config
        (``use_relative_actions`` / ``relative_exclude_joints``).
        """
        # Lazy import — keep optional dependency surface contained.
        from lerobot.policies.rrt_to_goal import check_chunk_collision

        # Peek without consuming. Returns None if no chunk cached yet
        # (e.g., very first tick before select_action populated the queue).
        chunk = self.inner_policy.get_pending_action_chunk()
        if chunk is None or chunk.shape[0] == 0:
            return False, None, None, None

        # Apply horizon_frames cap if configured.
        horizon = self._future_chunk_config.horizon_frames
        if horizon is not None and chunk.shape[0] > horizon:
            chunk = chunk[:horizon]

        # Determine action format from inner_policy.config. Diffusion / Pi0
        # / Pi0.5 all expose `use_relative_actions: bool` on their config.
        # Default to abs if the attribute isn't present (e.g., custom policies).
        inner_cfg = self.inner_policy.config
        action_format = "rel" if getattr(inner_cfg, "use_relative_actions", False) else "abs"

        # Denormalize each chunk step through the postprocessor with the
        # AbsoluteActionsProcessorStep state set to zero. This gives us the
        # unnormalized rel-deltas (or abs targets) without the per-step
        # last_state addition that would otherwise turn the chunk into N
        # independent "from current state" actions rather than the
        # cumulative trajectory we need.
        chunk_raw = self._denormalize_chunk_to_raw(chunk)
        if chunk_raw is None:
            # Denormalization failed (e.g., postprocessor not configured for
            # this codepath); skip the shield this tick rather than crashing.
            return False, None, None, None

        # Slice to the DOF arm dims — drop gripper. The wrapper conventionally
        # uses joint indices 1..1+num_dofs for the planning pybullet client,
        # so chunk_raw[:, :num_dofs] is the arm-only future trajectory.
        chunk_dof = np.asarray(chunk_raw[:, : self.num_dofs], dtype=np.float64)

        # IMPORTANT: the anchor we add to each chunk action must MATCH what
        # inference adds when popping that action — otherwise we predict a
        # different absolute position from where the robot will actually go.
        # The relative_action_processor only refreshes its ``_last_state``
        # when the policy's chunk queue is empty (i.e., on chunk regen);
        # during the 8 ticks the chunk plays out, ``_last_state`` stays
        # FIXED at the chunk-gen-time obs state. So inference does
        # ``action[k] = chunk[k] + _last_state_chunk_gen`` for all k,
        # NOT ``chunk[k] + obs_state_at_tick_k``.
        #
        # Read that anchor here, slice to the arm DOF. If it's not set
        # yet (very first preprocessor call) fall back to the wrapper's
        # current actual_q — at that single tick, they're the same value
        # anyway (chunk WAS just generated).
        q_current_dof = None
        for _step in self.postprocessor.steps:
            if isinstance(_step, AbsoluteActionsProcessorStep):
                # Explicit cast — Pyright can't narrow dataclass attributes
                # through the ProcessorStep base class even with isinstance,
                # so spell out the concrete type for the attribute accesses.
                abs_step = cast(AbsoluteActionsProcessorStep, _step)
                if abs_step.enabled and abs_step.relative_step is not None:
                    rel_step = abs_step.relative_step
                    if rel_step._last_state is not None:
                        anchor = rel_step._last_state.detach().cpu().numpy().reshape(-1)
                        q_current_dof = np.asarray(anchor, dtype=np.float64)[: self.num_dofs]
                break
        if q_current_dof is None:
            # No cached anchor (abs-mode policy or very-first-tick edge case)
            # → use the wrapper's actual_q. For abs-mode, q_current_dof is
            # unused anyway (action_format='abs' → future_qs = chunk_arr).
            q_current = self._latest_actual_q
            if q_current is None:
                return False, None, None, None
            q_current_dof = np.asarray(q_current, dtype=np.float64).reshape(-1)[: self.num_dofs]

        # Inherit clearance / skip-pair config from the planner that the
        # source already constructed (so the shield's contract matches RRT's).
        planner = self._rrt_source.state.planner
        self_skip_pairs = None
        obstacle_skip_pairs: set[tuple[int, int]] | None = None
        ob_clear = self._future_chunk_config.obstacle_clearance
        self_clear = self._future_chunk_config.self_collision_clearance
        if planner is not None:
            if ob_clear is None:
                ob_clear = planner._collision_kwargs.get("obstacle_clearance")
            if self_clear is None:
                self_clear = planner._collision_kwargs.get("self_collision_clearance")
            self_skip_pairs = planner._collision_kwargs.get("self_collision_skip_pairs")
            # Forward env-declared (robot_link, obstacle_body_id) skips
            # (e.g. base_link ⟷ table) so the shield uses the same skip
            # contract as RRT planning / is_in_collision_at.
            obstacle_skip_pairs = planner._skip_pairs if planner._skip_pairs else None

        # Joint indices the planner uses for the arm DOFs in the wrapper's
        # pybullet client. The wrapper convention is 1..1+num_dofs (see
        # _sync_joints).
        joint_indices = list(range(1, 1 + self.num_dofs))

        # Pass the env's actual gripper config so the shield's FK projection
        # checks the geometry the env ACTUALLY has, not the URDF default
        # OPEN-gripper that `check_links_in_collision(q=...)` would otherwise
        # force via its internal `set_robot_joint_positions → open_gripper`.
        # Without this, for grasp tasks where the policy progressively closes
        # the gripper, the shield false-fires `obstacle_collision` whenever
        # the policy's predicted chunk takes the EE near the goal — the
        # OPEN fingers overlap the goal object even though the CLOSED fingers
        # in the actual env wouldn't. Mirrors the gripper-snap fix in
        # `RRTToGoalPlanner.is_q_in_collision` (controller-side check).
        actual_gripper_q: float | None = None
        if self._latest_actual_q is not None and self._latest_actual_q.size > self.num_dofs:
            actual_gripper_q = float(self._latest_actual_q[self.num_dofs])

        collides, step, kind = check_chunk_collision(
            pb_client=self._pb_client,
            robot_id=self._robot_id,
            joint_indices=joint_indices,
            q_current=q_current_dof,
            chunk_dof_actions=chunk_dof,
            action_format=action_format,
            obstacle_ids=self._obstacle_ids,
            obstacle_clearance=ob_clear,
            self_collision_clearance=self_clear,
            self_collision_skip_pairs=self_skip_pairs,
            skip_pairs=obstacle_skip_pairs,
            actual_gripper_q=actual_gripper_q,
        )
        # DIAGNOSTIC (why-doesn't-shield-fire-earlier): for the NO-COLLISION
        # case, probe chunk-step 0 (immediate command) and chunk-step -1
        # (deepest lookahead the shield can see) for their closest-pair
        # distance. Log ONLY when either is inside a 5 cm proximity window
        # so we see the approach ramp up without spamming steady-state ticks.
        # This reveals whether the policy's chunk projects into the obstacle
        # BEFORE the shield's threshold hits (i.e., shield had a chance to
        # fire but chose not to under the current threshold) vs. the chunk
        # never reaches the obstacle geometrically until step 0 (i.e., the
        # policy jumps commanded position across a chunk boundary, giving
        # the shield no runway). Falls back silently when planner isn't
        # ready (mirror-loaded, no obstacles) or describe returns None.
        planner = self._rrt_source.state.planner
        if not collides and planner is not None and chunk_dof.shape[0] > 0:
            _steps_to_probe = [0]
            if chunk_dof.shape[0] > 1:
                _steps_to_probe.append(int(chunk_dof.shape[0]) - 1)
            _probes: list[tuple[int, float, str, str]] = []
            _min_dist = float("inf")
            for _k in _steps_to_probe:
                _base_q = q_current_dof + chunk_dof[_k] if action_format == "rel" else chunk_dof[_k]
                if actual_gripper_q is not None:
                    _probe_q = np.concatenate(
                        [
                            np.asarray(_base_q, dtype=np.float64),
                            np.asarray([actual_gripper_q], dtype=np.float64),
                        ]
                    )
                else:
                    _probe_q = np.asarray(_base_q, dtype=np.float64)
                _info = planner.describe_collision_at(
                    _probe_q,
                    obstacle_clearance=ob_clear,
                    self_collision_clearance=self_clear,
                )
                if _info is not None:
                    _d = float(_info.get("distance_m", float("inf")))
                    _probes.append((_k, _d, _info.get("link_a_name", "?"), _info.get("link_b_name", "?")))
                    if _d < _min_dist:
                        _min_dist = _d
        # Re-derive the offending future joint config so the caller can
        # probe `planner.describe_collision_at(future_q)` to identify the
        # violating link pair. Mirrors the math `check_chunk_collision`
        # uses internally — `rel` = anchor + chunk[k], `abs` = chunk[k].
        # Append actual gripper config (when known) so describe's gripper
        # snap matches the shield's check geometry.
        offending_q: np.ndarray | None = None
        if collides and step is not None:
            base_q = q_current_dof + chunk_dof[step] if action_format == "rel" else chunk_dof[step]
            if actual_gripper_q is not None:
                offending_q = np.concatenate(
                    [np.asarray(base_q, dtype=np.float64), np.asarray([actual_gripper_q], dtype=np.float64)]
                )
            else:
                offending_q = np.asarray(base_q, dtype=np.float64)
        return collides, step, kind, offending_q

    def _denormalize_chunk_to_raw(self, chunk: Tensor) -> np.ndarray | None:
        """Denormalize a queued action chunk to raw policy-space actions.

        The inner policy's action queue stores NORMALIZED actions of shape
        ``(n_steps, B=1, action_dim)``. We need them in RAW units (radians
        for joint dims) AND we need rel-format chunks to remain as rel
        deltas (not as per-step "where to go from current state").

        Historical implementation temporarily zeroed the SHARED
        ``rel_step._last_state`` (used by the outer postprocessor too) so
        the AbsoluteActionsProcessorStep would add 0 during a normal
        `self.postprocessor(chunk[k])` pass, then restored via try/finally.
        That mutation, even bracketed, made the shared anchor the wrong
        value for the duration of the shield loop — visible per-tick jitter
        in the returned action when the outer postprocessor happened to
        read the shared anchor at the wrong instant. Fixed here by running
        only the pre-abs-step tail of the pipeline (unnormalize → device)
        directly on each chunk step, then returning the rel-delta values
        without ever touching `rel_step._last_state`.

        Returns ``(n_steps, action_dim)`` numpy array, or None on failure
        (caller should skip the shield rather than guess at format).
        """
        # Find the AbsoluteActionsProcessorStep so we can (a) confirm this
        # is a rel-format pipeline (only then does zero-anchor apply) and
        # (b) run every OTHER step in the pipeline manually — skipping the
        # abs step to keep chunk values as pre-abs (rel-delta) numbers.
        abs_step_idx = None
        for idx, _step in enumerate(self.postprocessor.steps):
            if isinstance(_step, AbsoluteActionsProcessorStep):
                abs_step_idx = idx
                break
        if abs_step_idx is None:
            # No abs step in the pipeline → postprocessor output IS the
            # denormalized abs-format action. Run the full pipeline per-step
            # and stack. No shared-state mutation risk.
            out_rows: list[np.ndarray] = []
            for k in range(chunk.shape[0]):
                row = self.postprocessor(chunk[k]).detach().cpu().numpy().reshape(-1)
                out_rows.append(row)
            return np.stack(out_rows, axis=0) if out_rows else None
        abs_step = cast(AbsoluteActionsProcessorStep, self.postprocessor.steps[abs_step_idx])
        # DEBUG: snapshot the shared anchor before/after this call so we can
        # spot mutations that leak past the intended scope (see
        # `debug_shield_trace_anchor` docstring on SharedAutonomyConfig).
        _trace_anchor = None
        if self._debug_shield_trace_anchor and abs_step.relative_step is not None:
            _pre = abs_step.relative_step._last_state
            _trace_anchor = (
                _pre.detach().clone() if _pre is not None else None,
                id(_pre) if _pre is not None else None,
            )
        if not abs_step.enabled or abs_step.relative_step is None:
            # Abs step present but disabled → same as no-abs-step case.
            out_rows = []
            for k in range(chunk.shape[0]):
                row = self.postprocessor(chunk[k]).detach().cpu().numpy().reshape(-1)
                out_rows.append(row)
            return np.stack(out_rows, axis=0) if out_rows else None

        # Rel-format pipeline: run every step EXCEPT the abs step on each
        # chunk element. Bypass the pipeline's own __call__ (which wraps
        # input in a transition + iterates all steps) by manually building
        # the transition, running the non-abs steps in order, then
        # unwrapping the action tensor.
        from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action

        out_rows = []
        for k in range(chunk.shape[0]):
            transition = policy_action_to_transition(chunk[k])
            for idx, _step in enumerate(self.postprocessor.steps):
                if idx == abs_step_idx:
                    continue
                transition = _step(transition)
            action_t = transition_to_policy_action(transition)
            out_rows.append(action_t.detach().cpu().numpy().reshape(-1))
        result = np.stack(out_rows, axis=0) if out_rows else None
        # DEBUG trace post-condition: the shared anchor MUST be byte-identical
        # after this call (identity + value). Any drift means a downstream
        # step secretly mutated `rel_step._last_state` and the outer decoder
        # will now produce wrong absolute actions. Log LOUDLY when this fires.
        if _trace_anchor is not None and abs_step.relative_step is not None:
            _pre_val, _pre_id = _trace_anchor
            _post = abs_step.relative_step._last_state
            _post_id = id(_post) if _post is not None else None
            _drift = None
            if _pre_val is not None and _post is not None:
                _drift = float((_post - _pre_val).abs().max().item())
            if _pre_id != _post_id or (_drift is not None and _drift > 1e-9):
                logger.error(
                    "SA wrapper _denormalize_chunk_to_raw MUTATED shared anchor: "
                    "id changed %s→%s, max-abs drift=%s. This corrupts the outer "
                    "postprocessor decode and causes per-tick oscillation.",
                    _pre_id,
                    _post_id,
                    f"{_drift:.6e}" if _drift is not None else "n/a",
                )
        return result

    @property
    def _rrt(self):
        """Back-compat view of the RRT source's runtime state.

        Returns a thin proxy so `wrapper._rrt.mode`, `wrapper._rrt.target_steps`,
        `wrapper._rrt.planner`, etc. all read/write the underlying
        `RRTGuidanceSource.state` (which is the same `RRTRuntimeState` dataclass
        that used to live directly on the wrapper). See
        `lerobot.policies.guidance.views._RRTBackCompatView` for the proxy
        implementation.
        """
        return _RRTBackCompatView(self._rrt_source)

    @property
    def auto_pause_on_rrt_finish(self) -> bool:
        """Whether to pause the wrapper when RRT reaches its goal naturally.

        Mirrored on the source. External code (lerobot-eval --intervention,
        last_mile/helpers.py) sets this on the wrapper; the property
        forwards the write to the source.
        """
        return self._rrt_source.auto_pause_on_finish

    @auto_pause_on_rrt_finish.setter
    def auto_pause_on_rrt_finish(self, value: bool) -> None:
        self._rrt_source.auto_pause_on_finish = bool(value)

    def set_env_for_teleport(self, env: object) -> None:
        """Register the gym env handle used to teleport the sim's joint state
        before RRT execution begins. Should be the un-vectorized,
        un-wrapped env (or a single-env sub-handle) that exposes
        ``robot_server.teleport_joint_state(splatsim_robot, joint_state)``.

        Called once by the intervention recorder right after env creation.
        """
        self._rrt_source.set_env_for_teleport(env)

    # ── Obs-driven source: back-compat property shims for migrated state ─ #
    # External callers don't read these directly (audit), but inline wrapper
    # code (e.g. `select_action`'s pre-flush block, `_cancel_rrt`) still
    # touches them. Properties forward to the source so the existing code
    # keeps working transparently.

    @property
    def _guided_chunk(self):
        return self._obs_teleop_source._guided_chunk

    @_guided_chunk.setter
    def _guided_chunk(self, value) -> None:
        self._obs_teleop_source._guided_chunk = value

    @property
    def _chunk_step(self) -> int:
        return self._obs_teleop_source._chunk_step

    @_chunk_step.setter
    def _chunk_step(self, value: int) -> None:
        self._obs_teleop_source._chunk_step = value

    @property
    def _had_guidance_last_step(self) -> bool:
        return self._obs_teleop_source._had_guidance_last_step

    @_had_guidance_last_step.setter
    def _had_guidance_last_step(self, value: bool) -> None:
        self._obs_teleop_source._had_guidance_last_step = bool(value)

    @property
    def _last_decoded_guidance_chunk(self):
        return self._obs_teleop_source._last_decoded_guidance_chunk

    @_last_decoded_guidance_chunk.setter
    def _last_decoded_guidance_chunk(self, value) -> None:
        self._obs_teleop_source._last_decoded_guidance_chunk = value

    def is_rrt_active(self) -> bool:
        """True while RRT is planning or executing."""
        return self._rrt_source.is_active()

    def disable_recording(self) -> None:
        """Turn off all recording-related behavior.

        Clears two pieces of state:
          * ``_teleop_context``: detaches the singleton
            ``TeleopRecordingContext``, so per-step ``select_action``
            bookkeeping (frame_source, has_guidance, etc.) becomes a no-op.
          * the RRT source's teleport-to-q_start flag: disables the
            "teleport the sim robot to the RRT plan's q_start before
            execution" optimization. That feature exists to make the
            recorded RRT trajectory start pristine (no catch-up frames
            from physics interpolation). When we're not recording, the
            catch-up frames don't matter; and the teleport requires a
            separately-set env handle which the non-recording eval path
            doesn't supply, so leaving the flag on just produces a
            misleading "Skipping teleport — recorded intervention will
            start with catch-up frames" warning.

        ``_wrap_with_shared_autonomy`` always attaches a
        ``TeleopRecordingContext`` and leaves the teleport flag at its
        default ``True``, because the primary caller is
        ``lerobot-eval --intervention``. External callers using SA for help
        rather than data collection (e.g. the last-mile RRT helper) should
        call this method after wrapping.
        """
        self._teleop_context = None
        self._rrt_source.set_teleport_enabled(False)

    def trigger_rrt_to_goal(self) -> None:
        """Toggle: start RRT-to-goal if idle, cancel if planning/executing.

        Blocks when the source's `blocking_plan` is True (the default and the
        eval/recording path). See `RRTGuidanceSource.trigger` for details.
        """
        self._rrt_source.trigger()

    def _maintain_inner_obs_history(self, batch: dict[str, Tensor]) -> None:
        """Push the current batch into the inner policy's obs-history queue
        WITHOUT running the (expensive) forward pass. No-op for policies that
        don't maintain a `_queues[OBS_*]` deque (PI0 / PI0.5 read obs directly
        from the batch on each predict_action_chunk call, so there's no history
        to keep current).

        Mirrors the pre-forward-pass portion of DiffusionPolicy.select_action:
        image stacking + populate_queues. Skips the DDPM denoising loop. Used
        by `_lightweight_inner_call` during RRT / oracle-goal execution so
        the inner policy's obs history stays current — otherwise the fresh
        chunk generated at RRT-end would fill its obs queue by copying the
        FIRST obs it saw post-RRT, losing the recent history.
        """
        inner = self.inner_policy
        queues = getattr(inner, "_queues", None)
        if not isinstance(queues, dict):
            return  # PI0 / PI0.5: no obs queue to update
        _batch = batch
        image_features = getattr(getattr(inner, "config", None), "image_features", None)
        if image_features:
            from lerobot.utils.constants import OBS_IMAGES

            _batch = dict(batch)
            _batch[OBS_IMAGES] = torch.stack([_batch[k] for k in image_features], dim=-4)
        from lerobot.policies.utils import populate_queues

        populate_queues(queues, _batch)

    def _lightweight_inner_call(self, batch: dict[str, Tensor]) -> Tensor:
        """Fast substitute for `inner_policy.select_action(batch)` used during
        RRT / oracle-goal EXECUTING mode, when the inner action would be
        DISCARDED (intervention branch overrides it).

        Two cases:
          * Inner has a cached chunk (queue non-empty) → delegate to the full
            select_action: it's cheap (populate_queues + popleft, no forward
            pass) AND we get a real action tensor for downstream dtype/device
            references. Same cost as before this optimization.
          * Inner's chunk is empty → the full select_action would trigger a
            full DDPM denoise / Pi0 flow-matching forward pass here (~100-500 ms
            depending on model + device). We SKIP that. Manually push obs into
            the inner's history queue so the eventual fresh chunk (generated
            at intervention-end when the wrapper's `_flush_inner_action_queue`
            fires) has an up-to-date history to condition on. Return a
            zero-filled dummy tensor with the correct dtype / device / shape
            for downstream `inner_action.dtype / .device / .shape[0]` uses
            (get_hold_action, _normalize_policy_guidance_action's device
            match, oracle/obs_teleop next_action ctx fields).

        Net effect: RRT execution never triggers an inner forward pass. For
        n_action_steps=32 and a 100-tick RRT, saves ~3 forward passes = a few
        hundred ms of GPU time. For n_action_steps=1 it's dramatic.
        """
        inner = self.inner_policy
        # Diffusion caches in inner._queues[ACTION]; PI0 / PI0.5 caches in
        # inner._action_queue. Either non-empty ⇒ next select_action call
        # will just pop, not run the model — cheap to run normally.
        diffusion_cached = False
        _q = getattr(inner, "_queues", None)
        if isinstance(_q, dict):
            from lerobot.utils.constants import ACTION as _A

            _dq = _q.get(_A)
            diffusion_cached = _dq is not None and len(_dq) > 0
        _pi_q = getattr(inner, "_action_queue", None)
        pi_cached = _pi_q is not None and len(_pi_q) > 0

        if diffusion_cached or pi_cached:
            _act = inner.select_action(batch)
            # Defensive: some policy adapters (Peft-wrapped, custom) can
            # under-specify the select_action contract and return None on the
            # non-forward-pass path. Downstream code (RRT playback, hold-action
            # emission) needs a real Tensor for dtype/device — fall through to
            # the manual dummy path when the delegated call disappoints us.
            if _act is not None:
                return _act

        # Fresh forward pass would fire (or the delegate returned None) — skip it.
        self._maintain_inner_obs_history(batch)
        # Cache dtype/device so long RRT executions don't hit
        # next(inner.parameters()) every tick. Also guarantees a stable
        # (dtype, device) for the dummy tensor even if inner_policy briefly
        # loses its parameter iter (mid-move to a new device, etc.).
        template = self._inner_action_template
        if template is None:
            _param = next(inner.parameters(), None)
            _dtype = _param.dtype if _param is not None else torch.float32
            _device = _param.device if _param is not None else torch.device("cpu")
            # Full action dim = arm DOFs + gripper — matches what the model
            # would have returned.
            template = torch.zeros((1, self.num_dofs + 1), dtype=_dtype, device=_device)
            self._inner_action_template = template
        return template.clone()

    def _flush_inner_action_queue(self) -> None:
        """Drop the inner policy's cached actions without resetting its obs queue.

        Both PI0.5 and Diffusion buffer a chunk's worth of actions and pop one
        per select_action call. After RRT execution that buffer is stale (the
        robot has been driven by the planner). Clearing only the action queue
        forces predict_action_chunk to fire again on the next call — but the
        observation history (n_obs_steps) is preserved, which matters for
        policies whose obs window is longer than 1 step.
        """
        cleared_queue = False
        inner = self.inner_policy
        # PI0.5
        action_q = getattr(inner, "_action_queue", None)
        if action_q is not None and hasattr(action_q, "clear"):
            action_q.clear()
            cleared_queue = True
        # Diffusion (and any other policy following the shared `_queues[ACTION]` pattern)
        queues = getattr(inner, "_queues", None)
        if isinstance(queues, dict):
            from lerobot.utils.constants import ACTION

            q = queues.get(ACTION)
            if q is not None and hasattr(q, "clear"):
                q.clear()
                cleared_queue = True

        if not cleared_queue:
            raise RuntimeError(
                "Failed to flush inner policy's action queue: no known queue attribute found. "
            )

    def _cancel_rrt(self) -> None:
        """Cancel RRT and clear obs-driven cached state. See `_cancel_intervention`."""
        self._cancel_intervention(self._rrt_source)

    def _cancel_oracle_goal(self) -> None:
        """Cancel the OracleGoal sequence and clear obs-driven cached state.

        Same cleanup as `_cancel_rrt` — the differences between the two sources
        are in chunk construction (planner vs interpolator), not in cancellation.
        """
        self._cancel_intervention(self._oracle_goal_source)

    def _cancel_intervention(self, source) -> None:
        """Source-agnostic cancel: clear the source's chunk, flush stale inner
        policy actions, and reset the obs-teleop blend buffer so the next call
        generates fresh actions from the post-cancel pose.

        forward_flow_ratio is intentionally not parked (no source's execution
        branch reads it), so there's nothing to restore.
        """
        source.cancel()
        self._flush_inner_action_queue()
        self._obs_teleop_source.cancel()

    def _finish_rrt(self) -> None:
        """Goal reached: cancel + clean caches, then auto-pause unless disabled."""
        self._cancel_rrt()
        if self.auto_pause_on_rrt_finish:
            self._run_event.clear()
            logger.info("RRT goal reached; auto-paused. Resume to continue.")
        else:
            logger.info("RRT goal reached; auto-pause disabled (running headless).")

    def _log_rrt_drift_summary(self, reason: str) -> None:
        """One-line per-chunk drift rollup at chunk end. Pairs with the
        per-tick `RRT drift @ step` logs from the playback loop: those show the
        shape, this gives the verdict. `drift@step1` is the teleport landing
        error (robot vs chunk start); `drift_max` is the worst over the chunk.
        Both ~0 ⇒ clean execution; large step1 ⇒ teleport didn't land / scene
        ejection; small step1 but large max ⇒ open-loop runaway (clean start,
        robot fell behind). Called from the wrapper's RRT playback exits."""
        rrt = self._rrt
        d1 = getattr(self, "_rrt_drift_at_step1", None)
        dmax = getattr(self, "_rrt_drift_max", 0.0)
        n = len(rrt.chunk) if rrt.chunk is not None else 0
        logger.info(
            "RRT chunk drift summary (%s): steps=%d/%d  drift@step1=%s rad  drift_max=%.4f rad",
            reason,
            rrt.step,
            n,
            f"{d1:.4f}" if d1 is not None else "n/a",
            dmax,
        )

    def _project_delta_for_collision(
        self,
        q: np.ndarray,
        delta_pos: np.ndarray,
        delta_rot: np.ndarray,
        skip_collision: bool = False,
    ) -> np.ndarray:
        """Compute IK for delta, projecting delta_pos onto obstacle surfaces if needed.

        1. Try full delta via IK.
        2. If the result collides, project delta_pos to remove components pointing
           into each obstacle (standard surface-projection / constraint stacking).
        3. Retry IK with the projected delta.
        4. If still colliding, hold in place (return q).
        """
        q_new = self._compute_next_joints(q, delta_pos, delta_rot)

        if skip_collision:
            return q_new

        # Check collision at proposed joint config
        self._sync_joints(q_new[: self.num_dofs])
        p.performCollisionDetection(physicsClientId=self._pb_client)

        contacts = []
        for obs_id in self._obstacle_ids:
            contacts.extend(p.getContactPoints(self._robot_id, obs_id, physicsClientId=self._pb_client) or [])

        if not contacts:
            return q_new

        # Project delta_pos: for each contact normal pointing away from the obstacle,
        # remove the component of delta_pos that opposes it (i.e., moves into the obstacle).
        projected_pos = delta_pos.copy()
        for contact in contacts:
            normal = np.array(contact[7])  # contactNormalOnB: from obstacle toward robot
            dot = float(np.dot(projected_pos, normal))
            if dot < 0:  # moving into the obstacle
                projected_pos = projected_pos - dot * normal

        q_projected = self._compute_next_joints(q, projected_pos, delta_rot)

        # Safety check: if still colliding after projection, hold in place
        self._sync_joints(q_projected[: self.num_dofs])
        p.performCollisionDetection(physicsClientId=self._pb_client)
        for obs_id in self._obstacle_ids:
            if p.getContactPoints(self._robot_id, obs_id, physicsClientId=self._pb_client):
                return q[: self.num_dofs]

        return q_projected

    # ---- motion limits ----------------------------------------------------- #

    def _apply_velocity_limit(
        self, q_proposed: np.ndarray, q_prev: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Uniform velocity scaling: if any joint exceeds v_max, scale the whole delta vector
        down proportionally. This preserves the EE direction (unlike per-joint clipping).

        Only applied to the joint dims (first num_dofs); gripper passes through unchanged.

        Returns (q_actual, dq_actual) where dq_actual should be stored as _prev_dq for the
        next step (needed if you later add acceleration/jerk limits — see below).
        """
        n = self.num_dofs
        v_max = self._max_joint_delta  # 0.5 / self._fps  # max position delta per step (rad)

        dq = q_proposed[:n] - q_prev[:n]
        v_mag = np.max(np.abs(dq))
        if v_mag > v_max:
            dq = dq * (v_max / v_mag)  # scale whole vector, not per-joint clip

        # Use q_proposed as base so gripper (and any extra dims) are always present,
        # regardless of whether q_prev was set with or without gripper.
        q_actual = q_proposed.copy()
        q_actual[:n] = q_prev[:n] + dq

        # To also add acceleration and jerk limits, track dq_prev and ddq_prev across steps
        # and apply constraints in reverse order (jerk → accel → vel) before the vel limit:
        #
        # a_max = 1.0 / self._fps
        # j_max = 10.0 / self._fps
        #
        # d2q = dq - dq_prev                      # proposed acceleration
        # d3q = d2q - ddq_prev                    # proposed jerk
        #
        # # 1) Jerk limit: scale jerk vector uniformly
        # j_mag = np.max(np.abs(d3q))
        # if j_mag > j_max:
        #     d3q = d3q * (j_max / j_mag)
        # d2q = ddq_prev + d3q                    # accel after jerk constraint
        #
        # # 2) Acceleration limit: scale (jerk-constrained) accel vector uniformly
        # a_mag = np.max(np.abs(d2q))
        # if a_mag > a_max:
        #     d2q = d2q * (a_max / a_mag)
        # dq = dq_prev + d2q                      # velocity after accel constraint
        #
        # # 3) Velocity limit (same as above, applied to the now-cascaded dq)
        #
        # Also update _prev_ddq = dq_actual - dq_prev alongside _prev_dq each step.

        # The gripper passed through

        return q_actual, dq

    # ---- policy helpers ---------------------------------------------------- #

    def _normalize_policy_guidance_action(self, policy_guidance_action: Tensor) -> Tensor:
        """Normalize raw policy guidance action to policy's internal space.

        Zero-fills NaN/Inf dimensions (e.g., gripper always closed in training data
        where normalization stats have zero variance).

        When the policy uses relative actions, the postprocessor's AbsoluteActionsProcessorStep
        will add the current state to produce absolute joint positions. To make the round-trip
        correct (normalize → postprocess → absolute guidance), the raw absolute guidance must
        first be converted to relative (guidance - state) before normalizing, matching how
        training actions were preprocessed.
        """
        policy_guidance_action = policy_guidance_action.clone()

        # If relative actions are enabled, convert absolute guidance → relative so that
        # (a) normalization uses the correct relative-action stats, and
        # (b) the postprocessor's AbsoluteActionsProcessorStep adds state back cleanly.
        for _step in self.postprocessor.steps:
            if isinstance(_step, AbsoluteActionsProcessorStep):
                if _step.enabled and _step.relative_step is not None:
                    state = _step.relative_step._last_state
                    if state is not None:
                        mask = _step.relative_step._build_mask(policy_guidance_action.shape[-1])
                        policy_guidance_action = to_relative_actions(policy_guidance_action, state, mask)
                break

        normalized = self.inverse_postprocessor(policy_guidance_action)
        bad = ~torch.isfinite(normalized)
        if bad.any():
            logger.warning(
                f"inverse_postprocessor produced {bad.sum().item()} non-finite value(s) "
                f"(NaN/Inf) in policy_guidance_action. Zeroing affected entries. "
                f"Check normalization stats for zero-variance dims."
            )
            normalized = normalized.masked_fill(bad, 0.0)
        return normalized

    def _build_guidance_noise_from_chunk(
        self, guidance_chunk: Tensor, ratio: float, base_noise: Tensor | None = None
    ) -> tuple[Tensor, float] | None:
        """Build partially-noised guidance using the correct noise schedule.

        For diffusion (DDPM/DDIM):
            x_tsw = scheduler.add_noise(guidance, noise, t_sw)
            where t_sw = int(ratio * num_train_timesteps)
            Denoising then runs from t_sw down to 0.

        For flow matching (PI0.5):
            x_tsw = ratio * noise + (1 - ratio) * guidance
            Denoising then starts from t=ratio instead of t=1.0.

        ratio=0 → pure human (no denoising), ratio=1 → pure policy (handled before this call).

        Returns (x_tsw, ratio) to pass as (noise=x_tsw, sa_noise_ratio=ratio) kwargs,
        or None if the inner policy doesn't expose the needed interface.
        """
        device = guidance_chunk.device
        batch_size = guidance_chunk.shape[0]

        # --- Diffusion (DDPM/DDIM) path ---
        diffusion_model = getattr(self.inner_policy, "diffusion", None)
        noise_scheduler = (
            getattr(diffusion_model, "noise_scheduler", None) if diffusion_model is not None else None
        )
        if noise_scheduler is not None:
            # The UNet operates on the full horizon (e.g. 16), but guidance_chunk is only
            # n_action_steps (e.g. 8). Embed the guidance at the correct position within
            # the full horizon and fill the rest with pure noise.
            horizon = self.config.horizon
            n_obs_steps = self.config.n_obs_steps
            action_dim = guidance_chunk.shape[2]
            if base_noise is not None:
                full_noise = base_noise.clone()
            else:
                full_noise = torch.randn(
                    batch_size, horizon, action_dim, dtype=guidance_chunk.dtype, device=device
                )
            # guidance occupies [n_obs_steps-1, n_obs_steps-1+n_action_steps) in the horizon.
            # Fill non-guidance positions with plausible values (not pure noise) so the UNet
            # sees a coherent full-horizon sequence during denoising.
            start = n_obs_steps - 1
            end = start + guidance_chunk.shape[1]
            full_guidance = torch.zeros(
                batch_size, horizon, action_dim, dtype=guidance_chunk.dtype, device=device
            )
            # Past positions [0:start]: repeat first guidance step
            for t in range(start):
                full_guidance[:, t, :] = guidance_chunk[:, 0, :]
            # Guidance region
            full_guidance[:, start:end, :] = guidance_chunk
            # Future positions [end:horizon]: repeat last guidance step
            for t in range(end, horizon):
                full_guidance[:, t, :] = guidance_chunk[:, -1, :]
            # Sync to the exact discrete inference timesteps so the injected
            # noise variance matches what the UNet expects on its first step.
            # Using raw `int(ratio * num_train_timesteps)` can land between
            # inference steps, causing SNR mismatch and jagged outputs.
            if not hasattr(noise_scheduler, "timesteps") or noise_scheduler.timesteps is None:
                num_inf_steps = getattr(diffusion_model, "num_inference_steps", 100)
                noise_scheduler.set_timesteps(num_inf_steps, device=device)
            timesteps = noise_scheduler.timesteps  # e.g. [999, 899, ..., 0]
            start_step_idx = int((1.0 - ratio) * len(timesteps))
            start_step_idx = max(0, min(start_step_idx, len(timesteps) - 1))
            t_sw = timesteps[start_step_idx]
            t_tensor = torch.full((batch_size,), t_sw, dtype=torch.long, device=device)
            x_tsw = noise_scheduler.add_noise(full_guidance, full_noise, t_tensor)
            return x_tsw

        # --- Flow matching (PI0.5) path ---
        if getattr(self.config, "max_action_dim", None) is None:
            # policy doesn't expose needed config
            raise NotImplementedError(
                "Inner policy does not support noise injection for guided execution. "
                "Please use a compatible policy (e.g. diffusion with noise_scheduler, or flow model with max_action_dim) or set forward_flow_ratio=1.0 for pure policy control."
            )
        # sample_actions expects (batch_size, chunk_size, max_action_dim). If n_action_steps < chunk_size,
        # pad guidance to chunk_size with repeated boundary values for a coherent sequence.
        chunk_size = self.config.chunk_size
        n_action_steps = guidance_chunk.shape[1]
        if n_action_steps < chunk_size:
            full_guidance = torch.zeros(
                batch_size, chunk_size, guidance_chunk.shape[2], dtype=guidance_chunk.dtype, device=device
            )
            full_guidance[:, :n_action_steps, :] = guidance_chunk
            for t in range(n_action_steps, chunk_size):
                full_guidance[:, t, :] = guidance_chunk[:, -1, :]
            guidance_chunk = full_guidance
        noise = base_noise.clone() if base_noise is not None else torch.randn_like(guidance_chunk)
        x_tsw = ratio * noise + (1.0 - ratio) * guidance_chunk
        return x_tsw

    def reset(self):
        self._obs_teleop_source.reset()
        self._desired_q = None
        self._latest_actual_q = None
        self._actual_q_history.clear()
        self._frames_since_last_rrt_end = 0
        self._prev_rrt_mode = RRTMode.IDLE
        self._prev_dq = None
        self._shield_cooldown_ticks = 0
        self._shield_check_tick_counter = 0
        # Clear RRT chunk state on episode boundary; keep the planner instance
        # so its obstacle cache survives if the env config hash matches next episode.
        self._rrt_source.reset()
        self._oracle_goal_source.reset()
        if self._start_paused:
            self._run_event.clear()
        return self.inner_policy.reset()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        return self.inner_policy.predict_action_chunk(batch, **kwargs)

    @torch.no_grad()
    def get_hold_action(self, inner_action: Tensor) -> Tensor:
        assert self._desired_q is not None
        raw = torch.tensor(
            self._desired_q.reshape(-1), dtype=inner_action.dtype, device=inner_action.device
        ).unsqueeze(0)  # [1, num_dofs+1]
        return self._normalize_policy_guidance_action(raw)

    @torch.no_grad()
    def get_full_teleop_action(self, delta: Tensor):
        """
        Pure teleop mode: apply FK+IK from _desired_q (not obs, to avoid lag).

        delta: [batch_size, 7] tensor [dx,dy,dz,droll,dpitch,dyaw,gripper]
        inner_action: fallback action from inner policy (for dtype/device)

        Returns normalized action tensor.
        """
        batch_size = delta.shape[0]
        delta_np = delta.cpu().numpy()
        device = delta.device

        assert self._desired_q is not None, "_desired_q must be seeded before get_full_teleop_action"
        actions = np.zeros((batch_size, self.num_dofs + 1), dtype=np.float64)
        q_seed = self._desired_q.reshape(-1).copy()
        for b in range(batch_size):
            d_pos, d_rot, d_gripper = delta_np[b][:3], delta_np[b][3:6], delta_np[b][6]
            q_new = self._project_delta_for_collision(
                q_seed, d_pos, d_rot, skip_collision=self.skip_collision
            )
            q_seed = q_new[: self.num_dofs].copy()
            actions[b] = np.concatenate([q_new, [float(d_gripper)]])

        self._last_raw_action = actions[-1]  # [num_dofs+1] float64, for _desired_q update
        raw_action = torch.tensor(actions, dtype=delta.dtype, device=device)
        action = self._normalize_policy_guidance_action(raw_action)
        return action

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], base_noise: Tensor | None = None) -> Tensor:
        self._run_event.wait()  # blocks while paused
        self._last_raw_action = None  # reset; set by get_full_teleop_action if called

        # Cache the oracle env config (obstacle geometry + task goal) sent by the
        # SplatSim server when env.include_oracle_info=true. Loading obstacles here
        # benefits both the IK collision projection and the RRT-to-goal mode.
        # The RRT source owns the obstacle adoption logic now — it also tears
        # down the wrapper's hardcoded fallback obstacles on first oracle load.
        oracle_cfg = batch.pop("oracle_env_config", None)
        if oracle_cfg is not None:
            # Reset per-scenario shield-disabled latch on new oracle receipt.
            # SplatSimEnv rebuilds oracle_env_config on env.reset(), so a
            # different dict identity here == new scenario. Clear the latch
            # so the shield can re-evaluate goal availability against the
            # new task (target may now be published for the new episode).
            if id(oracle_cfg) != self._last_applied_oracle_id:
                self._shield_disabled_no_goal = False
                self._last_applied_oracle_id = id(oracle_cfg)
            # Lazy URDF load: when the SA config didn't pin robot_name, use
            # the splat_name published by the sim on first oracle receipt.
            # Guarded on _urdf_loaded so a subsequent oracle receipt is a
            # no-op (URDFs don't hot-reload safely).
            if not self._urdf_loaded:
                robot_info = oracle_cfg.get("robot") or {}
                splat_name = robot_info.get("splat_name")
                if splat_name:
                    self._load_urdf(splat_name)
                else:
                    raise RuntimeError(
                        "SA wrapper: robot_name was not set in "
                        "SharedAutonomyConfig AND oracle_env_config did not "
                        "include robot.splat_name. Set "
                        "--policy.shared_autonomy_config.robot_name=<name> "
                        "explicitly, or ensure the sim publishes robot info."
                    )
            self._rrt_source.update_oracle_config(oracle_cfg)
            self._oracle_goal_source.update_oracle_config(oracle_cfg)
        elif not self._urdf_loaded:
            # No oracle AND no explicit robot_name → we can't proceed. Fail
            # fast on this first tick rather than crash later in _sync_joints.
            raise RuntimeError(
                "SA wrapper: URDF not loaded (robot_name not set in "
                "SharedAutonomyConfig) and no oracle_env_config on first "
                "select_action. Either set "
                "--policy.shared_autonomy_config.robot_name=<name> or "
                "enable --env.include_oracle_info=true."
            )

        # The obs-driven source pops OBS_GUIDANCE_CHUNK from the batch and computes
        # has_guidance for this tick. Done BEFORE inner_policy.select_action so the
        # inner policy doesn't see the (consumed) guidance key in its obs batch.
        self._obs_teleop_source.update(GuidanceCallCtx(batch=batch))
        has_guidance = self._obs_teleop_source.has_guidance

        obs_state = batch.get(OBS_STATE)

        if obs_state is None:
            raise RuntimeError("No obs.state available for shared autonomy wrapper")
        # TODO this is really only designed to handle 1 teleoperator and 1 policy (batch size = 1)
        assert obs_state.shape[0] == 1

        ratio = self.forward_flow_ratio
        rrt_active = self._rrt.mode == RRTMode.EXECUTING and self._rrt.chunk is not None
        if self._teleop_context is not None:
            self._teleop_context.ratio = ratio
            # Treat user guidance OR RRT execution as "real" frames so the recorder
            # keeps them after trim and counts them toward min_episode_length.
            self._teleop_context.has_guidance = has_guidance or rrt_active
            # Tag the frame source for the recorder. RRT execution always tags
            # RRT; otherwise we mirror the legacy "ratio==0 means teleop" rule
            # so the recorder only records pure-teleop segments. Anything else
            # (pure policy, blend) is POLICY and not recorded.
            if rrt_active:
                self._teleop_context.frame_source = FrameSource.RRT
            elif ratio == 0.0:
                self._teleop_context.frame_source = FrameSource.TELEOP
            else:
                self._teleop_context.frame_source = FrameSource.POLICY

        # No inner policy reset needed here — the obs queue is always updated by
        # inner_policy.select_action (called unconditionally below), so it stays
        # current regardless of whether we're blending or not.

        # If RRT is about to be cancelled this step, pre-flush the inner policy's
        # cached action chunk so the next inner_policy.select_action call hits an
        # empty queue and triggers predict_action_chunk against the up-to-date
        # obs queue. Without this, inner_action would be drained from the chunk
        # predicted at/before RRT start — telling the robot to move toward a
        # pre-RRT pose for one frame after cancel, which shows up as a stutter.
        # The matching _cancel_rrt below also flushes (idempotent) so the cancel
        # state stays consistent if the order is ever rearranged.
        rrt_will_cancel = (
            self._rrt.mode == RRTMode.EXECUTING
            and self._rrt.chunk is not None
            and (has_guidance or self._rrt.cancel_requested)
        )
        if rrt_will_cancel:
            self._flush_inner_action_queue()
            self._obs_teleop_source.cancel()

        # Inner policy call — full or lightweight depending on whether an
        # intervention (RRT / oracle-goal) is currently EXECUTING.
        #
        # Full select_action: pushes obs into `_queues[OBS_*]`, if the ACTION
        # queue is empty runs a full forward pass (DDPM denoising for
        # diffusion; flow-matching solve for Pi0/0.5) — the expensive part.
        #
        # During intervention EXECUTING: the wrapper's downstream branches
        # (RRT playback ~2080, oracle-goal playback ~2100) DISCARD the inner
        # action and emit an intervention waypoint instead. Running the full
        # forward pass just to throw the result away is pure waste. The
        # `_lightweight_inner_call` helper (a) pops cached actions cheaply
        # when the queue is non-empty, or (b) pushes obs into the history
        # queue and returns a dummy tensor (correct dtype/device/shape) when
        # the queue would otherwise trigger a fresh forward pass.
        #
        # `_flush_inner_action_queue` fires on intervention-end (see
        # `_cancel_intervention`, `_finish_rrt`) so the next tick's full
        # select_action call generates a fresh chunk against an up-to-date
        # (RRT-tracked) obs history, without needing to remember to bootstrap
        # anything.
        #
        # PLANNING mode: we still call full select_action because the blocking
        # plan is likely to just run once per intervention cycle, and any obs
        # decode / normalization state the inner might have side-effected
        # on the pre-plan tick should stay valid. Cheap in practice —
        # PLANNING lasts one tick before flipping to EXECUTING.
        _intervention_executing = self._rrt.mode == RRTMode.EXECUTING or self._oracle_goal_source.is_active()
        if _intervention_executing:
            inner_action = self._lightweight_inner_call(batch)
        else:
            inner_action = self.inner_policy.select_action(batch)
        # Belt-and-suspenders: downstream branches (RRT playback, oracle-goal
        # playback, get_hold_action) index `inner_action.dtype/.device/.shape[0]`
        # unconditionally. Any code path that ever produces None here would
        # take down the whole select_action call (has happened when a policy
        # adapter under-specifies its select_action contract). Fall back to
        # the cached dummy template — same shape/dtype/device as the model's
        # real output, safe for every downstream consumer.
        if inner_action is None:
            template = self._inner_action_template
            if template is None:
                _param = next(self.inner_policy.parameters(), None)
                _dtype = _param.dtype if _param is not None else torch.float32
                _device = _param.device if _param is not None else torch.device("cpu")
                template = torch.zeros((1, self.num_dofs + 1), dtype=_dtype, device=_device)
                self._inner_action_template = template
            inner_action = template.clone()

        # Sync _desired_q from the ACTUAL observed joint state (not the cumulative
        # commanded value). The wrapper's pybullet client uses resetJointState
        # which is a teleport — no physics, so a previously-commanded pose may
        # have phased through an obstacle in our private client even though the
        # env's physics-enabled simulator stopped the real robot at the surface.
        # Re-syncing from obs every step keeps our private client matched to
        # reality, which fixes RRT plans starting from inside-an-obstacle, IK
        # seeded at a phantom pose, and teleop deltas accumulating from a place
        # the robot isn't actually at.
        #
        # obs.state was normalized by the policy preprocessor — so the right
        # inverse is the preprocessor's UnnormalizerProcessorStep, NOT the
        # action postprocessor. The action postprocessor includes an
        # AbsoluteActionsProcessorStep that adds the cached state to convert
        # relative deltas back to absolute joints — applying it to obs.state
        # double-adds and causes a constant offset.
        #
        # ``self.inverse_preprocessor`` carries the right UnnormalizerProcessor
        # (configured with the preprocessor's stats), but its top-level
        # ``to_transition`` puts the input in the ACTION slot, while the
        # unnormalize step expects obs in the OBSERVATION slot. Bypass the
        # bogus to_transition by building the transition manually with
        # obs_state as the observation, then run the steps directly.
        actual_q_t = None
        try:
            from lerobot.processor.converters import create_transition
            from lerobot.processor.normalize_processor import UnnormalizerProcessorStep
            from lerobot.types import TransitionKey

            transition = create_transition(observation={OBS_STATE: obs_state})
            for _step in self.inverse_preprocessor.steps:
                if isinstance(_step, UnnormalizerProcessorStep):
                    transition = _step(transition)
                    break
            obs_dict = transition.get(TransitionKey.OBSERVATION)
            if isinstance(obs_dict, dict) and OBS_STATE in obs_dict:
                actual_q_t = obs_dict[OBS_STATE]
        except Exception:
            actual_q_t = None

        if actual_q_t is not None:
            actual_q = actual_q_t[0].detach().cpu().numpy().astype(np.float64)
            # observation.state is normally [num_dofs joints + gripper] =
            # num_dofs+1 entries. But when --exclude_gripper_from_state is
            # set at the preprocessor, obs.state is arm-only (num_dofs
            # entries). We MUST still keep `_desired_q` and `_latest_actual_q`
            # sized to num_dofs+1 — the RRT-playback branch (~line 2082)
            # reads gripper via `_desired_q[-1]`, and a short array would
            # make `[-1]` return the last ARM joint (e.g. joint_3 ≈ -1.35 rad)
            # as the gripper command. The env's Robotiq interprets that as
            # a close-fingers command mid-RRT → collision → physics kick →
            # spurious episode split. Pad the missing gripper slot with
            # 0.0 (open pose). The wrapper doesn't observe or plan gripper
            # motion in the exclude-gripper mode, so "always open" is the
            # correct constant assumption (matches the parallel fix in
            # is_q_in_collision).
            _flat = actual_q.reshape(-1)
            if _flat.size < self.num_dofs + 1:
                _pad = np.zeros(self.num_dofs + 1, dtype=np.float64)
                _pad[: _flat.size] = _flat[: self.num_dofs]
                _flat = _pad  # gripper slot stays 0.0 (open)
            self._desired_q = _flat[: self.num_dofs + 1].copy()
            # Preserve a copy of the actual observation for RRT's q_start —
            # _desired_q gets overwritten with the commanded action at the end
            # of select_action, so by the time the planner thread reads it the
            # value reflects "where we want the robot to go next", not "where
            # the robot is right now". When commanded ≠ actual (collision,
            # mid-chunk replay, etc.) the latter is what RRT needs.
            self._latest_actual_q = _flat[: self.num_dofs + 1].copy()
            # Also push into the rolling history so RRT can pull q_start from
            # N steps ago (pre-jump pose), not just the current actual_q.
            self._actual_q_history.append(self._latest_actual_q.copy())
            # Teleport landing check. If a teleport fired on a previous tick,
            # `_pending_teleport_landing` holds the arm pose the robot was
            # teleported to (= the planned chunk start). This is the FIRST
            # real obs decode since, so `_latest_actual_q` is where the robot
            # actually ended up. A clean landing reads ~0; a large error means
            # the rewind pose was invalid (physics ejected the robot) or the
            # teleport didn't take over ZMQ, so the open-loop chunk now runs
            # away from the real robot. Log it, then clear the pending flag.
            if self._pending_teleport_landing is not None:
                land_err = float(
                    np.linalg.norm(self._latest_actual_q[: self.num_dofs] - self._pending_teleport_landing)
                )
                logger.info(
                    "RRT teleport landing error: %.4f rad "
                    "(actual robot pose one tick later vs planned chunk start; "
                    "~0 = clean, large = ejected/teleport-missed → chunk will run away)",
                    land_err,
                )
                self._pending_teleport_landing = None
            # Track post-intervention idle frames. Bumped on every policy-
            # driven tick (RRT IDLE), zeroed the moment RRT leaves IDLE for
            # a new cycle. rrt_source._do_plan() caps its lookback sample
            # at this counter so it can't rewind into a prior RRT cycle's
            # trajectory — see the comment on the ctor field for rationale.
            # NOTE: this needs `self._rrt` (the back-compat view onto the
            # RRT source's state), which exists from ctor regardless of
            # whether RRT is the active guidance source.
            #
            # Off-by-one detail: this check runs at the TOP of select_action,
            # BEFORE the RRT chunk is consumed for this tick. On the very
            # last RRT-EXECUTING frame R, mode is still EXECUTING here,
            # then the chunk's last waypoint pops and mode flips to IDLE
            # for frame R+1's check. We want the counter to represent
            # "number of POLICY-driven frames we can rewind into" — i.e.,
            # 0 at frame R+1 (the first post-RRT frame; rewinding 0 frames
            # leaves us at the post-RRT config), 1 at R+2, ... NOT 1 at
            # R+1. Without this off-by-one fix, rrt_source._do_plan's
            # lookback cap lands on the state at frame R (DURING the last
            # RRT waypoint) instead of R+1 (where RRT left the robot),
            # producing the "rewinds to a frame before RRT ended" artifact.
            # Fix: only increment after the SECOND consecutive IDLE
            # observation, using a prev-mode latch.
            _was_idle = self._prev_rrt_mode == RRTMode.IDLE
            if self._rrt.mode == RRTMode.IDLE:
                if _was_idle:
                    self._frames_since_last_rrt_end += 1
                else:
                    # Just transitioned EXECUTING/PLANNING -> IDLE this
                    # tick. The post-RRT state IS this tick's _actual_q,
                    # so "0 frames back" already lands on it.
                    self._frames_since_last_rrt_end = 0
            else:
                self._frames_since_last_rrt_end = 0
            self._prev_rrt_mode = self._rrt.mode
        elif self._desired_q is None:
            # Last-resort initial seed from the policy's postprocessed action.
            self._desired_q = self.postprocessor(inner_action).cpu().numpy().reshape(-1)
        assert self._desired_q is not None  # narrowed for the type checker

        # Reflect the (just-synced) actual joint state into the wrapper's
        # pybullet client so RRT planning, IK, and collision projection all
        # see a pose matching the env's real robot.
        self._sync_joints(self._desired_q[: self.num_dofs])

        # --- future_chunk predictive shield --------------------------------
        # When `rrt_collision_detection="future_chunk"`, FK-check the inner
        # policy's already-cached chunk against the obstacle world. If a
        # future waypoint would collide AND RRT isn't already running,
        # preempt the policy and trigger RRT from the CURRENT continuous-
        # motion state (no_lookback=True). The recorded intervention episode
        # therefore starts velocity-continuous, in-distribution.
        # Decrement the post-fail cooldown by one every tick. Skips shield
        # evaluation while >0. Reset by a successful plan (below) or by
        # episode reset.
        if self._shield_cooldown_ticks > 0:
            self._shield_cooldown_ticks -= 1
        # Rate-limiter: only run the shield's FK sweep every N ticks. Bumps
        # ONLY when the outer gate would otherwise have run the check, so
        # cooldown skips + mode-EXECUTING skips don't consume rate-limit
        # budget and delay the next real check post-cooldown.
        _shield_gate_open = (
            self._collision_detection_mode in ("future_chunk", "hybrid")
            and self._rrt.mode == RRTMode.IDLE
            and not self._shield_disabled_no_goal
            and self._shield_cooldown_ticks == 0
        )
        _rate_limit_hit = False
        if _shield_gate_open:
            _rate_limit_hit = self._shield_check_tick_counter % self._shield_check_every_n_ticks != 0
            self._shield_check_tick_counter += 1
        if _shield_gate_open and not _rate_limit_hit:
            shield_collides, shield_step, shield_kind, shield_offending_q = (
                self._check_future_chunk_collision()
            )
            # DEBUG: force a synthetic "collision" so the full trigger path
            # runs every tick regardless of the real FK check. Reproduces
            # heavy-shield-firing pathologies (queue thrashing, shared-
            # anchor mutation) even in scenarios with no real collisions.
            if self._debug_shield_force_trigger and not shield_collides:
                shield_collides = True
                shield_kind = "debug_forced"
                shield_step = 0
            # NOTE: the "current-config already in collision → skip shield"
            # gate that used to live here has been removed. The planner has
            # three escape methods (policy-history rewind, contact-normal,
            # self-collision gradient), and any one of them can succeed
            # even when the current config is in collision. Skipping the
            # trigger meant the arm could get stuck sliding against an
            # obstacle for many ticks without ever attempting an escape.
            # Now: shield still fires, RRT attempts all escape methods, and
            # only when planning FAILS does the post-trigger cooldown gate
            # engage (see below) to avoid per-tick escape-chain thrashing.
            if shield_collides and not self._shield_can_plan():
                # RRT can't recover without a task goal — firing the shield
                # would infinite-loop (trigger → source aborts with "no
                # goal" → shield sees same predicted collision → repeat).
                # Log ONCE per scenario and latch off; oracle change on
                # next episode re-enables via _last_applied_oracle_id.
                logger.warning(
                    "Future-chunk shield detected predicted %s collision at step %d "
                    "but oracle_env_config publishes no task.target_ee_pos — "
                    "RRT has nothing to plan toward. Disabling shield for this "
                    "scenario to prevent infinite retriggering.",
                    shield_kind or "unknown",
                    shield_step if shield_step is not None else -1,
                )
                self._shield_disabled_no_goal = True
                shield_collides = False  # skip the trigger block below
            if shield_collides:
                # Debug telemetry: surface which link pair tripped the shield
                # so we can tell whether grasp-finger ⟷ target-object proximity
                # is firing (false-positive for grasp tasks) vs a real
                # arm-vs-obstacle collision. `shield_offending_q` is the predicted
                # future joint config at the offending chunk step (computed inside
                # `_check_future_chunk_collision` via the same anchor+chunk math
                # the shield uses, with actual gripper snapped on).
                pair_str = ""
                planner = self._rrt_source.state.planner
                if planner is not None and shield_offending_q is not None:
                    info = planner.describe_collision_at(
                        shield_offending_q,
                        obstacle_clearance=self._future_chunk_config.obstacle_clearance,
                        self_collision_clearance=self._future_chunk_config.self_collision_clearance,
                    )
                    if info is not None:
                        flag = "VIOLATION" if info["in_violation"] else "ok"
                        pair_str = (
                            f" [debug] pair={info['kind']} {info['link_a_name']} ⟷ "
                            f"{info['link_b_name']} (dist={info['distance_m'] * 1000:.2f}mm, "
                            f"threshold={info['threshold_m'] * 1000:.2f}mm) [{flag}]"
                        )
                logger.info(
                    "Future-chunk shield: predicted %s collision at chunk step %d — "
                    "triggering RRT from current state (no rewind).%s",
                    shield_kind or "unknown",
                    shield_step if shield_step is not None else -1,
                    pair_str,
                )
                # Synchronous (blocking) RRT trigger from current state.
                # In future_chunk mode the source's _do_plan reads
                # q_start = wrapper._latest_actual_q and skips teleport.
                # NOTE: We flush the inner-policy chunk queue ONLY if the
                # plan actually succeeded (RRT is now EXECUTING). If the
                # plan failed (start in collision, escape failed, no goal,
                # etc.), RRT stays IDLE and flushing would just force the
                # diffusion policy to re-predict a fresh chunk with new
                # noise NEXT tick — producing per-tick action jitter (the
                # exact "robot shakes every frame" pathology). The current
                # chunk was FK-predicted to collide, but a re-predicted
                # chunk with different noise is no more likely to be safe
                # AND the resulting per-tick noise-driven commanded jumps
                # tend to push the robot further into collision, causing
                # amplitude-growing shakes. Better to let the current
                # chunk keep executing (it's about to collide but the env
                # will handle that via terminate_on_collision) than to
                # noise-thrash the arm every tick indefinitely.
                self._rrt_source.trigger(no_lookback=True)
                if self._rrt.mode == RRTMode.EXECUTING:
                    # Plan succeeded — safe to flush; the RRT chunk about
                    # to play out supersedes anything left in the inner
                    # queue.
                    self._flush_inner_action_queue()
                    self._shield_cooldown_ticks = 0
                else:
                    # Plan failed — suppress the shield for the next N ticks
                    # so we don't retrigger every frame against the same
                    # unresolvable collision. Inner policy keeps driving.
                    self._shield_cooldown_ticks = self._SHIELD_COOLDOWN_ON_PLAN_FAIL
                    logger.info(
                        "Future-chunk shield: RRT plan failed; suppressing shield "
                        "for %d ticks to avoid per-frame queue thrash.",
                        self._SHIELD_COOLDOWN_ON_PLAN_FAIL,
                    )
                # Refresh local view so the EXECUTING branch picks up.

        # Capture q_prev BEFORE any action computation so velocity limiting sees the true
        # previous position. get_full_teleop_action pre-updates _desired_q internally,
        # so reading it afterward would give dq=0 (a no-op).
        # q_prev_for_vel_limit = self._desired_q.reshape(-1).copy() if self._desired_q is not None else None

        # --- RRT-to-Goal mode: highest priority among non-paused branches. ---
        # Cancellation: user takes over (has_guidance) or explicit cancel button.
        rrt = self._rrt
        if rrt.mode == RRTMode.EXECUTING and rrt.chunk is not None:
            if has_guidance or rrt.cancel_requested:
                # print("cancel rrt")
                self._log_rrt_drift_summary("cancelled")
                self._cancel_rrt()
                # _cancel_rrt cleared stale inner-policy + obs-blend caches.
                # Return a hold action for this tick; next tick falls through
                # to the obs-driven source naturally.
                return self.get_hold_action(inner_action)
            elif rrt.step >= len(rrt.chunk):
                # print('finish rrt: chunk exhausted (step %d >= chunk length %d)' % (rrt.step, len(rrt.chunk)))
                # Goal reached: restore prior ratio, auto-pause for the next step.
                self._log_rrt_drift_summary("completed")
                self._finish_rrt()
                action = self.get_hold_action(inner_action)
                # Skip the existing branches below; jump to _desired_q update.
                assert self._desired_q is not None  # seeded above
                self._last_raw_action = self._desired_q.reshape(-1).copy()
                return action
            else:
                # print("rrt executing: step %d / %d" % (rrt.step, len(rrt.chunk)))
                # ── Drift diagnostic (this is the LIVE RRT playback path —
                # the wrapper drives the chunk directly here; the source's
                # next_action() is NOT used for RRT). Each tick the robot
                # should have reached the PREVIOUS waypoint (commanded last
                # tick). `_latest_actual_q` was refreshed from this tick's obs
                # above. step-1 drift = the teleport landing error (robot vs
                # chunk start); drift that GROWS step-over-step = open-loop
                # runaway (robot can't track while the command marches on).
                if rrt.step == 0:
                    self._rrt_drift_max = 0.0
                    self._rrt_drift_at_step1 = None
                    self._rrt_drift_streak = 0
                if rrt.step >= 1 and self._latest_actual_q is not None:
                    prev_wp = np.asarray(rrt.chunk[rrt.step - 1][: self.num_dofs], dtype=np.float64)
                    actual = self._latest_actual_q.reshape(-1)[: self.num_dofs].astype(np.float64)
                    drift_vec = actual - prev_wp
                    drift = float(np.linalg.norm(drift_vec))
                    self._rrt_drift_max = max(getattr(self, "_rrt_drift_max", 0.0), drift)
                    if rrt.step == 1:
                        self._rrt_drift_at_step1 = drift
                    # Drift-abort guard: the robot has stopped following the
                    # command (wedged on contact the env's penetration-only
                    # in_collision check misses; happens with OR without
                    # lookback). Count consecutive over-threshold ticks; once
                    # sustained, cancel the chunk so the open-loop runaway never
                    # reaches the recorded dataset. The cancel fires on the NEXT
                    # tick's `rrt.cancel_requested` branch (_cancel_rrt), and the
                    # recorder drops the resulting short fragment.
                    if self._rrt_abort_on_drift_rad > 0.0:
                        if drift > self._rrt_abort_on_drift_rad:
                            self._rrt_drift_streak += 1
                        else:
                            self._rrt_drift_streak = 0
                        if (
                            not rrt.cancel_requested
                            and self._rrt_drift_streak >= self._rrt_abort_on_drift_ticks
                        ):
                            worst_j = int(np.abs(drift_vec).argmax())
                            logger.warning(
                                "RRT drift-abort: |actual-commanded|=%.3f rad for %d consecutive "
                                "ticks (> %.2f) at step %d/%d (worst joint %d: actual=%.3f "
                                "commanded=%.3f) — robot not tracking; cancelling chunk and "
                                "DISCARDING the episode so no drift frames reach the dataset.",
                                drift,
                                self._rrt_drift_streak,
                                self._rrt_abort_on_drift_rad,
                                rrt.step,
                                len(rrt.chunk),
                                worst_j,
                                float(actual[worst_j]),
                                float(prev_wp[worst_j]),
                            )
                            # Dump the closest robot-vs-obstacle + robot-vs-robot
                            # link pairs so you can see WHAT is physically
                            # contacting the arm when drift piles up. When a
                            # pair shows dist ≤ 0 (mesh penetration) AND is
                            # SKIPPED in the planner's contract, the planner
                            # accepts that config kinematically but PyBullet
                            # generates contact force there → joint stall.
                            # Pairs are labelled SKIPPED / not-skipped so it's
                            # immediately obvious which side of the semantic
                            # gap they sit on. See `_dump_drift_collision_diagnostic`.
                            self._dump_drift_collision_diagnostic(actual)
                            rrt.cancel_requested = True
                            # Drop the ENTIRE in-progress RRT episode — not just
                            # the post-abort frames, but the whole detection-delay
                            # run that accumulated while drift was building. The
                            # recorder consumes this at the top of its next step()
                            # via _discard_episode(), which clears both its frame
                            # buffer AND the dataset's in-memory episode buffer
                            # (nothing is on disk until save_episode), so zero
                            # drift frames are committed. No-op when not recording.
                            ctx = getattr(self, "_teleop_context", None)
                            if ctx is not None:
                                ctx.discard_requested = True
                            # Request a re-plan per `rrt_drift_trigger`. The
                            # cancelled chunk reaches IDLE next tick; the
                            # InterventionController then fires the trigger
                            # through the SAME `_trigger_source` path the other
                            # triggers use (reason="drift_stall"), so the
                            # lookback choice stays consistent. "discard" leaves
                            # this None → no re-plan, control returns to policy.
                            self._rrt_drift_replan_no_lookback = {
                                "lookback": False,
                                "no_lookback": True,
                            }.get(self._rrt_drift_trigger, None)
                    # Gated by debug_rrt_drift_log — off by default because these
                    # fire every 10th RRT-chunk tick and drown the rest of the
                    # log. Enable when debugging phantom-state / rel-action-
                    # postprocessor drift bugs. Includes both a per-joint worst
                    # summary and a full-vector dump so the live
                    # `_latest_actual_q` and chunk trajectories can be aligned
                    # against the RECORDED observation.state / action frame-by-
                    # frame offline (distinguishes (a) phantom actual, from
                    # (b) fabricated commanded rel-action ramp).
                    if self._debug_rrt_drift_log:
                        if rrt.step == 1 or rrt.step % 10 == 0 or drift > 0.1:
                            worst_j = int(np.abs(drift_vec).argmax())
                            logger.info(
                                "RRT drift @ step %d/%d: |actual-commanded|=%.4f rad "
                                "(worst joint %d: actual=%.4f commanded=%.4f Δ=%+.4f; max-so-far %.4f)",
                                rrt.step,
                                len(rrt.chunk),
                                drift,
                                worst_j,
                                float(actual[worst_j]),
                                float(prev_wp[worst_j]),
                                float(drift_vec[worst_j]),
                                self._rrt_drift_max,
                            )
                        if rrt.step == 1 or rrt.step % 20 == 0:
                            logger.info(
                                "RRT drift raw @ step %d/%d: actual=%s commanded(chunk[step-1])=%s",
                                rrt.step,
                                len(rrt.chunk),
                                np.array2string(actual, precision=4, suppress_small=True),
                                np.array2string(prev_wp, precision=4, suppress_small=True),
                            )
                wp = rrt.chunk[rrt.step][: self.num_dofs]
                rrt.step += 1
                gripper = float(self._desired_q[-1]) if self._desired_q is not None else 0.0
                raw7 = np.concatenate([wp, [gripper]]).astype(np.float64)
                self._last_raw_action = raw7  # picked up by the post-block _desired_q update
                raw_t = torch.tensor(raw7, dtype=inner_action.dtype, device=inner_action.device).unsqueeze(0)
                action = self._normalize_policy_guidance_action(raw_t)
                # _desired_q is updated in the post-block via _last_raw_action.
                self._desired_q = raw7.copy()
                return action

        # --- Oracle-goal source: method-triggered (like RRT), VERBATIM playback. ---
        # Active iff an external caller (e.g. lerobot-eval --intervention with
        # method=oracle_goal) called `self._oracle_goal_source.trigger()`. The
        # chunk is a linear-interpolation from q_start to q_goal_bias; emit one
        # waypoint per step, tagged FrameSource.BLEND_INTERVENTION_100.
        if self._oracle_goal_source.is_active():
            if has_guidance:
                # User guidance arrived: cancel the oracle-goal sequence and
                # let the obs-teleop source take over on this and future ticks.
                self._oracle_goal_source.state.cancel_requested = True
            og_result = self._oracle_goal_source.next_action(
                GuidanceCallCtx(
                    batch=batch,
                    desired_q=self._desired_q,
                    actual_q_history=self._actual_q_history,
                    latest_actual_q=self._latest_actual_q,
                    inner_action=inner_action,
                    inner_dtype=inner_action.dtype,
                    inner_device=inner_action.device,
                    oracle_env_config=oracle_cfg,
                )
            )
            if og_result.flush_inner_queue_after:
                self._flush_inner_action_queue()
                self._obs_teleop_source.cancel()
            if self._teleop_context is not None and og_result.frame_source is not None:
                # Override the upfront frame_source tagging (which guessed POLICY
                # / TELEOP / RRT). OracleGoal emits BLEND_INTERVENTION_<XXX>.
                self._teleop_context.frame_source = og_result.frame_source
                self._teleop_context.has_guidance = True
            if og_result.raw7 is not None:
                self._desired_q = og_result.raw7.reshape(-1).copy()
            return og_result.action

        # --- Obs-driven path (pure teleop / blend): delegated to ObservationTeleopGuidanceSource. ---
        # Source already saw OBS_GUIDANCE_CHUNK during its update() above; it picks the
        # right sub-case (pure teleop at ratio=0, drain, or blend rebuild) internally.
        # When the source isn't active (ratio>0, no guidance, nothing draining), the
        # wrapper just returns the inner policy's output directly.
        if self._obs_teleop_source.is_active():
            action = self._obs_teleop_source.next_action(
                GuidanceCallCtx(
                    batch=batch,
                    desired_q=self._desired_q,
                    actual_q_history=self._actual_q_history,
                    latest_actual_q=self._latest_actual_q,
                    inner_action=inner_action,
                    inner_dtype=inner_action.dtype,
                    inner_device=inner_action.device,
                    oracle_env_config=oracle_cfg,
                ),
                base_noise=base_noise,
            ).action
        else:
            action = inner_action

        # Update _desired_q from the action we're about to send, so all modes
        # accumulate in raw joint space (like KeyboardInterfaceAgent._desired_q).
        # When get_full_teleop_action was called, use its raw float64 IK result
        # to avoid precision loss from the normalize→denormalize roundtrip.
        if self._last_raw_action is not None:
            self._desired_q = self._last_raw_action.reshape(-1).copy()
        else:
            self._desired_q = self.postprocessor(action).cpu().numpy().reshape(-1)

        # DEBUG: gated by debug_shield_trace_anchor. Logs the wrapper's exit
        # source (inner/guidance/rrt/oracle) and the commanded absolute joint
        # target this tick — same value the outer postprocessor + env will
        # see. Comparing consecutive ticks makes per-tick jitter obvious.
        #
        # ALSO logs at chunk-boundary ticks the raw rel-action delta the inner
        # policy predicted AND the anchor the postprocessor added, so we can
        # separate "policy predicted a big rel_0 (undertrained)" from
        # "anchor drifted from actual current state (wrapper/preproc bug)".
        # For a well-trained rel-action policy, rel_0 should be ~0 rad (the
        # first waypoint of a fresh chunk is "stay where you are"); anchor
        # should equal the just-observed actual joint state. Any drift
        # between anchor and latest_actual_q on a chunk-boundary tick is
        # the smoking gun for an anchor bug.
        if self._debug_shield_trace_anchor:
            _source = "inner"
            if self._rrt.mode == RRTMode.EXECUTING:
                _source = "rrt"
            elif self._oracle_goal_source.is_active():
                _source = "oracle_goal"
            # Chunk-boundary diagnostics: fires ONLY when the inner policy's
            # action queue was empty on entry (i.e., it re-predicted a fresh
            # chunk this tick). Detected by peeking the queue AFTER select
            # (has n_action_steps - 1 entries if a fresh chunk was just
            # generated + one popped; if the queue had entries before, it'd
            # be one fewer now — heuristic based on chunk-size threshold).
            try:
                _pending = self.inner_policy.get_pending_action_chunk()
                _q_len = 0 if _pending is None else int(_pending.shape[0])
                _ncfg = getattr(self.inner_policy.config, "n_action_steps", None)
                _chunk_boundary = (
                    _ncfg is not None and _q_len >= _ncfg - 1  # fresh chunk minus the 1 popped this tick
                )
            except Exception:
                _chunk_boundary = False
            if _chunk_boundary and self._latest_actual_q is not None:
                # Fetch anchor from the (shared) rel_step and denormalize the
                # inner_action so we can compare in raw radians.
                _anchor_raw = None
                _rel0_raw = None
                for _step in self.postprocessor.steps:
                    if isinstance(_step, AbsoluteActionsProcessorStep):
                        _rs = _step.relative_step
                        if _rs is not None and _rs._last_state is not None:
                            _a = _rs._last_state
                            if _a.ndim == 3:
                                _a = _a[..., -1, :]
                            _anchor_raw = _a.detach().cpu().numpy().reshape(-1)
                        break
                # Denormalize inner_action to get raw rel_0 (bypass abs step).
                try:
                    _raw = self._denormalize_chunk_to_raw(inner_action.unsqueeze(0))
                    if _raw is not None:
                        _rel0_raw = _raw[0]
                except Exception:  # nosec B110 - debug logging path; missing denormalized value is optional
                    pass
                _actual = self._latest_actual_q.reshape(-1)[: self.num_dofs]
                _drift = None
                if _anchor_raw is not None:
                    _drift = _anchor_raw[: self.num_dofs] - _actual
                logger.info(
                    "SA-CHUNK-BOUNDARY anchor=[%s] actual=[%s] anchor-actual=[%s] rel_0_raw=[%s]",
                    " ".join(
                        f"{x:+.4f}" for x in (_anchor_raw[: self.num_dofs] if _anchor_raw is not None else [])
                    ),
                    " ".join(f"{x:+.4f}" for x in _actual),
                    " ".join(f"{x:+.4f}" for x in _drift) if _drift is not None else "n/a",
                    " ".join(
                        f"{x:+.4f}" for x in (_rel0_raw[: self.num_dofs] if _rel0_raw is not None else [])
                    ),
                )
            elif self._obs_teleop_source.is_active():
                _source = "obs_teleop"
            _dq = self._desired_q.reshape(-1)[: self.num_dofs]
            logger.info(
                "SA-tick src=%s q_cmd=[%s]",
                _source,
                " ".join(f"{x:+.4f}" for x in _dq),
            )

        return action

    def _dump_drift_collision_diagnostic(self, actual_q: np.ndarray) -> None:
        """Log the closest robot-vs-obstacle + robot-vs-self mesh distances
        at the currently-stuck config, labelling each pair by its SKIP
        status in the planner's contract. Called from the drift-abort
        branch of `select_action` — helps identify pairs where the planner
        accepts a config (skipped from its collision check) but PyBullet's
        constraint solver still generates contact force, holding a joint
        stuck.

        Semantic gap this reveals:
          * Planner says "no collision" for a SKIPPED pair regardless of
            actual mesh overlap → RRT plans a chunk through that config
          * PyBullet's solver enforces non-penetration on ALL meshes,
            skipped or not → generates contact force → joint fails to
            track the commanded ruckig path → drift accumulates → abort
          * The diagnostic makes the offending pair visible so you can
            move it from SELF_COLLISION_SKIP_PAIRS to
            SELF_COLLISION_SKIP_PAIRS_EVAL_TERMINATE_EXTRA (planner
            catches it, eval terminate still ignores it).

        Best-effort: silently no-ops if any pybullet call fails; skips
        pair enumeration if the planner state isn't available.
        """
        try:
            # Snap the wrapper's pybullet client to the stuck config so
            # getClosestPoints reflects the ACTUAL stuck geometry, not
            # whatever was left from the last planner tick.
            self._sync_joints(actual_q)

            num_joints = p.getNumJoints(self._robot_id, physicsClientId=self._pb_client)

            # Read the planner's skip contract so we can label pairs.
            planner = getattr(self._rrt_source.state, "planner", None)
            strict_self_skips: set[frozenset[int]] = set()
            obstacle_skip_pairs: set[tuple[int, int]] = set()
            if planner is not None:
                # STRICT list (used by RRT + escape).
                _s = planner._collision_kwargs.get("self_collision_skip_pairs") or []
                strict_self_skips = {frozenset((int(a), int(b))) for a, b in _s}
                obstacle_skip_pairs = set(getattr(planner, "_skip_pairs", set()))

            def _link_name(link_i: int) -> str:
                if link_i == -1:
                    return "base(-1)"
                info = p.getJointInfo(self._robot_id, link_i, physicsClientId=self._pb_client)
                return f"{info[12].decode('utf-8')}({link_i})"

            def _obs_name(oid: int) -> str:
                # Wrapper doesn't hold names; the planner does when available.
                if planner is not None:
                    return f"{planner._obstacle_names.get(oid, str(oid))}(id={oid})"
                return f"obstacle({oid})"

            # Robot-vs-obstacle: enumerate every (link, obstacle) pair whose
            # closest points are within 5 cm. Sorted most-penetrating first.
            obs_hits: list[tuple[float, int, int, bool]] = []
            for obs_id in getattr(self, "_obstacle_ids", []):
                for link_i in range(-1, num_joints):
                    pts = p.getClosestPoints(
                        bodyA=self._robot_id,
                        bodyB=obs_id,
                        distance=0.05,
                        linkIndexA=link_i,
                        physicsClientId=self._pb_client,
                    )
                    for pt in pts:
                        dist = float(pt[8])
                        skipped = (link_i, obs_id) in obstacle_skip_pairs
                        obs_hits.append((dist, link_i, obs_id, skipped))
                        break  # closest point per (link, obstacle) pair
            obs_hits.sort(key=lambda t: t[0])

            # Robot self-collision: enumerate every non-adjacent link pair
            # within 2 cm. Sorted most-penetrating first.
            self_hits: list[tuple[float, int, int, str]] = []
            for a in range(num_joints):
                for b in range(a + 1, num_joints):
                    # Cheap adjacency check: parent link relation.
                    _pa = p.getJointInfo(self._robot_id, a, physicsClientId=self._pb_client)[16]
                    _pb = p.getJointInfo(self._robot_id, b, physicsClientId=self._pb_client)[16]
                    if _pa == b or _pb == a:
                        continue
                    pts = p.getClosestPoints(
                        self._robot_id,
                        self._robot_id,
                        distance=0.02,
                        linkIndexA=a,
                        linkIndexB=b,
                        physicsClientId=self._pb_client,
                    )
                    for pt in pts:
                        dist = float(pt[8])
                        if frozenset((a, b)) in strict_self_skips:
                            status = "SKIPPED (strict)"
                        else:
                            status = "not-skipped"
                        self_hits.append((dist, a, b, status))
                        break
            self_hits.sort(key=lambda t: t[0])

            # Log — cap at top-6 per bucket to keep the log short.
            lines = ["[drift-abort collision diagnostic]"]
            lines.append(f"  Robot-vs-obstacle (top {min(6, len(obs_hits))} closest, ≤ 50 mm):")
            for dist, link_i, obs_id, skipped in obs_hits[:6]:
                tag = "SKIPPED" if skipped else "not-skipped"
                lines.append(
                    f"    {_link_name(link_i):<30} vs {_obs_name(obs_id):<28} "
                    f"dist={dist * 1000:>+7.2f} mm  ({tag})"
                )
            lines.append(f"  Robot self-collision (top {min(6, len(self_hits))} closest, ≤ 20 mm):")
            for dist, a, b, status in self_hits[:6]:
                lines.append(
                    f"    {_link_name(a):<30} vs {_link_name(b):<30} dist={dist * 1000:>+7.2f} mm  ({status})"
                )
            logger.warning("\n".join(lines))
        except Exception as e:
            # Diagnostic must not crash the eval — swallow anything unexpected.
            logger.debug("drift-abort collision diagnostic failed: %s", e)

    def get_optim_params(self):
        return self.inner_policy.get_optim_params()

    def forward(self, batch, **kwargs):
        return self.inner_policy.forward(batch, **kwargs)

    def eval(self):
        self.inner_policy.eval()
        return self

    def train(self, mode=True):
        self.inner_policy.train(mode)
        return self

    def parameters(self, recurse=True):
        return self.inner_policy.parameters(recurse)

    def to(self, *args, **kwargs):
        self.inner_policy.to(*args, **kwargs)
        return self

    # For video saving compatibility (lerobot_eval.py line 280)
    def use_original_modules(self):
        if hasattr(self.inner_policy, "use_original_modules"):
            self.inner_policy.use_original_modules()
