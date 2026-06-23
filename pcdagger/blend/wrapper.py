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

    # Arm joint limits (matches SplatSim PybulletRobotServerBase).
    # Override in subclass or set on instance for different robots.
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
        robot_name: str = "robot_iphone_w_engine_new",
        max_joint_delta: float = 0.016,
        num_dofs: int = 6,
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
        rrt_self_collision_skip_pairs: list[list[int]] | None = None,
        rrt_diagnostic_log_pairs: str = "off",
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
        # Frames-since-last-RRT-cycle-end counter. Used to cap the RRT
        # source's lookback so it never rewinds into a prior RRT cycle's
        # trajectory (which would teleport the env's robot to a config the
        # POLICY never actually drove through). Incremented in select_action
        # whenever RRT is IDLE (policy driving); reset to 0 the moment RRT
        # leaves IDLE (a new cycle started). Episode reset() also zeros it.
        # See rrt_source._do_plan() lookback path for the cap site.
        self._frames_since_last_rrt_end: int = 0
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

        self.num_dofs = num_dofs
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
            self_collision_skip_pairs=rrt_self_collision_skip_pairs,
            diagnostic_log_pairs=rrt_diagnostic_log_pairs,
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
            f"robot={robot_name}, "
            f"rrt_collision_detection={self._collision_detection_mode}"
        )

        # Load pybullet DIRECT client for FK+IK (same pattern as KeyboardInterfaceAgent)
        robot_config = SplatObjectConfig(name="robot", splat_name=robot_name)
        urdf_path = resolve_splatsim_path(robot_config.urdf_path)
        ee_link_name = robot_config.wrist_camera_link_name

        self._pb_client = p.connect(p.GUI if show_slider else p.DIRECT)
        # TODO(hardcoded): base_position from robot_iphone_w_engine_new config
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
        self._robot_id = p.loadURDF(
            urdf_path,
            useFixedBase=True,
            basePosition=[0, 0, -0.088],
            flags=urdf_flags,
            physicsClientId=self._pb_client,
        )
        self._ee_link = self._find_ee_link(ee_link_name)
        self._num_pb_joints = p.getNumJoints(self._robot_id, physicsClientId=self._pb_client)
        # One-time AABB log per robot link at the rest pose. Matches the
        # diagnostic we use for obstacles in load_obstacles, so you can
        # eyeball that all gripper / arm links have non-degenerate
        # collision geometry after the URDF flag change.
        self._log_robot_link_aabbs()

        # Count movable (non-fixed) joints for null-space IK arrays.
        self._num_movable_joints = sum(
            1
            for i in range(self._num_pb_joints)
            if p.getJointInfo(self._robot_id, i, physicsClientId=self._pb_client)[2] != p.JOINT_FIXED
        )

        self._obstacle_ids: list[int] = []
        self._load_static_obstacles()

        if show_slider:
            from lerobot.policies.shared_autonomy_gui import launch_ratio_slider

            launch_ratio_slider(self)

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

    def _check_future_chunk_collision(self) -> tuple[bool, int | None, str | None]:
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
            return False, None, None

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
            return False, None, None

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
                return False, None, None
            q_current_dof = np.asarray(q_current, dtype=np.float64).reshape(-1)[: self.num_dofs]

        # Inherit clearance / skip-pair config from the planner that the
        # source already constructed (so the shield's contract matches RRT's).
        planner = self._rrt_source.state.planner
        skip_pairs = None
        ob_clear = self._future_chunk_config.obstacle_clearance
        self_clear = self._future_chunk_config.self_collision_clearance
        if planner is not None:
            if ob_clear is None:
                ob_clear = planner._collision_kwargs.get("obstacle_clearance")
            if self_clear is None:
                self_clear = planner._collision_kwargs.get("self_collision_clearance")
            skip_pairs = planner._collision_kwargs.get("self_collision_skip_pairs")

        # Joint indices the planner uses for the arm DOFs in the wrapper's
        # pybullet client. The wrapper convention is 1..1+num_dofs (see
        # _sync_joints).
        joint_indices = list(range(1, 1 + self.num_dofs))

        return check_chunk_collision(
            pb_client=self._pb_client,
            robot_id=self._robot_id,
            joint_indices=joint_indices,
            q_current=q_current_dof,
            chunk_dof_actions=chunk_dof,
            action_format=action_format,
            obstacle_ids=self._obstacle_ids,
            obstacle_clearance=ob_clear,
            self_collision_clearance=self_clear,
            self_collision_skip_pairs=skip_pairs,
        )

    def _denormalize_chunk_to_raw(self, chunk: Tensor) -> np.ndarray | None:
        """Denormalize a queued action chunk to raw policy-space actions.

        The inner policy's action queue stores NORMALIZED actions of shape
        ``(n_steps, B=1, action_dim)``. We need them in RAW units (radians
        for joint dims) AND we need rel-format chunks to remain as rel
        deltas (not as per-step "where to go from current state"), so we
        bypass the AbsoluteActionsProcessorStep's add-last-state behavior
        by temporarily zeroing its state during the postprocessor pass.

        Returns ``(n_steps, action_dim)`` numpy array, or None if no
        AbsoluteActionsProcessorStep was found in the pipeline (caller
        should skip the shield in that case rather than guess at format).
        """
        # Find the AbsoluteActionsProcessorStep so we can snapshot + zero
        # the relative_step._last_state.
        abs_step = None
        for _step in self.postprocessor.steps:
            if isinstance(_step, AbsoluteActionsProcessorStep):
                abs_step = _step
                break
        if abs_step is None or not abs_step.enabled or abs_step.relative_step is None:
            # No add-state step → postprocessor output IS the denormalized
            # action directly. Run it per-step and stack.
            out_rows: list[np.ndarray] = []
            for k in range(chunk.shape[0]):
                # chunk[k] has shape (B=1, action_dim). The postprocessor
                # expects a single tensor in that shape (it's how the
                # wrapper already calls it via `self.postprocessor(inner_action)`).
                row = self.postprocessor(chunk[k]).detach().cpu().numpy().reshape(-1)
                out_rows.append(row)
            return np.stack(out_rows, axis=0) if out_rows else None

        rel_step = abs_step.relative_step
        saved_state = rel_step._last_state
        try:
            # Zero out the cached state so AbsoluteActionsProcessorStep
            # adds 0 to every action, leaving the unnormalized rel-delta
            # (or unnormalized abs target if the policy is abs-mode).
            if saved_state is not None:
                rel_step._last_state = torch.zeros_like(saved_state)
            else:
                # If no state has been cached yet, fall back to skipping
                # the abs step entirely (chunk denormalization still works
                # via the other steps in the pipeline).
                abs_step.enabled = False
            out_rows = []
            for k in range(chunk.shape[0]):
                row = self.postprocessor(chunk[k]).detach().cpu().numpy().reshape(-1)
                out_rows.append(row)
            return np.stack(out_rows, axis=0) if out_rows else None
        finally:
            # Restore the original cached state (and the enabled flag if we
            # touched it) so the next normal select_action call sees the
            # right denormalization context.
            rel_step._last_state = saved_state
            if saved_state is None:
                abs_step.enabled = True

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
        self._prev_dq = None
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
            self._rrt_source.update_oracle_config(oracle_cfg)
            self._oracle_goal_source.update_oracle_config(oracle_cfg)

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

        # Always call inner_policy.select_action to keep obs queues updated (e.g. diffusion
        # maintains n_obs_steps history in _queues). Discard output when guidance overrides.
        inner_action = self.inner_policy.select_action(batch)

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
            # observation.state is [num_dofs joints + gripper] = num_dofs+1 entries.
            self._desired_q = actual_q.reshape(-1)[: self.num_dofs + 1]
            # Preserve a copy of the actual observation for RRT's q_start —
            # _desired_q gets overwritten with the commanded action at the end
            # of select_action, so by the time the planner thread reads it the
            # value reflects "where we want the robot to go next", not "where
            # the robot is right now". When commanded ≠ actual (collision,
            # mid-chunk replay, etc.) the latter is what RRT needs.
            self._latest_actual_q = actual_q.reshape(-1)[: self.num_dofs + 1].copy()
            # Also push into the rolling history so RRT can pull q_start from
            # N steps ago (pre-jump pose), not just the current actual_q.
            self._actual_q_history.append(self._latest_actual_q.copy())
            # Track post-intervention idle frames. Bumped on every policy-
            # driven tick (RRT IDLE), zeroed the moment RRT leaves IDLE for
            # a new cycle. rrt_source._do_plan() caps its lookback sample
            # at this counter so it can't rewind into a prior RRT cycle's
            # trajectory — see the comment on the ctor field for rationale.
            # NOTE: this needs `self._rrt` (the back-compat view onto the
            # RRT source's state), which exists from ctor regardless of
            # whether RRT is the active guidance source.
            if self._rrt.mode == RRTMode.IDLE:
                self._frames_since_last_rrt_end += 1
            else:
                self._frames_since_last_rrt_end = 0
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
        if self._collision_detection_mode in ("future_chunk", "hybrid") and self._rrt.mode == RRTMode.IDLE:
            shield_collides, shield_step, shield_kind = self._check_future_chunk_collision()
            if shield_collides:
                logger.info(
                    "Future-chunk shield: predicted %s collision at chunk step %d — "
                    "triggering RRT from current state (no rewind).",
                    shield_kind or "unknown",
                    shield_step if shield_step is not None else -1,
                )
                # Flush the colliding chunk so it doesn't drain in parallel
                # with the RRT chunk execution about to start.
                self._flush_inner_action_queue()
                # Synchronous (blocking) RRT trigger from current state.
                # In future_chunk mode the source's _do_plan reads
                # q_start = wrapper._latest_actual_q and skips teleport.
                self._rrt_source.trigger(no_lookback=True)
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
                self._cancel_rrt()
                # _cancel_rrt cleared stale inner-policy + obs-blend caches.
                # Return a hold action for this tick; next tick falls through
                # to the obs-driven source naturally.
                return self.get_hold_action(inner_action)
            elif rrt.step >= len(rrt.chunk):
                # print('finish rrt: chunk exhausted (step %d >= chunk length %d)' % (rrt.step, len(rrt.chunk)))
                # Goal reached: restore prior ratio, auto-pause for the next step.
                self._finish_rrt()
                action = self.get_hold_action(inner_action)
                # Skip the existing branches below; jump to _desired_q update.
                assert self._desired_q is not None  # seeded above
                self._last_raw_action = self._desired_q.reshape(-1).copy()
                return action
            else:
                # print("rrt executing: step %d / %d" % (rrt.step, len(rrt.chunk)))
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

        return action

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
