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

import logging
import threading
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import pybullet as p
import torch
from scipy.spatial.transform import Rotation
from splatsim.configs.env_config import SplatObjectConfig
from splatsim.utils.paths import resolve_splatsim_path
from torch import Tensor, nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyProcessorPipeline

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
        self._had_guidance_last_step: bool = False
        self._desired_q: np.ndarray | None = None  # raw joint-space IK seed [num_dofs]
        self._teleop_context: TeleopRecordingContext | None = None  # set by policy factory
        self._start_paused = start_paused
        self._run_event = threading.Event()
        if not start_paused:
            self._run_event.set()

        # Wrapper-managed blended chunk buffer
        self._guided_chunk: Tensor | None = None  # [B, n_action_steps, action_dim]
        self._chunk_step: int = 99999999999  # how many steps have been returned from _guided_chunk

        self.num_dofs = num_dofs
        self._max_joint_delta = max_joint_delta
        self._prev_dq: np.ndarray | None = None  # previous joint velocity (raw, [num_dofs])
        self.skip_collision: bool = False  # set True for visualization (dataset guidance is known-safe)
        self.policy_guidance_representation = policy_guidance_representation

        logger.info(f"SharedAutonomyPolicyWrapper: ratio={forward_flow_ratio}, robot={robot_name}")

        # Load pybullet DIRECT client for FK+IK (same pattern as KeyboardInterfaceAgent)
        robot_config = SplatObjectConfig(name="robot", splat_name=robot_name)
        urdf_path = resolve_splatsim_path(robot_config.urdf_path)
        ee_link_name = robot_config.wrist_camera_link_name

        self._pb_client = p.connect(p.DIRECT)
        # TODO(hardcoded): base_position from robot_iphone_w_engine_new config
        self._robot_id = p.loadURDF(
            urdf_path, useFixedBase=True, basePosition=[0, 0, -0.088], physicsClientId=self._pb_client
        )
        self._ee_link = self._find_ee_link(ee_link_name)
        self._num_pb_joints = p.getNumJoints(self._robot_id, physicsClientId=self._pb_client)

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
        """
        # The training gripper is near-constant, so normalization has a huge scale factor
        # and any guidance with gripper outside the training range explodes after normalization.
        policy_guidance_action = policy_guidance_action.clone()

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
        self._guided_chunk = None
        self._chunk_step = 0
        self._had_guidance_last_step = False
        self._desired_q = None
        self._prev_dq = None
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

        # Extract delta guidance (7-d: [dx,dy,dz,droll,dpitch,dyaw,gripper]).
        # All-NaN means no key is held.
        guidance_chunk_raw = batch.pop(OBS_GUIDANCE_CHUNK, None)  # [B, n_remaining, action_dim] or None
        obs_state = batch.get(OBS_STATE)

        print("batch keys:", batch.keys())

        print("guidance_chunk_raw:", guidance_chunk_raw)

        if obs_state is None:
            raise RuntimeError("No obs.state available for shared autonomy wrapper")
        # TODO this is really only designed to handle 1 teleoperator and 1 policy (batch size = 1)
        assert obs_state.shape[0] == 1

        ratio = self.forward_flow_ratio
        # TODO if this is a setting with multiple envs, prob need to have has_guidance have an entry per batch
        has_guidance = guidance_chunk_raw is not None and not torch.isnan(guidance_chunk_raw).all()
        if self._teleop_context is not None:
            self._teleop_context.ratio = ratio
            self._teleop_context.has_guidance = has_guidance

        # We stay in "guided execution" mode until the current buffer is fully consumed,
        # even if the user releases the key mid-chunk.
        chunk_exhausted = self._guided_chunk is None or self._chunk_step >= self.config.n_action_steps
        draining = self._guided_chunk is not None and not chunk_exhausted
        in_guidance_mode = has_guidance or draining

        self._had_guidance_last_step = has_guidance

        # No inner policy reset needed here — the obs queue is always updated by
        # inner_policy.select_action (called unconditionally below), so it stays
        # current regardless of whether we're blending or not.

        # Always call inner_policy.select_action to keep obs queues updated (e.g. diffusion
        # maintains n_obs_steps history in _queues). Discard output when guidance overrides.
        inner_action = self.inner_policy.select_action(batch)

        # Seed _desired_q from inner_action on first call
        if self._desired_q is None:
            self._desired_q = self.postprocessor(inner_action).cpu().numpy().reshape(-1)

        # Capture q_prev BEFORE any action computation so velocity limiting sees the true
        # previous position. get_full_teleop_action pre-updates _desired_q internally,
        # so reading it afterward would give dq=0 (a no-op).
        # q_prev_for_vel_limit = self._desired_q.reshape(-1).copy() if self._desired_q is not None else None

        # --- Pure teleop (ratio=0): bypass policy chunk, do FK+IK from current state ---
        if ratio == 0.0:
            if has_guidance:
                if self.policy_guidance_representation == PolicyGuidanceRepresentation.ABSOLUTE_POS:
                    action = self._normalize_policy_guidance_action(guidance_chunk_raw[:, 0, :])
                else:
                    action = self.get_full_teleop_action(guidance_chunk_raw[:, 0, :])
            else:
                action = self.get_hold_action(inner_action)

        elif not in_guidance_mode:
            # No pending guidance: clear buffer and use inner policy output.
            # if self._guided_chunk is not None:
            #     self._guided_chunk = None
            #     self._chunk_step = 0
            action = inner_action

        # --- Guided execution mode, ratio < 1.0 ---

        # Drain path: return the next action from an existing blended chunk without
        # re-blending. Taken when:
        #   - No new guidance delta (user released key), OR
        #   - ONCE_PER_CHUNK mode and the current blended chunk is still valid.
        elif (
            self._guided_chunk is not None
            and not chunk_exhausted
            and (not has_guidance or self.blend_mode == BlendMode.ONCE_PER_CHUNK)
        ):
            action = self._guided_chunk[:, self._chunk_step, :]
            self._chunk_step += 1
            print("draining blended chunk, step", self._chunk_step)

        # Do the blend
        else:
            print("starting blended execution with new guidance, ratio", ratio)
            # max_action_dim: PI0.5 pads actions to this size; diffusion uses raw action_dim.
            max_action_dim = getattr(self.config, "max_action_dim", None)
            batch_size = guidance_chunk_raw.shape[0] if guidance_chunk_raw is not None else obs_state.shape[0]

            # Determine anchor chunk for IK:
            # - If chunk exhausted or no buffer yet: get a fresh policy chunk via predict_action_chunk.
            #   Note: inner_policy.select_action was already called above (obs queues updated for diffusion),
            #   so predict_action_chunk will read from up-to-date obs queues.
            # - Otherwise: reuse _guided_chunk so guidance accumulates on top of itself.
            if chunk_exhausted or self._guided_chunk is None:
                noise_kwargs = {"noise": base_noise} if base_noise is not None else {}
                anchor_chunk = self.inner_policy.predict_action_chunk(
                    batch, **noise_kwargs
                )  # [batch_size, chunk_size, action_dim]
                self._chunk_step = 0
                print("predicting new chunk from policy for guidance, chunk_step reset to 0")
            else:
                # The anchor is the previously blended chunk
                anchor_chunk = self._guided_chunk  # [batch_size, n_action_steps, action_dim]
                print("reusing previous blended chunk as anchor for new guidance")

            device = anchor_chunk.device
            # apply guidance to future steps only
            anchor_len = anchor_chunk.shape[1]  # chunk_size or n_action_steps
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
                self.policy_guidance_representation == PolicyGuidanceRepresentation.ABSOLUTE_POS
                and guidance_chunk_raw is not None
            ):
                # Full per-step guidance chunk provided: normalize each step and overwrite
                # the corresponding positions in the anchor. This avoids the single-point
                # ramp-to-endpoint approximation that dilutes guidance on early steps.
                # guidance_chunk_raw: [B, n_remaining, action_dim] covering [chunk_step, anchor_len)
                n_provided = guidance_chunk_raw.shape[1]
                n_remaining = anchor_len - self._chunk_step
                n_fill = min(n_provided, n_remaining)
                for t_rel in range(n_fill):
                    step_raw = guidance_chunk_raw[:, t_rel, :]  # [B, action_dim]
                    step_norm = self._normalize_policy_guidance_action(step_raw)
                    t_abs = self._chunk_step + t_rel
                    guidance_chunk[:, t_abs, :action_dim] = step_norm
                # If guidance is shorter than remaining chunk, repeat last step
                print(
                    f"Applied {n_fill} steps of provided guidance chunk to absolute positions. {n_remaining - n_fill} steps remain; repeating last provided step for those."
                    if n_fill < n_remaining
                    else f"Applied full provided guidance chunk of {n_provided} steps to absolute positions."
                )
                if n_fill < n_remaining:
                    last_norm = guidance_chunk[:, self._chunk_step + n_fill - 1, :action_dim]
                    for t_abs in range(self._chunk_step + n_fill, anchor_len):
                        guidance_chunk[:, t_abs, :action_dim] = last_norm
                print(
                    "first 10 guidance chunk after applying provided absolute positions:",
                    guidance_chunk[:, :10, :action_dim],
                )

            # elif self.policy_guidance_representation == PolicyGuidanceRepresentation.ABSOLUTE_POS:
            #     # Single-point guidance: normalize and overwrite all remaining steps uniformly.
            #     guidance_end_norm = self._normalize_policy_guidance_action(delta)  # [B, action_dim]
            #     for t_abs in range(self._chunk_step, anchor_len):
            #         guidance_chunk[:, t_abs, :action_dim] = guidance_end_norm
            #     print(
            #         "applied single-point guidance to all future steps in chunk from step", self._chunk_step
            #     )

            elif (
                self.policy_guidance_representation == PolicyGuidanceRepresentation.DELTA
                and guidance_chunk_raw is not None
            ):
                # Per-step DELTA chunk: apply each delta sequentially via FK+IK,
                # building the guidance trajectory step-by-step from _desired_q.
                # guidance_chunk_raw: [B, n_provided, 7] — each step is an EE delta
                # [dx, dy, dz, droll, dpitch, dyaw, gripper].
                guidance_chunk_np = guidance_chunk_raw.cpu().numpy()
                n_provided = guidance_chunk_np.shape[1]
                n_remaining = anchor_len - self._chunk_step
                for b in range(batch_size):
                    raw_end = self.postprocessor(anchor_chunk[:, self._chunk_step, :])
                    q_seed = (
                        self._desired_q.copy()
                        if self._desired_q is not None
                        else raw_end[b, : self.num_dofs].cpu().numpy()
                    )
                    # Apply provided deltas step-by-step
                    last_delta = guidance_chunk_np[b, min(n_provided - 1, 0)]  # fallback for padding
                    for t_rel in range(n_remaining):
                        if t_rel < n_provided:
                            d = guidance_chunk_np[b, t_rel]
                            last_delta = d
                        else:
                            d = last_delta  # keep applying the last delta
                        d_pos, d_rot, d_gripper = d[:3], d[3:6], d[6]
                        q_new = self._project_delta_for_collision(
                            q_seed, d_pos, d_rot, skip_collision=self.skip_collision
                        )
                        raw_step = np.concatenate([q_new, [float(d_gripper)]])
                        step_t = torch.tensor(raw_step, dtype=anchor_chunk.dtype, device=device).unsqueeze(0)
                        step_norm = self._normalize_policy_guidance_action(step_t)
                        t_abs = self._chunk_step + t_rel
                        guidance_chunk[:, t_abs, :action_dim] = step_norm
                        q_seed = q_new[: self.num_dofs].copy()

            # elif self.policy_guidance_representation == PolicyGuidanceRepresentation.DELTA:
            #     # Single-step DELTA: apply IK at the end of the chunk (full accumulated
            #     # delta), ramp linearly from anchor at chunk_step to guidance at last step.
            #     assert guidance_chunk_raw is not None, "DELTA mode requires a guidance chunk in the batch"
            #     raw_end = self.postprocessor(anchor_chunk[:, anchor_len - 1, :])  # [B, num_dofs+1]
            #     guidance_end = np.zeros((batch_size, action_dim), dtype=np.float64)
            #     n_remaining = anchor_len - self._chunk_step
            #     for b in range(batch_size):
            #         d = guidance_chunk_raw[b, :, :].cpu().numpy()
            #         q = self._desired_q.copy() if self._desired_q is not None else raw_end[b, : self.num_dofs]
            #         accumulated_pos = d[:3] * n_remaining
            #         r_single = Rotation.from_euler("XYZ", d[3:6])
            #         r_accumulated = Rotation.identity()
            #         for _ in range(n_remaining):
            #             r_accumulated = r_accumulated * r_single
            #         accumulated_rot = r_accumulated.as_euler("XYZ")
            #         q_new = self._project_delta_for_collision(q, accumulated_pos, accumulated_rot)
            #         guidance_end[b] = np.concatenate([q_new, [float(d[6])]])

            #     guidance_end_t = torch.tensor(guidance_end, dtype=anchor_chunk.dtype, device=device)
            #     guidance_end_norm = self._normalize_policy_guidance_action(guidance_end_t)
            #     joint_delta = guidance_end_norm - anchor_chunk[:, anchor_len - 1, :]
            #     if max_action_dim > action_dim:
            #         joint_delta = torch.cat(
            #             [
            #                 joint_delta,
            #                 torch.zeros(
            #                     batch_size,
            #                     max_action_dim - action_dim,
            #                     dtype=joint_delta.dtype,
            #                     device=device,
            #                 ),
            #             ],
            #             dim=1,
            #         )
            #     n_steps = anchor_len - self._chunk_step
            #     for t_abs in range(self._chunk_step, anchor_len):
            #         alpha = (t_abs - self._chunk_step) / max(n_steps - 1, 1)
            #         guidance_chunk[:, t_abs, :] = guidance_chunk[:, t_abs, :] + alpha * joint_delta

            else:
                raise NotImplementedError(
                    f"Unsupported policy_guidance_representation: {self.policy_guidance_representation}"
                )

            # ratio=0 is handled by the early return above; this path is only for 0 < ratio < 1.

            # build the guidance noise regardless of blend strategy to consume the rng seed
            # It's ok to do this duplicate work for interpolate because that is a debug mode, anyways
            x_tsw = self._build_guidance_noise_from_chunk(guidance_chunk, ratio, base_noise=base_noise)
            if self.guidance_blend_strategy == GuidanceBlendStrategy.INTERPOLATE:
                # Simple linear interpolation in clean action space — no denoising.
                # guidance_chunk[:, :, :action_dim] is the normalized guidance.
                # anchor_chunk is the pure policy output (normalized).
                # blended = ratio * anchor + (1-ratio) * guidance
                blended = anchor_chunk.clone()
                g = guidance_chunk[:, :, :action_dim]
                blended[:, :, :action_dim] = ratio * anchor_chunk + (1.0 - ratio) * g
                self._guided_chunk = blended
            elif self.guidance_blend_strategy == GuidanceBlendStrategy.DENOISE:
                blended = self.inner_policy.predict_action_chunk(batch, noise=x_tsw, sa_noise_ratio=ratio)
                self._guided_chunk = blended
            else:
                raise NotImplementedError(
                    f"Unsupported guidance_blend_strategy: {self.guidance_blend_strategy}"
                )

            action = self._guided_chunk[:, self._chunk_step, :]
            self._chunk_step += 1

        # self._last_action = action

        # Apply velocity limit when not in pure policy mode.
        # Converts to raw joint space (where the limit is physically meaningful in radians),
        # scales the whole joint delta vector uniformly (preserves EE direction), then
        # converts back to normalized space.
        # TODO put this velocity limit back for all ratios after bugs are ironed out
        # if ratio != 0.0 and ratio != 1.0 and q_prev_for_vel_limit is not None:
        #     """
        #     TODO after other bugs are ironed out:
        #     _desired_q seeded from policy output, not obs state: line 548 seeds it from self.postprocessor(inner_action). So the very first velocity limit step (line 688) computes dq = q_proposed - inner_action_raw, not dq = q_proposed - actual_robot_state. This can produce an artificially large delta on the first step and trigger velocity limiting incorrectly.
        #     """
        #     q_proposed = self.postprocessor(action).cpu().numpy().reshape(-1)
        #     q_prev = q_prev_for_vel_limit
        #     q_actual, dq_actual = self._apply_velocity_limit(q_proposed, q_prev)
        #     self._prev_dq = dq_actual
        #     action = self._normalize_policy_guidance_action(
        #         torch.tensor(q_actual, dtype=action.dtype, device=action.device).unsqueeze(0)
        #     )

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
