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

Extracts policy_guidance_action (a 7-d delta vector [dx,dy,dz,droll,dpitch,dyaw,gripper])
from the observation dict, then applies FK→IK guidance to the full predicted action chunk
and re-runs partial diffusion/flow-matching denoising with the guided chunk as the noise
anchor. This means guidance is applied coherently across the entire action window.

Works with any noise/flow-based policy (PI0.5, Diffusion) without modifying lerobot_eval.py.
"""

import logging
import threading

import numpy as np
import pybullet as p
import torch
from scipy.spatial.transform import Rotation
from torch import Tensor, nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyProcessorPipeline

logger = logging.getLogger(__name__)

OBS_HUMAN_ACTION = "observation.policy_guidance_action"
OBS_STATE = "observation.state"


def _launch_ratio_slider(wrapper: "SharedAutonomyPolicyWrapper") -> None:
    """Launch a Tkinter slider in a background daemon thread to live-edit forward_flow_ratio."""
    import tkinter as tk

    def _run():
        root = tk.Tk()
        root.title("Shared Autonomy")
        root.resizable(False, False)

        tk.Label(root, text="forward_flow_ratio", font=("Helvetica", 12)).pack(padx=16, pady=(12, 0))

        var = tk.DoubleVar(value=wrapper.forward_flow_ratio)
        label = tk.Label(root, text=f"{wrapper.forward_flow_ratio:.2f}", font=("Courier", 14, "bold"))
        label.pack(pady=(0, 4))

        def on_change(val):
            ratio = round(float(val), 2)
            wrapper.forward_flow_ratio = ratio
            label.config(text=f"{ratio:.2f}")

        slider = tk.Scale(
            root,
            from_=0.0,
            to=1.0,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            length=300,
            variable=var,
            command=on_change,
            showvalue=False,
        )
        slider.pack(padx=16, pady=(0, 16))

        tk.Label(root, text="0 = pure guidance    1 = pure policy", font=("Helvetica", 9), fg="gray").pack(
            pady=(0, 12)
        )

        root.mainloop()

    t = threading.Thread(target=_run, daemon=True, name="sa-ratio-slider")
    t.start()


class SharedAutonomyPolicyWrapper(PreTrainedPolicy):
    """Wraps a policy to blend human EE-delta guidance with diffusion/flow policy output.

    The keyboard agent sends a 7-d delta [dx,dy,dz,droll,dpitch,dyaw,gripper] as
    observation.policy_guidance_action (or all-NaN when no key is held).

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

    def __init__(
        self,
        inner_policy: PreTrainedPolicy,
        inverse_postprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        inverse_preprocessor: PolicyProcessorPipeline | None,
        forward_flow_ratio: float,
        show_slider: bool = True,
        robot_name: str = "robot_iphone_w_engine_new",
        max_joint_delta: float = 0.02,
        num_dofs: int = 6,
    ):
        # Bypass PreTrainedPolicy.__init__ — we proxy the inner policy's config
        nn.Module.__init__(self)
        self.config = inner_policy.config
        self.inner_policy = inner_policy
        self.inverse_postprocessor = inverse_postprocessor
        self.postprocessor = postprocessor  # normalized → raw joints
        self.inverse_preprocessor = inverse_preprocessor  # normalized obs.state → raw joints
        self.forward_flow_ratio = forward_flow_ratio
        self._last_action: Tensor | None = None
        self._had_guidance_last_step: bool = False

        # Wrapper-managed blended chunk buffer
        self._guided_chunk: Tensor | None = None  # [B, n_action_steps, action_dim]
        self._chunk_step: int = 99999999999  # how many steps have been returned from _guided_chunk

        self.num_dofs = num_dofs
        self._max_joint_delta = max_joint_delta

        logger.info(f"SharedAutonomyPolicyWrapper: ratio={forward_flow_ratio}, robot={robot_name}")

        # Load pybullet DIRECT client for FK+IK (same pattern as KeyboardInterfaceAgent)
        from splatsim.configs.env_config import SplatObjectConfig
        from splatsim.utils.paths import resolve_splatsim_path

        robot_config = SplatObjectConfig(name="robot", splat_name=robot_name)
        urdf_path = resolve_splatsim_path(robot_config.urdf_path)
        ee_link_name = robot_config.wrist_camera_link_name

        self._pb_client = p.connect(p.DIRECT)
        self._robot_id = p.loadURDF(urdf_path, useFixedBase=True, physicsClientId=self._pb_client)
        self._ee_link = self._find_ee_link(ee_link_name)
        self._num_pb_joints = p.getNumJoints(self._robot_id, physicsClientId=self._pb_client)

        if show_slider:
            _launch_ratio_slider(self)

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

    def _compute_next_joints(self, q: np.ndarray, delta_pos: np.ndarray, delta_rot: np.ndarray) -> np.ndarray:
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

        n = self._num_pb_joints
        joint_poses = p.calculateInverseKinematics(
            self._robot_id,
            self._ee_link,
            target_pos,
            target_quat,
            restPoses=rest + [0.0] * (n - self.num_dofs),
            jointDamping=[0.1] * n,
            physicsClientId=self._pb_client,
        )
        q_ik = np.array(joint_poses[: self.num_dofs])
        if np.max(np.abs(q_ik - q)) > 0.15:
            return q  # reject singularity / far branch
        delta_q = np.clip(q_ik - q, -self._max_joint_delta, self._max_joint_delta)
        return q + delta_q

    # ---- policy helpers ---------------------------------------------------- #

    def _normalize_policy_guidance_action(self, policy_guidance_action: Tensor) -> Tensor:
        """Normalize raw policy guidance action to policy's internal space.

        Zero-fills NaN/Inf dimensions (e.g., gripper always closed in training data
        where normalization stats have zero variance).
        """
        normalized = self.inverse_postprocessor(policy_guidance_action)
        bad = ~torch.isfinite(normalized)
        if bad.any():
            logger.warning(
                f"inverse_postprocessor produced {bad.sum().item()} non-finite value(s) "
                f"(NaN/Inf) in policy_guidance_action. Zeroing affected entries. "
                f"Check normalization stats for zero-variance dims."
            )
            normalized = normalized.masked_fill(bad, 0.0)
        out_of_range = normalized.abs() > 2.0
        if out_of_range.any():
            bad_dims = out_of_range.nonzero(as_tuple=False)
            logger.warning(
                f"policy_guidance_action has {out_of_range.sum().item()} value(s) outside [-2, 2] "
                f"after normalization (max abs: {normalized.abs().max().item():.2f}). "
                f"Offending dims (indices): {bad_dims.tolist()}. "
                f"Values: {normalized[out_of_range].tolist()}. Clamping to [-2, 2]."
            )
            normalized = normalized.clamp(-2.0, 2.0)
        return normalized

    def _build_guidance_noise_from_chunk(
        self, guidance_chunk: Tensor, ratio: float
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
            t_sw = int(ratio * noise_scheduler.config.num_train_timesteps)
            t_tensor = torch.full((batch_size,), t_sw, dtype=torch.long, device=device)
            x_tsw = noise_scheduler.add_noise(full_guidance, full_noise, t_tensor)
            return x_tsw, ratio

        # --- Flow matching (PI0.5) path ---
        if getattr(self.config, "max_action_dim", None) is None:
            return None  # policy doesn't expose needed config
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
        noise = torch.randn_like(guidance_chunk)
        x_tsw = ratio * noise + (1.0 - ratio) * guidance_chunk
        return x_tsw, ratio

    def reset(self):
        self._guided_chunk = None
        self._chunk_step = 0
        self._last_action = None
        self._had_guidance_last_step = False
        return self.inner_policy.reset()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        return self.inner_policy.predict_action_chunk(batch, **kwargs)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        # Extract delta guidance (7-d: [dx,dy,dz,droll,dpitch,dyaw,gripper]).
        # All-NaN means no key is held.
        print("\n\n--- Shared Autonomy: select_action called ---")
        policy_guidance_delta = batch.pop(OBS_HUMAN_ACTION, None)
        print("curr observation.state", batch.get(OBS_STATE))

        ratio = self.forward_flow_ratio

        has_guidance = policy_guidance_delta is not None and not torch.isnan(policy_guidance_delta).all()

        # We stay in "guided execution" mode until the current buffer is fully consumed,
        # even if the user releases the key mid-chunk.
        n_action_steps = getattr(self.config, "n_action_steps", 1)
        chunk_exhausted = self._chunk_step >= n_action_steps
        draining = self._guided_chunk is not None and not chunk_exhausted
        in_guidance_mode = has_guidance or draining

        print(
            "n_action_steps",
            n_action_steps,
            "chunk_step",
            self._chunk_step,
            "has_guidance",
            has_guidance,
            "in_guidance_mode",
            in_guidance_mode,
        )

        self._had_guidance_last_step = has_guidance

        # Reset inner policy exactly once when guidance session begins so its obs queue
        # starts accumulating from this point. By the time guidance ends (after n_action_steps),
        # the inner policy's obs queue is fresh and its action queue is empty — seamless handoff.
        guidance_just_started = has_guidance and self._guided_chunk is None and self._chunk_step == 0
        if guidance_just_started:
            self.inner_policy.reset()

        # Always call inner_policy.select_action to keep obs queues updated (e.g. diffusion
        # maintains n_obs_steps history in _queues). Discard output when guidance overrides.
        inner_action = self.inner_policy.select_action(batch)

        if ratio == 1.0 or not in_guidance_mode:
            # Pure policy or no pending guidance: clear buffer and use inner policy output.
            if self._guided_chunk is not None:
                self._guided_chunk = None
                self._chunk_step = 0
            action = inner_action
            if ratio == 0.0 and not has_guidance and self._last_action is not None:
                action = self._last_action  # pure-human mode: hold last position
                print("Holding last action (pure human mode), action", action)
            else:
                print("Pure policy control (no guidance applied), action", action)

            self._last_action = action
            return action

        print(" --- Shared Autonomy: applying guidance ---")

        # --- Guided execution mode, ratio < 1.0 ---

        # Fast path: buffer draining with no new guidance delta — advance step and return.
        if not has_guidance and self._guided_chunk is not None:
            action = self._guided_chunk[:, self._chunk_step, :]
            self._chunk_step += 1
            self._last_action = action
            print("Draining guided chunk buffer... (no new guidance), action", action)
            return action

        # max_action_dim: PI0.5 pads actions to this size; diffusion uses raw action_dim.
        max_action_dim = getattr(self.config, "max_action_dim", None)
        batch_size = policy_guidance_delta.shape[0]

        # Determine anchor chunk for IK:
        # - If chunk exhausted or no buffer yet: get a fresh policy chunk via predict_action_chunk.
        #   Note: inner_policy.select_action was already called above (obs queues updated for diffusion),
        #   so predict_action_chunk will read from up-to-date obs queues.
        # - Otherwise: reuse _guided_chunk so guidance accumulates on top of itself.
        if chunk_exhausted or self._guided_chunk is None:
            anchor_chunk = self.inner_policy.predict_action_chunk(
                batch
            )  # [batch_size, chunk_size, action_dim]
            print(
                "New anchor chunk from policy (fresh), anchor_chunk shape",
                anchor_chunk.shape,
                "first action",
                anchor_chunk[:, 0, :].cpu().numpy(),
            )
            self._chunk_step = 0
            # To sync the reset time of the inner policy with this guidance policy, reset the observation queues of the
        else:
            anchor_chunk = self._guided_chunk  # [batch_size, n_action_steps, action_dim]
            print(
                "Using cached anchor chunk (accumulated)",
                anchor_chunk.shape,
                "first action",
                anchor_chunk[:, 0, :].cpu().numpy(),
            )

        delta_np = policy_guidance_delta.cpu().numpy()  # [batch_size, 7]
        device = anchor_chunk.device
        t_start = self._chunk_step  # apply guidance to future steps only
        anchor_len = anchor_chunk.shape[1]  # chunk_size or n_action_steps

        # Clone anchor and apply IK delta in-place to steps [t_start, anchor_len).
        # Zero-pad the last dim to max_action_dim if needed (required by PI0.5 model input).
        guidance_chunk = anchor_chunk.clone()
        action_dim = anchor_chunk.shape[2]
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

        # Apply IK at the last step of the chunk (full accumulated delta), then ramp up
        # linearly from 0 at t_start to the full delta at the last step.
        # This preserves curvature near the current step while steering the end of the
        # chunk toward the human's desired EE direction.
        raw_end = self.postprocessor(anchor_chunk[:, anchor_len - 1, :])  # [batch_size, num_dofs+1]
        raw_end_np = raw_end.cpu().numpy()
        step_end = np.zeros((batch_size, anchor_chunk.shape[2]), dtype=np.float64)
        n_remaining = anchor_len - t_start
        for b in range(batch_size):
            d = delta_np[b]
            if np.isnan(d).any():
                step_end[b] = raw_end_np[b]
            else:
                q = raw_end_np[b, : self.num_dofs].astype(float)
                # Accumulate n_remaining steps of delta: position scales linearly,
                # rotation composes n_remaining times (small-angle euler).
                accumulated_pos = d[:3] * n_remaining
                r_single = Rotation.from_euler("XYZ", d[3:6])
                r_accumulated = Rotation.identity()
                for _ in range(n_remaining):
                    r_accumulated = r_accumulated * r_single
                accumulated_rot = r_accumulated.as_euler("XYZ")
                q_new = self._compute_next_joints(q, accumulated_pos, accumulated_rot)
                step_end[b] = np.concatenate([q_new, [float(d[6])]])

        g_end = torch.tensor(step_end, dtype=anchor_chunk.dtype, device=device)  # [B, action_dim]
        norm_end = self._normalize_policy_guidance_action(g_end)  # [B, action_dim]

        # Joint-space delta at last step (normalized), padded to max_action_dim.
        joint_delta = norm_end - anchor_chunk[:, anchor_len - 1, :]  # [B, action_dim]
        if max_action_dim > action_dim:
            joint_delta = torch.cat(
                [
                    joint_delta,
                    torch.zeros(
                        batch_size, max_action_dim - action_dim, dtype=joint_delta.dtype, device=device
                    ),
                ],
                dim=1,
            )

        n_steps = anchor_len - t_start
        for t in range(t_start, anchor_len):
            alpha = (t - t_start) / max(n_steps - 1, 1)  # 0.0 at t_start, 1.0 at last step
            guidance_chunk[:, t, :] = guidance_chunk[:, t, :] + alpha * joint_delta

        # Apply noise scheduling and re-run guided denoising.
        noise_result = self._build_guidance_noise_from_chunk(guidance_chunk, ratio)
        if noise_result is None:
            raise NotImplementedError(
                "Inner policy does not support noise injection for guided execution. "
                "Please use a compatible policy (e.g. diffusion with noise_scheduler, or flow model with max_action_dim) or set forward_flow_ratio=1.0 for pure policy control."
            )
        x_tsw, sa_ratio = noise_result
        print("Built guidance noise with ratio", sa_ratio, "x_tsw shape", x_tsw.shape)

        blended = self.inner_policy.predict_action_chunk(
            batch, noise=x_tsw, sa_noise_ratio=sa_ratio
        )  # [B, n_action_steps, the dim of the action action (not necessarily action_dim, which is the max action size)]
        self._guided_chunk = blended
        print(
            "Blended chunk from policy with guidance noise (normalized), blended shape",
            blended.shape,
            "first action",
            blended[:, 0, :].cpu().numpy(),
        )
        # import pdb; pdb.set_trace()

        action = self._guided_chunk[:, self._chunk_step, :]
        print("Returning chunk_step", self._chunk_step, "from blended chunk as next action, action", action)
        self._chunk_step += 1

        self._last_action = action
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
