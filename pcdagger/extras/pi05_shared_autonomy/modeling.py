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
Shared Autonomy processor for PI0.5 flow matching.

Based on "To the Noise and Back: Diffusion for Shared Autonomy" adapted for flow matching.
Reference: https://arxiv.org/abs/2302.12244
"""

import logging
from collections import deque

import torch
from torch import Tensor

from lerobot.policies.pi05.configuration_shared_autonomy import SharedAutonomyConfig

logger = logging.getLogger(__name__)


class SharedAutonomyProcessor:
    """Shared Autonomy processor for PI0.5 flow matching.

    Implements partial forward flow followed by reverse denoising,
    blending human actions with learned model behavior.

    The core idea is to adapt the "To the Noise and Back" diffusion approach
    to flow matching:

    1. Start with human action a_human
    2. Apply partial forward flow: x_tsw = t_sw * noise + (1-t_sw) * a_human
    3. Run reverse denoising from x_tsw at time t_sw down to t=0

    This preserves human intent (fidelity) while correcting toward the
    learned behavior distribution (conformity). The forward_flow_ratio
    parameter controls this trade-off.
    """

    def __init__(self, sa_config: SharedAutonomyConfig):
        """Initialize the Shared Autonomy processor.

        Args:
            sa_config: Configuration for shared autonomy behavior.
        """
        self.sa_config = sa_config
        self._human_action_buffer = deque(maxlen=sa_config.human_action_buffer_size)

        # Debug tracking (similar to RTC)
        self.tracker = None
        if sa_config.debug:
            logger.info("Shared Autonomy debug mode enabled")
            # TODO: Implement debug tracker similar to RTC if needed

    def set_human_action(self, human_action: Tensor):
        """Store human action for next inference.

        Args:
            human_action: Human-provided action tensor.
                Shape: [B, chunk_size, action_dim] or [B, action_dim]
                If 2D, will be expanded to chunk size during processing.
        """
        self._human_action_buffer.append(human_action)

    def get_human_action(self) -> Tensor | None:
        """Retrieve buffered human action.

        Returns:
            Buffered human action tensor, or None if buffer is empty.
        """
        if len(self._human_action_buffer) > 0:
            return self._human_action_buffer.popleft()
        return None

    def has_human_action(self) -> bool:
        """Check if human action is available.

        Returns:
            True if human action is buffered, False otherwise.
        """
        return len(self._human_action_buffer) > 0

    def compute_initial_state(
        self,
        noise: Tensor,
        human_action: Tensor,
    ) -> Tensor:
        """Apply partial forward flow to human action.

        This implements the forward process of flow matching:
            x_t = t * noise + (1-t) * action

        By setting t = forward_flow_ratio, we create an initial state
        that is partially corrupted with noise, preserving some of the
        human action while allowing the denoising process to correct it.

        Args:
            noise: Sampled Gaussian noise [B, chunk_size, action_dim]
            human_action: Human-provided action [B, chunk_size, action_dim]

        Returns:
            Initial state x_tsw for denoising [B, chunk_size, action_dim]

        Raises:
            ValueError: If shapes of noise and human_action don't match.
        """
        # Handle case where human_action is [B, action_dim] - broadcast to chunk
        if human_action.ndim == 2 and noise.ndim == 3:
            # Expand: [B, action_dim] -> [B, 1, action_dim] -> [B, chunk_size, action_dim]
            human_action = human_action.unsqueeze(1).expand_as(noise)

        if noise.shape != human_action.shape:
            raise ValueError(
                f"Shape mismatch: noise {noise.shape} vs human_action {human_action.shape}"
            )

        t_sw = self.sa_config.forward_flow_ratio

        # Flow matching forward: x_t = t * noise + (1-t) * action
        x_tsw = t_sw * noise + (1 - t_sw) * human_action

        if self.sa_config.debug:
            logger.debug(
                f"Applied partial forward flow with t_sw={t_sw:.3f}, "
                f"noise_weight={t_sw:.3f}, action_weight={1-t_sw:.3f}"
            )

        return x_tsw

    def modify_denoising_params(
        self,
        num_inference_steps: int,
    ) -> tuple[float, float, int]:
        """Compute modified denoising parameters for partial flow.

        Instead of denoising from t=1.0 to t=0.0, we denoise from
        t=forward_flow_ratio to t=0.0, preserving the partial forward
        flow we applied.

        Args:
            num_inference_steps: Number of denoising steps (typically 10)

        Returns:
            Tuple of (t_start, dt, num_steps):
                - t_start: Starting time for denoising (= forward_flow_ratio)
                - dt: Time step size (negative, e.g., -0.04 for t_sw=0.4)
                - num_steps: Number of denoising steps (unchanged)

        Example:
            For forward_flow_ratio=0.4 and num_steps=10:
            - t_start = 0.4
            - dt = -0.04
            - Timesteps: 0.4, 0.36, 0.32, ..., 0.08, 0.04, 0.0
        """
        t_start = self.sa_config.forward_flow_ratio
        dt = -t_start / num_inference_steps

        if self.sa_config.debug:
            logger.debug(
                f"Modified denoising: t_start={t_start:.3f}, dt={dt:.4f}, "
                f"num_steps={num_inference_steps}"
            )

        return t_start, dt, num_inference_steps

    def apply_to_first_action_only(
        self,
        actions: Tensor,
        human_action: Tensor,
    ) -> Tensor:
        """Apply shared autonomy only to the first action in the chunk.

        This is useful when human input is sparse (one action per chunk)
        but the model generates 50 actions at once. We blend the first
        action with human input and let the model generate the rest
        autonomously.

        Args:
            actions: Model-generated action chunk [B, chunk_size, action_dim]
            human_action: Human-provided action [B, action_dim]

        Returns:
            Modified actions with first action replaced [B, chunk_size, action_dim]
        """
        if self.sa_config.apply_to_first_action_only:
            # Replace only the first action
            actions_modified = actions.clone()
            if human_action.ndim == 2:
                # [B, action_dim] -> first position in chunk
                actions_modified[:, 0, :] = human_action
            else:
                # [B, chunk_size, action_dim] -> use first action from chunk
                actions_modified[:, 0, :] = human_action[:, 0, :]
            return actions_modified

        return actions
