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
Shared Autonomy configuration class for PI0.5 flow matching.

Based on "To the Noise and Back: Diffusion for Shared Autonomy" adapted for flow matching.
Reference: https://arxiv.org/abs/2302.12244
"""

from dataclasses import dataclass


@dataclass
class SharedAutonomyConfig:
    """Configuration for Shared Autonomy inference.

    Implements "To the Noise and Back" ideas adapted for flow matching,
    allowing human input to be blended with model predictions via partial
    forward flow followed by reverse denoising.

    The key parameter is forward_flow_ratio (t_sw), which controls the
    trade-off between fidelity (preserving human intent) and conformity
    (following learned behavior distribution):

    - t_sw = 0.0: No intervention, return human action directly (high fidelity)
    - t_sw = 0.4: Moderate blending (recommended default)
    - t_sw = 1.0: Full model control, ignore human input (high conformity)

    Flow matching forward process: x_t = t * noise + (1-t) * action
    Starting from x_tsw instead of pure noise preserves human intent.
    """

    # Infrastructure
    enabled: bool = False

    # Core shared autonomy settings
    forward_flow_ratio: float = 0.4  # t_sw switching time (0.0-1.0)

    # Human input settings
    human_action_buffer_size: int = 1
    apply_to_first_action_only: bool = True  # Only blend first action in chunk

    # Debug settings
    debug: bool = False
    debug_maxlen: int = 100

    def __post_init__(self):
        """Validate shared autonomy configuration parameters."""
        if not 0.0 <= self.forward_flow_ratio <= 1.0:
            raise ValueError(
                f"forward_flow_ratio must be in [0, 1], got {self.forward_flow_ratio}"
            )
        if self.human_action_buffer_size <= 0:
            raise ValueError(
                f"human_action_buffer_size must be positive, got {self.human_action_buffer_size}"
            )
        if self.debug_maxlen <= 0:
            raise ValueError(f"debug_maxlen must be positive, got {self.debug_maxlen}")
