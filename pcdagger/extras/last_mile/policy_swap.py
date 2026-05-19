#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Policy load/unload helpers for the swap-to-alt-policy backend.

Centralized so the same `del + gc.collect + torch.cuda.empty_cache` pattern
is used everywhere. Mirrors patterns in:
* policies/sarm/compute_rabc_weights.py:376,425
* my_scripts/augment_dataset_with_blending.py:615,760
"""

from __future__ import annotations

import gc
import logging
from typing import Any

import torch

from lerobot.policies.pretrained import PreTrainedPolicy

logger = logging.getLogger(__name__)


def cache_state_dict_to_cpu(policy: PreTrainedPolicy) -> dict[str, torch.Tensor]:
    """Snapshot a policy's state_dict on CPU (detached, cloned).

    The snapshot is independent of the GPU tensors — safe to call ``del policy``
    afterward without invalidating the snapshot. For PI05 (~3.4 B params,
    bfloat16) this costs ~6.8 GB host RAM; for ACT-sized policies it's
    negligible.
    """
    return {k: v.detach().to("cpu", copy=True) for k, v in policy.state_dict().items()}


def unload_policy_inplace(owner: Any, attr_name: str) -> None:
    """Drop a policy reference from ``owner.attr_name`` and free its GPU memory.

    Sets the attribute to ``None``, runs ``gc.collect`` so the released GPU
    tensors become reclaimable, then ``torch.cuda.empty_cache`` to actually
    hand the memory back to the allocator. No-op if the attribute is
    already None.
    """
    if getattr(owner, attr_name, None) is None:
        return
    setattr(owner, attr_name, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_policy_from_path(path: str, device: str | torch.device) -> PreTrainedPolicy:
    """Load a policy from a pretrained checkpoint dir, move to device, eval()."""
    logger.info("Loading alt policy from %s", path)
    alt = PreTrainedPolicy.from_pretrained(path)
    alt = alt.to(device)
    alt.eval()
    return alt


def restore_policy_from_state(
    config,
    state_dict: dict[str, torch.Tensor],
    device: str | torch.device,
) -> PreTrainedPolicy:
    """Rebuild a policy from a saved config + CPU state_dict, move to device.

    Uses the same instantiation path as ``PreTrainedPolicy.from_pretrained``
    but skips the disk read — feeds the cached state_dict in directly.
    """
    from lerobot.policies.factory import get_policy_class

    policy_cls = get_policy_class(config.type)
    policy = policy_cls(config)
    policy.load_state_dict(state_dict, strict=False)
    policy = policy.to(device)
    policy.eval()
    return policy
