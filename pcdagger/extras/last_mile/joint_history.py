#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Rolling joint-state ring buffer.

Used immediately by ``StallDetector`` (the "no-progress over N steps" trigger).
Also a future migration target for:
* ``SharedAutonomyPolicyWrapper._actual_q_history`` (deque) — pre-jump
  lookback for RRT q_start.
* ``InterventionController``'s stall trigger in lerobot.scripts.intervention_controller.

Both of those currently work fine; the migration is a separate refactor
(not in this commit).
"""

from __future__ import annotations

import collections
from collections.abc import Iterable

import numpy as np
import torch
from torch import Tensor


def _to_numpy_1d(q) -> np.ndarray:
    """Coerce a joint vector (Tensor, numpy array, or list) to a 1-D float32
    numpy array, detached from any autograd graph and on CPU."""
    if isinstance(q, Tensor):
        return q.detach().cpu().reshape(-1).to(dtype=torch.float32).numpy()
    arr = np.asarray(q, dtype=np.float32).reshape(-1)
    return arr


class JointHistoryBuffer:
    """Rolling ring buffer of recent raw joint states.

    Stores at most ``maxlen`` entries; older entries are dropped on push.
    All entries are stored as 1-D float32 numpy arrays of identical shape
    (asserted on push after the first entry).
    """

    def __init__(self, maxlen: int) -> None:
        if maxlen < 1:
            raise ValueError(f"maxlen must be ≥ 1, got {maxlen}")
        self._buf: collections.deque[np.ndarray] = collections.deque(maxlen=maxlen)

    @property
    def maxlen(self) -> int:
        return self._buf.maxlen  # type: ignore[return-value]

    def __len__(self) -> int:
        return len(self._buf)

    def push(self, q) -> None:
        arr = _to_numpy_1d(q)
        if len(self._buf) > 0 and arr.shape != self._buf[0].shape:
            raise ValueError(
                f"JointHistoryBuffer: shape mismatch on push — got {arr.shape}, "
                f"buffer holds {self._buf[0].shape}"
            )
        self._buf.append(arr)

    def oldest(self) -> np.ndarray:
        """Return the oldest entry. Raises if buffer is empty.

        Useful for RRT pre-jump lookback: the planner starts from a pose the
        robot was at BEFORE the inner policy's current (presumably bad) chunk
        began driving it toward a collision.
        """
        if len(self._buf) == 0:
            raise IndexError("JointHistoryBuffer is empty")
        return self._buf[0]

    def latest(self) -> np.ndarray:
        if len(self._buf) == 0:
            raise IndexError("JointHistoryBuffer is empty")
        return self._buf[-1]

    def range_l2(self) -> float:
        """L2 norm of the per-joint range (max − min) across the buffer.

        Stall metric: small value means the robot's joint state has stayed
        nearly constant over the buffer window. Returns 0.0 when the buffer
        holds fewer than 2 entries (can't measure motion yet).
        """
        if len(self._buf) < 2:
            return 0.0
        stacked = np.stack(list(self._buf), axis=0)  # [N, n_joints]
        per_joint_range = stacked.max(axis=0) - stacked.min(axis=0)
        return float(np.linalg.norm(per_joint_range))

    def clear(self) -> None:
        self._buf.clear()

    def __iter__(self) -> Iterable[np.ndarray]:
        return iter(self._buf)
