"""End-effector kinematics helpers built around a ``SharedAutonomyPolicyWrapper``.

These run pybullet FK by re-using the wrapper's internal client (no extra sim
process required) to translate joint sequences into EE positions / deltas.

Extracted from ``visualize_shared_autonomy_DEPRECATED`` so downstream callers
don't have to depend on a deprecated module.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def compute_ee_from_states(wrapper, states_raw: np.ndarray) -> np.ndarray:
    """Run FK on a sequence of raw joint states. Returns ``[N, 3]`` EE positions."""
    positions = []
    for q in states_raw:
        wrapper._sync_joints(q[: wrapper.num_dofs])
        pos, _ = wrapper._get_ee_pose()
        positions.append(pos.copy())
    return np.array(positions)


def absolute_positions_to_ee_deltas(
    wrapper,
    init_state_raw: np.ndarray,
    target_states_raw: np.ndarray,
) -> np.ndarray:
    """Convert a sequence of absolute joint positions to EE-space deltas.

    Given an initial joint state and a sequence of target states, compute the
    7-d delta ``[dx, dy, dz, droll, dpitch, dyaw, gripper]`` between consecutive
    poses. The positional delta is expressed in the **current EE frame**
    (matching how ``_compute_next_joints`` applies it), and the rotation delta
    is the relative Euler XYZ rotation from current to next.

    ``init_state_raw``:    ``[action_dim]`` raw joints for the starting pose.
    ``target_states_raw``: ``[N, action_dim]`` raw joints for each target step.

    Returns ``[N, 7]`` array of EE deltas.
    """
    num_dofs = wrapper.num_dofs
    n_steps = target_states_raw.shape[0]
    deltas = np.zeros((n_steps, 7), dtype=np.float64)

    # Start from init state
    q_prev = init_state_raw[:num_dofs]
    wrapper._sync_joints(q_prev)
    pos_prev, quat_prev = wrapper._get_ee_pose()
    r_prev = Rotation.from_quat(quat_prev)

    for t in range(n_steps):
        q_next = target_states_raw[t, :num_dofs]
        wrapper._sync_joints(q_next)
        pos_next, quat_next = wrapper._get_ee_pose()
        r_next = Rotation.from_quat(quat_next)

        # Positional delta in EE (local) frame
        delta_pos_world = pos_next - pos_prev
        delta_pos_local = r_prev.inv().apply(delta_pos_world)

        # Rotation delta: R_delta = R_prev^{-1} * R_next
        r_delta = r_prev.inv() * r_next
        delta_rot = r_delta.as_euler("XYZ")

        # Gripper: pass the target value directly (not a delta — matches wrapper convention)
        gripper = float(target_states_raw[t, num_dofs]) if target_states_raw.shape[1] > num_dofs else 0.0

        deltas[t] = np.concatenate([delta_pos_local, delta_rot, [gripper]])

        # Advance
        pos_prev = pos_next.copy()
        r_prev = r_next
        q_prev = q_next

    return deltas


def compute_ee_trajectories(
    wrapper,
    init_obs_state_raw: np.ndarray,
    action_chunks_by_ratio: dict[float, np.ndarray],
) -> dict[float, np.ndarray]:
    """Compute end-effector XYZ positions for each ratio's action chunk using pybullet FK.

    ``init_obs_state_raw``: raw (unnormalized) joint angles from the dataset,
    shape ``[action_dim]``.

    Returns ``{ratio: np.ndarray [n_action_steps + 1, 3]}`` where index 0 is
    the initial EE position from the observation state.
    """
    wrapper._sync_joints(init_obs_state_raw[: wrapper.num_dofs])
    # init_pos, _ = wrapper._get_ee_pose()

    trajectories: dict[float, np.ndarray] = {}
    for ratio, chunk in action_chunks_by_ratio.items():
        positions = []  # Don't include the observation in the ratio action chunk
        # positions = [init_pos.copy()]
        for t in range(chunk.shape[0]):
            wrapper._sync_joints(chunk[t][: wrapper.num_dofs])
            pos, _ = wrapper._get_ee_pose()
            positions.append(pos.copy())
        trajectories[ratio] = np.array(positions)  # [n_action_steps + 1, 3]

    return trajectories
