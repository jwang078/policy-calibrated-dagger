"""Characterize the forearm↔wrist_2 mesh penetration observed at drift-abort.

Loads sisbot.urdf headless into pybullet, sets all joints to the stuck config
from the diagnostic dump, then sweeps each of the 6 arm DOFs one at a time
recording getClosestPoints(forearm_link, wrist_2_link).

Prints:
  - Which DOF actually controls the pair distance
  - The joint-value range where they interpenetrate
  - The nearest safe joint value (distance ≥ threshold) in each direction
  - The min-clearance envelope over the entire arm reachable space

Usage:
  uv run python3 my_scripts/analyze_forearm_wrist2_penetration.py
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pybullet as pb

# Config that drift-aborted in the logs (chunk step-27, wrist_1 stuck):
STUCK_CONFIG = np.array([0.4641, -1.3778, 0.9901, 0.4169, 0.571, 1.4729])
COMMANDED_END = np.array([0.5393, -1.3556, 1.0868, 0.7055, 0.5641, 1.5237])

# The two links from the diagnostic:
LINK_A_NAME = "forearm_link"
LINK_B_NAME = "wrist_2_link"

# Threshold used elsewhere in the codebase (RRT self_collision_clearance).
RRT_CLEARANCE_THRESHOLD = 0.020  # 20 mm

DEFAULT_URDF = Path("/home/jennyw2/code/SplatSim/splatsim/robot_definitions/urdf/sisbot.urdf")


def load_robot(urdf_path: Path) -> tuple[int, dict[str, int], list[int]]:
    """Load sisbot.urdf headless. Return (bodyId, name->linkIdx, movable_joint_indices)."""
    pb.connect(pb.DIRECT)
    # sisbot.urdf uses relative paths — cd to its parent so meshes resolve.
    pb.setAdditionalSearchPath(str(urdf_path.parent))
    body = pb.loadURDF(
        str(urdf_path),
        useFixedBase=True,
        flags=pb.URDF_USE_SELF_COLLISION,
    )
    name_to_idx: dict[str, int] = {}
    movable: list[int] = []
    n = pb.getNumJoints(body)
    for j in range(n):
        info = pb.getJointInfo(body, j)
        joint_type = info[2]
        child_link_name = info[12].decode("utf-8")
        name_to_idx[child_link_name] = j
        if joint_type in (pb.JOINT_REVOLUTE, pb.JOINT_PRISMATIC):
            # Only the 6 UR arm joints go in the DOF vector — skip mimic gripper joints.
            # The 6 arm joint names, in order:
            if info[1].decode("utf-8") in {
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            }:
                movable.append(j)
    return body, name_to_idx, movable


def set_arm_joints(body: int, movable: list[int], q: np.ndarray) -> None:
    """Snap the 6 UR arm DOFs. Leaves gripper joints at whatever they were."""
    for idx, val in zip(movable, q, strict=True):
        pb.resetJointState(body, idx, float(val))


def pair_distance(body: int, link_a: int, link_b: int, max_dist: float = 0.10) -> float:
    """Signed nearest-point distance between two links of the same body.
    Positive = clearance, negative = interpenetration. Uses getClosestPoints
    which does not require simulation stepping."""
    pts = pb.getClosestPoints(
        bodyA=body,
        bodyB=body,
        linkIndexA=link_a,
        linkIndexB=link_b,
        distance=max_dist,
    )
    if not pts:
        return math.inf
    return min(p[8] for p in pts)  # contactDistance


def sweep_single_dof(
    body: int,
    movable: list[int],
    link_a: int,
    link_b: int,
    base_q: np.ndarray,
    dof_idx: int,
    lo: float,
    hi: float,
    n: int,
) -> np.ndarray:
    """Return array [(joint_val, distance)] as (n,2). All other DOFs held at base_q."""
    out = np.empty((n, 2), dtype=np.float64)
    vals = np.linspace(lo, hi, n)
    q = base_q.copy()
    for i, v in enumerate(vals):
        q[dof_idx] = v
        set_arm_joints(body, movable, q)
        out[i, 0] = v
        out[i, 1] = pair_distance(body, link_a, link_b)
    return out


def find_zero_crossings(sweep: np.ndarray) -> list[tuple[float, str]]:
    """Return list of (joint_val, direction) where distance crosses 0."""
    crossings: list[tuple[float, str]] = []
    for i in range(1, len(sweep)):
        d0, d1 = sweep[i - 1, 1], sweep[i, 1]
        if math.isinf(d0) or math.isinf(d1):
            continue
        if d0 == 0.0 and d1 == 0.0:
            continue
        if (d0 < 0.0) != (d1 < 0.0):
            # Linear-interp the crossing.
            t = d0 / (d0 - d1)
            v = float(sweep[i - 1, 0] + t * (sweep[i, 0] - sweep[i - 1, 0]))
            direction = "→clear" if d1 > 0 else "→penetrate"
            crossings.append((v, direction))
    return crossings


def summary_row(name: str, sweep: np.ndarray) -> str:
    """One-line summary of a sweep."""
    dmin = np.nanmin(sweep[:, 1])
    dmax = np.nanmax(sweep[:, 1])
    imin = int(np.nanargmin(sweep[:, 1]))
    imax = int(np.nanargmax(sweep[:, 1]))
    range_v = sweep[-1, 0] - sweep[0, 0]
    return (
        f"  {name:22s}  "
        f"min={dmin * 1000:+7.2f} mm @ q={sweep[imin, 0]:+.3f}   "
        f"max={dmax * 1000:+7.2f} mm @ q={sweep[imax, 0]:+.3f}   "
        f"range_q={range_v:.2f} rad"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    ap.add_argument(
        "--n_sweep",
        type=int,
        default=401,
        help="Samples per DOF sweep (default 401 → 0.0157 rad/sample over 2π).",
    )
    ap.add_argument(
        "--n_random", type=int, default=5000, help="Random configs for envelope characterization."
    )
    args = ap.parse_args()

    print(f"Loading URDF: {args.urdf}")
    body, name_to_idx, movable = load_robot(args.urdf)
    print(f"Loaded body_id={body}, num_joints={pb.getNumJoints(body)}, arm DOFs={len(movable)}")

    link_a = name_to_idx[LINK_A_NAME]
    link_b = name_to_idx[LINK_B_NAME]
    print(f"Analyzing pair: {LINK_A_NAME}({link_a})  vs  {LINK_B_NAME}({link_b})")
    print()
    dof_names = [
        "0: shoulder_pan",
        "1: shoulder_lift",
        "2: elbow",
        "3: wrist_1",
        "4: wrist_2",
        "5: wrist_3",
    ]

    # 1. Distance at the exact stuck config.
    set_arm_joints(body, movable, STUCK_CONFIG)
    d_stuck = pair_distance(body, link_a, link_b)
    set_arm_joints(body, movable, COMMANDED_END)
    d_end = pair_distance(body, link_a, link_b)
    print("Pair distance at the two configs from the drift-abort log:")
    print(f"  stuck   q={STUCK_CONFIG}   dist={d_stuck * 1000:+7.2f} mm")
    print(f"  cmd_end q={COMMANDED_END}   dist={d_end * 1000:+7.2f} mm")
    print()

    # 2. Per-DOF sweep, base = stuck config.
    print(f"Per-DOF sweep (holding other DOFs at STUCK config, {args.n_sweep} samples over [-pi, +pi]):")
    per_dof_sweeps: dict[int, np.ndarray] = {}
    for dof_idx, name in enumerate(dof_names):
        sweep = sweep_single_dof(
            body,
            movable,
            link_a,
            link_b,
            STUCK_CONFIG,
            dof_idx,
            lo=-math.pi,
            hi=math.pi,
            n=args.n_sweep,
        )
        per_dof_sweeps[dof_idx] = sweep
        print(summary_row(name, sweep))
    print()

    # 3. Which DOF's sweep dominates? Report the one with the tightest range at penetration.
    print("Zero-distance crossings (transitions between clearance and interpenetration):")
    for dof_idx, name in enumerate(dof_names):
        crossings = find_zero_crossings(per_dof_sweeps[dof_idx])
        if not crossings:
            print(f"  {name:22s}  (no zero crossings — distance sign constant along this DOF)")
            continue
        crossings_str = "  ".join(f"q={v:+.3f} rad ({d})" for v, d in crossings)
        print(f"  {name:22s}  {crossings_str}")
    print()

    # 4. For the dominant DOF, find the nearest safe value to the current stuck config.
    stuck_dof_idx_guess = 3  # from the log: worst joint 3 = wrist_1 in the DOF array
    sweep = per_dof_sweeps[stuck_dof_idx_guess]
    stuck_val = STUCK_CONFIG[stuck_dof_idx_guess]
    idx_stuck = int(np.argmin(np.abs(sweep[:, 0] - stuck_val)))
    # Nearest sample with distance ≥ RRT threshold, searching outward.
    clear_mask = sweep[:, 1] >= RRT_CLEARANCE_THRESHOLD
    if clear_mask.any():
        clear_indices = np.where(clear_mask)[0]
        nearest_clear = clear_indices[np.argmin(np.abs(clear_indices - idx_stuck))]
        clear_q = sweep[nearest_clear, 0]
        clear_d = sweep[nearest_clear, 1]
        print(
            f"Nearest wrist_1 value from stuck ({stuck_val:+.4f}) with dist ≥ {RRT_CLEARANCE_THRESHOLD * 1000:.0f} mm:"
        )
        print(f"  q={clear_q:+.4f} rad (Δ={clear_q - stuck_val:+.4f})   dist={clear_d * 1000:+.2f} mm")
    else:
        print("No wrist_1 value along this DOF gives ≥ RRT threshold clearance to the pair.")
    print()

    # 5. Random envelope over full joint-limit range: how often are they penetrating?
    print(f"Random envelope ({args.n_random} configs uniform in [-pi, +pi]^6):")
    rng = np.random.default_rng(42)
    qs = rng.uniform(-math.pi, math.pi, size=(args.n_random, 6))
    dists = np.empty(args.n_random)
    for i, q in enumerate(qs):
        set_arm_joints(body, movable, q)
        dists[i] = pair_distance(body, link_a, link_b, max_dist=0.20)
    finite = dists[np.isfinite(dists)]
    n_penetrate = int(np.sum(finite < 0.0))
    n_lt_threshold = int(np.sum(finite < RRT_CLEARANCE_THRESHOLD))
    print(f"  configs sampled:                 {args.n_random}")
    print(f"  configs with pair within 200 mm: {len(finite)}")
    print(
        f"  configs with dist < 0    (mesh overlap):        {n_penetrate:5d}  ({100 * n_penetrate / args.n_random:5.2f}%)"
    )
    print(
        f"  configs with dist < {RRT_CLEARANCE_THRESHOLD * 1000:.0f} mm (RRT threshold): {n_lt_threshold:5d}  ({100 * n_lt_threshold / args.n_random:5.2f}%)"
    )
    if len(finite):
        print(f"  min distance seen: {np.min(finite) * 1000:+.2f} mm")
    print()

    # 6. Final take.
    print("Interpretation:")
    print("  * pair distance < 0     → meshes overlap; solver can pin joint (drift source)")
    print("  * pair distance < 20 mm → RRT would reject this config IF (3,5) weren't in SKIP list")
    print("  * If a large fraction of joint-limit space penetrates, the URDF collision mesh is oversized")
    print("    (typical UR5 fix: replace forearm.stl / wrist2.stl with a slimmed convex hull)")

    pb.disconnect()


if __name__ == "__main__":
    main()
