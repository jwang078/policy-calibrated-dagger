#!/usr/bin/env python3
"""Audit the SELF_COLLISION_SKIP_PAIRS list on the small_engine URDF.

Purpose: identify entries in ``SmallEnginePybulletRobotServer.SELF_COLLISION_SKIP_PAIRS``
whose two links AREN'T actually structurally overlapping — pairs that vary
across joint configs and can enter genuine self-collision at runtime, but are
being ignored by the planner because they were added to the skip list under
the "structural" bucket in error. Those are the false-skips that produce the
"planner accepts config → physics disagrees at runtime → joint spike"
symptom.

Method:
  1. Load ``sisbot.urdf`` in a fresh DIRECT-mode PyBullet client (no env
     setup, no obstacles).
  2. Sample joint configs from up to two sources (union):
       * uniform-random over each joint's URDF limits (broad C-space
         coverage — includes wrist rotations outside the workload's
         typical distribution).
       * per-frame joint states from an optional parquet dataset
         (workload-representative — captures the joint distribution the
         real recorder actually produces).
  3. For every non-adjacent link pair (a<b), record ``getClosestPoints``
     ``contactDistance`` at every sample. Pairs are enumerated over
     ``range(-1, num_joints)`` — same iteration as ``check_links_in_collision``.
  4. Classify each pair by the distance distribution:
       IN_SKIPS_STRUCTURAL — currently skipped, range < 1 mm AND max < 0
         → constant URDF overlap, correctly skipped.
       IN_SKIPS_SUSPECT   — currently skipped, range > 10 mm OR
                             (min < 0 AND max > 0) → pair articulates in
                             and out of collision — CANDIDATE TO UNSKIP.
       IN_SKIPS_BORDERLINE — currently skipped, 1 mm ≤ range ≤ 10 mm →
                             judgment call; keep for now.
       UNSKIPPED_CLEAR    — not skipped, min > 5 mm → correctly unskipped.
       UNSKIPPED_TIGHT    — not skipped, 0 ≤ min ≤ 5 mm → checked
                             normally; the planner already rejects any
                             samples where these penetrate.
       UNSKIPPED_PEN_SEEN — not skipped, min < 0 → real self-collision
                             surface observed at some sample. If this
                             fires, the planner is correctly rejecting
                             those configs today — nothing to do.

The ONLY actionable class is IN_SKIPS_SUSPECT — those are what to prune.

Usage:
    # Baseline: 5k uniform random configs (~30 seconds).
    python my_scripts/audit_self_collision_skip_pairs.py

    # Add workload samples from a parquet dataset (recommended before
    # pruning — a pair might only articulate under a specific workload):
    python my_scripts/audit_self_collision_skip_pairs.py \\
        --parquet ~/.cache/huggingface/lerobot/JennyWWW/splatsim_approach_lever_12_clean/data/chunk-000/file-000.parquet

    # Show every non-adjacent pair (not just skipped ones) — surfaces
    # UNSKIPPED_PEN_SEEN cases (pairs that penetrate but the planner
    # correctly catches):
    python my_scripts/audit_self_collision_skip_pairs.py --check_all_pairs

    # Focus on a specific bucket by name filter:
    python my_scripts/audit_self_collision_skip_pairs.py \\
        --link_name_filter finger

Notes on interpretation:
  * The classification thresholds (1 mm / 10 mm / 5 mm) are heuristics. When
    the pair sits near the boundary, the workload-representative samples
    (parquet) are the authoritative signal — that's what the recorder will
    actually produce at runtime.
  * A large negative min value (e.g. −13 mm) with a small range indicates
    a URDF mesh artifact — legitimately structural, keep skipped.
  * A pair that shows min=−5 mm max=+50 mm indicates a joint config where
    the meshes DO penetrate exists in the sampled distribution — the
    planner is silently accepting it because the pair is skipped. This is
    exactly the class producing joint spikes.

Deps: pybullet, numpy, pyarrow (only if --parquet is used).
"""

from __future__ import annotations

import argparse
import itertools
import random
import sys
import time
from pathlib import Path

import numpy as np
import pybullet as p

# ── Skip list snapshot ────────────────────────────────────────────────────
# Snapshot of SmallEnginePybulletRobotServer.SELF_COLLISION_SKIP_PAIRS
# (see /home/jennyw2/code/SplatSim/splatsim/robots/sim_robot_pybullet_small_engine.py:47).
# Inlined so the audit script can run without importing the full splatsim
# env stack. Post-audit prune (17 pairs). Keep in sync when the class
# attribute changes — the audit's IN_SKIPS_* classification buckets key
# off this list; drift silently changes what the audit reports.
CURRENT_SKIP_PAIRS = [
    (4, 6),
    # Re-added CRITICAL pairs — kinematically penetrate but the physics
    # engine tolerates WITHOUT producing solver kicks for THESE two pairs.
    # (0, 2) base_link is dynamics-free; (3, 5) forearm↔wrist_2 is a UR
    # wrist geometry floor with no measurable kick. Tested in traj-gen.
    #
    # PAIRS DELIBERATELY NOT ADDED — the wrist-camera / wrist_1 mesh
    # overlaps DO kick the solver when RRT paths pass through them:
    #   (2, 4)  upper_arm ↔ wrist_1     — extreme arm-curl configs
    #   (3, 19) forearm ↔ wrist_camera  — original joint-spike pair
    #   (4, 19) wrist_1 ↔ wrist_camera  — same wrist-cam mesh class
    # If you re-add these to the skip list, expect teleports + trailing
    # joint spikes in the trajectory-gen output. Sim's `is_robot_in_collision`
    # (eval-terminate) will fire on them, which is the correct behavior.
    (0, 2),
    (3, 5),
    (6, 19),
    (7, 19),
    (5, 7),
    (5, 8),
    (5, 9),
    (5, 13),
    (5, 14),
    (5, 18),
    (6, 9),
    (6, 13),
    (6, 14),
    (6, 18),
    (11, 13),
    (12, 13),
    (16, 18),
    (17, 18),
]

# Pairs that the audit classifies as CRITICAL (they penetrate) but that
# we've DELIBERATELY kept skipped because the penetration is a URDF mesh
# artifact — physics tolerates it, no solver kicks, no observed joint
# spikes. Kept in a set so `_classify` can emit `EXPECTED_MESH_ARTIFACT`
# instead of `CRITICAL_MUST_UNSKIP`, and `--assert_clean` doesn't false-
# fail on these entries. If a NEW pair shows CRITICAL_MUST_UNSKIP in the
# audit output, it's a real one to investigate (not one of these known-
# artifact overlaps).
KNOWN_MESH_ARTIFACT_PAIRS = {
    frozenset((0, 2)),  # base_link ↔ upper_arm — mesh overlap at shoulder
    # (base has no dynamics that get kicked)
    frozenset((3, 5)),  # forearm ↔ wrist_2 — UR wrist floor (~12 mm)
    # (no measurable solver kick observed)
    # ⚠ Wrist-camera / wrist_1 pairs are NOT whitelisted — see
    # CURRENT_SKIP_PAIRS's "PAIRS DELIBERATELY NOT ADDED" note.
    # The audit's CRITICAL_MUST_UNSKIP flag correctly identifies (2, 4),
    # (3, 19), and (4, 19) as genuine collisions the planner must reject.
}


# Canonical URDF path in the SplatSim tree. Uses the robot_definitions
# variant because it INCLUDES the wrist_camera_link (referenced by index 19
# in the current skip list). The submodule pybullet-playground URDF omits
# the camera and so would fail every (X, 19) pair audit.
DEFAULT_URDF = Path("/home/jennyw2/code/SplatSim/splatsim/robot_definitions/urdf/sisbot.urdf")

# Classification thresholds (metres). See the module docstring for rationale.
STRUCTURAL_RANGE_MAX = 0.001  # 1 mm — max range for "constant / URDF-fixed"
# 20 mm — min distance a pair must maintain across ALL sampled configs to
# qualify as REDUNDANT_UNSKIP_OK (safe to unskip because it never gets
# anywhere near the runtime self_collision_clearance buffer). Bumped from
# 10 mm to 20 mm to match the wider obstacle_clearance the runtime env
# uses — the audit should be conservative and treat "min > 10 mm" as
# BORDERLINE (still within one clearance-buffer's width of collision)
# rather than definitely-safe.
SAFELY_CLEAR_MIN = 0.020  # 20 mm
UNSKIPPED_TIGHT_MAX = 0.005  # 5 mm

# Query threshold for getClosestPoints. Must be larger than any real
# robot-self distance we care about; smaller queries return () when the
# actual distance exceeds the query, which we can't distinguish from
# "closest points is far but we don't know how far". 1 m is comfortably
# above any UR5e self distance.
QUERY_DISTANCE_M = 1.0


def _load_robot(urdf: Path, gui: bool = False) -> tuple[int, int, list[int], list[tuple[float, float]]]:
    """Bring up a DIRECT-mode pybullet client, load sisbot.urdf, return
    ``(pb_client_id, robot_id, movable_joint_indices, joint_limits)``.

    ``movable_joint_indices`` is the list of joint indices with type ≠ FIXED
    — those are the ones we can sample. ``joint_limits`` is a same-length
    list of ``(lower, upper)`` from the URDF (falls back to ±π for continuous
    joints whose URDF limits are 0..0).
    """
    cid = p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(str(urdf.parent), physicsClientId=cid)
    flags = p.URDF_USE_SELF_COLLISION
    robot_id = p.loadURDF(
        str(urdf),
        useFixedBase=True,
        flags=flags,
        physicsClientId=cid,
    )
    n_joints = p.getNumJoints(robot_id, physicsClientId=cid)
    movable: list[int] = []
    limits: list[tuple[float, float]] = []
    for j in range(n_joints):
        info = p.getJointInfo(robot_id, j, physicsClientId=cid)
        # info[2] is joint type. Fixed = 4.
        if info[2] == p.JOINT_FIXED:
            continue
        movable.append(j)
        lo, hi = float(info[8]), float(info[9])
        # URDF continuous joints often report lo==hi==0; treat that as ±π.
        if hi <= lo:
            lo, hi = -np.pi, np.pi
        limits.append((lo, hi))
    return cid, robot_id, movable, limits


def _parse_mimic_joints(urdf: Path) -> dict[str, tuple[str, float, float]]:
    """Return ``{mimic_joint_name: (parent_joint_name, multiplier, offset)}``
    parsed from the URDF's <mimic> elements. PyBullet ignores mimic
    constraints at runtime — this lets the audit enforce them manually so
    the sampled configs are physically achievable (a real Robotiq 2F-85
    can't articulate its 6 gripper joints independently).
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(str(urdf))
    root = tree.getroot()
    out: dict[str, tuple[str, float, float]] = {}
    for joint in root.findall("joint"):
        mim = joint.find("mimic")
        if mim is None:
            continue
        jname = joint.get("name") or ""
        pname = mim.get("joint") or ""
        mult = float(mim.get("multiplier", "1.0"))
        offs = float(mim.get("offset", "0.0"))
        if jname and pname:
            out[jname] = (pname, mult, offs)
    return out


def _joint_name(cid: int, robot_id: int, j: int) -> str:
    info = p.getJointInfo(robot_id, j, physicsClientId=cid)
    return info[1].decode("utf-8")


def _build_mimic_index_map(
    cid: int,
    robot_id: int,
    movable: list[int],
    name_to_mimic: dict[str, tuple[str, float, float]],
) -> tuple[dict[int, tuple[int, float, float]], set[int]]:
    """Convert the name-keyed mimic table into ``{mimic_joint_index:
    (parent_joint_index, multiplier, offset)}`` and return the set of
    INDEPENDENT joint indices (movable joints that are NOT mimics)."""
    name_to_idx: dict[str, int] = {}
    for j in movable:
        name_to_idx[_joint_name(cid, robot_id, j)] = j
    mimic_idx: dict[int, tuple[int, float, float]] = {}
    for mim_name, (parent_name, mult, offs) in name_to_mimic.items():
        if mim_name in name_to_idx and parent_name in name_to_idx:
            mimic_idx[name_to_idx[mim_name]] = (name_to_idx[parent_name], mult, offs)
    independent = {j for j in movable if j not in mimic_idx}
    return mimic_idx, independent


def _link_name(cid: int, robot_id: int, link_i: int) -> str:
    if link_i == -1:
        return "base_link(-1)"
    info = p.getJointInfo(robot_id, link_i, physicsClientId=cid)
    return f"{info[12].decode('utf-8')}({link_i})"


def are_adjacent_links(cid: int, robot_id: int, a: int, b: int) -> bool:
    """URDF adjacency: two links are adjacent iff one is the direct
    parent of the other in the joint tree. Matches the check used inside
    ``check_links_in_collision`` — adjacent pairs are filtered out because
    the URDF joint constraint keeps them permanently in contact by design.

    Link index i corresponds to joint i's child link. Base link is -1 and
    has no getJointInfo entry — we test it by treating "child of joint j
    == -1" as the base-adjacent signal.
    """
    if a == b:
        return True
    # For a joint index y (y >= 0), info[16] is its PARENT LINK index.
    # For the base link (-1), there's no getJointInfo call to make; we
    # cover base-adjacency in the branch below that queries the OTHER
    # link's parent instead.
    for x, y in ((a, b), (b, a)):
        if y < 0:
            continue  # base link has no joint entry — handled via the other order
        info = p.getJointInfo(robot_id, y, physicsClientId=cid)
        parent = info[16]
        if parent == x:
            return True
    return False


def _sample_uniform(n: int, limits: list[tuple[float, float]], rng: random.Random) -> list[list[float]]:
    """n independent uniform-random configs across ``limits``.

    NOTE: this samples EVERY movable joint independently, including URDF
    mimic joints. The caller must apply mimic constraints via
    ``_apply_with_mimics`` before doing collision queries — otherwise the
    gripper's driver-follower structure is violated and the finger meshes
    read as impossibly-configured.
    """
    out = []
    for _ in range(n):
        out.append([rng.uniform(lo, hi) for lo, hi in limits])
    return out


def _sample_from_parquet(path: Path, n_movable: int) -> list[list[float]]:
    """Read ``observation.state`` from a LeRobot parquet dataset. Returns
    the raw joint arrays truncated/padded to ``n_movable`` DOFs so we can
    apply them to the URDF's movable joints. Returns an empty list if
    the file can't be read.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print(f"[warn] pyarrow not available — skipping parquet sample source {path}")
        return []
    try:
        tbl = pq.read_table(str(path), columns=["observation.state"])
    except Exception as e:
        print(f"[warn] parquet read failed ({path}): {e}")
        return []
    col = tbl.column("observation.state").to_pylist()
    out = []
    for row in col:
        arr = list(row)
        # Recorder stores arm+gripper state; audit URDF has arm+gripper
        # movable joints. Align by prefix — if arr is shorter than n_movable,
        # left-pad with 0 (gripper stays open); if longer, truncate.
        if len(arr) < n_movable:
            arr = arr + [0.0] * (n_movable - len(arr))
        elif len(arr) > n_movable:
            arr = arr[:n_movable]
        out.append(arr)
    return out


def _apply(cid: int, robot_id: int, movable: list[int], q: list[float]) -> None:
    for j, qi in zip(movable, q):
        p.resetJointState(robot_id, j, float(qi), 0.0, physicsClientId=cid)


def _apply_with_mimics(
    cid: int,
    robot_id: int,
    movable: list[int],
    q: list[float],
    mimic_idx: dict[int, tuple[int, float, float]],
) -> None:
    """Apply a joint config while enforcing URDF mimic constraints.

    Step 1: set every joint to its sampled value (raw). This puts the
    parent joints at their sampled value; mimic joints are also set but
    will be immediately overwritten below.
    Step 2: for each mimic joint, look up its parent's applied value and
    overwrite the mimic joint with ``parent * multiplier + offset``.
    That makes the sampled config physically achievable regardless of
    the raw values the sampler picked for the mimic joints.
    """
    _apply(cid, robot_id, movable, q)
    for mim_j, (parent_j, mult, offs) in mimic_idx.items():
        parent_val = p.getJointState(robot_id, parent_j, physicsClientId=cid)[0]
        p.resetJointState(robot_id, mim_j, parent_val * mult + offs, 0.0, physicsClientId=cid)


def _pair_distance(cid: int, robot_id: int, a: int, b: int) -> float:
    """Closest-point distance between robot_id link a and link b. Uses a
    1 m query threshold so we always get a result (falls back to +inf if
    getClosestPoints returns empty — shouldn't happen at 1 m but guarded
    anyway)."""
    pts = p.getClosestPoints(
        robot_id,
        robot_id,
        QUERY_DISTANCE_M,
        linkIndexA=a,
        linkIndexB=b,
        physicsClientId=cid,
    )
    if not pts:
        return float("inf")
    # Contact tuple layout: [8] is contactDistance (negative = penetration).
    return min(pt[8] for pt in pts)


def _classify(pair: tuple[int, int], stats: dict, skipped: bool) -> str:
    """Classify a pair by the sampled distance distribution.

    Priorities for the audit:
      * CRITICAL_MUST_UNSKIP — currently skipped but the pair reaches
        NEGATIVE distance (penetration) at some sampled config → the
        planner is silently accepting a physically-colliding pose here.
        This is the class producing joint spikes.
      * STRUCTURAL_KEEP     — currently skipped, distance is essentially
        constant across all configs (range < 1 mm) → real URDF mesh
        artifact, correctly skipped.
      * REDUNDANT_UNSKIP_OK — currently skipped, distance varies but
        stays comfortably far (min > SAFELY_CLEAR_MIN = 20 mm) → no real
        risk of collision, the skip has no observable effect. Safe to
        remove or keep.
      * BORDERLINE          — currently skipped, distance stays > 0 but
        can come within 20 mm of collision → judgment call: could touch
        the runtime self_collision_clearance buffer under some workload.
      * UNSKIPPED_PEN_SEEN  — not skipped and does penetrate somewhere.
        Planner correctly rejects those configs → nothing to do.
      * UNSKIPPED_CLEAR     — not skipped and never near collision.
      * UNSKIPPED_TIGHT     — not skipped, gets close but doesn't
        penetrate; planner catches the near-miss.
    """
    if skipped:
        if stats["min"] < 0.0 and stats["range"] > STRUCTURAL_RANGE_MAX:
            # Actively penetrating AND range wide → normally CRITICAL, but
            # override to EXPECTED_MESH_ARTIFACT when this pair is
            # explicitly whitelisted as a URDF-mesh-overlap artifact that
            # PyBullet's constraint solver tolerates without producing
            # forces. Whitelisted pairs are still shown in the report but
            # don't trip --assert_clean (they're a design decision, not a
            # regression). See KNOWN_MESH_ARTIFACT_PAIRS at the top of
            # this module for the current set + rationale.
            if frozenset(pair) in KNOWN_MESH_ARTIFACT_PAIRS:
                return "EXPECTED_MESH_ARTIFACT"
            return "CRITICAL_MUST_UNSKIP"
        if stats["range"] < STRUCTURAL_RANGE_MAX:
            # Range essentially zero (mm-level oscillation is fine) →
            # constant URDF geometry. Whether the constant is negative
            # (mesh overlap) or positive (rigid attachment) doesn't
            # matter — the pair doesn't articulate.
            return "STRUCTURAL_KEEP"
        if stats["min"] > SAFELY_CLEAR_MIN:
            # Wide range but stays comfortably clear of collision (min >
            # SAFELY_CLEAR_MIN = 20 mm) — skip has no effect on runtime
            # because the pair is always further apart than any realistic
            # self_collision_clearance value. Safe to remove.
            return "REDUNDANT_UNSKIP_OK"
        return "BORDERLINE"
    if stats["min"] < 0.0:
        return "UNSKIPPED_PEN_SEEN"
    if stats["min"] > UNSKIPPED_TIGHT_MAX:
        return "UNSKIPPED_CLEAR"
    return "UNSKIPPED_TIGHT"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urdf", type=Path, default=DEFAULT_URDF, help=f"sisbot URDF. Default {DEFAULT_URDF}.")
    ap.add_argument("--n_samples", type=int, default=5000, help="Uniform-random samples across joint limits.")
    ap.add_argument(
        "--parquet",
        type=Path,
        default=None,
        help="Optional LeRobot parquet — sample observation.state from this to add workload-representative configs.",
    )
    ap.add_argument(
        "--check_all_pairs",
        action="store_true",
        help="Also check pairs NOT in the current skip list — surfaces UNSKIPPED_PEN_SEEN cases.",
    )
    ap.add_argument(
        "--link_name_filter",
        type=str,
        default=None,
        help="Restrict output to pairs where at least one link name contains this substring (case-insensitive).",
    )
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for the uniform sampler.")
    ap.add_argument("--gui", action="store_true", help="Run with pybullet GUI (for visual inspection).")
    ap.add_argument(
        "--sort_by",
        choices=["range", "min", "class", "pair"],
        default="class",
        help="Column to sort the report by.",
    )
    ap.add_argument(
        "--assert_clean",
        action="store_true",
        help="Regression-guard mode: exit non-zero if any pair in "
        "the current skip list classifies as CRITICAL_MUST_UNSKIP "
        "on the sampled distribution. Intended for CI / periodic "
        "re-runs to catch regressions where a future URDF change "
        "or workload shift causes a skipped pair to start "
        "penetrating.",
    )
    args = ap.parse_args()

    if not args.urdf.exists():
        print(f"ERROR: URDF not found at {args.urdf}", file=sys.stderr)
        return 2

    cid, robot_id, movable, limits = _load_robot(args.urdf, gui=args.gui)
    n_movable = len(movable)
    n_joints = p.getNumJoints(robot_id, physicsClientId=cid)
    print(f"[urdf] loaded {args.urdf.name}: {n_joints} joints total, {n_movable} movable")
    print(f"[urdf] movable joint indices: {movable}")
    print(f"[urdf] joint limits (rad): {[(round(lo, 2), round(hi, 2)) for lo, hi in limits]}")

    # Parse mimic constraints from the URDF so sampled configs are
    # physically achievable. Without this, the raw sampler picks
    # independent values for each of the 6 Robotiq gripper joints, and
    # the finger meshes end up in impossible relative poses — spuriously
    # tripping the mesh-overlap check.
    mimic_map_by_name = _parse_mimic_joints(args.urdf)
    mimic_idx, independent = _build_mimic_index_map(cid, robot_id, movable, mimic_map_by_name)
    if mimic_idx:
        print(f"[mimic] {len(mimic_idx)} mimic joint(s) will be derived from parents at query time:")
        for mim_j, (par_j, mult, offs) in mimic_idx.items():
            print(
                f"        {_joint_name(cid, robot_id, mim_j):>28} = "
                f"{_joint_name(cid, robot_id, par_j)} * {mult:+g} + {offs:+g}"
            )

    # Assemble sample sources.
    rng = random.Random(args.seed)
    configs = _sample_uniform(args.n_samples, limits, rng)
    parquet_added = 0
    if args.parquet is not None:
        parquet_configs = _sample_from_parquet(args.parquet, n_movable)
        configs.extend(parquet_configs)
        parquet_added = len(parquet_configs)
    print(f"[samples] {args.n_samples} uniform + {parquet_added} parquet = {len(configs)} total configs")

    # Enumerate pairs.
    skipped_set = {frozenset((a, b)) for a, b in CURRENT_SKIP_PAIRS}
    link_indices = list(range(-1, n_joints))
    all_pairs = list(itertools.combinations(link_indices, 2))
    # Filter: non-adjacent only (adjacency is skipped by check_links_in_collision).
    pairs = [(a, b) for a, b in all_pairs if not are_adjacent_links(cid, robot_id, a, b)]
    if not args.check_all_pairs:
        pairs = [(a, b) for a, b in pairs if frozenset((a, b)) in skipped_set]
    if args.link_name_filter:
        sub = args.link_name_filter.lower()
        pairs = [
            (a, b)
            for a, b in pairs
            if sub in _link_name(cid, robot_id, a).lower() or sub in _link_name(cid, robot_id, b).lower()
        ]
    print(f"[pairs] auditing {len(pairs)} pair(s) (--check_all_pairs={args.check_all_pairs})")

    # Measure. Per-pair over all configs.
    t0 = time.time()
    per_pair_dists: dict[tuple[int, int], list[float]] = {pair: [] for pair in pairs}
    for i, q in enumerate(configs):
        _apply_with_mimics(cid, robot_id, movable, q, mimic_idx)
        for pair in pairs:
            per_pair_dists[pair].append(_pair_distance(cid, robot_id, *pair))
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  ...{i + 1}/{len(configs)} configs ({elapsed:.1f}s)")

    # Compute stats + classify.
    rows = []
    for pair in pairs:
        d = np.array(per_pair_dists[pair], dtype=np.float64)
        stats = {
            "min": float(d.min()),
            "max": float(d.max()),
            "range": float(d.max() - d.min()),
            "mean": float(d.mean()),
            "std": float(d.std()),
            "frac_pen": float(np.mean(d < 0.0)),
            "frac_close": float(np.mean(d < 0.005)),
        }
        cls = _classify(pair, stats, skipped=frozenset(pair) in skipped_set)
        rows.append({"pair": pair, "cls": cls, **stats})

    # Sort.
    def _key(r):
        if args.sort_by == "range":
            return -r["range"]
        if args.sort_by == "min":
            return r["min"]
        if args.sort_by == "pair":
            return r["pair"]
        # class: group by classification, then by range desc.
        order = [
            "CRITICAL_MUST_UNSKIP",  # actionable first
            "UNSKIPPED_PEN_SEEN",
            "BORDERLINE",
            "UNSKIPPED_TIGHT",
            "EXPECTED_MESH_ARTIFACT",
            "REDUNDANT_UNSKIP_OK",
            "STRUCTURAL_KEEP",
            "UNSKIPPED_CLEAR",
        ]
        return (order.index(r["cls"]) if r["cls"] in order else 99, -r["range"])

    rows.sort(key=_key)

    # Report.
    print()
    print("=" * 130)
    print(
        f"{'pair':<16} {'class':<20} {'link_a':<28} {'link_b':<28} "
        f"{'min(mm)':>10} {'max(mm)':>10} {'range(mm)':>12} {'frac_pen':>10}"
    )
    print("-" * 130)
    for r in rows:
        a, b = r["pair"]
        print(
            f"({a:>3},{b:>3})".ljust(16),
            f"{r['cls']:<20}",
            f"{_link_name(cid, robot_id, a):<28}",
            f"{_link_name(cid, robot_id, b):<28}",
            f"{r['min'] * 1000:>10.2f}",
            f"{r['max'] * 1000:>10.2f}",
            f"{r['range'] * 1000:>12.2f}",
            f"{r['frac_pen']:>10.3f}",
        )
    print("=" * 130)

    # Summary counts by class.
    from collections import Counter

    counts = Counter(r["cls"] for r in rows)
    print("\nClass summary:")
    for cls, ct in counts.most_common():
        print(f"  {cls:<22} {ct}")

    # Actionable lists (in priority order).
    critical = [r for r in rows if r["cls"] == "CRITICAL_MUST_UNSKIP"]
    if critical:
        print(f"\n▶ {len(critical)} pair(s) MUST BE REMOVED — planner is masking real self-collisions:")
        for r in critical:
            a, b = r["pair"]
            print(
                f"    ({a}, {b})  # {_link_name(cid, robot_id, a)} ↔ {_link_name(cid, robot_id, b)} "
                f"— range {r['range'] * 1000:.1f} mm, min {r['min'] * 1000:.2f} mm, "
                f"{r['frac_pen'] * 100:.1f}% of samples penetrated"
            )
    else:
        print(
            "\n▶ No CRITICAL_MUST_UNSKIP pairs — no active penetration masking in the sampled distribution."
        )

    redundant = [r for r in rows if r["cls"] == "REDUNDANT_UNSKIP_OK"]
    if redundant:
        print(
            f"\n▶ {len(redundant)} pair(s) safe to remove "
            f"(min > {SAFELY_CLEAR_MIN * 1000:.0f} mm — never near collision):"
        )
        for r in redundant:
            a, b = r["pair"]
            print(
                f"    ({a}, {b})  # {_link_name(cid, robot_id, a)} ↔ {_link_name(cid, robot_id, b)} "
                f"— range {r['range'] * 1000:.1f} mm, min {r['min'] * 1000:.2f} mm"
            )

    borderline = [r for r in rows if r["cls"] == "BORDERLINE"]
    if borderline:
        print(
            f"\n▶ {len(borderline)} BORDERLINE pair(s) — min < {SAFELY_CLEAR_MIN * 1000:.0f} mm "
            f"but no penetration in sample; keep for now:"
        )
        for r in borderline:
            a, b = r["pair"]
            print(
                f"    ({a}, {b})  # {_link_name(cid, robot_id, a)} ↔ {_link_name(cid, robot_id, b)} "
                f"— range {r['range'] * 1000:.1f} mm, min {r['min'] * 1000:.2f} mm"
            )
    print(
        f"\n[timing] {time.time() - t0:.1f}s total ({len(configs)} configs × {len(pairs)} pairs = {len(configs) * len(pairs)} distance queries)"
    )

    p.disconnect(physicsClientId=cid)

    # Regression-guard: fail loudly if any skipped pair is masking a real
    # collision. Runs after the report so the operator can see WHICH pair
    # regressed before the non-zero exit.
    if args.assert_clean and critical:
        print(
            f"\n[FAIL] --assert_clean: {len(critical)} pair(s) in the skip "
            f"list are masking real penetration. See CRITICAL_MUST_UNSKIP "
            f"rows above.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
