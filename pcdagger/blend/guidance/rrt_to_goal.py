"""RRT-to-goal planner for the shared autonomy wrapper.

Wraps SplatSim's RRT (`splatsim.utils.rrt_path_utils.get_path`) and ruckig time
parametrization with a small interface tailored to the wrapper's needs:

  * Loads obstacle bodies from a serialized env config (sent over ZMQ from the
    SplatSim server) into a caller-provided pybullet client.
  * Plans a joint-space path from the current joints to a fixed goal config.
  * Returns waypoints sampled at the wrapper's control rate.

The planner does not own a pybullet client. It writes into the client owned by
the wrapper (created in ``SharedAutonomyPolicyWrapper.__init__``); this client
is private to the wrapper and never touches the env-side simulator, so the
planner works whether the env is sim or real-robot.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import threading
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pybullet as p

logger = logging.getLogger(__name__)


class RRTMode(Enum):
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"


@dataclass
class RRTRuntimeState:
    """All RRT-mode state owned by the shared autonomy wrapper.

    Bundled into a single dataclass so the wrapper class stays tidy. Access from
    the wrapper looks like ``self._rrt.mode``, ``self._rrt.chunk[i]``, etc.
    """

    mode: RRTMode = RRTMode.IDLE
    chunk: np.ndarray | None = None  # [T, num_dofs] joint waypoints
    step: int = 0  # next index into chunk
    # Optional hint set by the caller (e.g. the intervention controller)
    # BEFORE triggering: the number of waypoints the caller intends to
    # execute before cancelling and handing control back. Used purely for
    # informative logging ("executing X / Y waypoints"); the wrapper itself
    # always drains the full chunk unless explicitly cancelled.
    target_steps: int | None = None
    cancel_requested: bool = False
    planner: RRTToGoalPlanner | None = None
    oracle_env_config: dict | None = None
    oracle_config_hash: str | None = None  # detect config changes
    lock: threading.Lock = field(default_factory=threading.Lock)


class RRTPlanningError(RuntimeError):
    """Raised when RRT planning fails with a recognizable cause (start/goal
    in collision, no path found within iteration budget, etc.)."""


def _hash_config(cfg: dict) -> str:
    # Hash is for cache invalidation only, not security. usedforsecurity=False
    # silences bandit's B324 warning about SHA1.
    return hashlib.sha1(repr(cfg).encode("utf-8"), usedforsecurity=False).hexdigest()


class RRTToGoalPlanner:
    """Plans a joint-space trajectory to a fixed goal using SplatSim's RRT.

    Owns no pybullet client; all calls write into ``pb_client``. Not thread-safe
    on its own — the caller serializes access (the wrapper holds an RRT lock and
    only invokes the planner from a single worker thread).
    """

    def __init__(
        self,
        pb_client: int,
        robot_id: int,
        joint_indices: list[int],
        ee_link_index: int,
        num_dofs: int,
        fps: int,
        lower_limits: np.ndarray | None = None,
        upper_limits: np.ndarray | None = None,
        num_ik_candidates: int = 16,
        max_joint_vel: float = 0.5,
        max_joint_acc: float = 1.0,
        max_joint_jerk: float = 10.0,
    ) -> None:
        self._pb_client = pb_client
        self._robot_id = robot_id
        self._joint_indices = list(joint_indices)
        self._ee_link_index = ee_link_index
        self._num_dofs = num_dofs
        self._fps = fps
        self._lower_limits = (
            np.asarray(lower_limits, dtype=np.float64)
            if lower_limits is not None
            else -np.pi * np.ones(num_dofs)
        )
        self._upper_limits = (
            np.asarray(upper_limits, dtype=np.float64)
            if upper_limits is not None
            else np.pi * np.ones(num_dofs)
        )
        self._num_ik_candidates = num_ik_candidates
        self._max_joint_vel = max_joint_vel
        self._max_joint_acc = max_joint_acc
        self._max_joint_jerk = max_joint_jerk
        self._loaded_obstacle_ids: list[int] = []  # only oracle-loaded bodies
        self._obstacle_names: dict[int, str] = {}
        self._skip_pairs: set[tuple[int, int]] = set()
        self._loaded_config_hash: str | None = None
        # PyBullet's calculateInverseKinematics expects null-space arrays
        # (lowerLimits / upperLimits / jointDamping / restPoses) sized to the
        # number of MOVABLE joints in the URDF, not just the arm DOFs we plan
        # over. Cache the count once so the IK calls below can pad correctly.
        self._num_movable_joints = sum(
            1
            for j in range(p.getNumJoints(self._robot_id, physicsClientId=self._pb_client))
            if p.getJointInfo(self._robot_id, j, physicsClientId=self._pb_client)[2] != p.JOINT_FIXED
        )

    # ------------------------------------------------------------------ #
    #  Obstacle loading                                                  #
    # ------------------------------------------------------------------ #

    def load_obstacles(self, env_config: dict) -> list[int]:
        """Populate the wrapper's pybullet client with obstacles from ``env_config``.

        Idempotent: hashes the input and short-circuits when the config hasn't
        changed since the last load. On a cache miss, removes only the bodies
        this planner previously loaded (leaves any wrapper-owned hardcoded
        fallback obstacles untouched), then loads the new set.

        Returns the list of pybullet body IDs for the oracle obstacles.
        """
        cfg_hash = _hash_config(env_config)
        if cfg_hash == self._loaded_config_hash:
            return list(self._loaded_obstacle_ids)
        logger.info("load_obstacles: cache miss (hash %s) — loading fresh", cfg_hash[:12])

        # Tear down previously-loaded oracle bodies.
        for body_id in self._loaded_obstacle_ids:
            with contextlib.suppress(p.error):
                p.removeBody(body_id, physicsClientId=self._pb_client)
        self._loaded_obstacle_ids.clear()
        self._obstacle_names.clear()
        self._skip_pairs.clear()

        for obj in env_config.get("objects", []):
            body_id = self._load_one_object(obj)
            if body_id is None:
                continue
            self._loaded_obstacle_ids.append(body_id)
            name = obj.get("name", f"body_{body_id}")
            self._obstacle_names[body_id] = name
            for link_idx in obj.get("skip_collision_robot_links") or []:
                self._skip_pairs.add((int(link_idx), body_id))
            # Verify the collision shape actually exists at the expected place
            # by reading the body's AABB in our pybullet client. If a body has
            # no collision shape, getAABB returns the base-frame AABB only
            # (~zero-volume), which is the smoking gun for "planner ignores
            # this obstacle".
            try:
                num_links = p.getNumJoints(body_id, physicsClientId=self._pb_client)
                aabbs = [
                    p.getAABB(body_id, linkIndex=link_i, physicsClientId=self._pb_client)
                    for link_i in range(-1, num_links)
                ]
                logger.info(
                    "_load_one_object: name=%s body_id=%d num_links=%d AABBs=%s",
                    name,
                    body_id,
                    num_links,
                    aabbs,
                )
            except p.error as e:
                logger.warning("getAABB failed for %s (body_id=%d): %s", name, body_id, e)

        self._loaded_config_hash = cfg_hash
        logger.info(
            "RRTToGoalPlanner.load_obstacles: %d obstacle(s) loaded (%s)",
            len(self._loaded_obstacle_ids),
            ", ".join(self._obstacle_names.values()) or "<none>",
        )
        return list(self._loaded_obstacle_ids)

    def _load_one_object(self, obj: dict) -> int | None:
        """Create a pybullet body for ``obj`` (a serialized ObjectConfig dict).

        Returns the body id, or None if the type is unsupported / no collision
        geometry is available.
        """
        obj_type = obj.get("__type__", "")
        position = self._resolve_position(obj)
        quat = self._resolve_quat(obj)
        scale = obj.get("current_scale") or [1.0, 1.0, 1.0]
        logger.info(
            "_load_one_object: name=%s type=%s position=%s quat=%s scale=%s "
            "(raw: current_pos=%s, base_pos=%s, range_x=%s, range_y=%s, range_z=%s)",
            obj.get("name"),
            obj_type,
            position,
            quat,
            scale,
            obj.get("current_position"),
            obj.get("base_position"),
            obj.get("position_range_x"),
            obj.get("position_range_y"),
            obj.get("position_range_z"),
        )

        if obj_type == "CuboidObjectConfig":
            # SplatSim's create_box uses the raw size with no scale multiplication
            # (CuboidObjectConfig has no scaling_range; size is fixed at load time).
            size = obj.get("size") or (1.0, 1.0, 1.0)
            half = [s / 2.0 for s in size]
            shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=self._pb_client)
            return p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=shape,
                basePosition=list(position),
                baseOrientation=list(quat),
                physicsClientId=self._pb_client,
            )

        if obj_type == "SplatObjectConfig":
            urdf_path = obj.get("urdf_path")
            if not urdf_path:
                logger.debug("Skipping splat obstacle '%s' (no urdf_path)", obj.get("name"))
                return None
            try:
                from splatsim.utils.paths import resolve_splatsim_path

                resolved = resolve_splatsim_path(urdf_path)
            except Exception:
                resolved = urdf_path
            # Mirror SplatSim's two-step load: open the URDF at `base_position`
            # with identity orientation (matches SplatSim's `load_urdf`), then
            # call `resetBasePositionAndOrientation` to teleport it to the
            # final placement with the actual quaternion (matches their
            # `randomize_object_pose` which always applies the final pose via
            # reset). Some URDFs bake in mesh transforms that interact badly
            # with a non-identity quaternion at load time, so combining both
            # steps into a single loadURDF call can render the visual mesh in
            # the wrong place even though the link origin is correct.
            base_position = obj.get("base_position") or [0.0, 0.0, 0.0]
            # PyBullet only supports uniform globalScaling. SplatSim approximates
            # per-axis scaling with the geometric mean (cbrt of product); see
            # randomize_object_scale in sim_robot_pybullet_base.py.
            if scale:
                physics_scale = float(np.cbrt(float(scale[0]) * float(scale[1]) * float(scale[2])))
            else:
                physics_scale = 1.0
            try:
                body_id = p.loadURDF(
                    str(resolved),
                    basePosition=list(base_position),
                    baseOrientation=[0.0, 0.0, 0.0, 1.0],
                    useFixedBase=True,
                    globalScaling=physics_scale,
                    physicsClientId=self._pb_client,
                )
            except p.error as e:
                logger.warning("Failed to load splat obstacle '%s' from %s: %s", obj.get("name"), resolved, e)
                return None
            p.resetBasePositionAndOrientation(
                body_id,
                list(position),
                list(quat),
                physicsClientId=self._pb_client,
            )
            return body_id

        logger.debug("Unsupported object type '%s' (name=%s) — skipping", obj_type, obj.get("name"))
        return None

    @staticmethod
    def _resolve_position(obj: dict) -> tuple[float, float, float]:
        """Mirror SplatSim's pose-placement formula.

        ``randomize_object_pose`` (sim_robot_pybullet_base.py) computes
        ``pos = [x + bp[0], y + bp[1], z + bp[2]]`` where ``(x,y,z)`` is the
        sample from ``position_range_*`` (or its midpoint when no random) and
        ``bp = base_position``. Picking just one of the two — as the previous
        implementation did — drops the YAML-provided height offset
        (e.g. ``small_engine_new`` has ``base_position=[0, 0, 0.180955]`` to
        sit on the table; the table-relative xy comes from position_range).

        Preference order:
          1. ``current_position`` if the sim has updated it past the default
             [0,0,0] (i.e. a get_observations has fired post-placement).
          2. ``initial_position`` (set by ``randomize_object_pose`` at episode
             start) for the same reason.
          3. ``base_position + position_range_midpoint`` as the formula
             fallback when neither live field is populated yet.
        """
        for key in ("current_position", "initial_position"):
            v = obj.get(key)
            if v and any(abs(float(x)) > 1e-12 for x in v):
                return (float(v[0]), float(v[1]), float(v[2]))
        bp = obj.get("base_position") or [0.0, 0.0, 0.0]
        rx = obj.get("position_range_x") or (0.0, 0.0)
        ry = obj.get("position_range_y") or (0.0, 0.0)
        rz = obj.get("position_range_z") or (0.0, 0.0)
        return (
            float(bp[0]) + (float(rx[0]) + float(rx[1])) / 2.0,
            float(bp[1]) + (float(ry[0]) + float(ry[1])) / 2.0,
            float(bp[2]) + (float(rz[0]) + float(rz[1])) / 2.0,
        )

    @staticmethod
    def _resolve_quat(obj: dict) -> tuple[float, float, float, float]:
        """Pick the most informative orientation, with the same precedence as
        ``_resolve_position``. Skips current_quat / initial_quat when they're
        still the identity default (since asdict will always include them as
        [0,0,0,1] until the sim updates them); falls through to base_quat,
        which is YAML-provided and meaningful for static configs.
        """
        identity = (0.0, 0.0, 0.0, 1.0)
        for key in ("current_quat", "initial_quat"):
            v = obj.get(key)
            if v and len(v) == 4 and any(abs(float(x) - d) > 1e-9 for x, d in zip(v, identity, strict=True)):
                return (float(v[0]), float(v[1]), float(v[2]), float(v[3]))
        bq = obj.get("base_quat")
        if bq and len(bq) == 4:
            return (float(bq[0]), float(bq[1]), float(bq[2]), float(bq[3]))
        return identity

    # ------------------------------------------------------------------ #
    #  Planning                                                          #
    # ------------------------------------------------------------------ #

    def plan(
        self,
        q_start: np.ndarray,
        target_ee_pos: np.ndarray,
        target_ee_quat: np.ndarray,
        q_goal_bias: np.ndarray | None = None,
    ) -> np.ndarray:
        """Plan a joint-space trajectory to an end-effector pose.

        Mirrors SplatSim's ``TrajectoryGenerator._resolve_ee_pose_to_q_candidates``
        + ``_plan_with_fallback_goals``: solves IK to ``(target_ee_pos, target_ee_quat)``
        from multiple seeds — the first attempt seeded by ``q_goal_bias`` (so demos
        converge to a canonical configuration when feasible), the rest from random
        seeds — and runs RRT against each collision-free candidate until one
        succeeds.  This is much less constrained than fixing ``q_goal_bias`` as
        the only goal, which often has no valid path through cluttered scenes.

        Returns waypoints of shape (T, num_dofs) sampled at ``self._fps``.
        Raises ``RRTPlanningError`` if no IK candidate can be reached.
        """
        from splatsim.utils.rrt_path_utils import (
            check_links_in_collision,
            get_path,
            ruckig_parametrize_path,
        )

        q_start = np.asarray(q_start, dtype=np.float64).reshape(-1)[: self._num_dofs]
        target_ee_pos = np.asarray(target_ee_pos, dtype=np.float64).reshape(-1)[:3]
        target_ee_quat = np.asarray(target_ee_quat, dtype=np.float64).reshape(-1)[:4]
        if q_goal_bias is not None:
            q_goal_bias = np.asarray(q_goal_bias, dtype=np.float64).reshape(-1)[: self._num_dofs]

        # Snapshot every joint (positions and velocities) so we can fully restore
        # the robot's pose after planning — the wrapper's pybullet client is
        # shared with FK/IK code paths and would otherwise be left in an
        # arbitrary state.
        n_joints = p.getNumJoints(self._robot_id, physicsClientId=self._pb_client)
        saved_joint_states: list[tuple[int, float, float]] = []
        for i in range(n_joints):
            s = p.getJointState(self._robot_id, i, physicsClientId=self._pb_client)
            saved_joint_states.append((i, float(s[0]), float(s[1])))

        # Disable viewport rendering while planning. The IK candidate sampler
        # and RRT itself call resetJointState many times — on a GUI client
        # each one triggers a redraw, which is both visually noisy ("the robot
        # jumps to random positions") and a major slowdown. No-op for DIRECT
        # clients, so this is safe to leave on always.
        with contextlib.suppress(p.error):
            p.configureDebugVisualizer(
                p.COV_ENABLE_RENDERING,
                0,
                physicsClientId=self._pb_client,
            )

        try:
            # If the start config is in collision (the policy got stuck), try to
            # escape along the aggregated outward contact normal before planning.
            # The robot is the only thing in the wrapper's pybullet client besides
            # the loaded obstacles, so all contacts are robot↔obstacle.
            escape_path: np.ndarray | None = None
            if check_links_in_collision(
                self._robot_id,
                self._joint_indices,
                q_start,
                self._loaded_obstacle_ids,
                obstacle_names=self._obstacle_names,
                skip_pairs=self._skip_pairs,
                verbose=True,
                physics_client_id=self._pb_client,
            ):
                logger.info("Start config in collision; attempting contact-normal escape...")
                escape_path = self._escape_collision(q_start)
                if escape_path is None:
                    raise RRTPlanningError("Start in collision and contact-normal escape failed to clear it")
                logger.info("Escape produced %d waypoint(s); replanning from new start", len(escape_path))
                q_start = escape_path[-1].copy()

            # Resolve the target EE pose to multiple collision-free joint-space
            # candidates. First attempt is seeded with q_goal_bias (when provided).
            candidates = self._resolve_ee_pose_to_q_candidates(target_ee_pos, target_ee_quat, q_goal_bias)
            if len(candidates) == 0:
                raise RRTPlanningError("No collision-free IK solution found for target EE pose")
            logger.info("Resolved EE goal to %d collision-free IK candidate(s)", len(candidates))

            # Try RRT against EVERY candidate, score each by joint-space arc
            # length, keep the shortest. Compared to "stop at first success",
            # this is more expensive (we plan all candidates) but produces
            # noticeably tidier corrections — DAgger data quality matters more
            # than per-cycle latency for our use case.
            path = None
            chosen_q_goal = None
            best_length = float("inf")
            for i, q_goal in enumerate(candidates):
                logger.info("Trying RRT against IK candidate %d/%d", i + 1, len(candidates))
                attempt = get_path(
                    q_start,
                    q_goal,
                    self._robot_id,
                    self._joint_indices,
                    self._loaded_obstacle_ids,
                    self._lower_limits,
                    self._upper_limits,
                    self._fps,
                    obstacle_names=self._obstacle_names,
                    skip_pairs=self._skip_pairs,
                    verbose=True,
                    physics_client_id=self._pb_client,
                )
                if attempt is None:
                    continue
                attempt_arr = np.asarray(attempt, dtype=np.float64)
                if attempt_arr.shape[0] < 2:
                    length = 0.0
                else:
                    length = float(np.sum(np.linalg.norm(np.diff(attempt_arr, axis=0), axis=1)))
                logger.info(
                    "Candidate %d/%d produced a path with joint-space length=%.3f",
                    i + 1,
                    len(candidates),
                    length,
                )
                if length < best_length:
                    best_length = length
                    path = attempt
                    chosen_q_goal = q_goal

            if path is None or chosen_q_goal is None:
                raise RRTPlanningError(f"RRT failed for all {len(candidates)} IK goal candidate(s)")
            logger.info(
                "Picked shortest path among %d candidate(s): joint-space length=%.3f",
                len(candidates),
                best_length,
            )

            # Snap endpoints to the exact start/goal so smoothing doesn't drift.
            path = list(path)
            path[0] = q_start
            path[-1] = chosen_q_goal

            # Ruckig-parametrize the RRT portion only. The escape segment is
            # prepended raw (un-smoothed) so each escape waypoint becomes one
            # env-step command — this gives the simulator's PD controller a
            # position error per step large enough to overcome contact forces
            # while the robot is wedged against an obstacle. With ruckig
            # smoothing, the escape is split at the (escape→RRT) sharp angle
            # and ramps from 0 velocity, producing many tiny sub-samples that
            # individually can't push the simulator's robot out of contact.
            rrt_waypoints = np.asarray(path, dtype=np.float64)
            if rrt_waypoints.shape[0] >= 2:
                max_vel = np.full(self._num_dofs, self._max_joint_vel)
                max_acc = np.full(self._num_dofs, self._max_joint_acc)
                max_jerk = np.full(self._num_dofs, self._max_joint_jerk)
                traj = np.asarray(
                    ruckig_parametrize_path(
                        rrt_waypoints,
                        max_vel,
                        max_acc,
                        max_jerk,
                        control_hz=self._fps,
                    ),
                    dtype=np.float64,
                )
            else:
                traj = rrt_waypoints

            # Prepend escape waypoints raw. ``escape_path[-1] == path[0] == q_escaped``,
            # which matches ``traj[0]`` (ruckig preserves endpoints), so drop the
            # duplicate from the escape side to keep contiguous sampling.
            if escape_path is not None and len(escape_path) > 1:
                traj = np.concatenate([escape_path[:-1], traj], axis=0)

            return traj
        finally:
            for i, pos, vel in saved_joint_states:
                p.resetJointState(self._robot_id, i, pos, vel, physicsClientId=self._pb_client)
            # Re-enable GUI viewport rendering so the restored pose is visible.
            with contextlib.suppress(p.error):
                p.configureDebugVisualizer(
                    p.COV_ENABLE_RENDERING,
                    1,
                    physicsClientId=self._pb_client,
                )

    def _ik_null_space_kwargs(self, seed_q: np.ndarray) -> dict:
        """Build the null-space kwargs for calculateInverseKinematics, padded
        to ``self._num_movable_joints``. PyBullet silently disables damping
        (and the other null-space hints) when the array sizes don't match the
        URDF's total movable-joint count, so we extend the planner's per-arm
        arrays with permissive defaults for the trailing gripper joints.
        """
        n_extra = max(0, self._num_movable_joints - self._num_dofs)
        ll = self._lower_limits.tolist() + [-np.pi] * n_extra
        ul = self._upper_limits.tolist() + [np.pi] * n_extra
        jr = [u - lo for lo, u in zip(ll, ul, strict=True)]
        rp = list(seed_q) + [0.0] * n_extra
        return {
            "lowerLimits": ll,
            "upperLimits": ul,
            "jointRanges": jr,
            "restPoses": rp,
            "jointDamping": [0.1] * self._num_movable_joints,
        }

    # ------------------------------------------------------------------ #
    #  Collision escape (used when the policy got the robot stuck)       #
    # ------------------------------------------------------------------ #

    def _escape_collision(
        self,
        q_start: np.ndarray,
        max_iters: int = 60,
        step_size: float = 0.01,
        max_per_iter_joint_jump: float = 0.2,
        stall_iters: int = 6,
        lift_fallback_step: float = 0.015,
    ) -> np.ndarray | None:
        """Move the arm out of collision along the aggregated outward contact normal.

        Each iteration: query getClosestPoints between the robot and every loaded
        oracle obstacle, weight each (link, obstacle) pair by how deep it is past
        the planner's collision clearance, sum the contact normals (which point
        from obstacle → robot), and apply a small EE-position step in that
        direction via IK. Stops when no pair is within the clearance buffer (or
        when ``max_iters`` is reached).

        Returns a joint-space trajectory ``[N, num_dofs]`` whose first row is
        ``q_start`` and last row is a collision-free config (with the standard
        clearance), or ``None`` if escape did not converge. Caller is responsible
        for restoring the robot's joint state afterwards (the surrounding
        ``plan()`` ``finally`` block handles this).

        Self-collisions and within-clearance pairs whose ``(link, obstacle)`` is
        in ``skip_pairs`` are ignored, matching the planner's collision check.
        """
        from splatsim.utils.rrt_path_utils import _COLLISION_CLEARANCE, check_links_in_collision

        q = np.asarray(q_start, dtype=np.float64).copy()
        waypoints: list[np.ndarray] = [q.copy()]
        prev_max_pen: float | None = None
        no_progress_iters = 0
        lift_mode = False  # switch to a deterministic +z lift after stall

        for it in range(max_iters):
            # Snap pybullet to the current candidate config.
            for j_idx, qi in zip(self._joint_indices, q, strict=False):
                p.resetJointState(self._robot_id, j_idx, float(qi), physicsClientId=self._pb_client)

            escape_dir = np.zeros(3)
            max_pen = 0.0
            n_pairs = 0
            for obs_id in self._loaded_obstacle_ids:
                close_points = (
                    p.getClosestPoints(
                        self._robot_id,
                        obs_id,
                        distance=_COLLISION_CLEARANCE,
                        physicsClientId=self._pb_client,
                    )
                    or []
                )
                for c in close_points:
                    link_idx = c[3]  # linkIndexA
                    if (link_idx, obs_id) in self._skip_pairs:
                        continue
                    normal = np.asarray(c[7], dtype=np.float64)  # B → A
                    dist = float(c[8])
                    weight = max(_COLLISION_CLEARANCE - dist, 1e-3)
                    escape_dir = escape_dir + normal * weight
                    max_pen = max(max_pen, max(0.0, -dist))
                    n_pairs += 1

            if n_pairs == 0:
                # No obstacle close-pairs left, but the planner's standard
                # collision check (which ALSO covers self-collisions between
                # non-adjacent robot links) may still report colliding — e.g.
                # when the policy curled the arm into itself with the gripper
                # near the shoulder. Contact-normal escape can't help with
                # self-tangles (no obstacle to repel from), but a +z lift
                # straightens the arm and breaks the tangle. Switch to lift
                # mode and continue iterating.
                still_colliding = check_links_in_collision(
                    self._robot_id,
                    self._joint_indices,
                    None,
                    self._loaded_obstacle_ids,
                    obstacle_names=self._obstacle_names,
                    skip_pairs=self._skip_pairs,
                    verbose=False,
                    physics_client_id=self._pb_client,
                )
                if not still_colliding:
                    break  # truly clear
                if not lift_mode:
                    logger.info(
                        "No obstacle pairs but check_links_in_collision still "
                        "reports collision (likely self-collision from curled arm) "
                        "— switching to +z lift fallback.",
                    )
                    lift_mode = True

            # Track penetration progress; if max_pen hasn't decreased meaningfully
            # over `stall_iters` iterations we're stuck (e.g. contact normals
            # cancel each other out, or IK can't realise the requested EE step).
            # Switch to a deterministic +z lift in world frame as a fallback.
            if prev_max_pen is not None and max_pen >= prev_max_pen - 1e-4:
                no_progress_iters += 1
            else:
                no_progress_iters = 0
            prev_max_pen = max_pen
            if no_progress_iters >= stall_iters and not lift_mode:
                logger.info(
                    "Escape stalled after %d iter(s) at penetration=%.4fm; "
                    "switching to deterministic +z lift fallback.",
                    it + 1,
                    max_pen,
                )
                lift_mode = True

            if lift_mode:
                escape_dir = np.array([0.0, 0.0, 1.0])
                step = lift_fallback_step
            else:
                norm = float(np.linalg.norm(escape_dir))
                # Wedged between opposing surfaces (~zero net normal) — fall
                # back to a straight up-lift.
                escape_dir = np.array([0.0, 0.0, 1.0]) if norm < 1e-9 else escape_dir / norm
                # Step further when actually penetrating; up to ~6× the base step.
                step = step_size * (1.0 + min(max_pen / _COLLISION_CLEARANCE, 5.0))

            # FK at the current config gives us the EE pose to displace.
            ee_state = p.getLinkState(
                self._robot_id,
                self._ee_link_index,
                computeForwardKinematics=True,
                physicsClientId=self._pb_client,
            )
            ee_pos = np.asarray(ee_state[4], dtype=np.float64)
            ee_quat = np.asarray(ee_state[5], dtype=np.float64)
            target_pos = ee_pos + escape_dir * step

            joint_poses = p.calculateInverseKinematics(
                self._robot_id,
                self._ee_link_index,
                target_pos.tolist(),
                ee_quat.tolist(),
                **self._ik_null_space_kwargs(q),
                maxNumIterations=200,
                residualThreshold=1e-5,
                physicsClientId=self._pb_client,
            )
            q_new = np.asarray(joint_poses[: self._num_dofs], dtype=np.float64)
            q_new = ((q_new + np.pi) % (2 * np.pi)) - np.pi

            # Clamp to limits instead of bailing — escape is best-effort
            # recovery, not precise IK. If clamping leaves q_new == q (no
            # forward progress), the stall detector above will trip after
            # `stall_iters` iterations and switch to the deterministic +z
            # lift fallback. Bailing on the very first IK overshoot makes the
            # escape return None on iter 0 in the common case where IK wants
            # to move joint 1 through its 0-rad upper limit.
            q_new = np.clip(q_new, self._lower_limits, self._upper_limits)
            # If IK overshoots the per-iter jump cap, scale the step down so we
            # still make forward progress (rather than skipping the iteration).
            max_jump = float(np.max(np.abs(q_new - q)))
            if max_jump > max_per_iter_joint_jump:
                scale = max_per_iter_joint_jump / max_jump
                q_new = q + (q_new - q) * scale

            waypoints.append(q_new.copy())
            q = q_new
            if (it + 1) % 10 == 0:
                logger.info(
                    "Escape iter %d/%d: max_pen=%.4fm, n_pairs=%d, mode=%s",
                    it + 1,
                    max_iters,
                    max_pen,
                    n_pairs,
                    "lift" if lift_mode else "contact-normal",
                )

        # Final verification that we cleared the standard collision check.
        for j_idx, qi in zip(self._joint_indices, q, strict=False):
            p.resetJointState(self._robot_id, j_idx, float(qi), physicsClientId=self._pb_client)
        if check_links_in_collision(
            self._robot_id,
            self._joint_indices,
            None,
            self._loaded_obstacle_ids,
            obstacle_names=self._obstacle_names,
            skip_pairs=self._skip_pairs,
            verbose=False,
            physics_client_id=self._pb_client,
        ):
            logger.warning(
                "Escape failed after %d iter(s) — final config still in collision "
                "(max_pen this loop=%.4fm; lift_mode_used=%s). Returning None.",
                len(waypoints) - 1,
                prev_max_pen if prev_max_pen is not None else 0.0,
                lift_mode,
            )
            return None

        logger.info(
            "Escape succeeded after %d waypoint(s) (lift_mode_used=%s).",
            len(waypoints) - 1,
            lift_mode,
        )
        return np.asarray(waypoints, dtype=np.float64)

    # ------------------------------------------------------------------ #
    #  IK candidate resolution (mirrors SplatSim's TrajectoryGenerator)  #
    # ------------------------------------------------------------------ #

    def _resolve_ee_pose_to_q_candidates(
        self,
        ee_pos: np.ndarray,
        ee_quat: np.ndarray,
        q_goal_bias: np.ndarray | None,
    ) -> list[np.ndarray]:
        """Sample multiple IK solutions for the EE pose, biased toward q_goal_bias.

        First attempt seeds with q_goal_bias (when provided), the rest with
        random configurations, so the canonical solution is preferred when
        feasible and we still find alternate IK branches when it isn't.
        """
        candidates: list[np.ndarray] = []
        for i in range(self._num_ik_candidates):
            seed_q = q_goal_bias if (i == 0 and q_goal_bias is not None) else None
            q = self._solve_ik(ee_pos, ee_quat, seed_q=seed_q)
            if q is not None:
                candidates.append(q)
        return self._deduplicate_q_candidates(candidates)

    def _solve_ik(
        self,
        ee_pos: np.ndarray,
        ee_quat: np.ndarray,
        seed_q: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Solve IK for a target EE pose. Returns a collision-free q or None.

        Mirrors ``TrajectoryGenerator._solve_ik`` (splatsim/utils/trajectory_generation.py)
        but operates on the wrapper's private pybullet client.
        """
        from splatsim.utils.rrt_path_utils import check_links_in_collision

        is_biased = seed_q is not None
        if seed_q is None:
            seed_q = np.random.uniform(self._lower_limits, self._upper_limits)
        for idx, qi in zip(self._joint_indices, seed_q, strict=False):
            p.resetJointState(self._robot_id, idx, float(qi), physicsClientId=self._pb_client)

        # Pass null-space arrays sized to the URDF's full movable-joint count
        # so PyBullet actually engages null-space IK (it silently disables it
        # when the array sizes don't match), giving us bias toward seed_q.
        q_solution = p.calculateInverseKinematics(
            self._robot_id,
            self._ee_link_index,
            list(ee_pos),
            list(ee_quat),
            **self._ik_null_space_kwargs(np.asarray(seed_q, dtype=np.float64)),
            maxNumIterations=100000,
            residualThreshold=1e-10,
            physicsClientId=self._pb_client,
        )
        q_solution = np.array(q_solution[: len(self._joint_indices)])

        # Wrap to [-pi, pi]
        q_solution = ((q_solution + np.pi) % (2 * np.pi)) - np.pi

        if np.any(q_solution < self._lower_limits) or np.any(q_solution > self._upper_limits):
            return None

        # If we explicitly seeded with q_goal_bias, reject IK results that wandered
        # too far — null-space IK on a 6-DOF arm can still flip branches.
        if is_biased:
            wrapped_diff = ((q_solution - seed_q + np.pi) % (2 * np.pi)) - np.pi
            max_drift = float(np.max(np.abs(wrapped_diff)))
            if max_drift > np.pi / 3:  # 60° per-joint tolerance
                logger.debug(
                    "IK seeded with q_goal_bias drifted %.1f° from seed; falling back to random-seed IK.",
                    np.degrees(max_drift),
                )
                return None

        if check_links_in_collision(
            self._robot_id,
            self._joint_indices,
            q_solution,
            self._loaded_obstacle_ids,
            obstacle_names=self._obstacle_names,
            skip_pairs=self._skip_pairs,
            verbose=False,
            physics_client_id=self._pb_client,
        ):
            return None

        # Verify IK accuracy via FK
        for idx, qi in zip(self._joint_indices, q_solution, strict=False):
            p.resetJointState(self._robot_id, idx, float(qi), physicsClientId=self._pb_client)
        link_state = p.getLinkState(
            self._robot_id,
            self._ee_link_index,
            computeForwardKinematics=True,
            physicsClientId=self._pb_client,
        )
        actual_pos = np.array(link_state[0])
        if np.linalg.norm(actual_pos - ee_pos) > 0.005:  # 5 mm tolerance
            return None
        actual_quat = np.array(link_state[1])
        dot = float(np.clip(abs(np.dot(actual_quat, ee_quat)), -1.0, 1.0))
        if np.degrees(2 * np.arccos(dot)) > 5.0:  # 5° tolerance
            return None

        return q_solution

    @staticmethod
    def _deduplicate_q_candidates(
        candidates: list[np.ndarray], threshold_rad: float = 0.1
    ) -> list[np.ndarray]:
        """Remove near-duplicate joint configurations."""
        if len(candidates) <= 1:
            return candidates
        kept: list[np.ndarray] = []
        for c in candidates:
            if all(
                float(np.max(np.abs(((c - k + np.pi) % (2 * np.pi)) - np.pi))) > threshold_rad for k in kept
            ):
                kept.append(c)
        return kept


def extract_task_goal(
    env_config: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None] | None:
    """Pull the RRT goal (target EE pose + optional q_goal_bias seed) from the env config.

    Returns ``(target_ee_pos, target_ee_quat, q_goal_bias_or_none)`` when the
    task has a defined target EE pose, or ``None`` otherwise. The caller should
    surface ``None`` as a planning error rather than silently falling back.
    """
    task = env_config.get("task") if env_config else None
    if not task:
        return None
    pos = task.get("target_ee_pos")
    quat = task.get("target_ee_quat")
    if pos is None or quat is None:
        return None
    bias = task.get("q_goal_bias")
    return (
        np.asarray(pos, dtype=np.float64),
        np.asarray(quat, dtype=np.float64),
        np.asarray(bias, dtype=np.float64) if bias is not None else None,
    )
