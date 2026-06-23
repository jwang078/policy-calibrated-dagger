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
import itertools
import logging
import threading
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pybullet as p

from lerobot.policies.guidance.base import GuidanceMode


class PathSelectionStrategy(Enum):
    """How `RRTToGoalPlanner.plan()` picks among IK-goal-candidate paths.

    All strategies score each successful candidate path and pick the minimum;
    they differ only in what they minimize:

    * ``EE_ARC_LENGTH`` (default) — Euclidean distance traversed by the EE
      link in cartesian space. Penalizes wide swings that hurt DAgger data
      quality even when the joint-space length is small.
    * ``JOINT_ARC_LENGTH`` — sum of joint-space L2 distances between
      consecutive waypoints. Legacy behavior; tends to prefer paths that
      happen to land near `q_start` in configuration space even if the EE
      swings wide.
    * ``JOINT_VELOCITY_MATCH`` — cosine distance between the candidate's
      initial direction (averaged over the first few path samples) and
      the robot's recent direction (averaged over the last few samples
      before the trigger). Picks the path that starts off in the same
      direction the robot was already moving, minimizing the velocity
      discontinuity at the trigger moment. Direction-only (not
      magnitude-matching) because the candidate's raw waypoint deltas
      are in different units than the robot's per-tick velocity — see
      ``_path_velocity_deviation``. When the robot's recent velocity is
      near zero (typical from a collision/stall trigger), falls back to
      EE_ARC_LENGTH. Requires `recent_joint_velocity` to be passed to
      `plan()`; raises `RRTPlanningError` if not.
    * ``MIN_PAIR_CLEARANCE`` — picks the candidate whose path maintains
      the LARGEST minimum distance between any non-adjacent robot link
      pair, evaluated at every waypoint. "Larger min" = "more comfortable
      arm pose at every point along the path." Specifically targets the
      pretzeled-pose failure mode: BiRRT can find feasible paths that
      pass through configurations where normally-distant links (e.g.
      gripper near shoulder) come close together — those configurations
      aren't COLLISIONS, but they're hard for diffusion policies to
      learn from. Picking the path with the largest min-pair gap
      systematically avoids them when multiple candidates exist.
      The structurally-close pairs declared via the planner's
      ``self_collision_skip_pairs`` (URDF noise like UR base_link vs
      upper_arm_link, ~4 mm apart at every config) are EXCLUDED from
      the scoring so they don't dominate the min — otherwise every path
      would tie at ~4 mm and the strategy would be useless.
      Cost: one ``getClosestPoints`` query per non-adjacent pair per
      waypoint. For a UR-class robot (~24 links → ~250 non-adjacent
      pairs after filtering) with 20 waypoints, this is ~5K queries per
      candidate path. With ``rrt_num_path_candidates_per_ik=5`` that's
      ~25K per IK goal — ~250 ms at 0.01 ms/query. Use a query distance
      cap so far-apart pairs early-out cheaply.
    Note: scoring metrics that depend ONLY on the goal state (e.g. joint
    distance from start to candidate goal) belong in
    ``IkGoalSelectionStrategy`` instead — those don't need a planned path
    to evaluate and are properties of the IK candidate, not the path.
    """

    EE_ARC_LENGTH = "ee_arc_length"
    JOINT_ARC_LENGTH = "joint_arc_length"
    JOINT_VELOCITY_MATCH = "joint_velocity_match"
    MIN_PAIR_CLEARANCE = "min_pair_clearance"


class IkGoalSelectionStrategy(Enum):
    """How `RRTToGoalPlanner.plan()` picks AMONG the IK candidates BEFORE
    running RRT, based on goal-state geometry alone (no path required).

    When this strategy is set on the planner, candidates are scored by
    their goal-state property (lower = better), tried in order, and the
    FIRST successful RRT plan wins — ``path_selection`` is unused because
    each path is already to a different goal, so cross-path comparison
    is meaningless once we've decided which goal to commit to.

    When the strategy is left unset (None), the planner falls back to
    running RRT against EVERY IK candidate and using ``path_selection``
    to score the resulting paths (historical multi-candidate behavior).

    * ``JOINT_DISTANCE`` — minimize ``||q_candidate - q_start||``. For
      redundant arms (7-DOF, multiple IK solutions per EE pose) this
      picks the goal configuration requiring the LEAST joint
      reconfiguration. Biases the planner toward keeping the policy
      "in its current mode" — elbow-up stays elbow-up, wrist-flip stays
      unflipped. Useful when the policy's training data is multimodal
      (multiple IK branches) and you want intervention data to
      consistently commit to whichever branch the policy is already on.
      Works even when the policy was stationary at trigger time (unlike
      JOINT_VELOCITY_MATCH which needs a velocity history).
    """

    JOINT_DISTANCE = "joint_distance"


logger = logging.getLogger(__name__)


# `RRTMode` is aliased to the unified `GuidanceMode` so external callers like
# `InterventionController` (which imports `RRTMode` and compares against
# `RRTMode.IDLE/PLANNING/EXECUTING`) keep working unchanged after the SA-wrapper
# guidance-source refactor. The two enums have byte-identical members and string
# values; the alias is transparent.
RRTMode = GuidanceMode


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
    # Last successfully-planned IK goal (joint config). Set by RRTGuidanceSource
    # from the planner's `_last_chosen_q_goal` after each successful plan.
    # Consumed by the retry-on-collision path so we know which IK branch
    # to exclude when re-planning.
    chosen_q_goal: np.ndarray | None = None
    # IK branches whose paths collided when executed in sim. Appended on
    # each retry, passed back into planner.plan() via `exclude_q_goals` so
    # subsequent plans skip them. Reset to [] at scenario start (the
    # source's normal reset path) — within-scenario only.
    excluded_q_goals: list[np.ndarray] = field(default_factory=list)
    # When True for the next _do_plan invocation: skip the pre-jump
    # lookback sampling AND the teleport-to-q_start entirely. q_start is
    # read from the wrapper's CURRENT joint state (no rewind), and ruckig
    # is invoked with `start_vel = recent_joint_velocity` so the
    # parametrized trajectory begins at velocity-continuous matching the
    # robot's actual motion. Set per-trigger by RRTGuidanceSource.trigger();
    # consumed (and cleared back to False) by _do_plan at the start of
    # each planning call.
    no_lookback: bool = False


class RRTPlanningError(RuntimeError):
    """Raised when RRT planning fails with a recognizable cause (start/goal
    in collision, no path found within iteration budget, etc.)."""


def _canonical_for_hash(value) -> str:
    """Render value to a canonical string so functionally-identical configs
    hash to the same key.

    Specifically:
      * Sort dict keys (Python preserves insertion order, but the env can
        emit the same dict with different key orderings across calls).
      * Round floats to 6 decimal places — the env's quaternion conversion
        sometimes flips -0.0 vs 0.0 and emits sub-nanometer position
        jitter; without rounding, repr() of those differs and invalidates
        the cache for the same physical scene.
    """
    if isinstance(value, dict):
        items = sorted(value.items(), key=lambda kv: str(kv[0]))
        return "{" + ",".join(f"{_canonical_for_hash(k)}:{_canonical_for_hash(v)}" for k, v in items) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_canonical_for_hash(v) for v in value) + "]"
    if isinstance(value, float):
        rounded = round(value, 6)
        # Normalize -0.0 → 0.0 so the sign bit doesn't break the cache.
        if rounded == 0:
            rounded = 0.0
        return repr(rounded)
    return repr(value)


def _hash_config(cfg: dict) -> str:
    """Hash JUST the obstacle-relevant portion of the oracle env config.

    The full ``env_config`` dict includes transient per-step fields like
    ``current_ee_pos`` that change every tick — hashing the whole dict
    invalidates the cache every step and forces a full obstacle reload.
    ``load_obstacles`` only reads ``cfg["objects"]``, so that's the only
    thing the cache key needs to track.

    Uses a canonical-rendering helper that sorts dict keys and rounds floats
    so a static scene produces a stable hash across env ticks, even when
    the env emits ``-0.0`` vs ``0.0`` or sub-µm float jitter.

    Hash is for cache invalidation only, not security. usedforsecurity=False
    silences bandit's B324 warning about SHA1.
    """
    objs = cfg.get("objects", []) if isinstance(cfg, dict) else []
    return hashlib.sha1(_canonical_for_hash(objs).encode("utf-8"), usedforsecurity=False).hexdigest()


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
        path_selection: PathSelectionStrategy = PathSelectionStrategy.EE_ARC_LENGTH,
        velocity_match_window: int = 3,
        segment_at_sharp_corners: bool = True,
        ik_goal_selection: IkGoalSelectionStrategy | str | None = None,
        num_path_candidates_per_ik: int = 1,
        max_path_attempts_per_ik: int = 5,
        path_perturbation_scale: float = 0.001,
        obstacle_clearance: float | None = None,
        self_collision_clearance: float | None = None,
        self_collision_skip_pairs: list[tuple[int, int]] | None = None,
        diagnostic_log_pairs: str = "off",
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
        self._path_selection = path_selection
        # MIN_PAIR_CLEARANCE diagnostic toggle (validated upstream by the
        # SA config's __post_init__). Controls the structural-offender
        # probe inside `_path_min_pair_clearance`. See the SA config for
        # the semantics of "off" / "first" / "always".
        if diagnostic_log_pairs not in ("off", "first", "always"):
            raise ValueError(
                f"diagnostic_log_pairs must be one of 'off'/'first'/'always', got {diagnostic_log_pairs!r}"
            )
        self._diagnostic_log_pairs = diagnostic_log_pairs
        # Forwarded to ruckig_parametrize_path. True (default) = historical
        # per-segment mode with zero velocity at each sharp corner. False =
        # single-call ruckig with intermediate_positions (no forced internal
        # stops). Empirically indistinguishable on typical manipulation
        # RRT plans; True is the safer default.
        self._segment_at_sharp_corners = bool(segment_at_sharp_corners)
        # When set, the planner SHORT-CIRCUITS the multi-candidate
        # path-scoring loop: candidates are sorted by their IK-goal score
        # (lower = better, see IkGoalSelectionStrategy) and tried in
        # order — the FIRST successful RRT plan wins. `path_selection` is
        # ignored in that mode because there's no path-vs-path comparison
        # to make (each path goes to a different goal). When None
        # (default), the original "try all + score by path_selection"
        # behavior runs. Accepts the enum or its string value
        # ("joint_distance") for ergonomic config wiring.
        if ik_goal_selection is None:
            self._ik_goal_selection = None
        elif isinstance(ik_goal_selection, str):
            self._ik_goal_selection = IkGoalSelectionStrategy(ik_goal_selection)
        else:
            self._ik_goal_selection = ik_goal_selection
        # Per-IK multi-path scoring (ports SplatSim's
        # TrajectoryGenerator._generate_multiple_path_candidates pattern).
        # When num_path_candidates_per_ik > 1, the planner runs RRT
        # multiple times per IK candidate — first attempt at exact
        # endpoints, subsequent attempts with both q_start and q_goal
        # randomly perturbed by ±path_perturbation_scale to nudge RRT's
        # sampler down different branches — then `path_selection` picks
        # the best path among them for that IK. This is what makes
        # `path_selection` non-trivial when `ik_goal_selection` is also
        # set: each IK gets several path candidates, the best one wins
        # for that IK, and then the IK ordering decides which IK's best
        # path is used.
        # max_path_attempts_per_ik caps the total RRT calls per IK
        # (counter resets between successes, so it's actually max attempts
        # BETWEEN successes — matches SplatSim's loop semantics).
        self._num_path_candidates_per_ik = int(num_path_candidates_per_ik)
        self._max_path_attempts_per_ik = int(max_path_attempts_per_ik)
        self._path_perturbation_scale = float(path_perturbation_scale)
        # Collision clearances threaded into every check_links_in_collision
        # + get_path call so RRT plans paths with the configured margin.
        # None = use SplatSim's defaults (_COLLISION_CLEARANCE = 0.01 m
        # obstacle, self = 0.0 m). Stored as `_obstacle_clearance_override`
        # and `_self_collision_clearance_override`; downstream sites consult
        # them via the `_obstacle_clearance_arg` / `_self_clearance_arg`
        # helpers below so we don't have to special-case None at every
        # callsite.
        self._obstacle_clearance_override = (
            float(obstacle_clearance) if obstacle_clearance is not None else None
        )
        self._self_collision_clearance_override = (
            float(self_collision_clearance) if self_collision_clearance is not None else None
        )
        # Pre-build the kwargs dicts the callsites pass to
        # check_links_in_collision / get_path. None = empty dict so SplatSim's
        # defaults stand (omit kwarg → check_links_in_collision uses
        # _COLLISION_CLEARANCE / self_collision_clearance=0.0).
        self._collision_kwargs: dict = {}
        if self._obstacle_clearance_override is not None:
            self._collision_kwargs["obstacle_clearance"] = self._obstacle_clearance_override
        if self._self_collision_clearance_override is not None:
            self._collision_kwargs["self_collision_clearance"] = self._self_collision_clearance_override
        # Skip-pair list goes into the same kwargs dict so every
        # `check_links_in_collision(**self._collision_kwargs)` / `get_path(...)`
        # callsite picks it up automatically — same pattern as the clearance
        # overrides. Empty / None list means we omit the kwarg entirely so
        # SplatSim's defaults stand (no skips).
        if self_collision_skip_pairs:
            self._collision_kwargs["self_collision_skip_pairs"] = [
                tuple(p) for p in self_collision_skip_pairs
            ]
        # Published by plan() before ruckig — last successful IK goal as
        # joint config. Consumed by RRTGuidanceSource's retry-on-collision
        # to track which IK branch to exclude when re-planning.
        self._last_chosen_q_goal: np.ndarray | None = None
        # Window over which to average velocities for the JOINT_VELOCITY_MATCH
        # strategy. Applied to BOTH the candidate path's leading edge and the
        # robot's trailing velocity history. 3 samples ≈ 100 ms at 30 Hz —
        # enough to smooth jitter, short enough that "recent" still means recent.
        self._velocity_match_window = int(velocity_match_window)
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

    def _effective_obstacle_clearance(self) -> float:
        """Override or SplatSim's _COLLISION_CLEARANCE default (0.01).

        Used by `_escape_collision` whose contact-normal math is keyed off
        the SAME clearance threshold the BiRRT collision_fn uses, so the
        escape moves the robot to a state the planner considers safe.
        """
        if self._obstacle_clearance_override is not None:
            return self._obstacle_clearance_override
        # Lazy import to keep splatsim out of module-level import surface.
        from splatsim.utils.rrt_path_utils import _COLLISION_CLEARANCE

        return _COLLISION_CLEARANCE

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
        recent_joint_velocity: np.ndarray | None = None,
        exclude_q_goals: list[np.ndarray] | None = None,
        ruckig_start_vel: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Plan a joint-space trajectory to an end-effector pose.

        Mirrors SplatSim's ``TrajectoryGenerator._resolve_ee_pose_to_q_candidates``
        + ``_plan_with_fallback_goals``: solves IK to ``(target_ee_pos, target_ee_quat)``
        from multiple seeds — the first attempt seeded by ``q_goal_bias`` (so demos
        converge to a canonical configuration when feasible), the rest from random
        seeds — and runs RRT against each collision-free candidate until one
        succeeds.  This is much less constrained than fixing ``q_goal_bias`` as
        the only goal, which often has no valid path through cluttered scenes.

        `exclude_q_goals`: optional list of joint configurations to FILTER OUT
        of the IK candidate set before planning. Used by the retry-on-collision
        path: when a path executed in sim collides (typically because ruckig
        smoothing curved through an obstacle the RRT-raw path didn't), the
        source adds that path's q_goal to this list and re-calls plan() — the
        filter discards any candidate within ~0.05 rad (per-joint L2) of an
        excluded goal, so we don't immediately re-pick the same IK branch.
        Empty / None preserves historical behavior.

        After a successful plan, the chosen q_goal is also published on
        ``self._last_chosen_q_goal`` so callers can record it without
        scraping the trajectory's terminal pose.

        Returns ``(traj, escape_end_q)`` where:
          * ``traj`` is the ruckig-smoothed RRT chunk of shape (T, num_dofs)
            sampled at ``self._fps``. **NEW (formerly traj contained
            prepended escape waypoints):** the escape segment is no longer
            included in ``traj``. Callers that have access to the env must
            teleport the robot to ``escape_end_q`` before executing ``traj``,
            so the env's robot is physically at ``traj[0]`` at chunk t=0.
          * ``escape_end_q`` is the collision-free joint config the planner
            escaped to (== ``traj[0]``), or ``None`` if no escape was needed.
            Used as a signal to the source: "if non-None, env teleport is
            required". When the historical lookback path also wants to
            teleport q_start_full into the env, ``escape_end_q``'s teleport
            REPLACES that (you don't want to teleport to the wedged config
            first; the planner already moved past it).

        Why no longer prepend escape: the escape waypoints were intentionally
        un-smoothed (large per-step deltas to overcome PD-controller contact
        forces in sim), which produced 10×-mean-delta outlier frames at the
        start of recorded intervention episodes. Those outliers contaminated
        the DAgger training distribution (diffusion policy's score field
        learned to associate wedged-state observations with discrete
        pushout actions — a sim-PD artifact, not a transferable skill).
        Teleporting in the env achieves the same physical end-state without
        recording the artifact. The planner's iterative escape search is
        unchanged — only the env-side replay is bypassed.

        Raises ``RRTPlanningError`` if no IK candidate can be reached.
        """
        from splatsim.utils.rrt_path_utils import check_links_in_collision
        # ruckig_parametrize_path is no longer used inline here — the per-IK
        # loop now calls self._smooth_and_check_collision which owns both
        # the parametrize and the dense collision check.

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
                **self._collision_kwargs,
            ):
                logger.info(
                    "Start config in collision; attempting escape chain (contact-normal → self-collision gradient)..."
                )
                escape_path = self._try_escape_chain(q_start)
                if escape_path is None:
                    raise RRTPlanningError(
                        "Start in collision and all escape modes "
                        "(contact-normal, self-collision gradient) failed to clear it"
                    )
                logger.info("Escape produced %d waypoint(s); replanning from new start", len(escape_path))
                q_start = escape_path[-1].copy()

            # Resolve the target EE pose to multiple collision-free joint-space
            # candidates. First attempt is seeded with q_goal_bias (when provided).
            candidates = self._resolve_ee_pose_to_q_candidates(target_ee_pos, target_ee_quat, q_goal_bias)
            if len(candidates) == 0:
                raise RRTPlanningError("No collision-free IK solution found for target EE pose")
            n_before_exclude = len(candidates)
            # Retry-on-collision filter: drop any IK candidate whose joint
            # config is close to one previously found to collide when its
            # path was executed. The tolerance is generous (0.05 rad per-
            # joint L2) so we drop the SAME IK branch, not a genuinely
            # different solution that happens to be near it.
            if exclude_q_goals:
                tol = 0.05
                filtered = []
                for q in candidates:
                    q_arr = np.asarray(q)
                    if any(np.linalg.norm(q_arr - np.asarray(eq)) < tol for eq in exclude_q_goals):
                        continue
                    filtered.append(q)
                if len(filtered) < n_before_exclude:
                    logger.info(
                        "Collision-history filter dropped %d of %d IK candidate(s)",
                        n_before_exclude - len(filtered),
                        n_before_exclude,
                    )
                candidates = filtered
                if not candidates:
                    raise RRTPlanningError(
                        f"All {n_before_exclude} IK candidate(s) were excluded by "
                        f"the collision-history filter — no untried branch remains"
                    )
            logger.info("Resolved EE goal to %d collision-free IK candidate(s)", len(candidates))

            # Planning loop. The two scoring axes work TOGETHER (not
            # mutually exclusive):
            #
            #   * `ik_goal_selection` (when set) — sorts IK candidates by
            #     goal-state geometry (e.g. joint distance from q_start),
            #     best-first. The planner walks through candidates in
            #     this order. When None, candidates keep their original
            #     order from the IK solver.
            #
            #   * `path_selection` — scores the actual planned path(s) to
            #     a given IK goal. With one RRT attempt per IK there's
            #     a single path per candidate, but path_selection still
            #     determines which IK candidate's path wins when MULTIPLE
            #     candidates yield successful RRT plans.
            #
            # Algorithm:
            #   For each IK candidate (in IK-sorted order):
            #     - Run RRT once. If failure, fall through to next IK.
            #     - On success, score the path by path_selection.
            #     - Track the running best. If `early_exit_on_first_ik` is
            #       True (the IK-goal-selection contract: "first IK with
            #       any successful path wins"), break immediately. If
            #       False (no IK ordering — original "try-all-and-rank"
            #       behavior), keep going to find the minimum-path-score
            #       winner across all IKs.
            if self._path_selection == PathSelectionStrategy.JOINT_VELOCITY_MATCH:
                if recent_joint_velocity is None:
                    raise RRTPlanningError(
                        "PathSelectionStrategy.JOINT_VELOCITY_MATCH requires `recent_joint_velocity` "
                        "to be passed to plan(); none provided. Either supply a velocity history or "
                        "switch to EE_ARC_LENGTH / JOINT_ARC_LENGTH."
                    )
                recent_vel = np.asarray(recent_joint_velocity, dtype=np.float64).reshape(-1)[: self._num_dofs]
            else:
                recent_vel = None

            path_score_label, path_score_units = {
                PathSelectionStrategy.EE_ARC_LENGTH: ("EE arc-length", "m"),
                PathSelectionStrategy.JOINT_ARC_LENGTH: ("joint arc-length", "rad"),
                PathSelectionStrategy.JOINT_VELOCITY_MATCH: ("joint-velocity deviation", "rad/step"),
                # MIN_PAIR_CLEARANCE scoring returns NEGATED min distance — the
                # score "label" reflects the underlying quantity (min pair gap,
                # meters), but lower is better as elsewhere because of the
                # negation. Reader interpretation: more negative = larger
                # actual gap = safer path.
                PathSelectionStrategy.MIN_PAIR_CLEARANCE: ("neg min-pair clearance", "m"),
            }[self._path_selection]

            # Decide the IK try-order.
            if self._ik_goal_selection is not None:
                ik_score_label, ik_score_units = {
                    IkGoalSelectionStrategy.JOINT_DISTANCE: ("joint Δ from start", "rad"),
                }[self._ik_goal_selection]
                ordered = sorted(
                    enumerate(candidates),
                    key=lambda iq: self._score_ik_candidate(iq[1], q_start),
                )
                early_exit_on_first_ik = True
                ik_order_log = f"IK-sorted by {ik_score_label}"
            else:
                ordered = list(enumerate(candidates))
                early_exit_on_first_ik = False
                ik_order_log = "IK in original solver order"
                ik_score_label, ik_score_units = None, None

            path = None
            chosen_q_goal = None
            best_path_score = float("inf")
            chosen_ik_score: float | None = None
            traj: np.ndarray | None = None  # cached smoothed path of the winning IK
            _loop_ordered = ordered  # alias kept so logging uses a stable name
            for tried, (orig_i, q_goal) in enumerate(ordered, start=1):
                ik_score = (
                    self._score_ik_candidate(q_goal, q_start) if self._ik_goal_selection is not None else None
                )
                if ik_score is not None:
                    logger.info(
                        "IK candidate %d/%d (orig idx %d) %s=%.3f %s — running RRT (up to %d path candidate(s))",
                        tried,
                        len(_loop_ordered),
                        orig_i + 1,
                        ik_score_label,
                        ik_score,
                        ik_score_units,
                        self._num_path_candidates_per_ik,
                    )
                else:
                    logger.info(
                        "Trying RRT against IK candidate %d/%d (up to %d path candidate(s))",
                        tried,
                        len(_loop_ordered),
                        self._num_path_candidates_per_ik,
                    )
                # Generate one OR more RRT paths to this IK candidate.
                # With num_path_candidates_per_ik=1 (default) this is a
                # single get_path call; with >1, the helper perturbs
                # endpoints to force RRT to find distinct paths.
                candidate_paths = self._generate_paths_for_ik(q_start, q_goal)
                if not candidate_paths:
                    continue
                # Among this IK's path candidates, pick the best-scored one
                # whose RUCKIG-SMOOTHED form is collision-free. Score all
                # candidates, sort best-first, then ruckig + dense-check
                # each in score order — take the first that passes. Falling
                # back to a lower-scored path WITHIN the same IK goal is
                # almost always preferable to giving up on the IK goal and
                # trying the next IK (which may itself fail the same way).
                # When num_path_candidates_per_ik=1, this collapses to "try
                # the single path; if its ruckig collides, treat this IK
                # as failed".
                _scored_cps: list[tuple[float, np.ndarray]] = [
                    (self._score_candidate(cp, recent_vel), cp) for cp in candidate_paths
                ]
                _scored_cps.sort(key=lambda x: x[0])
                local_best_score = float("inf")
                local_best_path: np.ndarray | None = None
                local_best_traj: np.ndarray | None = None
                for _rank, (_s, _cp) in enumerate(_scored_cps, start=1):
                    # Snap endpoints to the exact start/goal so smoothing
                    # doesn't drift (same fix as the post-loop block used
                    # to do for the global winner).
                    _cp_snapped = list(_cp)
                    _cp_snapped[0] = q_start
                    _cp_snapped[-1] = q_goal
                    _cp_smoothed, _coll_idx = self._smooth_and_check_collision(
                        np.asarray(_cp_snapped, dtype=np.float64),
                        ruckig_start_vel,
                    )
                    if _coll_idx is not None:
                        logger.info(
                            "  path %d/%d (score=%.4f): ruckig-smoothed collides "
                            "at waypoint %d/%d — trying next path candidate.",
                            _rank,
                            len(_scored_cps),
                            _s,
                            _coll_idx,
                            _cp_smoothed.shape[0],
                        )
                        continue
                    # Found one whose smoothed form is collision-free.
                    local_best_score = _s
                    local_best_path = _cp
                    local_best_traj = _cp_smoothed
                    break
                if local_best_path is None:
                    logger.warning(
                        "IK candidate %d/%d: all %d path(s) had ruckig-smoothed "
                        "collisions — moving to next IK candidate.",
                        tried,
                        len(_loop_ordered),
                        len(_scored_cps),
                    )
                    continue
                assert local_best_traj is not None
                # MIN_PAIR_CLEARANCE's raw score is the NEGATED min distance —
                # flip the sign and rename the label for the log so the user
                # sees "min-pair clearance = 0.012 m" instead of the
                # confusing "neg min-pair clearance = -0.012 m". The
                # underlying score field stays negated for sort consistency
                # with the other strategies' lower-is-better convention.
                if self._path_selection == PathSelectionStrategy.MIN_PAIR_CLEARANCE:
                    _display_val = -local_best_score
                    _display_label = "min-pair clearance"
                    _display_units = "m (more = better)"
                else:
                    _display_val = local_best_score
                    _display_label = path_score_label
                    _display_units = path_score_units
                logger.info(
                    "IK candidate %d/%d: %d path(s) generated; best path %s=%.4f %s",
                    tried,
                    len(_loop_ordered),
                    len(candidate_paths),
                    _display_label,
                    _display_val,
                    _display_units,
                )
                if local_best_score < best_path_score:
                    best_path_score = local_best_score
                    path = local_best_path
                    traj = local_best_traj  # cache the smoothed path so we don't re-ruckig below
                    chosen_q_goal = q_goal
                    chosen_ik_score = ik_score
                    if early_exit_on_first_ik:
                        # IK-goal-selection contract: best IK with ANY successful
                        # plan wins. Within that IK, path_selection picked the
                        # best of its candidate paths above. Other IKs aren't
                        # tried.
                        break

            if path is None or chosen_q_goal is None:
                raise RRTPlanningError(f"RRT failed for all {len(_loop_ordered)} IK goal candidate(s)")
            if chosen_ik_score is not None:
                # Same per-strategy display fix as the per-IK log above —
                # MIN_PAIR_CLEARANCE shows positive min distance.
                if self._path_selection == PathSelectionStrategy.MIN_PAIR_CLEARANCE:
                    _disp_score = -best_path_score
                    _disp_label = "min-pair clearance"
                    _disp_units = "m (more = better)"
                else:
                    _disp_score = best_path_score
                    _disp_label = path_score_label
                    _disp_units = path_score_units
                logger.info(
                    "Picked path (%s; chose first IK with successful plan) — IK %s=%.3f %s, path %s=%.4f %s",
                    ik_order_log,
                    ik_score_label,
                    chosen_ik_score,
                    ik_score_units,
                    _disp_label,
                    _disp_score,
                    _disp_units,
                )
            else:
                # Same per-strategy display fix.
                if self._path_selection == PathSelectionStrategy.MIN_PAIR_CLEARANCE:
                    _disp_score = -best_path_score
                    _disp_label = "min-pair clearance"
                    _disp_units = "m (more = better)"
                else:
                    _disp_score = best_path_score
                    _disp_label = path_score_label
                    _disp_units = path_score_units
                logger.info(
                    "Picked best path (%s) by %s=%.4f %s",
                    ik_order_log,
                    _disp_label,
                    _disp_score,
                    _disp_units,
                )

            # Publish the chosen IK goal so callers (RRTGuidanceSource) can
            # track it across plan() calls without having to scrape the
            # trajectory's terminal pose. Set BEFORE ruckig so the value
            # reflects the actual IK solution, not the post-ruckig endpoint
            # (which might drift by tiny amounts from boundary effects).
            self._last_chosen_q_goal = np.asarray(chosen_q_goal, dtype=np.float64).copy()

            # `traj` is already populated from the per-IK loop above
            # (_smooth_and_check_collision was called there per path
            # candidate; the winning path's smoothed form was cached).
            # Historically, ruckig smoothing happened here as a single
            # post-loop step on the global winner, and the smoothed path
            # was never collision-checked — so a ruckig spline that curved
            # through obstacles would silently reach the env. Per-IK
            # smoothing+checking moved that logic up, so we just use the
            # cached `traj` here.
            #
            # Why escape isn't prepended: the escape segment was historically
            # PREPENDED raw (un-smoothed) so each escape waypoint became one
            # env-step command — needed because the simulator's PD controller
            # wouldn't overcome contact forces with smooth ruckig sub-samples.
            # That left 10×-mean-delta outlier frames in the recorded dataset
            # that corrupted the diffusion policy's score field. Now: escape
            # segment is NOT included in `traj` — `escape_end_q` is returned
            # separately so callers can teleport the env's robot directly to
            # the post-escape config, achieving the same physical end-state
            # without recording the artifact.
            assert traj is not None  # guaranteed when path/chosen_q_goal are set

            # Escape segment is NO LONGER prepended (see docstring + previous
            # comment for the rationale). Instead, return the post-escape
            # config so the caller can teleport the env's robot directly to
            # `traj[0]`. ``escape_path[-1]`` equals ``traj[0]`` by construction
            # (the planner ran from `q_start = escape_path[-1].copy()`), so the
            # teleport puts the robot exactly where the ruckig chunk begins.
            escape_end_q: np.ndarray | None = None
            if escape_path is not None and len(escape_path) >= 1:
                escape_end_q = np.asarray(escape_path[-1], dtype=np.float64)

            return traj, escape_end_q
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

    def _smooth_and_check_collision(
        self,
        rrt_waypoints: np.ndarray,
        ruckig_start_vel: np.ndarray | None,
    ) -> tuple[np.ndarray, int | None]:
        """Ruckig-parametrize a raw RRT path and dense-check the smoothed
        result for collisions. Returns ``(smoothed_traj, first_colliding_idx)``;
        ``first_colliding_idx is None`` means the smoothed path is
        collision-free.

        Used inside ``plan()``'s per-IK loop to try each candidate path's
        smoothed form BEFORE settling on one. Without this, the BiRRT raw
        check (which only validates discrete waypoints) lets ruckig's
        continuous spline curve through obstacles at sharp corners; the
        bad chunk then reaches the env, robot collides, and the controller
        has to retry from inside an already-contaminated teleop buffer.

        Cost dominated by the per-waypoint pybullet getClosestPoints calls.
        Bails on the first collision so a known-bad path is cheap to reject.
        Caller is responsible for joint-state restore (plan()'s finally).
        """
        from splatsim.utils.rrt_path_utils import (
            check_links_in_collision,
            ruckig_parametrize_path,
        )

        if rrt_waypoints.shape[0] < 2:
            # Single-waypoint path — nothing to ruckig and nothing the
            # raw collision check would have missed.
            return rrt_waypoints, None

        max_vel = np.full(self._num_dofs, self._max_joint_vel)
        max_acc = np.full(self._num_dofs, self._max_joint_acc)
        max_jerk = np.full(self._num_dofs, self._max_joint_jerk)
        _ruckig_kwargs: dict = {}
        if ruckig_start_vel is not None:
            _ruckig_kwargs["start_vel"] = np.asarray(ruckig_start_vel, dtype=np.float64).reshape(-1)[
                : self._num_dofs
            ]
        traj = np.asarray(
            ruckig_parametrize_path(
                rrt_waypoints,
                max_vel,
                max_acc,
                max_jerk,
                control_hz=self._fps,
                segment_at_sharp_corners=self._segment_at_sharp_corners,
                **_ruckig_kwargs,
            ),
            dtype=np.float64,
        )
        for k in range(traj.shape[0]):
            if check_links_in_collision(
                self._robot_id,
                self._joint_indices,
                traj[k],
                self._loaded_obstacle_ids,
                obstacle_names=self._obstacle_names,
                skip_pairs=self._skip_pairs,
                verbose=False,
                physics_client_id=self._pb_client,
                **self._collision_kwargs,
            ):
                return traj, k
        return traj, None

    def _score_candidate(
        self,
        path: np.ndarray,
        recent_joint_velocity: np.ndarray | None,
    ) -> float:
        """Dispatch to the active path-selection strategy. Lower = better.

        `recent_joint_velocity` is consulted only when the strategy is
        JOINT_VELOCITY_MATCH; for other strategies it is ignored.
        """
        strategy = self._path_selection
        if strategy == PathSelectionStrategy.EE_ARC_LENGTH:
            return self._path_ee_arc_length(path)
        if strategy == PathSelectionStrategy.JOINT_ARC_LENGTH:
            return self._path_joint_arc_length(path)
        if strategy == PathSelectionStrategy.JOINT_VELOCITY_MATCH:
            assert recent_joint_velocity is not None  # caller guarantees, see plan()
            return self._path_velocity_deviation(path, recent_joint_velocity)
        if strategy == PathSelectionStrategy.MIN_PAIR_CLEARANCE:
            return self._path_min_pair_clearance(path)
        raise ValueError(f"Unknown PathSelectionStrategy: {strategy!r}")

    def _score_ik_candidate(
        self,
        q_candidate: np.ndarray,
        q_start: np.ndarray,
    ) -> float:
        """Score a candidate IK goal under the active IkGoalSelectionStrategy.
        Lower = better. Pure goal-state geometry — no planned path needed.
        """
        strategy = self._ik_goal_selection
        if strategy == IkGoalSelectionStrategy.JOINT_DISTANCE:
            return float(np.linalg.norm(np.asarray(q_candidate) - np.asarray(q_start)))
        raise ValueError(f"Unknown IkGoalSelectionStrategy: {strategy!r}")

    def _generate_paths_for_ik(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
    ) -> list[np.ndarray]:
        """Generate up to `num_path_candidates_per_ik` distinct RRT paths
        from q_start to q_goal. First attempt uses exact endpoints;
        subsequent attempts perturb both endpoints by ±perturbation_scale
        to nudge RRT's random sampler down different branches. Each
        successful path's terminal point is snapped back to the exact
        q_goal so the final pose is consistent across candidates.

        Ports the multi-candidate pattern from SplatSim's
        `TrajectoryGenerator._generate_multiple_path_candidates`: the
        per-IK loop runs at most `max_path_attempts_per_ik` consecutive
        attempts between successes, so num_attempts is a soft cap, not
        a hard total. Returns the list of successful paths (may be
        empty if every attempt failed).

        Skipped (and reduces to a single get_path call) when
        num_path_candidates_per_ik == 1 — preserves the original single-
        attempt behavior with zero overhead.
        """
        # Lazy import — keeps the module-level import surface free of
        # splatsim (an optional dep) and matches the pattern used in
        # plan(). Resolves to the same `get_path` used there.
        from splatsim.utils.rrt_path_utils import get_path

        num_target = self._num_path_candidates_per_ik
        if num_target <= 1:
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
                **self._collision_kwargs,
            )
            return [np.asarray(attempt, dtype=np.float64)] if attempt is not None else []

        paths: list[np.ndarray] = []
        attempts = 0
        max_attempts = self._max_path_attempts_per_ik
        scale = self._path_perturbation_scale
        while len(paths) < num_target and attempts < max_attempts:
            attempts += 1
            if len(paths) == 0:
                plan_start = q_start
                plan_goal = q_goal
            else:
                plan_start = np.clip(
                    q_start + np.random.uniform(-scale, scale, size=q_start.shape),
                    self._lower_limits,
                    self._upper_limits,
                )
                plan_goal = np.clip(
                    q_goal + np.random.uniform(-scale, scale, size=q_goal.shape),
                    self._lower_limits,
                    self._upper_limits,
                )
            attempt = get_path(
                plan_start,
                plan_goal,
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
                **self._collision_kwargs,
            )
            if attempt is None:
                continue
            arr = np.asarray(attempt, dtype=np.float64)
            # Snap terminal pose back to the exact q_goal so every
            # candidate's endpoint is identical (perturbation only
            # affects the middle of the path).
            if not np.allclose(arr[-1], q_goal):
                arr = np.vstack([arr, q_goal])
            paths.append(arr)
            attempts = 0  # reset between successes — matches SplatSim semantics
        return paths

    def _path_joint_arc_length(self, path: np.ndarray) -> float:
        """Sum of joint-space L2 distances between consecutive waypoints.

        Cheap: no pybullet calls. Tends to favor candidates that land close
        to `q_start` in configuration space even if the EE swings wide;
        use EE_ARC_LENGTH if you care more about cartesian path tidiness.
        """
        if path.shape[0] < 2:
            return 0.0
        deltas = np.diff(path, axis=0)
        return float(np.sum(np.linalg.norm(deltas, axis=1)))

    def _path_velocity_deviation(
        self,
        path: np.ndarray,
        recent_joint_velocity: np.ndarray,
    ) -> float:
        """Cosine distance between candidate's initial DIRECTION and the
        robot's recent DIRECTION (lower = better aligned).

        An earlier version computed an L2 distance between magnitudes
        directly, but that conflates two different units: the candidate's
        ``leading_deltas`` are spatial deltas between raw RRT waypoints
        (which aren't uniformly time-spaced), while ``recent_joint_velocity``
        is a real per-step velocity (rad / control-tick). Comparing their
        magnitudes ranked paths in a way that didn't survive Ruckig's
        time-parametrization, producing sustained high-velocity stretches
        in some recorded trajectories (e.g. one joint at 5 rad/s for many
        consecutive frames). Direction-only comparison sidesteps the
        unit mismatch — we ask "does the candidate START in the same
        direction the robot was already moving" without trying to match
        magnitudes that aren't comparable.

        Fallback: when the robot's recent velocity is near zero (typical
        when RRT triggers from a collision/stall — no direction to match),
        we delegate to EE arc-length so the candidate selection isn't
        random. Threshold is sized to per-step deltas at 30 Hz, where
        ~5e-4 rad/step ≈ 1.5e-2 rad/s of total joint motion across the
        ``velocity_match_window`` — below that, the direction is dominated
        by sensor noise.
        """
        if path.shape[0] < 2:
            # Degenerate path — no motion to compare against. Return a large
            # penalty so any candidate with real motion wins over this one.
            return float("inf")
        recent_vel = recent_joint_velocity.reshape(-1)[: self._num_dofs]
        recent_norm = float(np.linalg.norm(recent_vel))
        if recent_norm < 5e-4:
            # No meaningful direction — fall back to EE arc length to avoid
            # picking among candidates at random.
            return self._path_ee_arc_length(path)
        window = max(1, min(self._velocity_match_window, path.shape[0] - 1))
        leading_deltas = np.diff(path[: window + 1], axis=0)  # [window, num_dofs]
        candidate_vel = leading_deltas.mean(axis=0)
        cand_norm = float(np.linalg.norm(candidate_vel))
        if cand_norm < 1e-9:
            # Candidate path starts with all-zero deltas (degenerate RRT
            # sampling). Maximally misaligned with any nonzero recent_vel.
            return 1.0
        cos_sim = float(np.dot(candidate_vel, recent_vel) / (cand_norm * recent_norm))
        # Clip to [-1, 1] to guard against rounding errors, then 1 - cos:
        # range becomes [0, 2], with 0 = perfectly aligned, 2 = opposite.
        cos_sim = max(-1.0, min(1.0, cos_sim))
        return 1.0 - cos_sim

    def _path_ee_arc_length(self, path: np.ndarray) -> float:
        """Sum of Euclidean distances between consecutive end-effector world
        positions along a joint-space ``path`` of shape ``[N, num_dofs]``.

        Uses fast FK: resetJointState (no physics step) on the arm joints,
        then ``getLinkState(..., computeForwardKinematics=True)``. This leaves
        the robot at the path's terminal config — callers in ``plan()`` are
        responsible for restoring joint state from the snapshot taken on entry.
        """
        if path.shape[0] < 2:
            return 0.0
        ee_positions = np.empty((path.shape[0], 3), dtype=np.float64)
        for k in range(path.shape[0]):
            for j_idx, qi in zip(self._joint_indices, path[k].tolist(), strict=True):
                p.resetJointState(self._robot_id, j_idx, float(qi), physicsClientId=self._pb_client)
            link_state = p.getLinkState(
                self._robot_id,
                self._ee_link_index,
                computeForwardKinematics=True,
                physicsClientId=self._pb_client,
            )
            ee_positions[k] = link_state[0]  # worldLinkFramePosition
        return float(np.sum(np.linalg.norm(np.diff(ee_positions, axis=0), axis=1)))

    @staticmethod
    def _densify_path_for_scoring(path: np.ndarray, max_step_rad: float = 0.05) -> np.ndarray:
        """Linearly interpolate between consecutive waypoints so adjacent
        dense waypoints differ by no more than ``max_step_rad`` (per-joint
        L-infinity). Used by ``_path_min_pair_clearance`` to score the
        trajectory the robot will actually execute, not just RRT's sparse
        sample points — RRT can return paths with 4-6 waypoints whose
        interpolated configs in between pass through tighter self-collision
        configs than the waypoints themselves.

        Densification is PURELY a scoring-side concern: callers of
        ``_path_min_pair_clearance`` pass the sparse RRT path, this helper
        densifies internally, the SAME sparse path is what gets returned
        from ``plan()`` and handed to ruckig. So the actual execution
        trajectory is unchanged; only the score reflects the dense view.

        Args:
            path: ``[N, num_dofs]`` sparse joint-space path from RRT.
            max_step_rad: max allowed per-joint L-infinity step between
                consecutive dense waypoints. 0.05 rad (~3°) matches the
                RRT-Connect collision-check resolution, so links move on
                the order of 10 mm between dense waypoints — enough to
                catch self-collisions in the 5-50 mm gap range we care
                about for scoring.

        Returns:
            ``[M, num_dofs]`` densified path with M >= N. The first and
            last waypoints are preserved exactly; intermediate sparse
            waypoints are preserved as the boundary points of segments.
        """
        if path.shape[0] < 2:
            return path  # nothing to interpolate between
        dense_chunks: list[np.ndarray] = []
        for k in range(path.shape[0] - 1):
            q0 = path[k]
            q1 = path[k + 1]
            # Number of intermediate steps so per-step L-infinity delta is
            # at most max_step_rad. n_steps=1 means just [q0, q1] (no extra).
            max_delta = float(np.max(np.abs(q1 - q0)))
            n_steps = max(1, int(np.ceil(max_delta / max_step_rad)))
            # Sample n_steps+1 points from q0 to q1 INCLUSIVE; drop the
            # last so the next segment's first point isn't duplicated.
            ts = np.linspace(0.0, 1.0, n_steps + 1)
            seg = q0[None, :] + ts[:, None] * (q1 - q0)[None, :]
            dense_chunks.append(seg[:-1])
        # Append the final waypoint that we trimmed off the last segment.
        dense_chunks.append(path[-1:].reshape(1, -1))
        return np.concatenate(dense_chunks, axis=0)

    def _path_min_pair_clearance(self, path: np.ndarray) -> float:
        """Negated minimum non-adjacent-link-pair distance over a joint-space
        path. Lower returned value = larger clearance = SAFER path.

        For each waypoint along the path, the robot is snapped to that joint
        config and every non-adjacent link pair (excluding the
        URDF-structurally-close pairs declared via
        ``self_collision_skip_pairs``) is queried for its actual minimum
        distance. The smallest such distance across the path is what's
        returned (negated, since the planner's scoring convention is
        lower-is-better and we want LARGER clearance to win).

        IMPORTANT: the input ``path`` is RRT's SPARSE waypoint sequence,
        but scoring is done on a DENSIFIED copy (linear interpolation,
        per-joint L-infinity step ≤ 0.05 rad) so configurations between
        the sparse waypoints aren't missed. The sparse path is what
        ruckig sees — densification is internal to scoring only. See
        ``_densify_path_for_scoring`` for the rationale.

        Query distance cap is set to ``_PAIR_CLEARANCE_QUERY_CAP`` (10 cm
        by default) so far-apart pairs early-out cheaply in pybullet's
        ``getClosestPoints``. A waypoint where every pair is > cap apart
        contributes +cap to the running minimum (saturated), which keeps
        the score comparable across paths with mostly-roomy waypoints.

        Side effect: leaves the robot at the dense path's terminal config
        (= the sparse path's terminal config). The ``plan()`` caller takes
        a joint-state snapshot on entry and restores from it after scoring.
        """
        if path.shape[0] < 1:
            return 0.0  # degenerate; no waypoints to evaluate
        # Densify so RRT's sparse waypoints don't hide tighter
        # configurations between them. Hardcoded step matches the BiRRT
        # collision-check resolution so the scorer never "sees" finer
        # granularity than the planner could have rejected at planning
        # time — keeps the scoring grounded in what was actually feasible.
        scoring_path = self._densify_path_for_scoring(path, max_step_rad=0.05)
        # 10 cm cap — beyond this the pair isn't influencing the score.
        # Larger cap = more compute per query; smaller = more pairs saturate
        # at the cap, making the score less discriminating.
        _PAIR_CLEARANCE_QUERY_CAP = 0.10  # noqa: N806 — read as a constant, kept uppercase for clarity

        # Lazy import — matches the pattern used elsewhere in this file
        # (e.g. plan_segment) to keep splatsim out of the module-level
        # import surface.
        from splatsim.utils.rrt_path_utils import are_adjacent_links

        # Build the structural-skip set once. Stored on the planner via
        # self._collision_kwargs["self_collision_skip_pairs"] (a list of
        # (a, b) tuples) when set by the SA config / env oracle dispatch.
        _skip_raw = self._collision_kwargs.get("self_collision_skip_pairs") or []
        _skip_set = {frozenset((int(a), int(b))) for a, b in _skip_raw}

        # All robot links (-1 = body base + every joint's child link). Match
        # check_links_in_collision's default to keep the scoring consistent
        # with the feasibility check.
        n_joints = p.getNumJoints(self._robot_id, physicsClientId=self._pb_client)
        all_links = list(range(-1, n_joints))

        # Precompute non-adjacent + non-skip pair list once (it's a property
        # of the URDF, not the path). Saves the adjacency check on every
        # waypoint.
        pair_list: list[tuple[int, int]] = []
        for a, b in itertools.combinations(all_links, 2):
            if frozenset((a, b)) in _skip_set:
                continue
            if are_adjacent_links(self._robot_id, a, b, physics_client_id=self._pb_client):
                continue
            pair_list.append((a, b))

        # Track the minimum across the DENSE path (interior + waypoints),
        # AND remember which pair set it. The full per-pair min/max table
        # is dumped on EVERY call (per-scenario, per-RRT-trigger) so the
        # user can spot structural offenders across all paths the robot
        # actually plans through — not just the first one (which may not
        # be representative of all path topologies).
        min_dist_over_path = _PAIR_CLEARANCE_QUERY_CAP
        min_pair: tuple[int, int] | None = None
        # Diagnostic mode determines (a) whether to log the per-pair table
        # at all, (b) whether to use the larger 5m getClosestPoints cap
        # for tracking, and (c) whether to log once per planner instance
        # or every call. See SharedAutonomyConfig.rrt_diagnostic_log_pairs.
        diag_mode = self._diagnostic_log_pairs
        diag_first_pending = diag_mode == "first" and not getattr(self, "_min_pair_diag_logged", False)
        should_log_table = diag_mode == "always" or diag_first_pending
        # Cap rules: when logging the full table we use 5m so far-apart
        # pairs are also captured; when off (or already-first-logged) we
        # use the lean 10cm cap so scoring loop stays fast (~200ms vs
        # ~500-1000ms). Score itself is invariant — see clamp below.
        scoring_cap = 5.0 if should_log_table else _PAIR_CLEARANCE_QUERY_CAP
        # Per-pair MIN/MAX only tracked when logging the table — small
        # dict overhead, but pointless if we're not going to print it.
        per_pair_min: dict[tuple[int, int], float] = {} if should_log_table else {}
        per_pair_max: dict[tuple[int, int], float] = {} if should_log_table else {}
        for k in range(scoring_path.shape[0]):
            # Snap to this waypoint's joint config (FK only, no physics).
            for j_idx, qi in zip(self._joint_indices, scoring_path[k].tolist(), strict=True):
                p.resetJointState(self._robot_id, j_idx, float(qi), physicsClientId=self._pb_client)
            for a, b in pair_list:
                pts = p.getClosestPoints(
                    self._robot_id,
                    self._robot_id,
                    distance=scoring_cap,
                    linkIndexA=a,
                    linkIndexB=b,
                    physicsClientId=self._pb_client,
                )
                if not pts:
                    continue  # > cap → saturated, doesn't update the min
                # pts[i][8] is the contact-distance field (negative = penetration).
                d = float(pts[0][8])
                # min_dist_over_path tracking clamps to the original
                # _PAIR_CLEARANCE_QUERY_CAP so the score the planner sees
                # is identical regardless of whether we used the lean
                # 10cm cap or the diagnostic's larger 5m cap.
                if d < _PAIR_CLEARANCE_QUERY_CAP and d < min_dist_over_path:
                    min_dist_over_path = d
                    min_pair = (a, b)
                if should_log_table:
                    prev_min = per_pair_min.get((a, b), scoring_cap)
                    if d < prev_min:
                        per_pair_min[(a, b)] = d
                    prev_max = per_pair_max.get((a, b), float("-inf"))
                    if d > prev_max:
                        per_pair_max[(a, b)] = d

        link_name = (  # noqa: E731 — compact local helper used only inside this loop
            lambda i: "WORLD"
            if i == -1
            else p.getJointInfo(self._robot_id, i, physicsClientId=self._pb_client)[12].decode()
        )
        # Dump EVERY non-adjacent pair that produced a close-points hit
        # over this path, sorted ascending by min distance. Includes a
        # range column (max-min) so structural pairs (range ~ 0 mm = rigid
        # sub-chain) are visually distinct from articulating pairs (range
        # > tens of mm = real motion). Gated by diag_mode — see comments
        # at the top of this block.
        # _STRUCTURAL_RANGE_MM_HINT: range below this is auto-flagged in
        # the log as "STRUCTURAL?" — a hint, not a hard threshold. The
        # user makes the final call on whether to skip.
        if should_log_table:
            if diag_first_pending:
                # First-mode bookkeeping — mark so subsequent calls
                # skip the table dump and use the lean cap.
                self._min_pair_diag_logged = True
            _STRUCTURAL_RANGE_MM_HINT = 5.0  # noqa: N806 — read as a constant, kept uppercase for clarity
            ranked = sorted(per_pair_min.items(), key=lambda kv: kv[1])
            logger.info(
                "MIN_PAIR_CLEARANCE structural-offender probe — "
                "ALL %d non-adjacent pairs that had a hit over this scored "
                "path (%d waypoints). Columns: min / max / range; pairs "
                "with range < %.1f mm are likely structural (joint motion "
                "doesn't move them apart) and good skip-list candidates:",
                len(ranked),
                scoring_path.shape[0],
                _STRUCTURAL_RANGE_MM_HINT,
            )
            for (a, b), dmin in ranked:
                dmax = per_pair_max[(a, b)]
                drange = dmax - dmin
                flag = "STRUCTURAL?" if drange * 1000.0 < _STRUCTURAL_RANGE_MM_HINT else "          "
                logger.info(
                    "  %s pair (%d,%d) %s vs %s : min %.3f / max %.3f / range %.3f mm",
                    flag,
                    a,
                    b,
                    link_name(a),
                    link_name(b),
                    dmin * 1000.0,
                    dmax * 1000.0,
                    drange * 1000.0,
                )
            n_structural = sum(
                1
                for (a, b), dmin in ranked
                if (per_pair_max[(a, b)] - dmin) * 1000.0 < _STRUCTURAL_RANGE_MM_HINT
            )
            logger.info(
                "MIN_PAIR_CLEARANCE structural-offender probe summary: "
                "%d/%d pairs flagged STRUCTURAL? (range < %.1f mm on this path).",
                n_structural,
                len(ranked),
                _STRUCTURAL_RANGE_MM_HINT,
            )
        # Per-call floor-pair line: always-on when diag is "always" or
        # when diag is "first" (since the per-pair table is also gated
        # by the same conditions, this stays consistent — first-mode
        # gets the table + floor line on first call, then silence).
        # In "off" mode there's no diagnostic output at all.
        if should_log_table and min_pair is not None:
            logger.info(
                "  → score floor set by pair (%d,%d) %s vs %s at %.3f mm",
                min_pair[0],
                min_pair[1],
                link_name(min_pair[0]),
                link_name(min_pair[1]),
                min_dist_over_path * 1000.0,
            )

        # Negate so lower returned value = larger clearance = better path.
        return -min_dist_over_path

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
        from splatsim.utils.rrt_path_utils import check_links_in_collision

        # Use the configured obstacle clearance (or SplatSim's default if
        # not overridden) so the escape's getClosestPoints query, weight
        # formula, and step-size scaling all use the SAME threshold as the
        # BiRRT collision_fn. Otherwise we could exit escape thinking the
        # robot is clear at 0.01 m while the planner refuses to plan
        # because it sees a violation at 0.04 m.
        _eff_clearance = self._effective_obstacle_clearance()

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
                        distance=_eff_clearance,
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
                    weight = max(_eff_clearance - dist, 1e-3)
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
                    **self._collision_kwargs,
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
                step = step_size * (1.0 + min(max_pen / _eff_clearance, 5.0))

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
            **self._collision_kwargs,
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

    def _set_joints_to(self, q: np.ndarray) -> None:
        """Snap pybullet joints to a config (no physics step). Used to restore
        the planning client's state between escape attempts in
        ``_try_escape_chain`` and at the start of gradient-escape iters.
        """
        for j_idx, qi in zip(self._joint_indices, q, strict=False):
            p.resetJointState(self._robot_id, j_idx, float(qi), physicsClientId=self._pb_client)

    def _find_worst_self_collision_pair(
        self,
        query_cap: float = 0.10,
    ) -> tuple[tuple[int, int] | None, float]:
        """Find the non-adjacent link pair with the smallest current clearance.

        Iterates all non-adjacent link pairs (respecting
        ``self_collision_skip_pairs``), queries getClosestPoints, returns
        the pair with the lowest distance (most penetrating / closest).
        Pairs farther than ``query_cap`` apart are ignored (pybullet
        returns no points for those — early-out per pair).

        Returns ``(pair, distance)`` or ``(None, +inf)`` if no pair is
        within ``query_cap``. Distance can be NEGATIVE (penetration).

        Reads the planner's currently-snapped joint state — caller must
        ensure pybullet is at the desired q before calling.
        """
        from splatsim.utils.rrt_path_utils import are_adjacent_links

        _skip_raw = self._collision_kwargs.get("self_collision_skip_pairs") or []
        _skip_set = {frozenset((int(a), int(b))) for a, b in _skip_raw}
        n_joints = p.getNumJoints(self._robot_id, physicsClientId=self._pb_client)

        worst_dist = float("inf")
        worst_pair: tuple[int, int] | None = None
        for a, b in itertools.combinations(list(range(-1, n_joints)), 2):
            if frozenset((a, b)) in _skip_set:
                continue
            if are_adjacent_links(self._robot_id, a, b, physics_client_id=self._pb_client):
                continue
            pts = p.getClosestPoints(
                self._robot_id,
                self._robot_id,
                distance=query_cap,
                linkIndexA=a,
                linkIndexB=b,
                physicsClientId=self._pb_client,
            )
            if not pts:
                continue
            d = float(pts[0][8])
            if d < worst_dist:
                worst_dist = d
                worst_pair = (a, b)
        return worst_pair, worst_dist

    def _escape_self_collision_gradient(
        self,
        q_start: np.ndarray,
        max_iters: int = 60,
        step_size: float = 0.02,
        eps: float = 0.005,
    ) -> np.ndarray | None:
        """Escape SELF-collision via finite-difference gradient ascent on the
        worst-pair clearance.

        The default ``_escape_collision`` handles OBSTACLE collisions
        (contact-normal IK steps) and falls back to ``+z lift`` when no
        obstacle pair is detected. But the lift fallback can't fix
        self-collisions: e.g., forearm vs wrist_camera depends on
        wrist_1/wrist_2/wrist_3 joint angles, not EE height — lifting
        the assembly leaves the offending pair untouched.

        This method instead does direct joint-space search. Each iter:
          1. Find the worst-clearance non-adjacent link pair (the "active pair").
          2. Compute per-joint finite-difference gradient of that pair's
             clearance w.r.t. each joint: ∂d/∂q_i ≈ (d(q+ε e_i) − d(q−ε e_i)) / 2ε.
          3. Step q in the gradient direction (toward larger clearance).
          4. Re-check; stop if the standard collision_fn clears.

        ``query_cap`` for pair distances is 10 cm; pairs beyond that don't
        influence the gradient (gracefully — pybullet returns no points).

        Cost: 2 × num_dofs getClosestPoints queries per iter (plus the
        worst-pair scan, which is ~N² pair queries each iter, but bounded
        by skip_pairs filtering). Typical: ~12-25 queries per iter ×
        20-60 iters = 0.3-1.5 sec total. Cheap enough for an escape that
        only fires on failed plans.

        Returns waypoints ``[N, num_dofs]`` with ``waypoints[0] == q_start``
        and ``waypoints[-1]`` a cleared config, or None on failure.

        Side effect: leaves pybullet at the final attempted q. Caller is
        responsible for restoring (the surrounding plan() finally block
        already snapshots and restores).
        """
        from splatsim.utils.rrt_path_utils import check_links_in_collision

        # Use the configured self_collision_clearance as the success threshold.
        # The standard check is "clearance >= threshold". Default 0.0 = no
        # penetration. We add a tiny positive epsilon so the gradient
        # has a target slightly above the threshold (prevents oscillating
        # at exactly the threshold boundary).
        target_clearance = self._collision_kwargs.get("self_collision_clearance", 0.0) or 0.0
        target_clearance_plus = target_clearance + 1e-4

        q = np.asarray(q_start, dtype=np.float64).copy()
        waypoints: list[np.ndarray] = [q.copy()]

        for it in range(max_iters):
            self._set_joints_to(q)
            # Find worst non-adjacent self-pair at current q.
            worst_pair, worst_dist = self._find_worst_self_collision_pair()

            if worst_pair is not None and worst_dist >= target_clearance_plus:  # noqa: SIM102
                # Worst pair cleared. Confirm via the standard collision_fn
                # (which also checks obstacle collisions) before returning.
                if not check_links_in_collision(
                    self._robot_id,
                    self._joint_indices,
                    None,
                    self._loaded_obstacle_ids,
                    obstacle_names=self._obstacle_names,
                    skip_pairs=self._skip_pairs,
                    verbose=False,
                    physics_client_id=self._pb_client,
                    **self._collision_kwargs,
                ):
                    logger.info(
                        "Self-collision gradient escape: cleared after %d iter(s) "
                        "(final worst pair %s at %.4fm).",
                        it,
                        worst_pair,
                        worst_dist,
                    )
                    return np.asarray(waypoints, dtype=np.float64)
            if worst_pair is None:
                # No non-adjacent pair within query cap → either we already
                # cleared OR the collision is between adjacent links (which
                # we don't optimize). Random kick + continue.
                q_new = q + np.random.uniform(-step_size, step_size, self._num_dofs) * 0.5
                q_new = np.clip(q_new, self._lower_limits, self._upper_limits)
                waypoints.append(q_new.copy())
                q = q_new
                continue

            # Compute finite-difference gradient of worst-pair clearance.
            # Two queries per joint (±ε), so 2 * num_dofs getClosestPoints calls.
            a, b = worst_pair
            grad = np.zeros(self._num_dofs)
            for i in range(self._num_dofs):
                d_pair = 0.0
                for sign, delta in ((+1, +eps), (-1, -eps)):
                    q_perturbed = q.copy()
                    q_perturbed[i] = float(
                        np.clip(
                            q_perturbed[i] + delta,
                            self._lower_limits[i],
                            self._upper_limits[i],
                        )
                    )
                    self._set_joints_to(q_perturbed)
                    pts = p.getClosestPoints(
                        self._robot_id,
                        self._robot_id,
                        distance=0.10,
                        linkIndexA=a,
                        linkIndexB=b,
                        physicsClientId=self._pb_client,
                    )
                    d = float(pts[0][8]) if pts else 0.10
                    d_pair += sign * d
                grad[i] = d_pair / (2 * eps)
            # Restore pybullet to current q (perturbations mutated it).
            self._set_joints_to(q)

            grad_norm = float(np.linalg.norm(grad))
            if grad_norm < 1e-6:
                # Flat region — random kick to escape the saddle.
                q_new = q + np.random.uniform(-step_size, step_size, self._num_dofs)
            else:
                q_new = q + step_size * grad / grad_norm
            q_new = np.clip(q_new, self._lower_limits, self._upper_limits)
            waypoints.append(q_new.copy())
            q = q_new

            if (it + 1) % 10 == 0:
                logger.info(
                    "Self-collision gradient escape iter %d/%d: worst pair (%d,%d) at %.4fm, ||grad||=%.4f",
                    it + 1,
                    max_iters,
                    worst_pair[0],
                    worst_pair[1],
                    worst_dist,
                    grad_norm,
                )

        # Final check.
        self._set_joints_to(q)
        if check_links_in_collision(
            self._robot_id,
            self._joint_indices,
            None,
            self._loaded_obstacle_ids,
            obstacle_names=self._obstacle_names,
            skip_pairs=self._skip_pairs,
            verbose=False,
            physics_client_id=self._pb_client,
            **self._collision_kwargs,
        ):
            logger.warning(
                "Self-collision gradient escape failed after %d iter(s) "
                "(final worst pair %s at %.4fm). Returning None.",
                max_iters,
                worst_pair,
                worst_dist,
            )
            return None
        logger.info(
            "Self-collision gradient escape succeeded after %d waypoint(s).",
            len(waypoints) - 1,
        )
        return np.asarray(waypoints, dtype=np.float64)

    def _try_escape_chain(self, q_start: np.ndarray) -> np.ndarray | None:
        """Run all escape modes in sequence, restoring pybullet joints to
        ``q_start`` between attempts so each mode starts from the same
        ground-truth pose.

        Order:
          1. Contact-normal escape (``_escape_collision``, existing). Best
             for obstacle collisions — pushes the EE along the aggregated
             contact normal. Falls back to ``+z lift`` internally when it
             stalls or sees no obstacle pairs.
          2. Self-collision gradient escape
             (``_escape_self_collision_gradient``, new). Best for wrist-
             pretzel / arm-on-arm self-collisions that ``+z lift`` can't
             fix because the offending pair's clearance depends on joint
             angles, not EE position.

        Returns the first successful escape's waypoints (with
        ``waypoints[0] == q_start``), or None if both modes failed. On
        failure, the pybullet state is restored to q_start so the caller
        sees a consistent rollback point.
        """
        # Ensure pybullet is at q_start before the first attempt.
        self._set_joints_to(q_start)
        waypoints = self._escape_collision(q_start)
        if waypoints is not None:
            return waypoints
        # Contact-normal failed — restore pybullet to q_start (it's been
        # mutated by 60 iters of escape attempts) and try the gradient mode.
        logger.info(
            "Contact-normal escape failed — restoring q_start and trying self-collision gradient escape.",
        )
        self._set_joints_to(q_start)
        waypoints = self._escape_self_collision_gradient(q_start)
        if waypoints is not None:
            return waypoints
        # Both failed — leave pybullet at q_start for the caller.
        self._set_joints_to(q_start)
        return None

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
            **self._collision_kwargs,
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


def check_chunk_collision(
    pb_client: int,
    robot_id: int,
    joint_indices: list[int],
    q_current: np.ndarray,
    chunk_dof_actions: np.ndarray,
    action_format: str,
    obstacle_ids: list[int],
    obstacle_clearance: float | None = None,
    self_collision_clearance: float | None = None,
    self_collision_skip_pairs: list[tuple[int, int]] | None = None,
    obstacle_names: list[str] | None = None,
    link_indices_to_check: list[int] | None = None,
) -> tuple[bool, int | None, str | None]:
    """Forward-kinematics safety sweep over a predicted action chunk.

    Snaps the planning-client robot through each future joint config that
    would result from executing ``chunk_dof_actions`` and reports whether
    any of them collides (obstacle or self) under the same collision contract
    RRT uses (``check_links_in_collision``).

    Used by the SharedAutonomyPolicyWrapper's future-chunk predictive shield:
    when ``rrt_collision_detection=future_chunk``, this is called every
    select_action tick to decide whether to preempt the policy and trigger
    RRT BEFORE the colliding waypoint actually executes. No teleport / no
    rewind — the wrapper triggers from the robot's CURRENT, continuous-motion
    state.

    Args:
        pb_client: pybullet physics client id (the planning one, with
            obstacles already loaded by RRTToGoalPlanner.load_obstacles).
        robot_id: planning robot's body id.
        joint_indices: movable joint indices for the DOF arm (length = num_dofs).
        q_current: (num_dofs,) current robot joint state. Used as the
            integration base for ``action_format='rel'`` and to restore the
            robot pose on exit.
        chunk_dof_actions: (n_steps, num_dofs) — the future joint actions
            from the policy chunk, already DOF-sliced (gripper dim dropped)
            and denormalized into raw joint-space units (radians).
        action_format: ``"rel"`` — offset from chunk-START state (NOT a
            per-step delta; see body of this function for the math). Each
            ``future_q[k] = q_current + chunk[k]``. ``"abs"`` — absolute
            joint targets per step (``future_q[k] = chunk[k]``).
        obstacle_ids: bodies in the planning client to check robot links
            against (matches what RRT uses).
        obstacle_clearance / self_collision_clearance / self_collision_skip_pairs:
            same semantics as the planner's ``_collision_kwargs``. Defaulted
            to SplatSim's built-in defaults when None.
        obstacle_names: optional pretty-print names for logging the offender.

    Returns:
        ``(any_collides, first_step_idx, kind)``:
        - any_collides: True if any future config collides.
        - first_step_idx: 0-indexed offset into chunk_dof_actions where
          collision was first detected, or None if no collision.
        - kind: ``"obstacle"`` or ``"self"`` (from check_links_in_collision)
          identifying which kind of collision tripped the check, or None.

    Side effect: leaves the planning robot at q_current on exit (so
    subsequent RRT planning starts from the same config). Each step
    snapshot uses p.resetJointState, which doesn't run physics.
    """
    if chunk_dof_actions.shape[0] == 0:
        return False, None, None
    if action_format not in ("rel", "abs"):
        raise ValueError(f"action_format must be 'rel' or 'abs', got {action_format!r}")

    # Lazy import — keep optional dependency surface contained.
    from splatsim.utils.rrt_path_utils import check_links_in_collision

    n_dof = len(joint_indices)
    if chunk_dof_actions.shape[1] != n_dof:
        raise ValueError(
            f"chunk_dof_actions has {chunk_dof_actions.shape[1]} action dims; "
            f"expected num_dofs={n_dof}. Caller must DOF-slice before passing."
        )
    q_current_arr = np.asarray(q_current, dtype=np.float64).reshape(-1)[:n_dof]

    # Build the absolute future joint trajectory.
    #
    # IMPORTANT — 'rel' is NOT a per-step delta format. It's "offset from
    # the chunk-START obs state". The training-time `to_relative_actions`
    # (and inference-time `to_absolute_actions`) broadcasts a SINGLE state
    # across all chunk timesteps (see
    # ``relative_action_processor.to_absolute_actions:122-126`` —
    # ``state_offset.unsqueeze(-2)`` widens the state to time dim, then
    # ``+=`` adds it to every step). So target k = chunk[k] + q_current,
    # NOT cumsum(chunk[..k]) + q_current. Using cumsum here would
    # overestimate the predicted motion by ~k×, causing the FK shield to
    # hallucinate collisions far beyond where the policy is actually
    # committing to go.
    #
    # For 'abs', chunk[k] IS the absolute joint target k directly.
    chunk_arr = np.asarray(chunk_dof_actions, dtype=np.float64)
    if action_format == "rel":  # noqa: SIM108 — branches are differently commented; ternary would lose the explanatory comments
        future_qs = q_current_arr[None, :] + chunk_arr
    else:  # abs
        future_qs = chunk_arr

    # Snap & check each future q. Return early on first collision so the
    # cost is bounded by the position of the offending step.
    collide_kwargs = {}
    if obstacle_clearance is not None:
        collide_kwargs["obstacle_clearance"] = obstacle_clearance
    if self_collision_clearance is not None:
        collide_kwargs["self_collision_clearance"] = self_collision_clearance
    if self_collision_skip_pairs:
        collide_kwargs["self_collision_skip_pairs"] = self_collision_skip_pairs
    if obstacle_names is not None:
        collide_kwargs["obstacle_names"] = obstacle_names
    # When the caller doesn't scope the link set, default to everything
    # downstream of (and including) the first DOF — i.e., skip the static
    # base_link (0) and world frame (-1) which never move and are always
    # close to the table top, causing spurious obstacle-clearance hits.
    # The RRT planner does this scoping for its own collision checks; the
    # shield must do the same or every shielded chunk gets flagged as
    # "base_link vs table" no matter what the policy actually predicts.
    if link_indices_to_check is None:
        n_joints = p.getNumJoints(robot_id, physicsClientId=pb_client)
        link_indices_to_check = list(range(1, n_joints))
    collide_kwargs["link_indices_to_check"] = link_indices_to_check

    try:
        for k in range(future_qs.shape[0]):
            # verbose=False keeps the per-tick log quiet. The caller's
            # "Future-chunk shield: predicted X collision at step Y"
            # line is the only shield output in production. Flip to True
            # temporarily if you need per-pair attribution for debugging.
            colliding, kind = check_links_in_collision(
                robot_id,
                joint_indices,
                future_qs[k].tolist(),
                obstacle_ids,
                verbose=False,
                physics_client_id=pb_client,
                return_kind=True,
                **collide_kwargs,
            )
            if colliding:
                return True, k, kind
        return False, None, None
    finally:
        # Restore the robot to q_current so subsequent RRT planning starts
        # from the same physical state the wrapper just queried.
        for j_idx, qi in zip(joint_indices, q_current_arr.tolist(), strict=True):
            p.resetJointState(robot_id, j_idx, float(qi), physicsClientId=pb_client)


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
