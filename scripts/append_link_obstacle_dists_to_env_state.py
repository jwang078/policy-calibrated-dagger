#!/usr/bin/env python3
"""Retrofit an existing planar-3joint LeRobotDataset with per-link min-obstacle
distances appended to `observation.environment_state`.

Motivation: adding EE xy to env_state let policies shortcut-learn an EE-space
controller that ignores obstacles (obstacles collide with ARM LINKS, not the EE
tip). Appending SIGNED per-link min-distances to any obstacle gives the policy a
salient obstacle signal it can't ignore. The SplatSim base class now emits these
dims automatically for new recordings; this script retrofits existing datasets
recorded before the change so they can be mixed with new data in one training
run (env_state shapes must match across the sub-datasets a MultiLeRobotDataset
loads).

Layout change (planar_3joint with 2 obstacles):
  BEFORE (width=8):
    [block_x, block_z, obstacle_1_x, obstacle_1_z, obstacle_2_x, obstacle_2_z,
     ee_x, ee_z]
  AFTER  (width=11):
    [block_x, block_z, obstacle_1_x, obstacle_1_z, obstacle_2_x, obstacle_2_z,
     ee_x, ee_z, link_1_min_dist, link_2_min_dist, link_3_min_dist]

Obstacle positions come DIRECTLY from `splatsim_object_configs` in
episodes.parquet (each episode records its per-object `initial_position`), so no
scenario-index lookup is required.

The distance metric mirrors `check_links_in_collision` /
`min_distance_to_obstacles` in rrt_path_utils.py — PyBullet's `getClosestPoints`
signed `contactDistance` (index 8). Negative means the link is INSIDE the
obstacle's inflated volume by that amount (same semantics as the base
class's runtime feature). We take min across obstacles per link.

Modifies the dataset IN PLACE with a `_pre_link_dists_backup` sibling directory
created first — matches the lerobot-edit-dataset convention.

Usage:
  bash -c 'source ~/miniforge3/etc/profile.d/conda.sh && conda activate splatsim \
    && python my_scripts/append_link_obstacle_dists_to_env_state.py \
        --repo_id=JennyWWW/planar_3joint_2 \
        --n_obstacles=2 \
        [--max_query_dist=1.0] [--sentinel=1.0] [--dry_run]'
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pybullet as p

# Repo-relative URDF path recorded in splatsim_robot_config.urdf_path is like
# "./splatsim/robot_definitions/urdf/planar_3joint.urdf" — resolve it against
# the SplatSim repo root.
SPLATSIM_ROOT = Path.home() / "code" / "SplatSim"

log = logging.getLogger("append_link_obstacle_dists")


def _resolve_urdf_path(rel_or_abs: str) -> Path:
    p_ = Path(rel_or_abs)
    if p_.is_absolute() and p_.exists():
        return p_
    # Recorded paths start with "./splatsim/..." — resolve against SplatSim repo root.
    if rel_or_abs.startswith("./"):
        rel_or_abs = rel_or_abs[2:]
    candidate = SPLATSIM_ROOT / rel_or_abs
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Could not resolve robot URDF path {rel_or_abs!r} — tried {candidate}. "
        f"Set SPLATSIM_ROOT env var if the SplatSim repo lives elsewhere."
    )


def _load_urdf_scene(
    client_id: int, robot_urdf: Path, obstacle_specs: list[dict]
) -> tuple[int, list[int], list[int]]:
    """Load the arm + obstacle boxes; return (robot_id, obstacle_ids, movable_link_indices).

    Movable link indices: PyBullet joint types other than JOINT_FIXED (=4),
    capped at 3 (num_dofs) so a hypothetical gripper joint doesn't sneak in.
    """
    robot_id = p.loadURDF(
        str(robot_urdf), basePosition=[0, 0, 0], useFixedBase=True, physicsClientId=client_id
    )

    # Enumerate movable arm links (matches _resolve_link_obstacle_dist_links in
    # sim_robot_pybullet_base.py: JOINT_FIXED = 4, cap at num_dofs=3 for planar).
    movable_links = []
    n_joints = p.getNumJoints(robot_id, physicsClientId=client_id)
    for j in range(n_joints):
        info = p.getJointInfo(robot_id, j, physicsClientId=client_id)
        if info[2] != 4:  # not JOINT_FIXED
            movable_links.append(j)
    movable_links = movable_links[:3]  # planar num_dofs

    # Load each obstacle as a box collision body at its recorded initial_position.
    obstacle_ids = []
    for spec in obstacle_specs:
        half = np.asarray(spec["size"], dtype=float) / 2.0
        pos = np.asarray(spec["initial_position"], dtype=float).tolist()
        quat = np.asarray(spec.get("initial_quat", [0, 0, 0, 1]), dtype=float).tolist()
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half.tolist(), physicsClientId=client_id)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half.tolist(), physicsClientId=client_id)
        body = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=pos,
            baseOrientation=quat,
            physicsClientId=client_id,
        )
        obstacle_ids.append(body)
    return robot_id, obstacle_ids, movable_links


def _compute_link_obstacle_min_dists(
    client_id: int,
    robot_id: int,
    joint_state: np.ndarray,
    movable_links: list[int],
    obstacle_ids: list[int],
    max_query: float,
    sentinel: float,
) -> list[float]:
    """Set robot joints to `joint_state`, return per-link min signed distance
    (across obstacles). Uses the same computation the runtime feature uses
    (`getClosestPoints` contactDistance).
    """
    # Set joint states. `joint_state` is (state_dim,) — for planar 3joint that's
    # [j1, j2, j3, gripper=0]. Only apply to movable joints; skip anything past
    # the movable count so a stray gripper dim doesn't error.
    for j_idx, j_val in zip(movable_links, joint_state[: len(movable_links)]):
        p.resetJointState(robot_id, j_idx, float(j_val), physicsClientId=client_id)

    out: list[float] = []
    for link_i in movable_links:
        min_d = sentinel
        for obs_id in obstacle_ids:
            pts = p.getClosestPoints(
                bodyA=robot_id,
                bodyB=obs_id,
                distance=max_query,
                linkIndexA=int(link_i),
                linkIndexB=-1,
                physicsClientId=client_id,
            )
            for pt in pts:
                d = float(pt[8])
                if d < min_d:
                    min_d = d
        out.append(min_d)
    return out


def _extract_obstacle_specs(episode_row: dict, name_pattern: str = "obstacle") -> list[dict]:
    """From an episodes.parquet row, pull out the obstacle specs (dicts with
    `initial_position` and `size`). Matches the auto-select pattern the base
    class uses at runtime (`ORACLE_LINK_OBSTACLE_NAME_PATTERN`)."""
    object_configs = episode_row["splatsim_object_configs"]
    specs = []
    for cfg in object_configs:
        name = cfg.get("name", "")
        if not name.startswith(name_pattern):
            continue
        specs.append(
            {
                "name": name,
                "initial_position": cfg["initial_position"],
                "size": cfg["size"],
                "initial_quat": cfg.get("initial_quat", [0.0, 0.0, 0.0, 1.0]),
            }
        )
    return specs


def _extract_robot_urdf(episode_row: dict) -> Path:
    cfg = episode_row["splatsim_robot_config"]
    urdf_path = cfg.get("urdf_path")
    if not urdf_path:
        raise ValueError("episode row is missing splatsim_robot_config.urdf_path")
    return _resolve_urdf_path(urdf_path)


def retrofit_dataset(
    repo_id: str, n_obstacles: int, max_query: float, sentinel: float, dry_run: bool
) -> None:
    hf_home = Path.home() / ".cache" / "huggingface" / "lerobot"
    root = hf_home / repo_id
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset not cached at {root}. Run a training/eval that touches it first, or download from hub."
        )

    info_path = root / "meta" / "info.json"
    info = json.load(open(info_path))
    old_shape = info["features"]["observation.environment_state"]["shape"]
    old_dim = int(old_shape[0])
    expected_new_dim = old_dim + 3  # 3 movable links → 3 new dims (matches planar num_dofs)

    log.info("=" * 72)
    log.info(f"Dataset: {repo_id}")
    log.info(f"  root: {root}")
    log.info(f"  current env_state dim: {old_dim}  →  new: {expected_new_dim}")
    log.info(f"  n_obstacles (expected): {n_obstacles}")
    log.info(f"  max_query_dist: {max_query} m   sentinel: {sentinel}")
    if dry_run:
        log.info("  --dry_run: no files will be modified")
    log.info("=" * 72)

    if old_dim == expected_new_dim:
        log.error(f"env_state already has width {old_dim} — did you already retrofit? Aborting.")
        sys.exit(1)

    # Load episodes.parquet — has the per-episode obstacle placements + URDF.
    ep_files = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not ep_files:
        raise FileNotFoundError(f"No episodes parquet under {root / 'meta' / 'episodes'}")
    ep_df = pd.concat([pd.read_parquet(f) for f in ep_files], ignore_index=True)
    log.info(f"Loaded {len(ep_df)} episode entries from {len(ep_files)} parquet file(s)")

    # Sanity: obstacle count in every episode's config == n_obstacles arg.
    sample_specs = _extract_obstacle_specs(ep_df.iloc[0].to_dict())
    if len(sample_specs) != n_obstacles:
        log.warning(
            f"Episode 0 has {len(sample_specs)} obstacles in splatsim_object_configs; "
            f"got --n_obstacles={n_obstacles} on CLI. Using what's in the parquet ({len(sample_specs)})."
        )
    log.info(f"Sample obstacle names: {[s['name'] for s in sample_specs]}")

    # Backup before modifying anything on disk.
    backup_root = root.with_name(root.name + "_pre_link_dists_backup")
    if not dry_run:
        if backup_root.exists():
            log.warning(
                f"Backup dir already exists at {backup_root}; leaving it as-is (assume prior aborted retrofit)."
            )
        else:
            log.info(f"Creating backup: {root} → {backup_root}")
            shutil.copytree(root, backup_root)

    # One PyBullet DIRECT client per RUN (not per episode) — reuse it. But we
    # need to CLEAR the scene per episode because each episode has different
    # obstacle positions. Pattern: reset the client's world at each episode
    # boundary rather than re-connecting.
    client_id = p.connect(p.DIRECT)
    log.info(f"Created PyBullet DIRECT client (id={client_id})")

    try:
        # Data parquets to update — one per (chunk, file) pair.
        data_files = sorted((root / "data").glob("chunk-*/file-*.parquet"))
        if not data_files:
            raise FileNotFoundError(f"No data parquet files under {root / 'data'}")
        log.info(f"Data parquet files to update: {len(data_files)}")

        # Group episodes by (data/chunk_index, data/file_index) so we can update
        # each parquet file in one pass. The mapping episode→file is in the
        # episodes.parquet columns 'data/chunk_index' + 'data/file_index'.
        ep_df["_data_key"] = list(zip(ep_df["data/chunk_index"], ep_df["data/file_index"]))
        eps_per_file = dict(list(ep_df.groupby("_data_key")))
        log.info(f"Episodes span {len(eps_per_file)} distinct data parquet file(s)")

        n_frames_total = 0
        n_frames_collided = 0

        for data_path in data_files:
            # Derive (chunk, file) from filename convention chunk-000/file-000.parquet.
            chunk_idx = int(data_path.parent.name.split("-")[1])
            file_idx = int(data_path.stem.split("-")[1])
            key = (chunk_idx, file_idx)
            eps = eps_per_file.get(key)
            if eps is None or len(eps) == 0:
                log.warning(f"  {data_path.relative_to(root)}: no episode metadata for key {key}; skipping.")
                continue

            log.info(f"  → {data_path.relative_to(root)}  ({len(eps)} episodes)")
            df = pd.read_parquet(data_path)
            # Build a new column: for each frame, the extended env_state list.
            new_env_state: list[np.ndarray] = [None] * len(df)  # type: ignore[list-item]

            for _, ep_row in eps.iterrows():
                ep_idx = int(ep_row["episode_index"])
                from_i = int(ep_row["dataset_from_index"])
                to_i = int(ep_row["dataset_to_index"])
                obstacle_specs = _extract_obstacle_specs(ep_row.to_dict())
                urdf_path = _extract_robot_urdf(ep_row.to_dict())

                # Fresh scene per episode.
                p.resetSimulation(physicsClientId=client_id)
                p.setGravity(0, 0, 0, physicsClientId=client_id)
                robot_id, obstacle_ids, movable_links = _load_urdf_scene(client_id, urdf_path, obstacle_specs)

                # Take frames for this episode from the data parquet's frame_index.
                mask = df["episode_index"] == ep_idx
                ep_slice = df[mask]
                if len(ep_slice) == 0:
                    log.warning(f"    episode {ep_idx}: no frames in this parquet; skipping.")
                    continue

                for _, frame_row in ep_slice.iterrows():
                    joint_state = np.asarray(frame_row["observation.state"], dtype=np.float64)
                    env_state_old = np.asarray(frame_row["observation.environment_state"], dtype=np.float64)
                    link_dists = _compute_link_obstacle_min_dists(
                        client_id,
                        robot_id,
                        joint_state,
                        movable_links,
                        obstacle_ids,
                        max_query,
                        sentinel,
                    )
                    combined = np.concatenate([env_state_old, np.asarray(link_dists, dtype=np.float64)])
                    row_idx = int(frame_row["index"]) - int(df["index"].iloc[0])
                    new_env_state[row_idx] = combined
                    n_frames_total += 1
                    if any(d < 0 for d in link_dists):
                        n_frames_collided += 1

            # Sanity — every row should have been written.
            missing = [i for i, v in enumerate(new_env_state) if v is None]
            if missing:
                raise RuntimeError(
                    f"Retrofit incomplete for {data_path}: {len(missing)} unwritten frames "
                    f"(first few row indices: {missing[:5]}). Aborting before writing."
                )

            # Overwrite the observation.environment_state column with the new arrays.
            df["observation.environment_state"] = pd.Series(new_env_state, index=df.index)
            if not dry_run:
                df.to_parquet(data_path, index=False)

        log.info("")
        log.info(f"Frames processed: {n_frames_total}")
        log.info(f"Frames with a link inside an obstacle (any link_dist < 0): {n_frames_collided}")

        # Update meta/info.json to reflect the new env_state width.
        info["features"]["observation.environment_state"]["shape"] = [expected_new_dim]
        if not dry_run:
            with open(info_path, "w") as f:
                json.dump(info, f, indent=2)
            log.info(f"Updated {info_path} (env_state shape → [{expected_new_dim}])")

        # Recompute per-episode env_state stats since they now cover more dims.
        # NOTE: we DON'T update the top-level meta/stats.json here — the user
        # should re-run compute_relative_stats.sh (rel-action stats) and the
        # main stats pass separately, since both depend on the new env_state
        # width via aggregate_stats. Print a reminder.
        log.info("")
        log.info("Retrofit complete.")
        log.info("Next steps:")
        log.info(
            "  1. Rerun `lerobot-edit-dataset --operation.type recompute_stats "
            f"--repo_id {repo_id} --new_repo_id {repo_id} --operation.overwrite true`"
        )
        log.info("     to refresh top-level meta/stats.json for the new env_state layout.")
        log.info(
            "  2. Rerun `my_scripts/compute_relative_stats.sh --dataset_repo="
            + repo_id
            + " --chunk_sizes=<horizon>` to refresh per-chunk rel-action sidecars."
        )
        log.info(f"  3. Bump env profile ENV_STATE_DIM from {old_dim} to {expected_new_dim}")
        log.info("     (already done for planar.sh / planar_oracle.sh — see the profile files).")
        log.info(f"Backup preserved at: {backup_root}")

    finally:
        p.disconnect(physicsClientId=client_id)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--repo_id", required=True, help='HF repo id, e.g. "JennyWWW/planar_3joint_2"')
    ap.add_argument(
        "--n_obstacles",
        type=int,
        default=2,
        help="Expected number of obstacles per episode (planar env default: 2)",
    )
    ap.add_argument(
        "--max_query_dist",
        type=float,
        default=1.0,
        help="PyBullet getClosestPoints query cap in meters (matches ORACLE_LINK_OBSTACLE_DIST_MAX_QUERY)",
    )
    ap.add_argument(
        "--sentinel",
        type=float,
        default=1.0,
        help="Sentinel returned when no obstacle is within max_query (matches ORACLE_LINK_OBSTACLE_DIST_SENTINEL)",
    )
    ap.add_argument("--dry_run", action="store_true", help="Compute everything but don't write files")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    retrofit_dataset(
        repo_id=args.repo_id,
        n_obstacles=args.n_obstacles,
        max_query=args.max_query_dist,
        sentinel=args.sentinel,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
