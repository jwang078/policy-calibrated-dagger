#!/usr/bin/env python
"""Retrofit planar LeRobot datasets with the gripper EE position appended to
observation.environment_state.

Background: the planar SplatSim envs now emit the end-effector position as the
LAST coords of observation.environment_state (ORACLE_STATE_INCLUDE_EE_POS in
sim_robot_pybullet_planar.py), so a state-only policy sees the gripper's (x,z)
directly instead of having to learn FK from the joint angles. Datasets recorded
before that change are len(coords) narrower; because the EE pose is a pure
function of the recorded observation.state joints, this script reconstructs it
exactly (pybullet FK on the same URDF the sim uses) — no re-recording needed.

Edits IN PLACE (videos/images untouched) after backing up meta/ to
meta_backup_pre_ee/. Idempotent: datasets whose env-state names already end
with the EE entries are skipped. Touches:
  1. data/**/*.parquet                 — append FK EE coords to each frame's env_state
  2. meta/episodes/**/*.parquet        — extend per-episode env_state stats (incl. quantiles)
  3. meta/stats.json                   — extend aggregated env_state stats
  4. meta/info.json                    — widen the feature shape + names (ee_x, ee_z)

Usage:
  python3 my_scripts/append_ee_to_env_state.py JennyWWW/planar_3joint_2 \
      JennyWWW/planar_d100_05dag_diff_r_dag1 [...] [--dry_run]

Each positional arg is a dataset root dir or a repo_id under $HF_LEROBOT_HOME.
Afterwards re-run my_scripts/compute_relative_stats.sh on each dataset to
refresh its *_recomputed_stats sidecar (the weighted-sampling stats source).
"""

import argparse
import glob
import json
import os
import shutil

import numpy as np
import pandas as pd

ENVS = "observation.environment_state"
STATE = "observation.state"
AXES = "xyz"
# Stats whose per-dim vectors get extended with the appended EE dims. `count`
# is a scalar and stays untouched.
VECTOR_STATS = ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99")
QUANTILES = {"q01": 0.01, "q10": 0.10, "q50": 0.50, "q90": 0.90, "q99": 0.99}


def _dataset_root(spec: str) -> str:
    if os.path.isdir(spec):
        return os.path.abspath(spec)
    home = os.environ.get("HF_LEROBOT_HOME") or os.path.expanduser("~/.cache/huggingface/lerobot")
    root = os.path.join(home, spec)
    if not os.path.isdir(root):
        raise SystemExit(f"Dataset not found: neither a directory nor under {home}: {spec}")
    return root


class PlanarFK:
    """FK for the planar arm via pybullet DIRECT on the same URDF the sim loads
    (fixed base at the origin, like SplatSim's objects.yaml planar_3joint entry).
    The arm joints are the FIRST `num_arm_joints` movable joints in URDF order —
    matching the server's get_joint_state/teleport convention (joint 0 is the
    fixed world_joint; gripper joints come after the arm and don't move the EE
    reference link)."""

    def __init__(self, urdf_path: str, ee_link_name: str, num_arm_joints: int):
        import pybullet as p
        import pybullet_utils.bullet_client as bc

        self._p = bc.BulletClient(connection_mode=p.DIRECT)
        self._rid = self._p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True)
        movable, self._ee_link = [], None
        for j in range(self._p.getNumJoints(self._rid)):
            info = self._p.getJointInfo(self._rid, j)
            if info[2] != p.JOINT_FIXED:
                movable.append(j)
            if info[12].decode("utf-8") == ee_link_name:
                self._ee_link = j
        if self._ee_link is None:
            raise SystemExit(f"EE link '{ee_link_name}' not found in {urdf_path}")
        if len(movable) < num_arm_joints:
            raise SystemExit(f"URDF has {len(movable)} movable joints < num_arm_joints={num_arm_joints}")
        self._arm_joints = movable[:num_arm_joints]

    def ee_positions(self, joint_states: np.ndarray) -> np.ndarray:
        """[N, >=num_arm_joints] joint angles -> [N, 3] world EE positions."""
        out = np.empty((len(joint_states), 3), dtype=np.float64)
        for n, q in enumerate(joint_states):
            for j, qj in zip(self._arm_joints, q):
                self._p.resetJointState(self._rid, j, float(qj))
            out[n] = self._p.getLinkState(self._rid, self._ee_link, computeForwardKinematics=True)[0]
        return out


def migrate_dataset(root: str, fk_cache: dict, args) -> None:
    print(f"\n=== {root}")
    info_path = os.path.join(root, "meta", "info.json")
    info = json.load(open(info_path))
    feats = info["features"]
    if ENVS not in feats:
        raise SystemExit(f"{root}: no {ENVS} feature — not an oracle dataset")

    coord_indices = tuple(int(i) for i in args.coord_indices.split(","))
    ee_names = [f"ee_{AXES[i]}" for i in coord_indices]
    old_dim = int(feats[ENVS]["shape"][0])
    names = list(feats[ENVS].get("names") or [f"env_{i}" for i in range(old_dim)])
    if names[-len(ee_names) :] == ee_names:
        print(f"  already has {ee_names} — skipping")
        return
    state_dim = int(feats[STATE]["shape"][0])
    num_arm_joints = state_dim - 1  # trailing entry is the gripper command

    fk_key = (args.urdf, args.ee_link_name, num_arm_joints)
    if fk_key not in fk_cache:
        fk_cache[fk_key] = PlanarFK(*fk_key)
    fk = fk_cache[fk_key]

    meta_bak = os.path.join(root, "meta_backup_pre_ee")
    if not args.dry_run:
        if os.path.exists(meta_bak):
            print(f"  backup already exists (prior run): {meta_bak}")
        else:
            shutil.copytree(os.path.join(root, "meta"), meta_bak)
            print(f"  backed up meta/ -> {meta_bak}")

    # ── 1. data parquets: FK per frame, append to env_state ──────────────────
    per_episode: dict[int, list[np.ndarray]] = {}
    all_vals: list[np.ndarray] = []
    for f in sorted(glob.glob(os.path.join(root, "data", "**", "*.parquet"), recursive=True)):
        df = pd.read_parquet(f)
        states = np.stack([np.asarray(v, dtype=np.float64) for v in df[STATE]])
        ee = fk.ee_positions(states[:, :num_arm_joints])[:, coord_indices].astype(np.float32)
        widths = {np.asarray(v).shape[-1] for v in df[ENVS]}
        if widths != {old_dim}:
            raise SystemExit(f"  {f}: env_state widths {widths} != info.json dim {old_dim}")
        df[ENVS] = [np.concatenate([np.asarray(v, dtype=np.float32), e]) for v, e in zip(df[ENVS], ee)]
        for ep, grp in pd.Series(list(ee), index=df["episode_index"]).groupby(level=0):
            per_episode.setdefault(int(ep), []).append(np.stack(grp.to_list()))
        all_vals.append(ee)
        print(f"  {os.path.relpath(f, root)}: {len(df)} frames -> env_state [{old_dim + len(ee_names)}]")
        if not args.dry_run:
            df.to_parquet(f + ".tmp", index=False)
            os.replace(f + ".tmp", f)
    # Empty shell datasets (0 episodes, schema only) have no data/stats to
    # extend — just widen the schema below so future appends match.
    ep_vals = {ep: np.concatenate(chunks) for ep, chunks in per_episode.items()}
    flat = np.concatenate(all_vals) if all_vals else np.empty((0, len(coord_indices)), dtype=np.float32)

    # ── 2. per-episode stats in meta/episodes ────────────────────────────────
    def _ep_stat(vals: np.ndarray, stat: str) -> np.ndarray:
        if stat in QUANTILES:
            return np.quantile(vals, QUANTILES[stat], axis=0)
        return getattr(np, stat)(vals, axis=0)

    prefix = f"stats/{ENVS}/"
    for f in sorted(glob.glob(os.path.join(root, "meta", "episodes", "**", "*.parquet"), recursive=True)):
        df = pd.read_parquet(f)
        for stat in VECTOR_STATS:
            col = prefix + stat
            if col not in df.columns:
                continue
            df[col] = [
                np.concatenate([np.asarray(v), _ep_stat(ep_vals[int(ep)], stat)])
                for v, ep in zip(df[col], df["episode_index"])
            ]
        print(f"  {os.path.relpath(f, root)}: extended per-episode stats for {len(df)} episodes")
        if not args.dry_run:
            df.to_parquet(f + ".tmp", index=False)
            os.replace(f + ".tmp", f)

    # ── 3. aggregated meta/stats.json ────────────────────────────────────────
    stats_path = os.path.join(root, "meta", "stats.json")
    if os.path.exists(stats_path) and len(flat):
        stats = json.load(open(stats_path))
        env_stats = stats[ENVS]
        for stat in VECTOR_STATS:
            if stat in env_stats:
                env_stats[stat] = list(env_stats[stat]) + [float(x) for x in _ep_stat(flat, stat)]
        print(f"  stats.json: extended {ENVS} aggregates")
        if not args.dry_run:
            json.dump(stats, open(stats_path, "w"), indent=4)

    # ── 4. meta/info.json feature schema ─────────────────────────────────────
    feats[ENVS]["shape"] = [old_dim + len(ee_names)]
    feats[ENVS]["names"] = names + ee_names
    print(f"  info.json: {ENVS} [{old_dim}] -> [{old_dim + len(ee_names)}]")
    if not args.dry_run:
        json.dump(info, open(info_path, "w"), indent=4)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="+", help="dataset root dirs or repo_ids under $HF_LEROBOT_HOME")
    ap.add_argument(
        "--urdf",
        default=os.path.expanduser("~/code/SplatSim/splatsim/robot_definitions/urdf/planar_3joint.urdf"),
        help="robot URDF (must match what the sim loads)",
    )
    ap.add_argument(
        "--ee_link_name", default="wrist_camera_link", help="EE reference link (matches _get_ee_link_index)"
    )
    ap.add_argument(
        "--coord_indices",
        default="0,2",
        help="world axes to append, comma-separated (planar ORACLE_STATE_COORD_INDICES = 0,2 = x,z)",
    )
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    fk_cache: dict = {}
    for spec in args.datasets:
        migrate_dataset(_dataset_root(spec), fk_cache, args)
    print(f"\nDone.{' (dry run — nothing written)' if args.dry_run else ''}")
    if not args.dry_run:
        print(
            "Re-run my_scripts/compute_relative_stats.sh on each dataset to refresh its *_recomputed_stats sidecar."
        )


if __name__ == "__main__":
    main()
