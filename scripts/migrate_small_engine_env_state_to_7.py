#!/usr/bin/env python
"""Migrate small_engine datasets from the 15-wide env_state era to the 7-wide layout.

Background: `observation.environment_state` for the small_engine envs was
redesigned (2026-08-04, SplatSim `sim_robot_pybullet_small_engine.py`):

  OLD (15): [engine(x,y,z), table(x,y,z), wall(x,y,z), box1(x,y,z), box2(x,y,z)]
  NEW  (7): [box1(x,y), box2(x,y), ee(x,y,z)]

Engine/table/wall are pinned per scenario (randomize_pose=False) — 9 constant
dims; the boxes sit ON the table so their z is constant too. The EE position is
a pure function of the recorded observation.state joints, so this script
reconstructs it exactly (pybullet FK on the same URDF + base position the sim
uses, URDF link frame of `wrist_camera_link` — matching the server's
`get_current_ee_pose`). No re-recording needed.

Safety rails baked in:
  * asserts the assumed layout empirically — dims 0-8 (engine/table/wall) and
    11, 14 (box z) must be CONSTANT across every frame of the dataset before
    anything is written;
  * FK sanity print — distance of each episode's final-frame EE from
    FK(q_goal_bias), the canonical goal config (successful demos should be
    near zero);
  * full data/ + meta/ backup to `_pre_es7_backup/` (videos untouched);
  * idempotent — datasets already at width 7 are skipped.

Touches (same contract as append_ee_to_env_state.py / strip_link_..._.py):
  1. data/**/*.parquet          — env_state rebuilt per frame
  2. meta/episodes/**/*.parquet — per-episode env_state stats rebuilt
  3. meta/stats.json            — aggregated env_state stats rebuilt
  4. meta/info.json             — feature shape [15]->[7] + explicit names

Afterwards refresh rel-action stats sidecars on TRAINING datasets:
  bash my_scripts/compute_relative_stats.sh --dataset_repo=<repo_id>
(the eval benchmark needs no sidecar — eval-benchmark-loss uses the policy's
own stats).

Usage:
  python3 my_scripts/migrate_small_engine_env_state_to_7.py \
      JennyWWW/splatsim_approach_lever_13_smooth \
      JennyWWW/eval_splatsim_approach_lever_13_benchmark [--dry_run]
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

OLD_DIM = 15
# Old layout: object index * 3 + axis. Boxes are objects 3 (box1) and 4 (box2).
BOX_XY_SLICE = [9, 10, 12, 13]  # box1_x, box1_y, box2_x, box2_y
# Dims that MUST be constant across the whole dataset for the layout
# assumption to hold: engine/table/wall xyz (0-8) + box z's (11, 14).
CONST_DIMS = list(range(9)) + [11, 14]
NEW_NAMES = ["box1_x", "box1_y", "box2_x", "box2_y", "ee_x", "ee_y", "ee_z"]
NEW_DIM = len(NEW_NAMES)

VECTOR_STATS = ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99")
QUANTILES = {"q01": 0.01, "q10": 0.10, "q50": 0.50, "q90": 0.90, "q99": 0.99}

# Sim-side constants (UR5 small_engine). URDF + base_position MUST match what
# the sim loads (SplatObjectConfig 'robot_iphone_w_engine_curtain'); q_goal_bias
# is the env's canonical goal config, used only for the FK sanity print.
DEFAULT_URDF = os.path.expanduser("~/code/SplatSim/splatsim/robot_definitions/urdf/sisbot.urdf")
DEFAULT_BASE_POSITION = (0.0, 0.0, -0.088)
DEFAULT_EE_LINK = "wrist_camera_link"
NUM_ARM_JOINTS = 6
Q_GOAL_BIAS = (1.223, -1.587, 2.082, -0.925, -0.496, -1.124)


def _dataset_root(spec: str) -> str:
    if os.path.isdir(spec):
        return os.path.abspath(spec)
    home = os.environ.get("HF_LEROBOT_HOME") or os.path.expanduser("~/.cache/huggingface/lerobot")
    root = os.path.join(home, spec)
    if not os.path.isdir(root):
        raise SystemExit(f"Dataset not found: neither a directory nor under {home}: {spec}")
    return root


class UR5FK:
    """FK via pybullet DIRECT on the sim's own URDF + base position.

    Arm joints are the FIRST `NUM_ARM_JOINTS` movable joints in URDF order
    (matches the server's get_joint_state / teleport convention). EE position
    is the URDF LINK FRAME (getLinkState index 4), matching the server's
    `get_current_ee_pose` — not the link COM (index 0); identical for
    wrist_camera_link but we mirror the sim exactly on principle.
    """

    def __init__(self, urdf_path: str, base_position, ee_link_name: str):
        import pybullet as p
        import pybullet_utils.bullet_client as bc

        self._p = bc.BulletClient(connection_mode=p.DIRECT)
        self._rid = self._p.loadURDF(urdf_path, list(base_position), useFixedBase=True)
        movable, self._ee_link = [], None
        for j in range(self._p.getNumJoints(self._rid)):
            info = self._p.getJointInfo(self._rid, j)
            if info[2] != p.JOINT_FIXED:
                movable.append(j)
            if info[12].decode("utf-8") == ee_link_name:
                self._ee_link = j
        if self._ee_link is None:
            raise SystemExit(f"EE link '{ee_link_name}' not found in {urdf_path}")
        if len(movable) < NUM_ARM_JOINTS:
            raise SystemExit(f"URDF has {len(movable)} movable joints < {NUM_ARM_JOINTS}")
        self._arm_joints = movable[:NUM_ARM_JOINTS]

    def ee_positions(self, joint_states: np.ndarray) -> np.ndarray:
        """[N, >=6] joint angles -> [N, 3] world EE positions (link frame)."""
        out = np.empty((len(joint_states), 3), dtype=np.float64)
        for n, q in enumerate(joint_states):
            for j, qj in zip(self._arm_joints, q[:NUM_ARM_JOINTS]):
                self._p.resetJointState(self._rid, j, float(qj))
            out[n] = self._p.getLinkState(self._rid, self._ee_link, computeForwardKinematics=True)[4]
        return out


def _stat(vals: np.ndarray, stat: str) -> np.ndarray:
    if stat in QUANTILES:
        return np.quantile(vals, QUANTILES[stat], axis=0)
    return getattr(np, stat)(vals, axis=0)


def migrate_dataset(root: str, fk: UR5FK, goal_ee: np.ndarray, args) -> None:
    print(f"\n=== {root}")
    info_path = os.path.join(root, "meta", "info.json")
    info = json.load(open(info_path))
    feats = info["features"]
    if ENVS not in feats:
        raise SystemExit(f"{root}: no {ENVS} feature — not an oracle dataset")
    cur_dim = int(feats[ENVS]["shape"][0])
    if cur_dim == NEW_DIM and list(feats[ENVS].get("names") or []) == NEW_NAMES:
        print(f"  already migrated ({NEW_DIM}-wide) — skipping")
        return
    if cur_dim != OLD_DIM:
        raise SystemExit(f"  env_state width {cur_dim} != expected {OLD_DIM} — refusing to guess layout")

    data_files = sorted(glob.glob(os.path.join(root, "data", "**", "*.parquet"), recursive=True))

    # ── Pass 0: validate the layout assumption on EVERY frame ────────────────
    ref = None
    n_frames = 0
    for f in data_files:
        env = np.stack([np.asarray(v, dtype=np.float64) for v in pd.read_parquet(f, columns=[ENVS])[ENVS]])
        if env.shape[1] != OLD_DIM:
            raise SystemExit(f"  {f}: env_state width {env.shape[1]} != {OLD_DIM}")
        if ref is None:
            ref = env[0, CONST_DIMS]
        dev = np.abs(env[:, CONST_DIMS] - ref).max()
        if dev > 1e-5:
            raise SystemExit(
                f"  {f}: supposedly-constant dims {CONST_DIMS} vary by {dev:.2e} — "
                f"layout assumption violated, NOT migrating"
            )
        n_frames += len(env)
    print(
        f"  layout validated: dims {CONST_DIMS} constant across {n_frames} frames "
        f"(engine/table/wall pinned, box z = table height)"
    )

    # ── Backup data/ + meta/ (videos untouched) ──────────────────────────────
    bak = os.path.join(root, "_pre_es7_backup")
    if not args.dry_run:
        if os.path.exists(bak):
            print(f"  backup already exists (prior run): {bak}")
        else:
            os.makedirs(bak)
            shutil.copytree(os.path.join(root, "data"), os.path.join(bak, "data"))
            shutil.copytree(os.path.join(root, "meta"), os.path.join(bak, "meta"))
            print(f"  backed up data/ + meta/ -> {bak}")

    # ── Pass 1: rewrite data parquets ────────────────────────────────────────
    per_episode: dict[int, list[np.ndarray]] = {}
    all_vals: list[np.ndarray] = []
    ep_final: dict[int, np.ndarray] = {}  # last-frame EE per episode (FK sanity)
    for f in data_files:
        df = pd.read_parquet(f)
        env = np.stack([np.asarray(v, dtype=np.float64) for v in df[ENVS]])
        states = np.stack([np.asarray(v, dtype=np.float64) for v in df[STATE]])
        ee = fk.ee_positions(states)
        new = np.concatenate([env[:, BOX_XY_SLICE], ee], axis=1).astype(np.float32)
        df[ENVS] = list(new)
        for ep, grp in pd.DataFrame({"i": range(len(df))}, index=df["episode_index"]).groupby(level=0):
            idx = grp["i"].to_numpy()
            per_episode.setdefault(int(ep), []).append(new[idx])
            ep_final[int(ep)] = ee[idx[-1]]
        all_vals.append(new)
        print(f"  {os.path.relpath(f, root)}: {len(df)} frames -> env_state [{NEW_DIM}]")
        if not args.dry_run:
            df.to_parquet(f + ".tmp", index=False)
            os.replace(f + ".tmp", f)

    # FK sanity: successful demos should end near FK(q_goal_bias).
    if ep_final:
        d = np.array([np.linalg.norm(v - goal_ee) for v in ep_final.values()])
        print(
            f"  FK sanity — final-frame EE vs FK(q_goal_bias): "
            f"median {np.median(d) * 1000:.1f} mm, p90 {np.quantile(d, 0.9) * 1000:.1f} mm "
            f"(over {len(d)} episodes; near-zero medians ⇒ FK convention matches the sim)"
        )

    ep_vals = {ep: np.concatenate(chunks) for ep, chunks in per_episode.items()}
    flat = np.concatenate(all_vals) if all_vals else np.empty((0, NEW_DIM), dtype=np.float32)

    # ── Pass 2: rebuild per-episode env_state stats ──────────────────────────
    prefix = f"stats/{ENVS}/"
    for f in sorted(glob.glob(os.path.join(root, "meta", "episodes", "**", "*.parquet"), recursive=True)):
        df = pd.read_parquet(f)
        for stat in VECTOR_STATS:
            col = prefix + stat
            if col not in df.columns:
                continue
            df[col] = [_stat(ep_vals[int(ep)], stat) for ep in df["episode_index"]]
        print(f"  {os.path.relpath(f, root)}: rebuilt per-episode stats for {len(df)} episodes")
        if not args.dry_run:
            df.to_parquet(f + ".tmp", index=False)
            os.replace(f + ".tmp", f)

    # ── Pass 3: rebuild aggregated meta/stats.json ───────────────────────────
    stats_path = os.path.join(root, "meta", "stats.json")
    if os.path.exists(stats_path) and len(flat):
        stats = json.load(open(stats_path))
        env_stats = stats.get(ENVS, {})
        for stat in VECTOR_STATS:
            if stat in env_stats:
                env_stats[stat] = [float(x) for x in _stat(flat, stat)]
        stats[ENVS] = env_stats
        print(f"  stats.json: rebuilt {ENVS} aggregates")
        if not args.dry_run:
            json.dump(stats, open(stats_path, "w"), indent=4)

    # ── Pass 4: meta/info.json schema ────────────────────────────────────────
    feats[ENVS]["shape"] = [NEW_DIM]
    feats[ENVS]["names"] = NEW_NAMES
    print(f"  info.json: {ENVS} [{OLD_DIM}] -> [{NEW_DIM}] names={NEW_NAMES}")
    if not args.dry_run:
        json.dump(info, open(info_path, "w"), indent=4)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="+", help="dataset root dirs or repo_ids under $HF_LEROBOT_HOME")
    ap.add_argument("--urdf", default=DEFAULT_URDF)
    ap.add_argument("--ee_link_name", default=DEFAULT_EE_LINK)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    fk = UR5FK(args.urdf, DEFAULT_BASE_POSITION, args.ee_link_name)
    goal_ee = fk.ee_positions(np.array([Q_GOAL_BIAS]))[0]
    print(f"FK(q_goal_bias) = {np.round(goal_ee, 4).tolist()}  (canonical goal EE for sanity checks)")

    for spec in args.datasets:
        migrate_dataset(_dataset_root(spec), fk, goal_ee, args)

    print("\nDone. For TRAINING datasets now refresh the rel-stats sidecars:")
    print("  bash my_scripts/compute_relative_stats.sh --dataset_repo=<repo_id>")


if __name__ == "__main__":
    main()
