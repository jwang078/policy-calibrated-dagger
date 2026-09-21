#!/usr/bin/env python3
"""Strip the trailing per-link min-obstacle-distance dims from
`observation.environment_state` in a planar-3joint LeRobotDataset.

Inverse of append_link_obstacle_dists_to_env_state.py. The link-distance
feature was retired from the planar env (2026-07-30 —
ORACLE_STATE_INCLUDE_LINK_OBSTACLE_DIST is now False in SplatSim's
PlanarPybulletRobotServer), so datasets recorded/retrofitted during the
11-wide window must be migrated back to the 8-wide layout to mix with new
recordings in one training run.

Layout change (planar_3joint with 2 obstacles):
  BEFORE (width=11):
    [block_x, block_z, obstacle_1_x, obstacle_1_z, obstacle_2_x, obstacle_2_z,
     ee_x, ee_z, link_1_min_dist, link_2_min_dist, link_3_min_dist]
  AFTER  (width=8):
    [block_x, block_z, obstacle_1_x, obstacle_1_z, obstacle_2_x, obstacle_2_z,
     ee_x, ee_z]

The link dists are a pure TRAILING suffix, so this is a truncation — no
pybullet or scene reconstruction needed. Updated in place (with a
`_pre_strip_link_dists_backup` sibling of data/ + meta/ first; videos are
untouched so they aren't copied):
  1. data/chunk-*/file-*.parquet  — env_state column truncated per frame.
  2. meta/info.json               — feature shape + names truncated.
  3. meta/stats.json              — env_state per-dim stat arrays truncated.
  4. meta/episodes/*.parquet      — per-episode stats/…/env_state arrays truncated.

After running, refresh the rel-action stats sidecars:
  bash my_scripts/compute_relative_stats.sh --dataset_repo=<repo_id>

Usage:
  python my_scripts/strip_link_obstacle_dists_from_env_state.py \
      --repo_id=JennyWWW/planar_3joint_3 --n_strip=3 [--dry_run]
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

log = logging.getLogger("strip_link_obstacle_dists")

ENV_KEY = "observation.environment_state"
# Per-dim stat arrays to truncate; "count" is a scalar and stays.
PER_DIM_STATS = ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99")


def strip_dataset(repo_id: str, n_strip: int, dry_run: bool) -> None:
    hf_home = Path.home() / ".cache" / "huggingface" / "lerobot"
    root = hf_home / repo_id
    if not root.exists():
        raise FileNotFoundError(f"Dataset not cached at {root}.")

    info_path = root / "meta" / "info.json"
    info = json.load(open(info_path))
    if ENV_KEY not in info["features"]:
        log.error(f"{repo_id} has no {ENV_KEY} feature — nothing to strip.")
        sys.exit(1)
    old_dim = int(info["features"][ENV_KEY]["shape"][0])
    new_dim = old_dim - n_strip

    log.info("=" * 72)
    log.info(f"Dataset: {repo_id}")
    log.info(f"  root: {root}")
    log.info(f"  env_state dim: {old_dim}  →  {new_dim}  (stripping trailing {n_strip})")
    if dry_run:
        log.info("  --dry_run: no files will be modified")
    log.info("=" * 72)

    if new_dim <= 0:
        log.error(f"Stripping {n_strip} from width {old_dim} leaves {new_dim} dims. Aborting.")
        sys.exit(1)

    # Backup data/ + meta/ (videos untouched → not copied).
    backup_root = root.with_name(root.name + "_pre_strip_link_dists_backup")
    if not dry_run:
        if backup_root.exists():
            log.warning(f"Backup dir already exists at {backup_root}; leaving as-is (prior aborted run?).")
        else:
            log.info(f"Creating backup of data/ + meta/: {backup_root}")
            backup_root.mkdir()
            shutil.copytree(root / "data", backup_root / "data")
            shutil.copytree(root / "meta", backup_root / "meta")

    # 1. Data parquets: truncate the env_state column per frame.
    data_files = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No data parquet files under {root / 'data'}")
    n_frames = 0
    for data_path in data_files:
        df = pd.read_parquet(data_path)
        widths = {len(v) for v in df[ENV_KEY]}
        if widths != {old_dim}:
            raise RuntimeError(
                f"{data_path}: env_state widths {sorted(widths)} != expected {{{old_dim}}}. "
                f"Mixed-width dataset — refusing to truncate blindly."
            )
        df[ENV_KEY] = df[ENV_KEY].map(lambda v: np.asarray(v, dtype=np.float32)[:new_dim])
        n_frames += len(df)
        if not dry_run:
            df.to_parquet(data_path, index=False)
        log.info(f"  data: {data_path.relative_to(root)}  ({len(df)} frames)")

    # 2. meta/info.json: shape + names.
    info["features"][ENV_KEY]["shape"] = [new_dim]
    names = info["features"][ENV_KEY].get("names")
    if isinstance(names, list) and len(names) == old_dim:
        info["features"][ENV_KEY]["names"] = names[:new_dim]
    if not dry_run:
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)
    log.info(f"  meta/info.json: shape → [{new_dim}]")

    # 3. meta/stats.json: per-dim arrays. Truncation is exact — per-dim stats
    # are independent, so dropping trailing dims can't invalidate the rest.
    stats_path = root / "meta" / "stats.json"
    if stats_path.exists():
        stats = json.load(open(stats_path))
        env_stats = stats.get(ENV_KEY)
        if env_stats:
            for k in PER_DIM_STATS:
                v = env_stats.get(k)
                if isinstance(v, list) and len(v) == old_dim:
                    env_stats[k] = v[:new_dim]
            if not dry_run:
                with open(stats_path, "w") as f:
                    json.dump(stats, f, indent=2)
            log.info("  meta/stats.json: env_state stat arrays truncated")

    # 4. Per-episode stats columns in meta/episodes parquets.
    ep_files = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    for ep_path in ep_files:
        ep_df = pd.read_parquet(ep_path)
        changed = False
        for k in PER_DIM_STATS:
            col = f"stats/{ENV_KEY}/{k}"
            if col not in ep_df.columns:
                continue
            ep_df[col] = ep_df[col].map(
                lambda v: np.asarray(v).reshape(-1)[:new_dim] if np.asarray(v).size == old_dim else v
            )
            changed = True
        if changed and not dry_run:
            ep_df.to_parquet(ep_path, index=False)
        log.info(f"  meta/episodes: {ep_path.relative_to(root)}  ({len(ep_df)} episodes)")

    log.info("")
    log.info(f"Done: {n_frames} frames truncated {old_dim} → {new_dim}.")
    log.info("Next steps:")
    log.info(f"  1. bash my_scripts/compute_relative_stats.sh --dataset_repo={repo_id}")
    log.info("     (refreshes stats_rel*.json sidecars, which embed env_state stats)")
    log.info(f"  2. Backup preserved at: {backup_root}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--repo_id", required=True)
    ap.add_argument(
        "--n_strip", type=int, default=3, help="Trailing dims to remove (planar 3-joint link dists: 3)"
    )
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    strip_dataset(repo_id=args.repo_id, n_strip=args.n_strip, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
