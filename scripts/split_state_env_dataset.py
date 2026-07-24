#!/usr/bin/env python
"""Migrate a LeRobot v3 dataset from the OLD oracle layout (object coords packed
into observation.state) to the SPLIT layout the diffusion policy expects:

    observation.state             = [joints, gripper]        (first `state_dim` dims)
    observation.environment_state = [object coords]          (remaining dims)

Background: older SplatSim oracle recordings concatenated privileged object
coords onto observation.state. The diffusion policy requires an image OR a
distinct observation.environment_state feature (FeatureType.ENV) and normalizes
the two independently, so the coords must live in their OWN feature. The recording
pipeline now emits the split directly (build_lerobot_frame /
build_lerobot_features); this script retrofits datasets recorded before that.

It edits IN PLACE (parquet-level, videos/images untouched — fast + lossless) and
first backs up meta/ to meta_backup_pre_split/. Idempotent: a dataset that already
has observation.environment_state is left alone.

Touches four places:
  1. data/**/*.parquet                — split the observation.state column
  2. meta/episodes/**/*.parquet        — split per-episode stats/observation.state/*
  3. meta/stats.json                   — split the aggregated observation.state stats
  4. meta/info.json                    — resize state feature + add env feature

Usage:
  python my_scripts/split_state_env_dataset.py \
      --repo_id=JennyWWW/planar_3joint_oracle_simple \
      --state_dim=4 --env_state_dim=2 [--dry_run]

state_dim + env_state_dim MUST equal the old observation.state width.
"""

import argparse
import glob
import json
import os
import shutil

import numpy as np
import pandas as pd

STATE = "observation.state"
ENVS = "observation.environment_state"


def _dataset_root(repo_id: str) -> str:
    home = os.environ.get("HF_LEROBOT_HOME") or os.path.expanduser("~/.cache/huggingface/lerobot")
    root = os.path.join(home, repo_id)
    if not os.path.isdir(root):
        raise SystemExit(f"Dataset not found on disk: {root}")
    return root


def _split_vec(v, sd):
    """Split a per-dim stat/data vector into (state[:sd], env[sd:]).
    Non-per-dim arrays (e.g. the scalar `count`) are returned unchanged for both."""
    a = np.asarray(v)
    if a.ndim >= 1 and a.shape[-1] > sd:
        return a[..., :sd], a[..., sd:]
    return a, a  # count / scalar stats: identical for both features


def migrate_info(root, sd, ed, dry):
    p = os.path.join(root, "meta", "info.json")
    info = json.load(open(p))
    feats = info["features"]
    if ENVS in feats:
        print("  info.json already has environment_state — skipping")
        return False
    old = feats[STATE]
    old_dim = int(old["shape"][0])
    if old_dim != sd + ed:
        raise SystemExit(f"observation.state width {old_dim} != state_dim({sd}) + env_state_dim({ed})")
    names = old.get("names") or [f"s{i}" for i in range(old_dim)]
    # Rebuild features dict preserving order, inserting env right after state.
    new_feats = {}
    for k, v in feats.items():
        if k == STATE:
            new_feats[STATE] = {**v, "shape": [sd], "names": list(names[:sd])}
            new_feats[ENVS] = {
                "dtype": "float32",
                "shape": [ed],
                "names": [f"env_{i}" for i in range(ed)],
            }
        else:
            new_feats[k] = v
    info["features"] = new_feats
    print(f"  info.json: {STATE} [{old_dim}] -> [{sd}] + {ENVS} [{ed}]")
    if not dry:
        json.dump(info, open(p, "w"), indent=4)
    return True


def migrate_stats_json(root, sd, dry):
    p = os.path.join(root, "meta", "stats.json")
    st = json.load(open(p))
    if ENVS in st:
        print("  stats.json already split — skipping")
        return
    s = st[STATE]
    env = {}
    new_s = {}
    for stat, val in s.items():
        a, b = _split_vec(val, sd)
        new_s[stat] = a.tolist()
        env[stat] = b.tolist()
    st[STATE] = new_s
    st[ENVS] = env
    print("  stats.json: split observation.state -> state + environment_state")
    if not dry:
        json.dump(st, open(p, "w"), indent=4)


def migrate_episode_stats(root, sd, dry):
    files = sorted(glob.glob(os.path.join(root, "meta", "episodes", "**", "*.parquet"), recursive=True))
    prefix = f"stats/{STATE}/"
    env_prefix = f"stats/{ENVS}/"
    for f in files:
        df = pd.read_parquet(f)
        if any(c.startswith(env_prefix) for c in df.columns):
            print(f"  {os.path.relpath(f, root)}: already split — skipping")
            continue
        state_cols = [c for c in df.columns if c.startswith(prefix)]
        # Build new columns; insert env stats right after the matching state col.
        new_cols_order = []
        env_data = {}
        for c in df.columns:
            new_cols_order.append(c)
            if c.startswith(prefix):
                stat = c[len(prefix) :]
                env_col = env_prefix + stat
                sa, ea = zip(*(_split_vec(v, sd) for v in df[c]))
                df[c] = list(sa)
                env_data[env_col] = list(ea)
                new_cols_order.append(env_col)
        for k, v in env_data.items():
            df[k] = v
        df = df[new_cols_order]
        print(f"  {os.path.relpath(f, root)}: split {len(state_cols)} stat cols")
        if not dry:
            df.to_parquet(f + ".tmp", index=False)
            os.replace(f + ".tmp", f)


def migrate_data(root, sd, dry):
    files = sorted(glob.glob(os.path.join(root, "data", "**", "*.parquet"), recursive=True))
    for f in files:
        df = pd.read_parquet(f)
        if ENVS in df.columns:
            print(f"  {os.path.relpath(f, root)}: already split — skipping")
            continue
        cols = []
        env_col = []
        for v in df[STATE]:
            a = np.asarray(v, dtype=np.float32)
            env_col.append(a[sd:])
        # truncate state, insert env right after
        df[STATE] = [np.asarray(v, dtype=np.float32)[:sd] for v in df[STATE]]
        for c in df.columns:
            cols.append(c)
            if c == STATE:
                cols.append(ENVS)
        df[ENVS] = env_col
        df = df[cols]
        print(f"  {os.path.relpath(f, root)}: {len(df)} frames split")
        if not dry:
            df.to_parquet(f + ".tmp", index=False)
            os.replace(f + ".tmp", f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", default="JennyWWW/planar_3joint_oracle_simple")
    ap.add_argument("--state_dim", type=int, required=True, help="new observation.state width (num_dofs + 1)")
    ap.add_argument(
        "--env_state_dim", type=int, required=True, help="observation.environment_state width (object coords)"
    )
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    root = _dataset_root(args.repo_id)
    sd, ed = args.state_dim, args.env_state_dim
    print(f"Dataset: {root}")
    print(f"Split: observation.state [{sd + ed}] -> state [{sd}] + environment_state [{ed}]")
    print(f"{'DRY RUN — no writes' if args.dry_run else 'WRITING IN PLACE'}\n")

    # Back up meta/ (small; the risky bit) before touching anything.
    meta_bak = os.path.join(root, "meta_backup_pre_split")
    if not args.dry_run:
        if os.path.exists(meta_bak):
            print(f"Backup already exists (prior run): {meta_bak}")
        else:
            shutil.copytree(os.path.join(root, "meta"), meta_bak)
            print(f"Backed up meta/ -> {meta_bak}\n")

    print("[1/4] info.json")
    changed = migrate_info(root, sd, ed, args.dry_run)
    print("[2/4] stats.json")
    migrate_stats_json(root, sd, args.dry_run)
    print("[3/4] meta/episodes/*.parquet")
    migrate_episode_stats(root, sd, args.dry_run)
    print("[4/4] data/*.parquet")
    migrate_data(root, sd, args.dry_run)

    print("\nDone." + ("" if changed else " (dataset was already split)"))
    if not args.dry_run:
        print("Re-run compute_relative_stats.sh to refresh the relative-action sidecar for the new schema.")


if __name__ == "__main__":
    main()
