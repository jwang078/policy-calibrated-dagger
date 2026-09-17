"""Snapshot loader for the paper figures.

Every figure script reads its raw inputs (episode states/actions from the
LeRobot datasets in ~/.cache/huggingface, a policy's config.json, per-episode
open-loop deltas) through these helpers. The first call copies exactly what the
figure needs into ``paper_plots/data/`` (small .npz/.json files, committed);
later calls read the snapshot, so the figures regenerate after the dataset
cache, ``outputs/`` or ``~/data`` are cleaned up.

Set ``FIGDATA_REFRESH=1`` to re-read the sources and overwrite the snapshots.
"""

import glob
import json
import os
import shutil

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CACHE = os.environ.get("LEROBOT_CACHE_DIR", os.path.expanduser("~/.cache/huggingface/lerobot/JennyWWW"))
REFRESH = os.environ.get("FIGDATA_REFRESH", "0") == "1"


def _path(name):
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, name)


def _parquet(repo, columns):
    import pandas as pd

    files = sorted(glob.glob(f"{CACHE}/{repo}/data/**/*.parquet", recursive=True))
    if not files:
        raise FileNotFoundError(f"{repo}: no parquet under {CACHE} and no snapshot in {DATA_DIR}")
    return pd.concat([pd.read_parquet(f, columns=columns) for f in files])


def episode(repo, ep, columns=("observation.state", "action")):
    """dict column -> (T, dim) array for one episode of a dataset."""
    columns = tuple(columns)
    p = _path(f"{repo}__ep{ep}.npz")
    if os.path.exists(p) and not REFRESH:
        z = np.load(p)
        if all(c in z.files for c in columns):
            return {c: z[c] for c in columns}
    df = _parquet(repo, ["episode_index", "frame_index", *columns])
    g = df[df.episode_index == ep].sort_values("frame_index")
    if len(g) == 0:
        raise KeyError(f"{repo}: no episode {ep}")
    out = {c: np.stack([np.asarray(v, dtype=float) for v in g[c]]) for c in columns}
    if os.path.exists(p):  # keep columns snapshotted earlier
        out = {**{k: v for k, v in np.load(p).items()}, **out}
    np.savez_compressed(p, **out)
    return {c: out[c] for c in columns}


def all_states(repo, narm):
    """(N, narm) array of every frame's first `narm` state dims (for the dataset PCA plane)."""
    p = _path(f"{repo}__states{narm}.npz")
    if os.path.exists(p) and not REFRESH:
        return np.load(p)["states"]
    df = _parquet(repo, ["observation.state"])
    states = np.stack([np.asarray(v, dtype=float)[:narm] for v in df["observation.state"]]).astype(np.float32)
    np.savez_compressed(p, states=states)
    return states


def json_file(name, source):
    """A small JSON read from `source` once (e.g. a checkpoint's config.json), then from the snapshot."""
    p = _path(name)
    if not os.path.exists(p) or REFRESH:
        shutil.copy(os.path.expanduser(source), p)
    return json.load(open(p))


def deltas_episode(name, source_npz, ep):
    """The `ep<ep>_*` arrays of one open-loop sigma-deltas file (tables_repro/analysis or a figure folder)."""
    p = _path(name)
    if os.path.exists(p) and not REFRESH:
        return np.load(p, allow_pickle=True)
    z = np.load(os.path.expanduser(source_npz), allow_pickle=True)
    keep = {k: z[k] for k in z.files if k.startswith(f"ep{ep}_")}
    if not keep:
        raise KeyError(f"{source_npz}: no episode {ep}")
    np.savez_compressed(p, **keep)
    return np.load(p, allow_pickle=True)
