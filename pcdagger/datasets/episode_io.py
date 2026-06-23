"""Small, dependency-free helpers for reading per-episode rows out of a
LeRobotDataset's parquet shards directly (without going through
``LeRobotDataset.__getitem__``, which would decode videos etc.).

Extracted from ``visualize_shared_autonomy_DEPRECATED`` so downstream callers
don't have to depend on a deprecated module.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def find_parquet_files(dataset_dir: Path) -> list[Path]:
    """Return every ``*.parquet`` file under ``dataset_dir`` in sorted order.

    Raises ``FileNotFoundError`` if there are none (i.e. the dataset's data
    directory is empty or the wrong path was passed).
    """
    files = sorted(dataset_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {dataset_dir}")
    return files


def load_episode_frames(
    dataset_dir: Path,
    episode_index: int,
    frame_index: int,
    n_frames: int,
) -> pd.DataFrame:
    """Load ``n_frames`` consecutive rows from a specific episode starting at
    ``frame_index``.

    Reads every parquet shard under ``dataset_dir`` with a pushdown filter on
    ``episode_index``, concatenates the hits, sorts by ``frame_index``, slices
    to the requested window, and returns the result as a single DataFrame.

    Raises ``ValueError`` if the episode isn't found or doesn't have enough
    frames at/after ``frame_index``.
    """
    parquet_files = find_parquet_files(dataset_dir)
    dfs = []
    for f in parquet_files:
        df = pd.read_parquet(f, filters=[("episode_index", "==", episode_index)])
        if len(df) > 0:
            dfs.append(df)
    if not dfs:
        raise ValueError(f"Episode {episode_index} not found in {dataset_dir}")
    df = pd.concat(dfs, ignore_index=True).sort_values("frame_index").reset_index(drop=True)

    mask = df["frame_index"] >= frame_index
    df = df[mask].head(n_frames).reset_index(drop=True)
    if len(df) < n_frames:
        raise ValueError(
            f"Episode {episode_index} only has {len(df)} frames from frame_index={frame_index}, "
            f"need {n_frames}."
        )
    return df


def get_available_episodes(dataset_dir: Path, min_episode_index: int = 301) -> list[int]:
    """Return sorted list of episode indices ``>= min_episode_index`` in the dataset."""
    parquet_files = find_parquet_files(dataset_dir)
    indices: set[int] = set()
    for f in parquet_files:
        ep_col = pd.read_parquet(f, columns=["episode_index"])
        indices.update(ep_col["episode_index"].unique().tolist())
    return sorted(i for i in indices if i >= min_episode_index)


def load_task_description(dataset_dir: Path) -> dict[int, str]:
    """Load ``task_index → task_description`` mapping from dataset metadata.

    Returns a dict like ``{1: "upright_small_engine_new"}``. Falls back to an
    empty dict if ``meta/tasks.parquet`` is missing.
    """
    meta_dir = dataset_dir.parent / "meta"
    tasks_file = meta_dir / "tasks.parquet"
    if not tasks_file.exists():
        return {}
    tasks_df = pd.read_parquet(tasks_file).reset_index()
    # The parquet has __index_level_0__ (task description string) and task_index (int).
    result: dict[int, str] = {}
    for _, row in tasks_df.iterrows():
        desc = row.get("__index_level_0__") or row.get("index") or ""
        idx = int(row.get("task_index", 0))
        if desc:
            result[idx] = str(desc)
    return result
