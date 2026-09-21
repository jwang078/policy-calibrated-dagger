#!/usr/bin/env python3
"""
Script to trim the first N timesteps from every episode in a LeRobot dataset.

This creates a new dataset with the specified number of frames removed from
the beginning of each episode.

Usage:
    python scripts/trim_episode_starts.py \
        --input_repo_id JennyWWW/my_dataset \
        --output_repo_id JennyWWW/my_dataset_trimmed \
        --trim_frames 5

    # Or with local paths:
    python scripts/trim_episode_starts.py \
        --input_path /path/to/dataset \
        --output_path /path/to/output \
        --trim_frames 5
"""

import argparse
import glob
import json
import logging
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_info(dataset_path: Path) -> dict:
    """Load info.json from dataset."""
    info_path = dataset_path / "meta" / "info.json"
    with open(info_path) as f:
        return json.load(f)


def save_info(info: dict, dataset_path: Path) -> None:
    """Save info.json to dataset."""
    info_path = dataset_path / "meta" / "info.json"
    info_path.parent.mkdir(parents=True, exist_ok=True)
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)


def load_episodes_metadata(dataset_path: Path) -> pd.DataFrame:
    """Load all episode metadata parquet files."""
    episodes_dir = dataset_path / "meta" / "episodes"
    parquet_files = sorted(glob.glob(str(episodes_dir / "*/*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No episode parquet files found in {episodes_dir}")

    dfs = [pd.read_parquet(f) for f in parquet_files]
    return pd.concat(dfs, ignore_index=True)


def save_episodes_metadata(episodes_df: pd.DataFrame, dataset_path: Path) -> None:
    """Save episode metadata to parquet."""
    episodes_path = dataset_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    episodes_df.to_parquet(episodes_path, index=False)


def load_data_parquets(dataset_path: Path) -> pd.DataFrame:
    """Load all data parquet files from dataset."""
    data_dir = dataset_path / "data"
    parquet_files = sorted(glob.glob(str(data_dir / "*/*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No data parquet files found in {data_dir}")

    logger.info(f"Loading {len(parquet_files)} data parquet files...")
    dfs = [pd.read_parquet(f) for f in tqdm(parquet_files, desc="Loading parquets")]
    return pd.concat(dfs, ignore_index=True)


def save_data_parquets(df: pd.DataFrame, dataset_path: Path, chunks_size: int = 1000) -> None:
    """Save data to parquet files in chunks."""
    data_dir = dataset_path / "data"

    # For simplicity, save all data to a single chunk/file
    # Could be extended to handle chunking based on size
    output_path = data_dir / "chunk-000" / "file-000.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving {len(df)} frames to {output_path}")
    df.to_parquet(output_path, index=False)


def load_stats(dataset_path: Path) -> dict | None:
    """Load stats.json if it exists."""
    stats_path = dataset_path / "meta" / "stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            return json.load(f)
    return None


def save_stats(stats: dict, dataset_path: Path) -> None:
    """Save stats.json."""
    stats_path = dataset_path / "meta" / "stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)


def copy_tasks(input_path: Path, output_path: Path) -> None:
    """Copy tasks.parquet if it exists."""
    tasks_src = input_path / "meta" / "tasks.parquet"
    if tasks_src.exists():
        tasks_dst = output_path / "meta" / "tasks.parquet"
        tasks_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(tasks_src, tasks_dst)


def trim_dataset(
    input_path: Path,
    output_path: Path,
    trim_frames: int,
    min_episode_length: int = 10,
) -> None:
    """
    Trim the first N frames from every episode in a dataset.

    Args:
        input_path: Path to input dataset
        output_path: Path to output dataset
        trim_frames: Number of frames to remove from the start of each episode
        min_episode_length: Minimum episode length after trimming (skip shorter episodes)
    """
    logger.info(f"Loading dataset from {input_path}")
    logger.info(f"Will trim first {trim_frames} frames from each episode")

    # Load dataset components
    info = load_info(input_path)
    episodes_meta = load_episodes_metadata(input_path)
    data_df = load_data_parquets(input_path)
    stats = load_stats(input_path)

    logger.info(f"Original dataset: {info['total_episodes']} episodes, {info['total_frames']} frames")

    # Sort data by episode_index and frame_index for proper ordering
    if "frame_index" in data_df.columns:
        data_df = data_df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    else:
        data_df = data_df.sort_values(["episode_index", "index"]).reset_index(drop=True)

    # Process each episode
    trimmed_frames = []
    new_episodes = []
    global_frame_idx = 0
    skipped_episodes = 0

    unique_episodes = sorted(data_df["episode_index"].unique())

    for ep_idx in tqdm(unique_episodes, desc="Trimming episodes"):
        # Get frames for this episode
        ep_mask = data_df["episode_index"] == ep_idx
        ep_df = data_df[ep_mask].copy()

        original_length = len(ep_df)

        # Skip if episode is too short after trimming
        if original_length <= trim_frames:
            logger.warning(f"Episode {ep_idx} has only {original_length} frames, skipping entirely")
            skipped_episodes += 1
            continue

        new_length = original_length - trim_frames
        if new_length < min_episode_length:
            logger.warning(f"Episode {ep_idx} would have only {new_length} frames after trimming, skipping")
            skipped_episodes += 1
            continue

        # Trim first N frames
        ep_df_trimmed = ep_df.iloc[trim_frames:].copy()

        # Update indices
        ep_df_trimmed["index"] = range(global_frame_idx, global_frame_idx + len(ep_df_trimmed))

        # Update frame_index to start from 0
        if "frame_index" in ep_df_trimmed.columns:
            ep_df_trimmed["frame_index"] = range(len(ep_df_trimmed))

        # Update timestamp to start from 0
        if "timestamp" in ep_df_trimmed.columns:
            timestamps = ep_df_trimmed["timestamp"].values
            ep_df_trimmed["timestamp"] = timestamps - timestamps[0]

        trimmed_frames.append(ep_df_trimmed)

        # Build new episode metadata
        new_ep_meta = {
            "episode_index": ep_idx,
            "length": len(ep_df_trimmed),
            "dataset_from_index": global_frame_idx,
            "dataset_to_index": global_frame_idx + len(ep_df_trimmed),
            "data/chunk_index": 0,
            "data/file_index": 0,
        }

        # Copy over other metadata fields from original episode
        orig_ep_row = episodes_meta[episodes_meta["episode_index"] == ep_idx]
        if len(orig_ep_row) > 0:
            orig_ep = orig_ep_row.iloc[0]
            for col in orig_ep.index:
                if col not in new_ep_meta and not col.startswith("dataset_") and col != "length":
                    new_ep_meta[col] = orig_ep[col]

        new_episodes.append(new_ep_meta)
        global_frame_idx += len(ep_df_trimmed)

    if not trimmed_frames:
        raise ValueError("No episodes remaining after trimming!")

    # Combine all trimmed frames
    new_data_df = pd.concat(trimmed_frames, ignore_index=True)

    # Create new episodes metadata DataFrame
    new_episodes_df = pd.DataFrame(new_episodes)

    # Update info
    new_info = info.copy()
    new_info["total_episodes"] = len(new_episodes)
    new_info["total_frames"] = len(new_data_df)

    logger.info(f"New dataset: {new_info['total_episodes']} episodes, {new_info['total_frames']} frames")
    logger.info(f"Skipped {skipped_episodes} episodes (too short)")

    # Save everything
    logger.info(f"Saving trimmed dataset to {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    save_info(new_info, output_path)
    save_episodes_metadata(new_episodes_df, output_path)
    save_data_parquets(new_data_df, output_path)

    # Copy tasks
    copy_tasks(input_path, output_path)

    # Copy or recalculate stats (for now, just copy)
    if stats is not None:
        save_stats(stats, output_path)
        logger.info("Copied stats (note: may need recalculation for accurate values)")

    logger.info("Done!")


def get_dataset_path(repo_id: str | None, local_path: str | None) -> Path:
    """Get dataset path from repo_id or local path."""
    if local_path:
        return Path(local_path)
    elif repo_id:
        # Default HuggingFace cache location
        from lerobot.utils.constants import HF_LEROBOT_HOME

        return HF_LEROBOT_HOME / repo_id
    else:
        raise ValueError("Either repo_id or local path must be specified")


def main():
    parser = argparse.ArgumentParser(
        description="Trim the first N frames from every episode in a LeRobot dataset"
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input_repo_id",
        type=str,
        help="Input dataset repo ID (e.g., 'JennyWWW/my_dataset')",
    )
    input_group.add_argument(
        "--input_path",
        type=str,
        help="Local path to input dataset",
    )

    # Output options
    parser.add_argument(
        "--output_repo_id",
        type=str,
        help="Output dataset repo ID (will save to HF cache)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Local path for output dataset",
    )

    # Trimming options
    parser.add_argument(
        "--trim_frames",
        type=int,
        default=5,
        help="Number of frames to trim from the start of each episode (default: 5)",
    )
    parser.add_argument(
        "--min_episode_length",
        type=int,
        default=10,
        help="Minimum episode length after trimming (shorter episodes are skipped, default: 10)",
    )

    args = parser.parse_args()

    # Validate output
    if not args.output_repo_id and not args.output_path:
        parser.error("Either --output_repo_id or --output_path must be specified")

    # Get paths
    input_path = get_dataset_path(args.input_repo_id, args.input_path)

    if args.output_path:
        output_path = Path(args.output_path)
    else:
        from lerobot.utils.constants import HF_LEROBOT_HOME

        output_path = HF_LEROBOT_HOME / args.output_repo_id

    # Verify input exists
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found at {input_path}")

    # Check output doesn't exist (safety)
    if output_path.exists():
        response = input(f"Output path {output_path} already exists. Overwrite? [y/N]: ")
        if response.lower() != "y":
            logger.info("Aborting")
            return
        shutil.rmtree(output_path)

    trim_dataset(
        input_path=input_path,
        output_path=output_path,
        trim_frames=args.trim_frames,
        min_episode_length=args.min_episode_length,
    )


if __name__ == "__main__":
    main()
