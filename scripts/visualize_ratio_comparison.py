#!/usr/bin/env python3
"""
Compare blending-ratio datasets side-by-side for one or more episodes.

Produces an MP4 with one column per ratio (0.2 → 1.0), all synchronized
to the same timestep.  Multiple episodes are concatenated into one long video.
Text overlays show the ratio and current episode/frame.

Usage:
    python visualize_ratio_comparison.py --episode 0
    python visualize_ratio_comparison.py --episode 1-5
    python visualize_ratio_comparison.py --episode 0,3,7
    python visualize_ratio_comparison.py --episode 0 --image-key observation.images.wrist_rgb_letterbox
    python visualize_ratio_comparison.py --episode 1-5 --output my_comparison.mp4
"""

import argparse
import gc
import glob
import os
import sys

import cv2
import imageio
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Allow imports from the SplatSim scripts directory (lerobot_parquet_utils, etc.)
_SPLATSIM_SCRIPTS = os.path.expanduser("~/code/SplatSim/scripts")
if _SPLATSIM_SCRIPTS not in sys.path:
    sys.path.insert(0, _SPLATSIM_SCRIPTS)

from lerobot_parquet_utils import parse_episodes  # noqa: E402
from view_lerobot_parquet_videos import decode_image  # noqa: E402

# ---------------------------------------------------------------------------
# Dataset registry: ratio → dataset path
# ---------------------------------------------------------------------------

_BASE_DIR = os.path.expanduser("~/.cache/huggingface/lerobot/JennyWWW")

RATIO_DATASETS: list[tuple[float, str]] = [
    (0.2, os.path.join(_BASE_DIR, "splatsim_approach_lever_11_50failsrrtpi05_piabsden02")),
    (0.4, os.path.join(_BASE_DIR, "splatsim_approach_lever_11_50failsrrtpi05_piabsden04")),
    (0.6, os.path.join(_BASE_DIR, "splatsim_approach_lever_11_50failsrrtpi05_piabsden06")),
    (0.8, os.path.join(_BASE_DIR, "splatsim_approach_lever_11_50failsrrtpi05_piabsden08")),
    (1.0, os.path.join(_BASE_DIR, "splatsim_approach_lever_11_50failsrrtpi05_piabsden10")),
]

# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

_LABEL_H = 36  # pixels for the top label bar
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.65
_THICKNESS = 1
_TEXT_COLOR = (220, 220, 220)
_BAR_COLOR = (30, 30, 30)


def _make_label_bar(width: int, line1: str, line2: str) -> np.ndarray:
    """Two-line label bar in BGR."""
    bar = np.full((_LABEL_H, width, 3), _BAR_COLOR, dtype=np.uint8)
    cv2.putText(bar, line1, (6, 14), _FONT, _FONT_SCALE, _TEXT_COLOR, _THICKNESS, cv2.LINE_AA)
    cv2.putText(bar, line2, (6, 30), _FONT, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
    return bar


def _frame_column(img_rgb: np.ndarray, ratio: float, episode: int, frame_idx: int, total: int) -> np.ndarray:
    """Return a BGR column: label bar + image."""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    line1 = f"ratio={ratio:.1f}"
    line2 = f"ep {episode}  frame {frame_idx + 1}/{total}"
    bar = _make_label_bar(w, line1, line2)

    return np.concatenate([bar, img_bgr], axis=0)


def _build_strip(columns: list[np.ndarray], sep: int = 4) -> np.ndarray:
    """Horizontal strip from BGR column images, with a thin separator."""
    max_h = max(c.shape[0] for c in columns)
    padded = []
    for c in columns:
        if c.shape[0] < max_h:
            pad = np.zeros((max_h - c.shape[0], c.shape[1], 3), dtype=np.uint8)
            c = np.concatenate([c, pad], axis=0)
        padded.append(c)
        if sep > 0:
            padded.append(np.zeros((max_h, sep, 3), dtype=np.uint8))
    return np.concatenate(padded[:-1], axis=1)  # drop trailing separator


# ---------------------------------------------------------------------------
# Per-dataset loading
# ---------------------------------------------------------------------------


def _parquet_folder(dataset_root: str) -> str:
    return os.path.join(dataset_root, "data", "chunk-000")


def _load_episode_frames(dataset_root: str, episode: int, image_key: str) -> list[np.ndarray]:
    """Return decoded RGB frames for one episode, loading only the needed image column."""
    folder = _parquet_folder(dataset_root)
    parquet_files = sorted(glob.glob(os.path.join(folder, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files in {folder}")

    needed_cols = ["episode_index", "frame_index", image_key]
    chunks = []
    for f in parquet_files:
        # Check columns exist before loading (avoids reading files that don't have
        # the episode at all, and validates image_key early with a clear error).
        schema = pq.read_schema(f)
        if image_key not in schema.names:
            raise KeyError(
                f"Image key '{image_key}' not found. Available image columns: "
                f"{[c for c in schema.names if 'image' in c.lower()]}"
            )
        tbl = pq.read_table(
            f,
            columns=needed_cols,
            filters=[("episode_index", "=", episode)],
        )
        if tbl.num_rows:
            chunks.append(tbl.to_pandas())

    if not chunks:
        raise ValueError(f"Episode {episode} not found in {folder}")

    df = pd.concat(chunks, ignore_index=True).sort_values("frame_index")
    frames = [decode_image(row[image_key]) for _, row in df.iterrows()]
    del df, chunks
    return frames


# ---------------------------------------------------------------------------
# Main comparison video builder
# ---------------------------------------------------------------------------


def _render_episode(
    writer,
    episode: int,
    image_key: str,
) -> None:
    """Load one episode from all ratio datasets and append its frames to writer."""
    all_frames: list[tuple[float, list[np.ndarray]]] = []
    for ratio, dataset_root in RATIO_DATASETS:
        print(f"  Loading ratio={ratio:.1f} ...")
        frames = _load_episode_frames(dataset_root, episode, image_key)
        print(f"    → {len(frames)} frames")
        all_frames.append((ratio, frames))

    min_len = min(len(frames) for _, frames in all_frames)
    if any(len(frames) != min_len for _, frames in all_frames):
        lengths = {r: len(f) for r, f in all_frames}
        print(f"  Episode lengths differ across ratios: {lengths}  — trimming to {min_len}")

    for t in range(min_len):
        columns = [_frame_column(frames[t], ratio, episode, t, min_len) for ratio, frames in all_frames]
        strip_bgr = _build_strip(columns, sep=4)
        strip_rgb = cv2.cvtColor(strip_bgr, cv2.COLOR_BGR2RGB)
        writer.append_data(strip_rgb)

        if (t + 1) % 50 == 0:
            print(f"  Rendered {t + 1}/{min_len} frames")

    del all_frames
    gc.collect()


def build_comparison_video(
    episodes: list[int],
    output_path: str,
    image_key: str = "observation.images.base_rgb_letterbox",
    fps: int = 30,
) -> None:
    print(f"Building ratio comparison for episodes {episodes}")
    print(f"Image key: {image_key}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    writer = imageio.get_writer(output_path, fps=fps, codec="libx264", pixelformat="yuv420p")

    for episode in episodes:
        print(f"\nEpisode {episode}:")
        _render_episode(writer, episode, image_key)

    writer.close()
    print(f"\nSaved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate a side-by-side ratio comparison video from blending datasets."
    )
    parser.add_argument(
        "--episode",
        type=str,
        required=True,
        help="Episode(s) to visualize: single '0', range '1-5', or list '0,3,7'. Multiple episodes are concatenated.",
    )
    parser.add_argument(
        "--image-key",
        type=str,
        default="observation.images.base_rgb_letterbox",
        help="Dataset column to render (default: observation.images.base_rgb_letterbox).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output MP4 path. Defaults to outputs/ratio_comparison/episode_<spec>.mp4",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second (default: 30).",
    )
    args = parser.parse_args()

    episodes = parse_episodes(args.episode)
    ep_slug = args.episode.replace(",", "_").replace("-", "to")
    output_path = args.output or os.path.join("outputs", "ratio_comparison", f"episode_{ep_slug}.mp4")

    build_comparison_video(
        episodes=episodes,
        output_path=output_path,
        image_key=args.image_key,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
