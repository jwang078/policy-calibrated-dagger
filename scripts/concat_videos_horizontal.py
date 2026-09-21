#!/usr/bin/env python3
"""
Concatenates all videos in a folder horizontally (in one row) and saves as combined.mp4.
Videos of different lengths are padded with their last frame until the longest video finishes.
"""

import argparse
import glob
import os
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np


def get_video_info(video_path):
    """Get video properties: width, height, fps, frame_count."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    return width, height, fps, frame_count


def concat_videos_horizontal(
    input_folder, output_folder=None, output_filename="combined.mp4", max_videos=None
):
    """
    Concatenate all videos in input_folder horizontally.

    Args:
        input_folder: Path to folder containing video files
        output_folder: Path to output folder (default: input_folder/summary)
        output_filename: Name of output file (default: combined.mp4)
    """
    input_folder = Path(input_folder)

    # Find all video files
    video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(str(input_folder / ext)))

    # Sort for consistent ordering
    video_files = sorted(video_files)
    if max_videos is not None:
        video_files = video_files[:max_videos]

    if not video_files:
        raise ValueError(f"No video files found in {input_folder}")

    print(f"Found {len(video_files)} videos:")
    for vf in video_files:
        print(f"  - {os.path.basename(vf)}")

    # Get info for all videos
    video_infos = []
    for vf in video_files:
        w, h, fps, fc = get_video_info(vf)
        video_infos.append({"path": vf, "width": w, "height": h, "fps": fps, "frame_count": fc})
        print(f"  {os.path.basename(vf)}: {w}x{h}, {fps} fps, {fc} frames")

    # Use the fps from the first video (assume all are same)
    output_fps = video_infos[0]["fps"]

    # Each video is resized to target_width, preserving aspect ratio
    target_width = 224
    max_frames = max(vi["frame_count"] for vi in video_infos)
    scaled_heights = [int(vi["height"] * target_width / vi["width"]) for vi in video_infos]
    target_height = max(scaled_heights)  # pad shorter videos vertically if aspect ratios differ

    total_width = target_width * len(video_infos)

    print(f"\nOutput: {total_width}x{target_height}, {output_fps} fps, {max_frames} frames")

    # Setup output
    output_folder = input_folder / "summary" if output_folder is None else Path(output_folder)

    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = output_folder / output_filename

    # Open all video captures
    caps = [cv2.VideoCapture(vi["path"]) for vi in video_infos]

    # Store last frames for padding
    last_frames = [None] * len(caps)

    # Collect all frames first
    print("\nProcessing frames...")
    all_frames = []

    for frame_idx in range(max_frames):
        if frame_idx % 100 == 0:
            print(f"  Frame {frame_idx}/{max_frames}")

        frames_row = []

        for i, (cap, vi, scaled_h) in enumerate(zip(caps, video_infos, scaled_heights, strict=True)):
            ret, frame = cap.read()

            if ret:
                last_frames[i] = frame
            else:
                frame = last_frames[i]

            if frame is None:
                frame = np.zeros((vi["height"], vi["width"], 3), dtype=np.uint8)
                last_frames[i] = frame

            # Resize to target_width x scaled_h (preserves aspect ratio)
            resized = cv2.resize(frame, (target_width, scaled_h))

            # Pad vertically with black if this video is shorter than target_height
            if scaled_h < target_height:
                pad = np.zeros((target_height - scaled_h, target_width, 3), dtype=np.uint8)
                resized = np.vstack([resized, pad])

            frames_row.append(resized)

        # Concatenate horizontally
        combined_frame = np.hstack(frames_row)
        # Convert BGR to RGB for imageio
        combined_frame_rgb = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)
        all_frames.append(combined_frame_rgb)

    # Cleanup captures
    for cap in caps:
        cap.release()

    # Write with imageio using H.264 codec for compatibility
    print("\nWriting H.264 encoded video...")
    iio.imwrite(str(output_path), all_frames, fps=output_fps, codec="libx264", plugin="pyav")

    print(f"\nSaved combined video to: {output_path}")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Concatenate videos horizontally with frame padding for different lengths"
    )
    parser.add_argument("input_folder", help="Path to folder containing video files")
    parser.add_argument(
        "-o", "--output-folder", default=None, help="Output folder (default: input_folder/summary)"
    )
    parser.add_argument(
        "-n", "--output-name", default="combined.mp4", help="Output filename (default: combined.mp4)"
    )
    parser.add_argument(
        "-m",
        "--max-videos",
        type=int,
        default=None,
        help="Maximum number of videos to include (default: all)",
    )

    args = parser.parse_args()

    # Default output name to the input folder's grandparent (the training run folder)
    if args.output_name == "combined.mp4" and args.output_folder is None:
        output_name = Path(args.input_folder).parent.parent.name + ".mp4"
    elif args.output_name == "combined.mp4":
        output_name = Path(args.output_folder).name + ".mp4"
    else:
        output_name = args.output_name

    concat_videos_horizontal(args.input_folder, args.output_folder, output_name, args.max_videos)


if __name__ == "__main__":
    main()
