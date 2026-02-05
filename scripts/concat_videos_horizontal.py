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


def concat_videos_horizontal(input_folder, output_folder=None, output_filename="combined.mp4"):
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

    # Find max frame count and target height (use max height for quality)
    max_frames = max(vi["frame_count"] for vi in video_infos)
    target_height = max(vi["height"] for vi in video_infos)

    # Calculate scaled widths to maintain aspect ratio
    scaled_widths = []
    for vi in video_infos:
        scale = target_height / vi["height"]
        scaled_widths.append(int(vi["width"] * scale))

    total_width = sum(scaled_widths)

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

        for i, (cap, vi, scaled_w) in enumerate(zip(caps, video_infos, scaled_widths, strict=True)):
            ret, frame = cap.read()

            if ret:
                # Successfully read a frame
                last_frames[i] = frame
            else:
                # Video ended, use last frame (freeze frame)
                frame = last_frames[i]

            if frame is None:
                # Fallback: create black frame if no frame available
                frame = np.zeros((vi["height"], vi["width"], 3), dtype=np.uint8)
                last_frames[i] = frame

            # Resize to target height while maintaining aspect ratio
            resized = cv2.resize(frame, (scaled_w, target_height))
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

    args = parser.parse_args()

    concat_videos_horizontal(args.input_folder, args.output_folder, args.output_name)


if __name__ == "__main__":
    main()
