#!/usr/bin/env python3
"""Compare N folders of eval videos as a vertical strip (N rows).

For each group of mp4 files sharing the same basename across all folders:
  * play the matched videos in parallel, top→bottom in folder-arg order
  * when any video ends, freeze its last frame until the longest finishes;
    a small "DONE" badge appears on the frozen panel
  * after they all finish, emit a black gap (default ~1 second at fps)
  * then move on to the next matched group

Labels (training-run directory basename, auto-derived from each folder path)
are overlaid top-left of each video. Override with --labels NAME1 NAME2 ...

Files NOT present in EVERY folder are skipped with a warning. Groups are
processed in natural sort order (eval_episode_0, eval_episode_1, ..., 10).

Examples:
    # 2 folders
    python my_scripts/compare_eval_videos.py \\
        outputs/training/<runA>/eval/.../splatsim_0 \\
        outputs/training/<runB>/eval/.../splatsim_0

    # 3+ folders
    python my_scripts/compare_eval_videos.py \\
        outputs/training/<runA>/.../splatsim_0 \\
        outputs/training/<runB>/.../splatsim_0 \\
        outputs/training/<runC>/.../splatsim_0
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import imageio
import numpy as np


def natural_sort_key(name: str) -> list:
    """Sort eval_episode_0, _1, _2, ..., _10 in numeric order."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", name)]


def derive_label_from_folder(folder: Path) -> str:
    """Walk up the folder path until we find the directory whose parent is
    ``training`` — that's the training-run dir and its name is the label.

    Handles all the layouts we use:
      .../<run>/eval/videos_step_<N>/splatsim_0          (inline lerobot-train)
      .../<run>/eval/eval_benchmark_<N>/videos/splatsim_0 (standalone lerobot-eval)
      .../<run>/dagger/interventions/videos/splatsim_0   (dagger orchestrator)

    Falls back to the folder's own name if no such ancestor is found.
    """
    folder = folder.resolve()
    cur = folder
    # Stop when we hit the filesystem root (parent == self).
    while cur.parent != cur:
        if cur.parent.name == "training":
            return cur.name
        cur = cur.parent
    return folder.name


def _find_videos(folder: Path, pattern: str) -> dict[str, Path]:
    """Return {basename: path} for video files under `folder`.

    Search order:
      1. Direct children of `folder` matching `pattern`.
      2. If none found, one level deeper (``<folder>/*/<pattern>``) — handles
         the lerobot/splatsim layout where mp4s live under ``splatsim_<N>/``.
    Duplicate basenames across multiple level-1 subdirs are warned about and
    only the first is kept (so e.g. passing the ``eval/`` parent dir doesn't
    silently mix episodes across step folders).
    """
    files = sorted(folder.glob(pattern))
    if not files:
        files = sorted(folder.glob("*/" + pattern))
    out: dict[str, Path] = {}
    for f in files:
        if f.name in out:
            print(
                f"WARNING: duplicate basename {f.name} found under {folder}; "
                f"keeping {out[f.name]}, skipping {f}. "
                f"Pass a more specific path if this isn't what you want."
            )
            continue
        out[f.name] = f
    return out


def match_videos(folders: list[Path], pattern: str) -> list[tuple[Path, ...]]:
    """Return tuples of paths (one per folder) sharing the same basename
    across ALL folders, in natural-sorted order.
    """
    per_folder = [_find_videos(folder, pattern) for folder in folders]
    name_sets = [set(d.keys()) for d in per_folder]
    common = sorted(set.intersection(*name_sets) if name_sets else set(), key=natural_sort_key)
    # Per-folder warning for files that won't be paired.
    for i, (folder, names) in enumerate(zip(folders, name_sets, strict=True)):
        missing_in_others = names - set(common)
        if missing_in_others:
            label = chr(ord("A") + i) if i < 26 else f"#{i}"
            extras = sorted(missing_in_others, key=natural_sort_key)
            preview = ", ".join(extras[:5]) + (" ..." if len(extras) > 5 else "")
            print(f"WARNING: skipping {len(extras)} file(s) only in folder {label} ({folder}): {preview}")
    return [tuple(d[name] for d in per_folder) for name in common]


def get_video_info(video_path: Path) -> tuple[int, int, float, int]:
    """(width, height, fps, frame_count) for one video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return w, h, fps, n


def _fit_font_scale(text: str, available_w: int, font: int, max_scale: float = 0.6) -> float:
    """Return the largest font scale (≤ max_scale) such that `text` fits within
    `available_w` pixels. Uses linear extrapolation from a unit-scale probe.
    """
    if not text:
        return max_scale
    (probe_w, _), _ = cv2.getTextSize(text, font, 1.0, 1)
    if probe_w <= 0:
        return max_scale
    # Target 95% of available so we don't crowd the edges.
    target = available_w * 0.95
    scale = target / probe_w
    return max(0.2, min(max_scale, scale))


def overlay_label(frame: np.ndarray, label: str) -> np.ndarray:
    """Draw `label` at top-left of `frame` with a semi-transparent background.

    Font scale is auto-shrunk to fit within the frame width with a small margin.
    Mutates a copy and returns it.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    margin = max(6, int(frame.shape[1] * 0.03))
    available_w = max(20, frame.shape[1] - 2 * margin)
    font_scale = _fit_font_scale(label, available_w, font, max_scale=0.6)
    thickness = max(1, int(round(font_scale * 1.6)))
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    pad = max(3, int(round(font_scale * 5)))
    x0, y0 = margin, margin
    x1, y1 = x0 + text_w + 2 * pad, y0 + text_h + 2 * pad + baseline
    out = frame.copy()
    overlay = out.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, dst=out)
    cv2.putText(
        out,
        label,
        (x0 + pad, y0 + pad + text_h),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return out


def overlay_status(frame: np.ndarray, text: str = "DONE") -> np.ndarray:
    """Draw a small 'DONE' badge centred near the bottom of `frame`.

    Used to mark the panel that's reached its last frame and is now frozen
    waiting for the longer video in the pair to finish.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    margin = max(6, int(frame.shape[1] * 0.03))
    available_w = max(40, frame.shape[1] - 2 * margin)
    # Smaller than the label region: ~0.5 max so it doesn't dominate.
    font_scale = _fit_font_scale(text, available_w, font, max_scale=0.5)
    thickness = max(1, int(round(font_scale * 1.8)))
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad = max(3, int(round(font_scale * 5)))
    x0 = (frame.shape[1] - text_w) // 2 - pad
    y1 = frame.shape[0] - margin
    y0 = y1 - text_h - 2 * pad - baseline
    x1 = x0 + text_w + 2 * pad
    out = frame.copy()
    overlay = out.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.6, out, 0.4, 0, dst=out)
    cv2.putText(
        out,
        text,
        (x0 + pad, y1 - pad - baseline),
        font,
        font_scale,
        (220, 220, 220),  # gray
        thickness,
        cv2.LINE_AA,
    )
    return out


def resize_to_width(frame: np.ndarray, target_w: int) -> np.ndarray:
    """Resize keeping aspect; returns same frame if already at target_w."""
    h, w = frame.shape[:2]
    if w == target_w:
        return frame
    target_h = int(round(h * target_w / w))
    return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)


def play_group_streaming(
    writer,  # imageio writer (returned by imageio.get_writer)
    paths: list[Path],
    labels: list[str],
    canvas_w: int,
    panel_heights: list[int],
    success_frame_threshold: int,
    final_hold_frames: int,
) -> None:
    """Stream N videos frame-by-frame to `writer`, freezing each one's last
    frame until the longest finishes. All panels are resized to `canvas_w`
    and stacked vertically (top→bottom in `paths` order).

    A panel "succeeds" iff its full frame count is strictly less than
    ``success_frame_threshold`` (i.e. the rollout ended before the cap, not
    truncated at the cap). Successful panels get a 'DONE' badge at the
    bottom both during their freeze period AND for ``final_hold_frames``
    extra frames after every panel has finished — so the very-last panel
    to finish still gets a visible DONE badge when there was no other panel
    left to freeze against.
    """
    n = len(paths)
    # Pre-compute "did this panel finish under the threshold?" by reading
    # the video's frame count. Strict `<` so an episode that exactly hits
    # the cap (typically a truncation) doesn't get a DONE badge.
    succeeded: list[bool] = []
    for p in paths:
        _, _, _, n_frames = get_video_info(p)
        succeeded.append(0 < n_frames < success_frame_threshold)

    caps = [cv2.VideoCapture(str(p)) for p in paths]
    last_frames: list[np.ndarray | None] = [None] * n
    done = [False] * n

    while not all(done):
        frames: list[np.ndarray | None] = []
        for i in range(n):
            if not done[i]:
                ret, frame = caps[i].read()
                if not ret:
                    done[i] = True
                    frame = last_frames[i]
                else:
                    last_frames[i] = frame
            else:
                frame = last_frames[i]
            frames.append(frame)

        # Build the stacked output. Per-panel: replace missing frames (rare —
        # 0-frame video) with black, resize/fit to (panel_h, canvas_w),
        # overlay the run-name label, and stamp DONE on frozen panels that
        # made the success threshold.
        any_still_running = not all(done)
        composed: list[np.ndarray] = []
        for i, frame in enumerate(frames):
            panel_h = panel_heights[i]
            if frame is None:
                frame = np.zeros((panel_h, canvas_w, 3), dtype=np.uint8)
            else:
                frame = resize_to_width(frame, canvas_w)
                frame = _fit_height(frame, panel_h)
            frame = overlay_label(frame, labels[i])
            if done[i] and any_still_running and succeeded[i]:
                frame = overlay_status(frame)
            composed.append(frame)

        stacked_bgr = np.vstack(composed)
        # imageio/ffmpeg wants RGB; cv2 reads/writes in BGR.
        writer.append_data(cv2.cvtColor(stacked_bgr, cv2.COLOR_BGR2RGB))

    for cap in caps:
        cap.release()

    # Final hold: after every panel has finished, write a short pause where
    # successful panels keep their DONE badge. Ensures the last panel to
    # finish — which had no "other panel still playing" to freeze against —
    # still gets a visible DONE before the black gap. Skipped when no panel
    # made the threshold (nothing to celebrate) or when configured to 0.
    if final_hold_frames > 0 and any(succeeded):
        composed_hold: list[np.ndarray] = []
        for i in range(n):
            panel_h = panel_heights[i]
            frame = last_frames[i]
            if frame is None:
                frame = np.zeros((panel_h, canvas_w, 3), dtype=np.uint8)
            else:
                frame = resize_to_width(frame, canvas_w)
                frame = _fit_height(frame, panel_h)
            frame = overlay_label(frame, labels[i])
            if succeeded[i]:
                frame = overlay_status(frame)
            composed_hold.append(frame)
        held_rgb = cv2.cvtColor(np.vstack(composed_hold), cv2.COLOR_BGR2RGB)
        for _ in range(final_hold_frames):
            writer.append_data(held_rgb)


def _fit_height(frame: np.ndarray, target_h: int) -> np.ndarray:
    """Pad with black or center-crop to make the height exactly target_h."""
    h, w = frame.shape[:2]
    if h == target_h:
        return frame
    if h < target_h:
        pad = target_h - h
        top = pad // 2
        bottom = pad - top
        return cv2.copyMakeBorder(frame, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    # h > target_h: center-crop.
    extra = h - target_h
    top = extra // 2
    return frame[top : top + target_h, :, :]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "folders",
        type=Path,
        nargs="+",
        help="Two or more folders of mp4s. Stacked top→bottom in arg order.",
    )
    ap.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help=(
            "One label per folder, overlaid top-left of each row. Default: "
            "auto-derive from each folder path (the training-run dir basename)."
        ),
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output mp4 path. Default: outputs/comparison_videos/compare_<labels>.mp4",
    )
    ap.add_argument(
        "--gap-seconds",
        type=float,
        default=1.0,
        help="Black-screen gap between groups, in seconds (default 1.0).",
    )
    ap.add_argument(
        "--success-threshold-seconds",
        type=float,
        default=20.0,
        help=(
            "Time threshold under which a rollout is treated as 'successful' "
            "(and earns a DONE badge). Translated to frames at each video's "
            "fps. Default 20.0 seconds."
        ),
    )
    ap.add_argument(
        "--final-hold-frames",
        type=int,
        default=10,
        help=(
            "After every panel in a group has finished, pause on the final "
            "frame for this many frames with the DONE badge still visible on "
            "successful panels. Ensures the last panel to finish has a "
            "visible DONE before the black gap. Default 10. Set to 0 to disable."
        ),
    )
    ap.add_argument(
        "--pattern",
        default="*.mp4",
        help="Glob pattern for video files in each folder (default '*.mp4').",
    )
    ap.add_argument(
        "--canvas-width",
        type=int,
        default=None,
        help=(
            "Output width in pixels. Default: max width across all folders' first "
            "matched video. All panels resize to this width keeping their aspect."
        ),
    )
    args = ap.parse_args()

    folders = [f.resolve() for f in args.folders]
    if len(folders) < 2:
        ap.error("At least two folders are required.")
    for f in folders:
        if not f.is_dir():
            ap.error(f"Not a directory: {f}")

    if args.labels is not None:
        if len(args.labels) != len(folders):
            ap.error(
                f"--labels has {len(args.labels)} entries but {len(folders)} folder(s) "
                f"were provided. Pass one label per folder."
            )
        labels = list(args.labels)
    else:
        labels = [derive_label_from_folder(f) for f in folders]

    groups = match_videos(folders, args.pattern)
    if not groups:
        print(f"No {args.pattern} files common to all {len(folders)} folders. Exiting.")
        return 1

    for i, label in enumerate(labels):
        position = "Top" if i == 0 else ("Bottom" if i == len(labels) - 1 else f"Row {i + 1}")
        print(f"{position} label:    {label}")
    print(f"Matched groups ({len(groups)}):")
    for group in groups:
        print(f"  {group[0].name}")

    # Inspect the first group's videos to determine output dims and fps.
    infos = [get_video_info(p) for p in groups[0]]
    widths = [w for (w, _, _, _) in infos]
    heights = [h for (_, h, _, _) in infos]
    fpss = [fps for (_, _, fps, _) in infos]
    fps0 = fpss[0] if fpss[0] > 0 else 30.0
    for i, fps_i in enumerate(fpss[1:], start=1):
        if abs(fps_i - fps0) > 0.5:
            print(
                f"WARNING: fps for folder {i} ({fps_i:.2f}) differs from folder 0 "
                f"({fps0:.2f}); using folder-0 fps for the output."
            )
    fps = fps0

    canvas_w = args.canvas_width or max(widths)
    # Each panel resizes to canvas_w keeping its aspect; lock the panel
    # heights from the first group so the writer's frame size stays
    # consistent across groups.
    panel_heights = [int(round(h * canvas_w / w)) for w, h in zip(widths, heights, strict=True)]
    out_h = sum(panel_heights)
    out_w = canvas_w
    panel_summary = ", ".join(f"row {i} {canvas_w}×{ph}" for i, ph in enumerate(panel_heights))
    print(f"Output dims: {out_w}×{out_h} @ {fps:.2f} fps ({panel_summary})")

    if args.output is None:
        # Default to outputs/comparison_videos/ at the repo root so the
        # comparison artifacts live alongside the other auto-generated stuff.
        repo_root = Path(__file__).resolve().parents[1]
        default_dir = repo_root / "outputs" / "comparison_videos"
        # Avoid path-unsafe chars in the auto-derived labels by replacing /.
        # Cap individual labels to keep the joined filename under typical FS
        # limits when many folders are passed.
        safe = ["_".join(label.replace("/", "_").split())[:80] for label in labels]
        stem = "compare_" + "__vs__".join(safe)
        # Hard ceiling at ~200 chars total stem to avoid ENAMETOOLONG.
        if len(stem) > 200:
            stem = stem[:200].rstrip("_") + f"__and{len(labels) - 2}more"
        output_path = (default_dir / f"{stem}.mp4").resolve()
    else:
        output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Output:       {output_path}")

    # Use imageio (ffmpeg backend) so the output is H.264 / yuv420p —
    # widely-compatible (browsers, VSCode preview, etc.). OpenCV's mp4v
    # fourcc produces MPEG-4 Part 2 which many viewers can't decode.
    # macro_block_size=1 lets us keep odd dimensions instead of imageio
    # silently rounding to a multiple of 16.
    writer = imageio.get_writer(
        str(output_path),
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=1,
    )

    gap_frames = max(0, int(round(args.gap_seconds * fps)))
    # Black frame in RGB (matches imageio's expected channel order).
    black_frame_rgb = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    # Translate the success threshold from seconds → frames at the output
    # fps. Same fps is used for every panel (matches the writer's), so this
    # converts cleanly. If a panel's source video has a different fps from
    # the writer, the comparison is still in writer-frame terms — close
    # enough since we already warned about fps mismatches above.
    success_frame_threshold = max(0, int(round(args.success_threshold_seconds * fps)))
    print(
        f"Success threshold: {args.success_threshold_seconds:.1f}s "
        f"= {success_frame_threshold} frames @ {fps:.2f} fps"
    )

    try:
        for i, group in enumerate(groups):
            print(f"[{i + 1}/{len(groups)}] {group[0].name}")
            play_group_streaming(
                writer=writer,
                paths=list(group),
                labels=labels,
                canvas_w=canvas_w,
                panel_heights=panel_heights,
                success_frame_threshold=success_frame_threshold,
                final_hold_frames=args.final_hold_frames,
            )
            # Gap between groups, but not after the last one so the video ends
            # on the final frame rather than fading to black.
            if i < len(groups) - 1:
                for _ in range(gap_frames):
                    writer.append_data(black_frame_rgb)
    finally:
        writer.close()

    print(f"Done. Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
