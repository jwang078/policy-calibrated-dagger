#!/usr/bin/env python3
"""Extract one mp4 per episode from a LeRobot dataset's embedded image bytes.

Why this exists: lerobot-eval saves intervention recordings as
`<output_dir>/videos/splatsim_0/eval_episode_<N>.mp4`, but
`augment_dataset_with_blending.py` (which produces the blend datasets used
in rerun-blends mode) saves frames as PNG bytes embedded in the dataset's
parquet rows. So you can't `visualize_intervention_episode.py` a blend
lineage directly — there's no mp4 to read. This script pulls the bytes
out, decodes them, and writes them as H.264 mp4s into a folder you can
inspect, diff, or annotate the same way as eval_episode_<N>.mp4.

Default behavior:
    * one mp4 per LeRobot episode (matches the dataset's natural episode
      boundaries — for blend datasets, each "episode" is one closed-loop
      replay of a source intervention cycle)
    * camera = `observation.images.base_rgb_stretch` (the stretched 224x224
      base camera, matches what lerobot-eval renders by default)
    * H.264 + yuv420p + faststart, so the output is VSCode/browser-playable
    * dataset's recorded fps (from meta/info.json) is reproduced exactly

Usage:
    python my_scripts/extract_dataset_videos.py <dataset_repo_id> \\
        [--cache_dir ~/.cache/huggingface/lerobot] \\
        [--camera observation.images.base_rgb_stretch] \\
        [--out_dir <path>] \\
        [--episodes 0 1 2]    # optional subset

    # Compare blend trajectory vs source intervention trajectory:
    python my_scripts/extract_dataset_videos.py JennyWWW/lever_g0_30ep_03dag_diff_r_dag1_blend010
    python my_scripts/extract_dataset_videos.py JennyWWW/lever_g0_30ep_03dag_diff_r_dag1

    # Default --out_dir lands next to the dataset cache:
    #   <cache_dir>/<repo_id>/extracted_videos/episode_<N>.mp4
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2  # type: ignore[import-not-found]
import numpy as np
import pyarrow.parquet as pq

DEFAULT_CACHE = Path.home() / ".cache" / "huggingface" / "lerobot"
DEFAULT_CAMERA = "observation.images.base_rgb_stretch"


def _decode_image_bytes(b: bytes) -> np.ndarray:
    """PNG/JPEG bytes → BGR numpy array suitable for cv2.VideoWriter."""
    arr = np.frombuffer(b, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
    if img is None:
        raise RuntimeError(
            f"cv2.imdecode failed on {len(b)} bytes (header: {b[:8]!r}). "
            "Image bytes may be corrupt or in an unsupported format."
        )
    return img


def _load_dataset_meta(dataset_root: Path) -> dict:
    """Read meta/info.json for fps, total episodes, etc."""
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Dataset meta missing: {info_path}. Is this a valid LeRobot dataset?")
    return json.loads(info_path.read_text())


def _iter_episode_frames(dataset_root: Path, camera_col: str) -> dict[int, list[bytes]]:
    """Walk every parquet file in chunk-000/, group frames by episode_index.

    Returns:
        {episode_index: [frame_bytes_for_frame_0, frame_bytes_for_frame_1, ...]}
        Frames are ordered by `frame_index` within each episode.
    """
    parquet_dir = dataset_root / "data" / "chunk-000"
    if not parquet_dir.is_dir():
        raise FileNotFoundError(f"No data/chunk-000/ under {dataset_root}")

    # Collect all (episode_index, frame_index, image_bytes) triples first so we
    # can sort within-episode regardless of how rows were split across parquet
    # files. Small enough to fit in memory for typical datasets (~7K frames *
    # ~100 KB per image ≈ 700 MB max; smaller in practice for stretch 224x224).
    rows: list[tuple[int, int, bytes]] = []
    for pq_file in sorted(parquet_dir.glob("file-*.parquet")):
        t = pq.read_table(pq_file, columns=[camera_col, "episode_index", "frame_index"])
        col = t[camera_col].to_pylist()
        epi = t["episode_index"].to_pylist()
        fri = t["frame_index"].to_pylist()
        for img_struct, ep, fr in zip(col, epi, fri, strict=True):
            rows.append((int(ep), int(fr), img_struct["bytes"]))

    # Group + sort.
    by_ep: dict[int, list[tuple[int, bytes]]] = {}
    for ep, fr, b in rows:
        by_ep.setdefault(ep, []).append((fr, b))
    out: dict[int, list[bytes]] = {}
    for ep, lst in by_ep.items():
        lst.sort(key=lambda x: x[0])
        # Validate contiguous frame_index — sanity check that we didn't miss any.
        expected = list(range(lst[0][0], lst[0][0] + len(lst)))
        actual = [x[0] for x in lst]
        if actual != expected:
            print(
                f"[extract] WARNING: episode {ep} has non-contiguous frame_index "
                f"(first {actual[:5]}, expected {expected[:5]}); frames will still "
                f"be written in sorted order but timing may drift.",
                file=sys.stderr,
            )
        out[ep] = [b for _, b in lst]
    return out


def _write_episode_mp4(
    frames: list[bytes],
    fps: float,
    out_path: Path,
    ffmpeg_bin: str,
) -> None:
    """Decode each frame's bytes and write an H.264 + yuv420p mp4.

    Same encode pipeline as visualize_intervention_episode.py: cv2 writes
    a temp mp4v file, ffmpeg re-encodes to libx264 + yuv420p + faststart
    so VSCode, browsers, and any modern player accept the result.
    """
    if not frames:
        raise ValueError("Cannot write an empty episode (no frames).")

    # Decode the first frame to learn dimensions.
    first = _decode_image_bytes(frames[0])
    height, width = first.shape[:2]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix="extract_dataset_", dir=out_path.parent)
    tmp_video = Path(tmp_dir) / "raw_mp4v.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_video), fourcc, fps, (width, height))
    if not writer.isOpened():
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError(f"Failed to open temp video writer: {tmp_video}")
    try:
        writer.write(first)
        for b in frames[1:]:
            writer.write(_decode_image_bytes(b))
    finally:
        writer.release()

    # H.264 re-encode for VSCode compat. Pad to even dims (libx264 requires
    # even width/height); 224x224 is already even but this guards future
    # camera-size changes.
    cmd = [
        ffmpeg_bin,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(tmp_video),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-movflags",
        "+faststart",
        "-preset",
        "veryfast",
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg re-encode failed (exit {proc.returncode}):\n{proc.stderr.strip()}")


def extract(
    dataset_repo_id: str,
    cache_dir: Path,
    camera: str,
    out_dir: Path | None,
    episodes: list[int] | None,
) -> Path:
    dataset_root = cache_dir / dataset_repo_id
    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"Dataset not found under cache: {dataset_root}. Check the repo_id and --cache_dir."
        )

    info = _load_dataset_meta(dataset_root)
    fps = float(info.get("fps", 30))
    total_eps = int(info.get("total_episodes", 0))
    total_frames = int(info.get("total_frames", 0))

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install via "
            "`conda install -c conda-forge ffmpeg` or `apt install ffmpeg`."
        )

    if out_dir is None:
        out_dir = dataset_root / "extracted_videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[extract] dataset: {dataset_root}")
    print(f"[extract]   {total_eps} episodes, {total_frames} frames, fps={fps}")
    print(f"[extract]   camera column: {camera}")
    print(f"[extract] writing → {out_dir}")

    # Pull frames out of parquet, grouped by episode.
    by_ep = _iter_episode_frames(dataset_root, camera)
    available = sorted(by_ep.keys())
    if not available:
        raise RuntimeError(f"No episodes found in {dataset_root}/data/chunk-000/")

    # Filter to requested episodes if specified; otherwise extract all.
    if episodes is not None:
        missing = [e for e in episodes if e not in by_ep]
        if missing:
            raise ValueError(
                f"Requested episode(s) {missing} not in dataset "
                f"(available: {available[:5]}...{available[-5:]} of {len(available)})"
            )
        episodes_to_write = episodes
    else:
        episodes_to_write = available

    for ep in episodes_to_write:
        frames = by_ep[ep]
        out_path = out_dir / f"episode_{ep}.mp4"
        _write_episode_mp4(frames, fps, out_path, ffmpeg_bin)
        print(f"[extract]   episode {ep}: {len(frames)} frames → {out_path.name}")

    print(f"[extract] Done. Wrote {len(episodes_to_write)} mp4 file(s) to {out_dir}")
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "dataset_repo_id",
        help="HF repo id (used as <cache_dir>/<repo_id> path), e.g. "
        "JennyWWW/lever_g0_30ep_03dag_diff_r_dag1_blend010",
    )
    p.add_argument(
        "--cache_dir",
        type=Path,
        default=DEFAULT_CACHE,
        help=f"LeRobot dataset cache root (default: {DEFAULT_CACHE}).",
    )
    p.add_argument(
        "--camera",
        default=DEFAULT_CAMERA,
        help=f"Image column to extract (default: {DEFAULT_CAMERA}). Other "
        "options typically include observation.images.base_rgb_letterbox, "
        "observation.images.wrist_rgb_stretch, observation.images.wrist_rgb_letterbox.",
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Where to write the mp4s (default: <cache>/<repo_id>/extracted_videos/).",
    )
    p.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=None,
        help="Optional subset of episode indices to extract. Default: all.",
    )
    args = p.parse_args()
    extract(
        dataset_repo_id=args.dataset_repo_id,
        cache_dir=args.cache_dir,
        camera=args.camera,
        out_dir=args.out_dir,
        episodes=args.episodes,
    )


if __name__ == "__main__":
    main()
