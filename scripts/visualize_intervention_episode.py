#!/usr/bin/env python3
"""Annotate one intervention episode video with phase labels (policy / rrt /
trigger reason at each cycle's start).

Inputs:
    training_dir   path to a DAgger training output dir, e.g.
                   outputs/training/diffusion_<...>_ft_dag<N>
    episode_idx    0-indexed episode to visualize. Maps directly to the
                   `eval_episode_<idx>.mp4` filename AND to
                   `scenario_idx == <idx>` in intervention_per_scenario.csv.

Output (default):  <training_dir>/dagger/interventions/videos/splatsim_0/
                       eval_episode_<idx>_annotated.mp4

How phases are labeled (per cycle i, with trigger_steps[i] = T and
rrt_steps_executed[i] = L):

    frames [0, T]            "policy"  ── T is the last policy frame; the
                                          trigger fired in the controller
                                          tick AFTER this frame was rendered.
    pause after frame T      "<trigger reason>" repeated for --pause_frames
                                          frames (default 30 ≈ 1 s @ 30 fps).
    frames [T+1, T+L]        "rrt"      ── L is rrt_steps_executed. When L
                                          is 0 (planning failed for the cycle,
                                          OR an old CSV with the trailing-0
                                          bug), no RRT-labeled frames are
                                          emitted — the pause alone marks the
                                          trigger event.
    frames [T+L+1, …]        "policy"   ── until the next trigger or EOF.

Usage:
    # Standard single-pane annotated video:
    python my_scripts/visualize_intervention_episode.py \\
        outputs/training/diffusion_<...>_ft_dag2 \\
        3 \\
        [--pause_frames 30] \\
        [--output some_file.mp4]

    # Side-by-side with a blend dataset on the right pane:
    python my_scripts/visualize_intervention_episode.py \\
        outputs/training/diffusion_<...>_ft_dag2 \\
        3 \\
        --side_by_side JennyWWW/lever_g0_30ep_03dag_diff_r_dag1_blend010

In side-by-side mode, LEFT is the source's eval_episode_<idx>.mp4 with the
standard annotations; RIGHT shows the matching blend dataset frames during
each RRT cycle, and a "blend diverges here" placeholder during the source's
policy phases. The blend dataset's `source_scenario_idx` per-episode field
is what pairs blend recordings to source cycles — both datasets emit one
episode per intervention cycle, so the cycle ordering matches exactly.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2  # type: ignore[import-not-found]
import numpy as np

try:
    import pyarrow.parquet as pq  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover — only needed for --side_by_side
    pq = None

# Visual constants — kept simple so anyone can tweak without reading further.
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.9
FONT_THICKNESS = 2
LABEL_POS = (12, 32)  # top-left, with a bit of padding
COLOR_POLICY = (60, 220, 60)  # green
COLOR_RRT = (60, 60, 220)  # red
COLOR_TRIGGER = (60, 220, 220)  # yellow — distinct from rrt/policy
OUTLINE_COLOR = (0, 0, 0)
OUTLINE_THICKNESS_EXTRA = 4  # outline is text-thickness + this


def _draw_label(frame: np.ndarray, text: str, color: tuple[int, int, int]) -> None:
    """Draw `text` on `frame` at LABEL_POS with a black outline for legibility."""
    # Outline first, then fill on top.
    cv2.putText(
        frame,
        text,
        LABEL_POS,
        FONT,
        FONT_SCALE,
        OUTLINE_COLOR,
        FONT_THICKNESS + OUTLINE_THICKNESS_EXTRA,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        LABEL_POS,
        FONT,
        FONT_SCALE,
        color,
        FONT_THICKNESS,
        cv2.LINE_AA,
    )


def _find_csv_row(csv_path: Path, scenario_idx: int) -> dict[str, str]:
    """Locate the row where `scenario_idx` matches the requested episode."""
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["scenario_idx"]) == scenario_idx:
                return row
    raise ValueError(
        f"No row with scenario_idx={scenario_idx} in {csv_path}. "
        f"The CSV may not cover this episode (e.g. a partial recording)."
    )


def _parse_int_list(s: str) -> list[int]:
    """Parse a comma-separated list of ints. Empty string → empty list."""
    s = s.strip()
    if not s:
        return []
    return [int(x) for x in s.split(",")]


def _build_phase_label(
    frame_idx: int,
    trigger_steps: list[int],
    rrt_steps_executed: list[int],
) -> tuple[str, tuple[int, int, int]]:
    """For a given video-frame index, return (label, color).

    Walks the cycle list to decide. Policy by default; switches to "rrt"
    when the frame is strictly inside (trigger_steps[i], trigger_steps[i] +
    rrt_steps_executed[i]] for some cycle i.
    """
    for ts, exec_len in zip(trigger_steps, rrt_steps_executed, strict=True):
        if exec_len > 0 and ts < frame_idx <= ts + exec_len:
            return "rrt", COLOR_RRT
    return "policy", COLOR_POLICY


def _cycle_idx_at_frame(
    frame_idx: int,
    trigger_steps: list[int],
    rrt_steps_executed: list[int],
) -> int | None:
    """Which cycle (CSV ordinal) the given source frame_idx belongs to.

    Returns the cycle index if the frame is inside the RRT-driven window
    (trigger_steps[i], trigger_steps[i] + rrt_steps_executed[i]], or None
    if the frame is in a policy phase. Cycles with rrt_steps_executed == 0
    occupy zero-width windows and never match.
    """
    for i, (ts, L) in enumerate(zip(trigger_steps, rrt_steps_executed, strict=True)):
        if L > 0 and ts < frame_idx <= ts + L:
            return i
    return None


def _load_blend_frames_for_scenario(
    blend_dataset_path: Path,
    scenario_idx: int,
    rrt_steps_executed: list[int],
    camera_col: str,
) -> list[list[np.ndarray]]:
    """Decode the blend-dataset frames that correspond to each CSV cycle of
    the requested source scenario.

    Returns a list of length len(rrt_steps_executed). Entry i is:
      * A list of `rrt_steps_executed[i]` BGR numpy frames (one per env tick
        during this cycle's RRT execution) when the cycle's RRT actually
        ran, OR
      * An empty list when rrt_steps_executed[i] == 0 (no blend recording
        exists for that cycle because RRT didn't execute).

    Pairing logic: the blend dataset has one episode per source intervention
    cycle. Both datasets share a `source_scenario_idx` per-episode field, so
    we filter the blend meta to the requested scenario and take episodes in
    `episode_index` order — that matches the CSV's cycle order (cycles with
    rrt_steps_executed == 0 are skipped in the dataset).
    """
    if pq is None:
        raise RuntimeError(
            "pyarrow not available; install via `pip install pyarrow` (it is "
            "already a lerobot dep). Required for --side_by_side blend loading."
        )

    meta_path = blend_dataset_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Blend dataset meta missing: {meta_path}")
    meta_tbl = pq.read_table(meta_path, columns=["episode_index", "source_scenario_idx"]).to_pandas()
    blend_eps_for_scenario = sorted(
        int(x) for x in meta_tbl[meta_tbl["source_scenario_idx"] == scenario_idx]["episode_index"].tolist()
    )

    # Cycles in CSV order that have a corresponding blend recording (=
    # rrt_steps_executed > 0). 1:1 with `blend_eps_for_scenario`.
    nonzero_cycle_idxs = [i for i, L in enumerate(rrt_steps_executed) if L > 0]
    if len(nonzero_cycle_idxs) != len(blend_eps_for_scenario):
        print(
            f"[side_by_side] WARNING: scenario {scenario_idx} has "
            f"{len(nonzero_cycle_idxs)} CSV cycle(s) with rrt_steps_executed > 0 "
            f"but the blend dataset has {len(blend_eps_for_scenario)} episode(s) "
            f"for this scenario. Pairing by position; mismatches may show stale "
            f"blend frames against unrelated source cycles.",
            file=sys.stderr,
        )

    # Read all parquet data once and bucket rows by episode_index to avoid
    # re-scanning files per cycle. Each cycle's blend recording is small
    # (≤ ~200 frames × ~50 KB PNG = ~10 MB), so memory is fine even with
    # all 68 episodes loaded.
    data_dir = blend_dataset_path / "data" / "chunk-000"
    target_eps = set(blend_eps_for_scenario)
    by_ep: dict[int, list[tuple[int, bytes]]] = {ep: [] for ep in target_eps}
    for pf in sorted(data_dir.glob("file-*.parquet")):
        tbl = pq.read_table(pf, columns=[camera_col, "episode_index", "frame_index"])
        epi = tbl["episode_index"].to_pylist()
        fri = tbl["frame_index"].to_pylist()
        img = tbl[camera_col].to_pylist()
        for ep, fr, im in zip(epi, fri, img, strict=True):
            if ep in target_eps:
                by_ep[int(ep)].append((int(fr), im["bytes"]))

    # Per-cycle decoded BGR frames, trimmed to rrt_steps_executed[i].
    out: list[list[np.ndarray]] = [[] for _ in rrt_steps_executed]
    for ci, blend_ep in zip(nonzero_cycle_idxs, blend_eps_for_scenario, strict=True):
        rows = sorted(by_ep[blend_ep], key=lambda x: x[0])
        L = rrt_steps_executed[ci]
        # Blend episodes are padded to a minimum length (teleop_min_episode_length).
        # We only need the first L frames — the rest are HOLD frames at the
        # terminal state and would falsely imply the blend was still moving.
        decoded: list[np.ndarray] = []
        for _, b in rows[:L]:
            arr = np.frombuffer(b, dtype=np.uint8)
            img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise RuntimeError(
                    f"Failed to decode blend frame (episode {blend_ep}, "
                    f"{len(b)} bytes). Image may be corrupt."
                )
            decoded.append(img_bgr)
        if len(decoded) < L:
            print(
                f"[side_by_side] NOTE: blend episode {blend_ep} has only "
                f"{len(decoded)} frame(s) but cycle {ci} needs {L}. Right "
                f"side will hold on the last available blend frame for the "
                f"remaining {L - len(decoded)} frame(s).",
                file=sys.stderr,
            )
        out[ci] = decoded
    return out


def _render_side_by_side_frame(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    left_label: str,
    left_color: tuple[int, int, int],
    right_label: str,
    right_color: tuple[int, int, int],
    header_label: str | None = None,
    header_color: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Compose left and right BGR frames into a single side-by-side image.

    Always emits a 40 px header band on top (blank when ``header_label`` is
    None) so every frame has identical dimensions. This is critical because
    the cv2 VideoWriter is initialized with a fixed (width, height) and
    silently rejects frames whose dims don't match — produces "Failed to
    write frame" warnings + a near-empty output file. The phase labels
    (``left_label``, ``right_label``) go in each pane's top-left corner.
    """
    h = max(left_bgr.shape[0], right_bgr.shape[0])

    # Pad each pane to the common height so concatenation lines up.
    def _pad_to_h(img: np.ndarray) -> np.ndarray:
        if img.shape[0] == h:
            return img
        pad = np.zeros((h - img.shape[0], img.shape[1], 3), dtype=img.dtype)
        return np.vstack([img, pad])

    left = _pad_to_h(left_bgr.copy())
    right = _pad_to_h(right_bgr.copy())
    _draw_label(left, left_label, left_color)
    _draw_label(right, right_label, right_color)

    # Thin black separator between panes for visual distinctness.
    sep = np.zeros((h, 2, 3), dtype=left.dtype)
    body = np.hstack([left, sep, right])

    # Header band — ALWAYS present (40 px tall). Blank when no label so the
    # output canvas stays a constant size across every frame, matching the
    # VideoWriter's fixed dimensions.
    header = np.zeros((40, body.shape[1], 3), dtype=body.dtype)
    if header_label:
        text_size, _ = cv2.getTextSize(header_label, FONT, FONT_SCALE, FONT_THICKNESS)
        tx = max(12, (header.shape[1] - text_size[0]) // 2)
        ty = (40 + text_size[1]) // 2
        cv2.putText(
            header,
            header_label,
            (tx, ty),
            FONT,
            FONT_SCALE,
            header_color or COLOR_TRIGGER,
            FONT_THICKNESS,
            cv2.LINE_AA,
        )
    return np.vstack([header, body])


def _make_placeholder_frame(
    width: int, height: int, text: str, color: tuple[int, int, int] = (180, 180, 180)
) -> np.ndarray:
    """Build a dark BGR frame with a centered single-line caption.

    Used for the right pane in --side_by_side mode when the source is in a
    policy phase (no blend recording exists outside RRT cycles).
    """
    img = np.full((height, width, 3), 30, dtype=np.uint8)  # near-black
    text_size, _ = cv2.getTextSize(text, FONT, 0.55, 1)
    tx = max(6, (width - text_size[0]) // 2)
    ty = (height + text_size[1]) // 2
    cv2.putText(img, text, (tx, ty), FONT, 0.55, color, 1, cv2.LINE_AA)
    return img


def _resolve_interventions_dir(root: Path) -> Path:
    """Pick the directory under which we expect to find
    ``intervention_per_scenario.csv`` + ``videos/splatsim_0/eval_episode_*.mp4``.

    Two layouts are supported:
      1. **Training-dir layout** (canonical DAgger output):
         ``<root>/dagger/interventions/{intervention_per_scenario.csv, videos/}``.
      2. **Eval-dir layout** (when lerobot-eval is invoked standalone, e.g.
         for a one-off shield smoke test): the CSV + videos sit directly
         under ``<root>/`` since there's no DAgger wrapper around the eval.

    We auto-detect by looking for the CSV at each location.
    """
    nested = root / "dagger" / "interventions"
    if (nested / "intervention_per_scenario.csv").is_file():
        return nested
    if (root / "intervention_per_scenario.csv").is_file():
        return root
    raise FileNotFoundError(
        f"Couldn't find intervention_per_scenario.csv under either:\n"
        f"  {nested / 'intervention_per_scenario.csv'} (training-dir layout)\n"
        f"  {root / 'intervention_per_scenario.csv'} (eval-dir layout)\n"
        f"Pass either a DAgger training_dir OR a standalone lerobot-eval output_dir."
    )


def annotate(
    training_dir: Path,
    episode_idx: int,
    pause_frames: int,
    output_path: Path | None,
    side_by_side_blend_repo_id: str | None = None,
    cache_dir: Path | None = None,
    blend_camera_col: str = "observation.images.base_rgb_stretch",
) -> Path:
    interventions_dir = _resolve_interventions_dir(training_dir)
    video_path = interventions_dir / "videos" / "splatsim_0" / f"eval_episode_{episode_idx}.mp4"
    csv_path = interventions_dir / "intervention_per_scenario.csv"

    if not video_path.is_file():
        raise FileNotFoundError(
            f"Video not found: {video_path}\n"
            f"  Check that `max_episodes_rendered` covered episode {episode_idx} "
            f"during the recording."
        )
    if not csv_path.is_file():
        raise FileNotFoundError(f"Intervention CSV not found: {csv_path}")

    row = _find_csv_row(csv_path, episode_idx)
    triggers_raw = row.get("triggers", "")
    trigger_steps = _parse_int_list(row.get("trigger_steps", ""))
    rrt_steps_executed = _parse_int_list(row.get("rrt_steps_executed", ""))
    # Renamed labels: map pre-rename CSV rows to the current names so the
    # rendered annotation matches what the controller emits now. Add new
    # entries here if labels get renamed again.
    _TRIGGER_REASON_ALIASES = {"future_chunk_shield": "future_chunk_coll"}
    triggers = (
        [_TRIGGER_REASON_ALIASES.get(t, t) for t in triggers_raw.split(",") if t] if triggers_raw else []
    )

    # Sanity: parallel lists. We require rrt_steps_executed to be present —
    # without it we can't draw RRT-phase frames. Older CSVs (pre this column)
    # need to be re-recorded.
    if "rrt_steps_executed" not in row:
        raise ValueError(
            f"CSV at {csv_path} is missing the `rrt_steps_executed` column "
            f"(this is an older CSV format). Re-record interventions with a "
            f"current lerobot-eval to get the column."
        )
    if not (len(triggers) == len(trigger_steps) == len(rrt_steps_executed)):
        raise ValueError(
            f"CSV cycle columns are not parallel for scenario {episode_idx}: "
            f"triggers={len(triggers)} trigger_steps={len(trigger_steps)} "
            f"rrt_steps_executed={len(rrt_steps_executed)}"
        )

    # ── open source video ──────────────────────────────────────────────── #
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps:  # 0 or NaN
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_input_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ── optionally load blend frames for the side-by-side comparison ───── #
    # When `side_by_side_blend_repo_id` is set, we'll display two columns:
    #   LEFT  = the source's eval_episode_<idx>.mp4 with the standard
    #           policy/rrt/trigger annotations.
    #   RIGHT = the blend dataset's frames for each cycle's RRT execution,
    #           or a "policy phase — no blend coverage" placeholder during
    #           source policy phases.
    # Blend cycles align in time to source cycles because both come from
    # the same env stepped at the same rate; the right pane just freezes
    # on a placeholder outside RRT windows so the user sees clearly that
    # blend trajectories diverge from source-policy continuations.
    blend_frames_by_cycle: list[list[np.ndarray]] = []
    blend_dataset_path: Path | None = None
    if side_by_side_blend_repo_id is not None:
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "huggingface" / "lerobot"
        blend_dataset_path = cache_dir / side_by_side_blend_repo_id
        if not blend_dataset_path.is_dir():
            raise FileNotFoundError(
                f"Blend dataset not found: {blend_dataset_path}. Pass --cache_dir "
                f"if your LeRobot cache is elsewhere."
            )
        blend_frames_by_cycle = _load_blend_frames_for_scenario(
            blend_dataset_path=blend_dataset_path,
            scenario_idx=episode_idx,
            rrt_steps_executed=rrt_steps_executed,
            camera_col=blend_camera_col,
        )

    # ── open destination video ─────────────────────────────────────────── #
    # Strategy: cv2 writes a TEMP mp4 with `mp4v` (no extra deps, works
    # everywhere cv2 does). After we've written every frame, we re-encode
    # the temp file with ffmpeg into H.264 + yuv420p — the format that
    # VSCode's preview, browsers, QuickTime, etc. actually accept.
    # `mp4v` alone produces a file most lightweight previewers reject as
    # "can't open video file". The re-encode is fast (≈ real-time-ish on
    # 224x448 input) and the temp file is deleted at the end.
    if output_path is None:
        suffix = "_side_by_side" if side_by_side_blend_repo_id else "_annotated"
        output_path = video_path.parent / f"eval_episode_{episode_idx}{suffix}.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install it (`conda install -c conda-forge "
            "ffmpeg` or `apt install ffmpeg`) so the annotated video can be "
            "re-encoded into VSCode/browser-compatible H.264."
        )

    # Temp file lives next to the destination so we don't cross filesystem
    # boundaries on the final move/replace.
    tmp_dir = tempfile.mkdtemp(prefix="viz_intervention_", dir=output_path.parent)
    tmp_video = Path(tmp_dir) / "raw_mp4v.mp4"

    # Determine output dimensions:
    #   single-pane mode: same as source video
    #   side-by-side mode: source width + 2px separator + blend width on the
    #     same height (blend frames padded to source height if shorter),
    #     plus a 40px header band for global captions ("TRIGGER: …",
    #     "BLEND DIVERGED" notes, etc.)
    if side_by_side_blend_repo_id is not None:
        # Need the blend frame dimensions. Use the first non-empty cycle's
        # first frame; fall back to a 224x224 placeholder if every cycle
        # has rrt_steps_executed == 0 (degenerate but possible).
        sample_blend_frame: np.ndarray | None = None
        for cycle_frames in blend_frames_by_cycle:
            if cycle_frames:
                sample_blend_frame = cycle_frames[0]
                break
        if sample_blend_frame is None:
            # All cycles had rrt_steps_executed == 0; side-by-side is degenerate.
            print(
                f"[side_by_side] WARNING: scenario {episode_idx} has no cycles with "
                f"rrt_steps_executed > 0; right pane will be all 'no blend' placeholder.",
                file=sys.stderr,
            )
            blend_h, blend_w = height, height  # square placeholder matching source height
        else:
            blend_h, blend_w, _ = sample_blend_frame.shape
        # Final canvas: header band on top + (source | sep | blend) below.
        # Both source and blend rows are padded to max(source_h, blend_h).
        pane_h = max(height, blend_h)
        out_width = width + 2 + blend_w
        out_height = 40 + pane_h
    else:
        out_width = width
        out_height = height

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_video), fourcc, fps, (out_width, out_height))
    if not writer.isOpened():
        shutil.rmtree(tmp_dir, ignore_errors=True)
        cap.release()
        raise RuntimeError(f"Failed to open intermediate video for writing: {tmp_video}")

    # Summary log so the user knows what mapping is being applied.
    print(f"[viz] Source video:   {video_path}")
    print(f"[viz]   fps={fps:.3f}  size={width}x{height}  frames={n_input_frames}")
    if side_by_side_blend_repo_id is not None:
        print(f"[viz] Side-by-side blend dataset: {blend_dataset_path}")
        print(f"[viz]   camera column: {blend_camera_col}")
        print(f"[viz]   blend frames per cycle: {[len(f) for f in blend_frames_by_cycle]}")
        print(f"[viz]   output canvas: {out_width}x{out_height}")
    print(f"[viz] CSV row scenario_idx={episode_idx}, cycles={len(triggers)}:")
    for i, (t, ts, L) in enumerate(zip(triggers, trigger_steps, rrt_steps_executed, strict=True)):
        end = ts + L if L > 0 else None
        if end is not None:
            print(
                f"[viz]   cycle {i}: trigger='{t}' fires @ frame {ts} → "
                f"RRT runs frames [{ts + 1}, {end}] ({L} frames)"
            )
        else:
            print(
                f"[viz]   cycle {i}: trigger='{t}' fires @ frame {ts} → "
                f"rrt_steps_executed=0 (no RRT phase visualized; pause only)"
            )
    print(
        f"[viz] Pause length at each trigger: {pause_frames} frames "
        f"(~{pause_frames / fps:.2f} s @ {fps:.0f} fps)"
    )
    print(f"[viz] Writing → {output_path}")

    # ── main loop ──────────────────────────────────────────────────────── #
    cycle_idx = 0  # index of the next unhandled trigger
    frame_idx = 0
    n_written = 0

    def _right_pane_for_frame(f_idx: int) -> tuple[np.ndarray, str, tuple[int, int, int]]:
        """Build the right-side frame + (label, color) for source frame f_idx.

        Returns:
          - During an RRT cycle window: the blend dataset's matching frame
            (or the last available blend frame held, if blend ran shorter
            than the source's RRT phase for this cycle).
          - During a policy phase (pre/post any cycle): a dark placeholder
            stating the blend trajectory diverged from this point.
        """
        # Side-by-side mode is gated on the outer `if`, so this helper is
        # only called when blend_frames_by_cycle is populated.
        ci = _cycle_idx_at_frame(f_idx, trigger_steps, rrt_steps_executed)
        if ci is not None:
            offset = f_idx - trigger_steps[ci] - 1  # 0-indexed within the RRT phase
            cycle_frames = blend_frames_by_cycle[ci]
            if cycle_frames:
                if offset < len(cycle_frames):
                    return cycle_frames[offset], f"blend cycle {ci}", COLOR_RRT
                # Source's RRT phase extends past what blend recorded → hold.
                return cycle_frames[-1], f"blend cycle {ci} (held)", COLOR_TRIGGER
            # No blend recording for this cycle (rrt_steps_executed=0 path).
            ph = _make_placeholder_frame(blend_w, blend_h, "no blend recording for this cycle")
            return ph, "no blend", COLOR_POLICY
        # Policy phase — blend trajectory diverged at the end of the previous
        # RRT cycle (if any) and what the source video shows now is NOT what
        # the blend would have done. Make that explicit.
        msg = "policy phase (blend diverges here)" if blend_frames_by_cycle else "policy phase"
        ph = _make_placeholder_frame(blend_w, blend_h, msg)
        return ph, "blend idle / diverged", COLOR_POLICY

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        left_label, left_color = _build_phase_label(frame_idx, trigger_steps, rrt_steps_executed)

        if side_by_side_blend_repo_id is not None:
            right_frame, right_label, right_color = _right_pane_for_frame(frame_idx)
            composed = _render_side_by_side_frame(
                left_bgr=frame,
                right_bgr=right_frame,
                left_label=f"SOURCE: {left_label}",
                left_color=left_color,
                right_label=right_label,
                right_color=right_color,
                # No header band on regular frames (cleaner); only during pauses.
                header_label=None,
            )
            writer.write(composed)
        else:
            annotated = frame.copy()
            _draw_label(annotated, left_label, left_color)
            writer.write(annotated)
        n_written += 1

        # If this frame index matches the next trigger, insert the pause
        # AFTER writing it (so the pause sits between this policy frame and
        # the first RRT frame at trigger_steps[i] + 1).
        if cycle_idx < len(trigger_steps) and frame_idx == trigger_steps[cycle_idx]:
            trigger_text = triggers[cycle_idx]
            for _ in range(pause_frames):
                if side_by_side_blend_repo_id is not None:
                    # During the trigger pause, freeze BOTH panes on their
                    # current frame and add a global header announcing the
                    # trigger. Right pane gets the about-to-start blend's
                    # first frame so the viewer sees the divergence preview.
                    cycle_frames = (
                        blend_frames_by_cycle[cycle_idx] if cycle_idx < len(blend_frames_by_cycle) else []
                    )
                    if cycle_frames:
                        right_pause = cycle_frames[0]
                        right_pause_label = f"blend cycle {cycle_idx} (about to start)"
                    else:
                        right_pause = _make_placeholder_frame(blend_w, blend_h, "no blend recording")
                        right_pause_label = "no blend"
                    paused = _render_side_by_side_frame(
                        left_bgr=frame,
                        right_bgr=right_pause,
                        left_label=f"SOURCE: {left_label}",
                        left_color=left_color,
                        right_label=right_pause_label,
                        right_color=COLOR_TRIGGER,
                        header_label=f"TRIGGER: {trigger_text}",
                        header_color=COLOR_TRIGGER,
                    )
                    writer.write(paused)
                else:
                    paused = frame.copy()
                    _draw_label(paused, f"TRIGGER: {trigger_text}", COLOR_TRIGGER)
                    writer.write(paused)
                n_written += 1
            cycle_idx += 1

        frame_idx += 1

    cap.release()
    writer.release()

    # ── re-encode to H.264 + yuv420p for VSCode/browser compatibility ──── #
    # `-c:v libx264 -pix_fmt yuv420p` is the canonical "plays everywhere"
    # combo. `-movflags +faststart` puts the moov atom up front so
    # previewers can start playback without scanning the whole file.
    # `-y` overwrites the destination if it already exists.
    # The H.264 encoder requires even dimensions; pad just in case the
    # render dimensions are odd (they shouldn't be for SplatSim, but
    # forward-compat is cheap).
    print(f"[viz] Re-encoding {tmp_video.name} → H.264 yuv420p for VSCode compat…")
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
        str(output_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    # Clean up the temp regardless of success/failure.
    shutil.rmtree(tmp_dir, ignore_errors=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg re-encode failed (exit {proc.returncode}):\n{proc.stderr.strip()}")

    print(
        f"[viz] Done. Read {frame_idx} source frames; wrote {n_written} total "
        f"(+{n_written - frame_idx} pause frames across {cycle_idx} trigger(s))."
    )
    print(f"[viz] Final video → {output_path}")
    if cycle_idx < len(trigger_steps):
        print(
            f"[viz] WARNING: {len(trigger_steps) - cycle_idx} trigger(s) were past EOF; "
            f"the source video is shorter than the CSV implies. Re-record or check "
            f"max_episodes_rendered.",
            file=sys.stderr,
        )
    return output_path


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "training_dir",
        type=Path,
        help=(
            "Path to one of: (a) a DAgger training output dir "
            "(e.g. outputs/training/diffusion_<...>_ft_dag2 — the "
            "script then looks under <dir>/dagger/interventions/), or "
            "(b) a standalone lerobot-eval output dir "
            "(e.g. outputs/eval/2026-06-16/15-30-00_splatsim_diffusion — "
            "the script looks for intervention_per_scenario.csv + "
            "videos/splatsim_0/ directly under <dir>). Auto-detected via "
            "presence of intervention_per_scenario.csv."
        ),
    )
    p.add_argument(
        "episode_idx",
        type=int,
        help="0-indexed episode (matches eval_episode_<N>.mp4 AND scenario_idx in the CSV).",
    )
    p.add_argument(
        "--pause_frames",
        type=int,
        default=30,
        help="Frames to pause at each trigger moment (default 30 ≈ 1 s @ 30 fps).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path. Default: <video_dir>/eval_episode_<idx>_annotated.mp4 "
        "(or _side_by_side.mp4 when --side_by_side is set).",
    )
    p.add_argument(
        "--side_by_side",
        type=str,
        default=None,
        metavar="BLEND_REPO_ID",
        help=(
            "HF repo id of a blend dataset to display in a right-hand pane "
            "alongside the source video, e.g. "
            "JennyWWW/lever_g0_30ep_03dag_diff_r_dag1_blend010. The blend "
            "dataset's per-episode `source_scenario_idx` field is used to "
            "match the requested scenario's cycles to the right blend "
            "recordings. The right pane shows the blend's frame for each "
            "tick of the source's RRT phase; during source policy phases "
            "the right pane shows a 'blend diverges here' placeholder so "
            "viewers see clearly that the source-policy continuation isn't "
            "what the blend would have done."
        ),
    )
    p.add_argument(
        "--cache_dir",
        type=Path,
        default=Path.home() / ".cache" / "huggingface" / "lerobot",
        help="LeRobot dataset cache root (used to resolve --side_by_side's "
        "BLEND_REPO_ID to a local path). Ignored without --side_by_side.",
    )
    p.add_argument(
        "--blend_camera",
        type=str,
        default="observation.images.base_rgb_stretch",
        help="Image column to use from the blend dataset for the right pane. "
        "Default matches what eval-side videos render.",
    )
    args = p.parse_args()
    annotate(
        training_dir=args.training_dir,
        episode_idx=args.episode_idx,
        pause_frames=args.pause_frames,
        output_path=args.output,
        side_by_side_blend_repo_id=args.side_by_side,
        cache_dir=args.cache_dir,
        blend_camera_col=args.blend_camera,
    )


if __name__ == "__main__":
    main()
