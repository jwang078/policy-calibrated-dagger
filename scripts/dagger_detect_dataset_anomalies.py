#!/usr/bin/env python3
"""Detect anomalies in LeRobot intervention datasets that hurt DAgger training.

Ten anomaly classes, each with a distinct underlying bug. Eight are
default-on; the two expensive/redundant ones are opt-in (see CLI flags).

  CLASS A: TELEPORT — single-frame state jump within an episode that's far
           above the typical RRT-controlled motion in the same episode.
    Cause:
      (1) `request_retry_after_collision` mid-cycle: controller observed
          in_collision during RRT execution, source aborted + replanned +
          (if new q_start in collision) escape-teleported the env. All
          within one controller.tick → wrapper sees no frame_source
          transition → post-teleport frame appends to same recorded episode.
      (2) Padding-window stitch (now fixed): cycle 1 ended too short, then
          a NEW cycle arrived during the padding window → without the
          padding→real_frame _finish_episode call, two cycles concatenated.
    Impact: chunks straddling the teleport produce 1+ rad rel-action values,
    aggregated normalization range stretches 10×+, base data normalized into
    a tiny band → policy learns to output near-zero actions → freezes near
    goal at deployment. The dominant DAgger failure mode in the d30_*
    lineages.

  CLASS B: PADDED_FROZEN — episode is mostly held-still repeated frames,
           indicating a cycle that was too short for `teleop_min_episode_length`
           and got padded by the recorder. Mid-run Δstate is ~0.
    Cause: RRT cycle reached goal in <60 frames → `_finish_episode`'s pad
    branch repeated the last committed frame to hit min_episode_length.
    Impact: ~50 training samples per padded episode where (obs_history,
    action) = (near_goal, hold_position). Trains the diffusion policy's
    score field to "freeze when close to goal" → same near-goal-freeze
    deployment failure as Class A, just from a different mechanism.
    Fix in recorder: `--env.teleop_pad_short_episodes=false` (drops these
    instead of padding). Pre-fix datasets may still contain them.

  CLASS C: TINY_EPISODE — episode shorter than `min_useful_episode_len`
           (default 30 frames). Too short to produce useful training chunks
           at n_obs_steps=2 + n_action_steps=8.
    Cause: `force_episode_split_next_real_frame` edge cases (mid-cycle
    teleport split a cycle into two parts), `teleop_pad_short_episodes=false`
    drops <60 frame cycles cleanly, but some edge paths still emit short
    fragments. Pre-`pad_short_episodes` flag data also has un-padded shorts.
    Impact: each tiny episode yields a handful of chunks at most, but those
    chunks span "near-target/held-still" data that bias the policy toward
    freezing. Less severe than padded_frozen but additive when many exist.

  CLASS D: NAN_OR_INF — any NaN/Inf value in observation.state or action.
    Cause: numerical instability in pybullet/IK (rare), corrupt parquet
    write, or a sim env returning NaN on collision-overlap geometry.
    Impact: silent training corruption — the loss either NaNs (training
    fails) or, worse, the optimizer survives one bad batch and silently
    poisons one direction of the weight space. Always worth deleting.

  CLASS E: LEADING_IDLE — first K frames (default 15) of an episode have
           Δstate-L2 below `frozen_threshold`.
    Cause: leading trim (recorder removes pre-planner-start idle frames)
    didn't fire — recorder buffered idle frames before the RRT chunk
    started playing. The PADDED_FROZEN check measures median Δstate from
    frame 20+, so a 15-frame leading idle slips through.
    Impact: every (obs_history, action) sample drawn from the leading
    region is (held-still, no-motion), reinforcing the diffusion policy's
    "freeze near init pose" mode.

  CLASS F: TRAILING_IDLE — last K frames (default 15) of an episode have
           Δstate-L2 below `frozen_threshold`.
    Cause: trailing trim (recorder removes post-goal-reach idle frames)
    didn't fire — RRT cycle finished but the recorder kept capturing
    held-still frames before the next state transition.
    Impact: same near-goal-freeze poisoning as PADDED_FROZEN, but on long
    episodes the trailing-idle frames don't move the mid-run median enough
    to trigger that class — this class catches the long-episode case.

  CLASS G: JOINT_VELOCITY (opt-in via --check_joint_velocity) — per-axis
           single-step |Δq| above joint_velocity_threshold_rad. Mostly
           redundant with TELEPORT (a single joint moving 0.3 rad → L2 =
           0.3 → triggers TELEPORT). Useful only when you set a stricter
           per-joint threshold than TELEPORT's L2 abs threshold.

  CLASS H: GRIPPER_DRIFT — gripper dim (default -1 = last state column)
           drifts within an episode by more than gripper_drift_tolerance
           (default 0.01).
    Cause: planner / recorder unintentionally touched the gripper. For the
    grip0 base datasets the gripper stays pinned at 0; any motion is a bug.
    Optional second check (--gripper_expected_value=N) flags episodes whose
    mean-gripper is far from N — catches "constant within episode but wrong
    value" cases (e.g., a whole round recorded with gripper stuck at 0.5).

  CLASS I: IMAGE_INTENSITY (opt-in via --check_images) — single-frame
           change in mean pixel intensity for any observation.images.*
           column exceeds image_intensity_threshold (default 50/255).
    Cause: camera dropout, render glitch, scene re-setup mid-episode, or
    a state teleport whose image jump happens to be large. Costly because
    every PNG cell must be decoded — opt-in only.

  CLASS J: TRACKING_ERROR — the MEASURED joint state diverges from the
           COMMANDED action over the episode: max ||state - action|| (over
           arm joints, gripper excluded) exceeds tracking_error_threshold_rad
           (default 0.15). The only class that compares state AGAINST action
           rather than looking at state alone.
    Cause: the RRT/ruckig guidance command is computed open-loop, but the
    robot physically stalls partway (contact with an obstacle, the manipulated
    object, or self) while the commanded target keeps marching along the
    planned path. The gap grows monotonically — clean episodes hold
    ||state-action|| ≈ 0.02-0.04 rad; divergent ones reach 0.7-1.5 rad.
    Impact: the dataset pairs `action` (the command the robot never reached)
    with `observation.state` (where it actually is), off-manifold by up to
    ~1.5 rad — far more corrosive to imitation learning than a single bad
    frame. Catches the divergence JOINT_SPIKE misses: a gradual runaway never
    produces a single-frame Δ above the spike floor, so joint_spike stays
    silent while the episode is thoroughly corrupted.
    Note: auto-skips RELATIVE-encoded action columns (stored per-step deltas
    ≈ 0 → frame-0 ||state-action|| ≈ ||state||); the metric only applies to
    absolute-target action columns (what the RRT recorder writes). See
    TRACKING_ERROR_REL_GUARD_RAD.

Detection metrics:
  TELEPORT       — any frame i where ||state[i+1]-state[i]|| > abs_threshold
                   OR > ratio_threshold × episode-median Δstate.
  PADDED_FROZEN  — median Δstate over frames 20..end is below
                   frozen_threshold. Skips episodes shorter than 30 frames.
  TINY_EPISODE   — len(state) < min_useful_episode_len.
  NAN_OR_INF     — np.isfinite(state).all() is False, or same on action.
  LEADING_IDLE   — max Δstate over frames 1..(edge_idle_frames+1) below
                   frozen_threshold (frame 0→1 excluded so a teleport
                   landing doesn't mask the check).
  TRAILING_IDLE  — max Δstate over the last `edge_idle_frames` is below
                   frozen_threshold.
  JOINT_VELOCITY — any (frame i, joint j) where |state[i+1,j] - state[i,j]|
                   > joint_velocity_threshold_rad.
  GRIPPER_DRIFT  — max(state[:,gripper_dim]) - min(...) > gripper_drift_tol
                   OR |mean(grip) - expected| > tol (if expected set).
  IMAGE_INTENSITY — any frame i, image_key k where
                    |mean_pixel(image[i+1]) - mean_pixel(image[i])|
                    > image_intensity_threshold (0..255 scale).
  TRACKING_ERROR  — max over frames of ||state[i] - action[i]|| (arm joints,
                    gripper dim excluded) > tracking_error_threshold_rad.
                    Skipped when frame-0 error > TRACKING_ERROR_REL_GUARD_RAD
                    (relative-encoded action column).

Most classes detect in observation.state space: observation.state is the
env's GROUND-TRUTH joint config — env teleports show up directly. The
TRACKING_ERROR class is the deliberate exception — it compares state against
the commanded action precisely to catch the case where action does NOT track
state (the robot stalls while the open-loop command runs away).

Usage:
    # Default (auto-scan): every DAgger intervention dataset under
    # ~/.cache/huggingface/lerobot/$HF_USER/. Prints a per-lineage summary
    # so the bad lineage / round pops out at a glance.
    python my_scripts/dagger_detect_dataset_anomalies.py

    # Auto-scan, filtered to one lineage family:
    python my_scripts/dagger_detect_dataset_anomalies.py \\
        --lineage_filter=d30_nopad

    # Scan one explicit dataset, full per-episode detail:
    python my_scripts/dagger_detect_dataset_anomalies.py \\
        --dataset_root ~/.cache/huggingface/lerobot/JennyWWW/lever_g0_d30_nopad_03dag_diff_r_dag4

    # Scan multiple datasets, compact summary only:
    python my_scripts/dagger_detect_dataset_anomalies.py \\
        --dataset_roots ~/.cache/huggingface/lerobot/JennyWWW/lever_g0_d30_nopad_03dag_diff_r_dag* \\
        --quiet

    # Tighter teleport detection (catch sub-0.3 rad lookback jumps too):
    python my_scripts/dagger_detect_dataset_anomalies.py --dataset_root ... \\
        --abs_threshold_rad 0.15 --ratio_threshold 5

    # Print the dagger_cleanup_lineage.sh commands for the anomalous rounds:
    python my_scripts/dagger_detect_dataset_anomalies.py --dataset_root ... \\
        --print_cleanup_command

    # Get a fully-resolved cleanup + resume command (uses the lineage's recorded
    # sidecar — emits the exact orchestrator OR sweep-wrapper invocation that
    # spawned this lineage, plus `--from_round=N --force_restart`). The sweep
    # command is recoverable only when the lineage was originally spawned via
    # dagger_orchestrate_sweep.sh (which now records its argv into the sidecar
    # alongside the orchestrator's own argv).
    python my_scripts/dagger_detect_dataset_anomalies.py --dataset_root ... \\
        --print_cleanup_command \\
        --sidecar outputs/training/<lineage>_ft_dag<N>/dagger/config.json

    # Suppress one or more classes (repeat the flag — multi-select):
    python my_scripts/dagger_detect_dataset_anomalies.py --dataset_root ... \\
        --skip teleport --skip padded   # show only the 4 new classes
    python my_scripts/dagger_detect_dataset_anomalies.py --dataset_root ... \\
        --skip nan                      # everything except NaN/Inf sanity check
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# Canonical DAgger naming lives in my_scripts/dagger_naming.py per CLAUDE.md —
# don't duplicate the `<prefix>_<a|r>_dag<N>` regex here. We import
# parse_dataset_short for forward parsing and reuse its ParsedDatasetName.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dagger_naming import parse_dataset_short  # noqa: E402

# Trailing suffix detector — used purely for display (e.g. distinguishing
# `_dag1` from `_dag1_old` in the per-lineage summary) since parse_dataset_short
# folds suffixes into its `kind` enum (intervention vs merged vs blend) rather
# than exposing them. Cheap to keep alongside the import.
_DATASET_SUFFIX_RE = re.compile(r"^.+_[ar]_dag\d+(?P<suffix>_.*)?$")
_DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "huggingface" / "lerobot"
_DEFAULT_HF_USER = os.environ.get("HF_USER", "JennyWWW")


def _parsed_intervention(name: str):
    """Parse a dataset basename and return its ParsedDatasetName when it
    looks like a raw DAgger intervention dataset. Returns None for base /
    merged / blend / unrecognized names.

    Tolerant fallback: dagger_naming's canonical regex restricts the suffix
    grammar to `blend\\d{3}(_nocoll)?|m` (the orchestrator-managed artifact
    kinds), so ad-hoc backups like `_dag1_old` and `--retrain_suffix`
    variants like `_dag5_v2` parse as kind='base'. Those are real data we
    want to surface in the lineage scan, so when canonical parse falls back
    to 'base' BUT the name still matches the looser `*_[ar]_dag<N>_<suffix>`
    shape (suffix not blend/m), we hand-construct an intervention-typed
    ParsedDatasetName from the loose match. Keeps dagger_naming's strict
    contract intact for everyone else while letting this script see the
    full on-disk reality.
    """
    parsed = parse_dataset_short(name)
    if parsed.kind == "intervention":
        return parsed
    if parsed.kind != "base":
        # merged/blend → genuinely derived datasets; don't surface them.
        return None
    # Try the loose-suffix fallback. parse_dataset_short failed → see if the
    # name matches `<prefix>_<a|r>_dag<N>_<suffix>` for any non-managed suffix.
    m = _LOOSE_INTERVENTION_RE.match(name)
    if not m:
        return None
    # Build a ParsedDatasetName via the canonical dataclass (same fields).
    from dagger_naming import ParsedDatasetName

    return ParsedDatasetName(
        kind="intervention",
        name=name,
        prefix=m.group("prefix"),
        infix=m.group("infix"),
        round=int(m.group("round")),
        blend_pct=None,
        is_nocoll=False,
    )


def _suffix_of(name: str) -> str:
    """Return the trailing `_<suffix>` after `_dag<N>` (e.g. `_old`, `_v2`),
    or empty string if there isn't one. parse_dataset_short only models
    intervention/merged/blend kinds; the dag-number-trailing freeform suffix
    (used by manual backups and `--retrain_suffix`) isn't captured there."""
    m = _DATASET_SUFFIX_RE.match(name)
    return (m.group("suffix") or "") if m else ""


# Loose-grammar fallback for `_parsed_intervention`: same shape as canonical
# parser but accepts ANY suffix after `_dag<N>` (so `_old`/`_v2` parse). Caller
# is responsible for first checking parse_dataset_short — this is only used
# when that returned kind='base' (i.e. the canonical regex didn't match).
_LOOSE_INTERVENTION_RE = re.compile(r"^(?P<prefix>.+)_(?P<infix>[ar])_dag(?P<round>\d+)(?:_.+)?$")


# Single-frame Δstate over 0.3 rad in joint-L2 is past what ruckig-bounded
# RRT produces at 30 Hz on a UR5e (typical max ~0.15 rad/frame). 0.5+ rad
# is unambiguously a teleport. Default 0.3 catches small lookback teleports
# too while staying above legitimate RRT motion.
DEFAULT_ABS_THRESHOLD_RAD = 0.3
DEFAULT_RATIO_THRESHOLD = 10.0
# Minimum absolute Δstate (rad) required for the ratio-vs-median teleport
# check to fire. Guards against the false-positive case where a PADDED
# episode (last-frame-repeat to hit min_episode_length) has most frames
# frozen at Δs=0, dragging the median down to ~1e-6 — at which point the
# ratio check fires on ANY real motion (e.g., normal ruckig ramp at
# Δs=0.001 rad is 1000× the artificial median). 0.05 rad/tick (=1.5 rad/s
# at 30Hz) sits well above normal ruckig motion (~0.03 rad/tick max from
# rest) but well below any genuine teleport (typically 0.1+ rad/tick).
# The absolute-threshold check (abs_thr) STILL fires on real teleports
# regardless of this floor — it just gates the ratio-based check.
DEFAULT_RATIO_MIN_ABS_RAD = 0.05

# An episode whose mid-run frames have effectively zero motion (less than
# 0.0001 rad/frame in joint-L2) was almost certainly padded by repeating the
# last committed frame. Conservative: real RRT motion is 0.02-0.05 rad/frame
# typical, so even very slow motion is well above this.
DEFAULT_FROZEN_THRESHOLD_RAD = 1e-4

# Frame index at which to start sampling "mid-run" median. Skips the initial
# accel-from-rest ramp (ruckig start_vel=0 produces small deltas in the
# first ~5-10 frames). Episodes shorter than this can't be classed as padded.
MIDRUN_START_FRAME = 20

# Min episode length to consider for padded-frozen detection. Below this,
# the mid-run sample is too small to be meaningful — skip.
MIN_LEN_FOR_PADDED_CHECK = 30

# An episode shorter than this is too small to produce meaningful training
# samples (the diffusion policy at n_obs_steps=2 + n_action_steps=8 needs
# at least ~10 frames per chunk, so <30 frames yields very few chunks and
# they're all very short-range). force_episode_split / mid-cycle teleport
# edge cases can sneak through here even when teleop_min_episode_length is
# set; this catches those.
DEFAULT_MIN_USEFUL_EPISODE_LEN = 30

# Number of frames at the start (or end) of an episode whose Δstate-L2 must
# be below frozen_threshold for the leading_idle / trailing_idle classes to
# fire. 15 ≈ 0.5 sec at 30 Hz — long enough to be clearly a recorder-trim
# bug rather than ruckig's natural start_vel=0 acceleration ramp (~5 frames).
DEFAULT_EDGE_IDLE_FRAMES = 15

# Per-joint single-step Δ threshold for the joint_velocity class. Slightly
# below TELEPORT's L2 threshold because per-joint is stricter (one axis at
# 0.3 rad would trigger TELEPORT's L2 too; a per-joint threshold of 0.3 here
# would be exactly redundant). 0.2 catches sustained-too-fast motion that
# the L2 might miss when other joints are still.
DEFAULT_JOINT_VELOCITY_THRESHOLD = 0.2

# gripper_drift: max Δgripper allowed within an episode. The grip0 base
# dataset has gripper pinned at 0.0 — RRT shouldn't touch it, so any
# nonzero motion (above noise floor 0.01) is a bug. Set ~10× higher than
# float32 quantization noise so we don't false-positive on rounding.
DEFAULT_GRIPPER_DRIFT_TOLERANCE = 0.01

# image_intensity_jump: max single-frame Δmean_pixel_intensity (0-255 scale)
# before flagging. The arm naturally occludes/reveals background so frame-
# to-frame Δmean of 5-20 is typical; >50 is a render glitch / camera
# dropout / wholesale teleport-induced re-render.
DEFAULT_IMAGE_INTENSITY_THRESHOLD = 50.0

# joint_spike: per-joint single-frame ratio-vs-median check. Catches
# single-joint stutters that the L2-norm TELEPORT check can't see — e.g.
# wrist_3 lurches -0.05 rad in one frame while the other 5 joints stay
# smooth at ~0.015 rad/frame; L2 sums to 0.06 (< abs_thr 0.3, ~2.5×
# L2-median) but the per-joint move IS 5× THAT joint's own median, and
# visually shows as a noticeable stutter. Two thresholds (same gate
# pattern as TELEPORT's ratio + ratio_min_abs floor):
#   * RATIO — per-joint Δ must exceed this multiple of THAT joint's
#     own median |Δ|. Default 5× — single-frame deviations 5× above the
#     joint's normal step are unmistakably anomalous (smooth motion
#     varies by ~30% step-to-step, not 5×).
#   * MIN_ABS — per-joint Δ must ALSO exceed this absolute floor in
#     rad/frame. Default 0.02 rad/frame (≈0.6 rad/s) — above normal
#     ruckig motion (~0.015 rad/frame steady-state on grasp tasks),
#     below the typical stutter magnitude (~0.03-0.05 rad/frame).
#     Without the floor, a near-stationary joint (median ≈ 0) trips
#     the ratio check on any tiny motion (same artifact the TELEPORT
#     class's ratio_min_abs floor guards against).
DEFAULT_JOINT_SPIKE_RATIO = 5.0
DEFAULT_JOINT_SPIKE_MIN_ABS_RAD = 0.02

# tracking_error: max ||state - action|| (arm joints, gripper excluded) above
# this flags an episode. Clean RRT execution tracks within ~0.02-0.04 rad;
# open-loop command runaway (robot stalls, command marches on) reaches
# 0.7-1.5 rad. 0.15 sits well clear of the clean baseline and of legitimate
# transient lag, while catching every divergence observed in the d30 lineages.
DEFAULT_TRACKING_ERROR_THRESHOLD_RAD = 0.15
# Frame-0 ||state - action|| above this means the action column stores
# RELATIVE actions (per-step deltas ≈ 0, so the metric reads ≈ ||state||
# everywhere — a false positive on every frame). Absolute-target columns (what
# the RRT recorder writes) start frame-0 well-tracked (state[0] ≈ action[0]),
# so a large frame-0 error is a reliable relative-encoding signal → skip.
TRACKING_ERROR_REL_GUARD_RAD = 0.5

# ── Class L: norm_clip (default OFF — needs an action-delta stats sidecar) ──
# The diffusion policy normalizes ACTION via MIN_MAX
# (normalize_processor.py: norm = 2*(x-min)/(max-min) - 1 → clamped to [-1,1]).
# Under --action_format=rel the normalized quantity is the relative action
# chunk delta_k = action[t+k] - state[t] over an H-step horizon (arm joints;
# gripper masked — see relative_action_processor.to_relative_actions), whose
# stats live in stats_rel8.json. A delta outside the sidecar [min,max]
# normalizes OUTSIDE [-1,1] and gets CLIPPED at train time → a corrupted
# label the policy can never reproduce. This is the aggregated-norm footgun
# from CLAUDE.md made measurable: pass the AGGREGATED sidecar (min-of-mins /
# max-of-maxes across base + every round, what weighted-sampling training
# actually normalizes with) and a round whose own deltas exceed another
# source's range will show clipping here. Flag an episode when the fraction
# of clipped normalized values exceeds this floor.
DEFAULT_NORM_CLIP_FRAC_THRESHOLD = 0.01
# Relative-action chunk horizon. stats_rel8.json ⇒ 8. Each anchor frame t
# contributes deltas action[t : t+H] - state[t]; far-horizon steps have the
# largest deltas, so they clip first — the single-step (H=1) view understates.
DEFAULT_REL_HORIZON = 8


def load_action_delta_stats(stats_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Read action min/max (the relative-action delta range) from a
    stats_rel8.json-style sidecar. Returns (min, max) float64 arrays over the
    full action dim (gripper included; scan_episode selects arm dims), or None
    if the file is missing / malformed / lacks min+max."""
    try:
        d = json.loads(Path(stats_path).read_text())
    except (OSError, json.JSONDecodeError):
        return None
    act = d.get("action")
    if not isinstance(act, dict) or "min" not in act or "max" not in act:
        return None
    amin = np.asarray(act["min"], dtype=np.float64)
    amax = np.asarray(act["max"], dtype=np.float64)
    if amin.shape != amax.shape or amin.ndim != 1:
        return None
    return amin, amax


def _default_rel_stats_path(dataset_root: Path) -> Path:
    """Canonical on-disk location of a dataset's rel-action stats sidecar:
    <repo>/outputs/dataset_stats/<dataset_name>/stats_rel8.json. Mirrors where
    compute_relative_stats.sh writes them."""
    repo_root = Path(__file__).resolve().parent.parent
    return repo_root / "outputs" / "dataset_stats" / dataset_root.name / "stats_rel8.json"


def scan_episode(
    state: np.ndarray,
    abs_thr: float,
    ratio_thr: float,
    ratio_min_abs: float,
    frozen_thr: float,
    min_useful_len: int,
    edge_idle_frames: int,
    joint_velocity_thr: float,
    joint_spike_ratio: float,
    joint_spike_min_abs: float,
    tracking_error_thr: float,
    gripper_dim: int,
    gripper_drift_tol: float,
    gripper_expected_value: float | None,
    image_intensity_thr: float,
    action: np.ndarray | None,
    image_intensities: dict[str, np.ndarray] | None,
    detect_teleport: bool,
    detect_padded: bool,
    detect_tiny: bool,
    detect_nan: bool,
    detect_leading_idle: bool,
    detect_trailing_idle: bool,
    detect_joint_velocity: bool,
    detect_joint_spike: bool,
    detect_tracking_error: bool,
    detect_gripper: bool,
    detect_image_intensity: bool,
    detect_norm_clip: bool = False,
    action_delta_min: np.ndarray | None = None,
    action_delta_max: np.ndarray | None = None,
    rel_horizon: int = DEFAULT_REL_HORIZON,
    norm_clip_frac_thr: float = DEFAULT_NORM_CLIP_FRAC_THRESHOLD,
) -> dict:
    """Return per-episode anomaly findings, one dict key per detection class.

    Class summary (all optional, all toggleable via the detect_* args):
      * 'teleports':         list of (frame_idx, delta_rad, ratio_to_median).
                             Single-frame state jump beyond abs_thr OR ratio_thr.
      * 'padded_frozen':     dict if mid-run (frame 20+) Δstate median is below
                             frozen_thr → recorder padded a too-short cycle.
      * 'tiny_episode':      dict if len(state) < min_useful_len.
      * 'nan_or_inf':        dict if any NaN/Inf in state or action.
      * 'leading_idle':      dict if first `edge_idle_frames` of Δstate are
                             all below frozen_thr.
      * 'trailing_idle':     dict if last `edge_idle_frames` of Δstate are
                             all below frozen_thr.
      * 'joint_velocity':    list of (frame_idx, joint_index, delta_rad). Any
                             per-joint single-step Δ above joint_velocity_thr
                             (default 0.2 rad/frame). Mostly redundant with
                             TELEPORT but per-joint catches single-axis
                             over-speed that L2 norm sometimes misses.
      * 'gripper_drift':     dict if gripper dim's max-Δ within episode is
                             above gripper_drift_tol (default 0.01). The
                             grip0 base datasets pin gripper at 0; anything
                             that moves it is an RRT / recorder bug.
                             If gripper_expected_value is set, also flags
                             episodes whose mean-gripper is far from it.
      * 'image_intensity':   list of (frame_idx, image_key, delta_intensity).
                             Single-frame Δmean_pixel above image_intensity_thr
                             (default 50/255). Catches camera dropouts or
                             render glitches that don't manifest in state.
                             Requires image_intensities to be precomputed by
                             the caller (involves PNG decode → opt-in).
    """
    out: dict = {
        "teleports": [],
        "padded_frozen": None,
        "tiny_episode": None,
        "nan_or_inf": None,
        "leading_idle": None,
        "trailing_idle": None,
        "joint_velocity": [],
        "joint_spike": [],
        "tracking_error": None,
        "norm_clip": None,
        "gripper_drift": None,
        "image_intensity": [],
    }
    if len(state) < 3:
        # Too short to do anything meaningful — at minimum, flag as tiny if
        # the user asked. NaN check still runs (state of 1-2 frames could
        # still have NaN values worth flagging).
        if detect_nan and not np.all(np.isfinite(state)):
            out["nan_or_inf"] = {"location": "state", "n_frames": int(len(state))}
        if detect_tiny and len(state) < min_useful_len:
            out["tiny_episode"] = {"episode_length": int(len(state)), "threshold": min_useful_len}
        return out

    deltas = np.linalg.norm(np.diff(state, axis=0), axis=1)
    median = float(np.median(deltas))
    safe_median = max(median, 1e-6)

    # ── Class A: teleport ──
    # Two-condition check: abs_thr catches genuine teleports outright
    # (typical Δ > 0.3 rad single-frame jump). The ratio check catches
    # unusually-large-but-sub-abs jumps — but ONLY when the absolute Δ
    # also crosses ratio_min_abs. Without that floor, padded episodes
    # (most frames frozen → median ≈ 0 → safe_median = 1e-6) trip the
    # ratio condition on ANY real motion (e.g., normal ruckig ramp at
    # Δ=0.001 rad reads as 1000× the artificial median). The abs check
    # is unaffected and still fires on real teleports regardless.
    if detect_teleport:
        for i, d in enumerate(deltas):
            d = float(d)
            ratio = d / safe_median
            if d > abs_thr or (ratio > ratio_thr and d > ratio_min_abs):
                # delta[i] = state[i+1] - state[i]; frame index of the
                # POST-teleport state is i+1.
                out["teleports"].append((i + 1, d, ratio))

    # ── Class B: padded_frozen ──
    if detect_padded and len(state) >= MIN_LEN_FOR_PADDED_CHECK:
        midrun_slice = deltas[MIDRUN_START_FRAME:]
        if len(midrun_slice) > 0:
            midrun_median = float(np.median(midrun_slice))
            if midrun_median < frozen_thr:
                out["padded_frozen"] = {
                    "midrun_median": midrun_median,
                    "midrun_start": MIDRUN_START_FRAME,
                    "episode_length": int(len(state)),
                }

    # ── Class C: tiny_episode ──
    if detect_tiny and len(state) < min_useful_len:
        out["tiny_episode"] = {
            "episode_length": int(len(state)),
            "threshold": min_useful_len,
        }

    # ── Class D: nan_or_inf ──
    if detect_nan:
        if not np.all(np.isfinite(state)):
            out["nan_or_inf"] = {"location": "state", "n_frames": int(len(state))}
        elif action is not None and not np.all(np.isfinite(action)):
            out["nan_or_inf"] = {"location": "action", "n_frames": int(len(state))}

    # ── Class E: leading_idle ──
    # Skip the very first delta (state[0]→state[1]) because a lookback
    # teleport lands there legitimately on rrt-onset; we want to catch
    # SUSTAINED zero motion at the start, not a single teleport landing.
    if detect_leading_idle and len(deltas) >= edge_idle_frames + 1:
        leading = deltas[1 : edge_idle_frames + 1]
        leading_max = float(np.max(leading))
        if leading_max < frozen_thr:
            out["leading_idle"] = {
                "n_idle_frames": int(edge_idle_frames),
                "leading_max_delta": leading_max,
            }

    # ── Class F: trailing_idle ──
    if detect_trailing_idle and len(deltas) >= edge_idle_frames:
        trailing = deltas[-edge_idle_frames:]
        trailing_max = float(np.max(trailing))
        if trailing_max < frozen_thr:
            out["trailing_idle"] = {
                "n_idle_frames": int(edge_idle_frames),
                "trailing_max_delta": trailing_max,
            }

    # ── Class G: joint_velocity (per-joint, default off — see CLI doc) ──
    # state.shape == (T, dof); per-joint Δ = np.diff axis=0. Compare each
    # element to joint_velocity_thr. Mostly redundant with TELEPORT's L2
    # check (a single joint moving 0.3 rad → L2 = 0.3 → triggers there too)
    # but useful when threshold is set below TELEPORT's abs_thr to catch
    # sustained but per-frame-modest over-speed that L2 misses.
    if detect_joint_velocity:
        per_joint = np.abs(np.diff(state, axis=0))
        # np.where returns (rows, cols) over the (T-1, dof) array.
        rows, cols = np.where(per_joint > joint_velocity_thr)
        for i, j in zip(rows, cols):
            out["joint_velocity"].append((int(i) + 1, int(j), float(per_joint[i, j])))

    # ── Class G2: joint_spike (per-joint ratio-vs-median, default ON) ──
    # Catches single-frame stutters that L2-TELEPORT misses: e.g. wrist_3
    # lurches -0.05 rad in one frame while the other 5 joints stay smooth
    # at ~0.015 rad — L2 sums to 0.06 (< abs_thr 0.3, only 2.5× L2-median)
    # but the per-joint move IS 5× THAT joint's own median, and visually
    # shows as a noticeable stutter. Two-gate check (ratio AND min_abs,
    # same pattern as TELEPORT's ratio_min_abs floor): the ratio catches
    # the per-joint anomaly shape; the abs floor prevents false positives
    # on joints that barely move at all (median ≈ 0).
    if detect_joint_spike:
        per_joint = np.abs(np.diff(state, axis=0))  # (T-1, dof)
        # Per-joint median (axis=0 reduces over time). For a joint that
        # never moves median ≈ 0; the floor below handles that.
        joint_medians = np.median(per_joint, axis=0)
        safe_joint_medians = np.maximum(joint_medians, 1e-6)
        joint_ratios = per_joint / safe_joint_medians
        # Both conditions: per-joint ratio AND absolute floor.
        hit_mask = (joint_ratios > joint_spike_ratio) & (per_joint > joint_spike_min_abs)
        rows, cols = np.where(hit_mask)
        for i, j in zip(rows, cols):
            out["joint_spike"].append((int(i) + 1, int(j), float(per_joint[i, j]), float(joint_ratios[i, j])))

    # ── Class J: tracking_error (measured state vs commanded action) ──
    # The only class that reads `action`: every other class looks at state
    # alone. Catches open-loop command runaway — the robot stalls (contact /
    # self-collision / object), the ruckig command keeps marching along the
    # planned path, and ||state - action|| grows monotonically from ~0.03 rad
    # to ~1 rad. JOINT_SPIKE misses this: a gradual drift never produces a
    # single-frame Δ above the spike floor, so it stays silent while the
    # (obs, action) pairs go badly off-manifold.
    if detect_tracking_error and action is not None and action.shape == state.shape:
        # Exclude the gripper dim: its action encoding can differ from state,
        # and the grip0 lineages pin it at 0 (a constant offset there would
        # falsely inflate the norm). Negative gripper_dim wraps.
        gdim = gripper_dim if gripper_dim >= 0 else state.shape[1] + gripper_dim
        arm_cols = [c for c in range(state.shape[1]) if c != gdim]
        if arm_cols:
            err = np.linalg.norm(state[:, arm_cols] - action[:, arm_cols], axis=1)
            # Relative-encoded action guard: an absolute-target column starts
            # frame-0 well-tracked (state[0] ≈ action[0]); a large frame-0
            # error means the column holds per-step deltas → metric N/A, skip.
            if float(err[0]) <= TRACKING_ERROR_REL_GUARD_RAD:
                max_i = int(np.argmax(err))
                max_err = float(err[max_i])
                if max_err > tracking_error_thr:
                    # argmax on a boolean array → index of the FIRST True.
                    first_i = int(np.argmax(err > tracking_error_thr))
                    out["tracking_error"] = {
                        "max_error_rad": max_err,
                        "max_error_frame": max_i,
                        "first_exceed_frame": first_i,
                        "end_error_rad": float(err[-1]),
                        "threshold": tracking_error_thr,
                    }

    # ── Class L: norm_clip (rel-action delta outside the MIN_MAX range) ──
    # The diffusion ACTION normalizer is MIN_MAX: norm = 2*(d-min)/(max-min)-1,
    # then clamped to [-1,1]. With --action_format=rel the normalized quantity
    # is the relative chunk delta d_k = action[t+k] - state[t] over an H-step
    # horizon (arm joints; gripper masked). Any d outside [min,max] (the stats
    # passed in — ideally the AGGREGATED range training actually uses) maps
    # outside [-1,1] and is clipped → a label the policy can't reproduce. We
    # reconstruct the chunk deltas, normalize, and flag the episode when too
    # large a fraction clip. Catches the aggregated-norm compression footgun
    # that none of the state-only classes can see.
    if (
        detect_norm_clip
        and action is not None
        and action.shape == state.shape
        and action_delta_min is not None
        and action_delta_max is not None
    ):
        gdim = gripper_dim if gripper_dim >= 0 else state.shape[1] + gripper_dim
        arm = [c for c in range(state.shape[1]) if c != gdim]
        amin = np.asarray(action_delta_min, dtype=np.float64)
        amax = np.asarray(action_delta_max, dtype=np.float64)
        if amin.shape[0] >= state.shape[1] and arm:
            amin_a = amin[arm]
            amax_a = amax[arm]
            denom = np.where((amax_a - amin_a) == 0.0, 1.0, amax_a - amin_a)
            T = state.shape[0]
            H = max(1, int(rel_horizon))
            n_total = 0
            n_clipped = 0
            max_abs_norm = 0.0
            for t in range(T):
                kmax = min(H, T - t)
                # chunk deltas anchored at state[t]: action[t:t+kmax] - state[t]
                d = action[t : t + kmax][:, arm] - state[t, arm]  # (kmax, n_arm)
                norm = 2.0 * (d - amin_a) / denom - 1.0
                n_total += norm.size
                n_clipped += int(np.count_nonzero(np.abs(norm) > 1.0))
                if norm.size:
                    max_abs_norm = max(max_abs_norm, float(np.max(np.abs(norm))))
            if n_total > 0:
                frac = n_clipped / n_total
                if frac > norm_clip_frac_thr:
                    out["norm_clip"] = {
                        "clip_fraction": frac,
                        "n_clipped": n_clipped,
                        "n_total": n_total,
                        "max_abs_norm": max_abs_norm,
                        "rel_horizon": H,
                        "threshold": norm_clip_frac_thr,
                    }

    # ── Class H: gripper_drift ──
    # state[:, gripper_dim] is the gripper position. For grip0 base datasets
    # this stays at 0 throughout; RRT shouldn't touch it. If it moves more
    # than gripper_drift_tol within the episode, flag. Optional second check
    # against gripper_expected_value catches "constant but wrong value" cases
    # (e.g. an entire round recorded with gripper stuck at 0.5).
    if detect_gripper and state.shape[1] > 0:
        # Negative indices wrap (gripper_dim=-1 = last column).
        dim = gripper_dim if gripper_dim >= 0 else state.shape[1] + gripper_dim
        if 0 <= dim < state.shape[1]:
            grip = state[:, dim]
            grip_min, grip_max = float(np.min(grip)), float(np.max(grip))
            drift = grip_max - grip_min
            triggered = drift > gripper_drift_tol
            mean_off = None
            if gripper_expected_value is not None:
                mean_val = float(np.mean(grip))
                mean_off = abs(mean_val - gripper_expected_value)
                if mean_off > gripper_drift_tol:
                    triggered = True
            if triggered:
                out["gripper_drift"] = {
                    "gripper_dim": dim,
                    "drift_range": drift,
                    "grip_min": grip_min,
                    "grip_max": grip_max,
                    "expected_value": gripper_expected_value,
                    "mean_offset_from_expected": mean_off,
                    "tolerance": gripper_drift_tol,
                }

    # ── Class I: image_intensity ──
    # image_intensities is {camera_key: (T,) mean-pixel-intensity array},
    # precomputed by the caller (PNG decode is expensive — keep it out of
    # the hot path here). Flag frames where Δintensity exceeds the threshold;
    # one row per (frame, camera) hit.
    if detect_image_intensity and image_intensities:
        for cam_key, intensity in image_intensities.items():
            if len(intensity) < 2:
                continue
            d_int = np.abs(np.diff(intensity))
            spikes = np.where(d_int > image_intensity_thr)[0]
            for i in spikes:
                out["image_intensity"].append((int(i) + 1, cam_key, float(d_int[i])))

    return out


def _image_mean_intensities(g_image_col) -> np.ndarray:
    """Decode the dict-wrapped PNG bytes in a Series-like column to a (T,)
    array of mean pixel intensities (0..255). Uses PIL for decode + numpy
    for the mean. Expensive — called only when image-intensity detection
    is on. Returns a length-0 array on any decode failure so the caller
    can silently skip the column."""
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        return np.array([])
    import io

    out = np.empty(len(g_image_col), dtype=np.float32)
    for i, cell in enumerate(g_image_col):
        # HF Image feature wraps PNG bytes as {'bytes': b'…', 'path': None}.
        # Tolerate either dict form or raw bytes.
        if isinstance(cell, dict):
            data = cell.get("bytes")
        elif isinstance(cell, (bytes, bytearray)):
            data = cell
        else:
            return np.array([])
        if not data:
            return np.array([])
        try:
            with Image.open(io.BytesIO(data)) as im:
                out[i] = float(np.asarray(im, dtype=np.uint8).mean())
        except Exception:
            return np.array([])
    return out


def scan_dataset(
    dataset_root: Path,
    abs_thr: float,
    ratio_thr: float,
    ratio_min_abs: float,
    frozen_thr: float,
    min_useful_len: int,
    edge_idle_frames: int,
    joint_velocity_thr: float,
    joint_spike_ratio: float,
    joint_spike_min_abs: float,
    tracking_error_thr: float,
    gripper_dim: int,
    gripper_drift_tol: float,
    gripper_expected_value: float | None,
    image_intensity_thr: float,
    detect_teleport: bool,
    detect_padded: bool,
    detect_tiny: bool,
    detect_nan: bool,
    detect_leading_idle: bool,
    detect_trailing_idle: bool,
    detect_joint_velocity: bool,
    detect_joint_spike: bool,
    detect_tracking_error: bool,
    detect_gripper: bool,
    detect_image_intensity: bool,
    detect_norm_clip: bool = False,
    action_delta_min: np.ndarray | None = None,
    action_delta_max: np.ndarray | None = None,
    rel_horizon: int = DEFAULT_REL_HORIZON,
    norm_clip_frac_thr: float = DEFAULT_NORM_CLIP_FRAC_THRESHOLD,
) -> dict:
    """Walk every parquet under dataset_root, return per-class anomaly summary."""
    chunk_dirs = sorted((dataset_root / "data").glob("chunk-*"))
    if not chunk_dirs:
        if dataset_root.name.startswith("chunk-"):
            chunk_dirs = [dataset_root]
        else:
            raise FileNotFoundError(
                f"No data/chunk-* dirs found under {dataset_root}. "
                f"Pass either the dataset root (containing data/, meta/) or "
                f"a specific chunk-NNN dir."
            )

    teleports: dict[int, list[tuple[int, float, float]]] = {}
    padded: dict[int, dict] = {}
    tiny: dict[int, dict] = {}
    nan: dict[int, dict] = {}
    leading_idle: dict[int, dict] = {}
    trailing_idle: dict[int, dict] = {}
    joint_velocity: dict[int, list[tuple[int, int, float]]] = {}
    joint_spike: dict[int, list[tuple[int, int, float, float]]] = {}
    tracking_error: dict[int, dict] = {}
    norm_clip: dict[int, dict] = {}
    gripper_drift: dict[int, dict] = {}
    image_intensity: dict[int, list[tuple[int, str, float]]] = {}
    ep_lengths: dict[int, int] = {}

    # Read only the columns each enabled class needs. Action column reads
    # are non-trivially expensive (object dtype + np.stack); image columns
    # are MUCH more expensive (object → dict → PNG decode → np.mean), so
    # only opt into them when their class is actually on.
    columns = ["episode_index", "observation.state"]
    if detect_nan or detect_tracking_error or detect_norm_clip:
        columns.append("action")
    image_columns: list[str] = []
    if detect_image_intensity:
        # Discover image columns by peeking at the first parquet's schema.
        # All chunks in a dataset share the same schema, so one peek suffices.
        first_parquet = next(iter(sorted(chunk_dirs[0].glob("file-*.parquet"))), None)
        if first_parquet is not None:
            try:
                head = pd.read_parquet(first_parquet, columns=None).head(0)
                image_columns = [c for c in head.columns if c.startswith("observation.images.")]
                columns.extend(image_columns)
            except Exception:
                # If schema peek fails, fall back to scanning without images.
                pass

    corrupt_parquets: list[tuple[Path, str]] = []
    for chunk_dir in chunk_dirs:
        for parquet in sorted(chunk_dir.glob("file-*.parquet")):
            # Corrupt parquets are themselves an anomaly worth surfacing —
            # but blowing up the whole multi-dataset scan when ONE file is
            # bad is the wrong UX. Catch + record + skip; surface in the
            # summary so the user can investigate manually.
            try:
                df = pd.read_parquet(parquet, columns=columns)
            except Exception as e:
                corrupt_parquets.append((parquet, f"{type(e).__name__}: {e}"))
                continue
            for ep_id, g in df.groupby("episode_index", sort=False):
                state = np.stack(g["observation.state"].values).astype(np.float32)
                action = None
                if (detect_nan or detect_tracking_error or detect_norm_clip) and "action" in g.columns:
                    action = np.stack(g["action"].values).astype(np.float32)
                # Precompute per-camera mean-intensity arrays for this
                # episode. Empty dict if image detection is off (cheap exit).
                image_intensities = None
                if detect_image_intensity and image_columns:
                    image_intensities = {}
                    for cam in image_columns:
                        if cam in g.columns:
                            arr = _image_mean_intensities(g[cam].values)
                            if len(arr) > 0:
                                image_intensities[cam] = arr
                ep_lengths[int(ep_id)] = int(len(state))
                hits = scan_episode(
                    state,
                    abs_thr,
                    ratio_thr,
                    ratio_min_abs,
                    frozen_thr,
                    min_useful_len,
                    edge_idle_frames,
                    joint_velocity_thr,
                    joint_spike_ratio,
                    joint_spike_min_abs,
                    tracking_error_thr,
                    gripper_dim,
                    gripper_drift_tol,
                    gripper_expected_value,
                    image_intensity_thr,
                    action,
                    image_intensities,
                    detect_teleport,
                    detect_padded,
                    detect_tiny,
                    detect_nan,
                    detect_leading_idle,
                    detect_trailing_idle,
                    detect_joint_velocity,
                    detect_joint_spike,
                    detect_tracking_error,
                    detect_gripper,
                    detect_image_intensity,
                    detect_norm_clip=detect_norm_clip,
                    action_delta_min=action_delta_min,
                    action_delta_max=action_delta_max,
                    rel_horizon=rel_horizon,
                    norm_clip_frac_thr=norm_clip_frac_thr,
                )
                if hits["teleports"]:
                    teleports[int(ep_id)] = hits["teleports"]
                if hits["padded_frozen"] is not None:
                    padded[int(ep_id)] = hits["padded_frozen"]
                if hits["tiny_episode"] is not None:
                    tiny[int(ep_id)] = hits["tiny_episode"]
                if hits["nan_or_inf"] is not None:
                    nan[int(ep_id)] = hits["nan_or_inf"]
                if hits["leading_idle"] is not None:
                    leading_idle[int(ep_id)] = hits["leading_idle"]
                if hits["trailing_idle"] is not None:
                    trailing_idle[int(ep_id)] = hits["trailing_idle"]
                if hits["joint_velocity"]:
                    joint_velocity[int(ep_id)] = hits["joint_velocity"]
                if hits["joint_spike"]:
                    joint_spike[int(ep_id)] = hits["joint_spike"]
                if hits["tracking_error"] is not None:
                    tracking_error[int(ep_id)] = hits["tracking_error"]
                if hits["norm_clip"] is not None:
                    norm_clip[int(ep_id)] = hits["norm_clip"]
                if hits["gripper_drift"] is not None:
                    gripper_drift[int(ep_id)] = hits["gripper_drift"]
                if hits["image_intensity"]:
                    image_intensity[int(ep_id)] = hits["image_intensity"]

    return {
        "dataset_root": dataset_root,
        "n_episodes": len(ep_lengths),
        "teleports": teleports,
        "padded": padded,
        "tiny": tiny,
        "nan": nan,
        "leading_idle": leading_idle,
        "trailing_idle": trailing_idle,
        "joint_velocity": joint_velocity,
        "joint_spike": joint_spike,
        "tracking_error": tracking_error,
        "norm_clip": norm_clip,
        "gripper_drift": gripper_drift,
        "image_intensity": image_intensity,
        "ep_lengths": ep_lengths,
        # Dataset-level (not per-episode) anomaly: corrupt parquet files
        # that couldn't even be opened to be scanned. List of
        # (parquet_path, error_str) tuples; empty when everything read clean.
        "corrupt_parquets": corrupt_parquets,
    }


# Tuple of (result_key, display_name, short_label) — drives every "iterate
# the classes" loop downstream (print_report, summary, totals, --skip,
# delete-set union). Adding a new class = appending one row here + adding
# the detection in scan_episode + plumbing the detect_* arg through main.
ANOMALY_CLASSES = [
    ("teleports", "TELEPORTS", "teleport"),
    ("padded", "PADDED_FROZEN", "padded"),
    ("tiny", "TINY_EPISODE", "tiny"),
    ("nan", "NAN_OR_INF", "nan"),
    ("leading_idle", "LEADING_IDLE", "leading_idle"),
    ("trailing_idle", "TRAILING_IDLE", "trailing_idle"),
    ("joint_velocity", "JOINT_VELOCITY", "joint_velocity"),
    ("joint_spike", "JOINT_SPIKE", "joint_spike"),
    ("tracking_error", "TRACKING_ERROR", "tracking_error"),
    ("norm_clip", "NORM_CLIP", "norm_clip"),
    ("gripper_drift", "GRIPPER_DRIFT", "gripper_drift"),
    ("image_intensity", "IMAGE_INTENSITY", "image_intensity"),
]


def _affected_episodes(result: dict) -> set[int]:
    """Union of episode IDs that hit ANY anomaly class in this result."""
    out: set[int] = set()
    for key, _, _ in ANOMALY_CLASSES:
        out.update(result.get(key, {}).keys())
    return out


def _format_episode_detail(class_key: str, ep_id: int, ep_len: int, info) -> list[str]:
    """Render a single (episode, anomaly_class) hit for the verbose report.

    Returns a list of lines (since teleports can have multiple per episode).
    Each class formats its own diagnostic numbers; the leading `ep N (len=L):`
    prefix is uniform so the verbose dump aligns visually.
    """
    prefix = f"     ep {ep_id:>4} (len={ep_len:>4}):"
    if class_key == "teleports":
        # info is a list of (frame_idx, delta, ratio) tuples.
        return [f"{prefix} teleport at frame {f:>4}, Δs={d:.4f} rad, {r:.1f}× median" for f, d, r in info]
    if class_key == "padded":
        return [
            f"{prefix} midrun (from frame {info['midrun_start']}) median Δs="
            f"{info['midrun_median']:.6f} rad — episode is mostly frozen"
        ]
    if class_key == "tiny":
        return [
            f"{prefix} only {info['episode_length']} frame(s) — below useful threshold ({info['threshold']})"
        ]
    if class_key == "nan":
        return [
            f"{prefix} NaN/Inf detected in '{info['location']}' column "
            f"(episode has {info['n_frames']} frame(s))"
        ]
    if class_key == "leading_idle":
        return [
            f"{prefix} first {info['n_idle_frames']} frame(s) idle "
            f"(max Δs={info['leading_max_delta']:.6f} rad) — leading-trim bug"
        ]
    if class_key == "trailing_idle":
        return [
            f"{prefix} last {info['n_idle_frames']} frame(s) idle "
            f"(max Δs={info['trailing_max_delta']:.6f} rad) — trailing-trim bug"
        ]
    if class_key == "joint_velocity":
        # info is a list of (frame_idx, joint_index, delta) tuples — can be
        # many per episode for a sustained over-speed run, so cap the print.
        lines = [
            f"{prefix} per-joint Δ over threshold at frame {f}, joint {j}: {d:.4f} rad"
            for f, j, d in info[:5]
        ]
        if len(info) > 5:
            lines.append(f"     ... ({len(info) - 5} more joint-velocity hit(s) suppressed)")
        return lines
    if class_key == "joint_spike":
        # info is list of (frame_idx, joint_index, delta_rad, ratio_to_joint_median).
        # These are the L2-misses — single-joint single-frame stutters
        # (visually noticeable). Print joint + delta + per-joint ratio.
        lines = [
            f"{prefix} joint-spike at frame {f}, joint {j}: Δ={d:.4f} rad ({r:.1f}× that joint's median)"
            for f, j, d, r in info[:5]
        ]
        if len(info) > 5:
            lines.append(f"     ... ({len(info) - 5} more joint-spike hit(s) suppressed)")
        return lines
    if class_key == "tracking_error":
        # info is a dict: max ||state-action||, where it peaks, where it first
        # crosses threshold, and the end-of-episode error (how far the robot
        # ended from the command).
        return [
            f"{prefix} state diverges from commanded action — "
            f"max ||Δ||={info['max_error_rad']:.3f} rad at frame {info['max_error_frame']} "
            f"(first >{info['threshold']:.2f} at frame {info['first_exceed_frame']}, "
            f"end-of-ep {info['end_error_rad']:.3f} rad)"
        ]
    if class_key == "norm_clip":
        # info dict: fraction of rel-action chunk deltas that normalize outside
        # [-1,1] (clipped at train time), plus the worst-case |normalized| value.
        return [
            f"{prefix} {info['clip_fraction'] * 100:.1f}% of rel-action deltas clip "
            f"(>{info['threshold'] * 100:.1f}% threshold) — {info['n_clipped']}/{info['n_total']} "
            f"values outside [-1,1] under MIN_MAX norm, max |norm|={info['max_abs_norm']:.2f} "
            f"(H={info['rel_horizon']})"
        ]
    if class_key == "gripper_drift":
        # Two sub-checks can independently trigger this class — show only
        # what actually fired so the diagnostic doesn't mislead.
        reasons: list[str] = []
        if info["drift_range"] > info["tolerance"]:
            reasons.append(
                f"in-episode drift {info['drift_range']:.4f} > tol {info['tolerance']} "
                f"(range [{info['grip_min']:.4f}, {info['grip_max']:.4f}])"
            )
        if (
            info["expected_value"] is not None
            and info["mean_offset_from_expected"] is not None
            and info["mean_offset_from_expected"] > info["tolerance"]
        ):
            reasons.append(
                f"mean off by {info['mean_offset_from_expected']:.4f} from expected={info['expected_value']}"
            )
        return [f"{prefix} gripper (dim {info['gripper_dim']}): " + "; ".join(reasons)]
    if class_key == "image_intensity":
        # info is a list of (frame, camera_key, delta_intensity).
        lines = [f"{prefix} {cam} intensity jump at frame {f}: Δ={d:.1f}/255" for f, cam, d in info[:5]]
        if len(info) > 5:
            lines.append(f"     ... ({len(info) - 5} more image-intensity hit(s) suppressed)")
        return lines
    return [f"{prefix} (no detail formatter for '{class_key}')"]


def print_report(result: dict, verbose: bool, max_verbose_episodes: int = 10) -> None:
    root = result["dataset_root"]
    n_total = result["n_episodes"]
    affected = _affected_episodes(result)
    counts = {key: len(result.get(key, {})) for key, _, _ in ANOMALY_CLASSES}
    counts_summary = (
        ", ".join(f"{short}: {counts[key]}" for key, _, short in ANOMALY_CLASSES if counts[key] > 0) or "none"
    )

    print(f"\n── {root.name}")
    print(f"   {len(affected)}/{n_total} episodes anomalous ({counts_summary})")

    # Surface dataset-level corruption (couldn't open a parquet file at all).
    # Doesn't go through the per-episode pipeline so it's reported separately.
    for parquet, err in result.get("corrupt_parquets", []):
        print(f"   ⚠ CORRUPT PARQUET: {parquet.name} ({err})")

    for key, display_name, _ in ANOMALY_CLASSES:
        hits_map = result.get(key, {})
        if not hits_map:
            continue
        if verbose:
            print(f"   {display_name}:")
            sorted_eps = sorted(hits_map.keys())
            # Cap per-class detail lines. 0 = unlimited. Default 10 keeps
            # single-lineage inspections readable when a whole round is
            # anomalous (e.g. gripper_drift firing on every episode).
            if max_verbose_episodes and len(sorted_eps) > max_verbose_episodes:
                display_eps = sorted_eps[:max_verbose_episodes]
                n_suppressed = len(sorted_eps) - max_verbose_episodes
            else:
                display_eps = sorted_eps
                n_suppressed = 0
            for ep_id in display_eps:
                for line in _format_episode_detail(key, ep_id, result["ep_lengths"][ep_id], hits_map[ep_id]):
                    print(line)
            if n_suppressed > 0:
                remaining_compact = ", ".join(str(e) for e in sorted_eps[max_verbose_episodes:])
                print(
                    f"     ... ({n_suppressed} more episode(s) suppressed; "
                    f"raise via --max_verbose_episodes=N, 0=unlimited)"
                )
                print(f"     remaining ep ids: [{remaining_compact}]")
        else:
            compact = ", ".join(str(e) for e in sorted(hits_map.keys()))
            print(f"   {display_name.lower()} episodes: [{compact}]")


def _format_argv(argv: list[str]) -> str:
    """Pretty-print an argv list as a shell-safe one-flag-per-line block."""
    return " \\\n        ".join(shlex.quote(a) for a in argv)


def _emit_template_cleanup(root: Path, round_n: int | None, affected: list[int]) -> None:
    """Fallback: no sidecar provided → emit a TEMPLATE with <placeholders>."""
    if round_n is None:
        print(
            f"     # Couldn't parse round number from dataset basename '{root.name}'.\n"
            f"     # Identify your lineage's training dir and run:\n"
            f"     bash my_scripts/dagger_cleanup_lineage.sh <round-N's training dir> --from_round=<N>"
        )
        return
    print(
        f"     # 1) Find the lineage's training dir for round {round_n}, e.g.:\n"
        f"     #      outputs/training/<base_policy_name>_ft_dag{round_n}/\n"
        f"     # 2) Wipe round {round_n} onwards (preserves rounds 1..{round_n - 1}):\n"
        f"     bash my_scripts/dagger_cleanup_lineage.sh <train_dir>_ft_dag{round_n} --from_round={round_n} -y\n"
        f"     # 3) Re-run the original orchestrator/sweep command with --resume;\n"
        f"     #    auto-resumes from round {round_n} step 1 (re-records intervention\n"
        f"     #    branching off round {round_n - 1}'s preserved policy).\n"
        f"     # Pass --sidecar=<train_dir>/dagger/config.json to this script to get\n"
        f"     # the EXACT cleanup + resume commands instead of this template."
    )
    if round_n == 1:
        print(
            f"\n     # NOTE: this IS the first round — there's no round 0 to branch from.\n"
            f"     # The full lineage would be wiped (--from_round=1 == historical behavior).\n"
            f"     # If only specific episodes are bad, you may want lerobot-edit-dataset instead:\n"
            f"     #   lerobot-edit-dataset --root {root} --operation.type delete_episodes \\\n"
            f'     #       --operation.episode_indices "{affected}"\n'
            f"     # …followed by `bash my_scripts/compute_relative_stats.sh` to refresh the sidecar."
        )


def _emit_resolved_cleanup(sidecar_path: Path, sidecar: dict, round_n: int) -> None:
    """Emit a fully-resolved cleanup + resume command using sidecar data.

    Resume command source-of-truth:
      - sidecar.sweep_invocation present → reproduce the sweep wrapper command
        (it's what actually spawned the lineage).
      - sidecar.sweep_invocation absent  → reproduce the orchestrator command
        directly.
    --resume is appended (idempotent — already-complete iterations are skipped).
    """
    train_dir = sidecar.get("training_output_dir") or ""
    if not train_dir:
        print("     # sidecar missing training_output_dir; falling back to template.")
        return
    cleanup_cmd = (
        f"bash my_scripts/dagger_cleanup_lineage.sh \\\n        "
        f"{shlex.quote(train_dir)} \\\n        "
        f"--from_round={round_n} -y"
    )
    print(f"     # 1) Wipe round {round_n} onwards (preserves rounds 1..{round_n - 1}):")
    print(f"     {cleanup_cmd}")
    sweep = sidecar.get("sweep_invocation")
    if sweep and sweep.get("argv"):
        # Sweep wrapper resume: filter out --force_restart if present in
        # original argv (the lineage we just wiped will replay; the rest of
        # the sweep continues via resume detection). Add --resume / -y so
        # the wrapper doesn't block on stdin.
        argv = [a for a in sweep["argv"] if a != "--force_restart" and a != "--resume"]
        argv = [*argv, "--resume"]
        wrapper = sweep.get("wrapper") or "my_scripts/dagger_orchestrate_sweep.sh"
        print(f"\n     # 2) Resume the SWEEP (re-runs the wiped iteration's rounds {round_n}+;")
        print("     #    other iterations auto-skip already-complete rounds):")
        print(f"     bash {wrapper} \\\n        {_format_argv(argv)}")
    else:
        argv = [
            a
            for a in sidecar.get("orchestrator_invocation", {}).get("argv", [])
            if a != "--force_restart" and a != "--resume"
        ]
        argv = [*argv, "--resume"]
        print(f"\n     # 2) Resume the ORCHESTRATOR (re-runs round {round_n}+; earlier rounds auto-skip):")
        print(f"     bash my_scripts/dagger_orchestrate.sh \\\n        {_format_argv(argv)}")
    print(f"\n     # (sidecar: {sidecar_path})")


def print_cleanup_command(result: dict, sidecar_path: Path | None = None) -> None:
    """Print a copy-pasteable cleanup + resume command for re-running this
    round (and all downstream rounds) of the lineage.

    Why not lerobot-edit-dataset? Deleting individual episodes via
    edit-dataset would leave the rel-action stats sidecar stale (computed
    from the polluted episodes), the downstream rounds' merged datasets /
    blends / trained policies all still embed the pollution, and the
    policy's normalization range was already stretched by the bad episodes.
    The clean fix is to nuke round N and every round downstream of it,
    re-record round N's intervention with the upstream-fixed recorder, and
    let dagger_orchestrate.sh re-chain rounds N..end.

    With --sidecar provided, the resume command is fully resolved (sweep
    wrapper if the lineage was sweep-spawned, else direct orchestrator).
    Without, a TEMPLATE with placeholders is emitted instead.
    """
    affected = sorted(_affected_episodes(result))
    if not affected:
        return
    root = result["dataset_root"]
    parsed = _parsed_intervention(root.name)
    round_n_str = str(parsed.round) if parsed else "<N>"
    print(f"\n   To re-record round {round_n_str} (and re-chain all downstream rounds):")
    if parsed is None or parsed.round is None:
        _emit_template_cleanup(root, None, affected)
        return
    round_n: int = parsed.round
    if sidecar_path is None:
        _emit_template_cleanup(root, round_n, affected)
        return
    try:
        sidecar = json.loads(sidecar_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        print(f"     # ERROR reading --sidecar {sidecar_path}: {e}")
        print("     # falling back to template:")
        _emit_template_cleanup(root, round_n, affected)
        return
    _emit_resolved_cleanup(sidecar_path, sidecar, round_n)
    if round_n == 1:
        print(
            f"\n     # NOTE: round 1 has no round 0 to branch from — wiping it restarts"
            f"\n     # the lineage from scratch (--from_round=1). For surgical episode removal:"
            f"\n     #   lerobot-edit-dataset --root {root} --operation.type delete_episodes \\\n"
            f'     #       --operation.episode_indices "{affected}"'
            f"\n     # …followed by `bash my_scripts/compute_relative_stats.sh` to refresh the sidecar."
        )


def discover_interventions_in_dir(
    parent_dir: Path,
    lineage_stem: str | None = None,
    lineage_infix: str | None = None,
    name_substring: str | None = None,
) -> list[Path]:
    """Scan `parent_dir` for raw DAgger intervention datasets.

    Optional filters (all AND-composed):
      - `lineage_stem`: keep only datasets whose `parse_dataset_short.prefix`
        matches exactly (use to expand a single dataset path to its full
        same-lineage sibling set).
      - `lineage_infix`: keep only `'a'` (abs-action) or `'r'` (rel-action)
        datasets — defaulting to None means accept both (a same-prefix lineage
        is unlikely to mix infixes in practice, but the filter is here for
        completeness).
      - `name_substring`: substring match on basename — used by `--lineage_filter`
        when the user wants a loose "anything matching X" scope rather than an
        exact prefix.

    Derived datasets (merged `_m`, blends `_blend*`, `_blend*_nocoll`) are
    skipped — they're downstream of the raw intervention so they'd just
    re-flag the same underlying anomalies. `_old` / `_v2`-style suffixes
    after `_dag<N>` are KEPT (separate recordings worth checking).

    Sorted by (prefix, infix, round, suffix) for natural per-lineage grouping.
    """
    if not parent_dir.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(parent_dir.iterdir()):
        if not p.is_dir():
            continue
        parsed = _parsed_intervention(p.name)
        if parsed is None:
            continue
        if lineage_stem is not None and parsed.prefix != lineage_stem:
            continue
        if lineage_infix is not None and parsed.infix != lineage_infix:
            continue
        if name_substring and name_substring not in p.name:
            continue
        out.append(p)

    def _key(p: Path):
        parsed = _parsed_intervention(p.name)
        assert parsed is not None  # filtered above
        return (parsed.prefix or "", parsed.infix or "", parsed.round or 0, _suffix_of(p.name))

    out.sort(key=_key)
    return out


def auto_discover_intervention_datasets(
    cache_root: Path, hf_user: str, lineage_filter: str | None
) -> list[Path]:
    """No-positional-filter wrapper: scan the whole `<cache_root>/<hf_user>/`
    user dir. Thin shim around discover_interventions_in_dir kept for callsite
    readability."""
    return discover_interventions_in_dir(
        cache_root / hf_user,
        name_substring=lineage_filter,
    )


def _dataset_root_of(path: Path) -> Path:
    """Normalize a user-supplied path to the DATASET ROOT dir (the one whose
    basename is the DAgger dataset name and which contains data/ + meta/).

    Tab-completion routinely lands the user on a deeper path —
    `.../<dataset>/data/chunk-000` or `.../<dataset>/data` — but lineage
    expansion + the naming parse both key off the DATASET basename, so a
    chunk/data path would never parse as an intervention dataset and
    expansion would silently no-op (the reported bug). Climb up in those
    cases so `<dataset>/data/chunk-NNN` and `<dataset>/data` both resolve to
    `<dataset>`. Any other path is returned unchanged.
    """
    if path.name.startswith("chunk-"):
        # .../<dataset>/data/chunk-NNN → .../<dataset>
        return path.parent.parent
    if path.name == "data":
        # .../<dataset>/data → .../<dataset>
        return path.parent
    return path


def expand_dataset_root_to_lineage(dataset_root: Path) -> list[Path]:
    """Given a single intervention dataset path, return every sibling in the
    same parent dir that shares its lineage stem + action infix. Mirrors how
    `dagger_cleanup_lineage.sh --detect_siblings` expands a single training
    dir to a same-prefix-K family on disk; here we work in dataset-name space
    instead of training-dir-name space, but the same `parse_dataset_short`
    canonical module owns the parse.

    Accepts a path pointing at the dataset root OR at a deeper `data/` or
    `data/chunk-NNN` dir (normalized via `_dataset_root_of` first). Returns
    `[dataset_root]` unchanged if the input isn't a recognizable intervention
    dataset (e.g. user passed a base dataset). Caller treats that as "scan
    exactly what was given, no expansion."
    """
    resolved = _dataset_root_of(dataset_root)
    parsed = _parsed_intervention(resolved.name)
    if parsed is None or parsed.prefix is None or parsed.infix is None:
        return [dataset_root]
    return discover_interventions_in_dir(
        resolved.parent,
        lineage_stem=parsed.prefix,
        lineage_infix=parsed.infix,
    )


def lineage_stem_of(dataset_root: Path) -> str | None:
    """Return the lineage stem (everything before `_<a|r>_dag<N>`) of a
    dataset, or None if it doesn't parse as an intervention dataset."""
    parsed = _parsed_intervention(dataset_root.name)
    return parsed.prefix if parsed else None


def print_lineage_summary(results: list[dict]) -> None:
    """Group results by lineage and print a compact per-round table.

    Rounds with anomalies get a ⚠ marker; clean rounds list as "0/N". When a
    lineage has any anomalous round, the lineage line itself gets a marker
    too. Lineages with no DAgger naming match are bundled under "(misc)".
    """
    by_lineage: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        stem = lineage_stem_of(r["dataset_root"]) or "(misc)"
        by_lineage[stem].append(r)
    print("\n── Per-lineage summary ──")
    for stem in sorted(by_lineage):
        rows = by_lineage[stem]

        # Round number → for "(misc)" path, parse_dataset_short returns
        # kind="base" with round=None, so fall back to 0 there.
        def _round_of(r: dict) -> int:
            parsed = _parsed_intervention(r["dataset_root"].name)
            return parsed.round if parsed and parsed.round is not None else 0

        rows.sort(key=_round_of)
        lineage_anomalous = sum(len(_affected_episodes(r)) for r in rows)
        flag = " ⚠" if lineage_anomalous > 0 else ""
        print(f"  {stem}{flag}  ({len(rows)} round(s), {lineage_anomalous} anomalous episode(s))")
        for r in rows:
            n_total = r["n_episodes"]
            n_aff = len(_affected_episodes(r))
            round_n = _round_of(r)
            # Include the trailing suffix (`_old`, `_v2`, etc.) so users can
            # tell apart parallel datasets that share a round number, e.g.
            # `_dag1` vs `_dag1_old` (manual backup) in the same lineage.
            suffix = _suffix_of(r["dataset_root"].name)
            extras = ""
            if n_aff > 0:
                # Compact per-class breakdown for the single anomalous round.
                pairs = ", ".join(
                    f"{short}: {len(r.get(key, {}))}"
                    for key, _, short in ANOMALY_CLASSES
                    if len(r.get(key, {})) > 0
                )
                extras = f" ({pairs}) ⚠"
            print(f"      round {round_n:>2}{suffix:<6}: {n_aff:>3}/{n_total:<3} anomalous{extras}")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument(
        "--dataset_root",
        type=Path,
        help="Dataset root containing data/chunk-*/ and meta/ (or a chunk-NNN dir directly).",
    )
    src.add_argument(
        "--dataset_roots",
        type=Path,
        nargs="+",
        help="Multiple dataset roots — scan each, one report per.",
    )
    # Auto-scan controls — kick in when neither --dataset_root nor --dataset_roots
    # is given. Globs `<cache_root>/<hf_user>/` for raw intervention datasets
    # (`*_[ar]_dag<N>`), filters out merged (`_m`) and blends (`_blend*`).
    p.add_argument(
        "--cache_root",
        type=Path,
        default=_DEFAULT_CACHE_ROOT,
        help=f"LeRobot HF cache root for auto-scan. Default {_DEFAULT_CACHE_ROOT}.",
    )
    p.add_argument(
        "--hf_user",
        type=str,
        default=_DEFAULT_HF_USER,
        help=f"HF user subdir under --cache_root for auto-scan. Default '{_DEFAULT_HF_USER}' "
        "(override via $HF_USER env or this flag).",
    )
    p.add_argument(
        "--lineage_filter",
        type=str,
        default=None,
        help="Substring filter on dataset basename (auto-scan only). Use to "
        "scope a huge cache to one lineage family, e.g. --lineage_filter=d30_nopad.",
    )
    p.add_argument(
        "--abs_threshold_rad",
        type=float,
        default=DEFAULT_ABS_THRESHOLD_RAD,
        help=f"Teleport: Δstate > this absolute threshold (rad). Default {DEFAULT_ABS_THRESHOLD_RAD}.",
    )
    p.add_argument(
        "--ratio_threshold",
        type=float,
        default=DEFAULT_RATIO_THRESHOLD,
        help=f"Teleport: Δstate > this × episode median. Default {DEFAULT_RATIO_THRESHOLD}×.",
    )
    p.add_argument(
        "--ratio_min_abs_rad",
        type=float,
        default=DEFAULT_RATIO_MIN_ABS_RAD,
        help=f"Teleport ratio-check FLOOR (rad): the ratio-vs-median check "
        f"only fires when Δstate ALSO exceeds this absolute floor. Guards "
        f"against false positives on padded episodes where the median Δ "
        f"is artificially crushed to ~0 by the frozen tail, causing the "
        f"ratio check to fire on normal motion. The absolute-threshold "
        f"check (--abs_threshold_rad) is unaffected. Default "
        f"{DEFAULT_RATIO_MIN_ABS_RAD} rad/tick.",
    )
    p.add_argument(
        "--frozen_threshold_rad",
        type=float,
        default=DEFAULT_FROZEN_THRESHOLD_RAD,
        help=f"Threshold below which Δstate counts as 'frozen' — drives the "
        f"padded_frozen, leading_idle, and trailing_idle classes. "
        f"Default {DEFAULT_FROZEN_THRESHOLD_RAD}.",
    )
    p.add_argument(
        "--min_useful_episode_len",
        type=int,
        default=DEFAULT_MIN_USEFUL_EPISODE_LEN,
        help=f"tiny_episode: flag episodes shorter than this. "
        f"Default {DEFAULT_MIN_USEFUL_EPISODE_LEN} frames "
        f"(< this is too short for useful diffusion training chunks).",
    )
    p.add_argument(
        "--edge_idle_frames",
        type=int,
        default=DEFAULT_EDGE_IDLE_FRAMES,
        help=f"leading_idle / trailing_idle: number of edge frames that must "
        f"ALL be below frozen_threshold to fire. "
        f"Default {DEFAULT_EDGE_IDLE_FRAMES} frames (~0.5s at 30Hz).",
    )
    p.add_argument(
        "--joint_velocity_threshold_rad",
        type=float,
        default=DEFAULT_JOINT_VELOCITY_THRESHOLD,
        help=f"joint_velocity: per-joint single-step |Δq| above this is flagged. "
        f"Default {DEFAULT_JOINT_VELOCITY_THRESHOLD} rad/frame. "
        f"NOTE: significant overlap with TELEPORT (which catches any L2-Δstate "
        f"≥ abs_threshold). joint_velocity is off by default for that reason; "
        f"enable with --check_joint_velocity if you want per-axis granularity.",
    )
    p.add_argument(
        "--check_joint_velocity",
        action="store_true",
        help="Enable per-joint joint_velocity class. OFF by default since mostly redundant with TELEPORT.",
    )
    p.add_argument(
        "--joint_spike_ratio",
        type=float,
        default=DEFAULT_JOINT_SPIKE_RATIO,
        help=f"joint_spike: per-joint Δ must exceed this multiple of THAT "
        f"joint's own median |Δ| to flag. Catches single-frame stutters "
        f"that L2-TELEPORT misses (e.g. wrist_3 lurches 5× its normal "
        f"step while other joints stay smooth). Default "
        f"{DEFAULT_JOINT_SPIKE_RATIO}×.",
    )
    p.add_argument(
        "--joint_spike_min_abs_rad",
        type=float,
        default=DEFAULT_JOINT_SPIKE_MIN_ABS_RAD,
        help=f"joint_spike: per-joint Δ must ALSO exceed this absolute "
        f"floor (rad/frame). Guards against false positives on joints "
        f"that barely move (median ≈ 0 → ratio explodes on any motion). "
        f"Default {DEFAULT_JOINT_SPIKE_MIN_ABS_RAD} rad/frame.",
    )
    p.add_argument(
        "--no_check_joint_spike",
        action="store_true",
        help="Disable the joint_spike class (per-joint single-frame "
        "stutter detection). ON by default since it catches real "
        "visually-noticeable artifacts the L2-norm TELEPORT check misses; "
        "pass this flag to turn it off if it's too noisy on your data.",
    )
    p.add_argument(
        "--tracking_error_threshold_rad",
        type=float,
        default=DEFAULT_TRACKING_ERROR_THRESHOLD_RAD,
        help=f"tracking_error: flag an episode when max ||state - action|| "
        f"(arm joints, gripper excluded) exceeds this. Catches open-loop "
        f"command runaway — the robot stalls on contact while the ruckig "
        f"command marches on, so the recorded action no longer matches the "
        f"observed state. Clean RRT tracks within ~0.03 rad; divergent "
        f"episodes reach 0.7-1.5 rad. Default "
        f"{DEFAULT_TRACKING_ERROR_THRESHOLD_RAD} rad.",
    )
    p.add_argument(
        "--no_check_tracking_error",
        action="store_true",
        help="Disable the tracking_error class (measured-state vs "
        "commanded-action divergence). ON by default — it catches the "
        "open-loop runaway corruption that JOINT_SPIKE misses. Auto-skips "
        "relative-encoded action columns, so it's safe to leave on.",
    )
    p.add_argument(
        "--detect_norm_clip",
        action="store_true",
        help="Enable the NORM_CLIP class (OFF by default — needs an action-delta "
        "stats sidecar). Reconstructs the relative-action chunk deltas "
        "(action[t:t+H] - state[t], arm joints), MIN_MAX-normalizes them with "
        "--stats_rel8, and flags episodes where too many deltas land outside "
        "[-1,1] (clipped at train time). Pass the AGGREGATED sidecar to measure "
        "the weighted-sampling compression footgun. Auto-resolves the sidecar to "
        "outputs/dataset_stats/<dataset>/stats_rel8.json when --stats_rel8 omitted.",
    )
    p.add_argument(
        "--stats_rel8",
        type=Path,
        default=None,
        help="Path to the action-delta stats sidecar (stats_rel8.json) for "
        "NORM_CLIP. If omitted, auto-resolves per dataset to "
        "outputs/dataset_stats/<dataset_name>/stats_rel8.json. Point this at an "
        "AGGREGATED sidecar to check clipping against the actual training "
        "normalizer range rather than each round's own (which can't clip itself).",
    )
    p.add_argument(
        "--rel_horizon",
        type=int,
        default=DEFAULT_REL_HORIZON,
        help=f"NORM_CLIP: relative-action chunk horizon H (stats_rel8 ⇒ 8). Each "
        f"anchor frame contributes deltas action[t:t+H] - state[t]. "
        f"Default {DEFAULT_REL_HORIZON}.",
    )
    p.add_argument(
        "--norm_clip_frac_threshold",
        type=float,
        default=DEFAULT_NORM_CLIP_FRAC_THRESHOLD,
        help=f"NORM_CLIP: flag an episode when this fraction of its rel-action "
        f"deltas clip outside [-1,1]. Default {DEFAULT_NORM_CLIP_FRAC_THRESHOLD}.",
    )
    p.add_argument(
        "--gripper_dim",
        type=int,
        default=-1,
        help="Gripper index in observation.state. Default -1 (last column). "
        "Used by the gripper_drift, tracking_error, and norm_clip classes.",
    )
    p.add_argument(
        "--gripper_drift_tolerance",
        type=float,
        default=DEFAULT_GRIPPER_DRIFT_TOLERANCE,
        help=f"gripper_drift: max allowed in-episode Δgripper. "
        f"Default {DEFAULT_GRIPPER_DRIFT_TOLERANCE} (just above float32 "
        f"quantization noise). RRT shouldn't touch the gripper, so any motion "
        f"above this is a recorder/planner bug in the grip0 lineages.",
    )
    p.add_argument(
        "--gripper_expected_value",
        type=float,
        default=None,
        help="Optional. If set, gripper_drift also flags episodes whose mean "
        "gripper value is more than --gripper_drift_tolerance away from this. "
        "Use --gripper_expected_value=0 to assert the grip0 base's constant.",
    )
    p.add_argument(
        "--image_intensity_threshold",
        type=float,
        default=DEFAULT_IMAGE_INTENSITY_THRESHOLD,
        help=f"image_intensity: max allowed single-frame Δmean_pixel (0..255). "
        f"Default {DEFAULT_IMAGE_INTENSITY_THRESHOLD}. Catches camera dropouts / "
        f"render glitches that don't manifest in state. Requires --check_images.",
    )
    p.add_argument(
        "--check_images",
        action="store_true",
        help="Enable image_intensity class. OFF by default since decoding the "
        "PNG bytes for every observation.images.* column for every frame is "
        "EXPENSIVE (10-60× slower than the default state-only scan). Use when "
        "you want to catch camera/render bugs the state-side classes miss.",
    )
    p.add_argument(
        "--skip",
        choices=[short for _, _, short in ANOMALY_CLASSES],
        action="append",
        default=[],
        help="Suppress one anomaly class. Repeat to skip multiple. Choices: "
        + ", ".join(short for _, _, short in ANOMALY_CLASSES)
        + ". "
        "Useful for triaging only the classes you care about.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Per-dataset summary lines only; suppress per-episode details.",
    )
    p.add_argument(
        "--max_verbose_episodes",
        type=int,
        default=10,
        help="Verbose mode: cap on per-class per-episode detail lines "
        "(default 10). When exceeded, the remaining episode IDs are still "
        "printed compactly so users can spot which rounds are worst without "
        "flooding the terminal. Pass 0 for unlimited. No effect in --quiet mode.",
    )
    p.add_argument(
        "--print_cleanup_command",
        "--print_delete_command",  # legacy alias from the earlier teleport-only version
        dest="print_cleanup_command",
        action="store_true",
        help="After the report, emit a copy-pasteable dagger_cleanup_lineage.sh "
        "command to re-record round N and re-chain all downstream rounds. "
        "(Sharing code with cleanup_lineage avoids leaving stale rel-action stats "
        "sidecars / polluted merged datasets / etc. that lerobot-edit-dataset would.)",
    )
    p.add_argument(
        "--sidecar",
        type=Path,
        default=None,
        help="Path to a single lineage's <train_dir>/dagger/config.json. When "
        "provided alongside --print_cleanup_command, the resume command is "
        "fully resolved using sidecar data: dagger_orchestrate_sweep.sh if the "
        "lineage was sweep-spawned (sweep_invocation recorded in the sidecar), "
        "else dagger_orchestrate.sh. Without --sidecar, a template with "
        "<placeholders> is emitted instead.",
    )
    p.add_argument(
        "--no_expand_to_lineage",
        action="store_true",
        help="By default, passing a single --dataset_root that matches the "
        "DAgger intervention naming (`*_[ar]_dag<N>`) is auto-expanded to ALL "
        "siblings in the same parent dir with matching lineage stem + action "
        "infix — so you can copy-paste any one round's path and get the full "
        "lineage scanned. Set this flag to disable that and scan ONLY the "
        "exact dataset you passed.",
    )
    args = p.parse_args()

    # args.skip is a LIST (action="append") — empty list = detect everything
    # that's enabled by default (joint_velocity + image_intensity are opt-in
    # via their own flags, so --skip on them is a no-op).
    skipped = set(args.skip)
    detect_teleport = "teleport" not in skipped
    detect_padded = "padded" not in skipped
    detect_tiny = "tiny" not in skipped
    detect_nan = "nan" not in skipped
    detect_leading_idle = "leading_idle" not in skipped
    detect_trailing_idle = "trailing_idle" not in skipped
    detect_gripper = "gripper_drift" not in skipped
    detect_joint_velocity = args.check_joint_velocity and "joint_velocity" not in skipped
    detect_joint_spike = (not args.no_check_joint_spike) and "joint_spike" not in skipped
    detect_tracking_error = (not args.no_check_tracking_error) and "tracking_error" not in skipped
    detect_norm_clip = args.detect_norm_clip and "norm_clip" not in skipped
    detect_image_intensity = args.check_images and "image_intensity" not in skipped

    # Resolve dataset list. Precedence:
    #   1. --dataset_root (single, by default auto-expanded to the full lineage)
    #   2. --dataset_roots (multiple, scanned as-is — no expansion)
    #   3. auto-scan <cache_root>/<hf_user>/*_[ar]_dag<N> (intervention only)
    auto_scanned = False
    expanded_from: Path | None = None
    if args.dataset_root:
        if args.no_expand_to_lineage:
            roots = [args.dataset_root]
        else:
            expanded = expand_dataset_root_to_lineage(args.dataset_root)
            # expand_dataset_root_to_lineage returns [root] unchanged when the
            # input doesn't parse — so a non-DAgger path stays standalone with
            # no surprise globbing.
            if len(expanded) > 1 or (len(expanded) == 1 and expanded[0] != args.dataset_root):
                expanded_from = args.dataset_root
                roots = expanded
                # Report the resolved dataset name + the dir the siblings live
                # in (not args.dataset_root, which may be a deeper data/chunk-*
                # path the user tab-completed onto — see _dataset_root_of).
                resolved = _dataset_root_of(args.dataset_root)
                print(
                    f"[expand] {resolved.name} → {len(roots)} sibling(s) "
                    f"in {resolved.parent} (--no_expand_to_lineage to disable)"
                )
            else:
                roots = expanded
    elif args.dataset_roots:
        roots = args.dataset_roots
    else:
        auto_scanned = True
        roots = auto_discover_intervention_datasets(
            args.cache_root,
            args.hf_user,
            args.lineage_filter,
        )
        if not roots:
            print(
                f"ERROR: auto-scan found no DAgger intervention datasets under "
                f"{args.cache_root}/{args.hf_user}.\n"
                f"  Pass --dataset_root / --dataset_roots explicitly, or use "
                f"--cache_root / --hf_user to point elsewhere.",
                file=sys.stderr,
            )
            return 2
        filt_note = f" (filter='{args.lineage_filter}')" if args.lineage_filter else ""
        print(
            f"[auto-scan] {len(roots)} intervention dataset(s) under "
            f"{args.cache_root}/{args.hf_user}{filt_note}"
        )

    # In auto-scan or lineage-expansion mode, per-dataset detail would be
    # hundreds of lines for a cluttered cache — suppress unless the user
    # explicitly opted out of quiet. (Single-dataset mode keeps verbose
    # default since the user clearly wanted to inspect that one specifically.)
    effective_quiet = args.quiet or auto_scanned or (expanded_from is not None)

    total_episodes = 0
    total_affected = 0
    per_class_totals: dict[str, int] = {key: 0 for key, _, _ in ANOMALY_CLASSES}
    results = []
    for root in roots:
        # NORM_CLIP needs an action-delta sidecar. Use the explicit --stats_rel8
        # if given (applies to all roots), else auto-resolve per dataset to
        # outputs/dataset_stats/<name>/stats_rel8.json. Missing/malformed → the
        # class silently no-ops for that root (arrays stay None).
        action_delta_min = action_delta_max = None
        if detect_norm_clip:
            stats_path = args.stats_rel8 or _default_rel_stats_path(_dataset_root_of(root))
            loaded = load_action_delta_stats(stats_path)
            if loaded is None:
                print(
                    f"   [norm_clip] no usable action stats at {stats_path} — "
                    f"skipping NORM_CLIP for {root.name}",
                    file=sys.stderr,
                )
            else:
                action_delta_min, action_delta_max = loaded
        try:
            result = scan_dataset(
                root,
                args.abs_threshold_rad,
                args.ratio_threshold,
                args.ratio_min_abs_rad,
                args.frozen_threshold_rad,
                args.min_useful_episode_len,
                args.edge_idle_frames,
                args.joint_velocity_threshold_rad,
                args.joint_spike_ratio,
                args.joint_spike_min_abs_rad,
                args.tracking_error_threshold_rad,
                args.gripper_dim,
                args.gripper_drift_tolerance,
                args.gripper_expected_value,
                args.image_intensity_threshold,
                detect_teleport,
                detect_padded,
                detect_tiny,
                detect_nan,
                detect_leading_idle,
                detect_trailing_idle,
                detect_joint_velocity,
                detect_joint_spike,
                detect_tracking_error,
                detect_gripper,
                detect_image_intensity,
                detect_norm_clip=detect_norm_clip,
                action_delta_min=action_delta_min,
                action_delta_max=action_delta_max,
                rel_horizon=args.rel_horizon,
                norm_clip_frac_thr=args.norm_clip_frac_threshold,
            )
        except FileNotFoundError as e:
            print(f"SKIP {root}: {e}", file=sys.stderr)
            continue
        results.append(result)
        total_episodes += result["n_episodes"]
        total_affected += len(_affected_episodes(result))
        for key, _, _ in ANOMALY_CLASSES:
            per_class_totals[key] += len(result.get(key, {}))
        print_report(
            result,
            verbose=not effective_quiet,
            max_verbose_episodes=args.max_verbose_episodes,
        )
        if args.print_cleanup_command:
            print_cleanup_command(result, sidecar_path=args.sidecar)

    # Per-lineage rollup — most useful in auto-scan mode where the user sees
    # many datasets at once and wants a single glance to find the bad lineage.
    # Print only when scanning more than one dataset; for a single root the
    # per-dataset report above already covers it.
    if len(results) > 1:
        print_lineage_summary(results)

    n_anomalous_datasets = sum(1 for r in results if _affected_episodes(r))
    n_corrupt_files = sum(len(r.get("corrupt_parquets", [])) for r in results)
    print(
        f"\n════════════════════════════════════════\n"
        f"Total: {total_affected}/{total_episodes} episodes anomalous across "
        f"{len(results)} dataset(s)  "
        f"({n_anomalous_datasets} dataset(s) with anomalies)."
    )
    # Per-class breakdown — only show classes that had any hits, plus a note
    # about skipped classes so users know what wasn't scanned.
    for key, display_name, _ in ANOMALY_CLASSES:
        print(f"  {display_name:<15}: {per_class_totals[key]} episode(s)")
    if n_corrupt_files > 0:
        print(f"  ⚠ CORRUPT PARQUET FILES: {n_corrupt_files} (see per-dataset reports above)")
    if skipped:
        print(f"  skipped classes : {sorted(skipped)}")
    thresh_parts = [
        f"abs={args.abs_threshold_rad} rad",
        f"ratio={args.ratio_threshold}× median (floor={args.ratio_min_abs_rad} rad)",
        f"frozen={args.frozen_threshold_rad} rad",
        f"min_len={args.min_useful_episode_len}",
        f"edge_idle={args.edge_idle_frames} frames",
        f"gripper_tol={args.gripper_drift_tolerance}",
    ]
    if detect_joint_velocity:
        thresh_parts.append(f"joint_vel={args.joint_velocity_threshold_rad} rad/frame")
    if detect_joint_spike:
        thresh_parts.append(
            f"joint_spike={args.joint_spike_ratio}× joint-median "
            f"(floor={args.joint_spike_min_abs_rad} rad/frame)"
        )
    if detect_tracking_error:
        thresh_parts.append(f"tracking_error={args.tracking_error_threshold_rad} rad")
    if detect_norm_clip:
        thresh_parts.append(f"norm_clip={args.norm_clip_frac_threshold} frac (H={args.rel_horizon})")
    if detect_image_intensity:
        thresh_parts.append(f"image_intensity={args.image_intensity_threshold}/255")
    print("  thresholds: " + ", ".join(thresh_parts) + "\n════════════════════════════════════════")
    return 1 if total_affected > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
