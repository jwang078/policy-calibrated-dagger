#!/usr/bin/env python
r"""Sim-in-the-loop variant of visualize_shared_autonomy.py.

The parquet-driven script (``visualize_shared_autonomy.py``) feeds the same frozen
dataset frame to the policy every step, so observations go stale as soon as the policy
diverges from the demo. This script drives a real splatsim env each step so
observations stay in sync with the actually-executed actions.

**Required setup** — splatsim must run out-of-process (the wrapper already holds a
pybullet GUI client in this process and a second in-process pybullet client would
crash). Launch the simulator once:

    cd ~/code/SplatSim && \\
        python scripts/launch_nodes.py \\
            --robot sim_ur_pybullet_small_engine_new_interactive \\
            --robot_port 6001 \\
            --robot_name robot_iphone_w_engine_curtain \\
            --eval_benchmark_repo_id <benchmark_dataset_repo_id> \\
            --sync_physics_to_client \\
            --strict_goal_tolerances

``--sync_physics_to_client`` is NOT optional for rollout work: without it,
physics integrates in WALL-CLOCK time, so every slow tick (a chunk-rebuild
denoise takes ~0.3 s) lets the PD controller keep integrating toward the last
target — the robot visibly LURCHES at every chunk boundary, in the video and
in the achieved states, while the commanded actions look perfectly smooth
(measured 2026-08-25: pixel-motion spikes ~20x the per-tick median, all at
the chunk-boundary phase; synced rerun of the same config had zero). The
orchestrator's managed sims pass it by default; manual launches must too.

Then point this script at it:

    python my_scripts/visualize_shared_autonomy_sim.py \\
        --policy_path .../pretrained_model \\
        --dataset_repo_id JennyWWW/splatsim_approach_lever_7_lowres_5path_10fails \\
        --episode_index 305 \\
        --forward_flow_ratios 0.0 0.05 0.2 0.4 0.8 1.0 \\
        --blend_strategy denoise --guidance_repr delta --blend_interval_frac \\
        --env_task upright_small_engine_new \\
        --env_external_port 6001

For example:
# 1. Launch splatsim out-of-process (once, stays up). --strict_goal_tolerances
#    matches the orchestrator's recording/blend sims (loose eval thresholds
#    would end rollouts "close enough" to the goal and freeze the video into
#    the post-success hold).
cd ~/code/SplatSim && python scripts/launch_nodes.py \
    --robot sim_ur_pybullet_small_engine_new_interactive \
    --robot_port 6001 \
    --robot_name robot_iphone_w_engine_curtain \
    --eval_benchmark_repo_id JennyWWW/eval_splatsim_approach_lever_benchmark_1000 \
    --sync_physics_to_client \
    --strict_goal_tolerances

# 2. Run visualize (in another terminal)
python my_scripts/visualize_shared_autonomy_sim.py \
    --policy_path outputs/training/pi05_approach_lever_11_biasend_5path_delta_basewrist/checkpoints/006000/pretrained_model \
    --dataset_repo_id JennyWWW/splatsim_approach_lever_7_lowres_5path_10fails \
    --episode_index 305 \
    --forward_flow_ratios 0.0 0.05 0.2 0.4 0.8 1.0 \
    --blend_strategy denoise --guidance_repr delta --blend_interval_frac \
    --env_task upright_small_engine_new --env_external_port 6001

The benchmark scenario is resolved from the episode's ``source_scenario_idx``
metadata when present (intervention datasets — same resolution the blend script
uses), falling back to ``episode_index``, and pinned on every per-ratio reset via
``vec_env.reset(options={"benchmark_start_index": ...})`` so all ratios roll out in
the SAME scenario (a bare seeded reset would let the server's EVAL_BENCHMARK
counter advance one scenario per reset).
``--frame_index`` slices the guidance (demo) actions from the dataset AND teleports
the robot to the demo's pose at that frame after the scenario reset (SplatSim
servers support the teleport over ZMQ; other servers fall back to the
episode-initial pose).

**Sweep-parity debugging** — to reproduce what ``dagger_orchestrate_sweep.sh``'s
blend step (``augment_dataset_with_blending.py``) actually runs, point at the
ROUND'S INTERVENTION DATASET and mirror its blend flags (defaults here already
match on start frame / rollout length / guidance construction):

    python my_scripts/visualize_shared_autonomy_sim.py \\
        --policy_path <the round's branching policy>/pretrained_model \\
        --dataset_repo_id JennyWWW/<intervention_dataset> \\
        --episode_index 0 \\
        --forward_flow_ratios 0.0 0.7 1.0 \\
        --blend_strategy denoise --guidance_repr absolute_pos \\
        --fixed_base_noise=false --clip_sample=false --progress_guidance=true \\
        --env_external_port 6005 --no_show

(``--fixed_base_noise=true``, the default, instead pins one noise draw shared
across ratios/ticks for cross-ratio comparability — cleaner plots, but NOT what
the sweep executes unless it passes ``--fixed_base_noise=true`` too.)

Imports plotting / IO helpers from the sibling parquet script
(``visualize_shared_autonomy.py``) and batch-building helpers from
``visualize_shared_autonomy_sim.py`` itself (which ``augment_dataset_with_blending.py``
also imports).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

# matplotlib's default TkAgg backend initializes Tcl/Tk at import time, which then
# crashes with "Tcl_AsyncDelete: async handler deleted by the wrong thread" once
# splatsim's pybullet GUI thread is running. Force the non-interactive Agg backend
# *before* any pyplot import (the sibling visualize_shared_autonomy module does
# `import matplotlib.pyplot as plt` at module load).
import matplotlib  # noqa: E402

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

# Allow importing the sibling parquet-driven script directly. Also expose
# the repo root on sys.path so `from my_scripts.X import Y` works even when
# this module is invoked from inside `my_scripts/` (e.g. when
# augment_dataset_with_blending.py is launched via
# `python my_scripts/augment_dataset_with_blending.py` — Python sets
# sys.path[0] to `my_scripts/`, not the repo root).
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Sibling-module imports. These previously came from
# ``my_scripts.visualize_shared_autonomy_DEPRECATED``; they've been split into
# topic-focused library modules so this script doesn't depend on a deprecated
# file. Bare module names (no ``my_scripts.`` prefix) so they resolve when
# this script is invoked directly via ``python my_scripts/…``.
from lib_dataset_episode_io import (  # type: ignore[import-not-found]  # noqa: E402
    find_parquet_files,
    get_available_episodes,
    load_episode_frames,
    load_episodes_meta,
    load_task_description,
)
from lib_ee_kinematics import (  # type: ignore[import-not-found]  # noqa: E402
    absolute_positions_to_ee_deltas,
    compute_ee_from_states,
    compute_ee_trajectories,
)
from lib_sa_plotting import (  # type: ignore[import-not-found]  # noqa: E402
    plot_ee_trajectories_3d,
    plot_joint_angles,
)
from lib_sa_policy_loading import (  # type: ignore[import-not-found]  # noqa: E402
    apply_clip_sample_override,
    load_wrapped_policy,
)
from lib_sa_rollout import (  # type: ignore[import-not-found]  # noqa: E402,F401
    # THE shared rollout core — the sweep's blend script executes the same
    # code. _build_sim_batch/_run_filler_phase/progress_guidance_index are
    # re-exported here for back-compat importers.
    _apply_rename_map,
    _build_sim_batch,
    _run_filler_phase,
    check_sim_strict_goal_tolerances,
    progress_guidance_index,
    run_blended_rollout,
    warn_if_sim_physics_unsynced,
)

from lerobot.envs import close_envs  # noqa: E402
from lerobot.envs.factory import make_env, make_env_config, make_env_pre_post_processors  # noqa: E402
from lerobot.policies.shared_autonomy_wrapper import (  # noqa: E402
    BlendMode,
    GuidanceBlendStrategy,
    PolicyGuidanceRepresentation,
)
from lerobot.utils.lerobot_dataset_utils import make_default_rename_map, resolve_dataset_dir  # noqa: E402

# ── sim-server launch hint ────────────────────────────────────────────────────

# launch_nodes.py --robot variant per env task. Mirrors the ENV_TASK →
# ROBOT_VARIANT pairs in my_scripts/env_profiles/*.sh (keep in sync).
_TASK_TO_ROBOT_VARIANT = {
    "planar_3joint": "sim_pybullet_planar_interactive",
    "planar_3joint_oracle": "sim_pybullet_planar_oracle_interactive",
    "planar_3joint_oracle_simple": "sim_pybullet_planar_oracle_simple_interactive",
    "upright_small_engine_new": "sim_ur_pybullet_small_engine_new_interactive",
}

# Splat names that are stale on disk but still recorded in old checkpoints'
# train_configs. The sim must NOT be launched with these (e.g. the
# robot_iphone_w_engine_new splat now renders a murky, mistracked wrist view);
# the env side remaps them to the current splat, so the printed launch hint
# and the env construction both use a working scene. Policy-side robot_name is
# left as recorded. Pass --env_robot_name explicitly to bypass the remap.
_STALE_ENV_ROBOT_NAMES = {
    "robot_iphone_w_engine_new": "robot_iphone_w_engine_curtain",
}


def format_sim_launch_command(
    *,
    env_task: str,
    robot_name: str | None,
    port: int,
    eval_benchmark_repo_id: str | None,
) -> str:
    """The launch_nodes.py invocation this script expects.

    It starts the splatsim server on ``port`` (matching env task / robot /
    benchmark).
    """
    variant = _TASK_TO_ROBOT_VARIANT.get(env_task, f"<launch_nodes.py robot variant for task '{env_task}'>")
    lines = [
        "cd ~/code/SplatSim && python -u scripts/launch_nodes.py \\",
        f"    --robot {variant} \\",
        f"    --robot_port {port} \\",
    ]
    if robot_name:
        lines.append(f"    --robot_name {robot_name} \\")
    if eval_benchmark_repo_id:
        lines.append(f"    --eval_benchmark_repo_id {eval_benchmark_repo_id} \\")
    # --sync_physics_to_client: physics steps only on client commands, so the
    # sim never races ahead in wallclock time while the policy is thinking.
    # Without it, slow policies produce jumpy rollouts that misrepresent them.
    #
    # --strict_goal_tolerances: recording-grade success thresholds (planar
    # 1 cm vs the loose 60 mm eval default). Matches the sim the DAgger
    # orchestrator launches for intervention recording + blending, so
    # sweep-parity rollouts don't terminate (and freeze into post-success
    # hold) as soon as the arm is merely "close enough" to the goal. Applied
    # at server startup only — no runtime toggle.
    lines.append("    --headless --control_gui --sync_physics_to_client --strict_goal_tolerances")
    return "\n".join(lines)


def check_sim_server_reachable(host: str, port: int, launch_hint: str) -> None:
    """Fail fast if nothing is listening on host:port.

    A ZMQ REQ socket never errors on a dead endpoint — the first reset
    request just queues forever, so without this check a missing server
    looks like a silent freeze.
    """
    import socket

    try:
        with socket.create_connection((host, port), timeout=2):
            pass
    except OSError:
        raise SystemExit(
            f"\nNo splatsim server listening on {host}:{port} — the ZMQ client "
            f"would hang silently. Launch the sim in another terminal, then "
            f"re-run this script:\n\n{launch_hint}\n"
        )


# ── env construction ──────────────────────────────────────────────────────────


def build_splatsim_env(
    *,
    task: str,
    robot_name: str,
    camera_names: list[str],
    image_resize_modes: list[str],
    fps: int,
    episode_length: int,
    external_port: int | None,
    external_host: str = "127.0.0.1",
    eval_benchmark_repo_id: str | None = None,
    eval_benchmark_subset: list[int] | None = None,
    num_dofs: int | None = None,
    state_dim: int | None = None,
    action_dim: int | None = None,
    env_state_dim: int | None = None,
    policy_cfg: Any,
):
    """Build a splatsim vec env (n_envs=1) plus the env-specific pre/post processors.

    When ``external_port`` is set the env connects to an already-running splatsim
    server via ZMQ; ``eval_benchmark_repo_id`` and ``eval_benchmark_subset`` are
    configured on the server side and are ignored here.

    ``num_dofs`` / ``state_dim`` / ``action_dim`` / ``env_state_dim`` must match
    what the sim server actually publishes — the SplatsimEnv defaults are
    UR5-shaped (6/7/7/0), and gymnasium's SyncVectorEnv pre-allocates its obs
    buffer from the declared observation_space, so a mismatched server (e.g.
    planar arm: 3/4/4/8) fails at the first reset with "Output array is the
    wrong shape". ``None`` keeps the config default.

    Returns (vec_env, env_cfg, env_preprocessor, env_postprocessor).
    """
    dim_overrides = {
        k: v
        for k, v in {
            "num_dofs": num_dofs,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "env_state_dim": env_state_dim,
        }.items()
        if v is not None
    }
    env_cfg = make_env_config(
        "splatsim",
        task=task,
        robot_name=robot_name,
        camera_names=camera_names,
        image_resize_modes=image_resize_modes,
        fps=fps,
        episode_length=episode_length,
        external_port=external_port,
        external_host=external_host,
        eval_benchmark_repo_id=eval_benchmark_repo_id,
        eval_benchmark_subset=eval_benchmark_subset,
        **dim_overrides,
    )
    env_dict = make_env(env_cfg, n_envs=1, use_async_envs=False)
    vec_env = env_dict["splatsim"][0]
    env_pre, env_post = make_env_pre_post_processors(env_cfg, policy_cfg)
    return vec_env, env_cfg, env_pre, env_post


# ── action chunk collection ───────────────────────────────────────────────────


@torch.no_grad()
def get_sim_action_chunk_for_ratio(
    wrapper,
    obs_preprocessor,
    vec_env,
    env_preprocessor,
    env_postprocessor,
    *,
    seed_joint_state: np.ndarray,
    seed_joint_velocity: np.ndarray | None = None,
    episode_index_for_seed: int,
    guidance_actions_raw: np.ndarray,
    ratio: float,
    blend_interval_frac: float,
    base_noise: torch.Tensor | None,
    total_steps: int,
    rename_map: dict[str, str],
    device: str,
    task_description: str | None,
    progress_guidance: bool = False,
    progress_guidance_window: int = 45,
    progress_guidance_soft_hold: float = 0.0,
    progress_guidance_hard_lag: int = 8,
    blend_ratio_goal_taper: int = 0,
    guidance_from_dart_labels: bool = False,
    blend_dev_regulation: bool = False,
    blend_dev_full_below: float = 3.0,
    blend_dev_zero_above: float = 8.0,
    demo_states_raw: np.ndarray | None = None,
    frame_sink: dict[str, list[np.ndarray]] | None = None,
    expected_env_state: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Thin adapter over :func:`lib_sa_rollout.run_blended_rollout`.

    The SAME core the sweep's blend step (``augment_dataset_with_blending``)
    executes. This wrapper only (a) derives the blend mode from ``--blend_interval_frac``
    (1.0 → ONCE_PER_CHUNK; anything below → EVERY_STEP, whose re-blend cadence
    the rollout core throttles to every
    ``ceil(blend_interval_frac * n_action_steps)`` ticks) and (b) pins the
    benchmark scenario on EVERY per-ratio reset via
    ``benchmark_start_index`` (a bare seeded reset would let the server's
    EVAL_BENCHMARK counter advance one scenario per reset).

    ``frame_sink``: when given, every tick's camera images (all non-``_stretch``
    keys in ``env_obs['pixels']``) are appended to ``frame_sink[key]`` — one
    RGB uint8 (H, W, 3) frame per tick, including frozen post-success hold
    ticks, so video length always matches the plotted trajectory length.
    """
    on_step = None
    if frame_sink is not None:

        def on_step(t: int, env_obs: dict, action_1d: np.ndarray, is_hold: bool) -> None:
            del t, action_1d, is_hold
            for key, img in (env_obs.get("pixels") or {}).items():
                if key.endswith("_stretch") or img is None:
                    continue
                frame = np.asarray(img)
                if frame.ndim == 4:  # vec-env batched (1, H, W, 3)
                    frame = frame[0]
                frame_sink.setdefault(key, []).append(frame.copy())

    result = run_blended_rollout(
        wrapper=wrapper,
        obs_preprocessor=obs_preprocessor,
        vec_env=vec_env,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        seed_joint_state=seed_joint_state,
        seed_joint_velocity=seed_joint_velocity,
        guidance_actions_raw=guidance_actions_raw,
        ratio=ratio,
        blend_mode=BlendMode.ONCE_PER_CHUNK if blend_interval_frac >= 1.0 else BlendMode.EVERY_STEP,
        blend_interval_frac=blend_interval_frac,
        total_steps=total_steps,
        rename_map=rename_map,
        device=device,
        task_description=task_description,
        seed=[episode_index_for_seed],
        benchmark_start_index=episode_index_for_seed,
        base_noise=base_noise,
        progress_guidance=progress_guidance,
        progress_guidance_window=progress_guidance_window,
        progress_guidance_soft_hold=progress_guidance_soft_hold,
        progress_guidance_hard_lag=progress_guidance_hard_lag,
        blend_ratio_goal_taper=blend_ratio_goal_taper,
        guidance_from_dart_labels=guidance_from_dart_labels,
        blend_dev_regulation=blend_dev_regulation,
        blend_dev_full_below=blend_dev_full_below,
        blend_dev_zero_above=blend_dev_zero_above,
        demo_states_raw=demo_states_raw,
        expected_env_state=expected_env_state,
        on_step=on_step,
    )
    return result.raw_actions, result.decoded_guidance_full


@torch.no_grad()
def get_sim_action_chunks_for_ratios(
    wrapper,
    obs_preprocessor,
    vec_env,
    env_preprocessor,
    env_postprocessor,
    *,
    seed_joint_state: np.ndarray,
    seed_joint_velocity: np.ndarray | None = None,
    episode_index_for_seed: int,
    guidance_actions_raw: np.ndarray,
    ratios: list[float],
    rename_map: dict[str, str],
    device: str,
    task_description: str | None,
    blend_interval_frac: float,
    total_steps: int,
    fixed_base_noise: bool = True,
    progress_guidance: bool = False,
    progress_guidance_window: int = 45,
    progress_guidance_soft_hold: float = 0.0,
    progress_guidance_hard_lag: int = 8,
    blend_ratio_goal_taper: int = 0,
    guidance_from_dart_labels: bool = False,
    blend_dev_regulation: bool = False,
    blend_dev_full_below: float = 3.0,
    blend_dev_zero_above: float = 8.0,
    demo_states_raw: np.ndarray | None = None,
    record_videos: bool = True,
    expected_env_state: np.ndarray | None = None,
) -> tuple[dict[float, np.ndarray], dict[float, np.ndarray], dict[float, dict[str, list[np.ndarray]]]]:
    """Run :func:`get_sim_action_chunk_for_ratio` for each ratio.

    ``fixed_base_noise=True`` (default) pins ONE noise draw shared across all
    ratios AND all ticks — the tool's historical cross-ratio comparability
    mode. ``False`` matches the sweep's blend default: the wrapper draws a
    FRESH torch.randn internally on every denoise (base_noise=None), so
    consecutive every_step samples are independent — expect shake.

    ``record_videos=True`` additionally collects each rollout's camera frames
    (every non-``_stretch`` key in the env's pixels dict) and returns them as
    the third element: ``{ratio: {image_key: [frame, ...]}}``.
    """
    # DAG_GLOBAL_SEED overrides the global torch seed (default 42): the DDPM
    # sampler draws its per-step variance noise from the GLOBAL stream, so at
    # ratio 1.0 passthrough (inner select_action takes no generator) this is
    # the ONLY seed that varies the rollout — --sample_seed does nothing there.
    _gseed = int(os.environ.get("DAG_GLOBAL_SEED", "42"))
    torch.manual_seed(_gseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_gseed)
    base_noise: torch.Tensor | None = None
    if fixed_base_noise:
        if getattr(wrapper.config, "max_action_dim", None) is not None:
            noise_shape = (1, wrapper.config.chunk_size, wrapper.config.max_action_dim)
        else:
            action_dim = wrapper.config.output_features["action"].shape[0]
            noise_shape = (1, wrapper.config.horizon, action_dim)
        base_noise = torch.randn(noise_shape, device=device)

    results: dict[float, np.ndarray] = {}
    decoded_guidance_by_ratio: dict[float, np.ndarray] = {}
    frames_by_ratio: dict[float, dict[str, list[np.ndarray]]] = {}

    progress = tqdm(
        ratios,
        desc="Computing sim action chunks",
        unit="ratio",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
    )
    for ratio in progress:
        progress.set_postfix_str(f"ratio={ratio:.2f}")
        # Re-seed the global RNG at the START of every rollout, not just once
        # before the loop. DDPM's noise_scheduler.step() draws fresh variance
        # noise from the global RNG at EVERY denoising step (base_noise only
        # pins the initial sample), so without a per-rollout re-seed each
        # ratio continues the stream where the previous rollout left it and
        # blends against a DIFFERENT policy sample — breaking cross-ratio
        # comparability (blends land outside the [guidance, policy] envelope).
        torch.manual_seed(_gseed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(_gseed)
        actions, decoded = get_sim_action_chunk_for_ratio(
            wrapper,
            obs_preprocessor,
            vec_env,
            env_preprocessor,
            env_postprocessor,
            seed_joint_state=seed_joint_state,
            seed_joint_velocity=seed_joint_velocity,
            episode_index_for_seed=episode_index_for_seed,
            guidance_actions_raw=guidance_actions_raw,
            ratio=ratio,
            blend_interval_frac=blend_interval_frac,
            base_noise=base_noise,
            total_steps=total_steps,
            rename_map=rename_map,
            device=device,
            task_description=task_description,
            progress_guidance=progress_guidance,
            progress_guidance_window=progress_guidance_window,
            progress_guidance_soft_hold=progress_guidance_soft_hold,
            progress_guidance_hard_lag=progress_guidance_hard_lag,
            blend_ratio_goal_taper=blend_ratio_goal_taper,
            guidance_from_dart_labels=guidance_from_dart_labels,
            blend_dev_regulation=blend_dev_regulation,
            blend_dev_full_below=blend_dev_full_below,
            blend_dev_zero_above=blend_dev_zero_above,
            demo_states_raw=demo_states_raw,
            frame_sink=frames_by_ratio.setdefault(ratio, {}) if record_videos else None,
            expected_env_state=expected_env_state,
        )
        results[ratio] = actions
        if decoded is not None:
            decoded_guidance_by_ratio[ratio] = decoded

    return results, decoded_guidance_by_ratio, frames_by_ratio


# ── video writing ─────────────────────────────────────────────────────────────


def _write_rollout_mp4(frames: list[np.ndarray], fps: float, out_path: Path) -> None:
    """Write RGB uint8 frames as an H.264 + yuv420p mp4 (VSCode/browser-safe).

    Same encode pipeline as extract_dataset_videos.py: cv2 writes a temp mp4v
    file, ffmpeg re-encodes to libx264 + yuv420p + faststart. If ffmpeg is not
    on PATH, the mp4v temp file is kept as the output instead (still playable
    in VLC, just not in VSCode's built-in viewer).
    """
    import shutil
    import subprocess
    import tempfile

    import cv2  # type: ignore[import-not-found]

    if not frames:
        raise ValueError(f"No frames collected for {out_path.name}")
    height, width = frames[0].shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix="sa_sim_video_", dir=out_path.parent)
    tmp_video = Path(tmp_dir) / "raw_mp4v.mp4"

    writer = cv2.VideoWriter(str(tmp_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError(f"Failed to open temp video writer: {tmp_video}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        shutil.move(str(tmp_video), str(out_path))
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"WARNING: ffmpeg not found — kept mp4v encoding for {out_path}")
        return
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


def _annotate_ratio(frame: np.ndarray, ratio: float) -> np.ndarray:
    """Return a copy of an RGB frame with 'ratio=X.XX' stamped in the top-left corner."""
    import cv2  # type: ignore[import-not-found]

    annotated = frame.copy()
    height = annotated.shape[0]
    scale = max(0.5, height / 480.0)
    thickness = max(1, int(round(2 * scale)))
    origin = (int(10 * scale), int(30 * scale))
    text = f"ratio={ratio:.2f}"
    # Black outline behind white text so it reads on any background.
    cv2.putText(
        annotated, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA
    )
    cv2.putText(
        annotated, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA
    )
    return annotated


# ── main ──────────────────────────────────────────────────────────────────────


def _parse_bool(s: str) -> bool:
    """Accept the same `--flag=true/false` spelling the blend script's draccus CLI uses."""
    v = s.strip().lower()
    if v in ("1", "true", "yes", "y"):
        return True
    if v in ("0", "false", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean, got {s!r}")


def _parse_blend_interval_frac(s: str) -> float:
    """Blend cadence in [0, 1]; also accepts legacy true/false spellings."""
    v = s.strip().lower()
    if v in ("true", "yes", "y"):
        return 1.0
    if v in ("false", "no", "n"):
        return 0.0
    try:
        f = float(s)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Expected a float in [0, 1] or true/false, got {s!r}") from e
    if not 0.0 <= f <= 1.0:
        raise argparse.ArgumentTypeError(f"--blend_interval_frac must be in [0, 1], got {s!r}")
    return f


def parse_args():
    """Build and parse the CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Sim-in-the-loop visualization of SharedAutonomyPolicyWrapper predictions. "
            "Requires an external splatsim ZMQ server (see script docstring)."
        )
    )
    parser.add_argument("--policy_path", required=True)
    parser.add_argument(
        "--dataset_repo_id",
        default=None,
        help=(
            "HuggingFace dataset repo ID. If omitted, auto-resolved from the "
            "checkpoint's train_config.json (dataset.repo_id)."
        ),
    )
    parser.add_argument("--dataset_dir", default=None)
    parser.add_argument(
        "--task_description",
        default=None,
        help=(
            "Task description string for PI0.5 preprocessing. If omitted, "
            "resolved from the dataset's tasks.parquet, falling back to "
            "--env_task."
        ),
    )
    parser.add_argument("--episode_index", type=int, default=None)
    parser.add_argument(
        "--frame_index",
        type=int,
        default=0,
        help=(
            "Starting frame within episode: guidance is sliced from here AND the robot is "
            "teleported to the demo's pose at this frame (works over ZMQ — the SplatSim "
            "server dispatches teleport_joint_state). Default 0 = replay from the episode "
            "start (sweep-parity). -1 = random."
        ),
    )
    parser.add_argument(
        "--forward_flow_ratios", nargs="+", type=float, default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    )
    parser.add_argument(
        "--image_resize_mode",
        default="letterbox",
        choices=["stretch", "letterbox"],
        help="Parquet column lookup and rename_map default.",
    )
    parser.add_argument(
        "--camera_names",
        nargs="+",
        default=None,
        help=(
            "Defaults to the checkpoint train_config's env.camera_names "
            "(may be empty for state-only policies), falling back to "
            "['base_rgb', 'wrist_rgb']."
        ),
    )
    parser.add_argument("--rename_map", type=json.loads, default=None)
    parser.add_argument(
        "--robot_name",
        default=None,
        help=(
            "Defaults to the checkpoint train_config's env.robot_name, "
            "falling back to 'robot_iphone_w_engine_curtain'."
        ),
    )
    parser.add_argument(
        "--num_dofs",
        type=int,
        default=None,
        help="Defaults to the checkpoint train_config's env.num_dofs, falling back to 6.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_action_steps", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--no_show", action="store_true")
    parser.add_argument(
        "--blend_interval_frac",
        "--drain_chunk",  # deprecated alias
        dest="blend_interval_frac",
        type=_parse_blend_interval_frac,
        nargs="?",
        const=1.0,
        default=0.0,
        help=(
            "Interval between guidance re-blends, as a fraction of the executed chunk "
            "(n_action_steps): 1.0 (or the bare flag) = blend once per chunk, 0.0 (default) = "
            "re-blend every step, fraction f = re-blend every ceil(f * n_action_steps) ticks "
            "(e.g. 0.5 = twice per chunk). --drain_chunk is a deprecated alias; legacy "
            "true/false spellings accepted."
        ),
    )
    parser.add_argument("--blend_strategy", default="denoise", choices=["denoise", "interpolate"])
    parser.add_argument("--guidance_repr", default="absolute_pos", choices=["absolute_pos", "delta"])
    # Tube re-blend, matching the blend pipeline (wrapper default is 8.0; the
    # orchestrate lineages run 16.0). 0 disables. The viz never ABORTS on
    # breach (visualize everything); only the predictive ratio-cut re-blend
    # is mirrored here.
    parser.add_argument(
        "--blend_tube_steps",
        type=float,
        default=None,
        help="Tube radius in demo med-steps for the predictive re-blend "
        "(None = keep wrapper default 8.0; pipeline uses 16.0; 0 disables)",
    )
    parser.add_argument("--guidance_from_dart_labels", type=lambda s: s.lower() != "false", default=False)
    parser.add_argument("--anchor_prefix_steps", type=int, default=0)
    parser.add_argument("--anchor_suffix_steps", type=int, default=0)
    parser.add_argument("--anchor_suffix_to_goal", type=lambda s: s.lower() == "true", default=False)
    parser.add_argument("--anchor_every_denoise_step", type=lambda s: s.lower() != "false", default=True)
    parser.add_argument(
        "--total_steps",
        type=int,
        default=None,
        help=(
            "Rollout length in ticks. Default: the remaining episode length from --frame_index "
            "(sweep-parity: the blend script replays whole episodes). The pre-parity behavior "
            "was one chunk length (pass --total_steps=64 to recover it)."
        ),
    )
    # ── Sweep-parity blend knobs (same names/semantics as augment_dataset_with_blending) ──
    parser.add_argument(
        "--progress_guidance",
        type=_parse_bool,
        default=False,
        help=(
            "Re-index the demo by robot progress (monotonic windowed nearest-state match) "
            "instead of wall-clock, exactly as the blend script does. Requires "
            "observation.state in the dataset for state-grid matching (falls back to the "
            "action matrix with a +1 shift)."
        ),
    )
    parser.add_argument(
        "--blend_dev_regulation",
        type=lambda x: x.lower() in ("1", "true", "yes"),
        default=False,
        help="tube-regulate the ratio by corridor deviation (requested ratio = maximum)",
    )
    parser.add_argument(
        "--blend_dev_full_below",
        type=float,
        default=3.0,
        help="full ratio while deviation <= this many med_steps",
    )
    parser.add_argument(
        "--blend_dev_zero_above",
        type=float,
        default=8.0,
        help="ratio annealed to 0 at this deviation (med_steps)",
    )
    parser.add_argument(
        "--blend_ratio_goal_taper",
        type=int,
        default=0,
        help="anneal the blend ratio to 0 over the last N demo indices (0 = off)",
    )
    parser.add_argument(
        "--progress_guidance_soft_hold",
        type=float,
        default=0.0,
        help="clock advance rate (x demo pace) while lag in (lag_tol, hard_lag); 0 = legacy hard hold",
    )
    parser.add_argument(
        "--progress_guidance_hard_lag",
        type=int,
        default=8,
        help="lag (demo indices) beyond which the clock hard-holds regardless of soft_hold",
    )
    parser.add_argument("--progress_guidance_window", type=int, default=45)
    parser.add_argument(
        "--fixed_base_noise",
        type=_parse_bool,
        default=True,
        help=(
            "true (default): pin ONE noise draw shared across all ratios and ticks — the "
            "tool's historical cross-ratio comparability mode. false: sweep-parity — the "
            "wrapper draws fresh noise every denoise (independent every_step samples)."
        ),
    )
    parser.add_argument(
        "--record_videos",
        type=_parse_bool,
        default=True,
        help=(
            "Record each ratio rollout's camera observations (every non-stretch image key the "
            "env returns, e.g. base_rgb_letterbox / wrist_rgb_letterbox) as mp4 files in the "
            "output dir. When the policy itself uses no cameras, --video_camera_names are "
            "requested from the sim just for recording."
        ),
    )
    parser.add_argument(
        "--video_camera_names",
        nargs="+",
        default=None,
        help=(
            "Cameras to request from the sim for video recording when the env would otherwise "
            "have none (state-only policies). Defaults to ['base_rgb', 'wrist_rgb']."
        ),
    )
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=42,
        help=(
            "Seed a fresh torch.Generator for EVERY blend-path model call (prior draw + all "
            "scheduler.step variance draws), so each call uses identical randomness and "
            "consecutive per-tick samples differ only via the observation — no fresh-sample "
            "shake, fully reproducible. Pass -1 to disable (legacy global-RNG sampling)."
        ),
    )
    parser.add_argument(
        "--clip_sample",
        type=_parse_bool,
        default=None,
        help=(
            "DEBUG: override the checkpoint's DDPM clip_sample (None = keep trained value). "
            "The sweep's blend step often runs --clip_sample=false."
        ),
    )
    parser.add_argument(
        "--rtc_prev_chunk",
        type=_parse_bool,
        nargs="?",
        const=True,
        default=False,
        help=(
            "RTC-style previous-chunk guidance: pass the previous blended chunk's unexecuted "
            "remainder into each re-blend's denoise (annealed gradient correction per step) so "
            "consecutive chunks commit to the same mode — cross-tick consistency without pinning "
            "the noise. Additive to the x_tsw ratio blend; diffusion policies only. "
            "See SharedAutonomyConfig.rtc_* for the knobs below."
        ),
    )
    parser.add_argument("--rtc_max_guidance_weight", type=float, default=10.0)
    parser.add_argument("--rtc_execution_horizon", type=int, default=None)
    parser.add_argument("--rtc_inference_delay", type=int, default=0)
    parser.add_argument(
        "--resample_noise_per_reblend",
        type=lambda x: str(x).lower() in ("1", "true", "yes"),
        default=False,
        help="With --sample_seed set: give each re-blend a FRESH but "
        "reproducible noise draw (salt the seed by a rebuild counter) instead "
        "of every rebuild sharing one draw.",
    )
    parser.add_argument(
        "--anchor_from_prev_blend",
        type=lambda x: str(x).lower() in ("1", "true", "yes"),
        default=False,
        help="UNPINNED anchor: base each re-blend on the previous BLENDED chunk "
        "(re-anchored) instead of a fresh pure-policy predict — deliberate "
        "contraction onto the guidance (DART-style perturb-and-recover).",
    )
    parser.add_argument(
        "--rtc_prefix_attention_schedule", choices=["linear", "exp", "zeros", "ones"], default="linear"
    )

    # ── Env / simulator config ────────────────────────────────────────────────
    parser.add_argument(
        "--env_task",
        default=None,
        help=(
            "Defaults to the checkpoint train_config's env.task, falling back to 'upright_small_engine_new'."
        ),
    )
    parser.add_argument("--env_robot_name", default=None, help="Defaults to --robot_name.")
    parser.add_argument("--env_camera_names", nargs="+", default=None, help="Defaults to --camera_names.")
    parser.add_argument(
        "--env_image_resize_modes", nargs="+", default=None, help="Defaults to [--image_resize_mode]."
    )
    parser.add_argument("--env_fps", type=int, default=30)
    parser.add_argument("--env_episode_length", type=int, default=1_000_000)
    parser.add_argument(
        "--env_external_port",
        type=int,
        default=6001,
        help=(
            "ZMQ port of the already-running splatsim server. The server must be "
            "launched separately (see script docstring). Default: 6001."
        ),
    )
    parser.add_argument("--env_external_host", default="127.0.0.1")

    # WRAPPER PASSTHROUGH: unrecognized `--name=value` / `--name value`
    # arguments are applied directly as `wrapper.<name> = <coerced value>`
    # after the wrapper is built (see _apply_wrapper_passthrough) — new
    # SharedAutonomyPolicyWrapper knobs work here without touching this
    # script. Unknown attribute names and single-dash arguments still fail
    # loudly, so typos are caught.
    args, unknown = parser.parse_known_args()
    passthrough: list[tuple[str, str]] = []
    i = 0
    while i < len(unknown):
        tok = unknown[i]
        if not tok.startswith("--"):
            parser.error(f"unrecognized argument: {tok}")
        name, eq, val = tok[2:].partition("=")
        if not eq:
            if i + 1 >= len(unknown) or unknown[i + 1].startswith("--"):
                parser.error(f"passthrough argument --{name} needs a value")
            val = unknown[i + 1]
            i += 1
        passthrough.append((name.replace("-", "_"), val))
        i += 1
    args._wrapper_passthrough = passthrough
    return args


def _apply_wrapper_passthrough(wrapper, passthrough: list[tuple[str, str]]) -> None:
    """Set unrecognized CLI args as wrapper attributes.

    Type-coerced from the attribute's current value (bool/int/float/str;
    'none' -> None). Errors on names the wrapper does not already have —
    typos must not pass silently.
    """
    for name, raw in passthrough:
        if not hasattr(wrapper, name):
            raise SystemExit(
                f"--{name} is neither a script argument nor a SharedAutonomyPolicyWrapper "
                f"attribute — typo, or the knob does not exist."
            )
        cur = getattr(wrapper, name)
        low = raw.strip().lower()
        value: object
        if low in ("none", "null"):
            value = None
        elif isinstance(cur, bool):
            value = low in ("1", "true", "yes")
        elif isinstance(cur, int) and not isinstance(cur, bool):
            value = int(raw)
        elif isinstance(cur, float):
            value = float(raw)
        elif cur is None:
            try:
                value = float(raw) if "." in raw or "e" in low else int(raw)
            except ValueError:
                value = raw
        else:
            value = raw
        setattr(wrapper, name, value)
        print(f"[passthrough] wrapper.{name} = {value!r}")


def main():
    """CLI entry point."""
    args = parse_args()

    # Load the checkpoint's train_config.json once — used to auto-resolve both
    # the dataset repo id and the env/robot settings below.
    train_cfg_path = Path(args.policy_path) / "train_config.json"
    train_cfg: dict = {}
    if train_cfg_path.is_file():
        try:
            train_cfg = json.loads(train_cfg_path.read_text())
        except (json.JSONDecodeError, OSError):
            train_cfg = {}

    # Auto-resolve env/robot settings from the checkpoint's env section.
    # Explicit CLI flags win; the historical hardcoded defaults are the
    # last-resort fallback for checkpoints without an env section.
    env_json = train_cfg.get("env") or {}

    def _auto(flag: str, cli_value, cfg_key: str, fallback):
        if cli_value is not None:
            return cli_value
        value = env_json.get(cfg_key)
        if value is None:
            return fallback
        print(f"Auto-resolved --{flag} from checkpoint: {value}")
        return value

    args.env_task = _auto("env_task", args.env_task, "task", "upright_small_engine_new")
    args.robot_name = _auto("robot_name", args.robot_name, "robot_name", "robot_iphone_w_engine_curtain")
    args.num_dofs = _auto("num_dofs", args.num_dofs, "num_dofs", 6)
    args.camera_names = _auto("camera_names", args.camera_names, "camera_names", ["base_rgb", "wrist_rgb"])

    env_robot_name = args.env_robot_name or args.robot_name
    if args.env_robot_name is None and env_robot_name in _STALE_ENV_ROBOT_NAMES:
        _remapped = _STALE_ENV_ROBOT_NAMES[env_robot_name]
        print(
            f"env robot_name '{env_robot_name}' is a stale splat — using "
            f"'{_remapped}' for the sim (pass --env_robot_name to override)."
        )
        env_robot_name = _remapped
    env_camera_names = args.env_camera_names or list(args.camera_names)
    env_image_resize_modes = args.env_image_resize_modes or [args.image_resize_mode]
    if args.record_videos:
        # Make sure the sim renders SOMETHING to record even when the policy is
        # state-only (planar checkpoints auto-resolve camera_names=[]). Extra
        # image keys in the obs are harmless to the policy — its preprocessor
        # only consumes the features it was trained on.
        video_cams = args.video_camera_names or ["base_rgb", "wrist_rgb"]
        env_camera_names = env_camera_names + [c for c in video_cams if c not in env_camera_names]

    # Remind how to start the sim this script needs, then fail fast if it isn't
    # up yet (a bare ZMQ connect to a dead port hangs with no error).
    sim_launch_hint = format_sim_launch_command(
        env_task=args.env_task,
        robot_name=env_robot_name,
        port=args.env_external_port,
        eval_benchmark_repo_id=env_json.get("eval_benchmark_repo_id"),
    )
    print(
        f"\nThis script requires a splatsim server already running on "
        f"{args.env_external_host}:{args.env_external_port}. To launch it for "
        f"this policy (in another terminal):\n\n{sim_launch_hint}\n"
    )
    check_sim_server_reachable(args.env_external_host, args.env_external_port, sim_launch_hint)

    # Auto-resolve dataset_repo_id from the checkpoint if not passed. Prevents
    # the silent dataset-mismatch bug (e.g. dataset-11 checkpoint visualized
    # against dataset-7 frames).
    dataset_repo_id = args.dataset_repo_id
    if dataset_repo_id is None:
        _ds_cfg = train_cfg.get("dataset", {})
        # Weighted-sampling checkpoints leave repo_id empty and list their
        # sub-datasets in repo_ids; source[0] is the base dataset by
        # orchestrator convention.
        dataset_repo_id = _ds_cfg.get("repo_id") or None
        if dataset_repo_id is None and _ds_cfg.get("repo_ids"):
            dataset_repo_id = _ds_cfg["repo_ids"][0]
        if dataset_repo_id is None:
            raise SystemExit(
                f"Could not auto-resolve --dataset_repo_id from "
                f"{train_cfg_path}. Pass --dataset_repo_id explicitly."
            )
        print(f"Auto-resolved --dataset_repo_id from checkpoint: {dataset_repo_id}")

    dataset_dir = resolve_dataset_dir(dataset_repo_id, args.dataset_dir)
    print(f"Dataset dir: {dataset_dir}")
    task_map = load_task_description(dataset_dir)
    print(f"Task map: {task_map}")

    rename_map = args.rename_map or make_default_rename_map(args.camera_names, args.image_resize_mode)
    print(f"Rename map: {rename_map}")

    print(f"Loading policy from {args.policy_path} …")
    wrapper, obs_preprocessor = load_wrapped_policy(
        policy_path=args.policy_path,
        forward_flow_ratio=1.0,
        robot_name=args.robot_name,
        num_dofs=args.num_dofs,
        device=args.device,
    )
    wrapper.guidance_blend_strategy = GuidanceBlendStrategy(args.blend_strategy)
    if args.sample_seed >= 0:
        wrapper.sample_seed = args.sample_seed
        print(f"Per-call sample generator enabled (seed={args.sample_seed}).")
    wrapper.policy_guidance_representation = PolicyGuidanceRepresentation(args.guidance_repr)
    if args.blend_tube_steps is not None:
        wrapper.blend_tube_steps = float(args.blend_tube_steps)
    wrapper.anchor_prefix_steps = args.anchor_prefix_steps
    wrapper.anchor_suffix_steps = args.anchor_suffix_steps
    wrapper.anchor_suffix_to_goal = args.anchor_suffix_to_goal
    wrapper.anchor_every_denoise_step = args.anchor_every_denoise_step
    wrapper.rtc_prev_chunk_guidance = args.rtc_prev_chunk
    wrapper.rtc_max_guidance_weight = args.rtc_max_guidance_weight
    wrapper.rtc_execution_horizon = args.rtc_execution_horizon
    wrapper.rtc_inference_delay = args.rtc_inference_delay
    wrapper.anchor_from_prev_blend = args.anchor_from_prev_blend
    wrapper.resample_noise_per_reblend = args.resample_noise_per_reblend
    _apply_wrapper_passthrough(wrapper, getattr(args, "_wrapper_passthrough", []))
    wrapper.rtc_prefix_attention_schedule = args.rtc_prefix_attention_schedule
    if args.rtc_prev_chunk:
        # Set post-init, so re-run the wrapper's init-time policy-type check.
        if getattr(wrapper.inner_policy.config, "type", None) != "diffusion":
            raise SystemExit("--rtc_prev_chunk requires a diffusion inner policy.")
        print(
            f"RTC prev-chunk guidance ON (max_gw={args.rtc_max_guidance_weight}, "
            f"exec_horizon={args.rtc_execution_horizon}, delay={args.rtc_inference_delay}, "
            f"schedule={args.rtc_prefix_attention_schedule})."
        )
    wrapper.skip_collision = True
    apply_clip_sample_override(wrapper, args.clip_sample)
    if args.n_action_steps is not None:
        prev = wrapper.config.n_action_steps
        wrapper.config.n_action_steps = args.n_action_steps
        print(f"Overrode n_action_steps: {prev} → {args.n_action_steps}")
    n_obs_steps = wrapper.config.n_obs_steps
    n_action_steps = wrapper.config.n_action_steps
    chunk_len = getattr(wrapper.config, "chunk_size", None) or getattr(wrapper.config, "horizon", None)
    if chunk_len is None:
        raise ValueError("Could not determine policy chunk length (chunk_size/horizon).")
    print(
        f"Policy: {wrapper.config.type}, n_obs_steps={n_obs_steps}, "
        f"n_action_steps={n_action_steps}, chunk_len={chunk_len}"
    )

    # Pick episode + frame.
    if args.episode_index is None:
        available = get_available_episodes(dataset_dir, min_episode_index=0)
        if not available:
            raise RuntimeError(f"No episodes found in {dataset_dir}.")
        episode_index = random.choice(available)
        print(f"Selected random episode: {episode_index}")
    else:
        episode_index = args.episode_index
        print(f"Using episode: {episode_index}")

    parquet_files = find_parquet_files(dataset_dir)
    ep_df_list = [
        pd.read_parquet(
            f, columns=["episode_index", "frame_index"], filters=[("episode_index", "==", episode_index)]
        )
        for f in parquet_files
    ]
    ep_df_list = [d for d in ep_df_list if len(d) > 0]
    if not ep_df_list:
        raise ValueError(f"Episode {episode_index} not found.")
    ep_info = pd.concat(ep_df_list).sort_values("frame_index").reset_index(drop=True)
    ep_length = len(ep_info)
    print(f"Episode {episode_index} has {ep_length} frames.")

    if args.frame_index == -1:
        n_needed_min = n_obs_steps + chunk_len
        max_start = ep_length - n_needed_min
        if max_start < 0:
            raise ValueError(f"Episode {episode_index} too short ({ep_length} frames); need {n_needed_min}.")
        frame_index = random.randint(0, max_start)
        print(f"Selected random frame_index: {frame_index}")
    else:
        frame_index = args.frame_index
        if not frame_index >= 0 or frame_index + n_obs_steps + 1 > ep_length:
            raise ValueError(
                f"frame_index={frame_index} leaves no guidance frames in episode "
                f"{episode_index} ({ep_length} frames, n_obs_steps={n_obs_steps})."
            )
        print(f"Using frame_index: {frame_index}")
    if frame_index > 0:
        print(
            f"NOTE: frame_index={frame_index} — the robot will be TELEPORTED to the demo's "
            f"pose at this frame after the scenario reset (SplatSim servers dispatch "
            f"teleport_joint_state over ZMQ), so guidance stays aligned with the robot. "
            f"Non-SplatSim servers without that method fall back to the scenario-initial "
            f"pose (watch for a large initial guidance\u2194robot gap in the plots)."
        )

    # Rollout length: default = the remaining episode (sweep-parity — the blend
    # script replays whole episodes and its guidance is the FULL action matrix).
    if args.total_steps is not None:
        total_steps = args.total_steps
    else:
        total_steps = ep_length - n_obs_steps - frame_index
    n_avail = ep_length - frame_index
    n_needed = min(n_obs_steps + total_steps, n_avail)

    frames_df = load_episode_frames(dataset_dir, episode_index, frame_index, n_needed)
    obs_frames = frames_df.iloc[:n_obs_steps]
    guidance_frames = frames_df.iloc[n_obs_steps:]
    guidance_actions_raw = np.stack(
        [np.array(row["action"], dtype=np.float32) for _, row in guidance_frames.iterrows()]
    )
    demo_states_raw = None
    if "observation.state" in guidance_frames.columns:
        demo_states_raw = np.stack(
            [np.array(row["observation.state"], dtype=np.float32) for _, row in guidance_frames.iterrows()]
        )
    print(
        f"Loaded {len(obs_frames)} obs + {len(guidance_frames)} guidance frames "
        f"(action_dim={guidance_actions_raw.shape[1]}, "
        f"demo_states={'yes' if demo_states_raw is not None else 'no'}); "
        f"rollout total_steps={total_steps}."
    )
    if args.progress_guidance and demo_states_raw is None:
        print(
            "NOTE: --progress_guidance without observation.state in the dataset — matching "
            "against the action matrix with a +1 shift (same fallback as the blend script)."
        )

    # Task resolution chain: --task_description override → per-episode lookup
    # in task_map → --env_task fallback. PI0.5 requires a non-empty task; the
    # final fallback ensures the script can't reach the preprocessor with None.
    task_idx = int(frames_df.iloc[0].get("task_index", 1))
    if args.task_description is not None:
        task_description = args.task_description
        print(f"Task: '{task_description}' (from --task_description override)")
    else:
        task_description = task_map.get(task_idx)
        if task_description:
            print(f"Task: '{task_description}' (task_index={task_idx})")
        else:
            task_description = args.env_task
            print(f"No task in dataset for task_index={task_idx}; using --env_task='{task_description}'")

    seed_joint_state = np.array(obs_frames.iloc[-1]["action"], dtype=np.float32)
    # Handoff velocity at the seed frame (3-frame FD, rad/s) — mirrors the
    # blend script: seed the sim + policy obs history MOVING like the source
    # episode's handoff instead of at rest (rest-seeding lurches the first
    # command and contradicts the source's conditioning).
    seed_joint_velocity = None
    if "observation.state" in frames_df.columns and len(frames_df) > n_obs_steps + 2:
        _s_lo = np.array(frames_df.iloc[n_obs_steps - 1]["observation.state"], dtype=np.float32)
        _s_hi = np.array(frames_df.iloc[n_obs_steps + 2]["observation.state"], dtype=np.float32)
        seed_joint_velocity = (_s_hi - _s_lo) / 3.0 * 30.0

    guidance_actions_raw_for_plot = guidance_actions_raw
    if args.guidance_repr == "delta":
        print("Converting absolute positions to EE deltas for DELTA mode …")
        guidance_actions_raw = absolute_positions_to_ee_deltas(
            wrapper, seed_joint_state, guidance_actions_raw
        )

    # Benchmark scenario: prefer the episode's recorded source_scenario_idx
    # (written by intervention recording; the blend script resolves it the same
    # way) so the rollout runs in the SAME scene geometry the demo was recorded
    # in. Identity fallback (scenario = episode_index) for plain training
    # datasets without the metadata.
    scenario_index = episode_index
    episodes_meta = load_episodes_meta(dataset_dir)
    if not episodes_meta.empty and "source_scenario_idx" in episodes_meta.columns:
        _row = episodes_meta.loc[episodes_meta["episode_index"] == episode_index, "source_scenario_idx"]
        if not _row.empty and pd.notna(_row.iloc[0]):
            scenario_index = int(_row.iloc[0])
            print(f"Resolved benchmark scenario from dataset meta: source_scenario_idx={scenario_index}")
    if scenario_index == episode_index:
        print(
            f"Benchmark scenario = episode_index = {scenario_index} (no source_scenario_idx "
            f"metadata; make sure the server's benchmark scenario {scenario_index} matches "
            f"this episode's scene geometry)."
        )

    # Connect to the external simulator. The server is already running in
    # EVAL_BENCHMARK mode; we select scenarios via reset(seed=[episode_index]).
    print(
        f"Connecting to splatsim at {args.env_external_host}:{args.env_external_port} "
        f"(task={args.env_task}) …"
    )
    # Feature dims from the checkpoint's env config (see build_splatsim_env's
    # docstring — the UR5-shaped defaults break non-UR5 servers at reset).
    # Fallback for checkpoints without an env section: joints + gripper.
    vec_env, _env_cfg, env_pre, env_post = build_splatsim_env(
        task=args.env_task,
        robot_name=env_robot_name,
        camera_names=env_camera_names,
        image_resize_modes=env_image_resize_modes,
        fps=args.env_fps,
        episode_length=args.env_episode_length,
        external_port=args.env_external_port,
        external_host=args.env_external_host,
        eval_benchmark_repo_id=None,  # configured on the server side
        eval_benchmark_subset=None,
        num_dofs=args.num_dofs,
        state_dim=env_json.get("state_dim") or args.num_dofs + 1,
        action_dim=env_json.get("action_dim") or args.num_dofs + 1,
        env_state_dim=env_json.get("env_state_dim") or 0,
        policy_cfg=wrapper.config,
    )
    warn_if_sim_physics_unsynced(vec_env)
    # Warn-only here (the viz is a debug tool and inspecting loose-sim
    # behavior is legitimate); the blend script hard-fails on a loose sim.
    check_sim_strict_goal_tolerances(vec_env, required=False)

    try:
        print(f"Computing sim rollouts for ratios: {args.forward_flow_ratios} …")
        action_chunks, decoded_guidance_by_ratio, frames_by_ratio = get_sim_action_chunks_for_ratios(
            wrapper,
            obs_preprocessor,
            vec_env,
            env_pre,
            env_post,
            seed_joint_state=seed_joint_state,
            seed_joint_velocity=seed_joint_velocity,
            episode_index_for_seed=scenario_index,
            guidance_actions_raw=guidance_actions_raw,
            ratios=args.forward_flow_ratios,
            rename_map=rename_map,
            device=args.device,
            task_description=task_description,
            blend_interval_frac=args.blend_interval_frac,
            total_steps=total_steps,
            fixed_base_noise=args.fixed_base_noise,
            progress_guidance=args.progress_guidance,
            progress_guidance_window=args.progress_guidance_window,
            progress_guidance_soft_hold=args.progress_guidance_soft_hold,
            progress_guidance_hard_lag=args.progress_guidance_hard_lag,
            blend_ratio_goal_taper=args.blend_ratio_goal_taper,
            guidance_from_dart_labels=args.guidance_from_dart_labels,
            blend_dev_regulation=args.blend_dev_regulation,
            blend_dev_full_below=args.blend_dev_full_below,
            blend_dev_zero_above=args.blend_dev_zero_above,
            demo_states_raw=demo_states_raw,
            record_videos=args.record_videos,
            expected_env_state=(
                np.asarray(frames_df.iloc[n_obs_steps]["observation.environment_state"], dtype=np.float64)
                if "observation.environment_state" in frames_df.columns
                else None
            ),
        )
        print("Done computing rollouts.")
    finally:
        close_envs({"splatsim": {0: vec_env}})

    decoded_guidance: np.ndarray | None = None
    if decoded_guidance_by_ratio:
        sample_ratio = next(iter(decoded_guidance_by_ratio.keys()))
        decoded_guidance = decoded_guidance_by_ratio[sample_ratio]
        print(f"Captured decoded guidance overlay from ratio={sample_ratio}.")

    print("Computing EE trajectories via pybullet FK …")
    obs_states_raw = np.stack([np.array(row["action"], dtype=np.float32) for _, row in obs_frames.iterrows()])
    init_obs_state_raw = obs_states_raw[-1]
    ee_trajectories = compute_ee_trajectories(
        wrapper=wrapper, init_obs_state_raw=init_obs_state_raw, action_chunks_by_ratio=action_chunks
    )
    obs_ee_positions = compute_ee_from_states(wrapper, obs_states_raw)
    guidance_ee_positions = compute_ee_from_states(wrapper, guidance_actions_raw_for_plot)
    decoded_guidance_ee_positions = (
        compute_ee_from_states(wrapper, decoded_guidance) if decoded_guidance is not None else None
    )

    action_dim = next(iter(action_chunks.values())).shape[1]
    joint_names = [f"joint_{i + 1}" for i in range(min(args.num_dofs, action_dim))]
    if action_dim > args.num_dofs:
        joint_names.append("gripper")

    if args.output_dir is None:
        policy_tag = (train_cfg.get("policy") or {}).get("type") or "policy"
        repr_tag = "delta" if args.guidance_repr == "delta" else "abspos"
        if args.blend_interval_frac >= 1.0:
            blend_interval_tag = "onestep"
        elif args.blend_interval_frac <= 0.0:
            blend_interval_tag = "everystep"
        else:
            blend_interval_tag = f"blendint{int(round(args.blend_interval_frac * 100)):03d}"
        anchor_tag = (
            f"anchor{args.anchor_prefix_steps}p{args.anchor_suffix_steps}s"
            if (args.anchor_prefix_steps > 0 or args.anchor_suffix_steps > 0)
            else "noanchor"
        )
        if args.anchor_suffix_to_goal:
            anchor_tag += "g"
        nas_tag = f"nas{n_action_steps}"
        noise_tag = "" if args.fixed_base_noise else "_freshnoise"
        pg_tag = "_pg" if args.progress_guidance else ""
        clip_tag = "" if args.clip_sample is None else ("_clip" if args.clip_sample else "_noclip")
        rtc_tag = "_rtc" if args.rtc_prev_chunk else ""
        parent = f"shared_autonomy_sim_ep{episode_index}_frame{frame_index}"
        name = (
            f"{policy_tag}_{args.blend_strategy}_{repr_tag}_{blend_interval_tag}_{anchor_tag}_{nas_tag}"
            f"{noise_tag}{pg_tag}{clip_tag}{rtc_tag}_sim"
        )
        output_dir: Path = Path("outputs/viz") / parent / name
    else:
        output_dir = Path(args.output_dir)
    print(f"Output dir: {output_dir}")
    joint_angles_path = output_dir / "joint_angles.png"
    ee_traj_path = output_dir / "ee_trajectory.html"

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "rollout_data.npz",
        guidance_actions_raw=guidance_actions_raw_for_plot,
        decoded_guidance=decoded_guidance if decoded_guidance is not None else np.array([]),
        obs_states_raw=obs_states_raw,
        **{f"ratio_{r:.2f}": chunk for r, chunk in action_chunks.items()},
    )
    print(f"Saved raw rollout arrays → {output_dir / 'rollout_data.npz'}")

    if frames_by_ratio:
        print("Writing rollout videos …")
        combined_frames: dict[str, list[np.ndarray]] = {}
        for ratio in sorted(frames_by_ratio):
            for image_key, frames in sorted(frames_by_ratio[ratio].items()):
                if not any(frame.any() for frame in frames):
                    # The gym client zero-fills cameras the sim server doesn't
                    # render (e.g. wrist_rgb on the planar arm) — skip them.
                    print(f"Skipping {image_key} at ratio={ratio:.2f}: all-black (camera not rendered)")
                    continue
                video_path = output_dir / f"ratio_{ratio:.2f}_{image_key}.mp4"
                _write_rollout_mp4(frames, fps=float(args.env_fps), out_path=video_path)
                print(f"Saved video → {video_path}")
                combined_frames.setdefault(image_key, []).extend(
                    _annotate_ratio(frame, ratio) for frame in frames
                )
        # One back-to-back compilation per camera: every ratio's rollout in
        # ascending-ratio order, each frame stamped with its ratio.
        for image_key, frames in sorted(combined_frames.items()):
            combined_path = output_dir / f"combined_ratios_{image_key}.mp4"
            _write_rollout_mp4(frames, fps=float(args.env_fps), out_path=combined_path)
            print(f"Saved combined video → {combined_path}")

    print("Plotting joint angles …")
    plot_joint_angles(
        action_chunks_by_ratio=action_chunks,
        joint_names=joint_names,
        episode_index=episode_index,
        frame_index=frame_index,
        obs_states_raw=obs_states_raw,
        guidance_actions_raw=guidance_actions_raw_for_plot,
        decoded_guidance_raw=decoded_guidance,
        output_path=joint_angles_path,
        no_show=args.no_show,
    )

    print("Plotting EE trajectories …")
    plot_ee_trajectories_3d(
        ee_trajectories_by_ratio=ee_trajectories,
        episode_index=episode_index,
        frame_index=frame_index,
        obs_ee_positions=obs_ee_positions,
        guidance_ee_positions=guidance_ee_positions,
        decoded_guidance_ee_positions=decoded_guidance_ee_positions,
        output_path=ee_traj_path,
        no_show=args.no_show,
    )

    # Ratio 1.0 (pure-policy bypass) often dwarfs the blended trajectories'
    # spread — when it's present alongside other ratios, emit a second set of
    # plots without it so the blend comparison stays readable.
    action_chunks_no100 = {r: c for r, c in action_chunks.items() if r < 1.0}
    if 0 < len(action_chunks_no100) < len(action_chunks):
        ee_trajectories_no100 = {r: t for r, t in ee_trajectories.items() if r < 1.0}
        # color_ratios pins each ratio's hue to its color in the full-set plots.
        all_ratios = sorted(action_chunks.keys())
        print("Plotting joint angles (without ratio 1.0) …")
        plot_joint_angles(
            action_chunks_by_ratio=action_chunks_no100,
            joint_names=joint_names,
            episode_index=episode_index,
            frame_index=frame_index,
            obs_states_raw=obs_states_raw,
            guidance_actions_raw=guidance_actions_raw_for_plot,
            decoded_guidance_raw=decoded_guidance,
            output_path=output_dir / "joint_angles_no100.png",
            no_show=args.no_show,
            color_ratios=all_ratios,
        )
        print("Plotting EE trajectories (without ratio 1.0) …")
        plot_ee_trajectories_3d(
            ee_trajectories_by_ratio=ee_trajectories_no100,
            episode_index=episode_index,
            frame_index=frame_index,
            obs_ee_positions=obs_ee_positions,
            guidance_ee_positions=guidance_ee_positions,
            decoded_guidance_ee_positions=decoded_guidance_ee_positions,
            output_path=output_dir / "ee_trajectory_no100.html",
            no_show=args.no_show,
            color_ratios=all_ratios,
        )

    print("Done.")


if __name__ == "__main__":
    main()
