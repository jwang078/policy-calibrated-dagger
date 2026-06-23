#!/usr/bin/env python
"""Sim-in-the-loop variant of visualize_shared_autonomy.py.

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
            --robot_name robot_iphone_w_engine_new \\
            --eval_benchmark_repo_id <benchmark_dataset_repo_id>

Then point this script at it:

    python my_scripts/visualize_shared_autonomy_sim.py \\
        --policy_path .../pretrained_model \\
        --dataset_repo_id JennyWWW/splatsim_approach_lever_7_lowres_5path_10fails \\
        --episode_index 305 \\
        --forward_flow_ratios 0.0 0.05 0.2 0.4 0.8 1.0 \\
        --blend_strategy denoise --guidance_repr delta --drain_chunk \\
        --env_task upright_small_engine_new \\
        --env_external_port 6001

For example:
# 1. Launch splatsim out-of-process (once, stays up)
cd ~/code/SplatSim && python scripts/launch_nodes.py \
    --robot sim_ur_pybullet_small_engine_new_interactive \
    --robot_port 6001 \
    --robot_name robot_iphone_w_engine_new \
    --eval_benchmark_repo_id JennyWWW/eval_splatsim_approach_lever_benchmark_1000

# 2. Run visualize (in another terminal)
python my_scripts/visualize_shared_autonomy_sim.py \
    --policy_path outputs/training/pi05_approach_lever_11_biasend_5path_delta_basewrist/checkpoints/006000/pretrained_model \
    --dataset_repo_id JennyWWW/splatsim_approach_lever_7_lowres_5path_10fails \
    --episode_index 305 \
    --forward_flow_ratios 0.0 0.05 0.2 0.4 0.8 1.0 \
    --blend_strategy denoise --guidance_repr delta --drain_chunk \
    --env_task upright_small_engine_new --env_external_port 6001

The ``--episode_index`` selects which benchmark scenario to load on the server via
``vec_env.reset(seed=[episode_index])``. ``--frame_index`` is used only to slice the
guidance (demo) actions from the dataset; the robot always starts from the episode's
initial pose as seeded by the server.

Imports plotting / IO helpers from the sibling parquet script
(``visualize_shared_autonomy.py``) and batch-building helpers from
``visualize_shared_autonomy_sim.py`` itself (which ``augment_dataset_with_blending.py``
also imports).
"""

from __future__ import annotations

import argparse
import json
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
    load_wrapped_policy,
)

from lerobot.envs import close_envs  # noqa: E402
from lerobot.envs.factory import make_env, make_env_config, make_env_pre_post_processors  # noqa: E402
from lerobot.envs.utils import preprocess_observation  # noqa: E402
from lerobot.policies.shared_autonomy_wrapper import (  # noqa: E402
    BlendMode,
    GuidanceBlendStrategy,
    PolicyGuidanceRepresentation,
)
from lerobot.utils.lerobot_dataset_utils import make_default_rename_map, resolve_dataset_dir  # noqa: E402
from lerobot.utils.sim_seeding import seed_splatsim_env_to_state  # noqa: E402

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
    policy_cfg: Any,
):
    """Build a splatsim vec env (n_envs=1) plus the env-specific pre/post processors.

    When ``external_port`` is set the env connects to an already-running splatsim
    server via ZMQ; ``eval_benchmark_repo_id`` and ``eval_benchmark_subset`` are
    configured on the server side and are ignored here.

    Returns (vec_env, env_cfg, env_preprocessor, env_postprocessor).
    """
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
    )
    env_dict = make_env(env_cfg, n_envs=1, use_async_envs=False)
    vec_env = env_dict["splatsim"][0]
    env_pre, env_post = make_env_pre_post_processors(env_cfg, policy_cfg)
    return vec_env, env_cfg, env_pre, env_post


# ── obs → policy batch ────────────────────────────────────────────────────────


def _apply_rename_map(obs: dict[str, torch.Tensor], rename_map: dict[str, str]) -> dict[str, torch.Tensor]:
    """Rename observation keys per ``rename_map``. Keys not present pass through."""
    if not rename_map:
        return obs
    return {rename_map.get(k, k): v for k, v in obs.items()}


def _build_sim_batch(
    env_obs: dict[str, np.ndarray],
    *,
    env_preprocessor,
    obs_preprocessor,
    rename_map: dict[str, str],
    device: str,
    task_description: str | None,
    guidance_chunk: np.ndarray | None,
) -> dict[str, torch.Tensor]:
    """env_obs (gym vec env format) → policy-ready preprocessed batch.

    Mirrors the lerobot_eval.py sequence:
    preprocess_observation → env_preprocessor → rename_map → obs_preprocessor.
    Optionally injects ``task`` and the guidance chunk.

    Also imported by ``augment_dataset_with_blending.py``.
    """
    obs = preprocess_observation(env_obs)
    obs = env_preprocessor(obs) if env_preprocessor is not None else obs
    obs = _apply_rename_map(obs, rename_map)
    obs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in obs.items()}
    # Task must be injected *before* the policy preprocessor (PI0.5 tokenizes it there).
    if task_description is not None:
        obs["task"] = [task_description]
    obs = obs_preprocessor(obs)
    if guidance_chunk is not None:
        chunk_t = torch.tensor(guidance_chunk, dtype=torch.float32, device=device).unsqueeze(0)
        obs["observation.policy_guidance_chunk"] = chunk_t
    return obs


# ── shared sim-loop helpers ───────────────────────────────────────────────────


def _run_filler_phase(
    wrapper,
    obs_preprocessor,
    env_preprocessor,
    env_obs: dict,
    *,
    guidance_chunk: np.ndarray,
    rename_map: dict[str, str],
    device: str,
    task_description: str | None,
    seed_joint_state: np.ndarray,
) -> None:
    """Drain the inner policy's first throwaway chunk so the obs queue has the
    right history before the real phase begins. Does NOT step the env.

    Also snaps ``wrapper._desired_q`` to ``seed_joint_state`` after filler so the
    wrapper's IK anchor isn't polluted by the throwaway chunk's actions.

    Imported by ``augment_dataset_with_blending.py``.
    """
    n_obs_steps: int = wrapper.config.n_obs_steps
    n_action_steps: int = wrapper.config.n_action_steps
    n_filler_drain = n_action_steps - (n_obs_steps - 1)
    for _ in range(n_filler_drain):
        batch = _build_sim_batch(
            env_obs,
            env_preprocessor=env_preprocessor,
            obs_preprocessor=obs_preprocessor,
            rename_map=rename_map,
            device=device,
            task_description=task_description,
            guidance_chunk=guidance_chunk,
        )
        wrapper.select_action(batch)
    for _ in range(n_obs_steps - 1):
        batch = _build_sim_batch(
            env_obs,
            env_preprocessor=env_preprocessor,
            obs_preprocessor=obs_preprocessor,
            rename_map=rename_map,
            device=device,
            task_description=task_description,
            guidance_chunk=guidance_chunk,
        )
        wrapper.select_action(batch)
    wrapper._desired_q = np.asarray(seed_joint_state, dtype=np.float32)[: wrapper.num_dofs].copy()


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
    episode_index_for_seed: int,
    guidance_actions_raw: np.ndarray,
    ratio: float,
    drain_chunk: bool,
    base_noise: torch.Tensor,
    total_steps: int,
    rename_map: dict[str, str],
    device: str,
    task_description: str | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Sim-in-the-loop variant: seed env, run filler, then loop
    ``env.step`` ↔ ``wrapper.select_action`` for ``total_steps`` iterations.

    ``episode_index_for_seed`` is forwarded to ``vec_env.reset(seed=[...])`` so the
    server loads the matching benchmark scenario. For local in-process envs the robot
    is also teleported to ``seed_joint_state``; for ZMQ envs it starts at the
    episode-initial pose.
    """
    n_action_steps: int = wrapper.config.n_action_steps
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")

    wrapper.reset()
    wrapper.forward_flow_ratio = ratio
    wrapper.blend_mode = BlendMode.ONCE_PER_CHUNK if drain_chunk else BlendMode.EVERY_STEP

    env_obs = seed_splatsim_env_to_state(
        vec_env,
        joint_state=seed_joint_state,
        num_dofs=wrapper.num_dofs,
        seed=[episode_index_for_seed],
    )

    # ─── Filler phase ────────────────────────────────────────────────────────
    _run_filler_phase(
        wrapper,
        obs_preprocessor,
        env_preprocessor,
        env_obs,
        guidance_chunk=guidance_actions_raw,
        rename_map=rename_map,
        device=device,
        task_description=task_description,
        seed_joint_state=seed_joint_state,
    )

    # ─── Real phase ──────────────────────────────────────────────────────────
    raw_actions: list[np.ndarray] = []
    decoded_guidance_full: np.ndarray | None = None
    success = False
    hold_action: np.ndarray | None = None

    for t in range(total_steps):
        # ── Hold mode: episode succeeded, don't step env again ────────────────
        # Stepping after termination triggers AutoresetMode.NEXT_STEP and would
        # bring in the next scene's images, causing a sharp visual transition.
        if success:
            assert hold_action is not None
            raw_actions.append(hold_action)
            continue

        at_chunk_boundary = t % n_action_steps == 0
        suppress_guidance = drain_chunk and not at_chunk_boundary and ratio not in (0.0, 1.0)
        guidance_chunk = None if suppress_guidance else guidance_actions_raw[t:]

        batch = _build_sim_batch(
            env_obs,
            env_preprocessor=env_preprocessor,
            obs_preprocessor=obs_preprocessor,
            rename_map=rename_map,
            device=device,
            task_description=task_description,
            guidance_chunk=guidance_chunk,
        )

        action_norm = wrapper.select_action(batch, base_noise=base_noise)
        raw_action = wrapper.postprocessor(action_norm)

        if env_postprocessor is not None:
            from lerobot.utils.constants import ACTION

            _post_out = env_postprocessor({ACTION: raw_action})
            if _post_out is not None:
                raw_action = _post_out[ACTION]

        action_numpy = raw_action.detach().to("cpu").numpy()
        env_obs, _reward, _term, _trunc, _info = vec_env.step(action_numpy)
        raw_actions.append(action_numpy.reshape(-1))

        if at_chunk_boundary and wrapper._last_decoded_guidance_chunk is not None:
            chunk_decode = wrapper._last_decoded_guidance_chunk[0]
            if decoded_guidance_full is None:
                decoded_guidance_full = np.zeros(
                    (total_steps, chunk_decode.shape[1]), dtype=chunk_decode.dtype
                )
            end_t = min(t + n_action_steps, total_steps)
            decoded_guidance_full[t:end_t] = chunk_decode[: end_t - t]

        # Check for success / termination.
        terminated = bool(_term[0]) if hasattr(_term, "__len__") else bool(_term)
        if terminated and not success:
            success = True
            agent_pos = env_obs.get("agent_pos")
            hold_action = (
                np.asarray(agent_pos[0], dtype=np.float32)
                if agent_pos is not None
                else action_numpy.reshape(-1)
            )
            print(
                f"[ratio={ratio}] Episode succeeded at t={t + 1}/{total_steps}. "
                f"Holding for {total_steps - t - 1} remaining steps.",
                flush=True,
            )

    return np.stack(raw_actions), decoded_guidance_full


@torch.no_grad()
def get_sim_action_chunks_for_ratios(
    wrapper,
    obs_preprocessor,
    vec_env,
    env_preprocessor,
    env_postprocessor,
    *,
    seed_joint_state: np.ndarray,
    episode_index_for_seed: int,
    guidance_actions_raw: np.ndarray,
    ratios: list[float],
    rename_map: dict[str, str],
    device: str,
    task_description: str | None,
    drain_chunk: bool,
    total_steps: int,
) -> tuple[dict[float, np.ndarray], dict[float, np.ndarray]]:
    """Run :func:`get_sim_action_chunk_for_ratio` for each ratio with a shared base_noise."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    if getattr(wrapper.config, "max_action_dim", None) is not None:
        noise_shape = (1, wrapper.config.chunk_size, wrapper.config.max_action_dim)
    else:
        action_dim = wrapper.config.output_features["action"].shape[0]
        noise_shape = (1, wrapper.config.horizon, action_dim)
    base_noise = torch.randn(noise_shape, device=device)

    results: dict[float, np.ndarray] = {}
    decoded_guidance_by_ratio: dict[float, np.ndarray] = {}

    progress = tqdm(
        ratios,
        desc="Computing sim action chunks",
        unit="ratio",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
    )
    for ratio in progress:
        progress.set_postfix_str(f"ratio={ratio:.2f}")
        actions, decoded = get_sim_action_chunk_for_ratio(
            wrapper,
            obs_preprocessor,
            vec_env,
            env_preprocessor,
            env_postprocessor,
            seed_joint_state=seed_joint_state,
            episode_index_for_seed=episode_index_for_seed,
            guidance_actions_raw=guidance_actions_raw,
            ratio=ratio,
            drain_chunk=drain_chunk,
            base_noise=base_noise,
            total_steps=total_steps,
            rename_map=rename_map,
            device=device,
            task_description=task_description,
        )
        results[ratio] = actions
        if decoded is not None:
            decoded_guidance_by_ratio[ratio] = decoded

    return results, decoded_guidance_by_ratio


# ── main ──────────────────────────────────────────────────────────────────────


def parse_args():
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
        default=None,
        help=(
            "Starting frame within episode for guidance slice. For ZMQ envs the robot "
            "always starts at the episode-initial pose regardless of this value."
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
    parser.add_argument("--camera_names", nargs="+", default=["base_rgb", "wrist_rgb"])
    parser.add_argument("--rename_map", type=json.loads, default=None)
    parser.add_argument("--robot_name", default="robot_iphone_w_engine_new")
    parser.add_argument("--num_dofs", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_action_steps", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--no_show", action="store_true")
    parser.add_argument("--drain_chunk", action="store_true")
    parser.add_argument("--blend_strategy", default="denoise", choices=["denoise", "interpolate"])
    parser.add_argument("--guidance_repr", default="absolute_pos", choices=["absolute_pos", "delta"])
    parser.add_argument("--n_anchor_steps", type=int, default=0)

    # ── Env / simulator config ────────────────────────────────────────────────
    parser.add_argument("--env_task", default="upright_small_engine_new")
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

    return parser.parse_args()


def main():
    args = parse_args()

    env_robot_name = args.env_robot_name or args.robot_name
    env_camera_names = args.env_camera_names or list(args.camera_names)
    env_image_resize_modes = args.env_image_resize_modes or [args.image_resize_mode]

    # Auto-resolve dataset_repo_id from the checkpoint if not passed. Prevents
    # the silent dataset-mismatch bug (e.g. dataset-11 checkpoint visualized
    # against dataset-7 frames).
    dataset_repo_id = args.dataset_repo_id
    if dataset_repo_id is None:
        train_cfg_path = Path(args.policy_path) / "train_config.json"
        if train_cfg_path.is_file():
            try:
                _cfg = json.loads(train_cfg_path.read_text())
                dataset_repo_id = _cfg.get("dataset", {}).get("repo_id")
            except (json.JSONDecodeError, OSError):
                dataset_repo_id = None
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
    wrapper.policy_guidance_representation = PolicyGuidanceRepresentation(args.guidance_repr)
    wrapper.n_anchor_steps = args.n_anchor_steps
    wrapper.skip_collision = True
    if args.n_action_steps is not None:
        prev = wrapper.config.n_action_steps
        wrapper.config.n_action_steps = args.n_action_steps
        print(f"Overrode n_action_steps: {prev} → {args.n_action_steps}")
    n_obs_steps = wrapper.config.n_obs_steps
    n_action_steps = wrapper.config.n_action_steps
    total_steps = getattr(wrapper.config, "chunk_size", None) or getattr(wrapper.config, "horizon", None)
    if total_steps is None:
        raise ValueError("Could not determine policy chunk length (chunk_size/horizon).")
    print(
        f"Policy: {wrapper.config.type}, n_obs_steps={n_obs_steps}, "
        f"n_action_steps={n_action_steps}, total_steps={total_steps}"
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

    n_needed = n_obs_steps + total_steps
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

    max_start = ep_length - n_needed
    if max_start < 0:
        raise ValueError(f"Episode {episode_index} too short ({ep_length} frames); need {n_needed}.")
    if args.frame_index is None:
        frame_index = random.randint(0, max_start)
        print(f"Selected random frame_index: {frame_index}")
    else:
        frame_index = args.frame_index
        print(f"Using frame_index: {frame_index}")

    frames_df = load_episode_frames(dataset_dir, episode_index, frame_index, n_needed)
    obs_frames = frames_df.iloc[:n_obs_steps]
    guidance_frames = frames_df.iloc[n_obs_steps : n_obs_steps + total_steps]
    guidance_actions_raw = np.stack(
        [np.array(row["action"], dtype=np.float32) for _, row in guidance_frames.iterrows()]
    )
    print(
        f"Loaded {len(obs_frames)} obs + {len(guidance_frames)} guidance frames (action_dim={guidance_actions_raw.shape[1]})."
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

    guidance_actions_raw_for_plot = guidance_actions_raw
    if args.guidance_repr == "delta":
        print("Converting absolute positions to EE deltas for DELTA mode …")
        guidance_actions_raw = absolute_positions_to_ee_deltas(
            wrapper, seed_joint_state, guidance_actions_raw
        )

    # Connect to the external simulator. The server is already running in
    # EVAL_BENCHMARK mode; we select scenarios via reset(seed=[episode_index]).
    print(
        f"Connecting to splatsim at {args.env_external_host}:{args.env_external_port} "
        f"(task={args.env_task}) …"
    )
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
        policy_cfg=wrapper.config,
    )

    try:
        print(f"Computing sim rollouts for ratios: {args.forward_flow_ratios} …")
        action_chunks, decoded_guidance_by_ratio = get_sim_action_chunks_for_ratios(
            wrapper,
            obs_preprocessor,
            vec_env,
            env_pre,
            env_post,
            seed_joint_state=seed_joint_state,
            episode_index_for_seed=episode_index,
            guidance_actions_raw=guidance_actions_raw,
            ratios=args.forward_flow_ratios,
            rename_map=rename_map,
            device=args.device,
            task_description=task_description,
            drain_chunk=args.drain_chunk,
            total_steps=total_steps,
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
        import json as _json

        train_cfg = args.policy_path.rstrip("/") + "/train_config.json"
        try:
            with open(train_cfg) as f:
                policy_tag = _json.load(f)["policy"]["type"]
        except Exception:
            policy_tag = "policy"
        repr_tag = "delta" if args.guidance_repr == "delta" else "abspos"
        drain_tag = "onestep" if args.drain_chunk else "everystep"
        anchor_tag = f"anchor{args.n_anchor_steps}" if args.n_anchor_steps > 0 else "noanchor"
        nas_tag = f"nas{n_action_steps}"
        parent = f"shared_autonomy_sim_ep{episode_index}_frame{frame_index}"
        name = f"{policy_tag}_{args.blend_strategy}_{repr_tag}_{drain_tag}_{anchor_tag}_{nas_tag}_sim"
        output_dir: Path = Path("outputs/viz") / parent / name
    else:
        output_dir = Path(args.output_dir)
    print(f"Output dir: {output_dir}")
    joint_angles_path = output_dir / "joint_angles.png"
    ee_traj_path = output_dir / "ee_trajectory.html"

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

    print("Done.")


if __name__ == "__main__":
    main()
