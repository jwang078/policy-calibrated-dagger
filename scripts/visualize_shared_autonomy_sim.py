#!/usr/bin/env python
"""Sim-in-the-loop variant of visualize_shared_autonomy.py.

The parquet-driven script (`visualize_shared_autonomy.py`) feeds the same dataset
frame's observation to the policy every step. With small `--n_action_steps`, the
wrapper re-predicts mid-rollout but the observations are stale → zigzag artifacts
at chunk boundaries. This script drives a real splatsim env so observations stay
in sync with the actually-executed actions.

Imports plotting / IO helpers from `visualize_shared_autonomy.py` (sibling file).

Example:
    python my_scripts/visualize_shared_autonomy_sim.py \\
        --policy_path .../pretrained_model \\
        --dataset_repo_id JennyWWW/splatsim_approach_lever_7_lowres_5path_10fails \\
        --episode_index 305 --frame_index 8 \\
        --forward_flow_ratios 0.0 0.05 0.2 0.4 0.8 1.0 \\
        --blend_strategy denoise --guidance_repr delta --drain_chunk \\
        --n_action_steps 5 \\
        --env_task upright_small_engine_new \\
        --env_camera_names base_rgb wrist_rgb \\
        --env_image_resize_modes letterbox --image_resize_mode letterbox \\
        --env_fps 30 --env_episode_length 400
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from math import ceil  # noqa: F401  (parity with sibling script)
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

# Allow importing the sibling parquet-driven script directly.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from visualize_shared_autonomy import (  # type: ignore[import-not-found]  # noqa: E402
    absolute_positions_to_ee_deltas,
    compute_ee_from_states,
    compute_ee_trajectories,
    find_parquet_files,
    get_available_episodes,
    load_episode_frames,
    load_task_description,
    load_wrapped_policy,
    plot_ee_trajectories_3d,
    plot_joint_angles,
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
    eval_benchmark_repo_id: str,
    eval_benchmark_subset: list[int],
    policy_cfg: Any,
):
    """Build a splatsim vec env (n_envs=1) plus the env-specific pre/post processors.

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
        eval_benchmark_repo_id=eval_benchmark_repo_id,
        eval_benchmark_subset=eval_benchmark_subset,
    )
    env_dict = make_env(env_cfg, n_envs=1, use_async_envs=False)
    vec_env = env_dict["splatsim"][0]
    env_pre, env_post = make_env_pre_post_processors(env_cfg, policy_cfg)
    return vec_env, env_cfg, env_pre, env_post


# ── obs → policy batch ────────────────────────────────────────────────────────


def _apply_rename_map(obs: dict[str, torch.Tensor], rename_map: dict[str, str]) -> dict[str, torch.Tensor]:
    """Rename observation keys in-place per ``rename_map``. Keys not present pass through."""
    if not rename_map:
        return obs
    out: dict[str, torch.Tensor] = {}
    for k, v in obs.items():
        out[rename_map.get(k, k)] = v
    return out


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

    Mirrors the sequence in lerobot_eval.py: preprocess_observation → env_preprocessor →
    rename_map → obs_preprocessor. Optionally injects ``task`` and the guidance chunk.
    """
    obs = preprocess_observation(env_obs)  # numpy → torch with OBS_STATE / OBS_IMAGES.* keys
    obs = env_preprocessor(obs) if env_preprocessor is not None else obs
    obs = _apply_rename_map(obs, rename_map)

    # Move tensors to device before the policy preprocessor sees them, matching the parquet path.
    obs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in obs.items()}

    # Task must be present *before* the policy preprocessor (PI0.5 tokenizes it there).
    if task_description is not None:
        obs["task"] = [task_description]

    obs = obs_preprocessor(obs)

    if guidance_chunk is not None:
        chunk_t = torch.tensor(guidance_chunk, dtype=torch.float32, device=device).unsqueeze(0)
        obs["observation.policy_guidance_chunk"] = chunk_t

    return obs


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
    guidance_actions_raw: np.ndarray,
    ratio: float,
    drain_chunk: bool,
    base_noise: torch.Tensor,
    total_steps: int,
    rename_map: dict[str, str],
    device: str,
    task_description: str | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Sim-in-the-loop variant: seed env, run filler with seeded obs, then loop
    ``env.step`` ↔ ``wrapper.select_action`` for ``total_steps`` iterations."""
    n_obs_steps: int = wrapper.config.n_obs_steps
    n_action_steps: int = wrapper.config.n_action_steps
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")

    print(f"[ratio={ratio}] wrapper.reset() …", flush=True)
    wrapper.reset()
    wrapper.forward_flow_ratio = ratio
    wrapper.blend_mode = BlendMode.ONCE_PER_CHUNK if drain_chunk else BlendMode.EVERY_STEP

    # Seed sim to (episode-start objects, frame_index robot pose).
    print(f"[ratio={ratio}] seeding env to state {seed_joint_state.tolist()} …", flush=True)
    env_obs = seed_splatsim_env_to_state(vec_env, joint_state=seed_joint_state, num_dofs=wrapper.num_dofs)
    print(f"[ratio={ratio}] seed done; env_obs keys: {list(env_obs.keys())}", flush=True)

    first_guidance_chunk = guidance_actions_raw  # full chunk; wrapper truncates as needed

    # ─── Filler phase ────────────────────────────────────────────────────────
    # Drain the inner policy's first throwaway chunk while feeding the seeded env_obs
    # repeatedly. This populates the inner policy's obs queue with the right history
    # before the real phase begins. We do NOT step the env during filler.
    n_filler_drain = n_action_steps - (n_obs_steps - 1)
    print(
        f"[ratio={ratio}] filler phase: draining {n_filler_drain} + {n_obs_steps - 1} actions …", flush=True
    )
    for i in range(n_filler_drain):
        batch = _build_sim_batch(
            env_obs,
            env_preprocessor=env_preprocessor,
            obs_preprocessor=obs_preprocessor,
            rename_map=rename_map,
            device=device,
            task_description=task_description,
            guidance_chunk=first_guidance_chunk,
        )
        if i == 0:
            print(f"[ratio={ratio}] first wrapper.select_action call …", flush=True)
        wrapper.select_action(batch)  # discard
        if i == 0:
            print(f"[ratio={ratio}] first wrapper.select_action returned", flush=True)
    for _ in range(n_obs_steps - 1):
        batch = _build_sim_batch(
            env_obs,
            env_preprocessor=env_preprocessor,
            obs_preprocessor=obs_preprocessor,
            rename_map=rename_map,
            device=device,
            task_description=task_description,
            guidance_chunk=first_guidance_chunk,
        )
        wrapper.select_action(batch)  # discard

    # Filler corrupts the IK anchor; snap _desired_q to the seeded joints.
    wrapper._desired_q = np.asarray(seed_joint_state, dtype=np.float32)[: wrapper.num_dofs].copy()
    print(f"[ratio={ratio}] filler done; entering real phase …", flush=True)

    # ─── Real phase ──────────────────────────────────────────────────────────
    raw_actions: list[np.ndarray] = []
    decoded_guidance_full: np.ndarray | None = None
    for t in range(total_steps):
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
        raw_action = wrapper.postprocessor(action_norm)  # (1, action_dim) tensor

        # Mirror lerobot_eval: env_postprocessor on action transition (no-op for splatsim).
        if env_postprocessor is not None:
            from lerobot.utils.constants import ACTION

            transition = env_postprocessor({ACTION: raw_action})
            raw_action = transition[ACTION]

        action_numpy = raw_action.detach().to("cpu").numpy()  # (1, action_dim)
        # Splatsim action_space is (7,); env.step on a SyncVectorEnv expects (n_envs, 7).
        # action_numpy already has the right shape (batch=1, action_dim=7).
        if t < 2:
            print(
                f"[ratio={ratio}, t={t}] action_numpy shape={action_numpy.shape} "
                f"first={action_numpy.reshape(-1).tolist()[:3]}; calling vec_env.step …",
                flush=True,
            )
        env_obs, _reward, _term, _trunc, _info = vec_env.step(action_numpy)
        if t < 2:
            print(f"[ratio={ratio}, t={t}] vec_env.step returned", flush=True)

        raw_actions.append(action_numpy.reshape(-1))

        # Stitch decoded guidance per blend boundary so the orange overlay tracks the
        # executed trajectory across multiple regenerations.
        if at_chunk_boundary and wrapper._last_decoded_guidance_chunk is not None:
            chunk_decode = wrapper._last_decoded_guidance_chunk[0]  # [anchor_len, joint_dim]
            if decoded_guidance_full is None:
                decoded_guidance_full = np.zeros(
                    (total_steps, chunk_decode.shape[1]), dtype=chunk_decode.dtype
                )
            end_t = min(t + n_action_steps, total_steps)
            decoded_guidance_full[t:end_t] = chunk_decode[: end_t - t]

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
    guidance_actions_raw: np.ndarray,
    ratios: list[float],
    rename_map: dict[str, str],
    device: str,
    task_description: str | None,
    drain_chunk: bool,
    total_steps: int,
) -> tuple[dict[float, np.ndarray], dict[float, np.ndarray]]:
    """Run :func:`get_sim_action_chunk_for_ratio` for each ratio with a shared base_noise."""
    # Match the parquet path's deterministic noise — same shape detection logic.
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

    for ratio in ratios:
        actions, decoded = get_sim_action_chunk_for_ratio(
            wrapper,
            obs_preprocessor,
            vec_env,
            env_preprocessor,
            env_postprocessor,
            seed_joint_state=seed_joint_state,
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
            "Drives a real splatsim env each step so observations stay in sync with the "
            "executed rollout (unlike visualize_shared_autonomy.py, which feeds the same "
            "dataset frame every step)."
        )
    )
    # ── Identical to visualize_shared_autonomy.py ────────────────────────────
    parser.add_argument("--policy_path", required=True, help="Path to pretrained model directory.")
    parser.add_argument(
        "--dataset_repo_id",
        default="JennyWWW/splatsim_approach_lever_7_lowres_5path_10fails",
        help="HuggingFace dataset repo ID.",
    )
    parser.add_argument(
        "--dataset_dir",
        default=None,
        help="Local directory containing parquet files (auto-resolved if not set).",
    )
    parser.add_argument("--episode_index", type=int, default=None)
    parser.add_argument("--frame_index", type=int, default=None)
    parser.add_argument(
        "--forward_flow_ratios",
        nargs="+",
        type=float,
        default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    )
    parser.add_argument(
        "--image_resize_mode",
        default="letterbox",
        choices=["stretch", "letterbox"],
        help="Image resize mode used for parquet column lookup AND the rename_map default.",
    )
    parser.add_argument("--camera_names", nargs="+", default=["base_rgb", "wrist_rgb"])
    parser.add_argument(
        "--rename_map",
        type=json.loads,
        default=None,
        help="JSON rename map; defaults to {cam}_{mode} → {cam} per camera.",
    )
    parser.add_argument("--robot_name", default="robot_iphone_w_engine_new")
    parser.add_argument("--num_dofs", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--n_action_steps",
        type=int,
        default=None,
        help=(
            "Override the policy's n_action_steps. Smaller values cause the policy "
            "to regenerate and re-blend more frequently within the same window."
        ),
    )
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--no_show", action="store_true")
    parser.add_argument("--drain_chunk", action="store_true")
    parser.add_argument("--blend_strategy", default="denoise", choices=["denoise", "interpolate"])
    parser.add_argument("--guidance_repr", default="absolute_pos", choices=["absolute_pos", "delta"])
    parser.add_argument("--n_anchor_steps", type=int, default=0)

    # ── Sim-in-the-loop env config ───────────────────────────────────────────
    parser.add_argument("--env_task", default="upright_small_engine_new")
    parser.add_argument(
        "--env_robot_name",
        default=None,
        help="Defaults to --robot_name (used both for FK and the splatsim env).",
    )
    parser.add_argument("--env_camera_names", nargs="+", default=None, help="Defaults to --camera_names.")
    parser.add_argument(
        "--env_image_resize_modes",
        nargs="+",
        default=None,
        help="Defaults to [--image_resize_mode]. The env produces obs keys per mode.",
    )
    parser.add_argument("--env_fps", type=int, default=30)
    parser.add_argument("--env_episode_length", type=int, default=400)
    parser.add_argument(
        "--env_external_port",
        type=int,
        default=None,
        help="Connect to an already-running splatsim server (ZMQ). State seeding is "
        "NOT supported in this mode — only --env_external_port=None is supported.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    if args.env_external_port is not None:
        raise NotImplementedError(
            "Connecting to an external splatsim server (ZMQ) is not supported by "
            "the sim-seeding helper. Run with --env_external_port omitted so this "
            "script launches its own splatsim env."
        )

    # Default the env_* fields to their non-env counterparts where applicable.
    env_robot_name = args.env_robot_name or args.robot_name
    env_camera_names = args.env_camera_names or list(args.camera_names)
    env_image_resize_modes = args.env_image_resize_modes or [args.image_resize_mode]

    # Resolve dataset / task map.
    dataset_dir = resolve_dataset_dir(args.dataset_repo_id, args.dataset_dir)
    print(f"Dataset dir: {dataset_dir}")
    task_map = load_task_description(dataset_dir)
    print(f"Task map: {task_map}")

    # Rename map (parity with parquet script).
    if args.rename_map is None:
        rename_map = make_default_rename_map(args.camera_names, args.image_resize_mode)
    else:
        rename_map = args.rename_map
    print(f"Rename map: {rename_map}")

    # Load policy.
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
        prev_n_action_steps = wrapper.config.n_action_steps
        wrapper.config.n_action_steps = args.n_action_steps
        print(f"Overrode n_action_steps: {prev_n_action_steps} → {args.n_action_steps}")
    n_obs_steps = wrapper.config.n_obs_steps
    n_action_steps = wrapper.config.n_action_steps
    total_steps = getattr(wrapper.config, "chunk_size", None) or getattr(wrapper.config, "horizon", None)
    if total_steps is None:
        raise ValueError("Could not determine policy chunk length (chunk_size/horizon).")
    print(
        f"Policy type: {wrapper.config.type}, n_obs_steps={n_obs_steps}, "
        f"n_action_steps={n_action_steps}, total_steps={total_steps}"
    )
    print(
        f"Blend strategy: {wrapper.guidance_blend_strategy.value}, "
        f"guidance repr: {wrapper.policy_guidance_representation.value}"
    )

    # Pick episode + frame.
    if args.episode_index is None:
        available = get_available_episodes(dataset_dir, min_episode_index=301)
        if not available:
            raise RuntimeError(f"No episodes >= 301 in {dataset_dir}.")
        episode_index = random.choice(available)
        print(f"Selected random episode: {episode_index}")
    else:
        episode_index = args.episode_index
        print(f"Using episode: {episode_index}")

    n_needed = n_obs_steps + total_steps
    parquet_files = find_parquet_files(dataset_dir)
    ep_df_list = []
    for f in parquet_files:
        df_ep = pd.read_parquet(
            f, columns=["episode_index", "frame_index"], filters=[("episode_index", "==", episode_index)]
        )
        if len(df_ep) > 0:
            ep_df_list.append(df_ep)
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

    # Load demo frames for guidance.
    frames_df = load_episode_frames(dataset_dir, episode_index, frame_index, n_needed)
    obs_frames = frames_df.iloc[:n_obs_steps]
    guidance_frames = frames_df.iloc[n_obs_steps : n_obs_steps + total_steps]
    guidance_actions_raw = np.stack(
        [np.array(row["action"], dtype=np.float32) for _, row in guidance_frames.iterrows()]
    )
    print(
        f"Loaded {len(obs_frames)} obs + {len(guidance_frames)} guidance frames "
        f"(action_dim={guidance_actions_raw.shape[1]})."
    )

    # Task description for PI0.5.
    task_idx = int(frames_df.iloc[0].get("task_index", 1))
    task_description = task_map.get(task_idx)
    if task_description is not None:
        print(f"Task: '{task_description}' (task_index={task_idx})")

    # Seed state = the recorded action at frame_index (matches parquet path's _desired_q).
    seed_joint_state = np.array(obs_frames.iloc[-1]["action"], dtype=np.float32)

    # DELTA-mode guidance prep — freeze the conversion using the seeded init_state.
    guidance_actions_raw_for_plot = guidance_actions_raw
    if args.guidance_repr == "delta":
        print("Converting absolute positions to EE deltas for DELTA mode …")
        guidance_actions_raw = absolute_positions_to_ee_deltas(
            wrapper, seed_joint_state, guidance_actions_raw
        )

    # Build env. The eval_benchmark_subset is a single-episode list so each reset()
    # restores object poses for *this* episode. The robot is then teleported to the
    # seed_joint_state (mid-episode) inside seed_splatsim_env_to_state.
    print(f"Building splatsim env (task={args.env_task}, episode_subset=[{episode_index}]) …")
    vec_env, env_cfg, env_pre, env_post = build_splatsim_env(
        task=args.env_task,
        robot_name=env_robot_name,
        camera_names=env_camera_names,
        image_resize_modes=env_image_resize_modes,
        fps=args.env_fps,
        episode_length=args.env_episode_length,
        external_port=args.env_external_port,
        eval_benchmark_repo_id=args.dataset_repo_id,
        eval_benchmark_subset=[episode_index],
        policy_cfg=wrapper.config,
    )

    try:
        # Run all ratios.
        print(f"Computing sim rollouts for ratios: {args.forward_flow_ratios} …")
        action_chunks, decoded_guidance_by_ratio = get_sim_action_chunks_for_ratios(
            wrapper,
            obs_preprocessor,
            vec_env,
            env_pre,
            env_post,
            seed_joint_state=seed_joint_state,
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

    # Pick a representative decoded guidance overlay (any ratio that hit the blend).
    decoded_guidance: np.ndarray | None = None
    if decoded_guidance_by_ratio:
        sample_ratio = next(iter(decoded_guidance_by_ratio.keys()))
        decoded_guidance = decoded_guidance_by_ratio[sample_ratio]
        print(f"Captured decoded guidance overlay from ratio={sample_ratio}.")

    # EE trajectories.
    print("Computing EE trajectories via pybullet FK …")
    obs_states_raw = np.stack([np.array(row["action"], dtype=np.float32) for _, row in obs_frames.iterrows()])
    init_obs_state_raw = obs_states_raw[-1]
    ee_trajectories = compute_ee_trajectories(
        wrapper=wrapper,
        init_obs_state_raw=init_obs_state_raw,
        action_chunks_by_ratio=action_chunks,
    )
    obs_ee_positions = compute_ee_from_states(wrapper, obs_states_raw)
    guidance_ee_positions = compute_ee_from_states(wrapper, guidance_actions_raw_for_plot)
    decoded_guidance_ee_positions = (
        compute_ee_from_states(wrapper, decoded_guidance) if decoded_guidance is not None else None
    )

    # Joint names for plots.
    action_dim = next(iter(action_chunks.values())).shape[1]
    joint_names = [f"joint_{i + 1}" for i in range(min(args.num_dofs, action_dim))]
    if action_dim > args.num_dofs:
        joint_names.append("gripper")

    # Output dir naming — mirror the parquet path with a `_sim` suffix.
    if args.output_dir is None:
        train_config_path = Path(args.policy_path) / "train_config.json"
        with open(train_config_path) as f:
            policy_tag = json.load(f)["policy"]["type"]
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
