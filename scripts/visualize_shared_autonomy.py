#!/usr/bin/env python
"""Visualize SharedAutonomyPolicyWrapper predictions for different forward_flow_ratios.

For a single observation window loaded from the training dataset, this script shows
how the predicted action trajectory changes as forward_flow_ratio varies from 0
(pure human demonstration) to 1 (pure policy).

Two visualizations are produced:
  1. Joint angles grid  — matplotlib, one subplot per joint + gripper
  2. EE trajectory 3D   — plotly interactive, end-effector XYZ over the action chunk

Example:
    python my_scripts/visualize_shared_autonomy.py \
        --policy_path outputs/training/pi05_approach_lever_7_lowres_5path_10fails_basewrist/checkpoints/006000/pretrained_model \
        --forward_flow_ratios 0.0 0.2 0.4 0.6 0.8 1.0
"""

from __future__ import annotations

import argparse
import io
import json
import random
from math import ceil
from pathlib import Path
from types import SimpleNamespace

import einops
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

# ── lerobot imports ──────────────────────────────────────────────────────────
from lerobot.configs.shared_autonomy import SharedAutonomyConfig
from lerobot.policies.factory import _wrap_with_shared_autonomy, get_policy_class
from lerobot.policies.shared_autonomy_wrapper import (
    BlendMode,
    GuidanceBlendStrategy,
    PolicyGuidanceRepresentation,
)
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import POLICY_PREPROCESSOR_DEFAULT_NAME

# ── policy loading ────────────────────────────────────────────────────────────


def load_wrapped_policy(
    policy_path: str | Path,
    forward_flow_ratio: float = 1.0,
    robot_name: str = "robot_iphone_w_engine_new",
    num_dofs: int = 6,
    device: str = "cpu",
):
    """Load inner policy and wrap with SharedAutonomyPolicyWrapper (no slider).

    Uses ABSOLUTE_POS guidance representation so that dataset joint positions are
    passed directly as guidance without FK→IK conversion.

    Returns (wrapper, obs_preprocessor).
    """
    policy_path = Path(policy_path)
    with open(policy_path / "config.json") as f:
        config_data = json.load(f)
    policy_type = config_data["type"]

    policy_cls = get_policy_class(policy_type)
    inner_policy = policy_cls.from_pretrained(str(policy_path))
    inner_policy = inner_policy.to(device).eval()

    cfg = SimpleNamespace(
        pretrained_path=str(policy_path),
        device=device,
        output_features=getattr(inner_policy.config, "output_features", None),
        input_features=getattr(inner_policy.config, "input_features", None),
        normalization_mapping=getattr(inner_policy.config, "normalization_mapping", None),
        shared_autonomy_config=SharedAutonomyConfig(
            enabled=True,
            forward_flow_ratio=forward_flow_ratio,
            show_slider=False,
            start_paused=False,
            robot_name=robot_name,
            num_dofs=num_dofs,
        ),
    )

    wrapper = _wrap_with_shared_autonomy(inner_policy, cfg)
    wrapper.policy_guidance_representation = PolicyGuidanceRepresentation.ABSOLUTE_POS
    wrapper.guidance_blend_strategy = GuidanceBlendStrategy.DENOISE  # default, overridden by caller
    wrapper = wrapper.to(device).eval()

    # Observation preprocessor (normalizes obs.state; images pass through unchanged).
    # Use default batch_to_transition / transition_to_batch so it accepts / returns dicts.
    obs_preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=str(policy_path),
        config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
    )

    return wrapper, obs_preprocessor


# ── parquet data loading ──────────────────────────────────────────────────────


def find_parquet_files(dataset_dir: Path) -> list[Path]:
    files = sorted(dataset_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {dataset_dir}")
    return files


def load_episode_frames(
    dataset_dir: Path,
    episode_index: int,
    frame_index: int,
    n_frames: int,
) -> pd.DataFrame:
    """Load n_frames consecutive rows from a specific episode starting at frame_index."""
    parquet_files = find_parquet_files(dataset_dir)
    dfs = []
    for f in parquet_files:
        df = pd.read_parquet(f, filters=[("episode_index", "==", episode_index)])
        if len(df) > 0:
            dfs.append(df)
    if not dfs:
        raise ValueError(f"Episode {episode_index} not found in {dataset_dir}")
    df = pd.concat(dfs, ignore_index=True).sort_values("frame_index").reset_index(drop=True)

    # Slice: find frames with frame_index >= frame_index and take n_frames of them
    mask = df["frame_index"] >= frame_index
    df = df[mask].head(n_frames).reset_index(drop=True)
    if len(df) < n_frames:
        raise ValueError(
            f"Episode {episode_index} only has {len(df)} frames from frame_index={frame_index}, "
            f"need {n_frames}."
        )
    return df


def get_available_episodes(dataset_dir: Path, min_episode_index: int = 301) -> list[int]:
    """Return sorted list of episode indices >= min_episode_index in the dataset."""
    parquet_files = find_parquet_files(dataset_dir)
    indices = set()
    for f in parquet_files:
        ep_col = pd.read_parquet(f, columns=["episode_index"])
        indices.update(ep_col["episode_index"].unique().tolist())
    return sorted(i for i in indices if i >= min_episode_index)


def decode_image(image_data) -> np.ndarray:
    """Decode a parquet image dict (with 'bytes' key) to uint8 RGB numpy array."""
    img_bytes = image_data["bytes"] if isinstance(image_data, dict) else image_data
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(img)


def load_task_description(dataset_dir: Path) -> dict[int, str]:
    """Load task_index → task_description mapping from dataset metadata.

    Returns a dict like {1: "upright_small_engine_new"}.
    Falls back to empty dict if not found.
    """
    meta_dir = dataset_dir.parent / "meta"
    tasks_file = meta_dir / "tasks.parquet"
    if not tasks_file.exists():
        return {}
    tasks_df = pd.read_parquet(tasks_file).reset_index()
    # The parquet has __index_level_0__ (task description string) and task_index (int).
    result = {}
    for _, row in tasks_df.iterrows():
        desc = row.get("__index_level_0__") or row.get("index") or ""
        idx = int(row.get("task_index", 0))
        if desc:
            result[idx] = str(desc)
    return result


def frame_to_obs_batch(
    row: pd.Series,
    camera_names: list[str],
    image_resize_mode: str,
    rename_map: dict[str, str],
    device: str,
    task_description: str | None = None,
) -> dict:
    """Convert a single parquet row to a preprocessor-ready observation dict.

    Images: decoded → uint8 → float32 [0,1] channel-first, batch-dim=1.
    State:  raw float32, batch-dim=1.
    The rename_map is applied so keys match what the policy expects.
    task_description: if provided, added as "task": [task_description] for PI0.5.
    """
    obs: dict = {}

    for cam in camera_names:
        parquet_col = f"observation.images.{cam}_{image_resize_mode}"
        img_np = decode_image(row[parquet_col])  # (H, W, C) uint8
        img_t = torch.from_numpy(img_np).unsqueeze(0)  # (1, H, W, C)
        img_t = einops.rearrange(img_t, "b h w c -> b c h w").contiguous()
        img_t = img_t.float() / 255.0  # (1, C, H, W) float32 [0,1]
        raw_key = f"observation.images.{cam}_{image_resize_mode}"
        policy_key = rename_map.get(raw_key, raw_key)
        obs[policy_key] = img_t.to(device)

    state = np.array(row["observation.state"], dtype=np.float32)
    obs["observation.state"] = torch.from_numpy(state).unsqueeze(0).to(device)

    if task_description is not None:
        obs["task"] = [task_description]

    return obs


# ── action chunk collection ───────────────────────────────────────────────────


@torch.no_grad()
def get_action_chunk_for_ratio(
    wrapper,
    obs_preprocessor,
    obs_frames: pd.DataFrame,
    guidance_actions_raw: np.ndarray,
    ratio: float,
    camera_names: list[str],
    image_resize_mode: str,
    rename_map: dict[str, str],
    device: str,
    task_description: str | None = None,
    drain_chunk: bool = True,
    base_noise: torch.Tensor | None = None,
) -> np.ndarray:
    """Compute a single predicted action chunk (raw joint space) for a given forward_flow_ratio."""

    def _build_batch(
        row: pd.Series,
        guidance_action: np.ndarray | None = None,
        guidance_chunk: np.ndarray | None = None,
    ) -> dict:
        """Build a preprocessed batch from one observation row + optional guidance.

        guidance_action: [action_dim] single-step guidance (sets OBS_HUMAN_ACTION).
        guidance_chunk:  [n_steps, action_dim] per-step guidance (sets OBS_GUIDANCE_CHUNK).
                         When provided alongside guidance_action, the wrapper uses the
                         full chunk for per-step blending instead of the ramp approximation.
        """
        raw_obs = frame_to_obs_batch(
            row,
            camera_names,
            image_resize_mode,
            rename_map,
            device,
            task_description=task_description,
        )
        preprocessed = obs_preprocessor(raw_obs)
        # if guidance_action is not None:
        #     guidance_t = torch.tensor(guidance_action, dtype=torch.float32, device=device).unsqueeze(0)
        #     preprocessed["observation.policy_guidance_action"] = guidance_t
        if guidance_chunk is not None:
            chunk_t = torch.tensor(guidance_chunk, dtype=torch.float32, device=device).unsqueeze(0)
            preprocessed["observation.policy_guidance_chunk"] = chunk_t
        return preprocessed

    # Use the first guidance action for all obs-filling steps (constant context).
    first_guidance = guidance_actions_raw[0]
    n_obs_steps = wrapper.config.n_obs_steps
    n_action_steps = wrapper.config.n_action_steps

    wrapper.reset()
    wrapper.forward_flow_ratio = ratio
    wrapper.blend_mode = BlendMode.ONCE_PER_CHUNK if drain_chunk else BlendMode.EVERY_STEP

    raw_actions: list[np.ndarray] = []

    # Filler phase: drain the first (throwaway) action chunk while populating
    # the inner policy's obs queue with the correct observation history.
    #
    # For diffusion (n_obs_steps=2): we need the obs queue to be [obs0, obs1]
    # when predict_action_chunk fires for the real phase. Strategy:
    #   - Call select_action (n_action_steps - n_obs_steps + 1) times with obs0.
    #     This triggers predict (throwaway) and leaves (n_obs_steps-1) actions
    #     in the queue. Obs queue = [obs0, obs0].
    #   - Call select_action (n_obs_steps - 1) times feeding obs0..obs_{n-2}.
    #     These pop the remaining filler actions (no new predict). Each call
    #     pushes its obs into the queue, shifting it to [obs0, obs1, ...].
    #   - Now the action queue is empty and the obs queue has the correct window.
    #     The next select_action call triggers predict with the right obs.
    #
    # For PI0.5 (n_obs_steps=1): n_obs_steps-1=0 so we just drain the full
    # chunk with n_action_steps calls, no extra obs-shifting needed.

    # Step 1: drain most of the filler chunk
    filler_row = obs_frames.iloc[0]
    n_filler_drain = n_action_steps - (n_obs_steps - 1)
    for _ in range(n_filler_drain):
        batch = _build_batch(
            filler_row, guidance_action=first_guidance, guidance_chunk=guidance_actions_raw[:]
        )
        wrapper.select_action(batch)  # discard

    # Step 2: pop remaining filler actions while shifting the obs queue
    for i in range(n_obs_steps - 1):
        row = obs_frames.iloc[i]
        batch = _build_batch(row, guidance_action=first_guidance, guidance_chunk=guidance_actions_raw[:])
        wrapper.select_action(batch)  # discard (last filler actions)

    # The filler phase corrupted the IK anchor. Snap it back to the true initial state!
    last_obs_action = np.array(obs_frames.iloc[-1]["action"], dtype=np.float32)
    wrapper._desired_q = last_obs_action[: wrapper.num_dofs].copy()

    # Real phase: action queue is now empty, obs queue has correct history.
    # The first call triggers predict_action_chunk with the right obs window.
    #
    # drain_chunk mode: provide guidance only on t=0 to trigger one clean blend,
    # then let the wrapper drain the blended chunk (no re-blending, temporally
    # coherent). Without drain_chunk, guidance is provided every step.
    last_row = obs_frames.iloc[-1]
    for t in range(n_action_steps):
        if drain_chunk and t > 0 and ratio not in (0.0, 1.0):
            # No guidance → wrapper takes drain path, returns from same chunk.
            # Only suppress guidance for intermediate ratios — ratio=0.0 needs
            # guidance every step (otherwise get_hold_action returns same point).
            batch = _build_batch(last_row)
        else:
            guidance = guidance_actions_raw[t]
            # Pass the full remaining chunk so the wrapper can use per-step guidance
            # instead of ramp-to-endpoint approximation (ABSOLUTE_POS mode only).
            remaining_chunk = guidance_actions_raw[t:]  # [n_remaining, action_dim]
            batch = _build_batch(last_row, guidance_action=guidance, guidance_chunk=remaining_chunk)
        # use base_noise to make the plots plot the same noise across ratios
        action_norm = wrapper.select_action(batch, base_noise=base_noise)
        raw = wrapper.postprocessor(action_norm).detach().cpu().numpy().reshape(-1)
        raw_actions.append(raw)

    all_raw_actions = np.stack(raw_actions)

    return all_raw_actions


@torch.no_grad()
def get_action_chunks_for_ratios(
    wrapper,
    obs_preprocessor,
    obs_frames: pd.DataFrame,
    guidance_actions_raw: np.ndarray,
    ratios: list[float],
    camera_names: list[str],
    image_resize_mode: str,
    rename_map: dict[str, str],
    device: str,
    task_description: str | None = None,
    drain_chunk: bool = False,
) -> dict[float, np.ndarray]:
    """Compute predicted action chunks (raw joint space) for each forward_flow_ratio.

    Uses wrapper.select_action() directly so that changes to select_action (velocity
    limiting, guidance blending, etc.) are automatically picked up.

    The wrapper must be initialized with ABSOLUTE_POS guidance representation so that
    the dataset joint positions are used directly without FK→IK conversion.

    For each ratio:
      - Resets wrapper, sets wrapper.forward_flow_ratio = ratio.
      - Filler phase drains a throwaway chunk to populate the obs queue.
      - Real phase collects n_action_steps actions.

    If drain_chunk=True, guidance is provided only on the first real-phase call.
    The blended chunk is then drained without re-blending, giving temporally coherent
    actions from a single denoising pass. Without this flag, guidance is provided every
    step (re-blends each step with fresh random noise).

    Returns {ratio: np.ndarray [n_action_steps, action_dim]} in raw joint space.
    """
    # Generate deterministic base noise once so all ratios use the same noise.
    # Shape depends on policy type:
    #   - Flow matching (PI0.5): (1, chunk_size, max_action_dim)
    #   - Diffusion (DDPM/DDIM): (1, horizon, action_dim)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    if getattr(wrapper.config, "max_action_dim", None) is not None:
        # Flow matching path
        noise_shape = (1, wrapper.config.chunk_size, wrapper.config.max_action_dim)
    else:
        # Diffusion path
        action_dim = wrapper.config.output_features["action"].shape[0]
        noise_shape = (1, wrapper.config.horizon, action_dim)
    base_noise = torch.randn(noise_shape, device=device)

    results: dict[float, np.ndarray] = {}

    for ratio in ratios:
        all_raw_actions = get_action_chunk_for_ratio(
            wrapper,
            obs_preprocessor,
            obs_frames,
            guidance_actions_raw,
            ratio,
            camera_names,
            image_resize_mode,
            rename_map,
            device,
            task_description,
            drain_chunk,
            base_noise=base_noise,
        )

        results[ratio] = all_raw_actions

    return results


# ── FK-based EE trajectory ────────────────────────────────────────────────────


def compute_ee_from_states(wrapper, states_raw: np.ndarray) -> np.ndarray:
    """Run FK on a sequence of raw joint states. Returns [N, 3] EE positions."""
    positions = []
    for q in states_raw:
        wrapper._sync_joints(q[: wrapper.num_dofs])
        pos, _ = wrapper._get_ee_pose()
        positions.append(pos.copy())
    return np.array(positions)


def absolute_positions_to_ee_deltas(
    wrapper,
    init_state_raw: np.ndarray,
    target_states_raw: np.ndarray,
) -> np.ndarray:
    """Convert a sequence of absolute joint positions to EE-space deltas.

    Given an initial joint state and a sequence of target states, compute the
    7-d delta [dx, dy, dz, droll, dpitch, dyaw, gripper] between consecutive
    poses. The positional delta is expressed in the **current EE frame** (matching
    how _compute_next_joints applies it), and the rotation delta is the relative
    Euler XYZ rotation from current to next.

    init_state_raw:    [action_dim] raw joints for the starting pose.
    target_states_raw: [N, action_dim] raw joints for each target step.

    Returns [N, 7] array of EE deltas.
    """
    num_dofs = wrapper.num_dofs
    n_steps = target_states_raw.shape[0]
    deltas = np.zeros((n_steps, 7), dtype=np.float64)

    # Start from init state
    q_prev = init_state_raw[:num_dofs]
    wrapper._sync_joints(q_prev)
    pos_prev, quat_prev = wrapper._get_ee_pose()
    r_prev = Rotation.from_quat(quat_prev)

    for t in range(n_steps):
        q_next = target_states_raw[t, :num_dofs]
        wrapper._sync_joints(q_next)
        pos_next, quat_next = wrapper._get_ee_pose()
        r_next = Rotation.from_quat(quat_next)

        # Positional delta in EE (local) frame
        delta_pos_world = pos_next - pos_prev
        delta_pos_local = r_prev.inv().apply(delta_pos_world)

        # Rotation delta: R_delta = R_prev^{-1} * R_next
        r_delta = r_prev.inv() * r_next
        delta_rot = r_delta.as_euler("XYZ")

        # Gripper: pass the target value directly (not a delta — matches wrapper convention)
        gripper = float(target_states_raw[t, num_dofs]) if target_states_raw.shape[1] > num_dofs else 0.0

        deltas[t] = np.concatenate([delta_pos_local, delta_rot, [gripper]])

        # Advance
        pos_prev = pos_next.copy()
        r_prev = r_next
        q_prev = q_next

    return deltas


def compute_ee_trajectories(
    wrapper,
    init_obs_state_raw: np.ndarray,
    action_chunks_by_ratio: dict[float, np.ndarray],
) -> dict[float, np.ndarray]:
    """Compute end-effector XYZ positions for each ratio's action chunk using pybullet FK.

    init_obs_state_raw: raw (unnormalized) joint angles from the dataset, shape [action_dim].

    Returns {ratio: np.ndarray [n_action_steps + 1, 3]} where index 0 is the initial
    EE position from the observation state.
    """
    wrapper._sync_joints(init_obs_state_raw[: wrapper.num_dofs])
    # init_pos, _ = wrapper._get_ee_pose()

    trajectories: dict[float, np.ndarray] = {}
    for ratio, chunk in action_chunks_by_ratio.items():
        positions = []  # Don't include the observation in the ratio action chunk
        # positions = [init_pos.copy()]
        for t in range(chunk.shape[0]):
            wrapper._sync_joints(chunk[t][: wrapper.num_dofs])
            pos, _ = wrapper._get_ee_pose()
            positions.append(pos.copy())
        trajectories[ratio] = np.array(positions)  # [n_action_steps + 1, 3]

    return trajectories


# ── plotting ──────────────────────────────────────────────────────────────────


def _ratio_colors(ratios: list[float]):
    """Return a list of hex colors, one per ratio, using the plasma colormap."""
    cmap = plt.colormaps["plasma"].resampled(max(len(ratios), 2))
    return [cmap(i / max(len(ratios) - 1, 1)) for i in range(len(ratios))]


def plot_joint_angles(
    action_chunks_by_ratio: dict[float, np.ndarray],
    joint_names: list[str],
    episode_index: int,
    frame_index: int,
    obs_states_raw: np.ndarray | None = None,
    guidance_actions_raw: np.ndarray | None = None,
    output_path: Path | None = None,
    no_show: bool = False,
):
    """Grid of joint angle subplots, one colored line per forward_flow_ratio.

    obs_states_raw: optional [n_obs_steps, action_dim] raw joint states shown in gray
                    at timesteps [-n_obs_steps, …, -1] before the predicted chunk.
    guidance_actions_raw: optional [n_action_steps, action_dim] GT actions shown in green
                          at timesteps [0, …, n_action_steps-1] for reference.
    """
    ratios = sorted(action_chunks_by_ratio.keys())
    colors = _ratio_colors(ratios)
    n_dims = len(joint_names)
    n_cols = 3
    n_rows = ceil(n_dims / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))
    axes = np.array(axes).reshape(-1)

    n_obs = obs_states_raw.shape[0] if obs_states_raw is not None else 0

    for dim_idx in range(n_dims):
        ax = axes[dim_idx]

        # Observation context: gradient from light gray (far from t=0) to dark gray (t=-1).
        if obs_states_raw is not None and obs_states_raw.shape[0] > 0:
            obs_ts = np.arange(-n_obs, 0)
            n = len(obs_ts)
            light, dark = 0.82, 0.38
            # Per-point gray level: 0=earliest (lightest), n-1=latest (darkest)
            grays = [light + (dark - light) * (i / max(n - 1, 1)) for i in range(n)]
            # Draw line segments between consecutive points
            for i in range(n - 1):
                seg_gray = (grays[i] + grays[i + 1]) / 2
                ax.plot(
                    obs_ts[i : i + 2],
                    obs_states_raw[i : i + 2, dim_idx],
                    color=str(seg_gray),
                    linewidth=1.8,
                    linestyle="--",
                    zorder=1,
                )
            # Draw markers separately so each gets its own color
            for i in range(n):
                label = "observation" if (i == 0 and dim_idx == 0) else "_nolegend_"
                ax.plot(
                    [obs_ts[i]],
                    [obs_states_raw[i, dim_idx]],
                    color=str(grays[i]),
                    marker="o",
                    markersize=4,
                    markerfacecolor=str(grays[i]),
                    linewidth=0,
                    zorder=2,
                    label=label,
                )

        # Guidance (GT) actions in green.
        if guidance_actions_raw is not None and guidance_actions_raw.shape[0] > 0:
            g_ts = np.arange(guidance_actions_raw.shape[0])
            ax.plot(
                g_ts,
                guidance_actions_raw[:, dim_idx],
                color="green",
                linewidth=1.8,
                linestyle=":",
                marker="o",
                markersize=4,
                label="guidance" if dim_idx == 0 else "_nolegend_",
                zorder=3,
            )

        for ratio, color in zip(ratios, colors, strict=True):
            chunk = action_chunks_by_ratio[ratio]
            timesteps = np.arange(chunk.shape[0])
            ax.plot(
                timesteps, chunk[:, dim_idx], color=color, label=f"ratio={ratio:.2f}", linewidth=1.8, zorder=2
            )

        if n_obs > 0:
            ax.axvline(0, color="gray", linewidth=0.8, linestyle=":", alpha=0.6)
        ax.set_title(joint_names[dim_idx], fontsize=9)
        ax.set_xlabel("timestep")
        ax.set_ylabel("joint angle (rad)")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots.
    for idx in range(n_dims, len(axes)):
        axes[idx].set_visible(False)

    # Shared legend — collect from all axes to include the "observation" entry.
    all_handles, all_labels = [], []
    for ax in axes[:n_dims]:
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
            if label not in all_labels:
                all_handles.append(handle)
                all_labels.append(label)
    fig.legend(all_handles, all_labels, loc="lower right", fontsize=9, ncol=2)
    fig.suptitle(
        f"Joint angles — episode {episode_index}, frame {frame_index}",
        fontsize=11,
        y=1.01,
    )
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), bbox_inches="tight", dpi=150)
        print(f"Saved joint angles plot → {output_path}")
    if not no_show:
        plt.show()
    plt.close(fig)


def plot_ee_trajectories_3d(
    ee_trajectories_by_ratio: dict[float, np.ndarray],
    episode_index: int,
    frame_index: int,
    obs_ee_positions: np.ndarray | None = None,
    guidance_ee_positions: np.ndarray | None = None,
    output_path: Path | None = None,
    no_show: bool = False,
):
    """Interactive 3D plotly figure with one EE trajectory per forward_flow_ratio.

    obs_ee_positions: optional [n_obs_steps, 3] EE positions shown in gray before t=0.
    guidance_ee_positions: optional [n_action_steps, 3] GT EE positions shown in green.
    """
    ratios = sorted(ee_trajectories_by_ratio.keys())
    cmap = plt.colormaps["plasma"].resampled(max(len(ratios), 2))

    def to_hex(c):
        r, g, b, _ = c
        return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"

    fig = go.Figure()

    # Observation context: gradient from light gray (far from t=0) to dark gray (t=-1).
    if obs_ee_positions is not None and len(obs_ee_positions) > 0:
        n_obs_pts = len(obs_ee_positions)
        light, dark = 0.82, 0.38
        grays = [light + (dark - light) * (i / max(n_obs_pts - 1, 1)) for i in range(n_obs_pts)]

        def _gray_rgb(g: float) -> str:
            v = int(round(g * 255))
            return f"rgb({v},{v},{v})"

        # Line segments (one trace per segment so each can have its own color)
        for i in range(n_obs_pts - 1):
            seg_color = _gray_rgb((grays[i] + grays[i + 1]) / 2)
            fig.add_trace(
                go.Scatter3d(
                    x=obs_ee_positions[i : i + 2, 0],
                    y=obs_ee_positions[i : i + 2, 1],
                    z=obs_ee_positions[i : i + 2, 2],
                    mode="lines",
                    line={"color": seg_color, "width": 3, "dash": "dash"},
                    showlegend=False,
                )
            )
        # Markers (one trace with per-point colors; first gets the legend entry)
        fig.add_trace(
            go.Scatter3d(
                x=obs_ee_positions[:, 0],
                y=obs_ee_positions[:, 1],
                z=obs_ee_positions[:, 2],
                mode="markers",
                name="observation",
                marker={
                    "color": [_gray_rgb(g) for g in grays],
                    "size": [8] * (n_obs_pts - 1) + [10],
                    "symbol": "circle",
                    "opacity": 0.9,
                },
            )
        )

    # Guidance (GT) EE trajectory in green.
    if guidance_ee_positions is not None and len(guidance_ee_positions) > 0:
        n_g = len(guidance_ee_positions)
        sizes_g = [12] + [5] * (n_g - 2) + [8]
        symbols_g = ["diamond"] + ["circle"] * (n_g - 2) + ["x"]
        fig.add_trace(
            go.Scatter3d(
                x=guidance_ee_positions[:, 0],
                y=guidance_ee_positions[:, 1],
                z=guidance_ee_positions[:, 2],
                mode="lines+markers",
                name="guidance",
                line={"color": "green", "width": 3, "dash": "dot"},
                marker={"color": "green", "size": sizes_g, "symbol": symbols_g, "opacity": 0.9},
            )
        )

    for i, ratio in enumerate(ratios):
        traj = ee_trajectories_by_ratio[ratio]  # [n_action_steps+1, 3]
        color = to_hex(cmap(i / max(len(ratios) - 1, 1)))
        n = traj.shape[0]
        sizes = [12] + [5] * (n - 2) + [8]
        # ['circle', 'circle-open', 'cross', 'diamond',
        # 'diamond-open', 'square', 'square-open', 'x']
        symbols = ["diamond"] + ["circle"] * (n - 2) + ["x"]
        fig.add_trace(
            go.Scatter3d(
                x=traj[:, 0],
                y=traj[:, 1],
                z=traj[:, 2],
                mode="lines+markers",
                name=f"ratio={ratio:.2f}",
                line={"color": color, "width": 4},
                marker={"color": color, "size": sizes, "symbol": symbols, "opacity": 0.9},
            )
        )

    fig.update_layout(
        title={
            "text": f"EE trajectory — episode {episode_index}, frame {frame_index}",
            "font": {"size": 14},
        },
        scene={
            "xaxis_title": "X (m)",
            "yaxis_title": "Y (m)",
            "zaxis_title": "Z (m)",
            "aspectmode": "data",
        },
        legend={"font": {"size": 12}},
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        print(f"Saved EE trajectory → {output_path}")
    if not no_show:
        fig.show()


# ── main ──────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize SharedAutonomyPolicyWrapper predictions at different forward_flow_ratios."
    )
    parser.add_argument(
        "--policy_path",
        required=True,
        help="Path to pretrained model directory.",
    )
    parser.add_argument(
        "--dataset_repo_id",
        default="JennyWWW/splatsim_approach_lever_7_lowres_5path_10fails",
        help="HuggingFace dataset repo ID.",
    )
    parser.add_argument(
        "--dataset_dir",
        default=None,
        help=(
            "Local directory containing parquet files. "
            "Defaults to ~/.cache/huggingface/lerobot/{repo_id}/data."
        ),
    )
    parser.add_argument(
        "--episode_index",
        type=int,
        default=None,
        help="Episode to use (default: random from episodes >= 301).",
    )
    parser.add_argument(
        "--frame_index",
        type=int,
        default=None,
        help="Starting frame within episode (default: random).",
    )
    parser.add_argument(
        "--forward_flow_ratios",
        nargs="+",
        type=float,
        default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        help="List of forward_flow_ratio values to visualize.",
    )
    parser.add_argument(
        "--image_resize_mode",
        default="stretch",
        choices=["stretch", "letterbox"],
        help="Image resize mode used in the dataset (affects column names).",
    )
    parser.add_argument(
        "--camera_names",
        nargs="+",
        default=["base_rgb", "wrist_rgb"],
        help="Camera names.",
    )
    parser.add_argument(
        "--rename_map",
        type=json.loads,
        default=None,
        help=(
            'JSON rename map, e.g. \'{"observation.images.base_rgb_stretch": '
            '"observation.images.base_rgb"}\'. '
            "Defaults to renaming {cam}_{mode} → {cam} for each camera."
        ),
    )
    parser.add_argument(
        "--robot_name",
        default="robot_iphone_w_engine_new",
        help="Robot name for pybullet FK.",
    )
    parser.add_argument("--num_dofs", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to save plots (PNG + HTML). If not set, plots are only shown.",
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="Do not call plt.show() / fig.show().",
    )
    parser.add_argument(
        "--drain_chunk",
        action="store_true",
        help=(
            "Provide guidance only on the first call, then drain the blended chunk "
            "without re-blending. Produces temporally coherent action chunks. "
            "Without this flag, guidance is provided every step (re-blends each step)."
        ),
    )
    parser.add_argument(
        "--blend_strategy",
        default="denoise",
        choices=["denoise", "interpolate"],
        help=(
            "How to blend guidance with policy output. "
            "'denoise' uses partial denoising (flow matching / diffusion). "
            "'interpolate' uses simple linear interpolation in clean action space."
        ),
    )
    parser.add_argument(
        "--guidance_repr",
        default="absolute_pos",
        choices=["absolute_pos", "delta"],
        help=(
            "How guidance is represented. "
            "'absolute_pos' passes raw joint positions directly. "
            "'delta' converts consecutive positions to EE-space deltas via FK, "
            "then applies FK+IK per step in the wrapper. Both should produce "
            "equivalent results given the same dataset guidance."
        ),
    )
    parser.add_argument(
        "--n_anchor_steps",
        type=int,
        default=0,
        help=(
            "Number of action steps at the start of each chunk to anchor exactly to guidance "
            "via inpainting inside the denoising loop. 0 = full-chunk blending only (default)."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve dataset directory.
    if args.dataset_dir is not None:
        dataset_dir = Path(args.dataset_dir)
    else:
        dataset_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / args.dataset_repo_id / "data"
    print(f"Dataset dir: {dataset_dir}")

    # Load task description mapping (needed by PI0.5 preprocessor).
    task_map = load_task_description(dataset_dir)
    print(f"Task map: {task_map}")

    # Build default rename_map if not provided.
    if args.rename_map is None:
        rename_map = {
            f"observation.images.{cam}_{args.image_resize_mode}": f"observation.images.{cam}"
            for cam in args.camera_names
        }
    else:
        rename_map = args.rename_map
    print(f"Rename map: {rename_map}")

    # Load policy.
    print(f"Loading policy from {args.policy_path} …")
    wrapper, obs_preprocessor = load_wrapped_policy(
        policy_path=args.policy_path,
        forward_flow_ratio=1.0,  # initial value, overridden per ratio in visualization
        robot_name=args.robot_name,
        num_dofs=args.num_dofs,
        device=args.device,
    )
    wrapper.guidance_blend_strategy = GuidanceBlendStrategy(args.blend_strategy)
    wrapper.policy_guidance_representation = PolicyGuidanceRepresentation(args.guidance_repr)
    wrapper.n_anchor_steps = args.n_anchor_steps
    wrapper.skip_collision = True  # dataset guidance is known-safe; collision detection adds IK drift
    n_obs_steps = wrapper.config.n_obs_steps
    n_action_steps = wrapper.config.n_action_steps
    print(f"Policy type: {wrapper.config.type}, n_obs_steps={n_obs_steps}, n_action_steps={n_action_steps}")
    print(
        f"Blend strategy: {wrapper.guidance_blend_strategy.value}, guidance repr: {wrapper.policy_guidance_representation.value}"
    )

    # Select episode.
    if args.episode_index is None:
        available = get_available_episodes(dataset_dir, min_episode_index=301)
        if not available:
            raise RuntimeError(
                f"No episodes with index >= 301 found in {dataset_dir}. "
                "Use --episode_index to specify one manually."
            )
        episode_index = random.choice(available)
        print(f"Selected random episode: {episode_index}")
    else:
        episode_index = args.episode_index
        print(f"Using episode: {episode_index}")

    # Load episode to determine length, then pick frame.
    n_needed = n_obs_steps + n_action_steps
    # Load full episode quickly to find frame count.
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
        raise ValueError(
            f"Episode {episode_index} is too short ({ep_length} frames); need at least {n_needed}."
        )

    if args.frame_index is None:
        frame_index = random.randint(0, max_start)
        print(f"Selected random starting frame_index: {frame_index}")
    else:
        frame_index = args.frame_index
        print(f"Using frame_index: {frame_index}")

    # Load the actual frame data.
    frames_df = load_episode_frames(dataset_dir, episode_index, frame_index, n_needed)

    obs_frames = frames_df.iloc[:n_obs_steps]
    guidance_frames = frames_df.iloc[n_obs_steps : n_obs_steps + n_action_steps]
    guidance_actions_raw = np.stack(
        [np.array(row["action"], dtype=np.float32) for _, row in guidance_frames.iterrows()]
    )  # [n_action_steps, action_dim]
    print(
        f"Loaded {len(obs_frames)} obs frames + {len(guidance_frames)} guidance frames. "
        f"Action dim: {guidance_actions_raw.shape[1]}"
    )

    # Resolve task description for this episode (used by PI0.5 preprocessor).
    task_idx = int(frames_df.iloc[0].get("task_index", 1))
    task_description = task_map.get(task_idx)
    if task_description is not None:
        print(f"Task: '{task_description}' (task_index={task_idx})")
    else:
        print(f"No task description found for task_index={task_idx} — skipping task injection.")

    # For DELTA mode: convert absolute joint positions to EE-space deltas.
    # The wrapper's DELTA path expects [dx,dy,dz,droll,dpitch,dyaw,gripper] per step.
    # We keep guidance_actions_raw_for_plot as the original absolute positions for plotting.
    guidance_actions_raw_for_plot = guidance_actions_raw
    if args.guidance_repr == "delta":
        # Initial state = last observation frame's action (robot state when guidance starts)
        init_state = np.array(obs_frames.iloc[-1]["action"], dtype=np.float32)
        print("Converting absolute positions to EE deltas for DELTA mode …")
        guidance_actions_raw = absolute_positions_to_ee_deltas(wrapper, init_state, guidance_actions_raw)
        print(
            f"EE deltas shape: {guidance_actions_raw.shape}, "
            f"pos range: [{guidance_actions_raw[:, :3].min():.4f}, {guidance_actions_raw[:, :3].max():.4f}], "
            f"rot range: [{guidance_actions_raw[:, 3:6].min():.4f}, {guidance_actions_raw[:, 3:6].max():.4f}]"
        )

    # Compute action chunks for each ratio.
    print(f"Computing action chunks for ratios: {args.forward_flow_ratios} …")
    action_chunks = get_action_chunks_for_ratios(
        wrapper=wrapper,
        obs_preprocessor=obs_preprocessor,
        obs_frames=obs_frames,
        guidance_actions_raw=guidance_actions_raw,
        ratios=args.forward_flow_ratios,
        camera_names=args.camera_names,
        image_resize_mode=args.image_resize_mode,
        rename_map=rename_map,
        device=args.device,
        task_description=task_description,
        drain_chunk=args.drain_chunk,
    )
    print("Done computing action chunks.")

    # Compute EE trajectories.
    print("Computing EE trajectories via pybullet FK …")
    obs_states_raw = np.stack(
        [np.array(row["action"], dtype=np.float32) for _, row in obs_frames.iterrows()]
    )  # [n_obs_steps, action_dim]
    init_obs_state_raw = obs_states_raw[-1]

    ee_trajectories = compute_ee_trajectories(
        wrapper=wrapper,
        init_obs_state_raw=init_obs_state_raw,
        action_chunks_by_ratio=action_chunks,
    )
    obs_ee_positions = compute_ee_from_states(wrapper, obs_states_raw)  # [n_obs_steps, 3]
    guidance_ee_positions = compute_ee_from_states(
        wrapper, guidance_actions_raw_for_plot
    )  # [n_action_steps, 3]
    print("Done computing EE trajectories.")

    # Build joint names.
    action_dim = next(iter(action_chunks.values())).shape[1]
    joint_names = [f"joint_{i + 1}" for i in range(min(args.num_dofs, action_dim))]
    if action_dim > args.num_dofs:
        joint_names.append("gripper")

    # Resolve output paths.
    if args.output_dir is None:
        # Auto-generate from run parameters.
        train_config_path = Path(args.policy_path) / "train_config.json"
        with open(train_config_path) as f:
            policy_tag = json.load(f)["policy"]["type"]
        repr_tag = "delta" if args.guidance_repr == "delta" else "abspos"
        drain_tag = "onestep" if args.drain_chunk else "everystep"
        anchor_tag = f"anchor{args.n_anchor_steps}" if args.n_anchor_steps > 0 else "noanchor"
        parent = f"shared_autonomy_ep{episode_index}_frame{frame_index}"
        name = f"{policy_tag}_{args.blend_strategy}_{repr_tag}_{drain_tag}_{anchor_tag}"
        output_dir = Path("outputs/viz") / parent / name
    else:
        output_dir = Path(args.output_dir)
    print(f"Output dir: {output_dir}")

    joint_angles_path = (output_dir / "joint_angles.png") if output_dir else None
    ee_traj_path = (output_dir / "ee_trajectory.html") if output_dir else None

    # Plot joint angles.
    print("Plotting joint angles …")
    plot_joint_angles(
        action_chunks_by_ratio=action_chunks,
        joint_names=joint_names,
        episode_index=episode_index,
        frame_index=frame_index,
        obs_states_raw=obs_states_raw,
        guidance_actions_raw=guidance_actions_raw_for_plot,
        output_path=joint_angles_path,
        no_show=args.no_show,
    )

    # Plot EE trajectories.
    print("Plotting EE trajectories …")
    plot_ee_trajectories_3d(
        ee_trajectories_by_ratio=ee_trajectories,
        episode_index=episode_index,
        frame_index=frame_index,
        obs_ee_positions=obs_ee_positions,
        guidance_ee_positions=guidance_ee_positions,
        output_path=ee_traj_path,
        no_show=args.no_show,
    )

    print("Done.")


if __name__ == "__main__":
    main()
