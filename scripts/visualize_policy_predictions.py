#!/usr/bin/env python
"""Debug a policy checkpoint by comparing its predictions to dataset ground truth.

Loads a few consecutive frames from the dataset, runs the policy through the same
preprocessor → select_action → postprocessor pipeline as eval, then plots:

  1. Relative action deltas  — raw policy output before absolute conversion
     (should be small ~0 offsets, not huge jumps)
  2. Predicted absolute actions vs GT actions  — per joint + gripper
  3. State used for relative→absolute conversion  — shows if the state is correct

This tests the full eval pipeline without an env, isolating model/pipeline issues
from env issues.

Example:
    python my_scripts/visualize_policy_predictions.py \\
        --checkpoint outputs/training/pi05_approach_lever_7_lowres_5path_delta_basewrist/checkpoints/001000/pretrained_model \\
        --dataset_dir ~/.cache/huggingface/lerobot/JennyWWW/splatsim_approach_lever_7_lowres_5path \\
        --episode 5 --start_frame 0 --n_frames 5 \\
        --camera_names base_rgb wrist_rgb \\
        --image_resize_mode letterbox \\
        --task_description "upright_small_engine_new"
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import einops
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from lerobot.policies.factory import _reconnect_relative_absolute_steps, get_policy_class
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME
from lerobot.utils.lerobot_dataset_utils import resolve_dataset_dir

JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]


# ── data helpers ──────────────────────────────────────────────────────────────


def decode_image(image_data) -> np.ndarray:
    img_bytes = image_data["bytes"] if isinstance(image_data, dict) else image_data
    return np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))


def load_episode_frames(dataset_dir: Path, episode: int, start_frame: int, n_frames: int) -> pd.DataFrame:
    parquet_files = sorted(dataset_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files under {dataset_dir}")
    dfs = []
    for f in parquet_files:
        df = pd.read_parquet(f, filters=[("episode_index", "==", episode)])
        if len(df):
            dfs.append(df)
    if not dfs:
        raise ValueError(f"Episode {episode} not found in {dataset_dir}")
    df = pd.concat(dfs, ignore_index=True).sort_values("frame_index").reset_index(drop=True)
    mask = df["frame_index"] >= start_frame
    df = df[mask].head(n_frames).reset_index(drop=True)
    if len(df) < n_frames:
        print(f"Warning: only {len(df)} frames available from frame {start_frame} (wanted {n_frames})")
    return df


def row_to_obs(
    row: pd.Series, camera_names: list[str], image_resize_mode: str, task_description: str | None, device: str
) -> dict:
    """Build a raw observation dict from a parquet row (before preprocessor)."""
    obs: dict = {}
    for cam in camera_names:
        col = f"observation.images.{cam}_{image_resize_mode}"
        img_np = decode_image(row[col])
        img_t = torch.from_numpy(img_np).unsqueeze(0)
        img_t = einops.rearrange(img_t, "b h w c -> b c h w").float() / 255.0
        obs[col] = img_t.to(device)
    state = np.array(row["observation.state"], dtype=np.float32)
    obs["observation.state"] = torch.from_numpy(state).unsqueeze(0).to(device)
    if task_description is not None:
        obs["task"] = [task_description]
    return obs


# ── inference ─────────────────────────────────────────────────────────────────


@torch.no_grad()
def _fresh_predict_at_frame(
    policy,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    frames: pd.DataFrame,
    target_idx: int,
    camera_names: list[str],
    image_resize_mode: str,
    task_description: str | None,
    device: str,
) -> np.ndarray:
    """Get a fresh policy prediction at target_idx (the first action of a newly-predicted chunk).

    Handles policies with n_obs_steps > 1 (e.g. diffusion) by draining a throwaway
    chunk while filling the inner obs queue, then shifting the obs queue to the
    correct history window before the real predict call. Mirrors the approach in
    visualize_shared_autonomy.py.
    """
    policy.reset()
    n_obs_steps = getattr(policy.config, "n_obs_steps", 1)
    n_action_steps = getattr(policy.config, "n_action_steps", 1)

    def _preprocess_at(i: int):
        row = frames.iloc[max(i, 0)]  # clip to first loaded frame for missing history
        return preprocessor(row_to_obs(row, camera_names, image_resize_mode, task_description, device))

    # Phase 1: drain most of the throwaway chunk using target obs (fills obs queue with obs[t])
    n_filler_drain = n_action_steps - (n_obs_steps - 1)
    target_batch = _preprocess_at(target_idx)
    for _ in range(n_filler_drain):
        _ = policy.select_action(target_batch)  # discard

    # Phase 2: pop remaining filler actions while shifting obs queue to the true history window
    # [obs[t - (n_obs-1)], ..., obs[t-1]]. The next real call will push obs[t] to complete it.
    for i in range(n_obs_steps - 1):
        hist_idx = target_idx - (n_obs_steps - 1 - i)
        _ = policy.select_action(_preprocess_at(hist_idx))  # discard

    # Phase 3: real prediction — action queue is now empty, obs queue has correct history.
    # This call triggers fresh predict_action_chunk anchored at state[t] and returns chunk[0].
    action_norm = policy.select_action(target_batch)
    action_abs = postprocessor(action_norm).detach().cpu().numpy().reshape(-1)
    return action_abs


@torch.no_grad()
def run_policy_on_frames(
    policy,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    frames: pd.DataFrame,
    camera_names: list[str],
    image_resize_mode: str,
    task_description: str | None,
    device: str,
) -> dict:
    """Run policy on each frame independently (fresh chunk per frame) and collect results.

    Returns a dict with:
      - "pred_absolute": (N, action_dim) predicted actions in absolute joint space
      - "pred_relative": (N, action_dim) predicted relative deltas (before absolute conversion)
      - "gt_actions": (N, action_dim) ground-truth actions from dataset (absolute)
      - "states": (N, state_dim) observation.state used for each step
    """
    pred_absolute = []
    pred_relative = []
    gt_actions = []
    states = []

    for i, (_, row) in enumerate(frames.iterrows()):
        # Ground truth
        gt_actions.append(np.array(row["action"], dtype=np.float32))
        state_np = np.array(row["observation.state"], dtype=np.float32)
        states.append(state_np)

        # Fresh prediction at this frame (handles n_obs_steps > 1 properly)
        action_abs = _fresh_predict_at_frame(
            policy,
            preprocessor,
            postprocessor,
            frames,
            i,
            camera_names,
            image_resize_mode,
            task_description,
            device,
        )
        pred_absolute.append(action_abs)

        # Derive the relative delta: absolute - state (for joints), absolute (for gripper).
        # With a fresh prediction each frame, this is now apples-to-apples with gt_delta.
        pred_relative.append(action_abs - state_np)

    return {
        "pred_absolute": np.stack(pred_absolute),
        "pred_relative": np.stack(pred_relative),
        "gt_actions": np.stack(gt_actions),
        "states": np.stack(states),
    }


# ── plotting ──────────────────────────────────────────────────────────────────


def plot_results(results: dict, episode: int, start_frame: int, save_path: Path | None = None):
    pred_abs = results["pred_absolute"]  # (N, 7)
    pred_rel = results["pred_relative"]  # (N, 7)
    gt = results["gt_actions"]  # (N, 7)
    states = results["states"]  # (N, 7)
    n_frames, action_dim = pred_abs.shape
    joint_names = JOINT_NAMES[:action_dim]
    xs = np.arange(n_frames) + start_frame

    fig, axes = plt.subplots(3, action_dim, figsize=(4 * action_dim, 9))
    fig.suptitle(
        f"Policy predictions — episode {episode}, frames {start_frame}–{start_frame + n_frames - 1}",
        fontsize=13,
    )

    for j, name in enumerate(joint_names):
        # Row 0: absolute predicted vs GT
        ax = axes[0, j]
        ax.plot(xs, gt[:, j], "k-o", markersize=4, label="GT (absolute)")
        ax.plot(xs, pred_abs[:, j], "r--s", markersize=4, label="Pred (absolute)")
        ax.plot(xs, states[:, j], "b:^", markersize=3, label="State")
        ax.set_title(name, fontsize=9)
        if j == 0:
            ax.set_ylabel("absolute (rad/m)", fontsize=8)
            ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

        # Row 1: relative delta predicted vs GT relative delta
        gt_delta = gt[:, j] - states[:, j]
        ax = axes[1, j]
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.plot(xs, gt_delta, "k-o", markersize=4, label="GT delta")
        ax.plot(xs, pred_rel[:, j], "r--s", markersize=4, label="Pred delta")
        if j == 0:
            ax.set_ylabel("relative delta", fontsize=8)
            ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

        # Row 2: absolute prediction error
        ax = axes[2, j]
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.plot(xs, pred_abs[:, j] - gt[:, j], "m-o", markersize=4)
        if j == 0:
            ax.set_ylabel("pred - GT error", fontsize=8)
            ax.set_xlabel("frame index", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    # Print diagnostics
    print("\n── Diagnostics ──────────────────────────────────────────────────")
    print(
        f"{'Joint':<12} {'GT delta mean':>14} {'GT delta std':>13} {'Pred delta mean':>16} {'Pred delta std':>14} {'Pred-GT abs err':>16}"
    )
    for j, name in enumerate(joint_names):
        gt_delta = gt[:, j] - states[:, j]
        pd_delta = pred_rel[:, j]
        err = pred_abs[:, j] - gt[:, j]
        print(
            f"{name:<12} {gt_delta.mean():>14.4f} {gt_delta.std():>13.4f} "
            f"{pd_delta.mean():>16.4f} {pd_delta.std():>14.4f} {err.mean():>16.4f}"
        )
    print()

    # Sanity check: large relative deltas are a red flag
    max_pred_delta = np.abs(pred_rel[:, :-1]).max()  # exclude gripper
    max_gt_delta = np.abs(gt[:, :-1] - states[:, :-1]).max()
    print(f"Max |predicted relative delta| (joints 1-6): {max_pred_delta:.4f}")
    print(f"Max |GT relative delta|        (joints 1-6): {max_gt_delta:.4f}")
    if max_pred_delta > 5 * max_gt_delta + 0.1:
        print("⚠️  Predicted deltas are MUCH larger than GT — policy may be predicting in wrong space")
    elif max_pred_delta < 0.001:
        print("⚠️  Predicted deltas are near zero — policy may not have learned yet (undertrained)")
    else:
        print("✓  Predicted delta magnitude looks comparable to GT")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"\nSaved plot to {save_path}")
    else:
        plt.show()


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True, help="Path to pretrained_model dir")
    parser.add_argument(
        "--dataset_repo_id",
        default="JennyWWW/splatsim_approach_lever_7_lowres_5path_10fails",
        help="HuggingFace dataset repo ID. Used to auto-resolve --dataset_dir if not given.",
    )
    parser.add_argument(
        "--dataset_dir",
        default=None,
        help=(
            "Local directory containing parquet files (the data/ dir). "
            "Defaults to auto-resolved from --dataset_repo_id (flat layout under "
            "$HF_LEROBOT_HOME or snapshot cache under $HF_LEROBOT_HUB_CACHE)."
        ),
    )
    parser.add_argument("--episode", type=int, default=0, help="Episode index to sample from")
    parser.add_argument("--start_frame", type=int, default=0, help="Starting frame within the episode")
    parser.add_argument("--n_frames", type=int, default=5, help="Number of frames to run")
    parser.add_argument("--camera_names", nargs="+", default=["base_rgb", "wrist_rgb"])
    parser.add_argument("--image_resize_mode", default="letterbox")
    parser.add_argument(
        "--task_description", default=None, help="Task description string for language-conditioned policies"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save", default=None, help="Save plot to this path instead of showing")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    data_dir = resolve_dataset_dir(args.dataset_repo_id, args.dataset_dir)
    print(f"Dataset data dir: {data_dir}")

    # ── load policy ──────────────────────────────────────────────────────────
    print(f"Loading policy from {checkpoint}")
    with open(checkpoint / "config.json") as f:
        config_data = json.load(f)
    policy_cls = get_policy_class(config_data["type"])
    policy = policy_cls.from_pretrained(str(checkpoint)).to(args.device).eval()

    preprocessor = PolicyProcessorPipeline.from_pretrained(
        str(checkpoint), config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        str(checkpoint),
        config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
        to_transition=policy_action_to_transition,  # select_action returns Tensor (=PolicyAction)
        to_output=transition_to_policy_action,  # output is also Tensor
    )

    print(f"Preprocessor steps: {[s.__class__.__name__ for s in preprocessor.steps]}")
    print(f"Postprocessor steps: {[s.__class__.__name__ for s in postprocessor.steps]}")

    # Wire absolute↔relative steps and attach the policy so the relative step only
    # refreshes its anchor state on chunk boundaries (matches lerobot-eval behavior).
    _reconnect_relative_absolute_steps(preprocessor, postprocessor, policy=policy)

    # ── load data ────────────────────────────────────────────────────────────
    print(f"\nLoading {args.n_frames} frames from episode {args.episode}, frame {args.start_frame}")
    frames = load_episode_frames(data_dir, args.episode, args.start_frame, args.n_frames)
    print(f"Loaded {len(frames)} frames (frame indices: {frames['frame_index'].tolist()})")

    # ── run inference ────────────────────────────────────────────────────────
    print(f"\nRunning policy on {len(frames)} frames...")
    results = run_policy_on_frames(
        policy,
        preprocessor,
        postprocessor,
        frames,
        args.camera_names,
        args.image_resize_mode,
        args.task_description,
        args.device,
    )

    # ── plot ─────────────────────────────────────────────────────────────────
    save_path = Path(args.save) if args.save else None
    plot_results(results, args.episode, args.start_frame, save_path)


if __name__ == "__main__":
    main()
