"""Dataset augmentation via closed-loop blended-policy rollouts.

Loads source episodes from an intervention-style LeRobot dataset, replays each
one through ``SharedAutonomyPolicyWrapper`` at one or more blend ratios in
closed loop against splatsim, and writes the resulting (observation, action)
trajectories to a target LeRobotDataset. Each output episode is tagged with
``source_episode_idx``, ``blend_ratio``, and (when present) the source
episode's ``source_scenario_idx`` so the augmented dataset can be merged
with the original — or with arbitrary ratio subsets of it — for training.

Per-output-episode rollout length matches the source episode length: every
source-action timestep gets executed through the wrapper (with the source
action stream as guidance) and the resulting (env_obs, action) pair is
written to the target dataset.

Argument names are aligned with visualize_shared_autonomy_sim.py so you can
copy the same env / policy / guidance flags across both scripts.

Example (single episode, mirrors the visualize command):

    python my_scripts/augment_dataset_with_blending.py \\
        --policy_path=outputs/training/.../checkpoints/006000/pretrained_model \\
        --dataset_repo_id=JennyWWW/splatsim_..._rrt_pi05 \\
        --target_dataset_repo_id=JennyWWW/splatsim_..._rrt_pi05_blended \\
        --forward_flow_ratios='[0.0, 0.5, 1.0]' \\
        --episode_index=305 \\
        --blend_strategy=denoise --guidance_repr=absolute_pos --drain_chunk \\
        --env_task=upright_small_engine_new --env_external_port=6001

Example (bulk — all episodes 0–49):

    python my_scripts/augment_dataset_with_blending.py \\
        --policy_path=... --dataset_repo_id=... --target_dataset_repo_id=... \\
        --forward_flow_ratios='[0.0, 0.5, 1.0]' \\
        --episode_range='[0, 50]' \\
        --env_task=upright_small_engine_new --env_external_port=6001
"""

# NOTE: do not add `from __future__ import annotations` — parser.wrap reads
# the function's annotation at runtime to infer the draccus config class, and
# stringified annotations break that lookup.

import csv
import faulthandler
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any

# Surface Python+C tracebacks for SIGSEGV / SIGABRT / SIGFPE / SIGILL /
# SIGBUS — without this, native crashes from pybullet / CUDA / shared
# memory / forkserver workers print only "Aborted (core dumped)" with no
# context. The C frames in the traceback (libpython / libc / libcuda /
# pybullet) usually point at the failing call.
faulthandler.enable(all_threads=True)

# matplotlib's default TkAgg backend initializes Tcl/Tk at first pyplot
# import, which then crashes with "Tcl_AsyncDelete: async handler deleted by
# the wrong thread" once splatsim's pybullet GUI thread is running. Force
# the non-interactive Agg backend BEFORE any matplotlib import — our
# sibling import below (visualize_shared_autonomy.py) does
# `import matplotlib.pyplot as plt` at module load, which would otherwise
# bake in TkAgg before visualize_shared_autonomy_sim's module-level Agg
# call gets a chance to run.
import matplotlib  # noqa: E402

matplotlib.use("Agg")

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

# Sibling-script imports. visualize_shared_autonomy_sim.py owns the env /
# batch / seeding helpers we reuse.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from visualize_shared_autonomy_sim import (  # type: ignore[import-not-found]  # noqa: E402
    _build_sim_batch,
    _run_filler_phase,
)

from lerobot.configs import parser  # noqa: E402
from lerobot.envs.factory import (  # noqa: E402
    make_env,
    make_env_config,
    make_env_pre_post_processors,
)
from lerobot.policies.shared_autonomy_wrapper import (  # noqa: E402
    BlendMode,
    GuidanceBlendStrategy,
    PolicyGuidanceRepresentation,
)
from lerobot.utils.constants import ACTION  # noqa: E402
from lerobot.utils.import_utils import register_third_party_plugins  # noqa: E402
from lerobot.utils.lerobot_dataset_utils import make_default_rename_map, resolve_dataset_dir  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402
from lerobot.utils.sim_seeding import seed_splatsim_env_to_state  # noqa: E402
from lerobot.utils.utils import init_logging  # noqa: E402
from my_scripts.visualize_shared_autonomy_DEPRECATED import (  # type: ignore[import-not-found]  # noqa: E402
    find_parquet_files,
    load_episode_frames,
    load_task_description,
    load_wrapped_policy,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class AugmentationConfig:
    # ── Shared with visualize_shared_autonomy_sim.py ──────────────────────────
    # Use the same flag names so commands are easy to copy between scripts.

    # Source dataset (≡ visualize's --dataset_repo_id).
    dataset_repo_id: str = ""
    # Local override for the source parquet dir (≡ visualize's --dataset_dir).
    dataset_dir: str | None = None

    policy_path: str = ""

    # Blend ratios (≡ visualize's --forward_flow_ratios). Each
    # (source_episode × ratio) → one output episode in the target dataset.
    # 0.0 = pure human guidance; 1.0 = pure policy.
    forward_flow_ratios: list[float] = field(default_factory=lambda: [0.5])

    # Episode selection — choose ONE of the two (mutually exclusive):
    #   episode_index:   single int ("305"), inclusive range ("300-310"), or
    #                    the literal "all" / "ALL" (every episode in the source
    #                    dataset). Mirrors visualize_shared_autonomy_sim.py's
    #                    --episode_index, with the "all" shorthand added so
    #                    callers (e.g. dagger_orchestrate.sh) can be explicit
    #                    about meaning "every episode" instead of relying on
    #                    the field being unset.
    #   episode_indices: explicit JSON list, e.g. '[3, 8, 23]'.
    # If both are None, every episode is processed (same as --episode_index=all).
    episode_index: str | None = None
    episode_indices: list[int] | None = None

    # ── Augment-specific ──────────────────────────────────────────────────────
    target_dataset_repo_id: str = ""

    # Wrapper config.
    blend_mode: str = "once_per_chunk"  # "once_per_chunk" | "every_step"
    blend_strategy: str = "denoise"  # "denoise" | "interpolate"
    guidance_repr: str = "absolute_pos"  # "absolute_pos" | "delta"
    n_anchor_steps: int = 0
    n_action_steps: int | None = None  # None ⇒ keep policy's default
    drain_chunk: bool = True

    # Env config — must match the source dataset's image keys so the
    # augmented dataset can be merged with the source for training.
    env_task: str = "upright_small_engine_new"
    env_robot_name: str = "robot_iphone_w_engine_new"
    env_camera_names: list[str] = field(default_factory=lambda: ["base_rgb", "wrist_rgb"])
    env_image_resize_modes: list[str] = field(default_factory=lambda: ["letterbox", "stretch"])
    env_fps: int = 30
    env_episode_length: int = 1_000_000  # very large; we don't truncate

    # The benchmark dataset whose per-scenario object / robot poses are
    # restored on each env.reset(). Each source episode in
    # ``source_dataset_repo_id`` carries a ``source_scenario_idx`` that
    # indexes into this dataset (the intervention recording stored only that
    # pointer, not the per-episode geometry). Default matches the benchmark
    # we record corrections against.
    eval_benchmark_repo_id: str = "JennyWWW/eval_splatsim_approach_lever_benchmark_1000"

    # SplatSim must run out-of-process — the wrapper's pybullet GUI client
    # in this process can't coexist with an in-process env (pybullet refuses
    # a second local GUI), and AsyncVectorEnv's dummy-env lifecycle hits the
    # SplatSim Tk GUI's Tcl_AsyncDelete abort. Launch splatsim manually:
    #
    #     cd ~/code/SplatSim && \
    #         python scripts/launch_nodes.py \
    #             --robot sim_ur_pybullet_small_engine_new_interactive \
    #             --robot_port 6001 \
    #             --robot_name robot_iphone_w_engine_new \
    #             --eval_benchmark_repo_id JennyWWW/eval_splatsim_approach_lever_benchmark_1000
    #
    # Then point this script at it via --env_external_port. The simulator
    # stays up across runs; only the augmentation script restarts.
    env_external_port: int = 6001
    env_external_host: str = "127.0.0.1"

    # Frame schema for the new dataset. image_keys are derived as
    # ``[f"{cam}_{mode}" for cam in env_camera_names for mode in env_image_resize_modes]``
    # so the augmented dataset matches the intervention dataset's schema.
    num_dofs: int = 6

    # rename_map: {sim-side key → policy-side key}. None ⇒ default map
    # (each ``{cam}_{first_mode}`` → ``{cam}``).
    rename_map: dict[str, str] | None = None

    # Where the per-episode CSV goes; the dataset itself goes to
    # $HF_LEROBOT_HOME/{target_dataset_repo_id} via LeRobotDataset.create.
    output_dir: str = "outputs/augment_dataset_with_blending"

    # If True, push the finalized dataset to the Hub at the end (creates
    # the repo if missing). False keeps it local-only.
    push_to_hub: bool = False

    device: str = "cuda"
    seed: int = 0


def _resolve_episode_selection(cfg: AugmentationConfig, available_eps: list[int]) -> list[int]:
    """Resolve episode selection from --episode_index / --episode_indices.

    ``--episode_index`` accepts:
      * a single int ("305"),
      * an inclusive range ("300-310" → episodes 300, 301, …, 310),
      * the literal "all" (case-insensitive) → every available episode.
    ``--episode_indices`` accepts an explicit JSON list ('[3, 8, 23]').
    If neither is set, all available episodes are used (same as ``=all``).
    """
    if cfg.episode_index is not None and cfg.episode_indices is not None:
        raise ValueError("Set at most one of --episode_index and --episode_indices.")

    if cfg.episode_index is not None:
        s = str(cfg.episode_index).strip()
        if s.lower() == "all":
            return sorted(available_eps)
        if "-" in s:
            parts = s.split("-", 1)
            start, end = int(parts[0]), int(parts[1])
            selected = list(range(start, end + 1))  # inclusive
        else:
            selected = [int(s)]
    elif cfg.episode_indices is not None:
        selected = [int(i) for i in cfg.episode_indices]
    else:
        return sorted(available_eps)

    available_set = set(available_eps)
    missing = [e for e in selected if e not in available_set]
    if missing:
        raise ValueError(
            f"Selected episode(s) not present in source dataset: {missing[:10]}"
            + (f" (and {len(missing) - 10} more)" if len(missing) > 10 else "")
        )
    return selected


# ---------------------------------------------------------------------------
# Frame construction
# ---------------------------------------------------------------------------


def _build_frame(
    raw_obs: dict[str, Any],
    gym_obs: dict[str, Any],
    action: np.ndarray,
    image_keys: list[str],
    task: str,
) -> dict[str, Any]:
    """Build a LeRobot frame from this step's env observations and action.

    Mirrors ``TeleopRecordingWrapper._build_frame`` so the resulting dataset is
    schema-compatible with intervention recordings. ``raw_obs`` is what the
    splatsim robot server returns directly (has ``{cam}_{mode}`` image keys
    for every resize mode); ``gym_obs`` is the post-_to_gym_obs form (used
    only for the agent_pos state, since raw_obs doesn't put it under a
    canonical name).
    """
    state = np.asarray(gym_obs["agent_pos"], dtype=np.float32).reshape(-1)
    frame: dict[str, Any] = {
        "observation.state": state,
        "action": np.asarray(action, dtype=np.float32).reshape(-1),
        "task": task,
    }
    for key in image_keys:
        img = raw_obs.get(key)
        if img is None:
            raise RuntimeError(
                f"Image key '{key}' missing from raw_obs (have: {list(raw_obs)[:6]}…). "
                f"Make sure --env_image_resize_modes covers every mode the source "
                f"dataset used."
            )
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        frame[f"observation.images.{key}"] = np.asarray(img, dtype=np.float32)
    return frame


def _get_raw_obs(vec_env: gym.vector.VectorEnv) -> dict[str, Any]:
    """Pull raw_obs (with ``{cam}_{mode}`` image keys) from the single-env vec.

    ``vec_env.call`` returns a tuple of length n_envs; we only ever use n=1.
    """
    raw = vec_env.call("get_observations")
    if isinstance(raw, tuple | list):
        return raw[0]
    return raw  # type: ignore[return-value]


def _unbatch_obs(env_obs: dict[str, Any]) -> dict[str, Any]:
    """Strip the batch (n_envs=1) dim from a vec-env obs. Used for state lookup."""
    out: dict[str, Any] = {}
    for k, v in env_obs.items():
        if k == "pixels" and isinstance(v, dict):
            out[k] = {ck: cv[0] if hasattr(cv, "__len__") and len(cv) > 0 else cv for ck, cv in v.items()}
        elif hasattr(v, "__len__") and not isinstance(v, str | bytes) and len(v) > 0:
            out[k] = v[0]
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Closed-loop rollout
# ---------------------------------------------------------------------------


@dataclass
class RolloutResult:
    frames: list[dict[str, Any]]
    n_steps: int


@torch.no_grad()
def rollout_closed_loop_for_augmentation(
    *,
    wrapper,
    obs_preprocessor,
    vec_env: gym.vector.VectorEnv,
    env_preprocessor,
    env_postprocessor,
    seed_joint_state: np.ndarray,
    episode_seed: list[int] | None,
    guidance_actions_raw: np.ndarray,
    ratio: float,
    blend_mode: BlendMode,
    drain_chunk: bool,
    total_steps: int,
    rename_map: dict[str, str],
    image_keys: list[str],
    task_description: str,
    device: str,
) -> RolloutResult:
    """Run one closed-loop rollout and capture (raw_obs, action) per step.

    Uses the same filler / real-phase structure as
    ``visualize_shared_autonomy_sim.get_sim_action_chunk_for_ratio``.
    The difference: we capture (raw_obs, action) as LeRobot frames at each
    step instead of just collecting raw actions for plotting.

    ``episode_seed`` is forwarded to ``seed_splatsim_env_to_state`` so the
    server loads the correct benchmark scenario (e.g. ``[source_scenario_idx]``).
    """
    n_action_steps: int = wrapper.config.n_action_steps
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")

    wrapper.reset()
    wrapper.forward_flow_ratio = ratio
    wrapper.blend_mode = blend_mode

    # Reset the env to the right scenario and (for local envs) teleport the
    # robot to the per-episode start joints. A single call handles both.
    env_obs = seed_splatsim_env_to_state(
        vec_env,
        joint_state=seed_joint_state,
        num_dofs=wrapper.num_dofs,
        seed=episode_seed,
    )

    # ─── Filler phase (shared with visualize_shared_autonomy_sim) ────────────
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
    MIN_FRAMES = 60  # noqa: N806 - pad with hold frames if episode ends early

    frames: list[dict[str, Any]] = []
    success = False
    terminal_env_obs: dict | None = None
    terminal_raw_obs: dict | None = None
    hold_action: np.ndarray | None = None

    for t in range(total_steps):
        # ── Hold mode: episode succeeded, don't step env again ────────────────
        # Stepping after termination triggers AutoresetMode.NEXT_STEP and would
        # bring in the next scene's images, causing a sharp visual transition.
        if success:
            assert terminal_raw_obs is not None and terminal_env_obs is not None
            frames.append(
                _build_frame(
                    raw_obs=terminal_raw_obs,
                    gym_obs=_unbatch_obs(terminal_env_obs),
                    action=hold_action,
                    image_keys=image_keys,
                    task=task_description,
                )
            )
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

        action_norm = wrapper.select_action(batch)
        raw_action = wrapper.postprocessor(action_norm)

        if env_postprocessor is not None:
            _post_out = env_postprocessor({ACTION: raw_action})
            if _post_out is not None:
                raw_action = _post_out[ACTION]

        action_numpy = raw_action.detach().to("cpu").numpy()  # (1, action_dim)

        # Capture (s_t, a_t) — obs before the step, action we're about to send.
        raw_obs = _get_raw_obs(vec_env)
        frames.append(
            _build_frame(
                raw_obs=raw_obs,
                gym_obs=_unbatch_obs(env_obs),
                action=action_numpy.reshape(-1),
                image_keys=image_keys,
                task=task_description,
            )
        )

        env_obs, _reward, _term, _trunc, _info = vec_env.step(action_numpy)

        # Check for success / termination.
        terminated = bool(_term[0]) if hasattr(_term, "__len__") else bool(_term)
        if terminated and not success:
            success = True
            # Snapshot terminal state BEFORE the next step() would reset.
            terminal_env_obs = env_obs
            terminal_raw_obs = _get_raw_obs(vec_env)
            agent_pos = env_obs.get("agent_pos")
            hold_action = (
                np.asarray(agent_pos[0], dtype=np.float32)
                if agent_pos is not None
                else action_numpy.reshape(-1)
            )
            logger.info(
                "Episode succeeded at t=%d/%d. Holding for %d remaining frames.",
                t + 1,
                total_steps,
                total_steps - t - 1,
            )

    # Pad to MIN_FRAMES if the rollout was shorter (very short source episode).
    # IMPORTANT: use dict copies, not the same reference. dataset_writer.add_frame
    # does frame.pop("task") which mutates the dict in place — sharing references
    # would cause the second add_frame call to fail with "Missing features: {'task'}".
    if frames and len(frames) < MIN_FRAMES:
        last_frame = frames[-1]
        while len(frames) < MIN_FRAMES:
            frames.append(dict(last_frame))

    return RolloutResult(frames=frames, n_steps=len(frames))


# ---------------------------------------------------------------------------
# Source-dataset helpers
# ---------------------------------------------------------------------------


def _load_source_episodes_meta(source_dataset_dir: Path) -> pd.DataFrame:
    """Read the source dataset's episodes parquet so we can copy through any
    per-episode metadata (e.g. ``source_scenario_idx``) into the augmented
    dataset's per-episode metadata.
    """
    ep_files = sorted((source_dataset_dir / "meta" / "episodes").rglob("*.parquet"))
    if not ep_files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in ep_files], ignore_index=True)


def _episode_length(parquet_files: list[Path], episode_idx: int) -> int:
    """Count frames in a source episode without loading them."""
    n = 0
    for pf in parquet_files:
        df = pd.read_parquet(pf, columns=["episode_index"])
        n += int((df["episode_index"] == episode_idx).sum())
    return n


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


@dataclass
class AugmentedEpisodeResult:
    target_episode_idx: int
    source_episode_idx: int
    source_scenario_idx: int | None
    blend_ratio: float
    n_frames: int
    elapsed_s: float


def run_augmentation(
    cfg: AugmentationConfig,
    *,
    csv_path: Path | None = None,
) -> list[AugmentedEpisodeResult]:
    """Top-level loop. Builds env+wrapper once, then per (source_ep, ratio)
    seeds, rolls out, captures frames, and saves to the target dataset.
    """
    from splatsim.utils.lerobot_utils import (
        create_lerobot_dataset,
        finalize_lerobot_dataset,
        load_lerobot_dataset,
    )

    # ── Resolve source dataset on disk ─────────────────────────────────────
    # resolve_dataset_dir returns the `data/` subdirectory (used by
    # find_parquet_files / load_episode_frames). The episodes-meta parquet
    # lives at the dataset ROOT under `meta/episodes/`, so we need the parent.
    source_data_dir = Path(resolve_dataset_dir(cfg.dataset_repo_id, cfg.dataset_dir))
    if not source_data_dir.exists():
        raise FileNotFoundError(f"Source data dir does not exist: {source_data_dir}")
    source_root_dir = source_data_dir.parent
    logger.info("Source dataset root: %s (data subdir: %s)", source_root_dir, source_data_dir)

    source_episodes_meta = _load_source_episodes_meta(source_root_dir)
    parquet_files = find_parquet_files(source_data_dir)
    task_map = load_task_description(source_data_dir)

    # ── Resolve which source episodes to process ───────────────────────────
    if not source_episodes_meta.empty:
        available_eps = sorted(int(i) for i in source_episodes_meta["episode_index"].tolist())
    else:
        # Fall back to scanning data parquets for unique episode_index.
        seen: set[int] = set()
        for pf in parquet_files:
            df = pd.read_parquet(pf, columns=["episode_index"])
            seen.update(int(i) for i in df["episode_index"].unique())
        available_eps = sorted(seen)

    episode_indices = _resolve_episode_selection(cfg, available_eps)
    if not episode_indices:
        raise ValueError("No source episodes selected.")
    logger.info(
        "Will augment %d source episode(s) × %d ratio(s) = %d output episode(s)",
        len(episode_indices),
        len(cfg.forward_flow_ratios),
        len(episode_indices) * len(cfg.forward_flow_ratios),
    )

    # ── Connect to externally-launched splatsim via ZMQ ───────────────────
    # SplatSim must run out-of-process — see the AugmentationConfig docstring
    # and the launch_nodes.py invocation it shows.
    logger.info(
        "Connecting to splatsim ZMQ server at %s:%d …",
        cfg.env_external_host,
        cfg.env_external_port,
    )
    env_cfg_obj = make_env_config(
        "splatsim",
        task=cfg.env_task,
        robot_name=cfg.env_robot_name,
        camera_names=cfg.env_camera_names,
        image_resize_modes=cfg.env_image_resize_modes,
        fps=cfg.env_fps,
        episode_length=cfg.env_episode_length,
        external_port=cfg.env_external_port,
        external_host=cfg.env_external_host,
        eval_benchmark_repo_id=cfg.eval_benchmark_repo_id,
        eval_benchmark_subset=None,
        include_oracle_info=False,
    )
    env_dict = make_env(env_cfg_obj, n_envs=1, use_async_envs=False)
    vec_env = env_dict["splatsim"][0]

    # ── Build wrapped policy (acquires its own pybullet GUI in parent) ────
    logger.info("Loading wrapped policy from %s …", cfg.policy_path)
    wrapper, obs_preprocessor = load_wrapped_policy(
        policy_path=cfg.policy_path,
        device=cfg.device,
    )
    wrapper.guidance_blend_strategy = GuidanceBlendStrategy(cfg.blend_strategy)
    wrapper.policy_guidance_representation = PolicyGuidanceRepresentation(cfg.guidance_repr)
    wrapper.n_anchor_steps = cfg.n_anchor_steps
    if cfg.n_action_steps is not None:
        prev = wrapper.config.n_action_steps
        wrapper.config.n_action_steps = cfg.n_action_steps
        logger.info("Overrode n_action_steps: %d → %d", prev, cfg.n_action_steps)
    blend_mode_enum = BlendMode(cfg.blend_mode)
    wrapper.blend_mode = blend_mode_enum

    # Release the CPU copy of model weights that may linger after .to("cuda").
    # The safetensors file is mmap'd during from_pretrained; the OS caches those
    # pages (up to ~6 GiB for PI0.5) and can force other pages to swap out.
    # Explicit GC + CUDA cache flush reclaims that headroom before the rollout loop.
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Now that the policy config is loaded, build the env-side processors.
    env_pre, env_post = make_env_pre_post_processors(env_cfg_obj, wrapper.config)

    rename_map = cfg.rename_map or make_default_rename_map(
        cfg.env_camera_names, cfg.env_image_resize_modes[0]
    )
    logger.info("rename_map: %s", rename_map)

    # ── Build target dataset (matches source's image-key schema) ───────────
    image_keys = [f"{cam}_{mode}" for cam in cfg.env_camera_names for mode in cfg.env_image_resize_modes]
    existing = load_lerobot_dataset(cfg.target_dataset_repo_id)
    if existing is not None:
        logger.warning(
            "Target dataset %s already exists locally — appending to it. Delete the "
            "directory if you want a fresh dataset.",
            cfg.target_dataset_repo_id,
        )
        target_ds = existing
    else:
        target_ds = create_lerobot_dataset(
            cfg.target_dataset_repo_id,
            fps=cfg.env_fps,
            image_keys=image_keys,
            num_dofs=cfg.num_dofs,
        )

    # ── Per-source-episode CSV writer ──────────────────────────────────────
    csv_writer = None
    csv_file = None
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        # File spans the whole rollout; closed in the outer finally.
        csv_file = open(csv_path, "w", newline="")  # noqa: SIM115
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [
                "target_episode_idx",
                "source_episode_idx",
                "source_scenario_idx",
                "blend_ratio",
                "n_frames",
                "elapsed_s",
            ]
        )
        csv_file.flush()

    results: list[AugmentedEpisodeResult] = []
    target_ep_idx = int(target_ds.meta.total_episodes)

    try:
        # Single env, scenarios pinned via seed-based reset. The simulator-
        # side _handle_reset (in EVAL_BENCHMARK mode) interprets seed as
        # ``subset[seed % len(subset)]``; with subset=None (=> full benchmark)
        # this is just ``scenario_idx == seed``. So the loop restores each
        # source episode's underlying scenario by passing seed=source_scenario_idx.
        for source_ep in tqdm(episode_indices, desc="source episodes", leave=True):
            ep_length = _episode_length(parquet_files, source_ep)
            if ep_length <= 0:
                logger.warning("Source episode %d has 0 frames; skipping", source_ep)
                continue

            # Load enough frames to cover the full episode (n_obs_steps obs +
            # ep_length-n_obs_steps guidance). We only need actions+state+task
            # here; images come from the live env.
            n_obs_steps = wrapper.config.n_obs_steps
            frames_needed = ep_length
            frames_df = load_episode_frames(source_data_dir, source_ep, frame_index=0, n_frames=frames_needed)
            seed_joint_state = np.asarray(frames_df.iloc[n_obs_steps - 1]["action"], dtype=np.float32)
            guidance_actions_raw = np.stack(
                [
                    np.asarray(row["action"], dtype=np.float32)
                    for _, row in frames_df.iloc[n_obs_steps:].iterrows()
                ]
            )
            total_steps = guidance_actions_raw.shape[0]

            task_idx = int(frames_df.iloc[0].get("task_index", 1))
            task_description = (task_map.get(task_idx) if task_map else None) or cfg.env_task

            source_scenario_idx: int | None = None
            if not source_episodes_meta.empty and "source_scenario_idx" in source_episodes_meta.columns:
                row = source_episodes_meta.loc[
                    source_episodes_meta["episode_index"] == source_ep, "source_scenario_idx"
                ]
                if not row.empty and pd.notna(row.iloc[0]):
                    source_scenario_idx = int(row.iloc[0])

            if source_scenario_idx is None:
                # No source_scenario_idx in metadata (plain training dataset, not an
                # intervention recording). Fall back to using the episode index as the
                # scenario seed. This works when the eval_benchmark_repo_id scenarios
                # are numbered to match the source dataset's episode indices. Double-
                # check that the benchmark you passed has a matching scene for this
                # episode, otherwise the scene geometry will be wrong.
                logger.warning(
                    "Source episode %d has no source_scenario_idx; falling back to "
                    "episode_index=%d as the benchmark scenario seed. Make sure "
                    "--eval_benchmark_repo_id scenario %d matches this episode's "
                    "scene geometry.",
                    source_ep,
                    source_ep,
                    source_ep,
                )
                source_scenario_idx = source_ep

            logger.info(
                "Source ep %d → scenario %d, total_steps=%d",
                source_ep,
                source_scenario_idx,
                total_steps,
            )

            for ratio in cfg.forward_flow_ratios:
                t0 = time.time()
                rollout = rollout_closed_loop_for_augmentation(
                    wrapper=wrapper,
                    obs_preprocessor=obs_preprocessor,
                    vec_env=vec_env,
                    env_preprocessor=env_pre,
                    env_postprocessor=env_post,
                    seed_joint_state=seed_joint_state,
                    episode_seed=[source_scenario_idx],
                    guidance_actions_raw=guidance_actions_raw,
                    ratio=float(ratio),
                    blend_mode=blend_mode_enum,
                    drain_chunk=cfg.drain_chunk,
                    total_steps=total_steps,
                    rename_map=rename_map,
                    image_keys=image_keys,
                    task_description=task_description,
                    device=cfg.device,
                )

                # Commit one episode per (source_ep, ratio) pair.
                # Pass a copy of each frame — dataset_writer.add_frame does
                # frame.pop("task") which mutates the dict in place. Copying
                # here is defensive against shared references (e.g. padding).
                n_frames = len(rollout.frames)
                for frame in rollout.frames:
                    target_ds.add_frame(dict(frame))
                del rollout  # free image buffers before video encoding in save_episode
                gc.collect()
                episode_metadata: dict[str, Any] = {
                    "source_episode_idx": int(source_ep),
                    "blend_ratio": float(ratio),
                }
                if source_scenario_idx is not None:
                    episode_metadata["source_scenario_idx"] = int(source_scenario_idx)
                target_ds.save_episode(episode_metadata=episode_metadata)

                elapsed = time.time() - t0
                result = AugmentedEpisodeResult(
                    target_episode_idx=target_ep_idx,
                    source_episode_idx=int(source_ep),
                    source_scenario_idx=source_scenario_idx,
                    blend_ratio=float(ratio),
                    n_frames=n_frames,
                    elapsed_s=elapsed,
                )
                results.append(result)
                logger.info(
                    "Saved target ep %d ← source_ep=%d, ratio=%.2f, frames=%d (%.1fs).",
                    target_ep_idx,
                    source_ep,
                    ratio,
                    n_frames,
                    elapsed,
                )
                if csv_writer is not None and csv_file is not None:
                    csv_writer.writerow(
                        [
                            target_ep_idx,
                            int(source_ep),
                            source_scenario_idx if source_scenario_idx is not None else "",
                            f"{float(ratio):.4f}",
                            n_frames,
                            f"{elapsed:.2f}",
                        ]
                    )
                    csv_file.flush()
                target_ep_idx += 1
    finally:
        if csv_file is not None:
            csv_file.close()
        try:
            vec_env.close()
        except Exception:
            logger.exception("vec_env.close() raised — ignoring during shutdown.")

    finalize_lerobot_dataset(target_ds)

    # Write a README / dataset card so the augmentation provenance is visible
    # on the HuggingFace dataset page when push_to_hub=True.
    _write_dataset_readme(cfg, results)

    if cfg.push_to_hub:
        logger.info("Pushing %s to Hub …", cfg.target_dataset_repo_id)
        target_ds.push_to_hub()
    return results


def _write_dataset_readme(cfg: AugmentationConfig, results: list["AugmentedEpisodeResult"]) -> None:
    """Write a README.md dataset card into the target dataset directory.

    HuggingFace renders this as the dataset page description. When push_to_hub=True
    the file is uploaded automatically.
    """
    from lerobot.utils.constants import HF_LEROBOT_HOME

    dataset_root = HF_LEROBOT_HOME / cfg.target_dataset_repo_id
    if not dataset_root.exists():
        logger.warning("Dataset root %s not found; skipping README write.", dataset_root)
        return

    n_episodes = len(results)
    n_source = len({r.source_episode_idx for r in results})
    ratios_str = ", ".join(f"`{r}`" for r in sorted({r.blend_ratio for r in results}))

    # Resolve which source episodes were processed.
    ep_idx = cfg.episode_index
    if ep_idx is not None and "-" in str(ep_idx):
        ep_desc = f"episodes `{ep_idx}` (range)"
    elif ep_idx is not None:
        ep_desc = f"episode `{ep_idx}`"
    elif cfg.episode_indices is not None:
        ep_desc = f"episodes `{cfg.episode_indices}`"
    else:
        ep_desc = f"all episodes ({n_source} total)"

    readme = f"""\
---
tags:
  - lerobot
  - splatsim
  - shared-autonomy
  - augmented
---

# {cfg.target_dataset_repo_id.split("/")[-1]}

Augmented dataset generated by `augment_dataset_with_blending.py`.

## Source dataset
`{cfg.dataset_repo_id}` — {ep_desc}

## Policy used for blending
```
{cfg.policy_path}
```

## Augmentation parameters
| Parameter | Value |
|---|---|
| `forward_flow_ratios` | {cfg.forward_flow_ratios} |
| `blend_strategy` | `{cfg.blend_strategy}` |
| `guidance_repr` | `{cfg.guidance_repr}` |
| `drain_chunk` | `{cfg.drain_chunk}` |
| `blend_mode` | `{cfg.blend_mode}` |
| `n_anchor_steps` | `{cfg.n_anchor_steps}` |
| `n_action_steps` | `{cfg.n_action_steps}` |

## Environment
| Parameter | Value |
|---|---|
| `env_task` | `{cfg.env_task}` |
| `env_robot_name` | `{cfg.env_robot_name}` |
| `env_camera_names` | `{cfg.env_camera_names}` |
| `env_image_resize_modes` | `{cfg.env_image_resize_modes}` |
| `eval_benchmark_repo_id` | `{cfg.eval_benchmark_repo_id}` |

## Output
- **{n_episodes} output episode(s)** from {n_source} source episode(s) × {len(cfg.forward_flow_ratios)} ratio(s) ({ratios_str})
"""

    readme_path = dataset_root / "README.md"
    readme_path.write_text(readme, encoding="utf-8")
    logger.info("Wrote dataset README → %s", readme_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@parser.wrap()
def augment_main(cfg: AugmentationConfig):
    logging.info("Augmentation config:\n%s", pformat(cfg.__dict__))

    if not cfg.dataset_repo_id:
        raise ValueError("--dataset_repo_id is required.")
    if not cfg.target_dataset_repo_id:
        raise ValueError("--target_dataset_repo_id is required.")
    if not cfg.policy_path:
        raise ValueError("--policy_path is required.")
    if not cfg.forward_flow_ratios:
        raise ValueError("--ratios must contain at least one float.")

    set_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "augmentation_per_episode.csv"
    logging.info("Per-episode CSV: %s", csv_path)

    results = run_augmentation(cfg, csv_path=csv_path)

    logging.info(
        "Done. %d episode(s) saved across %d source × %d ratios. CSV: %s",
        len(results),
        len({r.source_episode_idx for r in results}),
        len({r.blend_ratio for r in results}),
        csv_path,
    )


def main():
    init_logging()
    register_third_party_plugins()
    # Draccus treats bool fields as requiring a value (--flag=true). To keep
    # the CLI identical to visualize_shared_autonomy_sim.py, convert bare
    # boolean flags (--drain_chunk, --push_to_hub) to --flag=true before
    # draccus sees them.
    _BOOL_FLAGS = {"--drain_chunk", "--push_to_hub"}  # noqa: N806
    sys.argv = [arg + "=true" if arg in _BOOL_FLAGS else arg for arg in sys.argv]
    augment_main()


if __name__ == "__main__":
    main()
