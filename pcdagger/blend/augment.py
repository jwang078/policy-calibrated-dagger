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
        --blend_strategy=denoise --guidance_repr=absolute_pos --blend_interval_frac \\
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
import json
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

# Sibling-module imports. These previously came from
# ``my_scripts.visualize_shared_autonomy_DEPRECATED`` (which only resolved by
# accident as a side effect of the sim visualizer's sys.path manipulation);
# they've been split into topic-focused library modules so this script doesn't
# depend on a deprecated file. Bare module names (no ``my_scripts.`` prefix)
# so they resolve when this script is invoked via
# ``python my_scripts/augment_dataset_with_blending.py``.
from lib_dataset_episode_io import (  # type: ignore[import-not-found]  # noqa: E402
    find_parquet_files,
    load_episode_frames,
    load_task_description,
)
from lib_sa_policy_loading import (  # type: ignore[import-not-found]  # noqa: E402
    apply_clip_sample_override,
    load_wrapped_policy,
)
from lib_sa_rollout import (  # type: ignore[import-not-found]  # noqa: E402,F401
    progress_guidance_index,  # re-export kept for external importers
    run_blended_rollout,
    warn_if_sim_physics_unsynced,
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
from lerobot.utils.import_utils import register_third_party_plugins  # noqa: E402
from lerobot.utils.lerobot_dataset_utils import make_default_rename_map, resolve_dataset_dir  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402
from lerobot.utils.sim_seeding import set_env_benchmark_indices  # noqa: E402
from lerobot.utils.utils import init_logging  # noqa: E402

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
    # Interval between guidance re-blends, as a fraction of the executed chunk
    # (n_action_steps), in [0, 1]: 1.0 = blend at chunk boundaries only
    # (legacy --drain_chunk=true), 0.0 = re-blend every step (legacy false),
    # fraction f = re-blend every ceil(f * n_action_steps) ticks (e.g. 0.5 →
    # twice per chunk). Fractional values REQUIRE --blend_mode=every_step —
    # once_per_chunk's drain path ignores mid-chunk guidance, so the extra
    # blend ticks would silently do nothing (validated at startup). The CLI
    # still accepts --drain_chunk as a deprecated alias, the bare flag
    # (→ 1.0), and true/false spellings (→ 1.0/0.0).
    blend_interval_frac: float = 1.0
    # Pin ONE noise draw per episode ("common random numbers") and reuse it for
    # every select_action call, instead of letting the wrapper draw fresh
    # torch.randn each tick.
    #
    # This is the fix for `blend_mode=every_step` jitter. In every_step mode the
    # wrapper re-runs a FULL denoising pass each tick and only action[0] of that
    # fresh chunk is executed — so with independent noise per tick, consecutive
    # executed actions are independent samples from the policy's action
    # distribution. The guidance sets their MEAN, but the tick-to-tick DELTA is
    # dominated by resampling noise rather than by the (smooth) trajectory,
    # which reads as shaking even at low ratios where guidance dominates.
    # Pinning the noise makes consecutive passes differ only because the
    # observation/guidance moved — which is the intended "continuous steering"
    # semantics of every_step.
    #
    # Irrelevant to once_per_chunk (one draw per chunk already), where enabling
    # it just makes runs reproducible. Default False = unchanged behavior for
    # existing blend datasets. Mirrors visualize_shared_autonomy_sim.py, which
    # has always pinned a shared base_noise across ratios.
    fixed_base_noise: bool = False
    # DEBUG: override the checkpoint's DDPM/DDIM `clip_sample` (None = keep the
    # trained value). clip_sample=True clamps the predicted clean action to
    # ±clip_sample_range at EVERY denoising step — with out-of-distribution
    # guidance (robot stuck, demo far ahead: encoded deltas can exceed the
    # rel-stats range → |normalized| > 1) each step slams into that clamp,
    # which is one suspected cause of blend-mode shaking. Setting false lets
    # the denoiser express the OOD target un-clamped: if the stuck-shake turns
    # into large smooth lurches, the clip-fight is confirmed; if unchanged,
    # the clamp wasn't the mechanism. EVAL-MISMATCHED — debug only, don't
    # train on datasets produced this way without understanding the tradeoff.
    clip_sample: bool | None = None
    # Show a translucent green "ghost" robot at the guidance target pose in
    # the SA wrapper's pybullet window (needs the wrapper GUI, e.g.
    # --keep_sa_gui at the orchestrator level). Purely visual — the ghost is
    # collision-disabled and invisible to the RRT/shield collision world.
    show_guidance_ghost: bool = False
    # PROGRESS-AWARE guidance indexing. Default (False) indexes the demo by
    # wall-clock tick: guidance = demo_actions[t:]. If the robot falls behind
    # (contact, stall, blend detour), the demo marches on anyway and guidance
    # runs unboundedly ahead — the "green ghost diverges and never comes back"
    # failure. True re-indexes each tick by PROGRESS: find the demo step whose
    # action (≈ demo state) is closest to the robot's CURRENT joints, searched
    # in a forward window from the previous match (monotonic — never rewinds),
    # and pass demo_actions[j*:]. A stuck robot then holds guidance at its
    # current demo point (waits for the robot) instead of racing ahead, and a
    # recovered robot re-converges to the demo where it actually is.
    progress_guidance: bool = False
    # Forward search window (demo steps) for the progress match. Bounds both
    # compute and how far a single tick can jump ahead.
    progress_guidance_window: int = 45
    # Start the replay at this source-episode frame instead of frame 0. The
    # robot is teleported to the demo's pose at that frame (works over ZMQ —
    # the SplatSim server dispatches teleport_joint_state), so guidance and
    # robot stay aligned for nonzero starts. Episodes whose remaining length
    # after start_frame can't fit n_obs_steps + 1 guidance frame are skipped
    # with a warning. Applies to EVERY source episode of the run.
    start_frame: int = 0

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
    env_state_dim: int = 0,
) -> dict[str, Any]:
    """Build a LeRobot frame from this step's env observations and action.

    Mirrors ``TeleopRecordingWrapper._build_frame`` so the resulting dataset is
    schema-compatible with intervention recordings — including
    ``observation.environment_state`` when the source dataset declares it
    (``env_state_dim > 0``). ``raw_obs`` is what the splatsim robot server
    returns directly (has ``{cam}_{mode}`` image keys for every resize mode);
    ``gym_obs`` is the post-_to_gym_obs form (used for agent_pos and
    environment_state, since raw_obs doesn't put them under canonical names).
    """
    state = np.asarray(gym_obs["agent_pos"], dtype=np.float32).reshape(-1)
    frame: dict[str, Any] = {
        "observation.state": state,
        "action": np.asarray(action, dtype=np.float32).reshape(-1),
        "task": task,
    }
    if env_state_dim > 0:
        env_state = gym_obs.get("environment_state")
        if env_state is None:
            raise RuntimeError(
                f"Source dataset declares observation.environment_state "
                f"(dim={env_state_dim}) but the env observation has no "
                f"'environment_state' key (have: {list(gym_obs)}). The sim server "
                f"is not publishing env_state — check env_state_dim wiring in "
                f"make_env_config."
            )
        env_state = np.asarray(env_state, dtype=np.float32).reshape(-1)
        if env_state.shape[0] != env_state_dim:
            raise RuntimeError(
                f"environment_state width mismatch: env published {env_state.shape[0]}, "
                f"source dataset schema says {env_state_dim}."
            )
        frame["observation.environment_state"] = env_state
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
    guidance_actions_raw: np.ndarray,
    ratio: float,
    blend_mode: BlendMode,
    blend_interval_frac: float,
    total_steps: int,
    progress_guidance: bool = False,
    progress_guidance_window: int = 45,
    demo_states_raw: np.ndarray | None = None,
    rename_map: dict[str, str],
    image_keys: list[str],
    task_description: str,
    device: str,
    playlist_pos: int | None = None,
    env_state_dim: int = 0,
    base_noise: torch.Tensor | None = None,
) -> RolloutResult:
    """Run one closed-loop rollout and capture (raw_obs, action) per step.

    Thin frame-capture adapter over :func:`lib_sa_rollout.run_blended_rollout`
    — the SAME core the debug visualizer (``visualize_shared_autonomy_sim``)
    drives, so whatever the visualizer validates is what this script records.
    The only caller-specific parts are injected callbacks: ``on_step`` builds
    a LeRobot frame from the pre-step raw obs, and ``on_success`` snapshots
    the terminal raw obs for the post-success hold frames.

    The scenario for this rollout is picked by the sim server's EVAL_BENCHMARK
    counter walking whatever playlist was set via ``set_env_benchmark_indices``
    (see ``run_augmentation`` — the playlist is installed once before the
    outer loop and mirrors the source ``source_scenario_idx`` sequence).
    ``playlist_pos`` pins the reset to that exact playlist slot via
    ``benchmark_start_index`` instead of trusting the server counter's
    position — GUI interactions (or the dropdown echo of a duplicate scenario
    id) can move the counter between rollouts, silently replaying the wrong
    scenario. ``None`` falls back to counter-order.
    """
    MIN_FRAMES = 60  # noqa: N806 - pad with hold frames if episode ends early

    frames: list[dict[str, Any]] = []
    terminal_raw_obs: dict | None = None

    def _on_success(terminal_env_obs: dict) -> None:
        # Snapshot terminal raw state BEFORE the next step() would reset.
        nonlocal terminal_raw_obs
        terminal_raw_obs = _get_raw_obs(vec_env)

    def _on_step(t: int, env_obs: dict, action_1d: np.ndarray, is_hold: bool) -> None:
        # Capture (s_t, a_t) — obs before the step, action we're about to send.
        # Hold ticks reuse the frozen terminal obs (stepping after termination
        # would trigger AutoresetMode.NEXT_STEP and pull in the next scene).
        raw_obs = terminal_raw_obs if is_hold else _get_raw_obs(vec_env)
        assert raw_obs is not None
        frames.append(
            _build_frame(
                raw_obs=raw_obs,
                gym_obs=_unbatch_obs(env_obs),
                action=action_1d,
                image_keys=image_keys,
                task=task_description,
                env_state_dim=env_state_dim,
            )
        )

    run_blended_rollout(
        wrapper=wrapper,
        obs_preprocessor=obs_preprocessor,
        vec_env=vec_env,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        seed_joint_state=seed_joint_state,
        guidance_actions_raw=guidance_actions_raw,
        ratio=ratio,
        blend_mode=blend_mode,
        blend_interval_frac=blend_interval_frac,
        total_steps=total_steps,
        rename_map=rename_map,
        device=device,
        task_description=task_description,
        seed=None,
        benchmark_start_index=playlist_pos,
        base_noise=base_noise,
        progress_guidance=progress_guidance,
        progress_guidance_window=progress_guidance_window,
        demo_states_raw=demo_states_raw,
        on_step=_on_step,
        on_success=_on_success,
        log=lambda msg: logger.info(msg),
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


def _existing_provenance_pairs(target_root: Path) -> set[tuple[int, float]]:
    """(source_episode_idx, blend_ratio) pairs already committed to the target.

    Read from the target's per-episode metadata parquets. The blend loop skips
    pairs in this set, making re-invocation IDEMPOTENT: a run that died between
    dataset completion and the stats sidecar (or mid-blending) can be resumed
    by simply re-running — previously the script re-blended EVERYTHING and
    appended, silently duplicating every episode (observed: r_dag1_blend* at
    2x, their _nocoll siblings at 3x, 2026-07-31).

    Ratios are rounded to 6 decimals for the set membership so float noise in
    parquet round-trips can't defeat the match. Episodes without provenance
    columns (foreign/legacy data) are ignored — they never match, so the run
    degrades to the old append behavior for them.
    """
    pairs: set[tuple[int, float]] = set()
    for f in sorted((Path(target_root) / "meta" / "episodes").glob("chunk-*/*.parquet")):
        try:
            df = pd.read_parquet(f)
        except Exception:
            continue
        if "source_episode_idx" not in df.columns or "blend_ratio" not in df.columns:
            continue
        for se, br in zip(df["source_episode_idx"], df["blend_ratio"]):
            if pd.notna(se) and pd.notna(br):
                pairs.add((int(se), round(float(br), 6)))
    return pairs


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
        build_lerobot_features,
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
    # Read authoritative feature shapes from the source dataset's meta/info.json.
    # `SplatSimEnv`'s state_dim / action_dim / env_state_dim default to UR5
    # (7/7/0). If we don't override them, gymnasium's SyncVectorEnv pre-allocates
    # its `out` buffer from the env's observation_space at those UR5 shapes, then
    # env.reset() returns whatever the sim server ACTUALLY publishes (planar arm:
    # state=(4,), env_state=(8,)) — `np.stack` compares out.shape to items[0]
    # shape and raises "ValueError: Output array is the wrong shape" before the
    # first tick. Reading from meta/info.json is authoritative because that JSON
    # was written by the sim server itself when the source dataset was recorded,
    # so its shapes MATCH what the sim will publish now. Alternative — trusting
    # cfg.num_dofs + adding cfg.env_state_dim — would drift the two sources
    # apart again. Same reasoning as the num_dofs/robot_name forwarding fix in
    # load_wrapped_policy above: single source of truth for shapes = the source
    # dataset itself.
    _source_info_path = source_root_dir / "meta" / "info.json"
    with open(_source_info_path) as _f:
        _source_info = json.load(_f)
    _source_feats = _source_info.get("features", {})
    _state_shape = _source_feats.get("observation.state", {}).get("shape", [7])
    _action_shape = _source_feats.get("action", {}).get("shape", [7])
    _env_state_shape = _source_feats.get("observation.environment_state", {}).get("shape", [0])
    source_state_dim = int(_state_shape[0]) if _state_shape else 7
    source_action_dim = int(_action_shape[0]) if _action_shape else 7
    source_env_state_dim = int(_env_state_shape[0]) if _env_state_shape else 0
    # Same reasoning as the dims fix above — derive camera names + resize
    # modes from the source dataset's `observation.images.{cam}_{mode}` keys,
    # rather than the config's UR5-style defaults (which include a wrist
    # camera that the planar env doesn't have — and searching for
    # `wrist_rgb_letterbox` in the sim's obs dict raises RuntimeError inside
    # `_build_frame`). Convention: dataset keys are
    # `observation.images.{cam_with_underscores}_{resize_mode}` — split on
    # the LAST underscore (resize modes are single tokens: letterbox / stretch).
    _KNOWN_RESIZE_MODES = ("letterbox", "stretch")
    _src_img_keys = sorted(k for k in _source_feats if k.startswith("observation.images."))
    _cams_seen: list[str] = []
    _modes_seen: list[str] = []
    for _k in _src_img_keys:
        _stem = _k[len("observation.images.") :]  # e.g. "base_rgb_letterbox"
        _mode = next((m for m in _KNOWN_RESIZE_MODES if _stem.endswith("_" + m)), None)
        if _mode is None:
            # No known suffix — treat the whole thing as the camera name
            # (e.g. a legacy dataset that didn't tag resize mode into the key).
            _cam = _stem
        else:
            _cam = _stem[: -len("_" + _mode)]
            if _mode not in _modes_seen:
                _modes_seen.append(_mode)
        if _cam not in _cams_seen:
            _cams_seen.append(_cam)
    source_camera_names = _cams_seen  # may be [] for state-only datasets
    source_image_resize_modes = _modes_seen if _modes_seen else ["letterbox"]
    logger.info(
        "Source dataset feature shapes: state_dim=%d, action_dim=%d, env_state_dim=%d",
        source_state_dim,
        source_action_dim,
        source_env_state_dim,
    )
    logger.info(
        "Source dataset image keys: cameras=%s, resize_modes=%s",
        source_camera_names,
        source_image_resize_modes,
    )
    # Overwrite the CLI defaults with the derived values BEFORE any downstream
    # consumer uses them (rename_map builder + `image_keys` frame-schema
    # builder later in this function). Without this, those consumers still
    # look for e.g. `wrist_rgb_letterbox` and blow up at `_build_frame` when
    # the sim's obs dict is missing that key.
    if list(cfg.env_camera_names) != source_camera_names:
        logger.info(
            "Overriding env_camera_names %s → %s (from source dataset)",
            list(cfg.env_camera_names),
            source_camera_names,
        )
        cfg.env_camera_names = source_camera_names
    if list(cfg.env_image_resize_modes) != source_image_resize_modes:
        logger.info(
            "Overriding env_image_resize_modes %s → %s (from source dataset)",
            list(cfg.env_image_resize_modes),
            source_image_resize_modes,
        )
        cfg.env_image_resize_modes = source_image_resize_modes
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
        num_dofs=cfg.num_dofs,
        state_dim=source_state_dim,
        action_dim=source_action_dim,
        env_state_dim=source_env_state_dim,
    )
    env_dict = make_env(env_cfg_obj, n_envs=1, use_async_envs=False)
    vec_env = env_dict["splatsim"][0]
    warn_if_sim_physics_unsynced(vec_env, log=logger.info)

    # ── Build wrapped policy (acquires its own pybullet GUI in parent) ────
    logger.info("Loading wrapped policy from %s …", cfg.policy_path)
    # `action_names_dataset_hint=cfg.dataset_repo_id` ensures the
    # RelativeActionsProcessorStep's `action_names` gets backfilled from a
    # dataset we KNOW is on disk (the blend source intervention dataset),
    # rather than the policy's training dataset which may have been deleted
    # by the orchestrator after training (e.g. per-round merged datasets).
    # Without this, `exclude_joints=['gripper']` is silently ignored and the
    # recorded blend gripper column leaks the gripper STATE into the action
    # column — see _backfill_rel_step_action_names docstring.
    wrapper, obs_preprocessor = load_wrapped_policy(
        policy_path=cfg.policy_path,
        device=cfg.device,
        action_names_dataset_hint=cfg.dataset_repo_id,
        # Pass the CLI-configured robot_name AND num_dofs through so the SA
        # wrapper loads the CORRECT URDF and sizes its internal state to the
        # CORRECT DoF count. Both fields default in load_wrapped_policy to
        # UR5 assumptions (robot_iphone_w_engine_new, num_dofs=6); without
        # forwarding, the wrapper's internal PyBullet loads a different
        # robot or expects a different action-vector width than the sim
        # server is publishing — observation vector shapes disagree and
        # gymnasium's SyncVectorEnv.reset fails with "ValueError: Output
        # array is the wrong shape" at np.stack. For planar_3joint this
        # matters especially because the URDF has 6 MOVABLE joints (3 arm
        # + 3 gripper), so `num_dofs` auto-inference from URDF gets 6, not
        # 3. Same env_robot_name / num_dofs used above for make_env_config;
        # single source of truth is cfg.env_robot_name / cfg.num_dofs.
        robot_name=cfg.env_robot_name,
        num_dofs=cfg.num_dofs,
        # Ghost needs a visible pybullet window on the wrapper's client.
        pb_gui=cfg.show_guidance_ghost,
    )
    apply_clip_sample_override(wrapper, cfg.clip_sample)

    wrapper.show_guidance_ghost = cfg.show_guidance_ghost
    wrapper.guidance_blend_strategy = GuidanceBlendStrategy(cfg.blend_strategy)
    wrapper.policy_guidance_representation = PolicyGuidanceRepresentation(cfg.guidance_repr)
    wrapper.n_anchor_steps = cfg.n_anchor_steps
    if cfg.n_action_steps is not None:
        prev = wrapper.config.n_action_steps
        wrapper.config.n_action_steps = cfg.n_action_steps
        logger.info("Overrode n_action_steps: %d → %d", prev, cfg.n_action_steps)
    blend_mode_enum = BlendMode(cfg.blend_mode)
    if not 0.0 <= cfg.blend_interval_frac <= 1.0:
        raise ValueError(f"--blend_interval_frac must be in [0, 1], got {cfg.blend_interval_frac}")
    if 0.0 < cfg.blend_interval_frac < 1.0 and blend_mode_enum != BlendMode.EVERY_STEP:
        raise ValueError(
            f"Fractional --blend_interval_frac={cfg.blend_interval_frac} requires "
            "--blend_mode=every_step: once_per_chunk drains the cached blended chunk even on "
            "ticks that carry guidance, so the requested mid-chunk re-blends would silently "
            f"never happen (got --blend_mode={cfg.blend_mode})."
        )
    if blend_mode_enum == BlendMode.EVERY_STEP and not cfg.fixed_base_noise:
        logger.warning(
            "blend_mode=every_step WITHOUT --fixed_base_noise: the wrapper re-runs a full "
            "denoising pass with a FRESH torch.randn every tick, and only action[0] of that "
            "chunk is executed — so consecutive executed actions are INDEPENDENT samples and "
            "the recorded trajectory will shake (worse at higher ratios, where more of x_tsw "
            "is noise). Pass --fixed_base_noise=true to pin one draw per episode, or use "
            "blend_mode=once_per_chunk (which matches how the policy is actually evaluated: "
            "one denoising pass per n_action_steps=%s ticks).",
            getattr(wrapper.config, "n_action_steps", "?"),
        )
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

    # ── Build target dataset (matches source's full schema) ────────────────
    image_keys = [f"{cam}_{mode}" for cam in cfg.env_camera_names for mode in cfg.env_image_resize_modes]
    # The schema this run will write: derived from the SOURCE dataset's dims
    # (state / action / env_state / image keys), so blends stay
    # schema-compatible with the intervention they were blended from. In
    # particular env_state_dim>0 declares observation.environment_state —
    # required downstream when training mixes blends with env-state datasets
    # (multi_source_feature_intersection would otherwise silently drop the
    # feature from EVERY source and an env-state-conditioned policy gets
    # neither images nor env_state).
    expected_features = build_lerobot_features(
        image_keys,
        cfg.num_dofs,
        state_dim=source_state_dim,
        env_state_dim=source_env_state_dim,
    )
    existing = load_lerobot_dataset(cfg.target_dataset_repo_id)
    if existing is not None:
        _existing_feats = existing.meta.features
        _mismatches = []
        for _key, _spec in expected_features.items():
            if _key not in _existing_feats:
                _mismatches.append(f"missing feature '{_key}' (expected shape {tuple(_spec['shape'])})")
            elif tuple(_existing_feats[_key]["shape"]) != tuple(_spec["shape"]):
                _mismatches.append(
                    f"feature '{_key}' has shape {tuple(_existing_feats[_key]['shape'])}, "
                    f"expected {tuple(_spec['shape'])}"
                )
        if _mismatches:
            raise RuntimeError(
                f"Target dataset {cfg.target_dataset_repo_id} already exists on disk but its "
                f"schema does not match what this run would write (source "
                f"{cfg.dataset_repo_id}: state_dim={source_state_dim}, "
                f"env_state_dim={source_env_state_dim}):\n  - "
                + "\n  - ".join(_mismatches)
                + f"\nIt is stale (e.g. created before the source gained these features). "
                f"Refusing to append. Delete it and re-run:\n"
                f"  rm -rf {existing.root}"
            )
        logger.warning(
            "Target dataset %s already exists locally — resuming into it (schema verified "
            "compatible). Episodes whose (source_episode_idx, blend_ratio) provenance is "
            "already present will be SKIPPED (idempotent resume); only missing pairs are "
            "blended. Delete the directory if you want a fresh dataset.",
            cfg.target_dataset_repo_id,
        )
        target_ds = existing
    else:
        target_ds = create_lerobot_dataset(
            cfg.target_dataset_repo_id,
            fps=cfg.env_fps,
            image_keys=image_keys,
            num_dofs=cfg.num_dofs,
            state_dim=source_state_dim,
            env_state_dim=source_env_state_dim,
        )

    # Idempotent-resume skip set (empty for a freshly created target).
    _done_pairs = _existing_provenance_pairs(target_ds.root)
    if _done_pairs:
        logger.info(
            "[resume] target already contains %d (source_ep, ratio) pair(s); they will be skipped.",
            len(_done_pairs),
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

    def _resolve_source_scenario_idx(source_ep: int) -> int:
        """Look up the scenario the source episode was originally recorded in.

        Reads ``source_scenario_idx`` from the source dataset's per-episode
        metadata (written by RRT/oracle intervention recording); falls back
        to ``source_ep`` itself when absent (identity assumption for plain
        training datasets — logged as a warning).
        """
        if not source_episodes_meta.empty and "source_scenario_idx" in source_episodes_meta.columns:
            row = source_episodes_meta.loc[
                source_episodes_meta["episode_index"] == source_ep, "source_scenario_idx"
            ]
            if not row.empty and pd.notna(row.iloc[0]):
                return int(row.iloc[0])
        logger.warning(
            "Source episode %d has no source_scenario_idx in metadata; "
            "falling back to episode_index=%d as the scenario id. Make "
            "sure --eval_benchmark_repo_id scenario %d matches this "
            "episode's scene geometry.",
            source_ep,
            source_ep,
            source_ep,
        )
        return source_ep

    # Pre-scan episodes to (a) filter out ones with 0 frames (skipped in the
    # rollout loop below) and (b) resolve source_scenario_idx up front. The
    # RESULTING list ordering IS the playlist: for each surviving source_ep,
    # we play its scenario once per ratio in cfg.forward_flow_ratios. This is
    # the same order the outer loop below iterates in, so the sim's counter
    # walks the playlist in perfect sync with our source_ep × ratio iteration.
    _valid_source_eps: list[int] = []
    _playlist: list[int] = []
    for source_ep in episode_indices:
        if _episode_length(parquet_files, source_ep) <= 0:
            logger.warning("Source episode %d has 0 frames; skipping (excluded from playlist too)", source_ep)
            continue
        _valid_source_eps.append(source_ep)
        _scen_idx = _resolve_source_scenario_idx(source_ep)
        for _ in cfg.forward_flow_ratios:
            _playlist.append(_scen_idx)
    logger.info(
        "Installing sim EVAL_BENCHMARK playlist: %d entries (%d source_eps × %d ratios). First 20: %s",
        len(_playlist),
        len(_valid_source_eps),
        len(cfg.forward_flow_ratios),
        _playlist[:20],
    )
    # ORDER + DUPLICATES preserved end-to-end (see set_env_benchmark_indices
    # + server-side set_eval_benchmark_indices docstrings). Each rollout's
    # env.reset() then just advances the sim's internal counter one slot;
    # no per-reset scenario arg needed downstream.
    set_env_benchmark_indices(vec_env, _playlist)

    # Walks _playlist in lockstep with the (source_ep × ratio) loop below.
    # Passed to each rollout as benchmark_start_index so every reset lands on
    # ITS slot even if the server counter drifted (e.g. GUI dropdown echo).
    _playlist_pos = 0

    try:
        for source_ep in tqdm(_valid_source_eps, desc="source episodes", leave=True):
            ep_length = _episode_length(parquet_files, source_ep)
            # Load enough frames to cover the full episode (n_obs_steps obs +
            # ep_length-n_obs_steps guidance). We only need actions+state+task
            # here; images come from the live env.
            n_obs_steps = wrapper.config.n_obs_steps
            # --start_frame slices the replay window mid-episode. The robot is
            # TELEPORTED to the demo's pose at that frame by the shared rollout
            # core (works for ZMQ SplatSim servers too — the server dispatches
            # teleport_joint_state), so guidance and robot stay aligned from
            # tick 0 even for nonzero starts.
            s0 = int(cfg.start_frame)
            if s0 < 0 or s0 + n_obs_steps + 1 > ep_length:
                logger.warning(
                    "Source ep %d: start_frame=%d leaves no guidance frames "
                    "(ep_length=%d, n_obs_steps=%d); skipping episode.",
                    source_ep,
                    s0,
                    ep_length,
                    n_obs_steps,
                )
                continue
            frames_df = load_episode_frames(
                source_data_dir, source_ep, frame_index=s0, n_frames=ep_length - s0
            )
            seed_joint_state = np.asarray(frames_df.iloc[n_obs_steps - 1]["action"], dtype=np.float32)
            guidance_actions_raw = np.stack(
                [
                    np.asarray(row["action"], dtype=np.float32)
                    for _, row in frames_df.iloc[n_obs_steps:].iterrows()
                ]
            )
            total_steps = guidance_actions_raw.shape[0]
            # Demo STATES on the same index grid as guidance_actions_raw:
            # demo_states_raw[k] = the state at the time raw[k] should execute.
            # Used by progress-aware guidance to match the robot's CURRENT
            # state — matching against ACTIONS would be off by one (action[k]'s
            # target ≈ state[k+1]): the matched action's target would be the
            # robot's current pose, i.e. guidance would command "stay put" and
            # the robot would permanently crawl behind the demo.
            demo_states_raw = None
            if cfg.progress_guidance and "observation.state" in frames_df.columns:
                demo_states_raw = np.stack(
                    [
                        np.asarray(row["observation.state"], dtype=np.float32)
                        for _, row in frames_df.iloc[n_obs_steps:].iterrows()
                    ]
                )

            task_idx = int(frames_df.iloc[0].get("task_index", 1))
            task_description = (task_map.get(task_idx) if task_map else None) or cfg.env_task

            source_scenario_idx = _resolve_source_scenario_idx(source_ep)
            logger.info(
                "Source ep %d → scenario %d, total_steps=%d",
                source_ep,
                source_scenario_idx,
                total_steps,
            )

            # One pinned noise draw per (source_ep, ratio) rollout when
            # --fixed_base_noise is set: constant WITHIN the episode (kills the
            # per-tick resampling jitter that makes every_step shake) but
            # independent ACROSS episodes/ratios (so the dataset keeps sample
            # diversity). Shape mirrors visualize_shared_autonomy_sim.py.
            _base_noise = None
            if cfg.fixed_base_noise:
                if getattr(wrapper.config, "max_action_dim", None) is not None:
                    _noise_shape = (1, wrapper.config.chunk_size, wrapper.config.max_action_dim)
                else:
                    _adim = wrapper.config.output_features["action"].shape[0]
                    _noise_shape = (1, wrapper.config.horizon, _adim)
                _base_noise = torch.randn(_noise_shape, device=cfg.device)

            for ratio in cfg.forward_flow_ratios:
                if (int(source_ep), round(float(ratio), 6)) in _done_pairs:
                    logger.info(
                        "[resume] source_ep=%d ratio=%.2f already in target — skipping.",
                        source_ep,
                        ratio,
                    )
                    # The playlist has one slot per (source_ep, ratio); keep
                    # position in lockstep even when skipping so later
                    # rollouts still reset to THEIR scenario slot.
                    _playlist_pos += 1
                    continue
                t0 = time.time()
                rollout = rollout_closed_loop_for_augmentation(
                    wrapper=wrapper,
                    obs_preprocessor=obs_preprocessor,
                    vec_env=vec_env,
                    env_preprocessor=env_pre,
                    env_postprocessor=env_post,
                    seed_joint_state=seed_joint_state,
                    guidance_actions_raw=guidance_actions_raw,
                    ratio=float(ratio),
                    blend_mode=blend_mode_enum,
                    blend_interval_frac=cfg.blend_interval_frac,
                    total_steps=total_steps,
                    progress_guidance=cfg.progress_guidance,
                    progress_guidance_window=cfg.progress_guidance_window,
                    demo_states_raw=demo_states_raw,
                    rename_map=rename_map,
                    image_keys=image_keys,
                    task_description=task_description,
                    device=cfg.device,
                    playlist_pos=_playlist_pos,
                    env_state_dim=source_env_state_dim,
                    base_noise=_base_noise,
                )
                _playlist_pos += 1

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
| `blend_interval_frac` | `{cfg.blend_interval_frac}` |
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

    # Draccus treats bool fields as requiring a value (--flag=true). Keep the
    # bare-flag CLI: --push_to_hub → --push_to_hub=true. --blend_interval_frac
    # is a float blend cadence (--drain_chunk is its deprecated alias): the
    # bare flag maps to 1.0 (legacy true = blend once per chunk) and
    # true/false spellings map to 1.0/0.0 for back-compat with existing
    # callers (e.g. --blend_extra_args='--drain_chunk=false').
    def _map_arg(arg: str) -> str:
        if arg == "--push_to_hub":
            return "--push_to_hub=true"
        for flag in ("--blend_interval_frac", "--drain_chunk"):
            if arg == flag:
                return "--blend_interval_frac=1.0"
            if arg.startswith(flag + "="):
                val = arg.split("=", 1)[1].strip().lower()
                if val in ("true", "yes"):
                    val = "1.0"
                elif val in ("false", "no"):
                    val = "0.0"
                return f"--blend_interval_frac={val}"
        return arg

    sys.argv = [_map_arg(arg) for arg in sys.argv]
    augment_main()


if __name__ == "__main__":
    main()
