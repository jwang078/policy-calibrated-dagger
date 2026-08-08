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
            --robot_name robot_iphone_w_engine_curtain \\
            --eval_benchmark_repo_id <benchmark_dataset_repo_id>

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
# 1. Launch splatsim out-of-process (once, stays up)
cd ~/code/SplatSim && python scripts/launch_nodes.py \
    --robot sim_ur_pybullet_small_engine_new_interactive \
    --robot_port 6001 \
    --robot_name robot_iphone_w_engine_curtain \
    --eval_benchmark_repo_id JennyWWW/eval_splatsim_approach_lever_benchmark_1000

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
    """The launch_nodes.py invocation that starts the splatsim server this
    script expects to find on ``port`` (matching env task / robot / benchmark)."""
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
    lines.append("    --headless --control_gui --sync_physics_to_client")
    return "\n".join(lines)


def check_sim_server_reachable(host: str, port: int, launch_hint: str) -> None:
    """Fail fast if nothing is listening on host:port. A ZMQ REQ socket never
    errors on a dead endpoint — the first reset request just queues forever,
    so without this check a missing server looks like a silent freeze."""
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
    demo_states_raw: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Thin adapter over :func:`lib_sa_rollout.run_blended_rollout` — the SAME
    core the sweep's blend step (``augment_dataset_with_blending``) executes.
    This wrapper only (a) derives the blend mode from ``--blend_interval_frac``
    (1.0 → ONCE_PER_CHUNK; anything below → EVERY_STEP, whose re-blend cadence
    the rollout core throttles to every
    ``ceil(blend_interval_frac * n_action_steps)`` ticks) and (b) pins the
    benchmark scenario on EVERY per-ratio reset via
    ``benchmark_start_index`` (a bare seeded reset would let the server's
    EVAL_BENCHMARK counter advance one scenario per reset).
    """
    result = run_blended_rollout(
        wrapper=wrapper,
        obs_preprocessor=obs_preprocessor,
        vec_env=vec_env,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        seed_joint_state=seed_joint_state,
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
        demo_states_raw=demo_states_raw,
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
    demo_states_raw: np.ndarray | None = None,
) -> tuple[dict[float, np.ndarray], dict[float, np.ndarray]]:
    """Run :func:`get_sim_action_chunk_for_ratio` for each ratio.

    ``fixed_base_noise=True`` (default) pins ONE noise draw shared across all
    ratios AND all ticks — the tool's historical cross-ratio comparability
    mode. ``False`` matches the sweep's blend default: the wrapper draws a
    FRESH torch.randn internally on every denoise (base_noise=None), so
    consecutive every_step samples are independent — expect shake.
    """
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
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
            blend_interval_frac=blend_interval_frac,
            base_noise=base_noise,
            total_steps=total_steps,
            rename_map=rename_map,
            device=device,
            task_description=task_description,
            progress_guidance=progress_guidance,
            progress_guidance_window=progress_guidance_window,
            demo_states_raw=demo_states_raw,
        )
        results[ratio] = actions
        if decoded is not None:
            decoded_guidance_by_ratio[ratio] = decoded

    return results, decoded_guidance_by_ratio


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
    parser.add_argument("--n_anchor_steps", type=int, default=0)
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
        "--clip_sample",
        type=_parse_bool,
        default=None,
        help=(
            "DEBUG: override the checkpoint's DDPM clip_sample (None = keep trained value). "
            "The sweep's blend step often runs --clip_sample=false."
        ),
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

    return parser.parse_args()


def main():
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
    wrapper.policy_guidance_representation = PolicyGuidanceRepresentation(args.guidance_repr)
    wrapper.n_anchor_steps = args.n_anchor_steps
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

    try:
        print(f"Computing sim rollouts for ratios: {args.forward_flow_ratios} …")
        action_chunks, decoded_guidance_by_ratio = get_sim_action_chunks_for_ratios(
            wrapper,
            obs_preprocessor,
            vec_env,
            env_pre,
            env_post,
            seed_joint_state=seed_joint_state,
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
            demo_states_raw=demo_states_raw,
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
        anchor_tag = f"anchor{args.n_anchor_steps}" if args.n_anchor_steps > 0 else "noanchor"
        nas_tag = f"nas{n_action_steps}"
        noise_tag = "" if args.fixed_base_noise else "_freshnoise"
        pg_tag = "_pg" if args.progress_guidance else ""
        clip_tag = "" if args.clip_sample is None else ("_clip" if args.clip_sample else "_noclip")
        parent = f"shared_autonomy_sim_ep{episode_index}_frame{frame_index}"
        name = (
            f"{policy_tag}_{args.blend_strategy}_{repr_tag}_{blend_interval_tag}_{anchor_tag}_{nas_tag}"
            f"{noise_tag}{pg_tag}{clip_tag}_sim"
        )
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
