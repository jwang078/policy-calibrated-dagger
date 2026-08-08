"""THE closed-loop shared-autonomy blend rollout — one implementation.

Both the DAgger sweep's dataset blending (``augment_dataset_with_blending.py``,
invoked by ``dagger_orchestrate*.sh``) and the debug visualizer
(``visualize_shared_autonomy_sim.py``) execute :func:`run_blended_rollout`
verbatim. There are no per-caller forks of the control flow: what the
visualizer validates IS what the sweep runs. Caller-specific concerns are
injected, not branched:

* frame capture for dataset writing → ``on_step`` / ``on_success`` callbacks
  (the visualizer passes none and just consumes the returned action matrix);
* scenario selection → ``seed`` / ``benchmark_start_index`` passthrough to
  :func:`seed_splatsim_env_to_state` (playlist position for the blend script,
  resolved scenario id for the visualizer);
* logging → the ``log`` callable.

Everything else — env seeding + teleport, filler phase, per-tick guidance
selection (wall-clock or progress-matched), batch building, ``select_action``
with optional pinned base noise, postprocessing, ghost updates, success/hold
handling, decoded-guidance capture — is shared line-for-line.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from lerobot.envs.utils import preprocess_observation
from lerobot.utils.constants import ACTION
from lerobot.utils.sim_seeding import seed_splatsim_env_to_state

# ── sim physics-mode check ────────────────────────────────────────────────────


def warn_if_sim_physics_unsynced(vec_env, log=print) -> bool | None:
    """Report whether the connected sim runs --sync_physics_to_client.

    Queries the server's get_env_config (works over ZMQ and in-process) for
    the ``sync_physics_to_client`` flag the server publishes. Returns
    True/False when the server reports it, None when undeterminable (older
    SplatSim without the field, or no get_env_config on this backend).

    Prints a one-line mode notice either way: OFF means physics integrates
    in wallclock time while the policy is thinking, so slow policies (e.g.
    diffusion at chunk boundaries) roll out against a sim that raced ahead —
    "jumpy" trajectories that misrepresent the policy.
    """
    synced: bool | None = None
    try:
        single = vec_env.envs[0] if hasattr(vec_env, "envs") else vec_env
        fn = getattr(single, "get_env_config", None)
        cfg = fn() if callable(fn) else None
        if isinstance(cfg, dict) and "sync_physics_to_client" in cfg:
            synced = bool(cfg["sync_physics_to_client"])
    except Exception:
        synced = None
    if synced is True:
        log("[sim] sync_physics_to_client=ON — sim physics is gated on this client's commands.")
    elif synced is False:
        log(
            "[sim] WARNING: sync_physics_to_client=OFF — the sim integrates physics in "
            "WALLCLOCK time while the policy thinks. Slow policies will look jumpy and "
            "off-policy. Relaunch launch_nodes.py with --sync_physics_to_client."
        )
    else:
        log(
            "[sim] NOTE: could not determine the sim's sync_physics_to_client mode "
            "(older SplatSim server?). If rollouts look jumpy, relaunch the sim with "
            "--sync_physics_to_client."
        )
    return synced


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
    """
    n_obs_steps: int = wrapper.config.n_obs_steps
    n_action_steps: int = wrapper.config.n_action_steps
    n_filler_drain = n_action_steps - (n_obs_steps - 1)
    for _ in range(n_filler_drain + (n_obs_steps - 1)):
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


# ── progress-matched guidance ─────────────────────────────────────────────────


def progress_guidance_index(demo_arm: np.ndarray, q_now: np.ndarray, j_prev: int, window: int) -> int:
    """Monotonic nearest-point-on-demo index for progress-aware guidance.

    Returns j* in [j_prev, min(len, j_prev+window)) minimizing
    ||demo_arm[j] - q_now||. Monotonic (never below j_prev) so loops in the
    demo can't yank guidance backwards; windowed so one tick can't leap far
    ahead. demo_arm: (T, num_dofs) demo actions (≈ states, abs joint targets).
    """
    lo = int(j_prev)
    hi = min(demo_arm.shape[0], lo + max(1, int(window)))
    if lo >= demo_arm.shape[0]:
        return demo_arm.shape[0] - 1
    seg = demo_arm[lo:hi]
    d = np.linalg.norm(seg - q_now.reshape(1, -1), axis=1)
    return lo + int(np.argmin(d))


# ── the rollout ───────────────────────────────────────────────────────────────


@dataclass
class BlendRolloutResult:
    raw_actions: np.ndarray  # (total_steps, action_dim) executed action targets
    decoded_guidance_full: np.ndarray | None  # chunk-boundary decoded-guidance overlay
    success: bool
    success_t: int | None  # tick at which the episode terminated (None if never)
    final_progress_cursor: int | None  # progress-guidance demo cursor (None when off)


@torch.no_grad()
def run_blended_rollout(
    *,
    wrapper,
    obs_preprocessor,
    vec_env,
    env_preprocessor,
    env_postprocessor,
    seed_joint_state: np.ndarray,
    guidance_actions_raw: np.ndarray,
    ratio: float,
    blend_mode,
    blend_interval_frac: float,
    total_steps: int,
    rename_map: dict[str, str],
    device: str,
    task_description: str | None,
    seed: list[int] | None = None,
    benchmark_start_index: int | None = None,
    base_noise: torch.Tensor | None = None,
    progress_guidance: bool = False,
    progress_guidance_window: int = 45,
    demo_states_raw: np.ndarray | None = None,
    on_step: Callable[[int, dict[str, Any], np.ndarray, bool], None] | None = None,
    on_success: Callable[[dict[str, Any]], None] | None = None,
    log: Callable[[str], None] = print,
) -> BlendRolloutResult:
    """Seed the env, run the filler phase, then loop env.step ↔ select_action.

    ``on_step(t, env_obs_batched, action_1d, is_hold)`` fires once per tick
    BEFORE the env steps (i.e. on the (s_t, a_t) pair), and once per hold tick
    after success (with the frozen terminal obs and the hold action).
    ``on_success(terminal_env_obs_batched)`` fires exactly once at the success
    transition, after the terminating step.
    """
    n_action_steps: int = wrapper.config.n_action_steps
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")

    # Blend cadence: guidance is provided (→ the wrapper re-blends) every
    # ``ceil(blend_interval_frac * n_action_steps)`` ticks within each executed
    # chunk; offset 0 of every chunk always blends. 0 → every tick (legacy
    # drain_chunk=False), 1 → chunk boundaries only (legacy drain_chunk=True),
    # e.g. 0.5 → twice per chunk. bool inputs coerce to the matching endpoint
    # via float().
    frac = float(blend_interval_frac)
    if not 0.0 <= frac <= 1.0:
        raise ValueError(f"blend_interval_frac must be in [0, 1], got {blend_interval_frac}")
    blend_interval = max(1, math.ceil(frac * n_action_steps))

    wrapper.reset()
    wrapper.forward_flow_ratio = ratio
    wrapper.blend_mode = blend_mode

    # Reset (+ optional scenario pin) and teleport the robot to
    # seed_joint_state. The teleport works for BOTH in-process pybullet AND
    # ZMQ SplatSim servers (PybulletRobotServerBase dispatches
    # teleport_joint_state; _ZMQBackend forwards it).
    env_obs = seed_splatsim_env_to_state(
        vec_env,
        joint_state=seed_joint_state,
        num_dofs=wrapper.num_dofs,
        seed=seed,
        benchmark_start_index=benchmark_start_index,
    )

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

    raw_actions: list[np.ndarray] = []
    decoded_guidance_full: np.ndarray | None = None
    success = False
    success_t: int | None = None
    hold_action: np.ndarray | None = None
    terminal_env_obs: dict[str, Any] | None = None

    # Progress-guidance state: match the robot's CURRENT state against demo
    # STATES on the raw-index grid (demo_states_raw[k] = state when raw[k]
    # executes), so j* is the robot's true demo position and raw[j*] commands
    # the NEXT state. `_match_shift` compensates the action-matrix fallback
    # (no states in source): action[k]'s target ≈ state[k+1], so a match at k
    # means the robot IS at raw-index k+1.
    _num_arm = max(1, guidance_actions_raw.shape[1] - 1)  # drop gripper dim
    if demo_states_raw is not None:
        _demo_arm = np.asarray(demo_states_raw[:, :_num_arm], dtype=np.float32)
        _match_shift = 0
    else:
        _demo_arm = np.asarray(guidance_actions_raw[:, :_num_arm], dtype=np.float32)
        _match_shift = 1
    _j_progress = 0

    for t in range(total_steps):
        # ── Hold mode: episode succeeded, don't step env again ────────────────
        # Stepping after termination triggers AutoresetMode.NEXT_STEP and would
        # bring in the next scene's images, causing a sharp visual transition.
        if success:
            assert hold_action is not None and terminal_env_obs is not None
            if on_step is not None:
                on_step(t, terminal_env_obs, hold_action, True)
            raw_actions.append(hold_action)
            continue

        chunk_offset = t % n_action_steps
        at_chunk_boundary = chunk_offset == 0
        suppress_guidance = chunk_offset % blend_interval != 0 and ratio not in (0.0, 1.0)
        if progress_guidance:
            # Re-index the demo by PROGRESS instead of wall-clock: match the
            # robot's current joints to the closest demo step (monotonic,
            # windowed — see progress_guidance_index). A stuck robot holds
            # guidance at its current demo point; a detoured robot re-enters
            # the demo where it actually is, so guidance and robot re-converge.
            _q_now = np.asarray(env_obs["agent_pos"], dtype=np.float32).reshape(-1)[: _demo_arm.shape[1]]
            _j_progress = progress_guidance_index(_demo_arm, _q_now, _j_progress, progress_guidance_window)
            _j_exec = min(_j_progress + _match_shift, guidance_actions_raw.shape[0] - 1)
            guidance_chunk = None if suppress_guidance else guidance_actions_raw[_j_exec:]
        else:
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
            _post_out = env_postprocessor({ACTION: raw_action})
            if _post_out is not None:
                raw_action = _post_out[ACTION]

        action_numpy = raw_action.detach().to("cpu").numpy()  # (1, action_dim)
        action_1d = action_numpy.reshape(-1)

        # BLUE ghost: the absolute target actually commanded this tick (the
        # rel action decoded on the current anchor). Compare against the GREEN
        # guidance ghost + the real robot in the wrapper's pybullet window.
        if wrapper.show_guidance_ghost:
            wrapper.update_action_ghost(action_1d)

        # Callback on the (s_t, a_t) pair — obs before the step, action about
        # to be sent (the blend script builds its dataset frame here).
        if on_step is not None:
            on_step(t, env_obs, action_1d, False)
        raw_actions.append(action_1d)

        env_obs, _reward, _term, _trunc, _info = vec_env.step(action_numpy)

        # Decoded-guidance overlay: what the guidance source actually fed the
        # blend, decoded back to raw joints (plot diagnostic).
        if at_chunk_boundary and wrapper._last_decoded_guidance_chunk is not None:
            chunk_decode = wrapper._last_decoded_guidance_chunk[0]
            if decoded_guidance_full is None:
                # NaN-init, not zeros: ticks never written (post-termination
                # hold, tail after the last chunk boundary) plot as a GAP
                # instead of a fake 0.0-rad guidance trace that reads like a
                # decode bug.
                decoded_guidance_full = np.full(
                    (total_steps, chunk_decode.shape[1]), np.nan, dtype=chunk_decode.dtype
                )
            end_t = min(t + n_action_steps, total_steps)
            decoded_guidance_full[t:end_t] = chunk_decode[: end_t - t]

        # Check for success / termination.
        terminated = bool(_term[0]) if hasattr(_term, "__len__") else bool(_term)
        if terminated and not success:
            success = True
            success_t = t
            # Snapshot terminal state BEFORE the next step() would reset.
            terminal_env_obs = env_obs
            agent_pos = env_obs.get("agent_pos")
            hold_action = np.asarray(agent_pos[0], dtype=np.float32) if agent_pos is not None else action_1d
            if on_success is not None:
                on_success(terminal_env_obs)
            log(
                f"[ratio={ratio}] Episode succeeded at t={t + 1}/{total_steps}. "
                f"Holding for {total_steps - t - 1} remaining steps."
            )

    if progress_guidance:
        log(
            f"[ratio={ratio}] progress-guidance final demo cursor {_j_progress}/{_demo_arm.shape[0]} "
            f"(wall-clock ticks {total_steps}; lag {total_steps - _j_progress})"
        )

    return BlendRolloutResult(
        raw_actions=np.stack(raw_actions),
        decoded_guidance_full=decoded_guidance_full,
        success=success,
        success_t=success_t,
        final_progress_cursor=_j_progress if progress_guidance else None,
    )
