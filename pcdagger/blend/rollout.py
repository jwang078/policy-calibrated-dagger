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
from lerobot.policies.shared_autonomy_wrapper import BlendMode
from lerobot.utils.constants import ACTION
from lerobot.utils.sim_seeding import seed_splatsim_env_to_state

# ── sim physics-mode check ────────────────────────────────────────────────────


def _fetch_sim_env_config(vec_env) -> dict | None:
    """Best-effort fetch of the sim server's env-config dict.

    Used for capability probes (sync_physics_to_client /
    strict_goal_tolerances).

    ``ZMQSplatSimGymEnv.get_env_config()`` returns None when the env was
    built with ``include_oracle_info=False`` — that gate is an ORACLE-INFO
    opt-in (obstacle geometry, task goal) and most blend/viz envs run with
    it off, which used to make these probes report "undeterminable" for
    every ZMQ run. Capability flags shouldn't be gated on oracle info, so
    when the public method yields nothing we fall back to the env's raw ZMQ
    client, which always supports the get_env_config RPC.
    """
    try:
        single = vec_env.envs[0] if hasattr(vec_env, "envs") else vec_env
        base = getattr(single, "unwrapped", single)
        fn = getattr(base, "get_env_config", None)
        cfg = fn() if callable(fn) else None
        if isinstance(cfg, dict):
            return cfg
        zmq_client = getattr(base, "_zmq_client", None)
        zfn = getattr(zmq_client, "get_env_config", None)
        cfg = zfn() if callable(zfn) else None
        return cfg if isinstance(cfg, dict) else None
    except Exception:
        return None


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
    cfg = _fetch_sim_env_config(vec_env)
    if isinstance(cfg, dict) and "sync_physics_to_client" in cfg:
        synced = bool(cfg["sync_physics_to_client"])
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


def check_sim_strict_goal_tolerances(vec_env, *, required: bool = False, log=print) -> bool | None:
    """Report whether the connected sim runs --strict_goal_tolerances.

    Queries the server's get_env_config (same probe as
    :func:`warn_if_sim_physics_unsynced`) for the ``strict_goal_tolerances``
    flag. Returns True/False when the server reports it, None when
    undeterminable (older SplatSim without the field, or no get_env_config
    on this backend).

    Why it matters: the default eval-time is_success thresholds are LOOSE
    (planar 60 mm, small_engine 30 mm / 10°), so a rollout terminates — and
    the recorder/visualizer freezes into post-success hold — as soon as the
    arm gets "close enough". Intervention recording and blend rollouts exist
    to capture the last-mile corrections and the state coverage near the
    goal, which is exactly what a loose sim cuts off. The DAgger
    orchestrator therefore always launches its managed sims with
    --strict_goal_tolerances; standalone/blend/viz runs against a
    user-managed sim should match it.

    ``required=True`` escalates a definitive False to SystemExit (the flag
    is applied at server __init__ — there is no runtime toggle, the server
    must be relaunched with --strict_goal_tolerances).
    """
    strict: bool | None = None
    cfg = _fetch_sim_env_config(vec_env)
    if isinstance(cfg, dict) and "strict_goal_tolerances" in cfg:
        strict = bool(cfg["strict_goal_tolerances"])
    if strict is True:
        log("[sim] strict_goal_tolerances=ON — success requires the strict (recording-grade) goal pose.")
    elif strict is False:
        msg = (
            "[sim] strict_goal_tolerances=OFF — the sim terminates episodes at the LOOSE "
            "eval-time success thresholds (e.g. planar 60 mm), cutting rollouts off far "
            "from the goal pose. Relaunch launch_nodes.py with --strict_goal_tolerances "
            "for recording-grade rollouts (no runtime toggle; a restart is required)."
        )
        if required:
            raise SystemExit(
                msg + "\nRefusing to record against a loose sim "
                "(pass --allow_loose_goal_tolerances to override)."
            )
        log("WARNING: " + msg)
    else:
        log(
            "[sim] NOTE: could not determine the sim's strict_goal_tolerances mode "
            "(older SplatSim server without the get_env_config field?). If rollouts "
            "terminate early near the goal, relaunch with --strict_goal_tolerances."
        )
    return strict


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
    seed_joint_velocity: np.ndarray | None = None,
    fps: float = 30.0,
) -> None:
    """Drain the inner policy's first throwaway chunk. Does NOT step the env.

    Ensures the obs queue has the right history before the real phase
    begins.

    ``seed_joint_velocity`` (rad/s): when the rollout seeds an episode that
    began with a MOVING handoff, the policy's obs history must encode that
    motion — a filler phase of identical frames tells the policy "you are at
    rest", contradicting the source episode's conditioning and producing a
    first-command lurch (measured 0.67 rad/s first-step vs the source's
    carried 0.24-0.30, planar_12 blends, 2026-08-18). Each filler tick's
    ``agent_pos`` is back-extrapolated along the seed velocity so the final
    ``n_obs_steps`` queue entries arrive at ``seed_joint_state`` moving at
    ``seed_joint_velocity``. Images stay the seeded frame (state carries the
    velocity signal for these policies).

    Also snaps ``wrapper._desired_q`` to ``seed_joint_state`` after filler so the
    wrapper's IK anchor isn't polluted by the throwaway chunk's actions.
    """
    n_obs_steps: int = wrapper.config.n_obs_steps
    n_action_steps: int = wrapper.config.n_action_steps
    n_filler_drain = n_action_steps - (n_obs_steps - 1)
    n_total = n_filler_drain + (n_obs_steps - 1)
    for i in range(n_total):
        tick_obs = env_obs
        if seed_joint_velocity is not None:
            v = np.asarray(seed_joint_velocity, dtype=np.float32).reshape(-1)
            q0 = np.asarray(seed_joint_state, dtype=np.float32).reshape(-1)
            ticks_back = float(n_total - 1 - i)
            q_i = q0.copy()
            n_arm = min(len(v), len(q_i))
            q_i[:n_arm] = q0[:n_arm] - v[:n_arm] * (ticks_back / float(fps))
            tick_obs = dict(env_obs)
            ap = np.asarray(env_obs["agent_pos"], dtype=np.float32).copy()
            ap[0, : len(q_i)] = q_i[: ap.shape[1]]
            tick_obs["agent_pos"] = ap
        batch = _build_sim_batch(
            tick_obs,
            env_preprocessor=env_preprocessor,
            obs_preprocessor=obs_preprocessor,
            rename_map=rename_map,
            device=device,
            task_description=task_description,
            guidance_chunk=guidance_chunk,
        )
        wrapper.select_action(batch)
    wrapper._desired_q = np.asarray(seed_joint_state, dtype=np.float32)[: wrapper.num_dofs].copy()
    # Drop the filler's throwaway blend chunks: they were built on synthetic
    # obs and would otherwise serve as the "previous chunk" for the first
    # REAL build's RTC prev-chunk conditioning — committing the launch to
    # filler garbage instead of letting the source's first-build guidance
    # seed fire (the guidance prefix as the plan the rollout is joining).
    _src = getattr(wrapper, "_obs_teleop_source", None)
    if _src is not None and hasattr(_src, "cancel"):
        _src.cancel()


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
    """Arrays captured from one blended rollout (actions, overlays, success info)."""

    raw_actions: np.ndarray  # (total_steps, action_dim) executed action targets
    decoded_guidance_full: np.ndarray | None  # chunk-boundary decoded-guidance overlay
    success: bool
    success_t: int | None  # tick at which the episode terminated (None if never)
    final_progress_cursor: int | None  # progress-guidance demo cursor (None when off)
    mean_effective_ratio: float | None = None  # tick-mean of the (taper/tube-scaled) ratio
    dev_steps_p50: float | None = None  # median per-tick corridor deviation, med_step units
    dev_steps_p95: float | None = None
    tube_breach: bool = False  # aborted early: realized dev crossed abort_dev_steps


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
    progress_guidance_lag_tol: int = 2,
    progress_guidance_soft_hold: float = 0.0,
    progress_guidance_hard_lag: int = 8,
    blend_ratio_goal_taper: int = 0,
    blend_dev_regulation: bool = False,
    blend_dev_full_below: float = 3.0,
    blend_dev_zero_above: float = 8.0,
    seed_joint_velocity: np.ndarray | None = None,
    fps: float = 30.0,
    demo_states_raw: np.ndarray | None = None,
    pad_after_success: bool = True,
    expected_env_state: np.ndarray | None = None,
    env_state_match_tol: float = 0.02,
    abort_dev_steps: float = 0.0,
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

    ``pad_after_success=False`` returns immediately at the success transition
    instead of emitting frozen hold ticks until ``total_steps`` — the result's
    ``raw_actions`` is then ``success_t + 1`` rows, not ``total_steps``.
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
        joint_velocities=seed_joint_velocity,
        num_dofs=wrapper.num_dofs,
        seed=seed,
        benchmark_start_index=benchmark_start_index,
    )

    # World-match guard: the seeded env's environment_state must reproduce the
    # source episode's (obstacles + object). A mismatch means the scenario
    # mapping is broken — e.g. the sim server was launched with a benchmark
    # subset that does not contain (or reorders) the requested scenario, so
    # ``benchmark_start_index`` lands on the WRONG WORLD: replays collide with
    # obstacles the demo never saw and recorded images are poisoned (found
    # 2026-08-19: a server pinned to subset=[0] silently ran every episode in
    # scenario 0's world). Fails loudly instead. ``env_state_match_tol`` <= 0
    # or no expected state disables (e.g. envs without environment_state).
    if expected_env_state is not None and env_state_match_tol > 0 and "environment_state" in env_obs:
        _got = np.asarray(env_obs["environment_state"][0], dtype=np.float64).reshape(-1)
        _want = np.asarray(expected_env_state, dtype=np.float64).reshape(-1)
        if _got.shape == _want.shape:
            _err = float(np.max(np.abs(_got - _want)))
            if _err > env_state_match_tol:
                raise RuntimeError(
                    f"WORLD MISMATCH after scenario seeding: environment_state differs from the "
                    f"source episode's by max|delta|={_err:.3f} (> {env_state_match_tol}). The sim "
                    f"server is not honoring benchmark_start_index={benchmark_start_index} — check "
                    f"that it runs in EVAL_BENCHMARK mode with a benchmark subset containing this "
                    f"scenario (identity subset recommended). Rollouts in a wrong world produce "
                    f"colliding replays and poisoned training images.\n"
                    f"  server  : {np.round(_got, 3)}\n  expected: {np.round(_want, 3)}"
                )

    # Filler chunks are DISCARDED (cancel() below) — searching the tube for
    # them wastes ~n_action_steps x 4 denoiser calls per episode before the
    # env ever steps (the pre-motion re-blend log burst, 2026-08-20).
    _tube_saved = float(getattr(wrapper, "blend_tube_steps", 0.0) or 0.0)
    wrapper.blend_tube_steps = 0.0
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
        seed_joint_velocity=seed_joint_velocity,
        fps=fps,
    )
    wrapper.blend_tube_steps = _tube_saved

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
    _j_clock = 0.0  # demo-pace playback cursor (float; see the progress block below)
    _pg_seg = np.linalg.norm(np.diff(_demo_arm.astype(np.float64), axis=0), axis=1)
    _pg_med_step = float(np.median(_pg_seg[_pg_seg > 1e-9])) if (_pg_seg > 1e-9).any() else 1e-3
    _r_eff_sum, _r_eff_n = 0.0, 0
    _dev_hist: list[float] = []
    _tube_engaged_since: int | None = None
    _tube_ticks_total = 0
    # EARLY BREACH ABORT (abort_dev_steps > 0): the planned-chunk tube
    # re-blend saturates under DENOISE mode attraction on hard basins, so
    # the executed state can still leave the tube (ep 9, 2026-08-21) — the
    # accept gate would reject the episode post hoc anyway; abort the
    # moment the realized deviation crosses the gate instead of burning the
    # rest of the episode. 3 consecutive ticks so a one-tick projection
    # glitch cannot kill a healthy rollout.
    _breach_run = 0
    _tube_breach = False

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
            # DEMO-PACE CLOCK, gated by robot progress. Two prior designs
            # both corrupted the demo's speed profile:
            #   * commanding the matched step self-throttled (cursor advances
            #     only after the obs confirms arrival -> 1/2-1/3 demo pace,
            #     move/freeze command cadence);
            #   * a pursuit lead (command j*+2) fixed the throttle but let
            #     the robot COMPRESS the demo's deliberately-slow phases
            #     (launch ramp, taper, curvature dips): pace 1.22, ratio-0
            #     "faithful" replays finishing 124/155 frames (2026-08-18).
            # The clock advances at most 1 demo index per wall tick (pace
            # can never exceed 1.0 = the demo's own timing), holds when the
            # robot lags more than `progress_guidance_lag_tol` behind it
            # (stall-hold: a stuck robot pins the guidance), and snaps
            # forward when the robot re-enters the demo ahead of it (a
            # deviated rollout rejoins where it actually is).
            # SOFT HOLD (progress_guidance_soft_hold > 0): a hard stall-hold
            # cascades at small blend ratios — the policy component keeps the
            # robot perpetually ~lag_tol behind, every hold tick slows the
            # commanded guidance, and the rollout crawls (measured: r=0.1
            # blends of dag1 at pace ~0.7, episodes 1.45x source length,
            # 2026-08-19). Instead of stopping, the clock advances at
            # `soft_hold` x demo pace while moderately lagging, and only
            # hard-holds beyond `hard_lag` indices (a truly stuck robot
            # still pins the guidance).
            _j_clock = max(_j_clock, float(_j_progress))
            _j_exec = min(int(_j_clock) + _match_shift, guidance_actions_raw.shape[0] - 1)
            _lag = _j_clock - _j_progress
            if _lag <= max(0, int(progress_guidance_lag_tol)):
                _rate = 1.0
            elif _lag < max(int(progress_guidance_lag_tol) + 1, int(progress_guidance_hard_lag)):
                _rate = float(np.clip(progress_guidance_soft_hold, 0.0, 1.0))
            else:
                _rate = 0.0
            _j_clock = min(_j_clock + _rate, float(guidance_actions_raw.shape[0] - 1))
            # Ratio scaling: the requested ratio is a MAXIMUM; two per-tick
            # scale-free regulators anneal it down (combined by min):
            #   * GOAL TAPER (blend_ratio_goal_taper > 0): -> 0 over the last
            #     N demo indices — DART's validity condition (the noisy
            #     supervisor must still complete the task) applied at the
            #     goal, where a constant ratio leaves a r*(policy-expert)
            #     equilibrium offset that blocks strict success.
            #   * TUBE REGULATION (blend_dev_regulation): full ratio while
            #     the state is within blend_dev_full_below med_steps of the
            #     demo corridor, annealing linearly to 0 at
            #     blend_dev_zero_above — deviation-feedback noise: the
            #     CONTROLLED quantity is the coverage-tube radius, not the
            #     mixing weight, so basin escapes are pulled back instead of
            #     retried/backed off, and 'ratio r' honestly means 'at most
            #     r, inside the declared tube'.
            if ratio not in (0.0, 1.0):
                _scale = 1.0
                if blend_ratio_goal_taper > 0:
                    _remaining = float(guidance_actions_raw.shape[0] - 1) - _j_clock
                    _scale = min(_scale, max(0.0, min(1.0, _remaining / float(blend_ratio_goal_taper))))
                # Deviation telemetry (always on under progress guidance —
                # feeds the realized-tube episode metadata even when the
                # state-feedback regulation below is disabled).
                _dev = (
                    float(np.linalg.norm(_q_now.astype(np.float64) - _demo_arm[_j_progress])) / _pg_med_step
                )
                _dev_hist.append(_dev)
                if abort_dev_steps > 0.0 and _dev > abort_dev_steps:
                    _breach_run += 1
                    if _breach_run >= 3:
                        _tube_breach = True
                        log(
                            f"[ratio={ratio}] TUBE BREACH ABORT at t={t}: realized dev "
                            f"{_dev:.1f} med_steps > gate {abort_dev_steps:.1f} for "
                            f"{_breach_run} consecutive tick(s) — ending the rollout now "
                            f"(the accept gate would reject it post hoc anyway)"
                        )
                        break
                else:
                    _breach_run = 0
                if blend_dev_regulation:
                    _span = max(1e-6, float(blend_dev_zero_above) - float(blend_dev_full_below))
                    _scale_tube = float(np.clip((float(blend_dev_zero_above) - _dev) / _span, 0.0, 1.0))
                    _scale = min(_scale, _scale_tube)
                    # Pull-back event logging: engage/release transitions (not
                    # per-tick — regulation can stay engaged for long spans).
                    if _scale_tube < 1.0:
                        _tube_ticks_total += 1
                        if _tube_engaged_since is None:
                            _tube_engaged_since = t
                            log(
                                f"[ratio={ratio}] tube pull-back ENGAGED at t={t}: dev "
                                f"{_dev:.1f} med_steps > {blend_dev_full_below:.1f}, "
                                f"r_eff {ratio * _scale:.3f}"
                            )
                    elif _tube_engaged_since is not None:
                        log(
                            f"[ratio={ratio}] tube pull-back released at t={t} after "
                            f"{t - _tube_engaged_since} tick(s): dev back to {_dev:.1f} med_steps"
                        )
                        _tube_engaged_since = None
                wrapper.forward_flow_ratio = ratio * _scale
                _r_eff_sum += ratio * _scale
                _r_eff_n += 1
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
        #
        # Cadence matches the REBUILD cadence, not the chunk boundary:
        # EVERY_STEP rebuilds (and re-decodes) the guidance every tick with
        # an advancing progress cursor, so painting 32 ticks ahead from a
        # boundary snapshot showed a projection up to n_action_steps-1 ticks
        # stale — the plotted "guidance" stepped/spiked at every boundary
        # and diverged from what the blend actually consumed (user-observed,
        # 2026-08-18). Per tick we record THIS tick's decoded guidance entry;
        # at boundaries (or ONCE_PER_CHUNK) we also paint the lookahead for
        # ticks not yet written.
        if wrapper._last_decoded_guidance_chunk is not None:
            chunk_decode = wrapper._last_decoded_guidance_chunk[0]
            if decoded_guidance_full is None:
                # NaN-init, not zeros: ticks never written (post-termination
                # hold, tail after the last chunk boundary) plot as a GAP
                # instead of a fake 0.0-rad guidance trace that reads like a
                # decode bug.
                decoded_guidance_full = np.full(
                    (total_steps, chunk_decode.shape[1]), np.nan, dtype=chunk_decode.dtype
                )
            if blend_mode == BlendMode.EVERY_STEP:
                # Now-anchored: the decode refreshes at RE-BLEND ticks (every
                # `blend_interval`); on drained ticks in between, this tick's
                # guidance is the decode's entry at the drain offset — using
                # entry 0 for every tick held a constant per interval and
                # painted a staircase (user-observed at interval 0.5,
                # 2026-08-18). At blend_interval=1 the offset is always 0.
                _off = (t % n_action_steps) % max(1, int(blend_interval))
                decoded_guidance_full[t] = chunk_decode[min(_off, chunk_decode.shape[0] - 1)]
            elif at_chunk_boundary:
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
            if pad_after_success:
                log(
                    f"[ratio={ratio}] Episode succeeded at t={t + 1}/{total_steps}. "
                    f"Holding for {total_steps - t - 1} remaining steps."
                )
            else:
                log(
                    f"[ratio={ratio}] Episode succeeded at t={t + 1}/{total_steps}. "
                    f"Truncating (pad_after_success=False); "
                    f"{total_steps - t - 1} guidance steps unused."
                )
                break

    if progress_guidance:
        n_ticks = len(raw_actions)
        log(
            f"[ratio={ratio}] progress-guidance final demo cursor {_j_progress}/{_demo_arm.shape[0]} "
            f"(wall-clock ticks {n_ticks}; lag {n_ticks - _j_progress})"
        )
        if blend_dev_regulation and _dev_hist:
            log(
                f"[ratio={ratio}] tube regulation: pulled back {_tube_ticks_total}/{len(_dev_hist)} "
                f"tick(s); dev med_steps p50 {float(np.percentile(_dev_hist, 50)):.1f} "
                f"p95 {float(np.percentile(_dev_hist, 95)):.1f} max {max(_dev_hist):.1f}; "
                f"mean r_eff {(_r_eff_sum / _r_eff_n if _r_eff_n else 0.0):.3f}"
            )

    return BlendRolloutResult(
        raw_actions=np.stack(raw_actions),
        decoded_guidance_full=decoded_guidance_full,
        success=success,
        success_t=success_t,
        final_progress_cursor=_j_progress if progress_guidance else None,
        mean_effective_ratio=(_r_eff_sum / _r_eff_n) if _r_eff_n else None,
        dev_steps_p50=float(np.percentile(_dev_hist, 50)) if _dev_hist else None,
        dev_steps_p95=float(np.percentile(_dev_hist, 95)) if _dev_hist else None,
        tube_breach=_tube_breach,
    )
