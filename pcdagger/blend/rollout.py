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
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from lerobot.envs.utils import preprocess_observation
from lerobot.policies.shared_autonomy_wrapper import BlendMode
from lerobot.utils.constants import ACTION
from lerobot.utils.sim_seeding import seed_splatsim_env_to_state

# ── collision flag extraction ─────────────────────────────────────────────────


def extract_in_collision_flag(info: dict) -> bool:
    """Pull ``in_collision`` from a Gymnasium vec-env info dict.

    Vec-env wraps a single-env's info as either ``info["in_collision"]``
    (sync-vec passthrough), ``info["info_metrics"]["in_collision"]``
    (splatsim folding), or ``info["final_info"][0][...]`` (terminated step).
    Robust against all three; missing -> False.
    """
    if not info:
        return False
    if "in_collision" in info:
        return bool(np.asarray(info["in_collision"]).any())
    metrics = info.get("info_metrics")
    if isinstance(metrics, dict) and "in_collision" in metrics:
        return bool(np.asarray(metrics["in_collision"]).any())
    final = info.get("final_info")
    if final is not None:
        for sub in final:
            if not sub:
                continue
            if "in_collision" in sub:
                return bool(np.asarray(sub["in_collision"]).any())
            sm = sub.get("info_metrics")
            if isinstance(sm, dict) and "in_collision" in sm:
                return bool(np.asarray(sm["in_collision"]).any())
    return False


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


def warn_if_sim_physics_unsynced(vec_env, log=print, *, required: bool = False) -> bool | None:
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
        msg = (
            "[sim] sync_physics_to_client=OFF — the sim integrates physics in WALLCLOCK "
            "time while the policy thinks: every slow tick (chunk-rebuild denoise) becomes "
            "a physical LURCH in the achieved motion that is invisible in the commanded "
            "actions (measured 2026-08-25: pixel-motion spikes ~20x the per-tick median at "
            "every chunk boundary). Relaunch launch_nodes.py with --sync_physics_to_client."
        )
        if required:
            raise SystemExit(
                msg + "\nRefusing to record against an unsynced sim "
                "(pass --allow_unsynced_physics to override)."
            )
        log("WARNING: " + msg)
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
    if os.environ.get("DAG_BATCH_DEBUG"):
        import builtins

        _n = getattr(builtins, "_dag_batch_dbg_count", 0)
        if _n < int(os.environ.get("DAG_BATCH_DEBUG_TICKS", "2")):
            builtins._dag_batch_dbg_count = _n + 1
            for k, v in sorted(obs.items()):
                if isinstance(v, torch.Tensor):
                    print(
                        f"[batch-dbg {_n}] {k}: shape {tuple(v.shape)} "
                        f"vals {v.detach().cpu().numpy().reshape(-1)[:12].round(4).tolist()}"
                    )
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


class WorldMismatchError(RuntimeError):
    """Scenario seeding cannot reproduce the source episode's world.

    Raised by the world-match guard in :func:`run_blended_rollout`. Callers
    that iterate many source episodes should catch this, drop the offending
    episode loudly, and continue — retrying is pointless (deterministic) and
    aborting the whole run wastes every remaining episode.
    """


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
    in_collision: np.ndarray | None = None  # per-captured-frame collision flag (aligned with on_step calls)


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
    guidance_delay_steps: int = 0,
    guidance_lapse_start: int = 0,
    blend_ratio_goal_taper: int = 0,
    blend_dev_regulation: bool = False,
    blend_dev_full_below: float = 3.0,
    blend_dev_zero_above: float = 8.0,
    # Build the guidance chunk from the DART label function (dart_labels.
    # chunk_labels) evaluated at the CURRENT state and projection cursor,
    # instead of the raw demo window at demo pace. The rollout's guidance
    # then equals the train-time supervision targets by construction
    # (same function, same clock): position k is where the DART servo says
    # the robot SHOULD be at future tick k given its current deviation —
    # the clock slows for correction, so chunk seams are continuous (no
    # cursor-relock snap) and the suffix anchor's aim point needs no
    # separate timestep correction. Requires progress_guidance=True (the
    # projection cursor IS the DART demo_index).
    guidance_from_dart_labels: bool = False,
    # Monotone aim clock for DART-track guidance (indices/med-steps). On each
    # rebuild the track's demo_index is the PREVIOUS track's clock at the
    # switch position — so the aim NEVER steps backward at seams — clamped to
    # at most this many indices ahead of the robot's projection, so it cannot
    # run away from a stalled robot either (the pure state-relock retreats by
    # the per-chunk shortfall; the pure clock-resume is unbounded). <=0
    # disables the resume (pure state-relock, the original behavior).
    dart_guidance_max_lead: float = 8.0,
    seed_joint_velocity: np.ndarray | None = None,
    fps: float = 30.0,
    demo_states_raw: np.ndarray | None = None,
    pad_after_success: bool = True,
    expected_env_state: np.ndarray | None = None,
    env_state_static_mask: np.ndarray | None = None,
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
    _lapse_anchor_saved = int(getattr(wrapper, "anchor_suffix_steps", 0) or 0)
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
            _delta = np.abs(_got - _want)
            # SCENARIO dims (static over the source episode's opening frames:
            # block/obstacles) must match strictly; robot-derived dims (e.g.
            # the planar oracle EE, which moves whenever the handoff was
            # mid-motion) get 10x slack — their "mismatch" is the source
            # robot's own tracking error, not a wrong world.
            if env_state_static_mask is not None and env_state_static_mask.shape == _delta.shape:
                _bound = np.where(env_state_static_mask, env_state_match_tol, 10.0 * env_state_match_tol)
            else:
                _bound = np.full_like(_delta, env_state_match_tol)
            _err = float(np.max(_delta - _bound))
            if _err > 0:
                _err = float(np.max(_delta))
                raise WorldMismatchError(
                    f"WORLD MISMATCH after scenario seeding: environment_state differs from the "
                    f"source episode's by max|delta|={_err:.3f} (> {env_state_match_tol}) at "
                    f"benchmark_start_index={benchmark_start_index}. Two known causes: (a) the sim "
                    f"serves the WRONG scenario (bad benchmark subset/ordering — most dims differ), "
                    f"or (b) the SOURCE EPISODE itself did not start from the pristine scenario "
                    f"(the pre-intervention policy roll displaced an object before the takeover — "
                    f"only that object's dims differ). Either way this world cannot be reproduced "
                    f"by scenario seeding, and rollouts in a wrong world produce colliding replays "
                    f"and poisoned training data.\n"
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
    # Per-captured-frame collision flags: the flag for the state each
    # on_step/raw_actions entry observes (from the info of the step that
    # PRODUCED that state). Recorded into the blend dataset as
    # ``frame_in_collision`` so collision filtering can happen at TRAIN time
    # (loader-side) instead of via a destructive replay filter.
    _coll_hist: list[bool] = []
    _last_info: dict = {}
    _dart_geom = None
    _q_hist: list[np.ndarray] = []
    _dart_prev_clock: list[float] | None = None
    _dart_prev_build_t = 0
    _dart_prev_proj: float | None = None
    if guidance_from_dart_labels:
        if not progress_guidance:
            raise ValueError("guidance_from_dart_labels requires progress_guidance=True")
        from dart_labels import (
            _interp_rows as _dart_interp_rows,
            chunk_labels as _dart_chunk_labels,
            demo_geometry as _dart_demo_geometry,
        )

        _dart_states = demo_states_raw if demo_states_raw is not None else guidance_actions_raw
        _dart_geom = _dart_demo_geometry(
            np.asarray(_dart_states, dtype=np.float64),
            np.asarray(guidance_actions_raw, dtype=np.float64),
            _num_arm,
        )

    for t in range(total_steps):
        # ── Hold mode: episode succeeded, don't step env again ────────────────
        # Stepping after termination triggers AutoresetMode.NEXT_STEP and would
        # bring in the next scene's images, causing a sharp visual transition.
        if success:
            assert hold_action is not None and terminal_env_obs is not None
            if on_step is not None:
                on_step(t, terminal_env_obs, hold_action, True)
            _coll_hist.append(extract_in_collision_flag(_last_info))
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
            # DART-track aim clock (see guidance_from_dart_labels): computed
            # HERE — before the ratio regulators — because the goal taper must
            # key on the clock the guidance actually runs on. Keying it on the
            # vestigial demo-pace _j_clock left the taper lagging the track:
            # near the demo end the track already aims at the final pose while
            # the taper still sees indices remaining, so the ratio stays up
            # and the r*(policy-expert) equilibrium offset stalls the robot
            # short of strict success until _j_clock catches up and the ratio
            # collapses at once (stall-then-leap, measured ep35 t=173-176).
            _dart_aim = float(_j_progress)
            if guidance_from_dart_labels:
                if dart_guidance_max_lead > 0 and _dart_prev_clock is not None:
                    _switch_pos = min(max(0, t - _dart_prev_build_t), len(_dart_prev_clock) - 1)
                    _dart_aim = min(
                        max(_dart_aim, float(_dart_prev_clock[_switch_pos])),
                        float(_j_progress) + float(dart_guidance_max_lead),
                    )
                _dart_aim = min(_dart_aim, float(guidance_actions_raw.shape[0] - 1))
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
                    # The taper removes the r*(policy-expert) equilibrium
                    # offset when the ROBOT nears the goal — so it must key on
                    # the robot's projection. The vestigial _j_clock LAGS it
                    # (ratio stays up too long -> stall short of strict
                    # success, then a leap when the clock catches up); the
                    # DART aim clock LEADS it (the track's clock saturates at
                    # the demo end ~a chunk before the robot arrives -> ratio
                    # collapses while the robot is still away -> the ratio-0
                    # branch teleport-commands the end pose). Both measured on
                    # ep35, 2026-08-25.
                    _taper_clock = float(_j_progress) if guidance_from_dart_labels else _j_clock
                    _remaining = float(guidance_actions_raw.shape[0] - 1) - _taper_clock
                    # NOTE: do NOT shift this ramp to reach zero early — a
                    # near-zero ratio while the robot still has lateral offset
                    # couples badly with the lead-capped aim (commands park at
                    # demo[aim] instead of gliding in; measured ep35: stall at
                    # 3.5 med + 3.6-med leap with a remaining-2 shift). The
                    # plain ramp gives a monotone approach with only <=0.5
                    # med-step settling wiggles in the final ~3 ticks, which
                    # the DART hold-free window excludes from serving anyway.
                    _scale = min(_scale, max(0.0, min(1.0, _remaining / float(blend_ratio_goal_taper))))
                # Deviation telemetry (always on under progress guidance —
                # feeds the realized-tube episode metadata even when the
                # state-feedback regulation below is disabled).
                _dev = (
                    float(np.linalg.norm(_q_now.astype(np.float64) - _demo_arm[_j_progress])) / _pg_med_step
                )
                _dev_hist.append(_dev)
                # Timestep correction for the suffix anchor (see
                # wrapper.anchor_clock_lag / anchor_suffix_steps): a robot
                # _dev med-steps off-corridor needs ~_dev ticks of the chunk
                # to correct before it can track demo pace (the DART servo's
                # clock slowdown), so the chunk-end aim point is the demo at
                # (cursor + T - _dev). Published every tick; the obs-teleop
                # source consumes it as an index shift at anchor-build time.
                if int(getattr(wrapper, "anchor_suffix_steps", 0) or 0) > 0 or getattr(
                    wrapper, "anchor_suffix_to_goal", False
                ):
                    wrapper.anchor_clock_lag = max(0, int(round(_dev)))
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
            # LAPSE (guidance_delay_steps > 0): for ticks in
            # [guidance_lapse_start, guidance_lapse_start + guidance_delay_steps)
            # the policy drives free — guidance fully renoised, suffix anchor
            # off — then shared control (re-)engages. lapse_start=0 simulates
            # 'the expert grabbed it tau ticks later'; a mid-episode window
            # simulates an attention lapse anywhere along the intervention,
            # supplying the departure states the guided measurement
            # structurally suppresses — at every trajectory region.
            if guidance_delay_steps > 0:
                _l0 = int(guidance_lapse_start)
                _l1 = _l0 + int(guidance_delay_steps)
                if _l0 <= t < _l1:
                    wrapper.forward_flow_ratio = 1.0
                    wrapper.anchor_suffix_steps = 0
                elif t == _l1 or (t < _l0):
                    wrapper.forward_flow_ratio = ratio * _scale if ratio not in (0.0, 1.0) else ratio
                    wrapper.anchor_suffix_steps = _lapse_anchor_saved
                _r_eff_n += 1
            if guidance_from_dart_labels:
                # DART-track guidance: same label function as training.
                _q_hist.append(_q_now.astype(np.float64).copy())
                if len(_q_hist) > 8:
                    _q_hist.pop(0)
                if suppress_guidance:
                    guidance_chunk = None
                else:
                    # PLAN-CONTINUOUS LAUNCH: the track launches from the demo
                    # corridor at the (monotone, pace-matched) aim clock — NOT
                    # from the robot's state. State-launched tracks restart at
                    # the robot every rebuild, which puts a backward value-step
                    # into the guidance whenever the robot lags its plan (the
                    # rejoin transient re-encoded each refresh). Labels must
                    # launch from the visited state (train-time semantics);
                    # guidance must not (rollout-time reference). The blend
                    # ratio and the suffix anchor are what pull the robot
                    # toward this reference; the max-lead clamp on the aim is
                    # what keeps the reference from running away from a robot
                    # that cannot follow.
                    # PACE-MATCHED speed budget ("account for the ticks that
                    # lead up to the anchor"): the servo's default budget
                    # (1.2x demo median) is the pace of a robot that FOLLOWS
                    # the track — but the executor is a ratio-blend of policy
                    # and track, which realizes less arc per tick. A track
                    # paced faster than the blend can move puts every future
                    # waypoint (and the suffix anchor) out of reach, so each
                    # rebuild reveals a shortfall: the guidance snaps back
                    # (state-relock) or ratchets ahead (clock-resume). Budget
                    # the track at the REALIZED arc pace of the previous
                    # inter-refresh window (slight optimism, floored so a
                    # stall can recover) and the plan aims where the blend
                    # will actually be — seams close at the source.
                    _speed_budget = 1.2
                    if _dart_prev_proj is not None and t > _dart_prev_build_t:
                        _i0 = int(np.clip(_dart_prev_proj, 0, len(_dart_geom.cum) - 1))
                        _i1 = int(np.clip(_j_progress, 0, len(_dart_geom.cum) - 1))
                        _arc = float(_dart_geom.cum[_i1] - _dart_geom.cum[_i0])
                        _rate = _arc / ((t - _dart_prev_build_t) * _dart_geom.med_step)
                        _speed_budget = float(min(1.2, max(0.3, 1.05 * _rate)))
                    # Monotone aim clock — computed above (before the ratio
                    # regulators, which the goal taper keys on it).
                    _aim = _dart_aim
                    _launch = _dart_interp_rows(_dart_geom.P, _aim)
                    _launch_prev = _dart_interp_rows(_dart_geom.P, max(0.0, _aim - _speed_budget))
                    _v0 = _launch - _launch_prev
                    _dart_info: dict = {}
                    _track = _dart_chunk_labels(
                        _launch,
                        _aim,
                        _dart_geom,
                        horizon=64,
                        speed_budget=_speed_budget,
                        prev_state=_launch_prev,
                        velocity=_v0,
                        info=_dart_info,
                    )
                    if os.environ.get("DAG_DART_AIM_DEBUG"):
                        _pc = None
                        if _dart_prev_clock is not None:
                            _sp = min(max(0, t - _dart_prev_build_t), len(_dart_prev_clock) - 1)
                            _pc = round(float(_dart_prev_clock[_sp]), 1)
                        print(
                            f"[dart-aim] t={t} chunk_offset={t % n_action_steps} "
                            f"dt_prev_build={t - _dart_prev_build_t} proj={_j_progress} "
                            f"prev_clock@switch={_pc} aim={_aim:.1f} sb={_speed_budget:.2f} "
                            f"track_clock0={float((_dart_info.get('clock') or [0])[0]):.1f} "
                            f"track_clock16={float((_dart_info.get('clock') or [0] * 17)[min(16, len(_dart_info.get('clock') or [0]) - 1)]):.1f}"
                        )
                    _dart_prev_clock = list(_dart_info.get("clock") or []) or None
                    _dart_prev_build_t = t
                    _dart_prev_proj = float(_j_progress)
                    _track = np.asarray(_track, dtype=np.float32)
                    if _track.shape[1] < guidance_actions_raw.shape[1]:
                        # Defensive: pad non-arm columns (gripper) from the demo
                        # at the track's own clock indices.
                        _clock = _dart_info.get("clock") or [float(_j_progress)] * len(_track)
                        _pad_rows = np.stack(
                            [
                                guidance_actions_raw[min(int(round(c)), guidance_actions_raw.shape[0] - 1)]
                                for c in _clock
                            ]
                        ).astype(np.float32)
                        _full = _pad_rows.copy()
                        _full[:, : _track.shape[1]] = _track
                        _track = _full
                    guidance_chunk = _track
                    # The track already embeds the servo's slowed clock — the
                    # suffix anchor must NOT apply a second timestep shift.
                    wrapper.anchor_clock_lag = 0
                    # Goal-hold start for anchor_suffix_to_goal: the dart
                    # track is fixed-horizon with its clock HOLDING at the
                    # demo end, so the hold lives INSIDE the provided rows
                    # (the fill's shortfall detection sees none). Publish the
                    # track-row index where the clock saturates; the obs-
                    # teleop source converts it to a chunk-tail length (the
                    # track may be longer than the chunk). Lag is already
                    # embedded in the servo clock — no further -lag applies.
                    _clk = _dart_info.get("clock") or []
                    _end_clk = float(len(_dart_geom.A) - 1)
                    wrapper.anchor_goal_hold_start = next(
                        (i for i, c in enumerate(_clk) if float(c) >= _end_clk - 1e-6), None
                    )
            else:
                guidance_chunk = None if suppress_guidance else guidance_actions_raw[_j_exec:]
                wrapper.anchor_goal_hold_start = None
        else:
            guidance_chunk = None if suppress_guidance else guidance_actions_raw[t:]
            wrapper.anchor_goal_hold_start = None

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
        if os.environ.get("DAG_DECODE_DEBUG") and t < int(os.environ.get("DAG_DECODE_DEBUG_TICKS", "4")):
            _steps_info = []
            for _s in wrapper.postprocessor.steps:
                _en = getattr(_s, "enabled", "n/a")
                _steps_info.append(f"{type(_s).__name__}(enabled={_en})")
            _rs = None
            for _s in wrapper.postprocessor.steps:
                _rs = getattr(_s, "relative_step", None) or _rs
            _anchor = None
            if _rs is not None and getattr(_rs, "_last_state", None) is not None:
                _anchor = _rs._last_state.detach().cpu().numpy().reshape(-1)[:3].round(3).tolist()
            print(f"[decode-dbg] t={t} steps={_steps_info}")
            print(
                f"[decode-dbg]   action_norm={action_norm.detach().cpu().numpy().reshape(-1)[:3].round(3).tolist()}"
            )
            print(
                f"[decode-dbg]   raw_action ={raw_action.detach().cpu().numpy().reshape(-1)[:3].round(3).tolist()}"
            )
            print(
                f"[decode-dbg]   q_now      ={np.asarray(env_obs['agent_pos']).reshape(-1)[:3].round(3).tolist()}"
            )
            print(
                f"[decode-dbg]   rel_anchor ={_anchor} rel_enabled={getattr(_rs, 'enabled', None) if _rs is not None else None}"
            )

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
        _coll_hist.append(extract_in_collision_flag(_last_info))
        if os.environ.get("DAG_COLL_DEBUG") and _coll_hist[-1]:
            print(f"[coll-dbg] t={t} IN COLLISION")
        raw_actions.append(action_1d)

        env_obs, _reward, _term, _trunc, _info = vec_env.step(action_numpy)
        _last_info = _info

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
    # Collision visibility: eval TERMINATES on collision (terminate_on_collision
    # in the training env config), but this rollout keeps stepping — a collided
    # robot stays pinned against the obstacle for the rest of the episode,
    # which reads as a policy "limit cycle" in videos/metrics unless flagged
    # (scenario 0 ratio-1.0 debugging, 2026-08-24: 102/155 ticks in collision
    # were mistaken for a wrapper chunking bug).
    if any(_coll_hist):
        _first_coll = _coll_hist.index(True)
        log(
            f"[ratio={ratio}] COLLISION: {sum(_coll_hist)}/{len(_coll_hist)} ticks in collision "
            f"(first at t={_first_coll}); true eval would have TERMINATED there."
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
        in_collision=np.asarray(_coll_hist, dtype=bool),
    )
