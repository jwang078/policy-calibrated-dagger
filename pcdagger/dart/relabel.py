"""DART label synthesis for relabeled blend datasets (train-time chunk labels).

The DART label for a chunked policy is a FUNCTION, not a value: from a
visited (perturbed) state, the expert's H-step response rejoins the demo
corridor and then follows the demo. A per-frame action column cannot store
that function because training chunks overlap — every frame is position k of
H different chunks, each needing a different value — so the label chunk is
synthesized where it is consumed (dataloader / visualizer) from two things:

  * the frame's own ``observation.state`` (already in the batch), and
  * ``relabel_demo_index`` — the continuous demo index of the state's
    projection onto the source demo's state polyline, computed ONCE at
    record time by the same monotone/windowed/rate-capped arc cursor the
    progress guidance uses (storing it offline preserves the jump-proof
    disambiguation; recomputing at train time could re-project ambiguously
    near self-close passes).

Synthesis (all quantities in the demo's own units — no per-env tuning):

    c_k      = min(rate*med_step, ease_out*d_k, ease_in*(k+1)*med_step)  # S-curve
    t_k      = min(med_step, sqrt((B*med_step)^2-c_k^2))  # leftover -> progress
    i_k+1    = i_k + t_k/med_step                       # clock slows while correcting
    label_k  = demo_action(i_k) + d_k * u               # B = speed_budget (1.2)

``med_step`` is the demo's median per-tick joint step (fps/DOF/speed baked
in), so ``rate=1.0`` means the label track rejoins at the demo's own cruise
speed and the commanded step at EVERY chunk position is <= (1+rate)x the
env's own convention. k=0 reproduces the validated per-frame label exactly;
by ~d0/(rate*med_step) ticks (~6 for a 0.1 rad deviation) the chunk IS the
demo — chunks converge to the true expert actions within the horizon, which
per-frame stored labels structurally cannot do.
"""

from __future__ import annotations

import glob
import math
import os
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class DemoGeometry:
    """Source-demo polyline geometry needed for projection and synthesis."""

    P: np.ndarray  # (T, n_arm) demo states (the polyline)
    A: np.ndarray  # (T, adim) demo actions (action[i] governs P[i] -> P[i+1])
    seg: np.ndarray  # (T-1, n_arm) segment vectors
    seg_len: np.ndarray  # (T-1,) segment lengths
    cum: np.ndarray  # (T,) cumulative arc length
    med_step: float  # median non-degenerate segment length
    acc_p95: float = 0.0  # p95 per-tick^2 accel of the demo path (rad/tick^2)
    acc_mean: float = 0.0  # mean per-tick^2 accel of the demo path (rad/tick^2)

    @property
    def n_arm(self) -> int:
        """Number of arm joints in the polyline."""
        return self.P.shape[1]


def demo_geometry(demo_states_raw: np.ndarray, demo_actions_raw: np.ndarray, n_arm: int) -> DemoGeometry:
    """Build the polyline geometry from a source episode's states/actions."""
    pts = np.asarray(demo_states_raw, dtype=np.float64)[:, :n_arm]
    acts = np.asarray(demo_actions_raw, dtype=np.float64)
    seg = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    live = seg_len[seg_len > 1e-9]
    med_step = float(np.median(live)) if len(live) else 1e-3
    acc = np.linalg.norm(np.diff(pts, n=2, axis=0), axis=1) if len(pts) > 2 else np.zeros(1)
    acc_p95 = float(np.percentile(acc, 95)) if len(acc) else 0.0
    acc_mean = float(np.mean(acc)) if len(acc) else 0.0
    return DemoGeometry(
        P=pts,
        A=acts,
        seg=seg,
        seg_len=seg_len,
        cum=cum,
        med_step=med_step,
        acc_p95=acc_p95,
        acc_mean=acc_mean,
    )


def project_states(
    states: np.ndarray,
    geom: DemoGeometry,
    index_window: int,
    rate_cap_steps: float = 3.0,
) -> np.ndarray:
    """Project a state sequence onto the demo polyline -> continuous demo indices.

    Monotone, windowed (``index_window`` demo steps ahead of the cursor),
    per-tick arc advance capped at ``rate_cap_steps`` demo steps — identical
    cursor semantics to the progress guidance, so shortcut-induced jumps are
    structurally impossible. Returns float indices ``i + f`` with
    ``f in [0, 1)`` along segment ``i``.
    """
    arr = np.asarray(states, dtype=np.float64)[:, : geom.n_arm]
    arc_window = max(1, int(index_window)) * geom.med_step
    s_prev = 0.0
    out = np.empty(len(arr), dtype=np.float64)
    for t, q in enumerate(arr):
        lo = max(int(np.searchsorted(geom.cum, s_prev)) - 1, 0)
        hi = min(int(np.searchsorted(geom.cum, s_prev + arc_window)) + 1, len(geom.seg_len))
        best_s, best_d = s_prev, np.inf
        for i in range(lo, hi):
            if geom.seg_len[i] < 1e-9:
                continue
            u = float(np.clip(np.dot(q - geom.P[i], geom.seg[i]) / (geom.seg_len[i] ** 2), 0.0, 1.0))
            proj = geom.P[i] + u * geom.seg[i]
            d = float(np.linalg.norm(q - proj))
            s_cand = max(float(geom.cum[i] + u * geom.seg_len[i]), s_prev)  # monotone
            if d < best_d:
                best_d, best_s = d, s_cand
        s_prev = min(best_s, s_prev + rate_cap_steps * geom.med_step)  # capped advance
        idx = int(np.clip(np.searchsorted(geom.cum, s_prev) - 1, 0, len(geom.seg_len) - 1))
        f = float(np.clip((s_prev - geom.cum[idx]) / max(geom.seg_len[idx], 1e-9), 0.0, 1.0))
        out[t] = idx + f
    return out


def project_state_local(q: np.ndarray, geom: DemoGeometry, center_index: float, window_steps: float) -> float:
    """Project ONE state onto the demo polyline near ``center_index``.

    Non-monotone local search over segments within ``window_steps`` demo
    steps of the center — used by state-noise augmentation, where the
    perturbed state's best projection may sit slightly ahead of or behind
    the anchor frame's own index (a tangential noise component must advance
    or retard the clock rather than masquerade as lateral deviation to
    "correct"). Returns a continuous index ``i + f``.
    """
    lo = max(0, int(np.floor(center_index - window_steps)))
    hi = min(len(geom.seg_len), int(np.ceil(center_index + window_steps)) + 1)
    best_i, best_d = float(np.clip(center_index, 0, len(geom.seg_len))), np.inf
    qa = np.asarray(q, dtype=np.float64)[: geom.n_arm]
    for i in range(lo, hi):
        if geom.seg_len[i] < 1e-9:
            continue
        u = float(np.clip(np.dot(qa - geom.P[i], geom.seg[i]) / (geom.seg_len[i] ** 2), 0.0, 1.0))
        d = float(np.linalg.norm(qa - (geom.P[i] + u * geom.seg[i])))
        if d < best_d:
            best_d, best_i = d, i + u
    return best_i


def _interp_rows(mat: np.ndarray, i: float) -> np.ndarray:
    """Linear interpolation of row ``i`` (float, clamped) of matrix ``mat``."""
    idx = int(np.clip(np.floor(i), 0, len(mat) - 1))
    nxt = min(idx + 1, len(mat) - 1)
    f = float(np.clip(i - idx, 0.0, 1.0))
    return (1.0 - f) * mat[idx] + f * mat[nxt]


def _cartesian_shape(
    e: np.ndarray, v_ref: np.ndarray, jac: np.ndarray, cartesian_lambda: float
) -> np.ndarray:
    """Damp the parts of the position error whose EE image is louder than the demo's.

    ``W = (s_t^2 + lam) (J^T J + lam I)^-1``, ``lam = cartesian_lambda * s_t^2``,
    ``s_t = |J u_t|`` the EE gain of the demo's direction of travel. Unit gain
    along ``u_t`` by construction; a direction with EE gain ``s`` is scaled by
    ``(s_t^2 + lam) / (s^2 + lam)``. A degenerate reference velocity (the demo
    at rest) leaves the error untouched — there is no tangential direction to
    normalize against, and damping everything would just stall the rejoin.
    """
    jac = np.asarray(jac, dtype=np.float64)
    nv = float(np.linalg.norm(v_ref))
    if nv < 1e-12:
        return e
    st2 = float(np.linalg.norm(jac @ (v_ref / nv)) ** 2)
    lam = max(float(cartesian_lambda), 0.0) * st2
    if st2 <= 0.0 or lam <= 0.0:
        return e
    n = jac.shape[1]
    return (st2 + lam) * np.linalg.solve(jac.T @ jac + lam * np.eye(n), e)


def chunk_labels(
    state: np.ndarray,
    demo_index: float,
    geom: DemoGeometry,
    horizon: int,
    rate: float = 1.0,
    ease_out: float = 0.3,
    speed_budget: float = 1.2,
    ease_in: float = 0.35,
    prev_state: np.ndarray | None = None,
    glide_ratio: float = 2.0,
    velocity: np.ndarray | None = None,
    accel_budget: float = 1.0,
    jacobian_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    cartesian_lambda: float = 4.0,
    info: dict | None = None,
) -> np.ndarray:
    """Synthesize the expert's ``horizon``-step response from one state.

    Demo clock advances one index per tick from ``demo_index`` (holding at
    the demo's end); the state's offset from its corridor projection decays
    along a fixed direction by ``min(rate * med_step, ease_out * d)`` per
    tick: full-cruise pursuit while far (bounded first step), proportional
    within ``rate * med_step / ease_out`` of the corridor so the closure
    velocity fades geometrically instead of stopping dead — a hard-zero
    linear decay would end with a one-tick velocity discontinuity of
    ``rate * med_step * fps`` (~16 rad/s^2 at planar cruise, ~18x the
    demo's own max accel) that a chunk-mimicking policy would learn.
    Merge deceleration is bounded by ``ease_out * rate * med_step`` per
    tick^2; residual offsets below 5% of a demo step snap closed. Returns
    ``(horizon, adim)`` absolute-position labels.
    """
    q = np.asarray(state, dtype=np.float64)[: geom.n_arm]
    proj0 = _interp_rows(geom.P, demo_index)
    offset = q - proj0
    d0 = float(np.linalg.norm(offset))
    u = offset / d0 if d0 > 1e-9 else np.zeros_like(offset)
    close = max(0.0, float(rate)) * geom.med_step
    ease = float(np.clip(ease_out, 0.0, 1.0))
    labels = np.empty((horizon, geom.A.shape[1]), dtype=np.float64)
    end = float(len(geom.A) - 1)
    d_k = d0
    # SPEED-BUDGETED rejoin: each label step spends at most
    # speed_budget * med_step of total motion, correction FIRST, and the
    # demo clock advances at whatever rate the leftover affords
    # (sqrt(budget^2 - c^2), capped at demo pace). While correcting hard the
    # clock runs slow (~0.66x at budget 1.2, correction 1.0), so the
    # rendezvous lands further in time at a demo point the robot can
    # feasibly reach — additive progress+correction demanded sustained
    # 1.4-1.6x-cruise labels the model's action prior never contains, which
    # a trained policy systematically undershoots (worst at wide tubes,
    # where rejoins last up to tube_steps ticks).
    b = max(1.0, float(speed_budget)) * geom.med_step
    end_i = float(len(geom.A) - 1)
    # QUINTIC MERGE (position+velocity continuous at both ends, accel ZERO
    # at both ends): when the state's own velocity is known and the offset
    # is non-trivial, the rejoin is a min-jerk quintic from (q, v0) to (demo
    # action at a budget-chosen rendezvous index, demo velocity there).
    # Launch direction AND curvature match the robot's current motion, so
    # the track initially continues like the policy and bends mid-flight.
    # Rendezvous time T is the larger of the speed budget (chord/T <= B) and
    # the accel budget (peak ~5.77*d0/T^2 <= accel_budget x demo MEAN accel);
    # the rendezvous index is the tangential distance the speed budget
    # affords: di = sqrt((T*B)^2 - d0^2).
    # No offset floor: even ON the corridor the quintic matters — an anchor
    # whose speed differs from demo pace (episode start: robot at rest, demo
    # cruising) needs the same smooth velocity ramp, and with d0 ~ 0 the
    # merge degenerates gracefully to demo-following.
    if velocity is not None or prev_state is not None:
        # CRITICALLY DAMPED SERVO (initial-value formulation). The previous
        # quintic was a BOUNDARY-value problem — pick a rendezvous point and
        # time, let a polynomial fill the middle — and nothing in that
        # construction penalizes distance from the demo, so any mismatch
        # between the boundary data and the (T, rendezvous) choice became
        # interior wandering (detours, loiter, speed bulges, rendezvous
        # collapse, wrong-way lobes: one root cause, five symptoms). Here
        # the label track is INTEGRATED from (q, v0) under
        #
        #     a_k = Kp * (p_ref - x_k) + Kd * (v_ref - v_k),  Kd = 2*sqrt(Kp)
        #
        # toward the demo's position/velocity field at the track's OWN
        # monotone projection cursor. Critical damping is the classical
        # no-overshoot condition: the lateral error decays monotonically —
        # a detour is impossible by construction — and demo-following,
        # braking a wrong-way launch, retiming a speed mismatch, and
        # decelerating into the demo's end are all the SAME law, not
        # special cases. Accel is clamped to the demo envelope and slewed
        # (bounded jerk); Kp is set for a ~45-tick settle, which puts the
        # corrective accel at ~Kp*d0: demo-MEAN scale for typical offsets,
        # envelope scale only at the tube edge.
        if velocity is not None:
            v0 = np.asarray(velocity, dtype=np.float64)[: geom.n_arm].copy()
        else:
            v0 = q - np.asarray(prev_state, dtype=np.float64)[: geom.n_arm]
        sp0 = float(np.linalg.norm(v0))
        b = max(1.0, float(speed_budget)) * geom.med_step
        if sp0 > 2.5 * b:
            v0 = v0 * (2.5 * b / sp0)  # sanity cap only — outlier estimates
            sp0 = 2.5 * b
        end_i = float(len(geom.A) - 1)
        a_clamp = 0.9 * max(2.0 * float(np.sqrt(geom.n_arm)) * geom.acc_p95, 0.2 * geom.med_step)
        # ADAPTIVE gain: correct as decisively as the envelope allows — the
        # initial corrective accel Kp*d0 sits at the clamp, so settle time
        # scales with sqrt(offset) (~18 ticks at 3 med-steps, ~37 at a tube
        # edge) instead of a fixed timid constant; the floor keeps
        # near-corridor anchors from amplifying mrad noise at envelope
        # accel. Critical damping (Kd = 2*sqrt(Kp)) still guarantees the
        # no-overshoot/no-detour property.
        kp = a_clamp / max(d0, 4.0 * geom.med_step)
        kd = 2.0 * float(np.sqrt(kp))
        a_slew = 0.5 * a_clamp  # bounded jerk: accel ramps over ~2 ticks
        v_cap = max(1.4 * b, 1.05 * sp0)
        x = q.copy()
        v = v0.copy()
        a_prev = np.zeros_like(q)
        # local monotone arc cursor (same semantics as project_states),
        # advanced by projecting the LABEL's own point each tick.
        s = float(np.interp(demo_index, np.arange(len(geom.cum)), geom.cum))
        arc_window = 12.0 * geom.med_step
        clock: list[float] = []
        hit_end = False
        for k in range(horizon):
            lo = max(int(np.searchsorted(geom.cum, s)) - 1, 0)
            hi = min(int(np.searchsorted(geom.cum, s + arc_window)) + 1, len(geom.seg_len))
            best_s, best_d = s, np.inf
            for i in range(lo, hi):
                if geom.seg_len[i] < 1e-9:
                    continue
                u_seg = float(np.clip(np.dot(x - geom.P[i], geom.seg[i]) / (geom.seg_len[i] ** 2), 0.0, 1.0))
                d_seg = float(np.linalg.norm(x - (geom.P[i] + u_seg * geom.seg[i])))
                if d_seg < best_d:
                    best_d = d_seg
                    best_s = max(float(geom.cum[i] + u_seg * geom.seg_len[i]), s)
            s = min(best_s, s + 3.0 * geom.med_step)  # capped, monotone
            seg_i = int(np.clip(np.searchsorted(geom.cum, s) - 1, 0, len(geom.seg_len) - 1))
            frac = float(np.clip((s - geom.cum[seg_i]) / max(geom.seg_len[seg_i], 1e-9), 0.0, 1.0))
            i_f = seg_i + frac
            clock.append(i_f)
            a_row = _interp_rows(geom.A, i_f)
            if i_f >= end_i - 1e-6:
                hit_end = True
                p_ref = geom.A[-1][: geom.n_arm]
                v_ref = np.zeros(geom.n_arm)
            else:
                p_ref = a_row[: geom.n_arm]
                v_ref = (_interp_rows(geom.A, min(i_f + 1.0, end_i)) - a_row)[: geom.n_arm]
            e = p_ref - x
            if jacobian_fn is not None:
                e = _cartesian_shape(e, v_ref, jacobian_fn(x), cartesian_lambda)
            a_cmd = kp * e + kd * (v_ref - v)
            na = float(np.linalg.norm(a_cmd))
            if na > a_clamp:
                a_cmd *= a_clamp / na
            da = a_cmd - a_prev
            nda = float(np.linalg.norm(da))
            if nda > a_slew:
                a_cmd = a_prev + da * (a_slew / nda)
            a_prev = a_cmd
            v = v + a_cmd
            nv = float(np.linalg.norm(v))
            if nv > v_cap:
                v *= v_cap / nv
            x = x + v
            row = a_row.copy()
            row[: geom.n_arm] = x
            labels[k] = row
        if info is not None:
            info.update(
                branch="servo",
                cartesian=jacobian_fn is not None,
                d0=float(d0),
                sp0=float(sp0),
                end_clamped=bool(hit_end),
                clock=clock,
            )
        return labels

    idx = float(demo_index)
    for k in range(horizon):
        labels[k] = _interp_rows(geom.A, min(idx, end))
        c_k = min(close, ease * d_k) if ease > 0.0 else close
        # EASE-IN (mirror of the ease-out): lateral correction ramps up over
        # the first ~1/ease_in ticks instead of starting at full rate — a
        # correction-first start launched ~56 deg off the direction of
        # travel ('explicitly switching directions'); with the ramp the
        # rejoin is an on-ramp S-curve that initially moves like the policy
        # and bends into the corridor progressively.
        if ease_in > 0.0:
            c_k = min(c_k, float(ease_in) * (k + 1) * geom.med_step)
        c_k = min(c_k, d_k)
        d_k = d_k - c_k
        if d_k < 0.05 * geom.med_step:
            d_k = 0.0
        if d_k > 0.0:
            labels[k, : geom.n_arm] += d_k * u
        t_k = min(geom.med_step, float(np.sqrt(max(0.0, b * b - c_k * c_k))))
        idx = min(idx + t_k / geom.med_step, end)
    if info is not None:
        info.update(branch="pursuit", braked=False, d0=float(d0))
    return labels


def per_frame_labels(
    states: np.ndarray,
    geom: DemoGeometry,
    index_window: int,
    rate: float = 1.0,
    rate_cap_steps: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Project a state sequence and return (demo_indices, k=0 labels per frame).

    The labels equal ``chunk_labels(...)[0]`` at each frame — the view the
    per-frame relabel visualizer plots.
    """
    idxs = project_states(states, geom, index_window, rate_cap_steps=rate_cap_steps)
    labels = np.stack(
        [
            chunk_labels(
                states[i],
                idxs[i],
                geom,
                horizon=1,
                rate=rate,
                prev_state=states[i - 1] if i > 0 else None,
            )[0]
            for i in range(len(states))
        ]
    )
    return idxs, labels


# ── Train-time wrapper ────────────────────────────────────────────────────


def load_source_geometries(
    source_repo_id: str, n_arm: int, root: str | None = None
) -> dict[int, DemoGeometry]:
    """Load every episode of a source dataset as DemoGeometry, keyed by episode."""
    import pandas as pd

    if root is None:
        from lerobot.utils.constants import HF_LEROBOT_HOME

        root = str(HF_LEROBOT_HOME / source_repo_id)
    else:
        root = os.path.join(str(root), source_repo_id)
    files = sorted(glob.glob(os.path.join(root, "data/**/*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no parquet files under {root}")
    df = pd.concat([pd.read_parquet(f) for f in files])
    out: dict[int, DemoGeometry] = {}
    for ep, g in df.groupby("episode_index"):
        out[int(ep)] = demo_geometry(
            np.stack(g["observation.state"].to_numpy()),
            np.stack(g["action"].to_numpy()),
            n_arm=n_arm,
        )
    return out


class DartChunkDataset:
    """Wrap a blend LeRobotDataset: replace loaded action chunks with DART labels.

    The wrapped dataset must carry ``relabel_demo_index`` (recorded by
    ``augment_dataset_with_blending --relabel_actions=guidance``) and
    per-episode ``source_episode_idx`` metadata pairing it to the source
    (intervention) dataset. Everything else — observations, sampling,
    delta_timestamps — passes through untouched; no policy is involved.

    Usage in a finetune script::

        ds = LeRobotDataset(blend_repo, delta_timestamps=...)
        ds = DartChunkDataset(ds, source_repo_id, n_arm=3, rate=1.0)
    """

    def __init__(
        self,
        dataset,
        source_repo_id: str | None = None,
        n_arm: int | None = None,
        rate: float = 1.0,
        ease_out: float = 0.3,
        root: str | None = None,
        collision_filter: str = "none",
        collision_margin: int = 10,
        self_relabel: bool = False,
        state_noise_std: float = 0.0,
        vel_noise_std: float = 0.0,
        state_noise_p: float = 1.0,
        state_noise_schedule: str = "",
        state_noise_scale: float = 1.0,
        state_noise_seed: int | None = None,
        state_noise_min_sigma: float = 0.0,
        mask_hold_tail: bool = False,
        raw_mix: float = 0.0,
    ):
        """Wrap ``dataset``, pairing its episodes to its source demos.

        ``source_repo_id=None`` resolves the source dataset from the blend
        dataset's own ``source_dataset_repo_id`` episode metadata (stamped by
        augment_dataset_with_blending). ``n_arm=None`` infers action_dim - 1
        (trailing gripper dim, the same convention the recorder used).
        """
        self.dataset = dataset
        self.rate = float(rate)
        self.ease_out = float(ease_out)
        if collision_filter not in ("none", "drop", "trim_first_collision"):
            raise ValueError(
                f"collision_filter must be none|drop|trim_first_collision, got {collision_filter!r}"
            )
        self.collision_filter = collision_filter
        self.collision_margin = int(collision_margin)
        # "Base DART" mode: treat every episode of THIS dataset as its own
        # demo (identity pairing, geometry from its own states/actions,
        # demo_index = frame_index) — lets intervention datasets be wrapped
        # without any relabel columns or blend rollouts.
        self.self_relabel = bool(self_relabel)
        # Train-time state-noise augmentation (classic DART): perturb the
        # anchor state by N(0, (std*med_step)^2) on the arm dims with
        # probability p, re-project locally, and synthesize the label from
        # the perturbed state. The anchor's assumed VELOCITY is the demo
        # tangent at the progress-aligned (projected) index — a random
        # offset has no velocity of its own, so the history rows are rebuilt
        # backward along that tangent. std is in DEMO MED-STEP units so one
        # number transfers across episodes/tasks.
        self.state_noise_std = float(state_noise_std)
        # Phase-space DART: Gaussian residual on the anchor's assumed
        # velocity around the demo tangent, med-steps PER TICK per arm dim
        # (cruise ~ 1). Applied only when a draw carries no velocity of its
        # own (constant-sigma and v1/v2 schedule paths); v3+ joint schedules
        # already sample a correlated velocity residual and win.
        self.vel_noise_std = float(vel_noise_std)
        self.state_noise_p = float(state_noise_p)
        self.state_noise_seed = state_noise_seed
        # Truncated sampling: reject-and-redraw DART position draws whose
        # vector norm is below min_sigma*sqrt(3) med-steps (the obs-jitter
        # shell) — the keep-the-label jitter augmentation already covers
        # sub-jitter perturbations with the opposite (ignore) lesson, so
        # every DART sample should be a meaningful correction example.
        # Units: med-steps per axis (0.55 = the 0.01 rad jitter's scale).
        self.state_noise_min_sigma = float(state_noise_min_sigma)
        # HOLD-TAIL MASKING (pair with --policy.do_mask_loss_for_padding=true):
        # instead of excluding every anchor whose chunk tail holds at the demo
        # end (which leaves the last horizon of frames without ANY noise
        # supervision), accept chunks with >= MIN_GENUINE moving steps and
        # mark the held tail in action_is_pad so the (masked) loss ignores it.
        # Reopens the endgame to noise anchors on every episode, including
        # short interventions whose valid window was previously empty.
        self.mask_hold_tail = bool(mask_hold_tail)
        self.MIN_GENUINE = 8
        self._noise_rng: np.random.Generator | None = None
        # Adaptive (blend-calibrated) noise schedule: JSON mapping
        # {source_repo_id: {source_ep: [sigma per demo frame]}} in med-step
        # units. When set, the per-anchor sigma is looked up at (source ep,
        # demo index) instead of the constant state_noise_std — the schedule
        # is measured from blend-rollout deviations (policy proposes the
        # amplitude, gaussian provides the support). state_noise_scale
        # multiplies the looked-up sigma (r=0.5 blends need scale 1.0 under
        # the lambda_g ~= lambda_pi assumption; see 2026-09-01 notes).
        self._noise_schedule: dict[int, np.ndarray] | None = None
        self._noise_schedule_path = str(state_noise_schedule or "")
        self.state_noise_scale = float(state_noise_scale)
        # Serve the ORIGINAL dataset item (real recorded action, untouched
        # obs, original index over the FULL frame range — hold zone included)
        # with this probability. Without it, a wrapped dataset contributes
        # ZERO genuine states/actions and ZERO goal-arrival supervision (the
        # anti-dawdle window excludes arrive-and-hold anchors by design):
        # measured 2026-08-26, dn base-finetunes collapsed 6-14 points vs
        # int-only while the per-round doses stayed neutral — prolonged
        # training on all-synthetic corrections with no endgame is harmful.
        # raw_mix restores the real data alongside the DART labels.
        self.raw_mix = float(raw_mix)
        # label cache: flat index -> synthesized float32 chunk (filled by the
        # window sweep; __getitem__ becomes a lookup — the dataloader-side
        # synthesis cost drops to ~zero).
        self._labels: dict[int, np.ndarray] = {}
        if n_arm is None:
            n_arm = max(1, int(dataset.meta.features["action"]["shape"][0]) - 1)
        self.n_arm = int(n_arm)
        if self.self_relabel:
            source_repo_id = source_repo_id or dataset.repo_id
            self.geoms = load_source_geometries(source_repo_id, n_arm=self.n_arm, root=root)
            self.ep_to_source = {ep: ep for ep in self.geoms}
            self.source_repo_id = source_repo_id
        else:
            self.ep_to_source, meta_source = self._episode_pairing(dataset)
            source_repo_id = source_repo_id or meta_source
            if source_repo_id is None:
                raise ValueError(
                    f"{dataset.root}: no source_repo_id given and episodes metadata lacks "
                    f"source_dataset_repo_id — re-record the blend or pass source_repo_id explicitly."
                )
            self.source_repo_id = source_repo_id
        self._fingerprinted = False
        if not self.self_relabel:
            self.geoms = load_source_geometries(source_repo_id, n_arm=self.n_arm, root=root)
        # State-noise consistency for robot-derived env_state dims: some
        # environments publish forward-kinematics-derived values (the planar
        # oracle state ends with the EE position) inside
        # observation.environment_state. Perturbing the joints without
        # updating those dims hands the policy an internally inconsistent
        # observation — and a shortcut to read the TRUE state and ignore the
        # noise. Detect the moving dims (within-episode ptp, same 0.005
        # convention as the blend script's static mask) and fit them as a
        # linear function of [cos(cumsum q), sin(cumsum q), 1] — exact for
        # planar FK — from the dataset's own frames; applied per obs row in
        # __getitem__. Disabled (with a warning) if the fit is poor.
        self._env_fk_w: np.ndarray | None = None
        self._env_fk_dims: np.ndarray | None = None
        if self._noise_active:
            self._fit_env_fk()
        self._valid = self._build_valid_window()

    def _ensure_schedule(self) -> None:
        """Load the schedule lazily (source_repo_id is set after __init__'s branch)."""
        if self._noise_schedule is not None or not self._noise_schedule_path:
            return
        import json as _json

        with open(self._noise_schedule_path) as _f:
            _sched_all = _json.load(_f)
        _sched = _sched_all.get(self.source_repo_id)
        if _sched is None:
            raise ValueError(
                f"{self._noise_schedule_path}: no entry for source repo {self.source_repo_id!r} "
                f"(has {list(_sched_all)})"
            )
        self._noise_schedule = {int(k): np.asarray(v, dtype=np.float64) for k, v in _sched.items()}

    def _sigma_at(self, source_ep: int, di: float) -> float:
        """Per-anchor scalar sigma (med-step units): schedule lookup or constant.

        For an anisotropic (v2) schedule this returns the RMS sigma —
        sqrt(trace Sigma) — used for window sizing; the actual draw uses
        the full covariance via :meth:`_noise_draw`.
        """
        self._ensure_schedule()
        if self._noise_schedule is None:
            return self.state_noise_std
        prof = self._noise_schedule.get(int(source_ep))
        if prof is None or len(prof) == 0:
            return self.state_noise_std if self.state_noise_std > 0 else 2.0
        i = int(np.clip(round(di), 0, len(prof) - 1))
        row = prof[i]
        if np.ndim(row) == 0:
            return float(row) * self.state_noise_scale
        if np.ndim(row) == 2 and np.shape(row)[-1] == 9:
            # v5 mixture-of-v4 (one component per chunk age): marginal RMS
            tot = float(
                np.mean(row[:, 0] ** 2 + row[:, 1] ** 2 + row[:, 2] ** 2 + row[:, 3] + row[:, 4] + row[:, 5])
            )
            return float(np.sqrt(max(tot, 1e-9))) * self.state_noise_scale
        if len(row) == 21:
            if self.n_arm == 6:
                # full 6x6 POSITION covariance (6-joint arms): diagonal at IU
                # indices 0, 6, 11, 15, 18, 20
                return (
                    float(np.sqrt(max(row[0] + row[6] + row[11] + row[15] + row[18] + row[20], 1e-9)))
                    * self.state_noise_scale
                )
            # v3 joint 6x6: position trace at IU indices 0, 6, 11
            return float(np.sqrt(max(row[0] + row[6] + row[11], 1e-9))) * self.state_noise_scale
        if len(row) == 9:
            # v4 one-sided: [mu(3), cov6 about mean]; sigma = sqrt total 2nd moment
            tot = row[0] ** 2 + row[1] ** 2 + row[2] ** 2 + row[3] + row[4] + row[5]
            return float(np.sqrt(max(tot, 1e-9))) * self.state_noise_scale
        # v2 row = [s11, s22, s33, s12, s13, s23] in med-step^2 units
        return float(np.sqrt(max(row[0] + row[1] + row[2], 1e-9))) * self.state_noise_scale

    _IU6 = [(i, j) for i in range(6) for j in range(i, 6)]

    def _noise_draw(self, source_ep: int, di: float, med_step: float):
        """One noise draw: (dq [rad], dv [rad/tick] or None), plus the
        phase-space velocity residual when ``vel_noise_std`` > 0 and the
        underlying draw carried none (constant-sigma / v1 / v2 paths).
        Drawn only on demand, so vel_noise_std=0 leaves the RNG stream --
        and therefore every historical position draw -- untouched."""
        dq, dv = self._noise_draw_pos(source_ep, di, med_step)
        if dv is None and self.vel_noise_std > 0.0:
            if self._noise_rng is None:
                self._noise_rng = np.random.default_rng(self.state_noise_seed)
            dv = self._noise_rng.normal(0.0, self.vel_noise_std * med_step, size=self.n_arm)
        return dq, dv

    def _noise_draw_pos(self, source_ep: int, di: float, med_step: float):
        """One position draw: returns (dq [rad], dv [rad/tick] or None).

        v1 scalar / v2 3x3 schedules sample position only (dv None); v3
        joint 6x6 schedules sample correlated (position, velocity residual).

        ``state_noise_min_sigma`` > 0 truncates the position draw's inner
        ball: draws whose position norm falls below min_sigma*sqrt(3)
        med-steps (the obs-jitter shell) are rejected and redrawn — every
        DART sample is then a meaningful correction example, never a
        sub-jitter perturbation the keep-the-label jitter augmentation
        already covers with the opposite (ignore) lesson.
        """
        if self.state_noise_min_sigma > 0:
            thresh = self.state_noise_min_sigma * math.sqrt(3.0) * med_step
            for _ in range(64):
                dq, dv = self._noise_draw_once(source_ep, di, med_step)
                if float(np.linalg.norm(dq[: min(3, self.n_arm)])) >= thresh:
                    return dq, dv
            # pathological sigma << threshold: project radially onto the shell
            n = float(np.linalg.norm(dq[: min(3, self.n_arm)]))
            if n > 1e-12:
                dq = dq * (thresh / n)
            return dq, dv
        return self._noise_draw_once(source_ep, di, med_step)

    def _noise_draw_once(self, source_ep: int, di: float, med_step: float):
        self._ensure_schedule()
        if self._noise_rng is None:
            self._noise_rng = np.random.default_rng(self.state_noise_seed)
        prof = None if self._noise_schedule is None else self._noise_schedule.get(int(source_ep))
        if prof is not None and len(prof) > 0 and np.ndim(prof[0]) == 2 and np.shape(prof[0])[-1] == 9:
            # v5 MIXTURE of v4 components, one per chunk age: Layer-1 sampled
            # literally (draw an execution age uniformly, then from that age's
            # Gaussian) instead of collapsing to the moment-matched Gaussian —
            # the mixture keeps the heavy tails the collapse loses.
            i = int(np.clip(round(di), 0, len(prof) - 1))
            comps = prof[i]
            r9 = comps[int(self._noise_rng.integers(len(comps)))]
            return self._v4_draw(r9, med_step)
        if prof is not None and len(prof) > 0 and np.ndim(prof[0]) > 0 and len(prof[0]) == 21:
            i = int(np.clip(round(di), 0, len(prof) - 1))
            sg = np.empty((6, 6))
            for (a, b), val in zip(self._IU6, prof[i], strict=True):
                sg[a, b] = sg[b, a] = float(val)
            sg = sg * (self.state_noise_scale * med_step) ** 2
            try:
                chol = np.linalg.cholesky(sg + 1e-12 * np.eye(6))
            except np.linalg.LinAlgError:
                dq = self._noise_rng.normal(0.0, self._sigma_at(source_ep, di) * med_step, size=self.n_arm)
                return dq, None
            z = chol @ self._noise_rng.standard_normal(6)
            if self.n_arm == 6:
                # 21 entries on a 6-joint arm = full position covariance, no
                # velocity component (the v3 pos3+vel3 layout is planar-only)
                return z, None
            dq = np.zeros(self.n_arm)
            dq[: min(3, self.n_arm)] = z[:3][: self.n_arm]
            return dq, z[3:]
        if prof is not None and len(prof) > 0 and np.ndim(prof[0]) > 0 and len(prof[0]) == 9:
            # v4 ONE-SIDED draw: mean shift toward the measured failure side
            # plus centered anisotropic scatter — no mirror-side symmetrization.
            i = int(np.clip(round(di), 0, len(prof) - 1))
            return self._v4_draw(prof[i], med_step)
        if prof is not None and len(prof) > 0 and np.ndim(prof[0]) > 0:
            i = int(np.clip(round(di), 0, len(prof) - 1))
            s11, s22, s33, s12, s13, s23 = (float(v) for v in prof[i])
            sg = np.array([[s11, s12, s13], [s12, s22, s23], [s13, s23, s33]])
            sg = sg * (self.state_noise_scale * med_step) ** 2
            try:
                chol = np.linalg.cholesky(sg + 1e-12 * np.eye(3))
            except np.linalg.LinAlgError:
                return self._noise_rng.normal(
                    0.0, self._sigma_at(source_ep, di) * med_step, size=self.n_arm
                ), None
            z = self._noise_rng.standard_normal(3)
            out = chol @ z
            if self.n_arm != 3:
                full = self._noise_rng.normal(0.0, self._sigma_at(source_ep, di) * med_step, size=self.n_arm)
                full[:3] = out[: min(3, self.n_arm)]
                return full, None
            return out, None
        return self._noise_rng.normal(0.0, self._sigma_at(source_ep, di) * med_step, size=self.n_arm), None

    def _v4_draw(self, row9, med_step: float):
        """Draw from one v4 component [mu(3), C11,C22,C33,C12,C13,C23]."""
        r9 = [float(v) for v in row9]
        mu = np.array(r9[:3]) * self.state_noise_scale * med_step
        c11, c22, c33, c12, c13, c23 = r9[3:]
        cg = np.array([[c11, c12, c13], [c12, c22, c23], [c13, c23, c33]])
        cg = cg * (self.state_noise_scale * med_step) ** 2
        try:
            chol = np.linalg.cholesky(cg + 1e-12 * np.eye(3))
            z = chol @ self._noise_rng.standard_normal(3)
        except np.linalg.LinAlgError:
            z = self._noise_rng.normal(0.0, np.sqrt(max(c11 + c22 + c33, 1e-9) / 3) * med_step, size=3)
        dq = np.zeros(self.n_arm)
        dq[: min(3, self.n_arm)] = (mu + z)[: self.n_arm]
        return dq, None

    @property
    def _noise_active(self) -> bool:
        return self.state_noise_std > 0.0 or bool(self._noise_schedule_path)

    def _env_fk_features(self, q: np.ndarray) -> np.ndarray:
        cs = np.cumsum(np.asarray(q, dtype=np.float64)[..., : self.n_arm], axis=-1)
        return np.concatenate([np.cos(cs), np.sin(cs), np.ones(cs.shape[:-1] + (1,))], axis=-1)

    def _fit_env_fk(self) -> None:
        import logging

        import pandas as pd

        if "observation.environment_state" not in getattr(self.dataset.meta, "features", {}):
            return
        files = sorted(glob.glob(os.path.join(str(self.dataset.root), "data/**/*.parquet"), recursive=True))
        cols = ["episode_index", "observation.state", "observation.environment_state"]
        df = pd.concat([pd.read_parquet(f, columns=cols) for f in files])
        env = np.stack(df["observation.environment_state"].to_numpy()).astype(np.float64)
        q = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
        ptp = np.median(
            np.stack(
                [
                    np.ptp(np.stack(g["observation.environment_state"].to_numpy()), axis=0)
                    for _, g in df.groupby("episode_index")
                ]
            ),
            axis=0,
        )
        moving = np.where(ptp > 0.005)[0]
        if len(moving) == 0:
            return
        feats = self._env_fk_features(q)
        w, *_ = np.linalg.lstsq(feats, env[:, moving], rcond=None)
        resid = np.abs(feats @ w - env[:, moving])
        rel = resid.max(axis=0) / np.maximum(np.ptp(env[:, moving], axis=0), 1e-9)
        if (rel > 0.02).any():
            logging.warning(
                "dart_relabel %s: robot-derived env dims %s but FK fit residual %.4f of range — "
                "state noise will leave env_state UNTOUCHED (inconsistent obs; consider disabling noise).",
                self.dataset.repo_id,
                moving.tolist(),
                float(rel.max()),
            )
            return
        self._env_fk_w = w
        self._env_fk_dims = moving
        logging.info(
            "dart_relabel %s: state noise will recompute robot-derived env dims %s "
            "(FK fit max residual %.5f, %.2f%% of range).",
            self.dataset.repo_id,
            moving.tolist(),
            float(resid.max()),
            100 * float(rel.max()),
        )

    def _cache_key(self, horizon: int) -> str:
        """Fingerprint for the persistent window/label cache.

        Any change to the label function source, the synthesis knobs, the
        horizon, the collision-filter settings, or the dataset content
        (frame count + newest data file mtime) produces a different key, so
        a stale cache can never be served — this is the safe version of
        "preprocess the dataset": disk-persisted, self-invalidating.
        """
        import hashlib
        import inspect

        files = sorted(glob.glob(os.path.join(str(self.dataset.root), "data/**/*.parquet"), recursive=True))
        newest = max((os.path.getmtime(f) for f in files), default=0.0)
        blob = "|".join(
            [
                inspect.getsource(chunk_labels),
                inspect.getsource(chunk_ok),
                f"h={horizon}",
                f"rate={self.rate}",
                f"ease={self.ease_out}",
                f"cf={self.collision_filter}",
                f"cm={self.collision_margin}",
                f"self={self.self_relabel}",
                f"n={len(self.dataset)}",
                f"eps={sorted(int(e) for e in getattr(self.dataset, 'episodes', None) or [])}",
                f"mtime={newest:.3f}",
                f"maskhold={self.mask_hold_tail}",
            ]
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def _build_valid_window(self) -> np.ndarray:
        """Flat indices whose label chunk is genuine motion end to end.

        Anchors near the demo's end synthesize chunks whose tail lands on
        the end and HOLDS; with an unmasked action-pad loss those frames
        train brake-early-and-sit (eval dawdling). Sampling is restricted to
        the exact set of anchors whose full-horizon chunk keeps moving,
        found by synthesizing every anchor once — and persisted to a
        self-invalidating disk cache so repeat launches skip the sweep.
        """
        import pandas as pd

        dts = getattr(self.dataset, "delta_timestamps", None) or {}
        horizon = len(dts.get("action", [])) or 64
        key = self._cache_key(horizon)
        cache_path = os.path.join(str(self.dataset.root), f"dart_label_cache_{key}.npz")
        if os.path.exists(cache_path):
            try:
                z = np.load(cache_path)
                if str(z["key"]) == key:
                    valid_cached = z["valid"].astype(np.int64)
                    labels_arr = z["labels"]
                    self._labels = {int(i): labels_arr[k] for k, i in enumerate(valid_cached)}
                    return valid_cached
            except Exception as e:  # unreadable/corrupt cache -> rebuild
                import logging

                logging.warning("dart_relabel: ignoring unreadable label cache %s (%s)", cache_path, e)
        files = sorted(glob.glob(os.path.join(str(self.dataset.root), "data/**/*.parquet"), recursive=True))
        cols = ["index", "episode_index", "frame_index", "observation.state"]
        has_di = "relabel_demo_index" in self.dataset.meta.features
        if has_di:
            cols.append("relabel_demo_index")
        elif not self.self_relabel:
            raise ValueError(f"{self.dataset.root}: no relabel_demo_index and self_relabel is off")
        has_vel = "relabel_velocity" in self.dataset.meta.features
        if has_vel:
            cols.append("relabel_velocity")
        has_coll = "frame_in_collision" in self.dataset.meta.features
        if has_coll:
            cols.append("frame_in_collision")
        elif self.collision_filter != "none":
            import logging

            logging.warning(
                "dart_relabel %s: collision_filter=%s requested but the dataset has no "
                "frame_in_collision column (recorded pre-feature) — no collision filtering applied.",
                self.dataset.repo_id,
                self.collision_filter,
            )
        df = pd.concat([pd.read_parquet(f, columns=cols) for f in files])
        # Episode-subsetted datasets (dataset.episodes set, e.g. the scarcity
        # fraction arms) serve items POSITIONALLY over the selected episodes,
        # while this sweep reads the full on-disk parquets whose `index`
        # column is the full-dataset global index. Filter to the subset and
        # remap `index` to subset positions, or the served indices run past
        # the wrapped dataset (observed: IndexError 4176 >= 2949 on f50_dn4).
        _eps_sel = getattr(self.dataset, "episodes", None)
        if _eps_sel is not None:
            df = df[df["episode_index"].isin({int(e) for e in _eps_sel})]
            df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
            df["index"] = np.arange(len(df), dtype=np.int64)
        valid: list[np.ndarray] = []
        for ep, g in df.groupby("episode_index"):
            src = self.ep_to_source.get(int(ep))
            if src is None or src not in self.geoms:
                continue
            geom = self.geoms[src]
            g = g.sort_values("frame_index")
            states_ep = np.stack(g["observation.state"].to_numpy()).astype(np.float64)[:, : self.n_arm]
            if has_di:
                dis_ep = np.array([float(np.reshape(v, -1)[-1]) for v in g["relabel_demo_index"].to_numpy()])
            else:
                # self-relabel: the state at frame t IS the polyline's point t.
                dis_ep = g["frame_index"].to_numpy().astype(np.float64)
            vels_ep = (
                np.stack(
                    [
                        np.reshape(np.asarray(v, dtype=np.float64), -1)[: self.n_arm]
                        for v in g["relabel_velocity"].to_numpy()
                    ]
                )
                if has_vel
                else None
            )

            # Loader-side collision filtering (mirrors the legacy replay
            # filter's semantics, non-destructively): "drop" excludes the
            # whole episode when any frame collided; "trim_first_collision"
            # excludes frames from (first collision - margin) onward.
            n_ep = len(states_ep)
            allowed = np.ones(n_ep, dtype=bool)
            if has_coll and self.collision_filter != "none":
                coll = np.array(
                    [
                        float(np.reshape(np.asarray(v), -1)[0]) > 0.5
                        for v in g["frame_in_collision"].to_numpy()
                    ]
                )
                if coll.any():
                    if self.collision_filter == "drop":
                        allowed[:] = False
                    else:
                        first = int(np.argmax(coll))
                        allowed[max(0, first - self.collision_margin) :] = False
            # exact per-frame sweep: hold-ness is only NEAR-monotone in the
            # frame index (velocity variation), so evaluate every anchor —
            # ~1 ms each, a one-time init cost of seconds per dataset. The
            # synthesized chunks are KEPT (self._labels) so training-time
            # __getitem__ is a lookup, not a synthesis.
            flat = g["index"].to_numpy().astype(np.int64)
            keep = np.zeros(n_ep, dtype=bool)
            for t in range(n_ep):
                if not allowed[t]:
                    continue
                info: dict = {}
                labels = chunk_labels(
                    states_ep[t],
                    dis_ep[t],
                    geom,
                    horizon=horizon,
                    rate=self.rate,
                    ease_out=self.ease_out,
                    prev_state=states_ep[t - 1] if t > 0 else None,
                    velocity=vels_ep[t] if vels_ep is not None else None,
                    info=info,
                )
                _ok, _reason = chunk_ok(labels, info, geom)
                if _ok or (
                    self.mask_hold_tail
                    and _reason == "hold"
                    and self._hold_start(labels[:, : self.n_arm]) >= self.MIN_GENUINE
                ):
                    keep[t] = True
                    self._labels[int(flat[t])] = labels.astype(np.float32)
            valid.append(flat[keep])
        if not valid:
            import logging

            logging.warning(
                "dart_relabel %s: no hold-free anchors found — sampling the full range.",
                self.dataset.repo_id,
            )
            return np.arange(len(self.dataset), dtype=np.int64)
        out = np.concatenate(valid)
        try:
            np.savez_compressed(
                cache_path,
                key=key,
                valid=out,
                labels=np.stack([self._labels[int(i)] for i in out]),
            )
            # Superseded caches (older fingerprints — e.g. from before a
            # resume-extension appended more samples_per_episode blends) are
            # never read again; delete them so extended datasets don't
            # accumulate ~MBs of stale label archives per regeneration.
            import contextlib

            for stale in glob.glob(os.path.join(str(self.dataset.root), "dart_label_cache_*.npz")):
                if os.path.abspath(stale) != os.path.abspath(cache_path):
                    with contextlib.suppress(OSError):
                        os.remove(stale)
        except Exception:
            import logging

            logging.warning(
                "dart_relabel %s: could not write label cache %s", self.dataset.repo_id, cache_path
            )
        return out

    @staticmethod
    def _episode_pairing(dataset) -> tuple[dict[int, int], str | None]:
        import pandas as pd

        files = sorted(
            glob.glob(os.path.join(str(dataset.root), "meta/episodes/**/*.parquet"), recursive=True)
        )
        if not files:
            raise ValueError(f"{dataset.root}: no episodes metadata parquet found")
        m = pd.concat([pd.read_parquet(f) for f in files])
        if "source_episode_idx" not in m.columns:
            raise ValueError(f"{dataset.root}: episodes metadata lacks source_episode_idx pairing")
        source = None
        if "source_dataset_repo_id" in m.columns:
            vals = m["source_dataset_repo_id"].dropna().unique()
            if len(vals) > 1:
                raise ValueError(f"{dataset.root}: multiple source_dataset_repo_id values: {vals}")
            if len(vals) == 1:
                source = str(vals[0])
        return {int(r.episode_index): int(r.source_episode_idx) for r in m.itertuples()}, source

    def __len__(self) -> int:
        """Length of the wrapped dataset."""
        return len(self.dataset)

    def __getattr__(self, name):
        """Delegate meta/stats/etc. so the wrapper is a drop-in for training."""
        if name == "dataset":
            # Only reachable when self.dataset is not yet set (e.g. during
            # unpickling in a spawn-context DataLoader worker, before
            # __dict__ is restored) — delegating would recurse forever.
            raise AttributeError(name)
        return getattr(self.dataset, name)

    def __getitem__(self, idx: int) -> dict:
        """Return the wrapped item with its action chunk replaced by DART labels."""
        import torch

        # Sample only from the precomputed hold-free window: indices map
        # uniformly onto valid anchors (len() is unchanged so multi-dataset
        # cumulative sizes stay correct).
        if self.raw_mix > 0.0:
            if self._noise_rng is None:
                self._noise_rng = np.random.default_rng(self.state_noise_seed)
            if self._noise_rng.random() < self.raw_mix:
                # genuine sample: original index (full range incl. the demo
                # endgame), original action, untouched observations.
                return self.dataset[idx]
        if len(self._valid) and len(self._valid) < len(self.dataset):
            idx = int(self._valid[idx % len(self._valid)])
        item = self.dataset[idx]
        action = item["action"]
        geom = self.geoms[self.ep_to_source[int(item["episode_index"])]]
        if "relabel_demo_index" in item:
            _di_raw = item["relabel_demo_index"]
            di = (
                float(_di_raw.reshape(-1)[-1])
                if isinstance(_di_raw, torch.Tensor)
                else float(np.reshape(_di_raw, -1)[-1])
            )
        else:
            # self-relabel: this frame IS point frame_index of its own demo.
            _fi = item["frame_index"]
            di = float(_fi.reshape(-1)[-1]) if isinstance(_fi, torch.Tensor) else float(_fi)
        # ── classic-DART state noise: perturb the anchor, re-project, and
        # synthesize the label from the perturbed state ──
        noised = False
        noise_vel: np.ndarray | None = None
        delta = _dvel = None  # set only when a valid (end-guarded) draw lands
        if self._noise_active:
            if self._noise_rng is None:
                self._noise_rng = np.random.default_rng(self.state_noise_seed)
            if self._noise_rng.random() < self.state_noise_p:
                _ep_raw = item["episode_index"]
                _ep_idx = int(
                    np.reshape(np.asarray(_ep_raw.cpu() if hasattr(_ep_raw, "cpu") else _ep_raw), -1)[-1]
                )
                _src_ep = self.ep_to_source.get(_ep_idx, _ep_idx)
                _std_here = self._sigma_at(_src_ep, di)
                # END-GUARD RESAMPLE: a draw whose along-track component
                # shifts the projected index near the demo end would
                # synthesize a hold-tail chunk (brake-early-and-sit
                # supervision) that the valid window only screened for
                # UNPERTURBED anchors. Reject draws whose projected index
                # leaves < horizon of genuine demo ahead; after a few
                # retries fall back to serving the anchor unperturbed
                # (its cached label is valid by construction).
                _di0 = di
                _end_free = float(self.MIN_GENUINE if self.mask_hold_tail else 64)
                _di_max = max(0.0, float(len(geom.P) - 1) - _end_free)
                for _try in range(6):
                    _delta, _dv = self._noise_draw(_src_ep, _di0, geom.med_step)
                    _q_try = item["observation.state"]
                    _q_try = (_q_try[-1] if _q_try.dim() == 2 else _q_try).cpu().numpy().copy()
                    _q_try[: self.n_arm] += np.asarray(_delta)[: self.n_arm]
                    _di_try = project_state_local(_q_try, geom, _di0, window_steps=3.0 * _std_here + 4.0)
                    if _di_try <= _di_max or _di0 > _di_max:
                        delta, _dvel, di = _delta, _dv, _di_try
                        break
            if delta is not None:
                state_t = item["observation.state"].clone()
                d_t = torch.as_tensor(delta, dtype=state_t.dtype, device=state_t.device)
                if state_t.dim() == 2:
                    state_t[:, : self.n_arm] += d_t
                else:
                    state_t[: self.n_arm] += d_t
                # A random offset has no velocity of its own, so assume the
                # perturbed anchor was moving PARALLEL TO THE DEMO at the
                # progress-aligned index: velocity = the demo tangent at the
                # projected di (per tick). Rebuild the earlier obs-history
                # rows backward along that tangent (a plain shared shift
                # would carry frame t's velocity while the label launches at
                # di), and hand the same tangent to the label synthesis.
                _hi = min(di + 1.0, float(len(geom.P) - 1))
                _lo = max(di - 1.0, 0.0)
                noise_vel = (
                    (_interp_rows(geom.P, _hi) - _interp_rows(geom.P, _lo)) / max(_hi - _lo, 1e-9)
                ).astype(np.float64)
                if _dvel is not None:
                    # joint (v3) schedule: sampled velocity residual widens the
                    # velocity distribution around the demo tangent
                    noise_vel = noise_vel + np.asarray(_dvel[: len(noise_vel)], dtype=np.float64)
                if state_t.dim() == 2 and state_t.shape[0] >= 2:
                    v_t = torch.as_tensor(noise_vel, dtype=state_t.dtype, device=state_t.device)
                    n_rows = state_t.shape[0]
                    for k in range(n_rows - 1):
                        state_t[k, : self.n_arm] = state_t[-1, : self.n_arm] - (n_rows - 1 - k) * v_t
                item["observation.state"] = state_t
                if self._env_fk_w is not None and "observation.environment_state" in item:
                    env_t = item["observation.environment_state"].clone()
                    qs = (state_t if state_t.dim() == 2 else state_t[None])[:, : self.n_arm].cpu().numpy()
                    pred = self._env_fk_features(qs) @ self._env_fk_w  # (rows, n_moving)
                    pv = torch.as_tensor(pred, dtype=env_t.dtype, device=env_t.device)
                    dims = torch.as_tensor(self._env_fk_dims, device=env_t.device)
                    if env_t.dim() == 2 and env_t.shape[0] == qs.shape[0]:
                        env_t[:, dims] = pv
                    elif env_t.dim() == 1:
                        env_t[dims] = pv[-1]
                    item["observation.environment_state"] = env_t
                noised = True
        cached = None if noised else self._labels.get(int(idx))
        if cached is not None and cached.shape[0] == action.shape[0] and self._fingerprinted:
            item["action"] = torch.as_tensor(
                cached[:, : action.shape[1]], dtype=action.dtype, device=action.device
            )
            if self.mask_hold_tail:
                _lab = self._mask_and_dehold(np.asarray(cached, dtype=np.float64), item, geom)
                item["action"] = torch.as_tensor(
                    _lab[:, : action.shape[1]], dtype=action.dtype, device=action.device
                )
            return item
        if not self._fingerprinted:
            # One-time (per process) positive evidence that relabeling is
            # LIVE: the synthesized chunk must differ from the stored
            # executed actions on real blend data. Grep training logs for
            # "dart labels ACTIVE".
            self._fingerprinted = True
            import numpy as _np

            _stored = action.detach().cpu().numpy()
            _q = (
                (
                    item["observation.state"][-1]
                    if item["observation.state"].dim() == 2
                    else item["observation.state"]
                )
                .cpu()
                .numpy()
            )
            _lab = chunk_labels(
                _q, di, geom, horizon=_stored.shape[0], rate=self.rate, ease_out=self.ease_out
            )
            _diff = float(_np.abs(_lab[:, : _stored.shape[1]] - _stored).mean())
            import logging as _logging

            _logging.info(
                "dart labels ACTIVE for %s: first sampled chunk |label - stored_action| mean %.4f rad "
                "(0.0000 would mean relabeling is inert — investigate)",
                self.dataset.repo_id,
                _diff,
            )
        state = item["observation.state"]
        q = state[-1] if state.dim() == 2 else state  # last obs step conditions the chunk
        prev = state[0].cpu().numpy() if (state.dim() == 2 and state.shape[0] >= 2) else None
        vel = item.get("relabel_velocity")
        if vel is not None:
            vel = vel.cpu().numpy() if isinstance(vel, torch.Tensor) else np.asarray(vel)
            vel = vel[-1] if vel.ndim == 2 else vel  # last row when delta-stacked
        if noise_vel is not None:
            vel = noise_vel  # demo tangent at the projected index (see noise branch)
        info: dict = {}
        labels = chunk_labels(
            q.cpu().numpy(),
            di,
            geom,
            horizon=action.shape[0],
            rate=self.rate,
            ease_out=self.ease_out,
            prev_state=prev,
            velocity=vel,
            info=info,
        )
        # NO ARRIVE-AND-HOLD SAMPLES: an anchor close to the demo's end gets
        # a chunk whose tail lands on the end and HOLDS (land-at-rest, or the
        # fill clamping at the last index). With the diffusion horizon at 64
        # and do_mask_loss_for_padding=False those hold frames TRAIN the
        # policy to brake early and sit near the goal (observed as eval
        # dawdling: failures that creep close and time out). Instead of
        # padding/holding, redirect the sample to an earlier anchor in the
        # SAME episode until the whole chunk is genuine motion; the endgame
        # stays supervised by the base/intervention data.
        item["action"] = torch.as_tensor(
            labels[:, : action.shape[1]], dtype=action.dtype, device=action.device
        )
        if self.mask_hold_tail:
            _lab = self._mask_and_dehold(np.asarray(labels, dtype=np.float64), item, geom)
            item["action"] = torch.as_tensor(
                _lab[:, : action.shape[1]], dtype=action.dtype, device=action.device
            )
        return item

    def _mask_and_dehold(self, labels_np, item, geom):
        """mask_hold_tail serving: mark the synthesized hold tail in
        action_is_pad and REPLACE the repeated end-position values with a
        constant-velocity continuation along the last genuine step — served
        labels never contain a stop pattern (Jenny 2026-09-03: with the loss
        mask on, repeats are dead weight at best and brake-early leakage at
        worst). Returns possibly-rewritten labels (numpy, full width)."""
        hs = self._hold_start(labels_np[:, : self.n_arm])
        if hs >= labels_np.shape[0]:
            return labels_np
        if "action_is_pad" in item:
            item["action_is_pad"] = item["action_is_pad"].clone()
            item["action_is_pad"][hs:] = True
        # canonical short-chunk semantics (Jenny): the label's REAL steps end
        # at the demo's final index; the remainder is ordinary padding —
        # values repeat the last real step (the standard pad filler) and the
        # mask covers them. No fabricated values, no stop-teaching (masked).
        return labels_np

    @staticmethod
    def _hold_start(labels: np.ndarray) -> int:
        """Index of the first label of the terminal zero-motion run (== len(labels) if none)."""
        d = np.linalg.norm(np.diff(labels, axis=0), axis=1)
        k = len(labels)
        while k >= 2 and d[k - 2] < 1e-9:
            k -= 1
        return k

    @staticmethod
    def _chunk_holds(labels: np.ndarray, info: dict) -> bool:
        """True when the chunk's tail stops moving (arrive-and-hold at the demo end)."""
        if info.get("end_clamped"):
            return True
        return bool(np.linalg.norm(labels[-1] - labels[-2]) < 1e-9)


def chunk_ok(labels: np.ndarray, info: dict, geom: DemoGeometry) -> tuple[bool, str]:
    """Single validity predicate for a synthesized chunk — shared by the
    train-time sampling window and the QA certificate, so training serves
    exactly what QA certifies.

    Rejects (reason string names the cause):
      * ``hold``  — the tail stops moving (demo-end hold zone); with an
        unmasked pad loss these frames teach brake-early-and-sit;
      * ``accel`` — a label-internal accel spike above the demo envelope
        (2*sqrt(n)*acc_p95, floored at 0.2*med_step ~ the env's junction
        noise scale) — boundary-geometry artifacts (partial fill steps at a
        clamped end, fast-anchor-on-slow-demo splices).
    """
    if info.get("end_clamped") or bool(np.linalg.norm(labels[-1] - labels[-2]) < 1e-9):
        return False, "hold"
    n = geom.n_arm
    a_lim = max(2.0 * float(np.sqrt(n)) * geom.acc_p95, 0.2 * geom.med_step)
    amax = float(np.linalg.norm(np.diff(labels[:, :n], n=2, axis=0), axis=1).max())
    if amax > a_lim + 1e-9:
        return False, "accel"
    return True, "ok"


def maybe_wrap_dart(
    dataset,
    root: str | None = None,
    rate: float = 1.0,
    ease_out: float = 0.3,
    collision_filter: str = "none",
    collision_margin: int = 10,
    self_relabel_pattern: str = "",
    state_noise_std: float = 0.0,
    vel_noise_std: float = 0.0,
    state_noise_p: float = 1.0,
    raw_mix: float = 0.0,
    state_noise_schedule: str = "",
    state_noise_scale: float = 1.0,
    state_noise_min_sigma: float = 0.0,
    mask_hold_tail: bool = False,
):
    """Wrap ``dataset`` in DartChunkDataset when it qualifies.

    Two qualification routes:
      * it carries ``relabel_demo_index`` (a relabeled BLEND dataset —
        the historical route for ``--dataset.dart_relabel=true``), or
      * ``self_relabel_pattern`` is set and matches ``dataset.repo_id``
        ("base DART": the dataset's own episodes serve as their own demos —
        intervention datasets need no relabel columns and no blend rollouts).

    ``state_noise_std`` (demo med-step units) adds classic-DART train-time
    state noise to every wrapped dataset: perturb the anchor state, locally
    re-project, synthesize the recovery label from the perturbed state.
    Every other dataset is returned unchanged, so the flags are safe to set
    globally in mixed (raw + blend + intervention) multi-source training.
    """
    import logging
    import re as _re

    has_col = "relabel_demo_index" in getattr(dataset.meta, "features", {})
    self_match = bool(self_relabel_pattern) and bool(
        _re.search(self_relabel_pattern, getattr(dataset, "repo_id", "") or "")
    )
    if not has_col and not self_match:
        return dataset
    wrapped = DartChunkDataset(
        dataset,
        root=root,
        rate=rate,
        ease_out=ease_out,
        collision_filter=collision_filter,
        collision_margin=collision_margin,
        self_relabel=not has_col,
        state_noise_std=state_noise_std,
        vel_noise_std=vel_noise_std,
        state_noise_p=state_noise_p,
        raw_mix=raw_mix,
        state_noise_schedule=state_noise_schedule,
        state_noise_scale=state_noise_scale,
        state_noise_min_sigma=state_noise_min_sigma,
        mask_hold_tail=mask_hold_tail,
    )
    logging.info(
        "dart_relabel: wrapping %s with DART chunk labels (source %s, n_arm=%d, "
        "collision_filter=%s, sampling window %d/%d anchors, label cache %d chunks)",
        dataset.repo_id,
        wrapped.source_repo_id,
        wrapped.n_arm,
        collision_filter,
        len(wrapped._valid),
        len(dataset),
        len(wrapped._labels),
    )
    return wrapped


class ClearPadFlags(torch.utils.data.Dataset):
    """Passthrough that clears action_is_pad — pairs with
    do_mask_loss_for_padding=true to KEEP training a dataset's copy-padded
    terminal steps (repeat-the-goal-pose settle supervision) while wrapped
    intervention datasets carry their synthesized-hold masks."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __getattr__(self, name):
        if name == "dataset":
            # unpickling creates the instance without __init__; without this
            # guard the lookup of self.dataset recurses forever
            raise AttributeError(name)
        return getattr(self.dataset, name)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if "action_is_pad" in item:
            item["action_is_pad"] = torch.zeros_like(item["action_is_pad"])
        return item
