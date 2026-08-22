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
import os
from dataclasses import dataclass

import numpy as np


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


def _interp_rows(mat: np.ndarray, i: float) -> np.ndarray:
    """Linear interpolation of row ``i`` (float, clamped) of matrix ``mat``."""
    idx = int(np.clip(np.floor(i), 0, len(mat) - 1))
    nxt = min(idx + 1, len(mat) - 1)
    f = float(np.clip(i - idx, 0.0, 1.0))
    return (1.0 - f) * mat[idx] + f * mat[nxt]


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
        kp = (4.7 / 45.0) ** 2  # critically damped, ~45-tick settle
        kd = 2.0 * float(np.sqrt(kp))
        a_clamp = 0.9 * max(2.0 * float(np.sqrt(geom.n_arm)) * geom.acc_p95, 0.2 * geom.med_step)
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
            a_cmd = kp * (p_ref - x) + kd * (v_ref - v)
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
        if n_arm is None:
            n_arm = max(1, int(dataset.meta.features["action"]["shape"][0]) - 1)
        self.n_arm = int(n_arm)
        self.ep_to_source, meta_source = self._episode_pairing(dataset)
        source_repo_id = source_repo_id or meta_source
        if source_repo_id is None:
            raise ValueError(
                f"{dataset.root}: no source_repo_id given and episodes metadata lacks "
                f"source_dataset_repo_id — re-record the blend or pass source_repo_id explicitly."
            )
        self.source_repo_id = source_repo_id
        self._fingerprinted = False
        self.geoms = load_source_geometries(source_repo_id, n_arm=self.n_arm, root=root)
        self._valid = self._build_valid_window()

    def _build_valid_window(self) -> np.ndarray:
        """Flat indices whose label chunk is genuine motion end to end.

        Anchors near the demo's end synthesize chunks whose tail lands on
        the end and HOLDS; with an unmasked action-pad loss those frames
        train brake-early-and-sit (eval dawdling). Sampling is restricted to
        the exact set of anchors whose full-horizon chunk keeps moving,
        found by synthesizing every anchor once at init (~1 ms each).
        """
        import pandas as pd

        dts = getattr(self.dataset, "delta_timestamps", None) or {}
        horizon = len(dts.get("action", [])) or 64
        files = sorted(glob.glob(os.path.join(str(self.dataset.root), "data/**/*.parquet"), recursive=True))
        cols = ["index", "episode_index", "frame_index", "observation.state", "relabel_demo_index"]
        has_vel = "relabel_velocity" in self.dataset.meta.features
        if has_vel:
            cols.append("relabel_velocity")
        df = pd.concat([pd.read_parquet(f, columns=cols) for f in files])
        valid: list[np.ndarray] = []
        for ep, g in df.groupby("episode_index"):
            src = self.ep_to_source.get(int(ep))
            if src is None or src not in self.geoms:
                continue
            geom = self.geoms[src]
            g = g.sort_values("frame_index")
            states_ep = np.stack(g["observation.state"].to_numpy()).astype(np.float64)[:, : self.n_arm]
            dis_ep = np.array([float(np.reshape(v, -1)[-1]) for v in g["relabel_demo_index"].to_numpy()])
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

            def _holds_at(t, states_ep=states_ep, dis_ep=dis_ep, vels_ep=vels_ep, geom=geom) -> bool:
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
                return not chunk_ok(labels, info, geom)[0]

            # exact per-frame sweep: hold-ness is only NEAR-monotone in the
            # frame index (velocity variation), so evaluate every anchor —
            # ~1 ms each, a one-time init cost of seconds per dataset.
            keep = np.array([not _holds_at(t) for t in range(len(states_ep))])
            valid.append(g["index"].to_numpy()[keep].astype(np.int64))
        if not valid:
            import logging

            logging.warning(
                "dart_relabel %s: no hold-free anchors found — sampling the full range.",
                self.dataset.repo_id,
            )
            return np.arange(len(self.dataset), dtype=np.int64)
        return np.concatenate(valid)

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
        if len(self._valid) and len(self._valid) < len(self.dataset):
            idx = int(self._valid[idx % len(self._valid)])
        item = self.dataset[idx]
        action = item["action"]
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
            _di = item["relabel_demo_index"]
            _di = (
                float(_di.reshape(-1)[-1])
                if isinstance(_di, torch.Tensor)
                else float(_np.reshape(_di, -1)[-1])
            )
            _geom = self.geoms[self.ep_to_source[int(item["episode_index"])]]
            _lab = chunk_labels(
                _q, _di, _geom, horizon=_stored.shape[0], rate=self.rate, ease_out=self.ease_out
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
        di = item["relabel_demo_index"]
        di = float(di.reshape(-1)[-1]) if isinstance(di, torch.Tensor) else float(np.reshape(di, -1)[-1])
        geom = self.geoms[self.ep_to_source[int(item["episode_index"])]]
        prev = state[0].cpu().numpy() if (state.dim() == 2 and state.shape[0] >= 2) else None
        vel = item.get("relabel_velocity")
        if vel is not None:
            vel = vel.cpu().numpy() if isinstance(vel, torch.Tensor) else np.asarray(vel)
            vel = vel[-1] if vel.ndim == 2 else vel  # last row when delta-stacked
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
        return item

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


def maybe_wrap_dart(dataset, root: str | None = None, rate: float = 1.0, ease_out: float = 0.3):
    """Wrap ``dataset`` in DartChunkDataset iff it carries ``relabel_demo_index``.

    The factory-side entry point for ``--dataset.dart_relabel=true``: relabeled
    blend datasets (recorded with ``--relabel_actions=guidance``) get their
    action chunks replaced by synthesized DART labels; every other dataset is
    returned unchanged, so the flag is safe to set globally in mixed
    (raw + blend) multi-source training.
    """
    import logging

    if "relabel_demo_index" not in getattr(dataset.meta, "features", {}):
        return dataset
    wrapped = DartChunkDataset(dataset, root=root, rate=rate, ease_out=ease_out)
    logging.info(
        "dart_relabel: wrapping %s with DART chunk labels (source %s, n_arm=%d, rate=%.2f, ease_out=%.2f)",
        dataset.repo_id,
        wrapped.source_repo_id,
        wrapped.n_arm,
        rate,
        ease_out,
    )
    return wrapped
