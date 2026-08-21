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
    return DemoGeometry(P=pts, A=acts, seg=seg, seg_len=seg_len, cum=cum, med_step=med_step)


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
    # HERMITE MERGE (heading-continuous at BOTH ends): when the state's own
    # velocity is known and the offset is non-trivial, the rejoin is a cubic
    # Hermite from (q, v0) to (demo action at a budget-chosen rendezvous
    # index, demo velocity there). The launch direction IS the robot's
    # current motion — no perpendicular departure from the blended path —
    # and the landing is tangent to the demo. Rendezvous time T comes from
    # the speed budget (chord/T <= B), the rendezvous index from the
    # tangential distance the budget affords: di = sqrt((T*B)^2 - d0^2).
    if (velocity is not None or prev_state is not None) and d0 > 0.5 * geom.med_step:
        # Launch tangent: prefer an externally SMOOTHED velocity (the
        # recorded relabel_velocity annotation, or a windowed estimate from
        # a full track) — a single-tick finite difference on a noisy blend
        # path can point anywhere, and a Hermite launched along that noise
        # looks perpendicular again (user diagnosis, 2026-08-21).
        if velocity is not None:
            v0 = np.asarray(velocity, dtype=np.float64)[: geom.n_arm].copy()
        else:
            v0 = (
                np.asarray(state, dtype=np.float64)[: geom.n_arm]
                - np.asarray(prev_state, dtype=np.float64)[: geom.n_arm]
            )
        sp0 = float(np.linalg.norm(v0))
        if sp0 < 0.2 * geom.med_step:
            # stationary robot: launch along the corridor tangent instead.
            v0 = _interp_rows(geom.P, min(demo_index + 1.0, end_i)) - proj0
            sp0 = float(np.linalg.norm(v0)) or 1e-9
        v0 = v0 * (min(sp0, b) / sp0)
        labels = np.empty((horizon, geom.A.shape[1]), dtype=np.float64)
        # Rendezvous time from the GLIDE RATIO (tangential:normal >= glide:1,
        # chord angle <= atan(1/glide) ~ 27 deg at 2.0): chord = d0*sqrt(1+g^2),
        # T = chord/B. The earlier correction-dominated choice (T = d0/0.8B)
        # made 53-deg chords that rendered as perpendicular spokes.
        g = max(0.5, float(glide_ratio))
        t_merge = max(2, int(np.ceil(d0 * float(np.sqrt(1.0 + g * g)) / b)))
        for _ in range(3):  # speed-check bumps
            di = float(np.sqrt(max(0.0, (t_merge * b) ** 2 - d0 * d0))) / geom.med_step
            i_r = min(end_i, float(demo_index) + di)
            a_r = _interp_rows(geom.A, i_r)
            tan_r = (
                _interp_rows(geom.A, min(i_r + 1.0, end_i)) - _interp_rows(geom.A, max(i_r - 1.0, 0.0))
            ) / 2.0
            pts = []
            ok = True
            prev_pt = np.asarray(state, dtype=np.float64)[: geom.A.shape[1]].copy()
            for k in range(min(t_merge, horizon)):
                t01 = (k + 1) / t_merge
                h00 = 2 * t01**3 - 3 * t01**2 + 1
                h10 = t01**3 - 2 * t01**2 + t01
                h01 = -2 * t01**3 + 3 * t01**2
                h11 = t01**3 - t01**2
                pt = a_r.copy()
                pt[: geom.n_arm] = (
                    h00 * q
                    + h10 * (v0 * t_merge)
                    + h01 * a_r[: geom.n_arm]
                    + h11 * (tan_r[: geom.n_arm] * t_merge)
                )
                # non-arm dims (gripper): linear from the current demo action.
                if geom.A.shape[1] > geom.n_arm:
                    a0_row = _interp_rows(geom.A, min(float(demo_index), end_i))
                    pt[geom.n_arm :] = (1 - t01) * a0_row[geom.n_arm :] + t01 * a_r[geom.n_arm :]
                if np.linalg.norm(pt[: geom.n_arm] - prev_pt[: geom.n_arm]) > 1.4 * b:
                    ok = False
                    break
                pts.append(pt)
                prev_pt = pt
            if ok or horizon - 1 <= t_merge:
                break
            t_merge = min(horizon - 1, int(np.ceil(t_merge * 1.5)))
        for k in range(horizon):
            if k < len(pts):
                labels[k] = pts[k]
            else:
                labels[k] = _interp_rows(geom.A, min(i_r + (k - len(pts) + 1), end_i))
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
        labels = chunk_labels(
            q.cpu().numpy(),
            di,
            geom,
            horizon=action.shape[0],
            rate=self.rate,
            ease_out=self.ease_out,
            prev_state=prev,
            velocity=vel,
        )
        item["action"] = torch.as_tensor(
            labels[:, : action.shape[1]], dtype=action.dtype, device=action.device
        )
        return item


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
