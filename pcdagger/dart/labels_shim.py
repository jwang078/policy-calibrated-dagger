"""DART label synthesis for blend datasets — record thin, synthesize on demand.

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

    i_k      = min(i0 + k, end)                    # demo clock: 1 index/tick
    d_k      = max(0, d0 - (k+1)*rate*med_step)    # offset closes at cruise
    label_k  = demo_action(i_k) + d_k * u          # u = offset direction

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
    P = np.asarray(demo_states_raw, dtype=np.float64)[:, :n_arm]
    A = np.asarray(demo_actions_raw, dtype=np.float64)
    seg = np.diff(P, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    live = seg_len[seg_len > 1e-9]
    med_step = float(np.median(live)) if len(live) else 1e-3
    return DemoGeometry(P=P, A=A, seg=seg, seg_len=seg_len, cum=cum, med_step=med_step)


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
    S = np.asarray(states, dtype=np.float64)[:, : geom.n_arm]
    arc_window = max(1, int(index_window)) * geom.med_step
    s_prev = 0.0
    out = np.empty(len(S), dtype=np.float64)
    for t, q in enumerate(S):
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


def _interp_rows(M: np.ndarray, i: float) -> np.ndarray:
    """Linear interpolation of row ``i`` (float, clamped) of matrix M."""
    idx = int(np.clip(np.floor(i), 0, len(M) - 1))
    nxt = min(idx + 1, len(M) - 1)
    f = float(np.clip(i - idx, 0.0, 1.0))
    return (1.0 - f) * M[idx] + f * M[nxt]


def chunk_labels(
    state: np.ndarray,
    demo_index: float,
    geom: DemoGeometry,
    horizon: int,
    rate: float = 1.0,
) -> np.ndarray:
    """Synthesize the expert's ``horizon``-step response from one state.

    Demo clock advances one index per tick from ``demo_index`` (holding at
    the demo's end); the state's offset from its corridor projection decays
    by ``rate * med_step`` per tick to zero along a fixed direction. Returns
    ``(horizon, adim)`` absolute-position labels; k=0 equals the validated
    per-frame bounded-rate label.
    """
    q = np.asarray(state, dtype=np.float64)[: geom.n_arm]
    proj0 = _interp_rows(geom.P, demo_index)
    offset = q - proj0
    d0 = float(np.linalg.norm(offset))
    u = offset / d0 if d0 > 1e-9 else np.zeros_like(offset)
    close = max(0.0, float(rate)) * geom.med_step
    labels = np.empty((horizon, geom.A.shape[1]), dtype=np.float64)
    end = float(len(geom.A) - 1)
    for k in range(horizon):
        labels[k] = _interp_rows(geom.A, min(demo_index + k, end))
        d_k = max(0.0, d0 - (k + 1) * close)
        if d_k > 0.0:
            labels[k, : geom.n_arm] += d_k * u
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
        [chunk_labels(s, i, geom, horizon=1, rate=rate)[0] for s, i in zip(states, idxs, strict=True)]
    )
    return idxs, labels


# ── Train-time wrapper ────────────────────────────────────────────────────


def load_source_geometries(source_repo_id: str, n_arm: int) -> dict[int, DemoGeometry]:
    """Load every episode of a source dataset as DemoGeometry, keyed by episode."""
    import pandas as pd

    root = os.path.expanduser(f"~/.cache/huggingface/lerobot/{source_repo_id}")
    files = sorted(glob.glob(root + "/data/**/*.parquet", recursive=True))
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

    def __init__(self, dataset, source_repo_id: str, n_arm: int, rate: float = 1.0):
        """Wrap ``dataset``, pairing its episodes to ``source_repo_id`` demos."""
        self.dataset = dataset
        self.rate = float(rate)
        self.n_arm = int(n_arm)
        self.geoms = load_source_geometries(source_repo_id, n_arm=n_arm)
        self.ep_to_source = self._episode_pairing(dataset)

    @staticmethod
    def _episode_pairing(dataset) -> dict[int, int]:
        import pandas as pd

        path = os.path.join(str(dataset.root), "meta/episodes/chunk-000/file-000.parquet")
        m = pd.read_parquet(path)
        if "source_episode_idx" not in m.columns:
            raise ValueError(f"{dataset.root}: episodes metadata lacks source_episode_idx pairing")
        return {int(r.episode_index): int(r.source_episode_idx) for r in m.itertuples()}

    def __len__(self) -> int:
        """Length of the wrapped dataset."""
        return len(self.dataset)

    def __getattr__(self, name):
        """Delegate meta/stats/etc. so the wrapper is a drop-in for training."""
        return getattr(self.dataset, name)

    def __getitem__(self, idx: int) -> dict:
        """Return the wrapped item with its action chunk replaced by DART labels."""
        import torch

        item = self.dataset[idx]
        action = item["action"]
        state = item["observation.state"]
        q = state[-1] if state.dim() == 2 else state  # last obs step conditions the chunk
        di = item["relabel_demo_index"]
        di = float(di.reshape(-1)[-1]) if isinstance(di, torch.Tensor) else float(np.reshape(di, -1)[-1])
        geom = self.geoms[self.ep_to_source[int(item["episode_index"])]]
        labels = chunk_labels(q.cpu().numpy(), di, geom, horizon=action.shape[0], rate=self.rate)
        item["action"] = torch.as_tensor(
            labels[:, : action.shape[1]], dtype=action.dtype, device=action.device
        )
        return item
