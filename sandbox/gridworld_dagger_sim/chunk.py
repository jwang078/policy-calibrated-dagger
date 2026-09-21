"""Chunk-mode policies for the gridworld DAgger simulator.

A chunk policy outputs a K-step sequence of discrete actions per decision,
instead of a single action. At each cell:

    P(chunk = c | cell)   for each c ∈ {R, D}^K

The agent samples one chunk, executes its K actions, lands at a new cell,
and resamples. This mirrors the action-chunk architecture of
`shared_autonomy_wrapper.py` (chunks of N=8 EE-delta vectors) and captures
"curved" multi-step plans without needing continuous actions.

Why this matters for blending: a mix of two categorical chunk distributions
is naturally mode-preserving. If base says `{"DDR": 1.0}` and intervention
says `{"RRD": 1.0}`, blending at r=0.5 gives `{"DDR": 0.5, "RRD": 0.5}` —
stochastic between two SAFE chunks, never an "average diagonal" chunk
(no such thing exists in a categorical action space). The Frankenstein
collision of continuous-mode INTERPOLATE doesn't arise.

This is the well-defined discrete approximation to a continuous action
vector — instead of trying to "blend" a `(dx, dy)` vector toward a
diagonal that's off the action manifold, the chunk policy stores a
categorical distribution over discrete multi-step plans and samples
between them. Blending is mode-preserving by construction.

This module is independent of `Policy` in `core.py` — it lives in its
own namespace because the action space, rollout, and blending math are
sufficiently different. `Grid` / `NodeType` / `Action` are reused from
`core.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from my_scripts.gridworld_dagger_sim.core import (
    Action,
    Grid,
    NodeType,
)

# Default chunk length used by experiments / from_trajectories if not specified.
DEFAULT_CHUNK_K: int = 3


# ──────────────────────────── Chunk encoding ─────────────────────────────────


def n_chunks(k: int) -> int:
    """Number of distinct K-action chunks (2^K)."""
    return 1 << k


def decode_chunk(idx: int, k: int) -> tuple[int, ...]:
    """Decode a chunk index to its K-tuple of actions.

    Bit `i` of `idx` (LSB = first action) selects RIGHT (=0) vs DOWN (=1).
    Examples for K=3:
        0b000 → (R, R, R)
        0b011 → (D, D, R)
        0b111 → (D, D, D)
    """
    return tuple(int((idx >> i) & 1) for i in range(k))


def encode_chunk(actions: tuple[int, ...] | list[int]) -> int:
    """Encode a sequence of {RIGHT=0, DOWN=1} actions as a chunk index."""
    idx = 0
    for i, a in enumerate(actions):
        if a not in (0, 1):
            raise ValueError(f"Invalid action {a}; must be 0 (RIGHT) or 1 (DOWN).")
        idx |= int(a) << i
    return idx


# ──────────────────────────────── Types ──────────────────────────────────────


@dataclass
class ChunkPolicy:
    """Per-cell categorical distribution over K-step action chunks.

    chunk_probs[r, c, i] = P(chunk index i | cell (r, c)).
    For each cell, the slice `chunk_probs[r, c, :]` is a probability
    distribution (sums to 1) over `2^K` chunks (encoded as integers).
    """

    grid_height: int
    grid_width: int
    chunk_k: int
    chunk_probs: np.ndarray  # (H, W, 2^K)
    visitation: np.ndarray  # (H, W) — data density (from_trajectories)

    @property
    def n_chunks(self) -> int:
        return self.chunk_probs.shape[-1]

    # ── Constructors ───────────────────────────────────────────────────────

    @classmethod
    def uniform(cls, grid: Grid, chunk_k: int = DEFAULT_CHUNK_K) -> ChunkPolicy:
        """Uniform-over-VALID-chunks at every non-terminal cell.

        A chunk is "valid from cell (r, c)" if executing its K actions in
        sequence never tries to move off the grid (e.g. RIGHT from the
        last column). Invalid chunks get probability 0; the remaining
        chunks share probability uniformly.

        Terminal cells get all-zero probs (never sampled at rollout).
        """
        H, W = grid.height, grid.width
        N = n_chunks(chunk_k)
        chunk_probs = np.zeros((H, W, N), dtype=np.float64)
        for r in range(H):
            for c in range(W):
                if grid.is_terminal(r, c):
                    continue
                valid_mask = _valid_chunks_from_cell(grid, r, c, chunk_k)
                count = int(valid_mask.sum())
                if count == 0:
                    continue
                chunk_probs[r, c, valid_mask] = 1.0 / count
        v = compute_visitation_chunked(grid, chunk_probs, chunk_k)
        return cls(
            grid_height=H,
            grid_width=W,
            chunk_k=chunk_k,
            chunk_probs=chunk_probs,
            visitation=v,
        )

    @classmethod
    def from_trajectories(
        cls,
        grid: Grid,
        trajectories: list[list[tuple[int, int]]],
        chunk_k: int = DEFAULT_CHUNK_K,
        fallback: ChunkPolicy | None = None,
    ) -> ChunkPolicy:
        """Build a chunk policy from a set of demo trajectories.

        For each cell visited by any demo, record the next-K-step chunk
        the demo executed from that cell. Empirical distribution =
        observed-chunk counts / total visits at cell.

        At cells visited by no demo, fall back to `fallback.chunk_probs`
        (defaults to `ChunkPolicy.uniform`).

        If a demo has fewer than K steps remaining at some cell (i.e.
        it's about to terminate), that cell is treated as "no chunk
        recorded" — it'll use the fallback.
        """
        if fallback is None:
            fallback = cls.uniform(grid, chunk_k=chunk_k)
        elif fallback.chunk_k != chunk_k:
            raise ValueError(f"chunk_k mismatch: requested {chunk_k}, fallback has {fallback.chunk_k}")
        H, W = grid.height, grid.width
        N = n_chunks(chunk_k)
        chunk_counts = np.zeros((H, W, N), dtype=np.float64)
        visit_counts = np.zeros((H, W), dtype=np.float64)
        for traj in trajectories:
            L = len(traj)
            visited_in_traj: set[tuple[int, int]] = set()
            for i in range(L):
                r, c = traj[i]
                if (r, c) not in visited_in_traj:
                    visit_counts[r, c] += 1
                    visited_in_traj.add((r, c))
                # Try to extract a K-step chunk starting at this cell.
                # Need K actions, so K next-cells beyond traj[i].
                if i + chunk_k >= L:
                    continue  # not enough remaining steps for a full chunk
                chunk_actions: list[int] = []
                ok = True
                for j in range(chunk_k):
                    rj, cj = traj[i + j]
                    nrj, ncj = traj[i + j + 1]
                    if nrj == rj and ncj == cj + 1:
                        chunk_actions.append(int(Action.RIGHT))
                    elif ncj == cj and nrj == rj + 1:
                        chunk_actions.append(int(Action.DOWN))
                    else:
                        ok = False
                        break
                if not ok:
                    continue
                idx = encode_chunk(chunk_actions)
                chunk_counts[r, c, idx] += 1
        # Build chunk_probs: visited-with-chunks cells get empirical,
        # cells with no recorded chunk fall back.
        chunk_probs = np.array(fallback.chunk_probs, copy=True)
        sums = chunk_counts.sum(axis=-1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            empirical = np.where(sums > 0, chunk_counts / sums, 0.0)
        has_chunks = sums.squeeze(-1) > 0
        chunk_probs[has_chunks] = empirical[has_chunks]
        # Defensive: ensure no probability sits on an invalid chunk
        # (could happen if fallback was non-uniform and we copied it).
        # Re-normalize after masking invalid chunks to zero.
        for r in range(H):
            for c in range(W):
                if grid.is_terminal(r, c):
                    chunk_probs[r, c] = 0.0
                    continue
                valid = _valid_chunks_from_cell(grid, r, c, chunk_k)
                chunk_probs[r, c, ~valid] = 0.0
                s = float(chunk_probs[r, c].sum())
                if s > 0:
                    chunk_probs[r, c] /= s
                else:
                    # No valid chunks have probability — use uniform-over-valid.
                    count = int(valid.sum())
                    if count > 0:
                        chunk_probs[r, c, valid] = 1.0 / count
        K = max(len(trajectories), 1)
        visitation = visit_counts / K
        return cls(
            grid_height=H,
            grid_width=W,
            chunk_k=chunk_k,
            chunk_probs=chunk_probs,
            visitation=visitation,
        )

    # ── Blending ───────────────────────────────────────────────────────────

    @classmethod
    def blend(
        cls,
        grid: Grid,
        intervention: ChunkPolicy,
        base: ChunkPolicy,
        ratio: float,
    ) -> ChunkPolicy:
        """Simple per-cell categorical mix:

            blend.chunk_probs[r, c] = (1 - r) · intervention.chunk_probs[r, c]
                                       +    r  · base.chunk_probs[r, c]

        Convention matches `Policy.blend`: ratio=0 → pure intervention,
        ratio=1 → pure base. Both inputs must have the same chunk_k.

        Because the action space is categorical over discrete chunks,
        the mix is naturally mode-preserving — there's no "average chunk"
        that lies off-manifold.
        """
        if not (0.0 <= ratio <= 1.0):
            raise ValueError(f"ratio must be in [0, 1], got {ratio}")
        if intervention.chunk_k != base.chunk_k:
            raise ValueError(f"chunk_k mismatch: intervention={intervention.chunk_k}, base={base.chunk_k}")
        chunk_probs = (1.0 - ratio) * intervention.chunk_probs + ratio * base.chunk_probs
        H, W = grid.height, grid.width
        v = compute_visitation_chunked(grid, chunk_probs, intervention.chunk_k)
        return cls(
            grid_height=H,
            grid_width=W,
            chunk_k=intervention.chunk_k,
            chunk_probs=chunk_probs,
            visitation=v,
        )

    @classmethod
    def weighted_sum(
        cls,
        grid: Grid,
        weighted: list[tuple[float, ChunkPolicy]],
    ) -> ChunkPolicy:
        """BC-correct (visitation-weighted) mix of chunk policies.

            π̂(chunk | s) = (Σ wᵢ · vᵢ(s) · πᵢ(chunk | s)) / (Σ wᵢ · vᵢ(s))

        Same semantics as `Policy.weighted_sum` in `core.py`, just over
        chunks instead of single actions. All inputs must share `chunk_k`.
        """
        if not weighted:
            raise ValueError("weighted_sum needs at least one (weight, policy) pair.")
        chunk_k = weighted[0][1].chunk_k
        for _, p in weighted:
            if p.chunk_k != chunk_k:
                raise ValueError(f"all inputs must share chunk_k; got {chunk_k} and {p.chunk_k}")
        weights = np.array([w for w, _ in weighted], dtype=np.float64)
        total = weights.sum()
        if total <= 0:
            raise ValueError("Weights must sum to a positive value.")
        weights = weights / total
        H, W = grid.height, grid.width
        N = n_chunks(chunk_k)
        num = np.zeros((H, W, N), dtype=np.float64)
        denom = np.zeros((H, W), dtype=np.float64)
        for w, pol in zip(weights, [p for _, p in weighted], strict=False):
            wv = w * pol.visitation  # (H, W)
            denom += wv
            num += wv[..., None] * pol.chunk_probs
        with np.errstate(invalid="ignore", divide="ignore"):
            chunk_probs = np.where(denom[..., None] > 0, num / denom[..., None], 0.0)
        zero_mask = denom <= 0
        if zero_mask.any():
            chunk_probs[zero_mask] = weighted[0][1].chunk_probs[zero_mask]
        v = compute_visitation_chunked(grid, chunk_probs, chunk_k)
        return cls(
            grid_height=H,
            grid_width=W,
            chunk_k=chunk_k,
            chunk_probs=chunk_probs,
            visitation=v,
        )


# ──────────────────────────── Chunk validity ─────────────────────────────────


def _valid_chunks_from_cell(grid: Grid, r: int, c: int, chunk_k: int) -> np.ndarray:
    """Boolean mask of length `2^K` — True iff the chunk can be executed
    from (r, c) without ever trying to move off the grid.

    A chunk's actions are applied sequentially; a step is "valid" if the
    move stays in bounds. If a step lands on a terminal cell mid-chunk
    that's fine — execution stops there at rollout time (chunks are
    allowed to "overshoot" terminals).
    """
    N = n_chunks(chunk_k)
    valid = np.zeros(N, dtype=bool)
    for idx in range(N):
        chunk = decode_chunk(idx, chunk_k)
        cur_r, cur_c = r, c
        ok = True
        for a in chunk:
            if a == Action.RIGHT:
                if cur_c + 1 >= grid.width:
                    ok = False
                    break
                cur_c += 1
            else:  # DOWN
                if cur_r + 1 >= grid.height:
                    ok = False
                    break
                cur_r += 1
            # Reaching a terminal mid-chunk is fine (the chunk just stops
            # being relevant; remaining steps are ignored at rollout).
            if grid.is_terminal(cur_r, cur_c):
                break
        valid[idx] = ok
    return valid


# ─────────────────────────────── Rollout ─────────────────────────────────────


def rollout_chunked(
    grid: Grid,
    policy: ChunkPolicy,
    rng: np.random.Generator,
) -> tuple[list[tuple[int, int]], NodeType]:
    """Chunk-mode rollout. Returns (cell_path, terminal NodeType).

    Per "decision":
      1. Determine current cell (r, c).
      2. If terminal, return its NodeType.
      3. Sample chunk index from `chunk_probs[r, c]`.
      4. Execute the chunk action-by-action, appending each new cell to
         the path. Stop early on terminal or off-grid.
      5. Loop until terminal or step budget exhausted.
    """
    r, c = 0, 0
    path: list[tuple[int, int]] = [(0, 0)]
    K = policy.chunk_k
    max_decisions = 4 * (grid.height + grid.width)  # generous safety cap
    for _ in range(max_decisions):
        if grid.is_terminal(r, c):
            return path, NodeType(int(grid.node_types[r, c]))
        p = policy.chunk_probs[r, c]
        s = float(p.sum())
        if s <= 1e-12:
            return path, NodeType.FAILURE  # no valid chunk
        # Renormalize defensively.
        chunk_idx = int(rng.choice(policy.n_chunks, p=p / s))
        chunk = decode_chunk(chunk_idx, K)
        for a in chunk:
            if grid.is_terminal(r, c):
                break
            if a == Action.RIGHT:
                if c + 1 >= grid.width:
                    return path, NodeType.FAILURE  # off-grid mid-chunk
                c += 1
            else:
                if r + 1 >= grid.height:
                    return path, NodeType.FAILURE
                r += 1
            path.append((r, c))
    return path, NodeType.FAILURE  # step budget exhausted


def evaluate_chunked(
    grid: Grid,
    policy: ChunkPolicy,
    n_rollouts: int,
    rng: np.random.Generator,
) -> dict:
    succ = 0
    fail = 0
    for _ in range(n_rollouts):
        _, outcome = rollout_chunked(grid, policy, rng)
        if outcome == NodeType.SUCCESS:
            succ += 1
        elif outcome == NodeType.FAILURE:
            fail += 1
    n = max(n_rollouts, 1)
    return {
        "n": n_rollouts,
        "succ": succ,
        "fail": fail,
        "succ_rate": succ / n,
        "fail_rate": fail / n,
    }


# ─────────────────────────────── Visitation ──────────────────────────────────


def compute_visitation_chunked(
    grid: Grid,
    chunk_probs: np.ndarray,
    chunk_k: int,
) -> np.ndarray:
    """Forward-DP visitation: probability of visiting each cell during a
    chunked rollout starting at (0, 0).

    Decision points (cells where the agent samples a new chunk) and the
    intermediate cells a chunk passes through must be tracked separately:
    only decision points spawn new chunks. If we propagated from every cell
    with v > 0, the mass leaving (0, 0) via a DDD chunk would also spawn
    chunks at (1, 0) and (2, 0), wildly over-counting.

    `v_dec` tracks the probability the agent makes a decision at each cell;
    `v_total` tracks the probability of visiting each cell at any point
    (decision-point or mid-chunk). We return `v_total` to match the
    "data density" semantics used by `from_trajectories`.
    """
    H, W = grid.height, grid.width
    v_dec = np.zeros((H, W), dtype=np.float64)
    v_total = np.zeros((H, W), dtype=np.float64)
    v_dec[0, 0] = 1.0
    v_total[0, 0] = 1.0
    # Every chunk action advances row+col by 1, so the landing cell of a
    # chunk starting at diagonal d is on diagonal d+K. Processing in diagonal
    # order means v_dec at a source cell is finalized before its successors.
    cells_by_diagonal: dict[int, list[tuple[int, int]]] = {}
    for r in range(H):
        for c in range(W):
            cells_by_diagonal.setdefault(r + c, []).append((r, c))
    for d in sorted(cells_by_diagonal):
        for r, c in cells_by_diagonal[d]:
            if v_dec[r, c] <= 0 or grid.is_terminal(r, c):
                continue
            for idx in range(chunk_probs.shape[-1]):
                p_chunk = float(chunk_probs[r, c, idx])
                if p_chunk <= 0:
                    continue
                mass = v_dec[r, c] * p_chunk
                cur_r, cur_c = r, c
                actions = decode_chunk(idx, chunk_k)
                completed_full_chunk = True
                for a in actions:
                    if a == Action.RIGHT:
                        if cur_c + 1 >= W:
                            completed_full_chunk = False
                            break
                        cur_c += 1
                    else:
                        if cur_r + 1 >= H:
                            completed_full_chunk = False
                            break
                        cur_r += 1
                    v_total[cur_r, cur_c] += mass
                    if grid.is_terminal(cur_r, cur_c):
                        completed_full_chunk = False
                        break
                if completed_full_chunk:
                    # `(cur_r, cur_c)` is the landing cell — agent samples
                    # a new chunk here.
                    v_dec[cur_r, cur_c] += mass
    return v_total
