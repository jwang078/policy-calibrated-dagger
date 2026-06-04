"""Unit tests for gridworld_dagger_sim/chunk.py.

Run as:  python -m pytest my_scripts/gridworld_dagger_sim/test_chunk.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from my_scripts.gridworld_dagger_sim.chunk import (
    ChunkPolicy,
    _valid_chunks_from_cell,
    compute_visitation_chunked,
    decode_chunk,
    encode_chunk,
    evaluate_chunked,
    n_chunks,
    rollout_chunked,
)
from my_scripts.gridworld_dagger_sim.core import (
    DEFAULT_GRID_ASCII,
    Action,
    Grid,
    NodeType,
)


def _default_grid() -> Grid:
    return Grid.from_ascii(DEFAULT_GRID_ASCII)


def _open_8x8_grid() -> Grid:
    """8x8 grid with a single FAILURE at (1, 1) so a bifurcation at (0, 0)
    between R-first and D-first chunks deterministically punishes the
    'average diagonal' equivalent.
    """
    rows = [["o"] * 8 for _ in range(8)]
    rows[1][1] = "x"
    rows[-1][-1] = "s"
    ascii_grid = "\n".join(" ".join(r) for r in rows)
    return Grid.from_ascii(ascii_grid)


# ────────────────────────── Chunk encoding round-trip ────────────────────────


def test_n_chunks():
    assert n_chunks(1) == 2
    assert n_chunks(3) == 8
    assert n_chunks(5) == 32


def test_decode_chunk_known_values():
    # 0b000 → (R, R, R); 0b011 → (D, D, R); 0b111 → (D, D, D)
    assert decode_chunk(0, 3) == (0, 0, 0)
    assert decode_chunk(0b011, 3) == (1, 1, 0)
    assert decode_chunk(0b111, 3) == (1, 1, 1)


def test_encode_decode_roundtrip():
    for k in (1, 2, 3, 4):
        for idx in range(n_chunks(k)):
            assert encode_chunk(decode_chunk(idx, k)) == idx


def test_encode_rejects_bad_actions():
    with pytest.raises(ValueError):
        encode_chunk([0, 2, 1])


# ─────────────────────────── _valid_chunks_from_cell ────────────────────────


def test_valid_chunks_interior_all_valid_when_far_from_edges():
    g = _open_8x8_grid()
    valid = _valid_chunks_from_cell(g, 0, 0, chunk_k=3)
    # From (0, 0) all 8 chunks of length 3 stay in-grid (no failure on way).
    assert valid.sum() == 8


def test_valid_chunks_excludes_off_grid_steps_from_bottom_row():
    g = _open_8x8_grid()
    valid = _valid_chunks_from_cell(g, 7, 5, chunk_k=3)
    # At (7, 5): any chunk starting with D goes off-grid (row 7+1=8).
    # Of the remaining R-first chunks, those reaching (7, 7) terminal still
    # count as valid since terminal-hit aborts the chunk.
    for idx in range(n_chunks(3)):
        actions = decode_chunk(idx, 3)
        first_d = actions[0] == Action.DOWN
        if first_d:
            assert not valid[idx]


def test_valid_chunks_allows_terminal_mid_chunk():
    """Hitting a terminal mid-chunk is fine — the chunk just stops there."""
    g = _open_8x8_grid()
    # From (7, 5): chunk RRR → (7, 6), (7, 7) terminal-hit → stop. Valid.
    valid = _valid_chunks_from_cell(g, 7, 5, chunk_k=3)
    assert valid[encode_chunk([Action.RIGHT, Action.RIGHT, Action.RIGHT])]


# ─────────────────────────── ChunkPolicy.uniform ────────────────────────────


def test_uniform_chunk_probs_sum_to_one_at_non_terminal():
    g = _default_grid()
    pol = ChunkPolicy.uniform(g, chunk_k=3)
    for r in range(g.height):
        for c in range(g.width):
            s = pol.chunk_probs[r, c].sum()
            if g.is_terminal(r, c):
                assert s == pytest.approx(0.0)
            else:
                assert s == pytest.approx(1.0)


def test_uniform_chunk_zero_on_invalid_chunks():
    g = _default_grid()
    pol = ChunkPolicy.uniform(g, chunk_k=3)
    # At any cell on the right edge (excl bottom row), chunks starting with R
    # are invalid — must get 0 probability.
    for r in range(g.height):
        c = g.width - 1
        if g.is_terminal(r, c):
            continue
        for idx in range(pol.n_chunks):
            actions = decode_chunk(idx, 3)
            if actions[0] == Action.RIGHT:
                assert pol.chunk_probs[r, c, idx] == pytest.approx(0.0)


# ──────────────────────── ChunkPolicy.from_trajectories ─────────────────────


def test_from_trajectories_records_chunks_on_path():
    g = _open_8x8_grid()
    traj = [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),
        (0, 6),
        (0, 7),
        (1, 7),
        (2, 7),
        (3, 7),
        (4, 7),
        (5, 7),
        (6, 7),
        (7, 7),
    ]
    pol = ChunkPolicy.from_trajectories(g, [traj], chunk_k=3)
    # At (0, 0): next 3 actions = R R R → chunk idx = 0.
    assert pol.chunk_probs[0, 0, 0] == pytest.approx(1.0)
    assert pol.chunk_probs[0, 0, 1:].sum() == pytest.approx(0.0)
    # Visitation along the path is 1.0 per visited cell.
    for r, c in traj:
        assert pol.visitation[r, c] == pytest.approx(1.0)


def test_from_trajectories_falls_back_at_unvisited_cells():
    g = _open_8x8_grid()
    # Single demo D-first then R-along-bottom-row.
    d_first = [(r, 0) for r in range(8)] + [(7, c) for c in range(1, 8)]
    pol = ChunkPolicy.from_trajectories(g, [d_first], chunk_k=3)
    # (0, 7) is unvisited; chunk_probs should equal the uniform-over-valid
    # fallback there.
    uniform = ChunkPolicy.uniform(g, chunk_k=3)
    np.testing.assert_allclose(pol.chunk_probs[0, 7], uniform.chunk_probs[0, 7], atol=1e-12)


def test_blend_at_bifurcation_avoids_diagonal_collision():
    """The headline: chunk-mode blend at r=0.5 between a R-first intervention
    and a D-first base distributes mass over two SAFE multi-step chunks
    (RRR or DDD), never over a 'diagonal' chunk — so the rollout never
    hits the (1, 1) FAILURE that a continuous diagonal vector would.
    """
    g = _open_8x8_grid()
    r_first = [(0, c) for c in range(8)] + [(r, 7) for r in range(1, 8)]
    d_first = [(r, 0) for r in range(8)] + [(7, c) for c in range(1, 8)]
    iv = ChunkPolicy.from_trajectories(g, [r_first], chunk_k=3)
    base = ChunkPolicy.from_trajectories(
        g,
        [d_first],
        chunk_k=3,
        fallback=iv,
    )
    blended = ChunkPolicy.blend(g, iv, base, ratio=0.5)
    rng = np.random.default_rng(0)
    res = evaluate_chunked(g, blended, 1000, rng)
    # Both R-first and D-first paths reach success; the categorical blend
    # picks one or the other per rollout — every rollout succeeds.
    assert res["succ_rate"] == pytest.approx(1.0, abs=0.0)


def test_blend_ratio_zero_and_one_chunks():
    g = _open_8x8_grid()
    r_first = [(0, c) for c in range(8)] + [(r, 7) for r in range(1, 8)]
    d_first = [(r, 0) for r in range(8)] + [(7, c) for c in range(1, 8)]
    iv = ChunkPolicy.from_trajectories(g, [r_first], chunk_k=3)
    base = ChunkPolicy.from_trajectories(g, [d_first], chunk_k=3, fallback=iv)
    b0 = ChunkPolicy.blend(g, iv, base, ratio=0.0)
    np.testing.assert_allclose(b0.chunk_probs[0, 0], iv.chunk_probs[0, 0], atol=1e-12)
    b1 = ChunkPolicy.blend(g, iv, base, ratio=1.0)
    np.testing.assert_allclose(b1.chunk_probs[0, 0], base.chunk_probs[0, 0], atol=1e-12)


def test_weighted_sum_identity_one_input():
    g = _default_grid()
    pol = ChunkPolicy.uniform(g, chunk_k=2)
    out = ChunkPolicy.weighted_sum(g, [(1.0, pol)])
    np.testing.assert_allclose(out.chunk_probs, pol.chunk_probs, atol=1e-12)


def test_weighted_sum_mismatched_k_raises():
    g = _default_grid()
    p1 = ChunkPolicy.uniform(g, chunk_k=2)
    p2 = ChunkPolicy.uniform(g, chunk_k=3)
    with pytest.raises(ValueError):
        ChunkPolicy.weighted_sum(g, [(0.5, p1), (0.5, p2)])


# ───────────────────────────── rollout_chunked ──────────────────────────────


def test_rollout_chunked_deterministic_on_one_hot_chunks():
    g = _open_8x8_grid()
    r_first = [(0, c) for c in range(8)] + [(r, 7) for r in range(1, 8)]
    pol = ChunkPolicy.from_trajectories(g, [r_first], chunk_k=3)
    rng = np.random.default_rng(0)
    path, outcome = rollout_chunked(g, pol, rng)
    assert outcome == NodeType.SUCCESS
    # First chunk RRR lands at (0, 3); chunks chained until success.
    assert path[0] == (0, 0)
    assert path[-1] == (7, 7)


def test_evaluate_chunked_returns_expected_keys():
    g = _open_8x8_grid()
    pol = ChunkPolicy.uniform(g, chunk_k=2)
    rng = np.random.default_rng(0)
    res = evaluate_chunked(g, pol, 100, rng)
    assert set(res.keys()) == {"n", "succ", "fail", "succ_rate", "fail_rate"}
    assert res["n"] == 100


# ────────────────────── compute_visitation_chunked bounds ───────────────────


def test_visitation_origin_one_and_bounded():
    g = _open_8x8_grid()
    pol = ChunkPolicy.uniform(g, chunk_k=3)
    v = pol.visitation
    assert v[0, 0] == pytest.approx(1.0)
    # NB: total cell visitation can exceed 1.0 only via genuine multi-arrival
    # paths under stochastic chunks; on this grid (uniform), any single cell
    # should be reachable by mass <= 1.
    assert v.max() <= 1.0 + 1e-9, f"v.max()={v.max()} should not exceed 1.0"


def test_visitation_one_hot_demo_matches_path_density():
    """With a deterministic chunk policy that follows a single demo,
    visitation at every cell on the demo path should be exactly 1.0."""
    g = _open_8x8_grid()
    r_first = [(0, c) for c in range(8)] + [(r, 7) for r in range(1, 8)]
    pol = ChunkPolicy.from_trajectories(g, [r_first], chunk_k=3)
    # `from_trajectories` populates visitation from visit_counts. The analytical
    # forward DP via compute_visitation_chunked on a deterministic policy
    # should agree at all visited cells.
    v_dp = compute_visitation_chunked(g, pol.chunk_probs, chunk_k=3)
    for r, c in r_first:
        assert v_dp[r, c] == pytest.approx(1.0, abs=1e-9), (
            f"cell {(r, c)} visitation under DP = {v_dp[r, c]}, expected 1.0"
        )
