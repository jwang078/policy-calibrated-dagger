"""Unit tests for gridworld_dagger_sim/core.py.

Run as:  python -m pytest my_scripts/gridworld_dagger_sim/test_core.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from my_scripts.gridworld_dagger_sim.core import (
    DEFAULT_BASE_TRAJECTORIES,
    DEFAULT_GRID_ASCII,
    Action,
    Grid,
    NodeType,
    Policy,
    SimState,
    build_policy_catalog,
    compute_visitation,
    default_state,
    evaluate,
    load_state,
    rollout,
    save_state,
)


def _default_grid() -> Grid:
    return Grid.from_ascii(DEFAULT_GRID_ASCII)


# ─────────────────────────────────── Grid ────────────────────────────────────


def test_grid_ascii_round_trip():
    g = _default_grid()
    assert g.height == 4 and g.width == 4
    assert g.to_ascii() == DEFAULT_GRID_ASCII.strip()


def test_grid_validate_rejects_regular_bottom_right():
    bad = "o o o o\no o o o\no o o o\nx x x o\n"
    with pytest.raises(ValueError, match="(?i)bottom-right"):
        Grid.from_ascii(bad)


def test_grid_outgoing_and_terminal():
    g = _default_grid()
    # interior regular cell: both directions available
    assert sorted(g.outgoing(1, 2)) == [Action.RIGHT, Action.DOWN]
    # right column (not bottom) regular cell: only DOWN
    assert g.outgoing(1, 3) == [Action.DOWN]
    # scattered failures: terminal, no outgoing
    assert g.outgoing(1, 1) == []
    assert g.is_terminal(1, 1) is True
    assert g.outgoing(3, 0) == []
    assert g.is_terminal(3, 0) is True
    assert g.outgoing(0, 3) == []
    assert g.is_terminal(0, 3) is True
    # success: terminal
    assert g.is_terminal(3, 3) is True


# ─────────────────────────── compute_visitation ──────────────────────────────


def test_visitation_starts_at_one_at_origin():
    g = _default_grid()
    p = Policy.uniform(g).probs
    v = compute_visitation(g, p)
    assert v[0, 0] == pytest.approx(1.0)
    assert (v >= 0).all() and (v <= 1.0 + 1e-9).all()
    # Total visitation mass in terminal cells = 1 (every rollout ends).
    terminals = np.array([[g.is_terminal(r, c) for c in range(g.width)] for r in range(g.height)])
    assert v[terminals].sum() == pytest.approx(1.0, abs=1e-9)


# ─────────────────────────── Policy constructors ─────────────────────────────


def test_uniform_policy_boundaries_and_interior():
    """`Policy.uniform` is 0.5/0.5 at interior cells; one-hot at boundaries;
    zeros at terminals."""
    g = _default_grid()
    u = Policy.uniform(g)
    # Interior regular cell with both R and D outgoing → 0.5/0.5
    assert u.probs[1, 2, Action.RIGHT] == pytest.approx(0.5)
    assert u.probs[1, 2, Action.DOWN] == pytest.approx(0.5)
    # Right-column regular (only DOWN valid) → [0, 1]
    assert u.probs[1, 3, Action.RIGHT] == pytest.approx(0.0)
    assert u.probs[1, 3, Action.DOWN] == pytest.approx(1.0)
    # Bottom-row regular (only RIGHT valid) → [1, 0]
    assert u.probs[3, 1, Action.RIGHT] == pytest.approx(1.0)
    assert u.probs[3, 1, Action.DOWN] == pytest.approx(0.0)
    # Terminal cell → zeros
    assert u.probs[1, 1].sum() == pytest.approx(0.0)  # FAILURE
    assert u.probs[3, 3].sum() == pytest.approx(0.0)  # SUCCESS


def test_from_trajectories_visited_and_unvisited():
    """Empirical at visited cells; fallback at unvisited."""
    g = _default_grid()
    uniform = Policy.uniform(g)
    traj = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 3), (3, 3)]
    p = Policy.from_trajectories(g, [traj], fallback=uniform)
    # Visited cells with an action: empirical one-hot from this single traj.
    assert p.probs[0, 0, Action.RIGHT] == pytest.approx(1.0)
    assert p.probs[0, 2, Action.DOWN] == pytest.approx(1.0)
    assert p.probs[2, 2, Action.RIGHT] == pytest.approx(1.0)
    # Unvisited regular cells: copy fallback (uniform 0.5/0.5 at interior).
    np.testing.assert_allclose(p.probs[1, 0], uniform.probs[1, 0], atol=1e-12)
    np.testing.assert_allclose(p.probs[2, 0], uniform.probs[2, 0], atol=1e-12)
    # Visitation = 1 along path, 0 elsewhere.
    assert p.visitation[0, 0] == pytest.approx(1.0)
    assert p.visitation[3, 3] == pytest.approx(1.0)
    assert p.visitation[1, 0] == pytest.approx(0.0)


def test_from_trajectories_default_fallback_is_uniform():
    g = _default_grid()
    p1 = Policy.from_trajectories(g, [])  # no fallback → defaults to uniform
    u = Policy.uniform(g)
    np.testing.assert_allclose(p1.probs, u.probs, atol=1e-12)


def test_from_trajectories_multi_disagreement_averages():
    """If two trajectories disagree at a state, the empirical is the mean."""
    g = _default_grid()
    uniform = Policy.uniform(g)
    # Pick a split at (1, 2): one goes R to (1,3), one goes D to (2,2).
    t1 = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 3)]
    t2 = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 3), (3, 3)]
    p = Policy.from_trajectories(g, [t1, t2], fallback=uniform)
    # At (1, 2), one traj chose R, the other D → empirical 0.5/0.5
    assert p.probs[1, 2, Action.RIGHT] == pytest.approx(0.5)
    assert p.probs[1, 2, Action.DOWN] == pytest.approx(0.5)
    # (1, 2) was visited by both → empirical visitation = 2/2 = 1.0
    assert p.visitation[1, 2] == pytest.approx(1.0)


def test_partial_trajectory_falls_back_to_fallback_at_endpoint():
    """Partial trajectories (ending non-terminal) leave the endpoint cell
    with no recorded action → uses fallback there. Endpoint still visited."""
    g = _default_grid()
    uniform = Policy.uniform(g)
    partial = [(0, 0), (0, 1), (0, 2), (1, 2)]  # ends mid-grid
    p = Policy.from_trajectories(g, [partial], fallback=uniform)
    assert p.probs[0, 0, Action.RIGHT] == pytest.approx(1.0)
    assert p.probs[0, 1, Action.RIGHT] == pytest.approx(1.0)
    assert p.probs[0, 2, Action.DOWN] == pytest.approx(1.0)
    # Endpoint: no action recorded → falls back to uniform.
    np.testing.assert_allclose(p.probs[1, 2], uniform.probs[1, 2], atol=1e-12)
    # All cells on the path (including endpoint) get visitation 1.0.
    for r, c in partial:
        assert p.visitation[r, c] == pytest.approx(1.0)


# ────────────────────────── BC-correct mixing ────────────────────────────────


def test_weighted_sum_identity_one_input():
    g = _default_grid()
    p = Policy.uniform(g)
    out = Policy.weighted_sum(g, [(1.0, p)])
    np.testing.assert_allclose(out.probs, p.probs, atol=1e-12)
    np.testing.assert_allclose(out.visitation, p.visitation, atol=1e-12)


def test_weighted_sum_two_identical_inputs():
    g = _default_grid()
    p = Policy.uniform(g)
    out = Policy.weighted_sum(g, [(0.5, p), (0.5, p)])
    np.testing.assert_allclose(out.probs, p.probs, atol=1e-12)


def test_blend_at_intervention_unvisited_cells_equals_base():
    """At intervention-unvisited cells, `Policy.blend` returns the base policy.

    Mechanism is FALLBACK EQUALITY (not visitation zeroing):
    `Policy.from_trajectories` populates intervention-unvisited cells from
    its `fallback=base`, so at those cells `intervention.probs = base.probs`,
    and the linear mix `(1-r)·base + r·base = base` regardless of ratio.
    """
    g = _default_grid()
    base = Policy.uniform(g)
    traj = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 3), (3, 3)]
    iv = Policy.from_trajectories(g, [traj], fallback=base)
    blended = Policy.blend(g, iv, base, ratio=0.5)
    # Pick an unvisited cell: (2, 1) is regular + unvisited in this traj.
    assert iv.visitation[2, 1] == pytest.approx(0.0)
    np.testing.assert_allclose(blended.probs[2, 1], base.probs[2, 1], atol=1e-12)
    np.testing.assert_allclose(blended.probs[1, 0], base.probs[1, 0], atol=1e-12)


def test_blend_vs_weighted_sum_differ_at_unequal_visitation_cells():
    """Simple `Policy.blend` and BC-correct `Policy.weighted_sum` agree at
    endpoints (ratio=0 or 1) and at cells where both inputs visit equally,
    but diverge at cells where `v_intervention(s) ≠ v_base(s)`.

    Pins the conceptual distinction: blend = action-level mix
    (analog of SA wrapper INTERPOLATE); weighted_sum = visitation-weighted
    MLE (analog of perfect BC on a union of sub-datasets).
    """
    g = _default_grid()
    # Build a base policy from demos that all start with R, so v_base(0,0)=1
    # but v_base(0,1) follows from base's actions; a single intervention
    # demo on a different path makes v_int ≠ v_base at key cells.
    base = Policy.from_trajectories(
        g,
        [
            [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 3), (3, 3)],
            [(0, 0), (0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 3)],
            [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (2, 3), (3, 3)],
        ],
        fallback=Policy.uniform(g),
    )
    iv = Policy.from_trajectories(
        g,
        [[(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 3), (3, 3)]],
        fallback=base,
    )
    # Endpoint agreement: ratio=0 gives intervention.probs (modulo
    # at-unvisited cells where intervention = base by fallback).
    b_simple_0 = Policy.blend(g, iv, base, ratio=0.0)
    b_bcorr_0 = Policy.weighted_sum(g, [(1.0, iv), (0.0, base)])
    np.testing.assert_allclose(b_simple_0.probs, b_bcorr_0.probs, atol=1e-12)
    # Endpoint agreement: ratio=1 → pure base for both.
    b_simple_1 = Policy.blend(g, iv, base, ratio=1.0)
    b_bcorr_1 = Policy.weighted_sum(g, [(0.0, iv), (1.0, base)])
    np.testing.assert_allclose(b_simple_1.probs, b_bcorr_1.probs, atol=1e-12)
    # Mid-ratio divergence at a cell where v_int and v_base differ.
    # (1, 2) is visited by both base demos 1 & 2 and by the intervention,
    # so v_int(1,2) = 1.0 but v_base(1,2) = P(R, R)·1 = 2/3.
    b_simple_half = Policy.blend(g, iv, base, ratio=0.5)
    b_bcorr_half = Policy.weighted_sum(g, [(0.5, iv), (0.5, base)])
    assert not np.allclose(b_simple_half.probs[1, 2], b_bcorr_half.probs[1, 2], atol=1e-6), (
        f"expected simple≠BC-correct at (1,2): simple={b_simple_half.probs[1, 2]}, "
        f"bcorr={b_bcorr_half.probs[1, 2]}"
    )


def test_blend_ratio_zero_and_one():
    """Convention: ratio=0.0 is pure intervention, ratio=1.0 is pure base."""
    g = _default_grid()
    base = Policy.uniform(g)
    traj = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 3), (3, 3)]
    iv = Policy.from_trajectories(g, [traj], fallback=base)
    # ratio = 0 → pure intervention.
    b0 = Policy.blend(g, iv, base, ratio=0.0)
    for r, c in traj[:-1]:
        np.testing.assert_allclose(b0.probs[r, c], iv.probs[r, c], atol=1e-12)
    np.testing.assert_allclose(b0.probs[2, 1], base.probs[2, 1], atol=1e-12)
    # ratio = 1 → pure base everywhere.
    b1 = Policy.blend(g, iv, base, ratio=1.0)
    np.testing.assert_allclose(b1.probs, base.probs, atol=1e-12)


# ─────────────────────────── Softening (policy noise) ────────────────────────


def test_softened_endpoints_and_boundary_invariance():
    g = _default_grid()
    base = Policy.uniform(g)
    traj = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 3), (3, 3)]
    iv = Policy.from_trajectories(g, [traj], fallback=base)

    # noise = 0 → unchanged
    same = iv.softened(g, noise=0.0)
    np.testing.assert_allclose(same.probs, iv.probs, atol=1e-12)
    np.testing.assert_allclose(same.visitation, iv.visitation, atol=1e-12)

    # noise = 1 on uniform → still uniform (already uniform)
    fully_noisy = base.softened(g, noise=1.0)
    assert fully_noisy.probs[1, 2, Action.RIGHT] == pytest.approx(0.5)
    assert fully_noisy.probs[1, 2, Action.DOWN] == pytest.approx(0.5)
    # Boundary cell (right column, only DOWN valid) → noise can't change one-hot
    assert fully_noisy.probs[1, 3, Action.RIGHT] == pytest.approx(0.0)
    assert fully_noisy.probs[1, 3, Action.DOWN] == pytest.approx(1.0)

    # noise = 0.5 on a one-hot intervention cell → halfway to 0.5/0.5
    half_noisy = iv.softened(g, noise=0.5)
    # (0, 0) in iv is one-hot RIGHT; softened with noise=0.5 → 0.75/0.25
    assert half_noisy.probs[0, 0, Action.RIGHT] == pytest.approx(0.75)
    assert half_noisy.probs[0, 0, Action.DOWN] == pytest.approx(0.25)


def test_softened_intervention_loses_perfect_success():
    """With noise > 0 on a deterministic expert, single-step success rate
    decays. Stochastic categorical sampling means deviations into unvisited
    cells (where base/fallback rules) can lead to failures."""
    g = _default_grid()
    base = Policy.uniform(g)
    traj = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 3), (3, 3)]
    iv = Policy.from_trajectories(g, [traj], fallback=base)

    rng = np.random.default_rng(0)
    r_clean = evaluate(g, iv, 2000, rng)
    assert r_clean["succ_rate"] == pytest.approx(1.0)

    iv_noisy = iv.softened(g, noise=0.2)
    rng = np.random.default_rng(0)
    r_noisy = evaluate(g, iv_noisy, 5000, rng)
    assert r_noisy["succ_rate"] < 0.95


# ────────────────────────────────── Rollout ──────────────────────────────────


def test_rollout_deterministic_on_one_hot_policy():
    g = _default_grid()
    base = Policy.uniform(g)
    traj = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 3), (3, 3)]
    iv = Policy.from_trajectories(g, [traj], fallback=base)
    rng = np.random.default_rng(0)
    path, outcome = rollout(g, iv, rng)
    assert path == traj
    assert outcome == NodeType.SUCCESS


def test_evaluate_stable_succ_rate_on_default_grid():
    g = _default_grid()
    p = Policy.uniform(g)
    rng = np.random.default_rng(0)
    a = evaluate(g, p, 5000, rng)
    rng2 = np.random.default_rng(1)
    b = evaluate(g, p, 5000, rng2)
    assert abs(a["succ_rate"] - b["succ_rate"]) < 0.03


# ────────────────────────────── State I/O ────────────────────────────────────


def test_default_state_has_base_trajectories():
    """Default state should ship with the example base demos."""
    s = default_state()
    assert len(s.base_trajectories) == len(DEFAULT_BASE_TRAJECTORIES)
    for t in s.base_trajectories:
        assert "name" in t and "path" in t
        assert t["path"][0] == (0, 0)
        # Every default demo ends at SUCCESS (3, 3).
        assert t["path"][-1] == (3, 3)


def test_state_roundtrip(tmp_path):
    state = default_state()
    state.interventions.append(
        {"name": "expert", "path": [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 3), (3, 3)]}
    )
    state.compositions.append(
        {
            "name": "blend_0.4",
            "rows": [{"policy": "base", "weight": 0.4}, {"policy": "intervention", "weight": 0.6}],
        }
    )
    p = tmp_path / "state.json"
    save_state(p, state)
    loaded = load_state(p)
    assert loaded.grid.to_ascii() == state.grid.to_ascii()
    # base_trajectories round-trip
    assert len(loaded.base_trajectories) == len(state.base_trajectories)
    for orig, ld in zip(state.base_trajectories, loaded.base_trajectories, strict=False):
        assert orig["name"] == ld["name"]
        assert orig["path"] == ld["path"]
    # interventions and compositions round-trip
    assert len(loaded.interventions) == 1
    assert loaded.interventions[0]["path"][0] == (0, 0)
    assert loaded.compositions[0]["name"] == "blend_0.4"


def test_build_policy_catalog_with_composition():
    state = default_state()
    state.interventions.append(
        {"name": "expert", "path": [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 3), (3, 3)]}
    )
    state.compositions.append(
        {
            "name": "blend_0.4",
            "rows": [{"policy": "base", "weight": 0.4}, {"policy": "intervention", "weight": 0.6}],
        }
    )
    cat = build_policy_catalog(state)
    assert set(cat.keys()) == {"base", "intervention", "blend_0.4"}
    # All three policies should achieve high success on the default demos.
    rng = np.random.default_rng(0)
    base_succ = evaluate(state.grid, cat["base"], 5000, rng)["succ_rate"]
    rng = np.random.default_rng(0)
    blend_succ = evaluate(state.grid, cat["blend_0.4"], 5000, rng)["succ_rate"]
    # Both should be near 100% — the default demos cover the safe region.
    assert base_succ > 0.95
    assert blend_succ > 0.95


def test_build_policy_catalog_empty_base_trajectories_falls_back_to_uniform():
    """If base_trajectories is empty, base policy is Policy.uniform."""
    state = SimState(grid=_default_grid())
    cat = build_policy_catalog(state)
    u = Policy.uniform(state.grid)
    np.testing.assert_allclose(cat["base"].probs, u.probs, atol=1e-12)
