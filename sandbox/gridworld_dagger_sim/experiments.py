"""Canned experiments that test what blending does to success rates.

Each experiment prints a labeled markdown table. Tweak the constants near the
top of each function (path of interventions, blend ratios, noise levels, …) to
explore your own scenarios.

Run:

    python -m my_scripts.gridworld_dagger_sim.experiments [--state PATH]
        [--rollouts N] [--seed N] [--experiments NAME [NAME ...]]

By default loads `examples/default_4x4.json` (matching the GUI launcher) and
runs ALL experiments. Pick a subset with `--experiments`:

    python -m my_scripts.gridworld_dagger_sim.experiments \\
        --experiments blend_x_noise intervention_quality

Each experiment is a function `expt_<name>` registered in `EXPERIMENTS`.
Add new scenarios by writing more `expt_*` functions and registering them.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

import numpy as np

from my_scripts.gridworld_dagger_sim.chunk import (
    ChunkPolicy,
    evaluate_chunked,
)
from my_scripts.gridworld_dagger_sim.cli import resolve_initial_state
from my_scripts.gridworld_dagger_sim.core import (
    Grid,
    Policy,
    SimState,
    build_policy_catalog,
    evaluate,
)

# Default sweep grids — override at call time by editing the function.
DEFAULT_RATIOS: tuple[float, ...] = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0)
DEFAULT_NOISES: tuple[float, ...] = (0.0, 0.05, 0.15, 0.30)


def _eval(grid: Grid, policy: Policy, n_rollouts: int, seed: int) -> float:
    """Single-line success rate (deterministic given seed)."""
    rng = np.random.default_rng(seed)
    return evaluate(grid, policy, n_rollouts, rng)["succ_rate"]


def _eval_chunked(grid: Grid, policy: ChunkPolicy, n_rollouts: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    return evaluate_chunked(grid, policy, n_rollouts, rng)["succ_rate"]


def _print_table(title: str, header: list[str], rows: list[list[str]]) -> None:
    """Print a markdown table. Columns are right-aligned to header width."""
    widths = [max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(header)]
    print(f"\n## {title}")
    print("| " + " | ".join(h.rjust(w) for h, w in zip(header, widths, strict=False)) + " |")
    print("| " + " | ".join("-" * w for w in widths) + " |")
    for row in rows:
        print("| " + " | ".join(cell.rjust(w) for cell, w in zip(row, widths, strict=False)) + " |")


# ──────────────────────────────────────────────────────────────────────────────
# Experiment 1: blend ratio sweep, no noise
# ──────────────────────────────────────────────────────────────────────────────


def expt_blend_sweep(state: SimState, args: argparse.Namespace) -> None:
    """Baseline: simple blend ratio sweep with no policy noise.

    Tells you whether the blend curve is monotonic on this scenario.
    Convention: r=0 → pure intervention, r=1 → pure base.
    """
    cat = build_policy_catalog(state)
    if "intervention" not in cat or cat["intervention"].visitation.sum() == 0:
        print("\n## Experiment: blend_sweep — SKIPPED (state has no interventions)")
        return
    rows = []
    for r in DEFAULT_RATIOS:
        blended = Policy.blend(state.grid, cat["intervention"], cat["base"], ratio=r)
        succ = _eval(state.grid, blended, args.rollouts, args.seed)
        rows.append([f"{r:.2f}", f"{succ:.3f}"])
    _print_table(
        "Experiment 1: blend ratio sweep, no noise (r=0 → pure intervention, r=1 → pure base)",
        ["ratio", "succ_rate"],
        rows,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Experiment 2: blend ratio × policy noise (does noise create a sweet spot?)
# ──────────────────────────────────────────────────────────────────────────────


def expt_blend_x_noise(state: SimState, args: argparse.Namespace) -> None:
    """2D sweep: blend ratio × policy noise. The classic question — does
    adding per-step BC error make mid-range blend ratios beat the endpoints?
    (the "curriculum / label-smoothing" mechanism from the writeup)
    """
    cat = build_policy_catalog(state)
    if "intervention" not in cat or cat["intervention"].visitation.sum() == 0:
        print("\n## Experiment: blend_x_noise — SKIPPED (state has no interventions)")
        return
    rows: list[list[str]] = []
    for r in DEFAULT_RATIOS:
        b = Policy.blend(state.grid, cat["intervention"], cat["base"], ratio=r)
        row = [f"{r:.2f}"]
        for eps in DEFAULT_NOISES:
            soft = b.softened(state.grid, eps) if eps > 0 else b
            row.append(f"{_eval(state.grid, soft, args.rollouts, args.seed):.3f}")
        rows.append(row)
    header = ["ratio"] + [f"ε={e:.2f}" for e in DEFAULT_NOISES]
    _print_table(
        "Experiment 2: blend ratio × policy noise (success rate)",
        header,
        rows,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Experiment 3: simple Policy.blend vs BC-correct Policy.weighted_sum
# ──────────────────────────────────────────────────────────────────────────────


def expt_simple_vs_bc_correct(state: SimState, args: argparse.Namespace) -> None:
    """Side-by-side: does the choice of mixing operation matter for outcomes?

    Both should agree at r=0 and r=1; they diverge at mid ratios when
    intervention and base visit cells with unequal frequencies.
    """
    cat = build_policy_catalog(state)
    if "intervention" not in cat or cat["intervention"].visitation.sum() == 0:
        print("\n## Experiment: simple_vs_bc_correct — SKIPPED (no interventions)")
        return
    rows = []
    for r in DEFAULT_RATIOS:
        simple = Policy.blend(state.grid, cat["intervention"], cat["base"], ratio=r)
        bc = Policy.weighted_sum(state.grid, [(1.0 - r, cat["intervention"]), (r, cat["base"])])
        s_succ = _eval(state.grid, simple, args.rollouts, args.seed)
        b_succ = _eval(state.grid, bc, args.rollouts, args.seed)
        rows.append([f"{r:.2f}", f"{s_succ:.3f}", f"{b_succ:.3f}", f"{b_succ - s_succ:+.3f}"])
    _print_table(
        "Experiment 3: Policy.blend (simple) vs Policy.weighted_sum (BC-correct)",
        ["ratio", "simple", "BC-correct", "Δ (BC − simple)"],
        rows,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Experiment 4: intervention quality (perfect vs partial vs misleading)
# ──────────────────────────────────────────────────────────────────────────────


def expt_intervention_quality(state: SimState, args: argparse.Namespace) -> None:
    """How does the SHAPE of the intervention affect the blend curve?

    Constructs several intervention variants on top of the loaded state's
    base policy and sweeps blend ratios for each.
    """
    cat = build_policy_catalog(state)
    base = cat["base"]
    # All paths below are valid on the default 4×4 grid:
    #   o o o x
    #   o x o o
    #   o o o o
    #   x o o s
    variants: dict[str, list[list[tuple[int, int]]]] = {
        "perfect (R R D D R D → success)": [[(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 3), (3, 3)]],
        "diff path  (D D R R R D → success)": [[(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (2, 3), (3, 3)]],
        "partial    (R R D, stops at (1,2))": [[(0, 0), (0, 1), (0, 2), (1, 2)]],
        "partial-deep (R R D D R, stops at (2,3))": [[(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 3)]],
        "misleading (R R, stops at (0,2))": [
            # Stops one step short of the danger cell (0,3). Base from (0,2)
            # may or may not survive — depends on base's local action there.
            [(0, 0), (0, 1), (0, 2)]
        ],
    }
    short_ratios = (0.0, 0.3, 0.5, 0.7, 1.0)
    header = ["intervention"] + [f"r={r:.1f}" for r in short_ratios]
    rows = []
    for name, trajs in variants.items():
        iv = Policy.from_trajectories(state.grid, trajs, fallback=base)
        row = [name]
        for r in short_ratios:
            b = Policy.blend(state.grid, iv, base, ratio=r)
            row.append(f"{_eval(state.grid, b, args.rollouts, args.seed):.3f}")
        rows.append(row)
    _print_table(
        "Experiment 4: how intervention shape affects the blend sweep",
        header,
        rows,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Experiment 5: how dense the base policy is (effect of # demos)
# ──────────────────────────────────────────────────────────────────────────────


def expt_base_density(state: SimState, args: argparse.Namespace) -> None:
    """How does the # of base demos (state coverage) interact with blending?

    Constructs base policies with progressively more demos and sweeps blend
    ratios for each. Sparse base → uniform fallback dominates more cells →
    base is "worse" → intervention helps more.
    """
    cat = build_policy_catalog(state)
    if "intervention" not in cat or cat["intervention"].visitation.sum() == 0:
        print("\n## Experiment: base_density — SKIPPED (no interventions in state)")
        return
    iv = cat["intervention"]
    # Successful demos through different paths. We'll feed increasingly
    # larger subsets to from_trajectories so the base gets denser.
    demo_pool = [
        [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 3), (3, 3)],  # top→diagonal
        [(0, 0), (0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 3)],  # top→right col
        [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (2, 3), (3, 3)],  # left→bottom
        [(0, 0), (1, 0), (2, 0), (2, 1), (3, 1), (3, 2), (3, 3)],  # left→bottom row
        [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (3, 2), (3, 3)],  # top→bottom row
    ]
    short_ratios = (0.0, 0.3, 0.5, 0.7, 1.0)
    header = ["#base demos"] + [f"r={r:.1f}" for r in short_ratios]
    rows = []
    for n in (0, 1, 2, 3, 5):
        if n == 0:
            base = Policy.uniform(state.grid)
        else:
            base = Policy.from_trajectories(
                state.grid,
                demo_pool[:n],
                fallback=Policy.uniform(state.grid),
            )
        # Rebuild intervention with this new base as the fallback (so unvisited
        # intervention cells use this base, not the state's).
        if "intervention" in cat:
            iv_paths = [t["path"] for t in state.interventions]
            iv = Policy.from_trajectories(state.grid, iv_paths, fallback=base)
        row = [str(n)]
        for r in short_ratios:
            b = Policy.blend(state.grid, iv, base, ratio=r)
            row.append(f"{_eval(state.grid, b, args.rollouts, args.seed):.3f}")
        rows.append(row)
    _print_table(
        "Experiment 5: how base-demo density affects the blend sweep",
        header,
        rows,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Experiment 6: HG-DAgger rescue interventions
# ──────────────────────────────────────────────────────────────────────────────


def expt_hg_dagger_rescue(state: SimState, args: argparse.Namespace) -> None:
    """HG-DAgger scenario.

    Story
    -----
    Base policy is rolled out under per-step noise (modeling imperfect BC).
    Occasionally it deviates from its safe path into cells the base demos
    never visited — at those cells, the base policy is just `Policy.uniform`,
    so a coin-flip can push the rollout into a failure cell.

    A human watches: when the base deviates onto a dangerous cell, they take
    over and steer the policy back to a safe path. Only the takeover segment
    is recorded as an "intervention trajectory" — it does NOT start at (0,0).

    Training merges {base demos, rescue interventions} via BC-correct
    weighted_sum. At cells the base never visited but the rescue did, the
    trained policy uses the rescue's empirical (one-hot) action instead of
    uniform — closing the noise-deviation failure mode.

    Setup (uses the loaded state's `base_trajectories`; intervention is hardcoded
    here so the experiment works regardless of what's in the loaded state).

    Output: success rate vs noise ε for three configurations:
      (1) base alone
      (2) base + rescue, simple Policy.blend at r=0.5
      (3) base + rescue, BC-correct Policy.weighted_sum at equal weights

    Expectation: (3) wins under noise because BC-correct uses one-hot rescue
    actions at base-uncovered cells, while (1) has only uniform fallback and
    (2) only partially shifts toward the rescue (since it mixes 50/50 with
    base's uniform values, not 100% rescue).
    """
    grid = state.grid
    base = build_policy_catalog(state)["base"]

    # Rescue interventions. Each is a takeover trajectory that does NOT start
    # at (0,0) — it starts at the cell where the human grabbed control.
    # We pick cells the base never visits, where noise deviations cause
    # failures because base's policy there is uniform.
    rescue_paths: list[list[tuple[int, int]]] = [
        # Rescue starting at (0,1) — drive R toward the safe right column.
        [(0, 1), (0, 2), (1, 2), (2, 2), (2, 3), (3, 3)],
        # Rescue starting at (0,2) — drive D to escape the (0,3) failure.
        [(0, 2), (1, 2), (2, 2), (2, 3), (3, 3)],
        # Rescue starting at (1,2) — drive on toward success.
        [(1, 2), (2, 2), (2, 3), (3, 3)],
    ]
    rescue_iv = Policy.from_trajectories(grid, rescue_paths, fallback=base)

    # BC-correct mix (the right model for "BC training on base ∪ rescue").
    mixed_bc = Policy.weighted_sum(grid, [(0.5, base), (0.5, rescue_iv)])
    # Simple per-cell mix at r=0.5 for comparison.
    mixed_simple = Policy.blend(grid, rescue_iv, base, ratio=0.5)

    rows = []
    for eps in (0.0, 0.05, 0.15, 0.30, 0.50):

        def s(pol: Policy, _eps: float = eps) -> Policy:
            return pol.softened(grid, _eps) if _eps > 0 else pol

        b = _eval(grid, s(base), args.rollouts, args.seed)
        mb = _eval(grid, s(mixed_bc), args.rollouts, args.seed)
        ms = _eval(grid, s(mixed_simple), args.rollouts, args.seed)
        rows.append([f"{eps:.2f}", f"{b:.3f}", f"{mb:.3f}", f"{ms:.3f}", f"{mb - b:+.3f}"])

    _print_table(
        "Experiment 6: HG-DAgger rescue under noise "
        "(rescue interventions start mid-grid at noise-prone cells)",
        ["noise ε", "base alone", "+rescue (BC-correct)", "+rescue (simple r=0.5)", "Δ vs base"],
        rows,
    )

    # Bonus: which cells does the rescue actually help at? Print a small
    # before/after table of action probs at the rescue start cells.
    print("\n_(Per-cell view: action probabilities at rescue start cells)_\n")
    rescue_cells = sorted({path[0] for path in rescue_paths})
    sub_rows = []
    for r, c in rescue_cells:
        sub_rows.append(
            [
                f"({r},{c})",
                f"[{base.probs[r, c, 0]:.2f}, {base.probs[r, c, 1]:.2f}]",
                f"[{rescue_iv.probs[r, c, 0]:.2f}, {rescue_iv.probs[r, c, 1]:.2f}]",
                f"[{mixed_bc.probs[r, c, 0]:.2f}, {mixed_bc.probs[r, c, 1]:.2f}]",
                f"[{mixed_simple.probs[r, c, 0]:.2f}, {mixed_simple.probs[r, c, 1]:.2f}]",
            ]
        )
    _print_table(
        "[P(right), P(down)] at the cells where the rescue starts",
        ["cell", "base", "rescue", "BC-correct mix", "simple mix r=0.5"],
        sub_rows,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Experiment 7: single-step vs chunk-mode blends
# ──────────────────────────────────────────────────────────────────────────────


def expt_chunk_vs_single_step(state: SimState, args: argparse.Namespace) -> None:
    """Same demos, same blend ratios — single-step Policy vs K-step ChunkPolicy.

    The chunk policy is the well-defined discrete approximation to a
    continuous action vector: at each cell it stores a categorical
    distribution over K-step action trajectories, samples one, then
    executes its K actions before re-sampling at the landing cell.

    Why this is useful for DAgger blending: blending two K-step categoricals
    at r=0.5 produces a 50/50 mix between two SAFE multi-step plans, never
    an "average" chunk that's off-manifold. Single-step blending of two
    one-hot policies at r=0.5 produces a 50/50 categorical too, but each
    step is resampled independently, so a rollout can mix actions from
    two different demos cell-by-cell and end up off-path.
    """
    grid = state.grid
    cat = build_policy_catalog(state)
    if "intervention" not in cat or cat["intervention"].visitation.sum() == 0:
        print("\n## Experiment: chunk_vs_single_step — SKIPPED (no interventions)")
        return
    base_p = cat["base"]
    iv_p = cat["intervention"]

    base_trajs = [t["path"] for t in state.base_trajectories]
    iv_trajs = [t["path"] for t in state.interventions]
    chunk_k = int(getattr(args, "chunk_k", 3))
    base_c = ChunkPolicy.from_trajectories(grid, base_trajs, chunk_k=chunk_k)
    iv_c = ChunkPolicy.from_trajectories(
        grid,
        iv_trajs,
        chunk_k=chunk_k,
        fallback=base_c,
    )

    rows = []
    for r in DEFAULT_RATIOS:
        bp = Policy.blend(grid, iv_p, base_p, ratio=r)
        bc = ChunkPolicy.blend(grid, iv_c, base_c, ratio=r)
        s_single = _eval(grid, bp, args.rollouts, args.seed)
        s_chunk = _eval_chunked(grid, bc, args.rollouts, args.seed)
        rows.append([f"{r:.2f}", f"{s_single:.3f}", f"{s_chunk:.3f}", f"{s_chunk - s_single:+.3f}"])
    _print_table(
        f"Experiment 7: single-step vs chunk-mode (K={chunk_k}) blends",
        ["ratio", "single-step", f"chunk K={chunk_k}", "Δ (chunk − single)"],
        rows,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Registry + CLI
# ──────────────────────────────────────────────────────────────────────────────


EXPERIMENTS: dict[str, Callable[[SimState, argparse.Namespace], None]] = {
    "blend_sweep": expt_blend_sweep,
    "blend_x_noise": expt_blend_x_noise,
    "simple_vs_bc_correct": expt_simple_vs_bc_correct,
    "intervention_quality": expt_intervention_quality,
    "base_density": expt_base_density,
    "hg_dagger_rescue": expt_hg_dagger_rescue,
    "chunk_vs_single_step": expt_chunk_vs_single_step,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--state", type=str, default=None, help="Path to a state JSON. Default: examples/default_4x4.json."
    )
    parser.add_argument(
        "--rollouts", type=int, default=5000, help="Rollouts per policy evaluation (default 5000)."
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for deterministic results.")
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="*",
        default=None,
        choices=list(EXPERIMENTS.keys()),
        help="Which experiments to run. Default = all.",
    )
    parser.add_argument(
        "--chunk_k", type=int, default=3, help="Chunk length for chunk-mode experiments (default 3)."
    )
    args = parser.parse_args()

    state = resolve_initial_state(args.state)

    selected = args.experiments or list(EXPERIMENTS.keys())
    print(f"\n[experiments] rollouts={args.rollouts}, seed={args.seed}, chunk_k={args.chunk_k}")
    print(f"[experiments] running: {', '.join(selected)}")
    for name in selected:
        EXPERIMENTS[name](state, args)
    print()


if __name__ == "__main__":
    main()
