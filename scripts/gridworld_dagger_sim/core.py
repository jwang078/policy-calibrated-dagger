"""Pure logic for the gridworld DAgger simulator.

Models a small directed grid with stochastic stateless policies, expert
intervention trajectories, and BC-correct (visitation-weighted) policy mixing.

No tkinter, no lerobot imports. stdlib + numpy only. Lift this file out and
it still runs standalone.

Action semantics: this module's `Policy` is *single-step discrete*. At every
cell the policy is a categorical distribution `[P(right), P(down)]`, and a
rollout samples one action per cell. The multi-step "chunk" policy that's
the well-defined discrete approximation to continuous actions lives in
`chunk.py` (`ChunkPolicy` etc.).

See README.md for the math and high-level walkthrough.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

import numpy as np

# ─────────────────────────── User-editable defaults ──────────────────────────

DEFAULT_GRID_ASCII = """\
o o o x
o x o o
o o o o
x o o s
"""
# Failure cells (x) deliberately placed so each "naive" strategy dies:
#   * "always right" hits (0,3) on move 3
#   * "always down"  hits (3,0) on move 3
#   * "right-then-down" or "down-then-right" early hit (1,1) on move 2
# Under DEFAULT_BASE_POLICY_PROBS (0.55, 0.45) success is ~25%, vs ~60% on a
# clean grid — enough room for an expert intervention to demonstrate gains.

# Default base-policy trajectories: a few demos that all reach success on the
# DEFAULT_GRID_ASCII. The base policy is the empirical action distribution
# from these (Policy.from_trajectories with fallback=Policy.uniform).
# Edit / add more trajectories freely to shape the default base.
DEFAULT_BASE_TRAJECTORIES: tuple[tuple[tuple[int, int], ...], ...] = (
    # R R D D R D — top-then-diagonal
    ((0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 3), (3, 3)),
    # R R D R D D — top-then-right-column
    ((0, 0), (0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 3)),
    # D D R R R D — left-then-bottom-row
    ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (2, 3), (3, 3)),
)

DEFAULT_BLEND_RATIOS = (0.1, 0.3, 0.5, 0.7, 0.9)


# ────────────────────────────────── Types ────────────────────────────────────


class NodeType(IntEnum):
    REGULAR = 0
    SUCCESS = 1
    FAILURE = 2


class Action(IntEnum):
    RIGHT = 0
    DOWN = 1


_ASCII_TO_TYPE = {"o": NodeType.REGULAR, "s": NodeType.SUCCESS, "x": NodeType.FAILURE}
_TYPE_TO_ASCII = {v: k for k, v in _ASCII_TO_TYPE.items()}


@dataclass
class Grid:
    height: int
    width: int
    node_types: np.ndarray  # (H, W) int, NodeType values

    @classmethod
    def from_ascii(cls, text: str) -> Grid:
        rows = [r.strip() for r in text.strip().splitlines() if r.strip()]
        if not rows:
            raise ValueError("Empty grid ASCII.")
        height = len(rows)
        # Cells are whitespace-separated single chars: "o o o o".
        parsed = [row.split() for row in rows]
        width = len(parsed[0])
        if any(len(r) != width for r in parsed):
            raise ValueError(f"Inconsistent row widths in ASCII grid: {[len(r) for r in parsed]}")
        nt = np.zeros((height, width), dtype=np.int8)
        for r, row in enumerate(parsed):
            for c, cell in enumerate(row):
                key = cell.lower()
                if key not in _ASCII_TO_TYPE:
                    raise ValueError(f"Bad cell '{cell}' at ({r},{c}). Use o/s/x.")
                nt[r, c] = int(_ASCII_TO_TYPE[key])
        g = cls(height=height, width=width, node_types=nt)
        g.validate()
        return g

    def to_ascii(self) -> str:
        lines = []
        for r in range(self.height):
            cells = [_TYPE_TO_ASCII[NodeType(int(self.node_types[r, c]))] for c in range(self.width)]
            lines.append(" ".join(cells))
        return "\n".join(lines)

    def validate(self) -> None:
        # Bottom-right has no outgoing moves; it must be terminal-by-type so
        # every rollout-end is either SUCCESS or FAILURE per the user invariant.
        br = NodeType(int(self.node_types[self.height - 1, self.width - 1]))
        if br == NodeType.REGULAR:
            raise ValueError(
                "Bottom-right cell must be SUCCESS or FAILURE (it's a non-terminal "
                "leaf otherwise — the user invariant says all leaves are s or x)."
            )

    def outgoing(self, r: int, c: int) -> list[Action]:
        if self.is_terminal(r, c):
            return []
        moves: list[Action] = []
        if c + 1 < self.width:
            moves.append(Action.RIGHT)
        if r + 1 < self.height:
            moves.append(Action.DOWN)
        return moves

    def is_terminal(self, r: int, c: int) -> bool:
        nt = NodeType(int(self.node_types[r, c]))
        if nt != NodeType.REGULAR:
            return True
        # REGULAR but no exits → bottom-right (validated to be non-regular above).
        # Should not happen unless validate() was skipped; guard anyway.
        return (c + 1 >= self.width) and (r + 1 >= self.height)


# ───────────────────── Visitation forward DP (DAG order) ─────────────────────


def compute_visitation(grid: Grid, probs: np.ndarray) -> np.ndarray:
    """Forward-DP marginal visitation under `probs` starting from (0, 0).

    `probs[r, c]` is read as a categorical `[P(right), P(down)]`. Terminal
    cells receive visit mass but don't propagate it onward. Returns float64
    (H, W) in [0, 1].
    """
    H, W = grid.height, grid.width
    v = np.zeros((H, W), dtype=np.float64)
    v[0, 0] = 1.0
    # Process in row-major order; since edges only go right or down, every
    # predecessor of (r,c) is at (r-1,c) or (r,c-1), already filled.
    for r in range(H):
        for c in range(W):
            if r == 0 and c == 0:
                continue  # already initialized
            acc = 0.0
            if r > 0 and not grid.is_terminal(r - 1, c):
                # came from above via DOWN
                acc += v[r - 1, c] * probs[r - 1, c, Action.DOWN]
            if c > 0 and not grid.is_terminal(r, c - 1):
                # came from left via RIGHT
                acc += v[r, c - 1] * probs[r, c - 1, Action.RIGHT]
            v[r, c] = acc
    return v


# ────────────────────────────────── Policy ───────────────────────────────────


@dataclass
class Policy:
    """Single-step discrete policy.

    `probs[r, c]` = categorical `[P(right), P(down)]`. Rollout samples one
    action per cell. For the multi-step "chunk" policy (the well-defined
    discrete approximation to a continuous action vector) see `chunk.py`.

    `visitation` carries either data density (from `from_trajectories`) or
    the analytical marginal visit probability (from `uniform` / blends).
    BC-correct mixing in `weighted_sum` uses this as the per-state weight.
    """

    probs: np.ndarray  # (H, W, 2) — [P(right), P(down)]
    visitation: np.ndarray  # (H, W)  — data density or marginal visit

    # ── Constructors ────────────────────────────────────────────────────────

    @classmethod
    def uniform(cls, grid: Grid) -> Policy:
        """Uniform-over-valid-actions at every non-terminal cell.

        Interior cells (both RIGHT and DOWN valid) → [0.5, 0.5].
        Boundary cells (only one direction valid) → one-hot toward valid action.
        Terminal cells → all-zero probs (never sampled).
        """
        H, W = grid.height, grid.width
        probs = np.zeros((H, W, 2), dtype=np.float64)
        for r in range(H):
            for c in range(W):
                if grid.is_terminal(r, c):
                    continue
                outs = grid.outgoing(r, c)
                for a in outs:
                    probs[r, c, int(a)] = 1.0 / len(outs)
        return cls(probs=probs, visitation=compute_visitation(grid, probs))

    @classmethod
    def from_trajectories(
        cls,
        grid: Grid,
        trajectories: list[list[tuple[int, int]]],
        fallback: Policy | None = None,
    ) -> Policy:
        """Build a policy from a set of trajectories.

        - **Visited cells**: empirical action distribution across the input
          trajectories (one-hot if a single trajectory is given; mixed if
          multiple trajectories disagree at a state).
        - **Unvisited cells**: `fallback.probs` if provided, else uniform-
          over-valid-actions (`Policy.uniform(grid).probs`).
        - **Visitation**: empirical frequency = (# trajectories that visit
          cell) / K. This is data density, not the analytical marginal visit;
          BC-correct mixing wants this.
        """
        if fallback is None:
            fallback = cls.uniform(grid)
        H, W = grid.height, grid.width
        K = max(len(trajectories), 1)
        action_counts = np.zeros((H, W, 2), dtype=np.float64)
        visit_counts = np.zeros((H, W), dtype=np.float64)
        for traj in trajectories:
            if not traj:
                continue
            visited_in_traj = set()
            for i in range(len(traj) - 1):
                r, c = traj[i]
                nr, nc = traj[i + 1]
                if (r, c) not in visited_in_traj:
                    visit_counts[r, c] += 1
                    visited_in_traj.add((r, c))
                if nr == r and nc == c + 1:
                    action_counts[r, c, Action.RIGHT] += 1
                elif nc == c and nr == r + 1:
                    action_counts[r, c, Action.DOWN] += 1
                else:
                    raise ValueError(
                        f"Invalid trajectory step {(r, c)} -> {(nr, nc)} (must be right or down)."
                    )
            # Final node visit (no action recorded).
            r, c = traj[-1]
            if (r, c) not in visited_in_traj:
                visit_counts[r, c] += 1
        # Build probs: visited cells from action_counts, unvisited copy fallback.
        probs = np.array(fallback.probs, copy=True)
        sums = action_counts.sum(axis=-1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            empirical = np.where(sums > 0, action_counts / sums, 0.0)
        has_actions = sums.squeeze(-1) > 0
        probs[has_actions] = empirical[has_actions]
        visitation = visit_counts / K
        return cls(probs=probs, visitation=visitation)

    # ── Mixing ──────────────────────────────────────────────────────────────

    @classmethod
    def weighted_sum(
        cls,
        grid: Grid,
        weighted: list[tuple[float, Policy]],
    ) -> Policy:
        """BC-correct (visitation-weighted) mix of sub-policies.

            π̂(a|s) = (Σ wᵢ · vᵢ(s) · πᵢ(a|s)) / (Σ wᵢ · vᵢ(s))

        Weights are normalized to sum=1 first. Cells where the denominator
        is zero (no sub-policy visits) fall back to the first input's probs
        as a defensive default.
        """
        if not weighted:
            raise ValueError("weighted_sum needs at least one (weight, policy) pair.")
        weights = np.array([w for w, _ in weighted], dtype=np.float64)
        total = weights.sum()
        if total <= 0:
            raise ValueError("Weights must sum to a positive value.")
        weights = weights / total
        H, W = grid.height, grid.width
        num = np.zeros((H, W, 2), dtype=np.float64)
        denom = np.zeros((H, W), dtype=np.float64)
        for w, pol in zip(weights, [p for _, p in weighted], strict=False):
            wv = w * pol.visitation  # (H, W)
            denom += wv
            num += wv[..., None] * pol.probs
        with np.errstate(invalid="ignore", divide="ignore"):
            probs = np.where(denom[..., None] > 0, num / denom[..., None], 0.0)
        # Fallback at no-visitation cells: copy first input's probs (terminals
        # included, where probs are zero anyway — never sampled).
        zero_mask = denom <= 0
        if zero_mask.any():
            probs[zero_mask] = weighted[0][1].probs[zero_mask]
        v = compute_visitation(grid, probs)
        return cls(probs=probs, visitation=v)

    @classmethod
    def blend(
        cls,
        grid: Grid,
        intervention: Policy,
        base: Policy,
        ratio: float,
    ) -> Policy:
        """Simple per-cell linear mix of action distributions:

            blend.probs[s] = (1 - ratio) · intervention.probs[s]
                              +     ratio · base.probs[s]

        Convention (matches SplatSim's `forward_flow_ratio`):
            ratio = 0.0  →  pure intervention (all expert guidance)
            ratio = 1.0  →  pure base policy  (full policy flow)

        **This is NOT visitation-weighted.** It models:
          - the SA wrapper's INTERPOLATE strategy
            (`shared_autonomy_wrapper.py:99-100`):
                blended = ratio * policy_output + (1 - ratio) * guidance
          - the empirical action distribution of a single closed-loop blend
            dataset generated by acting the mixed policy at every step.

        For modeling **perfect BC training on a UNION of independent
        sub-datasets** (e.g. base ∪ intervention ∪ several blend datasets),
        use `Policy.weighted_sum(...)` instead — that's visitation-weighted
        and is the right operation for that distinct scenario.

        Visitation is recomputed via forward DP on the resulting probs so
        the returned policy is usable as an input to further compositions.
        """
        if not (0.0 <= ratio <= 1.0):
            raise ValueError(f"ratio must be in [0, 1], got {ratio}")
        probs = (1.0 - ratio) * intervention.probs + ratio * base.probs
        return cls(probs=probs, visitation=compute_visitation(grid, probs))

    def softened(self, grid: Grid, noise: float) -> Policy:
        """Soften the per-cell action distribution toward uniform-over-valid-actions:

            probs' = (1 - noise) · probs + noise · uniform_over_valid_actions

        Models the per-step deviation probability that real BC training has
        (function-approximation error, finite-sample bias, optimization noise)
        and exposes the compounding-error problem that motivates DAgger: with
        a deterministic one-hot intervention policy and `noise=ε`, success
        decays roughly like `(1-ε)^path_length` instead of staying at 100%.

        Boundary cells (only one valid outgoing direction) are unchanged —
        uniform-over-valid is itself one-hot there, so noise has nothing to
        spread to. Only interior cells (both RIGHT and DOWN valid) actually
        get smoothed.
        """
        if not (0.0 <= noise <= 1.0):
            raise ValueError(f"noise must be in [0, 1], got {noise}")
        if noise == 0.0:
            return Policy(
                probs=np.array(self.probs, copy=True),
                visitation=np.array(self.visitation, copy=True),
            )
        H, W = grid.height, grid.width
        new_probs = np.zeros((H, W, 2), dtype=np.float64)
        for r in range(H):
            for c in range(W):
                if grid.is_terminal(r, c):
                    continue
                outs = grid.outgoing(r, c)
                uniform = np.zeros(2, dtype=np.float64)
                for a in outs:
                    uniform[int(a)] = 1.0 / len(outs)
                new_probs[r, c] = (1.0 - noise) * self.probs[r, c] + noise * uniform
        v = compute_visitation(grid, new_probs)
        return Policy(probs=new_probs, visitation=v)


# ────────────────────────────── Rollout / eval ───────────────────────────────


def rollout(
    grid: Grid,
    policy: Policy,
    rng: np.random.Generator,
) -> tuple[list[tuple[int, int]], NodeType]:
    """Single episode. Returns (cell_path, terminal NodeType).

    Stochastic categorical sampling at each non-terminal cell. At each cell,
    samples an action from `policy.probs[r, c]` restricted to the cell's
    valid outgoing directions, then steps to the corresponding adjacent
    cell. Terminates on success / failure / off-grid.
    """
    r, c = 0, 0
    path: list[tuple[int, int]] = [(0, 0)]
    while not grid.is_terminal(r, c):
        outs = grid.outgoing(r, c)
        if len(outs) == 1:
            a = outs[0]
        else:
            p = np.array([policy.probs[r, c, int(o)] for o in outs], dtype=np.float64)
            s = p.sum()
            if s <= 0:
                p = np.ones(len(outs)) / len(outs)
            else:
                p = p / s
            a = outs[int(rng.choice(len(outs), p=p))]
        if a == Action.RIGHT:
            c += 1
        else:
            r += 1
        path.append((r, c))
    return path, NodeType(int(grid.node_types[r, c]))


def evaluate(
    grid: Grid,
    policy: Policy,
    n_rollouts: int,
    rng: np.random.Generator,
) -> dict:
    """Batch rollouts. Returns aggregate counts + success / failure rates."""
    succ = 0
    fail = 0
    for _ in range(n_rollouts):
        _, outcome = rollout(grid, policy, rng)
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


# ──────────────────────────────── State I/O ──────────────────────────────────


@dataclass
class SimState:
    """Bag of editable state, used for JSON round-trips between GUI and CLI."""

    grid: Grid
    # Base-policy trajectories (demos). The base policy is the empirical action
    # distribution from these via `Policy.from_trajectories(fallback=uniform)`.
    base_trajectories: list[dict] = field(default_factory=list)
    # Expert interventions for DAgger. Fed into `from_trajectories(fallback=base)`.
    interventions: list[dict] = field(default_factory=list)
    # Saved policy compositions: lists of (policy_name, weight) tuples.
    compositions: list[dict] = field(default_factory=list)

    def to_json_obj(self) -> dict:
        return {
            "grid": {
                "height": self.grid.height,
                "width": self.grid.width,
                "ascii": self.grid.to_ascii(),
            },
            "base_trajectories": [
                {"name": t["name"], "path": [list(p) for p in t["path"]]} for t in self.base_trajectories
            ],
            "interventions": [
                {"name": iv["name"], "path": [list(p) for p in iv["path"]]} for iv in self.interventions
            ],
            "compositions": [
                {
                    "name": c["name"],
                    "rows": [{"policy": r["policy"], "weight": float(r["weight"])} for r in c["rows"]],
                }
                for c in self.compositions
            ],
        }

    @classmethod
    def from_json_obj(cls, obj: dict) -> SimState:
        g = Grid.from_ascii(obj["grid"]["ascii"])
        bases = []
        for t in obj.get("base_trajectories", []):
            bases.append({"name": t["name"], "path": [tuple(p) for p in t["path"]]})
        ivs = []
        for iv in obj.get("interventions", []):
            ivs.append({"name": iv["name"], "path": [tuple(p) for p in iv["path"]]})
        comps = []
        for c in obj.get("compositions", []):
            comps.append(
                {
                    "name": c["name"],
                    "rows": [{"policy": r["policy"], "weight": float(r["weight"])} for r in c["rows"]],
                }
            )
        return cls(grid=g, base_trajectories=bases, interventions=ivs, compositions=comps)


def save_state(path: str | Path, state: SimState) -> None:
    Path(path).write_text(json.dumps(state.to_json_obj(), indent=2))


def load_state(path: str | Path) -> SimState:
    return SimState.from_json_obj(json.loads(Path(path).read_text()))


def default_state() -> SimState:
    """Build a fresh default state with example base demos.

    `base_trajectories` is pre-populated from `DEFAULT_BASE_TRAJECTORIES` —
    a few successful paths through the default grid. The base policy is the
    empirical action distribution from those demos (visited cells) plus
    `Policy.uniform` at unvisited cells.
    """
    grid = Grid.from_ascii(DEFAULT_GRID_ASCII)
    base_trajs = [
        {"name": f"base_demo_{i + 1}", "path": [tuple(p) for p in path]}
        for i, path in enumerate(DEFAULT_BASE_TRAJECTORIES)
    ]
    return SimState(
        grid=grid,
        base_trajectories=base_trajs,
        interventions=[],
        compositions=[],
    )


# ───────────────────── Convenience: build named policy set ───────────────────


def build_policy_catalog(state: SimState) -> dict[str, Policy]:
    """Resolve the named policies present in `state`: base, intervention, and
    every saved composition (composed bottom-up via topological order on names).

    - **base**: `Policy.from_trajectories(grid, base_trajectories, fallback=uniform)`
    - **intervention**: `Policy.from_trajectories(grid, interventions, fallback=base)`
    - **compositions**: BC-correct visitation-weighted mixes of any earlier
      policies (recursive; declaration order with topological resolution).
    """
    if state.base_trajectories:
        base_trajs = [t["path"] for t in state.base_trajectories]
        base = Policy.from_trajectories(
            state.grid,
            base_trajs,
            fallback=Policy.uniform(state.grid),
        )
    else:
        # No base demos → base IS uniform.
        base = Policy.uniform(state.grid)
    catalog: dict[str, Policy] = {"base": base}

    if state.interventions:
        trajs = [iv["path"] for iv in state.interventions]
        intervention = Policy.from_trajectories(state.grid, trajs, fallback=base)
    else:
        # No interventions: intervention "policy" copies base probs but has
        # zero visitation, so it contributes nothing to BC-correct mixes.
        H, W = state.grid.height, state.grid.width
        intervention = Policy(
            probs=np.array(base.probs, copy=True),
            visitation=np.zeros((H, W)),
        )
    catalog["intervention"] = intervention

    # Resolve compositions in declaration order; if a row references a name
    # that isn't yet in catalog, defer it. Detect cycles by counting passes
    # without progress.
    pending = list(state.compositions)
    while pending:
        progressed = False
        remaining = []
        for comp in pending:
            needed = [row["policy"] for row in comp["rows"]]
            if all(n in catalog for n in needed):
                weighted = [(float(row["weight"]), catalog[row["policy"]]) for row in comp["rows"]]
                catalog[comp["name"]] = Policy.weighted_sum(state.grid, weighted)
                progressed = True
            else:
                remaining.append(comp)
        if not progressed:
            missing = {
                n for comp in remaining for n in [row["policy"] for row in comp["rows"]] if n not in catalog
            }
            raise ValueError(
                f"Cannot resolve compositions {[c['name'] for c in remaining]}: "
                f"missing or cyclic references to {sorted(missing)}"
            )
        pending = remaining
    return catalog
