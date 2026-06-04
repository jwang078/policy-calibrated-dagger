# gridworld_dagger_sim

A tiny, modular, lerobot-independent sandbox for exploring blended-DAgger
semantics on a directed gridworld.

- 2D grid (default 4×4, configurable). Start = top-left. Each non-terminal
  cell has only `right` and `down` moves.
- **Single-step policy** (`core.Policy`) = a `(H, W, 2)` table of
  `[P(right), P(down)]`.
- **Multi-step "chunk" policy** (`chunk.ChunkPolicy`) = a `(H, W, 2^K)`
  table of categorical distributions over K-step action chunks. This is the
  well-defined discrete approximation to a continuous action vector — at
  each decision point the policy samples one of `2^K` discrete K-step
  plans and executes it before re-sampling.
- Node types: regular `o`, success `s`, failure `x` (terminal).
- Hardcoded editable base policy + expert intervention trajectories (drawn in GUI).
- BC-correct (visitation-weighted) policy mixing for blends and arbitrary
  weighted sums of sub-policies.
- Two driver modes: GUI (interactive editing + visualization) and CLI (batch
  rollouts, prints markdown tables).

Self-contained: only `numpy` + stdlib `tkinter`. No lerobot or torch imports.

## Quick start

```bash
# GUI (default)
python -m my_scripts.gridworld_dagger_sim

# Batch eval one policy
python -m my_scripts.gridworld_dagger_sim --no_gui --policy base --rollouts 10000 --seed 0

# Sweep blend ratios on the default state. Convention: r=0.0 → pure intervention,
# r=1.0 → pure base (matches DAgger's `forward_flow_ratio`).
python -m my_scripts.gridworld_dagger_sim --no_gui \
    --state my_scripts/gridworld_dagger_sim/examples/default_4x4.json \
    --sweep_blend 0.0 0.2 0.4 0.6 0.8 1.0

# Same sweep but with per-step BC error (compounding-error / DAgger story).
# With ε=0.15, pure intervention (blend_0.00) drops from 100% to ~80% — the
# policy occasionally deviates into unvisited cells where base takes over.
python -m my_scripts.gridworld_dagger_sim --no_gui \
    --state my_scripts/gridworld_dagger_sim/examples/default_4x4.json \
    --sweep_blend 0.0 0.2 0.4 0.6 0.8 1.0 --policy_noise 0.15

# Ad-hoc weighted mix (DAgger-style)
python -m my_scripts.gridworld_dagger_sim --no_gui \
    --state my_scripts/gridworld_dagger_sim/examples/default_4x4.json \
    --mix "base:0.7,intervention:0.15,base:0.15"
```

Unit tests:

```bash
python -m pytest my_scripts/gridworld_dagger_sim/test_core.py -v
python -m pytest my_scripts/gridworld_dagger_sim/test_chunk.py -v
```

Canned experiments (success-rate tables across several axes — blend ratio,
policy noise, intervention shape, base-demo density, chunk-mode comparison,
etc.):

```bash
python -m my_scripts.gridworld_dagger_sim.experiments
# or pick a subset:
python -m my_scripts.gridworld_dagger_sim.experiments \
    --experiments blend_x_noise chunk_vs_single_step
```

The experiments live in `experiments.py` — each is a function
`expt_<name>(state, args)` registered in `EXPERIMENTS`. Copy/edit any of
them to explore your own scenarios.

## Files

| File                        | Purpose                                                                                                                                                                    |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `core.py`                   | Single-step `Policy` math: `Grid`, `Policy`, `compute_visitation`, `weighted_sum`, `blend`, `rollout`, `evaluate`.                                                         |
| `chunk.py`                  | Multi-step `ChunkPolicy` math: chunk encoding, `uniform`/`from_trajectories`, `blend`/`weighted_sum`, `rollout_chunked`, `evaluate_chunked`, `compute_visitation_chunked`. |
| `gui.py`                    | tkinter GUI: 4 tabs (Edit Grid / Edit Interventions / Compose Policy / Rollouts) sharing one canvas. Single-step `Policy` only.                                            |
| `cli.py`                    | argparse + markdown-table batch output.                                                                                                                                    |
| `__main__.py`               | Dispatches GUI vs CLI based on `--no_gui`.                                                                                                                                 |
| `examples/default_4x4.json` | Default state: the user's `x x x s` bottom row + one top-right expert path.                                                                                                |
| `test_core.py`              | Pytest unit tests for the single-step math.                                                                                                                                |
| `test_chunk.py`             | Pytest unit tests for the chunk-mode math.                                                                                                                                 |

## Math: two distinct mixing operations

The sim exposes **two different policy-mixing primitives** that model
different stages of the DAgger pipeline. Both are available on `Policy`
(single-step) and on `ChunkPolicy` (multi-step); pick the right one for
the situation.

### 1. `Policy.blend(intervention, base, ratio)` — simple per-cell mix

> blend.probs[s] = (1 − r) · intervention.probs[s] + r · base.probs[s]

Convention: `r = 0` → pure intervention, `r = 1` → pure base (matches
`forward_flow_ratio` in `shared_autonomy_wrapper.py`).

**This models**:

- The SA wrapper's **INTERPOLATE** strategy
  (`shared_autonomy_wrapper.py:99-100`):
  `blended = ratio * policy_output + (1 - ratio) * guidance`. Each step's
  executed action is the linear combination of policy and expert outputs.
- The empirical action distribution of a **single closed-loop blend
  dataset** produced by acting the mixed policy at every step. After
  perfect BC training on that single dataset, the recovered policy is
  the simple linear mix at visited states.

It's a per-cell formula that ignores visitation entirely. It is **not**
the MLE estimator over a union of sub-datasets — that's the next
operation.

### 2. `Policy.weighted_sum([(w₁, π₁), (w₂, π₂), …])` — BC-correct mix

> π̂(a | s) = (Σᵢ wᵢ · vᵢ(s) · πᵢ(a | s)) / (Σᵢ wᵢ · vᵢ(s))

where `vᵢ(s)` is sub-dataset i's marginal visitation probability of state s.

**This models** perfect behavior cloning on a **weighted union of
independent sub-datasets** (e.g. base ∪ intervention ∪ multiple blend
datasets). It's the exact MLE estimator: cells visited more often by
sub-policy i get more "vote" from that sub-policy.

### When the two differ

At endpoints (`r=0` or `r=1`) and at cells where `v_intervention(s) =
v_base(s)`, the two give the same result. They diverge at cells where
visitation differs.

Concretely on the default grid with 3 R-/D-first base demos and a
single intervention through (1, 2):

| ratio | `blend(I, B, r).probs[1,2]` | `weighted_sum([(1-r, I), (r, B)]).probs[1,2]` |
| ----: | --------------------------- | --------------------------------------------- |
|   0.0 | `[0.00, 1.00]`              | `[0.00, 1.00]`                                |
|   0.2 | `[0.10, 0.90]`              | `[0.07, 0.93]`                                |
|   0.4 | `[0.20, 0.80]`              | `[0.15, 0.85]`                                |
|   0.6 | `[0.30, 0.70]`              | `[0.25, 0.75]`                                |
|   0.8 | `[0.40, 0.60]`              | `[0.36, 0.64]`                                |
|   1.0 | `[0.50, 0.50]`              | `[0.50, 0.50]`                                |

BC-correct skews toward intervention at (1,2) because `v_int(1,2) = 1.0`
but `v_base(1,2) = 2/3` — intervention "visits more often" so it gets a
larger share of the mix.

### Modeling the full DAgger pipeline

Compose the two:

```python
# Each blend dataset is a single closed-loop rollout under (1-r)·expert + r·policy.
# Its empirical policy is the simple per-cell mix.
blend_03 = Policy.blend(intervention, base, 0.3)

# Final trained policy: BC on the union of {base, intervention, blend_0.3}.
# Use BC-correct visitation-weighted across the union.
trained = Policy.weighted_sum(grid, [
    (w_base,         base),
    (w_intervention, intervention),
    (w_blend03,      blend_03),
])
```

The DAgger pipeline uses operation (1) to generate blend datasets and
operation (2) to train on their weighted union.

### How visitation is computed

- For the **base policy** (and any analytical policy with action probs
  everywhere): forward DP on the DAG. `v(0,0) = 1`;
  `v(r,c) = v(r-1,c)·P(down | r-1,c) + v(r,c-1)·P(right | r,c-1)`
  (with predecessors that are terminal contributing 0).
- For the **intervention policy** derived from K trajectories:
  `v(s) = (# trajectories through s) / K`. Unvisited cells get `v = 0`
  and their `probs` row is copied from base (irrelevant under BC math
  because v=0, but keeps the table interpretable).
- For any **composition** produced by `weighted_sum`: forward-DP again,
  using the resulting policy's action probs. Compositions can therefore
  be reused as inputs to further compositions.

## Chunk-mode policies (`chunk.ChunkPolicy`)

The single-step `Policy` is a categorical distribution over `{R, D}` at
each cell. To model the action-chunk architecture of real policies (the
SA wrapper emits chunks of N=8 EE-delta vectors at a time), `chunk.py`
gives you a categorical distribution over `{R, D}^K` at each cell:

```python
from my_scripts.gridworld_dagger_sim.chunk import ChunkPolicy, rollout_chunked

base = ChunkPolicy.from_trajectories(grid, base_demos, chunk_k=3)
iv   = ChunkPolicy.from_trajectories(grid, iv_demos, chunk_k=3, fallback=base)
blended = ChunkPolicy.blend(grid, iv, base, ratio=0.5)
path, outcome = rollout_chunked(grid, blended, np.random.default_rng(0))
```

### Rollout semantics

At each decision point the agent samples one chunk from
`chunk_probs[r, c]` (a 2^K-way categorical), executes its K actions
in sequence, and arrives at the landing cell K steps later. If a
chunk hits a terminal mid-execution it stops there; otherwise the
agent re-samples at the landing cell. Decision points are therefore
the start cell plus every cell that's the landing point of a full
K-step chunk — not every cell along the trajectory.

### Why chunks are the well-defined discrete approximation to continuous

A continuous "blend" of two action vectors (`(1, 0)` and `(0, 1)` at
r=0.5) produces a diagonal `(0.5, 0.5)` that doesn't correspond to any
demo action — it's off-manifold and can land the agent on a failure
cell that neither demo would touch. Categorical chunk blending side-
steps this: if base says `{"DDR": 1.0}` and intervention says
`{"RRD": 1.0}`, blending at r=0.5 gives `{"DDR": 0.5, "RRD": 0.5}`.
The agent stochastically picks one safe multi-step plan or the other,
never an "average" plan, because no such average exists in the discrete
action space.

This is the same mechanism action-chunk diffusion policies use to
keep blended actions on the data manifold, without the additional
machinery of noise + projection.

### Visitation in chunk mode

`compute_visitation_chunked` does a forward DP over the chunk DAG with
**decision-point visitation** tracked separately from **total cell
visitation**: only decision-point cells spawn new chunks, while every
cell touched along a chunk's execution accumulates total-visit mass.
This avoids the obvious double-counting bug of treating every visited
cell as a re-decision point.

The total cell visitation is returned (matching the "data density"
semantics of `from_trajectories`) and used as the weighting factor in
`ChunkPolicy.weighted_sum`.

## GUI walkthrough

The window has a shared canvas on the left and five tabs:

### Tab 1 — Edit Grid

- Click a cell: cycles `REGULAR → FAILURE → SUCCESS → REGULAR`.
- The bottom-right cell is locked to non-`REGULAR` (preserves the leaf
  invariant; clicks cycle `SUCCESS ↔ FAILURE`).
- Resize, Reset, Save/Load JSON. "Show visitation overlay" tints each cell
  by its visit probability under the currently-displayed policy.

### Tab 2 — Edit Base Policy (from demos)

- Sets the **base policy** by drawing a few demo trajectories — same workflow
  as the intervention tab. The base policy is then the empirical action
  distribution at visited cells, falling back to `Policy.uniform` (i.e.
  uniform over the cell's valid actions) at unvisited cells.
- "New base demo" starts a draw. Click adjacent cells (right/down only) to
  extend; "Finish demo" commits any path with ≥1 forward step. Right-click
  cancels. Partial demos (ending mid-grid) are allowed and get the
  `_partial` suffix in their name — base falls back to uniform from the
  endpoint onward.
- Multiple demos that disagree at a state produce a stochastic policy at
  that state (the empirical mixture).
- Code-level defaults: `DEFAULT_BASE_TRAJECTORIES` in `core.py` defines
  the demos used by `default_state()`.

### Tab 3 — Edit Interventions

- "New trajectory" starts a draw. Clicking adjacent cells (right or down
  only) extends a dashed line. Right-click cancels.
- "Finish trajectory" commits the current path. Available **as soon as
  you've taken at least one step** — the trajectory does _not_ have to
  reach a terminal cell. Partial trajectories (ending mid-grid) are saved
  with a `_partial` suffix in the name.
- At a partial endpoint, the intervention "policy" has no recorded action,
  so it falls back to the base policy at that cell — base takes over
  from the endpoint. This lets you experiment with "expert helps you
  partway but doesn't finish the job."
- The list shows committed trajectories with their colors. Delete to remove.

### Tab 4 — Compose Policy

- Add rows `[policy_name] [weight slider]`. Resulting policy renders live
  with arrow widths ∝ probability and a 1k-rollout `succ_rate` estimate.
- "Add blend (slider)" pre-fills two coupled rows so one slider controls
  `r · intervention + (1-r) · base` — the natural blend visualization.
- Name + "Save composition" stores it; saved compositions appear as policy
  names in subsequent rows and in Tab 4.

### Tab 5 — Rollouts

- Pick a policy (base / intervention / any saved composition).
- Step-through: "Next rollout" plays one episode (drawn as a polyline,
  current cell highlighted, tally updated). Auto-play toggles continuous
  rollouts at a configurable speed.
- Batch: enter N, click "Run batch", get an aggregate `succ_rate` /
  `fail_rate` block.
- **Policy noise ε** slider (0.0–1.0). Before rollout, the selected policy
  is softened toward uniform-over-valid-actions:
  `probs' = (1-ε)·probs + ε·uniform`. Models per-step BC error
  (function approximation, finite-sample bias, optimization noise);
  exposes the compounding-error / DAgger-motivation story: with
  deterministic intervention and `ε > 0`, success decays roughly like
  `(1-ε)^path_length` instead of staying at 100%.

The GUI only drives single-step `Policy` math. Chunk-mode comparisons
are CLI / experiments-only for now — see `expt_chunk_vs_single_step`.

## Editing defaults in code

`core.py` exposes editable constants at the top:

```python
DEFAULT_GRID_ASCII = """\
o o o x
o x o o
o o o o
x o o s
"""

# Base-policy demos used by default_state(): a few successful paths.
DEFAULT_BASE_TRAJECTORIES = (
    ((0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 3), (3, 3)),  # R R D D R D
    ((0, 0), (0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 3)),  # R R D R D D
    ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (2, 3), (3, 3)),  # D D R R R D
)
DEFAULT_BLEND_RATIOS = (0.1, 0.3, 0.5, 0.7, 0.9)
```

The default grid scatters failures at `(0,3)`, `(1,1)`, and `(3,0)` so each
naive strategy dies:

- "always right" hits `(0,3)` on move 3
- "always down" hits `(3,0)` on move 3
- mixed early (`RD` or `DR`) hits `(1,1)` on move 2

The three default base demos thread the safe region (top, top-right column,
and bottom-left routes), so the resulting base policy hits ~100% success.
To stress-test the base, use **policy noise** (Tab 5 slider / `--policy_noise`)
or define fewer / less-covering base demos so the uniform fallback dominates
more cells.

## State JSON schema

```json
{
  "grid": {
    "height": 4,
    "width": 4,
    "ascii": "o o o x\no x o o\no o o o\nx o o s"
  },
  "base_trajectories": [
    {
      "name": "base_demo_1",
      "path": [
        [0, 0],
        [0, 1],
        [0, 2],
        [1, 2],
        [2, 2],
        [2, 3],
        [3, 3]
      ]
    },
    {
      "name": "base_demo_2",
      "path": [
        [0, 0],
        [0, 1],
        [0, 2],
        [1, 2],
        [1, 3],
        [2, 3],
        [3, 3]
      ]
    },
    {
      "name": "base_demo_3",
      "path": [
        [0, 0],
        [1, 0],
        [2, 0],
        [2, 1],
        [2, 2],
        [2, 3],
        [3, 3]
      ]
    }
  ],
  "interventions": [
    {
      "name": "expert_diagonal",
      "path": [
        [0, 0],
        [0, 1],
        [0, 2],
        [1, 2],
        [2, 2],
        [2, 3],
        [3, 3]
      ]
    }
  ],
  "compositions": [
    {
      "name": "blend_0.4",
      "rows": [
        { "policy": "base", "weight": 0.4 },
        { "policy": "intervention", "weight": 0.6 }
      ]
    }
  ]
}
```

Compositions can reference earlier compositions by name (resolved in
topological order; cycles raise).

## What this DOES NOT include

- A multi-round DAgger orchestrator. The simulator gives you the
  primitives (intervention → blend → weighted mix → trained policy →
  rollouts); a multi-round loop is a separate script you can build on
  top of `core.py`.
- Continuous action vectors / DENOISE projection. Earlier versions
  carried `PolicyMode.CONTINUOUS` and `PolicyMode.DENOISE` as alternate
  ways to model "blended actions land off-manifold"; both were dropped
  in favor of chunk policies, which are the cleanly discrete analogue.
- Matplotlib. All visualization is in tk Canvas primitives.
- Undo/redo in the editors.
