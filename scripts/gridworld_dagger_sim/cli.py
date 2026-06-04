"""Batch-mode CLI for the gridworld DAgger simulator.

Run via `python -m my_scripts.gridworld_dagger_sim --no_gui ...`.

Modes:
  --policy NAME           Evaluate one named policy.
  --sweep_blend r1 r2 ... Evaluate base + blend(intervention, base, ratio) for each ratio.
  --mix "A:wa,B:wb,..."   Evaluate one ad-hoc weighted mix.

All modes print a markdown-friendly results table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from my_scripts.gridworld_dagger_sim.core import (
    DEFAULT_BLEND_RATIOS,
    Policy,
    SimState,
    build_policy_catalog,
    default_state,
    evaluate,
    load_state,
)


def _examples_default_path() -> Path:
    """Conventional path the launcher checks for an implicit default state."""
    return Path(__file__).parent / "examples" / "default_4x4.json"


def resolve_initial_state(explicit_state_path: str | Path | None) -> SimState:
    """Resolve the SimState to launch with.

    Precedence:
      1. `--state PATH` if provided (errors if missing).
      2. `examples/default_4x4.json` if it exists on disk — so 'Save to examples…'
         with the name 'default_4x4' persists across relaunches.
      3. The hardcoded `default_state()` built from `DEFAULT_BASE_TRAJECTORIES`.

    Prints which source was used so the user isn't surprised.
    """
    if explicit_state_path is not None:
        print(f"[gridworld_dagger_sim] loading state ← {explicit_state_path}")
        return load_state(explicit_state_path)
    examples_default = _examples_default_path()
    if examples_default.is_file():
        print(f"[gridworld_dagger_sim] loading default ← {examples_default}")
        return load_state(examples_default)
    print("[gridworld_dagger_sim] no examples/default_4x4.json found — using hardcoded default_state()")
    return default_state()


def _print_markdown_table(rows: list[dict]) -> None:
    if not rows:
        print("(no results)")
        return
    print(f"| {'Policy':<18} | {'n':>6} | {'succ':>5} | {'fail':>5} | {'succ_rate':>9} | {'fail_rate':>9} |")
    print(f"| {'-' * 18} | {'-' * 6} | {'-' * 5} | {'-' * 5} | {'-' * 9} | {'-' * 9} |")
    for r in rows:
        print(
            f"| {r['name']:<18} | {r['n']:>6} | {r['succ']:>5} | {r['fail']:>5} | "
            f"{r['succ_rate']:>9.3f} | {r['fail_rate']:>9.3f} |"
        )


def _eval_named(
    state: SimState, name: str, n_rollouts: int, seed: int | None, policy_noise: float = 0.0
) -> dict:
    cat = build_policy_catalog(state)
    if name not in cat:
        raise ValueError(f"Unknown policy '{name}'. Available: {sorted(cat)}")
    pol = cat[name].softened(state.grid, policy_noise) if policy_noise > 0 else cat[name]
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    result = evaluate(state.grid, pol, n_rollouts, rng)
    result["name"] = name if policy_noise == 0 else f"{name}+noise{policy_noise:.2f}"
    return result


def _parse_mix(spec: str) -> list[tuple[str, float]]:
    rows = []
    for part in spec.split(","):
        if ":" not in part:
            raise ValueError(f"Bad mix part '{part}': expected NAME:WEIGHT")
        name, w = part.split(":", 1)
        rows.append((name.strip(), float(w)))
    return rows


def _eval_mix(
    state: SimState,
    mix_rows: list[tuple[str, float]],
    n_rollouts: int,
    seed: int | None,
    policy_noise: float = 0.0,
) -> dict:
    cat = build_policy_catalog(state)
    weighted = []
    for name, w in mix_rows:
        if name not in cat:
            raise ValueError(f"Unknown policy '{name}' in mix. Available: {sorted(cat)}")
        weighted.append((w, cat[name]))
    mixed = Policy.weighted_sum(state.grid, weighted)
    if policy_noise > 0:
        mixed = mixed.softened(state.grid, policy_noise)
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    result = evaluate(state.grid, mixed, n_rollouts, rng)
    suffix = f"+noise{policy_noise:.2f}" if policy_noise > 0 else ""
    result["name"] = "mix(" + ",".join(f"{n}:{w:.2f}" for n, w in mix_rows) + ")" + suffix
    return result


def _eval_blend_sweep(
    state: SimState, ratios: list[float], n_rollouts: int, seed: int | None, policy_noise: float = 0.0
) -> list[dict]:
    cat = build_policy_catalog(state)
    if "intervention" not in cat:
        raise ValueError("No intervention policy in state. Add at least one trajectory.")
    base = cat["base"]
    iv = cat["intervention"]

    def _maybe_soften(pol: Policy) -> Policy:
        return pol.softened(state.grid, policy_noise) if policy_noise > 0 else pol

    suffix = f"+noise{policy_noise:.2f}" if policy_noise > 0 else ""
    rows = []
    # Base baseline first.
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    base_r = evaluate(state.grid, _maybe_soften(base), n_rollouts, rng)
    base_r["name"] = "base" + suffix
    rows.append(base_r)
    for r in ratios:
        blended = _maybe_soften(Policy.blend(state.grid, iv, base, ratio=r))
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        out = evaluate(state.grid, blended, n_rollouts, rng)
        out["name"] = f"blend_{r:.2f}" + suffix
        rows.append(out)
    return rows


def run(args: argparse.Namespace) -> None:
    state = resolve_initial_state(args.state)

    if args.sweep_blend:
        rows = _eval_blend_sweep(
            state, args.sweep_blend, args.rollouts, args.seed, policy_noise=args.policy_noise
        )
        _print_markdown_table(rows)
        return
    if args.mix:
        mix_rows = _parse_mix(args.mix)
        result = _eval_mix(state, mix_rows, args.rollouts, args.seed, policy_noise=args.policy_noise)
        _print_markdown_table([result])
        return
    if args.policy:
        result = _eval_named(state, args.policy, args.rollouts, args.seed, policy_noise=args.policy_noise)
        _print_markdown_table([result])
        return
    raise SystemExit("In --no_gui mode you must pass one of --policy / --sweep_blend / --mix.")


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m my_scripts.gridworld_dagger_sim",
        description="Gridworld DAgger simulator. GUI by default; --no_gui for batch eval.",
    )
    p.add_argument(
        "--no_gui", action="store_true", help="Skip GUI; run batch evaluation per the other flags."
    )
    p.add_argument(
        "--state",
        type=Path,
        default=None,
        help="Path to a state JSON (see examples/default_4x4.json). Default: built-in default.",
    )
    p.add_argument(
        "--policy",
        type=str,
        default=None,
        help="Name of a single policy to evaluate (base, intervention, or a saved composition).",
    )
    p.add_argument(
        "--sweep_blend",
        type=float,
        nargs="*",
        default=None,
        metavar="RATIO",
        help=f"Evaluate base + blend(intervention, base, r) for each r. "
        f"Convention: r=0.0 → pure intervention, r=1.0 → pure base. "
        f"Example default: {' '.join(str(x) for x in DEFAULT_BLEND_RATIOS)}.",
    )
    p.add_argument(
        "--mix", type=str, default=None, help="Evaluate a single ad-hoc mix. Format: 'name1:w1,name2:w2,...'."
    )
    p.add_argument(
        "--rollouts", type=int, default=10000, help="Number of rollouts per policy (default 10000)."
    )
    p.add_argument("--seed", type=int, default=None, help="RNG seed; default = system entropy.")
    p.add_argument(
        "--policy_noise",
        type=float,
        default=0.0,
        help="If > 0, soften each cell's action distribution toward "
        "uniform-over-valid-actions by this fraction before rolling out. "
        "Models per-step BC error; exposes DAgger-style compounding errors. "
        "Default 0.0.",
    )
    return p
