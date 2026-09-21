#!/usr/bin/env python
"""Per-scenario complementarity between two DAgger lineages.

Even at equal aggregate success, two arms may succeed on DIFFERENT scenario
subsets — evidence the treatment changed the policy's coverage rather than
doing nothing. This pools each arm's training-time evals over a round range
(1 episode per scenario per round) via dagger_failed_scenarios.py and
cross-tabulates chronic failures.

Usage:
    python my_scripts/dagger_scenario_complementarity.py
        --arm_a outputs/training/diffusion_..._03dag_ft_dag{r}
        --arm_b outputs/training/diffusion_..._03dag_rr_b050_ft_dag{r}
        --rounds 7 8 9 10

The {r} placeholder is substituted per round. Run it while the training dirs
still exist — the per-scenario eval records live inside them (lesson of
2026-08-20: deleting a lineage's training dirs deletes its eval history).
"""

from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def failed_set(train_dir: str) -> set[int] | None:
    """Return the failed-scenario set from one training dir's latest eval."""
    out = subprocess.run(
        [sys.executable, str(HERE / "dagger_failed_scenarios.py"), f"--prev_train_dir={train_dir}"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if out.returncode != 0:
        return None
    return set(json.loads(out.stdout)["failed"])


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm_a", required=True, help="training-dir template with {r}, e.g. .../..._ft_dag{r}")
    ap.add_argument("--arm_b", required=True)
    ap.add_argument("--name_a", default="A")
    ap.add_argument("--name_b", default="B")
    ap.add_argument("--rounds", type=int, nargs="+", default=[7, 8, 9, 10])
    ap.add_argument(
        "--chronic_min",
        type=int,
        default=None,
        help="fails >= this many rounds = chronic (default: len(rounds)-1)",
    )
    args = ap.parse_args()
    chronic = args.chronic_min if args.chronic_min is not None else max(1, len(args.rounds) - 1)

    counts: dict[str, collections.Counter] = {}
    n_evals: dict[str, int] = {}
    for name, tpl in ((args.name_a, args.arm_a), (args.name_b, args.arm_b)):
        c: collections.Counter = collections.Counter()
        n = 0
        for r in args.rounds:
            fs = failed_set(tpl.format(r=r))
            if fs is None:
                print(f"[warn] no eval data: {tpl.format(r=r)}")
                continue
            n += 1
            for s in fs:
                c[s] += 1
        counts[name], n_evals[name] = c, n
    a, b = args.name_a, args.name_b
    if not n_evals[a] or not n_evals[b]:
        raise SystemExit("one arm has no usable evals — nothing to compare")

    all_s = sorted(set(counts[a]) | set(counts[b]))
    print(f"\npooled {n_evals[a]} evals ({a}) vs {n_evals[b]} evals ({b}); chronic = fails >= {chronic}")
    print(f"{'scen':>5} {a:>8} {b:>8}")
    for s in all_s:
        if counts[a].get(s, 0) >= 2 or counts[b].get(s, 0) >= 2:
            print(f"{s:>5} {counts[a].get(s, 0):>8} {counts[b].get(s, 0):>8}")
    both = [s for s in all_s if counts[a].get(s, 0) >= chronic and counts[b].get(s, 0) >= chronic]
    only_a = [s for s in all_s if counts[a].get(s, 0) >= chronic and counts[b].get(s, 0) <= 1]
    only_b = [s for s in all_s if counts[b].get(s, 0) >= chronic and counts[a].get(s, 0) <= 1]
    print(f"\nchronic in BOTH: {both}")
    print(f"chronic in {a} only ({b} solves them): {only_a}")
    print(f"chronic in {b} only ({a} solves them): {only_b}")
    print(
        f"\ncomplementarity: {len(only_a) + len(only_b)} scenario(s) separate the arms; "
        f"{len(both)} are hard for both."
    )


if __name__ == "__main__":
    main()
