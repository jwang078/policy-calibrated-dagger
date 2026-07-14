"""Retrofit historical `intervention_per_scenario.csv` files so their
`scenario_idx` column reports the underlying eval-benchmark index, not the
rollout-local counter.

Before this fix, the CSV wrote `0, 1, 2, ..., N-1` regardless of what subset
the rollout actually visited. When `--dagger_skip_succeeded_in_prev_eval`
prunes the subset (e.g. round 2 targets scenarios `[1,2,5,6,7,10,13,19,21,22,24,25]`),
the CSV's `0..11` didn't tell you those were actually benchmark eps 1, 2, 5, ...

This script scans a training root, finds each `intervention_per_scenario.csv`,
recovers the subset from the sibling `eval_*.log` file's `lerobot-eval` command
line (searching for `--env.eval_benchmark_subset=[...]`), and rewrites the
scenario_idx column in-place (with a `.pre_retrofit.bak` backup).

Usage:
  # Retrofit a single lineage:
  python3 my_scripts/retrofit_intervention_scenario_idx.py \
    --root=outputs/training

  # Dry run (show what would change, no writes):
  python3 my_scripts/retrofit_intervention_scenario_idx.py \
    --root=outputs/training --dry_run

Idempotency:
  A CSV where `scenario_idx` already matches subset order (no 0-based
  contiguous prefix mismatch) is left untouched — safe to re-run.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path

SUBSET_RE = re.compile(r"--env\.eval_benchmark_subset=\[([^\]]+)\]")


def find_subset_in_eval_log(log_path: Path) -> list[int] | None:
    """Parse the first `--env.eval_benchmark_subset=[...]` from an eval log.

    Returns the subset as a list of ints, or None if not found."""
    try:
        # Only need the first few KB — the `+ lerobot-eval ...` line is very early.
        with open(log_path) as f:
            head = f.read(50_000)
    except OSError:
        return None
    m = SUBSET_RE.search(head)
    if m is None:
        return None
    body = m.group(1)
    try:
        return [int(x.strip()) for x in body.split(",") if x.strip()]
    except ValueError:
        return None


def load_csv_rows(csv_path: Path) -> tuple[list[str], list[list[str]]]:
    with open(csv_path, newline="") as f:
        r = csv.reader(f)
        rows = list(r)
    if not rows:
        return [], []
    return rows[0], rows[1:]


def write_csv_rows(csv_path: Path, header: list[str], rows: list[list[str]]) -> None:
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def needs_retrofit(header: list[str], rows: list[list[str]], subset: list[int]) -> bool:
    """True iff the CSV's scenario_idx column is the raw rollout counter
    (0, 1, 2, ...) rather than the resolved subset indices."""
    if not rows or not subset:
        return False
    try:
        sid_col = header.index("scenario_idx")
    except ValueError:
        return False
    for row_idx, row in enumerate(rows):
        try:
            recorded = int(row[sid_col])
        except (ValueError, IndexError):
            return False
        expected = subset[row_idx % len(subset)]
        if recorded != expected:
            return recorded == row_idx  # raw counter → needs retrofit
    return False  # already matches subset — no-op


def retrofit_csv(csv_path: Path, subset: list[int], dry_run: bool) -> str:
    header, rows = load_csv_rows(csv_path)
    if not rows:
        return "empty"
    try:
        sid_col = header.index("scenario_idx")
    except ValueError:
        return "no scenario_idx column"
    if not needs_retrofit(header, rows, subset):
        return "already retrofitted or CSV pattern doesn't match rollout counter"

    # Rewrite the scenario_idx column.
    new_rows = []
    for row_idx, row in enumerate(rows):
        new_row = list(row)
        new_row[sid_col] = str(subset[row_idx % len(subset)])
        new_rows.append(new_row)

    if dry_run:
        return f"WOULD REWRITE {len(rows)} rows: {[r[sid_col] for r in rows][:5]}... → {[r[sid_col] for r in new_rows][:5]}..."

    # Backup + write.
    backup = csv_path.with_suffix(csv_path.suffix + ".pre_retrofit.bak")
    if not backup.exists():
        shutil.copy2(csv_path, backup)
    write_csv_rows(csv_path, header, new_rows)
    return f"REWROTE {len(rows)} rows (backup: {backup.name})"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Directory to scan for intervention_per_scenario.csv (recursive).",
    )
    ap.add_argument("--dry_run", action="store_true", help="Print what would change but don't write.")
    args = ap.parse_args()

    if not args.root.is_dir():
        print(f"ERROR: --root={args.root} is not a directory", file=sys.stderr)
        return 2

    csvs = sorted(args.root.rglob("intervention_per_scenario.csv"))
    if not csvs:
        print(f"No intervention_per_scenario.csv files found under {args.root}")
        return 0

    print(f"Scanning {len(csvs)} intervention_per_scenario.csv file(s) under {args.root}")
    print(f"Dry run: {args.dry_run}")
    print()

    n_retrofitted = 0
    n_skipped = 0
    n_no_subset = 0

    for csv_path in csvs:
        # eval_*.log lives in the sibling `dagger/` dir, one level up from the
        # `interventions/` dir that holds the CSV.
        dagger_dir = csv_path.parent.parent
        eval_logs = sorted(dagger_dir.glob("eval_*.log"))
        if not eval_logs:
            print(f"[SKIP] {csv_path.relative_to(args.root)}: no eval_*.log in {dagger_dir}")
            n_no_subset += 1
            continue

        # Pick the LATEST eval log (in case a round was resumed and there
        # are multiple). Its subset is what the CSV rows actually visited.
        subset = None
        for log_path in reversed(eval_logs):
            subset = find_subset_in_eval_log(log_path)
            if subset is not None:
                break
        if subset is None:
            print(
                f"[SKIP] {csv_path.relative_to(args.root)}: no --env.eval_benchmark_subset in any eval_*.log"
            )
            n_no_subset += 1
            continue

        rel = csv_path.relative_to(args.root)
        verdict = retrofit_csv(csv_path, subset, args.dry_run)
        if verdict.startswith(("REWROTE", "WOULD REWRITE")):
            n_retrofitted += 1
            print(f"[FIX ] {rel} (subset len={len(subset)}): {verdict}")
        else:
            n_skipped += 1
            print(f"[SKIP] {rel}: {verdict}")

    print()
    print(f"Summary: {n_retrofitted} retrofitted, {n_skipped} skipped, {n_no_subset} no subset found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
