#!/usr/bin/env python3
"""Compute the list of FAILED scenario indices from a prior DAgger round's
training-time eval, so the next round's intervention recording can target
only those scenarios instead of re-running all 30.

Mapping logic (verified against
`splatsim/robots/sim_robot_pybullet_base.py:2280-2310`):

    * SplatSim's EVAL_BENCHMARK mode advances through `eval_benchmark_subset`
      in scan order. With seed=K, the next reset jumps to
      `subset[(K % len(subset))]` (formula in `_handle_reset`).
    * lerobot-eval uses contiguous `seed = start_seed + episode_idx`. With
      `start_seed=0`, episode `i` runs scenario `subset[i % len(subset)]`,
      which for `n_episodes == len(subset)` collapses to `subset[i]`.
    * eval_info.json's `per_task[0].metrics.successes` is a list of length
      `n_episodes`. Position `i` ↔ scenario `subset[i]`.

So `failed = [subset[i] for i in range(n) if not successes[i]]`.

Output: JSON dict on stdout. Empty `failed` list (or non-zero exit) means
the caller should fall back to the default subset:

    {
        "failed": [2, 7, 12, 17],
        "succeeded": [0, 1, 3, 4, ...],
        "n_total": 30,
        "n_failed": 4,
        "source_eval_step": 78000,
        "source_eval_info": "/.../eval_info_step_078000.json",
        "subset_used": [0, 1, ..., 29]
    }

Exit codes:
    0  — success (failed list available; may be empty if 100% success).
    1  — no usable eval_info.json on disk for that train dir.
    2  — eval_info.json exists but is corrupt / missing per_task data.
    3  — eval_info.json exists but matching train_config.json with subset
         is missing — can't recover scenario indices.
"""

import argparse
import json
import sys
from pathlib import Path


def _newest_train_config(train_dir: Path) -> Path | None:
    """Walk the train dir's checkpoints/ subdirs and return the train_config.json
    from the highest-step checkpoint. We need this to discover what
    `eval_benchmark_subset` the eval ran against — the eval_info.json itself
    only records per-task success lists, not the subset.
    """
    candidates = sorted(
        train_dir.glob("checkpoints/*/pretrained_model/train_config.json"),
        key=lambda p: int(p.parent.parent.name) if p.parent.parent.name.isdigit() else -1,
    )
    return candidates[-1] if candidates else None


def main() -> int:
    p = argparse.ArgumentParser(prog="dagger_failed_scenarios")
    p.add_argument(
        "--prev_train_dir",
        type=Path,
        required=True,
        help="Path to the previous DAgger round's training output directory "
        "(e.g. outputs/training/<base>_ft_dag<R-1>).",
    )
    args = p.parse_args()

    train_dir = args.prev_train_dir.expanduser().resolve()
    if not train_dir.is_dir():
        print(f"ERROR: train dir does not exist: {train_dir}", file=sys.stderr)
        return 1

    eval_dir = train_dir / "eval"
    if not eval_dir.is_dir():
        print(f"ERROR: no eval dir under {train_dir}", file=sys.stderr)
        return 1

    # Pick the most-recent eval_info_step_*.json by mtime — handles the case
    # where multiple eval steps fired during training. Matches the cascade
    # ordering used by dagger_progress.sh.
    candidates = sorted(
        eval_dir.glob("eval_info_step_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        print(f"ERROR: no eval_info_step_*.json under {eval_dir}", file=sys.stderr)
        return 1
    eval_info_path = candidates[0]

    try:
        data = json.load(open(eval_info_path))
    except Exception as e:
        print(f"ERROR: failed to parse {eval_info_path}: {e}", file=sys.stderr)
        return 2

    per_task = data.get("per_task") or []
    if not per_task:
        print(f"ERROR: no per_task data in {eval_info_path}", file=sys.stderr)
        return 2
    # SplatSim has a single per_task block with the full successes list.
    successes = (per_task[0].get("metrics") or {}).get("successes") or []
    if not successes:
        print(
            f"ERROR: no per_task[0].metrics.successes in {eval_info_path}",
            file=sys.stderr,
        )
        return 2

    # Find the subset the eval used. Lives in the train_config.json — same
    # train run that produced the eval_info.json. Match by step if possible.
    step_str = eval_info_path.stem.replace("eval_info_step_", "")
    train_config_paths = [train_dir / f"checkpoints/{step_str}/pretrained_model/train_config.json"]
    newest = _newest_train_config(train_dir)
    if newest is not None and newest not in train_config_paths:
        train_config_paths.append(newest)

    subset = None
    train_config_path_used = None
    for tcp in train_config_paths:
        if not tcp.is_file():
            continue
        try:
            cfg = json.load(open(tcp))
        except Exception:
            continue
        env = cfg.get("env") or {}
        subset_candidate = env.get("eval_benchmark_subset")
        if isinstance(subset_candidate, list) and subset_candidate:
            subset = subset_candidate
            train_config_path_used = tcp
            break

    if subset is None:
        print(
            f"ERROR: could not find env.eval_benchmark_subset in any train_config.json under {train_dir}",
            file=sys.stderr,
        )
        return 3

    # Pairwise zip — clamp to the shorter of the two so we don't index OOB
    # if subset and successes are mismatched in length (shouldn't happen in
    # well-formed runs but defensive).
    n = min(len(successes), len(subset))
    failed = [int(subset[i]) for i in range(n) if not bool(successes[i])]
    succeeded = [int(subset[i]) for i in range(n) if bool(successes[i])]

    out = {
        "failed": failed,
        "succeeded": succeeded,
        "n_total": n,
        "n_failed": len(failed),
        "source_eval_step": int(step_str) if step_str.isdigit() else None,
        "source_eval_info": str(eval_info_path),
        "subset_used": subset[:n],
        "train_config_path": str(train_config_path_used) if train_config_path_used else None,
    }
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
