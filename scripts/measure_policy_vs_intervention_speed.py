#!/usr/bin/env python
"""Measure how fast the POLICY moves vs the RRT INTERVENTIONS, from recorded data.

Uses the same association `visualize_intervention_episode.py` uses:
``<training_dir>/dagger/interventions/intervention_per_scenario.csv`` gives, per
scenario, the absolute env tick of each trigger (``trigger_steps``) and the RRT
execution length (``rrt_steps_executed``); the round's intervention DATASET has
one episode per RRT cycle (paired by ``source_scenario_idx`` + cycle order), and
each episode's ``observation.state`` covers exactly the RRT execution window.

From that we recover:

* **RRT speed** — within each recorded episode: per-tick path speed
  (mean ||Δq||) and net speed (||q_end − q_start|| / ticks).
* **Policy net speed** — the policy drives ALONE between the end of cycle i
  (last state of episode i) and the trigger of cycle i+1 (first state of
  episode i+1), for Δt = trigger_steps[i+1] − (trigger_steps[i] + L_i) ticks.
  Also the scenario-start window: benchmark start state → first cycle's start,
  over trigger_steps[0] ticks.

  Exactness depends on the i+1 trigger's lookback dispatch (hybrid mode):
  collision-type triggers (in_collision / self_collision / obstacle_collision /
  future_chunk_coll) are NO-LOOKBACK → episode i+1 starts at the exact state
  where the policy stopped → EXACT measurement. Stall-type triggers rewind
  50–100 ticks first → the window is reported separately as approximate
  (effective ticks reduced by the midpoint of --lookback_min/max), and is also
  selection-biased slow (stall triggers fire BECAUSE the policy stopped).

Pass several training dirs (rounds of a lineage) to see the trend across
rounds — e.g. whether successive finetunes get slower (the compounding-
slowdown hypothesis).

Examples:
    python my_scripts/measure_policy_vs_intervention_speed.py \\
        outputs/training/diffusion_planar_3joint_8_delta_stateng_03dag_smooth_ft_dag1

    python my_scripts/measure_policy_vs_intervention_speed.py \\
        outputs/training/diffusion_planar_3joint_8_delta_stateng_03dag_smooth_ft_dag{1..9}
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
for p in (str(_HERE), str(_REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from dagger_naming import _argv_get_flag, parse_dataset_short  # type: ignore[import-not-found]  # noqa: E402
from lib_dataset_episode_io import (  # type: ignore[import-not-found]  # noqa: E402
    find_parquet_files,
    load_episodes_meta,
)
from visualize_intervention_episode import (  # type: ignore[import-not-found]  # noqa: E402
    _parse_int_list,
    _resolve_interventions_dir,
)

from lerobot.utils.lerobot_dataset_utils import resolve_dataset_dir  # noqa: E402

# Hybrid-mode no-lookback trigger reasons — keep in sync with
# InterventionController._collision_trigger_reasons.
COLLISION_TRIGGER_REASONS = {"in_collision", "self_collision", "obstacle_collision", "future_chunk_coll"}


# ── per-round data assembly ───────────────────────────────────────────────────


@dataclass
class SpeedSamples:
    """Per-tick joint-space speeds (rad/tick, L2 over all state dims)."""

    rrt_path: list[float] = field(default_factory=list)  # mean ||Δq|| within episodes
    rrt_net: list[float] = field(default_factory=list)  # ||q_end − q_start|| / ticks
    policy_net_exact: list[float] = field(default_factory=list)  # collision-trigger windows
    policy_net_stall: list[float] = field(default_factory=list)  # stall windows (approx, biased)

    def summary(self, key: str) -> str:
        v = getattr(self, key)
        if not v:
            return "  (none)  "
        return f"{np.mean(v):.5f}±{np.std(v):.5f} (n={len(v)})"


def resolve_intervention_repo(training_dir: Path, override: str | None) -> str:
    """The round's intervention dataset repo id, from the dagger config sidecar."""
    if override:
        return override
    sidecar_path = training_dir / "dagger" / "config.json"
    if sidecar_path.is_file():
        sidecar = json.loads(sidecar_path.read_text())
        for repo in sidecar.get("config", {}).get("weighted_repo_ids") or []:
            parsed = parse_dataset_short(repo.rpartition("/")[2])
            if parsed.kind == "intervention" and parsed.round == sidecar.get("round"):
                return repo
    raise SystemExit(
        f"Could not resolve the intervention repo from {sidecar_path} "
        f"(no weighted_repo_ids entry of kind=intervention matching the sidecar round). "
        f"Pass --intervention_repo_id explicitly."
    )


def resolve_benchmark_repo(training_dir: Path, override: str | None) -> str | None:
    """The eval benchmark repo (scenario start states), from the sidecar argv."""
    if override:
        return override
    sidecar_path = training_dir / "dagger" / "config.json"
    if sidecar_path.is_file():
        argv = json.loads(sidecar_path.read_text()).get("orchestrator_invocation", {}).get("argv") or []
        return _argv_get_flag(argv, "eval_benchmark")
    return None


def load_episode_states(data_dir: Path) -> dict[int, np.ndarray]:
    """``episode_index → [n_frames, state_dim] observation.state`` for a dataset."""
    frames = pd.concat(
        [
            pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.state"])
            for f in find_parquet_files(data_dir)
        ],
        ignore_index=True,
    )
    out: dict[int, np.ndarray] = {}
    for ep, g in frames.groupby("episode_index"):
        g = g.sort_values("frame_index")
        out[int(ep)] = np.stack([np.asarray(s, dtype=np.float64) for s in g["observation.state"]])
    return out


def measure_round(
    training_dir: Path,
    intervention_repo: str,
    benchmark_states: dict[int, np.ndarray] | None,
    lookback_mid: float,
    dataset_dir: str | None,
) -> SpeedSamples:
    """Extract all speed samples for one DAgger round."""
    interventions_dir = _resolve_interventions_dir(training_dir)
    with open(interventions_dir / "intervention_per_scenario.csv") as f:
        rows = list(csv.DictReader(f))

    data_dir = resolve_dataset_dir(intervention_repo, dataset_dir)
    ep_states = load_episode_states(data_dir)
    meta = load_episodes_meta(data_dir)
    eps_by_scenario: dict[int, list[int]] = {}
    for _, m in meta.sort_values("episode_index").iterrows():
        if pd.notna(m.get("source_scenario_idx")):
            eps_by_scenario.setdefault(int(m["source_scenario_idx"]), []).append(int(m["episode_index"]))

    samples = SpeedSamples()
    for row in rows:
        scenario = int(row["scenario_idx"])
        trigger_steps = _parse_int_list(row["trigger_steps"])
        rrt_lens = _parse_int_list(row["rrt_steps_executed"])
        reasons = [r.strip() for r in row["triggers"].split(",")] if row["triggers"].strip() else []
        # Same positional pairing as visualize_intervention_episode: dataset
        # episodes exist only for cycles that actually executed RRT (L > 0).
        nonzero = [i for i, ln in enumerate(rrt_lens) if ln > 0]
        eps = eps_by_scenario.get(scenario, [])
        if len(nonzero) != len(eps):
            print(
                f"  [warn] scenario {scenario}: {len(nonzero)} cycles with RRT vs {len(eps)} episodes — "
                f"skipping this scenario (pairing ambiguous).",
                file=sys.stderr,
            )
            continue

        # RRT speed within each recorded episode (trim CSV-length to drop
        # min-episode-length hold padding; states has L+0/-1 rows vs CSV L).
        cyc_first: dict[int, np.ndarray] = {}
        cyc_last: dict[int, np.ndarray] = {}
        for ci, ep in zip(nonzero, eps, strict=True):
            q = ep_states.get(ep)
            if q is None or q.shape[0] < 2:
                continue
            q = q[: min(q.shape[0], rrt_lens[ci])]
            cyc_first[ci], cyc_last[ci] = q[0], q[-1]
            ticks = q.shape[0] - 1
            samples.rrt_path.append(float(np.linalg.norm(np.diff(q, axis=0), axis=1).mean()))
            samples.rrt_net.append(float(np.linalg.norm(q[-1] - q[0]) / ticks))

        # Policy windows. Cycle i's RRT ends at tick trigger_steps[i] +
        # rrt_lens[i]; the policy then drives alone until trigger_steps[i+1].
        for i in range(len(trigger_steps)):
            if i not in cyc_first:
                continue
            reason = reasons[i] if i < len(reasons) else ""
            if i == 0:
                if benchmark_states is None or scenario not in benchmark_states:
                    continue
                q_from, dt = benchmark_states[scenario][0], trigger_steps[0]
            else:
                if (i - 1) not in cyc_last:
                    continue
                q_from = cyc_last[i - 1]
                dt = trigger_steps[i] - (trigger_steps[i - 1] + rrt_lens[i - 1])
            if dt <= 0:
                continue
            net = float(np.linalg.norm(cyc_first[i] - q_from))
            if reason in COLLISION_TRIGGER_REASONS:
                samples.policy_net_exact.append(net / dt)
            else:
                # Stall-type trigger: the start state of episode i is a REWOUND
                # state ~lookback ticks before the trigger. Approximate the
                # effective policy-driving window; also inherently biased slow.
                dt_eff = dt - lookback_mid
                if dt_eff > 0:
                    samples.policy_net_stall.append(net / dt_eff)
    return samples


def demo_reference(base_repo: str, dataset_dir: str | None, max_episodes: int) -> tuple[float, float]:
    """(path_speed, net_speed) of the base demos, for the reference row."""
    data_dir = resolve_dataset_dir(base_repo, dataset_dir)
    states = load_episode_states(data_dir)
    path, net = [], []
    for _, q in sorted(states.items())[:max_episodes]:
        if q.shape[0] < 2:
            continue
        path.append(float(np.linalg.norm(np.diff(q, axis=0), axis=1).mean()))
        net.append(float(np.linalg.norm(q[-1] - q[0]) / (q.shape[0] - 1)))
    return float(np.mean(path)), float(np.mean(net))


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("training_dirs", nargs="+", type=Path, help="DAgger training dir(s), one per round.")
    parser.add_argument(
        "--intervention_repo_id", default=None, help="Override sidecar-derived repo (single dir only)."
    )
    parser.add_argument(
        "--benchmark_repo_id", default=None, help="Eval benchmark dataset (scenario start states)."
    )
    parser.add_argument(
        "--base_repo_id",
        default=None,
        help="Base demo dataset for the reference row (default: sidecar naming.base_repo).",
    )
    parser.add_argument(
        "--lookback_min", type=int, default=50, help="pre_jump_lookback.steps_min used at recording time."
    )
    parser.add_argument(
        "--lookback_max", type=int, default=100, help="pre_jump_lookback.steps_max used at recording time."
    )
    parser.add_argument("--demo_max_episodes", type=int, default=100)
    parser.add_argument("--dataset_dir", default=None, help="Dataset cache root override.")
    args = parser.parse_args()
    if args.intervention_repo_id and len(args.training_dirs) > 1:
        raise SystemExit("--intervention_repo_id only makes sense with a single training dir.")
    lookback_mid = (args.lookback_min + args.lookback_max) / 2.0

    # Benchmark start states (for the scenario-start → first-trigger window).
    bench_repo = resolve_benchmark_repo(args.training_dirs[0], args.benchmark_repo_id)
    benchmark_states = None
    if bench_repo:
        print(f"Benchmark (scenario start states): {bench_repo}")
        benchmark_states = load_episode_states(resolve_dataset_dir(bench_repo, args.dataset_dir))
    else:
        print("No benchmark repo resolved — skipping scenario-start policy windows.")

    print(
        "\nSpeeds in rad/tick (L2 over state dims). policy_net_exact = windows ending in a "
        "collision-type trigger (no rewind — exact). policy_net_stall = windows ending in a "
        "stall-type trigger (rewind-corrected estimate; selection-biased slow).\n"
    )
    header = (
        f"{'round':<28} {'rrt_path':>24} {'rrt_net':>24} {'policy_net_exact':>26} {'policy_net_stall':>26}"
    )
    print(header)
    print("-" * len(header))
    base_repo = args.base_repo_id
    for td in args.training_dirs:
        repo = resolve_intervention_repo(td, args.intervention_repo_id)
        if base_repo is None:
            sc = td / "dagger" / "config.json"
            if sc.is_file():
                base_repo = json.loads(sc.read_text()).get("naming", {}).get("base_repo")
        s = measure_round(td, repo, benchmark_states, lookback_mid, args.dataset_dir)
        label = repo.rpartition("/")[2]
        print(
            f"{label[-26:]:<28} {s.summary('rrt_path'):>24} {s.summary('rrt_net'):>24} "
            f"{s.summary('policy_net_exact'):>26} {s.summary('policy_net_stall'):>26}"
        )

    if base_repo:
        path, net = demo_reference(base_repo, args.dataset_dir, args.demo_max_episodes)
        print("-" * len(header))
        print(f"{'base demos (' + base_repo.rpartition('/')[2] + ')':<28} {path:>24.5f} {net:>24.5f}")


if __name__ == "__main__":
    main()
