#!/usr/bin/env python3
"""Does DAgger intervention data cover the policy's FAILURE states, or is it
redundant with the base demos?

This tests "problem #2" — the planner-expert covariate-shift concern: RRT records
states on its OWN recovery trajectory, which (especially under pre_jump_lookback,
which rewinds toward a near-demo state before planning) may sit on the base-demo
manifold rather than on the off-distribution states where the policy actually
breaks. If so, abundant correction data can't teach recovery from real failures
and DAgger won't improve no matter how you re-weight it.

We don't have the policy's eval-rollout states on disk (eval saves only metrics +
videos). But the intervention dataset's per-episode structure is a usable proxy:
its episode-START frames are closest to the trigger/failure region, while the
episode-TAIL frames are RRT near the goal. So we compare, in base-demo joint
space:

  * novelty(intervention frames → base): nearest-neighbour distance from each
    intervention state to the base-demo manifold, vs the base's own
    nearest-neighbour spacing (the redundancy baseline).
  * the same split by within-episode position (start / mid / end).

Reading:
  * intervention novelty ≈ base baseline  → corrections sit ON the demo manifold
    (redundant); the policy's true failure states aren't represented → coverage
    problem #2 is REAL → re-weighting won't help.
  * starts NOVEL, tails ≈ baseline         → corrections bridge failure→demo
    (the healthy DAgger pattern) → coverage is fine → the lever is the mixing
    (dilution), so the re-weighting A/B IS worth running.

Reuses load_state_episodes (parquet-only) and the lineage config resolver.

Usage:
    python my_scripts/dagger_coverage_overlap.py \\
        --dataset_path ~/.cache/huggingface/lerobot/JennyWWW/lever_g0_d30_clean_03dag_diff_r_dag16
    # several rounds onto one figure:
    python my_scripts/dagger_coverage_overlap.py --dataset_path .../_r_dag1 --rounds 1 8 16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dagger_diagnose_lineage import CACHE, OUT_DIR, find_config_for_dataset  # noqa: E402
from dagger_naming import parse_dataset_short  # noqa: E402
from plot_state_deltas import load_state_episodes  # noqa: E402

GRIPPER_DIM = -1
EDGE_FRAC = 0.2  # first/last 20% of each episode = "start"/"end" buckets


def _arm(state: np.ndarray) -> np.ndarray:
    n = state.shape[1]
    gdim = GRIPPER_DIM if GRIPPER_DIM >= 0 else n + GRIPPER_DIM
    return state[:, [c for c in range(n) if c != gdim]]


def within_episode_fraction(episode: np.ndarray) -> np.ndarray:
    """For each frame, its position within its episode in [0,1)."""
    frac = np.zeros(len(episode), dtype=np.float64)
    for ep in np.unique(episode):
        idx = np.where(episode == ep)[0]
        if len(idx) > 1:
            frac[idx] = np.arange(len(idx)) / (len(idx) - 1)
    return frac


def nn_distance(query: np.ndarray, ref: np.ndarray, exclude_self: bool) -> np.ndarray:
    """Nearest-neighbour distance from each query row to the ref set. When
    exclude_self (ref is query), uses the 2nd neighbour to skip the point itself."""
    k = 2 if exclude_self else 1
    nn = NearestNeighbors(n_neighbors=k).fit(ref)
    d, _ = nn.kneighbors(query)
    return d[:, -1]


def subsample(x: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(x) <= n:
        return x
    return x[rng.choice(len(x), n, replace=False)]


def analyze_round(base_arm_z, base_mean, base_std, inter_root: Path, rng, max_pts):
    """Return per-round novelty stats + projected points for plotting."""
    istate, iepisode, _ = load_state_episodes(str(inter_root))
    iarm = _arm(istate)
    iarm_z = (iarm - base_mean) / base_std
    frac = within_episode_fraction(iepisode)

    # base redundancy baseline: base-to-base NN spacing.
    base_sub = subsample(base_arm_z, max_pts, rng)
    base_baseline = float(np.median(nn_distance(base_sub, base_arm_z, exclude_self=True)))

    buckets = {
        "start": frac <= EDGE_FRAC,
        "mid": (frac > EDGE_FRAC) & (frac < 1 - EDGE_FRAC),
        "end": frac >= 1 - EDGE_FRAC,
    }
    stats = {"baseline": base_baseline, "n_inter": len(iarm)}
    for name, mask in buckets.items():
        pts = iarm_z[mask]
        if len(pts) == 0:
            stats[name] = None
            continue
        pts_s = subsample(pts, max_pts, rng)
        d = nn_distance(pts_s, base_arm_z, exclude_self=False)
        stats[name] = float(np.median(d))
    # whole-intervention novelty
    allpts = subsample(iarm_z, max_pts, rng)
    stats["all"] = float(np.median(nn_distance(allpts, base_arm_z, exclude_self=False)))
    return stats, iarm_z, frac


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--dataset_path", type=Path, required=True, help="Any per-round intervention dataset in the lineage."
    )
    p.add_argument(
        "--rounds",
        type=int,
        nargs="+",
        default=None,
        help="Round numbers to analyze (default: just the one in --dataset_path).",
    )
    p.add_argument("--sidecar", type=Path, default=None)
    p.add_argument(
        "--max_points", type=int, default=4000, help="Subsample cap per source for the NN computation."
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    ds_name = args.dataset_path.name
    found = find_config_for_dataset(ds_name, args.sidecar)
    if found is None:
        print(f"ERROR: no dagger config references {ds_name}", file=sys.stderr)
        return 2
    _, sidecar = found
    cfg = sidecar.get("config", {})
    repos = cfg.get("weighted_repo_ids") or sidecar.get("weighted_repo_ids") or []
    base_repo = sidecar.get("naming", {}).get("base_repo") or repos[0]
    # map round -> repo_id from the weighted list (index 0 = base).
    round_to_repo = {}
    for r in repos[1:]:
        parsed = parse_dataset_short(r.split("/")[-1])
        if parsed.round is not None:
            round_to_repo[parsed.round] = r

    parsed_in = parse_dataset_short(ds_name)
    rounds = args.rounds or ([parsed_in.round] if parsed_in.round is not None else [])
    rounds = [r for r in rounds if r in round_to_repo]
    if not rounds:
        print(f"ERROR: no resolvable rounds among {args.rounds}", file=sys.stderr)
        return 2

    # base manifold (z-scored on arm joints).
    bstate, _, _ = load_state_episodes(str(CACHE / base_repo))
    barm = _arm(bstate)
    base_mean = barm.mean(0)
    base_std = np.where(barm.std(0) == 0, 1.0, barm.std(0))
    base_z = (barm - base_mean) / base_std
    pca = PCA(n_components=2).fit(subsample(base_z, 8000, rng))
    base_proj = pca.transform(subsample(base_z, args.max_points, rng))

    print(f"\nBASE: {base_repo}  ({len(barm)} frames)")
    print(f"{'round':<6} {'baseline':>9} {'start':>8} {'mid':>8} {'end':>8} {'all':>8}   verdict")
    print("-" * 78)

    fig, axes = plt.subplots(1, len(rounds), figsize=(6 * len(rounds), 5.5), squeeze=False)
    all_stats = []
    for ax, rn in zip(axes[0], rounds):
        inter_root = CACHE / round_to_repo[rn]
        stats, iarm_z, frac = analyze_round(base_z, base_mean, base_std, inter_root, rng, args.max_points)
        all_stats.append((rn, stats))

        def ratio(v):
            return None if v is None else v / stats["baseline"]

        rs, rm, re_, ra = ratio(stats["start"]), ratio(stats["mid"]), ratio(stats["end"]), ratio(stats["all"])
        # verdict from the start-vs-end novelty pattern
        verdict = "?"
        if rs is not None and re_ is not None:
            if ra < 1.3:
                verdict = "REDUNDANT w/ base (coverage #2 likely REAL)"
            elif rs > 1.5 and re_ < rs * 0.7:
                verdict = "bridges failure->demo (coverage OK; mixing is the lever)"
            else:
                verdict = "partially novel"
        print(
            f"dag{rn:<3} {stats['baseline']:>9.3f} "
            f"{(stats['start'] or float('nan')):>8.3f} {(stats['mid'] or float('nan')):>8.3f} "
            f"{(stats['end'] or float('nan')):>8.3f} {stats['all']:>8.3f}   "
            f"x{ra:.2f}  {verdict}"
        )

        # plot: base (grey) + intervention colored by within-episode fraction.
        # One deterministic subsample so projected points and their frac align.
        sub_rng = np.random.default_rng(args.seed)
        idx = (
            np.arange(len(iarm_z))
            if len(iarm_z) <= args.max_points
            else sub_rng.choice(len(iarm_z), args.max_points, replace=False)
        )
        ip = pca.transform(iarm_z[idx])
        ax.scatter(base_proj[:, 0], base_proj[:, 1], s=6, c="lightgrey", label="base demos", alpha=0.5)
        sc = ax.scatter(ip[:, 0], ip[:, 1], s=8, c=frac[idx], cmap="viridis", alpha=0.7)
        ax.set_title(f"dag{rn}: novelty x{ra:.2f} base\n(start→end = dark→yellow)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(sc, ax=axes[0, -1], label="within-episode fraction")

    lineage = parsed_in.prefix or ds_name
    fig.suptitle(f"Intervention coverage vs base manifold — {lineage}", y=1.02, fontsize=13)
    out = args.out or (OUT_DIR / f"{lineage}_coverage_overlap.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"\nWrote {out}")

    # overall verdict
    novelties = [s["all"] / s["baseline"] for _, s in all_stats]
    print(f"\n{'─' * 78}")
    mean_nov = float(np.mean(novelties))
    if mean_nov < 1.3:
        print(f"VERDICT: intervention states are ~base-redundant (mean novelty x{mean_nov:.2f}).")
        print("  → coverage problem #2 is LIKELY REAL: RRT recovers along demo-like paths,")
        print("    so the policy's true failure states aren't represented. Re-weighting the")
        print("    DAgger fraction probably WON'T help; the recording/expert is the issue.")
    else:
        print(f"VERDICT: intervention states ARE novel vs base (mean novelty x{mean_nov:.2f}).")
        print("  → corrections cover regions base doesn't. Coverage is plausibly fine, so the")
        print("    dilution/mixing lever (the re-weighting A/B) is worth running.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
