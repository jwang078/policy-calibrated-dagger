#!/usr/bin/env python3
"""PCA scatter of observation.state across base / intervention / blend
datasets for a single DAgger round.

Tests the state-distribution-interpolation hypothesis for blended DAgger:
do blend rollouts visit states that lie between pure-policy base
distribution and pure-expert intervention distribution? If yes, the per-
ratio scatter clouds should fill the gap between base and intervention,
with `b010` clouds nearest base and `b090` nearest intervention.

User feeds the script ANY one dataset path from the lineage (base /
intervention / blend / merged); the script resolves every related dataset
for the requested round, fits a single 2D PCA across them all, and
scatter-plots one color per source.

Usage (headline):
    python3 my_scripts/dagger_state_coverage_pca.py \\
        --dataset_path=~/.cache/huggingface/lerobot/JennyWWW/lever_grip0_d5jvm_diff_r_dag7_blend010 \\
        --round=6

See `--help` for full option set. `--dry-run` prints the resolved set of
dataset paths + the sidecar that provided base-repo provenance, without
loading state or plotting.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Share the canonical naming module with the orchestrator + dagger_plot.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dagger_naming import (  # noqa: E402
    enumerate_blend_paths_on_disk,
    find_sidecar_by_prefix,
    int_cache_path,
    load_initial_policy_train_config,
    load_sidecar,
    parse_dataset_short,
    resolve_base_repo,
)

DEFAULT_TRAINING_ROOT = Path.home() / "code" / "lerobot" / "outputs" / "training"
DEFAULT_LEROBOT_CACHE = Path.home() / ".cache" / "huggingface" / "lerobot"
DEFAULT_OUT_DIR = Path.home() / "code" / "lerobot" / "outputs" / "dagger" / "state_coverage"
DEFAULT_HF_USER = "JennyWWW"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dagger_state_coverage_pca",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dataset_path",
        required=True,
        type=Path,
        help="Any dataset directory in the lineage (base / intervention / blend / merged).",
    )
    p.add_argument(
        "--round",
        type=int,
        default=None,
        help="DAgger round to gather datasets for. Defaults to the round parsed from "
        "--dataset_path. REQUIRED if --dataset_path is a base dataset (no round in name).",
    )
    p.add_argument(
        "--base_repo_id",
        type=str,
        default=None,
        help="Explicit base-dataset repo id override (e.g. JennyWWW/foo). HIGHEST priority "
        "— skips sidecar / train_config inference entirely.",
    )
    p.add_argument(
        "--max_points_per_source",
        type=int,
        default=2000,
        help="Random subsample per source for plotting. Each 'point' is a "
        "consecutive action chunk of length n_action_steps (default 2000).",
    )
    p.add_argument(
        "--n_action_steps",
        type=int,
        default=None,
        help="Action-chunk length. Defaults to the base policy's "
        "policy.n_action_steps from its train_config.json. Pass explicitly "
        "to override (e.g. when the policy config is unreadable).",
    )
    p.add_argument("--seed", type=int, default=0, help="Subsampling RNG seed (default 0).")
    p.add_argument(
        "--num_base_trajectories",
        type=int,
        default=4,
        help="Number of base-dataset episodes to overlay as connected chunk-trajectory "
        "LINES on top of the scatter (default 4). Lets you visually compare whether "
        "base episodes follow the same direction in PCA-space as intervention. Set to 0 to disable.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"Output PNG. Default {DEFAULT_OUT_DIR}/<prefix>_dag<R>_state_coverage_pca.png",
    )
    p.add_argument(
        "--training_root",
        type=Path,
        default=DEFAULT_TRAINING_ROOT,
        help="Override outputs/training/ for sidecar search.",
    )
    p.add_argument(
        "--hf_user",
        type=str,
        default=DEFAULT_HF_USER,
        help=f"HF user prefix for cache paths (default {DEFAULT_HF_USER}).",
    )
    p.add_argument(
        "--lerobot_cache",
        type=Path,
        default=DEFAULT_LEROBOT_CACHE,
        help="Override the LeRobot cache root.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved dataset paths + sidecar provenance, then exit.",
    )
    return p


def resolve_inputs(args: argparse.Namespace) -> dict:
    """Parse --dataset_path + --round, resolve sibling intervention / blend
    paths, and resolve the base-dataset repo id. Returns a dict with keys:

        round, prefix, infix,
        intervention_path, blends (list[tuple[pct, Path]]),
        base_repo, base_path, base_source_tag,
        sidecar_path (Path or None)
    """
    dataset_path = args.dataset_path.expanduser().resolve()
    parsed = parse_dataset_short(dataset_path.name)

    # ── Round resolution ────────────────────────────────────────────────────
    round_n = args.round if args.round is not None else parsed.round
    if round_n is None:
        raise SystemExit(
            f"ERROR: --round is required when --dataset_path is a base dataset "
            f"(parsed kind={parsed.kind!r}, name={parsed.name!r}). Pass --round=<N>."
        )

    # ── Prefix / infix resolution ───────────────────────────────────────────
    # For base datasets the prefix isn't encoded in the name; in that case
    # the user MUST provide --dataset_path that points at a DAgger dataset
    # (intervention or blend) so the script can locate the lineage.
    if parsed.prefix is None:
        raise SystemExit(
            f"ERROR: --dataset_path={dataset_path} parsed as kind={parsed.kind!r}; the script needs "
            f"either an intervention / blend / merged dataset path to derive the lineage prefix, "
            f"OR you need to manually specify the lineage. (Passing a bare base-dataset path is "
            f"ambiguous — it could belong to multiple lineages.)"
        )

    prefix = parsed.prefix
    infix = parsed.infix
    assert infix is not None  # parse_dataset_short guarantees this when prefix is set

    # ── Sibling intervention + blends for the requested round ──────────────
    intervention_path = int_cache_path(args.lerobot_cache, args.hf_user, prefix, infix, round_n)
    blends = enumerate_blend_paths_on_disk(args.lerobot_cache, args.hf_user, prefix, infix, round_n)

    if not intervention_path.is_dir():
        raise SystemExit(
            f"ERROR: intervention dataset for round {round_n} missing on disk:\n"
            f"  {intervention_path}\n"
            f"  Check --round / --dataset_path / --hf_user."
        )

    # ── Base repo resolution via sidecar + train_config fallbacks ───────────
    sidecar_path = find_sidecar_by_prefix(args.training_root, prefix)
    sidecar = load_sidecar(sidecar_path) if sidecar_path else None
    base_repo, base_source = resolve_base_repo(
        sidecar, explicit_override=args.base_repo_id, hf_user=args.hf_user
    )
    if base_repo is None:
        raise SystemExit(
            f"ERROR: could not resolve base dataset repo id.\n"
            f"  prefix={prefix!r} round={round_n}\n"
            f"  sidecar={sidecar_path}\n"
            f"  Try passing --base_repo_id=<HF_USER>/<short> explicitly."
        )
    base_path = args.lerobot_cache / base_repo
    if not base_path.is_dir():
        raise SystemExit(
            f"ERROR: resolved base dataset not on disk:\n"
            f"  base_repo={base_repo} (source={base_source})\n"
            f"  expected: {base_path}\n"
            f"  Pass --base_repo_id explicitly to override."
        )

    return {
        "round": round_n,
        "prefix": prefix,
        "infix": infix,
        "intervention_path": intervention_path,
        "blends": blends,
        "base_repo": base_repo,
        "base_path": base_path,
        "base_source_tag": base_source,
        "sidecar_path": sidecar_path,
    }


def _print_resolved(resolved: dict) -> None:
    """Pretty-print the resolved dataset set (used by --dry-run + on real
    runs, before loading data)."""
    round_n = resolved["round"]
    print(f"Round: {round_n}")
    print(f"Prefix: {resolved['prefix']}  (infix={resolved['infix']!r})")
    print(f"base:         {resolved['base_path']}")
    sc = resolved["sidecar_path"]
    sc_str = str(sc) if sc else "<no sidecar found>"
    print(f"              [source={resolved['base_source_tag']} via sidecar {sc_str}]")
    print(f"intervention: {resolved['intervention_path']}")
    for pct, p in resolved["blends"]:
        print(f"blend {pct:>3d}%:   {p}")
    if not resolved["blends"]:
        print("(no blend datasets on disk for this round)")


def numpy_pca_2d(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:  # noqa: N803
    """Fit a 2D PCA basis via centered SVD. Returns (mean[D], components[2,D]).

    Caller projects via `(X - mean) @ components.T`. No sklearn dep —
    SVD on a centered N×D matrix is instant for the sizes we deal with
    (≈10⁴ chunks × ≤200 dims). Components are sorted by descending
    singular value (matches sklearn / numpy convention).
    """
    mean = X.mean(axis=0)
    centered = X - mean
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    return mean, vt[:2]


def _load_action_episodes(dataset_root: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read `action` + `episode_index` from a LeRobotDataset's parquet
    files, sorted by (episode_index, frame_index).

    Returns:
        action  — [N, D_action] float32
        episode — [N]            int64
    """
    import glob

    import pyarrow.parquet as pq

    files = sorted(glob.glob(f"{dataset_root}/data/chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files under {dataset_root}/data")
    action_parts: list[np.ndarray] = []
    episode_parts: list[np.ndarray] = []
    for f in files:
        t = pq.read_table(f, columns=["action", "episode_index", "frame_index"])
        df = t.to_pandas().sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
        action_parts.append(np.stack(df["action"].to_numpy()))
        episode_parts.append(df["episode_index"].to_numpy())
    action = np.concatenate(action_parts, axis=0).astype(np.float32)
    episode = np.concatenate(episode_parts, axis=0).astype(np.int64)
    return action, episode


def _sample_action_chunks(
    dataset_root: Path,
    n_action_steps: int,
    max_chunks: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Randomly sample up to `max_chunks` consecutive action chunks of
    length `n_action_steps` from a dataset, each chunk fully contained
    within a single episode. Returns [n_chunks, n_action_steps * D_action]
    float32 (each row = one flattened chunk).

    Each VALID (episode, start_frame_offset) pair is a candidate — i.e. any
    start such that all `n_action_steps` frames land in the same episode.
    The sample is drawn uniformly from those candidates (without replacement).
    Chunks DO NOT need to align with the policy's true chunk boundaries —
    sliding-window over consecutive same-episode frames is enough to
    represent the dataset's chunk-level distribution.
    """
    action, episode = _load_action_episodes(dataset_root)
    n_frames, d_action = action.shape
    chunk_len = n_action_steps
    if n_frames < chunk_len:
        return np.zeros((0, chunk_len * d_action), dtype=np.float32)
    # Valid starts: episode[s] == episode[s + chunk_len - 1] (chunk doesn't
    # cross an episode boundary). Vectorized over all candidate starts.
    starts = np.arange(n_frames - chunk_len + 1)
    valid_mask = episode[starts] == episode[starts + chunk_len - 1]
    valid_starts = starts[valid_mask]
    if len(valid_starts) == 0:
        return np.zeros((0, chunk_len * d_action), dtype=np.float32)
    if len(valid_starts) > max_chunks:
        chosen = rng.choice(valid_starts, size=max_chunks, replace=False)
        chosen.sort()
    else:
        chosen = valid_starts
    # Slice [n_chunks, chunk_len, d_action] then flatten so each chunk is one
    # row vector in the PCA input.
    chunks = np.stack([action[s : s + chunk_len] for s in chosen], axis=0)
    return chunks.reshape(len(chosen), chunk_len * d_action).astype(np.float32)


def _sample_episode_trajectories(
    dataset_root: Path,
    n_action_steps: int,
    num_episodes: int,
    rng: np.random.Generator,
) -> list[tuple[int, np.ndarray]]:
    """Pick `num_episodes` random episodes from a dataset and return
    each one's FULL chunk-trajectory (every sliding-window chunk of length
    `n_action_steps`, in time order). Returns a list of (episode_id, chunks)
    where chunks has shape [T_ep - n_action_steps + 1, n_action_steps * D_action].

    Episodes shorter than `n_action_steps` are skipped. Used for overlaying
    connected per-episode lines on top of the scatter."""
    action, episode = _load_action_episodes(dataset_root)
    chunk_len = n_action_steps
    ep_ids = sorted(set(episode.tolist()))
    # Drop too-short episodes BEFORE sampling so we don't waste a slot.
    eligible: list[int] = []
    for ep in ep_ids:
        if (episode == ep).sum() >= chunk_len:
            eligible.append(ep)
    if not eligible:
        return []
    if len(eligible) > num_episodes:
        chosen = rng.choice(eligible, size=num_episodes, replace=False)
        chosen = sorted(int(e) for e in chosen)
    else:
        chosen = eligible
    out: list[tuple[int, np.ndarray]] = []
    d_action = action.shape[1]
    for ep in chosen:
        ep_action = action[episode == ep]
        starts = np.arange(len(ep_action) - chunk_len + 1)
        chunks = np.stack([ep_action[s : s + chunk_len] for s in starts]).reshape(
            len(starts), chunk_len * d_action
        )
        out.append((int(ep), chunks.astype(np.float32)))
    return out


def _gather_sources(resolved: dict, n_action_steps: int, max_points: int, seed: int) -> list[dict]:
    """Load all source datasets + assemble per-source metadata for plotting.

    Returns a list of dicts, one per source, with keys:
        label   — short legend label, e.g. "base", "intervention", "b070"
        color   — matplotlib RGBA tuple
        marker  — single matplotlib marker char
        size    — scatter marker size
        alpha   — scatter alpha
        zorder  — drawing order (higher = on top)
        chunks  — [N, n_action_steps * D_action] float32 (one row per chunk)
    """
    rng = np.random.default_rng(seed)
    sources: list[dict] = []
    sample = lambda path: _sample_action_chunks(path, n_action_steps, max_points, rng)  # noqa: E731

    # Per-source visual scheme designed for maximum per-source color clarity
    # even when sources land at the same chunk (which they do — intervention
    # and blends overlap heavily because blends are closed-loop replays of
    # the same intervention episodes). Drawing order:
    #
    #   1. Base (zorder 1, filled, low alpha) — background context cloud.
    #   2. Blends (zorder 2, filled, medium alpha) — rainbow by ratio,
    #      drawn AS A FILLED FACE so each ratio's color appears at the chunk
    #      coordinate. Multiple overlapping blends mix visually via alpha.
    #   3. Intervention (zorder 3, HOLLOW outline) — black ring around each
    #      chunk coordinate. Hollow means the blend-color faces underneath
    #      stay visible THROUGH the ring, so you can see both "there's
    #      intervention here" AND "what blend colors share this location".
    #
    # Marker sizes are nested: base < blends < intervention. This lets the
    # bigger intervention rings encircle the colored blend dots so the
    # source identity reads at a glance.

    # 1. Base — distinct pastel blue, drawn underneath everything else.
    sources.append(
        {
            "label": "base",
            "color": (0.65, 0.81, 0.89, 1.0),  # ColorBrewer light blue (#a6cee3)
            "marker": "o",
            "size": 10,
            "alpha": 0.35,
            "zorder": 1,
            "filled": True,
            "chunks": sample(resolved["base_path"]),
        }
    )

    # 2. Blends — rainbow by ratio (b010 purple → b090 red). Pin the
    #    colormap to the FULL [0, 1] range so the same ratio always gets
    #    the same color across script invocations.
    blends = resolved["blends"]
    if blends:
        cmap = plt.get_cmap("rainbow")
        pcts = np.array([p for p, _ in blends], dtype=float) / 100.0
        for (pct, path), color_t in zip(blends, [cmap(p) for p in pcts], strict=True):
            sources.append(
                {
                    "label": f"b{pct:03d}",
                    "color": color_t,
                    "marker": "o",
                    "size": 22,
                    "alpha": 0.5,
                    "zorder": 2,
                    "filled": True,
                    "chunks": sample(path),
                }
            )

    # 3. Intervention — HOLLOW black ring (foreground reference). The empty
    #    interior lets the blend colors underneath remain visible, so the
    #    user can see which blend ratios share each intervention chunk.
    sources.append(
        {
            "label": "intervention",
            "color": (0.0, 0.0, 0.0, 1.0),
            "marker": "o",
            "size": 50,
            "alpha": 0.85,
            "zorder": 3,
            "filled": False,
            "chunks": sample(resolved["intervention_path"]),
        }
    )

    return sources


def _validate_chunk_dims(sources: list[dict]) -> int:
    """Assert all per-source chunk matrices share the same column count
    (= n_action_steps × D_action). Returns the shared D."""
    dims = {s["label"]: s["chunks"].shape[1] for s in sources}
    unique = set(dims.values())
    if len(unique) != 1:
        details = "\n".join(f"  {lbl}: D={d}" for lbl, d in dims.items())
        raise SystemExit(
            f"ERROR: action-chunk dim mismatch across sources:\n{details}\n"
            "All sources must be from the same robot (same DOF count + same n_action_steps) for PCA."
        )
    return unique.pop()


def plot_state_coverage_pca(
    resolved: dict,
    sources: list[dict],
    n_action_steps: int,
    out_path: Path,
    base_trajectories: list[tuple[int, np.ndarray]] | None = None,
) -> None:
    """Concat all source chunks, fit a single 2D PCA, project each source,
    and scatter-plot. Optionally overlay per-episode trajectory LINES from
    `base_trajectories` (list of (episode_id, [T, n_action_steps*D_action])
    chunk matrices, projected through the SAME PCA basis fit on the scatter
    data). Saves PNG to `out_path`."""
    all_chunks = np.concatenate([s["chunks"] for s in sources], axis=0)
    mean, components = numpy_pca_2d(all_chunks)

    fig, ax = plt.subplots(figsize=(11, 8))
    for s in sources:
        proj = (s["chunks"] - mean) @ components.T  # [N, 2]
        label = f"{s['label']}  (n={s['chunks'].shape[0]})"
        # `filled=False` sources draw as outlines (facecolors='none'), so the
        # colored blend dots underneath stay visible inside the ring.
        if s["filled"]:
            ax.scatter(
                proj[:, 0],
                proj[:, 1],
                s=s["size"],
                marker=s["marker"],
                alpha=s["alpha"],
                zorder=s["zorder"],
                c=[s["color"]],
                edgecolors="none",
                label=label,
            )
        else:
            ax.scatter(
                proj[:, 0],
                proj[:, 1],
                s=s["size"],
                marker=s["marker"],
                alpha=s["alpha"],
                zorder=s["zorder"],
                facecolors="none",
                edgecolors=[s["color"]],
                linewidths=1.2,
                label=label,
            )

    # Per-episode trajectory overlay (from `base_trajectories`). Each
    # episode's chunks projected through the same PCA basis and connected
    # as a line so the user can visually compare the trajectory DIRECTION
    # of base episodes against the intervention/blend streaks.
    if base_trajectories:
        # tab10 gives 10 distinct saturated colors; cycle if N > 10.
        tab = plt.get_cmap("tab10")
        for i, (ep_id, ep_chunks) in enumerate(base_trajectories):
            ep_proj = (ep_chunks - mean) @ components.T  # [T, 2]
            ax.plot(
                ep_proj[:, 0],
                ep_proj[:, 1],
                color=tab(i % 10),
                linewidth=1.8,
                alpha=0.9,
                zorder=4,  # ABOVE scatter so the line is clearly readable
                label=f"base ep {ep_id}  (T={len(ep_chunks)} chunks)",
            )
            # Mark the start of each trajectory with a filled dot so the
            # direction of motion is unambiguous.
            ax.scatter(
                ep_proj[0:1, 0],
                ep_proj[0:1, 1],
                s=60,
                c=[tab(i % 10)],
                edgecolors="black",
                linewidths=0.8,
                marker="o",
                zorder=5,
            )

    # Variance-explained labels — handy for judging whether 2D PCA is enough.
    _u, sv, _vt = np.linalg.svd(all_chunks - mean, full_matrices=False)
    var = (sv**2) / max(all_chunks.shape[0] - 1, 1)
    total_var = var.sum()
    pc1_pct = 100.0 * var[0] / total_var
    pc2_pct = 100.0 * var[1] / total_var

    ax.set_xlabel(f"PC1 ({pc1_pct:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pc2_pct:.1f}% var)")
    action_dim = all_chunks.shape[1] // n_action_steps
    ax.set_title(
        f"DAgger action-chunk coverage (round {resolved['round']}): "
        f"PCA scatter of {n_action_steps}-step action chunks\n"
        f"chunk dim = {n_action_steps}×{action_dim} = {all_chunks.shape[1]}  |  "
        f"lineage prefix = {resolved['prefix']}  |  "
        f"base = {resolved['base_repo']}",
        fontsize=10,
    )
    ax.legend(loc="best", fontsize=9, markerscale=1.5, framealpha=0.9)
    ax.grid(True, alpha=0.25)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _resolve_n_action_steps(args: argparse.Namespace, sidecar: dict | None) -> int:
    """Resolve the action-chunk length, in order: (1) --n_action_steps
    explicit override, (2) sidecar's initial policy's
    policy.n_action_steps. Errors if neither is available."""
    if args.n_action_steps is not None:
        return args.n_action_steps
    if sidecar is None:
        raise SystemExit(
            "ERROR: no sidecar found to auto-resolve n_action_steps; pass --n_action_steps=N explicitly."
        )
    tc = load_initial_policy_train_config(sidecar)
    if tc is None:
        raise SystemExit(
            "ERROR: could not load initial policy's train_config.json to auto-resolve n_action_steps; "
            "pass --n_action_steps=N explicitly."
        )
    n = (tc.get("policy") or {}).get("n_action_steps")
    if n is None:
        raise SystemExit(
            f"ERROR: policy.n_action_steps missing from initial policy's train_config.json "
            f"({tc.get('policy', {}).get('type', '?')!r} policy); pass --n_action_steps=N explicitly."
        )
    return int(n)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    resolved = resolve_inputs(args)
    sidecar = load_sidecar(resolved["sidecar_path"]) if resolved["sidecar_path"] else None
    n_action_steps = _resolve_n_action_steps(args, sidecar)
    _print_resolved(resolved)
    src_tag = "override" if args.n_action_steps is not None else "train_config"
    print(f"n_action_steps: {n_action_steps}  [source={src_tag}]")

    if args.dry_run:
        return 0

    # Default out path: outputs/dagger/state_coverage/<prefix>_dag<R>_state_coverage_pca.png
    if args.out is None:
        out_path = DEFAULT_OUT_DIR / f"{resolved['prefix']}_dag{resolved['round']}_state_coverage_pca.png"
    else:
        out_path = args.out.expanduser().resolve()

    print()
    n_sources = 1 + 1 + len(resolved["blends"])
    print(
        f"Loading action chunks (length={n_action_steps}) from {n_sources} dataset(s) "
        f"(subsample <= {args.max_points_per_source} chunks per source, seed={args.seed})..."
    )
    sources = _gather_sources(resolved, n_action_steps, args.max_points_per_source, args.seed)
    chunk_dim = _validate_chunk_dims(sources)
    print(
        f"  loaded {len(sources)} source(s), chunk dim D={chunk_dim} "
        f"(= {n_action_steps} × {chunk_dim // n_action_steps})"
    )
    for s in sources:
        print(f"    {s['label']:>14s}: n_chunks={s['chunks'].shape[0]}")

    # Optional: sample a few base episodes to overlay as connected
    # trajectory lines (lets the user see whether base episodes flow in
    # the same direction as intervention in PCA-space).
    base_trajectories: list[tuple[int, np.ndarray]] = []
    if args.num_base_trajectories > 0:
        traj_rng = np.random.default_rng(args.seed + 1)  # separate stream
        base_trajectories = _sample_episode_trajectories(
            resolved["base_path"], n_action_steps, args.num_base_trajectories, traj_rng
        )
        print(f"  + {len(base_trajectories)} base-episode trajectory overlay(s):")
        for ep_id, chunks in base_trajectories:
            print(f"      ep {ep_id}: {len(chunks)} chunks")

    print(f"\nFitting 2D PCA + plotting to {out_path}...")
    plot_state_coverage_pca(
        resolved,
        sources,
        n_action_steps,
        out_path,
        base_trajectories=base_trajectories,
    )
    print(f"  wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
