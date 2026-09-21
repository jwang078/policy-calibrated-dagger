#!/usr/bin/env python3
"""Print per-round DAgger training-data composition for a given policy dir.

Given a policy training directory like

    outputs/training/diffusion_..._ft_dag3

reads its sidecar `dagger/config.json`, walks the on-disk intervention /
blend datasets the orchestrator would have used, and reports for EACH
round r in 1..N the breakdown of frames that round's training set was
composed of: base dataset + per-round intervention(s) + per-round blend(s).

Mode is auto-detected from the sidecar (`config.use_weighted_sampling`):

- **merged mode** (default): each round's merge composes
    `base + sum_{r=1..R} [raw_int_r × (target_volume - n_blends)
                          + each_blend_r × 1]`
  This mirrors dagger_orchestrate.sh's step 4. Total intervention content
  per round is always `target_volume × raw_int_r` regardless of n_blends.
- **weighted mode**: the round trains directly against the union
    `{base, every round's raw intervention, every round's blends}`
  with a WeightedRandomSampler enforcing per-batch composition
    `base = 1 - dagger_data_fraction`,
    `each DAgger sub-dataset = dagger_data_fraction / num_subs`.
  We report both the on-disk sub-dataset sizes AND the effective per-batch
  fractions, since the "total frames per round" concept doesn't have a
  single natural value in this mode.

Examples:
    python my_scripts/dagger_dataset_distribution.py \\
        outputs/training/diffusion_..._d5jvm_resweep_ft_dag10

    python my_scripts/dagger_dataset_distribution.py \\
        outputs/training/diffusion_..._d5jvm_resweep_rr_b010_ft_dag10
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# dagger_naming sits next to this script.
sys.path.insert(0, str(Path(__file__).parent))
import dagger_naming as dn  # noqa: E402

_ROUND_DAG_RE = re.compile(r"_(?:ft_)?dag(\d+)(?:_|$)")


def _read_total_frames(dataset_root: Path) -> int | None:
    """Return `total_frames` from `<dataset_root>/meta/info.json`, or None
    if the dataset isn't on disk (deleted post-training, etc)."""
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        return None
    try:
        return int(json.loads(info_path.read_text())["total_frames"])
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return None


def _format_with_pct(count: int, total: int) -> str:
    pct = 100.0 * count / total if total else 0.0
    return f"{count:>10,}  ({pct:5.1f}%)"


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "training_dir",
        type=Path,
        help="Policy training dir, e.g. outputs/training/..._ft_dag3. "
        "Must contain dagger/config.json (the orchestrator sidecar).",
    )
    p.add_argument(
        "--hf_user",
        default="JennyWWW",
        help="HuggingFace user namespace for dataset lookups (default: JennyWWW).",
    )
    p.add_argument(
        "--lerobot_cache",
        type=Path,
        default=Path.home() / ".cache/huggingface/lerobot",
        help="LeRobot dataset cache root (default: ~/.cache/huggingface/lerobot).",
    )
    p.add_argument(
        "--base_repo",
        default=None,
        help="Override base-dataset repo id (skip sidecar resolution). Format: <user>/<dataset>.",
    )
    args = p.parse_args(argv)

    train_dir: Path = args.training_dir.resolve()
    sidecar_path = train_dir / "dagger" / "config.json"
    if not sidecar_path.is_file():
        print(f"ERROR: no sidecar at {sidecar_path}", file=sys.stderr)
        print(
            "       this script expects a DAgger orchestrator training dir "
            "(round 1 onwards). Did you point at a round-0 base policy dir?",
            file=sys.stderr,
        )
        return 1

    sidecar = dn.load_sidecar(sidecar_path)

    m = _ROUND_DAG_RE.search(train_dir.name)
    if not m:
        print(
            f"ERROR: cannot extract round number from {train_dir.name!r}; "
            f"expected a `..._dag<N>` or `..._ft_dag<N>` basename.",
            file=sys.stderr,
        )
        return 1
    n_rounds = int(m.group(1))

    cfg = sidecar.get("config") or {}
    naming = sidecar.get("naming") or {}
    rerun = sidecar.get("rerun_mode") or {}

    base_repo, base_src = dn.resolve_base_repo(
        sidecar, explicit_override=args.base_repo, hf_user=args.hf_user
    )
    if not base_repo:
        print(
            "ERROR: could not resolve base dataset repo from sidecar. "
            "Pass --base_repo=<user>/<dataset> explicitly.",
            file=sys.stderr,
        )
        return 1

    action_format = (cfg.get("action_format") or "rel").lower()
    if action_format not in ("rel", "abs"):
        print(f"ERROR: unknown action_format {action_format!r} in sidecar", file=sys.stderr)
        return 1
    infix = "r" if action_format == "rel" else "a"

    # In rerun-blends mode the intervention + blend datasets use the SOURCE
    # lineage's prefix (rerun_mode.source_int_short_prefix), not the rerun's
    # own base_dataset_short. CLAUDE.md ("DAgger rerun-blends mode") covers this.
    if rerun:
        int_prefix = rerun.get("source_int_short_prefix")
        if not int_prefix:
            print("ERROR: rerun_mode set but source_int_short_prefix missing", file=sys.stderr)
            return 1
    else:
        int_prefix = naming.get("base_dataset_short")
        if not int_prefix:
            print("ERROR: naming.base_dataset_short missing from sidecar", file=sys.stderr)
            return 1

    blends: list[float] = list(cfg.get("blends") or [])
    n_blends = len(blends)
    use_weighted = bool(cfg.get("use_weighted_sampling", False))
    target_volume = int(cfg.get("target_intervention_volume") or 0)
    dagger_fraction = float(cfg.get("dagger_data_fraction") or 0.0)
    raw_multiplier = target_volume - n_blends if not use_weighted else None

    if not use_weighted and target_volume <= 0:
        print(
            "WARNING: merged mode but target_intervention_volume is 0/None — "
            "treating raw multiplier as 1 (degenerate config).",
            file=sys.stderr,
        )
        raw_multiplier = 1

    cache_root: Path = args.lerobot_cache
    base_cache = cache_root / base_repo
    base_total = _read_total_frames(base_cache)
    if base_total is None:
        print(
            f"WARNING: base dataset not found on disk at {base_cache}; "
            f"reporting size as 0 (downstream percentages will be wrong).",
            file=sys.stderr,
        )
        base_total = 0

    # Pre-resolve every round's on-disk frame counts so we can show
    # per-round cumulative composition without re-reading files in the loop.
    round_data: list[dict] = []
    for r in range(1, n_rounds + 1):
        int_path = dn.int_cache_path(cache_root, args.hf_user, int_prefix, infix, r)
        rd = {
            "r": r,
            "int_path": int_path,
            "int_total": _read_total_frames(int_path),
            "blends": {},  # ratio → (path, total)
        }
        for ratio in blends:
            blend_path = dn.blend_cache_path(cache_root, args.hf_user, int_prefix, infix, r, ratio)
            rd["blends"][ratio] = (blend_path, _read_total_frames(blend_path))
        round_data.append(rd)

    # ── Header
    print("═" * 78)
    print(f"DAgger lineage:       {naming.get('base_dataset_short') or '?'}")
    print(f"Base dataset:         {base_repo}  [{base_src}]")
    print(f"  on-disk:            {base_cache}  ({base_total:,} frames)")
    print(f"Intervention prefix:  {int_prefix}" + ("  (rerun-blends mode)" if rerun else ""))
    print(f"Action format:        {action_format}  → infix '{infix}'")
    print(f"Rounds available:     1..{n_rounds}")
    if use_weighted:
        print("Mode:                 WEIGHTED-SAMPLING")
        print(
            f"  dagger_data_fraction = {dagger_fraction}  "
            f"→ base gets {100 * (1 - dagger_fraction):.1f}% per batch, "
            f"DAgger sub-datasets share {100 * dagger_fraction:.1f}%"
        )
    else:
        print("Mode:                 MERGED")
        print(f"  target_intervention_volume = {target_volume}")
        print(f"  blends = {blends}  (n_blends = {n_blends})")
        print(
            f"  per-round merge composes:  raw × {raw_multiplier} + "
            f"each_blend × 1  =  {target_volume} × raw equivalent"
        )
    print("═" * 78)
    print()

    # ── Per-round breakdown
    for cur_r in range(1, n_rounds + 1):
        if use_weighted:
            _print_weighted_round(cur_r, round_data, blends, base_total, dagger_fraction)
        else:
            _print_merged_round(cur_r, round_data, blends, base_total, raw_multiplier)

    return 0


def _print_merged_round(
    cur_r: int,
    round_data: list[dict],
    blends: list[float],
    base_total: int,
    raw_multiplier: int,
) -> None:
    """Print one round's merged composition."""
    # Sources for round cur_r's merged training set:
    #   base + sum_{r=1..cur_r}: raw × raw_multiplier + each blend × 1
    sources: list[tuple[str, int, bool]] = [("Base dataset", base_total, False)]
    for r in range(1, cur_r + 1):
        rd = round_data[r - 1]
        int_total = rd["int_total"]
        if int_total is None:
            sources.append((f"Round {r} intervention (×{raw_multiplier})  [MISSING]", 0, True))
        else:
            sources.append(
                (
                    f"Round {r} intervention (×{raw_multiplier})",
                    int_total * raw_multiplier,
                    False,
                )
            )
        for ratio in blends:
            tag = dn.blend_tag_for_ratio(ratio)
            _, blend_total = rd["blends"][ratio]
            if blend_total is None:
                sources.append((f"Round {r} blend{tag}  [MISSING]", 0, True))
            else:
                sources.append((f"Round {r} blend{tag}", blend_total, False))

    total = sum(cnt for _, cnt, _ in sources)

    # Aggregate buckets: base vs all-dagger
    dagger_total = total - base_total
    base_pct = 100.0 * base_total / total if total else 0.0
    dagger_pct = 100.0 * dagger_total / total if total else 0.0

    print(f"━━ DAgger round {cur_r}: merged training set ━━")
    print(f"   Total frames:     {total:>10,}")
    print(
        f"   Base vs DAgger:   base = {_format_with_pct(base_total, total)},  "
        f"dagger = {_format_with_pct(dagger_total, total)}"
    )
    print()
    for label, cnt, missing in sources:
        prefix = "  ! " if missing else "    "
        print(f"{prefix}{label:<46s}  {_format_with_pct(cnt, total)}")
    print()


def _print_weighted_round(
    cur_r: int,
    round_data: list[dict],
    blends: list[float],
    base_total: int,
    dagger_fraction: float,
) -> None:
    """Print one round's weighted-sampling composition (per-batch fractions
    + on-disk sub-dataset sizes)."""
    n_blends = len(blends)
    # Per CLAUDE.md "DAgger weighted multi-dataset sampling":
    #   base gets `1 - fraction`
    #   DAgger sub-datasets share `fraction` EQUALLY
    n_dagger_subs = cur_r * (1 + n_blends)
    base_frac = 1.0 - dagger_fraction
    per_sub_frac = dagger_fraction / n_dagger_subs if n_dagger_subs else 0.0

    print(f"━━ DAgger round {cur_r}: weighted-sampling composition ━━")
    print(f"   {n_dagger_subs} DAgger sub-dataset(s) (rounds 1..{cur_r}, each = 1 raw + {n_blends} blend(s))")
    print(
        f"   Per-batch fractions:  "
        f"base = {100 * base_frac:5.1f}%,  "
        f"each DAgger sub = {100 * per_sub_frac:5.2f}%"
    )
    print()
    print(f"   {'Source':<46s}  {'on-disk size':>12s}  {'per-batch':>10s}")
    print(f"   {'-' * 46}  {'-' * 12}  {'-' * 10}")
    print(f"   {'Base dataset':<46s}  {base_total:>12,}  {100 * base_frac:>9.1f}%")
    for r in range(1, cur_r + 1):
        rd = round_data[r - 1]
        int_total = rd["int_total"]
        if int_total is None:
            print(f" ! {f'Round {r} intervention  [MISSING]':<46s}  {'-':>12s}  {100 * per_sub_frac:>9.2f}%")
        else:
            print(f"   {f'Round {r} intervention':<46s}  {int_total:>12,}  {100 * per_sub_frac:>9.2f}%")
        for ratio in blends:
            tag = dn.blend_tag_for_ratio(ratio)
            _, blend_total = rd["blends"][ratio]
            if blend_total is None:
                print(
                    f" ! {f'Round {r} blend{tag}  [MISSING]':<46s}  {'-':>12s}  {100 * per_sub_frac:>9.2f}%"
                )
            else:
                print(f"   {f'Round {r} blend{tag}':<46s}  {blend_total:>12,}  {100 * per_sub_frac:>9.2f}%")
    print()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
