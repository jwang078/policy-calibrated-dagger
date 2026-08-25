#!/usr/bin/env python3
"""Mean ± std of success rate across repeated DAgger runs.

Groups the lineages produced by my_scripts/dagger_orchestrate_repeat.sh:
repetition 1 uses the bare tag (source lineage `..._03dag`, rerun lineages
`..._03dag_rr_b090`, ...), repetition k >= 2 suffixes the tag with k
(`..._03dag2`, `..._03dag2_rr_b090`). Lineages that differ only in that rep
number form a FAMILY. For each family this script plots success rate vs
DAgger round: a line through the per-round means, a gray shaded ±1 std band,
and (with --plot-each) faint per-rep curves. A combined figure overlays every family's mean +
band (source family = black, blend families colored by ratio).

Per-round success comes from dagger_plot.collect_lineage_rows — i.e. the
same reeval > train-time-eval_info > wandb-log cascade the other DAgger
plots use. Only main-curve rounds (variant == "ft", which includes round 0)
are aggregated; scratch/retrain variants are ignored — except the
`--final_mode=base_finetune` control (`..._dag<N>_bft`: one stationary
finetune from the round-0 base on the full aggregate), which is overlaid as a
star marker with its own mean ± std at round N.

Usage:
    python my_scripts/dagger_plot_repeats.py --rep_tag=03dag --model=diff
    # optional: --filter=planar_3joint_12   (substring; drops families from
    #           other experiments that happen to reuse the same tag)
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dagger_plot import DEFAULT_OUT_DIR, collect_lineage_rows, discover_lineages  # noqa: E402

# --model accepts the orchestrator's short spelling or the training-dir prefix.
MODEL_PREFIXES = {"diff": "diffusion", "pi": "pi05", "act": "act"}

BLEND_TAG_RE = re.compile(r"_b(\d{3}(?:_\d{3})*)$")


def family_and_rep(lineage: str, tag: str) -> tuple[str, int] | None:
    """Map a lineage name to (family, rep).

    The family is the lineage with the rep number stripped from the tag
    component; rep 1 has no number. Returns None when the tag doesn't appear
    as a `_<tag><digits?>` component.
    """
    pat = re.compile(rf"_{re.escape(tag)}(\d*)(?=_|$)")
    matches = list(pat.finditer(lineage))
    if len(matches) != 1:
        if len(matches) > 1:
            print(f"WARNING: tag '{tag}' appears {len(matches)}x in lineage '{lineage}'; skipping.")
        return None
    m = matches[0]
    rep = int(m.group(1)) if m.group(1) else 1
    family = lineage[: m.start()] + f"_{tag}" + lineage[m.end() :]
    return family, rep


def family_color(family: str, tag: str) -> tuple:
    """Color for a family's curve.

    Source family (no blends tag) → black; blend families → rainbow by the
    first (highest) ratio in the blends tag, matching dagger_plot's ratio hue
    direction.
    """
    m = BLEND_TAG_RE.search(family)
    if not m:
        return (0.0, 0.0, 0.0, 1.0)
    first_pct = int(m.group(1).split("_")[0])
    return matplotlib.colormaps.get_cmap("rainbow")(first_pct / 100.0)


def assign_family_colors(families, tag: str) -> dict:
    """{family: (color, linestyle)} with every family visually distinct.

    `family_color` is ratio-based, so two lineages that differ only outside the
    blends tag (e.g. `..._03dag_b050` vs `..._03dag_rr_b050`, separate lineages
    sharing ratio 0.5) would otherwise draw in the exact same color. Families
    that collide on a base color are spread over a lightness ramp and given
    distinct linestyles so each curve is readable on its own.
    """
    by_color: dict[tuple, list[str]] = {}
    for fam in sorted(families):
        by_color.setdefault(tuple(np.round(family_color(fam, tag), 6)), []).append(fam)

    styles = ["-", "--", "-.", ":"]
    out: dict[str, tuple] = {}
    for base, fams in by_color.items():
        rgb = np.asarray(base[:3])
        for i, fam in enumerate(fams):
            if len(fams) == 1:
                color = tuple(base)
            else:
                # Darken along a ramp: member 0 keeps the base ratio color, the
                # rest get progressively deeper shades of it (staying legible on
                # white, unlike ramping toward white).
                f = 1.0 - 0.55 * i / (len(fams) - 1)
                color = (*(rgb * f), 1.0)
            out[fam] = (color, styles[i % len(styles)])
    return out


def collect_family(
    lineages_by_rep: dict[int, str], model: str, prefer_reeval: bool
) -> dict[int, dict[int, float]]:
    """{rep: {round: succ}} for one family, main curve only."""
    out: dict[int, dict[int, float]] = {}
    for rep, lineage in sorted(lineages_by_rep.items()):
        per_round: dict[int, float] = {}
        for row in collect_lineage_rows(lineage, model, prefer_reeval=prefer_reeval):
            if row.get("variant") != "ft" or row.get("succ") is None:
                continue
            per_round[row["round"]] = float(row["succ"])
        if per_round:
            out[rep] = per_round
    return out


def collect_family_base_finetune(
    lineages_by_rep: dict[int, str], model: str, prefer_reeval: bool
) -> dict[int, dict[int, float]]:
    """{rep: {round: succ}} for the `--final_mode=base_finetune` runs of a family.

    These are off-curve: ONE stationary finetune from the round-0 base
    checkpoint on the full aggregate, sharing the round number of the last
    DAgger round (`..._dag<N>_bft`). Plotted as a separate marker — it's the
    matched-compute "no DAgger loop" control for that round's finetune.
    """
    out: dict[int, dict[int, float]] = {}
    for rep, lineage in sorted(lineages_by_rep.items()):
        per_round: dict[int, float] = {}
        for row in collect_lineage_rows(lineage, model, prefer_reeval=prefer_reeval):
            if not row.get("is_base_finetune") or row.get("succ") is None:
                continue
            per_round[row["round"]] = float(row["succ"])
        if per_round:
            out[rep] = per_round
    return out


def draw_base_finetune(ax, bft_reps: dict[int, dict[int, float]], max_round: int, color, label=None):
    """Scatter the base-finetune control(s) as mean ± std error bars.

    One point per round that has data, drawn with a star marker so it reads as
    off-curve against the DAgger line. Rounds beyond `max_round` (the plotted
    x-range) are dropped.
    """
    rounds = sorted({r for pr in bft_reps.values() for r in pr if r <= max_round})
    if not rounds:
        return False
    mean, err = [], []
    for r in rounds:
        vals = [pr[r] for pr in bft_reps.values() if r in pr]
        mean.append(float(np.mean(vals)))
        err.append(float(np.std(vals, ddof=1)) if len(vals) >= 2 else 0.0)
    ax.errorbar(
        rounds,
        mean,
        yerr=err,
        fmt="*",
        markersize=13,
        color=color,
        markeredgecolor="0.2",
        markeredgewidth=0.6,
        capsize=4,
        linestyle="none",
        label=label,
        zorder=5,
    )
    return True


def band_stats(reps: dict[int, dict[int, float]]) -> tuple[list[int], np.ndarray, np.ndarray, list[int]]:
    """(rounds, mean, std, n) over whatever reps have data at each round.

    std is the sample std (ddof=1); 0 where only one rep has the round.
    """
    rounds = sorted({r for per_round in reps.values() for r in per_round})
    mean, std, n = [], [], []
    for r in rounds:
        vals = [per_round[r] for per_round in reps.values() if r in per_round]
        mean.append(float(np.mean(vals)))
        std.append(float(np.std(vals, ddof=1)) if len(vals) >= 2 else 0.0)
        n.append(len(vals))
    return rounds, np.asarray(mean), np.asarray(std), n


def draw_band(
    ax,
    rounds,
    mean,
    std,
    color,
    band_color=None,
    band_alpha=0.3,
    label=None,
    band_from_round=1,
    linestyle="-",
):
    """Mean line + shaded ±1 std band on `ax`.

    The band starts at `band_from_round` (default 1): round 0 is the shared
    base training, identical across every rep/lineage, so it has no spread to
    show — the polygon simply starts at round 1 rather than pinching to a point.
    """
    rr = np.asarray(rounds)
    m = rr >= band_from_round
    if m.any():
        ax.fill_between(
            rr[m],
            (mean - std)[m],
            (mean + std)[m],
            color=band_color or color,
            alpha=band_alpha,
            linewidth=0,
        )
    ax.plot(rounds, mean, marker="o", linestyle=linestyle, color=color, label=label, markersize=4)


def main() -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rep_tag", required=True, help="Base run tag of repetition 1 (e.g. 03dag).")
    ap.add_argument(
        "--model",
        default="diff",
        help="Orchestrator model (diff/pi/act) or training-dir prefix (diffusion/pi05/act). Default diff.",
    )
    ap.add_argument(
        "--filter",
        default="",
        help="Only keep families whose name contains this substring (scopes out other "
        "experiments that reuse the same tag).",
    )
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR / "repeats"))
    ap.add_argument("--max_round", type=int, default=None, help="Ignore rounds with an index beyond this.")
    ap.add_argument(
        "--max_rounds",
        type=int,
        default=None,
        help="Cap the highest DAgger round taken from each rep: keeps round 0 through round N "
        "(N+1 points). Applied after --max_round; also shrinks the plotted x range.",
    )
    ap.add_argument("--no_reeval", action="store_true", help="Ignore reevals; use training-time eval only.")
    ap.add_argument(
        "--plot_each",
        dest="plot_each",
        action="store_true",
        help="Also draw the faint individual-rep curves (off by default).",
    )
    ap.add_argument(
        "--no_per_rep_lines",
        action="store_true",
        help="Deprecated no-op (individual-rep curves are now off unless --plot-each is passed).",
    )
    args = ap.parse_args()

    model = MODEL_PREFIXES.get(args.model, args.model)
    prefer_reeval = not args.no_reeval

    families: dict[str, dict[int, str]] = defaultdict(dict)
    for lineage in discover_lineages(model):
        fr = family_and_rep(lineage, args.rep_tag)
        if fr is None:
            continue
        family, rep = fr
        if args.filter and args.filter not in family:
            continue
        families[family][rep] = lineage
    if not families:
        print(f"No lineages matching tag '{args.rep_tag}' (model prefix '{model}', filter '{args.filter}').")
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── per-family figures + stats table ────────────────────────────────────
    family_stats: dict[str, tuple] = {}
    written: list[tuple[str, Path]] = []
    for family, by_rep in sorted(families.items()):
        reps = collect_family(by_rep, model, prefer_reeval)
        if not reps:
            print(f"[{family}] reps on disk: {sorted(by_rep)} — no eval data yet, skipping.")
            continue
        if args.max_round is not None:
            reps = {k: {r: s for r, s in pr.items() if r <= args.max_round} for k, pr in reps.items()}
            reps = {k: pr for k, pr in reps.items() if pr}
        if args.max_rounds is not None:
            reps = {k: {r: pr[r] for r in sorted(pr)[: args.max_rounds + 1]} for k, pr in reps.items()}
            reps = {k: pr for k, pr in reps.items() if pr}
        rounds, mean, std, n = band_stats(reps)
        bft_reps = collect_family_base_finetune(by_rep, model, prefer_reeval)
        family_stats[family] = (rounds, mean, std, n, reps, bft_reps)

        fig, ax = plt.subplots(figsize=(9, 5.5))
        if args.plot_each:
            for rep, per_round in sorted(reps.items()):
                rr = sorted(per_round)
                ax.plot(
                    rr,
                    [per_round[r] for r in rr],
                    "-",
                    color="0.55",
                    alpha=0.45,
                    linewidth=0.9,
                    label="individual reps" if rep == min(reps) else None,
                )
        draw_band(ax, rounds, mean, std, color="tab:blue", band_color="0.5", label="mean ± 1 std")
        draw_base_finetune(ax, bft_reps, max(rounds), color="tab:orange", label="base-finetune control")
        ax.set_xlabel("DAgger round")
        ax.set_ylabel("success rate (%)")
        ax.set_xticks(rounds)
        # A touch of x padding so a base-finetune star on the last round isn't
        # clipped by the axes spine.
        ax.set_xlim(min(rounds) - 0.35, max(rounds) + 0.45)
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
        n_reps = len(reps)
        ax.set_title(f"{model}_{family}\n{n_reps} repetition(s); band = ±1 sample std")
        ax.legend(loc="lower right", fontsize=9)
        fig.tight_layout()
        out = out_dir / f"repeats_{model}_{family}_succ.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        written.append((family, out))
        print(f"[{family}] reps={sorted(reps)} → {out}")
        for r, m_, s_, cnt in zip(rounds, mean, std, n):
            if r == 0:
                # round 0 is the shared base checkpoint (trained once for all reps),
                # so there is no across-rep variation to report.
                print(f"    round {r:>2}: {m_:6.1f} %          (shared base)")
            else:
                print(f"    round {r:>2}: {m_:6.1f} ± {s_:5.1f} %  (n={cnt})")
        for r in sorted({rr for pr in bft_reps.values() for rr in pr if rr <= max(rounds)}):
            vals = [pr[r] for pr in bft_reps.values() if r in pr]
            m_ = float(np.mean(vals))
            s_ = float(np.std(vals, ddof=1)) if len(vals) >= 2 else 0.0
            print(f"    round {r:>2}: {m_:6.1f} ± {s_:5.1f} %  (n={len(vals)})  [base-finetune]")

    if not family_stats:
        print("No family had any eval data; nothing plotted.")
        return 1

    # ── combined overlay ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    styles_by_family = assign_family_colors(family_stats.keys(), args.rep_tag)
    for family, (rounds, mean, std, _n, _reps, bft_reps) in sorted(family_stats.items()):
        color, linestyle = styles_by_family[family]
        # Label: the part of the family from the tag onward (short + unique).
        idx = family.find(args.rep_tag)
        label = family[idx:] if idx >= 0 else family
        draw_band(ax, rounds, mean, std, color=color, band_alpha=0.15, label=label, linestyle=linestyle)
        draw_base_finetune(ax, bft_reps, max(rounds), color=color, label=f"{label} bft")
    ax.set_xlabel("DAgger round")
    ax.set_ylabel("success rate (%)")
    ax.set_ylim(0, 100)
    _all_rounds = [r for (rounds, *_rest) in family_stats.values() for r in rounds]
    ax.set_xlim(min(_all_rounds) - 0.35, max(_all_rounds) + 0.45)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=0.3)
    ax.set_title(f"success rate across repetitions (mean ± 1 std) — tag '{args.rep_tag}'")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out = out_dir / f"repeats_{model}_{args.rep_tag}_combined_succ.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    written.append(("combined", out))
    print(f"[combined] {len(family_stats)} families → {out}")

    # ── figure paths again, together (the per-family ones scroll off above) ──
    print("\nfigures written:")
    for _name, path in written:
        print(f"    {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
