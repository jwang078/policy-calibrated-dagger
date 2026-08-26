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
    # combined figure with only some families (per-family figures still all written):
    python my_scripts/dagger_plot_repeats.py --rep_tag=03dag --combine='03dag,03dag_b050,03dag_rr_b050anc8'
    #           (or a glob: --combine='*b050*')
    # ...or pick them from a prompt (prints the --combine string to reuse):
    python my_scripts/dagger_plot_repeats.py --rep_tag=03dag --combine_interactive
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from fnmatch import fnmatch
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

# Unanchored on purpose: a family may carry a variant suffix after the blends
# tag (`_b050anc8`, `_b050a010`) and should still take its ratio's hue rather
# than falling through to the source family's black.
BLEND_TAG_RE = re.compile(r"_b(\d{3}(?:_\d{3})*)")


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

    `family_color` is ratio-based, so lineages differing only outside the blends
    tag (`..._03dag_b050`, `..._03dag_rr_b050`, `..._03dag_rr_b050anc8` — all
    ratio 0.5) share a base color, and every non-blend family would come back
    black. The first member of a colliding group keeps the ratio color; the
    rest take colors from a muted qualitative palette (deliberately unlike the
    saturated `rainbow` hues, so they can't be mistaken for another ratio)
    rather than shades of the same hue, which read as one curve at a glance.
    Linestyles vary within a group as a second cue.
    """
    by_color: dict[tuple, list[str]] = {}
    for fam in sorted(families):
        by_color.setdefault(tuple(np.round(family_color(fam, tag), 6)), []).append(fam)

    styles = ["-", "--", "-.", ":"]
    fallback = ["#7f7f7f", "#8c564b", "#9467bd", "#e377c2", "#bcbd22", "#4c566a", "#c49102"]
    fallback_used = 0
    out: dict[str, tuple] = {}
    for base, fams in by_color.items():
        for i, fam in enumerate(fams):
            if i == 0:
                color = tuple(base)
            else:
                color = fallback[fallback_used % len(fallback)]
                fallback_used += 1
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


def training_progress(dir_path: Path) -> tuple[int | None, int | None]:
    """(last checkpoint step, configured step target) for a training dir.

    Either element is None when it can't be read (no checkpoints yet, no
    train_config.json). The target is the `steps` the run was launched with —
    for a `_bft` run that's the compute-matched budget the orchestrator
    inferred from the last DAgger round.
    """
    ckpt_dir = dir_path / "checkpoints"
    done: int | None = None
    steps = sorted(int(d.name) for d in ckpt_dir.glob("[0-9]*") if d.is_dir() and d.name.isdigit())
    if steps:
        done = steps[-1]
    target: int | None = None
    for cfg in (ckpt_dir / "last" / "pretrained_model" / "train_config.json", dir_path / "train_config.json"):
        try:
            target = int(json.loads(cfg.read_text())["steps"])
            break
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return done, target


def training_is_complete(dir_path: Path) -> bool:
    """Did this training reach its configured step target?

    A run whose target can't be read (no train_config.json yet) or that has no
    checkpoints counts as NOT complete: the point of the check is that only a
    full-budget run is a valid matched-compute control, so an unverifiable one
    is excluded rather than assumed good. Every exclusion is printed, so it's
    visible rather than silent.
    """
    done, target = training_progress(dir_path)
    return target is not None and done is not None and done >= target


def collect_family_base_finetune(
    lineages_by_rep: dict[int, str], model: str, prefer_reeval: bool
) -> tuple[dict[int, dict[int, float]], list[tuple[str, int | None, int | None]]]:
    """({rep: {round: succ}}, [(dir name, steps done, steps target), ...]).

    These are off-curve: ONE stationary finetune from the round-0 base
    checkpoint on the full aggregate, sharing the round number of the last
    DAgger round (`..._dag<N>_bft`). Plotted as a separate marker — it's the
    matched-compute "no DAgger loop" control for that round's finetune.

    A `_bft` run still in progress is EXCLUDED (and returned in the second
    element for reporting): its eval reflects a partial step budget, so it is
    neither a valid control on its own nor something to average with completed
    reps of the same family.
    """
    out: dict[int, dict[int, float]] = {}
    unfinished: list[tuple[str, int | None, int | None]] = []
    for rep, lineage in sorted(lineages_by_rep.items()):
        per_round: dict[int, float] = {}
        for row in collect_lineage_rows(lineage, model, prefer_reeval=prefer_reeval):
            if not row.get("is_base_finetune") or row.get("succ") is None:
                continue
            d = Path(row["dir"]) if row.get("dir") else None
            if d is not None and not training_is_complete(d):
                done, target = training_progress(d)
                unfinished.append((d.name, done, target))
                continue
            per_round[row["round"]] = float(row["succ"])
        if per_round:
            out[rep] = per_round
    return out, unfinished


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


def family_label(family: str, tag: str) -> str:
    """Short, unique name for a family: the part from the rep tag onward.

    `diffusion_planar_..._03dag_rr_b050` → `03dag_rr_b050`. This is what the
    combined plot's legend shows and what --combine matches against.
    """
    idx = family.find(tag)
    return family[idx:] if idx >= 0 else family


def select_families(family_stats: dict, tag: str, patterns: list[str]) -> list[str]:
    """Families to overlay on the combined figure, in the order requested.

    Each pattern is matched against both the short label and the full family
    name, trying exact match, then glob (`*`/`?`), then substring — so
    `--combine=03dag_b050` picks exactly one family while `--combine='*b050*'`
    picks every b050 variant. A pattern that matches nothing raises ValueError
    with the available labels; duplicates across patterns are dropped, keeping
    first-requested order.
    """
    labels = {f: family_label(f, tag) for f in family_stats}
    chosen: list[str] = []
    for pat in patterns:
        exact = [f for f, lab in labels.items() if pat in (lab, f)]
        glob = [f for f, lab in labels.items() if fnmatch(lab, pat) or fnmatch(f, pat)]
        sub = [f for f, lab in labels.items() if pat in lab or pat in f]
        hits = exact or glob or sub
        if not hits:
            raise ValueError(
                f"--combine pattern {pat!r} matched no family. Available: "
                + ", ".join(sorted(labels.values()))
            )
        for f in sorted(hits):
            if f not in chosen:
                chosen.append(f)
    return chosen


def prompt_for_families(family_stats: dict, tag: str) -> list[str]:
    """Interactive picker: numbered list on stderr, selection read from stdin.

    Accepts numbers, ranges (`1-3`), `all`/empty for everything. Prints the
    equivalent --combine string afterwards so the choice can be pasted into the
    next invocation instead of re-answering the prompt.
    """
    ordered = sorted(family_stats)
    print("\nfamilies available for the combined overlay:", file=sys.stderr)
    for i, f in enumerate(ordered, 1):
        rounds = family_stats[f][0]
        n_reps = len(family_stats[f][4])
        print(
            f"  {i:>2}. {family_label(f, tag)}  ({n_reps} rep(s), rounds {min(rounds)}-{max(rounds)})",
            file=sys.stderr,
        )
    raw = input("select (e.g. '1,3-4', blank = all): ").strip()
    if not raw or raw.lower() == "all":
        return ordered
    picked: list[str] = []
    for tok in raw.replace(" ", ",").split(","):
        if not tok:
            continue
        if "-" in tok[1:]:
            a, b = tok.split("-", 1)
            idxs = range(int(a), int(b) + 1)
        else:
            idxs = [int(tok)]
        for i in idxs:
            if not 1 <= i <= len(ordered):
                raise ValueError(f"selection {i} out of range 1-{len(ordered)}")
            if ordered[i - 1] not in picked:
                picked.append(ordered[i - 1])
    combine = ",".join(family_label(f, tag) for f in picked)
    print(f"(reuse this selection non-interactively with --combine='{combine}')", file=sys.stderr)
    return picked


def combined_out_name(model: str, tag: str, selected: list[str], all_families: list[str]) -> str:
    """Filename for the combined figure; a subset gets its own name.

    Plotting every family keeps the historical `..._<tag>_combined_succ.png`.
    A subset is written to `..._<tag>_combined_<labels>_succ.png` (labels with
    the tag prefix stripped and joined by `+`), so a selective run never
    overwrites the full overlay — and two runs with the same selection land on
    the same file. Long selections fall back to a count + short hash.
    """
    if len(selected) == len(all_families):
        return f"repeats_{model}_{tag}_combined_succ.png"
    parts = []
    for f in selected:
        lab = family_label(f, tag)
        short = lab[len(tag) :].lstrip("_") or "base"
        parts.append(re.sub(r"[^A-Za-z0-9]+", "", short))
    joined = "+".join(parts)
    if len(joined) > 60:
        joined = f"{len(selected)}fams-{hashlib.sha1(joined.encode()).hexdigest()[:6]}"
    return f"repeats_{model}_{tag}_combined_{joined}_succ.png"


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
    n=None,
):
    """Mean line + shaded ±1 std band on `ax`.

    The band starts at `band_from_round` (default 1): round 0 is the shared
    base training, identical across every rep/lineage, so it has no spread to
    show — the polygon simply starts at round 1 rather than pinching to a point.

    Rounds with fewer than 2 reps (`n`, when given) are excluded too: their std
    is 0 by construction, so including them would taper the polygon down to the
    mean line and read as "the spread collapsed" rather than "only one rep got
    this far". The band is drawn per contiguous run of eligible rounds, so it
    ends abruptly at the last round that actually has a spread.
    """
    rr = np.asarray(rounds)
    m = rr >= band_from_round
    if n is not None:
        m &= np.asarray(n) >= 2
    # Contiguous runs of eligible rounds → one polygon each (no bridging over
    # an n=1 gap).
    start = None
    for i in range(len(rr) + 1):
        eligible = i < len(rr) and m[i]
        if eligible and start is None:
            start = i
        elif not eligible and start is not None:
            sl = slice(start, i)
            ax.fill_between(
                rr[sl],
                (mean - std)[sl],
                (mean + std)[sl],
                color=band_color or color,
                alpha=band_alpha,
                linewidth=0,
            )
            start = None
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
    ap.add_argument(
        "--combine",
        default="",
        help="Comma-separated families to overlay on the combined figure (default: all). "
        "Matched against the short label shown in the legend (e.g. '03dag,03dag_b050,"
        "03dag_rr_b050anc8'); exact match wins, then glob ('*b050*'), then substring. "
        "Per-family figures are still written for every family. A subset is saved under "
        "its own filename so it never overwrites the full overlay.",
    )
    ap.add_argument(
        "--combine_interactive",
        action="store_true",
        help="Pick the combined-figure families from a numbered prompt; prints the "
        "equivalent --combine string for reuse. Ignored when --combine is given.",
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
        bft_reps, bft_unfinished = collect_family_base_finetune(by_rep, model, prefer_reeval)
        for _name, _done, _target in bft_unfinished:
            _p = f"step {_done or 0}/{_target}" if _target else "step target unreadable"
            print(f"    [base-finetune SKIPPED — not a full run, {_p}] {_name}")
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
        draw_band(ax, rounds, mean, std, color="tab:blue", band_color="0.5", label="mean ± 1 std", n=n)
        draw_base_finetune(ax, bft_reps, max(rounds), color="tab:orange", label="base-finetune")
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
    # A family whose only round is 0 (the shared base, identical for everyone)
    # has no curve of its own to show — drawing it would add a legend entry for
    # an invisible line, so it is not a candidate for the overlay at all.
    def _draws_nothing(f: str) -> bool:
        rounds, _mean, _std, _n, _reps, bft = family_stats[f]
        if max(rounds) > 0:
            return False
        # Only round 0 — a bft star still counts, but draw_base_finetune clips
        # it to the family's own x-range, so it has to fall at round 0 too.
        return not any(r <= max(rounds) for pr in bft.values() for r in pr)

    plottable = {f: st for f, st in family_stats.items() if not _draws_nothing(f)}
    if len(plottable) < len(family_stats):
        print(
            "[combined] no rounds beyond the shared base, omitted: "
            + ", ".join(family_label(f, args.rep_tag) for f in sorted(set(family_stats) - set(plottable)))
        )
    if not plottable:
        print("[combined] nothing to overlay; skipping the combined figure.")
        print("\nfigures written:")
        for _name, path in written:
            print(f"    {path}")
        return 0

    # Which of those to overlay: --combine patterns, an interactive pick, or
    # (default) all of them. Colors are assigned from the FULL family set so a
    # family keeps the same color whether or not its siblings are plotted.
    all_families = sorted(plottable)
    if args.combine:
        try:
            selected = select_families(
                plottable, args.rep_tag, [p for p in args.combine.split(",") if p.strip()]
            )
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
    elif args.combine_interactive:
        try:
            selected = prompt_for_families(plottable, args.rep_tag)
        except (ValueError, EOFError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
    else:
        selected = all_families

    fig, ax = plt.subplots(figsize=(10, 6))
    styles_by_family = assign_family_colors(family_stats.keys(), args.rep_tag)
    for family in selected:
        rounds, mean, std, _n, _reps, bft_reps = family_stats[family]
        color, linestyle = styles_by_family[family]
        # Label: the part of the family from the tag onward (short + unique).
        label = family_label(family, args.rep_tag)
        draw_band(ax, rounds, mean, std, color=color, band_alpha=0.15, label=label, linestyle=linestyle, n=_n)
        draw_base_finetune(ax, bft_reps, max(rounds), color=color, label=f"{label} bft")
    ax.set_xlabel("DAgger round")
    ax.set_ylabel("success rate (%)")
    ax.set_ylim(0, 100)
    _all_rounds = [r for f in selected for r in family_stats[f][0]]
    ax.set_xlim(min(_all_rounds) - 0.35, max(_all_rounds) + 0.45)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=0.3)
    ax.set_title(f"success rate across repetitions (mean ± 1 std) — tag '{args.rep_tag}'")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out = out_dir / combined_out_name(model, args.rep_tag, selected, all_families)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    written.append(("combined", out))
    _of = "" if len(selected) == len(all_families) else f" of {len(all_families)}"
    print(f"[combined] {len(selected)}{_of} families → {out}")

    # ── figure paths again, together (the per-family ones scroll off above) ──
    print("\nfigures written:")
    for _name, path in written:
        print(f"    {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
