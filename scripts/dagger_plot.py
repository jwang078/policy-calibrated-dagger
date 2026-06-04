#!/usr/bin/env python3
"""Plot DAgger progress across rounds as a single line chart with overloaded Y axis.

Each metric is min-max normalized across rounds (independently) so every line
spans the full [0, 1] range — `(y - min) / (max - min)`. The legend shows the
metric name, its "better" direction, the value at the latest round, and the
[min, max] normalization range so you can read absolute numbers back from
the chart.

A "lineage" is the training-dir name part between ${MODEL_PREFIX}_ and the
trailing [_ft]_dag${N} suffix. Finetune (`..._ft_dag{N}`) and scratch
(`..._dag{N}`) rounds from the same lineage are folded together.

Usage:
    python my_scripts/dagger_plot.py [--base_short=STR] [--action=abs] [--out_dir=DIR]

When --base_short is omitted, every lineage discovered under TRAINING_ROOT is
plotted, with one PNG per lineage named dagger_progress_<model>_<lineage>.png.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Canonical DAgger naming. ROUND_SUFFIX_RE and lineage_of used to live here;
# they're now shared with the orchestrator + viz scripts via dagger_naming.py
# so forward / inverse mappings can't drift.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dagger_naming import ROUND_SUFFIX_RE, lineage_of  # noqa: E402

TRAINING_ROOT = Path.home() / "code" / "lerobot" / "outputs" / "training"
DEFAULT_OUT_DIR = Path.home() / "code" / "lerobot" / "outputs" / "dagger"


def parse_eval_dict(line: str) -> dict | None:
    """Extract the trailing python-dict literal from an `INFO ... Suite overall aggregated: {...}` line."""
    m = re.search(r"(\{.*\})", line)
    if not m:
        return None
    import ast

    try:
        return ast.literal_eval(m.group(1))
    except Exception:
        return None


def scan_round(dir_path: Path, round_n: int | None = None) -> dict | None:
    """Pull (round, succ, pos_err, in_coll, trunc, eval_step, variant) from a training dir's wandb log.

    `round_n` overrides the auto-detected round number from the dir name suffix.
    Used for round 0 (base policy dir has no _dag${N} suffix to parse).
    """
    retrain_suffix: str | None = None
    if round_n is None:
        m = ROUND_SUFFIX_RE.search(dir_path.name)
        if not m:
            return None
        round_n = int(m.group(1))
        retrain_suffix = m.group(2)  # None unless --retrain_round was used
    # Variant — three classes, off the main DAgger curve:
    #   "ft":      canonical finetune round (or round 0 base). Main curve.
    #   "scratch": post-loop final-scratch reference (shares round number
    #              with ft round N).
    #   "retrain": --retrain_round run (shares round number with the
    #              canonical ft round, but uses different hyperparameters).
    if round_n == 0:
        variant = "ft"
    elif retrain_suffix is not None:
        variant = "retrain"
    else:
        variant = "ft" if "_ft_dag" in dir_path.name else "scratch"
    logs = sorted(dir_path.glob("wandb/run-*/files/output.log"), key=lambda p: p.stat().st_mtime)
    if not logs:
        return {"round": round_n, "variant": variant, "retrain_suffix": retrain_suffix}
    log = logs[-1].read_text(errors="ignore")

    row: dict = {"round": round_n, "variant": variant, "retrain_suffix": retrain_suffix}
    # Last eval block.
    eval_lines = [ln for ln in log.splitlines() if "Suite overall aggregated" in ln]
    if eval_lines:
        d = parse_eval_dict(eval_lines[-1])
        if d:
            row["succ"] = d.get("pc_success")
            row["pos_err"] = d.get("avg_final_position_error_m")
            row["ori_err"] = d.get("avg_final_orientation_error_deg")
            row["in_coll"] = d.get("avg_in_collision")
            row["trunc"] = d.get("avg_truncated")
    # Eval step (most recent).
    step_matches = re.findall(r"Eval policy at step (\d+)", log)
    if step_matches:
        row["eval_step"] = int(step_matches[-1])
    # Final training loss. Two sources, tried in order:
    #   (1) wandb-summary.json — wandb writes the final value of every
    #       logged scalar at run end. `train/loss` is the canonical key
    #       (matches the wandb dashboard chart). Per-run, never overwritten
    #       on the local disk by later trainings using the same run_id.
    #   (2) Tail of `loss:<value>` lines in output.log — for runs that
    #       don't have wandb-summary.json (older SDK or crashed before flush).
    #       Less accurate (depends on log_freq + averages over the last 20
    #       logged steps), used as a fallback.
    # Search ALL wandb subdirs (latest first) so a hand-written eval-only
    # wandb dir (used to surface a corrected benchmark eval) doesn't
    # shadow the real training run's summary.
    import json as _json

    loss_val: float | None = None
    summary_files = sorted(
        dir_path.glob("wandb/run-*/files/wandb-summary.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for sf in summary_files:
        try:
            data = _json.loads(sf.read_text())
        except Exception:  # noqa: BLE001  # malformed/partial summary → skip
            continue  # nosec B112
        v = data.get("train/loss")
        if v is None:
            v = data.get("loss")
        if v is not None:
            loss_val = float(v)
            break
    if loss_val is None:
        loss_pattern = re.compile(r"step:\d+\S*\s+smpl:\S+\s+ep:\S+\s+epch:\S+\s+loss:([0-9.]+)")
        for candidate in reversed(logs):
            matches = loss_pattern.findall(candidate.read_text(errors="ignore"))
            if matches:
                tail = [float(x) for x in matches[-20:]]
                loss_val = sum(tail) / len(tail)
                break
    if loss_val is not None:
        row["loss"] = loss_val
    return row


def discover_lineages(model: str) -> list[str]:
    """Find every distinct lineage under TRAINING_ROOT.

    A lineage is identified by having at least one _dag${N} dir. The base
    policy dir alone (no dag rounds) doesn't count — it'd surface every
    standalone training dir under TRAINING_ROOT. The round-0 point inside
    each lineage's plot only appears when that lineage also has dag rounds.
    """
    lineages = set()
    for d in TRAINING_ROOT.glob(f"{model}_*_dag*"):
        if not d.is_dir():
            continue
        lin = lineage_of(d.name, model)
        if lin:
            lineages.add(lin)
    return sorted(lineages)


def read_lineage_rerun_metadata(lineage: str, model: str) -> dict | None:
    """Read the rerun-source pointer for a lineage from its earliest round's dagger/config.json sidecar.

    Returns a dict with keys `source_lineage` and (optionally) `source_run_tag` /
    `source_blends_tag` if this lineage was produced by --rerun_blends_from.
    Returns None if the lineage isn't a rerun or has no sidecar.

    Why earliest round: the orchestrator re-writes the sidecar on every
    invocation, but all rounds within a single lineage share the same
    rerun_mode (set once, at orchestrator startup). Reading from round 1 (the
    first dag round) is enough.
    """
    # Walk dag rounds in ascending order; pick the first one that has a sidecar.
    dirs = [
        d
        for d in TRAINING_ROOT.glob(f"{model}_{lineage}*_dag*")
        if d.is_dir() and lineage_of(d.name, model) == lineage
    ]
    if not dirs:
        return None
    dirs.sort(key=lambda p: int(ROUND_SUFFIX_RE.search(p.name).group(1)))  # type: ignore[union-attr]
    for d in dirs:
        sidecar = d / "dagger" / "config.json"
        if not sidecar.is_file():
            continue
        try:
            cfg = json.loads(sidecar.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        rerun_mode = cfg.get("rerun_mode")
        if not rerun_mode:
            return None
        # source_policy_basename is `<model_prefix>_<source_lineage>` — strip
        # the model prefix to recover the source's lineage key.
        src_policy = rerun_mode.get("source_policy_basename", "")
        prefix = f"{model}_"
        if not src_policy.startswith(prefix):
            return None
        return {
            "source_lineage": src_policy[len(prefix) :],
            "source_run_tag": rerun_mode.get("source_run_tag", ""),
            "source_blends_tag": rerun_mode.get("source_blends_tag", ""),
        }
    return None


def group_reruns_by_source(lineages: list[str], model: str) -> dict[str, list[str]]:
    """Return {source_lineage: [rerun_lineage, ...]} for every source lineage that has
    at least one rerun on disk. Source lineages with no reruns are not in the dict.
    """
    grouped: dict[str, list[str]] = {}
    for lin in lineages:
        meta = read_lineage_rerun_metadata(lin, model)
        if meta is None:
            continue
        src = meta["source_lineage"]
        grouped.setdefault(src, []).append(lin)
    return grouped


def collect_lineage_rows(lineage: str, model: str) -> list[dict]:
    """Return per-round dicts (succ, pos_err, ori_err, in_coll, trunc, variant) for a lineage.

    Same logic as plot_lineage's data-collection prelude, factored out so the
    per-metric overlay code can reuse it without duplicating the round-0
    discovery / variant-splitting.
    """
    dirs = [
        d
        for d in TRAINING_ROOT.glob(f"{model}_{lineage}*_dag*")
        if d.is_dir() and lineage_of(d.name, model) == lineage
    ]
    dirs.sort(key=lambda p: int(ROUND_SUFFIX_RE.search(p.name).group(1)))  # type: ignore[union-attr]
    rows = [r for r in (scan_round(d) for d in dirs) if r is not None]
    if rows:
        base_dir = TRAINING_ROOT / f"{model}_{lineage}"
        if not base_dir.is_dir() and "_basewrist_" in lineage:
            untagged = lineage.rsplit("_basewrist_", 1)[0] + "_basewrist"
            base_dir = TRAINING_ROOT / f"{model}_{untagged}"
        if base_dir.is_dir():
            row0 = scan_round(base_dir, round_n=0)
            if row0 is not None:
                rows = [row0] + rows
    return rows


# Single-metric comparison plot definitions. Each tuple is
# (metric_key, axis_label, direction-of-improvement).
COMPARISON_METRICS = [
    ("succ", "success rate (%)", "↑ better"),
    ("pos_err", "final position error (m)", "↓ better"),
    ("ori_err", "final orientation error (deg)", "↓ better"),
    ("in_coll", "in-collision steps", "↓ better"),
    ("trunc", "truncation rate", "↓ better"),
    # Final training loss (avg of last 20 logged steps). Direction is
    # "↓ better" for the comparison framing (lower training loss = better
    # fit to the round's training set), but note that "better fit" doesn't
    # automatically imply "better eval" — read alongside the eval metrics
    # above. Lineages trained on harder/bigger datasets (e.g. multi-blend
    # mixes) will naturally have higher absolute loss.
    ("loss", "final training loss", "↓ better"),
]


def _sort_lineages_for_display(rerun_lineages: list[str]) -> list[str]:
    """Stable sort by (string-length, name). Groups single-blend reruns
    (`rerun_v1_b010`, length 13) BEFORE two-blend reruns (`rerun_v1_b090_050`,
    length 17), and within each length-group sorts alphanumerically (which,
    for the blend-tag naming convention, corresponds to ascending ratio).

    Used to order the legend (line plots) and the x-axis (bar charts), and
    also drives the name-based rainbow color assignment so a given lineage's
    color is consistent across all metric plots.
    """
    return sorted(rerun_lineages, key=lambda name: (len(name), name))


def _name_rainbow_colors(rerun_lineages: list[str]) -> dict[str, tuple[float, float, float, float]]:
    """Assign each rerun a stable rainbow color based on its position in the
    sorted-for-display list (see _sort_lineages_for_display). The color
    identifies the LINEAGE, not its performance — so the same rerun has the
    same color across every metric plot.
    """
    sorted_names = _sort_lineages_for_display(rerun_lineages)
    n = len(sorted_names)
    rainbow = matplotlib.colormaps.get_cmap("rainbow")
    if n == 1:
        return {sorted_names[0]: rainbow(0.5)}
    return {name: rainbow(i / (n - 1)) for i, name in enumerate(sorted_names)}


def _per_rerun_avg_delta(
    source_lineage: str,
    rerun_lineages: list[str],
    model: str,
    metric: str,
    direction: str,
) -> tuple[dict[str, float | None], dict[str, float | None], dict[str, int]]:
    """Per-rerun mean, std, and N of (rerun-beats-source) delta across canonical
    ft rounds. Returns (mean_by_lineage, std_by_lineage, n_by_lineage).

    For ↓-better metrics, delta is sign-flipped (source − rerun) so higher =
    rerun wins. Reruns with no overlapping rounds get None for mean/std and 0
    for n. With only one overlapping round, std=0 (no spread to measure).

    Std is the sample standard deviation (Bessel-corrected, dividing by N-1)
    of the per-round deltas — descriptive of round-to-round consistency,
    not an inferential CI. See the title note on the bar chart.
    """
    import statistics

    is_lower_better = direction.startswith("↓")
    source_rows = collect_lineage_rows(source_lineage, model)
    src_by_round: dict[int, float] = {}
    for r in source_rows:
        if r.get("variant", "ft") != "ft" or r["round"] <= 0:
            continue
        v = r.get(metric)
        if v is not None:
            src_by_round[r["round"]] = v
    mean_by: dict[str, float | None] = {}
    std_by: dict[str, float | None] = {}
    n_by: dict[str, int] = {}
    for rerun in rerun_lineages:
        ds: list[float] = []
        for r in collect_lineage_rows(rerun, model):
            if r.get("variant", "ft") != "ft" or r["round"] <= 0:
                continue
            v = r.get(metric)
            if v is None:
                continue
            src_v = src_by_round.get(r["round"])
            if src_v is None:
                continue
            ds.append((src_v - v) if is_lower_better else (v - src_v))
        if ds:
            mean_by[rerun] = sum(ds) / len(ds)
            std_by[rerun] = statistics.stdev(ds) if len(ds) >= 2 else 0.0
            n_by[rerun] = len(ds)
        else:
            mean_by[rerun] = None
            std_by[rerun] = None
            n_by[rerun] = 0
    return mean_by, std_by, n_by


def plot_comparison_metric(
    source_lineage: str,
    rerun_lineages: list[str],
    model: str,
    metric: str,
    axis_label: str,
    direction: str,
    out_path: Path,
) -> int:
    """Plot ONE metric for source + all reruns on a single axis. Returns total lines drawn."""
    series: list[tuple[str, list[dict], dict]] = []
    # (lineage_key, rows, style_kwargs). Source first (drawn solid, on top).
    series.append(
        (
            source_lineage,
            collect_lineage_rows(source_lineage, model),
            {"linestyle": "-", "linewidth": 2.5, "marker": "o", "color": "black", "zorder": 3},
        )
    )
    # Name-based rainbow: each rerun has a stable color across ALL metric
    # plots (alphabetical-position → rainbow). Bar height tells you how good
    # the rerun is on a given metric; color tells you WHICH rerun it is.
    # Source stays black as the reference line.
    color_by_lineage = _name_rainbow_colors(rerun_lineages)
    rerun_markers = ["s", "^", "v", "D", "P", "X", "<", ">", "p", "h"]
    for i, rerun in enumerate(rerun_lineages):
        series.append(
            (
                rerun,
                collect_lineage_rows(rerun, model),
                {
                    "linestyle": "--",
                    "linewidth": 1.8,
                    "marker": rerun_markers[i % len(rerun_markers)],
                    "color": color_by_lineage[rerun],
                    "zorder": 2,
                },
            )
        )

    fig, ax = plt.subplots(figsize=(13, 6))
    drew_anything = 0
    for lineage_key, rows, style in series:
        ft_rows = [r for r in rows if r.get("variant", "ft") == "ft" and r.get(metric) is not None]
        scratch_rows = [r for r in rows if r.get("variant") == "scratch" and r.get(metric) is not None]
        if not ft_rows and not scratch_rows:
            continue
        xs = [r["round"] for r in ft_rows]
        ys = [r[metric] for r in ft_rows]
        if xs:
            latest_str = f"   latest={ys[-1]:.3g}" if ys else ""
            ax.plot(
                xs,
                ys,
                label=f"{lineage_key}{latest_str}",
                **style,
            )
            for x, y in zip(xs, ys, strict=True):
                ax.annotate(
                    f"{y:.3g}",
                    xy=(x, y),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                    color=style["color"],
                )
            drew_anything += len(xs)
        # Plot scratch rows as stars off the main curve.
        for r in scratch_rows:
            ax.scatter(
                [r["round"]],
                [r[metric]],
                marker="*",
                s=180,
                color=style["color"],
                edgecolors="black",
                linewidths=0.6,
                zorder=4,
            )
            ax.annotate(
                f"{r[metric]:.3g}\n(scratch)",
                xy=(r["round"], r[metric]),
                xytext=(6, -14),
                textcoords="offset points",
                ha="left",
                fontsize=7,
                color=style["color"],
                fontweight="bold",
            )
            drew_anything += 1

    if drew_anything == 0:
        plt.close(fig)
        return 0

    ax.set_xlabel("DAgger round  (★ = post-loop from-scratch ref)")
    ax.set_ylabel(axis_label)
    ax.set_title(
        f"DAgger comparison ({direction}): source vs reruns\nsource = {model}_{source_lineage}",
        fontsize=10,
    )
    # Integer ticks across the round range used by any series.
    all_xs = sorted({r["round"] for _, rows, _ in series for r in rows if r.get(metric) is not None})
    if all_xs:
        ax.set_xticks(all_xs)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return drew_anything


def plot_comparison_metric_avg_delta_bar(
    source_lineage: str,
    rerun_lineages: list[str],
    model: str,
    metric: str,
    axis_label: str,
    direction: str,
    out_path: Path,
    error_bar_type: str = "sem",
) -> int:
    """Bar chart: per-rerun "winning delta" averaged across canonical finetune
    DAgger rounds. Positive bars ALWAYS mean rerun beats source on average,
    regardless of metric direction:
      - For ↑-better metrics (succ):     delta = rerun − source
      - For ↓-better metrics (pos_err…): delta = source − rerun  (sign-flipped)
    Bars sorted high → low and colored by the same rank-based rainbow used in
    the companion line plot — best (highest bar) = red, worst = purple.

    Returns number of bars drawn (0 if no rerun had any common rounds with src).
    """
    color_by_lineage = _name_rainbow_colors(rerun_lineages)
    delta_by_lineage, std_by_lineage, n_rounds_by_lineage = _per_rerun_avg_delta(
        source_lineage, rerun_lineages, model, metric, direction
    )

    # Drop reruns with no overlap; alphabetical x-axis order matches the line
    # plot's legend so the same color = same name across all 5 metric plots.
    # Bar HEIGHT shows the rerun's improvement over source; bar COLOR is the
    # lineage's stable identity tag.
    valid_lineages = [r for r in rerun_lineages if delta_by_lineage[r] is not None]
    if not valid_lineages:
        return 0

    fig, ax = plt.subplots(figsize=(11, 6))
    xs = list(range(len(valid_lineages)))
    bar_means: list[float] = [float(delta_by_lineage[r] or 0.0) for r in valid_lineages]
    bar_stds: list[float] = [float(std_by_lineage[r] or 0.0) for r in valid_lineages]
    bar_colors = [color_by_lineage[r] for r in valid_lineages]
    bar_labels = [
        # Short tail after `_basewrist_` (e.g. `rerun_v1_b050`) so x-tick labels fit.
        r.split("_basewrist_", 1)[-1] if "_basewrist_" in r else r
        for r in valid_lineages
    ]
    bar_n_rounds = [n_rounds_by_lineage[r] for r in valid_lineages]

    # Derive the actual error-bar values from the requested type. SEM = SD/√N
    # (uncertainty in the mean estimate). SD = sample std (round-to-round
    # spread). For N=1 both reduce to 0. Title and annotation use the
    # corresponding symbol so the reader knows which they're seeing.
    import math

    if error_bar_type == "sd":
        bar_errs = list(bar_stds)
        err_legend = "±1 std of per-round deltas (round-to-round spread)"
    elif error_bar_type == "sem":
        bar_errs = [
            (sd / math.sqrt(n)) if (sd is not None and n >= 1) else 0.0
            for sd, n in zip(bar_stds, bar_n_rounds, strict=True)
        ]
        err_legend = "±1 SEM (sd / √n; uncertainty in the mean estimate)"
    else:
        raise ValueError(f"unknown error_bar_type={error_bar_type!r}; expected 'sd' or 'sem'")

    # Error bars in dark gray with caps so they don't visually overpower the
    # rainbow bar colors.
    bars = ax.bar(
        xs,
        bar_means,
        yerr=bar_errs,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.7,
        capsize=5,
        error_kw={"ecolor": "#333333", "elinewidth": 1.2, "capthick": 1.2, "zorder": 3},
    )
    ax.axhline(0, color="black", linewidth=1)

    # Annotation offset sized to the bar range INCLUDING error bars so the
    # mean±err text never collides with an error-bar cap.
    extents = (
        [m + e for m, e in zip(bar_means, bar_errs, strict=True)]
        + [m - e for m, e in zip(bar_means, bar_errs, strict=True)]
        + [0.0]
    )
    y_range = max(extents) - min(extents)
    offset = max(abs(y_range) * 0.02, 1e-9)
    for bar, val, err, n_r in zip(bars, bar_means, bar_errs, bar_n_rounds, strict=True):
        h = bar.get_height()
        cap_y = h + err if h >= 0 else h - err  # place text above (below) error-bar cap
        ax.annotate(
            f"{val:+.3g} ± {err:.3g}\n(n={n_r})",
            xy=(bar.get_x() + bar.get_width() / 2, cap_y),
            xytext=(0, 6 if h >= 0 else -22),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )

    # Vertical dotted separators between length-groups. Bars are pre-sorted by
    # (len(name), name), so groups of single-blend reruns (`rerun_v1_b010`,
    # len 13), two-blend reruns (`rerun_v1_b030_010`, len 17), etc. are
    # already contiguous — just find every index where the next label's
    # length changes and draw a separator at x = i + 0.5.
    for i in range(len(bar_labels) - 1):
        if len(bar_labels[i]) != len(bar_labels[i + 1]):
            ax.axvline(i + 0.5, color="gray", linestyle=":", linewidth=1.2, alpha=0.7, zorder=1)

    ax.set_xticks(xs)
    ax.set_xticklabels(bar_labels, rotation=15, ha="right", fontsize=9)
    # Both y-axis and title are stated from the "rerun's advantage" point of
    # view so positive values always mean "rerun beats source", regardless of
    # whether the underlying metric is ↑-better or ↓-better.
    # The signed-delta computation lives in `_per_rerun_avg_delta`:
    #   ↑-better metric (e.g. succ):  delta = rerun − source  (positive when rerun's value is higher)
    #   ↓-better metric (e.g. loss):  delta = source − rerun  (positive when rerun's value is lower)
    # Either way, a positive bar = rerun beats source. Label the formula to
    # match the actual computation so the plot reader can reproduce the math.
    formula_note = "rerun − source" if direction.startswith("↑") else "source − rerun"
    ax.set_ylabel(f"rerun improvement over source: mean Δ {axis_label}\n(positive = rerun better)")
    ax.set_title(
        "DAgger: per-rerun avg improvement over source (↑ taller bar = bigger improvement)\n"
        f"averaged across canonical finetune rounds (excl. dag0 + scratch variants); "
        f"plotted formula = {formula_note}; error bars = {err_legend}\n"
        f"source = {model}_{source_lineage}    [bar color = lineage identity (same color = same rerun across all metric plots)]",
        fontsize=10,
    )
    ax.grid(True, alpha=0.3, axis="y")
    y_lo, y_hi = ax.get_ylim()
    ax.set_ylim(y_lo - offset * 4, y_hi + offset * 4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return len(valid_lineages)


def plot_lineage(lineage: str, model: str, out_path: Path) -> int:
    """Plot one lineage; returns number of rounds plotted (0 if nothing)."""
    dirs = [
        d
        for d in TRAINING_ROOT.glob(f"{model}_{lineage}*_dag*")
        if d.is_dir() and lineage_of(d.name, model) == lineage
    ]
    dirs.sort(key=lambda p: int(ROUND_SUFFIX_RE.search(p.name).group(1)))  # type: ignore[union-attr]
    rows = [r for r in (scan_round(d) for d in dirs) if r is not None]
    # Prepend round 0 (base policy dir, no _dag suffix) only if at least one
    # dag round exists — otherwise a standalone base training would plot as
    # a single round-0 point.
    #
    # Lineage may include a run tag (e.g. `..._basewrist_d30`) that's only
    # present on the dag artifacts — the actual base policy dir is untagged
    # (`..._basewrist`). Try the tagged path first, then strip the tag.
    if rows:
        base_dir = TRAINING_ROOT / f"{model}_{lineage}"
        if not base_dir.is_dir() and "_basewrist_" in lineage:
            untagged = lineage.rsplit("_basewrist_", 1)[0] + "_basewrist"
            base_dir = TRAINING_ROOT / f"{model}_{untagged}"
        if base_dir.is_dir():
            row0 = scan_round(base_dir, round_n=0)
            if row0 is not None:
                rows = [row0] + rows
    if not rows:
        return 0

    # Split rows into the finetune progression (the main curve) and the
    # off-curve variants (post-loop scratch reference + retrain runs). All
    # three classes can share a round number with a canonical ft round;
    # plotting them all on the same line would overlap or jitter
    # unpredictably. Showing them separately makes the visual question
    # clearer for each class:
    #   "did training the final model from scratch on the merged data beat
    #    the last finetune round?" (scratch)
    #   "did retuning round N's hyperparameters improve over the original
    #    dag${N}?" (retrain)
    ft_rows = [r for r in rows if r.get("variant", "ft") == "ft"]
    scratch_rows = [r for r in rows if r.get("variant") == "scratch"]
    retrain_rows = [r for r in rows if r.get("variant") == "retrain"]
    if not ft_rows:
        # Edge case: ONLY scratch rows. Treat them as the main curve.
        ft_rows = scratch_rows
        scratch_rows = []

    rounds = [r["round"] for r in ft_rows]
    # Metric definitions per panel: (key, label, color, direction-of-improvement).
    # Split so the panel showing "higher = better" doesn't have its trend visually
    # contradicted by the panels showing "lower = better".
    panels = [
        (
            "Success rate  (higher is better)",
            [
                ("succ", "success rate (%)", "tab:green", "↑"),
            ],
        ),
        (
            "Failure-mode metrics  (lower is better)",
            [
                ("pos_err", "final position error (m)", "tab:red", "↓"),
                ("ori_err", "final orientation error (deg)", "tab:blue", "↓"),
                ("in_coll", "in-collision steps", "tab:orange", "↓"),
                ("trunc", "truncation rate", "tab:purple", "↓"),
            ],
        ),
        (
            "Final training loss  (lower is better)",
            [
                ("loss", "final training loss", "tab:brown", "↓"),
            ],
        ),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    fig.suptitle(f"DAgger progress: {model}_{lineage}[_ft]_dag*", fontsize=11)
    for ax, (panel_title, metrics) in zip(axes, panels, strict=True):
        for key, label, color, direction in metrics:
            ys_raw = [r.get(key) for r in ft_rows]
            if not any(y is not None for y in ys_raw):
                continue
            ys_clean = [(x, y) for x, y in zip(rounds, ys_raw, strict=True) if y is not None]
            xs = [x for x, _ in ys_clean]
            ys = [y for _, y in ys_clean]
            # Also pull the off-curve variant references for the same metric,
            # so the normalization includes them (otherwise the markers can
            # fall outside the plot area).
            scratch_xs_ys = [(r["round"], r.get(key)) for r in scratch_rows if r.get(key) is not None]
            retrain_xs_ys = [
                (r["round"], r.get(key), r.get("retrain_suffix") or "")
                for r in retrain_rows
                if r.get(key) is not None
            ]
            ys_all = ys + [y for _, y in scratch_xs_ys] + [y for _, y, _ in retrain_xs_ys]
            lo, hi = min(ys_all), max(ys_all)
            if hi > lo:
                normed = [(y - lo) / (hi - lo) for y in ys]
                scratch_normed = [(x, (y - lo) / (hi - lo), y) for x, y in scratch_xs_ys]
                retrain_normed = [(x, (y - lo) / (hi - lo), y, s) for x, y, s in retrain_xs_ys]
            else:
                normed = [0.5 for _ in ys]
                scratch_normed = [(x, 0.5, y) for x, y in scratch_xs_ys]
                retrain_normed = [(x, 0.5, y, s) for x, y, s in retrain_xs_ys]
            latest = ys[-1]
            ax.plot(
                xs,
                normed,
                marker="o",
                linewidth=2,
                markersize=7,
                color=color,
                label=f"{label}   {direction} better   latest={latest:.3g}   (norm range [{lo:.3g}, {hi:.3g}])",
            )
            for x, y_raw, y_norm in zip(xs, ys, normed, strict=True):
                ax.annotate(
                    f"{y_raw:.3g}",
                    xy=(x, y_norm),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    color=color,
                )
            # Plot scratch reference points as stars (distinct marker), with
            # the raw value annotated. No connecting line — they're not part
            # of the round-over-round progression.
            for x, y_norm, y_raw in scratch_normed:
                ax.scatter(
                    [x],
                    [y_norm],
                    marker="*",
                    s=200,
                    color=color,
                    edgecolors="black",
                    linewidths=0.8,
                    zorder=5,
                )
                ax.annotate(
                    f"{y_raw:.3g}\n(scratch)",
                    xy=(x, y_norm),
                    xytext=(8, -18),
                    textcoords="offset points",
                    ha="left",
                    fontsize=8,
                    color=color,
                    fontweight="bold",
                )
            # Plot retrain reference points as diamonds, annotated with the
            # retrain suffix so multiple retrain variants of the same round
            # remain distinguishable.
            for x, y_norm, y_raw, suffix in retrain_normed:
                ax.scatter(
                    [x],
                    [y_norm],
                    marker="D",
                    s=80,
                    color=color,
                    edgecolors="black",
                    linewidths=0.8,
                    zorder=5,
                )
                ax.annotate(
                    f"{y_raw:.3g}\n({suffix})" if suffix else f"{y_raw:.3g}\n(retrain)",
                    xy=(x, y_norm),
                    xytext=(8, 10),
                    textcoords="offset points",
                    ha="left",
                    fontsize=7,
                    color=color,
                    fontweight="bold",
                )
        ax.set_title(panel_title, fontsize=10)
        ax.set_ylabel("normalized\n(min-max per metric, across rounds)")
        ax.set_xticks(rounds)
        ax.set_ylim(-0.05, 1.15)
        ax.grid(True, alpha=0.3)
        # Park legend outside the axes so it never overlaps the data.
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9, framealpha=0.9)
    axes[-1].set_xlabel("DAgger round  (★ = post-loop from-scratch ref, ◆ = --retrain_round variant)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    # Print path on its own line so terminals can recognize it as ctrl-clickable.
    scratch_note = f", {len(scratch_rows)} scratch reference(s)" if scratch_rows else ""
    retrain_note = f", {len(retrain_rows)} retrain variant(s)" if retrain_rows else ""
    print(f"  {len(rounds)} rounds (dag{rounds[0]}..dag{rounds[-1]}){scratch_note}{retrain_note}:")
    print(f"  {out_path.resolve()}")
    return len(rounds) + len(scratch_rows) + len(retrain_rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--base_short",
        default=None,
        help="Lineage filter: restrict to ${base_short}_${action}_basewrist[_${run_tag}]. Omit to plot every lineage found.",
    )
    ap.add_argument(
        "--action",
        default="abs",
        help="Action format tag (abs|delta). Only used with --base_short. Default abs.",
    )
    ap.add_argument(
        "--run_tag",
        default=None,
        help="Optional run tag appended to the lineage (e.g. 'd30'). Only used with --base_short.",
    )
    ap.add_argument("--model", default="pi05", help="Policy prefix in dir name. Default pi05.")
    ap.add_argument(
        "--out_dir",
        default=str(DEFAULT_OUT_DIR),
        help=f"Directory for PNG files. Default: {DEFAULT_OUT_DIR}",
    )
    ap.add_argument(
        "--error_bar_type",
        default="sem",
        choices=["sd", "sem"],
        help=(
            "Error bar style on the avg-delta bar charts. "
            "'sem' (default) = sd/√n, shows uncertainty in the mean estimate "
            "(narrow bars at large n, useful for ranking decisions). "
            "'sd' = round-to-round spread (wider bars; useful for assessing rerun consistency)."
        ),
    )
    ap.add_argument(
        "--filter",
        default=None,
        nargs="+",
        help=(
            "One or more substring filters on lineage names. Each value is "
            "wrapped as `*FILTER*` and matched (case-sensitive) with OR "
            "semantics — a lineage shows up if ANY filter value appears in "
            "its name. Only matching lineages get per-lineage plots; "
            "comparison plots are skipped if either the source OR all its "
            "reruns are filtered out. NOTHING is regenerated for "
            "non-matching lineages. Example: `--filter grip0 g0` keeps any "
            "lineage containing either substring."
        ),
    )
    args = ap.parse_args()

    if args.base_short:
        lineages = [f"{args.base_short}_{args.action}_basewrist"]
    else:
        lineages = discover_lineages(args.model)
        if not lineages:
            print(f"ERROR: no DAgger training dirs found under {TRAINING_ROOT}", file=sys.stderr)
            sys.exit(1)

    if args.filter:
        # OR semantics: keep a lineage if ANY filter substring is in its name.
        before = len(lineages)
        lineages = [lin for lin in lineages if any(f in lin for f in args.filter)]
        if not lineages:
            quoted = " ".join(f"'{f}'" for f in args.filter)
            print(
                f"ERROR: --filter {quoted} matched none of the "
                f"{before} discovered lineage(s) for --model={args.model} (OR semantics).",
                file=sys.stderr,
            )
            sys.exit(1)

    out_dir = Path(args.out_dir)
    n_plotted = 0
    for lin in lineages:
        # Path matches dagger_progress.sh's `plot_path` so the table's "Plot:"
        # line and the file dagger_plot.py writes resolve to the same PNG.
        out_path = out_dir / f"dagger_progress_{args.model}_{lin}.png"
        if plot_lineage(lin, args.model, out_path) > 0:
            n_plotted += 1

    if n_plotted == 0:
        print("ERROR: no rows extracted for any lineage", file=sys.stderr)
        sys.exit(1)

    # Per-metric overlay comparison plots: source vs all its rerun-blends
    # lineages on the same axis, one PNG per metric. Source lineages with no
    # reruns on disk are skipped silently (nothing to compare).
    grouped = group_reruns_by_source(lineages, args.model)
    # When --filter is set, also skip groups whose SOURCE lineage doesn't
    # match the filter. group_reruns_by_source's key is the source name,
    # which may have been excluded by the lineage filter above — without
    # this gate we'd regenerate a comparison plot using a source we already
    # said we don't care about.
    if args.filter and grouped:
        grouped = {src: reruns for src, reruns in grouped.items() if any(f in src for f in args.filter)}
    if grouped:
        print()
        print(
            f"Detected {sum(len(rs) for rs in grouped.values())} rerun lineage(s) across {len(grouped)} source(s); generating overlay comparison plots:"
        )
        for source_lin, rerun_lineages_list in sorted(grouped.items()):
            # Model-prefixed dirname mirrors the per-lineage plot naming so
            # pi05 and diffusion lineages with the same suffix don't share a
            # comparison dir.
            comp_dir = out_dir / f"comparison_{args.model}_{source_lin}"
            # Sort by (length, name) so single-blend reruns (b010, b030, …)
            # appear before multi-blend reruns (b090_050, b090_070, …) and
            # within each length-group the lineages sort alphanumerically.
            sorted_reruns = _sort_lineages_for_display(rerun_lineages_list)
            for metric_key, axis_label, direction in COMPARISON_METRICS:
                out_path = comp_dir / f"{metric_key}.png"
                lines = plot_comparison_metric(
                    source_lin,
                    sorted_reruns,
                    args.model,
                    metric_key,
                    axis_label,
                    direction,
                    out_path,
                )
                if lines > 0:
                    print(f"  {out_path.resolve()}")
                # Companion bar chart: per-rerun mean (rerun − source) across
                # canonical ft rounds. Same metric, same colors.
                bar_path = comp_dir / f"{metric_key}_avg_delta_bar.png"
                bars = plot_comparison_metric_avg_delta_bar(
                    source_lin,
                    sorted_reruns,
                    args.model,
                    metric_key,
                    axis_label,
                    direction,
                    bar_path,
                    error_bar_type=args.error_bar_type,
                )
                if bars > 0:
                    print(f"  {bar_path.resolve()}")
            print(f"  source={source_lin}   reruns={', '.join(sorted_reruns)}")


if __name__ == "__main__":
    main()
