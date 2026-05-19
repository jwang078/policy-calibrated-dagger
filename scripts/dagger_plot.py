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
plotted, with one PNG per lineage named dagger_progress_<lineage>.png.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRAINING_ROOT = Path.home() / "code" / "lerobot" / "outputs" / "training"
DEFAULT_OUT_DIR = Path.home() / "code" / "lerobot" / "outputs" / "dagger"

# Matches `_ft_dag5`, `_dag10`, or `_ft_dag5_${retrain_suffix}` at end of dirname.
# Group 1: round number. Group 2 (optional): the retrain suffix without its
# leading underscore.
ROUND_SUFFIX_RE = re.compile(r"(?:_ft)?_dag(\d+)(?:_([^/]+))?$")


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
    return row


def lineage_of(dir_name: str, model: str) -> str | None:
    """Map a training dir basename to its lineage key (or None if not a DAgger round dir)."""
    prefix = f"{model}_"
    if not dir_name.startswith(prefix):
        return None
    rest = dir_name[len(prefix) :]
    m = ROUND_SUFFIX_RE.search(rest)
    if not m:
        return None
    return rest[: m.start()]


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
    ]

    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
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
    args = ap.parse_args()

    if args.base_short:
        lineages = [f"{args.base_short}_{args.action}_basewrist"]
    else:
        lineages = discover_lineages(args.model)
        if not lineages:
            print(f"ERROR: no DAgger training dirs found under {TRAINING_ROOT}", file=sys.stderr)
            sys.exit(1)

    out_dir = Path(args.out_dir)
    n_plotted = 0
    for lin in lineages:
        out_path = out_dir / f"dagger_progress_{lin}.png"
        if plot_lineage(lin, args.model, out_path) > 0:
            n_plotted += 1

    if n_plotted == 0:
        print("ERROR: no rows extracted for any lineage", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
