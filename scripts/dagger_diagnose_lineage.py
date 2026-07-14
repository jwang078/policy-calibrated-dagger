#!/usr/bin/env python3
"""Cross-round DAgger data-health dashboard for a weighted-sampling lineage.

Where dagger_detect_dataset_anomalies.py inspects ONE dataset, this walks the
whole lineage (base + every round) and correlates data-health metrics against
the per-round eval success, to localize WHY a DAgger lineage isn't improving.

It answers the questions that the flat-success symptom raises:

  * SIGNAL DILUTION — under --use_weighted_sampling each source gets a fixed
    per-batch share (base = 1 - dagger_data_fraction; each round splits the
    rest). With many rounds, a round's hard-won corrections can be <2% of every
    batch → drowned. We read the actual weights from the dagger config sidecar
    and print the effective share + base:dagger ratio.

  * VOLUME TREND — a working DAgger needs FEWER interventions over time. Flat or
    rising intervention volume across rounds ⇒ non-convergence.

  * NORMALIZATION COMPRESSION (the aggregated-norm footgun, CLAUDE.md) — with
    norm_mode=aggregated the policy normalizes every source by the
    min-of-mins/max-of-maxes across ALL sources. If one source's action-delta
    range is wide, every other source's data compresses into a sub-band of
    [-1,1]. We compute each source's normalized span (q01..q99 under the
    aggregated range) straight from the stats_rel8 sidecars — small span ⇒
    that source is compressed.

  * NORM CLIPPING — fraction of a source's relative-action chunk deltas that
    land outside the aggregated [-1,1] (clipped at train time). ~0 under
    aggregated mode (the aggregate is the union); non-zero would indicate a
    stats/mode mismatch. Reuses the NORM_CLIP class added to the anomaly
    scanner.

  * ANOMALY RATE — teleports / tracking-error / padded episodes per round via
    the anomaly scanner, as a data-quality trend.

Reuses (no duplication): dagger_naming (parse + sidecar), dagger_plot.scan_round
(per-round eval metrics), dagger_detect_dataset_anomalies.scan_dataset +
load_action_delta_stats (per-dataset scan + NORM_CLIP).

Usage:
    python my_scripts/dagger_diagnose_lineage.py \\
        --dataset_path ~/.cache/huggingface/lerobot/JennyWWW/lever_g0_d30_clean_03dag_diff_r_dag16

    # fast (sidecar-only: dilution + volume + compression, no parquet scan):
    python my_scripts/dagger_diagnose_lineage.py --dataset_path ... --no_scan_data
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dagger_detect_dataset_anomalies import (  # noqa: E402
    ANOMALY_CLASSES,
    DEFAULT_ABS_THRESHOLD_RAD,
    DEFAULT_EDGE_IDLE_FRAMES,
    DEFAULT_FROZEN_THRESHOLD_RAD,
    DEFAULT_GRIPPER_DRIFT_TOLERANCE,
    DEFAULT_IMAGE_INTENSITY_THRESHOLD,
    DEFAULT_JOINT_SPIKE_MIN_ABS_RAD,
    DEFAULT_JOINT_SPIKE_RATIO,
    DEFAULT_JOINT_VELOCITY_THRESHOLD,
    DEFAULT_MIN_USEFUL_EPISODE_LEN,
    DEFAULT_NORM_CLIP_FRAC_THRESHOLD,
    DEFAULT_RATIO_MIN_ABS_RAD,
    DEFAULT_RATIO_THRESHOLD,
    DEFAULT_REL_HORIZON,
    DEFAULT_TRACKING_ERROR_THRESHOLD_RAD,
    _affected_episodes,
    _dataset_root_of,
    load_action_delta_stats,
    scan_dataset,
)
from dagger_naming import parse_dataset_short  # noqa: E402
from dagger_plot import scan_round  # noqa: E402

CACHE = Path.home() / ".cache" / "huggingface" / "lerobot"
REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_ROOT = REPO_ROOT / "outputs" / "training"
OUT_DIR = REPO_ROOT / "outputs" / "dagger" / "diagnostics"
GRIPPER_DIM = -1
# Batch share below this fraction → a round's corrections are effectively
# drowned by the base data. Used purely to flag the recommendation.
DILUTION_FLAG_FRAC = 0.05
# A source whose central-98% action-delta band occupies less than this fraction
# of [-1,1] under the aggregated normalizer is compressed.
COMPRESSION_FLAG_SPAN = 0.35
# Default batch size (config sidecar doesn't always record it) — only used to
# annotate "frames seen / round"; the headline dilution number is the weight.
DEFAULT_BATCH_SIZE = 64


def find_config_for_dataset(ds_name: str, explicit: Path | None) -> tuple[Path, dict] | None:
    """Locate the dagger config.json that governs this dataset's lineage.

    Robust to prefix/base_dataset_short mismatches: scans every
    <TRAIN_ROOT>/*/dagger/config.json and matches the one whose
    config.weighted_repo_ids contains this dataset (or a same-prefix sibling),
    picking the highest round (the most complete lineage snapshot). An explicit
    --sidecar short-circuits the search.
    """
    if explicit is not None:
        return explicit, json.loads(Path(explicit).read_text())
    prefix = parse_dataset_short(ds_name).prefix
    best: tuple[Path, dict] | None = None
    best_round = -1
    for cfg_path in sorted(TRAIN_ROOT.glob("*/dagger/config.json")):
        try:
            d = json.loads(cfg_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        cfg = d.get("config", {})
        repos = cfg.get("weighted_repo_ids") or d.get("weighted_repo_ids") or []
        names = [r.split("/")[-1] for r in repos]
        match = ds_name in names or (
            prefix is not None and any((parse_dataset_short(n).prefix == prefix) for n in names if n)
        )
        if match and int(d.get("round", -1)) > best_round:
            best = (cfg_path, d)
            best_round = int(d.get("round", -1))
    return best


def _read_info(repo_id: str) -> tuple[int, int]:
    """(total_episodes, total_frames) from a dataset's meta/info.json, or (0,0)."""
    info = CACHE / repo_id / "meta" / "info.json"
    try:
        d = json.loads(info.read_text())
        return int(d.get("total_episodes", 0)), int(d.get("total_frames", 0))
    except (OSError, json.JSONDecodeError):
        return 0, 0


def _arm_dims(n: int) -> list[int]:
    gdim = GRIPPER_DIM if GRIPPER_DIM >= 0 else n + GRIPPER_DIM
    return [c for c in range(n) if c != gdim]


def aggregate_delta_range(stats_paths: list[str]) -> tuple[np.ndarray, np.ndarray] | None:
    """min-of-mins / max-of-maxes of the action-delta across all sidecars —
    the range norm_mode=aggregated actually normalizes with."""
    mins, maxs = [], []
    for sp in stats_paths:
        loaded = load_action_delta_stats(Path(sp))
        if loaded is not None:
            mins.append(loaded[0])
            maxs.append(loaded[1])
    if not mins:
        return None
    n = min(m.shape[0] for m in mins)
    agg_min = np.min(np.stack([m[:n] for m in mins]), axis=0)
    agg_max = np.max(np.stack([m[:n] for m in maxs]), axis=0)
    return agg_min, agg_max


def compression_span(stats_path: str, agg_min: np.ndarray, agg_max: np.ndarray) -> float | None:
    """Fraction of [-1,1] that this source's central-98% action-delta band
    (q01..q99) occupies under the aggregated normalizer, averaged over arm
    joints. Small ⇒ this source is compressed by the aggregate range."""
    try:
        d = json.loads(Path(stats_path).read_text())
        act = d["action"]
        q01 = np.asarray(act["q01"], dtype=np.float64)
        q99 = np.asarray(act["q99"], dtype=np.float64)
    except (OSError, json.JSONDecodeError, KeyError):
        return None
    n = min(q01.shape[0], agg_min.shape[0])
    arm = _arm_dims(n)
    denom = np.where((agg_max[:n] - agg_min[:n]) == 0.0, 1.0, agg_max[:n] - agg_min[:n])
    # normalized full range is 2.0 (-1..1); fraction = (q99-q01)/denom.
    frac = (q99[:n] - q01[:n]) / denom
    return float(np.mean(frac[arm]))


def scan_one(root: Path, agg_min, agg_max, scan_data: bool) -> dict:
    """Anomaly scan of one source against the aggregated normalizer (for
    NORM_CLIP). Returns the per-class episode counts + n_episodes, or empty
    when scan_data is False."""
    if not scan_data:
        return {}
    res = scan_dataset(
        root,
        DEFAULT_ABS_THRESHOLD_RAD,
        DEFAULT_RATIO_THRESHOLD,
        DEFAULT_RATIO_MIN_ABS_RAD,
        DEFAULT_FROZEN_THRESHOLD_RAD,
        DEFAULT_MIN_USEFUL_EPISODE_LEN,
        DEFAULT_EDGE_IDLE_FRAMES,
        DEFAULT_JOINT_VELOCITY_THRESHOLD,
        DEFAULT_JOINT_SPIKE_RATIO,
        DEFAULT_JOINT_SPIKE_MIN_ABS_RAD,
        DEFAULT_TRACKING_ERROR_THRESHOLD_RAD,
        GRIPPER_DIM,
        DEFAULT_GRIPPER_DRIFT_TOLERANCE,
        None,
        DEFAULT_IMAGE_INTENSITY_THRESHOLD,
        True,
        True,
        True,
        True,
        True,
        True,  # teleport,padded,tiny,nan,lead,trail
        False,
        True,
        True,
        True,
        False,  # jvel,jspike,track,gripper,image
        detect_norm_clip=(agg_min is not None),
        action_delta_min=agg_min,
        action_delta_max=agg_max,
        rel_horizon=DEFAULT_REL_HORIZON,
        norm_clip_frac_thr=DEFAULT_NORM_CLIP_FRAC_THRESHOLD,
    )
    out = {"n_episodes": res["n_episodes"], "n_affected": len(_affected_episodes(res))}
    for key, _, _ in ANOMALY_CLASSES:
        out[key] = len(res.get(key, {}))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--dataset_path",
        type=Path,
        required=True,
        help="Any per-round intervention dataset in the lineage (the cross-round "
        "view is resolved from its dagger config sidecar).",
    )
    p.add_argument(
        "--sidecar",
        type=Path,
        default=None,
        help="Explicit dagger config.json. Default: auto-find by scanning "
        "outputs/training/*/dagger/config.json for one whose weighted_repo_ids "
        "include this dataset.",
    )
    p.add_argument(
        "--no_scan_data",
        action="store_true",
        help="Skip the per-dataset parquet anomaly/clip scan (fast: dilution + "
        "volume + compression from sidecars only).",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"For the 'frames seen / round' estimate. Default {DEFAULT_BATCH_SIZE}.",
    )
    args = p.parse_args()

    ds_root = _dataset_root_of(args.dataset_path)
    found = find_config_for_dataset(ds_root.name, args.sidecar)
    if found is None:
        print(
            f"ERROR: no dagger config.json under {TRAIN_ROOT} references "
            f"{ds_root.name}. Pass --sidecar explicitly.",
            file=sys.stderr,
        )
        return 2
    cfg_path, sidecar = found
    cfg = sidecar.get("config", {})
    repos = cfg.get("weighted_repo_ids") or sidecar.get("weighted_repo_ids") or []
    weights = cfg.get("weighted_sample_weights") or sidecar.get("weighted_sample_weights") or []
    stats_paths = cfg.get("weighted_stats_paths") or sidecar.get("weighted_stats_paths") or []
    if not repos:
        print(
            f"ERROR: sidecar {cfg_path} has no weighted_repo_ids — is this a "
            f"--use_weighted_sampling lineage? (merge-mode lineages aren't "
            f"supported by this dashboard.)",
            file=sys.stderr,
        )
        return 2

    data_frac = cfg.get("dagger_data_fraction")
    finetune_steps = cfg.get("finetune_steps")
    norm_mode = cfg.get("norm_mode")
    action_format = cfg.get("action_format")
    train_out = sidecar.get("training_output_dir", "")
    # round r training dir = <lineage>_ft_dag{r}; strip the trailing dag number.
    import re

    train_base = re.sub(r"_ft_dag\d+$", "", str(train_out))

    lineage = parse_dataset_short(ds_root.name).prefix or ds_root.name
    print(f"\n{'=' * 78}")
    print(f"DAgger lineage diagnostic: {lineage}")
    print(f"  sidecar:       {cfg_path}")
    print(f"  sources:       {len(repos)} (base + {len(repos) - 1} round(s))")
    print(
        f"  weighted samp: data_fraction={data_frac}  norm_mode={norm_mode}  "
        f"action_format={action_format}  finetune_steps={finetune_steps}"
    )
    print(f"{'=' * 78}")

    agg = aggregate_delta_range(stats_paths) if stats_paths else None
    agg_min, agg_max = agg if agg is not None else (None, None)

    rows: list[dict] = []
    for i, repo_id in enumerate(repos):
        name = repo_id.split("/")[-1]
        weight = float(weights[i]) if i < len(weights) else float("nan")
        sp = stats_paths[i] if i < len(stats_paths) else None
        is_base = i == 0
        parsed = parse_dataset_short(name)
        round_n = 0 if is_base else (parsed.round if parsed.round is not None else i)
        eps, frames = _read_info(repo_id)

        span = compression_span(sp, agg_min, agg_max) if (sp and agg_min is not None) else None

        # per-round eval success
        succ = pos_err = ori_err = None
        if not is_base and train_base:
            rdir = Path(f"{train_base}_ft_dag{round_n}")
            if rdir.is_dir():
                sr = scan_round(rdir, round_n=round_n)
                if sr:
                    succ, pos_err, ori_err = sr.get("succ"), sr.get("pos_err"), sr.get("ori_err")

        # data scan (anomaly + clip) against the aggregated range
        root = CACHE / repo_id
        scan = scan_one(root, agg_min, agg_max, not args.no_scan_data) if root.is_dir() else {}

        rows.append(
            {
                "i": i,
                "role": "base" if is_base else f"dag{round_n}",
                "round": round_n,
                "name": name,
                "episodes": eps,
                "frames": frames,
                "weight": weight,
                "succ": succ,
                "pos_err": pos_err,
                "ori_err": ori_err,
                "norm_span": span,
                "n_episodes_scanned": scan.get("n_episodes"),
                "n_affected": scan.get("n_affected"),
                "teleports": scan.get("teleports"),
                "tracking_error": scan.get("tracking_error"),
                "norm_clip": scan.get("norm_clip"),
                "padded": scan.get("padded"),
            }
        )

    _print_table(rows)
    base_share, dagger_total, per_round = _dilution(rows)
    _print_recommendations(
        rows, base_share, dagger_total, per_round, data_frac, finetune_steps, args.batch_size, norm_mode
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"{lineage}_health.csv"
    _write_csv(rows, csv_path)
    png_path = OUT_DIR / f"{lineage}_health.png"
    _plot(rows, lineage, base_share, png_path)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {png_path}")
    return 0


def _print_table(rows: list[dict]) -> None:
    hdr = (
        f"{'src':<7} {'eps':>4} {'frames':>7} {'share%':>7} {'succ%':>6} "
        f"{'pos_err':>7} {'norm_span':>9} {'clip':>4} {'teleP':>5} {'trackE':>6}"
    )
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:

        def f(v, fmt, na="-"):
            return na if v is None else format(v, fmt)

        print(
            f"{r['role']:<7} {r['episodes']:>4} {r['frames']:>7} "
            f"{r['weight'] * 100:>6.2f} {f(r['succ'], '6.0f'):>6} "
            f"{f(r['pos_err'], '7.3f'):>7} {f(r['norm_span'], '9.2f'):>9} "
            f"{f(r['norm_clip'], '4d'):>4} {f(r['teleports'], '5d'):>5} "
            f"{f(r['tracking_error'], '6d'):>6}"
        )


def _dilution(rows: list[dict]) -> tuple[float, float, float]:
    base_share = rows[0]["weight"] if rows else float("nan")
    dagger_rows = list(rows[1:])
    dagger_total = sum(r["weight"] for r in dagger_rows)
    per_round = dagger_total / len(dagger_rows) if dagger_rows else float("nan")
    return base_share, dagger_total, per_round


def _print_recommendations(
    rows, base_share, dagger_total, per_round, data_frac, finetune_steps, batch_size, norm_mode
) -> None:
    n_rounds = len(rows) - 1
    print(f"\n{'─' * 78}\nDIAGNOSIS\n{'─' * 78}")
    # 1. dilution
    seen = (per_round * (finetune_steps or 0) * batch_size) if finetune_steps else None
    print(
        f"• Signal dilution: base gets {base_share * 100:.1f}% of every batch; "
        f"the {n_rounds} DAgger rounds split {dagger_total * 100:.1f}% → "
        f"{per_round * 100:.2f}% each."
    )
    if seen is not None:
        print(
            f"    ≈ {seen:.0f} frames/round seen over {finetune_steps} finetune steps (batch={batch_size})."
        )
    if per_round < DILUTION_FLAG_FRAC:
        print(
            f"    ⚠ each round < {DILUTION_FLAG_FRAC * 100:.0f}% of the batch — corrections "
            f"are likely DROWNED. Raise --dagger_data_fraction, run fewer/larger "
            f"rounds, or recency-weight recent rounds."
        )
    # 2. volume trend
    dag_frames = [r["frames"] for r in rows[1:] if r["frames"]]
    if len(dag_frames) >= 4:
        first_half = np.mean(dag_frames[: len(dag_frames) // 2])
        second_half = np.mean(dag_frames[len(dag_frames) // 2 :])
        trend = (
            "shrinking ✓"
            if second_half < 0.8 * first_half
            else ("rising ⚠" if second_half > 1.2 * first_half else "flat ⚠")
        )
        print(
            f"• Intervention volume: {first_half:.0f}→{second_half:.0f} frames (first→second half) — {trend}."
        )
        if "flat" in trend or "rising" in trend:
            print(
                "    ⚠ volume not shrinking ⇒ policy isn't needing fewer interventions "
                "⇒ non-convergence (consistent with flat success)."
            )
    # 3. compression
    spans = [(r["role"], r["norm_span"]) for r in rows if r["norm_span"] is not None]
    if spans:
        base_span = spans[0][1] if spans[0][0] == "base" else None
        worst = min(spans, key=lambda x: x[1])
        print(
            f"• Norm compression (norm_mode={norm_mode}): base span="
            f"{base_span if base_span is None else f'{base_span:.2f}'}, "
            f"most-compressed {worst[0]}={worst[1]:.2f} of [-1,1]."
        )
        if base_span is not None and base_span < COMPRESSION_FLAG_SPAN:
            print(
                f"    ⚠ base data occupies <{COMPRESSION_FLAG_SPAN:.0%} of the normalized "
                f"range — intervention deltas stretch the aggregate, compressing base "
                f"toward zero (policy biased to tiny/no-op actions)."
            )
    # 4. clipping
    clipped = [(r["role"], r["norm_clip"]) for r in rows if r.get("norm_clip") not in (None, 0)]
    if clipped:
        print(
            "• Norm clipping vs aggregated range: "
            + ", ".join(f"{role}={n}ep" for role, n in clipped)
            + " — unexpected under aggregated mode; check stats/mode alignment."
        )
    else:
        print(
            "• Norm clipping vs aggregated range: none (expected for "
            "norm_mode=aggregated — the aggregate is the union of all sources)."
        )
    # 5. anomalies
    anom = [(r["role"], r["n_affected"]) for r in rows if r.get("n_affected") not in (None, 0)]
    if anom:
        print("• Per-source anomalous episodes: " + ", ".join(f"{role}={n}" for role, n in anom))


def _write_csv(rows: list[dict], path: Path) -> None:
    cols = [
        "i",
        "role",
        "round",
        "name",
        "episodes",
        "frames",
        "weight",
        "succ",
        "pos_err",
        "ori_err",
        "norm_span",
        "n_episodes_scanned",
        "n_affected",
        "teleports",
        "tracking_error",
        "norm_clip",
        "padded",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})


def _plot(rows: list[dict], lineage: str, base_share: float, path: Path) -> None:
    dag = list(rows[1:])
    rounds = [r["round"] for r in dag]
    fig, axes = plt.subplots(3, 2, figsize=(15, 14))

    # (1) volume per round
    ax = axes[0, 0]
    ax.bar(rounds, [r["frames"] for r in dag], color="tab:blue")
    ax.set_title("(1) Intervention volume per round (should shrink if converging)")
    ax.set_xlabel("round")
    ax.set_ylabel("frames")

    # (2) success vs round (+ pos_err twin)
    ax = axes[0, 1]
    sr = [(r["round"], r["succ"]) for r in dag if r["succ"] is not None]
    if sr:
        ax.plot([x for x, _ in sr], [y for _, y in sr], "o-", color="tab:green", label="succ%")
    ax.axhline(100, ls="--", color="gray", lw=1)
    ax.set_ylabel("success %", color="tab:green")
    ax.set_xlabel("round")
    ax.set_title("(2) Eval success & position error vs round")
    ax2 = ax.twinx()
    pe = [(r["round"], r["pos_err"]) for r in dag if r["pos_err"] is not None]
    if pe:
        ax2.plot([x for x, _ in pe], [y for _, y in pe], "s-", color="tab:red", label="pos_err")
    ax2.set_ylabel("pos_err (m)", color="tab:red")

    # (3) batch share per source — the dilution headline
    ax = axes[1, 0]
    labels = [r["role"] for r in rows]
    colors = ["tab:gray"] + ["tab:orange"] * (len(rows) - 1)
    ax.bar(range(len(rows)), [r["weight"] * 100 for r in rows], color=colors)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_title(
        f"(3) Per-source batch share (base={base_share * 100:.0f}%, "
        f"each round≈{(1 - base_share) / max(1, len(rows) - 1) * 100:.1f}%)"
    )
    ax.set_ylabel("% of each training batch")

    # (4) compression: normalized span per source
    ax = axes[1, 1]
    spans = [r["norm_span"] if r["norm_span"] is not None else 0 for r in rows]
    ax.bar(range(len(rows)), spans, color=colors)
    ax.axhline(
        COMPRESSION_FLAG_SPAN, ls="--", color="red", lw=1, label=f"compressed < {COMPRESSION_FLAG_SPAN}"
    )
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_title("(4) Norm span (q01..q99 frac of [-1,1] under aggregated range)")
    ax.set_ylabel("fraction of [-1,1]")
    ax.legend(fontsize=8)

    # (5) anomaly rate per round
    ax = axes[2, 0]
    scanned = [r for r in dag if r.get("n_episodes_scanned")]
    if scanned:
        for key, color in [
            ("teleports", "tab:purple"),
            ("tracking_error", "tab:red"),
            ("padded", "tab:brown"),
            ("norm_clip", "tab:cyan"),
        ]:
            ys = [(r["round"], (r.get(key) or 0) / max(1, r["n_episodes_scanned"]) * 100) for r in scanned]
            ax.plot([x for x, _ in ys], [y for _, y in ys], "o-", label=key, color=color)
        ax.set_title("(5) Anomalous-episode rate per round")
        ax.set_xlabel("round")
        ax.set_ylabel("% of episodes")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "data scan skipped (--no_scan_data)", ha="center", va="center")
        ax.set_axis_off()

    # (6) success vs per-round volume (does more data help?)
    ax = axes[2, 1]
    pts = [(r["frames"], r["succ"]) for r in dag if r["succ"] is not None and r["frames"]]
    if pts:
        ax.scatter([x for x, _ in pts], [y for _, y in pts], color="tab:green")
        for r in dag:
            if r["succ"] is not None and r["frames"]:
                ax.annotate(str(r["round"]), (r["frames"], r["succ"]), fontsize=7)
    ax.set_title("(6) Success vs round volume (flat cloud ⇒ data isn't the lever)")
    ax.set_xlabel("round frames")
    ax.set_ylabel("success %")

    fig.suptitle(f"DAgger data-health dashboard — {lineage}", fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(path, dpi=110)


if __name__ == "__main__":
    sys.exit(main())
