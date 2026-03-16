#!/usr/bin/env python3
"""
Summarize evaluation results across multiple runs.

Usage:
    python summarize_evals.py <eval_output_folder>

Example:
    python summarize_evals.py /home/jennyw2/code/lerobot/outputs/eval_output/2026-01-28-102155_5episodes_successhightolerance_approachlever
"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt  # noqa: E402

# Set larger font sizes for all figure text (approximately double the defaults)
plt.rcParams.update(
    {
        "font.size": 20,
        "axes.titlesize": 21,
        "axes.labelsize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "legend.title_fontsize": 18,
    }
)

# ---------------------------------------------------------------------------
# Metric registry — single source of truth for all metric configuration.
# To add a new metric, add one MetricConfig entry here. Everything else
# (EvalResult fields, JSON loading, DataFrame columns, plots, summary tables)
# is driven from this list.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricConfig:
    name: str  # column name / JSON key (e.g. "pc_success")
    higher_is_better: bool  # direction for "best checkpoint" selection
    required: bool = False  # True = always has a value (default 0.0); False = nullable
    label: str | None = None  # display label override (None → auto-generate from name)
    plot: bool = True  # whether to generate plots for this metric
    summary_display: bool = False  # whether to show in best_checkpoints summary table
    per_task_key: str | None = None  # key in per_task metrics/info_metrics to get raw episode values
    per_task_is_info: bool = False  # if True, look in info_metrics sub-dict
    use_boxplot: bool = False  # True = show distribution (box plot); False = bar chart (binary/0-1 metrics)


METRICS: list[MetricConfig] = [
    MetricConfig(
        "avg_sum_reward",
        higher_is_better=True,
        required=True,
        summary_display=True,
        per_task_key="sum_rewards",
        use_boxplot=True,
    ),
    MetricConfig(
        "avg_max_reward",
        higher_is_better=True,
        required=True,
        plot=False,
        per_task_key="max_rewards",
        use_boxplot=True,
    ),
    MetricConfig(
        "pc_success",
        higher_is_better=True,
        required=True,
        label="% Success",
        summary_display=True,
        per_task_key="successes",
        use_boxplot=False,  # binary 0/1
    ),
    MetricConfig(
        "avg_episode_length",
        higher_is_better=False,
        summary_display=True,
        per_task_key="episode_length",
        per_task_is_info=True,
        use_boxplot=True,
    ),
    MetricConfig(
        "avg_final_position_error_m",
        higher_is_better=False,
        summary_display=True,
        per_task_key="final_position_error_m",
        per_task_is_info=True,
        use_boxplot=True,
    ),
    MetricConfig(
        "avg_final_orientation_error_deg",
        higher_is_better=False,
        per_task_key="final_orientation_error_deg",
        per_task_is_info=True,
        use_boxplot=True,
    ),
    MetricConfig(
        "avg_cam_looks_at_goal_score",
        higher_is_better=True,
        summary_display=True,
        per_task_key="cam_looks_at_goal_score",
        per_task_is_info=True,
        use_boxplot=True,
    ),
    MetricConfig(
        "avg_action_delta",
        higher_is_better=False,
        per_task_key="action_delta",
        per_task_is_info=True,
        use_boxplot=True,
    ),
    MetricConfig(
        "avg_action_accel",
        higher_is_better=False,
        per_task_key="action_accel",
        per_task_is_info=True,
        use_boxplot=True,
    ),
    MetricConfig(
        "avg_action_jerk",
        higher_is_better=False,
        per_task_key="action_jerk",
        per_task_is_info=True,
        use_boxplot=True,
    ),
    MetricConfig(
        "avg_truncated",
        higher_is_better=False,
        per_task_key="truncated",
        per_task_is_info=True,
        use_boxplot=False,  # binary 0/1
    ),
    MetricConfig(
        "avg_in_collision",
        higher_is_better=False,
        per_task_key="in_collision",
        per_task_is_info=True,
        use_boxplot=True,
    ),
    MetricConfig(
        "avg_episode_length_without_truncation",
        higher_is_better=False,
        label="Avg Episode Length W/O Truncation",
        per_task_key="episode_length_without_truncation",
        per_task_is_info=True,
        use_boxplot=True,
    ),
]

# Derived lookups
METRIC_BY_NAME: dict[str, MetricConfig] = {m.name: m for m in METRICS}
METRIC_HIGHER_IS_BETTER: dict[str, bool] = {m.name: m.higher_is_better for m in METRICS}
METRIC_NAMES: list[str] = [m.name for m in METRICS]
REQUIRED_METRICS: list[str] = [m.name for m in METRICS if m.required]
OPTIONAL_METRICS: list[str] = [m.name for m in METRICS if not m.required]
PLOT_METRICS: list[str] = [m.name for m in METRICS if m.plot and m.required]
OPTIONAL_PLOT_METRICS: list[str] = [m.name for m in METRICS if m.plot and not m.required]
SUMMARY_DISPLAY_METRICS: list[str] = [m.name for m in METRICS if m.summary_display]
BOXPLOT_METRIC_NAMES: set[str] = {m.name for m in METRICS if m.use_boxplot}


def select_best_checkpoint_per_experiment(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """For each experiment (exp_name), keep only the row with the best checkpoint for the given metric."""
    higher_is_better = METRIC_HIGHER_IS_BETTER.get(metric, True)
    df_valid = df[df[metric].notna()]
    if higher_is_better:
        idx = df_valid.groupby("exp_name")[metric].idxmax()
    else:
        idx = df_valid.groupby("exp_name")[metric].idxmin()
    return df.loc[idx].reset_index(drop=True)


def compute_normalized_scores(df: pd.DataFrame) -> pd.Series:
    """Compute a composite normalized score (0-1) for each row.

    For each metric in METRIC_HIGHER_IS_BETTER, min-max normalize across the rows
    (flipping for lower-is-better so that 1.0 is always best). Average across all
    available metrics. Rows with no valid metrics get score 0.
    """
    normalized = pd.DataFrame(index=df.index)
    for metric, higher_is_better in METRIC_HIGHER_IS_BETTER.items():
        if metric not in df.columns:
            continue
        col = df[metric]
        valid = col.notna()
        if not valid.any():
            continue
        min_val = col[valid].min()
        max_val = col[valid].max()
        if max_val == min_val:
            # All same value — everyone gets 1.0
            normalized[metric] = np.where(valid, 1.0, np.nan)
        elif higher_is_better:
            normalized[metric] = np.where(valid, (col - min_val) / (max_val - min_val), np.nan)
        else:
            normalized[metric] = np.where(valid, (max_val - col) / (max_val - min_val), np.nan)

    # Average across metrics, ignoring NaN
    return normalized.mean(axis=1).fillna(0.0)


def select_overall_best_checkpoint_per_experiment(df: pd.DataFrame) -> pd.DataFrame:
    """For each experiment, pick the checkpoint with the highest normalized composite score.

    Ties are broken by lowest checkpoint_numeric (fewest training steps).
    """
    scores = compute_normalized_scores(df)
    df_scored = df.assign(_score=scores)

    best_indices = []
    for _, group in df_scored.groupby("exp_name"):
        max_score = group["_score"].max()
        tied = group[group["_score"] == max_score]
        best_indices.append(tied["checkpoint_numeric"].idxmin())

    return df.loc[best_indices].reset_index(drop=True)


@dataclass
class EvalResult:
    """Parsed evaluation result with metadata.

    Metric fields below must match the ``name`` values in the METRICS registry.
    Required metrics (``required=True``) use ``float``; optional use ``float | None``.
    """

    folder_name: str
    method: str  # "diffusion" or "pi0.5"
    task: str  # e.g., "approach_lever"
    trajectory_gen: str  # "1st RRT", "5th RRT", or "5path"
    cameras: str  # "external", "wrist", or "external+wrist"
    checkpoint: str  # e.g., "25000", "50000", "last"
    checkpoint_numeric: int  # numeric value for sorting
    has_2enc: bool  # whether it uses 2 encoders
    has_90crop: bool  # whether it uses 90 crop
    dataset: str | None  # e.g., "splatsim_approach_lever_1strrtpath"
    exp_name: str  # experiment name (folder name without checkpoint suffix)
    n_episodes: int  # number of eval episodes (metadata, not a metric)

    # Metrics — see METRICS registry for configuration (direction, labels, etc.)
    avg_sum_reward: float
    avg_max_reward: float
    pc_success: float
    avg_episode_length: float | None
    avg_final_position_error_m: float | None
    avg_final_orientation_error_deg: float | None
    avg_cam_looks_at_goal_score: float | None
    avg_action_delta: float | None
    avg_action_accel: float | None
    avg_action_jerk: float | None
    avg_truncated: float | None
    avg_in_collision: float | None
    avg_episode_length_without_truncation: float | None

    # Per-episode stds from per_task (None if < 2 samples or not available)
    avg_sum_reward_std: float | None
    avg_max_reward_std: float | None
    pc_success_std: float | None
    avg_episode_length_std: float | None
    avg_final_position_error_m_std: float | None
    avg_final_orientation_error_deg_std: float | None
    avg_cam_looks_at_goal_score_std: float | None
    avg_action_delta_std: float | None
    avg_action_accel_std: float | None
    avg_action_jerk_std: float | None
    avg_truncated_std: float | None
    avg_in_collision_std: float | None
    avg_episode_length_without_truncation_std: float | None

    # Raw per-episode values for boxplot metrics (None if not available)
    avg_sum_reward_episodes: list[float] | None
    avg_max_reward_episodes: list[float] | None
    pc_success_episodes: list[float] | None
    avg_episode_length_episodes: list[float] | None
    avg_final_position_error_m_episodes: list[float] | None
    avg_final_orientation_error_deg_episodes: list[float] | None
    avg_cam_looks_at_goal_score_episodes: list[float] | None
    avg_action_delta_episodes: list[float] | None
    avg_action_accel_episodes: list[float] | None
    avg_action_jerk_episodes: list[float] | None
    avg_truncated_episodes: list[float] | None
    avg_in_collision_episodes: list[float] | None
    avg_episode_length_without_truncation_episodes: list[float] | None


def parse_folder_name(folder_name: str) -> dict[str, Any]:
    """Parse folder name to extract metadata."""
    result: dict[str, Any] = {
        "method": None,
        "task": None,
        "trajectory_gen": None,
        "cameras": None,
        "checkpoint": None,
        "has_2enc": False,
        "has_90crop": False,
    }

    # Determine method
    if folder_name.startswith("diffusion_"):
        result["method"] = "diffusion"
        remaining = folder_name[len("diffusion_") :]
    elif folder_name.startswith("pi05_"):
        result["method"] = "pi0.5"
        remaining = folder_name[len("pi05_") :]
    else:
        return result

    # Extract task (approach_lever)
    task_match = re.match(r"(approach_lever)_(.+)", remaining)
    if task_match:
        result["task"] = task_match.group(1).replace("_", " ")
        remaining = task_match.group(2)

    # Check for special flags
    result["has_2enc"] = "_2enc_" in remaining or remaining.endswith("_2enc")
    result["has_90crop"] = "_90crop_" in remaining or "_90crop" in remaining

    # Extract trajectory generation method
    # Note: 1path/1strrtpath = "1st RRT", 5path/5thrrtpath = "5th RRT"
    if "1strrtpath" in remaining or "1path" in remaining:
        result["trajectory_gen"] = "1st RRT"
    elif "5thrrtpath" in remaining or "5path" in remaining:
        result["trajectory_gen"] = "5th RRT"

    # Extract cameras
    # Check for combined cameras first
    if "basewristrgb" in remaining or "basewrist" in remaining:
        result["cameras"] = "external+wrist"
    elif "basergb" in remaining or "_base_" in remaining or remaining.endswith("_base"):
        result["cameras"] = "external"
    elif "wristrgb" in remaining or "_wrist_" in remaining or remaining.endswith("_wrist"):
        result["cameras"] = "wrist"

    # More specific camera check using regex for pi0.5 style naming
    if result["cameras"] is None:
        if re.search(r"_base_\d+$", remaining) or re.search(r"_base_last$", remaining):
            result["cameras"] = "external"
        elif re.search(r"_wrist_\d+$", remaining) or re.search(r"_wrist_last$", remaining):
            result["cameras"] = "wrist"
        elif re.search(r"_basewrist_\d+$", remaining) or re.search(r"_basewrist_last$", remaining):
            result["cameras"] = "external+wrist"

    # Extract checkpoint
    checkpoint_match = re.search(r"_(\d{6}|last)$", remaining)
    if checkpoint_match:
        result["checkpoint"] = checkpoint_match.group(1)

    return result


def get_checkpoint_numeric(checkpoint: str, method: str) -> int:
    """Convert checkpoint string to numeric value for sorting."""
    if checkpoint == "last":
        # Last checkpoint values based on method
        if method == "pi0.5":
            return 3000
        else:  # diffusion
            return 100000  # Could be 75000 or 100000, using 100000 as default
    else:
        return int(checkpoint)


# Camera suffixes to strip from dataset repo_id
CAMERA_SUFFIXES = ["_basewristrgb", "_basewrist", "_basergb", "_wristrgb", "_base", "_wrist"]


def extract_dataset_name(repo_id: str) -> str:
    """
    Extract the dataset name from a repo_id, stripping user prefix and camera suffix.

    Example:
        "JennyWWW/splatsim_approach_lever_1path_stretch_base" -> "splatsim_approach_lever_1path_stretch"
    """
    # Remove user prefix (e.g., "JennyWWW/")
    if "/" in repo_id:
        repo_id = repo_id.split("/", 1)[1]

    # Remove camera suffix
    for suffix in CAMERA_SUFFIXES:
        if repo_id.endswith(suffix):
            repo_id = repo_id[: -len(suffix)]
            break

    return repo_id


def _read_dataset_from_checkpoints(checkpoints_dir: Path) -> str | None:
    """Read dataset name from any checkpoint's train_config.json inside a checkpoints dir."""
    if not checkpoints_dir.exists():
        return None
    for checkpoint_dir in checkpoints_dir.iterdir():
        if not checkpoint_dir.is_dir():
            continue
        config_path = checkpoint_dir / "pretrained_model" / "train_config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                repo_id = config.get("dataset", {}).get("repo_id")
                if repo_id:
                    return extract_dataset_name(repo_id)
            except (json.JSONDecodeError, OSError):
                continue
    return None


def _get_camera_suffix(name: str) -> str | None:
    """Return the camera suffix of a folder name, or None."""
    for suffix in CAMERA_SUFFIXES:
        if name.endswith(suffix):
            return suffix
    return None


def _collect_training_dirs(training_base: Path) -> list[Path]:
    """Collect all training experiment directories (direct children and one level of subdirs)."""
    dirs: list[Path] = []
    if not training_base.exists():
        return dirs
    for entry in training_base.iterdir():
        if not entry.is_dir():
            continue
        checkpoints = entry / "checkpoints"
        if checkpoints.exists():
            dirs.append(entry)
        else:
            # Search one level deeper (date-prefixed subdirectories)
            for sub in entry.iterdir():
                if sub.is_dir() and (sub / "checkpoints").exists():
                    dirs.append(sub)
    return dirs


def get_dataset_from_training_folder(exp_name: str, training_base: Path) -> str | None:
    """
    Get the dataset name from the training folder's train_config.json.

    Tries exact name match first, then falls back to finding a training dir
    with the same camera suffix (different experiments that share a camera config
    use the same dataset).

    Args:
        exp_name: Experiment name (e.g., "diffusion_approach_lever_1strrtpath_basergb")
        training_base: Base path for training outputs

    Returns:
        Dataset name or None if not found
    """
    all_dirs = _collect_training_dirs(training_base)

    # 1. Try exact name match
    for d in all_dirs:
        if d.name == exp_name:
            result = _read_dataset_from_checkpoints(d / "checkpoints")
            if result:
                return result

    # 2. Fallback: find a training dir with the same camera suffix where all of
    #    its name tokens (underscore-separated) appear in exp_name's tokens.
    #    This handles cases like exp_name having an extra "lr1e-5" flag.
    exp_cam = _get_camera_suffix(exp_name)
    if exp_cam:
        exp_base_tokens = set(exp_name[: -len(exp_cam)].split("_"))
        best_match: Path | None = None
        best_match_len = 0
        for d in all_dirs:
            dir_cam = _get_camera_suffix(d.name)
            if dir_cam is None or dir_cam != exp_cam or d.name == exp_name:
                continue
            dir_base_tokens = set(d.name[: -len(dir_cam)].split("_"))
            # All training dir tokens must appear in exp_name tokens
            if dir_base_tokens.issubset(exp_base_tokens) and len(dir_base_tokens) > best_match_len:
                best_match = d
                best_match_len = len(dir_base_tokens)
        if best_match:
            result = _read_dataset_from_checkpoints(best_match / "checkpoints")
            if result:
                return result

    return None


def _extract_per_task_episode_values(per_task: list[dict], metric_cfg: MetricConfig) -> list[float] | None:
    """Extract all per-episode raw values for a metric from per_task entries.

    Returns a flat list of all episode values across all task entries, or None if not available.
    For 'successes', converts bool to float (1.0/0.0).
    """
    if metric_cfg.per_task_key is None:
        return None
    all_values: list[float] = []
    for task_entry in per_task:
        metrics = task_entry.get("metrics", {})
        source = metrics.get("info_metrics", {}) if metric_cfg.per_task_is_info else metrics
        values = source.get(metric_cfg.per_task_key)
        if values is None:
            continue
        for v in values:
            if isinstance(v, bool):
                all_values.append(1.0 if v else 0.0)
            elif v is None or (isinstance(v, float) and np.isnan(v)):
                continue  # skip sentinel values (e.g. truncated episodes for episode_length_without_truncation)
            else:
                all_values.append(float(v))
    return all_values if all_values else None


def _compute_per_task_stats(
    per_task: list[dict], metric_cfg: MetricConfig
) -> tuple[float | None, float | None]:
    """Compute mean and std for a metric from per_task raw episode values.

    Returns (mean, std). std is None if fewer than 2 samples.
    For pc_success, mean is returned as a percentage (0-100) to match overall format.
    """
    values = _extract_per_task_episode_values(per_task, metric_cfg)
    if not values:
        return None, None
    arr = np.array(values, dtype=float)
    mean_val = float(arr.mean())
    std_val = float(arr.std(ddof=1)) if len(arr) >= 2 else None
    # pc_success is stored as fraction in per_task (0/1), but overall uses 0-100
    if metric_cfg.name == "pc_success":
        mean_val = mean_val * 100.0
        if std_val is not None:
            std_val = std_val * 100.0
    return mean_val, std_val


def load_eval_results(eval_folder: Path, training_base: Path | None = None) -> list[EvalResult]:
    """Load all evaluation results from the folder."""
    results = []

    # Default training base if not provided
    if training_base is None:
        # Assume training outputs are at outputs/training relative to eval_output
        training_base = eval_folder.parent.parent / "training"

    for subdir in sorted(eval_folder.iterdir()):
        if not subdir.is_dir():
            continue

        eval_info_path = subdir / "eval_info.json"
        if not eval_info_path.exists():
            print(f"Warning: No eval_info.json in {subdir.name}")
            continue

        with open(eval_info_path) as f:
            data = json.load(f)

        # Parse folder name
        metadata = parse_folder_name(subdir.name)

        if metadata["method"] is None:
            print(f"Warning: Could not parse folder name: {subdir.name}")
            continue

        # Extract experiment name by stripping checkpoint suffix from folder name
        # e.g., "diffusion_approach_lever_1strrtpath_basergb_050000" -> "diffusion_approach_lever_1strrtpath_basergb"
        exp_name = re.sub(r"_(\d{6}|last)$", "", subdir.name)

        # Get dataset from training folder
        dataset = get_dataset_from_training_folder(exp_name, training_base)

        # Extract metrics from overall
        overall = data.get("overall", {})
        per_task = data.get("per_task", [])

        # Build metric kwargs: use per_task for means when available, else fall back to overall.
        # Also compute stds from per_task raw episode values.
        metric_values: dict[str, Any] = {}
        metric_stds: dict[str, Any] = {}
        metric_episodes: dict[str, Any] = {}
        for m in METRICS:
            default = 0.0 if m.required else None
            per_task_mean, per_task_std = _compute_per_task_stats(per_task, m)
            if per_task_mean is not None:
                metric_values[m.name] = per_task_mean
            else:
                metric_values[m.name] = overall.get(m.name, default)
            metric_stds[f"{m.name}_std"] = per_task_std
            metric_episodes[f"{m.name}_episodes"] = _extract_per_task_episode_values(per_task, m)

        result = EvalResult(
            folder_name=subdir.name,
            method=metadata["method"],
            task=metadata["task"],
            trajectory_gen=metadata["trajectory_gen"],
            cameras=metadata["cameras"],
            checkpoint=metadata["checkpoint"],
            checkpoint_numeric=get_checkpoint_numeric(metadata["checkpoint"], metadata["method"]),
            has_2enc=metadata["has_2enc"],
            has_90crop=metadata["has_90crop"],
            dataset=dataset,
            exp_name=exp_name,
            n_episodes=int(overall.get("n_episodes", 0)),
            **metric_values,
            **metric_stds,
            **metric_episodes,
        )
        results.append(result)

    return results


def create_dataframe(results: list[EvalResult]) -> pd.DataFrame:
    """Convert results to a pandas DataFrame."""
    data = []
    for r in results:
        row = {
            "folder_name": r.folder_name,
            "method": r.method,
            "task": r.task,
            "trajectory_gen": r.trajectory_gen,
            "cameras": r.cameras,
            "checkpoint": r.checkpoint,
            "checkpoint_numeric": r.checkpoint_numeric,
            "has_2enc": r.has_2enc,
            "has_90crop": r.has_90crop,
            "dataset": r.dataset,
            "exp_name": r.exp_name,
            "n_episodes": r.n_episodes,
        }
        for m in METRICS:
            row[m.name] = getattr(r, m.name)
            row[f"{m.name}_std"] = getattr(r, f"{m.name}_std")
            row[f"{m.name}_episodes"] = getattr(r, f"{m.name}_episodes")
        data.append(row)
    return pd.DataFrame(data)


CAMERA_ORDER = ["external", "wrist", "external+wrist"]

# Fixed y-axis limits per metric for cross-run comparability.
# (min, max) — None means use matplotlib auto-scaling for that metric.
METRIC_YLIM: dict[str, tuple[float, float]] = {
    "pc_success": (0, 100),
    "avg_sum_reward": (0, 1),
    "avg_max_reward": (0, 1),
    "avg_episode_length": (0, 500),
    "avg_episode_length_without_truncation": (0, 500),
    "avg_truncated": (0, 1),
    "avg_final_position_error_m": (0, 1.1),
    "avg_position_error_m": (0, 1.1),
    "avg_final_orientation_error_deg": (0, 200),
    "avg_orientation_error_deg": (0, 200),
    "avg_cam_looks_at_goal_score": (0, 150),
    "avg_final_cam_looks_at_goal_score": (0, 150),
    "avg_action_delta": (0, 1.0),
    "avg_final_action_delta": (0, 1.0),
    "avg_action_accel": (0, 0.5),
    "avg_final_action_accel": (0, 0.5),
    "avg_action_jerk": (0, 0.5),
    "avg_final_action_jerk": (0, 0.5),
    "avg_in_collision": (0, 500),
}


def get_metric_label(metric: str) -> str:
    """Convert metric name to a human-readable label."""
    # Synthetic metric not in the registry
    if metric == "avg_metric_rank":
        return "Avg Metric Rank (0-1, higher=better)"
    cfg = METRIC_BY_NAME.get(metric)
    if cfg and cfg.label:
        return cfg.label
    return metric.replace("_", " ").title()


def compute_method_normalized_stats(
    df: pd.DataFrame, metric: str, groupby_col: str
) -> tuple[pd.Series, pd.Series]:
    """
    Compute mean and std for a metric grouped by groupby_col, with equal weighting per method.

    Each method contributes equally to the final average, regardless of how many samples
    each method has. This prevents methods with more runs from dominating the average.

    Returns:
        (means, stds): Series indexed by groupby_col values
    """
    methods = list(df["method"].unique())
    n_methods = len(methods)

    if n_methods == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    # Compute per-method means for each group
    method_means = df.groupby([groupby_col, "method"])[metric].mean().unstack()

    # Average across methods (equal weight per method)
    normalized_means = method_means.mean(axis=1)

    # For std, we use pooled variance approach with equal method weights
    # First get per-method stds
    method_stds = df.groupby([groupby_col, "method"])[metric].std().unstack()

    # Compute combined std using the formula for combining distributions with equal weights
    # Combined variance = (1/n) * sum(var_i + (mean_i - combined_mean)^2)
    combined_vars: list[float] = []
    groups = list(normalized_means.index)

    for group in groups:
        group_mean = normalized_means.loc[group]  # type: ignore[call-overload]
        var_sum = 0.0
        valid_methods = 0
        for method in methods:
            if method in method_means.columns:
                mean_val = method_means.loc[group, method]  # type: ignore[call-overload]
                std_val = method_stds.loc[group, method]  # type: ignore[call-overload]
                if not (isinstance(mean_val, float) and np.isnan(mean_val)) and mean_val is not None:
                    std_is_nan = isinstance(std_val, float) and np.isnan(std_val)
                    method_var = std_val**2 if not std_is_nan else 0.0
                    # Variance contribution: within-method variance + between-method variance
                    var_sum += method_var + (mean_val - group_mean) ** 2
                    valid_methods += 1
        combined_vars.append(np.sqrt(var_sum / valid_methods) if valid_methods > 0 else np.nan)

    normalized_stds = pd.Series(combined_vars, index=normalized_means.index)

    return normalized_means, normalized_stds


def _pooled_std_for_group(group_df: pd.DataFrame, metric: str) -> float | None:
    """Compute pooled per-episode std for a group of rows (NOT divided by sqrt(N)).

    Use this if you want the spread of individual episodes rather than uncertainty of the mean.
    For error bars on summary charts, prefer _pooled_sem_for_group instead.
    """
    std_col = f"{metric}_std"
    if std_col not in group_df.columns:
        return None
    rows = group_df[[metric, std_col, "n_episodes"]].dropna(subset=[metric])
    if rows.empty:
        return None
    total_n = rows["n_episodes"].sum()
    if total_n < 2:
        return None
    grand_mean = (rows[metric] * rows["n_episodes"]).sum() / total_n
    var_sum = 0.0
    for _, row in rows.iterrows():
        n_i = row["n_episodes"]
        mean_i = row[metric]
        std_i = row[std_col]
        var_i = (
            (std_i**2) if (std_i is not None and not (isinstance(std_i, float) and np.isnan(std_i))) else 0.0
        )
        var_sum += n_i * (var_i + (mean_i - grand_mean) ** 2)
    return float(np.sqrt(var_sum / total_n))


def _wilson_ci_halfwidth(p_pct: float, n: int, z: float = 1.96) -> float:
    """Wilson score interval half-width for a proportion, returned in percentage points.

    p_pct: success rate in [0, 100]
    n: number of episodes
    z: z-score for desired confidence level (1.96 = 95%)
    """
    p = p_pct / 100.0
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    # Return symmetric half-width around the original mean (not the Wilson-adjusted center)
    # so it can be used directly as yerr on the bar chart.
    # We use the wider of the two asymmetric Wilson bounds as a conservative symmetric estimate.
    lo = (center - half) * 100.0
    hi = (center + half) * 100.0
    return float(max(p_pct - lo, hi - p_pct))


def _pooled_sem_for_group(group_df: pd.DataFrame, metric: str) -> float | None:
    """Compute pooled SEM for a group of rows using their per-episode mean, std, and n_episodes.

    For pc_success, uses Wilson score CI half-width instead of SEM (bounded, asymmetry-aware).
    For other metrics, pools variance across runs then divides by sqrt(N) to get SEM.

    Returns None if fewer than 2 total episodes across the group.
    """
    std_col = f"{metric}_std"
    if std_col not in group_df.columns:
        return None
    rows = group_df[[metric, std_col, "n_episodes"]].dropna(subset=[metric])
    if rows.empty:
        return None
    total_n = int(rows["n_episodes"].sum())
    if total_n < 2:
        return None

    # For binary success metric, use Wilson CI half-width
    if metric == "pc_success":
        grand_mean = float((rows[metric] * rows["n_episodes"]).sum() / total_n)
        return _wilson_ci_halfwidth(grand_mean, total_n)

    grand_mean = (rows[metric] * rows["n_episodes"]).sum() / total_n
    var_sum = 0.0
    for _, row in rows.iterrows():
        n_i = row["n_episodes"]
        mean_i = row[metric]
        std_i = row[std_col]
        var_i = (
            (std_i**2) if (std_i is not None and not (isinstance(std_i, float) and np.isnan(std_i))) else 0.0
        )
        var_sum += n_i * (var_i + (mean_i - grand_mean) ** 2)
    pooled_std = np.sqrt(var_sum / total_n)
    return float(pooled_std / np.sqrt(total_n))  # SEM


def _compute_grouped_errorbars(
    df: pd.DataFrame, metric: str, groupby_col: str
) -> tuple[pd.Series, pd.Series]:
    """Compute mean and SEM (or Wilson CI for pc_success) for each group."""
    means = df.groupby(groupby_col)[metric].mean()
    stds_dict = {}
    for group_val, group_df in df.groupby(groupby_col):
        stds_dict[group_val] = _pooled_sem_for_group(group_df, metric)
    stds = pd.Series(stds_dict).reindex(means.index)
    return means, stds


def _yerr_array(stds: pd.Series) -> np.ndarray | None:
    """Convert a stds Series to a yerr array for ax.bar(), replacing NaN with 0 (no error bar shown)."""
    arr = stds.to_numpy(dtype=float)
    # If all NaN, return None to suppress error bars entirely
    if np.all(np.isnan(arr)):
        return None
    arr = np.where(np.isnan(arr), 0.0, arr)
    return arr


_CI_NOTE = "Error bars: 95% Wilson CI"


def _savefig(fig_or_path, path: Path, *, errorbars: bool = True, dpi: int = 150) -> None:
    """Call tight_layout, optionally add CI note, save, and close the current figure."""
    if errorbars:
        plt.figtext(0.99, 0.01, _CI_NOTE, ha="right", va="bottom", fontsize=11, color="#666666")
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def _add_bar_labels(ax: plt.Axes, bars, values: np.ndarray, fontsize: int = 20) -> None:
    """Add value labels centered inside each bar in white."""
    for bar, val in zip(bars, values, strict=True):
        height = bar.get_height()
        if height <= 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height / 2,
            f"{val:.2f}",
            ha="center",
            va="center",
            fontsize=fontsize,
            color="white",
            fontweight="bold",
        )


def plot_metric_by_category(df: pd.DataFrame, metric: str, output_dir: Path, title_suffix: str = ""):
    """Create plots for a metric grouped by different categories."""

    # Filter out rows with NaN for this metric if needed
    df_valid = df[df[metric].notna()].copy()

    if df_valid.empty:
        print(f"No valid data for metric: {metric}")
        return

    metric_label = get_metric_label(metric)

    # 1. Bar plot by method
    fig, ax = plt.subplots(figsize=(10, 6))
    method_means, method_stds = _compute_grouped_errorbars(df_valid, metric, "method")
    yerr = _yerr_array(method_stds)
    bars = ax.bar(
        method_means.index,
        method_means.to_numpy(),
        yerr=yerr,
        capsize=5,
        error_kw={"ecolor": "#888888", "elinewidth": 1.5},
    )
    ax.set_xlabel("Method")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{metric_label} by Method{title_suffix}")
    _add_bar_labels(ax, bars, method_means.to_numpy())
    if metric in METRIC_YLIM:
        ax.set_ylim(*METRIC_YLIM[metric])
    _savefig(fig, output_dir / f"{metric}_by_method.png")

    # 2. Bar plot by dataset (method-normalized)
    df_with_dataset = df_valid[df_valid["dataset"].notna()]
    if not df_with_dataset.empty:
        _, ax = plt.subplots(figsize=(14, 8))
        dataset_means, _ = compute_method_normalized_stats(df_with_dataset, metric, "dataset")
        if not dataset_means.empty:
            # Use pooled per-episode stds for dataset grouping
            _, dataset_ep_stds = _compute_grouped_errorbars(df_with_dataset, metric, "dataset")
            dataset_ep_stds = dataset_ep_stds.reindex(dataset_means.index)
            yerr = _yerr_array(dataset_ep_stds)
            bars = ax.bar(
                range(len(dataset_means)),
                dataset_means.to_numpy(),
                yerr=yerr,
                capsize=5,
                error_kw={"ecolor": "#888888", "elinewidth": 1.5},
            )
            ax.set_xlabel("Dataset")
            ax.set_ylabel(metric_label)
            ax.set_title(f"{metric_label} by Dataset\n(method-normalized){title_suffix}")
            ax.set_xticks(range(len(dataset_means)))
            ax.set_xticklabels(dataset_means.index, rotation=10, ha="right", fontsize=16)
            _add_bar_labels(ax, bars, dataset_means.to_numpy(), fontsize=16)
            _savefig(None, output_dir / f"{metric}_by_dataset.png")

    # 3. Bar plot by trajectory generation (method-normalized)
    _, ax = plt.subplots(figsize=(10, 6))
    traj_means, _ = compute_method_normalized_stats(df_valid, metric, "trajectory_gen")
    _, traj_ep_stds = _compute_grouped_errorbars(df_valid, metric, "trajectory_gen")
    traj_ep_stds = traj_ep_stds.reindex(traj_means.index)
    yerr = _yerr_array(traj_ep_stds)
    bars = ax.bar(
        traj_means.index,
        traj_means.to_numpy(),
        yerr=yerr,
        capsize=5,
        error_kw={"ecolor": "#888888", "elinewidth": 1.5},
    )
    ax.set_xlabel("Trajectory Generation")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{metric_label} by Trajectory Generation\n(method-normalized){title_suffix}")
    _add_bar_labels(ax, bars, traj_means.to_numpy())
    if metric in METRIC_YLIM:
        ax.set_ylim(*METRIC_YLIM[metric])
    _savefig(None, output_dir / f"{metric}_by_trajectory_gen.png")

    # 4. Bar plot by cameras (method-normalized)
    _, ax = plt.subplots(figsize=(10, 6))
    cam_means, _ = compute_method_normalized_stats(df_valid, metric, "cameras")
    cam_means = cam_means.reindex(CAMERA_ORDER)
    _, cam_ep_stds = _compute_grouped_errorbars(df_valid, metric, "cameras")
    cam_ep_stds = cam_ep_stds.reindex(CAMERA_ORDER)
    yerr = _yerr_array(cam_ep_stds)
    bars = ax.bar(
        cam_means.index,
        cam_means.to_numpy(),
        yerr=yerr,
        capsize=5,
        error_kw={"ecolor": "#888888", "elinewidth": 1.5},
    )
    ax.set_xlabel("Cameras")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{metric_label} by Camera Config\n(method-normalized){title_suffix}")
    _add_bar_labels(ax, bars, cam_means.to_numpy())
    if metric in METRIC_YLIM:
        ax.set_ylim(*METRIC_YLIM[metric])
    _savefig(None, output_dir / f"{metric}_by_cameras.png")

    # 5. Grouped bar plot: method x trajectory_gen
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot = df_valid.pivot_table(values=metric, index="trajectory_gen", columns="method", aggfunc="mean")
    x = np.arange(len(pivot.index))
    width = 0.8 / len(pivot.columns)
    for i, method in enumerate(pivot.columns):
        offset = (i - (len(pivot.columns) - 1) / 2) * width
        values = pivot[method].fillna(0).to_numpy()
        # Compute per-cell pooled std (trajectory_gen x method)
        yerr_vals = []
        for traj in pivot.index:
            cell_df = df_valid[(df_valid["trajectory_gen"] == traj) & (df_valid["method"] == method)]
            yerr_vals.append(_pooled_sem_for_group(cell_df, metric) or 0.0)
        yerr_arr = np.array(yerr_vals)
        has_errs = yerr_arr.any()
        bars_group = ax.bar(
            x + offset,
            values,
            width,
            label=method,
            yerr=yerr_arr if has_errs else None,
            capsize=4,
            error_kw={"ecolor": "#888888", "elinewidth": 1.5} if has_errs else {},
        )
        _add_bar_labels(ax, bars_group, values, fontsize=13)
        # Add N/A labels for missing data
        for j, val in enumerate(pivot[method]):
            if pd.isna(val):
                ax.text(
                    x[j] + offset,
                    0,
                    "N/A",
                    ha="center",
                    va="bottom",
                    fontsize=14,
                    color="gray",
                    style="italic",
                )
    ax.set_xlabel("Trajectory Generation")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{metric_label}\nby Trajectory Generation and Method{title_suffix}")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=45)
    ax.legend(title="Method")
    if metric in METRIC_YLIM:
        ax.set_ylim(*METRIC_YLIM[metric])
    _savefig(fig, output_dir / f"{metric}_by_traj_and_method.png")

    # 6. Grouped bar plot: method x cameras
    fig, ax = plt.subplots(figsize=(12, 6.6))
    pivot = df_valid.pivot_table(values=metric, index="cameras", columns="method", aggfunc="mean")
    pivot = pivot.reindex(CAMERA_ORDER)
    x = np.arange(len(pivot.index))
    width = 0.8 / len(pivot.columns)
    for i, method in enumerate(pivot.columns):
        offset = (i - (len(pivot.columns) - 1) / 2) * width
        values = pivot[method].fillna(0).to_numpy()
        # Compute per-cell pooled std (cameras x method)
        yerr_vals = []
        for cam in pivot.index:
            cell_df = df_valid[(df_valid["cameras"] == cam) & (df_valid["method"] == method)]
            yerr_vals.append(_pooled_sem_for_group(cell_df, metric) or 0.0)
        yerr_arr = np.array(yerr_vals)
        has_errs = yerr_arr.any()
        cam_bars = ax.bar(
            x + offset,
            values,
            width,
            label=method,
            yerr=yerr_arr if has_errs else None,
            capsize=4,
            error_kw={"ecolor": "#888888", "elinewidth": 1.5} if has_errs else {},
        )
        _add_bar_labels(ax, cam_bars, values, fontsize=13)
        # Add N/A labels for missing data
        for j, val in enumerate(pivot[method]):
            if pd.isna(val):
                ax.text(
                    x[j] + offset,
                    0,
                    "N/A",
                    ha="center",
                    va="bottom",
                    fontsize=14,
                    color="gray",
                    style="italic",
                )
    ax.set_xlabel("Cameras")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{metric_label} by Camera Config and Method{title_suffix}")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=45)
    ax.legend(title="Method")
    if metric in METRIC_YLIM:
        ax.set_ylim(*METRIC_YLIM[metric])
    _savefig(fig, output_dir / f"{metric}_by_cameras_and_method.png")


def _collect_episodes_by_group(
    df: pd.DataFrame, metric: str, groupby_col: str
) -> tuple[list[str], list[list[float]]]:
    """Return (group_labels, list_of_episode_arrays) grouped by groupby_col."""
    eps_col = f"{metric}_episodes"
    labels = []
    episode_lists = []
    for group_val, group_df in df.groupby(groupby_col):
        episodes: list[float] = []
        if eps_col in group_df.columns:
            for val in group_df[eps_col]:
                if val is not None:
                    episodes.extend(val)
        labels.append(str(group_val))
        episode_lists.append(episodes)
    return labels, episode_lists


def _draw_boxplot(
    ax,
    labels: list[str],
    episode_lists: list[list[float]],
    metric_label: str,
    xlabel: str,
    title: str,
    rotation: int = 0,
    metric: str | None = None,
) -> None:
    """Draw a boxplot with overlaid jittered data points."""
    # Filter out empty groups
    pairs = [(lb, e) for lb, e in zip(labels, episode_lists, strict=False) if e]
    if not pairs:
        ax.set_title(title + "\n(no data)")
        return
    labels_clean, episode_lists_clean = zip(*pairs, strict=False)

    ax.boxplot(
        episode_lists_clean,
        tick_labels=labels_clean,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 2},
        boxprops={"facecolor": "#a8c8e8", "alpha": 0.7},
        whiskerprops={"linewidth": 1.5},
        capprops={"linewidth": 1.5},
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(metric_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    if metric and metric in METRIC_YLIM:
        ax.set_ylim(*METRIC_YLIM[metric])
    if rotation:
        ax.set_xticklabels(labels_clean, rotation=rotation, ha="right")


def _plot_grouped_boxplot_by_category(
    df: pd.DataFrame,
    metric: str,
    metric_label: str,
    methods: list[str],
    categories: list[str],
    category_col: str,
    *,
    group_label: str,
    title: str,
    out_path: Path,
) -> None:
    """Single figure: groups = categories (traj or camera), boxes per group = methods.

    Groups are spaced with a gap between them (no visible border). Shared y-axis.
    """
    eps_col = f"{metric}_episodes"
    n_methods = len(methods)
    n_cats = len(categories)
    if n_cats == 0 or n_methods == 0:
        return

    method_colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#f39c12"]

    # Box width and intra-group spacing
    width = 0.7 / max(n_methods, 1)
    group_gap = 1.5  # gap between category groups (in position units)

    # Assign x positions: each category group is centered, with group_gap between groups
    group_centers = np.arange(n_cats) * (1.0 + group_gap)

    all_positions: list[float] = []
    all_data: list[list[float]] = []
    all_colors: list[str] = []

    for gi, cat in enumerate(categories):
        cat_df = df[df[category_col] == cat]
        for mi, method in enumerate(methods):
            cell_df = cat_df[cat_df["method"] == method]
            episodes: list[float] = []
            for val in cell_df[eps_col]:
                if val is not None:
                    episodes.extend(v for v in val if not (isinstance(v, float) and np.isnan(v)))
            offset = (mi - (n_methods - 1) / 2) * width
            all_positions.append(float(group_centers[gi]) + offset)
            all_data.append(episodes)
            all_colors.append(method_colors[mi % len(method_colors)])

    # Filter empty boxes
    nonempty = [(p, d, c) for p, d, c in zip(all_positions, all_data, all_colors, strict=False) if d]
    if not nonempty:
        return
    positions_clean, data_clean, colors_clean = zip(*nonempty, strict=False)

    fig, ax = plt.subplots(figsize=(max(8, 2.5 * n_cats * n_methods), 6))
    bp = ax.boxplot(
        data_clean,
        positions=positions_clean,
        widths=width * 0.85,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 2},
        whiskerprops={"linewidth": 1.5},
        capprops={"linewidth": 1.5},
        manage_ticks=False,
    )
    for patch, color in zip(bp["boxes"], colors_clean, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks(group_centers)
    ax.set_xticklabels(categories)
    ax.set_xlabel(group_label)
    ax.set_ylabel(metric_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    if metric in METRIC_YLIM:
        ax.set_ylim(*METRIC_YLIM[metric])

    legend_handles = [
        mpatches.Patch(facecolor=method_colors[i % len(method_colors)], alpha=0.7, label=m)
        for i, m in enumerate(methods)
    ]
    ax.legend(handles=legend_handles, title="Method")
    _savefig(fig, out_path, errorbars=False)


def plot_metric_boxplots_by_category(df: pd.DataFrame, metric: str, output_dir: Path, title_suffix: str = ""):
    """Create box plots (with jittered points) for a metric grouped by different categories."""
    df_valid = df[df[metric].notna()].copy()
    if df_valid.empty:
        return
    eps_col = f"{metric}_episodes"
    if eps_col not in df_valid.columns:
        return
    # Skip if no episode data at all
    if not any(v is not None and len(v) > 0 for v in df_valid[eps_col]):
        return

    metric_label = get_metric_label(metric)

    # 1. By method
    fig, ax = plt.subplots(figsize=(10, 6))
    labels, episode_lists = _collect_episodes_by_group(df_valid, metric, "method")
    _draw_boxplot(
        ax,
        labels,
        episode_lists,
        metric_label,
        "Method",
        f"{metric_label} by Method{title_suffix}",
        metric=metric,
    )
    _savefig(fig, output_dir / f"{metric}_by_method.png", errorbars=False)

    # 2. By trajectory generation
    fig, ax = plt.subplots(figsize=(10, 6))
    labels, episode_lists = _collect_episodes_by_group(df_valid, metric, "trajectory_gen")
    _draw_boxplot(
        ax,
        labels,
        episode_lists,
        metric_label,
        "Trajectory Generation",
        f"{metric_label} by Trajectory Generation{title_suffix}",
        metric=metric,
    )
    _savefig(fig, output_dir / f"{metric}_by_trajectory_gen.png", errorbars=False)

    # 3. By cameras
    fig, ax = plt.subplots(figsize=(10, 6))
    # Preserve CAMERA_ORDER
    cam_groups = dict(zip(*_collect_episodes_by_group(df_valid, metric, "cameras"), strict=False))
    cam_labels = [c for c in CAMERA_ORDER if c in cam_groups]
    cam_episodes = [cam_groups[c] for c in cam_labels]
    _draw_boxplot(
        ax,
        cam_labels,
        cam_episodes,
        metric_label,
        "Cameras",
        f"{metric_label} by Camera Config{title_suffix}",
        metric=metric,
    )
    _savefig(fig, output_dir / f"{metric}_by_cameras.png", errorbars=False)

    # 4. By method x trajectory_gen — grouped by traj, methods side-by-side, shared y-axis
    methods = sorted(df_valid["method"].unique())
    trajs = [t for t in TRAJECTORY_ORDER if t in df_valid["trajectory_gen"].unique()]
    _plot_grouped_boxplot_by_category(
        df_valid,
        metric,
        metric_label,
        methods,
        trajs,
        "trajectory_gen",
        group_label="Trajectory Generation",
        title=f"{metric_label} by Method × Trajectory Gen{title_suffix}",
        out_path=output_dir / f"{metric}_by_method_traj.png",
    )

    # 5. By method x cameras — grouped by camera, methods side-by-side, shared y-axis
    cams = [c for c in CAMERA_ORDER if c in df_valid["cameras"].unique()]
    _plot_grouped_boxplot_by_category(
        df_valid,
        metric,
        metric_label,
        methods,
        cams,
        "cameras",
        group_label="Camera Config",
        title=f"{metric_label} by Method × Camera Config{title_suffix}",
        out_path=output_dir / f"{metric}_by_method_cameras.png",
    )


def plot_metric_over_checkpoints(df: pd.DataFrame, metric: str, output_dir: Path, title_suffix: str = ""):
    """Per-method line plots showing metric over checkpoints (one plot per method, separate scales)."""
    df_valid = df[df[metric].notna()].copy()
    if df_valid.empty:
        return

    metric_label = get_metric_label(metric)
    for method in sorted(df_valid["method"].unique()):
        method_df = df_valid[df_valid["method"] == method]
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot each experiment as its own line
        for exp_name, exp_df in method_df.groupby("exp_name"):
            exp_sorted = exp_df.sort_values("checkpoint_numeric")
            ax.plot(
                exp_sorted["checkpoint_numeric"],
                exp_sorted[metric],
                marker="o",
                label=exp_name,
                alpha=0.7,
            )

        ax.set_xlabel("Checkpoint")
        ax.set_ylabel(metric_label)
        ax.set_title(f"{metric_label}\nover Checkpoints — {method}{title_suffix}")
        ax.legend(fontsize=8, loc="best")
        ax.set_xscale("log")
        method_slug = method.replace(".", "").replace(" ", "_").lower()
        _savefig(fig, output_dir / f"{metric}_over_checkpoints_{method_slug}.png", errorbars=False)


def plot_heatmap(df: pd.DataFrame, metric: str, output_dir: Path):
    """Create a heatmap showing metric across method/trajectory/camera combinations."""
    df_valid = df[df[metric].notna()].copy()

    if df_valid.empty:
        return

    metric_label = get_metric_label(metric)

    # Create a combined category with consistent ordering
    df_valid["config"] = df_valid["trajectory_gen"] + " / " + df_valid["cameras"]

    # Define the order for configs (trajectory x camera)
    traj_order = ["1st RRT", "5th RRT"]
    config_order = [f"{t} / {c}" for t in traj_order for c in CAMERA_ORDER]

    # Pivot for heatmap
    pivot = df_valid.pivot_table(values=metric, index="config", columns="method", aggfunc="mean")
    # Reindex to get consistent ordering (only keep configs that exist)
    config_order_existing = [c for c in config_order if c in pivot.index]
    pivot = pivot.reindex(config_order_existing)

    fig, ax = plt.subplots(figsize=(10, max(8, len(pivot) * 0.4)))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric_label)

    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color="white" if val > pivot.values.max() * 0.5 else "black",
                )

    ax.set_title(f"{metric_label} Heatmap")
    _savefig(fig, output_dir / f"{metric}_heatmap.png", errorbars=False)


def plot_success_rate_comparison(df: pd.DataFrame, output_dir: Path):
    """Create a comprehensive success rate comparison plot."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Success rate by method and checkpoint (line plot)
    ax = axes[0, 0]
    for method in df["method"].unique():
        method_df = df[df["method"] == method]
        checkpoint_means = method_df.groupby("checkpoint_numeric")["pc_success"].mean().sort_index()
        ax.plot(checkpoint_means.index, checkpoint_means.values, marker="o", label=method, linewidth=2)
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Success Rate over Checkpoints")
    ax.legend()
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    # 2. Success rate by trajectory and method (grouped bar)
    ax = axes[0, 1]
    pivot = df.pivot_table(values="pc_success", index="trajectory_gen", columns="method", aggfunc="mean")
    pivot.plot(kind="bar", ax=ax, width=0.8)
    ax.set_xlabel("Trajectory Generation")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Success Rate by Trajectory and Method")
    ax.legend(title="Method")
    plt.sca(ax)
    plt.xticks(rotation=45)

    # 3. Success rate by cameras and method (grouped bar)
    ax = axes[1, 0]
    pivot = df.pivot_table(values="pc_success", index="cameras", columns="method", aggfunc="mean")
    pivot = pivot.reindex(CAMERA_ORDER)
    pivot.plot(kind="bar", ax=ax, width=0.8)
    ax.set_xlabel("Cameras")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Success Rate by Camera and Method")
    ax.legend(title="Method")
    plt.sca(ax)
    plt.xticks(rotation=45)

    # 4. Best configurations (top 10)
    ax = axes[1, 1]
    df_sorted = df.sort_values("pc_success", ascending=False).head(15)
    labels = [
        f"{r.method[:4]}/{r.trajectory_gen}/{r.cameras[:4]}/{r.checkpoint}" for _, r in df_sorted.iterrows()
    ]
    colors = ["#2ecc71" if r.method == "pi0.5" else "#3498db" for _, r in df_sorted.iterrows()]
    ax.barh(range(len(labels)), df_sorted["pc_success"].values, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=16)
    ax.set_xlabel("Success Rate (%)")
    ax.set_title("Top 15 Configurations by Success Rate")
    ax.invert_yaxis()

    _savefig(fig, output_dir / "success_rate_comparison.png", errorbars=False)


TRAJECTORY_ORDER = ["1st RRT", "5th RRT"]


def plot_metric_by_method(df: pd.DataFrame, metric: str, output_dir: Path):
    """Create one chart per method showing metric grouped by camera config and trajectory generation."""
    df_valid = df[df[metric].notna()].copy()

    if df_valid.empty:
        print(f"No valid data for metric: {metric}")
        return

    metric_label = get_metric_label(metric)

    # First pass: compute the maximum y value across all methods for consistent y-axis
    # Also collect all cameras and trajectories present across all methods
    max_y_value = 0.0
    grouped_data: dict[str, pd.DataFrame] = {}
    all_cameras: set[str] = set()
    all_trajectories: set[str] = set()

    for method in df_valid["method"].unique():
        method_df = df_valid[df_valid["method"] == method]

        if method_df.empty:
            continue

        # Calculate means for each camera + trajectory combination
        grouped = method_df.groupby(["cameras", "trajectory_gen"])[metric].mean().unstack(fill_value=0)

        if grouped.empty:
            continue

        grouped_data[method] = grouped
        max_y_value = max(max_y_value, grouped.values.max())
        all_cameras.update(grouped.index.tolist())
        all_trajectories.update(grouped.columns.tolist())

    # Determine consistent categories across all methods
    # Use predefined order, but only include categories that exist in at least one method
    consistent_cameras = [c for c in CAMERA_ORDER if c in all_cameras]
    consistent_trajectories = [t for t in TRAJECTORY_ORDER if t in all_trajectories]

    # Add headroom for error bars (20% padding)
    y_limit = max_y_value * 1.20

    use_boxplot = metric in BOXPLOT_METRIC_NAMES
    eps_col = f"{metric}_episodes"

    # Second pass: create the plots with consistent y-axis and x-axis categories
    for method, grouped in grouped_data.items():
        # Reindex to have consistent categories (missing ones will be NaN)
        grouped = grouped.reindex(index=consistent_cameras, columns=consistent_trajectories)
        method_df = df_valid[df_valid["method"] == method]
        safe_method = method.replace(".", "").replace(" ", "_")

        _, ax = plt.subplots(figsize=(10, 6))
        colors = ["#3498db", "#e74c3c"]  # Blue for 1st RRT, Red for 5th RRT

        has_episode_data = (
            eps_col in df_valid.columns
            and df_valid[eps_col].apply(lambda v: v is not None and len(v) > 0).any()
        )
        if use_boxplot and has_episode_data:
            # Grouped boxplot: one box per (camera, traj) cell, grouped by camera on x-axis
            n_traj = len(consistent_trajectories)
            width = 0.8 / max(n_traj, 1)
            positions_map: dict[tuple[str, str], float] = {}
            x = np.arange(len(consistent_cameras))
            for i, traj in enumerate(consistent_trajectories):
                offset = (i - (n_traj - 1) / 2) * width
                for j, cam in enumerate(consistent_cameras):
                    positions_map[(cam, traj)] = x[j] + offset

            # Collect episode data per cell and draw boxplots
            all_positions = []
            all_data = []
            all_colors = []
            for i, traj in enumerate(consistent_trajectories):
                for _j, cam in enumerate(consistent_cameras):
                    cell_df = method_df[(method_df["cameras"] == cam) & (method_df["trajectory_gen"] == traj)]
                    episodes: list[float] = []
                    for val in cell_df[eps_col]:
                        if val is not None:
                            episodes.extend(val)
                    if episodes:
                        all_positions.append(positions_map[(cam, traj)])
                        all_data.append(episodes)
                        all_colors.append(colors[i % len(colors)])

            if all_data:
                bp = ax.boxplot(
                    all_data,
                    positions=all_positions,
                    widths=width * 0.85,
                    patch_artist=True,
                    medianprops={"color": "black", "linewidth": 2},
                    whiskerprops={"linewidth": 1.5},
                    capprops={"linewidth": 1.5},
                    manage_ticks=False,
                )
                for patch, color in zip(bp["boxes"], all_colors, strict=False):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

            # Legend proxies
            legend_handles = [
                mpatches.Patch(facecolor=colors[i % len(colors)], alpha=0.7, label=traj)
                for i, traj in enumerate(consistent_trajectories)
            ]
            ax.legend(handles=legend_handles, title="Trajectory Gen")
            ax.set_xticks(x)
            ax.set_xticklabels(consistent_cameras)
            ax.grid(True, alpha=0.3, axis="y")
            ax.set_xlabel("Camera Config")
            ax.set_ylabel(metric_label)
            ax.set_title(f"{metric_label} - {method}")
            if metric in METRIC_YLIM:
                ax.set_ylim(*METRIC_YLIM[metric])
            _savefig(None, output_dir / f"{metric}_by_camera_traj_{safe_method}.png", errorbars=False)

        else:
            x = np.arange(len(grouped.index))
            width = 0.35
            n_traj = len(grouped.columns)

            for i, traj in enumerate(grouped.columns):
                offset = (i - (n_traj - 1) / 2) * width
                values = grouped[traj].fillna(0).to_numpy()
                yerr_vals = []
                for cam in grouped.index:
                    cell_df = method_df[(method_df["cameras"] == cam) & (method_df["trajectory_gen"] == traj)]
                    yerr_vals.append(_pooled_sem_for_group(cell_df, metric) or 0.0)
                yerr_arr = np.array(yerr_vals)
                has_errs = yerr_arr.any()
                bars = ax.bar(
                    x + offset,
                    values,
                    width,
                    label=traj,
                    color=colors[i % len(colors)],
                    yerr=yerr_arr if has_errs else None,
                    capsize=4,
                    error_kw={"ecolor": "#888888", "elinewidth": 1.5} if has_errs else {},
                )
                _add_bar_labels(ax, bars, values, fontsize=13)
                for j, original_value in enumerate(grouped[traj]):
                    if pd.isna(original_value):
                        ax.text(
                            x[j] + offset,
                            0,
                            "N/A",
                            ha="center",
                            va="bottom",
                            fontsize=14,
                            color="gray",
                            style="italic",
                        )

            ax.set_xticks(x)
            ax.set_xticklabels(grouped.index)
            ax.set_ylim(*METRIC_YLIM[metric]) if metric in METRIC_YLIM else ax.set_ylim(0, y_limit)
            ax.legend(title="Trajectory Gen")
            ax.grid(True, alpha=0.3, axis="y")
            ax.set_xlabel("Camera Config")
            ax.set_ylabel(metric_label)
            ax.set_title(f"{metric_label} - {method}")
            _savefig(None, output_dir / f"{metric}_by_camera_traj_{safe_method}.png")


def compile_best_checkpoint_videos(df: pd.DataFrame, eval_folder: Path, output_dir: Path) -> None:
    """For each experiment's best checkpoint, concatenate all eval episode videos into one file.

    Videos are saved to output_dir/videos/<folder_name>.mp4.
    Episodes across all task groups are concatenated in order (task_group sorted, then episode index).
    Requires ffmpeg on PATH.
    """
    import shutil
    import subprocess
    import tempfile

    if shutil.which("ffmpeg") is None:
        print("Warning: ffmpeg not found on PATH, skipping video compilation.")
        return

    df_best = select_overall_best_checkpoint_per_experiment(df)
    videos_out = output_dir / "videos"
    videos_out.mkdir(exist_ok=True)

    for _, row in df_best.iterrows():
        folder_name = row["folder_name"]
        eval_run_dir = eval_folder / folder_name / "videos"
        if not eval_run_dir.exists():
            print(f"  Skipping {folder_name}: no videos folder found")
            continue

        # Collect all episode mp4s across task group subdirs, sorted by subdir then filename
        episode_files: list[Path] = []
        for task_dir in sorted(eval_run_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            episode_files.extend(sorted(task_dir.glob("eval_episode_*.mp4")))

        if not episode_files:
            print(f"  Skipping {folder_name}: no episode videos found")
            continue

        out_path = videos_out / f"{folder_name}.mp4"

        # Write a concat list file and run ffmpeg
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            concat_file = Path(f.name)
            for ep in episode_files:
                f.write(f"file '{ep.resolve()}'\n")

        try:
            result = subprocess.run(  # nosec B607
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(concat_file),
                    "-c",
                    "copy",
                    str(out_path),
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print(f"  Saved: {out_path.name} ({len(episode_files)} episodes)")
            else:
                print(
                    f"  Error compiling {folder_name}: {result.stderr.splitlines()[-1] if result.stderr else 'unknown error'}"
                )
        finally:
            concat_file.unlink(missing_ok=True)


def _build_agg_dict(df: pd.DataFrame) -> dict:
    """Build aggregation dict from METRICS registry, including optional metrics only if data exists."""
    agg: dict = {}
    for m in METRICS:
        if m.required or df[m.name].notna().any():
            agg[m.name] = ["mean", "std"]
    return agg


def generate_summary_table(df: pd.DataFrame, output_dir: Path):
    """Generate a summary CSV and print summary statistics."""
    # Save full results
    df.to_csv(output_dir / "all_results.csv", index=False)

    # Summary by method
    method_summary = df.groupby("method").agg(_build_agg_dict(df)).round(3)
    method_summary.to_csv(output_dir / "summary_by_method.csv")

    # Summary by trajectory
    traj_summary = df.groupby("trajectory_gen").agg(_build_agg_dict(df)).round(3)
    traj_summary.to_csv(output_dir / "summary_by_trajectory.csv")

    # Summary by cameras
    cam_summary = df.groupby("cameras").agg(_build_agg_dict(df)).round(3)
    cam_summary.to_csv(output_dir / "summary_by_cameras.csv")

    # Summary by dataset
    df_with_dataset = df[df["dataset"].notna()]
    dataset_summary = None
    if not df_with_dataset.empty:
        dataset_summary = df_with_dataset.groupby("dataset").agg(_build_agg_dict(df_with_dataset)).round(3)
        dataset_summary.to_csv(output_dir / "summary_by_dataset.csv")

    # Best configurations
    best_configs = df.nlargest(10, "pc_success")[
        ["folder_name", "method", "trajectory_gen", "cameras", "checkpoint", "pc_success", "avg_sum_reward"]
    ]
    best_configs.to_csv(output_dir / "best_configurations.csv", index=False)

    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    print(f"\nTotal evaluations: {len(df)}")
    print(f"Methods: {df['method'].unique().tolist()}")
    print(f"Trajectory generation: {df['trajectory_gen'].unique().tolist()}")
    print(f"Camera configs: {df['cameras'].unique().tolist()}")
    datasets = df_with_dataset["dataset"].unique().tolist() if not df_with_dataset.empty else []
    print(f"Datasets: {datasets}")

    print("\n--- Summary by Method ---")
    print(method_summary.to_string())

    print("\n--- Summary by Trajectory Generation ---")
    print(traj_summary.to_string())

    print("\n--- Summary by Cameras ---")
    print(cam_summary.to_string())

    if dataset_summary is not None:
        print("\n--- Summary by Dataset ---")
        print(dataset_summary.to_string())

    print("\n--- Top 10 Configurations ---")
    print(best_configs.to_string(index=False))

    return method_summary, traj_summary, cam_summary, dataset_summary


def rank_experiments_by_normalized_score(df_best: pd.DataFrame) -> tuple[list[str], pd.Series]:
    """Rank experiments by normalized composite score of their best checkpoint.

    Args:
        df_best: DataFrame with one row per experiment (the best checkpoint for each).

    Returns:
        (exp_names ordered best to worst, scores Series aligned with df_best index)
    """
    scores = compute_normalized_scores(df_best)

    # Sort by score descending, then checkpoint_numeric ascending for ties
    ranking = df_best.assign(_score=scores).sort_values(
        ["_score", "checkpoint_numeric"], ascending=[False, True]
    )
    return ranking["exp_name"].tolist(), scores


def print_best_checkpoints_summary(df: pd.DataFrame, output_dir: Path):
    """Print and save a table showing all checkpoints per experiment, grouped by experiment.

    Experiments are ranked by normalized composite score (best experiment first).
    Within each experiment, the best checkpoint is listed first (exp_name filled in),
    followed by other checkpoints (exp_name blank) sorted by checkpoint_numeric.
    Each row includes its avg_metric_rank (0-1, higher is better).
    """
    df_best = select_overall_best_checkpoint_per_experiment(df)
    best_set = set(zip(df_best["exp_name"], df_best["checkpoint"], strict=True))

    # Rank experiments by normalized score
    experiment_order, _ = rank_experiments_by_normalized_score(df_best)

    # Compute scores for ALL checkpoints so non-best rows get scores too
    df_with_scores = df.copy()
    df_with_scores["avg_metric_rank"] = compute_normalized_scores(df).round(3)

    display_cols = ["exp_name", "checkpoint", "avg_metric_rank"]
    for m in METRICS:
        if m.summary_display and m.name in df.columns and df[m.name].notna().any():
            display_cols.append(m.name)

    # Build rows: for each experiment (in ranked order), best checkpoint first, then the rest
    rows = []
    for exp_name in experiment_order:
        group = df_with_scores[df_with_scores["exp_name"] == exp_name].sort_values("checkpoint_numeric")
        best_mask = group.apply(lambda r: (r["exp_name"], r["checkpoint"]) in best_set, axis=1)
        best_row = group[best_mask].iloc[0]

        # Best checkpoint row: exp_name filled in
        rows.append(best_row[display_cols].to_dict())

        # Other checkpoints: exp_name blank, sorted by avg_metric_rank descending
        for _, other_row in group[~best_mask].sort_values("avg_metric_rank", ascending=False).iterrows():
            row = other_row[display_cols].to_dict()
            row["exp_name"] = ""
            rows.append(row)

    result_df = pd.DataFrame(rows, columns=display_cols)

    print("\n" + "=" * 80)
    print("BEST CHECKPOINT PER EXPERIMENT (ranked by avg_metric_rank)")
    print("avg_metric_rank: 0-1 avg of min-max normalized metrics (1.0 = best). Best checkpoint first.")
    print("=" * 80)
    print(result_df.to_string(index=False))

    # Save to CSV
    result_df.to_csv(output_dir / "best_checkpoints.csv", index=False)
    print(f"\nSaved to: {output_dir / 'best_checkpoints.csv'}")


def main():
    parser = argparse.ArgumentParser(description="Summarize evaluation results")
    parser.add_argument("eval_folder", type=str, help="Path to the evaluation output folder")
    args = parser.parse_args()

    eval_folder = Path(args.eval_folder)
    if not eval_folder.exists():
        print(f"Error: Folder does not exist: {eval_folder}")
        return 1

    # Create output directory for plots
    output_dir = eval_folder / "summary"
    output_dir.mkdir(exist_ok=True)

    print(f"Loading results from: {eval_folder}")
    results = load_eval_results(eval_folder)
    print(f"Loaded {len(results)} evaluation results")

    if not results:
        print("No results found!")
        return 1

    # Create DataFrame
    df = create_dataframe(results)

    # Generate summary tables
    generate_summary_table(df, output_dir)

    # Print best checkpoint per experiment
    print_best_checkpoints_summary(df, output_dir)

    # Generate plots for each metric (using best checkpoint per experiment)
    print("\nGenerating plots (best checkpoint per experiment for each metric)...")

    for metric in PLOT_METRICS:
        df_best = select_best_checkpoint_per_experiment(df, metric)
        if metric in BOXPLOT_METRIC_NAMES:
            plot_metric_boxplots_by_category(df_best, metric, output_dir)
        else:
            plot_metric_by_category(df_best, metric, output_dir)
        plot_heatmap(df_best, metric, output_dir)
        plot_metric_by_method(df_best, metric, output_dir)
        # Line plot uses ALL checkpoints to show progression
        plot_metric_over_checkpoints(df, metric, output_dir)

    for metric in OPTIONAL_PLOT_METRICS:
        if df[metric].notna().any():
            df_best = select_best_checkpoint_per_experiment(df, metric)
            if metric in BOXPLOT_METRIC_NAMES:
                plot_metric_boxplots_by_category(df_best, metric, output_dir)
            else:
                plot_metric_by_category(df_best, metric, output_dir)
            plot_heatmap(df_best, metric, output_dir)
            plot_metric_by_method(df_best, metric, output_dir)
            plot_metric_over_checkpoints(df, metric, output_dir)
        else:
            print(f"No entries have {metric} metric")

    # avg_metric_rank plots (same best-checkpoint-per-experiment filtering as other metrics)
    df["avg_metric_rank"] = compute_normalized_scores(df)
    df_best_rank = select_best_checkpoint_per_experiment(df, "avg_metric_rank")
    plot_metric_by_category(df_best_rank, "avg_metric_rank", output_dir)
    plot_heatmap(df_best_rank, "avg_metric_rank", output_dir)
    plot_metric_by_method(df_best_rank, "avg_metric_rank", output_dir)
    plot_metric_over_checkpoints(df, "avg_metric_rank", output_dir)

    # Comprehensive success rate comparison (uses best checkpoint for pc_success)
    df_best_success = select_best_checkpoint_per_experiment(df, "pc_success")
    plot_success_rate_comparison(df_best_success, output_dir)

    # Compile best-checkpoint episode videos into per-experiment concat videos
    print("\nCompiling best-checkpoint eval videos...")
    compile_best_checkpoint_videos(df, eval_folder, output_dir)

    # Warn if any metric's observed range falls outside its fixed ylim
    ylim_warnings = []
    for metric, (ymin, ymax) in METRIC_YLIM.items():
        if metric not in df.columns:
            continue
        col = df[metric].dropna()
        if col.empty:
            continue
        obs_min, obs_max = float(col.min()), float(col.max())
        # Also check per-episode values (used in boxplots — can exceed the aggregated mean range)
        eps_col = f"{metric}_episodes"
        if eps_col in df.columns:
            for val in df[eps_col]:
                if val is not None:
                    for v in val:
                        if not (isinstance(v, float) and np.isnan(v)):
                            obs_min = min(obs_min, float(v))
                            obs_max = max(obs_max, float(v))
        if obs_min < ymin or obs_max > ymax:
            ylim_warnings.append(
                f"  {metric}: data [{obs_min:.4g}, {obs_max:.4g}]  ylim [{ymin:.4g}, {ymax:.4g}]"
            )
    if ylim_warnings:
        print("\nWARNING: metric values fall outside fixed ylim — update METRIC_YLIM:")
        for w in ylim_warnings:
            print(w)

    print(f"\nPlots saved to: {output_dir}")

    return 0


if __name__ == "__main__":
    exit(main())
