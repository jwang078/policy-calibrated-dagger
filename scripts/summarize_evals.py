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


@dataclass
class EvalResult:
    """Parsed evaluation result with metadata."""

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

    # Metrics
    avg_sum_reward: float
    avg_max_reward: float
    pc_success: float
    n_episodes: int
    avg_episode_length: float | None
    avg_final_position_error_m: float | None
    avg_final_orientation_error_deg: float | None
    avg_cam_looks_at_goal_score: float | None
    avg_action_delta: float | None


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


def get_dataset_from_training_folder(exp_name: str, training_base: Path) -> str | None:
    """
    Get the dataset name from the training folder's train_config.json.

    Args:
        exp_name: Experiment name (e.g., "diffusion_approach_lever_1strrtpath_basergb")
        training_base: Base path for training outputs

    Returns:
        Dataset name or None if not found
    """
    training_folder = training_base / exp_name / "checkpoints"
    if not training_folder.exists():
        return None

    # Find any checkpoint with train_config.json
    for checkpoint_dir in training_folder.iterdir():
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
            avg_sum_reward=overall.get("avg_sum_reward", 0.0),
            avg_max_reward=overall.get("avg_max_reward", 0.0),
            pc_success=overall.get("pc_success", 0.0),
            n_episodes=overall.get("n_episodes", 0),
            avg_episode_length=overall.get("avg_episode_length"),
            avg_final_position_error_m=overall.get("avg_final_position_error_m"),
            avg_final_orientation_error_deg=overall.get("avg_final_orientation_error_deg"),
            avg_cam_looks_at_goal_score=overall.get("avg_cam_looks_at_goal_score"),
            avg_action_delta=overall.get("avg_action_delta"),
        )
        results.append(result)

    return results


def create_dataframe(results: list[EvalResult]) -> pd.DataFrame:
    """Convert results to a pandas DataFrame."""
    data = []
    for r in results:
        data.append(
            {
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
                "avg_sum_reward": r.avg_sum_reward,
                "avg_max_reward": r.avg_max_reward,
                "pc_success": r.pc_success,
                "n_episodes": r.n_episodes,
                "avg_episode_length": r.avg_episode_length,
                "avg_final_position_error_m": r.avg_final_position_error_m,
                "avg_final_orientation_error_deg": r.avg_final_orientation_error_deg,
                "avg_cam_looks_at_goal_score": r.avg_cam_looks_at_goal_score,
                "avg_action_delta": r.avg_action_delta,
            }
        )
    return pd.DataFrame(data)


CAMERA_ORDER = ["external", "wrist", "external+wrist"]


def get_metric_label(metric: str) -> str:
    """Convert metric name to a human-readable label."""
    if metric == "pc_success":
        return "% Success"
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
    method_means = df_valid.groupby("method")[metric].mean()
    method_stds = df_valid.groupby("method")[metric].std()
    bars = ax.bar(method_means.index, method_means.values, yerr=method_stds.values, capsize=5)
    ax.set_xlabel("Method")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{metric_label} by Method{title_suffix}")
    for bar, val in zip(bars, method_means.values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + method_stds.max() * 0.1,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=20,
        )
    plt.tight_layout()
    plt.savefig(output_dir / f"{metric}_by_method.png", dpi=150)
    plt.close()

    # 2. Bar plot by dataset (method-normalized)
    df_with_dataset = df_valid[df_valid["dataset"].notna()]
    if not df_with_dataset.empty:
        _, ax = plt.subplots(figsize=(14, 8))
        dataset_means, dataset_stds = compute_method_normalized_stats(df_with_dataset, metric, "dataset")
        if not dataset_means.empty:
            bars = ax.bar(
                range(len(dataset_means)), dataset_means.to_numpy(), yerr=dataset_stds.to_numpy(), capsize=5
            )
            ax.set_xlabel("Dataset")
            ax.set_ylabel(metric_label)
            ax.set_title(f"{metric_label} by Dataset\n(method-normalized){title_suffix}")
            ax.set_xticks(range(len(dataset_means)))
            ax.set_xticklabels(dataset_means.index, rotation=10, ha="right", fontsize=16)
            for bar, val in zip(bars, dataset_means.values, strict=True):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + dataset_stds.max() * 0.1,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=16,
                )
            plt.tight_layout()
            plt.savefig(output_dir / f"{metric}_by_dataset.png", dpi=150)
            plt.close()

    # 3. Bar plot by trajectory generation (method-normalized)
    _, ax = plt.subplots(figsize=(10, 6))
    traj_means, traj_stds = compute_method_normalized_stats(df_valid, metric, "trajectory_gen")
    bars = ax.bar(traj_means.index, traj_means.to_numpy(), yerr=traj_stds.to_numpy(), capsize=5)
    ax.set_xlabel("Trajectory Generation")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{metric_label} by Trajectory Generation\n(method-normalized){title_suffix}")
    for bar, val in zip(bars, traj_means.values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + traj_stds.max() * 0.1,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=20,
        )
    plt.tight_layout()
    plt.savefig(output_dir / f"{metric}_by_trajectory_gen.png", dpi=150)
    plt.close()

    # 4. Bar plot by cameras (method-normalized)
    _, ax = plt.subplots(figsize=(10, 6))
    cam_means, cam_stds = compute_method_normalized_stats(df_valid, metric, "cameras")
    cam_means = cam_means.reindex(CAMERA_ORDER)
    cam_stds = cam_stds.reindex(CAMERA_ORDER)
    bars = ax.bar(cam_means.index, cam_means.to_numpy(), yerr=cam_stds.to_numpy(), capsize=5)
    ax.set_xlabel("Cameras")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{metric_label} by Camera Configuration\n(method-normalized){title_suffix}")
    for bar, val in zip(bars, cam_means.values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + cam_stds.max() * 0.1,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=20,
        )
    plt.tight_layout()
    plt.savefig(output_dir / f"{metric}_by_cameras.png", dpi=150)
    plt.close()

    # 5. Grouped bar plot: method x trajectory_gen
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot = df_valid.pivot_table(values=metric, index="trajectory_gen", columns="method", aggfunc="mean")
    x = np.arange(len(pivot.index))
    width = 0.8 / len(pivot.columns)
    for i, method in enumerate(pivot.columns):
        offset = (i - (len(pivot.columns) - 1) / 2) * width
        values = pivot[method].fillna(0).to_numpy()
        ax.bar(x + offset, values, width, label=method)
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
    plt.tight_layout()
    plt.savefig(output_dir / f"{metric}_by_traj_and_method.png", dpi=150)
    plt.close()

    # 6. Grouped bar plot: method x cameras
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot = df_valid.pivot_table(values=metric, index="cameras", columns="method", aggfunc="mean")
    pivot = pivot.reindex(CAMERA_ORDER)
    x = np.arange(len(pivot.index))
    width = 0.8 / len(pivot.columns)
    for i, method in enumerate(pivot.columns):
        offset = (i - (len(pivot.columns) - 1) / 2) * width
        values = pivot[method].fillna(0).to_numpy()
        ax.bar(x + offset, values, width, label=method)
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
    ax.set_title(f"{metric_label} by Camera Configuration and Method{title_suffix}")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=45)
    ax.legend(title="Method")
    plt.tight_layout()
    plt.savefig(output_dir / f"{metric}_by_cameras_and_method.png", dpi=150)
    plt.close()

    # 7. Line plot: metric over checkpoints by method
    fig, ax = plt.subplots(figsize=(12, 6))
    for method in df_valid["method"].unique():
        method_df = df_valid[df_valid["method"] == method]
        checkpoint_means = method_df.groupby("checkpoint_numeric")[metric].mean().sort_index()
        ax.plot(checkpoint_means.index, checkpoint_means.values, marker="o", label=method)
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{metric_label}\nover Checkpoints by Method{title_suffix}")
    ax.legend()
    ax.set_xscale("log")
    plt.tight_layout()
    plt.savefig(output_dir / f"{metric}_over_checkpoints.png", dpi=150)
    plt.close()


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
    plt.tight_layout()
    plt.savefig(output_dir / f"{metric}_heatmap.png", dpi=150)
    plt.close()


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

    plt.tight_layout()
    plt.savefig(output_dir / "success_rate_comparison.png", dpi=150)
    plt.close()


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

    # Add some headroom for the value labels (15% padding)
    y_limit = max_y_value * 1.15

    # Second pass: create the plots with consistent y-axis and x-axis categories
    for method, grouped in grouped_data.items():
        # Reindex to have consistent categories (missing ones will be NaN)
        grouped = grouped.reindex(index=consistent_cameras, columns=consistent_trajectories)

        # Create the plot
        _, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(grouped.index))
        width = 0.35
        n_traj = len(grouped.columns)

        colors = ["#3498db", "#e74c3c"]  # Blue for 1st RRT, Red for 5th RRT

        for i, traj in enumerate(grouped.columns):
            offset = (i - (n_traj - 1) / 2) * width
            # Replace NaN with 0 for plotting (will show as no bar)
            values = grouped[traj].fillna(0).to_numpy()
            bars = ax.bar(x + offset, values, width, label=traj, color=colors[i % len(colors)])

            # Add value labels on bars (only for non-zero values that were not NaN)
            # Add "N/A" for missing data
            for j, bar in enumerate(bars):
                height = bar.get_height()
                original_value = grouped[traj].iloc[j]
                if pd.isna(original_value):
                    # Show "N/A" at the baseline for missing data
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        0,
                        "N/A",
                        ha="center",
                        va="bottom",
                        fontsize=14,
                        color="gray",
                        style="italic",
                    )
                elif height > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height,
                        f"{height:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=16,
                    )

        ax.set_xlabel("Camera Configuration")
        ax.set_ylabel(metric_label)
        ax.set_title(f"{metric_label} - {method}")
        ax.set_xticks(x)
        ax.set_xticklabels(grouped.index)
        ax.set_ylim(0, y_limit)  # Set consistent y-axis range
        ax.legend(title="Trajectory Gen")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        safe_method = method.replace(".", "").replace(" ", "_")
        plt.savefig(output_dir / f"{metric}_by_camera_traj_{safe_method}.png", dpi=150)
        plt.close()


def generate_summary_table(df: pd.DataFrame, output_dir: Path):
    """Generate a summary CSV and print summary statistics."""
    # Save full results
    df.to_csv(output_dir / "all_results.csv", index=False)

    # Summary by method
    method_agg_dict = {
        "avg_sum_reward": ["mean", "std"],
        "pc_success": ["mean", "std"],
        "avg_episode_length": ["mean", "std"],
        "avg_final_position_error_m": ["mean", "std"],
        "avg_final_orientation_error_deg": ["mean", "std"],
        "avg_cam_looks_at_goal_score": ["mean", "std"],
    }
    # Only include avg_action_delta if any runs have it
    if df["avg_action_delta"].notna().any():
        method_agg_dict["avg_action_delta"] = ["mean", "std"]
    method_summary = df.groupby("method").agg(method_agg_dict).round(3)
    method_summary.to_csv(output_dir / "summary_by_method.csv")

    # Summary by trajectory
    traj_agg_dict = {
        "avg_sum_reward": ["mean", "std"],
        "pc_success": ["mean", "std"],
        "avg_episode_length": ["mean", "std"],
        "avg_final_position_error_m": ["mean", "std"],
        "avg_final_orientation_error_deg": ["mean", "std"],
        "avg_cam_looks_at_goal_score": ["mean", "std"],
    }
    if df["avg_action_delta"].notna().any():
        traj_agg_dict["avg_action_delta"] = ["mean", "std"]
    traj_summary = df.groupby("trajectory_gen").agg(traj_agg_dict).round(3)
    traj_summary.to_csv(output_dir / "summary_by_trajectory.csv")

    # Summary by cameras
    cam_agg_dict = {
        "avg_sum_reward": ["mean", "std"],
        "pc_success": ["mean", "std"],
        "avg_episode_length": ["mean", "std"],
        "avg_final_position_error_m": ["mean", "std"],
        "avg_final_orientation_error_deg": ["mean", "std"],
        "avg_cam_looks_at_goal_score": ["mean", "std"],
    }
    if df["avg_action_delta"].notna().any():
        cam_agg_dict["avg_action_delta"] = ["mean", "std"]
    cam_summary = df.groupby("cameras").agg(cam_agg_dict).round(3)
    cam_summary.to_csv(output_dir / "summary_by_cameras.csv")

    # Summary by dataset
    df_with_dataset = df[df["dataset"].notna()]
    dataset_summary = None
    if not df_with_dataset.empty:
        dataset_agg_dict = {
            "avg_sum_reward": ["mean", "std"],
            "pc_success": ["mean", "std"],
            "avg_episode_length": ["mean", "std"],
            "avg_final_position_error_m": ["mean", "std"],
            "avg_final_orientation_error_deg": ["mean", "std"],
            "avg_cam_looks_at_goal_score": ["mean", "std"],
        }
        if df_with_dataset["avg_action_delta"].notna().any():
            dataset_agg_dict["avg_action_delta"] = ["mean", "std"]
        dataset_summary = df_with_dataset.groupby("dataset").agg(dataset_agg_dict).round(3)
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
    print(f"Camera configurations: {df['cameras'].unique().tolist()}")
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

    # Generate plots for each metric
    print("\nGenerating plots...")

    # avg_sum_reward plots
    plot_metric_by_category(df, "avg_sum_reward", output_dir)
    plot_heatmap(df, "avg_sum_reward", output_dir)

    # avg_episode_length plots
    plot_metric_by_category(df, "avg_episode_length", output_dir)
    plot_heatmap(df, "avg_episode_length", output_dir)

    # avg_final_position_error_m plots
    plot_metric_by_category(df, "avg_final_position_error_m", output_dir)
    plot_heatmap(df, "avg_final_position_error_m", output_dir)

    # avg_final_orientation_error_deg plots
    plot_metric_by_category(df, "avg_final_orientation_error_deg", output_dir)
    plot_heatmap(df, "avg_final_orientation_error_deg", output_dir)

    # avg_cam_looks_at_goal_score plots
    plot_metric_by_category(df, "avg_cam_looks_at_goal_score", output_dir)
    plot_heatmap(df, "avg_cam_looks_at_goal_score", output_dir)

    # avg_action_delta plots (only for entries that have it)
    df_with_action_delta = df[df["avg_action_delta"].notna()]
    if not df_with_action_delta.empty:
        plot_metric_by_category(
            df_with_action_delta,
            "avg_action_delta",
            output_dir,
            title_suffix="\n(subset with action delta metric)",
        )
        plot_heatmap(df_with_action_delta, "avg_action_delta", output_dir)
        plot_metric_by_method(df_with_action_delta, "avg_action_delta", output_dir)
    else:
        print("No entries have avg_action_delta metric")

    # Per-method charts (camera x trajectory grouped bars)
    plot_metric_by_method(df, "avg_episode_length", output_dir)
    plot_metric_by_method(df, "avg_final_position_error_m", output_dir)
    plot_metric_by_method(df, "avg_final_orientation_error_deg", output_dir)
    plot_metric_by_method(df, "avg_cam_looks_at_goal_score", output_dir)

    # pc_success plots
    plot_metric_by_category(df, "pc_success", output_dir)
    plot_heatmap(df, "pc_success", output_dir)

    # Comprehensive success rate comparison
    plot_success_rate_comparison(df, output_dir)

    print(f"\nPlots saved to: {output_dir}")
    print("Generated files:")
    for f in sorted(output_dir.iterdir()):
        print(f"  - {f.name}")

    return 0


if __name__ == "__main__":
    exit(main())
