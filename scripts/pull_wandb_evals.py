#!/usr/bin/env python3
"""
Pull eval metrics from wandb runs and save them in eval_info.json format.

Usage:
    python pull_wandb_evals.py <run_name_or_id> [<run_name_or_id> ...]
    python pull_wandb_evals.py --project lerobot --filter "pi05_approach_lever_*"

Examples:
    python pull_wandb_evals.py pi05_approach_lever_1path_meanstate_quantileaction_base
    python pull_wandb_evals.py v4ftecaf  # by run ID
    python pull_wandb_evals.py --project lerobot --filter "pi05_*"
"""

import argparse
import fnmatch
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import wandb


def get_video_path_for_step(run, step: int) -> str | None:
    """Get the wandb file path for a video at a given step."""
    # Videos are stored as media/videos/eval/video_{step}_*.mp4
    for f in run.files():
        if f.name.startswith(f"media/videos/eval/video_{step}_") and f.name.endswith(".mp4"):
            return f.name
    return None


def download_video(run, video_path: str, output_dir: Path) -> str | None:
    """Download a video from wandb and save it to the output directory.

    Returns the local path to the downloaded video, or None if download failed.
    """
    # Create videos/splatsim_0 subdirectory to match lerobot-eval output structure
    videos_dir = output_dir / "videos" / "splatsim_0"
    videos_dir.mkdir(parents=True, exist_ok=True)

    # Download to a temp directory first, then move to final location
    with tempfile.TemporaryDirectory() as tmpdir:
        for f in run.files():
            if f.name == video_path:
                f.download(root=tmpdir, replace=True)
                downloaded_path = Path(tmpdir) / video_path

                # Rename to eval_episode_0.mp4 (wandb only stores one concatenated video per eval)
                final_path = videos_dir / "eval_episode_0.mp4"
                shutil.move(str(downloaded_path), str(final_path))
                return str(final_path)

    return None


def get_eval_metrics_from_run(run) -> dict[int, dict]:
    """Extract eval metrics for each training step from a wandb run."""
    history = run.history()

    # Find all eval-related columns
    eval_cols = [c for c in history.columns if c.startswith("eval/")]

    # Filter to rows where eval happened (non-null pc_success)
    if "eval/pc_success" not in history.columns:
        return {}

    eval_rows = history[history["eval/pc_success"].notna()]

    metrics_by_step = {}
    for _, row in eval_rows.iterrows():
        step = int(row.get("_step", row.get("eval/steps", 0)))

        metrics = {}
        for col in eval_cols:
            if pd.notna(row.get(col)):
                # Remove 'eval/' or 'eval/eval/' prefix for cleaner keys
                key = col
                if key.startswith("eval/eval/"):
                    key = key[len("eval/eval/") :]
                elif key.startswith("eval/"):
                    key = key[len("eval/") :]
                metrics[key] = row[col]

        if metrics:
            metrics_by_step[step] = metrics

    return metrics_by_step


def create_eval_info_json(
    metrics: dict,
    run_name: str,
    run_id: str,
    project: str,
    step: int,
    output_dir: str,
    video_paths: list[str] | None = None,
) -> dict:
    """Create eval_info.json format from wandb metrics."""
    if video_paths is None:
        video_paths = []

    # Map wandb metric names to eval_info.json format
    metric_mapping = {
        "avg_sum_reward": "avg_sum_reward",
        "avg_max_reward": "avg_max_reward",
        "pc_success": "pc_success",
        "avg_episode_length": "avg_episode_length",
        "avg_position_error_m": "avg_position_error_m",
        "avg_final_position_error_m": "avg_final_position_error_m",
        "avg_orientation_error_deg": "avg_orientation_error_deg",
        "avg_final_orientation_error_deg": "avg_final_orientation_error_deg",
        "avg_cam_looks_at_goal_score": "avg_cam_looks_at_goal_score",
        "avg_final_cam_looks_at_goal_score": "avg_final_cam_looks_at_goal_score",
        "avg_action_delta": "avg_action_delta",
        "avg_final_action_delta": "avg_final_action_delta",
    }

    # Build per_group metrics
    per_group_metrics = {}
    for wandb_key, json_key in metric_mapping.items():
        if wandb_key in metrics:
            per_group_metrics[json_key] = metrics[wandb_key]

    # Add n_episodes if available
    n_episodes = int(metrics.get("episodes", metrics.get("n_episodes", 5)))
    per_group_metrics["n_episodes"] = n_episodes
    per_group_metrics["video_paths"] = video_paths

    # Build overall metrics (same as per_group for single task)
    overall_metrics = per_group_metrics.copy()
    if "eval_s" in metrics:
        overall_metrics["eval_s"] = metrics["eval_s"]
    if "eval_ep_s" in metrics:
        overall_metrics["eval_ep_s"] = metrics["eval_ep_s"]

    eval_info = {
        "per_task": [
            {
                "task_group": "splatsim",
                "task_id": 0,
                "metrics": {
                    "sum_rewards": [],  # Per-episode data not available from wandb
                    "max_rewards": [],
                    "successes": [],
                    "video_paths": video_paths,
                    "info_metrics": {},  # Per-episode metrics not available
                },
            }
        ],
        "per_group": {"splatsim": per_group_metrics},
        "overall": overall_metrics,
        "_wandb_source": {
            "project": project,
            "run_name": run_name,
            "run_id": run_id,
            "step": step,
            "note": "Pulled from wandb API. Per-episode data not available.",
        },
    }

    return eval_info


def pull_run_evals(run, output_base_dir: Path, project: str, download_videos: bool = True):
    """Pull all eval checkpoints from a single wandb run."""
    run_name = run.name
    run_id = run.id
    print(f"\nProcessing run: {run_name} (id: {run_id})")

    metrics_by_step = get_eval_metrics_from_run(run)

    if not metrics_by_step:
        print(f"  No eval metrics found for {run_name}")
        return []

    created_dirs = []
    for step, metrics in sorted(metrics_by_step.items()):
        # Create output directory: {date}-wandb/{run_name}_{step:06d}
        step_str = f"{step:06d}"
        eval_subdir = output_base_dir / f"{run_name}_{step_str}"
        eval_subdir.mkdir(parents=True, exist_ok=True)

        # Download video if requested
        video_paths = []
        if download_videos:
            video_wandb_path = get_video_path_for_step(run, step)
            if video_wandb_path:
                local_video_path = download_video(run, video_wandb_path, eval_subdir)
                if local_video_path:
                    video_paths.append(local_video_path)

        # Create eval_info.json
        eval_info = create_eval_info_json(
            metrics, run_name, run_id, project, step, str(eval_subdir), video_paths
        )

        eval_info_path = eval_subdir / "eval_info.json"
        with open(eval_info_path, "w") as f:
            json.dump(eval_info, f, indent=2)

        video_status = f", video={'yes' if video_paths else 'no'}" if download_videos else ""
        print(
            f"  Step {step}: pc_success={metrics.get('pc_success', 'N/A')}, "
            f"avg_position_error_m={metrics.get('avg_position_error_m', 'N/A'):.4f}{video_status}"
            if "avg_position_error_m" in metrics
            else f"  Step {step}: pc_success={metrics.get('pc_success', 'N/A')}{video_status}"
        )

        created_dirs.append(str(eval_subdir))

    print(f"  Created {len(created_dirs)} eval_info.json files")
    return created_dirs


def main():
    parser = argparse.ArgumentParser(description="Pull eval metrics from wandb runs")
    parser.add_argument("runs", nargs="*", help="Run names or IDs to pull")
    parser.add_argument("--project", default="lerobot", help="Wandb project name")
    parser.add_argument("--entity", default=None, help="Wandb entity (username or team)")
    parser.add_argument(
        "--filter",
        dest="filter_pattern",
        default=None,
        help='Filter runs by name pattern (glob-style, e.g., "pi05_*")',
    )
    parser.add_argument("--output-dir", default="outputs/eval_output", help="Base output directory")
    parser.add_argument(
        "--no-videos", action="store_true", help="Skip downloading videos (faster, smaller output)"
    )
    parser.add_argument("--list", action="store_true", help="List matching runs without pulling")

    args = parser.parse_args()

    api = wandb.Api()

    # Determine which runs to process
    runs_to_process = []

    if args.filter_pattern:
        # Query all runs and filter by name pattern
        project_path = f"{args.entity}/{args.project}" if args.entity else args.project
        all_runs = api.runs(project_path)
        for run in all_runs:
            if fnmatch.fnmatch(run.name, args.filter_pattern):
                runs_to_process.append(run)
        print(f"Found {len(runs_to_process)} runs matching '{args.filter_pattern}'")
    elif args.runs:
        # Fetch specific runs by name or ID
        for run_ref in args.runs:
            try:
                # Try as run ID first
                project_path = f"{args.entity}/{args.project}" if args.entity else args.project
                run = api.run(f"{project_path}/{run_ref}")
                runs_to_process.append(run)
            except wandb.errors.CommError:
                # Try to find by name
                project_path = f"{args.entity}/{args.project}" if args.entity else args.project
                matching = list(api.runs(project_path, {"display_name": run_ref}))
                if matching:
                    runs_to_process.extend(matching)
                else:
                    print(f"Warning: Could not find run '{run_ref}'")
    else:
        parser.print_help()
        return

    if args.list:
        print("\nMatching runs:")
        for run in runs_to_process:
            print(f"  - {run.name} (id: {run.id}, state: {run.state})")
        return

    if not runs_to_process:
        print("No runs to process")
        return

    # Create date-based output directory (overwrites same-day runs)
    date_str = datetime.now().strftime("%Y-%m-%d")
    output_base_dir = Path(args.output_dir) / f"{date_str}-wandb"
    output_base_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput directory: {output_base_dir}")
    print("=" * 60)

    all_created = []
    download_videos = not args.no_videos
    for run in runs_to_process:
        created = pull_run_evals(run, output_base_dir, args.project, download_videos)
        all_created.extend(created)

    print("\n" + "=" * 60)
    print(f"Done! Created {len(all_created)} eval directories in {output_base_dir}")

    # Run summarize_evals.py to generate charts
    summarize_script = Path(__file__).parent / "summarize_evals.py"
    if summarize_script.exists() and all_created:
        print("\nGenerating summary charts...")
        result = subprocess.run(
            [sys.executable, str(summarize_script), str(output_base_dir)], capture_output=False
        )
        if result.returncode != 0:
            print(f"Warning: summarize_evals.py exited with code {result.returncode}")


if __name__ == "__main__":
    main()
