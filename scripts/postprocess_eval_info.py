#!/usr/bin/env python3
"""
Postprocess eval_info.json files to recompute success, truncated, and in_collision.

New definitions:
  - success: original success=True AND in_collision==0 for that episode
  - in_collision: binarized to 0 or 1 per episode (was previously a step-count sum)
  - truncated: original_truncated OR ended_in_collision (both are non-success terminations
    that cut the episode short of the task goal)

Recomputes all avg_* aggregates from the corrected per-episode values.

Usage:
    python postprocess_eval_info.py <eval_output_folder>

Example:
    python postprocess_eval_info.py outputs/eval_output/2026-03-09-191703
"""

import argparse
import json
from pathlib import Path

import numpy as np

MAX_EPISODE_STEPS = 400


def recompute_task_metrics(metrics: dict) -> dict:
    """Recompute successes and truncated in a per-task metrics dict, and update averages."""
    import copy

    metrics = copy.deepcopy(metrics)

    successes = metrics.get("successes", [])
    info = metrics.get("info_metrics", {})
    in_collision = info.get("in_collision", [])
    episode_length = info.get("episode_length", [])

    n = len(successes)
    orig_truncated = info.get("truncated", [0] * n)

    # Binarize in_collision: was a step-count sum, now 0 or 1 per episode
    new_in_collision = [int(in_collision[i] > 0) if i < len(in_collision) else 0 for i in range(n)]

    # Recompute successes: must be originally true AND no collision steps
    new_successes = [bool(successes[i]) and new_in_collision[i] == 0 for i in range(n)]

    # Recompute truncated: originally truncated OR ended in collision
    new_truncated = [
        int(bool(orig_truncated[i] if i < len(orig_truncated) else 0) or new_in_collision[i])
        for i in range(n)
    ]

    metrics["successes"] = new_successes
    if "info_metrics" in metrics:
        metrics["info_metrics"]["in_collision"] = new_in_collision
        metrics["info_metrics"]["truncated"] = new_truncated

    # Recompute sum_rewards and max_rewards from new successes (sparse reward = success)
    metrics["sum_rewards"] = [1.0 if s else 0.0 for s in new_successes]
    metrics["max_rewards"] = [1.0 if s else 0.0 for s in new_successes]

    # Recompute all avg_* fields
    metrics = _recompute_averages(metrics, new_successes, new_truncated, episode_length)

    return metrics


def _recompute_averages(
    metrics: dict, new_successes: list, new_truncated: list, episode_length: list
) -> dict:
    """Recompute avg_* aggregate fields from per-episode lists."""
    n = len(new_successes)
    if n == 0:
        return metrics

    metrics["avg_sum_reward"] = float(np.mean([1.0 if s else 0.0 for s in new_successes]))
    metrics["avg_max_reward"] = metrics["avg_sum_reward"]
    metrics["pc_success"] = float(np.mean(new_successes)) * 100.0

    info = metrics.get("info_metrics", {})

    for key, values in info.items():
        if len(values) == n:
            metrics[f"avg_{key}"] = float(np.mean(values))

    # Recompute avg_episode_length_without_truncation
    if episode_length and new_truncated:
        non_trunc = [episode_length[i] for i in range(n) if i < len(new_truncated) and not new_truncated[i]]
        metrics["avg_episode_length_without_truncation"] = float(np.mean(non_trunc)) if non_trunc else None

    return metrics


def recompute_overall(per_task: list, original_overall: dict) -> dict:
    """Recompute overall aggregates from all per-task metrics."""
    import copy

    overall = copy.deepcopy(original_overall)

    all_successes = []
    all_episode_lengths = []
    all_truncated = []

    for task in per_task:
        m = task["metrics"]
        all_successes.extend(m.get("successes", []))
        info = m.get("info_metrics", {})
        all_episode_lengths.extend(info.get("episode_length", []))
        all_truncated.extend(info.get("truncated", []))

    n = len(all_successes)
    if n == 0:
        return overall

    overall["avg_sum_reward"] = float(np.mean([1.0 if s else 0.0 for s in all_successes]))
    overall["avg_max_reward"] = overall["avg_sum_reward"]
    overall["pc_success"] = float(np.mean(all_successes)) * 100.0
    overall["n_episodes"] = n

    # Recompute per-info-metric averages from all tasks combined
    all_info: dict[str, list] = {}
    for task in per_task:
        for key, values in task["metrics"].get("info_metrics", {}).items():
            all_info.setdefault(key, []).extend(values)

    for key, values in all_info.items():
        if len(values) == n:
            overall[f"avg_{key}"] = float(np.mean(values))

    # avg_episode_length_without_truncation
    if all_episode_lengths and all_truncated and len(all_episode_lengths) == n:
        non_trunc = [all_episode_lengths[i] for i in range(n) if not all_truncated[i]]
        overall["avg_episode_length_without_truncation"] = float(np.mean(non_trunc)) if non_trunc else None

    return overall


def postprocess_eval_info(eval_info_path: Path) -> None:
    with open(eval_info_path) as f:
        data = json.load(f)

    # Recompute per-task metrics
    for task in data.get("per_task", []):
        task["metrics"] = recompute_task_metrics(task["metrics"])

    # Recompute per-group aggregates
    for group_name, group_agg in data.get("per_group", {}).items():
        # Collect tasks belonging to this group
        group_tasks = [t for t in data["per_task"] if t.get("task_group") == group_name]
        data["per_group"][group_name] = recompute_overall(group_tasks, group_agg)

    # Recompute overall
    if "overall" in data:
        data["overall"] = recompute_overall(data["per_task"], data["overall"])

    out_path = eval_info_path.parent / "eval_info_postprocessed.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_output_folder", type=Path)
    args = parser.parse_args()

    eval_folder = args.eval_output_folder
    eval_info_files = sorted(eval_folder.glob("*/eval_info.json"))

    if not eval_info_files:
        print(f"No eval_info.json files found in {eval_folder}")
        return

    for path in eval_info_files:
        print(f"Processing {path.parent.name}...")
        postprocess_eval_info(path)


if __name__ == "__main__":
    main()
