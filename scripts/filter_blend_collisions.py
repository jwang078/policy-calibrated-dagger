"""Collision-filtered blend dataset producer.

Reads a blend dataset (output of `augment_dataset_with_blending.py`), replays
each episode through a SplatSim env in headless physics-only mode, and writes
out a sibling dataset (`<blend>_nocoll`) containing only the surviving frames:

  * Episodes with NO collision get copied through full-length.
  * Episodes that collide get trimmed to `t_collide - pre_collision_margin`
    frames; if the resulting prefix is shorter than `min_episode_length`,
    the episode is dropped entirely.

Scenario loading reuses the same mechanism as `augment_dataset_with_blending.py`:
each blend episode carries `source_scenario_idx` in its per-episode metadata
(written by the augment script). At the start of each episode we call
`env.reset(seed=[source_scenario_idx])`, which makes the SplatSim server
load that scenario's objects + initial robot pose. No bespoke obstacle code.

SplatSim must be running out-of-process in --headless mode on the port given
by --env_external_port. Example:

    cd ~/code/SplatSim && \
        python scripts/launch_nodes.py \
            --robot sim_ur_pybullet_small_engine_new_interactive \
            --robot_port 6101 \
            --robot_name robot_iphone_w_engine_new \
            --eval_benchmark_repo_id JennyWWW/eval_splatsim_approach_lever_benchmark_1000 \
            --headless

Then:

    python my_scripts/filter_blend_collisions.py \
        --source_repo_id=JennyWWW/lever_grip0_d5jvm_diff_r_dag3_blend090 \
        --target_repo_id=JennyWWW/lever_grip0_d5jvm_diff_r_dag3_blend090_nocoll \
        --env_task=upright_small_engine_new \
        --env_robot_name=robot_iphone_w_engine_new \
        --env_external_port=6101 \
        --env_eval_benchmark_repo_id=JennyWWW/eval_splatsim_approach_lever_benchmark_1000

The output dataset preserves each kept episode's metadata
(`source_episode_idx`, `blend_ratio`, `source_scenario_idx`) plus three new
fields recording the filtering outcome (`pre_filter_n_frames`,
`first_collision_frame`, `trimmed_to_n_frames`).
"""

# NOTE: no `from __future__ import annotations` — draccus reads annotations
# at runtime.

import csv
import faulthandler
import logging
import sys
import time
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

faulthandler.enable(all_threads=True)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image as PILImage  # noqa: E402
from tqdm import tqdm  # noqa: E402

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# `_HERE` (= my_scripts dir) is added to sys.path above, so import sibling
# modules by their bare name. Note: do NOT add a ``my_scripts.`` package
# prefix here — that only works when the lerobot repo root itself is on
# sys.path (e.g. during ``pytest``), and the orchestrator launches this via
# ``python /…/my_scripts/filter_blend_collisions.py`` without that.
from lib_dataset_episode_io import (  # type: ignore[import-not-found]  # noqa: E402
    find_parquet_files,
    load_episode_frames,
)

from lerobot.configs import parser  # noqa: E402
from lerobot.envs.factory import make_env, make_env_config  # noqa: E402
from lerobot.utils.constants import DEFAULT_FEATURES  # noqa: E402
from lerobot.utils.import_utils import register_third_party_plugins  # noqa: E402
from lerobot.utils.lerobot_dataset_utils import resolve_dataset_dir  # noqa: E402
from lerobot.utils.utils import init_logging  # noqa: E402

# DEFAULT_FEATURES (timestamp, frame_index, episode_index, index, task_index)
# are auto-populated by ``LeRobotDataset.add_frame`` and rejected by
# ``validate_frame`` if the caller supplies them. We carry the keys in a frozen
# set so the per-frame copy below can drop them efficiently.
_DEFAULT_FEATURE_KEYS = frozenset(DEFAULT_FEATURES)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class FilterCollisionsConfig:
    # Source = the blend dataset to filter.
    source_repo_id: str = ""
    # Target = where the filtered (and possibly trimmed) episodes are written.
    # Conventionally `<source_repo_id>_nocoll` (see dagger_naming.nocoll_*).
    target_repo_id: str = ""

    # SplatSim connection (must be launched with --headless out-of-process —
    # the orchestrator's start_sim helper spawns this on an auxiliary port).
    env_task: str = "upright_small_engine_new"
    env_robot_name: str = "robot_iphone_w_engine_new"
    env_camera_names: list[str] = field(default_factory=lambda: ["base_rgb", "wrist_rgb"])
    env_image_resize_modes: list[str] = field(default_factory=lambda: ["letterbox", "stretch"])
    env_fps: int = 30
    env_episode_length: int = 1_000_000  # don't truncate; we control the loop
    env_external_port: int = 6101
    env_external_host: str = "127.0.0.1"

    # Scenario-source benchmark for env.reset(seed=...). Must match what was
    # in effect when the source blend dataset was created.
    env_eval_benchmark_repo_id: str = "JennyWWW/eval_splatsim_approach_lever_benchmark_1000"

    # Filter knobs.
    # Drop `pre_collision_margin` frames before the first colliding frame so
    # the policy isn't trained on "near-miss" approaches that ended in a
    # crash. Default 10 ≈ 1/3 sec at 30fps.
    pre_collision_margin: int = 10
    # Mirrors `teleop_min_episode_length` default in
    # src/lerobot/envs/configs.py:566. Episodes whose trimmed length falls
    # below this are dropped entirely.
    min_episode_length: int = 60

    # Where the per-episode CSV summary goes.
    output_dir: str = "outputs/filter_blend_collisions"

    # If True, push the finalized dataset to the Hub at the end.
    push_to_hub: bool = False

    seed: int = 0


@dataclass
class FilterEpisodeResult:
    """One row in the per-episode CSV. Captures the filter's decision."""

    source_episode_idx: int
    target_episode_idx: int | None  # None when dropped
    source_scenario_idx: int | None
    blend_ratio: float | None
    pre_filter_n_frames: int
    first_collision_frame: int | None  # None when no collision
    trimmed_to_n_frames: int
    kept: bool
    drop_reason: str | None
    elapsed_s: float


# ---------------------------------------------------------------------------
# Episode-meta + frame I/O (mirrors augment_dataset_with_blending.py)
# ---------------------------------------------------------------------------


def _load_source_episodes_meta(source_dataset_dir: Path) -> pd.DataFrame:
    """Read source episodes parquet for `source_scenario_idx` + `blend_ratio`."""
    ep_files = sorted((source_dataset_dir / "meta" / "episodes").rglob("*.parquet"))
    if not ep_files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in ep_files], ignore_index=True)


def _episode_length(parquet_files: list[Path], episode_idx: int) -> int:
    """Count frames in a source episode without loading them."""
    n = 0
    for pf in parquet_files:
        df = pd.read_parquet(pf, columns=["episode_index"])
        n += int((df["episode_index"] == episode_idx).sum())
    return n


# ---------------------------------------------------------------------------
# Collision detection — replay one episode through the env and find the first
# colliding frame (or None if the trajectory is clean).
# ---------------------------------------------------------------------------


def _find_first_collision_frame(
    vec_env,
    actions: np.ndarray,
    source_scenario_idx: int,
) -> int | None:
    """Reset env to `source_scenario_idx`, then step through `actions` until
    either the trajectory ends (return None) or `info["in_collision"]` is true
    for the first time (return that frame index).

    Termination handling: SplatSim envs default to ``terminate_on_success=True``,
    so a blend episode that succeeded mid-rollout will trigger ``terminated``
    here. On the NEXT ``step()`` Gymnasium's vec-env auto-resets the sub-env
    to a *different* scenario, and any ``in_collision`` flag from that point
    onward describes the wrong world. We break out of the loop the moment we
    see ``terminated`` or ``truncated`` without an active collision — the
    original episode is done from the env's point of view, anything past it
    is replay noise.
    """
    # env.reset(seed=[N]) → splatsim server loads scenario N's objects + robot.
    vec_env.reset(seed=[int(source_scenario_idx)])
    for t in range(len(actions)):
        # vec_env is a Gymnasium vector env wrapping a single sub-env, so
        # the action tensor needs a leading batch axis. info is also batched.
        action_batched = actions[t : t + 1]
        _obs, _reward, terminated, truncated, info = vec_env.step(action_batched)
        # SplatSim's check_metrics() records per-step collision in info_metrics
        # (see sim_robot_pybullet_small_engine.py). The Gymnasium vector
        # wrapper folds per-env info dicts into "final_info" (terminated/
        # truncated frames) or the parent info dict. Try both.
        if _extract_in_collision_flag(info):
            return t
        # Stop on success / truncation — the next step() would auto-reset
        # the sub-env into a different scenario and any future in_collision
        # flag would be a false positive. terminated/truncated are batched
        # like other vec-env outputs (np.ndarray, shape (1,)).
        if bool(np.asarray(terminated).any()) or bool(np.asarray(truncated).any()):
            return None
    return None


def _extract_in_collision_flag(info: dict[str, Any]) -> bool:
    """Pull `in_collision` from a Gymnasium vec-env info dict.

    Vec-env wraps a single-env's info as either:
      - info["in_collision"] = np.array([bool])        (sync-vec passthrough)
      - info["info_metrics"]["in_collision"] = ...     (splatsim folding)
      - info["final_info"][0]["in_collision"] = ...    (terminated step)
    Robust against all three; missing → returns False.
    """
    if not info:
        return False
    # Direct top-level (sync vec env passthrough).
    if "in_collision" in info:
        v = info["in_collision"]
        return bool(np.asarray(v).any())
    # Nested under info_metrics.
    metrics = info.get("info_metrics")
    if isinstance(metrics, dict) and "in_collision" in metrics:
        return bool(np.asarray(metrics["in_collision"]).any())
    # Final-info on terminated steps.
    final = info.get("final_info")
    if final is not None:
        for sub in final:
            if not sub:
                continue
            if "in_collision" in sub:
                return bool(np.asarray(sub["in_collision"]).any())
            sm = sub.get("info_metrics")
            if isinstance(sm, dict) and "in_collision" in sm:
                return bool(np.asarray(sm["in_collision"]).any())
    return False


# ---------------------------------------------------------------------------
# Trim + drop policy. Encapsulated so it's unit-testable.
# ---------------------------------------------------------------------------


def _row_to_frame(
    row: pd.Series,
    target_features: dict,
    task_for_index: dict[int, str],
) -> dict:
    """Convert one parquet row to a frame dict ``add_frame`` will accept.

    Three transforms vs. the raw parquet row:
      1. Drop columns in ``DEFAULT_FEATURES`` (``timestamp``, ``frame_index``,
         ``episode_index``, ``index``, ``task_index``). ``add_frame`` computes
         these and ``validate_frame`` rejects them as "extra features".
      2. Decode columns with ``dtype == "image"`` from the parquet's
         ``{'bytes': ...}`` encoding into ``PIL.Image`` objects (which
         ``add_frame`` accepts directly per ``validate_feature_image_or_video``).
      3. Look the task string up by ``task_index`` from the source dataset's
         ``meta/tasks.parquet`` mapping. The blend parquet stores
         ``task_index`` but not ``task``; ``add_frame`` requires ``task``.
    """
    out: dict[str, Any] = {}
    for key, spec in target_features.items():
        if key in _DEFAULT_FEATURE_KEYS:
            continue
        value = row[key]
        if spec.get("dtype") == "image" and isinstance(value, dict) and "bytes" in value:
            value = PILImage.open(BytesIO(value["bytes"])).convert("RGB")
        out[key] = value
    # add_frame pops the "task" key separately and resolves it to task_index
    # on its own. Use the source row's task_index to look up the original
    # task string so the new dataset's tasks.parquet stays aligned.
    out["task"] = task_for_index[int(row["task_index"])]
    return out


def _load_task_mapping(source_root_dir: Path) -> dict[int, str]:
    """Load the source dataset's ``task_index → task_string`` map from
    ``meta/tasks.parquet``. Falls back to an empty dict if the file is
    missing — callers can then default the task string at the call site."""
    tasks_path = source_root_dir / "meta" / "tasks.parquet"
    if not tasks_path.is_file():
        return {}
    df = pd.read_parquet(tasks_path)
    # tasks.parquet stores the task STRING as the index and task_index as a
    # column. Invert it for O(1) lookup.
    return {int(ti): str(task) for task, ti in df["task_index"].items()}


def decide_trim(
    n_frames: int,
    first_collision_frame: int | None,
    pre_collision_margin: int,
    min_episode_length: int,
) -> tuple[int, bool, str | None]:
    """Returns `(trimmed_to, kept, drop_reason)` where:
    - trimmed_to: number of leading frames to keep (0 if dropped).
    - kept: True if the episode should be written to the target dataset.
    - drop_reason: short string explaining why kept=False (else None).
    """
    if first_collision_frame is None:
        return n_frames, True, None
    new_len = max(0, first_collision_frame - pre_collision_margin)
    if new_len < min_episode_length:
        return 0, False, f"trimmed_len={new_len} < min={min_episode_length}"
    return new_len, True, None


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _run(
    cfg: FilterCollisionsConfig,
    *,
    csv_path: Path | None = None,
) -> list[FilterEpisodeResult]:
    """Top-level: build env once, iterate episodes, stream survivors to target."""
    from splatsim.utils.lerobot_utils import (
        create_lerobot_dataset,
        finalize_lerobot_dataset,
        load_lerobot_dataset,
    )

    # ── Resolve source dataset on disk ─────────────────────────────────────
    source_data_dir = Path(resolve_dataset_dir(cfg.source_repo_id, None))
    if not source_data_dir.exists():
        raise FileNotFoundError(f"Source data dir does not exist: {source_data_dir}")
    source_root_dir = source_data_dir.parent
    logger.info("Source dataset root: %s", source_root_dir)

    source_episodes_meta = _load_source_episodes_meta(source_root_dir)
    parquet_files = find_parquet_files(source_data_dir)
    # `task_index → task_string` map for converting parquet rows into
    # add_frame-friendly frames. Fall back to a single-entry default keyed by 0
    # using cfg.env_task so single-task datasets without a tasks.parquet still
    # work.
    task_for_index = _load_task_mapping(source_root_dir) or {0: cfg.env_task}

    if "episode_index" not in source_episodes_meta.columns:
        raise ValueError(
            f"Source dataset {cfg.source_repo_id} has no episodes/*.parquet — "
            "expected an augment_dataset_with_blending.py output."
        )
    if "source_scenario_idx" not in source_episodes_meta.columns:
        raise ValueError(
            f"Source dataset {cfg.source_repo_id} has no source_scenario_idx column "
            "in its episodes metadata. The filter needs this to load the right "
            "scenario per episode. Was the source produced by augment_dataset_with_blending.py?"
        )
    available_eps = sorted(int(e) for e in source_episodes_meta["episode_index"].unique())
    logger.info("Source has %d episode(s) to filter.", len(available_eps))

    # ── Connect to externally-launched headless splatsim via ZMQ ───────────
    logger.info(
        "Connecting to headless splatsim ZMQ server at %s:%d …",
        cfg.env_external_host,
        cfg.env_external_port,
    )
    env_cfg_obj = make_env_config(
        "splatsim",
        task=cfg.env_task,
        robot_name=cfg.env_robot_name,
        camera_names=cfg.env_camera_names,
        image_resize_modes=cfg.env_image_resize_modes,
        fps=cfg.env_fps,
        episode_length=cfg.env_episode_length,
        external_port=cfg.env_external_port,
        external_host=cfg.env_external_host,
        eval_benchmark_repo_id=cfg.env_eval_benchmark_repo_id,
        eval_benchmark_subset=None,
        include_oracle_info=False,
    )
    env_dict = make_env(env_cfg_obj, n_envs=1, use_async_envs=False)
    vec_env = env_dict["splatsim"][0]

    # ── Build target dataset (matches source schema) ──────────────────────
    image_keys = [f"{cam}_{mode}" for cam in cfg.env_camera_names for mode in cfg.env_image_resize_modes]
    existing = load_lerobot_dataset(cfg.target_repo_id)
    if existing is not None:
        logger.warning(
            "Target dataset %s already exists locally — appending. Delete the directory for a fresh dataset.",
            cfg.target_repo_id,
        )
        target_ds = existing
    else:
        target_ds = create_lerobot_dataset(
            cfg.target_repo_id,
            fps=cfg.env_fps,
            image_keys=image_keys,
            num_dofs=6,  # Same as augment_dataset_with_blending.py default.
        )

    # ── Per-episode CSV writer ─────────────────────────────────────────────
    csv_writer = None
    csv_file = None
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(csv_path, "w", newline="")  # noqa: SIM115
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [
                "source_episode_idx",
                "target_episode_idx",
                "source_scenario_idx",
                "blend_ratio",
                "pre_filter_n_frames",
                "first_collision_frame",
                "trimmed_to_n_frames",
                "kept",
                "drop_reason",
                "elapsed_s",
            ]
        )
        csv_file.flush()

    results: list[FilterEpisodeResult] = []
    target_ep_idx = int(target_ds.meta.total_episodes)

    try:
        for source_ep in tqdm(available_eps, desc="filter episodes", leave=True):
            t0 = time.time()
            ep_length = _episode_length(parquet_files, source_ep)
            if ep_length <= 0:
                logger.warning("Source episode %d has 0 frames; skipping.", source_ep)
                continue

            # Pull per-episode metadata fields we want to preserve / use.
            ep_meta_row = source_episodes_meta.loc[source_episodes_meta["episode_index"] == source_ep].iloc[0]
            source_scenario_idx = int(ep_meta_row["source_scenario_idx"])
            blend_ratio = (
                float(ep_meta_row["blend_ratio"])
                if "blend_ratio" in source_episodes_meta.columns and pd.notna(ep_meta_row["blend_ratio"])
                else None
            )
            source_episode_idx_meta = (
                int(ep_meta_row["source_episode_idx"])
                if "source_episode_idx" in source_episodes_meta.columns
                and pd.notna(ep_meta_row["source_episode_idx"])
                else int(source_ep)
            )

            # Load full episode frames (we need them BOTH for the action stream
            # to replay AND for the per-frame data to copy to the target).
            frames_df = load_episode_frames(source_data_dir, source_ep, frame_index=0, n_frames=ep_length)
            actions = np.stack(
                [np.asarray(row["action"], dtype=np.float32) for _, row in frames_df.iterrows()]
            )

            # Replay through env, find first collision.
            first_collision = _find_first_collision_frame(vec_env, actions, source_scenario_idx)
            trimmed_to, kept, drop_reason = decide_trim(
                ep_length, first_collision, cfg.pre_collision_margin, cfg.min_episode_length
            )

            if not kept:
                logger.info(
                    "Episode %d DROPPED (first_collision=%s, %s).",
                    source_ep,
                    first_collision,
                    drop_reason,
                )
                results.append(
                    FilterEpisodeResult(
                        source_episode_idx=source_episode_idx_meta,
                        target_episode_idx=None,
                        source_scenario_idx=source_scenario_idx,
                        blend_ratio=blend_ratio,
                        pre_filter_n_frames=ep_length,
                        first_collision_frame=first_collision,
                        trimmed_to_n_frames=0,
                        kept=False,
                        drop_reason=drop_reason,
                        elapsed_s=time.time() - t0,
                    )
                )
                if csv_writer is not None and csv_file is not None:
                    csv_writer.writerow(
                        [
                            source_episode_idx_meta,
                            "",
                            source_scenario_idx,
                            f"{blend_ratio:.4f}" if blend_ratio is not None else "",
                            ep_length,
                            first_collision if first_collision is not None else "",
                            0,
                            False,
                            drop_reason or "",
                            f"{time.time() - t0:.2f}",
                        ]
                    )
                    csv_file.flush()
                continue

            # KEPT: stream the first `trimmed_to` frames to the target dataset.
            # ``_row_to_frame`` drops DEFAULT_FEATURES columns, decodes image
            # bytes from the parquet's ``{'bytes': ...}`` encoding to PIL
            # images, and looks up the task string by task_index — without
            # those transforms, ``add_frame`` rejects the frame.
            for _, row in frames_df.iloc[:trimmed_to].iterrows():
                target_ds.add_frame(_row_to_frame(row, target_ds.meta.features, task_for_index))
            episode_metadata: dict[str, Any] = {
                "source_episode_idx": source_episode_idx_meta,
                "source_scenario_idx": source_scenario_idx,
                "pre_filter_n_frames": int(ep_length),
                "first_collision_frame": (int(first_collision) if first_collision is not None else -1),
                "trimmed_to_n_frames": int(trimmed_to),
            }
            if blend_ratio is not None:
                episode_metadata["blend_ratio"] = float(blend_ratio)
            target_ds.save_episode(episode_metadata=episode_metadata)

            logger.info(
                "Episode %d KEPT (first_collision=%s, trimmed_to=%d/%d).",
                source_ep,
                first_collision,
                trimmed_to,
                ep_length,
            )
            results.append(
                FilterEpisodeResult(
                    source_episode_idx=source_episode_idx_meta,
                    target_episode_idx=target_ep_idx,
                    source_scenario_idx=source_scenario_idx,
                    blend_ratio=blend_ratio,
                    pre_filter_n_frames=ep_length,
                    first_collision_frame=first_collision,
                    trimmed_to_n_frames=trimmed_to,
                    kept=True,
                    drop_reason=None,
                    elapsed_s=time.time() - t0,
                )
            )
            if csv_writer is not None and csv_file is not None:
                csv_writer.writerow(
                    [
                        source_episode_idx_meta,
                        target_ep_idx,
                        source_scenario_idx,
                        f"{blend_ratio:.4f}" if blend_ratio is not None else "",
                        ep_length,
                        first_collision if first_collision is not None else "",
                        trimmed_to,
                        True,
                        "",
                        f"{time.time() - t0:.2f}",
                    ]
                )
                csv_file.flush()
            target_ep_idx += 1
    finally:
        if csv_file is not None:
            csv_file.close()
        try:
            vec_env.close()
        except Exception:
            logger.exception("vec_env.close() raised — ignoring during shutdown.")

    finalize_lerobot_dataset(target_ds)
    # Optional Hub push. The orchestrator currently doesn't enable this for
    # `_nocoll` siblings (they're consumed locally by step 6b's training), but
    # leave the hook wired so standalone invocations can opt in.
    if cfg.push_to_hub:
        try:
            logger.info("Pushing %s to the Hub …", cfg.target_repo_id)
            target_ds.push_to_hub()
            logger.info("Pushed %s successfully.", cfg.target_repo_id)
        except Exception:
            logger.exception(
                "push_to_hub failed for %s — dataset is still saved locally at %s.",
                cfg.target_repo_id,
                target_ds.root,
            )
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@parser.wrap()
def main(cfg: FilterCollisionsConfig) -> None:
    init_logging()
    register_third_party_plugins()

    if not cfg.source_repo_id or not cfg.target_repo_id:
        raise ValueError("--source_repo_id and --target_repo_id are required.")

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_short = cfg.target_repo_id.split("/")[-1]
    csv_path = out_dir / f"{target_short}.csv"
    summary_path = out_dir / f"{target_short}_summary.json"

    results = _run(cfg, csv_path=csv_path)

    n_total = len(results)
    n_kept = sum(1 for r in results if r.kept)
    n_dropped = n_total - n_kept
    n_trimmed = sum(1 for r in results if r.kept and r.first_collision_frame is not None)
    n_full = n_kept - n_trimmed
    total_frames_in = sum(r.pre_filter_n_frames for r in results)
    total_frames_kept = sum(r.trimmed_to_n_frames for r in results if r.kept)

    # JSON summary alongside the per-episode CSV so post-hoc analysis
    # scripts (e.g. dagger_plot, audits) can pick up the aggregate without
    # re-aggregating the per-episode rows. Mirrors what's printed below.
    import json as _json

    summary = {
        "source_repo_id": cfg.source_repo_id,
        "target_repo_id": cfg.target_repo_id,
        "pre_collision_margin": cfg.pre_collision_margin,
        "min_episode_length": cfg.min_episode_length,
        "n_episodes_in": n_total,
        "n_episodes_kept_full": n_full,
        "n_episodes_kept_trimmed": n_trimmed,
        "n_episodes_dropped": n_dropped,
        "total_frames_in": total_frames_in,
        "total_frames_kept": total_frames_kept,
        "per_episode_csv": str(csv_path),
    }
    summary_path.write_text(_json.dumps(summary, indent=2) + "\n")

    print()
    print("=" * 70)
    print(f"Collision-filter summary for {cfg.target_repo_id}")
    print("=" * 70)
    print(f"  source dataset:                 {cfg.source_repo_id}")
    print(f"  pre_collision_margin:           {cfg.pre_collision_margin} frames")
    print(f"  min_episode_length:             {cfg.min_episode_length} frames")
    print(f"  n_episodes_in:                  {n_total}")
    print(f"  n_episodes_kept_full:           {n_full}")
    print(f"  n_episodes_kept_trimmed:        {n_trimmed}")
    print(f"  n_episodes_dropped:             {n_dropped}")
    print(f"  total_frames_in:                {total_frames_in}")
    print(f"  total_frames_kept:              {total_frames_kept}")
    print(f"  per-episode CSV:                {csv_path}")
    print(f"  summary JSON:                   {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
