#!/usr/bin/env python3
"""PCA scatter of per-frame data across base / intervention / blend
datasets for a single DAgger round.

Tests the state-distribution-interpolation hypothesis for blended DAgger:
do blend rollouts visit states that lie between pure-policy base
distribution and pure-expert intervention distribution? If yes, the per-
ratio scatter clouds should fill the gap between base and intervention,
with `b010` clouds nearest base and `b090` nearest intervention.

User feeds the script ANY one dataset path from the lineage (base /
intervention / blend / merged); the script resolves every related dataset
for the requested round, fits a single 2D PCA across them all, and
scatter-plots one color per source.

Three plot granularities (`--plot_mode`):
  * `observation` — each point = ONE frame's `data_column` value.
  * `chunk`       — each point = a window of `n_action_steps` consecutive
                    frames concatenated (matches the policy's action-chunk
                    length). Default; matches the original behavior.
  * `trajectory`  — each point = a WHOLE episode's `data_column` trace,
                    edge-padded / truncated to `--trajectory_length` frames
                    and flattened.

The `--data_column` flag picks which parquet column to read (default
`action`; common alternative `observation.state`). Despite the script's
name, the historical default has always been to plot ACTION chunks — pass
`--data_column observation.state` to actually look at the observation
distribution.

Usage (headline):
    python3 my_scripts/dagger_state_coverage_pca.py \\
        --dataset_path=~/.cache/huggingface/lerobot/JennyWWW/lever_grip0_d5jvm_diff_r_dag7_blend010 \\
        --round=6

See `--help` for full option set. `--dry-run` prints the resolved set of
dataset paths + the sidecar that provided base-repo provenance, without
loading data or plotting.
"""

from __future__ import annotations

import argparse
import colorsys
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Share the canonical naming module with the orchestrator + dagger_plot.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dagger_naming import (  # noqa: E402
    enumerate_blend_paths_on_disk,
    find_sidecar_by_prefix,
    int_cache_path,
    load_initial_policy_train_config,
    load_sidecar,
    parse_dataset_short,
    resolve_base_repo,
)

DEFAULT_TRAINING_ROOT = Path.home() / "code" / "lerobot" / "outputs" / "training"
DEFAULT_LEROBOT_CACHE = Path.home() / ".cache" / "huggingface" / "lerobot"
DEFAULT_OUT_DIR = Path.home() / "code" / "lerobot" / "outputs" / "dagger" / "state_coverage"
DEFAULT_HF_USER = "JennyWWW"

PLOT_MODES = ("observation", "chunk", "trajectory")

# Sentinel returned by --round=all. Carried through resolve_inputs +
# downstream as the "round" key; consumers branch on `isinstance(round, str)`
# to know whether to use per-round logic or to enumerate.
ROUND_ALL_SENTINEL = "all"


def _round_arg(s: str) -> int | str:
    """argparse type for --round. Accepts an integer (single round) or
    the literal string 'all' (aggregate over every round on disk)."""
    if s == ROUND_ALL_SENTINEL:
        return ROUND_ALL_SENTINEL
    try:
        return int(s)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"--round must be an integer or 'all', got {s!r}") from e


def _enumerate_lineage_rounds(
    lerobot_cache: Path,
    hf_user: str,
    prefix: str,
    infix: str,
) -> list[int]:
    """Scan the HF cache for every `<prefix>_<infix>_dag<N>` intervention
    dataset on disk. Returns the sorted list of N (ascending). Filters out
    dirs without parquet data so partial recordings don't pollute the set.
    """
    import re as _re

    parent = Path(lerobot_cache) / hf_user
    if not parent.is_dir():
        return []
    pat = _re.compile(rf"^{_re.escape(prefix)}_{_re.escape(infix)}_dag(\d+)$")
    found: list[int] = []
    for d in parent.iterdir():
        m = pat.match(d.name)
        if not m or not d.is_dir():
            continue
        # Skip if no parquet content (partially-formed dataset).
        if next(d.glob("data/chunk-*/file-*.parquet"), None) is None:
            continue
        found.append(int(m.group(1)))
    return sorted(found)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dagger_state_coverage_pca",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dataset_path",
        required=False,
        default=None,
        type=Path,
        help="Any dataset directory in the lineage (base / intervention / blend / merged). "
        "OPTIONAL if --policy_path + --round are given — in that case the lineage's "
        "round-N intervention dataset is auto-resolved from the policy's sidecar.",
    )
    p.add_argument(
        "--round",
        type=_round_arg,
        default=None,
        metavar="N | all",
        help="DAgger round to gather datasets for. Pass an integer N to plot just "
        "round N (default: round parsed from --dataset_path). Pass 'all' to "
        "aggregate intervention + blend data from EVERY round found on disk for "
        "this lineage into a single PCA projection — useful for checking "
        "whether the lineage's data distribution drifts over rounds. Base "
        "dataset is included as usual. Output goes to "
        "`{prefix}_dagAll/...` instead of `{prefix}_dag{N}/...`.",
    )
    p.add_argument(
        "--base_repo_id",
        type=str,
        default=None,
        help="Explicit base-dataset repo id override (e.g. JennyWWW/foo). HIGHEST priority "
        "— skips sidecar / train_config inference entirely.",
    )
    p.add_argument(
        "--plot_mode",
        type=str,
        default="chunk",
        choices=list(PLOT_MODES),
        help="Granularity of each scatter point: 'observation' (one frame), 'chunk' "
        "(n_action_steps consecutive frames concatenated; the default, matches the "
        "original behavior), or 'trajectory' (entire episode concatenated; padded/"
        "truncated to --trajectory_length).",
    )
    p.add_argument(
        "--data_column",
        type=str,
        default="action",
        help="Parquet column to project. Default 'action' (matches historical behavior). "
        "Common alternative: 'observation.state'. Must be a fixed-shape numeric column.",
    )
    p.add_argument(
        "--max_points_per_source",
        type=int,
        default=2000,
        help="Per-source approximate cap on the number of scatter points. A 'point' "
        "is one frame (observation), one non-overlapping chunk (chunk), or one "
        "episode (trajectory). In observation and chunk modes the points are sampled "
        "EPISODE-BY-EPISODE: episodes are picked in randomized order, and each picked "
        "episode is taken IN FULL (no truncation). Sampling stops AFTER the episode "
        "that first reaches the cap — the final episode is never cut short, so the "
        "returned count can slightly exceed `max_points_per_source`. In trajectory "
        "mode it's a simple random sample of episodes. The cap is applied to the "
        "TOTAL per-source point count, NOT per-round: with --round=all the per-round "
        "sample()'s internal cap can multiply across rounds, so this post-concat pass "
        "drops whole episodes randomly until the per-source total honors the cap. "
        "Default 2000.",
    )
    p.add_argument(
        "--n_action_steps",
        type=int,
        default=None,
        help="Chunk length (used only when --plot_mode=chunk and for the trajectory "
        "overlay). Defaults to the base policy's policy.n_action_steps from its "
        "train_config.json. Pass explicitly to override.",
    )
    p.add_argument(
        "--trajectory_length",
        type=int,
        default=200,
        help="Frame count to pad/truncate each episode to (used only when "
        "--plot_mode=trajectory). Shorter episodes are edge-padded (last value "
        "repeated); longer episodes are truncated. Default 200.",
    )
    p.add_argument("--seed", type=int, default=0, help="Subsampling RNG seed (default 0).")
    p.add_argument(
        "--base_highlight_fraction",
        type=float,
        default=0.10,
        help="Fraction of BASE episodes rendered at boosted alpha + bolder arrows "
        "so they read individually through the otherwise-faint base cloud. Rounded "
        "up — e.g. 0.10 of 30 episodes = 3 highlighted. Set to 0 to disable. "
        "Default 0.10.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            f"Output PNG. Default places ALL of this run's PNGs (main plot + "
            f"variants + cluster diagnostics) into a nested per-run subdirectory:\n"
            f"  {DEFAULT_OUT_DIR}/<prefix>_dag<R>/<mode>/coverage_pca.png\n"
            f"so all modes (chunk, observation, trajectory) of the same lineage-round "
            f"group together as siblings under <prefix>_dag<R>/. Pass "
            f"--out=/explicit/path.png to override and put all variants as siblings "
            f"in that path's parent dir (legacy single-dir behavior)."
        ),
    )
    p.add_argument(
        "--training_root",
        type=Path,
        default=DEFAULT_TRAINING_ROOT,
        help="Override outputs/training/ for sidecar search.",
    )
    p.add_argument(
        "--hf_user",
        type=str,
        default=DEFAULT_HF_USER,
        help=f"HF user prefix for cache paths (default {DEFAULT_HF_USER}).",
    )
    p.add_argument(
        "--lerobot_cache",
        type=Path,
        default=DEFAULT_LEROBOT_CACHE,
        help="Override the LeRobot cache root.",
    )
    p.add_argument(
        "--legacy_naming",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When sidecar lookup by the dataset's full prefix fails, retry with a "
        "trailing blends-tag (`_b<NNN>(_<NNN>)*`) stripped. Lets the script handle "
        "older rerun-blends dataset names whose prefix embeds the blends-tag but the "
        "sidecar only stores the source-lineage prefix. Default ON (best-effort "
        "fallback, warns when used). Pass --no-legacy_naming for strict matching.",
    )
    p.add_argument(
        "--plot_normalized",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In addition to the raw-value PCA plot, emit a SECOND plot in the "
        "policy's normalized space — observation.state passes through the "
        "policy's preprocessor (Normalizer), action passes through the inverse "
        "direction of the postprocessor (= Unnormalizer used as Normalizer). "
        "Requires a resolvable policy path; if none is found and "
        "--policy_path isn't passed, the normalized plot is skipped with a "
        "warning. Default ON. Pass --no-plot_normalized to skip.",
    )
    p.add_argument(
        "--plot_without_base",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In addition to the with-base plots, emit a sibling plot that EXCLUDES "
        "the base dataset from BOTH the PCA fit AND the scatter — so PCA axes "
        "are determined entirely by intervention+blends. Useful for spotting "
        "blend-vs-intervention structure that's otherwise crushed when the "
        "much larger base cloud dominates the PCA fit. Output filenames get a "
        "`_nobase` suffix. Combined with --plot_normalized this emits up to 4 "
        "plots per run. Default ON. Pass --no-plot_without_base to skip.",
    )
    p.add_argument(
        "--plot_only_base",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emit a sibling plot showing ONLY the base dataset — drops "
        "intervention + all blends from both the PCA fit and the scatter. "
        "Useful for diagnosing whether multimodality / fan-out structure in "
        "the full plot was already present in the base demos (i.e. predates "
        "DAgger) vs introduced by interventions. Output filenames get an "
        "`_onlybase` suffix. Default ON. Pass --no-plot_only_base to skip.",
    )
    p.add_argument(
        "--policy_path",
        type=Path,
        default=None,
        help="Override the auto-resolved acting policy path (the directory "
        "containing policy_preprocessor.json + policy_postprocessor.json). "
        "Used only by --plot_normalized. Auto-resolution: finds a sidecar in "
        "--training_root whose lineage matches --dataset_path and whose "
        "training dir ends with _ft_dag<round>, then uses that sidecar's "
        "rerun_mode.branching_policy_path (or initial_policy_path for round 1).",
    )
    p.add_argument(
        "--episodes",
        type=str,
        default=None,
        metavar="SPEC",
        help="Filter scatter points to specific episode indices, per source. "
        "Accepts: a single int (`4`), comma list (`4,6,8`), range (`2-14`), or "
        "any combination (`0-2,5,7-9`). Zero-indexed. The same indices are "
        "applied to EVERY source — intervention, each blend, AND base. For "
        "intervention/blend the indices refer to the round's recorded episodes "
        "(intervention episode N and blend episode N are the same scenario+ "
        "cycle); for base the indices refer to whatever episode N is in the "
        "base demo dataset (NOT related to intervention's scenario). Combine "
        "with --no-plot_without_base to drop base entirely. Default: no filter.",
    )
    p.add_argument(
        "--videos_per_cluster",
        type=int,
        default=0,
        metavar="N",
        help="With --cluster_endpoints=K, sample up to N representative "
        "episodes PER SOURCE per cluster (base, intervention, b010, b050, "
        "b090, etc — whatever's on disk) and concatenate them into ONE "
        "composite mp4 PER CLUSTER, showing each sample for the last "
        "--cluster_video_trim_seconds of its episode (default 2s, where the "
        "cluster-defining behavior actually happens). Between samples a "
        "brief title card announces the next (source, ep_id) so the cuts "
        "aren't disorienting. The composite shows base | wrist side-by-side "
        "per frame. Sampling is deterministic on --seed; sources with fewer "
        "than N episodes contribute all they have; sources with ZERO are "
        "silently skipped. Output: "
        "`<run_subdir>/<variant>_cluster_videos/cluster<i>_composite.mp4` "
        "(one mp4 per cluster instead of one per episode — easier to scrub "
        "through). Pair with the companion `_cluster_collages/` PNGs (still "
        "frames of each sample's endpoint) to first pick which sample looks "
        "interesting, then watch its segment in the composite. Default 0 "
        "(no videos).",
    )
    p.add_argument(
        "--cluster_video_trim_seconds",
        type=float,
        default=2.0,
        metavar="S",
        help="With --videos_per_cluster=N: trim each sampled episode to its "
        "LAST S seconds before concatenating into the per-cluster composite "
        "mp4. The clustering is on the END of each episode (last action "
        "chunk's PCA projection), so the cluster-defining behavior is at "
        "the trajectory's end — showing the full episode is mostly "
        "irrelevant approach motion. Default 2.0 (60 frames at fps=30). "
        "Set to 0 to disable trimming and show full episodes.",
    )
    p.add_argument(
        "--cluster_endpoints",
        type=int,
        default=None,
        metavar="K",
        help="When set to K, run k-means with K clusters on the per-episode "
        "endpoints (last chunk's PCA projection per episode, across all "
        "rendered sources) and emit ONE additional diagnostic PNG per plot "
        "variant: `<base>_endpoint_clustersK.png`. Endpoints colored by "
        "cluster, centroids marked with `×`. A per-source breakdown of how "
        "many episodes per source reached each cluster is also printed to "
        "stdout — useful for tasks where the policy can reach multiple "
        "valid end states (e.g. left vs right IK branches) and you want "
        "to know (a) how many distinct end states actually exist and (b) "
        "which sources prefer which. Skipped in trajectory mode (each "
        "scatter point already IS an episode there, so 'endpoint' isn't "
        "meaningful).",
    )
    p.add_argument(
        "--drop_outliers_above",
        type=float,
        default=None,
        metavar="THRESHOLD",
        help="If set, drop any scatter point in the policy-normalized plots "
        "(NOT the raw plots) whose absolute value in any feature dim exceeds "
        "THRESHOLD. Useful for filtering out chunks where the policy issued "
        "OOD actions — e.g. b090 chunks reaching ±5 in normalized space when "
        "you want to see the in-distribution cluster structure. A natural "
        "threshold is `1.5` (= 1.5× the training MIN_MAX range of [-1, 1]). "
        "Per-source dropped counts are logged. Default: off (no filtering).",
    )
    p.add_argument(
        "--force_gripper",
        type=float,
        default=None,
        metavar="VALUE",
        help="If set, overwrite the LAST dimension of every loaded `action` "
        "row with VALUE before any further processing (chunking, rel-action "
        "conversion, normalization, PCA). Useful when --data_column=action "
        "and the dataset's gripper column is contaminated (e.g. blend "
        "recordings where the rel/abs roundtrip leaked gripper STATE into "
        "the action column when `action_names` was None at training time). "
        "Pass `--force_gripper 0` to clamp gripper-action to zero everywhere. "
        "Only the action column is touched — observation.state is left alone. "
        "Applies to BOTH the raw and the policy-normalized plots. Default: "
        "no override (use the dataset's recorded value).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved dataset paths + sidecar provenance, then exit.",
    )
    return p


_LEGACY_BLENDS_TAG_RE = re.compile(r"_b\d{3}(?:_\d{3})*$")
_ROUND_DAG_RE = re.compile(r"_(?:ft_)?dag(\d+)(?:_|$)")


def _parse_episodes_spec(spec: str) -> set[int]:
    """Parse `--episodes` value into a set of zero-indexed integers.

    Accepts comma-separated tokens, each of which is either:
      * A single integer: `4`
      * A range `LO-HI` (inclusive): `2-5` → {2, 3, 4, 5}
    The combinations `0-2,5,7-9` → {0, 1, 2, 5, 7, 8, 9}. Whitespace tolerated.
    Raises SystemExit on malformed input so the user sees the problem early.
    """
    out: set[int] = set()
    for raw in spec.split(","):
        tok = raw.strip()
        if not tok:
            continue
        if "-" in tok:
            try:
                lo_s, hi_s = tok.split("-", 1)
                lo, hi = int(lo_s), int(hi_s)
            except ValueError:
                raise SystemExit(f"ERROR: bad --episodes range {tok!r}")
            if lo > hi:
                raise SystemExit(f"ERROR: --episodes range {tok!r} has lo > hi; use {hi}-{lo} instead.")
            out.update(range(lo, hi + 1))
        else:
            try:
                out.add(int(tok))
            except ValueError:
                raise SystemExit(f"ERROR: bad --episodes token {tok!r}")
    return out


def _find_acting_policy_path(
    training_root: Path,
    prefix: str,
    round_n: int,
    *,
    allow_legacy: bool,
) -> tuple[Path | None, str]:
    """Locate the policy that ACTED to produce round-N's recordings.

    The semantics depend on what `prefix` is:
      * For a non-rerun lineage, that's the lineage's own ft_dag(round-1)
        checkpoint (or initial policy for round 1).
      * For a rerun lineage, it's the SOURCE lineage's ft_dag(round-1) — which
        every sibling rerun stores under `rerun_mode.branching_policy_path` of
        its OWN round-`round_n` sidecar.

    Resolution:
      1. Look for ANY sidecar whose training dir ends in `_ft_dag<round_n>` or
         `_dag<round_n>` AND whose lineage prefix matches `prefix` (strict),
         or — if `allow_legacy` — matches `prefix` with a trailing
         `_b\\d{{3}}(_\\d{{3}})*` blends-tag stripped (source-lineage match).
      2. Return that sidecar's `rerun_mode.branching_policy_path` (resolved
         to absolute path). Falls back to `config.initial_policy_path` when
         branching is empty (typical for round 1 of a non-rerun lineage).
      3. Returns (None, reason) if no match — caller skips the normalized plot.
    """
    candidate_prefixes = [prefix]
    if allow_legacy:
        stripped = _LEGACY_BLENDS_TAG_RE.sub("", prefix)
        if stripped != prefix:
            candidate_prefixes.append(stripped)

    if not training_root.is_dir():
        return None, f"training_root {training_root} not found"

    for sc_path in sorted(training_root.glob("*/dagger/config.json")):
        train_dir_name = sc_path.parent.parent.name
        m = _ROUND_DAG_RE.search(train_dir_name)
        if not m or int(m.group(1)) != round_n:
            continue
        try:
            sc = load_sidecar(sc_path)
        except Exception:
            continue
        naming = sc.get("naming") or {}
        rerun = sc.get("rerun_mode") or {}
        base_short = naming.get("base_dataset_short") or ""
        src_int = rerun.get("source_int_short_prefix") or ""
        if not any(p in (base_short, src_int) for p in candidate_prefixes):
            continue
        # Pick the acting policy in this order of preference:
        #   (a) rerun_mode.branching_policy_path — exact pointer (rerun mode).
        #   (b) For non-rerun N >= 2: sibling training dir _ft_dag(N-1)
        #       (same lineage, prior round's checkpoint).
        #   (c) For round 1: config.initial_policy_path (the base policy that
        #       was finetuned to make ft_dag1).
        # WARNING: sidecar's `config.initial_policy_path` is always the BASE
        # (round-0) policy, NOT the prior round — so it's only correct for
        # round 1. For N >= 2, we MUST find the sibling _ft_dag(N-1) dir.
        branching = rerun.get("branching_policy_path")
        if branching:
            bp = Path(branching).expanduser()
            if not bp.is_absolute():
                bp = Path.cwd() / branching
            if bp.is_dir():
                return bp.resolve(), f"branching_policy_path from {train_dir_name}"

        # (b) Non-rerun N >= 2: derive prior round's training dir name by
        # replacing `_ft_dag<N>` (or `_dag<N>`) with `_ft_dag<N-1>` / `_dag<N-1>`.
        if round_n >= 2:
            prior_name = re.sub(
                rf"_ft_dag{round_n}(?=_|$)",
                f"_ft_dag{round_n - 1}",
                train_dir_name,
            )
            if prior_name == train_dir_name:
                prior_name = re.sub(
                    rf"(?<!ft)_dag{round_n}(?=_|$)",
                    f"_dag{round_n - 1}",
                    train_dir_name,
                )
            if prior_name != train_dir_name:
                prior_path = Path(training_root) / prior_name
                for cand in (
                    prior_path / "checkpoints" / "last" / "pretrained_model",
                    prior_path / "pretrained_model",
                ):
                    if cand.is_dir():
                        return (
                            cand.resolve(),
                            f"prior round dir {prior_name}/checkpoints/last/pretrained_model",
                        )

        # (c) Round 1 only: initial_policy_path is the acting (base) policy.
        if round_n == 1:
            ip = (sc.get("config") or {}).get("initial_policy_path")
            if ip:
                ip_p = Path(ip).expanduser()
                if not ip_p.is_absolute():
                    ip_p = Path.cwd() / ip
                for cand in (
                    ip_p / "checkpoints" / "last" / "pretrained_model",
                    ip_p / "pretrained_model",
                    ip_p,
                ):
                    if cand.is_dir():
                        return cand.resolve(), f"initial_policy_path from {train_dir_name}"
    return None, f"no matching ft_dag{round_n} sidecar for prefix {prefix!r}"


def _load_action_names_from_dataset(dataset_path: Path) -> list[str] | None:
    """Read `features.action.names` from <dataset>/meta/info.json.

    Used to backfill `RelativeActionsProcessorStep.action_names` on policies
    that were saved with `action_names=null` (the field is set on PI0/PI0.5/
    PI0FAST configs but NOT on DiffusionConfig, so trained diffusion policies
    persist null and the `exclude_joints=['gripper']` mask silently degrades
    to all-True). Returns the names list or None if the file or field is missing.
    """
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.is_file():
        return None
    try:
        info = json.loads(info_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    names = (info.get("features", {}).get("action", {}) or {}).get("names")
    return list(names) if names else None


def _load_policy_processor_steps(policy_path: Path, base_dataset_path: Path | None = None) -> dict:
    """Load the policy's pre/postprocessor pipelines and return:

        {
          "preprocessor":          full PolicyProcessorPipeline (preprocessor)
          "postprocessor":         full PolicyProcessorPipeline (postprocessor)
          "norm_step":             NormalizerProcessorStep from preprocessor  | None
          "unnorm_step":           UnnormalizerProcessorStep from postprocessor | None
          "abs_step":              AbsoluteActionsProcessorStep from postprocessor | None
                                   (carries `.relative_step` reference + `.enabled` gate)
        }

    The full pipelines are returned so callers can construct an
    `inverse_postprocessor` pipeline that mirrors `_wrap_with_shared_autonomy`'s
    recipe (NormalizerProcessorStep using the POSTPROCESSOR's stats +
    `zero_variance_denom=2.0` for zero-variance dim safety, then DeviceProcessorStep).

    Lazy import — keeps the absolute-PCA path free of heavy torch + lerobot
    imports for users who only want the raw plot.
    """
    from lerobot.policies.factory import (
        batch_to_transition,
        policy_action_to_transition,
        transition_to_batch,
        transition_to_policy_action,
    )
    from lerobot.processor.normalize_processor import (
        NormalizerProcessorStep,
        UnnormalizerProcessorStep,
    )
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.relative_action_processor import AbsoluteActionsProcessorStep
    from lerobot.utils.constants import (
        POLICY_POSTPROCESSOR_DEFAULT_NAME,
        POLICY_PREPROCESSOR_DEFAULT_NAME,
    )

    preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=str(policy_path),
        config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=str(policy_path),
        config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    # Re-wire AbsoluteActionsProcessorStep.relative_step → RelativeActionsProcessorStep
    # after deserialization (the reference is not serializable). Without this,
    # `abs_step.relative_step` is None and the SA-wrapper-style gate fails to
    # trigger the rel-conversion path. lerobot_eval / training do the same.
    from lerobot.policies.factory import _reconnect_relative_absolute_steps

    _reconnect_relative_absolute_steps(preprocessor, postprocessor)

    # Backfill RelativeActionsProcessorStep.action_names from the BASE
    # dataset's meta/info.json if it's missing.
    #
    # WHY: training-time `make_policy` populates `cfg.action_feature_names`
    # from dataset metadata, but only if the policy config class declares the
    # field via `hasattr`. PI0/PI0.5/PI0FAST configs declare it; DiffusionConfig
    # does NOT, so diffusion policies save `action_names: null` in their
    # preprocessor JSON. On reload the rel-step's `_build_mask` then falls
    # back to `[True] * action_dim` — exclude_joints=['gripper'] silently
    # becomes a no-op, gripper gets converted to rel (`0 - state ≈ -0.4`),
    # then normalized against its zero-variance stats → values blow up to
    # ~1e8 and dominate the PCA fit.
    from lerobot.processor.relative_action_processor import RelativeActionsProcessorStep

    rel_step = next(
        (s for s in preprocessor.steps if isinstance(s, RelativeActionsProcessorStep)),
        None,
    )
    if (
        rel_step is not None
        and rel_step.action_names is None
        and rel_step.exclude_joints
        and base_dataset_path is not None
    ):
        names = _load_action_names_from_dataset(base_dataset_path)
        if names is not None:
            rel_step.action_names = names
            print(
                f"[plot_normalized] backfilled rel-step.action_names from "
                f"{base_dataset_path / 'meta' / 'info.json'}: {names}  "
                f"(exclude_joints={rel_step.exclude_joints} now actually applies)"
            )
        else:
            print(
                f"[plot_normalized] WARNING: rel-step has exclude_joints="
                f"{rel_step.exclude_joints} but action_names=None and could "
                f"not be loaded from {base_dataset_path}/meta/info.json — "
                f"gripper exclusion will be ignored (all dims converted to rel)."
            )

    return {
        "preprocessor": preprocessor,
        "postprocessor": postprocessor,
        "norm_step": next(
            (s for s in preprocessor.steps if isinstance(s, NormalizerProcessorStep)),
            None,
        ),
        "unnorm_step": next(
            (s for s in postprocessor.steps if isinstance(s, UnnormalizerProcessorStep)),
            None,
        ),
        "abs_step": next(
            (s for s in postprocessor.steps if isinstance(s, AbsoluteActionsProcessorStep)),
            None,
        ),
    }


def _build_inverse_postprocessor(unnorm_step, device: str = "cpu"):
    """Construct an `inverse_postprocessor` pipeline that mirrors
    `_wrap_with_shared_autonomy`'s recipe in `lerobot_eval.py`:

        NormalizerProcessorStep(stats=postprocessor_stats,
                                features=postprocessor_features,
                                norm_map=postprocessor_norm_map,
                                zero_variance_denom=2.0)
        → DeviceProcessorStep

    `zero_variance_denom=2.0` keeps zero-variance dims (e.g. gripper always 0)
    finite — without it the QUANTILES branch would divide by ~eps and blow up.
    For MIN_MAX (most policies) this parameter doesn't apply, but matching the
    SA wrapper's construction is the canonical way to put a "raw human guidance
    action" into the policy's internal normalized space.
    """
    from lerobot.policies.factory import (
        policy_action_to_transition,
        transition_to_policy_action,
    )
    from lerobot.processor.device_processor import DeviceProcessorStep
    from lerobot.processor.normalize_processor import NormalizerProcessorStep
    from lerobot.processor.pipeline import PolicyProcessorPipeline

    return PolicyProcessorPipeline(
        steps=[
            NormalizerProcessorStep(
                features=unnorm_step.features,
                norm_map=unnorm_step.norm_map,
                stats=unnorm_step.stats,
                zero_variance_denom=2.0,
            ),
            DeviceProcessorStep(device=device),
        ],
        name="inverse_postprocessor",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )


def _make_value_transform(data_column: str, policy_path: Path, base_dataset_path: Path | None = None):
    """Build a per-frame value transform that puts `data_column` into the
    SAME space the policy operates on internally — mirroring exactly what
    `SharedAutonomyPolicyWrapper._normalize_policy_guidance_action` does:

      * `observation.state` → preprocessor.NormalizerProcessorStep (forward).
        Matches what the policy sees as input at training time.

      * `action` → if the postprocessor's AbsoluteActionsProcessorStep is
        ENABLED and has a paired `relative_step` (i.e. policy was trained
        with action_format=rel), first do
            `rel = to_relative_actions(action, obs_state, mask)`
        using the same mask the policy's training pipeline uses, then run
        through an `inverse_postprocessor` pipeline constructed the same way
        `_wrap_with_shared_autonomy` constructs it (NormalizerProcessorStep
        with postprocessor's stats + `zero_variance_denom=2.0` for
        zero-variance dim safety + DeviceProcessorStep).
        If the abs_step is disabled, just run through `inverse_postprocessor`.

    Returns `(transform_fn, note_string, needs_obs_state)` where:
      * `transform_fn(values, obs_state=None)` takes [N, D] float32 and returns
        [N, D] float32. `obs_state` is required when `needs_obs_state` is True.
      * `needs_obs_state` is True iff the abs_step is enabled (rel-conversion
        needs paired state).

    Raises if the requested column is neither observation.state nor action,
    or if the required step is missing from the loaded pipeline.
    """
    import torch

    from lerobot.configs.types import FeatureType

    steps = _load_policy_processor_steps(policy_path, base_dataset_path=base_dataset_path)
    norm_step = steps["norm_step"]
    unnorm_step = steps["unnorm_step"]
    abs_step = steps["abs_step"]

    if data_column == "observation.state":
        if norm_step is None:
            raise SystemExit(f"ERROR: no NormalizerProcessorStep found in preprocessor at {policy_path}")
        if "observation.state" not in getattr(norm_step, "_tensor_stats", {}):
            raise SystemExit(
                f"ERROR: preprocessor has no stats for 'observation.state' (available: "
                f"{list(norm_step._tensor_stats.keys())})"
            )

        def _tx(values: np.ndarray, obs_state: np.ndarray | None = None) -> np.ndarray:
            t = torch.as_tensor(values, dtype=torch.float32)
            out = norm_step._apply_transform(t, "observation.state", FeatureType.STATE, inverse=False)
            return out.detach().cpu().numpy().astype(np.float32)

        return _tx, "preprocessor.NormalizerProcessorStep (forward)", False

    if data_column == "action":
        if unnorm_step is None:
            raise SystemExit(f"ERROR: no UnnormalizerProcessorStep found in postprocessor at {policy_path}")

        # Build the SA-wrapper-style inverse_postprocessor pipeline.
        inv_pp = _build_inverse_postprocessor(unnorm_step, device="cpu")

        # Rel-conversion gate: postprocessor's AbsoluteActionsProcessorStep.
        # Mirrors SA wrapper's check at shared_autonomy_wrapper.py:730-737.
        abs_enabled = (
            abs_step is not None
            and bool(getattr(abs_step, "enabled", False))
            and getattr(abs_step, "relative_step", None) is not None
        )
        rel_step = abs_step.relative_step if abs_enabled else None

        from lerobot.processor.relative_action_processor import to_relative_actions

        def _tx(values: np.ndarray, obs_state: np.ndarray | None = None) -> np.ndarray:
            t_action = torch.as_tensor(values, dtype=torch.float32)
            if abs_enabled:
                if obs_state is None:
                    raise SystemExit(
                        "ERROR: action normalization needs paired observation.state "
                        "(postprocessor has an enabled AbsoluteActionsProcessorStep "
                        "→ policy uses relative actions). Internal plumbing bug."
                    )
                if obs_state.shape[0] != values.shape[0]:
                    raise SystemExit(
                        f"ERROR: action ({values.shape[0]} rows) vs obs_state "
                        f"({obs_state.shape[0]} rows) length mismatch."
                    )
                t_state = torch.as_tensor(obs_state, dtype=torch.float32)
                mask = rel_step._build_mask(t_action.shape[-1])
                print("observation.state", t_state[-1])
                print("action in abs", t_action[-1])
                t_action = to_relative_actions(t_action, t_state, mask)
                print("action in relative", t_action[-1])
            # Now run through the inverse_postprocessor pipeline — same call
            # shape as SA wrapper: `inverse_postprocessor(tensor) -> tensor`.
            out = inv_pp(t_action)
            print("action normalized", out[-1])
            return out.detach().cpu().numpy().astype(np.float32)

        if abs_enabled:
            note = (
                "to_relative_actions(action, obs_state, "
                f"mask=from postprocessor.AbsoluteActionsProcessorStep.relative_step "
                f"[exclude_joints={getattr(rel_step, 'exclude_joints', [])}]) "
                "→ inverse_postprocessor pipeline "
                "(NormalizerProcessorStep zero_variance_denom=2.0 + DeviceProcessorStep)"
            )
        else:
            note = (
                "inverse_postprocessor pipeline "
                "(NormalizerProcessorStep zero_variance_denom=2.0 + DeviceProcessorStep) "
                "[no rel-conversion: postprocessor.AbsoluteActionsProcessorStep disabled]"
            )
        return _tx, note, abs_enabled

    raise SystemExit(
        f"ERROR: --plot_normalized only supports --data_column in "
        f"{{observation.state, action}}; got {data_column!r}"
    )


def _find_sidecar_with_legacy_fallback(
    training_root: Path, prefix: str, *, allow_legacy: bool
) -> tuple[Path | None, str | None]:
    """Strict prefix match first. On miss, if `allow_legacy`, strip a trailing
    `_b<NNN>(_<NNN>)*` blends-tag and retry. Returns
    (sidecar_path_or_None, stripped_prefix_used_or_None). `stripped_prefix` is
    non-None only when the legacy fallback fired (whether or not it found a
    match), so the caller can log it.
    """
    sc = find_sidecar_by_prefix(training_root, prefix)
    if sc is not None:
        return sc, None
    if not allow_legacy:
        return None, None
    stripped = _LEGACY_BLENDS_TAG_RE.sub("", prefix)
    if stripped == prefix:
        return None, None
    sc = find_sidecar_by_prefix(training_root, stripped)
    return sc, stripped


def _resolve_to_pretrained_model_dir(p: Path) -> Path:
    """Return a directory containing `policy_preprocessor.json` given either:
      (a) a path that already IS a pretrained_model dir (has the JSON), OR
      (b) a training output dir (has `checkpoints/{last,<N>}/pretrained_model/...`).

    Mirrors the candidate order in
    `dagger_naming.load_initial_policy_train_config` and the orchestrator's
    `resolve_latest_checkpoint` so behavior is consistent with the rest of
    the lineage tooling.
    """
    p = p.resolve()
    if (p / "policy_preprocessor.json").is_file():
        return p
    candidates: list[Path] = []
    last_pm = p / "checkpoints" / "last" / "pretrained_model"
    if last_pm.is_dir():
        candidates.append(last_pm)
    ckpt_root = p / "checkpoints"
    if ckpt_root.is_dir():
        numbered = sorted(
            (c for c in ckpt_root.iterdir() if c.is_dir() and c.name.isdigit()),
            key=lambda x: int(x.name),
            reverse=True,
        )
        for c in numbered:
            pm = c / "pretrained_model"
            if pm.is_dir():
                candidates.append(pm)
    direct = p / "pretrained_model"
    if direct.is_dir():
        candidates.append(direct)
    for c in candidates:
        if (c / "policy_preprocessor.json").is_file():
            return c
    raise SystemExit(
        f"ERROR: --policy_path={p} doesn't contain policy_preprocessor.json, and "
        f"no checkpoints/{{last,<N>}}/pretrained_model subdir under it has one either.\n"
        f"  Tried: {[str(c) for c in candidates] or '(no checkpoints/ dir found)'}"
    )


def _autoderive_dataset_path_from_policy(
    policy_path: Path, round_n: int, hf_user: str, lerobot_cache: Path
) -> Path:
    """If the user passed `--policy_path` but no `--dataset_path`, derive the
    round-N intervention dataset path from the policy's sidecar.

    The sidecar's `naming.base_dataset_short` gives the lineage prefix (or
    `rerun_mode.source_int_short_prefix` in rerun-blends mode where the
    intervention datasets live under SOURCE's prefix, not the rerun's own).
    `config.action_format` gives the 'r'/'a' infix. Combined with `round_n`
    those identify the on-disk intervention dataset:
        <lerobot_cache>/<hf_user>/<prefix>_<infix>_dag<round_n>
    """
    if not policy_path.is_dir():
        raise SystemExit(
            f"ERROR: --policy_path={policy_path} is not a directory "
            f"(must be a training-output dir like outputs/training/..._ft_dag<N>)."
        )
    sidecar_path = policy_path / "dagger" / "config.json"
    if not sidecar_path.is_file():
        raise SystemExit(
            f"ERROR: --policy_path={policy_path} has no dagger/config.json sidecar.\n"
            f"  Cannot auto-derive --dataset_path. Pass --dataset_path=<path> directly."
        )
    sc = load_sidecar(sidecar_path)
    naming = sc.get("naming") or {}
    rerun = sc.get("rerun_mode") or {}
    cfg = sc.get("config") or {}

    prefix = rerun.get("source_int_short_prefix") if rerun else naming.get("base_dataset_short")
    if not prefix:
        raise SystemExit(
            f"ERROR: sidecar {sidecar_path} has no usable prefix "
            f"(rerun_mode.source_int_short_prefix / naming.base_dataset_short both empty)."
        )
    action_format = (cfg.get("action_format") or "rel").lower()
    if action_format not in ("rel", "abs"):
        raise SystemExit(f"ERROR: sidecar action_format {action_format!r} not rel/abs.")
    infix = "r" if action_format == "rel" else "a"

    derived = int_cache_path(lerobot_cache, hf_user, prefix, infix, round_n)
    print(
        f"[auto] derived --dataset_path from --policy_path + --round={round_n}:\n"
        f"       sidecar.naming.base_dataset_short = {naming.get('base_dataset_short')!r}\n"
        f"       sidecar.rerun_mode = {('present' if rerun else 'none')}  "
        f"→ intervention prefix = {prefix!r}, infix = {infix!r}\n"
        f"       → {derived}"
    )
    if not derived.is_dir():
        raise SystemExit(
            f"ERROR: derived dataset path {derived} does not exist on disk.\n"
            f"  Check --round / --hf_user / --lerobot_cache, or pass --dataset_path explicitly."
        )
    return derived


def resolve_inputs(args: argparse.Namespace) -> dict:
    """Parse --dataset_path + --round, resolve sibling intervention / blend
    paths, and resolve the base-dataset repo id. Returns a dict with keys:

        round              — int (single round) or ROUND_ALL_SENTINEL ('all')
        prefix, infix
        intervention_path  — Path (single round) or list[Path] (all rounds)
        blends             — list[(pct, Path)] (single) or
                             list[(pct, list[Path])] (all rounds — per-ratio
                             list of paths, ordered by round ascending)
        base_repo, base_path, base_source_tag,
        sidecar_path (Path or None)
        rounds_seen        — list[int] of rounds aggregated (informational;
                             length 1 for single-round, len(all_rounds) for all)
    """
    # `--round=all` short-circuits the policy_path autoderive (which targets
    # a single round). Single-round paths still autoderive as before.
    round_is_all = args.round == ROUND_ALL_SENTINEL
    if args.dataset_path is None:
        if round_is_all:
            if args.policy_path is None:
                raise SystemExit(
                    "ERROR: --round=all requires --policy_path (so the lineage "
                    "prefix can be derived). Pass --policy_path=<your training dir>."
                )
            # Autoderive against round 1 just to get the prefix/infix; we'll
            # discover ALL rounds via _enumerate_lineage_rounds below.
            args.dataset_path = _autoderive_dataset_path_from_policy(
                args.policy_path.expanduser().resolve(),
                1,
                args.hf_user,
                args.lerobot_cache,
            )
        else:
            if args.policy_path is None or args.round is None:
                raise SystemExit(
                    "ERROR: --dataset_path is required UNLESS both --policy_path and "
                    "--round are given (in which case the lineage's round-N "
                    "intervention dataset is auto-resolved from the policy's sidecar)."
                )
            args.dataset_path = _autoderive_dataset_path_from_policy(
                args.policy_path.expanduser().resolve(),
                args.round,
                args.hf_user,
                args.lerobot_cache,
            )
    dataset_path = args.dataset_path.expanduser().resolve()
    parsed = parse_dataset_short(dataset_path.name)

    # ── Round resolution ────────────────────────────────────────────────────
    round_n: int | str
    if round_is_all:
        round_n = ROUND_ALL_SENTINEL
    else:
        _resolved_round = args.round if args.round is not None else parsed.round
        if _resolved_round is None:
            raise SystemExit(
                f"ERROR: --round is required when --dataset_path is a base dataset "
                f"(parsed kind={parsed.kind!r}, name={parsed.name!r}). Pass --round=<N>."
            )
        round_n = int(_resolved_round)  # narrow to int for the rest of the function

    # ── Prefix / infix resolution ───────────────────────────────────────────
    # For base datasets the prefix isn't encoded in the name; in that case
    # the user MUST provide --dataset_path that points at a DAgger dataset
    # (intervention or blend) so the script can locate the lineage.
    if parsed.prefix is None:
        raise SystemExit(
            f"ERROR: --dataset_path={dataset_path} parsed as kind={parsed.kind!r}; the script needs "
            f"either an intervention / blend / merged dataset path to derive the lineage prefix, "
            f"OR you need to manually specify the lineage. (Passing a bare base-dataset path is "
            f"ambiguous — it could belong to multiple lineages.)"
        )

    prefix = parsed.prefix
    infix = parsed.infix
    assert infix is not None  # parse_dataset_short guarantees this when prefix is set

    # ── Sibling intervention + blends ──────────────────────────────────────
    # Two shapes depending on round mode:
    #   single round → `intervention_path: Path`, `blends: list[(pct, Path)]`
    #   --round=all  → `intervention_path: list[Path]` (per-round paths
    #                  sorted by round), `blends: list[(pct, list[Path])]`
    #                  (per-ratio list across rounds, sorted by round)
    if round_is_all:
        rounds_seen = _enumerate_lineage_rounds(args.lerobot_cache, args.hf_user, prefix, infix)
        if not rounds_seen:
            raise SystemExit(
                f"ERROR: --round=all found no rounds on disk for prefix={prefix!r} "
                f"infix={infix!r}.\n  Looked under: {Path(args.lerobot_cache) / args.hf_user}\n"
                f"  Expected dirs of the form `{prefix}_{infix}_dag{{N}}` with parquet content."
            )
        intervention_path = [
            int_cache_path(args.lerobot_cache, args.hf_user, prefix, infix, r) for r in rounds_seen
        ]
        # Aggregate blends by ratio across rounds. Each round may have a
        # different set of ratios on disk; we union them, listing only the
        # rounds where each ratio actually exists. Ordering: ratio ascending
        # outer, round ascending inner (so identical ratios from different
        # rounds end up adjacent in the load order, easier to debug).
        per_ratio: dict[int, list[Path]] = {}
        for r in rounds_seen:
            for pct, p in enumerate_blend_paths_on_disk(args.lerobot_cache, args.hf_user, prefix, infix, r):
                per_ratio.setdefault(int(pct), []).append(p)
        blends = [(pct, per_ratio[pct]) for pct in sorted(per_ratio.keys())]
        print(
            f"[round=all] aggregating rounds {rounds_seen} "
            f"({len(intervention_path)} intervention dataset(s), "
            f"{sum(len(v) for v in per_ratio.values())} blend dataset(s) across "
            f"{len(per_ratio)} ratio(s))."
        )
    else:
        assert isinstance(round_n, int)  # narrowed above; this branch is single-round
        rounds_seen = [round_n]
        intervention_path = int_cache_path(args.lerobot_cache, args.hf_user, prefix, infix, round_n)
        blends = enumerate_blend_paths_on_disk(args.lerobot_cache, args.hf_user, prefix, infix, round_n)
        if not intervention_path.is_dir():
            raise SystemExit(
                f"ERROR: intervention dataset for round {round_n} missing on disk:\n"
                f"  {intervention_path}\n"
                f"  Check --round / --dataset_path / --hf_user."
            )

    # ── Base repo resolution via sidecar + train_config fallbacks ───────────
    sidecar_path, legacy_prefix = _find_sidecar_with_legacy_fallback(
        args.training_root, prefix, allow_legacy=args.legacy_naming
    )
    if legacy_prefix is not None and sidecar_path is not None:
        print(
            f"[legacy_naming] strict prefix {prefix!r} had no sidecar; using "
            f"stripped prefix {legacy_prefix!r} → {sidecar_path}"
        )
    elif legacy_prefix is not None:
        print(
            f"[legacy_naming] strict prefix {prefix!r} had no sidecar; legacy "
            f"fallback prefix {legacy_prefix!r} also had none."
        )
    sidecar = load_sidecar(sidecar_path) if sidecar_path else None
    base_repo, base_source = resolve_base_repo(
        sidecar, explicit_override=args.base_repo_id, hf_user=args.hf_user
    )
    if base_repo is None:
        legacy_hint = (
            f"\n  legacy_naming fallback tried stripped prefix={legacy_prefix!r}, also unresolved."
            if legacy_prefix is not None
            else ""
        )
        raise SystemExit(
            f"ERROR: could not resolve base dataset repo id.\n"
            f"  prefix={prefix!r} round={round_n}\n"
            f"  sidecar={sidecar_path}{legacy_hint}\n"
            f"  Pass --base_repo_id=<HF_USER>/<short> explicitly to skip sidecar resolution."
        )
    base_path = args.lerobot_cache / base_repo
    if not base_path.is_dir():
        raise SystemExit(
            f"ERROR: resolved base dataset not on disk:\n"
            f"  base_repo={base_repo} (source={base_source})\n"
            f"  expected: {base_path}\n"
            f"  Pass --base_repo_id explicitly to override."
        )

    return {
        "round": round_n,
        "prefix": prefix,
        "infix": infix,
        "intervention_path": intervention_path,
        "blends": blends,
        "base_repo": base_repo,
        "base_path": base_path,
        "base_source_tag": base_source,
        "sidecar_path": sidecar_path,
        "rounds_seen": rounds_seen,  # [r] for single round, [r1, r2, ...] for --round=all
    }


def _print_resolved(resolved: dict) -> None:
    """Pretty-print the resolved dataset set (used by --dry-run + on real
    runs, before loading data)."""
    round_n = resolved["round"]
    rounds_seen = resolved.get("rounds_seen", [round_n])
    if round_n == ROUND_ALL_SENTINEL:
        print(f"Round: all  (aggregating rounds {rounds_seen})")
    else:
        print(f"Round: {round_n}")
    print(f"Prefix: {resolved['prefix']}  (infix={resolved['infix']!r})")
    print(f"base:         {resolved['base_path']}")
    sc = resolved["sidecar_path"]
    sc_str = str(sc) if sc else "<no sidecar found>"
    print(f"              [source={resolved['base_source_tag']} via sidecar {sc_str}]")
    ivp = resolved["intervention_path"]
    if isinstance(ivp, list):
        for r, p in zip(rounds_seen, ivp, strict=True):
            print(f"intervention dag{r}: {p}")
    else:
        print(f"intervention: {ivp}")
    for pct, p in resolved["blends"]:
        if isinstance(p, list):
            print(f"blend {pct:>3d}%:   {len(p)} dataset(s) across rounds:")
            for sub in p:
                print(f"                {sub}")
        else:
            print(f"blend {pct:>3d}%:   {p}")
    if not resolved["blends"]:
        print("(no blend datasets on disk for this round)")


def numpy_pca_2d(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # noqa: N803
    """Fit a 2D PCA basis via centered SVD. Returns (mean[D], components[2,D],
    singular_values[min(N,D)]).

    Caller projects via `(X - mean) @ components.T`. No sklearn dep —
    SVD on a centered N×D matrix is instant for the sizes we deal with
    (≈10⁴ chunks × ≤200 dims). Components are sorted by descending
    singular value (matches sklearn / numpy convention).
    """
    mean = X.mean(axis=0)
    centered = X - mean
    _u, sv, vt = np.linalg.svd(centered, full_matrices=False)
    return mean, vt[:2], sv


def numpy_kmeans_2d(
    points: np.ndarray,
    k: int,
    seed: int = 0,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Simple Lloyd's-algorithm k-means in 2D. Returns (labels[N], centroids[k,2]).

    No sklearn dep — k-means with ≤K=10 centroids on ≤1000 endpoints is
    well under 1ms and not a perf concern. Init via k-means++ for
    reproducibility (seed-controlled); single-restart so the result is
    deterministic given the same seed + data. Degenerate cases (fewer
    points than k) just return all points as their own cluster.
    """
    n = points.shape[0]
    if n == 0:
        return np.zeros(0, dtype=int), np.zeros((0, 2), dtype=float)
    if n <= k:
        return np.arange(n), points.astype(float).copy()
    rng = np.random.default_rng(seed)
    # k-means++ init: first centroid random, each subsequent picked with
    # probability ∝ squared-distance to nearest existing centroid.
    centroids = np.empty((k, 2), dtype=float)
    centroids[0] = points[rng.integers(n)]
    for c in range(1, k):
        d2 = np.min(((points[:, None, :] - centroids[None, :c, :]) ** 2).sum(axis=2), axis=1)
        d2_sum = d2.sum()
        if d2_sum == 0:
            # All remaining points coincide with existing centroids; pick any.
            centroids[c] = points[rng.integers(n)]
        else:
            centroids[c] = points[rng.choice(n, p=d2 / d2_sum)]
    # Lloyd iterations.
    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        dists = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = np.argmin(dists, axis=1)
        new_centroids = centroids.copy()
        for ci in range(k):
            mask = new_labels == ci
            if mask.any():
                new_centroids[ci] = points[mask].mean(axis=0)
        moved = np.linalg.norm(new_centroids - centroids, axis=1).max()
        labels = new_labels
        centroids = new_centroids
        if moved < tol:
            break
    return labels, centroids


def _per_episode_endpoints(
    sources: list[dict],
    mean: np.ndarray,
    components: np.ndarray,
) -> list[dict]:
    """For each source with per-frame episode IDs, find the LAST chunk of each
    episode and project it through the PCA basis. Returns a list of dicts:
        [{label, color, episode_id, pc1, pc2}, ...]
    one entry per episode per source. Sources without episode IDs (trajectory
    mode) are skipped — no meaningful per-episode endpoint there.
    """
    out: list[dict] = []
    for s in sources:
        eps = s.get("episodes")
        feats = s["features"]
        if eps is None or len(eps) == 0 or feats.shape[0] == 0:
            continue
        # For each episode, the LAST row in (episode, frame)-sorted order
        # is its endpoint chunk. _load_frames already sorts by (episode,
        # frame), so per-episode last-index = max index with that episode.
        last_idx_per_ep: dict[int, int] = {}
        for i in range(len(eps)):
            last_idx_per_ep[int(eps[i])] = i
        for ep, idx in last_idx_per_ep.items():
            proj = (feats[idx] - mean) @ components.T  # [2]
            out.append(
                {
                    "label": s["label"],
                    "color": s["color"],
                    "episode_id": int(ep),
                    "pc1": float(proj[0]),
                    "pc2": float(proj[1]),
                }
            )
    return out


def plot_cluster_trajectories(
    sources: list[dict],
    mean: np.ndarray,
    components: np.ndarray,
    endpoints: list[dict],
    labels: np.ndarray,
    k: int,
    out_path: Path,
    space_label: str,
    lineage_prefix: str,
    fixed_xlim: tuple[float, float] | None = None,
    fixed_ylim: tuple[float, float] | None = None,
) -> None:
    """Emit ONE plot per k-means cluster, each showing only the trajectories
    whose endpoints landed in that cluster. Uses the SAME PCA basis as the
    main scatter (mean, components) so cluster plots are directly comparable
    to the main plot and to each other. Per-source colors preserved.

    Axes are pinned to the FULL data range (across all sources, not just
    the cluster) so different cluster plots have identical axes — eye-
    comparable side by side.

    If `fixed_xlim` / `fixed_ylim` are supplied, they OVERRIDE the
    auto-computed range. Used by the variant cascade (nobase / onlybase /
    raw) so all variants of the same plot type share identical axes,
    making them directly comparable across variants too (not just within
    a single variant). Otherwise computed from the current `sources` list.

    Files: `<base>_clusterN_of_K.png` for cluster N ∈ {0..K-1}.
    """
    if not endpoints:
        return
    # Group cluster → source → set of episode_ids that ended in this cluster.
    cluster_eps: dict[int, dict[str, set[int]]] = {ci: {} for ci in range(k)}
    for ep_info, label in zip(endpoints, labels, strict=True):
        d = cluster_eps[int(label)].setdefault(ep_info["label"], set())
        d.add(ep_info["episode_id"])

    # Pre-compute global axis range across all source projections so every
    # cluster plot uses identical x/y limits — keeps the visual comparison
    # honest (otherwise matplotlib auto-zooms to whichever subset is densest).
    all_proj_min = np.array([np.inf, np.inf])
    all_proj_max = np.array([-np.inf, -np.inf])
    for s in sources:
        if s["features"].shape[0] == 0:
            continue
        proj_full = (s["features"] - mean) @ components.T
        all_proj_min = np.minimum(all_proj_min, proj_full.min(axis=0))
        all_proj_max = np.maximum(all_proj_max, proj_full.max(axis=0))
    # 5% margin so points on the edge aren't clipped.
    span = all_proj_max - all_proj_min
    xlim = (all_proj_min[0] - 0.05 * span[0], all_proj_max[0] + 0.05 * span[0])
    ylim = (all_proj_min[1] - 0.05 * span[1], all_proj_max[1] + 0.05 * span[1])
    # Variant cascade override: when nobase/onlybase share limits with the
    # full plot, every cluster*of* file in the lineage uses identical axes.
    if fixed_xlim is not None:
        xlim = fixed_xlim
    if fixed_ylim is not None:
        ylim = fixed_ylim

    cmap_cluster = plt.get_cmap("tab10")

    for ci in range(k):
        cluster_per_source = cluster_eps.get(ci, {})
        if not any(cluster_per_source.values()):
            continue
        fig, ax = plt.subplots(figsize=(11, 8))

        any_drawn = False
        for s in sources:
            eps = s.get("episodes")
            feats = s["features"]
            if eps is None or feats.shape[0] == 0:
                continue
            keep_episodes = cluster_per_source.get(s["label"], set())
            if not keep_episodes:
                continue
            mask = np.isin(eps, list(keep_episodes))
            if not mask.any():
                continue
            sub_feats = feats[mask]
            sub_eps = eps[mask]
            proj = (sub_feats - mean) @ components.T  # [N, 2]
            # Episode boundaries: indices where the episode_id changes.
            # _load_frames sorts by (episode, frame), so within each contiguous
            # run the rows are time-ordered → consecutive (i, i+1) pairs ARE
            # consecutive frames of the same episode.
            ep_change = np.diff(sub_eps) != 0
            seg_starts = np.concatenate([[0], np.where(ep_change)[0] + 1])
            seg_ends = np.concatenate([np.where(ep_change)[0] + 1, [len(sub_eps)]])
            # Quiver: per-frame → next-frame arrows within each episode.
            for st, en in zip(seg_starts, seg_ends, strict=True):
                if en - st < 2:
                    # Single-frame episode (rare) — just plot as a dot.
                    ax.scatter(
                        proj[st, 0],
                        proj[st, 1],
                        s=40,
                        marker="o",
                        c=[s["color"]],
                        alpha=0.6,
                        edgecolors="none",
                        zorder=2,
                    )
                    continue
                ax.quiver(
                    proj[st : en - 1, 0],
                    proj[st : en - 1, 1],
                    proj[st + 1 : en, 0] - proj[st : en - 1, 0],
                    proj[st + 1 : en, 1] - proj[st : en - 1, 1],
                    angles="xy",
                    scale_units="xy",
                    scale=1,
                    width=0.003,
                    color=[s["color"]],
                    alpha=0.55,
                    zorder=2,
                )
                # Endpoint marker (last frame of this episode) — bigger,
                # outlined so the END of each trajectory is unambiguous.
                ax.scatter(
                    proj[en - 1, 0],
                    proj[en - 1, 1],
                    s=70,
                    marker="o",
                    c=[s["color"]],
                    alpha=0.9,
                    edgecolors="black",
                    linewidths=0.6,
                    zorder=3,
                )
            # Legend entry (empty scatter just for the label).
            ax.scatter(
                [],
                [],
                c=[s["color"]],
                s=70,
                label=f"{s['label']}  (n={len(keep_episodes)})",
                edgecolors="black",
                linewidths=0.6,
            )
            any_drawn = True

        if not any_drawn:
            plt.close(fig)
            continue

        # Centroid of this cluster.
        cluster_pts = np.array(
            [[e["pc1"], e["pc2"]] for e, l in zip(endpoints, labels, strict=True) if int(l) == ci]
        )
        if len(cluster_pts) > 0:
            centroid = cluster_pts.mean(axis=0)
            ax.scatter(
                [centroid[0]],
                [centroid[1]],
                s=300,
                marker="x",
                c=[cmap_cluster(ci % 10)],
                linewidths=3.0,
                label=f"cluster {ci} centroid",
                zorder=5,
            )

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(
            f"Cluster {ci + 1}/{k} trajectories — {space_label} space\n"
            f"lineage prefix = {lineage_prefix}  |  "
            f"{sum(len(eps) for eps in cluster_per_source.values())} trajectories ending in this cluster\n"
            f"arrows = per-frame time flow (same PCA basis as the main plot)  |  "
            f"axes pinned to the full data range across all clusters"
        )
        legend = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
        ax.grid(True, alpha=0.3)
        cluster_out = out_path.with_name(out_path.stem + f"_cluster{ci}of{k}" + out_path.suffix)
        cluster_out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(cluster_out, bbox_inches="tight", bbox_extra_artists=[legend], dpi=150)
        plt.close(fig)
        print(f"  wrote {cluster_out}")


def _load_episode_camera_frames(
    dataset_path: Path,
    episode_id: int,
    camera_cols: list[str],
) -> dict[str, list[np.ndarray]]:
    """Read one episode's frames from a LeRobotDataset parquet, decode PNG
    bytes per camera column, return BGR numpy frames keyed by camera column.

    Mirrors the parquet-decoding pattern from
    my_scripts/visualize_intervention_episode.py:_load_blend_frames_for_scenario
    (read PNG bytes from parquet, cv2.imdecode each). Sorted by frame_index
    so the returned list is time-ordered. Empty list for any column with no
    frames decoded.
    """
    import glob

    import cv2  # type: ignore[import-not-found]
    import pyarrow.parquet as pq

    pf_paths = sorted(glob.glob(f"{dataset_path}/data/chunk-*/file-*.parquet"))
    if not pf_paths:
        raise FileNotFoundError(f"No parquet files under {dataset_path}/data")
    # Bucket rows: per-cam, list of (frame_index, png_bytes).
    by_cam: dict[str, list[tuple[int, bytes]]] = {c: [] for c in camera_cols}
    for pf in pf_paths:
        cols_to_read = [*camera_cols, "episode_index", "frame_index"]
        try:
            tbl = pq.read_table(pf, columns=cols_to_read)
        except Exception:
            # Column missing in this parquet (e.g. camera names differ across
            # dataset shards) — skip; we'll get an empty list back.
            continue
        ep_arr = tbl["episode_index"].to_pylist()
        fr_arr = tbl["frame_index"].to_pylist()
        for cam in camera_cols:
            try:
                img_arr = tbl[cam].to_pylist()
            except KeyError:
                continue
            for ep, fr, im in zip(ep_arr, fr_arr, img_arr, strict=True):
                if int(ep) == episode_id and im is not None:
                    by_cam[cam].append((int(fr), im["bytes"]))
    out: dict[str, list[np.ndarray]] = {}
    for cam, rows in by_cam.items():
        rows.sort(key=lambda x: x[0])
        decoded: list[np.ndarray] = []
        for _, b in rows:
            arr = np.frombuffer(b, dtype=np.uint8)
            img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img_bgr is not None:
                decoded.append(img_bgr)
        out[cam] = decoded
    return out


def _detect_camera_columns(dataset_path: Path) -> tuple[str | None, str | None]:
    """Find the actual `observation.images.base_rgb*` and `.wrist_rgb*` column
    names in this dataset's parquet schema. Datasets vary on the suffix
    (e.g. `_letterbox`, `_stretch`) depending on the image_resize_modes
    config used at recording time. Returns (base_col, wrist_col); either may
    be None if no matching column exists.
    """
    import glob

    import pyarrow.parquet as pq

    pf_paths = sorted(glob.glob(f"{dataset_path}/data/chunk-*/file-*.parquet"))
    if not pf_paths:
        return None, None
    schema = pq.read_schema(pf_paths[0])
    cols = schema.names
    base_col = next(
        (c for c in cols if c.startswith("observation.images.base_rgb")),
        None,
    )
    wrist_col = next(
        (c for c in cols if c.startswith("observation.images.wrist_rgb")),
        None,
    )
    return base_col, wrist_col


def _sample_cluster_episodes(
    endpoints: list[dict],
    labels: np.ndarray,
    k: int,
    n_per_cluster: int,
    seed: int = 0,
) -> dict[int, dict[str, list[int]]]:
    """Per-cluster, per-source episode sampler — shared by the video renderer
    and the endpoint collage so they use IDENTICAL episodes.

    Returns: {cluster_idx: {source_label: [ep_id, ...]}}. Sorted source order
    + a single np.random.default_rng(seed) → deterministic across calls
    with the same inputs.
    """
    out: dict[int, dict[str, list[int]]] = {}
    if n_per_cluster <= 0 or not endpoints:
        return out
    rng = np.random.default_rng(seed)
    for ci in range(k):
        cluster_eps_by_source: dict[str, list[int]] = {}
        for ep_info, lbl in zip(endpoints, labels, strict=True):
            if int(lbl) != ci:
                continue
            cluster_eps_by_source.setdefault(ep_info["label"], []).append(ep_info["episode_id"])
        sampled: dict[str, list[int]] = {}
        for src in sorted(cluster_eps_by_source.keys()):
            src_eps = cluster_eps_by_source[src]
            n_pick = min(n_per_cluster, len(src_eps))
            chosen = rng.choice(src_eps, size=n_pick, replace=False)
            sampled[src] = [int(e) for e in chosen]
        out[ci] = sampled
    return out


def render_cluster_videos(
    sources: list[dict],
    samples: dict[int, dict[str, list[int]]],
    out_dir: Path,
    fps: int = 30,
    trim_seconds: float = 2.0,
    gap_seconds: float = 0.6,
) -> None:
    """Render ONE composite mp4 per cluster — all sampled episodes for that
    cluster concatenated SEQUENTIALLY into a single file, each trimmed to
    the LAST `trim_seconds` of its episode so the cluster-defining behavior
    (end of trajectory, where the action chunks the clustering ran on
    actually came from) is the only thing shown.

    Between samples a `gap_seconds` title card announces the next
    (source, episode_id) so the cuts aren't disorienting — viewer always
    knows which sample they're watching. Within each sample, the
    within-frame layout stays base | wrist side-by-side (same as the
    previous per-episode videos).

    Replaces the previous per-episode output (one mp4 per sampled episode
    → 28+ files per cluster) with one mp4 per cluster — same content,
    much easier to scrub through. For per-episode deep dives, the
    collage PNG sibling shows the endpoint of every sample as still
    frames, which is the right granularity for "tell me which exact
    episode I want to look at" before launching the composite.

    Cleans `cluster<i>_composite.mp4` (and any stale per-episode `.mp4`s
    from the old pre-composite output format) before writing.
    """
    if not samples:
        return
    import shutil
    import subprocess

    if out_dir.is_dir():
        import re

        stale_pattern = re.compile(r"^cluster\d+_.*\.mp4$")
        for f in out_dir.iterdir():
            if f.is_file() and stale_pattern.match(f.name):
                f.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    cam_cache: dict[str, tuple[str | None, str | None]] = {}
    trim_frames = max(0, int(round(trim_seconds * fps)))
    gap_frames = max(1, int(round(gap_seconds * fps)))
    k_total = max(samples.keys()) + 1 if samples else 0
    written = 0
    skipped: list[str] = []

    for ci in sorted(samples.keys()):
        per_source = samples[ci]
        if not per_source:
            continue
        # Phase 1: load every sample's last-N frames + remember dims, so
        # we can pre-compute a single canvas size that works for ALL
        # samples in this cluster (libx264 needs fixed dims per stream).
        loaded: list[tuple[str, int, list[np.ndarray], list[np.ndarray]]] = []
        for src_label in sorted(per_source.keys()):
            src_obj = next((s for s in sources if s["label"] == src_label), None)
            if src_obj is None or "path" not in src_obj:
                skipped.append(f"cluster{ci}/{src_label} (no path)")
                continue
            ds_path: Path = src_obj["path"]
            if src_label not in cam_cache:
                cam_cache[src_label] = _detect_camera_columns(ds_path)
            base_col, wrist_col = cam_cache[src_label]
            if base_col is None or wrist_col is None:
                skipped.append(f"cluster{ci}/{src_label} (cams: base={base_col} wrist={wrist_col})")
                continue
            for ep_id in per_source[src_label]:
                try:
                    frames = _load_episode_camera_frames(ds_path, ep_id, [base_col, wrist_col])
                except Exception as e:
                    skipped.append(f"cluster{ci}/{src_label}/ep{ep_id} (load error: {e})")
                    continue
                bf = frames.get(base_col, [])
                wf = frames.get(wrist_col, [])
                if not bf or not wf:
                    skipped.append(
                        f"cluster{ci}/{src_label}/ep{ep_id} (empty: base={len(bf)} wrist={len(wf)})"
                    )
                    continue
                if trim_frames > 0:
                    bf = bf[-trim_frames:]
                    wf = wf[-trim_frames:]
                loaded.append((src_label, int(ep_id), bf, wf))
        if not loaded:
            continue

        # Phase 2: compute the cluster-wide canvas (max base+wrist dims
        # across all samples; pad smaller samples up to fit).
        max_bh = max(s[2][0].shape[0] for s in loaded)
        max_bw = max(s[2][0].shape[1] for s in loaded)
        max_wh = max(s[3][0].shape[0] for s in loaded)
        max_ww = max(s[3][0].shape[1] for s in loaded)
        pane_h = max(max_bh, max_wh)
        sep_w = 2
        body_w = max_bw + sep_w + max_ww
        header_h = 40
        canvas_h = header_h + pane_h
        canvas_w = body_w
        # libx264 yuv420p requires even dims.
        enc_h = canvas_h + (canvas_h % 2)
        enc_w = canvas_w + (canvas_w % 2)

        out_mp4 = out_dir / f"cluster{ci}_composite.mp4"
        # Total composite length for the log line.
        total_sample_frames = sum(min(len(b), len(w)) for _, _, b, w in loaded)
        total_gap_frames = gap_frames * len(loaded)  # title card before each sample
        total_dur = (total_sample_frames + total_gap_frames) / fps

        if shutil.which("ffmpeg") is None:
            # Fallback to mp4v (won't play in VSCode webview). Unusual env.
            import cv2

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_mp4), fourcc, float(fps), (enc_w, enc_h))
            if not writer.isOpened():
                skipped.append(f"cluster{ci}/composite (writer open failed)")
                continue
            _stream_fn = writer.write
            _stream_close = writer.release
            proc = None
        else:
            cmd = [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{enc_w}x{enc_h}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-movflags",
                "+faststart",
                str(out_mp4),
            ]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            assert proc.stdin is not None
            stdin = proc.stdin
            _stream_fn = lambda canvas: stdin.write(canvas.tobytes())  # noqa: E731
            _stream_close = stdin.close

        import cv2

        font = cv2.FONT_HERSHEY_SIMPLEX

        def _pad_pane(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
            """Pad bottom + right with black to (target_h, target_w)."""
            h, w = img.shape[:2]
            if h == target_h and w == target_w:
                return img
            out = np.zeros((target_h, target_w, 3), dtype=img.dtype)
            out[:h, :w] = img
            return out

        def _pad_canvas(canvas: np.ndarray) -> np.ndarray:
            """Pad canvas to encoder dims (even h/w required by libx264 yuv420p)."""
            h, w = canvas.shape[:2]
            if h < enc_h:
                canvas = np.vstack([canvas, np.zeros((enc_h - h, w, 3), dtype=canvas.dtype)])
            if canvas.shape[1] < enc_w:
                canvas = np.hstack(
                    [canvas, np.zeros((canvas.shape[0], enc_w - canvas.shape[1], 3), dtype=canvas.dtype)]
                )
            return canvas

        def _make_title_card(title: str, subtitle: str) -> np.ndarray:
            """Dark frame with centered two-line title — shown between samples.

            Rendered via matplotlib (not cv2.putText) so non-ASCII glyphs
            like em-dashes / multiplication signs / curly quotes render
            properly. cv2's built-in Hershey fonts only support basic ASCII
            and replace anything else with `?`.
            """
            fig = plt.figure(figsize=(enc_w / 100.0, enc_h / 100.0), dpi=100)
            # Slate background — matches the cv2-rendered title cards we
            # had before (color (25,25,25)) so the look is consistent
            # with the rest of the composite.
            fig.patch.set_facecolor((25 / 255.0, 25 / 255.0, 25 / 255.0))
            ax = fig.add_axes((0, 0, 1, 1))  # full-figure axes, no margins
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_axis_off()
            # Title slightly above center, subtitle below. fontsize=18 (down
            # from 22) gives the title a less-overwhelming presence in the
            # frame — still clearly readable but doesn't crowd the canvas
            # at small render dimensions. Subtitle stays at 14.
            ax.text(
                0.5,
                0.58,
                title,
                ha="center",
                va="center",
                fontsize=16,
                color=(180 / 255.0, 220 / 255.0, 255 / 255.0),
                family="DejaVu Sans",
                weight="bold",
            )
            ax.text(
                0.5,
                0.4,
                subtitle,
                ha="center",
                va="center",
                fontsize=14,
                color=(200 / 255.0, 200 / 255.0, 200 / 255.0),
                family="DejaVu Sans",
            )
            fig.canvas.draw()
            # Pull RGBA buffer → BGR for cv2/ffmpeg pipeline. RGBA path is
            # portable across matplotlib backends (tostring_rgb was removed
            # in newer mpl); buffer_rgba is the supported equivalent.
            buf = np.asarray(fig.canvas.buffer_rgba())
            img_bgr = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
            plt.close(fig)
            # Defensive: matplotlib's pixel dims can be off-by-one from
            # (enc_h, enc_w) due to DPI rounding. Resize to the exact
            # encoder canvas dims so the ffmpeg stdin write stays
            # frame-aligned.
            if img_bgr.shape[0] != enc_h or img_bgr.shape[1] != enc_w:
                img_bgr = cv2.resize(img_bgr, (enc_w, enc_h), interpolation=cv2.INTER_AREA)
            return img_bgr

        def _make_sample_frame(b: np.ndarray, w: np.ndarray, header: str) -> np.ndarray:
            """Header (40px) + base | sep | wrist body, padded to cluster canvas."""
            bp = _pad_pane(b, max_bh, max_bw)
            wp = _pad_pane(w, max_wh, max_ww)
            # Then pad each pane to the common pane_h.
            if bp.shape[0] < pane_h:
                bp = np.vstack([bp, np.zeros((pane_h - bp.shape[0], bp.shape[1], 3), dtype=bp.dtype)])
            if wp.shape[0] < pane_h:
                wp = np.vstack([wp, np.zeros((pane_h - wp.shape[0], wp.shape[1], 3), dtype=wp.dtype)])
            sep = np.zeros((pane_h, sep_w, 3), dtype=bp.dtype)
            body = np.hstack([bp.copy(), sep, wp.copy()])
            cv2.putText(body, "base", (10, 24), font, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(body, "base", (10, 24), font, 0.7, (60, 220, 60), 2, cv2.LINE_AA)
            cv2.putText(body, "wrist", (max_bw + sep_w + 10, 24), font, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(body, "wrist", (max_bw + sep_w + 10, 24), font, 0.7, (60, 220, 60), 2, cv2.LINE_AA)
            hdr = np.full((header_h, body.shape[1], 3), 30, dtype=np.uint8)
            cv2.putText(hdr, header, (12, 28), font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            return _pad_canvas(np.vstack([hdr, body]))

        try:
            for sample_idx, (src_label, ep_id, bf, wf) in enumerate(loaded):
                # Title card before each sample (including the first).
                title = f"cluster {ci}/{k_total}  —  sample {sample_idx + 1}/{len(loaded)}"
                subtitle = f"{src_label}   ·   ep {ep_id}   ·   last {trim_seconds:.1f}s"
                card = _make_title_card(title, subtitle)
                for _ in range(gap_frames):
                    _stream_fn(card)
                # Then the sample frames.
                n = min(len(bf), len(wf))
                for i in range(n):
                    header = f"cluster {ci}/{k_total} | {src_label} | ep {ep_id} | frame {i + 1}/{n}"
                    _stream_fn(_make_sample_frame(bf[i], wf[i], header))
        finally:
            _stream_close()
            if proc is not None:
                proc.wait()
                if proc.returncode != 0:
                    skipped.append(f"cluster{ci}/composite (ffmpeg exit {proc.returncode})")
                    continue
        print(f"  wrote {out_mp4}  ({len(loaded)} samples × ~{trim_seconds:.1f}s, total {total_dur:.1f}s)")
        written += 1
    if skipped:
        print(f"  [cluster_videos] skipped {len(skipped)} entry/entries:")
        for s in skipped[:10]:
            print(f"    - {s}")
        if len(skipped) > 10:
            print(f"    (+{len(skipped) - 10} more)")
    print(f"  [cluster_videos] wrote {written} composite mp4(s) total")


def render_cluster_endpoint_collage(
    sources: list[dict],
    samples: dict[int, dict[str, list[int]]],
    out_dir: Path,
) -> None:
    """For each cluster, render ONE collage PNG of the LAST-frame base+wrist
    images for every (source, episode) in `samples[cluster]`. Layout: rows
    = source, cols = sample index within source. Each cell shows base on
    the left, wrist on the right, with a `<source> | ep <id>` title.

    Companion to `render_cluster_videos`: both consume the same `samples`
    dict so the collage shows the exact same episodes the videos show, just
    as still endpoints instead of full trajectories. Useful for an
    at-a-glance comparison of where every sampled trajectory ended up
    (much faster to eyeball than scrubbing through 10+ videos).

    Cleans any prior `cluster*_endpoint_collage.png` files before writing.
    """
    if not samples:
        return
    import cv2  # type: ignore[import-not-found]

    out_dir.mkdir(parents=True, exist_ok=True)
    # Clean stale collages.
    import re

    stale_pattern = re.compile(r"^cluster\d+_endpoint_collage\.png$")
    for f in out_dir.iterdir():
        if f.is_file() and stale_pattern.match(f.name):
            f.unlink()

    cam_cache: dict[str, tuple[str | None, str | None]] = {}
    k_total = max(samples.keys()) + 1 if samples else 0
    for ci in sorted(samples.keys()):
        per_source_eps = samples[ci]
        if not per_source_eps:
            continue
        # Load each sample's last base+wrist frames into a per-cell BGR image
        # (base | sep | wrist, padded to common height).
        cell_imgs: dict[tuple[str, int], np.ndarray] = {}
        for src_label in sorted(per_source_eps.keys()):
            src_obj = next((s for s in sources if s["label"] == src_label), None)
            if src_obj is None or "path" not in src_obj:
                continue
            ds_path: Path = src_obj["path"]
            if src_label not in cam_cache:
                cam_cache[src_label] = _detect_camera_columns(ds_path)
            base_col, wrist_col = cam_cache[src_label]
            if base_col is None or wrist_col is None:
                continue
            for ep_id in per_source_eps[src_label]:
                try:
                    frames = _load_episode_camera_frames(ds_path, ep_id, [base_col, wrist_col])
                except Exception:
                    continue
                base_frames = frames.get(base_col, [])
                wrist_frames = frames.get(wrist_col, [])
                if not base_frames or not wrist_frames:
                    continue
                lb = base_frames[-1]
                lw = wrist_frames[-1]
                h = max(lb.shape[0], lw.shape[0])

                def _pad(img: np.ndarray) -> np.ndarray:
                    if img.shape[0] == h:
                        return img
                    return np.vstack([img, np.zeros((h - img.shape[0], img.shape[1], 3), dtype=img.dtype)])

                sep = np.zeros((h, 2, 3), dtype=lb.dtype)
                cell = np.hstack([_pad(lb), sep, _pad(lw)])
                cell_imgs[(src_label, ep_id)] = cell
        if not cell_imgs:
            continue

        sources_in_cluster = sorted({src for (src, _) in cell_imgs})
        max_cols = max(sum(1 for (s, _) in cell_imgs if s == src) for src in sources_in_cluster)
        n_rows = len(sources_in_cluster)
        n_cols = max(1, max_cols)

        # Build the grid figure. Per-cell aspect: roughly 2 panes wide × 1
        # pane tall — figure size scaled per cell so the images render at
        # readable resolution without re-sampling artifacts.
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 2.0))
        # Normalize axes to 2D array regardless of subplot count.
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        for ri, src in enumerate(sources_in_cluster):
            eps_for_src = per_source_eps.get(src, [])
            for col in range(n_cols):
                ax = axes[ri, col]
                ax.set_xticks([])
                ax.set_yticks([])
                if col < len(eps_for_src):
                    ep_id = eps_for_src[col]
                    cell = cell_imgs.get((src, ep_id))
                    if cell is not None:
                        ax.imshow(cv2.cvtColor(cell, cv2.COLOR_BGR2RGB))
                        ax.set_title(f"{src} | ep {ep_id}", fontsize=8)
                    else:
                        ax.set_facecolor("0.95")
                        ax.text(
                            0.5,
                            0.5,
                            "(load failed)",
                            ha="center",
                            va="center",
                            transform=ax.transAxes,
                            fontsize=8,
                            color="gray",
                        )
                else:
                    # Empty cell (this source had fewer samples than the max).
                    ax.set_facecolor("0.95")
                    ax.spines[:].set_visible(False)

        total_samples = sum(len(v) for v in per_source_eps.values())
        fig.suptitle(
            f"Cluster {ci}/{k_total} — endpoint frames "
            f"(base | wrist per cell, last frame of each sampled episode)\n"
            f"{total_samples} sample(s) across {len(sources_in_cluster)} source(s) — "
            f"same episodes as the companion videos",
            fontsize=11,
        )
        plt.tight_layout(rect=(0, 0, 1, 0.96))
        out_png = out_dir / f"cluster{ci}_endpoint_collage.png"
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out_png}")


def _detect_relative_action_config(
    policy_path: Path | None,
    base_dataset_path: Path | None,
) -> dict | None:
    """Return rel-action config when the loaded policy was trained with
    `use_relative_actions=True`, else None.

    Used by render_cluster_action_profiles to decide whether to plot
    last-chunk actions as REL deltas (= what the policy actually predicts)
    instead of ABS joint targets. The relative-mask is built lazily inside
    the renderer (needs n_dims) — this helper just returns the rel_step
    instance plus a few precomputed bits.

    Returns:
        {"rel_step": RelativeActionsProcessorStep,
         "exclude_joints": list[str],
         "action_names": list[str] | None}
        or None if no policy or policy doesn't use rel actions.
    """
    if policy_path is None:
        return None
    try:
        steps = _load_policy_processor_steps(policy_path, base_dataset_path=base_dataset_path)
    except Exception:
        return None
    from lerobot.processor.relative_action_processor import RelativeActionsProcessorStep

    rel_step = next(
        (s for s in steps["preprocessor"].steps if isinstance(s, RelativeActionsProcessorStep)),
        None,
    )
    if rel_step is None or not getattr(rel_step, "enabled", False):
        return None
    return {
        "rel_step": rel_step,
        "exclude_joints": list(getattr(rel_step, "exclude_joints", [])),
        "action_names": list(getattr(rel_step, "action_names", []) or []) or None,
    }


def _load_episode_last_state_chunk(
    dataset_path: Path,
    episode_id: int,
    chunk_len: int,
) -> np.ndarray | None:
    """Read the LAST `chunk_len` frames of observation.state for one episode.

    Used to convert an absolute-action last-chunk to its REL counterpart
    (action - state * mask) at render time, matching what the policy
    actually predicted internally. Returns [chunk_len, state_dim] float32,
    or None if not found. If the episode has fewer than chunk_len frames,
    the result is left-padded with the first frame so the shape is always
    [chunk_len, state_dim].
    """
    import glob

    import pyarrow.parquet as pq

    pf_paths = sorted(glob.glob(f"{dataset_path}/data/chunk-*/file-*.parquet"))
    rows: list[tuple[int, np.ndarray]] = []
    for pf in pf_paths:
        try:
            tbl = pq.read_table(pf, columns=["observation.state", "episode_index", "frame_index"])
        except Exception:
            continue
        eps = tbl["episode_index"].to_pylist()
        frs = tbl["frame_index"].to_pylist()
        states = tbl["observation.state"].to_pylist()
        for ep, fr, st in zip(eps, frs, states, strict=True):
            if int(ep) == episode_id and st is not None:
                rows.append((int(fr), np.asarray(st, dtype=np.float32)))
    if not rows:
        return None
    rows.sort(key=lambda x: x[0])
    if len(rows) < chunk_len:
        rows = [rows[0]] * (chunk_len - len(rows)) + rows
    last = rows[-chunk_len:]
    return np.stack([s for _, s in last], axis=0)


def _per_episode_last_feature(sources: list[dict]) -> dict[tuple[str, int], np.ndarray]:
    """Build a lookup: (source_label, episode_id) → that episode's LAST
    feature row (the chunk vector that was clustered on). Used by
    `render_cluster_action_profiles` to plot the raw action values, not
    their PCA projection — answers "what action distribution did this
    cluster actually capture?" when the projected endpoints / camera
    endpoints don't show an obvious visual difference.

    Skips sources without per-frame episode IDs (trajectory mode).
    """
    out: dict[tuple[str, int], np.ndarray] = {}
    for s in sources:
        eps = s.get("episodes")
        feats = s["features"]
        if eps is None or len(eps) == 0 or feats.shape[0] == 0:
            continue
        # _load_frames sorts by (episode, frame), so within each contiguous
        # episode run, the LAST row is the endpoint chunk. Overwriting the
        # dict on each row leaves the last-seen index per episode.
        last_idx_per_ep: dict[int, int] = {}
        for i in range(len(eps)):
            last_idx_per_ep[int(eps[i])] = i
        for ep, idx in last_idx_per_ep.items():
            out[(s["label"], int(ep))] = feats[idx]
    return out


def render_cluster_action_profiles(
    sources: list[dict],
    samples: dict[int, dict[str, list[int]]],
    chunk_len: int,
    data_column: str,
    out_dir: Path,
    base_dataset_path: Path | None = None,
    rel_action_config: dict | None = None,
) -> None:
    """For each cluster, render one PNG of per-dim curves showing the
    actual feature values (action chunk or state chunk, depending on
    --data_column) for every sampled episode. This is the visualization
    that DIRECTLY answers "what does this cluster's action distribution
    actually look like?" — useful when the projected PCA endpoint + camera
    end-frame collages look similar across clusters (which can happen when
    visually-similar end scenes have meaningfully-different last-chunk
    action commands, e.g. one decelerating vs one still-correcting).

    When `rel_action_config` is provided (== the policy was trained with
    use_relative_actions=True) AND data_column=="action", the per-sample
    chunks are converted from raw absolute actions to RELATIVE deltas:
        rel_chunk[t, d] = action[t, d] - state[t, d] * mask[d]
    matching what the policy actually predicts internally. The state for
    each sampled episode is re-loaded from its parquet's
    observation.state column for the last `chunk_len` frames; cached per
    (source, ep_id) so multiple cluster plots share the same loads. Dims
    excluded by the rel-step's `exclude_joints` (e.g. gripper) keep their
    absolute values via mask=False.

    Layout per cluster:
      - One subplot per feature dimension (= per joint when
        data_column=action), ~square grid (e.g. 7 dims → 3×3 with the
        last 2 cells hidden).
      - X-axis: step within chunk (0..chunk_len-1).
      - Y-axis: feature value at that step. Y-limits are pinned PER DIM
        across all clusters so the visual comparison is honest.
      - One line per sampled episode, colored by source. Per-source legend
        at the top of the figure.

    Joint names auto-loaded from base_dataset_path/meta/info.json when
    `data_column` looks action-/state-like; falls back to `dim 0..N-1`.
    """
    if not samples:
        return
    # Figure out per-step dimensionality. Chunk mode: feature_dim = chunk_len
    # × n_dims. Observation mode: chunk_len effectively 1 → feature_dim = n_dims.
    # Defensive: if the math doesn't work out (legacy mode, padded trajectory),
    # collapse to single-step with feature_dim dims.
    feat_dim = None
    for s in sources:
        if s["features"].shape[0] > 0:
            feat_dim = int(s["features"].shape[1])
            break
    if feat_dim is None:
        return
    n_steps = chunk_len if (chunk_len > 0 and feat_dim % chunk_len == 0) else 1
    n_dims = feat_dim // n_steps

    # Optional dim labels from the base dataset's meta/info.json (action /
    # observation.state both use the same robot-side dim names: joint_1..6,
    # gripper for a 7-DOF arm). Falls back to "dim N" labels.
    dim_labels: list[str] | None = None
    if base_dataset_path is not None:
        loaded = _load_action_names_from_dataset(base_dataset_path)
        if loaded is not None and len(loaded) == n_dims:
            dim_labels = loaded
    if dim_labels is None:
        dim_labels = [f"dim {i}" for i in range(n_dims)]

    out_dir.mkdir(parents=True, exist_ok=True)
    import re

    stale = re.compile(r"^cluster\d+_action_profile\.png$")
    for f in out_dir.iterdir():
        if f.is_file() and stale.match(f.name):
            f.unlink()

    feat_lookup = _per_episode_last_feature(sources)
    color_by_source = {s["label"]: s["color"] for s in sources}

    # Decide whether to convert abs → rel chunks. Only applies when (a)
    # the user clustered on `action` (rel conversion is undefined for
    # observation.state — that's already raw state) AND (b) the policy
    # was trained with `use_relative_actions=True`. The renderer pulls
    # per-episode last-chunk obs.state from disk and computes
    # `rel = action - state * mask` for each sampled (source, ep).
    rel_mask: np.ndarray | None = None
    state_chunk_cache: dict[tuple[str, int], np.ndarray] = {}
    rel_excluded: list[str] = []
    apply_rel = rel_action_config is not None and data_column == "action"
    if apply_rel and rel_action_config is not None:
        rel_step = rel_action_config["rel_step"]
        rel_mask = np.array(rel_step._build_mask(n_dims), dtype=bool)
        # Cache state chunks for ALL (source, ep_id) we'll plot — done
        # once up-front so the global y-range computation + the per-cluster
        # render loop both use the same converted values.
        wanted: set[tuple[str, int]] = set()
        for ci in samples:
            for src in samples[ci]:
                for ep_id in samples[ci][src]:
                    wanted.add((src, int(ep_id)))
        for src, ep_id in wanted:
            src_obj = next((s for s in sources if s["label"] == src), None)
            if src_obj is None or "path" not in src_obj:
                continue
            sc = _load_episode_last_state_chunk(src_obj["path"], ep_id, n_steps)
            if sc is not None and sc.shape == (n_steps, n_dims):
                state_chunk_cache[(src, ep_id)] = sc
        rel_excluded = list(rel_action_config.get("exclude_joints") or [])
        print(
            f"  [cluster_action_profiles] applying rel conversion "
            f"(policy use_relative_actions=True, exclude_joints={rel_excluded}) "
            f"to {len(state_chunk_cache)}/{len(wanted)} sampled episode(s) "
            f"with loadable state chunks"
        )

    def _maybe_convert(src: str, ep_id: int, chunk: np.ndarray) -> np.ndarray:
        """abs → rel = action - state * mask (if rel mode + state cached)."""
        if not apply_rel or rel_mask is None:
            return chunk
        state = state_chunk_cache.get((src, ep_id))
        if state is None:
            return chunk  # fall back to abs for this sample if state missing
        return chunk - state * rel_mask.astype(chunk.dtype)

    # Subplot grid: roughly square.
    rows = int(np.ceil(np.sqrt(n_dims)))
    cols = int(np.ceil(n_dims / rows))
    k_total = max(samples.keys()) + 1 if samples else 0

    # Pre-compute per-dim global y-range across ALL samples in ALL clusters
    # so each cluster's plot uses identical y-limits per joint — makes
    # cross-cluster comparison eyeball-able (otherwise matplotlib auto-zooms
    # each subplot to whatever its samples happen to span, and a joint that
    # only varies between -1.85 and -1.0 in one cluster vs -1.0 and +1.5 in
    # another would render at the same visual height, hiding the difference).
    dim_min = np.full(n_dims, np.inf)
    dim_max = np.full(n_dims, -np.inf)
    for ci in samples:
        for src in samples[ci]:
            for ep_id in samples[ci][src]:
                feat = feat_lookup.get((src, ep_id))
                if feat is None or len(feat) != n_steps * n_dims:
                    continue
                chunk = _maybe_convert(src, ep_id, np.asarray(feat).reshape(n_steps, n_dims))
                dim_min = np.minimum(dim_min, chunk.min(axis=0))
                dim_max = np.maximum(dim_max, chunk.max(axis=0))
    # 5% margin per dim so points on the edge don't get clipped; collapse
    # degenerate dims (min == max) to a small symmetric range so the plot
    # doesn't end up with zero-height axes (e.g. the forced-zero gripper).
    dim_ylim: list[tuple[float, float]] = []
    for di in range(n_dims):
        lo, hi = float(dim_min[di]), float(dim_max[di])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            dim_ylim.append((lo - 0.05, hi + 0.05) if np.isfinite(lo) else (-1.0, 1.0))
        else:
            margin = 0.05 * (hi - lo)
            dim_ylim.append((lo - margin, hi + margin))

    for ci in sorted(samples.keys()):
        per_source = samples[ci]
        if not per_source:
            continue
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(cols * 3.2, rows * 2.4),
            sharex=True,
        )
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        axes = np.asarray(axes).flatten()

        n_samples_plotted = 0
        seen_sources_in_legend: set[str] = set()
        for src in sorted(per_source.keys()):
            src_color = color_by_source.get(src, (0.5, 0.5, 0.5, 1.0))
            for ep_id in per_source[src]:
                feat = feat_lookup.get((src, ep_id))
                if feat is None or len(feat) != n_steps * n_dims:
                    continue
                chunk = _maybe_convert(src, ep_id, np.asarray(feat).reshape(n_steps, n_dims))
                for di in range(n_dims):
                    ax = axes[di]
                    legend_label = src if src not in seen_sources_in_legend else None
                    ax.plot(
                        range(n_steps),
                        chunk[:, di],
                        color=src_color,
                        alpha=0.6,
                        linewidth=1.2,
                        label=legend_label,
                    )
                seen_sources_in_legend.add(src)
                n_samples_plotted += 1

        for di in range(n_dims):
            axes[di].set_title(dim_labels[di], fontsize=10)
            axes[di].grid(True, alpha=0.3)
            axes[di].set_ylim(dim_ylim[di])  # pinned across clusters per dim
            # X-axis label only on the bottom row.
            if di >= n_dims - cols:
                axes[di].set_xlabel(f"step in chunk (0..{n_steps - 1})")
        for di in range(n_dims, len(axes)):
            axes[di].set_visible(False)

        # Aggregate legend at the top.
        handles, labels = [], []
        for ax in axes[:n_dims]:
            for h, l in zip(*ax.get_legend_handles_labels(), strict=True):
                if l not in labels:
                    handles.append(h)
                    labels.append(l)
        if handles:
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.98),
                ncol=min(len(handles), 6),
                fontsize=9,
                frameon=False,
            )

        space_tag = (
            f"relative {data_column} (action - state * mask, exclude={rel_excluded})"
            if apply_rel
            else data_column
        )
        fig.suptitle(
            f"Cluster {ci}/{k_total} — last-chunk {space_tag} per dim "
            f"({n_samples_plotted} sample(s); each line = one episode's "
            f"last {n_steps}-step chunk, colored by source)",
            fontsize=10,
            y=1.0,
        )
        plt.tight_layout(rect=(0, 0, 1, 0.94))
        out_png = out_dir / f"cluster{ci}_action_profile.png"
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out_png}")


def plot_endpoint_clusters(
    endpoints: list[dict],
    k: int,
    out_path: Path,
    space_label: str = "raw",
    lineage_prefix: str = "",
    seed: int = 0,
    precomputed_labels: np.ndarray | None = None,
    fixed_xlim: tuple[float, float] | None = None,
    fixed_ylim: tuple[float, float] | None = None,
) -> None:
    """Render a diagnostic scatter of per-episode endpoints colored by k-means
    cluster, plus centroids marked with ×. Prints per-source breakdown of
    cluster membership to stdout. No-op if there are no endpoints.

    If `precomputed_labels` is provided, k-means is skipped and those labels
    are used directly — lets callers share cluster assignments with companion
    plots (e.g. plot_cluster_trajectories) without re-running the algorithm.
    """
    if not endpoints:
        print("  [cluster_endpoints] no per-episode endpoints to cluster (trajectory mode? empty sources?).")
        return
    pts = np.array([[e["pc1"], e["pc2"]] for e in endpoints], dtype=float)
    effective_k = min(k, len(pts))
    if effective_k != k:
        print(
            f"  [cluster_endpoints] WARNING: requested k={k} but only {len(pts)} endpoint(s) available; using k={effective_k}."
        )
    if precomputed_labels is not None and len(precomputed_labels) == len(pts):
        labels = precomputed_labels
        # Re-compute centroids from labels (cheap; needed for the × markers).
        centroids = np.array(
            [
                pts[labels == ci].mean(axis=0) if (labels == ci).any() else np.zeros(2)
                for ci in range(effective_k)
            ]
        )
    else:
        labels, centroids = numpy_kmeans_2d(pts, k=effective_k, seed=seed)

    # Render: endpoints colored by cluster; cluster centroids as bold ×;
    # legend lists per-cluster sizes.
    fig, ax = plt.subplots(figsize=(11, 8))
    cmap = plt.get_cmap("tab10")
    for ci in range(effective_k):
        mask = labels == ci
        n_in = int(mask.sum())
        ax.scatter(
            pts[mask, 0],
            pts[mask, 1],
            s=70,
            marker="o",
            c=[cmap(ci % 10)],
            alpha=0.75,
            edgecolors="black",
            linewidths=0.5,
            label=f"cluster {ci}  (n={n_in})",
            zorder=3,
        )
    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        s=300,
        marker="x",
        c="black",
        linewidths=2.5,
        label="centroids",
        zorder=4,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    # Pin axes to caller-supplied limits so the endpoint_clusters PNG uses
    # the SAME range as the companion cluster*of* trajectory PNGs. Without
    # this, the onlybase / nobase variants would auto-scale to whichever
    # subset is in view, breaking visual comparison across variants.
    if fixed_xlim is not None:
        ax.set_xlim(fixed_xlim)
    if fixed_ylim is not None:
        ax.set_ylim(fixed_ylim)
    ax.set_title(
        f"Per-episode endpoint k-means (k={effective_k}) — {space_label} space\n"
        f"lineage prefix = {lineage_prefix}  |  {len(pts)} endpoint(s) across all sources\n"
        f"each dot = one episode's LAST chunk projected to the same PCA basis as the main plot"
    )
    legend = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    ax.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", bbox_extra_artists=[legend], dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")

    # Per-source breakdown: which sources have endpoints in which clusters?
    print(
        f"\n  [cluster_endpoints k={effective_k}, {space_label} space] cluster sizes + per-source breakdown:"
    )
    for ci in range(effective_k):
        mask = labels == ci
        n_in = int(mask.sum())
        c = centroids[ci]
        # Per-source counts
        per_src: dict[str, int] = {}
        for i in np.where(mask)[0]:
            lbl = endpoints[i]["label"]
            per_src[lbl] = per_src.get(lbl, 0) + 1
        src_str = ", ".join(f"{lbl}={cnt}" for lbl, cnt in sorted(per_src.items()))
        print(f"    cluster {ci}: {n_in} ep(s)  centroid=({c[0]:+.2f}, {c[1]:+.2f})  [{src_str}]")


# ────────────────────────── Per-frame parquet loading ────────────────────────


def _load_frames(
    dataset_root: Path,
    column: str,
    value_transform=None,
    needs_obs_state: bool = False,
    force_gripper: float | None = None,
    episodes_filter: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Read `<column>` + `episode_index` from a LeRobotDataset's parquet
    files, sorted by (episode_index, frame_index). When `value_transform` is
    provided, it's applied per-frame to the loaded values right before the
    sort — used to normalize raw joint/action values into the policy's
    internal space for the normalized-PCA plot variant.

    When `needs_obs_state` is True, additionally loads the `observation.state`
    column from the same parquets and passes the paired array to the value
    transform as a second argument. Required when the transform needs to
    compute relative actions (action - state*mask) before normalization.

    When `force_gripper` is set AND `column == "action"`, the LAST dim of every
    loaded action row is overwritten with `force_gripper` BEFORE the value
    transform runs (so rel-action conversion + normalization both see the
    forced value). `observation.state` is never touched even when loaded.

    Returns:
        values  — [N, D] float32  (the values from `column`, post-transform)
        episode — [N]    int64
    """
    import glob

    import pyarrow.parquet as pq

    files = sorted(glob.glob(f"{dataset_root}/data/chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files under {dataset_root}/data")
    extra_cols = ["observation.state"] if needs_obs_state and column != "observation.state" else []
    columns_to_read = [column, *extra_cols, "episode_index", "frame_index"]

    value_parts: list[np.ndarray] = []
    state_parts: list[np.ndarray] = []  # only populated if needs_obs_state
    episode_parts: list[np.ndarray] = []
    frame_parts: list[np.ndarray] = []
    skipped_in_progress: list[str] = []
    for f in files:
        try:
            t = pq.read_table(f, columns=columns_to_read)
        except Exception as e:
            # In-progress recordings show up as parquet files without a
            # valid footer ("Parquet magic bytes not found in footer" /
            # "Could not open Parquet input source"). Warn + skip so the
            # script can still plot whatever's complete on disk; the
            # enumerator's per-dir validation only checked file-000,
            # so chunks with multiple parquets can have later files mid-
            # write. Any OTHER read failure (missing schema column, etc.)
            # is still a hard error — that's a config/usage problem the
            # user needs to know about.
            msg = str(e).lower()
            if "magic bytes" in msg or "could not open parquet" in msg:
                skipped_in_progress.append(str(f))
                continue
            raise SystemExit(
                f"ERROR: failed to read columns {columns_to_read!r} from {f}: {e}\n"
                f"  Pass --data_column to a column that exists in the parquet schema."
            ) from e
        df = t.to_pandas()
        value_parts.append(np.stack(df[column].to_numpy()))
        if needs_obs_state:
            state_parts.append(np.stack(df["observation.state"].to_numpy()))
        episode_parts.append(df["episode_index"].to_numpy())
        frame_parts.append(df["frame_index"].to_numpy())
    if skipped_in_progress:
        print(
            f"  WARN: skipped {len(skipped_in_progress)} parquet file(s) under "
            f"{dataset_root.name} (recording still in progress — no footer):"
        )
        for p in skipped_in_progress[:3]:
            print(f"    - {Path(p).relative_to(dataset_root)}")
        if len(skipped_in_progress) > 3:
            print(f"    (+{len(skipped_in_progress) - 3} more)")
    if not value_parts:
        # Every parquet was mid-write — nothing usable. Return empty arrays
        # rather than crashing on np.concatenate; downstream code path
        # already tolerates zero-row sources (the PCA fit's all-feats
        # concat just produces an empty matrix). The 1st dim is N=0; the
        # 2nd dim is unknown without a valid schema, so use 0 — concat
        # against other non-empty sources still works because we
        # `np.concatenate(axis=0)` and the empty array's column count is
        # broadcast away in the empty-case.
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )
    values = np.concatenate(value_parts, axis=0).astype(np.float32)
    episode = np.concatenate(episode_parts, axis=0).astype(np.int64)
    frame = np.concatenate(frame_parts, axis=0).astype(np.int64)
    obs_state = np.concatenate(state_parts, axis=0).astype(np.float32) if needs_obs_state else None
    # Filter by selected episode indices (zero-indexed, per-source). Applied
    # BEFORE the value_transform so the rel-action pairing stays consistent
    # row-by-row (value_transform takes (values, obs_state) aligned 1:1).
    if episodes_filter is not None and len(episodes_filter):
        keep = np.isin(episode, np.fromiter(episodes_filter, dtype=np.int64))
        values = values[keep]
        episode = episode[keep]
        frame = frame[keep]
        if obs_state is not None:
            obs_state = obs_state[keep]
    # Pre-transform clamp: overwrite the last dim of the action column so all
    # downstream processing (rel-conversion, normalization, chunking, PCA)
    # uses the forced value. Skipped for non-action columns and for obs_state.
    if force_gripper is not None and column == "action":
        values[:, -1] = float(force_gripper)
    if value_transform is not None and len(values):
        values = value_transform(values, obs_state).astype(np.float32)
    # Final sort across files to guarantee episode-then-frame order regardless
    # of how parquet files happen to be laid out on disk.
    order = np.lexsort((frame, episode))
    return values[order], episode[order]


# ───────────────────────── Per-mode sampling primitives ─────────────────────


def _sample_observations(
    dataset_root: Path,
    column: str,
    max_points: int,
    rng: np.random.Generator,
    value_transform=None,
    needs_obs_state: bool = False,
    force_gripper: float | None = None,
    episodes_filter: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample whole episodes worth of frames, episode-by-episode.

    Iterates episodes in randomized order; for each episode, takes ALL of
    its frames in time order (no truncation). Stops AFTER the episode that
    first pushes the total to >= `max_points` — that final episode is kept
    in full, so the returned count may slightly exceed `max_points`. No
    episode is ever cut short. Returns `(features, episodes)` where
    `episodes[i]` is the episode id of row i; contiguous runs of equal
    values mark one episode's contribution.
    """
    values, episode = _load_frames(
        dataset_root,
        column,
        value_transform=value_transform,
        needs_obs_state=needs_obs_state,
        force_gripper=force_gripper,
        episodes_filter=episodes_filter,
    )
    n, d = values.shape
    if n == 0 or max_points <= 0:
        return np.zeros((0, d), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    ep_ids = np.unique(episode).copy()  # copy so shuffle doesn't mutate cache
    rng.shuffle(ep_ids)
    collected: list[np.ndarray] = []
    collected_eps: list[np.ndarray] = []
    total = 0
    for ep in ep_ids:
        ep_int = int(ep)
        ep_values = values[episode == ep_int]
        if len(ep_values) == 0:
            continue
        collected.append(ep_values)  # take the whole episode
        collected_eps.append(np.full(len(ep_values), ep_int, dtype=np.int64))
        total += len(ep_values)
        if total >= max_points:
            break  # stop AFTER the current episode, never truncate
    if not collected:
        return np.zeros((0, d), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return (
        np.concatenate(collected, axis=0).astype(np.float32),
        np.concatenate(collected_eps, axis=0),
    )


def _sample_chunks(
    dataset_root: Path,
    column: str,
    chunk_len: int,
    max_chunks: int,
    rng: np.random.Generator,
    value_transform=None,
    needs_obs_state: bool = False,
    force_gripper: float | None = None,
    episodes_filter: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample whole-episode runs of NON-OVERLAPPING `chunk_len`-windows.

    Iterates episodes in randomized order; for each episode, takes ALL of
    its non-overlapping windows starting at offsets 0, chunk_len,
    2·chunk_len, ... until the episode is exhausted. Episodes shorter than
    `chunk_len` are skipped (they contribute no chunks). Stops AFTER the
    episode that first pushes the total to >= `max_chunks` — that final
    episode is kept in full, so the returned chunk count may slightly
    exceed `max_chunks`. No episode is ever cut short.

    Chunks are NON-OVERLAPPING (sliding by `chunk_len`, not by 1) so each
    one represents a distinct policy-chunk-boundary action, matching how
    the real policy would emit them at rollout time.

    Returns `(features, episodes)` where each chunk row has its episode id
    in `episodes`.
    """
    values, episode = _load_frames(
        dataset_root,
        column,
        value_transform=value_transform,
        needs_obs_state=needs_obs_state,
        force_gripper=force_gripper,
        episodes_filter=episodes_filter,
    )
    n_frames, d = values.shape
    if n_frames < chunk_len or max_chunks <= 0:
        return (
            np.zeros((0, chunk_len * d), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )
    ep_ids = np.unique(episode).copy()
    rng.shuffle(ep_ids)
    collected: list[np.ndarray] = []
    collected_eps: list[int] = []
    total = 0
    for ep in ep_ids:
        ep_int = int(ep)
        ep_values = values[episode == ep_int]
        t_ep = len(ep_values)
        if t_ep < chunk_len:
            continue
        # Take ALL non-overlapping windows in this episode — no per-chunk cap.
        for start in range(0, t_ep - chunk_len + 1, chunk_len):
            collected.append(ep_values[start : start + chunk_len].reshape(1, chunk_len * d))
            collected_eps.append(ep_int)
            total += 1
        if total >= max_chunks:
            break  # stop AFTER the current episode, never truncate it
    if not collected:
        return (
            np.zeros((0, chunk_len * d), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )
    return (
        np.concatenate(collected, axis=0).astype(np.float32),
        np.asarray(collected_eps, dtype=np.int64),
    )


def _sample_trajectories(
    dataset_root: Path,
    column: str,
    traj_len: int,
    max_trajs: int,
    rng: np.random.Generator,
    value_transform=None,
    needs_obs_state: bool = False,
    force_gripper: float | None = None,
    episodes_filter: set[int] | None = None,
) -> np.ndarray:
    """Randomly sample up to `max_trajs` whole episodes from a dataset and
    return each as a fixed-length flattened vector of length `traj_len * D`.

    Episodes shorter than `traj_len` are edge-padded (last value repeated);
    episodes longer than `traj_len` are truncated. Returns
    [n_episodes, traj_len * D] float32.
    """
    values, episode = _load_frames(
        dataset_root,
        column,
        value_transform=value_transform,
        needs_obs_state=needs_obs_state,
        force_gripper=force_gripper,
        episodes_filter=episodes_filter,
    )
    if values.shape[0] == 0:
        return np.zeros((0, traj_len * 0), dtype=np.float32)
    d = values.shape[1]
    ep_ids = sorted({int(e) for e in np.unique(episode)})
    if not ep_ids:
        return np.zeros((0, traj_len * d), dtype=np.float32)
    if len(ep_ids) > max_trajs:
        chosen = rng.choice(np.array(ep_ids), size=max_trajs, replace=False)
        chosen = sorted(int(e) for e in chosen)
    else:
        chosen = ep_ids
    rows: list[np.ndarray] = []
    for ep in chosen:
        ep_values = values[episode == ep]  # [T_ep, D]
        t_ep = ep_values.shape[0]
        if t_ep == 0:
            continue
        if t_ep >= traj_len:
            padded = ep_values[:traj_len]
        else:
            # Edge-pad with the last observed value, repeated.
            pad = np.tile(ep_values[-1:], (traj_len - t_ep, 1))
            padded = np.concatenate([ep_values, pad], axis=0)
        rows.append(padded.reshape(traj_len * d))
    if not rows:
        return np.zeros((0, traj_len * d), dtype=np.float32)
    return np.stack(rows, axis=0).astype(np.float32)


# ──────────────────────────── Source assembly ───────────────────────────────


def _gather_sources(
    resolved: dict,
    plot_mode: str,
    data_column: str,
    chunk_len: int,
    traj_len: int,
    max_points: int,
    seed: int,
    value_transform=None,
    needs_obs_state: bool = False,
    force_gripper: float | None = None,
    episodes_filter: set[int] | None = None,
) -> list[dict]:
    """Load all source datasets + assemble per-source metadata for plotting.

    Returns a list of dicts, one per source, with keys:
        label    — short legend label, e.g. "base", "intervention", "b070"
        color    — matplotlib RGBA tuple
        marker   — single matplotlib marker char
        size     — scatter marker size
        alpha    — scatter alpha
        zorder   — drawing order (higher = on top)
        features — [N, F] float32 — F depends on plot_mode:
                     observation → F = D
                     chunk       → F = chunk_len * D
                     trajectory  → F = traj_len * D
    """
    rng = np.random.default_rng(seed)

    def sample(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
        """Returns (features, episodes_or_None). `episodes` is per-point
        episode IDs for observation / chunk modes (used to draw per-episode
        time-flow arrows on the scatter); None in trajectory mode where
        each point already IS an episode."""
        if plot_mode == "observation":
            return _sample_observations(
                path,
                data_column,
                max_points,
                rng,
                value_transform=value_transform,
                needs_obs_state=needs_obs_state,
                force_gripper=force_gripper,
                episodes_filter=episodes_filter,
            )
        if plot_mode == "chunk":
            return _sample_chunks(
                path,
                data_column,
                chunk_len,
                max_points,
                rng,
                value_transform=value_transform,
                needs_obs_state=needs_obs_state,
                force_gripper=force_gripper,
                episodes_filter=episodes_filter,
            )
        if plot_mode == "trajectory":
            return (
                _sample_trajectories(
                    path,
                    data_column,
                    traj_len,
                    max_points,
                    rng,
                    value_transform=value_transform,
                    needs_obs_state=needs_obs_state,
                    force_gripper=force_gripper,
                    episodes_filter=episodes_filter,
                ),
                None,
            )
        raise ValueError(f"Unknown plot_mode {plot_mode!r}")

    def sample_concat(p_or_paths: Path | list) -> tuple[np.ndarray, np.ndarray | None]:
        """Wrapper around `sample()` that transparently concatenates the
        result over a list of paths. Used by --round=all where each "source"
        (intervention, blend at a given ratio) is the union of that
        source's per-round datasets. For single-round mode, the caller
        still passes a bare Path and this just returns sample(path).

        max_points cap (--max_points_per_source) is enforced on the
        CONCATENATED total, not per-round. Without this, the per-round
        sample()'s cap of `max_points` would multiply with NUM_ROUNDS and
        intervention/blend sources would have ~10x more points than base
        (whose cap was applied at the source level). Subsampling at the
        EPISODE level preserves the per-episode arrow rendering and the
        'episode in full' semantic — we just drop whole episodes randomly
        until the cap is honored.
        """
        if isinstance(p_or_paths, list):
            feats_parts = []
            eps_parts: list[np.ndarray] = []
            ep_offset = 0
            any_eps_seen = False
            for sub in p_or_paths:
                sub_feats, sub_eps = sample(sub)
                feats_parts.append(sub_feats)
                if sub_eps is not None:
                    any_eps_seen = True
                    # Re-namespace episode ids across rounds so per-episode
                    # arrow rendering doesn't collide (round 1's ep 0 vs
                    # round 2's ep 0 are different recordings).
                    eps_parts.append(np.asarray(sub_eps, dtype=np.int64) + ep_offset)
                    ep_offset += int(sub_eps.max() + 1) if len(sub_eps) else ep_offset
            feats = np.concatenate(feats_parts, axis=0) if feats_parts else np.empty((0, 0))
            eps = np.concatenate(eps_parts, axis=0) if any_eps_seen and eps_parts else None
            # Post-concat per-source cap. Skip when the source has no
            # episode metadata (trajectory mode, each point IS an episode —
            # the per-round sample() already enforced the cap once and
            # multi-round trajectory is a small absolute count anyway).
            if max_points > 0 and eps is not None and feats.shape[0] > max_points:
                uniq = np.unique(eps)
                # Shuffle whole-episode order, then keep episodes until
                # cumulative point count crosses max_points (same semantic
                # as _sample_chunks: stop AFTER the episode that pushes
                # the total over the cap; never truncate an episode).
                shuffled = rng.permutation(uniq)
                ep_size = {int(e): int((eps == e).sum()) for e in uniq}
                keep_set: set[int] = set()
                total = 0
                for e in shuffled:
                    e_i = int(e)
                    keep_set.add(e_i)
                    total += ep_size[e_i]
                    if total >= max_points:
                        break
                keep_mask = np.isin(eps, np.fromiter(keep_set, dtype=np.int64))
                feats = feats[keep_mask]
                eps = eps[keep_mask]
            return feats, eps
        return sample(p_or_paths)

    sources: list[dict] = []

    # Per-source visual scheme designed for maximum per-source color clarity
    # even when sources land at the same point (which they do — intervention
    # and blends overlap heavily because blends are closed-loop replays of
    # the same intervention episodes). Drawing order:
    #
    #   1. Base (zorder 1, filled, low alpha) — background context cloud.
    #   2. Blends (zorder 2, filled, medium alpha) — rainbow by ratio,
    #      drawn AS A FILLED FACE so each ratio's color appears at the
    #      coordinate. Multiple overlapping blends mix visually via alpha.
    #   3. Intervention (zorder 3, HOLLOW outline) — black ring around each
    #      coordinate. Hollow means the blend-color faces underneath stay
    #      visible THROUGH the ring.
    #
    # Marker sizes are nested: base < blends < intervention. This lets the
    # bigger intervention rings encircle the colored blend dots so the
    # source identity reads at a glance.

    # 1. Base — pastel blue, drawn underneath everything else.
    base_feats, base_eps = sample(resolved["base_path"])
    sources.append(
        {
            "label": "base",
            "color": (0.65, 0.81, 0.89, 1.0),  # ColorBrewer light blue (#a6cee3)
            "marker": "o",
            "size": 10,
            "alpha": 0.35,
            "zorder": 1,
            "filled": True,
            "features": base_feats,
            "episodes": base_eps,
            "path": resolved["base_path"],  # for downstream video rendering
        }
    )

    # 2. Blends — rainbow by ratio (b010 purple → b090 red). Pin the
    #    colormap to the FULL [0, 1] range so the same ratio always gets
    #    the same color across script invocations.
    blends = resolved["blends"]
    if blends:
        cmap = plt.get_cmap("rainbow")
        pcts = np.array([p for p, _ in blends], dtype=float) / 100.0
        for (pct, path), color_t in zip(blends, [cmap(p) for p in pcts], strict=True):
            feats, eps = sample_concat(path)
            sources.append(
                {
                    "label": f"b{pct:03d}",
                    "color": color_t,
                    "marker": "o",
                    "size": 22,
                    "alpha": 0.5,
                    "zorder": 2,
                    "filled": True,
                    "features": feats,
                    "episodes": eps,
                    "path": path,  # Path or list[Path] (latter in --round=all mode)
                }
            )

    # 3. Intervention — HOLLOW black ring (foreground reference).
    iv_feats, iv_eps = sample_concat(resolved["intervention_path"])
    sources.append(
        {
            "label": "intervention",
            "color": (0.0, 0.0, 0.0, 1.0),
            "marker": "o",
            "size": 50,
            "alpha": 0.85,
            "zorder": 3,
            "filled": False,
            "features": iv_feats,
            "episodes": iv_eps,
            "path": resolved["intervention_path"],
        }
    )

    return sources


def _validate_feature_dims(sources: list[dict]) -> int:
    """Assert all per-source feature matrices share the same column count.
    Returns the shared dim."""
    dims = {s["label"]: s["features"].shape[1] for s in sources}
    unique = set(dims.values())
    if len(unique) != 1:
        details = "\n".join(f"  {lbl}: D={d}" for lbl, d in dims.items())
        raise SystemExit(
            f"ERROR: feature-dim mismatch across sources:\n{details}\n"
            "All sources must share the same column dimensionality for PCA."
        )
    return unique.pop()


# ─────────────────────────────── Plotting ───────────────────────────────────


def plot_state_coverage_pca(
    resolved: dict,
    sources: list[dict],
    plot_mode: str,
    data_column: str,
    chunk_len: int,
    traj_len: int,
    out_path: Path,
    base_highlight_fraction: float = 0.10,
    highlight_seed: int = 42,
    space_label: str = "raw",
    cluster_endpoints_k: int | None = None,
    videos_per_cluster: int = 0,
    cluster_video_trim_seconds: float = 2.0,
    rel_action_config: dict | None = None,
    pca_basis: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    cluster_xlim: tuple[float, float] | None = None,
    cluster_ylim: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float], tuple[float, float]]:
    """Concat all source features, fit a single 2D PCA, project each source,
    and scatter-plot. Each scatter source's per-episode time direction is
    drawn directly on the scatter via thin connecting lines + arrows (see
    the per-source overlay block below). Saves PNG to `out_path`.

    For the `base` source, a small fraction of its episodes (default 10%,
    rounded up) are rendered at higher alpha and slightly bolder arrows so
    they pop out of the otherwise-faint base cloud — useful when the base
    cloud is so dense it muddles individual trajectories. Controlled by
    `base_highlight_fraction` (set to 0 to disable).

    If `pca_basis` is supplied as `(mean, components, sv)` from a previous
    call, skip the local fit and reuse that basis — so multiple subplots
    over different SUBSETS of the same source set (e.g. all / nobase /
    onlybase) project to the SAME axes and points land at the SAME pixel
    coordinates. Used to keep nobase/onlybase variants comparable to the
    canonical "all sources" plot. The fitted (or reused) basis is always
    returned so the caller can pass it to follow-up emits.
    """
    # `all_feats` is the union of points actually being plotted in THIS
    # variant (full / nobase / onlybase). Always computed so the variance-
    # explained labels below reflect what's IN the scatter, even when the
    # basis came from a different subset.
    all_feats = np.concatenate([s["features"] for s in sources], axis=0)
    if all_feats.shape[0] == 0:
        raise SystemExit("ERROR: no data points loaded across any source — aborting plot.")
    if pca_basis is not None:
        # Reuse shared basis from a previous emit. Re-derive sv on the
        # SUBSET's projection so the "% var" axis labels describe how
        # much of THIS subset's variance lives along the shared axes —
        # honest report of how well the shared basis represents the
        # subset (vs cheating by reporting the original fit's sv).
        mean, components, _ = pca_basis
        proj_2d = (all_feats - mean) @ components.T
        sv = np.linalg.norm(proj_2d, axis=0)
    else:
        mean, components, sv = numpy_pca_2d(all_feats)
        proj_2d = (all_feats - mean) @ components.T
    # Cluster-plot axis limits: derived from THIS plot's projection if not
    # supplied. Caller threads them into variants alongside `pca_basis` so
    # every cluster*of* file in the lineage uses identical axes.
    if cluster_xlim is None or cluster_ylim is None:
        proj_min = proj_2d.min(axis=0)
        proj_max = proj_2d.max(axis=0)
        proj_span = proj_max - proj_min
        auto_xlim = (proj_min[0] - 0.05 * proj_span[0], proj_max[0] + 0.05 * proj_span[0])
        auto_ylim = (proj_min[1] - 0.05 * proj_span[1], proj_max[1] + 0.05 * proj_span[1])
        if cluster_xlim is None:
            cluster_xlim = auto_xlim
        if cluster_ylim is None:
            cluster_ylim = auto_ylim

    # Optional endpoint-clustering diagnostic. Runs k-means on per-episode
    # endpoints (last chunk per episode per source, projected through the
    # SAME PCA basis as the main plot) and emits a sibling PNG +
    # per-source breakdown to stdout. Useful for discovering whether the
    # task has more or fewer distinct end states than visual inspection
    # suggests. Doesn't modify the main plot — purely additive.
    if cluster_endpoints_k is not None and cluster_endpoints_k > 0:
        endpoints = _per_episode_endpoints(sources, mean, components)
        prefix_label = resolved.get("prefix") or ""
        cluster_out = out_path.with_name(
            out_path.stem + f"_endpoint_clusters{cluster_endpoints_k}" + out_path.suffix
        )
        # Compute k-means labels ONCE so the diagnostic scatter + per-cluster
        # trajectory plots all share the same cluster assignments. Seeded so
        # re-runs are deterministic.
        if endpoints:
            _ep_pts = np.array([[e["pc1"], e["pc2"]] for e in endpoints], dtype=float)
            _effective_k = min(cluster_endpoints_k, len(_ep_pts))
            _km_labels, _ = numpy_kmeans_2d(_ep_pts, k=_effective_k, seed=highlight_seed)
        else:
            _km_labels = np.zeros(0, dtype=int)
            _effective_k = 0
        plot_endpoint_clusters(
            endpoints,
            k=cluster_endpoints_k,
            out_path=cluster_out,
            space_label=space_label,
            lineage_prefix=prefix_label,
            seed=highlight_seed,
            precomputed_labels=_km_labels,
            fixed_xlim=cluster_xlim,
            fixed_ylim=cluster_ylim,
        )
        # Per-cluster trajectory plots: one PNG per cluster showing only the
        # episodes whose endpoints fall into it, using the SAME PCA basis.
        if _effective_k > 0:
            plot_cluster_trajectories(
                sources=sources,
                mean=mean,
                components=components,
                endpoints=endpoints,
                labels=_km_labels,
                k=_effective_k,
                out_path=out_path,
                space_label=space_label,
                lineage_prefix=prefix_label,
                fixed_xlim=cluster_xlim,
                fixed_ylim=cluster_ylim,
            )
            # Per-cluster representative VIDEOS (base + wrist side-by-side)
            # AND endpoint COLLAGE (one PNG per cluster showing only the
            # last frame of each sampled episode in a grid). Both consume
            # the same `samples` dict so they show the SAME episodes —
            # collage for at-a-glance comparison, videos for digging in.
            # Output dirs are namespaced under the variant's stem so each
            # of the raw / nobase / normalized / normalized-nobase
            # clusterings gets its own pair (clusters differ across
            # variants since the PCA basis differs).
            if videos_per_cluster > 0:
                # The cluster video / collage / action-profile renderers all
                # need a single source dataset path so they can decode the
                # camera frames + action chunks for sampled episodes. In
                # --round=all mode the intervention / blend sources have
                # `path = list[Path]` (one per round), and even though the
                # `base` source still has a single Path, generating
                # base-only videos while skipping the more interesting
                # intervention / blend variants is asymmetric and
                # confusing. Skip for ALL variants in --round=all so the
                # behavior is uniform: either you get videos for every
                # source, or for none of them. Re-run with --round=<N> to
                # get per-cluster media for a specific round.
                # Detect via the resolved round directly — checking only
                # the current variant's sources list would miss the
                # `onlybase` variant, whose base source IS still a single
                # Path even in --round=all mode (only intervention /
                # blends get split per round). We want uniform skip for
                # consistency, so the resolved-round check is the right
                # signal.
                _is_round_all = resolved.get("round") == ROUND_ALL_SENTINEL
                if _is_round_all:
                    print(
                        "\n  [cluster_videos] SKIPPED — --round=all aggregates "
                        "multi-round sources, so per-cluster video / collage / "
                        "action-profile renderers can't load frames from a "
                        "single dataset path. Re-run with --round=<N> (a "
                        "specific round) if you want these renderers."
                    )
                    return mean, components, sv, cluster_xlim, cluster_ylim
                samples = _sample_cluster_episodes(
                    endpoints,
                    _km_labels,
                    _effective_k,
                    n_per_cluster=videos_per_cluster,
                    seed=highlight_seed,
                )
                videos_dir = out_path.parent / f"{out_path.stem}_cluster_videos"
                collage_dir = out_path.parent / f"{out_path.stem}_cluster_collages"
                profiles_dir = out_path.parent / f"{out_path.stem}_cluster_action_profiles"
                print(
                    f"\n  [cluster_videos] rendering ONE composite mp4 per cluster "
                    f"(up to {videos_per_cluster} sample(s) per source, "
                    f"trimmed to last {cluster_video_trim_seconds:.1f}s each) "
                    f"→ {videos_dir}/"
                )
                render_cluster_videos(
                    sources=sources,
                    samples=samples,
                    out_dir=videos_dir,
                    trim_seconds=cluster_video_trim_seconds,
                )
                print(
                    f"\n  [cluster_collage] rendering one endpoint-frame collage per cluster → {collage_dir}/"
                )
                render_cluster_endpoint_collage(
                    sources=sources,
                    samples=samples,
                    out_dir=collage_dir,
                )
                # Action-profile plots: per cluster, show the actual
                # feature values (action chunk or state chunk depending on
                # --data_column) for every sampled episode. Answers "what
                # does this cluster's chunk distribution actually look
                # like?" when the camera collages look identical across
                # clusters because the visual scene at endpoints is similar
                # but the COMMANDED actions during the last chunk differ.
                print(
                    f"\n  [cluster_action_profiles] rendering one action-chunk "
                    f"profile plot per cluster → {profiles_dir}/"
                )
                render_cluster_action_profiles(
                    sources=sources,
                    samples=samples,
                    chunk_len=chunk_len,
                    data_column=data_column,
                    out_dir=profiles_dir,
                    base_dataset_path=resolved.get("base_path"),
                    rel_action_config=rel_action_config,
                )

    fig, ax = plt.subplots(figsize=(11, 8))
    highlight_rng = np.random.default_rng(highlight_seed)
    # Highlighted-base styling: darker shade of the same blue. Uniform RGB
    # scaling produces grey-navy on a pastel input (because pastel = nearly
    # equal RGB channels), so we operate in HLS: reduce lightness and bump
    # saturation, keeping the hue identical. Result for pastel blue
    # (#a6cee3) is a saturated medium-blue that's clearly darker than the
    # cloud but still unmistakably the same colour family.
    HIGHLIGHT_ALPHA = 0.95
    HIGHLIGHT_LIGHTNESS_FACTOR = 0.5  # 1.0 = unchanged, 0.0 = black
    HIGHLIGHT_SATURATION_BOOST = 2.0  # saturates immediately (clamped to 1.0)

    def _darken_preserve_hue(
        rgba: tuple,
        lightness_factor: float = HIGHLIGHT_LIGHTNESS_FACTOR,
        saturation_boost: float = HIGHLIGHT_SATURATION_BOOST,
    ) -> tuple:
        r, g, b, a = rgba
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        new_l = max(0.0, min(1.0, l * lightness_factor))
        new_s = max(0.0, min(1.0, s * saturation_boost))
        nr, ng, nb = colorsys.hls_to_rgb(h, new_l, new_s)
        return (nr, ng, nb, a)

    for s in sources:
        if s["features"].shape[0] == 0:
            continue  # empty source — skip to avoid empty-scatter labels
        proj = (s["features"] - mean) @ components.T  # [N, 2]
        label = f"{s['label']}  (n={s['features'].shape[0]})"
        eps = s.get("episodes")

        # For the `base` source, pick ~`base_highlight_fraction` of its
        # episodes (rounded up) to render at higher alpha so they read
        # individually through the otherwise-faint base cloud. `highlight_eps`
        # is a set of episode ids; `highlight_pts` is a set of point indices
        # used for trajectory mode (where each point IS one episode).
        highlight_eps: set[int] = set()
        highlight_pts: set[int] = set()
        if s["label"] == "base" and base_highlight_fraction > 0:
            if eps is not None and len(eps) > 0:
                unique_eps = np.unique(eps)
                n_hl = min(len(unique_eps), int(np.ceil(base_highlight_fraction * len(unique_eps))))
                if n_hl > 0:
                    chosen = highlight_rng.choice(unique_eps, size=n_hl, replace=False)
                    highlight_eps = {int(e) for e in chosen}
            else:
                # Trajectory mode has no per-point episode array; each scatter
                # point already IS one episode, so just bump ~fraction of them.
                n_pts = len(s["features"])
                n_hl = min(n_pts, int(np.ceil(base_highlight_fraction * n_pts)))
                if n_hl > 0:
                    idx = highlight_rng.choice(n_pts, size=n_hl, replace=False)
                    highlight_pts = {int(i) for i in idx}

        # Two render branches:
        #   * `eps is None` (trajectory mode): plain dots; each point is a
        #     whole episode and there's no within-episode flow. The
        #     highlighted point indices get a brighter second scatter pass
        #     on top.
        #   * `eps` provided: build a quiver of small arrows, one per
        #     consecutive (point[i] → point[i+1]) pair within an episode.
        #     The last point of each episode has no successor and is drawn
        #     as a small dot so the episode's endpoint is still visible.
        #     Arrows + endpoints belonging to highlighted episodes are
        #     rendered in a second pass at boosted alpha + bolder shaft.
        if eps is None or len(eps) < 2:
            n_pts = len(proj)
            mask_hl = np.zeros(n_pts, dtype=bool)
            if highlight_pts:
                mask_hl[list(highlight_pts)] = True
            highlight_color = _darken_preserve_hue(s["color"]) if s["label"] == "base" else s["color"]

            def _scatter(idx_mask: np.ndarray, alpha: float, color, *, with_label: bool):
                if not idx_mask.any():
                    return
                lbl = label if with_label else None
                sub = proj[idx_mask]
                if s["filled"]:
                    ax.scatter(
                        sub[:, 0],
                        sub[:, 1],
                        s=s["size"],
                        marker=s["marker"],
                        alpha=alpha,
                        zorder=s["zorder"],
                        c=[color],
                        edgecolors="none",
                        label=lbl,
                    )
                else:
                    ax.scatter(
                        sub[:, 0],
                        sub[:, 1],
                        s=s["size"],
                        marker=s["marker"],
                        alpha=alpha,
                        zorder=s["zorder"],
                        facecolors="none",
                        edgecolors=[color],
                        linewidths=1.2,
                        label=lbl,
                    )

            _scatter(~mask_hl, s["alpha"], s["color"], with_label=True)
            _scatter(mask_hl, HIGHLIGHT_ALPHA, highlight_color, with_label=False)
            continue

        # Find contiguous-episode run boundaries via diff.
        change_at = np.where(np.diff(eps) != 0)[0] + 1
        run_starts = np.concatenate([[0], change_at, [len(eps)]]).astype(int)
        # Arrow buckets — keep highlighted-episode arrows separate so we
        # can render them in a second quiver pass at boosted alpha.
        arrow_xy = {"normal": [], "highlight": []}
        arrow_uv = {"normal": [], "highlight": []}
        endpoint_xy = {"normal": [], "highlight": []}
        for ri in range(len(run_starts) - 1):
            a, b = int(run_starts[ri]), int(run_starts[ri + 1])
            run_proj = proj[a:b]
            ep_id = int(eps[a])
            bucket = "highlight" if ep_id in highlight_eps else "normal"
            if len(run_proj) == 1:
                endpoint_xy[bucket].append(run_proj[0])
                continue
            arrow_xy[bucket].append(run_proj[:-1])
            arrow_uv[bucket].append(np.diff(run_proj, axis=0))
            endpoint_xy[bucket].append(run_proj[-1])

        # For the highlight pass on base, use a DARKENED variant of the source
        # color (so highlighted episodes pop visually instead of just being
        # the same pale color at higher alpha). Other sources, if they ever
        # have highlights in the future, would reuse their own color as-is.
        highlight_color = _darken_preserve_hue(s["color"]) if s["label"] == "base" else s["color"]

        def _emit_quiver(
            key: str,
            alpha: float,
            width: float,
            hw: float,
            hl: float,
            *,
            color,
            with_label: bool,
        ):
            if not arrow_xy[key]:
                return
            xy = np.concatenate(arrow_xy[key], axis=0)
            uv = np.concatenate(arrow_uv[key], axis=0)
            # Highlighted base sits slightly above the normal base z-layer
            # so its arrows don't get covered by the normal cloud.
            zo = s["zorder"] + (0.2 if key == "highlight" else 0.0)
            ax.quiver(
                xy[:, 0],
                xy[:, 1],
                uv[:, 0],
                uv[:, 1],
                color=color,
                alpha=alpha,
                zorder=zo,
                angles="xy",
                scale_units="xy",
                scale=1,
                width=width,
                headwidth=hw,
                headlength=hl,
                headaxislength=hl - 0.5,
                label=(label if with_label else None),
            )

        # Normal-alpha quiver carries the legend entry.
        _emit_quiver(
            "normal",
            s["alpha"],
            0.0035,
            4.5,
            5.0,
            color=s["color"],
            with_label=True,
        )
        # Highlighted quiver: SAME geometry as normal, just darker color +
        # brighter alpha (no legend entry to avoid duplicates).
        _emit_quiver(
            "highlight",
            HIGHLIGHT_ALPHA,
            0.0035,
            4.5,
            5.0,
            color=highlight_color,
            with_label=False,
        )

        def _emit_endpoints(key: str, alpha: float, size_scale: float, *, color):
            if not endpoint_xy[key]:
                return
            ep = np.stack(endpoint_xy[key], axis=0)
            zo = s["zorder"] + (0.3 if key == "highlight" else 0.1)
            if s["filled"]:
                ax.scatter(
                    ep[:, 0],
                    ep[:, 1],
                    s=max(8, int(s["size"] * size_scale)),
                    marker=s["marker"],
                    alpha=alpha,
                    zorder=zo,
                    c=[color],
                    edgecolors="none",
                )
            else:
                ax.scatter(
                    ep[:, 0],
                    ep[:, 1],
                    s=max(20, int(s["size"] * size_scale)),
                    marker=s["marker"],
                    alpha=alpha,
                    zorder=zo,
                    facecolors="none",
                    edgecolors=[color],
                    linewidths=1.0,
                )

        _emit_endpoints("normal", s["alpha"], 0.5, color=s["color"])
        # Highlighted endpoints: same size as normal, just darker color + brighter alpha.
        _emit_endpoints("highlight", HIGHLIGHT_ALPHA, 0.5, color=highlight_color)

    # Small in-axes key describing the per-source time-flow encoding (which
    # is drawn in the per-source scatter loop above). Skipped in trajectory
    # mode where each scatter point IS one episode (no within-episode flow).
    if plot_mode != "trajectory" and any(s.get("episodes") is not None for s in sources):
        key_lines = [
            "per-source time flow:  each ▶ = one frame, pointing to its next "
            "frame in the same episode  •  dot = last frame of an episode",
        ]
        if base_highlight_fraction > 0:
            key_lines.append(
                f"highlighted base eps:  ~{base_highlight_fraction:.0%} of base "
                "episodes (rounded up) drawn at boosted alpha so they read individually"
            )
        ax.text(
            0.01,
            0.99,
            "\n".join(key_lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="#222",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#888", lw=0.6, alpha=0.9),
            zorder=7,
        )

    # Variance-explained labels — handy for judging whether 2D PCA is enough.
    # `sv` may have full min(N, D) components (local fit) or just 2 (shared
    # basis path), so compute total_var from the actual sum-of-squared
    # residuals (trace of the covariance matrix) — same answer in both
    # paths, and honest about the subset's variance when the basis is
    # shared with a different fit's subset.
    n_minus_1 = max(all_feats.shape[0] - 1, 1)
    total_var = float(np.sum((all_feats - mean) ** 2)) / n_minus_1
    var = (sv**2) / n_minus_1
    pc1_pct = 100.0 * var[0] / total_var if total_var > 0 else 0.0
    pc2_pct = 100.0 * var[1] / total_var if total_var > 0 else 0.0

    ax.set_xlabel(f"PC1 ({pc1_pct:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pc2_pct:.1f}% var)")
    feat_dim = all_feats.shape[1]
    if plot_mode == "observation":
        point_desc = f"per-frame {data_column} (D={feat_dim})"
    elif plot_mode == "chunk":
        d_col = feat_dim // chunk_len
        point_desc = f"{chunk_len}-step {data_column} chunks (D={chunk_len}×{d_col}={feat_dim})"
    else:  # trajectory
        d_col = feat_dim // traj_len
        point_desc = (
            f"whole-episode {data_column} (pad/truncate to T={traj_len}; D={traj_len}×{d_col}={feat_dim})"
        )
    ax.set_title(
        f"DAgger {plot_mode}-mode coverage (round {resolved['round']}) "
        f"— {space_label} space:\n"
        f"PCA scatter of {point_desc}\n"
        f"lineage prefix = {resolved['prefix']}  |  base = {resolved['base_repo']}",
        fontsize=10,
    )
    # Legend placed OUTSIDE the axes (to the right) so it never overlaps
    # scatter points. `bbox_extra_artists` + `bbox_inches="tight"` on savefig
    # makes sure the legend is included in the saved bounding box even though
    # it lives outside the axes.
    legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=9,
        markerscale=1.5,
        framealpha=0.9,
        borderaxespad=0.0,
    )
    ax.grid(True, alpha=0.25)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_extra_artists=[legend], bbox_inches="tight")
    plt.close(fig)
    return mean, components, sv, cluster_xlim, cluster_ylim


def _resolve_n_action_steps(args: argparse.Namespace, sidecar: dict | None) -> int:
    """Resolve the chunk length, in order: (1) --n_action_steps explicit
    override, (2) sidecar's initial policy's policy.n_action_steps. Errors
    if neither is available."""
    if args.n_action_steps is not None:
        return args.n_action_steps
    if sidecar is None:
        raise SystemExit(
            "ERROR: no sidecar found to auto-resolve n_action_steps; pass --n_action_steps=N explicitly."
        )
    tc = load_initial_policy_train_config(sidecar)
    if tc is None:
        raise SystemExit(
            "ERROR: could not load initial policy's train_config.json to auto-resolve n_action_steps; "
            "pass --n_action_steps=N explicitly."
        )
    n = (tc.get("policy") or {}).get("n_action_steps")
    if n is None:
        raise SystemExit(
            f"ERROR: policy.n_action_steps missing from initial policy's train_config.json "
            f"({tc.get('policy', {}).get('type', '?')!r} policy); pass --n_action_steps=N explicitly."
        )
    return int(n)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    resolved = resolve_inputs(args)
    sidecar = load_sidecar(resolved["sidecar_path"]) if resolved["sidecar_path"] else None

    # chunk_len is only needed in chunk mode (for both the scatter points
    # and the per-episode chunk-trace overlay). Resolve lazily so users in
    # observation / trajectory modes don't have to pass --n_action_steps.
    chunk_len = _resolve_n_action_steps(args, sidecar) if args.plot_mode == "chunk" else None

    _print_resolved(resolved)
    if chunk_len is not None:
        src_tag = "override" if args.n_action_steps is not None else "train_config"
        print(f"n_action_steps: {chunk_len}  [source={src_tag}]")
    print(f"plot_mode:     {args.plot_mode}")
    print(f"data_column:   {args.data_column}")
    if args.plot_mode == "trajectory":
        print(f"trajectory_length: {args.trajectory_length}  (edge-pad / truncate)")

    if args.dry_run:
        return 0

    # Default out path layout (v2):
    #     <DEFAULT_OUT_DIR>/<prefix>_dag<R>/<plot_mode>/coverage_pca.png
    # Per-lineage-round folder at the top level groups all modes (chunk,
    # observation, trajectory) of the same run under one parent dir, with
    # plot_mode as an inner subdir. Makes browsing one run's full output
    # set straightforward: open `<prefix>_dag<R>/` and see one child folder
    # per plot mode you've generated.
    # Explicit --out keeps the legacy single-dir behavior (all PNGs as
    # siblings of the user-supplied path).
    if args.out is None:
        run_subdir = f"{resolved['prefix']}_dag{resolved['round']}"
        out_path = DEFAULT_OUT_DIR / run_subdir / args.plot_mode / "coverage_pca.png"
    else:
        out_path = args.out.expanduser().resolve()

    print()
    n_sources = 1 + 1 + len(resolved["blends"])
    if args.plot_mode == "observation":
        unit = "frames"
    elif args.plot_mode == "chunk":
        unit = f"chunks (length={chunk_len})"
    else:
        unit = f"episodes (pad/truncate to T={args.trajectory_length})"
    print(
        f"Loading {unit} from {n_sources} dataset(s) "
        f"(subsample <= {args.max_points_per_source} per source, seed={args.seed})..."
    )
    if args.force_gripper is not None and args.data_column == "action":
        print(
            f"\n[force_gripper] overriding action[-1] = {args.force_gripper} on EVERY "
            f"loaded action row BEFORE any rel-conversion / normalization / chunking / PCA."
        )
    episodes_filter: set[int] | None = None
    if args.episodes is not None:
        episodes_filter = _parse_episodes_spec(args.episodes)
        sorted_eps = sorted(episodes_filter)
        print(
            f"\n[episodes] keeping only episode_index ∈ {sorted_eps} per source "
            f"({len(sorted_eps)} indices). For intervention/blend these are the round's "
            f"recorded episodes; for base these are unrelated demo indices."
        )
    sources = _gather_sources(
        resolved,
        plot_mode=args.plot_mode,
        data_column=args.data_column,
        chunk_len=chunk_len or 1,  # only consulted in chunk mode
        traj_len=args.trajectory_length,
        max_points=args.max_points_per_source,
        seed=args.seed,
        force_gripper=args.force_gripper,
        episodes_filter=episodes_filter,
    )
    feat_dim = _validate_feature_dims(sources)
    print(f"  loaded {len(sources)} source(s), feature dim D={feat_dim}")
    for s in sources:
        print(f"    {s['label']:>14s}: n={s['features'].shape[0]}")

    # Detect rel-action mode once (the policy's preprocessor is consulted).
    # When the user clustered on `action` and the policy was trained with
    # use_relative_actions=True, the cluster_action_profiles renderer
    # converts each per-episode last chunk from absolute to relative
    # (action - state * mask) so the plot shows what the policy actually
    # predicts. For data_column=observation.state this detection is a no-op
    # (rel conversion is undefined on raw state).
    rel_action_config: dict | None = None
    if args.data_column == "action":
        _policy_path_for_rel: Path | None = None
        if args.policy_path is not None:
            try:
                _policy_path_for_rel = _resolve_to_pretrained_model_dir(
                    args.policy_path.expanduser().resolve()
                )
            except SystemExit:
                _policy_path_for_rel = None
        else:
            # In --round=all, use the latest round for sidecar lookup — that's
            # the policy that generated the most recent intervention data and
            # is the canonical "acting policy" for normalization stats.
            _rel_lookup_round = (
                max(resolved["rounds_seen"]) if resolved["round"] == ROUND_ALL_SENTINEL else resolved["round"]
            )
            _policy_path_for_rel, _ = _find_acting_policy_path(
                args.training_root,
                resolved["prefix"],
                _rel_lookup_round,
                allow_legacy=args.legacy_naming,
            )
        rel_action_config = _detect_relative_action_config(
            _policy_path_for_rel,
            base_dataset_path=resolved.get("base_path"),
        )
        if rel_action_config is not None:
            print(
                f"\n[rel_action_detection] policy at {_policy_path_for_rel} uses "
                f"use_relative_actions=True (exclude_joints="
                f"{rel_action_config['exclude_joints']}). cluster_action_profiles "
                f"will convert last-chunk action → action - state*mask."
            )

    # All plot variants share these emission knobs.
    # `shared` carries the canonical (basis + cluster axis limits) tuple from
    # the "all sources" plot to its variant siblings (nobase / onlybase), so
    # they project to the SAME PCA axes AND their per-cluster plots use the
    # SAME x/y limits. Points land at the same pixel coordinates and the
    # cluster*of* files in the lineage are directly eye-comparable.
    def _emit(sources_list, out_p, space, shared=None):
        if shared is not None:
            pca_basis = (shared[0], shared[1], shared[2])
            cluster_xlim = shared[3]
            cluster_ylim = shared[4]
            action = "re-using shared PCA basis + cluster limits"
        else:
            pca_basis = None
            cluster_xlim = None
            cluster_ylim = None
            action = "fitting 2D PCA"
        print(f"\n{action} + plotting to {out_p}...")
        result = plot_state_coverage_pca(
            resolved,
            sources_list,
            plot_mode=args.plot_mode,
            data_column=args.data_column,
            chunk_len=chunk_len or 1,
            traj_len=args.trajectory_length,
            out_path=out_p,
            base_highlight_fraction=args.base_highlight_fraction,
            highlight_seed=args.seed + 42,
            space_label=space,
            cluster_endpoints_k=args.cluster_endpoints,
            videos_per_cluster=args.videos_per_cluster,
            cluster_video_trim_seconds=args.cluster_video_trim_seconds,
            rel_action_config=rel_action_config,
            pca_basis=pca_basis,
            cluster_xlim=cluster_xlim,
            cluster_ylim=cluster_ylim,
        )
        print(f"  wrote {out_p}")
        return result

    def _suffix_path(base_p: Path, suffix: str) -> Path:
        """Insert `_<suffix>` before the file extension."""
        return base_p.with_name(base_p.stem + f"_{suffix}" + base_p.suffix)

    def _drop_base(sources_list):
        """Filter out the `base` source so neither PCA fit nor scatter
        include it."""
        return [s for s in sources_list if s["label"] != "base"]

    def _keep_only_base(sources_list):
        """Keep ONLY the `base` source so neither PCA fit nor scatter
        include intervention/blends."""
        return [s for s in sources_list if s["label"] == "base"]

    # ── Plot #1: raw / unnormalized space, with base ────────────────────────
    # The basis + cluster axis limits fitted on ALL sources are shared with
    # the nobase / onlybase variants below — three plots, identical axes,
    # identical cluster*of* layout. Filename gets the `_unnormalized` suffix
    # so it's symmetric with the `_normalized` variants (no implicit default).
    out_path_raw = _suffix_path(out_path, "unnormalized")
    raw_shared = _emit(sources, out_path_raw, "raw")

    # ── Plot #2: raw / unnormalized space, WITHOUT base (PCA + cluster axes SHARED)
    if args.plot_without_base:
        sources_nb = _drop_base(sources)
        if not sources_nb:
            print("\n[plot_without_base] SKIPPED — no non-base sources to plot.")
        else:
            _emit(sources_nb, _suffix_path(out_path_raw, "nobase"), "raw (no base)", shared=raw_shared)

    # ── Plot #3: raw / unnormalized space, ONLY base (PCA + cluster axes SHARED)
    if args.plot_only_base:
        sources_ob = _keep_only_base(sources)
        if not sources_ob:
            print("\n[plot_only_base] SKIPPED — base source not in input set.")
        else:
            _emit(sources_ob, _suffix_path(out_path_raw, "onlybase"), "raw (only base)", shared=raw_shared)

    # ── Plot #3 + #4: policy-normalized space ──────────────────────────────
    if not args.plot_normalized:
        return 0

    if args.policy_path is not None:
        # Override may be a TRAINING dir or a pretrained_model dir; resolve to
        # whichever subdir actually contains policy_preprocessor.json.
        raw_override = args.policy_path.expanduser().resolve()
        policy_path = _resolve_to_pretrained_model_dir(raw_override)
        if policy_path != raw_override:
            policy_src = f"override → {policy_path.relative_to(raw_override)}"
        else:
            policy_src = "override"
    else:
        _norm_lookup_round = (
            max(resolved["rounds_seen"]) if resolved["round"] == ROUND_ALL_SENTINEL else resolved["round"]
        )
        policy_path, policy_src = _find_acting_policy_path(
            args.training_root,
            resolved["prefix"],
            _norm_lookup_round,
            allow_legacy=args.legacy_naming,
        )
    if policy_path is None:
        print(
            f"\n[plot_normalized] SKIPPED — could not auto-resolve a policy "
            f"({policy_src}). Pass --policy_path=<dir> to override, or "
            f"--no-plot_normalized to silence this."
        )
        return 0
    if args.data_column not in ("observation.state", "action"):
        print(
            f"\n[plot_normalized] SKIPPED — --data_column={args.data_column!r} "
            "is not normalizable (only 'observation.state' and 'action' are)."
        )
        return 0

    print(f"\n[plot_normalized] acting policy: {policy_path}  [{policy_src}]")
    try:
        value_transform, transform_note, needs_obs_state = _make_value_transform(
            args.data_column, policy_path, base_dataset_path=resolved["base_path"]
        )
    except SystemExit as e:
        print(f"\n[plot_normalized] SKIPPED — {e}")
        return 0
    print(f"[plot_normalized] transform: {transform_note}")
    if needs_obs_state:
        print(
            "[plot_normalized] also loading paired observation.state per frame "
            "for relative-action conversion."
        )

    sources_norm = _gather_sources(
        resolved,
        plot_mode=args.plot_mode,
        data_column=args.data_column,
        chunk_len=chunk_len or 1,
        traj_len=args.trajectory_length,
        max_points=args.max_points_per_source,
        seed=args.seed,
        value_transform=value_transform,
        needs_obs_state=needs_obs_state,
        force_gripper=args.force_gripper,
        episodes_filter=episodes_filter,
    )
    if args.drop_outliers_above is not None:
        threshold = float(args.drop_outliers_above)
        print(
            f"\n[drop_outliers_above={threshold}] filtering normalized scatter "
            f"points where any feature-dim |value| > {threshold} (raw plots "
            f"unaffected):"
        )
        for s in sources_norm:
            feats = s["features"]
            if feats.shape[0] == 0:
                continue
            keep = np.abs(feats).max(axis=1) <= threshold
            n_before = int(feats.shape[0])
            n_after = int(keep.sum())
            n_dropped = n_before - n_after
            if n_dropped > 0:
                s["features"] = feats[keep]
                if s.get("episodes") is not None:
                    s["episodes"] = s["episodes"][keep]
                print(
                    f"  {s['label']:>14s}: dropped {n_dropped}/{n_before} "
                    f"({100 * n_dropped / n_before:.1f}%)  "
                    f"→ {n_after} chunks remain"
                )
            else:
                print(f"  {s['label']:>14s}: nothing dropped (all {n_before} within threshold)")
    _validate_feature_dims(sources_norm)

    # Plot #4: normalized space, with base. PCA basis + cluster axis limits
    # are reused by the nobase / onlybase normalized variants — same-axes
    # guarantee within the normalized space, independent of the unnormalized-
    # space basis.
    norm_shared = _emit(
        sources_norm,
        _suffix_path(out_path, "normalized"),
        "policy-normalized",
    )

    # Plot #5: normalized space, WITHOUT base (PCA + cluster axes SHARED).
    if args.plot_without_base:
        sources_norm_nb = _drop_base(sources_norm)
        if not sources_norm_nb:
            print("\n[plot_without_base] SKIPPED (normalized) — no non-base sources.")
        else:
            _emit(
                sources_norm_nb,
                _suffix_path(out_path, "normalized_nobase"),
                "policy-normalized (no base)",
                shared=norm_shared,
            )

    # Plot #6: normalized space, ONLY base (PCA + cluster axes SHARED).
    if args.plot_only_base:
        sources_norm_ob = _keep_only_base(sources_norm)
        if not sources_norm_ob:
            print("\n[plot_only_base] SKIPPED (normalized) — base source not present.")
        else:
            _emit(
                sources_norm_ob,
                _suffix_path(out_path, "normalized_onlybase"),
                "policy-normalized (only base)",
                shared=norm_shared,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
