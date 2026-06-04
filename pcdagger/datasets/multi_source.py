"""Multi-source weighted-sampling wrapper around `MultiLeRobotDataset`.

When training samples from N sub-datasets with potentially different stats
(e.g., a base demo dataset plus several DAgger intervention datasets), the
naive options are:

  * Per-source normalization (each frame normalized via ITS source's stats),
    saved as PRE-normalized so the policy's downstream normalize layer is a
    no-op. This was the original design but has two problems:
      1. Eval-time normalization is asymmetric — the saved policy still needs
         SOME stats for live env observations, and there's no clean choice
         (BASE clips intervention-territory obs; combined compresses base
         obs). The per-source advantage gained at training is lost at eval.
      2. The "make the policy's normalizer a no-op" trick is fragile — the
         saved processor's stats override doesn't stick across `load_state_dict`
         (see `_stats_explicitly_provided` in
         `lerobot.processor.normalize_processor`). Easy to silently double-
         normalize.

  * Single aggregated stats over the union of sub-datasets (what this wrapper
    does now). Semantically equivalent to training on a manually-merged
    dataset with weighted sampling on top, but without writing the merge to
    disk. Training, save, and eval all use the SAME aggregated stats — one
    source of truth, no asymmetry, no bypass tricks needed.

This wrapper therefore does NOT normalize frames in `__getitem__`. It just:

  * Wraps `MultiLeRobotDataset` (sample-weighting is provided externally by
    a `WeightedRandomSampler` built in `lerobot_train.py` over
    `cumulative_sizes`).
  * Loads each sub-dataset's stats sidecar, validates consistency, aggregates
    them via the existing `aggregate_stats()` (min-of-mins, max-of-maxes,
    count-weighted mean/std, count-weighted quantiles when every source has
    them), and exposes the aggregated result via `.meta.stats`. The policy's
    preprocessor then normalizes exactly once, with stats that span the union
    of source distributions.

See `src/lerobot/configs/default.py:DatasetConfig` for the
`repo_ids`/`sample_weights`/`stats_paths` config surface that activates this
code path, and `src/lerobot/scripts/lerobot_train.py` for how the aggregated
stats reach the preprocessor.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.multi_dataset import MultiLeRobotDataset

logger = logging.getLogger(__name__)


# The five summary stats that aggregate_feature_stats() consumes
# unconditionally (it KeyErrors otherwise). We hand-check for these so we
# can emit a clear error message naming the offending source + feature
# instead of relying on the opaque KeyError that np.stack would throw inside
# aggregate_feature_stats.
_REQUIRED_BASIC_STAT_KEYS = frozenset({"mean", "std", "count", "min", "max"})


def _to_numpy_stats(stats: dict[str, dict[str, Any]]) -> dict[str, dict[str, np.ndarray]]:
    """Convert a JSON-loaded stats dict (lists) into numpy arrays in place.

    `aggregate_stats` expects np.ndarray values (it uses np.stack / np.min /
    np.max on them). Sidecars on disk are JSON, so values are Python lists or
    scalars.
    """
    out: dict[str, dict[str, np.ndarray]] = {}
    for feature_key, feature_stats in stats.items():
        out[feature_key] = {
            stat_name: np.asarray(stat_value, dtype=np.float64)
            for stat_name, stat_value in feature_stats.items()
        }
    return out


class _MetaProxy:
    """Lightweight proxy over a `LeRobotDatasetMetadata` that overrides
    `.stats` while forwarding every other attribute access to the inner meta.

    The proxy lets us hand callers an object that looks and feels like the
    base sub-dataset's metadata (so `dataset.meta.camera_keys`,
    `dataset.meta.features`, `make_policy(ds_meta=dataset.meta)`, etc. all
    work unchanged), while serving our AGGREGATED stats for `dataset.meta.stats`.

    `.stats` is a plain dict so existing call sites that mutate it in place
    (e.g. `dataset.meta.stats[key][stat_name] = X` in `make_dataset()`'s
    IMAGENET injection) keep working — the assignment lands on our proxy
    dict, not the inner sub-dataset's meta.
    """

    def __init__(self, inner_meta, aggregated_stats: dict[str, dict[str, np.ndarray]]):
        # Use object.__setattr__ to avoid triggering __setattr__ semantics
        # below if subclasses ever override it. Two underscores so attribute
        # lookup is unambiguous against any forwarded attr from inner_meta.
        object.__setattr__(self, "_inner", inner_meta)
        object.__setattr__(self, "_stats", aggregated_stats)

    @property
    def stats(self) -> dict[str, dict[str, np.ndarray]]:
        return self._stats

    @stats.setter
    def stats(self, new_stats: dict[str, dict[str, np.ndarray]]) -> None:
        # Some call sites in `lerobot_train.py` assign `dataset.meta.stats = X`
        # to override (e.g. when `--dataset.stats_path` is set). Honor that —
        # later reads see the new value.
        object.__setattr__(self, "_stats", new_stats)

    def __getattr__(self, name: str):
        # `__getattr__` only fires when normal lookup fails, so it won't
        # shadow `_inner`, `_stats`, or `stats`.
        return getattr(self._inner, name)


class MultiSourceNormalizingDataset(Dataset):
    """Aggregated-stats wrapper around `MultiLeRobotDataset`.

    Args:
        multi_dataset: An already-constructed `MultiLeRobotDataset` whose
            sub-dataset order is parallel to `stats_paths`.
        stats_paths: List of paths to stats JSON sidecars, one per sub-dataset.
            Each JSON has the standard `{feature_key: {mean, std, min, max,
            q01, q99, ...}}` schema written by lerobot's compute_stats pipeline.
        features: Kept for back-compat with the previous wrapper signature;
            not used by this implementation (kept so existing callers don't
            need to change).
        norm_map: Same — kept for back-compat, not used.

    The wrapper computes aggregated stats once at init by calling
    `aggregate_stats()` over the loaded sidecars, validates that every
    sub-dataset's sidecar has a compatible schema (same feature keys, same
    set of stat keys per feature, all required basics present), and exposes
    the aggregated result via `.meta.stats`. Frames are forwarded unchanged
    from the inner `MultiLeRobotDataset` — the policy's preprocessor handles
    the single normalization pass.

    The frame's `dataset_index` field (set by `MultiLeRobotDataset`) is
    preserved through this wrapper.
    """

    def __init__(
        self,
        multi_dataset: MultiLeRobotDataset,
        stats_paths: list[str],
        features: dict | None = None,  # back-compat; unused
        norm_map: dict | None = None,  # back-compat; unused
    ):
        super().__init__()
        del features, norm_map  # accepted for signature compat; not used here

        if len(stats_paths) != len(multi_dataset.repo_ids):
            raise ValueError(
                f"stats_paths length ({len(stats_paths)}) must match the number "
                f"of sub-datasets in multi_dataset ({len(multi_dataset.repo_ids)})."
            )
        self.multi_dataset = multi_dataset
        self.stats_paths = stats_paths

        # Cumulative sub-dataset sizes are still load-bearing for downstream
        # callers (the WeightedRandomSampler in `lerobot_train.py` builds
        # per-sample weights from these boundaries).
        self._cumulative_sizes: list[int] = multi_dataset.cumulative_sizes

        # Load and validate each sub-dataset's stats sidecar.
        per_source_stats_raw: list[dict[str, dict[str, list]]] = []
        for path in stats_paths:
            per_source_stats_raw.append(self._load_stats_json(Path(path)))
        self._validate_consistency(per_source_stats_raw, multi_dataset.repo_ids, stats_paths)

        # Aggregate. `aggregate_stats` from `compute_stats.py` does the
        # math: min-of-mins, max-of-maxes, count-weighted mean/std (via the
        # parallel-variance algorithm), count-weighted quantiles (only when
        # every source has the same quantile keys — we validate this above).
        per_source_stats = [_to_numpy_stats(s) for s in per_source_stats_raw]
        self._aggregated_stats: dict[str, dict[str, np.ndarray]] = aggregate_stats(per_source_stats)
        for i, (repo, stats) in enumerate(zip(multi_dataset.repo_ids, per_source_stats, strict=True)):
            logger.info(
                "MultiSourceNormalizingDataset: source[%d] %s contributes %d feature(s) with "
                "stat keys=%s (count=%s)",
                i,
                repo,
                len(stats),
                sorted({k for ft in stats.values() for k in ft}),
                {k: int(np.asarray(v["count"]).reshape(-1)[0]) for k, v in stats.items() if "count" in v},
            )
        logger.info(
            "MultiSourceNormalizingDataset: aggregated stats over %d source(s); "
            "%d feature(s); total frames=%d.",
            len(per_source_stats),
            len(self._aggregated_stats),
            len(multi_dataset),
        )

    # ── stats loading + consistency validation ─────────────────────────── #

    @staticmethod
    def _load_stats_json(path: Path) -> dict[str, dict[str, list]]:
        """Load a stats sidecar JSON, raising a clear error if missing."""
        if not path.exists():
            raise FileNotFoundError(
                f"Stats sidecar not found: {path}. Each sub-dataset in multi-dataset "
                "mode needs its own stats sidecar (rel-action sidecars are computed "
                "by the orchestrator at step 1b / 2b)."
            )
        with open(path) as f:
            return json.load(f)

    @staticmethod
    def _validate_consistency(
        per_source_stats: list[dict[str, dict[str, list]]],
        repo_ids: list[str],
        stats_paths: list[str],
    ) -> None:
        """Reject inconsistent sidecars before they cause an opaque aggregate failure.

        Three checks, each raising a clear ValueError naming the offending
        source + feature + missing key:

          1. Every source must define the SAME set of feature keys. Aggregation
             over a key present in only some sources would silently downweight
             the missing sources' contribution.
          2. For each feature, every source must have the SAME set of stat
             keys (mean/std/count/min/max plus any quantiles). If one source
             has only mean/std and another has only min/max, aggregation
             becomes ambiguous and the downstream policy normalizer would
             pick the wrong stats for its norm_map mode.
          3. The required basics (mean, std, count, min, max) must be present
             on every (source, feature) pair. `aggregate_feature_stats` uses
             these unconditionally; missing them gives a confusing KeyError
             deep inside numpy.
        """
        if not per_source_stats:
            raise ValueError("MultiSourceNormalizingDataset: no stats sidecars provided.")

        # Check 1: feature-key parity across sources.
        ref_features = set(per_source_stats[0])
        for i in range(1, len(per_source_stats)):
            cur_features = set(per_source_stats[i])
            if cur_features != ref_features:
                missing_from_cur = ref_features - cur_features
                extra_in_cur = cur_features - ref_features
                raise ValueError(
                    f"Multi-source stats sidecars have inconsistent feature keys.\n"
                    f"  source[0] ({repo_ids[0]}) @ {stats_paths[0]}\n"
                    f"    has features: {sorted(ref_features)}\n"
                    f"  source[{i}] ({repo_ids[i]}) @ {stats_paths[i]}\n"
                    f"    has features: {sorted(cur_features)}\n"
                    f"  Missing in source[{i}]: {sorted(missing_from_cur)}\n"
                    f"  Extra in source[{i}]:   {sorted(extra_in_cur)}\n"
                    f"  All sub-datasets must contribute stats for the SAME feature set."
                )

        # Check 2 + 3: per-feature, every source has the SAME set of stat
        # keys, AND that set includes the required basics.
        for feature_key in sorted(ref_features):
            ref_stat_keys = set(per_source_stats[0][feature_key])
            missing_basics = _REQUIRED_BASIC_STAT_KEYS - ref_stat_keys
            if missing_basics:
                raise ValueError(
                    f"Multi-source stats sidecar source[0] ({repo_ids[0]}) @ {stats_paths[0]}: "
                    f"feature '{feature_key}' is missing required stat key(s): "
                    f"{sorted(missing_basics)}. Required basics: "
                    f"{sorted(_REQUIRED_BASIC_STAT_KEYS)}. "
                    "All sources must provide these so aggregate_stats() can combine them."
                )
            for i in range(1, len(per_source_stats)):
                cur_stat_keys = set(per_source_stats[i][feature_key])
                if cur_stat_keys != ref_stat_keys:
                    missing_from_cur = ref_stat_keys - cur_stat_keys
                    extra_in_cur = cur_stat_keys - ref_stat_keys
                    raise ValueError(
                        f"Multi-source stats sidecars disagree on stat keys for feature "
                        f"'{feature_key}'.\n"
                        f"  source[0] ({repo_ids[0]}) @ {stats_paths[0]}\n"
                        f"    has stat keys: {sorted(ref_stat_keys)}\n"
                        f"  source[{i}] ({repo_ids[i]}) @ {stats_paths[i]}\n"
                        f"    has stat keys: {sorted(cur_stat_keys)}\n"
                        f"  Missing in source[{i}]: {sorted(missing_from_cur)}\n"
                        f"  Extra in source[{i}]:   {sorted(extra_in_cur)}\n"
                        f"  All sources must define the SAME stat keys for each feature so "
                        f"aggregate_stats() can combine them unambiguously. Recompute the "
                        f"per-source sidecars with a consistent compute_stats configuration."
                    )

    # ── Dataset protocol ───────────────────────────────────────────────── #

    def __len__(self) -> int:
        return len(self.multi_dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # No per-source normalization — the policy's preprocessor normalizes
        # once using the aggregated stats exposed via `self.meta.stats`.
        return self.multi_dataset[idx]

    # ── `.meta` proxy ──────────────────────────────────────────────────── #

    @property
    def meta(self) -> _MetaProxy:
        """Expose a meta proxy that serves AGGREGATED stats.

        Other meta attributes (camera_keys, features, fps, etc.) are forwarded
        from the FIRST sub-dataset's metadata — by convention that's the base
        dataset and provides the canonical schema. The validation above
        guarantees all sub-datasets have a compatible schema.
        """
        # Build lazily so any in-place stats mutation by callers (e.g.
        # IMAGENET injection in `make_dataset()`) survives across reads.
        if not hasattr(self, "_meta_proxy"):
            self._meta_proxy = _MetaProxy(
                self.multi_dataset._datasets[0].meta,
                self._aggregated_stats,
            )
        return self._meta_proxy

    # Forward attribute lookups to the inner MultiLeRobotDataset so callers
    # that read `.fps`, `.repo_ids`, `.stats`, etc. (typical for things like
    # the rename_map builder in lerobot_train.py) don't need to know about
    # the wrapper layer.
    def __getattr__(self, name: str):
        # `__getattr__` is only called when normal attribute lookup failed,
        # so this won't shadow `multi_dataset`, `_aggregated_stats`, etc.
        return getattr(self.multi_dataset, name)
