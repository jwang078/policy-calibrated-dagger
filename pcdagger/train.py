#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Train a policy.

Requires: pip install 'lerobot[training]'  (includes dataset + accelerate + wandb extras)

Launch with torchrun for distributed runs; every parallelism/acceleration knob lives on the
config (`--parallelism.*`, `--accelerator.*`) so a run is reproducible from its
train_config.json alone:

```bash
torchrun --nproc-per-node=8 $(which pcdagger-train) \
    --dataset.repo_id=... --policy.type=act \
    --parallelism.dp_shard=8 --accelerator.mixed_precision=bf16
```
"""

import dataclasses
import logging
import sys
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager, nullcontext
from pprint import pformat
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from accelerate import Accelerator

import torch
from termcolor import colored
from torch.optim import Optimizer
from tqdm import tqdm

from lerobot.common.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_metadata,
    publish_trained_model,
    push_checkpoint_to_hub,
    resume_after_prepare,
    resume_before_prepare,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.wandb_utils import WandBLogger
from lerobot.configs import JobConfig, parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets import EpisodeAwareSampler, compute_sampler_state
from pcdagger.lerobot_glue.dataset import make_train_eval_datasets
from lerobot.datasets.io_utils import cast_stats_to_numpy
from lerobot.distributed import (
    ParallelDims,
    finalize_sharded_policy,
    is_main_process,
    make_accelerator,
    set_fsdp_wrap_modules,
)
from lerobot.envs import close_envs, make_env, make_env_pre_post_processors
from lerobot.jobs import submit_to_hf
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies import PreTrainedPolicy, make_policy, make_pre_post_processors
from lerobot.policies.factory import (
    ProcessorConfigKwargs,
)
from pcdagger.compat import peft_available, reconnect_relative_absolute_steps
from pcdagger.lerobot_glue.policy import (
    _wrap_with_shared_autonomy,
    _wrap_with_temporal_ensemble,
)
from lerobot.rewards import make_reward_pre_post_processors
from lerobot.utils.collate import lerobot_collate_fn
from lerobot.utils.constants import PRETRAINED_MODEL_DIR, TRAINING_STATE_DIR
from lerobot.utils.import_utils import register_third_party_plugins, require_package
from lerobot.utils.io_utils import load_json
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import (
    cycle,
    format_big_number,
    has_method,
    init_logging,
    inside_slurm,
)

if TYPE_CHECKING or peft_available():
    from peft import PeftModel
else:
    PeftModel = None

from pcdagger.dagger.eval import eval_policy_all

EMA_STATE_FILENAME = "ema_state.pt"


@contextmanager
def _ema_weights(ema: Any, policy: PreTrainedPolicy) -> Iterator[None]:
    """Temporarily swap the EMA shadow weights into `policy`, restoring the live ones on exit."""
    params = list(policy.parameters())
    ema.store(params)
    ema.copy_to(params)
    try:
        yield
    finally:
        ema.restore(params)


@contextmanager
def _make_eval_envs(cfg: TrainPipelineConfig) -> Iterator[dict[str, dict[int, Any]]]:
    """Create evaluation environments for one run and always dispose of them."""
    envs = make_env(
        cfg.env,
        n_envs=cfg.eval.batch_size,
        use_async_envs=cfg.eval.use_async_envs,
    )
    try:
        yield envs
    finally:
        close_envs(envs)


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: "Accelerator",
    lr_scheduler=None,
    lock=None,
    sample_weighter=None,
) -> tuple[MetricsTracker, dict | None]:
    """
    Performs a single training step to update the policy's weights.

    This function executes the forward and backward passes, clips gradients, and steps the optimizer and
    learning rate scheduler. Accelerator handles mixed-precision training automatically, and — under
    gradient accumulation — suppresses gradient sync on non-final micro-batches and rescales the loss.

    Args:
        train_metrics (MetricsTracker): A MetricsTracker instance to record training statistics.
        policy (PreTrainedPolicy): The policy model to be trained (as returned by `accelerator.prepare`).
        batch (Any): A batch of training data.
        optimizer (Optimizer): The optimizer used to update the policy's parameters.
        grad_clip_norm (float): The maximum norm for gradient clipping (no clipping when <= 0).
        accelerator (Accelerator): The Accelerator instance for distributed training and mixed precision.
        lr_scheduler (LRScheduler | None, optional): An optional learning rate scheduler, stepped once
            per micro-batch. Defaults to None.
        lock (Lock | None, optional): An optional lock for thread-safe optimizer updates.
            Defaults to None.
        sample_weighter (SampleWeighter | None, optional): Optional SampleWeighter instance for
            per-sample loss weighting. Defaults to None.

    Returns:
        tuple[MetricsTracker, dict | None]: The updated MetricsTracker with new statistics for this
        step, and the dictionary of outputs from the policy's forward pass, for logging purposes.
    """
    start_time = time.perf_counter()
    policy.train()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Compute sample weights if a weighter is provided
    sample_weights = None
    weight_stats = None
    if sample_weighter is not None:
        sample_weights, weight_stats = sample_weighter.compute_batch_weights(batch)

    # Under gradient accumulation this context suppresses gradient sync (FSDP2:
    # set_requires_gradient_sync) on non-final micro-batches and divides the loss;
    # with gradient_accumulation_steps == 1 it is a transparent no-op.
    with accelerator.accumulate(policy):
        # Let accelerator handle mixed precision
        with accelerator.autocast():
            # `policy(...)`, never `policy.forward(...)`: FSDP2 all-gathers parameters through
            # nn.Module forward hooks, which only run via __call__.
            if sample_weights is not None:
                # Use per-sample loss for weighted training
                # Note: Policies supporting sample weighting must implement forward(batch, reduction="none")
                per_sample_loss, output_dict = policy(batch, reduction="none")

                # Weighted loss: each sample's contribution is scaled by its weight.
                # We divide by weight sum (not batch size) so that if some weights are zero,
                # the remaining samples contribute proportionally more, preserving gradient scale.
                # Weights are pre-normalized to sum to batch_size for stable training dynamics.
                epsilon = 1e-6
                loss = (per_sample_loss * sample_weights).sum() / (sample_weights.sum() + epsilon)

                # Log weighting statistics
                if output_dict is None:
                    output_dict = {}
                for key, value in weight_stats.items():
                    output_dict[f"sample_weight_{key}"] = value
            else:
                loss, output_dict = policy(batch)

            # TODO(rcadene): policy.unnormalize_outputs(out_dict)

        # Use accelerator's backward method
        accelerator.backward(loss)

        # Gradients are complete only on sync micro-batches; clipping partial gradients would
        # be meaningless. Always pass the full parameter list: accelerate's FSDP2 path requires
        # an exact match with the prepared model's parameters for a globally correct norm.
        grad_norm = None
        if accelerator.sync_gradients and grad_clip_norm > 0:
            grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)

        # Optimizer step (a no-op on non-final micro-batches under gradient accumulation)
        with lock if lock is not None else nullcontext():
            optimizer.step()
        optimizer.zero_grad()

        # Step through pytorch scheduler at every batch instead of epoch
        if lr_scheduler is not None:
            lr_scheduler.step()

    # Update internal buffers if policy has update method. These track optimizer updates
    # (EMA, target networks), not micro-batches: gate on the sync step under accumulation.
    if accelerator.sync_gradients and has_method(
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"
    ):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    train_metrics.loss = loss.item()
    if grad_norm is not None:
        train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    if torch.cuda.is_available():
        train_metrics.gpu_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
    # Aggregate the policy's scalar outputs for logging and rank-reduction across the log window.
    if output_dict:
        train_metrics.update_metrics(output_dict)
    return train_metrics, output_dict


def make_dataloaders(
    cfg: TrainPipelineConfig,
    dataset,
    eval_dataset,
    step: int,
    parallel_dims: ParallelDims,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader | None]:
    """Build the train (and optional eval) dataloader, including the sampler resume offset.

    The sampler offset is *derived* from `step` (`resume_before_prepare` loads step + RNG only):
    each loop step consumes `batch_size` samples on each of the `dp_world_size` distinct
    data-parallel workers — no grad-accumulation factor, since `step` counts micro-batches.

    Args:
        cfg (TrainPipelineConfig): The training config (batch size, workers, streaming, resume, seed).
        dataset (LeRobotDataset | MultiLeRobotDataset): The training dataset.
        eval_dataset (LeRobotDataset | None): Optional held-out split; when provided, an eval
            dataloader is built (subsampled per task when `cfg.max_eval_samples > 0`).
        step (int): The loop step to resume the sampler from (0 for a fresh run).
        parallel_dims (ParallelDims): The resolved parallelism topology; provides the device type
            and the fallback dp world size for the resume offset.

    Returns:
        tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader | None]: The train
        dataloader and the eval dataloader (None when no eval split exists).
    """
    active_cfg = cfg.trainable_config

    # Multi-dataset weighted sampling: when `sample_weights` is set, per-source share is
    # governed by the configured target weights instead of the dataset-size ratio. Each frame
    # in sub-dataset i gets weight `sample_weights[i] / size_i`, so the expected per-batch
    # share matches `sample_weights[i]`. `replacement=True` is required for float weights and
    # is the intended semantics ("every batch is X% intervention", however few frames exist).
    sampler = None
    if cfg.dataset.sample_weights is not None:
        per_sample_weights = torch.zeros(len(dataset), dtype=torch.double)
        cumulative_sizes = dataset.cumulative_sizes
        starts = [0, *cumulative_sizes[:-1]]
        for i, (start, end) in enumerate(zip(starts, cumulative_sizes, strict=True)):
            per_sample_weights[start:end] = cfg.dataset.sample_weights[i] / (end - start)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=per_sample_weights, num_samples=len(dataset), replacement=True
        )
        if is_main_process():
            logging.info(
                "Multi-dataset weighted sampling enabled: per-source weights=%s, sub-dataset sizes=%s",
                cfg.dataset.sample_weights,
                [end - start for start, end in zip(starts, cumulative_sizes, strict=True)],
            )

    # Two sampler modes: (1) weighted sampling already set `sampler` above — don't overwrite
    # it (a ConcatDataset exposes no per-episode indices for EpisodeAwareSampler anyway, and
    # per-source proportions are the whole point); (2) single-dataset non-streaming uses
    # EpisodeAwareSampler (upstream default).
    if not cfg.dataset.streaming and sampler is None:
        # All non-streaming (map-style) datasets use EpisodeAwareSampler.
        # The order is a pure function of (seed, epoch), so every rank independently produces the
        # same permutation. accelerate then shards it disjointly across data-parallel ranks via
        # BatchSamplerShard without needing a `generator` attribute to synchronize an RNG, and
        # resume is sample-exact.
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=getattr(active_cfg, "drop_n_last_frames", 0),
            shuffle=True,
            seed=cfg.seed if cfg.seed is not None else 0,
            absolute_to_relative_idx=dataset.absolute_to_relative_idx,
        )
        if cfg.resume and step > 0:
            # The resume offset depends on the (dp_world_size, batch_size) that produced `step`,
            # so use the values recorded in the checkpoint (falling back to the current ones for
            # older checkpoints that did not store them).
            metadata = load_training_metadata(cfg.checkpoint_path / TRAINING_STATE_DIR)
            saved_dp_world = metadata["dp_world_size"]
            saved_batch_size = metadata["batch_size"]
            ckpt_dp_world = saved_dp_world or parallel_dims.dp_world_size
            ckpt_batch_size = saved_batch_size or cfg.batch_size
            if is_main_process() and saved_dp_world not in (None, parallel_dims.dp_world_size):
                logging.warning(
                    f"Resuming with dp_world_size={parallel_dims.dp_world_size} but the "
                    f"checkpoint was written with dp_world_size={saved_dp_world}. The data order "
                    "resumes at the right epoch/offset, but per-rank sample-exactness requires "
                    "the same data-parallel world size."
                )
            if is_main_process() and saved_batch_size not in (None, cfg.batch_size):
                logging.warning(
                    f"Resuming with batch_size={cfg.batch_size} but the checkpoint was written "
                    f"with batch_size={saved_batch_size}. The data order resumes at the right "
                    "epoch/offset, but per-rank sample-exactness requires the same batch size."
                )
            sampler_state = compute_sampler_state(step, len(sampler), ckpt_batch_size, ckpt_dp_world)
            sampler.load_state_dict(sampler_state)
            if is_main_process():
                logging.info(
                    f"Resuming data order at epoch {sampler_state['epoch']}, "
                    f"sample {sampler_state['start_index']}"
                )
    else:
        shuffle = sampler is None

    device_type = parallel_dims.device_type
    # Only swap in the language-aware collate when the dataset actually
    # declares language columns; otherwise stay on PyTorch's default
    # collate so non-language training runs are unaffected.
    collate_fn = lerobot_collate_fn if dataset.meta.has_language_columns else None
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=device_type == "cuda",
        drop_last=False,
        collate_fn=collate_fn,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
        multiprocessing_context=cfg.dataloader_multiprocessing_context if cfg.num_workers > 0 else None,
    )

    # Build eval dataloader if a held-out split exists
    eval_dataloader = None
    if eval_dataset is not None:
        eval_ds = eval_dataset
        if cfg.max_eval_samples > 0 and hasattr(eval_dataset, "hf_dataset"):
            task_arr = eval_dataset.hf_dataset.data.column("task_index").to_numpy()
            unique_tasks = sorted(set(task_arr.tolist()))
            per_task = max(1, cfg.max_eval_samples // len(unique_tasks))
            selected: list[int] = []
            for t in unique_tasks:
                frames = (task_arr == t).nonzero()[0][:per_task]
                selected.extend(frames.tolist())
            eval_ds = torch.utils.data.Subset(eval_dataset, selected)

        eval_collate_fn = lerobot_collate_fn if dataset.meta.has_language_columns else None
        eval_dataloader = torch.utils.data.DataLoader(
            eval_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=device_type == "cuda",
            drop_last=False,
            collate_fn=eval_collate_fn,
            prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
            persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
            multiprocessing_context=cfg.dataloader_multiprocessing_context if cfg.num_workers > 0 else None,
        )
    return dataloader, eval_dataloader


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    """
    Main function to train a policy.

    This function orchestrates the entire training pipeline, including:
    - Setting up logging, seeding, and the distributed engine.
    - Creating the dataset, evaluation environment (if applicable), policy, and optimizer.
    - Handling resumption from a checkpoint (two-phase, around `accelerator.prepare`).
    - Running the main training loop, which involves fetching data batches and calling `update_policy`.
    - Periodically logging metrics, saving model checkpoints, and evaluating the policy.
    - Publishing the trained model to the Hugging Face Hub if configured.

    Args:
        cfg (TrainPipelineConfig): A `TrainPipelineConfig` object containing all training
            configurations, parsed from the CLI by `parser.wrap()`. On `--resume`, it is the config
            recorded in the checkpoint's `train_config.json`; when `cfg.job.is_remote`, the run is
            dispatched to HF Jobs instead of executing locally.
    """
    if cfg.job.is_remote:
        return submit_to_hf(cfg)

    require_package("accelerate", extra="training")

    cfg.validate()  # all fail-fasts fire here, before any distributed init

    # --- engine & topology --------------------------------------------------------------------
    # The factory is the ONLY accelerate configuration site: it guards against env-var
    # interference, resolves the declared parallelism degrees against the launched world, and
    # builds the Accelerator from the config mirrors.
    accelerator = make_accelerator(cfg)
    parallel_dims = ParallelDims.from_config(
        cfg.parallelism, accelerator.num_processes, accelerator.device.type
    )
    init_logging(accelerator=accelerator)

    if is_main_process():
        logging.info(pformat(cfg.to_dict()))

    if cfg.wandb.enable and cfg.wandb.project and is_main_process():
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process():
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    device = accelerator.device
    if cfg.cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # --- data (the main process downloads once; peers read the populated cache) ----------------
    if is_main_process():
        logging.info("Creating dataset")
        dataset, eval_dataset = make_train_eval_datasets(cfg)
    accelerator.wait_for_everyone()
    if not is_main_process():
        dataset, eval_dataset = make_train_eval_datasets(cfg)

    assert dataset is not None

    # Override dataset stats with a custom stats file if specified. This lets multiple
    # policies (e.g. diffusion with chunk_size=8, pi05 with chunk_size=50) share the same
    # dataset on disk while each using their own relative-action normalization stats.
    # Truthy check (not `is not None`) so callers can clear an inherited
    # `dataset.stats_path` from a resumed train_config.json by passing
    # `--dataset.stats_path=` (empty string). draccus interprets that
    # bare-equals as an empty string, not None, so the prior `is not None`
    # check would crash trying to load `Path("")`. The DAgger orchestrator's
    # weighted-sampling-mode finetune uses this idiom to disable the
    # legacy base-stats override (which silently clobbered
    # `MultiSourceNormalizingDataset`'s aggregation) — see the long comment
    # in `my_scripts/dagger_orchestrate.sh` for context.
    if cfg.dataset.stats_path:
        from pathlib import Path

        stats_path = Path(cfg.dataset.stats_path)
        logging.info(f"Overriding dataset stats from {stats_path}")
        dataset.meta.stats = cast_stats_to_numpy(load_json(stats_path))

    # --- policy (weight source decided by the resume rule) -------------------------------------
    # On resume, cfg was parsed FROM the checkpoint's train_config.json, so cfg.checkpoint_format
    # IS the recorded value: DCP-bearing formats skip the safetensors load here and stream the
    # sharded weights in after prepare (resume_after_prepare).
    defer_weight_load = cfg.resume and cfg.checkpoint_format.wants_dcp
    if cfg.is_reward_model_training:
        if is_main_process():
            logging.info("Creating reward model")
        from lerobot.rewards import make_reward_model

        policy = make_reward_model(
            cfg=cfg.reward_model,
            dataset_stats=dataset.meta.stats,
            dataset_meta=dataset.meta,
        )
        if not policy.is_trainable:
            raise ValueError(
                f"Reward model '{policy.name}' is zero-shot and cannot be trained via lerobot-train. "
                "Use it directly for inference via compute_reward() (e.g. offline precompute)."
            )
    else:
        if is_main_process():
            logging.info("Creating policy")
        policy = make_policy(
            cfg=cfg.policy,
            ds_meta=dataset.meta,
            rename_map=cfg.rename_map,
            defer_weight_load=defer_weight_load,
        )

    peft_model = None
    if cfg.peft is not None:
        if cfg.is_reward_model_training:
            raise ValueError("PEFT is only supported for policy training. ")
        require_package("peft", extra="peft")

        if isinstance(policy, PeftModel):
            logging.info("PEFT adapter already loaded from checkpoint, skipping wrap_with_peft.")
        else:
            logging.info("Using PEFT! Wrapping model.")
            peft_cli_overrides = dataclasses.asdict(cfg.peft)
            policy = policy.wrap_with_peft(peft_cli_overrides=peft_cli_overrides)
        peft_model = policy

    accelerator.wait_for_everyone()

    # --- processors (overrides built once, as one typed mapping) -------------------------------
    active_cfg = cfg.trainable_config
    processor_pretrained_path = active_cfg.pretrained_path

    processor_kwargs = ProcessorConfigKwargs()

    # Apply rename_map to dataset stats keys so the normalizer can find them after the rename step.
    # The rename step runs before normalization, so stats must be keyed by the post-rename names.
    # e.g. "observation.images.base_rgb_letterbox" -> "observation.images.base_rgb"
    #
    # In multi-dataset weighted-sampling mode, `dataset.meta.stats` is the
    # AGGREGATED stats computed by `MultiSourceNormalizingDataset` over all
    # sub-datasets' sidecars (min-of-mins, max-of-maxes, count-weighted
    # mean/std, etc.). The policy's normalize layer is fed those aggregated
    # stats and normalizes every frame ONCE. No special case needed here —
    # single-dataset and multi-dataset modes both flow through this same path.
    renamed_stats = {cfg.rename_map.get(k, k): v for k, v in dataset.meta.stats.items()}

    # observation_dim_slice → shrink the matching stats keys on the last axis
    # to match what the SelectObservationDimsProcessorStep will emit at
    # runtime. Without this, the normalizer holds full-width stats (e.g. [4]
    # for observation.state) but the tensor arriving at normalization time
    # has been sliced to [3] → shape mismatch on the (tensor - min) subtract.
    # Applied per stat key (min, max, mean, std, q01, ...); each stat value
    # is a tensor / ndarray whose last axis is the feature dim.
    #
    # Stats may be either torch.Tensor (from dataset.meta.stats natively) OR
    # numpy.ndarray (when --dataset.stats_path override is loaded via
    # cast_stats_to_numpy). Handle both — indexing the last axis with a slice
    # works uniformly across numpy and torch.
    if cfg.observation_dim_slice:
        import numpy as _np
        import torch as _torch

        for _obs_key, _keep_indices in cfg.observation_dim_slice.items():
            _target_key = cfg.rename_map.get(_obs_key, _obs_key)
            _feat_stats = renamed_stats.get(_target_key)
            if _feat_stats is None:
                continue
            _keep_list = list(_keep_indices)
            _max_idx = max(_keep_list)
            _new_feat_stats = {}
            for _stat_name, _stat_val in _feat_stats.items():
                _is_torch = isinstance(_stat_val, _torch.Tensor)
                _is_numpy = isinstance(_stat_val, _np.ndarray)
                if (_is_torch or _is_numpy) and _stat_val.ndim >= 1 and _stat_val.shape[-1] > _max_idx:
                    if _is_torch:
                        _idx = _torch.as_tensor(_keep_list, dtype=_torch.long, device=_stat_val.device)
                        _new_feat_stats[_stat_name] = _stat_val.index_select(dim=-1, index=_idx)
                    else:
                        _new_feat_stats[_stat_name] = _stat_val[..., _keep_list]
                else:
                    # Scalar stats (e.g. count) or per-pixel image stats — pass through.
                    _new_feat_stats[_stat_name] = _stat_val
            renamed_stats[_target_key] = _new_feat_stats

    if (processor_pretrained_path and not cfg.resume) or not processor_pretrained_path:
        processor_kwargs["dataset_stats"] = renamed_stats

    if cfg.is_reward_model_training:
        processor_kwargs["dataset_meta"] = dataset.meta

    if not cfg.is_reward_model_training:
        # Preprocessor/postprocessor override delivery differs between the
        # two paths through make_pre_post_processors (factory.py):
        #   * Resume path: PolicyProcessorPipeline.from_pretrained applies
        #     every override key wholesale (factory.py:338).
        #   * Fresh-init path: only rename_map is fished out and patched
        #     into the built pipeline post-hoc (factory.py:525-533). All
        #     other override keys are silently ignored on fresh init — the
        #     policy-specific factory builds normalizer/relative-actions
        #     steps from `dataset_stats` (passed separately) and from
        #     `policy_cfg` fields (`use_relative_actions`) directly.
        # So the two branches below carry exactly the overrides that
        # actually reach the pipeline in each mode. Do not add fresh-init
        # keys that the factory doesn't consume — they'd look like they
        # do something but don't.
        preprocessor_overrides: dict[str, Any] = {
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        }
        postprocessor_overrides: dict[str, Any] = {}
        if processor_pretrained_path is not None:
            preprocessor_overrides["device_processor"] = {"device": device.type}
            preprocessor_overrides["normalizer_processor"] = {
                "stats": renamed_stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            }
            postprocessor_overrides["unnormalizer_processor"] = {
                "stats": renamed_stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            }
            if getattr(active_cfg, "use_relative_actions", False):
                preprocessor_overrides["relative_actions_processor"] = {
                    "enabled": True,
                    "exclude_joints": getattr(active_cfg, "relative_exclude_joints", []),
                    "action_names": getattr(active_cfg, "action_feature_names", None),
                }
                postprocessor_overrides["absolute_actions_processor"] = {"enabled": True}
            # On resume, the checkpoint's saved processor stats are authoritative: they may have
            # been adapted by the policy (e.g. EVO1 pads state/action stats to max_state_dim),
            # and force-feeding raw dataset stats over them crashes normalization (#4006).
            if cfg.resume:
                preprocessor_overrides["normalizer_processor"].pop("stats", None)
                postprocessor_overrides["unnormalizer_processor"].pop("stats", None)
        processor_kwargs["preprocessor_overrides"] = preprocessor_overrides
        processor_kwargs["postprocessor_overrides"] = postprocessor_overrides
        # Observation dim slice: not delivered via preprocessor_overrides
        # because PolicyProcessorPipeline.from_pretrained validates that
        # every override key matches an existing saved step — a resume from
        # a checkpoint that predates this feature has no
        # `select_observation_dims_processor` in its saved pipeline, so the
        # override would fail (pipeline.py:_validate_overrides_used).
        # Instead we route the slice through a top-level kwarg that the
        # factory's post-hoc block reads, uniformly patching both fresh
        # init AND resume (inserting the step at position 0 when the saved
        # pipeline doesn't already carry it).
        if cfg.observation_dim_slice:
            processor_kwargs["observation_dim_slice"] = cfg.observation_dim_slice

    if cfg.is_reward_model_training:
        preprocessor, postprocessor = make_reward_pre_post_processors(
            cfg.reward_model,
            **processor_kwargs,
        )
    else:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=processor_pretrained_path,
            pretrained_revision=getattr(cfg.policy, "pretrained_revision", None),
            **processor_kwargs,
        )

    # Apply temporal-ensembling FIRST (innermost wrapper) so SA, if also enabled,
    # operates on smoothed chunks from TE's predict_action_chunk.
    te_cfg = getattr(cfg.policy, "temporal_ensemble_config", None)
    te_force_act = te_cfg is not None and getattr(te_cfg, "force_act_to_wrapper_mode", False)
    legacy_act_te = (
        getattr(cfg.policy, "type", None) == "act"
        and getattr(cfg.policy, "temporal_ensemble_coeff", None) is not None
        and not te_force_act  # user explicitly opted into the wrapper for ACT
    )
    if te_cfg is not None and te_cfg.enabled and not legacy_act_te:
        policy = _wrap_with_temporal_ensemble(policy, cfg.policy)

    sa_cfg = getattr(cfg.policy, "shared_autonomy_config", None)
    if sa_cfg is not None and sa_cfg.enabled:
        policy = _wrap_with_shared_autonomy(policy, cfg.policy)
        reconnect_relative_absolute_steps(preprocessor, policy.postprocessor, policy=policy)
    else:
        reconnect_relative_absolute_steps(preprocessor, postprocessor, policy=policy)

    # ── Debug: surface the normalization stats the policy ACTUALLY uses ──
    # This is ground truth AFTER all the layering that decides normalization:
    # single- vs multi-dataset, the norm_mode aggregation, any --dataset.stats_path
    # override, and (on resume) make_pre_post_processors loading saved processors +
    # load_state_dict's explicit-stats preservation. Reading the live
    # NormalizerProcessorStep.stats here tells you exactly what the policy
    # normalizes against — so you can confirm e.g. aggregated-multi-source vs
    # base-only, and catch a silent stats_path clobber.
    if is_main_process():
        try:
            import numpy as np  # not imported at module scope in this file

            from lerobot.processor.normalize_processor import (
                NormalizerProcessorStep,
                UnnormalizerProcessorStep,
            )

            def _log_active_norm_stats(pipeline, label: str) -> None:
                steps = getattr(pipeline, "steps", None) or []
                for step in steps:
                    if not isinstance(step, (NormalizerProcessorStep, UnnormalizerProcessorStep)):
                        continue
                    stats = getattr(step, "stats", None) or {}
                    logging.info(
                        "[norm-stats] %s %s: %d feature(s), _stats_explicitly_provided=%s",
                        label,
                        type(step).__name__,
                        len(stats),
                        getattr(step, "_stats_explicitly_provided", "?"),
                    )
                    for feat in sorted(stats):
                        fs = stats[feat] or {}
                        mn, mx = fs.get("min"), fs.get("max")
                        if "image" in feat or "pixel" in feat or mn is None or mx is None:
                            # images = IMAGENET stats; just confirm presence.
                            logging.info("[norm-stats]     %s: stat_keys=%s", feat, sorted(fs))
                            continue
                        mn = np.asarray(mn, dtype=np.float64).reshape(-1)
                        mx = np.asarray(mx, dtype=np.float64).reshape(-1)
                        logging.info(
                            "[norm-stats]     %s:\n        min=%s\n        max=%s",
                            feat,
                            np.array2string(mn, precision=4, suppress_small=True),
                            np.array2string(mx, precision=4, suppress_small=True),
                        )

            # `preprocessor` holds the input normalizer; the postprocessor holds
            # the action un-normalizer (under SA, it lives on policy.postprocessor).
            _log_active_norm_stats(preprocessor, "preprocessor")
            _post = getattr(policy, "postprocessor", None) or postprocessor
            _log_active_norm_stats(_post, "postprocessor")
            if cfg.dataset.repo_ids is not None:
                logging.info(
                    "[norm-stats] multi-dataset mode: norm_mode=%s over %d source(s); "
                    "the above is what the policy actually normalizes with.",
                    cfg.dataset.norm_mode,
                    len(cfg.dataset.repo_ids),
                )
        except Exception:
            logging.exception("[norm-stats] failed to introspect live normalizer stats (non-fatal)")

    # Created BEFORE prepare on the unsharded parameters — accelerate's FSDP2 path requires the
    # model and optimizer in one prepare() call and rebinds the param groups itself.
    if is_main_process():
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    # --- resume phase 1 + dataloaders ----------------------------------------------------------
    step = 0  # number of loop steps (= micro-batches consumed per data-parallel worker)
    if cfg.resume:
        step = resume_before_prepare(cfg)  # step + RNG only; sharded state loads after prepare

    dataloader, eval_dataloader = make_dataloaders(cfg, dataset, eval_dataset, step, parallel_dims)

    # Build a held-out-benchmark loss dataloader when configured. Uses the
    # SAME dataset the env-eval rolls out on (cfg.env.eval_benchmark_repo_id),
    # loaded as a LeRobotDataset with the same delta_timestamps the policy
    # was built with — so the batch shape matches what policy.forward expects.
    # This gives a per-frame LOSS on truly unseen scenarios (the benchmark is
    # a distinct dataset, unlike `eval_dataloader` which slices out episodes
    # from the training dataset). Logged as `eval_benchmark/loss` in wandb.
    # Silently no-op when the flag is disabled OR when no benchmark repo is
    # configured — safe to enable in shared training configs.
    eval_benchmark_dataloader = None
    _benchmark_repo = getattr(cfg.env, "eval_benchmark_repo_id", None) if cfg.env is not None else None
    if cfg.eval_benchmark_loss_freq > 0 and _benchmark_repo:
        if is_main_process:
            logging.info(
                f"[eval_benchmark_loss] enabling periodic loss on {_benchmark_repo} "
                f"every {cfg.eval_benchmark_loss_freq} steps "
                f"(max_batches={cfg.eval_benchmark_loss_max_batches or 'all'})"
            )
        # Reuse the same delta_timestamps the training dataset resolved with
        # so the returned batch shape matches what the policy consumes.
        from lerobot.datasets.factory import (
            resolve_delta_timestamps as _resolve_dt,
            unconsumed_camera_keys as _unconsumed_cams,
        )
        from lerobot.datasets.lerobot_dataset import (
            LeRobotDataset as _BenchDataset,
            LeRobotDatasetMetadata as _BenchMeta,
        )

        # Narrowing for pyright: eval_benchmark_loss_freq > 0 branch cannot
        # reach here without cfg.policy set (make_train_eval_datasets earlier
        # required it) and `dataset` bound.
        assert cfg.policy is not None
        # Same unconsumed-camera exclusion as make_dataset: benchmark repos in
        # image-mode (camera frames embedded in parquet rows) would otherwise
        # pay full per-row PNG materialization for tensors the loss never uses.
        _bench_meta = _BenchMeta(_benchmark_repo)
        _bench_excl = _unconsumed_cams(cfg.policy, _bench_meta, getattr(cfg, "rename_map", None))
        if _bench_excl and is_main_process:
            logging.info(f"[eval_benchmark_loss] excluding unconsumed camera feature(s): {_bench_excl}")
        _bench_ds = _BenchDataset(
            _benchmark_repo,
            delta_timestamps=_resolve_dt(cfg.policy, dataset.meta, exclude_keys=set(_bench_excl) or None),
            exclude_features=_bench_excl or None,
        )
        _bench_collate = lerobot_collate_fn if dataset.meta.has_language_columns else None
        # Deliberately NOT cfg.num_workers, and NOT persistent.
        #
        # This loader does a short burst of work (<= eval_benchmark_loss_max_batches)
        # once every eval_benchmark_loss_freq steps — typically minutes apart. Sized
        # like the train loader it would hold a SECOND full worker pool alive for the
        # entire run: with the `spawn` start method each worker re-imports torch and
        # reloads dataset metadata, so they cost ~1.5 GB EACH and never release it.
        # That is what OOM-killed a 30 GB box (15 live workers, ~23.7 GB, while the
        # benchmark pool sat idle >95% of the time).
        #
        # A small non-persistent pool pays a few seconds of worker startup per eval
        # instead — irrelevant next to the eval interval — and gives the memory back.
        _bench_workers = min(cfg.num_workers, 2)
        eval_benchmark_dataloader = torch.utils.data.DataLoader(
            _bench_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=_bench_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
            collate_fn=_bench_collate,
            prefetch_factor=cfg.prefetch_factor if _bench_workers > 0 else None,
            persistent_workers=False,
        )
    elif cfg.eval_benchmark_loss_freq > 0 and is_main_process:
        logging.warning(
            "[eval_benchmark_loss] eval_benchmark_loss_freq > 0 but env.eval_benchmark_repo_id "
            "is not set — skipping."
        )

    # --- prepare & resume phase 2 ---------------------------------------------------------------
    # The FSDP wrap-unit class names resolve right before prepare: user override, else the
    # policy's _fsdp_wrap_modules declaration — root-only wrapping is never silently accepted.
    set_fsdp_wrap_modules(accelerator, accelerator.unwrap_model(policy) if peft_model else policy)
    accelerator.wait_for_everyone()
    if eval_dataloader is not None:
        policy, optimizer, dataloader, lr_scheduler, eval_dataloader = accelerator.prepare(
            policy, optimizer, dataloader, lr_scheduler, eval_dataloader
        )
    else:
        policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            policy, optimizer, dataloader, lr_scheduler
        )
    finalize_sharded_policy(policy, parallel_dims)
    if cfg.resume:
        resume_after_prepare(cfg, accelerator, policy, optimizer, lr_scheduler)

    # Step counter at the start of THIS training-loop invocation. Fresh runs start
    # at 0; a resume starts at the restored step. env_eval_freq / eval_steps are
    # applied RELATIVE to this so a resume of N steps with freq=N evaluates at the
    # END of the run rather than at whichever global multiple lands inside it.
    initial_step_this_run = step

    # --- auxiliaries (after the core assembly, per the construction-order contract) -------------
    sample_weighter = None
    if cfg.sample_weighting is not None:
        from lerobot.utils.sample_weighting import make_sample_weighter

        if is_main_process():
            logging.info(f"Creating sample weighter: {cfg.sample_weighting.type}")
        sample_weighter = make_sample_weighter(
            cfg.sample_weighting,
            policy,
            device,
            dataset_root=cfg.dataset.root,
            dataset_repo_id=cfg.dataset.repo_id,
        )

    # --- banner (main process only; numel() reads metadata — on DTensors it is the GLOBAL shape,
    # so the totals are correct even after sharding) ---------------------------------------------
    # One loop step consumes one micro-batch on every dp worker; the optimizer sees
    # `samples_per_step x gradient_accumulation_steps` samples per update.
    samples_per_step = cfg.batch_size * parallel_dims.dp_world_size
    effective_batch_size = samples_per_step * cfg.accelerator.gradient_accumulation.steps
    if is_main_process():
        num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        num_total_params = sum(p.numel() for p in policy.parameters())
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
            logging.info("Creating environment processors")
            env_preprocessor, env_postprocessor = make_env_pre_post_processors(
                env_cfg=cfg.env, policy_cfg=cfg.policy
            )
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        logging.info(
            f"Effective batch size: {cfg.batch_size} x {parallel_dims.dp_world_size} dp workers "
            f"x {cfg.accelerator.gradient_accumulation.steps} grad accum = {effective_batch_size} "
            f"(topology: dp_replicate={parallel_dims.dp_replicate}, dp_shard={parallel_dims.dp_shard})"
        )
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # Multi-dataset weighted sampling: fork-only path. Explicitly gated on
    # cfg.dataset.use_weighted_sampling (not just `sample_weights is not None`)
    # so upstream users who happen to set sample_weights don't get silently
    # swapped onto a different sampler. Requires sample_weights to be set
    # (validated in DatasetConfig.__post_init__). Each frame in sub-dataset i
    # gets per-sample weight `sample_weights[i] / size_i`, so the expected
    # per-batch share matches `sample_weights[i]` exactly.
    # `replacement=True` is required for float weights and is correct semantics:
    # "every batch is X% intervention" is meaningful regardless of how few
    # intervention frames exist on disk.

    dl_iter = cycle(dataloader)
    policy.train()

    # EMA shadow of the policy weights (Chi et al. 2023, Diffusion Policy, section V.D). The shadow
    # lives on the main process only, which is safe under DDP where every rank holds identical
    # weights after each gradient sync. diffusers is imported lazily so the base training path does
    # not depend on it.
    ema = None
    if cfg.ema.enable:
        if parallel_dims.is_sharded:
            raise NotImplementedError(
                "--ema.enable=true is not supported with sharded training (FSDP2/HSDP/CP): the "
                "parameters are sharded across ranks. Use a replicated (DDP) or single-GPU run."
            )
        if cfg.peft is not None:
            raise NotImplementedError("--ema.enable=true is not supported together with PEFT adapters.")
        require_package("diffusers", extra="diffusion")
        if is_main_process():
            from diffusers.training_utils import EMAModel  # noqa: PLC0415

            # A constant --ema.decay is expressed through the schedule clamp: with
            # min_decay == max_decay, the warmup curve is pinned to that value at every step.
            min_decay = cfg.ema.min_decay if cfg.ema.decay is None else cfg.ema.decay
            max_decay = cfg.ema.max_decay if cfg.ema.decay is None else cfg.ema.decay
            ema = EMAModel(
                accelerator.unwrap_model(policy).parameters(),
                decay=max_decay,
                min_decay=min_decay,
                update_after_step=cfg.ema.update_after_step,
                use_ema_warmup=True,
                inv_gamma=cfg.ema.inv_gamma,
                power=cfg.ema.power,
            )
            ema.to(device)
            if cfg.ema.decay is not None:
                logging.info(
                    "EMA enabled: decay=%g (constant), update_after_step=%d, use_for_eval=%s",
                    cfg.ema.decay,
                    cfg.ema.update_after_step,
                    cfg.ema.use_for_eval,
                )
            else:
                logging.info(
                    "EMA enabled: max_decay=%g, inv_gamma=%g, power=%g, update_after_step=%d, use_for_eval=%s",
                    cfg.ema.max_decay,
                    cfg.ema.inv_gamma,
                    cfg.ema.power,
                    cfg.ema.update_after_step,
                    cfg.ema.use_for_eval,
                )
            if cfg.checkpoint_path is not None:
                ema_path = cfg.checkpoint_path / TRAINING_STATE_DIR / EMA_STATE_FILENAME
                if ema_path.exists():
                    ema.load_state_dict(torch.load(ema_path, map_location=device, weights_only=True))
                    logging.info("Resumed EMA shadow from %s", ema_path)
                else:
                    logging.warning(
                        "Resuming with --ema.enable=true but %s is missing; "
                        "restarting the shadow from the current weights.",
                        ema_path,
                    )

    train_metrics = {
        # Per-rank loss reflects only one shard of the global batch; mean recovers the loss the
        # data-parallel group is actually optimizing. grad_norm and lr are already identical on
        # every rank (post gradient sync / deterministic scheduler) so reducing them would be a
        # no-op collective.
        "loss": AverageMeter("loss", ":.3f", reduction="mean"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        # Report the slowest rank for bottleneck-style timings so multi-GPU runs surface the
        # true straggler instead of rank 0's view.
        "dataloading_s": AverageMeter("data_s", ":.3f", reduction="max"),
        "preprocessing_s": AverageMeter("prep_s", ":.3f", reduction="max"),
        "update_s": AverageMeter("updt_s", ":.3f", reduction="max"),
        "step_s": AverageMeter("step_s", ":.3f", reduction="max"),
        "samples_per_s": AverageMeter("smp/s", ":.0f"),
    }
    if torch.cuda.is_available():
        # max() because headroom is gated by the worst-case rank.
        train_metrics["gpu_mem_gb"] = AverageMeter("mem_gb", ":.2f", reduction="max")

    train_tracker = MetricsTracker(
        cfg.batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        dp_world_size=parallel_dims.dp_world_size,
    )

    if is_main_process():
        progbar = tqdm(
            total=cfg.steps - step,
            desc="Training",
            unit="step",
            disable=inside_slurm(),
            position=0,
            leave=True,
        )
        logging.info(
            f"Start offline training on a fixed dataset, with effective batch size: {effective_batch_size}"
        )

    # Persist the training config BEFORE the loop so an early crash (e.g. during
    # the first env eval / video save) still leaves a record of exactly how this
    # run was configured. save_checkpoint() also writes train_config.json, but
    # only at the first save step — which lands AFTER the first eval, so it's lost
    # if eval crashes first. This top-level snapshot is idempotent (overwritten by
    # each checkpoint's copy) and harmless on resume.
    if is_main_process and cfg.save_checkpoint:
        cfg.save_pretrained(cfg.output_dir)
        logging.info(f"Saved pre-training config snapshot: {cfg.output_dir / 'train_config.json'}")

    # Cache the observation-noise config once (dict lookups per-step are fine
    # but the None check should skip the whole inner loop). Only applied to
    # the TRAINING loop's batches — the eval-loss branch below deliberately
    # doesn't inject noise so eval-loss numbers reflect the clean data
    # distribution and stay comparable step-over-step.
    _obs_noise_std = cfg.dataset.observation_noise_std
    if _obs_noise_std and is_main_process:
        logging.info(f"[train] observation noise enabled: {_obs_noise_std}")

    # Eval envs live for the whole RUN, not per-eval. Building the sim is
    # expensive, and tearing it down mid-training is what triggers the
    # PyBullet/Tcl `Tcl_AsyncDelete` SIGABRT — which, if it fires between
    # evals, kills training before the final checkpoint exists. Created
    # lazily on the first eval so eval-less runs never build a sim, and
    # closed once at the end of train().
    _eval_envs_stack = ExitStack()
    _eval_envs: dict[str, dict[int, Any]] = {}

    for _ in range(step, cfg.steps):
        step_start = time.perf_counter()
        batch = next(dl_iter)
        preprocessing_start = time.perf_counter()
        train_tracker.dataloading_s = preprocessing_start - step_start
        for cam_key in dataset.meta.camera_keys:
            if cam_key in batch and batch[cam_key].dtype == torch.uint8:
                batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0
        # Per-key Gaussian noise on observation features — TRAIN-TIME ONLY.
        # Applied BEFORE the preprocessor so the noise σ is in raw feature
        # units (rad / meters), matching the units the user configures. The
        # preprocessor's normalizer then treats noisy inputs the same as
        # clean ones. Skipped when the key isn't in `batch` (e.g. an image
        # policy that doesn't consume env_state) so misconfigured keys are
        # a silent no-op rather than a KeyError mid-training.
        if _obs_noise_std:
            for _k, _s in _obs_noise_std.items():
                if (
                    _s > 0
                    and _k in batch
                    and isinstance(batch[_k], torch.Tensor)
                    and batch[_k].dtype.is_floating_point
                ):
                    batch[_k] = batch[_k] + torch.randn_like(batch[_k]) * _s
        batch = preprocessor(batch)
        train_tracker.preprocessing_s = time.perf_counter() - preprocessing_start

        train_tracker, _ = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
            sample_weighter=sample_weighter,
        )
        train_tracker.step_s = time.perf_counter() - step_start

        # Pull one optimizer step of the live weights into the EMA shadow (main process only).
        # The shadow tracks optimizer updates, not micro-batches: gate on the sync step under
        # gradient accumulation.
        if ema is not None and accelerator.sync_gradients:
            ema.step(accelerator.unwrap_model(policy).parameters())

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        if is_main_process():
            progbar.update(1)
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        # save_freq / env_eval_freq / eval_steps are applied RELATIVE to the
        # start of this training run (initial_step_this_run), not to the global
        # step counter. Fires every freq local steps AND unconditionally at
        # cfg.steps. Without this, a resume of N steps with freq=N would fire
        # mid-run at whichever multiple of freq happens to land inside the run
        # (e.g. a 75000->85000 finetune with save_freq=10000 saved at 80000
        # AND 85000), and skip the end.
        _local_step = step - initial_step_this_run
        is_saving_step = (_local_step > 0 and _local_step % cfg.save_freq == 0) or step == cfg.steps
        is_env_eval_step = cfg.env_eval_freq > 0 and (
            (_local_step > 0 and _local_step % cfg.env_eval_freq == 0) or step == cfg.steps
        )
        is_eval_step = (
            cfg.eval_steps > 0
            and eval_dataloader is not None
            and ((_local_step > 0 and _local_step % cfg.eval_steps == 0) or step == cfg.steps)
        )
        is_eval_benchmark_step = (
            cfg.eval_benchmark_loss_freq > 0
            and eval_benchmark_dataloader is not None
            and ((_local_step > 0 and _local_step % cfg.eval_benchmark_loss_freq == 0) or step == cfg.steps)
        )

        if is_log_step:
            # Collective reduce must run on every rank, before the main-process gate below.
            train_tracker.reduce_across_ranks()
            if is_main_process():
                if train_tracker.step_s.avg > 0:
                    train_tracker.samples_per_s = samples_per_step / train_tracker.step_s.avg
                logging.info(train_tracker)
                if wandb_logger:
                    # Policy sub-losses (latent_loss, action_loss, ...) are aggregated into the
                    # tracker by update_policy, so to_dict() already carries their windowed,
                    # rank-reduced averages — no per-step output_dict passthrough needed.
                    wandb_log_dict = train_tracker.to_dict()
                    # Log sample weighting statistics if enabled
                    if sample_weighter is not None:
                        weighter_stats = sample_weighter.get_stats()
                        wandb_log_dict.update({f"sample_weighting/{k}": v for k, v in weighter_stats.items()})
                    if ema is not None and ema.cur_decay_value is not None:
                        wandb_log_dict["ema/decay"] = ema.cur_decay_value
                        wandb_log_dict["ema/step"] = ema.optimization_step
                    wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if is_eval_step:
            policy.eval()
            eval_loss_sum = 0.0
            n_eval_batches = 0
            with torch.no_grad(), accelerator.autocast():
                for eval_batch in eval_dataloader:
                    for cam_key in dataset.meta.camera_keys:
                        if cam_key in eval_batch and eval_batch[cam_key].dtype == torch.uint8:
                            eval_batch[cam_key] = eval_batch[cam_key].to(dtype=torch.float32) / 255.0
                    eval_batch = preprocessor(eval_batch)
                    loss, _ = policy(eval_batch)  # __call__, so FSDP2 forward hooks run
                    eval_loss_sum += loss.item()
                    n_eval_batches += 1
            eval_loss = eval_loss_sum / max(n_eval_batches, 1)
            eval_loss = torch.tensor(eval_loss, device=device)
            eval_loss = accelerator.reduce(eval_loss, reduction="mean").item()
            policy.train()

            if is_main_process():
                logging.info(f"step {step}: eval_loss={eval_loss:.4f}")
                if wandb_logger:
                    wandb_logger.log_dict({"eval_loss": eval_loss}, step=step, mode="eval")

        if is_eval_benchmark_step:
            # Held-out-benchmark loss on the SAME dataset env-eval rolls out on.
            # Mirrors the is_eval_step branch above but the underlying dataset
            # is a distinct benchmark (never in training), so this loss is a
            # direct overfitting diagnostic vs train/loss. NO observation
            # noise is added here — we want the clean data-distribution loss
            # for a stable step-over-step comparison. `eval_benchmark_loss_max_batches`
            # bounds the periodic cost when the benchmark is large.
            policy.eval()
            _bench_loss_sum = 0.0
            _n_bench_batches = 0
            _max_batches = cfg.eval_benchmark_loss_max_batches or float("inf")
            with torch.no_grad(), accelerator.autocast():
                for bench_batch in eval_benchmark_dataloader:
                    if _n_bench_batches >= _max_batches:
                        break
                    for cam_key in dataset.meta.camera_keys:
                        if cam_key in bench_batch and bench_batch[cam_key].dtype == torch.uint8:
                            bench_batch[cam_key] = bench_batch[cam_key].to(dtype=torch.float32) / 255.0
                    bench_batch = preprocessor(bench_batch)
                    loss, _ = policy.forward(bench_batch)
                    _bench_loss_sum += loss.item()
                    _n_bench_batches += 1
            _bench_loss = _bench_loss_sum / max(_n_bench_batches, 1)
            _bench_loss = torch.tensor(_bench_loss, device=device)
            _bench_loss = accelerator.reduce(_bench_loss, reduction="mean").item()
            policy.train()

            if is_main_process:
                logging.info(
                    f"step {step}: eval_benchmark/loss={_bench_loss:.6f} "
                    f"(over {_n_bench_batches} batches of {_benchmark_repo})"
                )
                if wandb_logger:
                    wandb_logger.log_dict({"eval_benchmark/loss": _bench_loss}, step=step, mode="eval")

        if cfg.save_checkpoint and is_saving_step:
            # Collective: every rank participates (gathers / DCP shard writes); rank-0-only file
            # writes are gated inside save_checkpoint — no rank branches at the call site.
            if is_main_process():
                logging.info(f"Checkpoint policy after step {step}")
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                step=step,
                cfg=cfg,
                policy=policy,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                accelerator=accelerator,
            )
            if is_main_process():
                if ema is not None:
                    # Save the shadow for exact resume, plus a directly loadable copy of the EMA
                    # weights (lerobot-eval --policy.path=<checkpoint>/pretrained_model_ema).
                    torch.save(ema.state_dict(), checkpoint_dir / TRAINING_STATE_DIR / EMA_STATE_FILENAME)
                    unwrapped_policy = accelerator.unwrap_model(policy)
                    ema_dir = checkpoint_dir / f"{PRETRAINED_MODEL_DIR}_ema"
                    with _ema_weights(ema, unwrapped_policy):
                        unwrapped_policy.save_pretrained(ema_dir)
                        cfg.save_pretrained(ema_dir)
                        preprocessor.save_pretrained(ema_dir)
                        postprocessor.save_pretrained(ema_dir)
                update_last_checkpoint(checkpoint_dir)
                if cfg.save_checkpoint_to_hub:
                    push_checkpoint_to_hub(
                        checkpoint_dir,
                        cfg.policy.repo_id,
                        private=cfg.policy.private,
                    )
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)
            accelerator.wait_for_everyone()

        if cfg.env and is_env_eval_step:
            if is_main_process():
                step_id = get_step_identifier(step, cfg.steps)
                logging.info(f"Eval policy at step {step}")
                eval_policy_model = accelerator.unwrap_model(policy)
                # Evaluate the EMA weights when enabled: the swap happens only on the main
                # process (the other ranks wait at the barrier below) and is exactly undone
                # afterwards, so the live weights stay in sync across ranks.
                use_ema_for_eval = ema is not None and cfg.ema.use_for_eval
                if use_ema_for_eval:
                    logging.info("Evaluating the EMA weights")
                weights_cm = _ema_weights(ema, eval_policy_model) if use_ema_for_eval else nullcontext()
                if not _eval_envs:
                    _eval_envs.update(_eval_envs_stack.enter_context(_make_eval_envs(cfg)))
                eval_env = _eval_envs
                with weights_cm, torch.no_grad(), accelerator.autocast():
                    eval_info = eval_policy_all(
                        envs=eval_env,  # dict[suite][task_id] -> vec_env
                        policy=eval_policy_model,
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        n_episodes=cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                        max_episodes_rendered=5,
                        # Also save up to 5 failure videos that AREN'T already
                        # in the first 5 — debugging eval regressions otherwise
                        # only sees the first 5 episodes, which on a high-
                        # success policy are usually all wins.
                        max_episodes_rendered_failed=5,
                        start_seed=cfg.seed,
                        max_parallel_tasks=cfg.env.max_parallel_tasks,
                        close_envs_after_eval=False,  # training owns env lifetime; closed at end of train()
                    )
                # Persist the full eval_info to disk so downstream tooling
                # (dagger_progress.sh, dagger_plot.py, etc.) can access the
                # SAME per-task data the standalone lerobot-eval writes —
                # in particular per-task `successes` + `info_metrics.episode_length`
                # lists, which the wandb log's aggregated "Suite overall"
                # entry doesn't preserve. Same dict shape + same serializer
                # as `lerobot_eval.py:1002`. One file per eval step, keyed
                # by step_id, alongside the videos_step_<id>/ subdir.
                import json

                import numpy as np

                def _json_default(obj):
                    if isinstance(obj, np.generic):
                        return obj.item()
                    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

                eval_dir = cfg.output_dir / "eval"
                eval_dir.mkdir(parents=True, exist_ok=True)
                eval_info_path = eval_dir / f"eval_info_step_{step_id}.json"
                try:
                    with open(eval_info_path, "w") as f:
                        json.dump(eval_info, f, indent=2, default=_json_default)
                    logging.info("Wrote eval_info to %s", eval_info_path)
                except Exception as e:
                    # Non-fatal — training continues even if the dump
                    # fails. wandb log scrape is still the fallback.
                    logging.warning("Failed to write eval_info to %s: %s", eval_info_path, e)

                # overall metrics (suite-agnostic)
                aggregated = eval_info["overall"]

                # optional: per-suite logging
                for suite, suite_info in eval_info.items():
                    logging.info("Suite %s aggregated: %s", suite, suite_info)

                # Compact failure taxonomy per task: WHICH episodes failed and
                # WHY (collision vs timeout vs other), greppable from the
                # training log. The full per-episode telemetry (collision
                # steps, min goal distance, lengths) is already persisted in
                # eval/eval_info_step_<id>.json — this line exists so
                # scenario-class analyses don't require opening the JSON
                # (added 2026-09-06 for the dart-fix scenario taxonomy).
                try:
                    for _t in eval_info.get("per_task", []):
                        _m = _t.get("metrics", {})
                        _succ = _m.get("successes") or []
                        _im = _m.get("info_metrics", {})
                        _col_l = _im.get("in_collision") or []
                        _tr_l = _im.get("truncated") or []
                        _col = [i for i, s in enumerate(_succ) if not s and i < len(_col_l) and _col_l[i]]
                        _tout = [
                            i
                            for i, s in enumerate(_succ)
                            if not s and i < len(_tr_l) and _tr_l[i] and i not in _col
                        ]
                        _oth = [i for i, s in enumerate(_succ) if not s and i not in _col and i not in _tout]
                        logging.info(
                            "Failure taxonomy %s/%s: collision=%s timeout=%s other=%s",
                            _t.get("task_group"),
                            _t.get("task_id"),
                            _col,
                            _tout,
                            _oth,
                        )
                except Exception as _e:  # non-fatal: taxonomy is a convenience
                    logging.debug("failure-taxonomy logging skipped: %s", _e)

                # meters/tracker
                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size,
                    dataset.num_frames,
                    dataset.num_episodes,
                    eval_metrics,
                    initial_step=step,
                    dp_world_size=parallel_dims.dp_world_size,
                )
                eval_tracker.eval_s = aggregated.pop("eval_s")
                eval_tracker.avg_sum_reward = aggregated.pop("avg_sum_reward")
                eval_tracker.pc_success = aggregated.pop("pc_success")
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    # Log additional info_metrics from eval (e.g., avg_in_collision, avg_episode_length, etc.)
                    # These are any remaining keys in aggregated after popping the standard metrics
                    for metric_name, metric_value in aggregated.items():
                        if metric_name not in ("avg_max_reward", "n_episodes", "eval_ep_s", "video_paths"):
                            wandb_log_dict[metric_name] = metric_value
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    wandb_logger.log_video(eval_info["overall"]["video_paths"][0], step, mode="eval")

            accelerator.wait_for_everyone()

    if is_main_process():
        progbar.close()

    # Flush wandb BEFORE closing the eval envs: env teardown can SIGABRT
    # (PyBullet/Tcl), which skips atexit and would drop the still-uncommitted
    # final-step row (wandb only commits a step's data when a higher step is
    # logged or the run is finished — neither happens for the last step).
    if wandb_logger:
        wandb_logger.finish()

        logging.info("End of training")

    # --- publish (collective-safe: all ranks; the model commit gathers sharded weights) ---------
    if getattr(active_cfg, "push_to_hub", False):
        unwrapped = accelerator.unwrap_model(policy)
        model_to_publish = unwrapped.get_base_model() if peft_model is not None else unwrapped
        publish_trained_model(
            cfg,
            model_to_publish,
            preprocessor,
            postprocessor,
            dataset.meta,
            peft_model=unwrapped if peft_model is not None else None,
        )

        # The push above ships the live weights; when EMA is on, the weights that were
        # evaluated are the shadow, so push those too under a sibling `<repo_id>-ema` repo.
        # The shadow lives on the main process only, so this is rank-0-only by construction.
        # Non-fatal: the live model is already up if this fails.
        if ema is not None:
            ema_repo_id = f"{active_cfg.repo_id}-ema"
            orig_repo_id = unwrapped.config.repo_id
            try:
                unwrapped.config.repo_id = ema_repo_id
                with _ema_weights(ema, unwrapped):
                    unwrapped.push_model_to_hub(cfg, dataset_meta=dataset.meta)
                preprocessor.push_to_hub(ema_repo_id)
                postprocessor.push_to_hub(ema_repo_id)
                logging.info("Pushed EMA weights to %s", ema_repo_id)
            except Exception as exc:  # noqa: BLE001
                logging.warning("Failed to push EMA weights to %s: %s", ema_repo_id, exc)
            finally:
                unwrapped.config.repo_id = orig_repo_id

    # Close eval envs only now that the final checkpoint is on disk, so a
    # teardown abort can no longer cost us the run.
    _eval_envs_stack.close()

    # Properly clean up the distributed process group
    accelerator.wait_for_everyone()
    accelerator.end_training()


def _remote_target_in_argv() -> bool:
    """Detect a remote HF Jobs run request on the raw CLI, before draccus parsing.

    Returns:
        bool: True when the CLI requests a remote HF Jobs run (`--job.target=<non-local>`).
    """
    target = None
    args = sys.argv[1:]
    for i, tok in enumerate(args):
        if tok == "--job.target" and i + 1 < len(args):
            target = args[i + 1]
        elif tok.startswith("--job.target="):
            target = tok.split("=", 1)[1]
    return JobConfig.is_remote_target(target)


def main():
    register_third_party_plugins()
    if _remote_target_in_argv():
        # The policy device is resolved on the remote pod, not here, so silence the
        # client-side "Device '...' is not available" warning PreTrainedConfig emits
        # while parsing the config (it fires before train() can dispatch remotely).
        logging.getLogger("lerobot.configs.policies").setLevel(logging.ERROR)
    train()


if __name__ == "__main__":
    main()
