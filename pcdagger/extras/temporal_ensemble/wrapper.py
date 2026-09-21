#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Policy-agnostic temporal-ensemble wrapper.

Wraps any chunk-predicting policy (pi05, diffusion, ACT, ...) to apply ACT's
temporal-ensembling smoothing on top of its predicted action chunks. Pure
inference-time — the underlying model and training loop are untouched.

Mirrors the architecture of ``SharedAutonomyPolicyWrapper``: inherits from
``PreTrainedPolicy``, stores ``inner_policy``, intercepts ``select_action``
and ``reset``, delegates everything else.

The wrapper supports ``n_action_steps != 1`` for a speed/smoothness trade-off:
with ``n_action_steps=K``, the underlying model is queried every ``K`` steps
and the intervening ``K-1`` actions are read out of the smoothed ensembler
buffer (no new model calls between chunk boundaries).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.temporal_ensembler import TemporalEnsembler

if TYPE_CHECKING:
    from lerobot.configs.temporal_ensemble import TemporalEnsembleConfig

logger = logging.getLogger(__name__)


class _TEActionQueueProxy:
    """Length-and-clear proxy over a TE wrapper's intra-chunk counter.

    Exposed as ``wrapper._action_queue`` so two existing consumers Just Work:

    1. ``RelativeActionsProcessorStep._policy_queue_empty`` reads
       ``len(action_queue) == 0`` to detect chunk boundaries → returns True
       only when the wrapper is about to fetch a fresh chunk.
    2. ``SharedAutonomyPolicyWrapper._flush_inner_action_queue`` calls
       ``.clear()`` on the inner's queue → resets our counter so the next
       call fetches a fresh chunk (semantically: "discard cached chunk").

    Allocated once per wrapper; zero allocations per ``select_action`` call.
    """

    def __init__(self, wrapper: TemporalEnsemblePolicyWrapper) -> None:
        self._wrapper = wrapper

    def __len__(self) -> int:
        step = self._wrapper._intra_chunk_step
        if step == 0:
            return 0  # chunk boundary — next call fetches fresh chunk
        return self._wrapper._n_action_steps - step

    def clear(self) -> None:
        self._wrapper._intra_chunk_step = 0


class TemporalEnsemblePolicyWrapper(PreTrainedPolicy):
    """Wraps a chunk-predicting policy with ACT-style temporal ensembling.

    On each ``select_action`` call at a chunk boundary, runs the inner policy's
    ``predict_action_chunk`` to obtain a fresh chunk and pushes it through an
    online exponentially-weighted average maintained over time. Between chunk
    boundaries (when ``n_action_steps > 1``), pops smoothed actions directly
    from the ensembler's buffer without invoking the model.

    See ``TemporalEnsembler`` for the underlying ensembling algorithm and
    ``TemporalEnsembleConfig`` for the user-facing configuration.
    """

    config_class = PreTrainedConfig
    name = "temporal_ensemble_wrapper"

    def __init__(
        self,
        inner_policy: PreTrainedPolicy,
        te_cfg: TemporalEnsembleConfig,
    ) -> None:
        # Bypass PreTrainedPolicy.__init__ — we proxy the inner policy's config.
        nn.Module.__init__(self)
        self.config: PreTrainedConfig = inner_policy.config
        self.inner_policy = inner_policy
        self.te_cfg = te_cfg

        # Read n_action_steps from the inner policy's config at construction time
        # — every chunk-predicting policy in lerobot has this field.
        self._n_action_steps: int = int(getattr(inner_policy.config, "n_action_steps", 1))

        # Defer ensembler creation until first call so we can read chunk_size
        # from the chunk shape directly (uniform across pi05, diffusion, ACT).
        self._ensembler: TemporalEnsembler | None = None
        self._intra_chunk_step: int = 0

        # One allocation: shared proxy object reused for every queue check.
        self._action_queue = _TEActionQueueProxy(self)

        # Pinned noise for stochastic policies (pi05, diffusion). Sampled on
        # the first chunk-boundary call (lazy because we need the batch size),
        # reused for ``chunk_size`` outer steps, then re-sampled. The window
        # length matches the TE buffer's natural memory so within-window
        # ensembling stays coherent and across-window noise shifts give the
        # policy a chance to escape absorbing states. Cleared on reset().
        # ``None`` when ``pin_noise=False`` or when the inner doesn't expose
        # enough config to infer the shape.
        self._pinned_noise: Tensor | None = None
        # Counts *outer* steps (``select_action`` calls) since the last noise
        # sample. Compared against the inner's ``chunk_size`` to trigger a
        # refresh; reset on every refresh and on ``reset()``. Must count outer
        # steps (not chunk-boundary calls) because the refresh interval is
        # measured in real environment timesteps — with ``n_action_steps=K``,
        # one chunk-boundary call corresponds to K outer steps, so counting
        # calls would push the effective refresh interval to K * chunk_size.
        self._outer_steps_since_noise_refresh: int = 0

        # Cache whether the inner's ``select_action`` populates state that
        # ``predict_action_chunk`` reads. Diffusion needs it (populates
        # ``self._queues`` obs buffers); pi05/ACT don't (their select_action
        # is pure cache management around predict_action_chunk). Skipping the
        # redundant call halves chunk-boundary inference cost on pi05/ACT and
        # eliminates a wasted independent noise draw on pi05.
        self._inner_needs_select_action: bool = isinstance(getattr(inner_policy, "_queues", None), dict)

        logger.info(
            "TemporalEnsemblePolicyWrapper: coeff=%.4f, n_action_steps=%d. "
            "Model will be queried every %d step(s); smaller n_action_steps "
            "gives more smoothing at higher inference cost.",
            te_cfg.coeff,
            self._n_action_steps,
            self._n_action_steps,
        )

        # If the inner is ACT and it *also* has the legacy inline ensembler
        # configured (``temporal_ensemble_coeff`` set on ACTConfig), we have
        # two ensemblers running in series: ACT's inline one inside
        # ``inner.select_action`` (whose output the wrapper discards) and this
        # wrapper's own ensembler over the chunk from ``predict_action_chunk``.
        # Only reachable via ``TemporalEnsembleConfig.force_act_to_wrapper_mode=True``
        # (otherwise lerobot_{eval,train} skips wrapping ACT on the legacy path).
        # Surface this loudly so A/B tests are unambiguous.
        inner_legacy_coeff = getattr(inner_policy.config, "temporal_ensemble_coeff", None)
        if inner_legacy_coeff is not None and getattr(te_cfg, "force_act_to_wrapper_mode", False):
            logger.warning(
                "TemporalEnsemblePolicyWrapper: force_act_to_wrapper_mode=True. "
                "Inner ACT has temporal_ensemble_coeff=%.4f set; its inline "
                "ensembler will still run inside inner.select_action (output "
                "discarded by the wrapper), and THIS wrapper (coeff=%.4f) "
                "produces the executed action from inner.predict_action_chunk. "
                "Expect ~2x inference cost per chunk-boundary step. Use only "
                "for A/B-testing wrapper equivalence vs the inline path.",
                inner_legacy_coeff,
                te_cfg.coeff,
            )

    # ------------------------------------------------------------------
    # Inference path
    # ------------------------------------------------------------------

    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Return one ensembled action.

        Chunk-boundary steps (``_intra_chunk_step == 0``) advance the ensembler
        by one new chunk. Mid-chunk steps return the next slice of the smoothed
        forward buffer without invoking the model.
        """
        if self._intra_chunk_step == 0:
            # Chunk boundary: fetch a fresh chunk and advance the ensembler.
            #
            # Only call inner.select_action when the inner relies on it to
            # populate state (e.g. diffusion's obs queues). pi05 and ACT
            # don't — their select_action is pure cache management around
            # predict_action_chunk — so skipping the call halves chunk-boundary
            # cost. For stochastic policies (pi05 flow matching) it also
            # eliminates a wasted independent noise draw whose chunk we'd
            # immediately discard.
            if self._inner_needs_select_action:
                _ = self.inner_policy.select_action(batch, **kwargs)
                # Drop the inner's cached action so its queue length reflects
                # a fresh chunk on the next outer call (matters for the
                # relative-action processor's queue-empty anchor refresh).
                self._flush_inner_action_queue()

            inner_kwargs = self._build_inner_kwargs(batch, kwargs)
            chunk = self.inner_policy.predict_action_chunk(batch, **inner_kwargs)
            # Lazy ensembler init: read chunk_size from the actual chunk so we
            # uniformly handle pi05 (chunk_size), diffusion (n_action_steps),
            # ACT (chunk_size).
            if self._ensembler is None:
                self._ensembler = TemporalEnsembler(self.te_cfg.coeff, chunk.shape[1])
                if self._n_action_steps > chunk.shape[1]:
                    raise ValueError(
                        f"TemporalEnsemblePolicyWrapper requires n_action_steps "
                        f"({self._n_action_steps}) <= chunk size ({chunk.shape[1]}); "
                        f"check the inner policy's config."
                    )
            else:
                # K>1: we've consumed (n_action_steps - 1) actions from the buffer
                # since the last update via direct indexing. Tell the ensembler that
                # that much real time has elapsed so the new chunk integrates with
                # the buffer entries that correspond to the same future timesteps
                # (not the entries the ensembler would otherwise treat as "now").
                self._ensembler.skip(self._n_action_steps - 1)
            action = self._ensembler.update(chunk)
        else:
            # Mid-chunk: read the next entry from the smoothed forward buffer.
            #
            # After ``update()`` the buffer has length ``chunk_size - 1`` with
            # buffer[i] = action for step (last_update_step + 1 + i). Our
            # ``_intra_chunk_step`` k in [1, n_action_steps-1] wants the action
            # for step (last_update_step + k) = buffer[k - 1].
            assert self._ensembler is not None  # always created on the boundary path
            action = self._ensembler.ensembled_actions[:, self._intra_chunk_step - 1].clone()

        # Advance counters: intra-chunk position wraps at K; outer-step
        # counter monotonically increments and drives noise-refresh timing
        # (read inside ``_build_inner_kwargs``).
        self._intra_chunk_step = (self._intra_chunk_step + 1) % self._n_action_steps
        self._outer_steps_since_noise_refresh += 1
        return action

    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Return the smoothed forward action buffer.

        Used by outer wrappers (e.g. ``SharedAutonomyPolicyWrapper`` when its
        inner is a TE wrapper) that want to operate on the smoothed chunk
        rather than the raw underlying policy output.

        Returns a *clone* of the ensembler buffer because the buffer is
        mutated in-place by the next ``update()`` call (the ensembler uses
        ``*=`` and ``+=`` on it).
        """
        if self._ensembler is None or self._ensembler.ensembled_actions is None:
            # No chunk has been observed yet — fall back to the raw inner chunk.
            return self.inner_policy.predict_action_chunk(batch, **kwargs)
        return self._ensembler.ensembled_actions.clone()

    def reset(self) -> None:
        """Reset per-episode state (called at the start of each episode)."""
        self.inner_policy.reset()
        if self._ensembler is not None:
            self._ensembler.reset()
        self._intra_chunk_step = 0
        # Re-sample the pinned noise on the next chunk-boundary call so each
        # episode starts on its own flow trajectory.
        self._pinned_noise = None
        self._outer_steps_since_noise_refresh = 0

    def _build_inner_kwargs(self, batch: dict[str, Tensor], kwargs: dict) -> dict:
        """Inject ``noise=...`` into kwargs when ``pin_noise=True`` and shapes are known.

        Lazily samples the noise tensor on the first chunk-boundary call of
        each episode using the inner config's ``chunk_size`` + ``max_action_dim``
        (pi05's exact noise shape). Reused for the rest of the episode so the
        ensembler smooths along a coherent flow trajectory instead of averaging
        independent posterior samples.
        """
        if not getattr(self.te_cfg, "pin_noise", False):
            return kwargs
        # Refresh every chunk_size *outer steps* so the policy gets a chance
        # to switch modes (escapes absorbing states) without breaking within-
        # window ensembler coherence. The counter is incremented in
        # ``select_action`` once per outer step, so this interval is the same
        # whether ``n_action_steps`` is 1 or larger.
        refresh_interval = int(getattr(self.inner_policy.config, "chunk_size", 0) or 0)
        if (
            self._pinned_noise is not None
            and refresh_interval > 0
            and self._outer_steps_since_noise_refresh >= refresh_interval
        ):
            self._pinned_noise = None  # triggers re-sample below
        if self._pinned_noise is None:
            self._pinned_noise = self._try_sample_pinned_noise(batch)
            self._outer_steps_since_noise_refresh = 0
            if self._pinned_noise is None:
                # Inner policy doesn't expose pi05-shaped noise config; skip
                # pinning silently. (ACT lands here and ignores noise anyway.)
                return kwargs
        # Don't override an explicit noise kwarg from the caller (e.g. SA).
        if "noise" in kwargs:
            return kwargs
        out = dict(kwargs)
        out["noise"] = self._pinned_noise
        return out

    def _try_sample_pinned_noise(self, batch: dict[str, Tensor]) -> Tensor | None:
        """Return a freshly-sampled noise tensor matching pi05's expected shape, or None.

        Shape: ``(batch_size, chunk_size, max_action_dim)`` in float32 —
        matches ``Pi05Policy.model.sample_noise`` so the inner uses our tensor
        directly with no reshape/cast. Returns None if the inner config lacks
        ``max_action_dim`` (e.g. ACT, diffusion-with-different-conventions),
        in which case noise pinning is skipped silently.
        """
        inner_cfg = self.inner_policy.config
        chunk_size = getattr(inner_cfg, "chunk_size", None)
        max_action_dim = getattr(inner_cfg, "max_action_dim", None)
        if chunk_size is None or max_action_dim is None:
            return None
        ref = next((v for v in batch.values() if isinstance(v, Tensor)), None)
        if ref is None:
            return None
        noise = torch.randn(
            (ref.shape[0], int(chunk_size), int(max_action_dim)),
            device=ref.device,
            dtype=torch.float32,
        )
        logger.info(
            "TemporalEnsemblePolicyWrapper: pinned noise sampled "
            "(shape=%s, device=%s) — successive chunks will share this "
            "flow-matching trajectory until reset().",
            tuple(noise.shape),
            noise.device,
        )
        return noise

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flush_inner_action_queue(self) -> None:
        """Drop the inner policy's cached actions without resetting obs queues.

        Mirrors ``SharedAutonomyPolicyWrapper._flush_inner_action_queue``:
        handles both ``_action_queue`` (pi05, ACT-in-queue-mode) and
        ``_queues[ACTION]`` (diffusion) patterns.

        Some policies don't cache an action between ``select_action`` calls
        — ACT in inline-temporal-ensembler mode does a fresh forward pass
        every step and only its inline ensembler holds state, never an action
        queue. For those the flush is a no-op (nothing to clear), which is
        correct because the next ``inner.select_action`` will re-run the
        model anyway.
        """
        inner = self.inner_policy
        action_q = getattr(inner, "_action_queue", None)
        if action_q is not None and hasattr(action_q, "clear"):
            action_q.clear()
        queues = getattr(inner, "_queues", None)
        if isinstance(queues, dict):
            from lerobot.utils.constants import ACTION

            q = queues.get(ACTION)
            if q is not None and hasattr(q, "clear"):
                q.clear()

    # ------------------------------------------------------------------
    # Delegation
    # ------------------------------------------------------------------

    def forward(self, batch, **kwargs):
        return self.inner_policy.forward(batch, **kwargs)

    def get_optim_params(self):
        return self.inner_policy.get_optim_params()

    def train(self, mode: bool = True):
        self.inner_policy.train(mode)
        return self

    def eval(self):
        self.inner_policy.eval()
        return self

    def parameters(self, recurse: bool = True):
        return self.inner_policy.parameters(recurse)

    def to(self, *args, **kwargs):
        self.inner_policy.to(*args, **kwargs)
        return self

    def use_original_modules(self):
        # For video saving compatibility (lerobot_eval.py:280).
        if hasattr(self.inner_policy, "use_original_modules"):
            self.inner_policy.use_original_modules()
