#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Policy-agnostic temporal ensembler.

Extracted from ``policies/act/modeling_act.py`` so it can be reused by
``TemporalEnsemblePolicyWrapper`` for any chunk-predicting policy.

Implements Algorithm 2 of https://huggingface.co/papers/2304.13705 — an
online exponentially-weighted average over overlapping action-chunk
predictions. Each call to ``update(chunk)`` advances the running average
by one step and returns the oldest (most-averaged) action.
"""

import torch
from torch import Tensor


class TemporalEnsembler:
    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
        """Temporal ensembling as described in Algorithm 2 of https://huggingface.co/papers/2304.13705.

        The weights are calculated as wᵢ = exp(-temporal_ensemble_coeff * i) where w₀ is the oldest action.
        They are then normalized to sum to 1 by dividing by Σwᵢ. Here's some intuition around how the
        coefficient works:
            - Setting it to 0 uniformly weighs all actions.
            - Setting it positive gives more weight to older actions.
            - Setting it negative gives more weight to newer actions.
        NOTE: The default value for `temporal_ensemble_coeff` used by the original ACT work is 0.01. This
        results in older actions being weighed more highly than newer actions (the experiments documented in
        https://github.com/huggingface/lerobot/pull/319 hint at why highly weighing new actions might be
        detrimental: doing so aggressively may diminish the benefits of action chunking).

        Here we use an online method for computing the average rather than caching a history of actions in
        order to compute the average offline. For a simple 1D sequence it looks something like:

        ```
        import torch

        seq = torch.linspace(8, 8.5, 100)
        print(seq)

        m = 0.01
        exp_weights = torch.exp(-m * torch.arange(len(seq)))
        print(exp_weights)

        # Calculate offline
        avg = (exp_weights * seq).sum() / exp_weights.sum()
        print("offline", avg)

        # Calculate online
        for i, item in enumerate(seq):
            if i == 0:
                avg = item
                continue
            avg *= exp_weights[:i].sum()
            avg += item * exp_weights[i]
            avg /= exp_weights[: i + 1].sum()
        print("online", avg)
        ```
        """
        self.chunk_size = chunk_size
        self.ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size))
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self.reset()

    def reset(self):
        """Resets the online computation variables."""
        self.ensembled_actions = None
        # (chunk_size,) count of how many actions are in the ensemble for each time step in the sequence.
        self.ensembled_actions_count = None

    def update(self, actions: Tensor) -> Tensor:
        """
        Takes a (batch, chunk_size, action_dim) sequence of actions, update the temporal ensemble for all
        time steps, and pop/return the next batch of actions in the sequence.

        The buffer's *current* length determines how much of the new chunk overlaps with existing
        predictions. In the K=1 cadence (call every step), the buffer has length ``chunk_size - 1``
        after each pop, and integration uses ``actions[:, :-1]`` exactly as in the original ACT
        algorithm. For K>1 cadence (called every K steps after the wrapper's ``skip(K-1)`` has
        shortened the buffer to length ``chunk_size - K``), the first ``chunk_size - K`` entries of
        the new chunk integrate with the existing buffer (these are predictions for the same future
        timesteps from two different chunks); the last K entries append fresh.
        """
        self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
        self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(device=actions.device)
        if self.ensembled_actions is None:
            # Initializes `self._ensembled_action` to the sequence of actions predicted during the first
            # time step of the episode.
            self.ensembled_actions = actions.clone()
            # Note: The last dimension is unsqueeze to make sure we can broadcast properly for tensor
            # operations later.
            self.ensembled_actions_count = torch.ones(
                (self.chunk_size, 1), dtype=torch.long, device=self.ensembled_actions.device
            )
        else:
            # Existing buffer has shape (batch_size, overlap, action_dim) where overlap ∈ [0, chunk_size).
            # The first ``overlap`` entries of the new chunk align with the existing buffer (same real
            # timesteps); the remaining entries are predictions for timesteps the buffer hasn't seen.
            overlap = self.ensembled_actions.shape[1]
            if overlap > 0:
                self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
                self.ensembled_actions += (
                    actions[:, :overlap] * self.ensemble_weights[self.ensembled_actions_count]
                )
                self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
                self.ensembled_actions_count = torch.clamp(
                    self.ensembled_actions_count + 1, max=self.chunk_size
                )
            if overlap < self.chunk_size:
                # Append the fresh entries (chunk indices [overlap..chunk_size)) — these are predictions
                # for future timesteps the buffer hadn't covered yet, so they start with count = 1.
                fresh_actions = actions[:, overlap:]
                fresh_counts = torch.ones(
                    (self.chunk_size - overlap, 1),
                    dtype=torch.long,
                    device=actions.device,
                )
                if overlap == 0:
                    self.ensembled_actions = fresh_actions.clone()
                    self.ensembled_actions_count = fresh_counts
                else:
                    self.ensembled_actions = torch.cat([self.ensembled_actions, fresh_actions], dim=1)
                    self.ensembled_actions_count = torch.cat([self.ensembled_actions_count, fresh_counts])
        # "Consume" the first action.
        action, self.ensembled_actions, self.ensembled_actions_count = (
            self.ensembled_actions[:, 0],
            self.ensembled_actions[:, 1:],
            self.ensembled_actions_count[1:],
        )
        return action

    def skip(self, k: int) -> None:
        """Advance the ensembler's notion of time by ``k`` steps without
        integrating any new chunk. Pops ``k`` entries from the front of the
        buffer (they've already been consumed externally via direct buffer
        indexing in the wrapper's intra-chunk path).

        This is what makes K-step batched execution coherent. Without it, the
        next ``update()`` would integrate a chunk meant for "real time T+K"
        with buffer entries the ensembler thinks are for "ensembler time 1",
        producing a Frankenstein-average of predictions for different
        real-world timesteps.
        """
        if k <= 0 or self.ensembled_actions is None:
            return
        self.ensembled_actions = self.ensembled_actions[:, k:]
        self.ensembled_actions_count = self.ensembled_actions_count[k:]
