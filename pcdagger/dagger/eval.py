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
"""Evaluate a policy on an environment by running rollouts and computing metrics.

Requires: pip install 'lerobot[evaluation]' plus the policy extra (e.g. lerobot[pi])
          and the environment extra (e.g. lerobot[pusht]) if evaluating in simulation.

Usage examples:

You want to evaluate a model from the hub (eg: https://huggingface.co/lerobot/diffusion_pusht)
for 10 episodes.

```
pcdagger-eval \
    --policy.path=lerobot/diffusion_pusht \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --policy.use_amp=false \
    --policy.device=cuda
```

OR, you want to evaluate a model checkpoint from the LeRobot training script for 10 episodes.
```
pcdagger-eval \
    --policy.path=outputs/train/diffusion_pusht/checkpoints/005000/pretrained_model \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --policy.use_amp=false \
    --policy.device=cuda
```

Note that in both examples, the repo/folder should contain at least `config.json` and `model.safetensors` files.

You can learn about the CLI options for this script in the `EvalPipelineConfig` in lerobot/configs/eval.py
"""

import concurrent.futures as cf
import faulthandler
import json
import logging
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
from functools import partial
from pathlib import Path
from pprint import pformat
from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from pcdagger.dagger.intervention import InterventionContext

import einops
import gymnasium as gym
import numpy as np
import torch
from termcolor import colored
from torch import Tensor, nn
from tqdm import trange

from lerobot.configs import FeatureType, parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs import (
    check_env_attributes_and_types,
    close_envs,
    make_env,
    make_env_pre_post_processors,
    preprocess_observation,
)
from lerobot.envs.utils import NEW_ROLLOUT_OPTION
from lerobot.lerobot_types import PolicyAction
from lerobot.policies import PreTrainedPolicy, make_policy, make_pre_post_processors
from pcdagger.compat import peft_available, reconnect_relative_absolute_steps
from pcdagger.lerobot_glue.policy import (
    _wrap_with_last_mile,
    _wrap_with_shared_autonomy,
    _wrap_with_temporal_ensemble,
)
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import ACTION, DONE, OBS_IMAGE, OBS_IMAGES, OBS_STR, REWARD
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.import_utils import register_third_party_plugins, require_package
from lerobot.utils.io_utils import write_video
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import (
    init_logging,
    inside_slurm,
)

if TYPE_CHECKING or peft_available():
    from peft import PeftModel
else:
    PeftModel = None


logger = logging.getLogger(__name__)

# Print a Python traceback if the process receives SIGSEGV / SIGABRT / SIGFPE
# / SIGBUS / SIGILL. Without this, native crashes (pybullet, CUDA, etc.)
# show up as a bare "Aborted (core dumped)" in the terminal with no stack
# trace from Python. Side-effect-only at import time; safe to call once.
faulthandler.enable(file=sys.stderr, all_threads=True)


def _find_last_mile_wrapper(policy):
    """Walk the wrapper chain to find a LastMileWrapper instance, or None.

    Used to apply the last-mile help in raw joint space after the
    postprocessor. Returns None if the wrapper isn't in the chain (e.g. flag
    disabled), so the call site is a cheap no-op when off.
    """
    from pcdagger.extras.last_mile import LastMileWrapper

    p = policy
    while p is not None:
        if isinstance(p, LastMileWrapper):
            return p
        p = getattr(p, "inner_policy", None)
    return None


def _find_shared_autonomy_wrapper(policy):
    """Walk the wrapper chain to find a SharedAutonomyPolicyWrapper instance, or None.

    Used in intervention mode to set headless flags (`auto_pause_on_rrt_finish`,
    `_run_event.set()`) and to hand the env handle to the RRT source's
    pre-execution teleport.
    """
    from pcdagger.blend.wrapper import SharedAutonomyPolicyWrapper

    p = policy
    while p is not None:
        if isinstance(p, SharedAutonomyPolicyWrapper):
            return p
        p = getattr(p, "inner_policy", None)
    return None


def _extract_state_version(info: dict | None) -> int | None:
    """Pull the env-mutation clock stamp out of a (vector-)env info dict.

    Gym vector envs aggregate per-env infos into arrays keyed by the same
    name; a plain env returns the scalar directly. Intervention mode runs
    batch_size=1, so index 0 is THE env. Returns None when the env/server
    doesn't stamp observations (pre-state_version SplatSim, non-splatsim
    envs) so consumers can fall back to assuming freshness.
    """
    if not isinstance(info, dict):
        return None
    ver = info.get("state_version")
    if ver is None:
        return None
    try:
        arr = np.asarray(ver).reshape(-1)
        return int(arr[0]) if arr.size else None
    except (TypeError, ValueError):
        return None


def _env_features_to_dataset_features(env_features: dict) -> dict:
    """Convert EnvConfig.features to the dict format expected by LeRobotDataset.create()."""
    features = {}
    for key, ft in env_features.items():
        shape = tuple(ft.shape)
        if ft.type is FeatureType.VISUAL:
            features[key] = {"dtype": "video", "shape": shape, "names": ["height", "width", "channel"]}
        else:
            features[key] = {"dtype": "float32", "shape": shape, "names": None}
    features["next.reward"] = {"dtype": "float32", "shape": (1,), "names": None}
    features["next.success"] = {"dtype": "bool", "shape": (1,), "names": None}
    features["next.done"] = {"dtype": "bool", "shape": (1,), "names": None}
    return features


def _build_raw_frame(
    raw_obs: dict,
    env_idx: int,
    action: np.ndarray,
    reward: float,
    success: bool,
    done: bool,
    task: str,
    env_features: dict,
) -> dict:
    """Build a dataset frame from raw env observations for one env index.

    Keys in the frame match the keys in env_features so they align with the
    dataset schema created by _env_features_to_dataset_features().
    """
    frame: dict[str, Any] = {}
    for key in env_features:
        if key == ACTION:
            continue
        if key.startswith("next."):
            continue
        if "pixels" in raw_obs and isinstance(raw_obs["pixels"], dict):
            for cam_name, img in raw_obs["pixels"].items():
                candidate = f"{OBS_IMAGES}.{cam_name}"
                if candidate == key:
                    frame[key] = img[env_idx]
            if key in frame:
                continue
        if "pixels" in raw_obs and not isinstance(raw_obs["pixels"], dict) and key in ("pixels", OBS_IMAGE):
            frame[key] = raw_obs["pixels"][env_idx]
            continue
        if key in raw_obs and isinstance(raw_obs[key], np.ndarray):
            val = raw_obs[key][env_idx]
            if val.dtype == np.float64:
                val = val.astype(np.float32)
            frame[key] = val
    frame[ACTION] = action
    frame["next.reward"] = np.atleast_1d(np.float32(reward))
    frame["next.success"] = np.atleast_1d(np.bool_(success))
    frame["next.done"] = np.atleast_1d(np.bool_(done))
    frame["task"] = task
    return frame


def rollout(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    seeds: list[int] | None = None,
    scenario_indices: list[int] | None = None,
    return_observations: bool = False,
    render_callback: Callable[[gym.vector.VectorEnv], None] | None = None,
    recording_dir: Path | None = None,
    env_features: dict | None = None,
    recording_repo_id: str | None = None,
    recording_private: bool = False,
    predicted_latents_callback: Callable[[PreTrainedPolicy], None] | None = None,
    intervention_ctx: "InterventionContext | None" = None,
) -> dict:
    """Run a batched policy rollout once through a batch of environments.

    Note that all environments in the batch are run until the last environment is done. This means some
    data will probably need to be discarded (for environments that aren't the first one to be done).

    The return dictionary contains:
        (optional) "observation": A dictionary of (batch, sequence + 1, *) tensors mapped to observation
            keys. NOTE that this has an extra sequence element relative to the other keys in the
            dictionary. This is because an extra observation is included for after the environment is
            terminated or truncated.
        "action": A (batch, sequence, action_dim) tensor of actions applied based on the observations (not
            including the last observations).
        "reward": A (batch, sequence) tensor of rewards received for applying the actions.
        "success": A (batch, sequence) tensor of success conditions (the only time this can be True is upon
            environment termination/truncation).
        "done": A (batch, sequence) tensor of **cumulative** done conditions. For any given batch element,
            the first True is followed by True's all the way till the end. This can be used for masking
            extraneous elements from the sequences above.

    Args:
        env: The batch of environments.
        policy: The policy. Must be a PyTorch nn module.
        seeds: The environments are seeded once at the start of the rollout. If provided, this argument
            specifies the seeds for each of the environments.
        scenario_indices: For envs that select scenarios from an eval benchmark subset, the absolute
            rollout index of each env in this batch. Forwarded to ``env.reset(options=...)`` as
            ``{"benchmark_start_index": idx}`` per env. Decouples scenario selection from any stateful
            per-server counter — SplatSim's ``_handle_reset`` uses this option to force
            ``subset[idx % len(subset)]`` regardless of prior counter state. Non-SplatSim envs
            silently ignore the option.
        return_observations: Whether to include all observations in the returned rollout data. Observations
            are returned optionally because they typically take more memory to cache. Defaults to False.
        render_callback: Optional rendering callback to be used after the environments are reset, and after
            every step.
        predicted_latents_callback: Optional callback invoked after every ``select_action`` with the policy
            itself. World-model policies (e.g. LingBot-VA) stash predicted video latents on
            ``policy.last_predicted_latents``; this lets the caller concatenate chunks and decode once.
    Returns:
        The dictionary described above.
    """
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."

    # Reset the policy and environments. When scenario_indices are provided
    # AND the vector env has a single sub-env, pass benchmark_start_index in
    # options so SplatSim (or any env that honors the key) plays a
    # DETERMINISTIC scenario per rollout, immune to counter drift from
    # partial prior sessions.
    #
    # Gymnasium's SyncVectorEnv.reset fans a single `options` dict out to
    # every sub-env, so per-env options aren't natively supported. For
    # num_envs > 1 we skip the override — each parallel env is expected to
    # be a fresh independent server (own port + own counter), where drift
    # is not a cross-session concern. Intervention mode is single-env
    # (validated upstream), so it's covered. Non-SplatSim envs silently
    # ignore an unknown option key.
    policy.reset()
    # NEW_ROLLOUT_OPTION tells FreezeAfterEpisodeEnd this is a genuine new episode, as
    # opposed to Gymnasium's argument-less autoreset of a sub-env that already finished.
    reset_options: dict[str, Any] = {NEW_ROLLOUT_OPTION: True}
    if scenario_indices is not None and env.num_envs == 1:
        assert len(scenario_indices) == 1, (
            f"scenario_indices length {len(scenario_indices)} != env.num_envs 1"
        )
        reset_options["benchmark_start_index"] = int(scenario_indices[0])
    observation, info = env.reset(seed=seeds, options=reset_options)
    if render_callback is not None:
        render_callback(env)

    recording_datasets: list[LeRobotDataset] | None = None
    raw_observation = None
    task_desc = ""
    if recording_dir is not None and env_features is not None:
        features = _env_features_to_dataset_features(env_features)
        fps = env.unwrapped.metadata.get("render_fps", 30)
        recording_datasets = []
        multi_env = env.num_envs > 1
        base_repo_id = recording_repo_id or "eval_recording"
        for i in range(env.num_envs):
            root = str(recording_dir / f"env_{i}") if multi_env else str(recording_dir)
            repo_id = f"{base_repo_id}_env_{i}" if multi_env else base_repo_id
            recording_datasets.append(
                LeRobotDataset.create(
                    repo_id=repo_id,
                    fps=fps,
                    features=features,
                    root=root,
                    use_videos=True,
                )
            )
        raw_observation = deepcopy(observation)
        try:
            task_desc = list(env.call("task_description"))[0]
        except (AttributeError, NotImplementedError):
            task_desc = ""

    # ── intervention-mode per-scenario setup ──────────────────────────────── #
    # When `intervention_ctx` is set, lerobot-eval is running single-env
    # per-scenario mode (validated upstream). For this rollout call:
    #   * reset the controller (each rollout() invocation = one scenario)
    #   * advertise the scenario index to the TeleopRecordingContext so the
    #     recorder tags committed episodes with it
    #   * push splatsim scene metadata so the dataset captures it per-episode
    #   * flip the recorder into defer-commit mode (commit only on success;
    #     discard on failure)
    if intervention_ctx is not None:
        assert env.num_envs == 1, (
            f"Intervention mode requires env.num_envs=1, got {env.num_envs} (should have been caught upstream)."
        )
        ictrl = intervention_ctx.controller
        iteleop = intervention_ctx.teleop_context
        ictrl.reset_for_new_scenario()
        # source_scenario_idx tags each recorded episode with the UNDERLYING
        # eval-benchmark index (via benchmark_subset resolution), not the
        # rollout-local counter. This way dataset episodes downstream can be
        # joined against the eval benchmark by episode index — even when
        # --dagger_skip_succeeded_in_prev_eval prunes the subset to a
        # non-contiguous list. Matches the CSV's scenario_idx column
        # semantics.
        iteleop.source_scenario_idx = intervention_ctx.resolve_scenario_idx(intervention_ctx.scenario_idx)
        iteleop.defer_episode_saves = True
        try:
            env_cfgs = env.call("get_env_config")
            env_cfg_one = env_cfgs[0] if env_cfgs else None
        except Exception:
            logging.warning(
                "Could not fetch splatsim metadata from env; per-episode metadata will be incomplete."
            )
            env_cfg_one = None
        if env_cfg_one is not None:
            iteleop.splatsim_robot_config = env_cfg_one.get("splatsim_robot_config")
            iteleop.splatsim_object_configs = env_cfg_one.get("splatsim_object_configs")
            iteleop.splatsim_background_config = env_cfg_one.get("splatsim_background_config")
        else:
            iteleop.splatsim_robot_config = None
            iteleop.splatsim_object_configs = None
            iteleop.splatsim_background_config = None

    all_observations = []
    all_actions = []
    all_rewards = []
    all_successes = []
    all_dones = []
    all_info_metrics: dict[str, list[torch.Tensor]] = {}  # Custom metrics from info dict

    step = 0
    # Keep track of which environments are done.
    done = np.array([False] * env.num_envs)
    max_steps = env.call("_max_episode_steps")[0]
    progbar = trange(
        max_steps,
        desc=f"Running rollout with at most {max_steps} steps",
        disable=inside_slurm(),  # we dont want progress bar when we use slurm, since it clutters the logs
        leave=False,
    )
    check_env_attributes_and_types(env)
    try:
        while not np.all(done) and step < max_steps:
            # Numpy array to tensor and changing dictionary keys to LeRobot policy format.
            observation = preprocess_observation(observation)
            if return_observations:
                all_observations.append(deepcopy(observation))

            # Infer "task" from sub-environments (prefer natural language description).
            # env.call() works with both SyncVectorEnv and AsyncVectorEnv.
            try:
                observation["task"] = list(env.call("task_description"))
            except (AttributeError, NotImplementedError):
                try:
                    observation["task"] = list(env.call("task"))
                except (AttributeError, NotImplementedError):
                    observation["task"] = [""] * env.num_envs

            # Apply environment-specific preprocessing (e.g., LiberoProcessorStep for LIBERO)
            observation = env_preprocessor(observation)

            # DEBUG: snapshot raw (pre-normalization) joint state for the
            # last-mile debug wrapper to read after the policy preprocessor
            # has normalized the in-place value. Cheap; remove once the
            # diagnostic is gone.
            _raw_obs_state = observation.get("observation.state")
            if isinstance(_raw_obs_state, torch.Tensor):
                _raw_obs_state = _raw_obs_state.detach().clone()

            observation = preprocessor(observation)

            # Re-inject raw obs.state under a dedicated key so it survives
            # the normalizer step. Read by LastMileWrapper.
            if _raw_obs_state is not None:
                from pcdagger.extras.last_mile import RAW_STATE_KEY

                observation[RAW_STATE_KEY] = _raw_obs_state

            # Inject oracle env config AFTER the policy preprocessor, since the
            # pipeline drops keys that don't match its known observation /
            # complementary schema. Consumed directly by the shared-autonomy
            # wrapper (Python dict, not a tensor).
            try:
                oracle_cfgs = env.call("get_env_config")
                if oracle_cfgs is not None and any(c is not None for c in oracle_cfgs):
                    observation["oracle_env_config"] = oracle_cfgs[0]
            except (AttributeError, NotImplementedError):
                pass

            # Inject the env-mutation clock stamped on THIS observation (from
            # the info dict of the env.reset/env.step that produced it — NOT a
            # live query, which would read post-mutation). The SA wrapper
            # compares it against the version returned by its teleport RPC to
            # detect an observation captured before a controller-driven
            # teleport (the controller mutates the env AFTER env.step returned
            # this obs, so the snapshot in hand is one mutation stale). Absent
            # for envs/servers without the stamp.
            _obs_state_version = _extract_state_version(info)
            if _obs_state_version is not None:
                observation["state_version"] = _obs_state_version

            with torch.inference_mode():
                action = policy.select_action(observation)
            if predicted_latents_callback is not None:
                predicted_latents_callback(policy)
            action = postprocessor(action)

            # Apply the last-mile help in raw joint space (after the
            # postprocessor has converted action_norm → absolute joint
            # command). Operating here avoids a fragile inverse-normalization
            # round-trip that would incorrectly scale on delta-action policies.
            # The wrapper staged the help in select_action; we just consume it
            # here.
            _last_mile_wrapper = _find_last_mile_wrapper(policy)
            if _last_mile_wrapper is not None:
                action = _last_mile_wrapper.apply_help(action, _raw_obs_state)

            action_transition = {ACTION: action}
            action_transition = env_postprocessor(action_transition)
            action = action_transition[ACTION]

            # Convert to CPU / numpy.
            action_numpy: np.ndarray = action.to("cpu").numpy()
            assert action_numpy.ndim == 2, "Action dimensions should be (batch, action_dim)"

            # Apply the next action.
            observation, reward, terminated, truncated, info = env.step(action_numpy)
            if render_callback is not None:
                render_callback(env)

            # VectorEnv stores is_success in `info["final_info"][env_index]["is_success"]`. "final_info" isn't
            # available if none of the envs finished.
            if "final_info" in info:
                final_info = info["final_info"]
                if isinstance(final_info, dict):
                    is_success = final_info.get("is_success", [False] * env.num_envs)
                    successes = (
                        is_success.tolist()
                        if hasattr(is_success, "tolist")
                        else [bool(is_success)] * env.num_envs
                    )
                else:
                    # Gymnasium < 1.0 returns final_info as a per-env sequence/object array,
                    # with entries set to a dict only for envs that just finished.
                    successes = []
                    for item in final_info:
                        if isinstance(item, dict) and "is_success" in item:
                            successes.append(bool(item["is_success"]))
                        else:
                            successes.append(False)
            elif "is_success" in info:
                is_success = info["is_success"]
                successes = (
                    is_success.tolist()
                    if hasattr(is_success, "tolist")
                    else [bool(is_success)] * env.num_envs
                )
            else:
                successes = [False] * env.num_envs

            # ── intervention-mode tick ─────────────────────────────────────── #
            # The controller drives policy/intervention alternation: stall +
            # collision triggers, plan-failure backoff, controller-initiated
            # cancel after a random waypoint budget. A "advance" decision means
            # "this scenario is done" — mark all envs done to exit the loop.
            if intervention_ctx is not None:
                from pcdagger.dagger.intervention import (
                    _extract_collision_kind,
                    _extract_in_collision,
                    _extract_orientation_error_deg,
                    _extract_position_error_m,
                    _extract_success,
                )

                scn_success = _extract_success(info)
                in_collision = _extract_in_collision(info)
                collision_kind = _extract_collision_kind(info)
                position_error_m = _extract_position_error_m(info)
                orientation_error_deg = _extract_orientation_error_deg(info)
                decision = intervention_ctx.controller.tick(
                    success=scn_success,
                    in_collision=in_collision,
                    collision_kind=collision_kind,
                    position_error_m=position_error_m,
                    orientation_error_deg=orientation_error_deg,
                )
                if decision == "advance":
                    done = np.ones_like(done, dtype=bool)

            if recording_datasets is not None and raw_observation is not None:
                prev_done = done.copy()
                for env_idx in range(env.num_envs):
                    if prev_done[env_idx]:
                        continue
                    frame = _build_raw_frame(
                        raw_observation,
                        env_idx,
                        action_numpy[env_idx],
                        reward[env_idx],
                        successes[env_idx],
                        bool(terminated[env_idx] | truncated[env_idx]),
                        task_desc,
                        recording_datasets[env_idx].features,
                    )
                    recording_datasets[env_idx].add_frame(frame)
                    if terminated[env_idx] or truncated[env_idx]:
                        recording_datasets[env_idx].save_episode()
                raw_observation = deepcopy(observation)

            # Keep track of which environments are done so far.
            # Mark the episode as done if we reach the maximum step limit.
            # This ensures that the rollout always terminates cleanly at `max_steps`,
            # and allows logging/saving (e.g., videos) to be triggered consistently.
            done = terminated | truncated | done
            if step + 1 == max_steps:
                done = np.ones_like(done, dtype=bool)

            all_actions.append(torch.from_numpy(action_numpy))
            all_rewards.append(torch.from_numpy(reward))
            all_dones.append(torch.from_numpy(done))
            all_successes.append(torch.tensor(successes))

            # Track whether each env was truncated (timed out) at this step.
            if "truncated" not in all_info_metrics:
                all_info_metrics["truncated"] = []
            all_info_metrics["truncated"].append(torch.from_numpy(truncated.copy()))

            # Collect custom metrics from info dict (e.g. "in_collision",
            # "distance_to_goal"). Per-env boolean or scalar values that get
            # logged per step and aggregated below. When an episode terminates,
            # Gymnasium auto-resets and overwrites info with reset values, so
            # for terminated envs pull the pre-reset value from final_info.
            final_info = info.get("final_info", {})
            env_terminated = terminated | truncated
            for key, value in info.items():
                # state_version is the env-mutation clock (bookkeeping consumed
                # by the SA wrapper's teleport-staleness gate), not a metric —
                # aggregating it would emit a meaningless avg_state_version and
                # crash on rollouts where its presence varies across steps.
                if key in (
                    "final_obs",
                    "final_info",
                    "is_success",
                    "step_count",
                    "state_version",
                ) or key.startswith("_"):
                    continue
                if isinstance(value, np.ndarray) and value.shape == (env.num_envs,):
                    # Skip object-dtype arrays (strings, None mixes, arbitrary
                    # Python objects). torch.from_numpy would crash on them
                    # and they have no place in the numeric per-step
                    # aggregation. Sub-envs that need to surface a categorical
                    # to eval_info.json should publish a parallel `<key>_code:
                    # int` (see e.g. SplatSim's collision_kind /
                    # collision_kind_code pair).
                    if value.dtype == object:
                        continue
                    if key in final_info:
                        value = value.copy()
                        value[env_terminated] = final_info[key][env_terminated]
                    if key not in all_info_metrics:
                        all_info_metrics[key] = []
                    all_info_metrics[key].append(
                        torch.from_numpy(value.copy() if key not in final_info else value)
                    )

            step += 1
            running_success_rate = (
                einops.reduce(torch.stack(all_successes, dim=1), "b n -> b", "any").numpy().mean()
            )
            # refresh=False — let progbar.update() trigger the single refresh,
            # otherwise each loop iteration writes the bar twice. Invisible on
            # a real TTY (both `\r`-overwrite the same line) but doubles the
            # line count when stdout isn't TTY-detected by tqdm (e.g. piped/
            # tee'd).
            progbar.set_postfix(
                {"running_success_rate": f"{running_success_rate.item() * 100:.1f}%"},
                refresh=False,
            )
            progbar.update()
    finally:
        if recording_datasets is not None:
            for ds in recording_datasets:
                ds.finalize()
                if recording_repo_id is not None:
                    if ds.num_episodes > 0:
                        ds.push_to_hub(private=recording_private)
                    else:
                        logging.warning("No episodes recorded for %s — skipping push to hub.", ds.repo_id)

    # Track the final observation.
    if return_observations:
        observation = preprocess_observation(observation)
        all_observations.append(deepcopy(observation))

    # Stack the sequence along the first dimension so that we have (batch, sequence, *) tensors.
    ret = {
        ACTION: torch.stack(all_actions, dim=1),
        "reward": torch.stack(all_rewards, dim=1),
        "success": torch.stack(all_successes, dim=1),
        "done": torch.stack(all_dones, dim=1),
    }

    # Add custom info metrics (e.g., "in_collision") as (batch, sequence) tensors
    if all_info_metrics:
        ret["info_metrics"] = {key: torch.stack(values, dim=1) for key, values in all_info_metrics.items()}
    if return_observations:
        stacked_observations = {}
        for key in all_observations[0]:
            stacked_observations[key] = torch.stack([obs[key] for obs in all_observations], dim=1)
        ret[OBS_STR] = stacked_observations

    if hasattr(policy, "use_original_modules"):
        policy.use_original_modules()

    # ── intervention-mode post-rollout: commit + CSV row ─────────────────── #
    # Reads ``all_successes`` for the scenario verdict (CSV bookkeeping).
    # Buffered episodes (accumulated under teleop_ctx with
    # defer_episode_saves=True) are committed regardless of the verdict —
    # see the rationale at the commit call below.
    if intervention_ctx is not None:
        # ``all_successes`` is a list of [B]-shaped tensors; we have B==1 so
        # check if any tick reported success.
        scn_success = bool(torch.stack(all_successes, dim=1).any().item()) if all_successes else False

        # Force any in-progress recorded frames (e.g. env declared success
        # mid-intervention and we broke out before the wrapper saw
        # frame_source transition back to POLICY) into pending_episodes
        # under the still-set source_scenario_idx, before commit/discard.
        try:
            env.call("flush_in_progress_episode")
        except Exception:
            logging.exception("flush_in_progress_episode failed during intervention rollout cleanup.")

        # Commit UNCONDITIONALLY — including failed scenarios. Each recorded
        # chunk is RRT expert data that reached its own local goal; its
        # validity does not depend on whether the POLICY later completed the
        # scenario between interventions. The old success-gated commit
        # silently created survivorship bias: the hardest scenarios (the
        # exact states DAgger exists to supervise) contributed ZERO data
        # because the weak policy kept them from ever finishing — measured
        # 2026-08-18: scenario 11's first-ever wrap-around recovery chunks
        # (5 episodes, 619 frames) discarded on max_cycles_reached.
        n_committed_list = env.call("commit_pending_episodes")
        n_committed = int(sum(n_committed_list)) if n_committed_list else 0
        intervention_ctx.n_committed_episodes += n_committed

        ctrl = intervention_ctx.controller
        _resolved_idx = intervention_ctx.resolve_scenario_idx(intervention_ctx.scenario_idx)
        logging.info(
            "Scenario %d (benchmark ep %d) finished: success=%s cycles=%d status=%s "
            "(%d episode(s) committed%s)",
            intervention_ctx.scenario_idx,
            _resolved_idx,
            scn_success,
            ctrl.cycles_used,
            ctrl.last_status,
            n_committed,
            "" if scn_success else " from FAILED scenario — chunks are still expert data",
        )
        intervention_ctx.record_scenario_result(intervention_ctx.scenario_idx, scn_success)
        # Advance the counter for the next rollout() call (next scenario).
        intervention_ctx.scenario_idx += 1

    return ret


def eval_policy(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    n_episodes: int,
    max_episodes_rendered: int = 0,
    max_episodes_rendered_failed: int = 0,
    videos_dir: Path | None = None,
    return_episode_data: bool = False,
    start_seed: int | None = None,
    recording_dir: Path | None = None,
    env_features: dict | None = None,
    recording_repo_id: str | None = None,
    recording_private: bool = False,
    save_predicted_video: bool = False,
    intervention_ctx: "InterventionContext | None" = None,
) -> dict:
    """
    Args:
        env: The batch of environments.
        policy: The policy.
        n_episodes: The number of episodes to evaluate.
        max_episodes_rendered: Maximum number of episodes to render into videos.
            These are saved as `eval_episode_0.mp4`, `eval_episode_1.mp4`, etc.
            in scan-order regardless of success — captures the FIRST N episodes.
        max_episodes_rendered_failed: ADDITIONAL videos rendered only for episodes
            the policy failed (i.e. `success == False`). Saved as
            `eval_episode_failed_0.mp4`, `eval_episode_failed_1.mp4`, etc.
            Most useful for debugging — most failed episodes are NOT in the
            first N successful-or-not, so without this they'd never get
            visualized. Default 0 = legacy behavior (first-N only).
        videos_dir: Where to save rendered videos.
        return_episode_data: Whether to return episode data for online training. Incorporates the data into
            the "episodes" key of the returned dictionary.
        start_seed: The first seed to use for the first individual rollout. For all subsequent rollouts the
            seed is incremented by 1. If not provided, the environments are not manually seeded.
    Returns:
        Dictionary with metrics and data regarding the rollouts.
    """
    if (max_episodes_rendered > 0 or max_episodes_rendered_failed > 0) and not videos_dir:
        raise ValueError(
            "If max_episodes_rendered > 0 or max_episodes_rendered_failed > 0, videos_dir must be provided."
        )

    # World-model policies (e.g. LingBot-VA) opt into predicted-video saving via their config.
    save_predicted_video = save_predicted_video or bool(
        getattr(getattr(policy, "config", None), "save_predicted_video", False)
    )

    if not isinstance(policy, PreTrainedPolicy):
        exc = ValueError(
            f"Policy of type 'PreTrainedPolicy' is expected, but type '{type(policy)}' was provided."
        )
        if not peft_available():
            raise exc
        require_package("peft", extra="peft")
        if not isinstance(policy, PeftModel):
            raise exc

    start = time.time()
    # Preserve the mode for direct callers. eval_policy_all scopes the mode
    # around all tasks so parallel evaluations cannot race with each other.
    was_training = policy.training
    policy.eval()

    # Determine how many batched rollouts we need to get n_episodes. Note that if n_episodes is not evenly
    # divisible by env.num_envs we end up discarding some data in the last batch.
    n_batches = n_episodes // env.num_envs + int((n_episodes % env.num_envs) != 0)

    # Keep track of some metrics.
    sum_rewards = []
    max_rewards = []
    all_successes = []
    all_seeds = []
    all_info_metrics: dict[str, list[float]] = {}  # Aggregated custom metrics per episode
    threads = []  # for video saving threads
    n_episodes_rendered = 0  # for saving the correct number of videos
    n_failed_rendered = 0  # extra budget for failed-only videos
    any_render_budget = max_episodes_rendered > 0 or max_episodes_rendered_failed > 0
    # Whether the POLICY consumes image observations. When it does, the env is
    # already rendering cameras every step, so capturing video frames is cheap and
    # we keep the original "render every batch, filter afterward" behavior. When it
    # does NOT (e.g. a state-only oracle policy), the env renders nothing otherwise,
    # so video frames are pure added cost — we stop rendering as soon as the video
    # budgets are full (see render_frame).
    policy_uses_images = bool(getattr(getattr(policy, "config", None), "image_features", None))

    # Callback for visualization.
    # Frames must be captured WHILE the rollout runs (we can't go back and
    # re-render afterward), but we don't know which episodes will fail
    # until the rollout finishes. So when max_episodes_rendered_failed > 0
    # we have to render every env in every batch up to whatever is needed
    # — the post-rollout video write step then decides which episodes to
    # keep based on success status. When max_episodes_rendered_failed == 0
    # this collapses to the legacy "first N" capture.
    def render_frame(env: gym.vector.VectorEnv):
        # noqa: B023
        if not any_render_budget:
            return
        # Image-free policy: once BOTH budgets are full nothing more will be saved
        # (the save loop breaks on the same condition), so stop rendering — for a
        # state-only policy the env renders nothing otherwise, so this avoids
        # rendering every remaining episode purely to discard it. NOT applied when
        # the policy uses images: there the cameras run every step regardless, and
        # we keep the original render-every-batch behavior so failure-video
        # selection is byte-for-byte unchanged for the existing image pipelines.
        if (
            not policy_uses_images
            and n_episodes_rendered >= max_episodes_rendered
            and n_failed_rendered >= max_episodes_rendered_failed
        ):
            return
        # When the failed-video budget is set we need frames for ALL envs
        # in this batch (since any one of them might turn out to be a
        # failure we want to save). When only the first-N budget is set
        # and it's exhausted, no need to render anything more.
        if max_episodes_rendered_failed == 0 and n_episodes_rendered >= max_episodes_rendered:
            return
        if max_episodes_rendered_failed == 0:
            n_to_render_now = min(max_episodes_rendered - n_episodes_rendered, env.num_envs)
        else:
            # Render everything in the batch — post-batch logic filters.
            n_to_render_now = env.num_envs
        if isinstance(env, gym.vector.SyncVectorEnv):
            ep_frames.append(np.stack([env.envs[i].render() for i in range(n_to_render_now)]))  # noqa: B023
        elif hasattr(env, "call"):
            # Here we must render all frames and discard any we don't need.
            # Covers AsyncVectorEnv and _LazyAsyncVectorEnv (which wraps one).
            ep_frames.append(np.stack(env.call("render")[:n_to_render_now]))

    if any_render_budget:
        video_paths: list[str] = []

    if save_predicted_video:
        if not videos_dir:
            raise ValueError("If save_predicted_video is True, videos_dir must be provided.")
        predicted_video_paths: list[str] = []
        n_predicted_rendered = 0

    # Collect predicted-video latents across a rollout (world-model policies only). The latents are
    # concatenated and decoded once after the rollout, matching upstream LingBot-VA's visualization path.
    def collect_predicted_latents(policy: PreTrainedPolicy):
        latents = getattr(policy, "last_predicted_latents", None)
        if latents is not None:
            pred_latents.append(
                latents.detach().to("cpu") if hasattr(latents, "detach") else torch.as_tensor(latents).cpu()
            )
            policy.last_predicted_latents = None

    if return_episode_data:
        episode_data: dict | None = None

    # we dont want progress bar when we use slurm, since it clutters the logs
    progbar = trange(n_batches, desc="Stepping through eval batches", disable=inside_slurm())
    for batch_ix in progbar:
        # Cache frames for rendering videos. Each item will be (b, h, w, c), and the list indexes the rollout
        # step.
        if any_render_budget:
            ep_frames: list[np.ndarray] = []

        if save_predicted_video:
            pred_latents: list[torch.Tensor] = []

        if start_seed is None:
            seeds = None
        else:
            seeds = range(
                start_seed + (batch_ix * env.num_envs), start_seed + ((batch_ix + 1) * env.num_envs)
            )
        # Absolute rollout indices (0-based, independent of start_seed). SplatSim
        # env's reset uses these via options["benchmark_start_index"] to select
        # scenario = subset[idx % len(subset)] deterministically, so partial
        # prior sessions can't misalign scenario 0. Non-SplatSim envs ignore.
        scenario_indices = list(range(batch_ix * env.num_envs, (batch_ix + 1) * env.num_envs))
        rollout_data = rollout(
            env=env,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            seeds=list(seeds) if seeds else None,
            scenario_indices=scenario_indices,
            return_observations=return_episode_data,
            render_callback=render_frame if any_render_budget else None,
            recording_dir=recording_dir,
            env_features=env_features,
            recording_repo_id=recording_repo_id,
            recording_private=recording_private,
            predicted_latents_callback=collect_predicted_latents if save_predicted_video else None,
            intervention_ctx=intervention_ctx,
        )

        # Figure out where in each rollout sequence the first done condition was encountered (results after
        # this won't be included).
        n_steps = rollout_data["done"].shape[1]
        # Note: this relies on a property of argmax: that it returns the first occurrence as a tiebreaker.
        done_indices = torch.argmax(rollout_data["done"].to(int), dim=1)

        # Track episode lengths (number of steps taken)
        if "episode_length" not in all_info_metrics:
            all_info_metrics["episode_length"] = []
        all_info_metrics["episode_length"].extend((done_indices + 1).tolist())

        # Make a mask with shape (batch, n_steps) to mask out rollout data after the first done
        # (batch-element-wise). Note the `done_indices + 1` to make sure to keep the data from the done step.
        mask = (torch.arange(n_steps) <= einops.repeat(done_indices + 1, "b -> b s", s=n_steps)).int()
        # Extend metrics.
        batch_sum_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "sum")
        sum_rewards.extend(batch_sum_rewards.tolist())
        batch_max_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "max")
        max_rewards.extend(batch_max_rewards.tolist())
        batch_successes = einops.reduce((rollout_data["success"] * mask), "b n -> b", "any")
        all_successes.extend(batch_successes.tolist())

        # Aggregate custom info metrics (e.g., count steps where in_collision=True)
        if "info_metrics" in rollout_data:
            for metric_name, metric_data in rollout_data["info_metrics"].items():
                if metric_name not in all_info_metrics:
                    all_info_metrics[metric_name] = []
                # For boolean metrics (like in_collision), compute sum of True steps per episode
                # For scalar metrics, compute mean per episode
                if metric_data.dtype == torch.bool:
                    # Count how many steps have this condition true (masked to valid steps)
                    batch_metric = einops.reduce((metric_data.int() * mask), "b n -> b", "sum")
                else:
                    # For scalar metrics, compute mean over valid steps
                    batch_metric = einops.reduce((metric_data.float() * mask), "b n -> b", "sum") / mask.sum(
                        dim=1
                    ).clamp(min=1)
                all_info_metrics[metric_name].extend(batch_metric.tolist())

                # For scalar metrics, also track the final value at episode end
                if metric_data.dtype != torch.bool:
                    final_metric_name = f"final_{metric_name}"
                    if final_metric_name not in all_info_metrics:
                        all_info_metrics[final_metric_name] = []
                    # Get value at done_indices (last valid step) for each episode
                    batch_final_metric = metric_data[torch.arange(metric_data.size(0)), done_indices]
                    all_info_metrics[final_metric_name].extend(batch_final_metric.tolist())

                    # Also track the minimum value across valid steps per episode
                    min_metric_name = f"min_{metric_name}"
                    if min_metric_name not in all_info_metrics:
                        all_info_metrics[min_metric_name] = []
                    # Set invalid steps to +inf so they don't affect the minimum
                    data_float = metric_data.float().clone()
                    data_float[mask == 0] = float("inf")
                    batch_min_metric = data_float.min(dim=1).values
                    all_info_metrics[min_metric_name].extend(batch_min_metric.tolist())

        if seeds:
            all_seeds.extend(seeds)
        else:
            all_seeds.extend([None] * env.num_envs)

        # FIXME: episode_data is either None or it doesn't exist
        if return_episode_data:
            this_episode_data = _compile_episode_data(
                rollout_data,
                done_indices,
                start_episode_index=batch_ix * env.num_envs,
                start_data_index=(0 if episode_data is None else (episode_data["index"][-1].item() + 1)),
                fps=env.unwrapped.metadata["render_fps"],
            )
            if episode_data is None:
                episode_data = this_episode_data
            else:
                # Some sanity checks to make sure we are correctly compiling the data.
                assert episode_data["episode_index"][-1] + 1 == this_episode_data["episode_index"][0]
                assert episode_data["index"][-1] + 1 == this_episode_data["index"][0]
                # Concatenate the episode data.
                episode_data = {k: torch.cat([episode_data[k], this_episode_data[k]]) for k in episode_data}

        # Maybe render video for visualization.
        # Two independent budgets:
        #   1. `max_episodes_rendered` — saves the FIRST N episodes in scan
        #      order regardless of success. Files: `eval_episode_<i>.mp4`.
        #   2. `max_episodes_rendered_failed` — saves up to K episodes that
        #      FAILED, deduped against episodes already saved by (1).
        #      Files: `eval_episode_failed_<i>.mp4`.
        # Per-episode success comes from this batch's `batch_successes`
        # (computed above). The two budgets are processed in the same
        # loop so both can fire on the same episode (the failed counter
        # just increments; we don't double-write the same episode).
        if any_render_budget and len(ep_frames) > 0:
            # `videos_dir` is guaranteed non-None when any_render_budget
            # is True (asserted at function entry). assert for the
            # type-checker so it doesn't flag the mkdir/path-join below.
            assert videos_dir is not None
            batch_stacked_frames = np.stack(ep_frames, axis=1)  # (b, t, *)
            batch_succ_list = batch_successes.flatten().tolist()
            for ep_in_batch, (stacked_frames, done_index) in enumerate(
                zip(batch_stacked_frames, done_indices.flatten().tolist(), strict=False)
            ):
                if (
                    n_episodes_rendered >= max_episodes_rendered
                    and n_failed_rendered >= max_episodes_rendered_failed
                ):
                    break
                # Tier 1: first-N quota (unconditional on success).
                saved_in_tier1 = False
                if n_episodes_rendered < max_episodes_rendered:
                    videos_dir.mkdir(parents=True, exist_ok=True)
                    video_path = videos_dir / f"eval_episode_{n_episodes_rendered}.mp4"
                    video_paths.append(str(video_path))
                    thread = threading.Thread(
                        target=write_video,
                        args=(
                            str(video_path),
                            stacked_frames[: done_index + 1],  # + 1 to capture the last observation
                            env.unwrapped.metadata["render_fps"],
                        ),
                    )
                    thread.start()
                    threads.append(thread)
                    n_episodes_rendered += 1
                    saved_in_tier1 = True
                # Tier 2: failed-only quota. Skip if this episode succeeded,
                # or if we already saved it in tier 1 (no double-write —
                # the failed-budget is for failures NOT in the first N).
                #
                # Naming convention: `eval_episode_failed_<absolute_episode_idx>.mp4`
                # where `absolute_episode_idx` is the position of this episode
                # in the full eval sequence (NOT a sequential failure counter).
                # This way the filename directly identifies which scenario
                # failed — match against `eval_info.json`'s
                # `per_task[0].metrics.successes[<absolute_episode_idx>]`
                # to confirm. Useful when only a few episodes in a long
                # eval failed: filename tells you which ones without
                # cross-referencing a separate index.
                if (
                    not saved_in_tier1
                    and ep_in_batch < len(batch_succ_list)
                    and not bool(batch_succ_list[ep_in_batch])
                    and n_failed_rendered < max_episodes_rendered_failed
                ):
                    absolute_episode_idx = batch_ix * env.num_envs + ep_in_batch
                    videos_dir.mkdir(parents=True, exist_ok=True)
                    video_path = videos_dir / f"eval_episode_failed_{absolute_episode_idx}.mp4"
                    video_paths.append(str(video_path))
                    thread = threading.Thread(
                        target=write_video,
                        args=(
                            str(video_path),
                            stacked_frames[: done_index + 1],
                            env.unwrapped.metadata["render_fps"],
                        ),
                    )
                    thread.start()
                    threads.append(thread)
                    n_failed_rendered += 1

        # Maybe save the policy's predicted (imagined) video for this batch's rollout.
        if save_predicted_video and len(pred_latents) > 0:
            predicted_latent = torch.cat(pred_latents, dim=2)
            decoder = getattr(policy, "decode_predicted_latents", None) or getattr(
                policy, "_decode_predicted_video", None
            )
            if decoder is None:
                raise AttributeError(
                    "Policy config requested predicted-video saving, but the policy does not expose "
                    "`decode_predicted_latents` or `_decode_predicted_video`."
                )
            predicted_video = decoder(predicted_latent)
            if hasattr(predicted_video, "detach"):
                predicted_video = predicted_video.detach().to("cpu").numpy()
            videos_dir.mkdir(parents=True, exist_ok=True)
            predicted_video_path = videos_dir / f"pred_episode_{n_predicted_rendered}.mp4"
            predicted_video_paths.append(str(predicted_video_path))
            thread = threading.Thread(
                target=write_video,
                args=(
                    str(predicted_video_path),
                    predicted_video,
                    env.unwrapped.metadata["render_fps"],
                ),
            )
            thread.start()
            threads.append(thread)
            n_predicted_rendered += 1

        # refresh=False: see the matching comment on the inner-rollout
        # progbar.set_postfix call. The outer for-loop already triggers a
        # tqdm refresh on the next iteration; this just stages the postfix
        # for that refresh without double-writing.
        progbar.set_postfix(
            {"running_success_rate": f"{np.mean(all_successes[:n_episodes]).item() * 100:.1f}%"},
            refresh=False,
        )

    # Wait till all video rendering threads are done.
    for thread in threads:
        thread.join()

    # Compile eval info.
    # Build per-episode info with optional custom metrics
    per_episode_info = []
    for i, (sum_reward, max_reward, success, seed) in enumerate(
        zip(
            sum_rewards[:n_episodes],
            max_rewards[:n_episodes],
            all_successes[:n_episodes],
            all_seeds[:n_episodes],
            strict=True,
        )
    ):
        ep_info = {
            "episode_ix": i,
            "sum_reward": sum_reward,
            "max_reward": max_reward,
            "success": success,
            "seed": seed,
        }
        # Add custom info metrics for this episode
        for metric_name, metric_values in all_info_metrics.items():
            if i < len(metric_values):
                ep_info[metric_name] = metric_values[i]
        per_episode_info.append(ep_info)

    # Build aggregated metrics
    aggregated: dict[str, float | None] = {
        "avg_sum_reward": float(np.nanmean(sum_rewards[:n_episodes])),
        "avg_max_reward": float(np.nanmean(max_rewards[:n_episodes])),
        "pc_success": float(np.nanmean(all_successes[:n_episodes]) * 100),
        "eval_s": time.time() - start,
        "eval_ep_s": (time.time() - start) / n_episodes,
    }
    # Add aggregated custom info metrics (mean across episodes)
    for metric_name, metric_values in all_info_metrics.items():
        aggregated[f"avg_{metric_name}"] = float(np.nanmean(metric_values[:n_episodes]))

    # Compute avg episode length excluding truncated (timed-out) episodes
    if "episode_length" in all_info_metrics and "truncated" in all_info_metrics:
        ep_lens = all_info_metrics["episode_length"][:n_episodes]
        trunc_flags = all_info_metrics["truncated"][:n_episodes]
        non_truncated_lens = [
            episode_len for episode_len, truncated in zip(ep_lens, trunc_flags, strict=True) if not truncated
        ]
        aggregated["avg_episode_length_without_truncation"] = (
            float(np.mean(non_truncated_lens)) if non_truncated_lens else None
        )

    info = {
        "per_episode": per_episode_info,
        "aggregated": aggregated,
    }

    if return_episode_data:
        info["episodes"] = episode_data

    if max_episodes_rendered > 0:
        info["video_paths"] = video_paths

    if save_predicted_video:
        info["predicted_video_paths"] = predicted_video_paths

    policy.train(was_training)

    return info


def _compile_episode_data(
    rollout_data: dict, done_indices: Tensor, start_episode_index: int, start_data_index: int, fps: float
) -> dict:
    """Convenience function for `eval_policy(return_episode_data=True)`

    Compiles all the rollout data into a Hugging Face dataset.

    Similar logic is implemented when datasets are pushed to hub (see: `push_to_hub`).
    """
    ep_dicts = []
    total_frames = 0
    for ep_ix in range(rollout_data[ACTION].shape[0]):
        # + 2 to include the first done frame and the last observation frame.
        num_frames = done_indices[ep_ix].item() + 2
        total_frames += num_frames

        # Here we do `num_frames - 1` as we don't want to include the last observation frame just yet.
        ep_dict = {
            ACTION: rollout_data[ACTION][ep_ix, : num_frames - 1],
            "episode_index": torch.tensor([start_episode_index + ep_ix] * (num_frames - 1)),
            "frame_index": torch.arange(0, num_frames - 1, 1),
            "timestamp": torch.arange(0, num_frames - 1, 1) / fps,
            DONE: rollout_data["done"][ep_ix, : num_frames - 1],
            "next.success": rollout_data["success"][ep_ix, : num_frames - 1],
            REWARD: rollout_data["reward"][ep_ix, : num_frames - 1].type(torch.float32),
        }

        # For the last observation frame, all other keys will just be copy padded.
        for k in ep_dict:
            ep_dict[k] = torch.cat([ep_dict[k], ep_dict[k][-1:]])

        for key in rollout_data[OBS_STR]:
            ep_dict[key] = rollout_data[OBS_STR][key][ep_ix, :num_frames]

        ep_dicts.append(ep_dict)

    data_dict = {}
    for key in ep_dicts[0]:
        data_dict[key] = torch.cat([x[key] for x in ep_dicts])

    data_dict["index"] = torch.arange(start_data_index, start_data_index + total_frames, 1)

    return data_dict


@parser.wrap()
def eval_main(cfg: EvalPipelineConfig):
    logging.info(pformat(asdict(cfg)))

    # ── intervention-mode pre-flight validation ──────────────────────────── #
    # When `cfg.intervention` is set, lerobot-eval switches into single-env
    # per-scenario mode with an `InterventionController` driving an SA-wrapped
    # policy. Several config combinations are incompatible with that mode and
    # would fail downstream in confusing ways — error here instead.
    if cfg.intervention is not None:
        if cfg.eval.batch_size != 1:
            raise ValueError(
                f"Intervention mode requires --eval.batch_size=1, got {cfg.eval.batch_size}. "
                "The controller's state machine ticks per-step on a single env."
            )
        if cfg.eval.use_async_envs:
            raise ValueError(
                "Intervention mode is incompatible with --eval.use_async_envs=true. "
                "The controller needs synchronous access to env state each tick to decide "
                "whether to cancel an active intervention."
            )
        sa_cfg_pre = getattr(cfg.policy, "shared_autonomy_config", None)
        if sa_cfg_pre is None or not sa_cfg_pre.enabled:
            raise ValueError(
                "Intervention mode requires --policy.shared_autonomy_config.enabled=true. "
                "Both 'rrt' and 'oracle_goal' guidance sources live on the SA wrapper; "
                "without it there is no intervention path."
            )
        if not getattr(cfg.env, "teleop_dataset_repo_id", None):
            raise ValueError(
                "Intervention mode requires --env.teleop_dataset_repo_id=<repo>. "
                "An intervention run without a recording target is wasted compute."
            )

    # Check device is available
    if cfg.policy is not None:
        get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")

    logging.info(f"Making environment (batch_size={cfg.eval.batch_size}, async={cfg.eval.use_async_envs}).")
    envs = make_env(
        cfg.env,
        n_envs=cfg.eval.batch_size,
        use_async_envs=cfg.eval.use_async_envs,
        trust_remote_code=cfg.trust_remote_code,
    )

    logging.info("Making policy.")

    policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
        rename_map=cfg.rename_map,
    )

    policy.eval()

    # The inference device is automatically set to match the detected hardware, overriding any previous device settings from training to ensure compatibility.
    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": cfg.rename_map},
    }

    # observation_dim_slice: same delivery convention as lerobot-train — a
    # top-level kwarg (not a preprocessor_overrides key) because
    # `PolicyProcessorPipeline.from_pretrained` rejects overrides for steps
    # that don't exist in the saved config. The factory's post-hoc block
    # inserts/replaces the step so eval works uniformly whether the
    # checkpoint was trained with or without the slice.
    make_pp_kwargs: dict[str, Any] = {
        "policy_cfg": cfg.policy,
        "pretrained_path": cfg.policy.pretrained_path,
        "preprocessor_overrides": preprocessor_overrides,
    }
    if cfg.observation_dim_slice:
        make_pp_kwargs["observation_dim_slice"] = cfg.observation_dim_slice
    preprocessor, postprocessor = make_pre_post_processors(**make_pp_kwargs)

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

    # Outermost last-mile help wrapper. Applied AFTER TE+SA so it overrides
    # whatever final action the inner stack produces.
    last_mile_cfg = getattr(cfg.policy, "last_mile_config", None)
    if last_mile_cfg is not None and last_mile_cfg.enabled:
        policy = _wrap_with_last_mile(policy, cfg.policy)

    # ── intervention-mode setup ──────────────────────────────────────────── #
    # If we got here with cfg.intervention set, the validation block at the
    # top of eval_main already confirmed the SA wrapper is enabled and the
    # env/eval flags are intervention-compatible. Now wire the controller +
    # context + CSV and force the SA-wrapper flags the controller needs.
    intervention_ctx = None
    if cfg.intervention is not None:
        from lerobot_env_splatsim.recording import TeleopRecordingContext
        from pcdagger.dagger.intervention import (
            InterventionContext,
            InterventionController,
        )

        sa_policy = _find_shared_autonomy_wrapper(policy)
        if sa_policy is None:
            # Defensive — validation block above should make this unreachable.
            raise RuntimeError("Intervention mode set but SA wrapper not found in the wrapper chain.")
        # Headless: controller cannot tolerate pause-gate blocking.
        sa_policy.auto_pause_on_rrt_finish = False
        sa_policy._run_event.set()
        # Hand the env handle to the SA wrapper so the RRT source can teleport
        # the sim's joint state pre-execution. Walk the nested envs dict to
        # pull the single VectorEnv (validation guaranteed exactly one task).
        flat_envs = [vec for group in envs.values() for vec in group.values()]
        if len(flat_envs) != 1:
            raise ValueError(
                f"Intervention mode expects exactly one task/env (single-env), found {len(flat_envs)}."
            )
        sa_policy.set_env_for_teleport(flat_envs[0])

        ctrl = InterventionController(sa_policy, cfg.intervention)
        # Pass the eval-benchmark subset so the per-scenario CSV records the
        # underlying benchmark index (not the rollout-local counter). When
        # --dagger_skip_succeeded_in_prev_eval prunes the subset to a
        # non-contiguous list like [1, 5, 8, 13], the CSV's scenario_idx
        # column reports those actual indices instead of 0..3.
        env_subset = getattr(cfg.env, "eval_benchmark_subset", None)
        intervention_ctx = InterventionContext(
            controller=ctrl,
            teleop_context=TeleopRecordingContext.get_instance(),
            csv_path=Path(cfg.output_dir) / "intervention_per_scenario.csv",
            benchmark_subset=list(env_subset) if env_subset else None,
        )
        intervention_ctx.open_csv()
        logging.info("Intervention config: %s", pformat(asdict(cfg.intervention)))
        logging.info("Per-scenario CSV: %s", intervention_ctx.csv_path)

    # Create environment-specific preprocessor and postprocessor (e.g., for LIBERO environments)
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)

    recording_dir = Path(cfg.output_dir) / "recordings" if cfg.eval.recording else None
    max_episodes_rendered = 0 if cfg.eval.recording else 10
    videos_dir = None if cfg.eval.recording else Path(cfg.output_dir) / "videos"

    # Note: AMP (autocast) is applied only around policy inference, not env operations,
    # because some simulators (e.g., SplatSim with e3nn) don't support half precision.
    # For SplatSim users, keep cfg.policy.use_amp=False so autocast becomes a nullcontext.
    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():  # noqa: F821
        info = eval_policy_all(
            envs=envs,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=cfg.eval.n_episodes,
            max_episodes_rendered=max_episodes_rendered,
            videos_dir=videos_dir,
            return_episode_data=False,
            start_seed=cfg.seed,
            max_parallel_tasks=cfg.env.max_parallel_tasks,
            recording_dir=recording_dir,
            env_features=cfg.env.features if cfg.eval.recording else None,
            recording_repo_id=cfg.eval.recording_repo_id,
            recording_private=cfg.eval.recording_private,
            output_dir=Path(cfg.output_dir),
            intervention_ctx=intervention_ctx,
        )
        logger.info("Overall Aggregated Metrics:")
        logger.info(info["overall"])

        # Print per-suite stats
        for task_group, task_group_info in info.items():
            logger.info(f"\nAggregated Metrics for {task_group}:")
            logger.info(task_group_info)
    # Close all vec envs
    close_envs(envs)

    # Save info
    def _json_default(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    with open(Path(cfg.output_dir) / "eval_info.json", "w") as f:
        json.dump(info, f, indent=2, default=_json_default)

    # Finalize the per-scenario CSV (intervention mode only).
    if intervention_ctx is not None:
        intervention_ctx.close_csv()
        logging.info(
            "Intervention run complete: %d episode(s) committed to dataset across %d scenario(s).",
            intervention_ctx.n_committed_episodes,
            intervention_ctx.scenario_idx,
        )

    # Close all vec envs (intentionally after saving eval_info.json so results
    # are persisted even if env cleanup has issues)
    close_envs(envs)

    logging.info("End of eval")


# ---- typed payload returned by one task eval ----
class TaskMetrics(TypedDict, total=False):
    sum_rewards: list[float]
    max_rewards: list[float]
    successes: list[bool]
    video_paths: list[str]
    predicted_video_paths: list[str]
    info_metrics: dict[str, list[float]]  # Custom metrics from env info dict


ACC_KEYS = ("sum_rewards", "max_rewards", "successes", "video_paths", "predicted_video_paths", "info_metrics")


def eval_one(
    env: gym.vector.VectorEnv,
    *,
    policy: PreTrainedPolicy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    n_episodes: int,
    max_episodes_rendered: int,
    max_episodes_rendered_failed: int,
    videos_dir: Path | None,
    return_episode_data: bool,
    start_seed: int | None,
    recording_dir: Path | None = None,
    env_features: dict | None = None,
    recording_repo_id: str | None = None,
    recording_private: bool = False,
    intervention_ctx: "InterventionContext | None" = None,
) -> TaskMetrics:
    """Evaluates one task_id of one suite using the provided vec env."""

    task_videos_dir = videos_dir

    task_result = eval_policy(
        env=env,
        policy=policy,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        n_episodes=n_episodes,
        max_episodes_rendered=max_episodes_rendered,
        max_episodes_rendered_failed=max_episodes_rendered_failed,
        videos_dir=task_videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
        recording_dir=recording_dir,
        env_features=env_features,
        recording_repo_id=recording_repo_id,
        recording_private=recording_private,
        intervention_ctx=intervention_ctx,
    )

    per_episode = task_result["per_episode"]

    # Extract custom info metrics from per_episode data
    # These are any keys that aren't the standard ones
    standard_keys = {"episode_ix", "sum_reward", "max_reward", "success", "seed"}
    info_metrics: dict[str, list[float]] = {}
    if per_episode:
        for key in per_episode[0]:
            if key not in standard_keys:
                info_metrics[key] = [ep[key] for ep in per_episode]

    result = TaskMetrics(
        sum_rewards=[ep["sum_reward"] for ep in per_episode],
        max_rewards=[ep["max_reward"] for ep in per_episode],
        successes=[ep["success"] for ep in per_episode],
        video_paths=task_result.get("video_paths", []),
        predicted_video_paths=task_result.get("predicted_video_paths", []),
    )
    if "episode_length" in info_metrics and "truncated" in info_metrics:
        info_metrics["episode_length_without_truncation"] = [
            ep_len if not truncated else float("nan")
            for ep_len, truncated in zip(
                info_metrics["episode_length"], info_metrics["truncated"], strict=True
            )
        ]
    if info_metrics:
        result["info_metrics"] = info_metrics
    return result


def run_one(
    task_group: str,
    task_id: int,
    env,
    *,
    policy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    n_episodes: int,
    max_episodes_rendered: int,
    max_episodes_rendered_failed: int,
    videos_dir: Path | None,
    return_episode_data: bool,
    start_seed: int | None,
    recording_dir: Path | None = None,
    env_features: dict | None = None,
    recording_repo_id: str | None = None,
    recording_private: bool = False,
    intervention_ctx: "InterventionContext | None" = None,
):
    """
    Run eval_one for a single (task_group, task_id, env).
    Returns (task_group, task_id, task_metrics_dict).
    This function is intentionally module-level to make it easy to test.
    """
    task_videos_dir = None
    if videos_dir is not None:
        task_videos_dir = videos_dir / f"{task_group}_{task_id}"
        task_videos_dir.mkdir(parents=True, exist_ok=True)

    task_recording_dir = None
    task_repo_id = None
    if recording_dir is not None and env_features is not None:
        task_recording_dir = recording_dir / f"{task_group}_{task_id}"
        if recording_repo_id is not None:
            task_repo_id = f"{recording_repo_id}_{task_group}_{task_id}"

    metrics = eval_one(
        env,
        policy=policy,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        n_episodes=n_episodes,
        max_episodes_rendered=max_episodes_rendered,
        max_episodes_rendered_failed=max_episodes_rendered_failed,
        videos_dir=task_videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
        recording_dir=task_recording_dir,
        env_features=env_features,
        recording_repo_id=task_repo_id,
        recording_private=recording_private,
        intervention_ctx=intervention_ctx,
    )
    # ensure we always provide video_paths key to simplify accumulation
    if max_episodes_rendered > 0 or max_episodes_rendered_failed > 0:
        metrics.setdefault("video_paths", [])
    metrics.setdefault("predicted_video_paths", [])
    return task_group, task_id, metrics


def eval_policy_all(
    envs: dict[str, dict[int, gym.vector.VectorEnv]],
    policy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    n_episodes: int,
    *,
    max_episodes_rendered: int = 0,
    max_episodes_rendered_failed: int = 0,
    recording_dir: Path | None = None,
    env_features: dict | None = None,
    recording_repo_id: str | None = None,
    recording_private: bool = False,
    videos_dir: Path | None = None,
    return_episode_data: bool = False,
    start_seed: int | None = None,
    max_parallel_tasks: int = 1,
    output_dir: Path | None = None,
    close_envs_after_eval: bool = True,
    intervention_ctx: "InterventionContext | None" = None,
) -> dict:
    """
    Evaluate a nested `envs` dict: {task_group: {task_id: vec_env}}.
    This implementation flattens tasks, runs them sequentially or via ThreadPoolExecutor,
    accumulates per-group and overall statistics, and returns the same aggregate metrics
    schema as the single-env evaluator (avg_sum_reward / avg_max_reward / pc_success / timings)
    plus per-task infos.
    """
    start_t = time.time()

    # Flatten envs into list of (task_group, task_id, env)
    tasks = [(tg, tid, vec) for tg, group in envs.items() for tid, vec in group.items()]

    # accumulators: track metrics at both per-group level and across all groups
    group_acc: dict[str, dict[str, list]] = defaultdict(lambda: {k: [] for k in ACC_KEYS})
    overall: dict[str, list] = {k: [] for k in ACC_KEYS}
    per_task_infos: list[dict] = []

    # Track custom info metrics separately (dynamic keys)
    group_info_metrics: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    overall_info_metrics: dict[str, list] = defaultdict(list)

    def _json_default(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    def _save_partial_snapshot():
        if output_dir is None:
            return
        snapshot = {
            "per_task": list(per_task_infos),
            "partial": True,
        }
        out_path = output_dir / "eval_info.json"
        with open(out_path, "w") as f:
            json.dump(snapshot, f, indent=2, default=_json_default)

    # small inline helper to accumulate one task's metrics into accumulators
    def _accumulate_to(group: str, metrics: dict):
        # metrics expected to contain 'sum_rewards', 'max_rewards', 'successes', optionally 'video_paths'
        # but eval_one may store per-episode lists; we assume metrics uses scalars averaged per task as before.
        # To be robust, accept scalars or lists.
        def _append(key, value):
            if value is None:
                return
            if isinstance(value, list):
                group_acc[group][key].extend(value)
                overall[key].extend(value)
            else:
                group_acc[group][key].append(value)
                overall[key].append(value)

        _append("sum_rewards", metrics.get("sum_rewards"))
        _append("max_rewards", metrics.get("max_rewards"))
        _append("successes", metrics.get("successes"))
        for key in ("video_paths", "predicted_video_paths"):
            paths = metrics.get(key, [])
            if paths:
                group_acc[group][key].extend(paths)
                overall[key].extend(paths)

        # Handle custom info metrics
        info_metrics = metrics.get("info_metrics", {})
        for metric_name, metric_values in info_metrics.items():
            if isinstance(metric_values, list):
                group_info_metrics[group][metric_name].extend(metric_values)
                overall_info_metrics[metric_name].extend(metric_values)
            else:
                group_info_metrics[group][metric_name].append(metric_values)
                overall_info_metrics[metric_name].append(metric_values)

    # Choose runner (sequential vs threaded)
    task_runner = partial(
        run_one,
        policy=policy,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        n_episodes=n_episodes,
        max_episodes_rendered=max_episodes_rendered,
        max_episodes_rendered_failed=max_episodes_rendered_failed,
        videos_dir=videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
        recording_dir=recording_dir,
        env_features=env_features,
        recording_repo_id=recording_repo_id,
        recording_private=recording_private,
        intervention_ctx=intervention_ctx,
    )

    # Set the shared policy's mode before launching any workers. Restoring it
    # inside individual tasks would let one task enable training mode while
    # another task is still evaluating.
    was_training = policy.training
    policy.eval()
    try:
        if max_parallel_tasks <= 1:
            prefetch_thread: threading.Thread | None = None
            for i, (task_group, task_id, env) in enumerate(tasks):
                if prefetch_thread is not None:
                    prefetch_thread.join()
                    prefetch_thread = None

                try:
                    tg, tid, metrics = task_runner(task_group, task_id, env)
                    _accumulate_to(tg, metrics)
                    per_task_infos.append({"task_group": tg, "task_id": tid, "metrics": metrics})
                    _save_partial_snapshot()
                except BaseException as _eval_exc:
                    # Print the original exception before the finally clause runs
                    # env.close(), because close() itself can SIGABRT inside
                    # pybullet's GUI cleanup, which would replace the original
                    # traceback with a faulthandler trace pointing at cleanup
                    # instead of the real cause.
                    import traceback as _tb

                    print(
                        "\n=== task_runner raised an exception "
                        "(printed before env.close() because close may crash natively) ===",
                        flush=True,
                    )
                    _tb.print_exception(type(_eval_exc), _eval_exc, _eval_exc.__traceback__)
                    print("=== end of original-exception traceback ===\n", flush=True)
                    raise
                finally:
                    if close_envs_after_eval:
                        env.close()
                    # Prefetch next task's workers *after* closing current env to prevent
                    # GPU memory overlap between consecutive tasks.
                    if i + 1 < len(tasks):
                        next_env = tasks[i + 1][2]
                        if hasattr(next_env, "_ensure"):
                            prefetch_thread = threading.Thread(target=next_env._ensure, daemon=True)
                            prefetch_thread.start()
        else:
            with cf.ThreadPoolExecutor(max_workers=max_parallel_tasks) as executor:
                fut2meta = {}
                for task_group, task_id, env in tasks:
                    fut = executor.submit(task_runner, task_group, task_id, env)
                    fut2meta[fut] = (task_group, task_id, env)
                for fut in cf.as_completed(fut2meta):
                    tg, tid, env = fut2meta[fut]
                    try:
                        tg, tid, metrics = fut.result()
                        _accumulate_to(tg, metrics)
                        per_task_infos.append({"task_group": tg, "task_id": tid, "metrics": metrics})
                    finally:
                        if close_envs_after_eval:
                            env.close()
    finally:
        policy.train(was_training)

    # compute aggregated metrics helper (robust to lists/scalars)
    def _agg_from_list(xs):
        if not xs:
            return float("nan")
        arr = np.array(xs, dtype=float)
        return float(np.nanmean(arr))

    # compute per-group aggregates
    groups_aggregated = {}
    for group, acc in group_acc.items():
        group_agg = {
            "avg_sum_reward": _agg_from_list(acc["sum_rewards"]),
            "avg_max_reward": _agg_from_list(acc["max_rewards"]),
            "pc_success": _agg_from_list(acc["successes"]) * 100 if acc["successes"] else float("nan"),
            "n_episodes": len(acc["sum_rewards"]),
            "video_paths": list(acc["video_paths"]),
            "predicted_video_paths": list(acc["predicted_video_paths"]),
        }
        # Add custom info metrics for this group
        gim = group_info_metrics[group]
        for metric_name, metric_values in gim.items():
            group_agg[f"avg_{metric_name}"] = _agg_from_list(metric_values)
        if "episode_length" in gim and "truncated" in gim:
            non_trunc = [
                episode_len
                for episode_len, truncated in zip(gim["episode_length"], gim["truncated"], strict=True)
                if not truncated
            ]
            group_agg["avg_episode_length_without_truncation"] = (
                float(np.mean(non_trunc)) if non_trunc else None
            )
        groups_aggregated[group] = group_agg

    # overall aggregates
    overall_agg = {
        "avg_sum_reward": _agg_from_list(overall["sum_rewards"]),
        "avg_max_reward": _agg_from_list(overall["max_rewards"]),
        "pc_success": _agg_from_list(overall["successes"]) * 100 if overall["successes"] else float("nan"),
        "n_episodes": len(overall["sum_rewards"]),
        "eval_s": time.time() - start_t,
        "eval_ep_s": (time.time() - start_t) / max(1, len(overall["sum_rewards"])),
        "video_paths": list(overall["video_paths"]),
        "predicted_video_paths": list(overall["predicted_video_paths"]),
    }
    # Add custom info metrics to overall
    for metric_name, metric_values in overall_info_metrics.items():
        overall_agg[f"avg_{metric_name}"] = _agg_from_list(metric_values)
    if "episode_length" in overall_info_metrics and "truncated" in overall_info_metrics:
        non_trunc = [
            episode_len
            for episode_len, truncated in zip(
                overall_info_metrics["episode_length"], overall_info_metrics["truncated"], strict=True
            )
            if not truncated
        ]
        overall_agg["avg_episode_length_without_truncation"] = (
            float(np.mean(non_trunc)) if non_trunc else None
        )

    return {
        "per_task": per_task_infos,
        "per_group": groups_aggregated,
        "overall": overall_agg,
    }


def main():
    init_logging()
    register_third_party_plugins()
    eval_main()


if __name__ == "__main__":
    main()
