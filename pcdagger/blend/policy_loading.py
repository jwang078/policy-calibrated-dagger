"""``SharedAutonomyPolicyWrapper`` loader — assembles the wrapper + obs
preprocessor pipeline used by the visualization / debugging scripts in
``my_scripts/``.

Extracted from ``visualize_shared_autonomy_DEPRECATED`` so downstream callers
don't have to depend on a deprecated module.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from lerobot.configs.shared_autonomy import SharedAutonomyConfig
from lerobot.policies.factory import (
    _reconnect_relative_absolute_steps,
    _wrap_with_shared_autonomy,
    get_policy_class,
)
from lerobot.policies.shared_autonomy_wrapper import (
    GuidanceBlendStrategy,
    PolicyGuidanceRepresentation,
)
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import POLICY_PREPROCESSOR_DEFAULT_NAME
from lerobot.utils.lerobot_dataset_utils import resolve_dataset_dir


def load_wrapped_policy(
    policy_path: str | Path,
    forward_flow_ratio: float = 1.0,
    robot_name: str = "robot_iphone_w_engine_new",
    num_dofs: int = 6,
    device: str = "cpu",
    action_names_dataset_hint: str | Path | None = None,
):
    """Load inner policy and wrap with ``SharedAutonomyPolicyWrapper`` (no slider).

    Uses ``ABSOLUTE_POS`` guidance representation so that dataset joint
    positions are passed directly as guidance without FK→IK conversion.

    Returns ``(wrapper, obs_preprocessor)``.
    """
    policy_path = Path(policy_path)
    with open(policy_path / "config.json") as f:
        config_data = json.load(f)
    policy_type = config_data["type"]

    policy_cls = get_policy_class(policy_type)
    inner_policy = policy_cls.from_pretrained(str(policy_path))
    inner_policy = inner_policy.to(device).eval()

    cfg = SimpleNamespace(
        pretrained_path=str(policy_path),
        device=device,
        output_features=getattr(inner_policy.config, "output_features", None),
        input_features=getattr(inner_policy.config, "input_features", None),
        normalization_mapping=getattr(inner_policy.config, "normalization_mapping", None),
        shared_autonomy_config=SharedAutonomyConfig(
            enabled=True,
            forward_flow_ratio=forward_flow_ratio,
            show_slider=False,
            start_paused=False,
            robot_name=robot_name,
            num_dofs=num_dofs,
        ),
    )

    wrapper = _wrap_with_shared_autonomy(inner_policy, cfg)
    wrapper.policy_guidance_representation = PolicyGuidanceRepresentation.ABSOLUTE_POS
    wrapper.guidance_blend_strategy = GuidanceBlendStrategy.DENOISE  # default, overridden by caller
    wrapper = wrapper.to(device).eval()

    # Observation preprocessor (normalizes obs.state; images pass through unchanged).
    # Use default batch_to_transition / transition_to_batch so it accepts / returns dicts.
    obs_preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=str(policy_path),
        config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
    )
    _reconnect_relative_absolute_steps(obs_preprocessor, wrapper.postprocessor, policy=wrapper)
    _backfill_rel_step_action_names(obs_preprocessor, policy_path, dataset_hint=action_names_dataset_hint)

    return wrapper, obs_preprocessor


def _backfill_rel_step_action_names(
    preprocessor, policy_path: Path, *, dataset_hint: str | Path | None = None
) -> None:
    """If the ``RelativeActionsProcessorStep`` in ``preprocessor`` has
    ``action_names=None``, look up the action feature names from any available
    dataset and patch them in.

    Source precedence:
      1. ``dataset_hint`` (caller-supplied path or repo id — most reliable when
         the train_config's dataset has since been deleted, e.g. orchestrator
         merged datasets that get cleaned up after training).
      2. ``train_config.json`` → ``dataset.repo_id`` → ``meta/info.json``.

    WHY: training-time ``make_policy`` only populates ``cfg.action_feature_names``
    when the policy config class declares the field. PI0 / PI0.5 / PI0FAST
    configs declare it; DiffusionConfig does NOT (fixed in a later commit for
    new training, but existing checkpoints still have ``action_names: null``).
    Without this backfill, the rel-step's ``_build_mask`` falls back to
    ``[True] * action_dim`` — ``exclude_joints`` is silently ignored, all dims
    (including gripper) get converted to relative actions, and at
    blend-recording time the paired ``AbsoluteActionsProcessorStep`` adds the
    gripper STATE back to the policy's near-zero rel output, leaking
    ``gripper_state`` into the recorded action. Backfilling here makes
    ``exclude_joints=['gripper']`` actually apply.
    """
    from lerobot.processor.relative_action_processor import RelativeActionsProcessorStep

    rel_step = next(
        (s for s in preprocessor.steps if isinstance(s, RelativeActionsProcessorStep)),
        None,
    )
    if rel_step is None or rel_step.action_names is not None or not rel_step.exclude_joints:
        return

    # Build candidate list of dataset locations to try.
    candidates: list[tuple[str, Path]] = []
    if dataset_hint is not None:
        hint_p = Path(dataset_hint).expanduser()
        if hint_p.is_dir():
            candidates.append(("dataset_hint (path)", hint_p))
        else:
            # Treat as a repo id.
            try:
                candidates.append(("dataset_hint (repo_id)", Path(resolve_dataset_dir(str(dataset_hint)))))
            except Exception:
                pass
    train_cfg_path = Path(policy_path) / "train_config.json"
    if train_cfg_path.is_file():
        try:
            train_cfg = json.loads(train_cfg_path.read_text())
            repo_id = (train_cfg.get("dataset") or {}).get("repo_id")
            if repo_id:
                try:
                    candidates.append(("train_config.dataset.repo_id", Path(resolve_dataset_dir(repo_id))))
                except Exception:
                    pass
        except (OSError, json.JSONDecodeError):
            pass

    for label, ds_dir in candidates:
        # `resolve_dataset_dir` returns the `data/` subdir; meta/ lives in the
        # dataset ROOT. Look in `ds_dir/meta/` first (in case a caller passed
        # the root directly), then in `ds_dir.parent/meta/` (the data-subdir case).
        for meta_root in (ds_dir, ds_dir.parent):
            info_path = meta_root / "meta" / "info.json"
            if info_path.is_file():
                break
        else:
            continue
        try:
            info = json.loads(info_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        names = (info.get("features", {}).get("action", {}) or {}).get("names")
        if not names:
            continue
        rel_step.action_names = list(names)
        print(
            f"[load_wrapped_policy] backfilled rel-step.action_names = {list(names)} "
            f"from {info_path}  [source={label}]  "
            f"(exclude_joints={rel_step.exclude_joints} now actually applies)"
        )
        return

    print(
        f"[load_wrapped_policy] WARNING: rel-step has exclude_joints="
        f"{rel_step.exclude_joints} but action_names=None and no source dataset "
        f"could be resolved (tried: dataset_hint + train_config.json). "
        f"exclude_joints will be IGNORED — all dims including gripper will be "
        f"converted to relative actions, and recorded blend actions will leak "
        f"the gripper state into the action column."
    )
