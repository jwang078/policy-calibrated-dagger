"""Contract between pcdagger and its two dependencies, lerobot (the jwang078 fork) and SplatSim.

Run this first after updating either dependency (``bash scripts/check_dependency_contract.sh``). It needs
no GPU, no datasets and no checkpoints: it only imports things and reads signatures.

1. every ``lerobot`` / ``lerobot_env_splatsim`` / ``splatsim`` name that pcdagger/ and scripts/ import
   (scanned from the source with ``ast``, so the list never goes stale) still exists;
2. the fork-only hooks pcdagger relies on keep the signatures it calls them with;
3. the SplatSim environment plugin still registers ``--env.type=splatsim`` through lerobot's plugin
   discovery.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DEP_PREFIXES = ("lerobot", "lerobot_env_splatsim", "splatsim")


def _scan_imports() -> set[tuple[str, str | None]]:
    """(module, name) pairs imported from the dependencies anywhere in pcdagger/ and scripts/."""
    found: set[tuple[str, str | None]] = set()
    for root in ("pcdagger", "scripts"):
        for py in (REPO / root).rglob("*.py"):
            try:
                tree = ast.parse(py.read_text(), filename=str(py))
            except SyntaxError as e:  # a broken script is its own bug, not a dependency drift
                pytest.fail(f"{py}: {e}")
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    top = node.module.split(".")[0]
                    if top in DEP_PREFIXES:
                        for alias in node.names:
                            found.add((node.module, None if alias.name == "*" else alias.name))
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split(".")[0] in DEP_PREFIXES:
                            found.add((alias.name, None))
    return found


IMPORTS = sorted(_scan_imports(), key=lambda t: (t[0], t[1] or ""))


@pytest.mark.parametrize("module,name", IMPORTS, ids=[f"{m}:{n or '*'}" for m, n in IMPORTS])
def test_dependency_name_exists(module: str, name: str | None) -> None:
    try:
        mod = importlib.import_module(module)
    except ImportError:
        if name is None:
            raise
        # ``from pkg import submodule`` form
        importlib.import_module(f"{module}.{name}")
        return
    if name is not None and not hasattr(mod, name):
        importlib.import_module(f"{module}.{name}")


def _params(fn) -> set[str]:
    return set(inspect.signature(fn).parameters)


def test_fork_dataset_hooks() -> None:
    from lerobot.configs.default import DatasetConfig
    from lerobot.datasets.factory import resolve_delta_timestamps, unconsumed_camera_keys
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.multi_dataset import MultiLeRobotDataset

    assert {"exclude_keys", "rename_map"} <= _params(resolve_delta_timestamps)
    assert {"policy_cfg", "ds_meta", "rename_map"} <= _params(unconsumed_camera_keys)
    assert "exclude_features" in _params(LeRobotDataset.__init__)
    assert {"exclude_features", "episodes"} <= _params(MultiLeRobotDataset.__init__)
    assert isinstance(inspect.getattr_static(MultiLeRobotDataset, "cumulative_sizes"), property)
    fields = {f.name for f in DatasetConfig.__dataclass_fields__.values()}
    assert {
        "repo_ids", "sample_weights", "stats_paths", "norm_mode", "use_weighted_sampling", "stats_path",
        "multi_source_episodes", "multi_source_feature_intersection", "dart_relabel",
        "dart_state_noise_std", "dart_state_noise_schedule", "dart_raw_mix", "dart_mask_hold_tail",
        "dart_selective_mask", "blend_collision_filter", "observation_noise_std",
    } <= fields


def test_fork_policy_hooks() -> None:
    from lerobot.policies.common.flow_matching import euler_integrate
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionModel, DiffusionPolicy
    from lerobot.policies.pretrained import PreTrainedPolicy
    from lerobot.processor.relative_action_processor import RelativeActionsProcessorStep

    from pcdagger.compat import peft_available, reconnect_relative_absolute_steps

    assert callable(reconnect_relative_absolute_steps) and isinstance(peft_available(), bool)
    assert {"t_start", "dt", "post_step"} <= _params(euler_integrate)
    assert {"noise", "sa_noise_ratio", "chunk_anchor", "generator"} <= _params(DiffusionModel.generate_actions)
    assert {"noise", "sa_noise_ratio"} <= _params(DiffusionModel.conditional_sample)
    assert "noise" in _params(DiffusionPolicy.predict_action_chunk)
    assert callable(getattr(PreTrainedPolicy, "predict_action_chunk", None))
    assert callable(RelativeActionsProcessorStep.refresh_anchor)
    fields = {f.name for f in DiffusionConfig.__dataclass_fields__.values()}
    assert {"shared_autonomy_config", "temporal_ensemble_config", "last_mile_config"} <= fields


def test_fork_config_modules() -> None:
    from lerobot.configs.eval import EvalPipelineConfig
    from lerobot.configs.intervention import InterventionConfig  # noqa: F401
    from lerobot.configs.last_mile import LastMileConfig  # noqa: F401
    from lerobot.configs.shared_autonomy import SharedAutonomyConfig  # noqa: F401
    from lerobot.configs.temporal_ensemble import TemporalEnsembleConfig  # noqa: F401
    from lerobot.configs.train import TrainPipelineConfig

    assert "intervention" in {f.name for f in EvalPipelineConfig.__dataclass_fields__.values()}
    assert "rename_map" in {f.name for f in TrainPipelineConfig.__dataclass_fields__.values()}


def test_splatsim_env_plugin_registers() -> None:
    from lerobot.envs.configs import EnvConfig
    from lerobot.utils.import_utils import register_third_party_plugins

    register_third_party_plugins()
    assert "splatsim" in EnvConfig.get_known_choices()
    from lerobot_env_splatsim.recording import FrameSource, TeleopRecordingContext  # noqa: F401
    from lerobot_env_splatsim.seeding import seed_splatsim_env_to_state, set_env_benchmark_indices  # noqa: F401


def test_splatsim_helpers() -> None:
    from splatsim.utils.lerobot_utils import (  # noqa: F401
        build_lerobot_features,
        create_lerobot_dataset,
        finalize_lerobot_dataset,
        load_lerobot_dataset,
    )
    from splatsim.utils.paths import resolve_splatsim_path
    from splatsim.utils.rrt_to_goal import extract_task_goal, planner_kwargs_from_traj_config

    assert callable(resolve_splatsim_path) and callable(extract_task_goal)
    assert callable(planner_kwargs_from_traj_config)


def test_splatsim_robots_expose_tool_frame() -> None:
    """The shared-autonomy wrapper plans in the frame named by ``wrist_camera_link_name``; both paper
    robots must resolve it (it silently became None for the UR5 after the 2026-09-18 asset refactor,
    which broke every lever intervention recording with "Link 'None' not found in URDF")."""
    from splatsim.configs.env_config import SplatObjectConfig

    for robot in ("planar_3joint", "robot_iphone_w_engine_curtain"):
        cfg = SplatObjectConfig(name="robot", splat_name=robot)
        assert cfg.wrist_camera_link_name, f"{robot}: wrist_camera_link_name is unset"
        assert cfg.urdf_path


def test_console_scripts_import() -> None:
    from pcdagger.dagger.eval import main as eval_main
    from pcdagger.train import main as train_main

    assert callable(train_main) and callable(eval_main)
