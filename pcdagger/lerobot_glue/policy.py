"""Policy wrappers applied after ``lerobot.policies.factory.make_policy``.

These used to be ``_wrap_with_*`` helpers inside the lerobot fork's policy factory; they are
plain post-factory wrappers, so they live here and ``pcdagger.train`` / ``pcdagger.dagger.eval``
call :func:`wrap_policy` right after ``make_policy``.
"""

from __future__ import annotations

import logging

from lerobot.processor import (
    PolicyProcessorPipeline,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


def _wrap_with_shared_autonomy(policy, cfg):
    """Wrap a policy with SharedAutonomyPolicyWrapper.

    Builds an inverse postprocessor (raw action → normalized action) by loading
    the postprocessor from the pretrained checkpoint and extracting its stats.
    """
    from pcdagger.blend.wrapper import SharedAutonomyPolicyWrapper
    from lerobot.processor.device_processor import DeviceProcessorStep
    from lerobot.processor.normalize_processor import NormalizerProcessorStep, UnnormalizerProcessorStep

    sa_cfg = cfg.shared_autonomy_config

    # Load the postprocessor from pretrained to get normalization stats
    pretrained_path = cfg.pretrained_path
    if pretrained_path is None:
        raise ValueError(
            "Shared autonomy requires a pretrained model (need normalization stats). "
            "Set pretrained_path in the policy config."
        )

    postprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=pretrained_path,
        config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )

    # Extract features, norm_map, and stats from the postprocessor's UnnormalizerProcessorStep.
    # We use the postprocessor's own config rather than cfg.output_features so that the
    # features and stats keys are guaranteed to match (important when rename_map is used
    # during training and obs keys get renamed in the saved stats).
    postprocessor_features = None
    postprocessor_norm_map = None
    postprocessor_stats = None
    for step in postprocessor.steps:
        if hasattr(step, "stats") and step.stats:
            postprocessor_features = getattr(step, "features", None)
            postprocessor_norm_map = getattr(step, "norm_map", None)
            postprocessor_stats = step.stats
            break

    if postprocessor_stats is None:
        logging.warning(
            "Could not extract normalization stats from postprocessor. "
            "Inverse postprocessor will not normalize human actions."
        )

    # Fall back to cfg values if we couldn't extract from postprocessor
    features = postprocessor_features or cfg.output_features
    norm_map = postprocessor_norm_map or cfg.normalization_mapping

    # Build inverse postprocessor: normalizes raw actions to policy's internal space.
    # Use zero_variance_denom=2.0 so that action dimensions with zero training
    # variance (e.g. gripper always 0) map non-zero guidance values to finite
    # normalised values instead of ~1/eps (≈1e8), which would corrupt the denoiser.
    # This only affects QUANTILES dimensions where q99-q01 ≈ 0; all other dims
    # are unchanged. Standard lerobot train/eval paths leave zero_variance_denom=None.
    inverse_steps = [
        NormalizerProcessorStep(
            features=features,
            norm_map=norm_map,
            stats=postprocessor_stats,
            zero_variance_denom=2.0,
        ),
        DeviceProcessorStep(device=cfg.device),
    ]
    inverse_postprocessor = PolicyProcessorPipeline(
        steps=inverse_steps,
        name="inverse_postprocessor",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )

    # Load preprocessor to get observation.state normalization stats
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=pretrained_path,
        config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )

    preprocessor_features = None
    preprocessor_norm_map = None
    preprocessor_stats = None
    for step in preprocessor.steps:
        if hasattr(step, "stats") and step.stats:
            preprocessor_features = getattr(step, "features", None)
            preprocessor_norm_map = getattr(step, "norm_map", None)
            preprocessor_stats = step.stats
            break

    # Build inverse_preprocessor: normalized obs.state → raw joint values
    inverse_preprocessor = None
    if preprocessor_stats is not None:
        inverse_preprocessor = PolicyProcessorPipeline(
            steps=[
                UnnormalizerProcessorStep(
                    features=preprocessor_features or cfg.input_features,
                    norm_map=preprocessor_norm_map or cfg.normalization_mapping,
                    stats=preprocessor_stats,
                ),
                DeviceProcessorStep(device=cfg.device),
            ],
            name="inverse_preprocessor",
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )
    else:
        logging.warning(
            "Could not extract normalization stats from preprocessor. "
            "IK will use raw (possibly wrong) obs.state values."
        )

    wrapped = SharedAutonomyPolicyWrapper(
        inner_policy=policy,
        inverse_postprocessor=inverse_postprocessor,
        postprocessor=postprocessor,
        inverse_preprocessor=inverse_preprocessor,
        forward_flow_ratio=sa_cfg.forward_flow_ratio,
        show_slider=sa_cfg.show_slider,
        pybullet_gui=getattr(sa_cfg, "pybullet_gui", None),
        start_paused=sa_cfg.start_paused,
        robot_name=sa_cfg.robot_name,
        num_dofs=sa_cfg.num_dofs,
        blend_mode=sa_cfg.blend_mode,
        anchor_prefix_steps=getattr(sa_cfg, "anchor_prefix_steps", 0),
        anchor_suffix_steps=getattr(sa_cfg, "anchor_suffix_steps", 0),
        anchor_suffix_to_goal=getattr(sa_cfg, "anchor_suffix_to_goal", False),
        anchor_every_denoise_step=getattr(sa_cfg, "anchor_every_denoise_step", True),
        rtc_prev_chunk_guidance=getattr(sa_cfg, "rtc_prev_chunk_guidance", False),
        rtc_max_guidance_weight=getattr(sa_cfg, "rtc_max_guidance_weight", 10.0),
        rtc_execution_horizon=getattr(sa_cfg, "rtc_execution_horizon", None),
        rtc_inference_delay=getattr(sa_cfg, "rtc_inference_delay", 0),
        rtc_prefix_attention_schedule=getattr(sa_cfg, "rtc_prefix_attention_schedule", "linear"),
        fps=sa_cfg.fps,
        rrt_collision_detection=sa_cfg.rrt_collision_detection,
        rrt_camera_score_weight=getattr(sa_cfg, "rrt_camera_score_weight", None),
        rrt_ik_camera_weight=getattr(sa_cfg, "rrt_ik_camera_weight", None),
        rrt_wrist_camera_link_name=getattr(sa_cfg, "rrt_wrist_camera_link_name", None),
        rrt_pre_jump_lookback=sa_cfg.pre_jump_lookback,
        rrt_future_chunk=sa_cfg.future_chunk,
        rrt_teleport_to_q_start=sa_cfg.rrt_teleport_to_q_start,
        rrt_blocking_plan=sa_cfg.rrt_blocking_plan,
        rrt_path_selection=sa_cfg.rrt_path_selection,
        rrt_path_score_joint_arc_weight=sa_cfg.rrt_path_score_joint_arc_weight,
        rrt_segment_at_sharp_corners=sa_cfg.rrt_segment_at_sharp_corners,
        rrt_ik_goal_selection=sa_cfg.rrt_ik_goal_selection,
        rrt_ik_accept_arc_chord_ratio=sa_cfg.rrt_ik_accept_arc_chord_ratio,
        rrt_num_path_candidates_per_ik=sa_cfg.rrt_num_path_candidates_per_ik,
        rrt_max_path_attempts_per_ik=sa_cfg.rrt_max_path_attempts_per_ik,
        rrt_path_perturbation_scale=sa_cfg.rrt_path_perturbation_scale,
        rrt_num_ik_candidates=sa_cfg.rrt_num_ik_candidates,
        rrt_obstacle_clearance=sa_cfg.rrt_obstacle_clearance,
        rrt_self_collision_clearance=sa_cfg.rrt_self_collision_clearance,
        rrt_in_progress_obstacle_clearance=sa_cfg.rrt_in_progress_obstacle_clearance,
        rrt_in_progress_self_collision_clearance=sa_cfg.rrt_in_progress_self_collision_clearance,
        rrt_self_collision_skip_pairs=sa_cfg.rrt_self_collision_skip_pairs,
        rrt_diagnostic_log_pairs=sa_cfg.rrt_diagnostic_log_pairs,
        rrt_ik_skip_gripper_obstacle_pairs=sa_cfg.rrt_ik_skip_gripper_obstacle_pairs,
        rrt_escape_clearance_factor=sa_cfg.rrt_escape_clearance_factor,
        rrt_rewind_clearance_factor=sa_cfg.rrt_rewind_clearance_factor,
        rrt_final_approach_dist=sa_cfg.rrt_final_approach_dist,
        rrt_final_approach_vel_scale=sa_cfg.rrt_final_approach_vel_scale,
        rrt_final_approach_acc_scale=sa_cfg.rrt_final_approach_acc_scale,
        rrt_max_joint_vel=sa_cfg.rrt_max_joint_vel,
        rrt_max_joint_acc=sa_cfg.rrt_max_joint_acc,
        rrt_max_joint_jerk=sa_cfg.rrt_max_joint_jerk,
        rrt_smooth_iterations=sa_cfg.rrt_smooth_iterations,
        rrt_elastic_smooth_passes=sa_cfg.rrt_elastic_smooth_passes,
        rrt_uniform_path_speed=sa_cfg.rrt_uniform_path_speed,
        rrt_trajopt_passes=sa_cfg.rrt_trajopt_passes,
        rrt_trajopt_lr=sa_cfg.rrt_trajopt_lr,
        rrt_trajopt_smoothness_weight=sa_cfg.rrt_trajopt_smoothness_weight,
        rrt_trajopt_collision_weight=sa_cfg.rrt_trajopt_collision_weight,
        rrt_trajopt_collision_threshold=sa_cfg.rrt_trajopt_collision_threshold,
        rrt_trajopt_fd_step=sa_cfg.rrt_trajopt_fd_step,
        rrt_abort_on_drift_rad=sa_cfg.rrt_abort_on_drift_rad,
        rrt_abort_on_drift_ticks=sa_cfg.rrt_abort_on_drift_ticks,
        rrt_drift_trigger=sa_cfg.rrt_drift_trigger,
        shield_check_every_n_ticks=getattr(sa_cfg, "shield_check_every_n_ticks", 1),
        debug_shield_force_trigger=getattr(sa_cfg, "debug_shield_force_trigger", False),
        debug_shield_trace_anchor=getattr(sa_cfg, "debug_shield_trace_anchor", False),
        debug_rrt_drift_log=getattr(sa_cfg, "debug_rrt_drift_log", False),
    )

    # Connect shared context for teleop recording (if active)
    from lerobot_env_splatsim.recording import TeleopRecordingContext

    ctx = TeleopRecordingContext.get_instance()
    wrapped._teleop_context = ctx

    logging.info(
        f"Wrapped policy with SharedAutonomyPolicyWrapper (forward_flow_ratio={sa_cfg.forward_flow_ratio})"
    )
    return wrapped


def _wrap_with_temporal_ensemble(policy, cfg):
    """Wrap a policy with TemporalEnsemblePolicyWrapper.

    Pure inference-time smoothing of chunk boundaries via online
    exponentially-weighted averaging of overlapping chunks (ACT's Algorithm 2,
    generalised to any chunk-predicting policy).

    Validates ``n_action_steps <= chunk_size``. Smaller ``n_action_steps``
    gives more smoothing at higher inference cost; ``n_action_steps=1`` is
    the maximum-smoothness setting and queries the model every step.
    """
    from pcdagger.extras.temporal_ensemble.wrapper import TemporalEnsemblePolicyWrapper

    te_cfg = cfg.temporal_ensemble_config
    chunk_size = getattr(cfg, "chunk_size", None)
    n_action_steps = getattr(cfg, "n_action_steps", 1)
    if chunk_size is not None and n_action_steps >= chunk_size:
        # With n_action_steps == chunk_size the wrapper consumes every entry
        # from the smoothed buffer before the next update; the next update
        # then sees an empty buffer (after skip(K-1) = skip(chunk_size-1)) and
        # appends the new chunk as fresh entries with count=1. No averaging
        # ever happens — the wrapper degenerates to "no smoothing." Block this
        # so users don't silently get a no-op.
        raise ValueError(
            f"temporal_ensemble_config.enabled=True requires n_action_steps "
            f"({n_action_steps}) < chunk_size ({chunk_size}) — when equal, the "
            f"ensembler buffer empties between chunks and no smoothing occurs. "
            f"Lower n_action_steps to get chunk_size/n_action_steps averaged "
            f"predictions per future timestep (n_action_steps=1 = maximum smoothing)."
        )
    if n_action_steps != 1:
        logging.warning(
            "TemporalEnsemble: inner policy has n_action_steps=%d. The wrapper supports "
            "K>1 via ensembler.skip(), but smaller K gives MORE smoothing (more averaged "
            "predictions per future timestep). For maximum smoothness pass "
            "`--policy.n_action_steps=1` explicitly.",
            n_action_steps,
        )
    wrapped = TemporalEnsemblePolicyWrapper(policy, te_cfg)
    logging.info(
        "Wrapped policy with TemporalEnsemblePolicyWrapper (coeff=%.4f, n_action_steps=%d)",
        te_cfg.coeff,
        n_action_steps,
    )
    return wrapped


def _wrap_with_last_mile(policy, cfg):
    """Wrap a policy with LastMileWrapper.

    Eval-time help mechanism with pluggable detect + help backends. See
    ``lerobot.configs.last_mile.LastMileConfig`` for the backend list and
    per-backend params. The help is staged in the wrapper's ``select_action``
    and APPLIED by lerobot_eval after the postprocessor (raw joint space),
    via ``wrapper.apply_help``.

    When ``help_backend == "rrt_to_goal"``, an outer ``SharedAutonomyPolicyWrapper``
    must already be in the stack — the RRT helper delegates to its
    ``trigger_rrt_to_goal`` / ``is_rrt_active``. The factory walks the inner
    policy chain to find SA and registers it on the wrapper. Raises if SA is
    not found.
    """
    from pcdagger.extras.last_mile import LastMileWrapper
    from pcdagger.blend.wrapper import SharedAutonomyPolicyWrapper

    last_mile_cfg = cfg.last_mile_config
    wrapped = LastMileWrapper(inner_policy=policy, cfg=last_mile_cfg)
    logging.info(
        "Wrapped policy with LastMileWrapper (detect=%s, help=%s)",
        last_mile_cfg.detect_backend,
        last_mile_cfg.help_backend,
    )

    if last_mile_cfg.help_backend == "rrt_to_goal":
        # Walk the wrapper chain for an SA wrapper.
        sa = policy
        while sa is not None and not isinstance(sa, SharedAutonomyPolicyWrapper):
            sa = getattr(sa, "inner_policy", None)
        if sa is None:
            raise ValueError(
                "LastMileConfig.help_backend='rrt_to_goal' requires a "
                "SharedAutonomyPolicyWrapper to also be enabled (set "
                "shared_autonomy_config.enabled=true)."
            )
        wrapped.register_sa_wrapper(sa)
        logging.info("LastMileWrapper: registered SharedAutonomyPolicyWrapper for RRT help.")

    return wrapped


