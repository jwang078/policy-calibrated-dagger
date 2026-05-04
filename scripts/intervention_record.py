"""DAgger-style intervention recording across SplatSim eval-benchmark scenarios.

Runs a SharedAutonomyPolicyWrapper-wrapped policy through every scenario in an
eval-benchmark dataset. When the policy hasn't reached the env's success
condition after `policy_steps_before_rrt` (default 400), the controller toggles
the wrapper's RRT mode, lets it execute a random number of waypoints in
[rrt_steps_min, rrt_steps_max] (60..100 by default), then cancels — handing
control back to the policy. The cycle repeats up to `max_cycles_per_scenario`
times before advancing.

The wrapper's existing TeleopRecordingWrapper writes one dataset episode per
RRT correction segment (frames tagged FrameSource.RRT). The result is a
DAgger-style correction dataset to fold into retraining.

Usage: same CLI as ``lerobot-eval`` (it parses the same EvalPipelineConfig),
but drops --env.external_port (the script spawns SplatSim locally) and adds
--env.eval_benchmark_repo_id. See the plan file for a full launch command.
"""

# NOTE: do not add `from __future__ import annotations` — parser.wrap reads
# the function's annotation at runtime to infer the draccus config class, and
# stringified annotations break that lookup.

import csv
import logging
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm

from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs import (
    make_env,
    make_env_pre_post_processors,
    preprocess_observation,
)
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.policies.factory import (
    _reconnect_relative_absolute_steps,
    _wrap_with_shared_autonomy,
)
from lerobot.policies.rrt_to_goal import RRTMode
from lerobot.policies.shared_autonomy_wrapper import SharedAutonomyPolicyWrapper
from lerobot.utils.constants import ACTION
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Intervention controller
# ---------------------------------------------------------------------------


@dataclass
class InterventionConfig:
    policy_steps_before_rrt: int = 400
    # After an RRT cycle has actually executed, check the policy's progress
    # more often: pick a random threshold in
    # [policy_steps_between_rrt_min, policy_steps_between_rrt_max] for each
    # subsequent trigger. Set min == max to disable randomization.
    policy_steps_between_rrt_min: int = 80
    policy_steps_between_rrt_max: int = 120
    rrt_steps_min: int = 60
    rrt_steps_max: int = 200
    max_cycles_per_scenario: int = 10
    max_plan_failures: int = 5
    # After this many consecutive backoff rounds (each = max_plan_failures
    # failed plans in a row, none of which executed), give up on the scenario
    # and advance. Stops the controller from spinning forever on a
    # configuration where the robot is stuck in a way the planner can't
    # escape from. cycles_used does NOT increase across backoff rounds (only
    # executed cycles count), so without this cap the loop is unbounded.
    max_backoff_rounds_per_scenario: int = 3


class InterventionController:
    """State machine driving policy/RRT alternation across one scenario.

    See plan file (``Step 2`` table) for the full transition table. The
    controller never touches the env directly — it only reads the wrapper's
    ``_rrt.mode`` and calls ``trigger_rrt_to_goal()`` / ``_cancel_rrt()``.
    """

    def __init__(
        self,
        wrapper: SharedAutonomyPolicyWrapper,
        cfg: InterventionConfig,
    ) -> None:
        self.wrapper = wrapper
        self.cfg = cfg
        # per-scenario state — set in ``reset_for_new_scenario``
        self.policy_step_count: int = 0
        self.rrt_step_count: int = 0
        self.target_rrt_steps: int = 0
        # Threshold of policy steps required before the next RRT trigger.
        # Starts at ``policy_steps_before_rrt`` for the first cycle; after the
        # first executed cycle, gets resampled from
        # [policy_steps_between_rrt_min, policy_steps_between_rrt_max] each
        # time it's reset, so post-intervention we check in more often.
        self.next_policy_threshold: int = cfg.policy_steps_before_rrt
        self.cycles_used: int = 0
        self.plan_failures: int = 0
        self.controller_initiated_cancel: bool = False
        self.prev_mode: RRTMode = RRTMode.IDLE
        # True from the tick we trigger an RRT plan until the wrapper either
        # transitions into EXECUTING (planning succeeded) or is observed back
        # in IDLE without having executed (planning failed). Robust to fast
        # PLANNING→IDLE transitions that finish entirely between two ticks
        # (e.g. when start-in-collision rejects before any actual RRT runs).
        self.pending_rrt_trigger: bool = False
        self.unexpected_natural_finish: bool = False
        # Set after a backoff fires (max_plan_failures hit). While True, the
        # collision trigger is suppressed so we don't burst-retrigger on the
        # very next tick — the policy gets the full backoff window to do
        # something. Cleared once policy_step_count crosses
        # next_policy_threshold (or on scenario reset).
        self.in_backoff_cooldown: bool = False
        # Number of completed backoff rounds in this scenario. Reset on
        # scenario reset; advance the scenario when this hits the configured
        # cap (otherwise unbounded since cycles_used only counts executed cycles).
        self.backoff_rounds: int = 0
        self.last_status: str = "running"

    def reset_for_new_scenario(self) -> None:
        self.policy_step_count = 0
        self.rrt_step_count = 0
        self.target_rrt_steps = 0
        self.next_policy_threshold = self.cfg.policy_steps_before_rrt
        self.cycles_used = 0
        self.plan_failures = 0
        self.controller_initiated_cancel = False
        self.prev_mode = RRTMode.IDLE
        self.pending_rrt_trigger = False
        self.unexpected_natural_finish = False
        self.in_backoff_cooldown = False
        self.backoff_rounds = 0
        self.last_status = "running"

    def _resample_post_intervention_threshold(self) -> None:
        """Pick the next ``policy_step_count`` threshold to use AFTER an RRT
        cycle has executed. Random uniform draw from the configured between
        range so the controller checks in on the policy more often (and at
        slightly varied cadences) once it has demonstrated it's intervening.
        """
        lo = max(1, self.cfg.policy_steps_between_rrt_min)
        hi = max(lo, self.cfg.policy_steps_between_rrt_max)
        self.next_policy_threshold = random.randint(lo, hi)

    def tick(self, success: bool, in_collision: bool = False) -> str:
        """Advance one step. Returns ``"continue"`` or ``"advance"``.

        ``in_collision`` is the env's current collision state (read from
        ``info["in_collision"]``). When the policy is driving (mode == IDLE)
        and the robot is in collision, we trigger RRT immediately rather than
        waiting for ``policy_step_count`` to reach the threshold — collisions
        mean the policy is already failing, so there's no reason to keep
        accumulating bad transitions.
        """
        mode: RRTMode = self.wrapper._rrt.mode
        prev_mode = self.prev_mode
        # Capture mode for next tick BEFORE branches that might mutate it via
        # _cancel_rrt — we want the mode the wrapper had when this tick started.
        self.prev_mode = mode

        if success:
            self.last_status = "success"
            return "advance"

        # Natural RRT finish: was EXECUTING last tick, now IDLE, controller
        # didn't cancel. Wait one more step so the env has a chance to register
        # success on the goal pose; the next tick handles the verdict.
        if prev_mode == RRTMode.EXECUTING and mode == RRTMode.IDLE and not self.controller_initiated_cancel:
            logger.warning(
                "RRT chunk exhausted on its own (natural finish). Waiting one step "
                "to see if the env reports success on the planned goal pose..."
            )
            self.unexpected_natural_finish = True
            self.cycles_used += 1
            self.policy_step_count = 0
            self.rrt_step_count = 0
            self.backoff_rounds = 0
            self.in_backoff_cooldown = False
            # An RRT cycle ran to completion — shorten cadence for next check.
            self._resample_post_intervention_threshold()
            return "continue"

        if self.unexpected_natural_finish:
            # Env did not report success this step → goal-vs-success mismatch.
            logger.warning(
                "Natural RRT finish did not produce env success. Possible "
                "mismatch between RRT goal pose and env success condition; "
                "marking scenario and advancing."
            )
            self.last_status = "rrt_finished_no_success"
            return "advance"

        # Plan failure detection. We use a "pending trigger" flag set the
        # moment we call trigger_rrt_to_goal(); if the next observation of
        # IDLE arrives WITHOUT the wrapper ever entering EXECUTING, planning
        # failed. This is robust to the wrapper completing PLANNING → IDLE
        # entirely between two of our ticks (which happens whenever planning
        # rejects fast, e.g. start-in-collision).
        if self.pending_rrt_trigger and mode == RRTMode.IDLE:
            self.pending_rrt_trigger = False
            self.plan_failures += 1
            logger.info(
                "RRT plan failed (attempt %d/%d).",
                self.plan_failures,
                self.cfg.max_plan_failures,
            )
            if self.plan_failures < self.cfg.max_plan_failures:
                logger.info("Retrying RRT plan...")
                self.wrapper.trigger_rrt_to_goal()
                self.pending_rrt_trigger = True
                return "continue"
            self.backoff_rounds += 1
            logger.warning(
                "RRT plan failed %d times in a row (backoff round %d/%d); "
                "letting the policy run for another %d steps before the next "
                "attempt. Collision-triggered RRT is suppressed during this window.",
                self.cfg.max_plan_failures,
                self.backoff_rounds,
                self.cfg.max_backoff_rounds_per_scenario,
                self.next_policy_threshold,
            )
            if self.backoff_rounds >= self.cfg.max_backoff_rounds_per_scenario:
                logger.warning(
                    "Hit max %d backoff round(s) for this scenario; advancing.",
                    self.cfg.max_backoff_rounds_per_scenario,
                )
                self.last_status = "max_backoff_rounds"
                return "advance"
            self.plan_failures = 0
            self.policy_step_count = 0
            self.in_backoff_cooldown = True
            return "continue"

        if mode == RRTMode.PLANNING:
            return "continue"

        if mode == RRTMode.EXECUTING:
            # Planning succeeded — clear the pending flag so a future IDLE
            # transition is correctly treated as natural-finish (or our cancel),
            # not as a plan failure.
            self.pending_rrt_trigger = False
            self.rrt_step_count += 1
            if not self.controller_initiated_cancel and self.rrt_step_count >= self.target_rrt_steps:
                logger.info(
                    "Auto-cancelling RRT after %d step(s) (random target=%d).",
                    self.rrt_step_count,
                    self.target_rrt_steps,
                )
                self.wrapper._cancel_rrt()
                self.controller_initiated_cancel = True
                self.cycles_used += 1
                self.rrt_step_count = 0
                self.policy_step_count = 0
                # An RRT cycle just executed successfully — the planner is
                # working again, so clear backoff state.
                self.backoff_rounds = 0
                self.in_backoff_cooldown = False
                # An RRT cycle just executed — shorten the cadence for the
                # next check-in (sampled fresh each time for variation).
                self._resample_post_intervention_threshold()
                if self.cycles_used >= self.cfg.max_cycles_per_scenario:
                    logger.warning(
                        "Reached max %d intervention cycle(s) without success; advancing scenario.",
                        self.cfg.max_cycles_per_scenario,
                    )
                    self.last_status = "max_cycles_reached"
                    return "advance"
            return "continue"

        # mode == RRTMode.IDLE
        if self.cycles_used >= self.cfg.max_cycles_per_scenario:
            self.last_status = "max_cycles_reached"
            return "advance"

        # Reset the controller-cancel flag now that the cancel has settled.
        self.controller_initiated_cancel = False
        self.policy_step_count += 1
        # Two triggers, both gated on mode == IDLE: stall (policy_step_count
        # >= threshold) and collision (the policy has driven the robot into
        # an obstacle, no point in waiting). Trigger fires whichever first.
        # During backoff cooldown the collision trigger is suppressed so we
        # don't burst-retrigger right after a backoff (the policy gets the
        # full window to make progress on its own); the stall trigger still
        # fires on the threshold and lifts the cooldown.
        should_trigger_stall = self.policy_step_count >= self.next_policy_threshold
        should_trigger_collision = in_collision and not self.in_backoff_cooldown
        if should_trigger_stall:
            self.in_backoff_cooldown = False
        if should_trigger_stall or should_trigger_collision:
            self.target_rrt_steps = random.randint(self.cfg.rrt_steps_min, self.cfg.rrt_steps_max)
            reason = "in_collision" if should_trigger_collision else "stall"
            logger.info(
                "Triggering RRT (%s) after %d policy steps (cycle %d/%d, target=%d).",
                reason,
                self.policy_step_count,
                self.cycles_used + 1,
                self.cfg.max_cycles_per_scenario,
                self.target_rrt_steps,
            )
            self.plan_failures = 0
            self.rrt_step_count = 0
            # Reset the policy counter so a fast plan-fail on the next tick
            # can't burst-retrigger here on every step (the pending_rrt_trigger
            # branch above is the single source of truth for retries / backoff).
            self.policy_step_count = 0
            # Advertise our planned cancel point so the wrapper's "RRT
            # executing X / Y waypoints" log shows partial vs. total.
            self.wrapper._rrt.target_steps = self.target_rrt_steps
            self.wrapper.trigger_rrt_to_goal()
            self.pending_rrt_trigger = True
        return "continue"


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------


def _process_observation(
    observation: dict[str, Any],
    env: gym.vector.VectorEnv,
    env_preprocessor,
    preprocessor,
) -> dict[str, Any]:
    """Mirror lerobot_eval's per-step observation pipeline."""
    observation = preprocess_observation(observation)
    try:
        observation["task"] = list(env.call("task_description"))
    except (AttributeError, NotImplementedError):
        try:
            observation["task"] = list(env.call("task"))
        except (AttributeError, NotImplementedError):
            observation["task"] = [""] * env.num_envs

    observation = env_preprocessor(observation)
    observation = preprocessor(observation)

    # Inject oracle_env_config AFTER the preprocessor pipeline because the
    # standard converters drop unknown top-level keys.
    try:
        oracle_cfgs = env.call("get_env_config")
        if oracle_cfgs is not None and any(c is not None for c in oracle_cfgs):
            observation["oracle_env_config"] = oracle_cfgs[0]
    except (AttributeError, NotImplementedError):
        pass
    return observation


def _extract_info_bool(info: dict, key: str) -> bool:
    """Pull a single boolean metric out of either the live or final info dict.

    The simulator's ``check_metrics()`` is spread into ``info`` on every step
    (both local and ZMQ paths), so per-step env signals like ``is_success`` and
    ``in_collision`` are reachable here.
    """
    val = info["final_info"].get(key, False) if "final_info" in info else info.get(key, False)
    if hasattr(val, "tolist"):
        # Numpy array per-env; we run with num_envs=1, so just take the first.
        vals = val.tolist()
        return bool(vals[0]) if vals else False
    return bool(val)


def _extract_success(info: dict) -> bool:
    return _extract_info_bool(info, "is_success")


def _extract_in_collision(info: dict) -> bool:
    return _extract_info_bool(info, "in_collision")


@dataclass
class ScenarioResult:
    scenario_idx: int
    success: bool
    cycles_used: int
    status: str
    plan_failures: int


def run_intervention_rollout(
    env: gym.vector.VectorEnv,
    policy: SharedAutonomyPolicyWrapper,
    preprocessor,
    postprocessor,
    env_preprocessor,
    env_postprocessor,
    n_scenarios: int,
    intervention_cfg: InterventionConfig,
    csv_path: Path | None = None,
) -> list[ScenarioResult]:
    """Run the intervention loop over ``n_scenarios`` env scenarios."""
    assert env.num_envs == 1, (
        f"Intervention recording assumes a single env (eval.batch_size=1); got {env.num_envs}."
    )

    amp_device_type = policy.config.device if hasattr(policy.config, "device") else "cuda"
    use_amp = getattr(policy.config, "use_amp", False)
    max_steps_per_scenario = int(env.call("_max_episode_steps")[0])

    ctrl = InterventionController(policy, intervention_cfg)
    results: list[ScenarioResult] = []

    csv_writer = None
    csv_file = None
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        # File lifetime spans the whole rollout (closed in `finally`), so a
        # context manager would have to wrap the entire function body — keep
        # the explicit open + `finally: csv_file.close()` below instead.
        csv_file = open(csv_path, "w", newline="")  # noqa: SIM115
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["scenario_idx", "success", "cycles_used", "status", "plan_failures"])
        csv_file.flush()

    # The recording wrapper reads source_scenario_idx off the shared
    # TeleopRecordingContext when it saves an episode, so we push the index
    # there on each reset. Cleared in `finally` so a stale value from this run
    # can't bleed into a later non-controller-driven recording.
    from lerobot.policies.teleop_recording import TeleopRecordingContext

    teleop_ctx = TeleopRecordingContext.get_instance()
    # Defer dataset writes until we know whether each scenario succeeded.
    # On scenario success we commit_pending_episodes(); on failure we
    # discard_pending_episodes() so junk corrections don't pollute the dataset.
    teleop_ctx.defer_episode_saves = True

    try:
        for scenario_idx in range(n_scenarios):
            t0 = time.time()
            observation, _ = env.reset()
            policy.reset()
            ctrl.reset_for_new_scenario()
            teleop_ctx.source_scenario_idx = scenario_idx
            success = False
            step = 0

            pbar = tqdm(
                total=max_steps_per_scenario,
                desc=f"scenario {scenario_idx + 1}/{n_scenarios}",
                leave=False,
                dynamic_ncols=True,
            )
            try:
                while step < max_steps_per_scenario:
                    obs_b = _process_observation(observation, env, env_preprocessor, preprocessor)

                    with (
                        torch.inference_mode(),
                        torch.autocast(device_type=amp_device_type) if use_amp else nullcontext(),
                    ):
                        action = policy.select_action(obs_b)

                    action = postprocessor(action)
                    action_t = env_postprocessor({ACTION: action})[ACTION]
                    action_numpy = action_t.to("cpu").numpy()

                    observation, _reward, terminated, truncated, info = env.step(action_numpy)
                    success = _extract_success(info)
                    in_collision = _extract_in_collision(info)

                    decision = ctrl.tick(success=success, in_collision=in_collision)
                    step += 1
                    pbar.update(1)
                    pbar.set_postfix(
                        rrt=policy._rrt.mode.value,
                        cycle=f"{ctrl.cycles_used}/{intervention_cfg.max_cycles_per_scenario}",
                        plan_fail=ctrl.plan_failures,
                        coll=int(in_collision),
                        backoff=f"{ctrl.backoff_rounds}/{intervention_cfg.max_backoff_rounds_per_scenario}",
                        cd=int(ctrl.in_backoff_cooldown),
                    )

                    if decision == "advance":
                        break
                    if bool(np.any(terminated) | np.any(truncated)):
                        break
            finally:
                pbar.close()

            elapsed = time.time() - t0
            result = ScenarioResult(
                scenario_idx=scenario_idx,
                success=success,
                cycles_used=ctrl.cycles_used,
                status=ctrl.last_status,
                plan_failures=ctrl.plan_failures,
            )
            results.append(result)
            # If the scenario ended while still inside an RRT/TELEOP
            # recording (e.g. env declared success mid-RRT-execution and
            # the controller broke out before the wrapper saw the
            # frame_source transition back to POLICY), the in-progress
            # frames are stranded in the wrapper's _frame_buffer. Force
            # them through _finish_episode so they land in pending_episodes
            # under the still-set source_scenario_idx, before we commit /
            # discard.
            try:
                env.call("flush_in_progress_episode")
            except Exception:
                logger.exception("flush_in_progress_episode failed; continuing")
            # Commit the scenario's buffered episodes only if the scenario
            # actually succeeded — otherwise the corrections didn't help and
            # are not useful DAgger data, so drop them.
            if result.success:
                n_committed = sum(
                    env.call("commit_pending_episodes")  # one per sub-env
                )
            else:
                n_committed = 0
                env.call("discard_pending_episodes")
            logger.info(
                "Scenario %d/%d finished: success=%s cycles=%d status=%s (%.1fs, %d episode(s) %s)",
                scenario_idx + 1,
                n_scenarios,
                result.success,
                result.cycles_used,
                result.status,
                elapsed,
                n_committed if result.success else ctrl.cycles_used,
                "committed" if result.success else "discarded",
            )
            if csv_writer is not None and csv_file is not None:
                csv_writer.writerow(
                    [
                        result.scenario_idx,
                        int(result.success),
                        result.cycles_used,
                        result.status,
                        result.plan_failures,
                    ]
                )
                csv_file.flush()
    finally:
        # Order matters here. Drop any stranded in-progress / pending data
        # FIRST while source_scenario_idx and defer_episode_saves are still
        # set — that way any save_episode calls along the discard path see
        # the same metadata schema as during the run, preventing the
        # _flush_metadata_buffer pa.Table.from_pydict failure caused by
        # mixing rows that have source_scenario_idx with rows that don't.
        try:
            env.call("flush_in_progress_episode")
        except Exception:
            logger.exception("flush_in_progress_episode failed during shutdown.")
        try:
            env.call("discard_pending_episodes")
        except Exception:
            logger.exception("Failed to discard pending episodes during shutdown.")
        teleop_ctx.source_scenario_idx = None
        teleop_ctx.defer_episode_saves = False
        if csv_file is not None:
            csv_file.close()

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@dataclass
class _InterventionDefaults:
    """Just defaults — the script reads cfg.eval / cfg.env / cfg.policy from
    the standard EvalPipelineConfig and combines them with these intervention
    knobs at runtime. Override via env vars or by editing this dataclass."""

    intervention: InterventionConfig = field(default_factory=InterventionConfig)


@parser.wrap()
def intervention_main(cfg: EvalPipelineConfig):
    """Same arg surface as ``lerobot-eval`` plus intervention defaults."""
    logging.info(pformat({"eval_cfg": cfg.__class__.__name__}))

    if cfg.policy is not None:
        get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    if cfg.eval.batch_size != 1:
        raise ValueError(f"Intervention recording requires --eval.batch_size=1 (got {cfg.eval.batch_size}).")

    logging.info("Making environment (single env, EVAL_BENCHMARK if set)...")
    envs = make_env(
        cfg.env,
        n_envs=cfg.eval.batch_size,
        use_async_envs=cfg.eval.use_async_envs,
        trust_remote_code=cfg.trust_remote_code,
    )
    # Flatten {task_group: {task_id: vec_env}} → single vec_env (one task expected).
    flat = [vec for group in envs.values() for vec in group.values()]
    if len(flat) != 1:
        raise ValueError(
            f"Intervention recording expects a single task (got {len(flat)} from groups={list(envs)})."
        )
    env = flat[0]

    logging.info("Making policy + processors...")
    policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env, rename_map=cfg.rename_map)
    policy.eval()

    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": cfg.rename_map},
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )

    sa_cfg = getattr(cfg.policy, "shared_autonomy_config", None)
    if sa_cfg is None or not sa_cfg.enabled:
        raise ValueError(
            "Intervention recording requires --policy.shared_autonomy_config.enabled=true. "
            "RRT-to-goal lives on the wrapper; without it there is no intervention path."
        )
    policy = _wrap_with_shared_autonomy(policy, cfg.policy)
    _reconnect_relative_absolute_steps(preprocessor, policy.postprocessor, policy=policy)

    if not isinstance(policy, SharedAutonomyPolicyWrapper):
        raise RuntimeError("Expected SharedAutonomyPolicyWrapper after wrapping.")

    # Disable auto-pause on RRT natural finish so the headless loop keeps running.
    policy.auto_pause_on_rrt_finish = False
    # Make sure we start un-paused regardless of start_paused (the controller
    # never expects to be blocked by the pause gate).
    policy._run_event.set()

    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)

    intervention_cfg = InterventionConfig()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "intervention_per_scenario.csv"

    logging.info("Intervention config: %s", pformat(intervention_cfg.__dict__))
    logging.info("Per-scenario CSV: %s", csv_path)

    with torch.no_grad():
        results = run_intervention_rollout(
            env=env,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            n_scenarios=cfg.eval.n_episodes,
            intervention_cfg=intervention_cfg,
            csv_path=csv_path,
        )

    n_success = sum(1 for r in results if r.success)
    logging.info(
        "Done. %d/%d scenarios succeeded. CSV: %s",
        n_success,
        len(results),
        csv_path,
    )

    try:
        env.close()
    except Exception:
        logging.exception("env.close() raised — ignoring during shutdown.")


def main():
    init_logging()
    register_third_party_plugins()
    intervention_main()


if __name__ == "__main__":
    main()
