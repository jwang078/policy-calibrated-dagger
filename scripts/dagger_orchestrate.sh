#!/usr/bin/env bash
set -euo pipefail
# dagger_orchestrate.sh
#
# Automated multi-round DAgger pipeline. Each round does:
#   1. Record interventions on benchmark scenarios with the current policy
#   2. Compute sidecar stats for the per-round intervention dataset
#   3. Run augment_ratios_sweep.sh with --ratios "0.0" (hardlink-alias no-op)
#   4. Merge cumulatively (base + dag1 + ... + dagN) into a new training dataset
#   5. Compute sidecar stats for the merged dataset
#   6. Train: finetune (rounds 1..N-1 by default) or train-from-scratch (round N)
#   7. Cleanup: rm -rf round (N-1)'s merged dataset to bound disk use
#
# Hard constraints:
#   * Requires a SHARED EXTERNAL SplatSim ZMQ server running at --env_external_port
#     BEFORE this script is invoked. GPU memory can't host multiple SplatSim
#     instances plus training simultaneously. Launch with e.g.:
#       cd ~/code/SplatSim && python scripts/launch_nodes.py \
#         --robot sim_ur_pybullet_small_engine_new_interactive \
#         --robot_port 6001 --robot_name robot_iphone_w_engine_new \
#         --eval_benchmark_repo_id JennyWWW/eval_splatsim_approach_lever_benchmark_1000
#     Keep that running for the whole orchestration.
#
# Usage:
#   bash my_scripts/dagger_orchestrate.sh --base_short=STR --num_rounds=N [OPTIONS]
#
# Required:
#   --base_short=STR              Base dataset short name. The full repo id is
#                                 derived as ${HF_USER}/${BASE_SHORT} and the
#                                 per-round artifacts as ${HF_USER}/${BASE_SHORT}_dag{N}*.
#                                 The orchestrator no longer prepends "splatsim_"
#                                 to derived names — that prefix used to eat 9
#                                 chars off the 56-char repo limit and made many-
#                                 round runs impossible. If your existing base
#                                 dataset is named "splatsim_X" on disk/hub,
#                                 either rename it or symlink "X" → "splatsim_X"
#                                 in ~/.cache/huggingface/lerobot/${HF_USER}/ so
#                                 the orchestrator can find it under the prefix-less
#                                 name.
#   --num_rounds=N                Number of DAgger rounds (>=1).
#   --no_strip_splatsim_prefix    Keep the "splatsim_" prefix on derived dag/merged
#                                 dataset names (matches the base dataset exactly,
#                                 useful if the rest of your tooling expects the
#                                 prefix). Default behavior strips it to save 9
#                                 chars under the 56-char HF repo limit. Pass
#                                 --strip_splatsim_prefix to force the default
#                                 explicitly.
#
# Starting point (optional):
#   --initial_policy_path=PATH    Skip round-0 base training; use this checkpoint
#                                 as the round-1 input policy.
#   (unset)                       Run train_sweep.sh on the base dataset first.
#
# Shared external SplatSim:
#   --env_external_port=N         (default 6001)
#   --env_external_host=STR       (default 127.0.0.1)
#
# Per-round training mode:
#   --intermediate_mode=MODE      "finetune" or "scratch" used for EVERY dag
#                                 round 1..N (default "finetune"). Naming kept
#                                 for back-compat; effectively "--mode".
#   --final_mode=MODE             "scratch" (default) or "finetune". Controls
#                                 an OPTIONAL extra training step that runs
#                                 AFTER all N dag rounds, using round N's
#                                 final merged dataset:
#                                   scratch  → do an extra from-scratch train.
#                                              The plot-friendly mode: all N
#                                              dag rounds are uniform finetune
#                                              steps (the round-over-round
#                                              curve is monotonic), and the
#                                              extra scratch step gives you
#                                              a deployable policy trained on
#                                              the same final data.
#                                   finetune → skip the extra step; the last
#                                              dag round's finetune is the
#                                              deployable policy.
#
# Intervention recording:
#   --intervention_n_episodes=N   Forwarded to lerobot-eval as --eval.n_episodes
#                                 (default 100 — first 100 benchmark scenarios in order).
#   --intervention_sample_from_first=Y
#                                 Randomly sample --intervention_n_episodes scenarios
#                                 from the first Y indices of the benchmark dataset.
#                                 Unset (default): use first N in order.
#                                 Example: --intervention_n_episodes=20 --intervention_sample_from_first=100
#                                 → pick 20 random scenarios from indices [0..99].
#   --intervention_sample_seed=S  RNG seed for reproducible random sampling
#                                 (default 0). Same seed = same subset across reruns,
#                                 which is what you want for DAgger optimizing the
#                                 same scenarios round over round.
#   --target_intervention_volume=N
#                                 (REQUIRED, positive integer.) Total
#                                 intervention-related content per round, in
#                                 units of "1× raw intervention". The merge
#                                 composes:
#                                     raw × (N - n_blends) + each_blend × 1
#                                 so every condition (0 / 1 / 2 / … blends)
#                                 ends up with N× raw_intervention worth of
#                                 intervention-related content per round —
#                                 apples-to-apples across blend counts. The
#                                 1-blend case puts the leftover budget on
#                                 raw (raw:blend = (N-1):1, so 67/33 at N=3).
#                                 Must satisfy N >= n_blends (N == n_blends
#                                 drops raw and merges blends only); startup
#                                 errors if N < n_blends.
#                                 Typical values 1-3. Each round's merge is
#                                 ~N× the size of the raw intervention; watch
#                                 disk usage.
#
#                                 REPLACES the prior --intervention_oversample;
#                                 commands using the old flag name will fail
#                                 at the parser with an unknown-flag error.
#   --intervention_max_episode_length=N
#                                 Forwarded to lerobot-eval as
#                                 --env.episode_length (default 5000). This is the
#                                 MAX step cap per episode, NOT the length of any
#                                 single human intervention within the episode. The
#                                 SplatSimEnv config default of 400 is too short for
#                                 RRT-to-goal recovery to complete on hard scenarios.
#   --eval_benchmark_repo_id=ID   (default JennyWWW/eval_splatsim_approach_lever_benchmark_1000)
#   --intervention_extra_args=STR Raw passthrough to lerobot-eval
#                                 (e.g. --policy.shared_autonomy_config.enabled=true).
#   --intervention_method=METHOD  "rrt" (default) or "oracle_goal".
#                                 "rrt" uses the SA wrapper's RRT-to-goal planner
#                                 (recorded as FrameSource.RRT).
#                                 "oracle_goal" uses OracleGoalGuidanceSource —
#                                 a straight-line joint-space interpolation
#                                 from q_start → oracle q_goal_bias, no planner.
#                                 Recorded as FrameSource.BLEND_INTERVENTION_100
#                                 (verbatim playback). Use for DAgger label
#                                 generation when you want a deterministic
#                                 correction signal rather than RRT's variability.
#                                 Forwarded as --intervention.method=METHOD.
#   --intervention_oracle_goal_chunk_steps=N
#                                 (default 80) Number of waypoints in the
#                                 q_start → q_goal_bias linear interpolation chunk.
#                                 Only used iff --intervention_method=oracle_goal.
#                                 The intervention controller's rrt_steps_min/max
#                                 still picks a target_steps within this chunk
#                                 (typically less than N → partial playback).
#                                 Forwarded as
#                                 --intervention.oracle_goal_chunk_steps=N.
#   --blends="0.9 0.8"            Space-separated forward_flow_ratios to produce
#                                 EXTRA blended-intervention datasets per round
#                                 (alongside the raw intervention dataset).
#                                 For each ratio R in (0, 1), the orchestrator
#                                 invokes augment_dataset_with_blending.py to
#                                 closed-loop-replay the round's intervention
#                                 episodes through SplatSim with the SA wrapper's
#                                 forward_flow_ratio set to R, producing a
#                                 dataset named <int_short>_blend<NNN> (NNN =
#                                 3-digit int(R*100), e.g. blend090 / blend080).
#                                 These blend datasets are then merged into the
#                                 round's training dataset alongside the raw
#                                 intervention. The mix is controlled by
#                                 --target_intervention_volume (raw × (N -
#                                 n_blends) + each blend × 1). Empty/unset →
#                                 disabled (only raw intervention in merge).
#                                 Ratio=0.0 is silently dropped (= raw intervention
#                                 = full guidance; no blending needed). Ratio=1.0
#                                 IS ALLOWED (= pure policy = full noise / no
#                                 guidance contribution per wrapper math); when
#                                 it's the ONLY ratio in --blends the orchestrator
#                                 enters PURE_POLICY_MODE: training skips merge /
#                                 weighted-sampling and finetunes on the base
#                                 dataset only via resume_training.sh, producing
#                                 a matched-compute base-only baseline lineage
#                                 with per-round _blend100 rollout datasets.
#                                 Brackets/commas tolerated: --blends='[0.9, 0.8]'
#                                 is equivalent.
#   --blend_extra_args=STR        Raw passthrough to augment_dataset_with_blending.py
#                                 for every per-ratio invocation. Useful to override
#                                 blend_strategy / guidance_repr / blend_mode away
#                                 from the defaults (denoise / absolute_pos /
#                                 once_per_chunk / drain_chunk), or for any future
#                                 blend-side filter flag.
#   --filter_blend_collisions     Replay each blend dataset through a headless
#                                 splatsim and drop / trim episodes that hit
#                                 obstacles. Step 2 produces `_nocoll` siblings
#                                 (<blend>_nocoll) which step 4 substitutes into
#                                 the merge in place of the raw blends. The raw
#                                 blends stay on disk (cross-rerun cacheable);
#                                 toggling the flag off makes the merge use
#                                 them again without re-running step 2. Default
#                                 false. See filter_blend_collisions.py for
#                                 trim semantics (drop frames before collision,
#                                 drop episode if remaining length < 60).
#   --filter_collision_extra_args=STR
#                                 Raw passthrough to filter_blend_collisions.py.
#                                 Examples: --pre_collision_margin=20, or
#                                 --min_episode_length=90.
#   --filter_collision_env_port=N
#                                 Port for the auxiliary headless splatsim used
#                                 by the filter step. Default: --env_external_port
#                                 + 100, so the two sims don't collide.
#   --finetune_extra_args=STR     Raw passthrough to resume_training.sh for the
#                                 per-round FINETUNE training step (step 6 in
#                                 --intermediate_mode=finetune). Multi-flag, word-
#                                 split. Examples:
#                                   --finetune_extra_args='--eval.n_episodes=30'
#                                   --finetune_extra_args='--eval.n_episodes=20 --policy.use_amp=true'
#                                 Note: does NOT apply to scratch trainings via
#                                 train_sweep.sh (round 0 base, per-round scratch,
#                                 post-loop final scratch). Empty → no extras.
#   --use_weighted_sampling       Switch the per-round training pipeline from
#                                 "merge then train on a fixed-share dataset" to
#                                 "train on the per-source weighted union via
#                                 lerobot-train's --dataset.repo_ids / sample_weights
#                                 / stats_paths". When set, steps 4 (merge) and 5
#                                 (merged-rel-stats) are SKIPPED — the per-source
#                                 datasets and their stats sidecars (already on
#                                 disk from steps 1b/2b) are passed to step 6
#                                 directly. The DataLoader's WeightedRandomSampler
#                                 then enforces an EXACT per-batch share of base
#                                 vs DAgger data, regardless of how the dataset
#                                 sizes diverge across rounds. Default: false
#                                 (legacy merge-mode behavior). Currently requires
#                                 --action_format=rel (abs uses each dataset's
#                                 meta/stats.json and isn't wired here yet).
#                                 Mode-purity is enforced at startup: a lineage
#                                 that already has merge-mode `_m` artifacts
#                                 cannot be resumed in weighted mode (and vice
#                                 versa). Use a fresh --run_tag to start a new
#                                 lineage in the desired mode.
#   --dagger_data_fraction=F      Share of every batch sampled from DAgger
#                                 sub-datasets (raw intervention + any blends),
#                                 equal-split across them. Base gets the
#                                 remaining (1 - F). F must be in (0.0, 1.0).
#                                 Only meaningful with --use_weighted_sampling.
#                                 Default: 0.3.
#
# Rerun-blends mode (reuse an existing lineage's intervention data, iterate on
# blend ratios without re-collecting):
#   --rerun_blends_from=TAG[:BLENDS_TAG]
#                                 Activates rerun mode. Reads as "rerun blends
#                                 on top of the <TAG> lineage". The orchestrator
#                                 SKIPS step 1's intervention recording, REUSES
#                                 source's per-round intervention datasets, and
#                                 trains a NEW lineage that branches off SOURCE's
#                                 _ft_dag(r-1) at each round. Examples:
#                                   --rerun_blends_from=d5jvm
#                                       → source = no-blends d5jvm lineage
#                                   --rerun_blends_from=d5jvm:b090_080
#                                       → source = d5jvm lineage's b090_080
#                                         in-lineage-blends sub-lineage
#                                 Requirements (validated at startup):
#                                   * --blends must be non-empty (rerun w/o
#                                     blends would just reproduce source's
#                                     training)
#                                   * --run_tag must differ from the source's
#                                     run_tag (avoid lineage collision)
#                                   * source intervention dirs exist on disk
#                                     for rounds 1..NUM_ROUNDS
#                                   * source _ft_dag(r-1) training dir exists
#                                     for rounds 2..NUM_ROUNDS
#                                 --num_rounds becomes optional in rerun mode:
#                                 auto-detected from source lineage on disk
#                                 (highest contiguous N with source intervention
#                                 dataset present in HF cache). Override with
#                                 --num_rounds=N to subset.
#                                 Assumes source and current command share
#                                 task/dataset_tag/model/method. If they differ,
#                                 use the explicit prefix flags below.
#                                 Intervention-recording flags
#                                 (--intervention_n_episodes/_sample_from_first/
#                                 _sample_seed/_extra_args) are IGNORED in rerun
#                                 mode (a warning is emitted if set).
#   --reuse_intervention_from=PFX Advanced override: source intervention dataset
#                                 short prefix (everything before _<a|r>_dag<N>).
#                                 Use when --rerun_blends_from's auto-derivation
#                                 is wrong (source/current differ in
#                                 task/dataset_tag/model/method).
#   --reuse_policy_from=BASENAME  Advanced override: source training-dir
#                                 basename (everything before _ft_dag<N>). Must
#                                 be set together with --reuse_intervention_from.
#
# Model / action format (controls naming, must match train_sweep.sh's choices):
#   --model=MODEL                 "pi" / "diff" / "act"  (default "pi")
#   --action_format=FMT           "abs" / "rel". When --initial_policy_path is
#                                 set and this flag is omitted, the orchestrator
#                                 auto-detects it from the policy's train_config.json
#                                 (use_relative_actions: true→rel, false→abs). Pass
#                                 it explicitly to override the auto-detection.
#                                 Default when no initial policy: "rel".
#
# Finetune overrides (forwarded to resume_training.sh):
#   --finetune_steps=N            (default 4000)
#   --finetune_eval_freq=N        (default 2000)
#   --finetune_save_freq=N        (default 2000)
#   --extend_last_round           When set, the orchestrator re-runs step 6 for
#                                 the highest fully-complete round R if its
#                                 originally-trained target_steps is less than
#                                 what the CURRENT --finetune_steps value implies.
#                                 Resumes training from R's existing checkpoint
#                                 (no work lost), runs until the new target, and
#                                 leaves earlier complete rounds untouched. Use
#                                 this when bumping --finetune_steps to extend
#                                 the tail of a lineage in place. Default off →
#                                 complete rounds stay skipped (legacy behavior).
#   --finetune_batch_size=N       (default empty = inherit from base train_config.json)
#                                 Lower this if you hit CUDA OOM — the shared external
#                                 SplatSim costs ~2.75 GiB the original from-scratch
#                                 training had to itself, so the finetune step is
#                                 tighter on memory than the round-0 training was.
#   --finetune_decay_lr=FLOAT     Override the cosine scheduler's floor LR
#                                 (--policy.scheduler_decay_lr). Default empty
#                                 = AUTO-SET to the policy's peak optimizer_lr
#                                 (read from train_config.json — 2.5e-5 for
#                                 pi05, 1e-5 for diffusion). This forces a
#                                 CONSTANT peak LR through the finetune, which
#                                 is what you want for DAgger resumes past the
#                                 cosine decay end (otherwise runtime LR is
#                                 parked at the original ~1e-6 floor — too
#                                 small to actually move the policy in
#                                 200-2000 steps even with large gradients).
#                                 Pass an explicit value to override the
#                                 auto-set (e.g. 5e-5 for an aggressive run).
#
# Resume/restart:
#   --start_round=N               Explicitly start at round N (default: auto-detect).
#   --resume                      Skip the resume-prompt and auto-confirm resuming
#                                 from the detected partial-progress point. When
#                                 nothing remains to do (all rounds + optional
#                                 final-scratch complete), exits 0 instead of
#                                 prompting for restart. Designed for batch/sweep
#                                 invocations that shouldn't block on stdin.
#                                 Mutually exclusive with --force_restart.
#   --force_restart               Skip resume-prompt; restart from round 1, deleting
#                                 any existing dag1..dag{num_rounds} artifacts. Asks
#                                 for confirmation token.
#   --also_delete_blends          Opt-in modifier for --force_restart in rerun mode.
#                                 By default rerun-mode cleanup PRESERVES blend
#                                 datasets (<src_int>_dagN_blendXXX) since they're
#                                 deterministic functions of (source, ratio) and
#                                 are cross-rerun-cacheable: other rerun lineages
#                                 using the same ratio at the same source round
#                                 reuse them. Set this flag when you want a true
#                                 clean slate (knows no other rerun shares the
#                                 blends, or wants them regenerated). Has no
#                                 effect outside rerun mode: non-rerun lineages
#                                 own their blends, so they're always deleted.
#   --retrain_round=N             Re-run ONLY round N's training step (step 6), writing
#                                 to a suffixed output dir so the original training dir
#                                 isn't overwritten. Use this to iterate on finetune
#                                 hyperparameters (LR, steps, oversample) without
#                                 redoing intervention recording or merging.
#                                 Auto-detects what data exists: if round N's merged
#                                 dataset is on disk, runs step 6 only; otherwise
#                                 re-runs steps 4-6 (re-merge from existing
#                                 intervention datasets, recompute stats, train).
#                                 Skips the post-loop final-scratch phase (the chain
#                                 stops at the retrained round). Mutually exclusive
#                                 with --start_round/--force_restart.
#   --retrain_suffix=NAME         Suffix appended to the retrained round's training
#                                 output dir + policy run name (default "v2"). Must
#                                 be alphanumeric/underscore/hyphen. Example: dag6
#                                 → "_v2" → outputs/training/..._ft_dag6_v2/
#
# Merge routing:
#   --skip_alias_step             Skip step 3 (the augment_ratios_sweep.sh ratio=0
#                                 hardlink-alias). Pass the per-round intervention
#                                 datasets directly into merge step 4 by their raw
#                                 names ({base}_dag{N}). Use this when your base
#                                 dataset name plus "_dag{N}_pirel00" suffix would
#                                 exceed the 56-char HuggingFace repo limit. Default
#                                 OFF — the alias step is on by default so future
#                                 ratio>0 augmentations slot into the same pipeline.
#
# Offline mode:
#   --push_to_hub                 When set, push artifacts to HuggingFace Hub:
#                                 (a) the per-round intervention dataset (after
#                                     step 1), and
#                                 (b) the trained policy checkpoint (after step 6).
#                                 Default OFF — DAgger generates 3+ datasets and
#                                 1+ multi-GB policy per round; keeping them
#                                 local-only avoids minutes-to-hours of unwanted
#                                 hub uploads per round.
#
# SplatSim lifecycle:
#   --manage_splatsim             (default ON) The orchestrator launches its own
#                                 SplatSim ZMQ server on --env_external_port at
#                                 startup and shuts it down before each training
#                                 step, then re-launches it after training. This
#                                 lets lerobot-train spawn its own in-process sim
#                                 for inline eval (same CUDA context as training
#                                 → memory poolable → no OOM), while keeping the
#                                 external sim available for intervention recording
#                                 and the next round's interventions. The sim is
#                                 also killed on orchestrator exit.
#   --no_manage_splatsim          Opt out: the user is responsible for launching
#                                 and managing SplatSim manually on
#                                 --env_external_port (legacy behavior). The
#                                 orchestrator will neither kill nor relaunch it,
#                                 and training will keep --env.external_port set.
#                                 Risk: external sim's 2.7 GiB CUDA context is
#                                 hard-partitioned during training → potential OOM.
#   --headless                    Aggregator: disable every GUI/visualizer the
#                                 orchestrator controls — for fast batch runs.
#                                 Forwards:
#                                   * --headless to SplatSim's launch_nodes.py
#                                     in start_sim() (pybullet via p.DIRECT)
#                                   * --policy.shared_autonomy_config.show_slider=false
#                                     to step 1's lerobot-eval (disables the
#                                     Tkinter ratio slider AND the SA wrapper's
#                                     per-policy pybullet GUI window)
#                                   * --env.headless=true and the same
#                                     show_slider=false to step 6's training
#                                     (in-process inline-eval sim → p.DIRECT)
#                                 In --no_manage_splatsim mode, the training-side
#                                 --env.headless=true is suppressed (the user
#                                 owns their sim; training connects via
#                                 external_port and doesn't spawn pybullet).
#                                 Default: false (interactive behavior).
#   --splatsim_root=PATH          Root of the SplatSim repo (default $HOME/code/SplatSim).
#                                 launch_nodes.py is run from this dir.
#   --splatsim_robot=NAME         --robot arg for launch_nodes.py
#                                 (default sim_ur_pybullet_small_engine_new_interactive).
#   --splatsim_robot_name=NAME    --robot_name arg for launch_nodes.py
#                                 (default robot_iphone_w_engine_new).
#
# Misc:
#   --dry-run                     Print every command instead of executing.
#
# Example:
#   bash my_scripts/dagger_orchestrate.sh \
#       --base_short=approach_lever_7_lowres_5path \
#       --num_rounds=3 \
#       --initial_policy_path=outputs/training/pi05_approach_lever_7_lowres_5path_delta_basewrist \
#       --env_external_port=6001 --intervention_n_episodes=100 \
#       --intermediate_mode=finetune --final_mode=scratch

# ── USER CONFIG (defaults) ────────────────────────────────────────────────────
HF_USER="JennyWWW"
BASE_SHORT=""
BASE_REPO_OVERRIDE=""
STRIP_SPLATSIM_PREFIX=true   # strip splatsim_ from the dag-name prefix; flip with --no_strip_splatsim_prefix
# Optional tag inserted into derived dag dataset names AND derived policy
# training-dir names so a new dagger run doesn't overwrite a previous run
# that used the same base. Empty (default) = original naming. Set to
# e.g. "d30" for a diverse-30-scenario run. The tag costs N+1 chars in the
# dataset name; ensure num_rounds + base_short + tag fits HF's 56-char repo-
# name limit (pre-flight will fail otherwise with a clear message).
RUN_TAG=""
# Optional override of the dag dataset-name root, post-derivation. Lets you
# use a SHORT name for dag artifacts even when BASE_REPO has a long name
# (e.g. BASE_REPO=JennyWWW/splatsim_approach_lever_11_biasend_5path_grip0
# → DAG_SHORT_OVERRIDE=lever_grip0 → dag artifacts become
# lever_grip0_r_dag1, lever_grip0_r_dag1_m, etc.). The actual BASE_REPO
# data source is untouched; only dag artifact names change.
DAG_SHORT_OVERRIDE=""
NUM_ROUNDS=""
INITIAL_POLICY_PATH=""
ENV_EXTERNAL_PORT="6001"
ENV_EXTERNAL_HOST="127.0.0.1"
INTERMEDIATE_MODE="finetune"
FINAL_MODE="scratch"
INTERVENTION_N_EPISODES="100"
# Eval episode count for ALL inline / final-scratch evals. Empty → default
# to `$INTERVENTION_N_EPISODES` (single-source-of-truth ergonomic).
# Auto-forwarded as:
#   - `--eval.n_episodes=$EVAL_N_EPISODES` appended to the finetune step's
#     lerobot-train invocation AFTER `$FINETUNE_EXTRA_ARGS` so it always wins
#     over anything the user passed in finetune_extra_args (draccus uses
#     the last occurrence).
#   - `--eval_n_episodes=$EVAL_N_EPISODES` + `--eval_benchmark_subset=...`
#     forwarded to train_sweep.sh's new passthroughs in the final-scratch
#     step (previously hardcoded at 5, leaving final-scratch evals
#     incomparable to per-round finetune evals).
# Set explicitly only if you want a DIFFERENT eval count than the
# intervention recording's count (rare; e.g. you record on 30 scenarios but
# only want to eval on the first 5 for a quick sanity check).
EVAL_N_EPISODES=""
TARGET_INTERVENTION_VOLUME=""   # required; validated after BLENDS is parsed
INTERVENTION_MAX_EPISODE_LENGTH="5000"
INTERVENTION_SAMPLE_FROM_FIRST=""   # empty → run first N in order; set → random subset
INTERVENTION_SAMPLE_SEED="0"
EVAL_BENCHMARK_REPO_ID="JennyWWW/eval_splatsim_approach_lever_benchmark_1000"
INTERVENTION_EXTRA_ARGS=""
# RRT-planner collision clearances, in meters. Empty → use SplatSim's
# defaults (_COLLISION_CLEARANCE = 0.01 m obstacle, self = 0.0 m).
# Forwarded to intervention recording via
# --policy.shared_autonomy_config.rrt_obstacle_clearance / .rrt_self_collision_clearance.
# Increases give the policy drift margin along the planned path — but tight
# scenarios may become unplannable. Start small (0.02-0.04 m / 0.01-0.02 m)
# and watch for jumps in plan_failures / rrt_steps_executed=0 in the per-
# scenario CSV.
RRT_OBSTACLE_CLEARANCE=""
RRT_SELF_COLLISION_CLEARANCE=""
# Non-adjacent robot-link pairs to EXCLUDE from self-collision checks.
# JSON-encoded list of pairs (e.g. '[[0,2]]' for the UR robot's
# base_link(0) vs upper_arm_link(2)). Empty → no skips. Required
# whenever --rrt_self_collision_clearance is non-zero on a robot whose
# URDF has structurally-close non-adjacent links — without it the
# planner rejects every IK solution. Forwarded to intervention recording
# via --policy.shared_autonomy_config.rrt_self_collision_skip_pairs.
RRT_SELF_COLLISION_SKIP_PAIRS=""
# Blended-intervention data sources. Space-separated list of forward_flow_ratio
# floats; for each ratio R in (0,1) the orchestrator creates an additional
# per-round dataset alongside the raw intervention via
# augment_dataset_with_blending.py — same policy that recorded the intervention
# drives a closed-loop replay at SA-wrapper ratio=R. Each blend is then
# included in the merge alongside the raw intervention (oversampled the same
# way). Empty → disabled (back-compat). Ratio=0.0 is dropped with an info log
# (= full guidance = raw intervention; nothing to record). Ratio=1.0 IS
# ALLOWED (= pure policy; records _blend100). When ratio=1.0 is the ONLY
# entry, the orchestrator enters PURE_POLICY_MODE and skips merge/weighted-
# sampling at training time — see the BLENDS-parsing block below.
BLENDS_STR=""
BLEND_EXTRA_ARGS=""
# Collision-free blend filtering. When true, each blend dataset produced in
# step 2 gets a sibling `_nocoll` variant created by replaying its episodes
# through a headless splatsim and dropping (or trimming) episodes that hit
# obstacles. Step 4's merge then uses the `_nocoll` siblings instead of the
# raw blends. The raw blends stay on disk so they're cross-rerun-cacheable
# and the user can flip the flag off without re-running step 2.
# Default false → byte-identical to today's behavior.
FILTER_BLEND_COLLISIONS=false
# Extra args forwarded to filter_blend_collisions.py (e.g. to override
# --pre_collision_margin, --min_episode_length). Word-split.
FILTER_COLLISION_EXTRA_ARGS=""
# Auxiliary headless splatsim port (used only by the filter step). Defaults
# to main port + 100 so the two sims can coexist.
FILTER_COLLISION_ENV_PORT=""
# Raw passthrough to resume_training.sh for the per-round FINETUNE training
# step. Lets callers override individual lerobot-train flags (e.g.
# --eval.n_episodes, --batch_size adjustments) without baking each one into
# the orchestrator. Word-split, so multi-flag strings work:
# --finetune_extra_args='--eval.n_episodes=30 --policy.use_amp=true'. Empty
# (default) → no extra args. Note: only applies to the finetune training
# path (step 6 in --intermediate_mode=finetune); scratch trainings via
# train_sweep.sh aren't covered.
FINETUNE_EXTRA_ARGS=""
# Multi-dataset weighted-sampling mode. When true, step 4 (merge) and step 5
# (merged-dataset rel-action stats) are SKIPPED — instead, step 6 trains
# directly against the union of {base, all per-round raw intervention, all
# per-round blends} via lerobot-train's --dataset.repo_ids / sample_weights
# / stats_paths multi-dataset path. Per-source normalization happens inside
# the DataLoader; the policy's normalize layer is a no-op (see
# src/lerobot/datasets/multi_source_normalizing_dataset.py + lerobot_train.py).
# Equal share among all DAgger sub-datasets, BASE gets (1 - DAGGER_DATA_FRACTION).
# A lineage runs entirely in one mode — switching mid-lineage is rejected at
# startup (see mode-purity validation below).
USE_WEIGHTED_SAMPLING=false
DAGGER_DATA_FRACTION="0.3"
# --norm_mode controls how MultiSourceNormalizingDataset exposes stats to the
# policy in weighted-sampling mode. See
# src/lerobot/datasets/multi_source_normalizing_dataset.py docstring + the
# DatasetConfig.norm_mode docstring for the full trade-off explanation:
#   * aggregated (default) — min-of-mins/max-of-maxes across base + every
#     round's intervention + every blend. Both train and eval use the same.
#     Base data gets compressed if interventions add extreme values.
#   * base_only — expose source[0] (base) stats only. Intervention data
#     may produce normalized targets OUTSIDE [-1, 1] when its raw range
#     exceeds base's. Was the de-facto behavior before the `--dataset.stats_path`
#     override was decoupled from aggregation; preserved as an explicit
#     opt-in for A/B testing.
#   * per_source — NOT IMPLEMENTED (wrapper raises NotImplementedError).
# In merge mode (USE_WEIGHTED_SAMPLING=false), norm_mode is silently ignored
# because there's only ONE dataset and stats come from the merged sidecar.
NORM_MODE="aggregated"
# Skip-already-succeeded mode. When true (default), the per-round intervention
# recording on round R reads the previous round's training-time eval_info.json
# and runs ONLY on the scenarios that failed there — the policy already passed
# the others, so recording on them is wasted work + dilutes the per-scenario
# DAgger signal. Falls back to the full subset when:
#   * round == 1 (no prior eval data)
#   * the helper can't find / parse eval_info_step_*.json
#   * the prior eval was 100% success (no failures to target)
# Set to false to record interventions on every scenario in
# --env.eval_benchmark_subset every round (the historical behavior).
DAGGER_SKIP_SUCCEEDED_IN_PREV_EVAL=true
# Rerun-blends mode. When --rerun_blends_from=<source_run_tag>[:<source_blends_tag>]
# is set, the orchestrator REUSES an existing lineage's intervention datasets
# (skipping step 1's lerobot-eval recording) and trains a NEW lineage that
# branches off SOURCE's _ft_dag(r-1) at each round. See header doc + AGENTS.md.
RERUN_BLENDS_FROM=""
# Advanced overrides — explicit source dataset/policy prefixes (everything
# before _<a|r>_dag<N> and _ft_dag<N> respectively). Use when source and
# current command differ in task/dataset_tag/model/method, or when the source
# lineage has a non-standard name. If set, take precedence over the
# auto-derivation from --rerun_blends_from.
REUSE_INTERVENTION_FROM=""
REUSE_POLICY_FROM=""
# Intervention method selector. "rrt" uses the SA wrapper's RRT-to-goal
# planner (existing behavior); "oracle_goal" uses a straight-line joint-space
# interpolation from q_start to the oracle's q_goal_bias. Recorded frames are
# committed to the dataset in both cases (tagged FrameSource.RRT for rrt,
# FrameSource.BLEND_INTERVENTION_100 for oracle_goal VERBATIM playback).
INTERVENTION_METHOD="rrt"
INTERVENTION_ORACLE_GOAL_CHUNK_STEPS="80"
MODEL="pi"
ACTION_FORMAT=""   # empty → auto-detect from --initial_policy_path; fall back to "rel"
FINETUNE_STEPS="4000"
FINETUNE_EVAL_FREQ="2000"
FINETUNE_SAVE_FREQ="2000"
FINETUNE_BATCH_SIZE=""
FINETUNE_DECAY_LR=""
START_ROUND=""    # empty → auto-detect
FORCE_RESTART=false
# --resume: auto-confirm the resume prompt (default Y). Useful for batch /
# sweep invocations that shouldn't block on stdin. When prior work is fully
# complete (nothing to resume from), the orchestrator exits 0 instead of
# prompting for restart. Mutually exclusive with --force_restart.
RESUME=false
# --cleanup_only: when combined with --force_restart, deletes all lineage
# artifacts and exits 0 without starting a fresh run. Designed to be called
# by my_scripts/dagger_cleanup_lineage.sh, which derives the orchestrator
# flags from a training-dir path. Has no effect without --force_restart.
CLEANUP_ONLY=false
# --also_delete_blends: opt-in flag for --force_restart. In rerun-blends
# mode, blend datasets are named after the SOURCE (e.g.
# `<src_int>_dagN_blendXXX`) and are deterministic functions of
# (source_intervention, source_branching_policy, ratio). Two reruns asking
# for the same ratio at the same source round share these files via the
# blend cache. By default we preserve them on --force_restart so that other
# rerun lineages on disk aren't forced to regenerate. Set this flag to also
# delete blend datasets — useful when the user knows no other rerun shares
# them, or wants a clean slate. Has no effect outside rerun mode (a
# non-rerun lineage owns its own blends, so they're always deleted).
ALSO_DELETE_BLENDS=false
# --preserve_round_1_intervention: when set, --force_restart's cleanup leaves
# round 1's raw intervention dataset + alias + int-stats sidecar in place
# (the expensive human-in-the-loop recording). Everything else — round 1's
# merged dataset, training dir, blends, plus rounds 2..N entirely — still
# gets deleted, so the lineage restarts from "round 1 step 4 (merge)" rather
# than "round 1 step 1 (record interventions)". No-op in rerun mode where
# intervention paths are source-owned and never touched.
PRESERVE_ROUND_1_INTERVENTION=false
# --extend_last_round: when set, the orchestrator re-runs step 6 for the
# HIGHEST fully-complete round R *iff* the user's current --finetune_steps
# would have produced a higher target step count for R than what R was
# originally trained to. Useful when bumping --finetune_steps to extend
# the last round's training in place, without losing the steps already
# done. Earlier complete rounds are NOT touched; the chain is only extended
# at the tail. Default false → existing behavior (complete rounds stay
# skipped). Safe to leave on across sweep iterations.
#
# Mechanics: at resume-detection time, after deciding which rounds are
# complete, we read R's saved cfg.steps from
# <train_dir_R>/checkpoints/last/pretrained_model/train_config.json, compute
# its starting step (= the carried-forward CURRENT_STEP at the top of round R,
# i.e. previous round's actual final or the initial policy's step for R=1),
# and form planned_target = starting_step + FINETUNE_STEPS. If planned > saved,
# we demote ROUND_COMPLETED_STEPS[R] from 6 → 5 so step 6 re-runs. lerobot-train
# --resume=true loads checkpoints/last from R's output_dir and continues from
# the existing checkpoint until reaching --steps=planned_target, so no work is
# lost.
EXTEND_LAST_ROUND=false
RETRAIN_ROUND=""    # empty → not in retrain mode; set → only re-train this round to a suffixed dir
RETRAIN_SUFFIX="v2"
SKIP_ALIAS_STEP=false
PUSH_TO_HUB=false   # offline by default — see header for rationale
MANAGE_SPLATSIM=true
# --headless: aggregator flag that disables every GUI/visualizer the
# orchestrator controls — for fast batch runs where no human is watching.
# Specifically:
#   * `start_sim()` appends `--headless` to SplatSim's launch_nodes.py call
#     (pybullet connects via p.DIRECT instead of p.GUI; no rendering window).
#   * Step 1's lerobot-eval invocation appends
#     `--policy.shared_autonomy_config.show_slider=false`, which gates BOTH
#     the Tkinter ratio slider AND the SA wrapper's per-policy pybullet GUI
#     client (one field, two surfaces — see shared_autonomy_wrapper.py:240).
#   * Step 6 training (both scratch via train_sweep.sh and finetune via
#     resume_training.sh) appends `--env.headless=true` so lerobot-train's
#     in-process inline-eval sim connects via p.DIRECT, plus
#     `--policy.shared_autonomy_config.show_slider=false` in case the policy
#     has SA wrapper config (defensive — most training configs don't enable
#     SA but a few that resume from intervention-recording dirs do).
# In `--no_manage_splatsim` mode the user is responsible for the external
# sim's GUI mode; the training-side `--env.headless` injection is suppressed
# (the user is connecting to their own sim via external_port and the env
# never spawns an in-process pybullet client). Step 1's `show_slider=false`
# is still injected — it controls a wrapper-side surface, not the sim.
# Default false → byte-identical to today's interactive behavior.
HEADLESS=false
SPLATSIM_ROOT="$HOME/code/SplatSim"
SPLATSIM_ROBOT="sim_ur_pybullet_small_engine_new_interactive"
SPLATSIM_ROBOT_NAME="robot_iphone_w_engine_new"
DRY_RUN=false
# ─────────────────────────────────────────────────────────────────────────────

# Path constants.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LEROBOT_CACHE="$HOME/.cache/huggingface/lerobot"
STATS_BASE="$LEROBOT_ROOT/outputs/dataset_stats"

# Thin shim around the canonical naming module so forward-direction naming
# (config → name) lives in ONE place — `my_scripts/dagger_naming.py` — and
# can't drift between the bash orchestrator and the Python viz scripts. See
# the module docstring for available subcommands.
_py_dagger_name() {
    python3 "$SCRIPT_DIR/dagger_naming.py" "$@"
}

# Computed at the end of arg parsing (see "intervention scenario subset" block):
# JSON list of episode indices to pass as --env.eval_benchmark_subset, or empty
# when using default first-N-in-order behavior.
INTERVENTION_SUBSET_JSON=""

# ── parse args ────────────────────────────────────────────────────────────────
# Capture original argv before parsing so the dagger config sidecar can log
# the exact invocation that produced each per-round training dir.
ORIG_ARGV=( "$@" )
for arg in "$@"; do
    case "$arg" in
        --base_short=*)               BASE_SHORT="${arg#*=}" ;;
        --base_repo_id=*)             BASE_REPO_OVERRIDE="${arg#*=}" ;;
        --no_strip_splatsim_prefix)   STRIP_SPLATSIM_PREFIX=false ;;
        --strip_splatsim_prefix)      STRIP_SPLATSIM_PREFIX=true ;;
        --run_tag=*)                  RUN_TAG="${arg#*=}" ;;
        --dag_short_override=*)       DAG_SHORT_OVERRIDE="${arg#*=}" ;;
        --num_rounds=*)               NUM_ROUNDS="${arg#*=}" ;;
        --initial_policy_path=*)      INITIAL_POLICY_PATH="${arg#*=}" ;;
        --env_external_port=*)        ENV_EXTERNAL_PORT="${arg#*=}" ;;
        --env_external_host=*)        ENV_EXTERNAL_HOST="${arg#*=}" ;;
        --intermediate_mode=*)        INTERMEDIATE_MODE="${arg#*=}" ;;
        --final_mode=*)               FINAL_MODE="${arg#*=}" ;;
        --intervention_n_episodes=*)         INTERVENTION_N_EPISODES="${arg#*=}" ;;
        --eval_n_episodes=*)                 EVAL_N_EPISODES="${arg#*=}" ;;
        --target_intervention_volume=*)      TARGET_INTERVENTION_VOLUME="${arg#*=}" ;;
        --intervention_max_episode_length=*) INTERVENTION_MAX_EPISODE_LENGTH="${arg#*=}" ;;
        --intervention_sample_from_first=*)  INTERVENTION_SAMPLE_FROM_FIRST="${arg#*=}" ;;
        --intervention_sample_seed=*)        INTERVENTION_SAMPLE_SEED="${arg#*=}" ;;
        --eval_benchmark_repo_id=*)          EVAL_BENCHMARK_REPO_ID="${arg#*=}" ;;
        --intervention_extra_args=*)         INTERVENTION_EXTRA_ARGS="${arg#*=}" ;;
        --rrt_obstacle_clearance=*)          RRT_OBSTACLE_CLEARANCE="${arg#*=}" ;;
        --rrt_self_collision_clearance=*)    RRT_SELF_COLLISION_CLEARANCE="${arg#*=}" ;;
        --rrt_self_collision_skip_pairs=*)   RRT_SELF_COLLISION_SKIP_PAIRS="${arg#*=}" ;;
        --blends=*)                          BLENDS_STR="${arg#*=}" ;;
        --blend_extra_args=*)                BLEND_EXTRA_ARGS="${arg#*=}" ;;
        --filter_blend_collisions)           FILTER_BLEND_COLLISIONS=true ;;
        --filter_collision_extra_args=*)     FILTER_COLLISION_EXTRA_ARGS="${arg#*=}" ;;
        --filter_collision_env_port=*)       FILTER_COLLISION_ENV_PORT="${arg#*=}" ;;
        --finetune_extra_args=*)             FINETUNE_EXTRA_ARGS="${arg#*=}" ;;
        --use_weighted_sampling)             USE_WEIGHTED_SAMPLING=true ;;
        --dagger_data_fraction=*)            DAGGER_DATA_FRACTION="${arg#*=}" ;;
        --norm_mode=*)                       NORM_MODE="${arg#*=}" ;;
        --dagger_skip_succeeded_in_prev_eval=*) DAGGER_SKIP_SUCCEEDED_IN_PREV_EVAL="${arg#*=}" ;;
        --rerun_blends_from=*)               RERUN_BLENDS_FROM="${arg#*=}" ;;
        --reuse_intervention_from=*)         REUSE_INTERVENTION_FROM="${arg#*=}" ;;
        --reuse_policy_from=*)               REUSE_POLICY_FROM="${arg#*=}" ;;
        --intervention_method=*)             INTERVENTION_METHOD="${arg#*=}" ;;
        --intervention_oracle_goal_chunk_steps=*) INTERVENTION_ORACLE_GOAL_CHUNK_STEPS="${arg#*=}" ;;
        --model=*)                    MODEL="${arg#*=}" ;;
        --action_format=*)            ACTION_FORMAT="${arg#*=}" ;;
        --finetune_steps=*)           FINETUNE_STEPS="${arg#*=}" ;;
        --finetune_eval_freq=*)       FINETUNE_EVAL_FREQ="${arg#*=}" ;;
        --finetune_save_freq=*)       FINETUNE_SAVE_FREQ="${arg#*=}" ;;
        --finetune_batch_size=*)      FINETUNE_BATCH_SIZE="${arg#*=}" ;;
        --finetune_decay_lr=*)        FINETUNE_DECAY_LR="${arg#*=}" ;;
        --start_round=*)              START_ROUND="${arg#*=}" ;;
        --force_restart)              FORCE_RESTART=true ;;
        --resume)                     RESUME=true ;;
        --cleanup_only)               CLEANUP_ONLY=true ;;
        --also_delete_blends)         ALSO_DELETE_BLENDS=true ;;
        --preserve_round_1_intervention) PRESERVE_ROUND_1_INTERVENTION=true ;;
        --extend_last_round)          EXTEND_LAST_ROUND=true ;;
        --retrain_round=*)            RETRAIN_ROUND="${arg#*=}" ;;
        --retrain_suffix=*)           RETRAIN_SUFFIX="${arg#*=}" ;;
        --skip_alias_step)            SKIP_ALIAS_STEP=true ;;
        --push_to_hub)                PUSH_TO_HUB=true ;;
        --manage_splatsim)            MANAGE_SPLATSIM=true ;;
        --no_manage_splatsim)         MANAGE_SPLATSIM=false ;;
        --headless)                   HEADLESS=true ;;
        --splatsim_root=*)            SPLATSIM_ROOT="${arg#*=}" ;;
        --splatsim_robot=*)           SPLATSIM_ROBOT="${arg#*=}" ;;
        --splatsim_robot_name=*)      SPLATSIM_ROBOT_NAME="${arg#*=}" ;;
        --dry-run)                    DRY_RUN=true ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

if [[ -z "$BASE_SHORT" ]]; then
    echo "ERROR: --base_short is required" >&2; exit 1
fi

# Parse --rerun_blends_from=<run_tag>[:<blends_tag>] into its two components.
# Either piece may be empty; the source's blends_tag is empty when reusing a
# no-blends source lineage. RERUN_MODE_ENABLED is the single boolean every
# downstream branch checks to decide "is this a rerun command?"
SOURCE_RUN_TAG=""
SOURCE_BLENDS_TAG=""
if [[ -n "$RERUN_BLENDS_FROM" ]]; then
    if [[ "$RERUN_BLENDS_FROM" == *:* ]]; then
        SOURCE_RUN_TAG="${RERUN_BLENDS_FROM%%:*}"
        SOURCE_BLENDS_TAG="${RERUN_BLENDS_FROM#*:}"
    else
        SOURCE_RUN_TAG="$RERUN_BLENDS_FROM"
        SOURCE_BLENDS_TAG=""
    fi
    if [[ -z "$SOURCE_RUN_TAG" ]]; then
        echo "ERROR: --rerun_blends_from='$RERUN_BLENDS_FROM' must include a non-empty source run_tag" >&2; exit 1
    fi
fi
RERUN_MODE_ENABLED=false
if [[ -n "$RERUN_BLENDS_FROM" || -n "$REUSE_INTERVENTION_FROM" || -n "$REUSE_POLICY_FROM" ]]; then
    RERUN_MODE_ENABLED=true
fi

# In rerun mode --num_rounds is optional (auto-detected from source lineage on
# disk). Outside rerun mode it's still required. Either way: if present, must
# be a positive integer.
if [[ -n "$NUM_ROUNDS" ]]; then
    if ! [[ "$NUM_ROUNDS" =~ ^[0-9]+$ ]] || (( NUM_ROUNDS < 1 )); then
        echo "ERROR: --num_rounds=$NUM_ROUNDS must be a positive integer" >&2; exit 1
    fi
elif [[ "$RERUN_MODE_ENABLED" != "true" ]]; then
    echo "ERROR: --num_rounds=N (positive integer) is required" >&2; exit 1
fi
# Validate intervention sampling: when --intervention_sample_from_first=Y is set,
# Y must be a positive integer and >= --intervention_n_episodes (can't sample
# 100 items from a pool of 50).
if [[ -n "$INTERVENTION_SAMPLE_FROM_FIRST" ]]; then
    if ! [[ "$INTERVENTION_SAMPLE_FROM_FIRST" =~ ^[0-9]+$ ]] || (( INTERVENTION_SAMPLE_FROM_FIRST < 1 )); then
        echo "ERROR: --intervention_sample_from_first=$INTERVENTION_SAMPLE_FROM_FIRST must be a positive integer" >&2; exit 1
    fi
    if (( INTERVENTION_SAMPLE_FROM_FIRST < INTERVENTION_N_EPISODES )); then
        echo "ERROR: --intervention_sample_from_first ($INTERVENTION_SAMPLE_FROM_FIRST) must be >=" \
             "--intervention_n_episodes ($INTERVENTION_N_EPISODES) — can't sample more than the pool size" >&2; exit 1
    fi
fi

# Resolve EVAL_N_EPISODES default. Empty (user didn't pass --eval_n_episodes)
# → default to the intervention count, so all three eval contexts
# (intervention recording, per-round finetune inline eval, final-scratch
# inline eval) use the same scope by default.
[[ -z "$EVAL_N_EPISODES" ]] && EVAL_N_EPISODES="$INTERVENTION_N_EPISODES"
if ! [[ "$EVAL_N_EPISODES" =~ ^[0-9]+$ ]] || (( EVAL_N_EPISODES < 1 )); then
    echo "ERROR: --eval_n_episodes='$EVAL_N_EPISODES' must be a positive integer" >&2; exit 1
fi

# Retrain-mode validation. --retrain_round=N is mutually exclusive with
# --start_round/--force_restart since it pins the start at round N step 6 and
# uses a different code path that skips the resume-prompt entirely.
if [[ -n "$RETRAIN_ROUND" ]]; then
    if ! [[ "$RETRAIN_ROUND" =~ ^[0-9]+$ ]] || (( RETRAIN_ROUND < 1 )); then
        echo "ERROR: --retrain_round=$RETRAIN_ROUND must be a positive integer" >&2; exit 1
    fi
    # Upper bound depends on NUM_ROUNDS, which in rerun mode is auto-detected
    # later. Defer the upper-bound check to the rerun validation block.
    if [[ -n "$NUM_ROUNDS" ]] && (( RETRAIN_ROUND > NUM_ROUNDS )); then
        echo "ERROR: --retrain_round=$RETRAIN_ROUND must be between 1 and $NUM_ROUNDS" >&2; exit 1
    fi
    if [[ -z "$RETRAIN_SUFFIX" ]]; then
        echo "ERROR: --retrain_round requires --retrain_suffix=NAME (e.g. v2); cannot be empty" >&2; exit 1
    fi
    if ! [[ "$RETRAIN_SUFFIX" =~ ^[a-zA-Z0-9_-]+$ ]]; then
        echo "ERROR: --retrain_suffix='$RETRAIN_SUFFIX' must be alphanumeric/underscore/hyphen only" >&2; exit 1
    fi
    if [[ -n "$START_ROUND" || "$FORCE_RESTART" == true ]]; then
        echo "ERROR: --retrain_round is mutually exclusive with --start_round / --force_restart" >&2; exit 1
    fi
fi

# --resume is the inverse of --force_restart; setting both is a user error.
if [[ "$RESUME" == true && "$FORCE_RESTART" == true ]]; then
    echo "ERROR: --resume and --force_restart are mutually exclusive." >&2; exit 1
fi

case "$INTERMEDIATE_MODE" in finetune|scratch) ;; *) echo "ERROR: --intermediate_mode must be 'finetune' or 'scratch'" >&2; exit 1;; esac
case "$FINAL_MODE"        in finetune|scratch) ;; *) echo "ERROR: --final_mode must be 'finetune' or 'scratch'" >&2; exit 1;; esac
case "$MODEL"             in pi|diff|act)      ;; *) echo "ERROR: --model must be one of pi/diff/act" >&2; exit 1;; esac
case "$INTERVENTION_METHOD" in rrt|oracle_goal) ;; *) echo "ERROR: --intervention_method must be 'rrt' or 'oracle_goal'" >&2; exit 1;; esac

# --rrt_*_clearance: must be a non-negative float. Empty (unset) = use
# SplatSim defaults; allowed. Anything above ~0.1 m is suspicious (RRT
# may fail to plan most paths) — warn but don't reject. Values that fail
# the float parse are rejected hard.
_validate_clearance() {
    local _name="$1" _val="$2"
    [[ -z "$_val" ]] && return 0
    local _verdict
    _verdict=$(python3 -c "
import sys
try:
    v = float(sys.argv[1])
except ValueError:
    print('ERR_NOT_FLOAT'); sys.exit()
if v < 0:
    print('ERR_NEGATIVE')
elif v > 0.1:
    print('WARN_LARGE')
else:
    print('OK')
" "$_val" 2>/dev/null || echo "ERR_NOT_FLOAT")
    case "$_verdict" in
        OK) ;;
        WARN_LARGE)
            echo "WARNING: --$_name=$_val is > 0.1 m. RRT may fail to find paths in tight scenarios." >&2
            ;;
        ERR_NEGATIVE)
            echo "ERROR: --$_name=$_val must be non-negative." >&2; exit 1 ;;
        *)
            echo "ERROR: --$_name='$_val' is not a valid float." >&2; exit 1 ;;
    esac
}
_validate_clearance rrt_obstacle_clearance         "$RRT_OBSTACLE_CLEARANCE"
_validate_clearance rrt_self_collision_clearance   "$RRT_SELF_COLLISION_CLEARANCE"

# --rrt_self_collision_skip_pairs: must be a JSON list of 2-element int
# lists (e.g. '[[0,2]]') or empty. Validated via python3 — bash JSON
# parsing is the wrong tool. Friendly hint emitted when the user sets a
# non-zero self-collision clearance WITHOUT skip pairs on the
# small-engine env (the UR URDF needs [[0,2]] to plan at all).
if [[ -n "$RRT_SELF_COLLISION_SKIP_PAIRS" ]]; then
    _skip_verdict=$(python3 -c "
import sys, json
try:
    v = json.loads(sys.argv[1])
except Exception:
    print('ERR_NOT_JSON'); sys.exit()
if not isinstance(v, list):
    print('ERR_NOT_LIST'); sys.exit()
for pair in v:
    if not isinstance(pair, list) or len(pair) != 2:
        print('ERR_BAD_PAIR'); sys.exit()
    if not all(isinstance(x, int) for x in pair):
        print('ERR_NOT_INT'); sys.exit()
print('OK')
" "$RRT_SELF_COLLISION_SKIP_PAIRS" 2>/dev/null || echo "ERR_NOT_JSON")
    if [[ "$_skip_verdict" != "OK" ]]; then
        echo "ERROR: --rrt_self_collision_skip_pairs='$RRT_SELF_COLLISION_SKIP_PAIRS' must be a JSON list of 2-element int lists, e.g. '[[0,2]]'." >&2
        exit 1
    fi
fi
# Soft warning: self-collision clearance > 0 with NO skip pairs is almost
# certainly going to make every RRT IK fail on the UR URDF (base_link(0)
# vs upper_arm_link(2) is naturally ~4 mm apart). Don't reject — other
# robots might not have this issue — just nudge.
if [[ -n "$RRT_SELF_COLLISION_CLEARANCE" && "$RRT_SELF_COLLISION_CLEARANCE" != "0" && "$RRT_SELF_COLLISION_CLEARANCE" != "0.0" ]] \
   && [[ -z "$RRT_SELF_COLLISION_SKIP_PAIRS" ]]; then
    echo "WARNING: --rrt_self_collision_clearance=$RRT_SELF_COLLISION_CLEARANCE is set but --rrt_self_collision_skip_pairs is empty." >&2
    echo "  On the UR URDF (small-engine scene) this will reject every IK solution due to" >&2
    echo "  base_link(0) vs upper_arm_link(2) being ~4 mm apart at all valid joint configs." >&2
    echo "  Recommended: --rrt_self_collision_skip_pairs='[[0,2]]'" >&2
fi

# Parse --blends="0.9 0.8" into BLENDS bash array. Validates each entry is in
# [0, 1] inclusive but with endpoint semantics fixed:
#
#   ratio = 0.0 → SILENTLY DROPPED. Per wrapper math (x_tsw = ratio*noise +
#                 (1-ratio)*guidance), ratio=0.0 means full guidance = raw
#                 intervention output. No blending happens, no useful dataset
#                 produced beyond the existing raw intervention.
#   ratio = 1.0 → ALLOWED. Means full noise / no guidance contribution = pure
#                 policy. Produces a `_blend100` dataset per round. When the
#                 ONLY ratio is 1.0, the orchestrator enters PURE_POLICY_MODE
#                 (see below): training skips all DAgger data and just resumes
#                 finetuning on the base dataset, giving a matched-compute
#                 base-only baseline for comparison against DAgger lineages.
#   0 < ratio < 1 → OK, normal blend.
BLENDS=()
if [[ -n "$BLENDS_STR" ]]; then
    # Allow optional brackets / commas for ergonomics: "[0.9, 0.8]" → "0.9 0.8".
    _blends_clean=$(echo "$BLENDS_STR" | tr ',[]' '   ')
    # shellcheck disable=SC2206  # word-split is intentional here
    _blends_raw=( $_blends_clean )
    for r in "${_blends_raw[@]}"; do
        # Validate numeric in [0,1]. python3 used so we get float comparison.
        _verdict=$(python3 -c "
import sys
try:
    v = float(sys.argv[1])
except ValueError:
    print('ERR_NOT_FLOAT'); sys.exit()
if v == 0.0:
    print('DROP_ZERO')
elif v == 1.0:
    print('OK_ONE')
elif 0.0 < v < 1.0:
    print('OK')
else:
    print('ERR_OUT_OF_RANGE')
" "$r" 2>/dev/null || echo "ERR_NOT_FLOAT")
        case "$_verdict" in
            OK|OK_ONE)
                BLENDS+=( "$r" )
                ;;
            DROP_ZERO)
                echo "[blends] dropping ratio=0.0 (= raw intervention dataset; no blending needed)."
                ;;
            ERR_OUT_OF_RANGE|ERR_NOT_FLOAT)
                echo "ERROR: --blends entry '$r' is not a valid forward_flow_ratio. Each entry must be a float in [0.0, 1.0]; 0.0 is silently dropped (= raw intervention)." >&2
                exit 1
                ;;
        esac
    done
fi

# Detect PURE_POLICY_MODE: when the only blend is ratio=1.0, this iteration is
# a matched-compute base-only baseline. Training will skip DAgger data
# (intervention/blend merge + weighted sampling) and just resume_training.sh
# on the base dataset; per-round closed-loop rollout still runs at ratio=1.0
# to produce the `_blend100` rollout datasets the PCA viz auto-discovers.
#
# Why this trigger: the user expressed pure-policy via --combination_pool=1.0
# in the sweep wrapper; that surfaces here as --blends=1.0. Mixed iterations
# like --blends="0.5 1.0" are NOT pure-policy (they're a normal blended
# lineage that happens to also record _blend100 from the DAgger-trained
# policy). See dagger_orchestrate_sweep.sh callers + the design doc.
PURE_POLICY_MODE=false
# Trigger on any string form that floats to 1.0 ("1", "1.0", "1.00", " 1 ").
# The sweep wrapper formats ratios via f"{r:g}" so 1.0 arrives as the bare
# string "1" — comparing literally against "1.0" would silently miss.
if [[ "${#BLENDS[@]}" -eq 1 ]] && python3 -c "import sys; sys.exit(0 if abs(float(sys.argv[1]) - 1.0) < 1e-9 else 1)" "${BLENDS[0]}" 2>/dev/null; then
    PURE_POLICY_MODE=true
    echo "[pure_policy] --blends=${BLENDS[0]} (= 1.0) detected → PURE_POLICY_MODE: per-round training will use BASE DATASET ONLY (skipping merge / weighted-sampling sub-datasets); per-round _blend100 rollouts use the matched-compute pure-policy checkpoint."
fi

# Validate --target_intervention_volume now that BLENDS is built. Must be a
# positive integer N with N >= n_blends. raw_count = N - n_blends; N == n_blends
# is allowed and means the merge contains blends only (no raw intervention).
#
# In weighted-sampling mode --target_intervention_volume is irrelevant (no
# merge step happens) — silently default it to a sentinel that satisfies the
# downstream invariants (must be >= n_blends) but is never used. Required ONLY
# in merge mode.
if [[ "$USE_WEIGHTED_SAMPLING" != "true" ]]; then
    if [[ -z "$TARGET_INTERVENTION_VOLUME" ]]; then
        echo "ERROR: --target_intervention_volume=N is required (positive integer)." >&2
        echo "  N = total intervention-related content per round, in units of '1× raw intervention'." >&2
        echo "  Each round's merge = raw × (N - n_blends) + each_blend × 1." >&2
        echo "  Typical values: 1-3 (must be >= number of blend ratios; == drops raw)." >&2
        echo "  REPLACES the prior --intervention_oversample flag." >&2
        echo "  (Not required when --use_weighted_sampling is set — weighted mode skips the merge entirely.)" >&2
        exit 1
    fi
    if ! [[ "$TARGET_INTERVENTION_VOLUME" =~ ^[0-9]+$ ]] || (( TARGET_INTERVENTION_VOLUME < 1 )); then
        echo "ERROR: --target_intervention_volume=$TARGET_INTERVENTION_VOLUME must be a positive integer." >&2
        exit 1
    fi
    if (( TARGET_INTERVENTION_VOLUME < ${#BLENDS[@]} )); then
        echo "ERROR: --target_intervention_volume=$TARGET_INTERVENTION_VOLUME must be >= number of blends (${#BLENDS[@]})." >&2
        echo "  raw_count = N - n_blends would be negative." >&2
        echo "  Increase --target_intervention_volume or remove blends." >&2
        exit 1
    fi
else
    # Default sentinel for sidecar logging / format strings. Never consumed by
    # the merge path (which is gated off in weighted mode).
    [[ -z "$TARGET_INTERVENTION_VOLUME" ]] && TARGET_INTERVENTION_VOLUME=1
fi

# Validate --dagger_data_fraction (only meaningful in weighted mode). Must be
# a float in (0, 1) — at exactly 0 the DAgger data is invisible (defeats the
# point), at exactly 1 the base is invisible (defeats DAgger). Validate via
# python3 so floats compare correctly.
if [[ "$USE_WEIGHTED_SAMPLING" == "true" ]]; then
    _frac_verdict=$(python3 -c "
import sys
try:
    v = float(sys.argv[1])
except ValueError:
    print('ERR_NOT_FLOAT'); sys.exit()
if 0.0 < v < 1.0:
    print('OK')
else:
    print('ERR_OUT_OF_RANGE')
" "$DAGGER_DATA_FRACTION" 2>/dev/null || echo "ERR_NOT_FLOAT")
    if [[ "$_frac_verdict" != "OK" ]]; then
        echo "ERROR: --dagger_data_fraction='$DAGGER_DATA_FRACTION' must be a float in (0.0, 1.0)." >&2
        echo "  This is the share of every batch that comes from DAgger sub-datasets" >&2
        echo "  (raw intervention + any blends). Base gets the remaining (1 - fraction)." >&2
        exit 1
    fi
    # --norm_mode validation. per_source is rejected here (and again
    # inside the wrapper) — the option is reserved as a placeholder so
    # callers + sidecars know it exists, but the wrapper currently raises
    # NotImplementedError for it. See multi_source_normalizing_dataset.py
    # docstring for why.
    case "$NORM_MODE" in
        aggregated|base_only) ;;
        per_source)
            echo "ERROR: --norm_mode=per_source is reserved but not implemented." >&2
            echo "  See src/lerobot/datasets/multi_source_normalizing_dataset.py docstring." >&2
            echo "  Use 'aggregated' (default; min-of-mins/max-of-maxes across all sources) or" >&2
            echo "  'base_only' (source[0] stats only; intervention rows may exceed [-1,1])." >&2
            exit 1
            ;;
        *)
            echo "ERROR: --norm_mode='$NORM_MODE' is not recognized." >&2
            echo "  Valid: aggregated, base_only." >&2
            exit 1
            ;;
    esac
fi

# Auto-detect ACTION_FORMAT from --initial_policy_path's train_config.json when
# the user didn't pass --action_format explicitly. Reading use_relative_actions
# off the actual policy avoids the subtle stats-mismatch class of bugs (rel
# stats sidecar applied to an absolute-action policy → silent corruption).
if [[ -z "$ACTION_FORMAT" && -n "$INITIAL_POLICY_PATH" ]]; then
    DETECT_CFG=""
    for candidate in \
        "$INITIAL_POLICY_PATH/train_config.json" \
        "$INITIAL_POLICY_PATH/pretrained_model/train_config.json" \
        "$INITIAL_POLICY_PATH/checkpoints/last/pretrained_model/train_config.json"
    do
        if [[ -f "$candidate" ]]; then
            DETECT_CFG="$candidate"; break
        fi
    done
    if [[ -n "$DETECT_CFG" ]]; then
        DETECTED_REL="$(python3 -c "
import json,sys
c = json.load(open(sys.argv[1]))
print('true' if c.get('policy',{}).get('use_relative_actions') else 'false')
" "$DETECT_CFG")"
        if [[ "$DETECTED_REL" == "true" ]]; then
            ACTION_FORMAT="rel"
        else
            ACTION_FORMAT="abs"
        fi
        echo "Auto-detected --action_format=$ACTION_FORMAT from $DETECT_CFG (use_relative_actions=$DETECTED_REL)."
    else
        echo "WARNING: --action_format not specified and could not find train_config.json under $INITIAL_POLICY_PATH" >&2
        echo "  Falling back to --action_format=rel." >&2
        ACTION_FORMAT="rel"
    fi
fi
# Fallback when no initial policy was provided (from-scratch training).
[[ -z "$ACTION_FORMAT" ]] && ACTION_FORMAT="rel"

case "$ACTION_FORMAT"     in abs|rel)          ;; *) echo "ERROR: --action_format must be 'abs' or 'rel'" >&2; exit 1;; esac

# Single-letter action-format infix used in dag dataset names — see
# int_short_for_round / merged_short_for_round below for rationale (HF 56-char
# repo-name limit leaves only ~10 chars of suffix budget for long base names).
case "$ACTION_FORMAT" in
    rel) ACTION_INFIX="r" ;;
    abs) ACTION_INFIX="a" ;;
esac

# Base repo id — used by both the Round-0 block (to train from scratch) and
# the per-round merge step (which always merges from originals: base + all
# dag{1..r}).
#
# Auto-detect from the resumed policy's train_config.json (the
# dataset.repo_id field) when --initial_policy_path is set. This avoids a
# previously-silent bug where the orchestrator derived
# BASE_REPO=${HF_USER}/${BASE_SHORT} but the actual policy was trained on a
# differently-named variant (e.g. ${BASE_SHORT}_grip0). Mixing the wrong base
# into the merge silently corrupts training because the base's action
# distribution conflicts with what the policy already learned (and what the
# RRT intervention data continues to provide). Pass --base_repo_id to override.
BASE_REPO=""
if [[ -n "${BASE_REPO_OVERRIDE:-}" ]]; then
    BASE_REPO="$BASE_REPO_OVERRIDE"
    echo "Base repo (from --base_repo_id):    $BASE_REPO"
elif [[ -n "$INITIAL_POLICY_PATH" ]]; then
    # Reuse DETECT_CFG from the action-format auto-detect block above.
    if [[ -n "${DETECT_CFG:-}" && -f "$DETECT_CFG" ]]; then
        DETECTED_BASE="$(python3 -c "
import json, sys
c = json.load(open(sys.argv[1]))
print(c.get('dataset', {}).get('repo_id', ''))
" "$DETECT_CFG")"
        if [[ -n "$DETECTED_BASE" ]]; then
            BASE_REPO="$DETECTED_BASE"
            echo "Auto-detected base repo from policy's train_config.json: $BASE_REPO"
        fi
    fi
fi
if [[ -z "$BASE_REPO" ]]; then
    BASE_REPO="${HF_USER}/${BASE_SHORT}"
    echo "Base repo (derived from --base_short): $BASE_REPO"
fi
# Sanity check: the merge base should exist on disk. If not, fall back to the
# splatsim_-prefixed variant (legacy naming) and warn. This catches the case
# where the auto-detected repo_id is a stale path from before a rename.
if [[ ! -d "$LEROBOT_CACHE/$BASE_REPO" ]]; then
    LEGACY_BASE="${HF_USER}/splatsim_${BASE_REPO#${HF_USER}/}"
    if [[ -d "$LEROBOT_CACHE/$LEGACY_BASE" ]]; then
        echo "WARNING: $BASE_REPO not found on disk; falling back to legacy $LEGACY_BASE." >&2
        BASE_REPO="$LEGACY_BASE"
    else
        echo "WARNING: merge base $BASE_REPO not found on disk." >&2
        echo "  Expected: $LEROBOT_CACHE/$BASE_REPO" >&2
        echo "  Pass --base_repo_id=ID explicitly, or symlink the dataset under that name." >&2
    fi
fi

# BASE_DATASET_SHORT is the short name used to derive every per-round dataset
# name: intervention dataset, alias dataset, and merged training dataset. It
# matches the base dataset's name (so dag artifacts inherit identifying
# suffixes like _grip0, _lowres, etc.) but strips the HF user prefix and the
# optional "splatsim_" prefix to keep names short within the 56-char hub
# limit. Computed AFTER the BASE_REPO legacy-fallback above so the strip
# handles both `JennyWWW/foo` and `JennyWWW/splatsim_foo` uniformly.
#
# Example:
#   BASE_REPO          = JennyWWW/splatsim_approach_lever_11_biasend_5path_grip0
#   BASE_DATASET_SHORT = approach_lever_11_biasend_5path_grip0
#   dag1 dataset       = JennyWWW/approach_lever_11_biasend_5path_grip0_dag1
#   dag3_m merged      = JennyWWW/approach_lever_11_biasend_5path_grip0_dag3_m
# Compute the BASE_DATASET_STEM: the BASE_REPO basename minus optional
# `splatsim_` prefix, then optionally replaced by --dag_short_override.
# This is the "stem" that all tag suffixes (run_tag, model_tag, method_tag,
# blends_tag) hang off of. Extracted so the rerun-blends mode can re-derive
# source's dataset short with a different run_tag/blends_tag via the
# _derive_base_dataset_short_for_tags helper below.
BASE_DATASET_STEM="${BASE_REPO#${HF_USER}/}"
if [[ "$STRIP_SPLATSIM_PREFIX" == true ]]; then
    BASE_DATASET_STEM="${BASE_DATASET_STEM#splatsim_}"
fi
# Capture the BASE DATASET's actual short name BEFORE --dag_short_override
# replaces the stem. The base dataset's stats sidecar dir (produced by
# compute_relative_stats.sh against BASE_REPO) is keyed by the dataset's
# OWN short name, not by the dag-artifact stem. Weighted-mode training
# (which feeds the base + every round's intervention into lerobot-train as
# parallel --dataset.repo_ids / --dataset.stats_paths) needs the base
# sidecar at $STATS_BASE/$BASE_REPO_DATASET_SHORT/stats_rel{N}.json.
# Without this, --dag_short_override=foo points the loader at
# $STATS_BASE/foo/... which doesn't exist.
BASE_REPO_DATASET_SHORT="$BASE_DATASET_STEM"
# --dag_short_override lets the user completely replace the stem with a
# shorter name. Useful when the auto-derived stem is at the HF 56-char limit
# and would overflow after adding _dag${N}_m or adding a --run_tag. Doesn't
# affect BASE_REPO (the actual data source); only changes what the dag
# artifacts are named on disk and on the hub.
if [[ -n "$DAG_SHORT_OVERRIDE" ]]; then
    BASE_DATASET_STEM="$DAG_SHORT_OVERRIDE"
fi
# Model-type tag. "pi" (pi05) gets no tag — preserves back-compat with every
# existing pi05 DAgger lineage on disk (dag artifact names were originally
# model-agnostic). "diff" / "act" get an explicit tag so pi05 and diffusion/act
# runs on the same base + run_tag + method don't collide on the same dataset
# names. Training dirs are already disambiguated by the model prefix
# (pi05_/diffusion_/act_), but dag artifact names were not.
MODEL_TAG=""
case "$MODEL" in
    pi)   MODEL_TAG="" ;;
    diff) MODEL_TAG="diff" ;;
    act)  MODEL_TAG="act" ;;
    *) echo "ERROR: no MODEL_TAG mapping for model='$MODEL'" >&2; exit 1 ;;
esac
# Intervention-method tag. "rrt" gets no tag (preserves existing artifact
# names on disk and on the hub — back-compat with all DAgger runs to date).
# Other methods get a 2-char tag so the user can tell runs apart at a glance,
# and dagger_progress.sh / dagger_plot.py auto-discover them as separate
# lineages (each method has its own table + plot).
METHOD_TAG=""
case "$INTERVENTION_METHOD" in
    rrt)         METHOD_TAG="" ;;
    oracle_goal) METHOD_TAG="og" ;;
    *) echo "ERROR: no METHOD_TAG mapping for intervention_method='$INTERVENTION_METHOD'" >&2; exit 1 ;;
esac
# Blends-lineage tag. When --blends is set, this configuration produces a
# DIFFERENT progression of policies than the no-blends lineage (rounds 2+
# train on different merged data → ft_dag{N} checkpoints diverge), so the
# round-N dag artifacts must use distinct names — otherwise resume detection
# would false-positive on the no-blends lineage's artifacts. Empty $BLENDS
# → empty tag (back-compat: byte-identical to today). Sorted descending so
# `--blends='0.8 0.9'` and `--blends='0.9 0.8'` produce the same tag.
BLENDS_TAG=""
if (( ${#BLENDS[@]} > 0 )); then
    # Delegate to dagger_naming.format_blends_tag — sorts descending,
    # zero-pads each pct, joins with `_`, prefixes `b`.
    BLENDS_TAG="$(_py_dagger_name format_blends_tag --blends="${BLENDS[*]}")"
fi

# Helper: combine BASE_DATASET_STEM + MODEL_TAG + METHOD_TAG with a given
# (run_tag, blends_tag) pair. Used twice: once with the current command's
# $RUN_TAG/$BLENDS_TAG (= BASE_DATASET_SHORT), and once in rerun mode with
# the source lineage's run_tag/blends_tag (= SOURCE_INT_SHORT_PREFIX).
_derive_base_dataset_short_for_tags() {
    local run_tag="$1"
    local blends_tag="$2"
    _py_dagger_name derive_base_dataset_short \
        --stem="$BASE_DATASET_STEM" \
        --run_tag="$run_tag" \
        --model_tag="$MODEL_TAG" \
        --method_tag="$METHOD_TAG" \
        --blends_tag="$blends_tag"
}

BASE_DATASET_SHORT="$(_derive_base_dataset_short_for_tags "$RUN_TAG" "$BLENDS_TAG")"
echo "Dag dataset short prefix (dag artifacts named ${BASE_DATASET_SHORT}_${ACTION_INFIX}_dag{N}{,_m,_${MODEL}${ACTION_FORMAT}00}): $BASE_DATASET_SHORT"

# Blends summary (only if any are configured). Delegate ratio→tag via the
# canonical naming module so the formatting stays in lock-step with the
# blend_short_for_round helper (defined later in the file).
if (( ${#BLENDS[@]} > 0 )); then
    _blend_tags=()
    for r in "${BLENDS[@]}"; do
        _tag=$(_py_dagger_name blend_tag --ratio="$r")
        _blend_tags+=( "blend${_tag}(ratio=$r)" )
    done
    echo "Blend datasets per round:        ${_blend_tags[*]}"
    echo "  → each round's intervention will be replayed at each ratio above and stored separately."
fi

# ── intervention scenario subset (computed once, reused every round) ──────────
# When --intervention_sample_from_first=Y is set, pick INTERVENTION_N_EPISODES
# scenarios at random from [0..Y-1] using the given seed, and pass them as
# --env.eval_benchmark_subset=[...] to every per-round lerobot-eval invocation.
# Computed once and reused so every DAgger round operates on the SAME scenarios
# — that's the standard DAgger setup ("keep improving on the same hard set").
if [[ -n "$INTERVENTION_SAMPLE_FROM_FIRST" ]]; then
    INTERVENTION_SUBSET_JSON=$(python3 -c "
import random, sys
seed = int(sys.argv[1])
n = int(sys.argv[2])
pool = int(sys.argv[3])
random.seed(seed)
indices = sorted(random.sample(range(pool), n))
print('[' + ','.join(str(i) for i in indices) + ']')
" "$INTERVENTION_SAMPLE_SEED" "$INTERVENTION_N_EPISODES" "$INTERVENTION_SAMPLE_FROM_FIRST")
    echo "Intervention subset: $INTERVENTION_N_EPISODES scenarios sampled from [0..$((INTERVENTION_SAMPLE_FROM_FIRST-1))] (seed=$INTERVENTION_SAMPLE_SEED)"
    echo "  → $INTERVENTION_SUBSET_JSON"
    echo
fi

# Per-round training-output naming uses 'delta'/'rel' for relative and 'abs'/'abs' for absolute.
# train_sweep.sh writes "delta_basewrist" / "abs_basewrist" depending on USE_RELATIVE_ACTIONS.
if [[ "$ACTION_FORMAT" == "rel" ]]; then
    TRAIN_OUTPUT_ACTION_TAG="delta"
else
    TRAIN_OUTPUT_ACTION_TAG="abs"
fi
# Full model prefix for training output dirs (train_sweep.sh writes "pi05_", "diffusion_", "act_").
case "$MODEL" in
    pi)   TRAIN_OUTPUT_MODEL_PREFIX="pi05" ;;
    diff) TRAIN_OUTPUT_MODEL_PREFIX="diffusion" ;;
    act)  TRAIN_OUTPUT_MODEL_PREFIX="act" ;;
esac

# Base policy basename — every DAgger round writes to
# ${LEROBOT_ROOT}/outputs/training/${BASE_POLICY_NAME}_dag${N}/. Tying the dag
# output dirs to the actual initial policy name means it's always obvious where
# a given checkpoint came from (e.g. dag2 of which base), and the `_grip0` /
# `_grip2` / camera-suffix tags that distinguish base variants carry through
# into the dag dirs automatically.
#
# When --initial_policy_path is set we use its basename (walking up if the path
# points at /checkpoints/.../pretrained_model). When not set, fall back to the
# name train_sweep.sh would produce for a from-scratch round-0 base training.
BASE_POLICY_STEM=""
if [[ -n "$INITIAL_POLICY_PATH" ]]; then
    # Normalize the path: strip a trailing /pretrained_model and any
    # /checkpoints/<step|last>/ segment so basename returns the run dir name.
    _stripped="${INITIAL_POLICY_PATH%/}"
    _stripped="${_stripped%/pretrained_model}"
    _stripped="${_stripped%/checkpoints/*}"
    BASE_POLICY_STEM="$(basename "$_stripped")"
else
    # No initial policy: round 0 will train from scratch using train_sweep.sh,
    # which produces this name pattern. Round 1+ dag dirs hang off of it.
    BASE_POLICY_STEM="${TRAIN_OUTPUT_MODEL_PREFIX}_${BASE_SHORT}_${TRAIN_OUTPUT_ACTION_TAG}_basewrist"
fi

# Helper: combine BASE_POLICY_STEM + METHOD_TAG with a given (run_tag,
# blends_tag) pair. Mirrors _derive_base_dataset_short_for_tags but for
# training dirs. Note: no MODEL_TAG suffix on policy names — the model
# prefix (pi05_/diffusion_/act_) on the stem already disambiguates.
_derive_base_policy_name_for_tags() {
    local run_tag="$1"
    local blends_tag="$2"
    _py_dagger_name derive_base_policy_name \
        --stem="$BASE_POLICY_STEM" \
        --run_tag="$run_tag" \
        --method_tag="$METHOD_TAG" \
        --blends_tag="$blends_tag"
}

BASE_POLICY_NAME="$(_derive_base_policy_name_for_tags "$RUN_TAG" "$BLENDS_TAG")"
echo "Base policy name (dag dirs will be named ${BASE_POLICY_NAME}_dag{N}):  $BASE_POLICY_NAME"

# Source lineage prefixes for rerun-blends mode. In rerun mode the
# intervention + blend datasets read from SOURCE's lineage names; the merged
# datasets + training dirs write to the CURRENT command's BASE_DATASET_SHORT /
# BASE_POLICY_NAME (= NEW lineage). Outside rerun mode, source = current
# (back-compat: every helper that reads SOURCE_* sees identical values).
if [[ "$RERUN_MODE_ENABLED" == "true" ]]; then
    if [[ -n "$SOURCE_RUN_TAG" ]]; then
        SOURCE_INT_SHORT_PREFIX="$(_derive_base_dataset_short_for_tags "$SOURCE_RUN_TAG" "$SOURCE_BLENDS_TAG")"
        SOURCE_POLICY_BASENAME="$(_derive_base_policy_name_for_tags "$SOURCE_RUN_TAG" "$SOURCE_BLENDS_TAG")"
    else
        SOURCE_INT_SHORT_PREFIX="$BASE_DATASET_SHORT"
        SOURCE_POLICY_BASENAME="$BASE_POLICY_NAME"
    fi
    # Explicit overrides win when set (advanced cases — source and current
    # command differ in task/dataset_tag/model/method).
    [[ -n "$REUSE_INTERVENTION_FROM" ]] && SOURCE_INT_SHORT_PREFIX="$REUSE_INTERVENTION_FROM"
    [[ -n "$REUSE_POLICY_FROM"       ]] && SOURCE_POLICY_BASENAME="$REUSE_POLICY_FROM"
    echo "[rerun] Source intervention prefix: $SOURCE_INT_SHORT_PREFIX"
    echo "[rerun] Source policy basename:    $SOURCE_POLICY_BASENAME"
else
    SOURCE_INT_SHORT_PREFIX="$BASE_DATASET_SHORT"
    SOURCE_POLICY_BASENAME="$BASE_POLICY_NAME"
fi

# Rerun-blends mode: auto-detect NUM_ROUNDS from source lineage on disk if
# omitted. Scans the local HF cache for source intervention datasets named
# `<SOURCE_INT_SHORT_PREFIX>_<ACTION_INFIX>_dag<N>` and uses the highest
# contiguous N. Existing dagger lineages are always contiguous (dag1, dag2,
# ...), so contiguous-scan matches the invariant.
auto_detect_source_num_rounds() {
    local n=0 dir
    while :; do
        dir="${LEROBOT_CACHE}/${HF_USER}/${SOURCE_INT_SHORT_PREFIX}_${ACTION_INFIX}_dag$((n + 1))"
        [[ -d "$dir" ]] || break
        n=$((n + 1))
    done
    echo "$n"
}
if [[ "$RERUN_MODE_ENABLED" == "true" && -z "$NUM_ROUNDS" ]]; then
    NUM_ROUNDS="$(auto_detect_source_num_rounds)"
    if (( NUM_ROUNDS < 1 )); then
        echo "ERROR: --num_rounds omitted and no source rounds found under" >&2
        echo "  ${LEROBOT_CACHE}/${HF_USER}/${SOURCE_INT_SHORT_PREFIX}_${ACTION_INFIX}_dag1" >&2
        echo "  Check --rerun_blends_from / --reuse_intervention_from spelling, or" >&2
        echo "  ensure the source lineage's interventions are in the HF cache." >&2
        exit 1
    fi
    echo "[rerun] Auto-detected $NUM_ROUNDS source rounds; defaulting --num_rounds=$NUM_ROUNDS"
fi

# Deferred RETRAIN_ROUND upper-bound check (NUM_ROUNDS now resolved).
if [[ -n "$RETRAIN_ROUND" ]] && (( RETRAIN_ROUND > NUM_ROUNDS )); then
    echo "ERROR: --retrain_round=$RETRAIN_ROUND must be between 1 and $NUM_ROUNDS" >&2; exit 1
fi

# Rerun-blends startup validation. Runs once after NUM_ROUNDS is known; checks
# that the source lineage is fully on disk and that the new lineage won't
# collide with the source.
if [[ "$RERUN_MODE_ENABLED" == "true" ]]; then
    if [[ -n "$REUSE_INTERVENTION_FROM" && -z "$REUSE_POLICY_FROM" ]] \
    || [[ -z "$REUSE_INTERVENTION_FROM" && -n "$REUSE_POLICY_FROM" ]]; then
        echo "ERROR: --reuse_intervention_from and --reuse_policy_from must be set together." >&2
        echo "  (Use --rerun_blends_from for the simple form, or set both flags.)" >&2
        exit 1
    fi
    if (( ${#BLENDS[@]} == 0 )) && (( TARGET_INTERVENTION_VOLUME == 1 )); then
        echo "ERROR: rerun mode with no blends AND --target_intervention_volume=1 would reproduce source's training." >&2
        echo "  Add --blends, OR set --target_intervention_volume > 1 to create a meaningfully" >&2
        echo "  different lineage (= 'what would happen if I retrained with higher intervention" >&2
        echo "  oversample on the same recorded data')." >&2
        exit 1
    fi
    if [[ -n "$SOURCE_RUN_TAG" && "$RUN_TAG" == "$SOURCE_RUN_TAG" && "$BLENDS_TAG" == "$SOURCE_BLENDS_TAG" ]]; then
        echo "ERROR: --run_tag='$RUN_TAG' (+ blends_tag='$BLENDS_TAG') matches source." >&2
        echo "  Choose a different --run_tag so the new lineage's artifacts don't" >&2
        echo "  collide with the source lineage on disk." >&2
        exit 1
    fi
    for f in INTERVENTION_N_EPISODES INTERVENTION_SAMPLE_FROM_FIRST INTERVENTION_SAMPLE_SEED INTERVENTION_EXTRA_ARGS; do
        case "$f" in
            INTERVENTION_N_EPISODES)         _default="100" ;;
            INTERVENTION_SAMPLE_FROM_FIRST)  _default="" ;;
            INTERVENTION_SAMPLE_SEED)        _default="0" ;;
            INTERVENTION_EXTRA_ARGS)         _default="" ;;
        esac
        if [[ "${!f}" != "$_default" ]]; then
            echo "[rerun] WARN: --${f,,}='${!f}' ignored in rerun mode (no new intervention recorded)." >&2
        fi
    done
    # Inline the path expansions here (int_repo_for_round +
    # source_train_output_dir_for_round are defined further down in the
    # helpers section; we run this validation block before those helpers).
    #
    # Skip the source-on-disk existence checks when --cleanup_only is set:
    # in cleanup mode we're just rm-rfing the RERUN lineage's own artifacts,
    # so the source not being there is fine (and is in fact the common case
    # — you might be cleaning up rerun siblings AFTER deleting the source
    # lineage, e.g. when nuking a sweep family).
    if [[ "$CLEANUP_ONLY" != true ]]; then
        for r in $(seq 1 "$NUM_ROUNDS"); do
            _src_int_dir="$LEROBOT_CACHE/${HF_USER}/${SOURCE_INT_SHORT_PREFIX}_${ACTION_INFIX}_dag${r}"
            if [[ ! -d "$_src_int_dir" ]]; then
                echo "ERROR: source intervention missing on disk." >&2
                echo "  Expected at: $_src_int_dir" >&2
                exit 1
            fi
        done
        for r in $(seq 2 "$NUM_ROUNDS"); do
            _src_pol_dir="$LEROBOT_ROOT/outputs/training/${SOURCE_POLICY_BASENAME}_ft_dag$((r - 1))"
            if [[ ! -d "$_src_pol_dir/checkpoints" ]]; then
                echo "ERROR: source policy dir not found: $_src_pol_dir" >&2
                echo "  Need source's _ft_dag$((r - 1)) for blending + training at round $r." >&2
                exit 1
            fi
        done
    fi
fi

# Substring filter for `dagger_progress.sh --filter` so each progress refresh
# is scoped EXACTLY to this run's training dirs, not to every lineage that
# shares the same base_short. The old approach computed PROGRESS_BASE_SHORT
# by stripping prefix + `_${TAG}_basewrist` from BASE_POLICY_NAME, but for
# sweep iterations where BASE_POLICY_NAME ends in `_${RUN_TAG}_${BLENDS_TAG}`
# (not `_basewrist`) the strip silently fails and the query matches no dirs.
# Using BASE_POLICY_NAME (minus the model prefix) as a substring filter
# matches lineage names that dagger_progress.sh constructs by stripping the
# model prefix and the `_dag<N>` suffix — i.e. exactly this run's lineage.
PROGRESS_LINEAGE_FILTER="${BASE_POLICY_NAME#${TRAIN_OUTPUT_MODEL_PREFIX}_}"

# ── helpers ───────────────────────────────────────────────────────────────────

# Validate dataset names (copied from train_sweep.sh:73-126).
# Accepts a repo_id like "JennyWWW/splatsim_xxx" and aborts on failure.
validate_repo_name() {
    local repo="$1"
    local dataset_name="${repo#*/}"
    local errors=0
    if [[ "$dataset_name" =~ [^a-zA-Z0-9_.-] ]]; then
        echo "ERROR: dataset name has invalid chars (a-zA-Z0-9_.- only): '$dataset_name'" >&2; errors=1
    fi
    if [[ "$dataset_name" =~ ^[.-] || "$dataset_name" =~ [.-]$ ]]; then
        echo "ERROR: dataset name cannot start/end with - or .: '$dataset_name'" >&2; errors=1
    fi
    if [[ "$dataset_name" == *"--"* || "$dataset_name" == *".."* ]]; then
        echo "ERROR: dataset name cannot contain -- or ..: '$dataset_name'" >&2; errors=1
    fi
    if [[ "$dataset_name" == *.git || "$dataset_name" == *.ipynb ]]; then
        echo "ERROR: dataset name cannot end with .git or .ipynb: '$dataset_name'" >&2; errors=1
    fi
    if (( ${#repo} > 56 )); then
        echo "ERROR: dataset name exceeds 56 chars (${#repo}): '$repo'" >&2; errors=1
    fi
    if (( errors > 0 )); then return 1; fi
    return 0
}

# Write a per-round dagger config sidecar to <train_dir>/dagger/config.json.
# Logs the orchestrator invocation, computed naming, and (in rerun mode) the
# source-lineage pointers — primary mechanism by which my_scripts/dagger_plot.py
# auto-pairs rerun lineages with their source for the overlay comparison plots.
# Called once at the top of each round; re-writes idempotently on resume so the
# latest invocation's values win.
write_dagger_config_sidecar() {
    local round_n="$1"
    local out_path="$2"
    local round_train_output_dir="$3"
    run_or_echo mkdir -p "$(dirname "$out_path")"
    # In --dry-run, the JSON is not actually written (the mkdir above is a
    # no-op too). Bail to keep the dry-run clean — readers should never
    # encounter half-built sidecars from a dry-run.
    [[ "$DRY_RUN" == true ]] && { echo "[DRY-RUN] would write dagger config sidecar: $out_path"; return 0; }
    # Use python3 to construct the JSON safely (handles escaping, types).
    # Pass everything via env vars to dodge shell-quoting issues with argv.
    local _argv_json
    _argv_json="$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "${ORIG_ARGV[@]}")"
    DAG_CFG_ROUND="$round_n" \
    DAG_CFG_OUT_PATH="$out_path" \
    DAG_CFG_ARGV_JSON="$_argv_json" \
    DAG_CFG_BASE_DATASET_SHORT="$BASE_DATASET_SHORT" \
    DAG_CFG_BASE_POLICY_NAME="$BASE_POLICY_NAME" \
    DAG_CFG_TRAIN_OUTPUT_DIR="$round_train_output_dir" \
    DAG_CFG_RUN_TAG="$RUN_TAG" \
    DAG_CFG_BLENDS_TAG="$BLENDS_TAG" \
    DAG_CFG_MODEL_TAG="$MODEL_TAG" \
    DAG_CFG_METHOD_TAG="$METHOD_TAG" \
    DAG_CFG_RERUN_MODE_ENABLED="$RERUN_MODE_ENABLED" \
    DAG_CFG_SOURCE_RUN_TAG="$SOURCE_RUN_TAG" \
    DAG_CFG_SOURCE_BLENDS_TAG="$SOURCE_BLENDS_TAG" \
    DAG_CFG_SOURCE_INT_SHORT_PREFIX="$SOURCE_INT_SHORT_PREFIX" \
    DAG_CFG_SOURCE_POLICY_BASENAME="$SOURCE_POLICY_BASENAME" \
    DAG_CFG_BRANCHING_POLICY="${CURRENT_POLICY:-}" \
    DAG_CFG_BLENDS_STR="${BLENDS[*]}" \
    DAG_CFG_MODEL="$MODEL" \
    DAG_CFG_ACTION_FORMAT="$ACTION_FORMAT" \
    DAG_CFG_INTERMEDIATE_MODE="$INTERMEDIATE_MODE" \
    DAG_CFG_FINAL_MODE="$FINAL_MODE" \
    DAG_CFG_FINETUNE_STEPS="$FINETUNE_STEPS" \
    DAG_CFG_TARGET_INTERVENTION_VOLUME="$TARGET_INTERVENTION_VOLUME" \
    DAG_CFG_FILTER_BLEND_COLLISIONS="$FILTER_BLEND_COLLISIONS" \
    DAG_CFG_INTERVENTION_METHOD="$INTERVENTION_METHOD" \
    DAG_CFG_INITIAL_POLICY_PATH="${INITIAL_POLICY_PATH:-}" \
    DAG_CFG_BASE_REPO="${BASE_REPO:-}" \
    DAG_CFG_USE_WEIGHTED_SAMPLING="$USE_WEIGHTED_SAMPLING" \
    DAG_CFG_DAGGER_DATA_FRACTION="$DAGGER_DATA_FRACTION" \
    DAG_CFG_NORM_MODE="$NORM_MODE" \
    DAG_CFG_RRT_OBSTACLE_CLEARANCE="$RRT_OBSTACLE_CLEARANCE" \
    DAG_CFG_RRT_SELF_COLLISION_CLEARANCE="$RRT_SELF_COLLISION_CLEARANCE" \
    DAG_CFG_RRT_SELF_COLLISION_SKIP_PAIRS="$RRT_SELF_COLLISION_SKIP_PAIRS" \
    DAG_CFG_WEIGHTED_REPO_IDS_JSON="${DAG_CFG_WEIGHTED_REPO_IDS_JSON:-[]}" \
    DAG_CFG_WEIGHTED_WEIGHTS_JSON="${DAG_CFG_WEIGHTED_WEIGHTS_JSON:-[]}" \
    DAG_CFG_WEIGHTED_STATS_PATHS_JSON="${DAG_CFG_WEIGHTED_STATS_PATHS_JSON:-[]}" \
    DAG_CFG_HEADLESS="$HEADLESS" \
    python3 - <<'PY'
import json, os, datetime, socket, getpass
rerun_mode = None
if os.environ["DAG_CFG_RERUN_MODE_ENABLED"] == "true":
    rerun_mode = {
        "source_run_tag":          os.environ["DAG_CFG_SOURCE_RUN_TAG"],
        "source_blends_tag":       os.environ["DAG_CFG_SOURCE_BLENDS_TAG"],
        "source_int_short_prefix": os.environ["DAG_CFG_SOURCE_INT_SHORT_PREFIX"],
        "source_policy_basename":  os.environ["DAG_CFG_SOURCE_POLICY_BASENAME"],
        "branching_policy_path":   os.environ["DAG_CFG_BRANCHING_POLICY"],
    }
blends_str = os.environ["DAG_CFG_BLENDS_STR"].strip()
blends = [float(x) for x in blends_str.split()] if blends_str else []
config = {
    "schema_version": 1,
    "round": int(os.environ["DAG_CFG_ROUND"]),
    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "user": getpass.getuser(),
    "host": socket.gethostname(),
    "training_output_dir": os.environ["DAG_CFG_TRAIN_OUTPUT_DIR"],
    "rerun_mode": rerun_mode,
    "naming": {
        "base_dataset_short": os.environ["DAG_CFG_BASE_DATASET_SHORT"],
        "base_policy_name":   os.environ["DAG_CFG_BASE_POLICY_NAME"],
        "base_repo":          os.environ["DAG_CFG_BASE_REPO"],
        "run_tag":            os.environ["DAG_CFG_RUN_TAG"],
        "blends_tag":         os.environ["DAG_CFG_BLENDS_TAG"],
        "model_tag":          os.environ["DAG_CFG_MODEL_TAG"],
        "method_tag":         os.environ["DAG_CFG_METHOD_TAG"],
    },
    "config": {
        "model":                  os.environ["DAG_CFG_MODEL"],
        "action_format":          os.environ["DAG_CFG_ACTION_FORMAT"],
        "intermediate_mode":      os.environ["DAG_CFG_INTERMEDIATE_MODE"],
        "final_mode":             os.environ["DAG_CFG_FINAL_MODE"],
        "finetune_steps":         int(os.environ["DAG_CFG_FINETUNE_STEPS"]),
        "blends":                 blends,
        "target_intervention_volume": int(os.environ["DAG_CFG_TARGET_INTERVENTION_VOLUME"]),
        "filter_blend_collisions": os.environ["DAG_CFG_FILTER_BLEND_COLLISIONS"] == "true",
        "intervention_method":    os.environ["DAG_CFG_INTERVENTION_METHOD"],
        "initial_policy_path":    os.environ["DAG_CFG_INITIAL_POLICY_PATH"],
        "use_weighted_sampling":  os.environ["DAG_CFG_USE_WEIGHTED_SAMPLING"] == "true",
        "dagger_data_fraction":   float(os.environ["DAG_CFG_DAGGER_DATA_FRACTION"]),
        "norm_mode":              os.environ["DAG_CFG_NORM_MODE"],
        "rrt_obstacle_clearance":      float(os.environ["DAG_CFG_RRT_OBSTACLE_CLEARANCE"]) if os.environ.get("DAG_CFG_RRT_OBSTACLE_CLEARANCE") else None,
        "rrt_self_collision_clearance": float(os.environ["DAG_CFG_RRT_SELF_COLLISION_CLEARANCE"]) if os.environ.get("DAG_CFG_RRT_SELF_COLLISION_CLEARANCE") else None,
        "rrt_self_collision_skip_pairs": json.loads(os.environ["DAG_CFG_RRT_SELF_COLLISION_SKIP_PAIRS"]) if os.environ.get("DAG_CFG_RRT_SELF_COLLISION_SKIP_PAIRS") else None,
        "weighted_repo_ids":      json.loads(os.environ["DAG_CFG_WEIGHTED_REPO_IDS_JSON"]),
        "weighted_sample_weights": json.loads(os.environ["DAG_CFG_WEIGHTED_WEIGHTS_JSON"]),
        "weighted_stats_paths":   json.loads(os.environ["DAG_CFG_WEIGHTED_STATS_PATHS_JSON"]),
        "headless":               os.environ["DAG_CFG_HEADLESS"] == "true",
    },
    "orchestrator_invocation": {
        # Sort argv lexicographically so two sidecars whose user-side invocations
        # differ only in flag ORDER produce IDENTICAL JSON — diffing two
        # config.json files then surfaces only real differences (added/removed/
        # changed flags), not order noise. All consumers (`_argv_get_flag`,
        # `dagger_cleanup_lineage.sh`, the reeval script) look up by `--<key>=`
        # prefix and are order-independent. The original invocation order isn't
        # load-bearing for reproducibility either since the orchestrator's arg
        # parser uses `--key=value` form throughout (no positional or
        # order-dependent args).
        "argv": sorted(json.loads(os.environ["DAG_CFG_ARGV_JSON"])),
    },
}
out_path = os.environ["DAG_CFG_OUT_PATH"]
os.makedirs(os.path.dirname(out_path), exist_ok=True)
tmp_path = out_path + ".tmp"
with open(tmp_path, "w") as f:
    json.dump(config, f, indent=2)
os.replace(tmp_path, out_path)
PY
}

# Per-round name derivations. All dag artifact names hang off BASE_DATASET_SHORT
# (derived from BASE_REPO above, splatsim_ prefix stripped), so the dag names
# inherit every identifying suffix from the base dataset (_grip0, _lowres, etc.)
# while staying short enough to fit under HuggingFace's 56-char repo-name limit.
#
# Action-format infix (a single letter, "r" or "a") keeps abs and rel runs from
# colliding on the same dataset names — both run flavors can co-exist on disk
# under the same base dataset. Single-letter (not "rel"/"abs") because long base
# names like `approach_lever_11_biasend_5path_grip0` (37 chars) only leave a
# ~10-char suffix budget once `JennyWWW/` (9 chars) is accounted for. The infix
# sits between BASE_DATASET_SHORT and _dag${N}, so the dataset short for round 1
# of a rel run on the grip0 base is e.g.
# "approach_lever_11_biasend_5path_grip0_r_dag1".
# Intervention + alias + blend names use SOURCE_INT_SHORT_PREFIX so that in
# rerun mode they resolve to the SOURCE lineage's names (read-only artifacts);
# outside rerun mode SOURCE_INT_SHORT_PREFIX == BASE_DATASET_SHORT so behavior
# is unchanged.
# Per-round name helpers — all delegated to the canonical naming module via
# _py_dagger_name. Forward derivation (config → name) lives in
# my_scripts/dagger_naming.py so the bash orchestrator and downstream Python
# viz scripts share one source of truth.
int_short_for_round()    { _py_dagger_name int_short    --prefix="$SOURCE_INT_SHORT_PREFIX" --infix="$ACTION_INFIX" --round="$1"; }
int_repo_for_round()     { _py_dagger_name int_repo     --hf_user="$HF_USER" --prefix="$SOURCE_INT_SHORT_PREFIX" --infix="$ACTION_INFIX" --round="$1"; }
alias_short_for_round()  { _py_dagger_name alias_short  --prefix="$SOURCE_INT_SHORT_PREFIX" --infix="$ACTION_INFIX" --round="$1" --model="$MODEL" --action_format="$ACTION_FORMAT"; }
alias_repo_for_round()   { _py_dagger_name alias_repo   --hf_user="$HF_USER" --prefix="$SOURCE_INT_SHORT_PREFIX" --infix="$ACTION_INFIX" --round="$1" --model="$MODEL" --action_format="$ACTION_FORMAT"; }
# Merged dataset uses a compact "_m" suffix (not "_merged") to fit under the
# 56-char HuggingFace repo-name limit. The merged dataset is a derived artifact
# that gets deleted between rounds anyway; the cryptic suffix is acceptable.
# Merged datasets are THIS run's artifacts → use BASE_DATASET_SHORT (NEW lineage).
merged_short_for_round() { _py_dagger_name merged_short --base_dataset_short="$BASE_DATASET_SHORT" --infix="$ACTION_INFIX" --round="$1"; }
merged_repo_for_round()  { _py_dagger_name merged_repo  --hf_user="$HF_USER" --base_dataset_short="$BASE_DATASET_SHORT" --infix="$ACTION_INFIX" --round="$1"; }

# Blend datasets: per round per ratio. Built off int_short_for_round so they
# inherit every identifying suffix the raw intervention has (including, in
# rerun mode, the source lineage's run_tag/blends_tag — so two reruns asking
# for the same ratio at the same source round produce IDENTICAL blend dataset
# names → safe cross-rerun cache reuse). Tag is 3-digit zero-padded
# int(ratio*100) — e.g. 0.9 → "090", 0.1 → "010", 0.95 → "095".
_blend_tag_for_ratio()   { _py_dagger_name blend_tag    --ratio="$1"; }
blend_short_for_round()  { _py_dagger_name blend_short  --prefix="$SOURCE_INT_SHORT_PREFIX" --infix="$ACTION_INFIX" --round="$1" --ratio="$2"; }
blend_repo_for_round()   { _py_dagger_name blend_repo   --hf_user="$HF_USER" --prefix="$SOURCE_INT_SHORT_PREFIX" --infix="$ACTION_INFIX" --round="$1" --ratio="$2"; }
# Collision-filtered blend dataset (produced by filter_blend_collisions.py
# when --filter_blend_collisions is on). Naming: `<blend_short>_nocoll`.
nocoll_short_for_round() { _py_dagger_name nocoll_short --prefix="$SOURCE_INT_SHORT_PREFIX" --infix="$ACTION_INFIX" --round="$1" --ratio="$2"; }
nocoll_repo_for_round()  { _py_dagger_name nocoll_repo  --hf_user="$HF_USER" --prefix="$SOURCE_INT_SHORT_PREFIX" --infix="$ACTION_INFIX" --round="$1" --ratio="$2"; }

# Source training-dir lookup for rerun-blends mode. At round r, the rerun
# branches off SOURCE's _ft_dag(r-1) for both blending (step 2) and finetune
# (step 6). Returns the training dir path; resolve_latest_checkpoint walks
# inside it to find the actual checkpoint.
source_train_output_dir_for_round() {
    echo "$LEROBOT_ROOT/outputs/training/${SOURCE_POLICY_BASENAME}_ft_dag$1"
}

# Training mode for a given round number. As of the refactor away from
# "last round special-cases to FINAL_MODE", every dag round uses the same
# mode. FINAL_MODE now controls an optional post-loop phase, not a round.
mode_for_round() {
    echo "$INTERMEDIATE_MODE"
}

# Whether the post-loop "final from-scratch train on round N's data" phase
# runs. Disabled iff --intermediate_mode=scratch (round N already trained
# from scratch on the same data — the extra step would be a duplicate run).
do_final_scratch() {
    [[ "$FINAL_MODE" == "scratch" && "$INTERMEDIATE_MODE" != "scratch" ]]
}

# Output dir for the post-loop final from-scratch step. Naming pattern:
# ${BASE_POLICY_NAME}_dag${NUM_ROUNDS} (NO "_ft_" infix, since this step
# is from scratch, not a finetune). The finetune dag rounds live at
# ${BASE_POLICY_NAME}_ft_dag${r}, so the two are disambiguated by the
# presence/absence of "_ft_" — easy to distinguish in plots.
train_output_dir_final_scratch() {
    echo "$LEROBOT_ROOT/outputs/training/${BASE_POLICY_NAME}_dag${NUM_ROUNDS}"
}

# Training output dir naming. Mode-aware so finetune rounds get an "_ft"
# infix between the base policy name and the dag round suffix; scratch rounds
# don't (no finetuning was done). Examples for BASE_POLICY_NAME=foo:
#   foo_ft_dag3  ← finetune round 3
#   foo_dag10    ← scratch round 10 (typical for final_mode=scratch)
# Since mode is deterministic from round number, every callsite can hit the
# right path without passing extra state.
train_output_dir_for_round() {
    local round_n="$1"
    local mode
    mode="$(mode_for_round "$round_n")"
    local infix=""
    [[ "$mode" == "finetune" ]] && infix="_ft"
    # In retrain mode, the retrained round gets a suffix so we don't clobber
    # the original training dir. Other rounds (e.g. PREV_R when starting at
    # round N) resolve to their ORIGINAL non-suffixed dirs so we can pull the
    # input checkpoint from there.
    local suffix=""
    if [[ -n "$RETRAIN_ROUND" && "$round_n" == "$RETRAIN_ROUND" ]]; then
        suffix="_${RETRAIN_SUFFIX}"
    fi
    echo "$LEROBOT_ROOT/outputs/training/${BASE_POLICY_NAME}${infix}_dag${round_n}${suffix}"
}
# Sibling "collision-filtered" policy paths/names. Used by step 6b which only
# runs when --filter_blend_collisions is on. The sibling policy is trained on
# the same data mix as step 6 but with raw blends replaced by `_nocoll`
# siblings. We append `_nc` so the name differs from the raw policy without
# clobbering it; the raw policy at `train_output_dir_for_round` stays
# untouched.
nocoll_train_output_dir_for_round() {
    local round_n="$1"
    echo "$(train_output_dir_for_round "$round_n")_nc"
}
# Filename of nocoll_train_output_dir = the canonical policy.repo_id we use
# for the sibling policy (kept consistent with how step 6 derives FT_RUN_NAME
# from BASE_POLICY_NAME).
nocoll_run_name_for_round() {
    basename "$(nocoll_train_output_dir_for_round "$1")"
}
# Back-compat aliases. They both call through to the mode-aware function
# so external scripts that referenced the old function names keep working.
train_output_dir_scratch()  { train_output_dir_for_round "$1"; }
train_output_dir_finetune() { train_output_dir_for_round "$1"; }

# Disk existence checks for resume detection. We try hard to detect crashed/
# partial states so resume doesn't skip past an incomplete step. These checks
# are necessary-but-not-sufficient (see notes below each function).
dataset_exists() {
    local d="$LEROBOT_CACHE/$1"
    # 1. Descriptor must exist.
    [[ -f "$d/meta/info.json" ]] || return 1
    # 2. Reject when total_episodes == 0 — this is the most common partial-
    #    crash signature (env init wrote info.json but the script errored
    #    before any rollout). Falls back to "no episodes" if info.json is
    #    unparsable.
    local n_eps
    n_eps=$(python3 -c "
import json, sys
try:
    info = json.load(open(sys.argv[1]))
    print(int(info.get('total_episodes', 0)))
except Exception:
    print(0)
" "$d/meta/info.json" 2>/dev/null)
    [[ -n "$n_eps" && "$n_eps" -gt 0 ]] || return 1
    # 3. At least one data parquet must be on disk. info.json could claim
    #    episodes that didn't actually finish writing.
    compgen -G "$d/data/chunk-*/*.parquet" >/dev/null || return 1
    return 0
    # NOTE: this does NOT verify that all `total_episodes` parquets are
    # present and well-formed. A recording that crashed at episode 50 of 100
    # would still report total_episodes=50 (the recorder commits per-success)
    # and pass this check — which is fine, the partial dataset is still
    # usable. What this DOES catch: total_episodes=0 (the empty-info case),
    # and data/ being empty.
}
stats_exists() {
    local short="$1"
    # When --action_format=abs, steps 2/5 are skipped entirely and these
    # relative-action sidecars are never produced. Treat that case as
    # "completed" for resume-detection purposes.
    if [[ "$ACTION_FORMAT" != "rel" ]]; then
        return 0
    fi
    # Files are named by chunk size (stats_rel{N}.json), produced by
    # compute_relative_stats.sh — currently chunks 50 (pi05) and 8 (diffusion).
    # We check whichever sidecar corresponds to the active model's chunk size;
    # the other one isn't needed but compute_relative_stats.sh writes both
    # anyway, so a successful run yields both and either model is satisfied.
    local needed_chunk
    case "$TRAIN_OUTPUT_MODEL_PREFIX" in
        diffusion*) needed_chunk=8 ;;
        pi05*|pi0*) needed_chunk=50 ;;
        *)          needed_chunk="" ;;
    esac
    if [[ -z "$needed_chunk" ]]; then
        # No chunk applies (e.g. act → absolute actions only). Treat as present.
        return 0
    fi
    [[ -f "$STATS_BASE/$short/stats_rel${needed_chunk}.json" ]]
}

# Read the policy's n_action_steps (== chunk size for the relative-action stats
# sidecar) from a resumed checkpoint's train_config.json. Echoes empty if the
# config doesn't exist or doesn't carry the field. Pass either the
# pretrained_model dir or a path that contains one.
n_action_steps_from_policy_path() {
    local policy_path="$1"
    local resolved cfg
    resolved="$(readlink -f "$policy_path" 2>/dev/null || echo "$policy_path")"
    for cfg in \
        "$resolved/train_config.json" \
        "$resolved/pretrained_model/train_config.json" \
        "$resolved/checkpoints/last/pretrained_model/train_config.json"
    do
        if [[ -f "$cfg" ]]; then
            python3 -c "
import json, sys
c = json.load(open(sys.argv[1]))
n = c.get('policy', {}).get('n_action_steps')
print(n if n is not None else '')
" "$cfg"
            return 0
        fi
    done
    echo ""
}
training_exists() {
    local exp_dir="$1"
    # Find the "best" checkpoint dir to validate against. Prefer the `last`
    # symlink (canonical), but fall back to the highest-numbered numeric
    # checkpoint when `last` is missing — that's the same fallback the
    # `resolve_latest_checkpoint` helper uses, and a completed run whose
    # `last` symlink got cleaned up (manual disk cleanup, etc.) still has
    # its final numeric checkpoint dir on disk.
    local last_dir="" actual_steps=""
    if [[ -L "$exp_dir/checkpoints/last" || -d "$exp_dir/checkpoints/last" ]]; then
        last_dir="$exp_dir/checkpoints/last/pretrained_model"
        actual_steps=$(readlink -f "$exp_dir/checkpoints/last" 2>/dev/null \
            | xargs -r basename \
            | grep -oE '^[0-9]+$' || echo "")
    fi
    if [[ -z "$actual_steps" && -d "$exp_dir/checkpoints" ]]; then
        # Pick the highest-numbered numeric checkpoint dir under checkpoints/.
        local highest
        highest=$(find "$exp_dir/checkpoints" -mindepth 1 -maxdepth 1 -type d \
            -regextype posix-extended -regex '.*/[0-9]+$' 2>/dev/null \
            | sort -V | tail -1)
        if [[ -n "$highest" ]]; then
            last_dir="$highest/pretrained_model"
            actual_steps=$(basename "$highest" | grep -oE '^[0-9]+$' || echo "")
        fi
    fi
    # 1. Both descriptor AND weights file must exist (descriptor alone could
    #    be from a save that crashed mid-write).
    [[ -n "$last_dir" && -f "$last_dir/train_config.json" && -f "$last_dir/model.safetensors" ]] || return 1
    # 2. The last checkpoint's step number must match the planned total
    #    steps from train_config.json. Otherwise training was interrupted
    #    (crash, OOM, ctrl-C) and we should re-run from where it left off.
    #    lerobot-train's save_freq writes checkpoints/{step}/ on every save,
    #    and 'last' symlinks to the most recent. If steps=4000 in
    #    train_config.json but the resolved checkpoint is at step 002000,
    #    training was cut off.
    local target_steps
    target_steps=$(python3 -c "
import json, sys
try:
    cfg = json.load(open(sys.argv[1]))
    print(int(cfg.get('steps', 0)))
except Exception:
    print(0)
" "$last_dir/train_config.json" 2>/dev/null)
    [[ -n "$target_steps" && "$target_steps" -gt 0 ]] || return 1
    [[ -n "$actual_steps" ]] || return 1
    # Force base-10 arithmetic; otherwise "004000" gets parsed as octal and
    # silently mismatches 4000.
    (( 10#$actual_steps == target_steps )) || return 1
    return 0
    # NOTE: this does not verify that model.safetensors is non-corrupt (only
    # that the file exists with the right step count). A torch.load on a
    # corrupt safetensors would crash downstream — which is the right
    # behavior (better to fail loud at training-resume time than to silently
    # restart). It also assumes lerobot-train always saves at the final step,
    # which it does (the train loop calls save() once at step==total_steps).
}

# Resolve the latest checkpoint dir from a training experiment dir.
# Mirrors resume_training.sh's resolve_config. Tries (in order):
#   1. {exp}/checkpoints/last/pretrained_model
#   2. {exp}/checkpoints/<highest-numeric>/pretrained_model (when 'last' symlink missing)
#   3. {exp}/pretrained_model
#   4. {exp} itself (if it already contains train_config.json)
resolve_latest_checkpoint() {
    local exp_dir="$1"
    local candidate
    # 1. checkpoints/last
    candidate="$exp_dir/checkpoints/last/pretrained_model"
    [[ -d "$candidate" && -f "$candidate/train_config.json" ]] && { echo "$candidate"; return 0; }
    # 2. highest-numbered checkpoint under checkpoints/
    if [[ -d "$exp_dir/checkpoints" ]]; then
        local highest
        highest=$(ls -1 "$exp_dir/checkpoints" 2>/dev/null \
            | grep -E '^[0-9]+$' | sort -n | tail -1)
        if [[ -n "$highest" ]]; then
            candidate="$exp_dir/checkpoints/$highest/pretrained_model"
            [[ -d "$candidate" && -f "$candidate/train_config.json" ]] && { echo "$candidate"; return 0; }
        fi
    fi
    # 3. directly-nested pretrained_model
    candidate="$exp_dir/pretrained_model"
    [[ -d "$candidate" && -f "$candidate/train_config.json" ]] && { echo "$candidate"; return 0; }
    # 4. the exp_dir itself
    [[ -d "$exp_dir" && -f "$exp_dir/train_config.json" ]] && { echo "$exp_dir"; return 0; }
    echo "ERROR: cannot resolve a checkpoint dir from '$exp_dir'" >&2
    return 1
}

# Dry-run helper.
run_or_echo() {
    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY-RUN] $*"
    else
        echo "+ $*"
        "$@"
    fi
}

# ── SplatSim lifecycle helpers (used when --manage_splatsim is on) ────────────
# We track the child pid in MANAGED_SIM_PID. start_sim / stop_sim are idempotent
# so they can be called freely around training. The EXIT trap (installed once,
# right after these definitions) guarantees the sim is cleaned up even if the
# orchestrator dies mid-run.
MANAGED_SIM_PID=""
MANAGED_SIM_LOG=""
# Auxiliary headless sim — used by the collision-filter step when
# --filter_blend_collisions is on. Lifecycle mirrors the main sim: started
# lazily before the first filter call, stopped at orchestrator exit. Lives
# on a separate port so it can coexist with the main intervention sim.
MANAGED_FILTER_SIM_PID=""
MANAGED_FILTER_SIM_LOG=""
# Resolved aux port. Lazily filled by start_filter_sim() the first time it's
# called: explicit --filter_collision_env_port wins; else ENV_EXTERNAL_PORT + 100.
FILTER_COLLISION_ENV_PORT_RESOLVED=""

port_in_use() {
    # echoes pid bound to TCP port $1 (first one found), or empty.
    # Be defensive: lsof returns nonzero when there are no matches, and the
    # `| head` pipeline interacts badly with `set -o pipefail` — without the
    # `|| true` here the command substitution returns nonzero, which set -e
    # propagates and exits the script silently.
    local out
    out="$(lsof -ti "tcp:$1" -sTCP:LISTEN 2>/dev/null || true)"
    printf '%s\n' "${out%%$'\n'*}"
}

wait_for_port() {
    # Args: $1=port, $2=max_wait_seconds.
    # Returns 0 once a TCP connect succeeds; 1 on timeout.
    local port="$1" max_wait="$2"
    local i
    for ((i=1; i<=max_wait; i++)); do
        if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
            exec 3<&-; exec 3>&-
            return 0
        fi
        sleep 1
    done
    return 1
}

start_sim() {
    echo "[start_sim] entering (managed=$MANAGE_SPLATSIM, dry-run=$DRY_RUN, pid=${MANAGED_SIM_PID:-<unset>})"
    [[ "$MANAGE_SPLATSIM" == true ]] || return 0
    if [[ "$DRY_RUN" == true ]]; then
        # Idempotent in dry-run too — only print "would start" on transitions.
        [[ "$MANAGED_SIM_PID" == "DRYRUN" ]] && return 0
        local _hl=""
        [[ "$HEADLESS" == true ]] && _hl=" --headless"
        echo "[DRY-RUN] would start SplatSim on port $ENV_EXTERNAL_PORT:"
        echo "[DRY-RUN]   cwd: $SPLATSIM_ROOT"
        echo "[DRY-RUN]   cmd: python scripts/launch_nodes.py --robot $SPLATSIM_ROBOT --robot_port $ENV_EXTERNAL_PORT --hostname $ENV_EXTERNAL_HOST --robot_name $SPLATSIM_ROBOT_NAME --eval_benchmark_repo_id $EVAL_BENCHMARK_REPO_ID$_hl"
        MANAGED_SIM_PID="DRYRUN"
        return 0
    fi
    # Already running under our management?
    if [[ -n "$MANAGED_SIM_PID" ]] && kill -0 "$MANAGED_SIM_PID" 2>/dev/null; then
        return 0
    fi
    # Port already in use by someone else?
    local existing
    existing="$(port_in_use "$ENV_EXTERNAL_PORT")"
    if [[ -n "$existing" ]]; then
        echo "ERROR: port $ENV_EXTERNAL_PORT already in use by pid $existing." >&2
        echo "  Either kill it, change --env_external_port, or pass --no_manage_splatsim." >&2
        exit 1
    fi
    # Sim log lives inside the current round's training dir/dagger/ when a
    # round number is in scope ($r set by the per-round loop), else falls
    # back to the legacy shared location. Co-locating with the round
    # artifacts means dagger_cleanup_lineage.sh wipes the log automatically
    # when it rm's the round's training dir — no more 600+ orphan logs
    # accumulating under outputs/dagger/.
    local _log_dir=""
    if [[ -n "${r:-}" ]] && declare -F train_output_dir_for_round >/dev/null; then
        _log_dir="$(train_output_dir_for_round "$r")/dagger"
    fi
    if [[ -z "$_log_dir" ]]; then
        _log_dir="$LEROBOT_ROOT/outputs/dagger"
    fi
    mkdir -p "$_log_dir"
    MANAGED_SIM_LOG="$_log_dir/splatsim_$(date +%Y%m%d_%H%M%S).log"
    local launch_cmd=(
        python scripts/launch_nodes.py
        --robot              "$SPLATSIM_ROBOT"
        --robot_port         "$ENV_EXTERNAL_PORT"
        --hostname           "$ENV_EXTERNAL_HOST"
        --robot_name         "$SPLATSIM_ROBOT_NAME"
        --eval_benchmark_repo_id "$EVAL_BENCHMARK_REPO_ID"
    )
    # When the orchestrator's skip-succeeded path computed a per-round
    # failed-only subset, pass it through to launch_nodes.py so SplatSim's
    # internal `_eval_benchmark_subset` matches what lerobot's
    # --env.eval_benchmark_subset says. Empty = SplatSim defaults to the
    # full subset (legacy behavior). The variable holds a comma-separated
    # string ("3,7,10,23,25"); tyro expects each int as its own argv slot,
    # so split on commas before appending.
    if [[ -n "${EVAL_BENCHMARK_SUBSET_FOR_SIM:-}" ]]; then
        IFS=',' read -ra _ebs_arr <<< "$EVAL_BENCHMARK_SUBSET_FOR_SIM"
        launch_cmd+=( --eval_benchmark_subset "${_ebs_arr[@]}" )
    fi
    # --headless connects pybullet via p.DIRECT instead of p.GUI — no
    # visualizer window, ~50% less GPU memory, faster startup. Mirrors what
    # start_filter_sim() does unconditionally for the collision-filter sim.
    [[ "$HEADLESS" == true ]] && launch_cmd+=( --headless )
    echo "Starting SplatSim:"
    echo "  cwd:     $SPLATSIM_ROOT"
    echo "  cmd:     ${launch_cmd[*]}"
    echo "  log:     $MANAGED_SIM_LOG"
    # Launch sim in background. setsid puts it in its own session so it survives
    # SIGINT to our shell (we kill it explicitly in stop_sim / trap).
    (
        cd "$SPLATSIM_ROOT" || exit 1
        setsid "${launch_cmd[@]}" </dev/null >"$MANAGED_SIM_LOG" 2>&1
    ) &
    SIM_LAUNCH_BG_PID=$!
    echo "  launcher subshell pid: $SIM_LAUNCH_BG_PID (sim will fork under setsid)"
    # Capture the actual sim pid (the setsid grandchild) by polling for the
    # port to come up, then querying lsof.
    echo -n "  waiting for port $ENV_EXTERNAL_PORT to come up "
    if wait_for_port "$ENV_EXTERNAL_PORT" 300; then
        echo "ready."
    else
        echo
        echo "ERROR: SplatSim did not come up within 300s. Last 30 log lines:" >&2
        tail -30 "$MANAGED_SIM_LOG" >&2 || true
        exit 1
    fi
    MANAGED_SIM_PID="$(port_in_use "$ENV_EXTERNAL_PORT")"
    if [[ -z "$MANAGED_SIM_PID" ]]; then
        echo "WARNING: port is up but pid lookup via lsof returned empty." >&2
    else
        echo "  pid:     $MANAGED_SIM_PID"
    fi
    # Give the sim a beat to finish internal init beyond just port-bind.
    sleep 3
}

stop_sim() {
    [[ "$MANAGE_SPLATSIM" == true ]] || return 0
    if [[ "$MANAGED_SIM_PID" == "DRYRUN" ]]; then
        echo "[DRY-RUN] would stop SplatSim on port $ENV_EXTERNAL_PORT"
        MANAGED_SIM_PID=""
        return 0
    fi
    [[ -z "$MANAGED_SIM_PID" ]] && return 0
    if ! kill -0 "$MANAGED_SIM_PID" 2>/dev/null; then
        MANAGED_SIM_PID=""
        return 0
    fi
    echo "Stopping SplatSim (pid=$MANAGED_SIM_PID)..."
    # SIGTERM the whole process group so any pybullet/gaussian-splat subprocs die too.
    local pgid
    pgid="$(ps -o pgid= -p "$MANAGED_SIM_PID" 2>/dev/null | tr -d ' ')"
    if [[ -n "$pgid" ]]; then
        kill -TERM -"$pgid" 2>/dev/null || true
    else
        kill -TERM "$MANAGED_SIM_PID" 2>/dev/null || true
    fi
    # Wait up to 30s for graceful exit.
    local i
    for ((i=1; i<=30; i++)); do
        kill -0 "$MANAGED_SIM_PID" 2>/dev/null || break
        sleep 1
    done
    if kill -0 "$MANAGED_SIM_PID" 2>/dev/null; then
        echo "  sim did not exit on SIGTERM; sending SIGKILL."
        [[ -n "$pgid" ]] && kill -KILL -"$pgid" 2>/dev/null || kill -KILL "$MANAGED_SIM_PID" 2>/dev/null || true
    fi
    # Wait for the port to actually free (kernel may keep it in TIME_WAIT briefly).
    for ((i=1; i<=15; i++)); do
        [[ -z "$(port_in_use "$ENV_EXTERNAL_PORT")" ]] && break
        sleep 1
    done
    MANAGED_SIM_PID=""
    echo "  stopped."
}

# Headless auxiliary sim helpers, used only by the collision-filter step.
# Same lifecycle pattern as start_sim/stop_sim above, but pinned to
# $FILTER_COLLISION_ENV_PORT_RESOLVED and launched with --headless.
start_filter_sim() {
    [[ "$FILTER_BLEND_COLLISIONS" == "true" ]] || return 0
    [[ "$MANAGE_SPLATSIM" == true ]] || return 0
    # Resolve the aux port lazily: explicit flag wins; else main port + 100
    # so the two sims don't collide on the same TCP port.
    if [[ -z "$FILTER_COLLISION_ENV_PORT_RESOLVED" ]]; then
        if [[ -n "$FILTER_COLLISION_ENV_PORT" ]]; then
            FILTER_COLLISION_ENV_PORT_RESOLVED="$FILTER_COLLISION_ENV_PORT"
        else
            FILTER_COLLISION_ENV_PORT_RESOLVED=$(( ENV_EXTERNAL_PORT + 100 ))
        fi
    fi
    if [[ "$DRY_RUN" == true ]]; then
        [[ "$MANAGED_FILTER_SIM_PID" == "DRYRUN" ]] && return 0
        # Reuse the same clearance-scraping the live path does so the dry-run
        # preview matches what would actually be invoked. `|| true` because
        # grep exits non-zero when the flag is absent and `set -eo pipefail`
        # would otherwise abort the dry-run.
        local _dr_obs_clr _dr_self_clr
        _dr_obs_clr="$({ printf '%s' "$INTERVENTION_EXTRA_ARGS" \
            | grep -oE -- '--policy\.shared_autonomy_config\.rrt_obstacle_clearance=[0-9.eE+-]+' \
            | tail -1 | sed -E 's/.*=//'; } || true)"
        _dr_self_clr="$({ printf '%s' "$INTERVENTION_EXTRA_ARGS" \
            | grep -oE -- '--policy\.shared_autonomy_config\.rrt_self_collision_clearance=[0-9.eE+-]+' \
            | tail -1 | sed -E 's/.*=//'; } || true)"
        : "${_dr_obs_clr:=0.0}"
        : "${_dr_self_clr:=0.0}"
        echo "[DRY-RUN] would start HEADLESS SplatSim on port $FILTER_COLLISION_ENV_PORT_RESOLVED:"
        echo "[DRY-RUN]   cwd: $SPLATSIM_ROOT"
        echo "[DRY-RUN]   cmd: python scripts/launch_nodes.py --robot $SPLATSIM_ROBOT --robot_port $FILTER_COLLISION_ENV_PORT_RESOLVED --hostname $ENV_EXTERNAL_HOST --robot_name $SPLATSIM_ROBOT_NAME --eval_benchmark_repo_id $EVAL_BENCHMARK_REPO_ID --in_collision_obstacle_clearance $_dr_obs_clr --in_collision_self_collision_clearance $_dr_self_clr --headless"
        MANAGED_FILTER_SIM_PID="DRYRUN"
        return 0
    fi
    if [[ -n "$MANAGED_FILTER_SIM_PID" ]] && kill -0 "$MANAGED_FILTER_SIM_PID" 2>/dev/null; then
        return 0
    fi
    local existing
    existing="$(port_in_use "$FILTER_COLLISION_ENV_PORT_RESOLVED")"
    if [[ -n "$existing" ]]; then
        echo "ERROR: aux headless port $FILTER_COLLISION_ENV_PORT_RESOLVED already in use by pid $existing." >&2
        echo "  Either kill it, change --filter_collision_env_port, or skip the filter step." >&2
        exit 1
    fi
    # Same per-round-dir co-location as start_sim above. Filter sim is only
    # invoked from step 2 (blend collision filtering), which always runs in
    # a per-round context, so $r is reliably in scope here.
    local _flog_dir=""
    if [[ -n "${r:-}" ]] && declare -F train_output_dir_for_round >/dev/null; then
        _flog_dir="$(train_output_dir_for_round "$r")/dagger"
    fi
    if [[ -z "$_flog_dir" ]]; then
        _flog_dir="$LEROBOT_ROOT/outputs/dagger"
    fi
    mkdir -p "$_flog_dir"
    MANAGED_FILTER_SIM_LOG="$_flog_dir/splatsim_filter_$(date +%Y%m%d_%H%M%S).log"
    # Sync the filter sim's collision clearances with the SA wrapper's RRT
    # clearances. Without this, the filter would judge a frame "collision"
    # only on actual contact (0 m) while RRT plans treat anything within
    # `rrt_*_clearance` as a collision — the filter's `_nocoll` survivors
    # would still include trajectories RRT itself would reject.
    # We scrape the two values from $INTERVENTION_EXTRA_ARGS so the
    # filter follows whatever the user set on the wrapper. Defaults to 0
    # when those flags aren't present (preserves the env's historical
    # behavior).
    local _filter_obs_clr _filter_self_clr
    # `|| true` because grep exits non-zero when the flag is absent and
    # `set -eo pipefail` would otherwise abort start_filter_sim.
    _filter_obs_clr="$({ printf '%s' "$INTERVENTION_EXTRA_ARGS" \
        | grep -oE -- '--policy\.shared_autonomy_config\.rrt_obstacle_clearance=[0-9.eE+-]+' \
        | tail -1 | sed -E 's/.*=//'; } || true)"
    _filter_self_clr="$({ printf '%s' "$INTERVENTION_EXTRA_ARGS" \
        | grep -oE -- '--policy\.shared_autonomy_config\.rrt_self_collision_clearance=[0-9.eE+-]+' \
        | tail -1 | sed -E 's/.*=//'; } || true)"
    : "${_filter_obs_clr:=0.0}"
    : "${_filter_self_clr:=0.0}"
    local launch_cmd=(
        python scripts/launch_nodes.py
        --robot              "$SPLATSIM_ROBOT"
        --robot_port         "$FILTER_COLLISION_ENV_PORT_RESOLVED"
        --hostname           "$ENV_EXTERNAL_HOST"
        --robot_name         "$SPLATSIM_ROBOT_NAME"
        --eval_benchmark_repo_id "$EVAL_BENCHMARK_REPO_ID"
        --in_collision_obstacle_clearance "$_filter_obs_clr"
        --in_collision_self_collision_clearance "$_filter_self_clr"
        --headless
    )
    echo "Starting HEADLESS SplatSim (for collision filter):"
    echo "  cwd:     $SPLATSIM_ROOT"
    echo "  cmd:     ${launch_cmd[*]}"
    echo "  log:     $MANAGED_FILTER_SIM_LOG"
    (
        cd "$SPLATSIM_ROOT" || exit 1
        setsid "${launch_cmd[@]}" </dev/null >"$MANAGED_FILTER_SIM_LOG" 2>&1
    ) &
    echo "  launcher subshell pid: $! (sim will fork under setsid)"
    echo -n "  waiting for port $FILTER_COLLISION_ENV_PORT_RESOLVED to come up "
    if wait_for_port "$FILTER_COLLISION_ENV_PORT_RESOLVED" 300; then
        echo "ready."
    else
        echo
        echo "ERROR: headless SplatSim did not come up within 300s. Last 30 log lines:" >&2
        tail -30 "$MANAGED_FILTER_SIM_LOG" >&2 || true
        exit 1
    fi
    MANAGED_FILTER_SIM_PID="$(port_in_use "$FILTER_COLLISION_ENV_PORT_RESOLVED")"
    sleep 3
}

stop_filter_sim() {
    [[ "$MANAGE_SPLATSIM" == true ]] || return 0
    if [[ "$MANAGED_FILTER_SIM_PID" == "DRYRUN" ]]; then
        echo "[DRY-RUN] would stop HEADLESS SplatSim on port $FILTER_COLLISION_ENV_PORT_RESOLVED"
        MANAGED_FILTER_SIM_PID=""
        return 0
    fi
    [[ -z "$MANAGED_FILTER_SIM_PID" ]] && return 0
    if ! kill -0 "$MANAGED_FILTER_SIM_PID" 2>/dev/null; then
        MANAGED_FILTER_SIM_PID=""
        return 0
    fi
    echo "Stopping HEADLESS SplatSim (pid=$MANAGED_FILTER_SIM_PID)..."
    local pgid
    pgid="$(ps -o pgid= -p "$MANAGED_FILTER_SIM_PID" 2>/dev/null | tr -d ' ' || true)"
    [[ -n "$pgid" ]] && kill -TERM -"$pgid" 2>/dev/null || kill -TERM "$MANAGED_FILTER_SIM_PID" 2>/dev/null || true
    for ((i=1; i<=20; i++)); do
        kill -0 "$MANAGED_FILTER_SIM_PID" 2>/dev/null || break
        sleep 1
    done
    if kill -0 "$MANAGED_FILTER_SIM_PID" 2>/dev/null; then
        [[ -n "$pgid" ]] && kill -KILL -"$pgid" 2>/dev/null || kill -KILL "$MANAGED_FILTER_SIM_PID" 2>/dev/null || true
    fi
    MANAGED_FILTER_SIM_PID=""
    echo "  stopped."
}

# Cleanup on exit (success or crash). Idempotent. Both sims get stopped.
trap 'stop_sim; stop_filter_sim' EXIT

# ── pre-flight: validate every round's derived names ──────────────────────────
# Skipped in --cleanup_only mode: that path only deletes already-existing
# artifacts and never creates new datasets, so the 56-char HuggingFace
# repo-name limit doesn't apply. Without this gate, cleanup of lineages built
# before the limit was enforced (or with a longer-than-allowed --run_tag /
# --blends combo) would error out at this step and the user would be unable
# to delete them via dagger_cleanup_lineage.sh.
if [[ "$CLEANUP_ONLY" == true ]]; then
    echo "Pre-flight: SKIPPED (--cleanup_only — only deleting existing artifacts; their names don't need to satisfy the 56-char HF limit)."
else
echo "Pre-flight: validating derived repo names for $NUM_ROUNDS rounds..."
ANY_NAME_TOO_LONG=false
ALIAS_NAMES_OVERFLOW=false
for r in $(seq 1 "$NUM_ROUNDS"); do
    # Always validate the raw intervention name.
    if ! validate_repo_name "$(int_repo_for_round "$r")"; then
        ANY_NAME_TOO_LONG=true
    fi
    # Only validate the merged-output name in merge mode — weighted mode never
    # creates a merged dataset on disk, so its name length doesn't matter.
    # This is what lets long base names (which would overflow `_dag${N}_m`)
    # still run in weighted mode where the merge step is skipped entirely.
    if [[ "$USE_WEIGHTED_SAMPLING" != "true" ]]; then
        if ! validate_repo_name "$(merged_repo_for_round "$r")"; then
            ANY_NAME_TOO_LONG=true
        fi
    fi
    # Only validate the alias name when the alias step is enabled.
    if [[ "$SKIP_ALIAS_STEP" == false ]]; then
        if ! validate_repo_name "$(alias_repo_for_round "$r")"; then
            ANY_NAME_TOO_LONG=true
            ALIAS_NAMES_OVERFLOW=true
        fi
    fi
    # Step 6b (collision-filtered sibling policy): validate the policy.repo_id
    # length. Used as --policy.repo_id (wandb run name + HF push target if
    # push_to_hub=true). Wandb's run-name limit is 128 chars; HF Hub's
    # repo-name limit is 96. We validate against 128 so non-push runs can
    # still use long lineage names — the user will hit HF's 96 separately
    # at push time if they enable that. Append `_nc` to the existing dir name.
    if [[ "$FILTER_BLEND_COLLISIONS" == "true" ]]; then
        _nc_name="$(nocoll_run_name_for_round "$r")"
        if (( ${#_nc_name} > 128 )); then
            echo "ERROR: nocoll sibling policy name exceeds 128 chars (${#_nc_name}): '$_nc_name'" >&2
            echo "  Step 6b (--filter_blend_collisions) appends '_nc' to each round's training dir." >&2
            echo "  Shorten the lineage somewhere upstream (e.g. --run_tag, --base_short, or --dag_short_override)." >&2
            ANY_NAME_TOO_LONG=true
        fi
    fi
done
if [[ "$ANY_NAME_TOO_LONG" == true ]]; then
    echo "ERROR: one or more derived repo names violate the 56-char limit or other rules." >&2
    if [[ "$ALIAS_NAMES_OVERFLOW" == true ]]; then
        echo "  Note: the alias suffix '_${MODEL}${ACTION_FORMAT}00' adds 8 chars on top of '_dag{N}'." >&2
        echo "  Either:" >&2
        echo "    (a) pass --skip_alias_step to bypass augment_ratios_sweep.sh and merge" >&2
        echo "        intervention datasets directly (loses forward-compat with" >&2
        echo "        future ratio>0 blending augmentations on these rounds), OR" >&2
        echo "    (b) use a shorter base dataset name (current dag-name prefix '$BASE_DATASET_SHORT'," >&2
        echo "        length ${#BASE_DATASET_SHORT}; was derived from BASE_REPO=$BASE_REPO)." >&2
    else
        echo "  Use a shorter base dataset name (current dag-name prefix '$BASE_DATASET_SHORT'," >&2
        echo "  length ${#BASE_DATASET_SHORT}; was derived from BASE_REPO=$BASE_REPO) and retry." >&2
    fi
    exit 1
fi
echo "  ✓ all derived names valid (alias step: $([ "$SKIP_ALIAS_STEP" == true ] && echo SKIPPED || echo enabled))."
fi  # end of `if [[ "$CLEANUP_ONLY" == true ]]; then ... else ...`
echo

# ── mode-purity validation ────────────────────────────────────────────────────
# A single lineage must run end-to-end in one mode — merge OR weighted, never
# half-and-half. Two distinguishing artifacts on disk:
#   * merge-mode runs produce `<lineage>_dag${r}_m` merged datasets.
#   * weighted-mode runs write a per-round dagger/config.json sidecar with
#     `use_weighted_sampling=true` (added below in write_dagger_config_sidecar).
# Mixed-mode reuse silently corrupts: a weighted-mode resume on top of an
# already-merged round would skip the merge but try to retrain on incompatible
# stats; a merge-mode resume on top of a weighted-mode round would re-merge
# (slow + wasteful, but not corrupting). Reject both upfront.
echo "Mode-purity validation:"
if [[ "$USE_WEIGHTED_SAMPLING" == "true" ]]; then
    _conflicting=()
    for _r in $(seq 1 "$NUM_ROUNDS"); do
        _m_repo="$(merged_repo_for_round "$_r")"
        if [[ -d "$LEROBOT_CACHE/$_m_repo" ]]; then
            _conflicting+=( "round $_r: $LEROBOT_CACHE/$_m_repo" )
        fi
    done
    if (( ${#_conflicting[@]} > 0 )); then
        echo "ERROR: lineage already has merge-mode artifacts but --use_weighted_sampling is set." >&2
        echo "  Conflicting (merged datasets from a prior merge-mode run):" >&2
        for _c in "${_conflicting[@]}"; do echo "    $_c" >&2; done
        echo "  Resolve by either:" >&2
        echo "    (1) starting a weighted-mode lineage under a different --run_tag, OR" >&2
        echo "    (2) running my_scripts/dagger_cleanup_lineage.sh first." >&2
        exit 1
    fi
    echo "  ✓ no prior merge-mode artifacts; safe to run in weighted mode."
else
    _conflicting=()
    for _r in $(seq 1 "$NUM_ROUNDS"); do
        _train_dir="$(train_output_dir_for_round "$_r")"
        _sidecar="$_train_dir/dagger/config.json"
        if [[ -f "$_sidecar" ]]; then
            _flag=$(python3 -c "
import json, sys
try:
    c = json.load(open(sys.argv[1]))
except Exception:
    print('false'); sys.exit()
v = (c.get('config') or {}).get('use_weighted_sampling')
print('true' if v else 'false')
" "$_sidecar" 2>/dev/null || echo "false")
            if [[ "$_flag" == "true" ]]; then
                _conflicting+=( "round $_r: $_sidecar (use_weighted_sampling=true)" )
            fi
        fi
    done
    if (( ${#_conflicting[@]} > 0 )); then
        echo "ERROR: lineage already has weighted-sampling-mode artifacts but the new run defaults to merge mode." >&2
        echo "  Conflicting (per-round sidecars from a prior weighted-mode run):" >&2
        for _c in "${_conflicting[@]}"; do echo "    $_c" >&2; done
        echo "  Resolve by either:" >&2
        echo "    (1) passing --use_weighted_sampling to stay in weighted mode, OR" >&2
        echo "    (2) starting a merge-mode lineage under a different --run_tag, OR" >&2
        echo "    (3) running my_scripts/dagger_cleanup_lineage.sh first." >&2
        exit 1
    fi
    echo "  ✓ no prior weighted-mode artifacts; safe to run in merge mode."
fi
# Second mode-purity check: within weighted mode, the chosen --norm_mode
# must match any prior round's sidecar. Switching norm_mode mid-lineage
# (e.g. r1..r3 were trained with norm_mode=aggregated, then r4 attempts
# norm_mode=base_only) would silently change the normalization the policy
# expects, producing a hard distribution shift at finetune. Reject so the
# user picks a fresh --run_tag instead.
#
# Skipped in --cleanup_only mode: we're just rm-rf'ing existing artifacts,
# so a norm_mode mismatch between the resumed-via-sidecar argv and the
# prior rounds doesn't matter (no training happens). Same logic as the
# rerun-mode source-existence guard above.
if [[ "$USE_WEIGHTED_SAMPLING" == "true" && "$CLEANUP_ONLY" != true ]]; then
    _norm_conflicting=()
    for _r in $(seq 1 "$NUM_ROUNDS"); do
        _train_dir="$(train_output_dir_for_round "$_r")"
        _sidecar="$_train_dir/dagger/config.json"
        if [[ -f "$_sidecar" ]]; then
            _prior_norm=$(python3 -c "
import json, sys
try:
    c = json.load(open(sys.argv[1]))
except Exception:
    print('')
    sys.exit()
v = (c.get('config') or {}).get('norm_mode')
print(v if v else '')
" "$_sidecar" 2>/dev/null || echo "")
            # Pre-norm_mode sidecars have no 'norm_mode' key. Their
            # effective behavior was base_only (the --dataset.stats_path
            # override; see the long comment in the finetune block).
            # Treat absent → base_only for back-compat: a lineage built
            # before this flag existed can only be safely resumed with
            # --norm_mode=base_only.
            [[ -z "$_prior_norm" ]] && _prior_norm="base_only"
            if [[ "$_prior_norm" != "$NORM_MODE" ]]; then
                _norm_conflicting+=( "round $_r: $_sidecar (norm_mode=$_prior_norm)" )
            fi
        fi
    done
    if (( ${#_norm_conflicting[@]} > 0 )); then
        echo "ERROR: lineage has prior rounds with norm_mode != '$NORM_MODE'." >&2
        echo "  Switching normalization mid-lineage would silently shift the policy's" >&2
        echo "  input distribution. Conflicting sidecars:" >&2
        for _c in "${_norm_conflicting[@]}"; do echo "    $_c" >&2; done
        echo "  Resolve by either:" >&2
        echo "    (1) re-running with --norm_mode=$_prior_norm to match the prior rounds, OR" >&2
        echo "    (2) starting a new lineage under a different --run_tag." >&2
        echo "  NOTE: pre-existing lineages without a recorded norm_mode are treated as" >&2
        echo "        norm_mode=base_only for back-compat (the prior accidental behavior)." >&2
        exit 1
    fi
    echo "  ✓ norm_mode='$NORM_MODE' is consistent with prior rounds (or no prior rounds)."
fi
echo

# ── headless-mode banner ──────────────────────────────────────────────────────
# Single-line summary so the user can confirm at a glance that GUI-killing
# is active before launching what's likely a multi-hour batch run.
if [[ "$HEADLESS" == true ]]; then
    if [[ "$MANAGE_SPLATSIM" == true ]]; then
        echo "Headless mode ON: SplatSim launch + step-1 SA wrapper + step-6 inline-eval sim all run with no GUI."
    else
        echo "Headless mode ON: step-1 SA wrapper GUI disabled. (--no_manage_splatsim → orchestrator does not"
        echo "                  set --env.headless on training; the user's external sim's GUI is unchanged.)"
    fi
    echo
fi

# ── resume detection ──────────────────────────────────────────────────────────
# For each round r, count completed steps (1..7). Step 7 (cleanup) is treated
# as auto-complete once step 6 succeeds, so we only probe steps 1..6.
declare -a ROUND_COMPLETED_STEPS  # parallel array indexed 1..N
ROUND_COMPLETED_STEPS[0]=0  # unused

# Per-round raw-blend completion check. Required for step 6 (the un-filtered
# policy training). Empty $BLENDS → vacuously true.
raw_blends_complete_for_round() {
    local r="$1"
    (( ${#BLENDS[@]} == 0 )) && return 0
    local R blend_repo blend_short
    for R in "${BLENDS[@]}"; do
        blend_repo="$(blend_repo_for_round "$r" "$R")"
        blend_short="$(blend_short_for_round "$r" "$R")"
        dataset_exists "$blend_repo" || return 1
        if [[ "$ACTION_FORMAT" == "rel" ]]; then
            stats_exists "$blend_short" || return 1
        fi
    done
    return 0
}

# Per-round `_nocoll`-sibling completion check. Required for step 6b (the
# collision-filtered sibling policy). Only meaningful when
# --filter_blend_collisions is on; empty $BLENDS → vacuously true.
nocoll_blends_complete_for_round() {
    local r="$1"
    (( ${#BLENDS[@]} == 0 )) && return 0
    local R nocoll_repo nocoll_short
    for R in "${BLENDS[@]}"; do
        nocoll_repo="$(nocoll_repo_for_round "$r" "$R")"
        nocoll_short="$(nocoll_short_for_round "$r" "$R")"
        dataset_exists "$nocoll_repo" || return 1
        if [[ "$ACTION_FORMAT" == "rel" ]]; then
            stats_exists "$nocoll_short" || return 1
        fi
    done
    return 0
}

# Per-round step-2 completion = raw blends always; nocoll siblings additionally
# when --filter_blend_collisions is on. The execution side (step 2 in the
# round loop) is idempotent: short-circuits if these already exist.
all_blends_complete_for_round() {
    local r="$1"
    raw_blends_complete_for_round "$r" || return 1
    if [[ "$FILTER_BLEND_COLLISIONS" == "true" ]]; then
        nocoll_blends_complete_for_round "$r" || return 1
    fi
    return 0
}

for r in $(seq 1 "$NUM_ROUNDS"); do
    int_short="$(int_short_for_round "$r")"
    int_repo="$(int_repo_for_round "$r")"
    alias_repo="$(alias_repo_for_round "$r")"
    merged_short="$(merged_short_for_round "$r")"
    merged_repo="$(merged_repo_for_round "$r")"
    ft_dir="$(train_output_dir_for_round "$r")"
    scratch_dir="$ft_dir"  # both modes write to the same path under the new naming

    step=0
    # Check training output FIRST. If the trained policy is on disk, the round
    # is complete regardless of intermediate artifacts (the merged dataset and
    # its stats sidecar may have been legitimately deleted by step 7 cleanup
    # in the next round). Without this short-circuit, a sequential counter
    # would mark a fully-completed round as PARTIAL whenever cleanup ran.
    #
    # Note on --filter_blend_collisions: blends + _nocoll siblings are NEVER
    # cleaned (step 7 only touches the merged dataset). When the flag is on,
    # step 6 still trains the un-filtered (raw blend) policy with its
    # existing name — the filter produces an ADDITIONAL sibling policy
    # in a new step (6b) named with a _nocoll suffix. That step has its
    # own completeness check; it doesn't affect step 6's resume detection.
    if dataset_exists "$int_repo" \
       && { training_exists "$ft_dir" || training_exists "$scratch_dir"; } \
       && all_blends_complete_for_round "$r"; then
        step=6
        # Step 6b — collision-filtered sibling policy. Only meaningful when
        # --filter_blend_collisions is on. If its training output doesn't
        # exist yet, demote to step=5 so the orchestrator re-enters this
        # round at step 6+. Step 6 will no-op (raw policy already at target
        # step, lerobot-train detects that), then step 6b will run and
        # produce the nocoll sibling.
        if [[ "$FILTER_BLEND_COLLISIONS" == "true" ]] && (( ${#BLENDS[@]} > 0 )); then
            if ! training_exists "$(nocoll_train_output_dir_for_round "$r")"; then
                step=5
            fi
        fi
    else
        # Step 1 = record + stats-on-int (folded). Requires BOTH the dataset
        # AND its rel-action stats sidecar (when ACTION_FORMAT=rel). When the
        # dataset exists but sidecar is missing, resume detection treats step 1
        # as "not done" so that step 1 execution runs — but the step 1 EXECUTION
        # block detects the existing dataset and just regenerates the sidecar
        # instead of destructively re-recording.
        if dataset_exists "$int_repo"; then
            if [[ "$ACTION_FORMAT" == "rel" ]]; then
                stats_exists "$int_short" && step=1
            else
                step=1
            fi
        fi
        # Step 2 = all blends produced + their stats sidecars (or empty $BLENDS).
        (( step == 1 )) && all_blends_complete_for_round "$r" && step=2
        if [[ "$SKIP_ALIAS_STEP" == true ]]; then
            # Step 3 is a no-op when skipped; auto-advance.
            (( step == 2 )) && step=3
        else
            (( step == 2 )) && dataset_exists "$alias_repo"   && step=3
        fi
        # Step 4 = cumulative merge (merge-mode) OR skipped (weighted-mode).
        # In weighted mode the per-source datasets ARE the training inputs;
        # there's no merged artifact on disk. Auto-advance once step 3 is done.
        if [[ "$USE_WEIGHTED_SAMPLING" == "true" ]]; then
            (( step == 3 )) && step=4
        else
            (( step == 3 )) && dataset_exists "$merged_repo" && step=4
        fi
        # Step 5 = stats on merged dataset (merge-mode) OR skipped (weighted-
        # mode; per-source stats sidecars are already on disk from step 1b/2b
        # checks above, which are prerequisites for step 4 advancing here).
        if [[ "$USE_WEIGHTED_SAMPLING" == "true" ]]; then
            (( step == 4 )) && step=5
        else
            (( step == 4 )) && stats_exists "$merged_short" && step=5
        fi
    fi
    ROUND_COMPLETED_STEPS[$r]=$step
done

# ── --extend_last_round: demote the highest fully-complete round to PARTIAL
# when the current --finetune_steps would have produced a higher target than
# what's saved. Only touches the LAST complete round; earlier rounds are
# preserved (extending them would require redoing the whole chain).
# Reads:
#   * R = highest round with ROUND_COMPLETED_STEPS[R] == 6
#   * R's starting step (= previous round's saved cfg.steps, or initial
#     policy's saved cfg.steps for R=1) so we can compute planned_target =
#     starting_step + FINETUNE_STEPS.
#   * R's currently-saved cfg.steps (from its checkpoints/last/.../train_config.json).
# If planned > saved → set ROUND_COMPLETED_STEPS[R]=5 and log.
# Helper that returns cfg.steps from a train_config.json (or empty if missing).
_steps_from_train_config() {
    local cfg_path="$1"
    [[ -f "$cfg_path" ]] || { echo ""; return; }
    python3 -c "
import json, sys
try:
    c = json.load(open(sys.argv[1]))
    print(int(c.get('steps', 0)))
except Exception:
    print('')
" "$cfg_path"
}
if [[ "$EXTEND_LAST_ROUND" == "true" ]]; then
    # Find the highest fully-complete round.
    LAST_COMPLETE_ROUND=0
    for r in $(seq 1 "$NUM_ROUNDS"); do
        (( ${ROUND_COMPLETED_STEPS[$r]} == 6 )) && LAST_COMPLETE_ROUND=$r
    done
    if (( LAST_COMPLETE_ROUND > 0 )); then
        _R=$LAST_COMPLETE_ROUND
        # Starting step for round R.
        if (( _R == 1 )); then
            _start_cfg="$INITIAL_POLICY_PATH/checkpoints/last/pretrained_model/train_config.json"
            # Fall back to other common layouts for --initial_policy_path.
            [[ -f "$_start_cfg" ]] || _start_cfg="$INITIAL_POLICY_PATH/pretrained_model/train_config.json"
            [[ -f "$_start_cfg" ]] || _start_cfg="$INITIAL_POLICY_PATH/train_config.json"
        else
            _prev_dir="$(train_output_dir_for_round "$((_R - 1))")"
            _start_cfg="$_prev_dir/checkpoints/last/pretrained_model/train_config.json"
        fi
        _R_dir="$(train_output_dir_for_round "$_R")"
        _R_saved_cfg="$_R_dir/checkpoints/last/pretrained_model/train_config.json"
        _start_steps="$(_steps_from_train_config "$_start_cfg")"
        _saved_steps="$(_steps_from_train_config "$_R_saved_cfg")"
        if [[ -n "$_start_steps" && -n "$_saved_steps" ]]; then
            _planned_target=$(( _start_steps + FINETUNE_STEPS ))
            if (( _planned_target > _saved_steps )); then
                echo "[extend_last_round] Round $_R complete at saved target=$_saved_steps (started from $_start_steps);"
                echo "  new --finetune_steps=$FINETUNE_STEPS implies planned target=$_planned_target."
                echo "  Demoting round $_R to step 6 so step 6 re-runs and resumes from existing checkpoint."
                ROUND_COMPLETED_STEPS[$_R]=5
            else
                echo "[extend_last_round] Round $_R complete at saved target=$_saved_steps; current --finetune_steps=$FINETUNE_STEPS"
                echo "  implies planned target=$_planned_target (≤ saved). No extension needed."
            fi
        else
            echo "[extend_last_round] WARNING: could not read steps from round $_R's saved config or the previous-round/initial config." >&2
            echo "  start_cfg=$_start_cfg ($_start_steps), saved_cfg=$_R_saved_cfg ($_saved_steps)" >&2
            echo "  Skipping extension; behaving as if --extend_last_round were unset." >&2
        fi
    fi
fi

# Find the furthest round/step combo.
FURTHEST_ROUND=0
FURTHEST_STEP=0
for r in $(seq 1 "$NUM_ROUNDS"); do
    s=${ROUND_COMPLETED_STEPS[$r]}
    if (( s > 0 )); then
        FURTHEST_ROUND=$r
        FURTHEST_STEP=$s
    fi
    # Stop scanning once we hit an incomplete round (later rounds depend on this one).
    (( s < 6 )) && break
done

# Probe the optional post-loop final-scratch step.
FINAL_SCRATCH_DONE=false
if do_final_scratch; then
    if training_exists "$(train_output_dir_final_scratch)"; then
        FINAL_SCRATCH_DONE=true
    fi
fi

# Print the step legend so the user understands what "5/6 steps complete"
# refers to in the detection table below.
echo "Steps per round:"
echo "  1. Record interventions (lerobot-eval --intervention.method=...) + sidecar rel-action stats"
if (( ${#BLENDS[@]} > 0 )); then
    echo "  2. Produce blended-intervention datasets at ratios ${BLENDS[*]} (augment_dataset_with_blending.py) + sidecar stats"
else
    echo "  2. (SKIPPED — --blends is empty) Blend intervention at configured ratios"
fi
if [[ "$SKIP_ALIAS_STEP" == true ]]; then
    echo "  3. (SKIPPED — --skip_alias_step) Hardlink-alias under _${MODEL}${ACTION_FORMAT}00 naming"
else
    echo "  3. Hardlink-alias intervention dataset under _${MODEL}${ACTION_FORMAT}00 naming (augment_ratios_sweep.sh ratio=0)"
fi
if [[ "$USE_WEIGHTED_SAMPLING" == "true" ]]; then
    echo "  4. (SKIPPED — --use_weighted_sampling) Cumulative merge"
    echo "  5. (SKIPPED — --use_weighted_sampling) Merged-dataset rel-action stats"
    echo "  6. Train policy ($INTERMEDIATE_MODE) on weighted union of {base + every round's intervention + every round's blends}"
    echo "     → per-source DataLoader weights: base=$(python3 -c "print(round(1-float('$DAGGER_DATA_FRACTION'), 4))"), DAgger=${DAGGER_DATA_FRACTION} split equally across DAgger sub-datasets"
    case "$NORM_MODE" in
        aggregated)
            echo "     → norm_mode=aggregated: policy normalizer/unnormalizer use min-of-mins / max-of-maxes / count-weighted mean+std over ALL sub-datasets."
            ;;
        base_only)
            echo "     → norm_mode=base_only: policy normalizer/unnormalizer use ONLY source[0] (base) stats; intervention rows may normalize OUTSIDE [-1,1]."
            ;;
    esac
else
    echo "  4. Cumulative merge: base + dag1[..dag${NUM_ROUNDS}] → training dataset"
    echo "  5. Compute sidecar relative-action stats for the merged dataset"
    echo "  6. Train policy ($INTERMEDIATE_MODE) on the merged dataset"
fi
if do_final_scratch; then
    echo "  + (post-loop) Extra from-scratch train on round $NUM_ROUNDS's merged data"
    echo "                → $(train_output_dir_final_scratch)"
fi
echo

echo "Resume detection:"
for r in $(seq 1 "$NUM_ROUNDS"); do
    s=${ROUND_COMPLETED_STEPS[$r]}
    if (( s == 6 )); then
        verdict="COMPLETE (dataset ✓, policy ✓)"
    elif (( s >= 1 )); then
        verdict="PARTIAL — will redo from step $((s + 1))"
    else
        verdict="not started"
    fi
    echo "  Round $r: $s/6 steps complete — $verdict"
done
if do_final_scratch; then
    if [[ "$FINAL_SCRATCH_DONE" == true ]]; then
        echo "  Final scratch: COMPLETE (policy ✓)"
    else
        echo "  Final scratch: not done"
    fi
fi
echo
echo "Note: a round is only considered fully complete when BOTH its intervention"
echo "      dataset (step 1, non-empty) AND its trained policy checkpoint (step 6,"
echo "      with model.safetensors) are present on disk."
echo

# Restart-from-scratch handler. Prompts the user to confirm before deleting
# every dag artifact on disk. Sets EFFECTIVE_START_ROUND=1, EFFECTIVE_START_STEP=1
# on success. Exits the script on abort.
restart_from_scratch() {
    RESTART_PATHS=()
    for r in $(seq 1 "$NUM_ROUNDS"); do
        # In rerun-blends mode, the intervention + alias + int-stats artifacts
        # belong to the SOURCE lineage (read-only). The new lineage we're
        # building owns the merged + training + blend artifacts only. Never
        # delete source artifacts on --force_restart.
        #
        # --preserve_round_1_intervention skips round 1's int/alias/int-stats
        # too, since those represent the (expensive) human-recorded intervention
        # that the user often wants to keep across "restart all finetuning from
        # scratch" runs. Rounds 2..N still get cleaned because their
        # intervention recordings depend on (and would be inconsistent with)
        # the new fresh-policy chain.
        _skip_int_for_this_round=false
        if [[ "$RERUN_MODE_ENABLED" == "true" ]]; then
            _skip_int_for_this_round=true
        elif [[ "$PRESERVE_ROUND_1_INTERVENTION" == "true" && "$r" == "1" ]]; then
            _skip_int_for_this_round=true
        fi
        if [[ "$_skip_int_for_this_round" != "true" ]]; then
            RESTART_PATHS+=( "$LEROBOT_CACHE/$(int_repo_for_round "$r")" )
            RESTART_PATHS+=( "$LEROBOT_CACHE/$(alias_repo_for_round "$r")" )
            RESTART_PATHS+=( "$STATS_BASE/$(int_short_for_round "$r")" )
        fi
        RESTART_PATHS+=( "$LEROBOT_CACHE/$(merged_repo_for_round "$r")" )
        RESTART_PATHS+=( "$STATS_BASE/$(merged_short_for_round "$r")" )
        RESTART_PATHS+=( "$(train_output_dir_for_round "$r")" )
        # Collision-filtered sibling policy dir (step 6b). rm -rf is
        # idempotent on missing paths, so safe to include even for
        # lineages built without --filter_blend_collisions. Use the
        # _DIR_ variant (full path), not _RUN_NAME (basename only).
        RESTART_PATHS+=( "$(nocoll_train_output_dir_for_round "$r")" )
        # Blended-intervention datasets + their rel-action stats sidecars,
        # one per ratio in $BLENDS. No-op when $BLENDS is empty.
        # In rerun mode blends are cross-rerun-cacheable (see comment on
        # ALSO_DELETE_BLENDS); preserve unless the user opted in. In
        # non-rerun mode the lineage owns its blends → always delete.
        if [[ "$RERUN_MODE_ENABLED" != "true" || "$ALSO_DELETE_BLENDS" == "true" ]]; then
            for R in "${BLENDS[@]}"; do
                RESTART_PATHS+=( "$LEROBOT_CACHE/$(blend_repo_for_round "$r" "$R")" )
                RESTART_PATHS+=( "$STATS_BASE/$(blend_short_for_round "$r" "$R")" )
                # Collision-filtered siblings (always include in the cleanup
                # list; the underlying rm is idempotent on missing paths, so
                # this is safe whether the user ran with the flag or not).
                RESTART_PATHS+=( "$LEROBOT_CACHE/$(nocoll_repo_for_round "$r" "$R")" )
                RESTART_PATHS+=( "$STATS_BASE/$(nocoll_short_for_round "$r" "$R")" )
            done
        fi
        # Intervention output dir lives INSIDE the training dir
        # (<train>/dagger/interventions/), so the rm above already takes it
        # out. Legacy paths from older orchestrator layouts kept here so
        # --force_restart still cleans pre-migration runs:
        #   outputs/dagger/<train_basename>/   (mirror layout, briefly used)
        #   outputs/dagger/round_${r}/          (original layout)
        RESTART_PATHS+=( "$LEROBOT_ROOT/outputs/dagger/$(basename "$(train_output_dir_for_round "$r")")" )
        RESTART_PATHS+=( "$LEROBOT_ROOT/outputs/dagger/round_${r}" )
    done
    do_final_scratch && RESTART_PATHS+=( "$(train_output_dir_final_scratch)" )
    EXISTING_PATHS=()
    for p in "${RESTART_PATHS[@]}"; do
        [[ -e "$p" ]] && EXISTING_PATHS+=( "$p" )
    done
    if (( ${#EXISTING_PATHS[@]} == 0 )); then
        echo "No dag artifacts to delete; nothing to do."
    else
        echo "The following dag artifacts will be deleted (rm -rf):"
        for p in "${EXISTING_PATHS[@]}"; do
            echo "  $p"
        done
        echo
    fi
    echo -n "Type 'restart' to confirm: "
    read -r CONFIRM
    [[ "$CONFIRM" == "restart" ]] || { echo "Aborted."; exit 1; }
    echo "Clearing prior dag artifacts..."
    for p in "${EXISTING_PATHS[@]}"; do
        run_or_echo rm -rf "$p"
    done
    EFFECTIVE_START_ROUND=1
    EFFECTIVE_START_STEP=1
}

# Decide start_round. EFFECTIVE_START_ROUND > NUM_ROUNDS means "skip the main
# dag-round loop entirely and only run the post-loop final-scratch step."
ALL_ROUNDS_DONE=false
(( FURTHEST_ROUND == NUM_ROUNDS && FURTHEST_STEP == 6 )) && ALL_ROUNDS_DONE=true

if [[ -n "$RETRAIN_ROUND" ]]; then
    # Retrain mode: jump straight to round $RETRAIN_ROUND's training step,
    # skipping all earlier rounds and the data steps (1-5) for that round.
    # Auto-pick the entry step based on what data is on disk:
    #   - if the merged dataset (+ stats sidecar in rel mode) exists, start step 6.
    #   - else if all per-round intervention datasets 1..N exist, start step 4
    #     (re-merge from existing intervention datasets, recompute stats, train).
    #   - else error out — we can't reconstitute the data.
    EFFECTIVE_START_ROUND="$RETRAIN_ROUND"
    rt_merged_repo="$(merged_repo_for_round "$RETRAIN_ROUND")"
    rt_merged_short="$(merged_short_for_round "$RETRAIN_ROUND")"
    rt_have_merged=true
    if ! dataset_exists "$rt_merged_repo"; then
        rt_have_merged=false
    elif [[ "$ACTION_FORMAT" == "rel" ]] && ! stats_exists "$rt_merged_short"; then
        rt_have_merged=false
    fi
    if [[ "$rt_have_merged" == true ]]; then
        EFFECTIVE_START_STEP=6
        echo "Retrain mode: round $RETRAIN_ROUND, merged dataset present → running step 6 only."
    else
        # Need to re-merge. step 4 reads from either the alias datasets (default)
        # or the raw intervention datasets (--skip_alias_step). All rounds
        # 1..$RETRAIN_ROUND must have those sources available, otherwise the
        # merge will fail mid-flight. Validate up front.
        for prev in $(seq 1 "$RETRAIN_ROUND"); do
            if [[ "$SKIP_ALIAS_STEP" == true ]]; then
                rt_need_repo="$(int_repo_for_round "$prev")"
            else
                rt_need_repo="$(alias_repo_for_round "$prev")"
            fi
            if ! dataset_exists "$rt_need_repo"; then
                echo "ERROR: --retrain_round=$RETRAIN_ROUND with missing merged dataset needs" >&2
                echo "  $rt_need_repo (round $prev's merge source) on disk, but it is missing." >&2
                echo "  Re-run the orchestrator normally (without --retrain_round) to" >&2
                echo "  regenerate the data pipeline before retrying." >&2
                exit 1
            fi
            # Same check for every blend ratio. If any blend is missing, the
            # merge will fail mid-flight; tell the user to re-run normally
            # (which re-blends that round) before retrying.
            for R in "${BLENDS[@]}"; do
                rt_blend_repo="$(blend_repo_for_round "$prev" "$R")"
                if ! dataset_exists "$rt_blend_repo"; then
                    echo "ERROR: --retrain_round=$RETRAIN_ROUND with missing merged dataset needs" >&2
                    echo "  $rt_blend_repo (round $prev's blend at ratio=$R) on disk, but it is missing." >&2
                    echo "  Either remove --blends to skip blending, or re-run the orchestrator" >&2
                    echo "  normally (without --retrain_round) to regenerate the data pipeline." >&2
                    exit 1
                fi
            done
        done
        EFFECTIVE_START_STEP=4
        echo "Retrain mode: round $RETRAIN_ROUND, merged dataset missing → running steps 4-6."
    fi
    echo "  Retrain output dir: $(train_output_dir_for_round "$RETRAIN_ROUND")"
    echo "  (post-loop final-scratch phase skipped in retrain mode)"
elif [[ -n "$START_ROUND" ]]; then
    # Explicit override.
    if ! [[ "$START_ROUND" =~ ^[0-9]+$ ]] || (( START_ROUND < 1 || START_ROUND > NUM_ROUNDS )); then
        echo "ERROR: --start_round=$START_ROUND must be between 1 and $NUM_ROUNDS" >&2; exit 1
    fi
    EFFECTIVE_START_ROUND=$START_ROUND
    EFFECTIVE_START_STEP=1
    echo "Using explicit --start_round=$START_ROUND (begins at step 1)."
elif [[ "$FORCE_RESTART" == true ]]; then
    EFFECTIVE_START_ROUND=1
    EFFECTIVE_START_STEP=1
    if [[ "$DRY_RUN" != true ]]; then
        if [[ "$RERUN_MODE_ENABLED" == "true" ]]; then
            echo "--force_restart (rerun mode): this will rm -rf the NEW lineage's"
            if [[ "$ALSO_DELETE_BLENDS" == "true" ]]; then
                echo "  merged datasets + training dirs + blend datasets for dag1..dag${NUM_ROUNDS}."
                echo "  (--also_delete_blends set; blend datasets will be deleted too)"
            else
                echo "  merged datasets + training dirs for dag1..dag${NUM_ROUNDS}."
                echo "  Blend datasets (<src_int>_dagN_blendXXX) will be PRESERVED for cross-rerun"
                echo "  cache reuse — pass --also_delete_blends to delete them too."
            fi
            echo "  SOURCE intervention/alias/int-stats artifacts will be PRESERVED."
        else
            echo "--force_restart: this will rm -rf all dag1..dag${NUM_ROUNDS} datasets,"
            echo "  their stats sidecars, and per-round training output dirs."
            if [[ "$PRESERVE_ROUND_1_INTERVENTION" == "true" ]]; then
                echo "  --preserve_round_1_intervention set: round 1's raw intervention dataset,"
                echo "  alias, and int-stats sidecar will be PRESERVED (everything else is wiped)."
            fi
        fi
        echo -n "Type 'restart' to confirm: "
        read -r CONFIRM
        [[ "$CONFIRM" == "restart" ]] || { echo "Aborted."; exit 1; }
    fi
    echo "--force_restart: clearing prior dag artifacts..."
    for r in $(seq 1 "$NUM_ROUNDS"); do
        # In rerun-blends mode, source intervention/alias/int-stats artifacts
        # are read-only and MUST be preserved. Only the new lineage's merged
        # datasets + training dirs + blends get nuked. Mirrors the gate in
        # restart_from_scratch() above.
        #
        # --preserve_round_1_intervention extends the same skip to round 1
        # only — the human-recorded intervention dataset is preserved so the
        # lineage restarts from "round 1 step 4 (merge)" instead of replaying
        # step 1's expensive lerobot-eval recording.
        _skip_int_for_this_round=false
        if [[ "$RERUN_MODE_ENABLED" == "true" ]]; then
            _skip_int_for_this_round=true
        elif [[ "$PRESERVE_ROUND_1_INTERVENTION" == "true" && "$r" == "1" ]]; then
            _skip_int_for_this_round=true
        fi
        if [[ "$_skip_int_for_this_round" != "true" ]]; then
            run_or_echo rm -rf "$LEROBOT_CACHE/$(int_repo_for_round "$r")"
            run_or_echo rm -rf "$LEROBOT_CACHE/$(alias_repo_for_round "$r")"
            run_or_echo rm -rf "$STATS_BASE/$(int_short_for_round "$r")"
        fi
        run_or_echo rm -rf "$LEROBOT_CACHE/$(merged_repo_for_round "$r")"
        run_or_echo rm -rf "$STATS_BASE/$(merged_short_for_round "$r")"
        # Blended-intervention datasets + their rel-action stats sidecars
        # (one per ratio in $BLENDS). No-op when $BLENDS is empty.
        # Same gating as the RESTART_PATHS block above: in rerun mode,
        # blends are cross-rerun-cacheable, so preserve them unless the
        # user opted in via --also_delete_blends. In non-rerun mode the
        # lineage owns its blends → always delete.
        if [[ "$RERUN_MODE_ENABLED" != "true" || "$ALSO_DELETE_BLENDS" == "true" ]]; then
            for R in "${BLENDS[@]}"; do
                run_or_echo rm -rf "$LEROBOT_CACHE/$(blend_repo_for_round "$r" "$R")"
                run_or_echo rm -rf "$STATS_BASE/$(blend_short_for_round "$r" "$R")"
                # Collision-filtered siblings (always attempt; rm -rf is
                # idempotent on missing paths so this is safe regardless of
                # whether --filter_blend_collisions was used).
                run_or_echo rm -rf "$LEROBOT_CACHE/$(nocoll_repo_for_round "$r" "$R")"
                run_or_echo rm -rf "$STATS_BASE/$(nocoll_short_for_round "$r" "$R")"
            done
            # Glob-based sweep of orphaned blends. The explicit BLENDS loop
            # above only catches blends THIS lineage was launched with —
            # but blends use SOURCE'S naming (per CLAUDE.md, deterministic
            # cross-rerun cache reuse), so when cleaning up a SOURCE
            # lineage, BLENDS is empty (source never used them) and the
            # orphan blends created by rerun lineages survive. Without
            # this sweep, source cleanup leaves 40+ orphaned `_blend*` and
            # `_blend*_nocoll` datasets that get reused as stale cache by
            # the next rerun. Bash globbing wrapped in `|| true` to keep
            # `set -e` happy when no files match.
            # Cache path uses the full repo id (HF_USER/<int_short>); stats
            # uses just the short. Mirror the explicit-loop paths above so
            # we glob the right directories.
            _BLEND_GLOB_BASE="$LEROBOT_CACHE/$(int_repo_for_round "$r")_blend"
            for orphan in "$_BLEND_GLOB_BASE"*; do
                [[ -e "$orphan" ]] || continue
                run_or_echo rm -rf "$orphan"
            done
            _BLEND_STATS_GLOB_BASE="$STATS_BASE/$(int_short_for_round "$r")_blend"
            for orphan in "$_BLEND_STATS_GLOB_BASE"*; do
                [[ -e "$orphan" ]] || continue
                run_or_echo rm -rf "$orphan"
            done
        fi
        # New layout nests interventions under the training dir
        # (<train>/dagger/...), so this rm already takes them out.
        run_or_echo rm -rf "$(train_output_dir_for_round "$r")"
        # Legacy intervention layouts kept here for back-compat:
        run_or_echo rm -rf "$LEROBOT_ROOT/outputs/dagger/$(basename "$(train_output_dir_for_round "$r")")"
        run_or_echo rm -rf "$LEROBOT_ROOT/outputs/dagger/round_${r}"
        # Collision-filtered sibling policy dir (step 6b, --filter_blend_collisions).
        # Always attempted; rm -rf is a no-op when the dir doesn't exist, so
        # this is safe for lineages built without --filter_blend_collisions.
        # Without this, full cleanup leaves orphan _nc training dirs that
        # the user then has to clean up via a second --nc_only pass. Use
        # the _DIR_ variant (full path); _RUN_NAME returns basename only.
        _NC_TRAIN_DIR="$(nocoll_train_output_dir_for_round "$r")"
        run_or_echo rm -rf "$_NC_TRAIN_DIR"
        run_or_echo rm -rf "$LEROBOT_ROOT/outputs/dagger/$(basename "$_NC_TRAIN_DIR")"
    done
    do_final_scratch && run_or_echo rm -rf "$(train_output_dir_final_scratch)"
    if [[ "$CLEANUP_ONLY" == true ]]; then
        echo "--cleanup_only: deletion complete; exiting without starting a new run."
        exit 0
    fi
elif (( FURTHEST_ROUND == 0 )); then
    # No prior work detected.
    EFFECTIVE_START_ROUND=1
    EFFECTIVE_START_STEP=1
    echo "No prior DAgger artifacts found; starting from round 1, step 1."
elif [[ "$ALL_ROUNDS_DONE" == true ]] && { ! do_final_scratch || [[ "$FINAL_SCRATCH_DONE" == true ]]; }; then
    # Everything is done — either no final-scratch was requested, or it
    # already exists. Offer restart-from-scratch as the only do-something path.
    if do_final_scratch; then
        MSG="all $NUM_ROUNDS dag rounds AND the post-loop final-scratch step are complete"
    else
        MSG="all $NUM_ROUNDS dag rounds complete (no final-scratch requested with --final_mode=$FINAL_MODE)"
    fi
    echo "Pipeline detected: $MSG."
    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY-RUN] Would exit without running anything."
        exit 0
    fi
    if [[ "$RESUME" == true ]]; then
        # Nothing to resume — already fully complete. Exit cleanly so sweep
        # wrappers count this iteration as success and move on.
        echo "--resume: nothing to do (already complete). Exiting 0."
        exit 0
    fi
    echo -n "Re-run anyway? [n/restart-from-scratch] "
    read -r RESP
    case "$RESP" in
        restart-from-scratch|restart) restart_from_scratch ;;
        *) echo "Aborted."; exit 0 ;;
    esac
else
    # Auto-detected partial progress. Compute (next_r, next_s, msg) describing
    # the immediate next step. Cases:
    #   (a) all rounds done, final-scratch pending → NEXT_R=NUM_ROUNDS+1 (skip loop)
    #   (b) furthest round fully trained → start next round at step 1
    #   (c) furthest round mid-flight → resume at step (furthest_step+1)
    if [[ "$ALL_ROUNDS_DONE" == true ]]; then
        # do_final_scratch && !FINAL_SCRATCH_DONE (the other branches above caught the rest).
        NEXT_R=$((NUM_ROUNDS + 1))
        NEXT_S=1
        MSG="all $NUM_ROUNDS dag rounds complete; next is the post-loop final from-scratch train"
    elif (( FURTHEST_STEP == 6 )); then
        NEXT_R=$((FURTHEST_ROUND + 1))
        NEXT_S=1
        MSG="round $FURTHEST_ROUND fully complete; next is round $NEXT_R, step 1"
    else
        NEXT_R=$FURTHEST_ROUND
        NEXT_S=$((FURTHEST_STEP + 1))
        MSG="round $FURTHEST_ROUND has $FURTHEST_STEP/6 steps complete; next is round $FURTHEST_ROUND, step $NEXT_S"
    fi

    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY-RUN] Detected: $MSG. Would resume there."
        EFFECTIVE_START_ROUND=$NEXT_R
        EFFECTIVE_START_STEP=$NEXT_S
    elif [[ "$RESUME" == true ]]; then
        echo "Pipeline detected: $MSG"
        echo "--resume: auto-confirming. Resuming there."
        EFFECTIVE_START_ROUND=$NEXT_R
        EFFECTIVE_START_STEP=$NEXT_S
    else
        echo "Pipeline detected: $MSG"
        echo -n "Resume from there? [Y/n/restart-from-scratch] "
        read -r RESP
        case "$RESP" in
            ""|y|Y|yes)
                EFFECTIVE_START_ROUND=$NEXT_R
                EFFECTIVE_START_STEP=$NEXT_S
                ;;
            n|N|no)               echo "Aborted."; exit 0 ;;
            restart-from-scratch|restart) restart_from_scratch ;;
            *) echo "Unrecognized response. Aborting." >&2; exit 1 ;;
        esac
    fi
fi

# Stale-merged-dataset sweep. The orchestrator's step 7 cleans up round (r-1)'s
# merged dataset after round r's training, leaving at most one merged dataset on
# disk at a time. But if a prior run crashed (e.g. SIGABRT from PyBullet/Tcl
# teardown, or the user Ctrl-C'd between steps 6 and 7), step 7 may never have
# run and an older merged_m dir is now an orphan. Sweep them on every resume.
#
# Rules:
#   - Round R is "the round we're entering". Its merged dataset (dag{R}_m) is
#     potentially needed when starting at step >= 5; we KEEP it.
#   - When EFFECTIVE_START_ROUND > NUM_ROUNDS (i.e. entering the post-loop
#     final-scratch step), we KEEP round NUM_ROUNDS's merged dataset since the
#     final-scratch step trains on it.
#   - Everything else is stale and removed.
KEEP_MERGED_ROUND="$EFFECTIVE_START_ROUND"
(( KEEP_MERGED_ROUND > NUM_ROUNDS )) && KEEP_MERGED_ROUND="$NUM_ROUNDS"
echo "Stale-merged-dataset sweep (keeping at most round ${KEEP_MERGED_ROUND}'s merged dataset)..."
stale_found=0
for stale_r in $(seq 1 "$NUM_ROUNDS"); do
    if (( stale_r == KEEP_MERGED_ROUND )); then continue; fi
    stale_merged_repo="$(merged_repo_for_round "$stale_r")"
    stale_merged_short="$(merged_short_for_round "$stale_r")"
    stale_dataset_path="$LEROBOT_CACHE/$stale_merged_repo"
    stale_stats_path="$STATS_BASE/$stale_merged_short"
    found_this_round=0
    if [[ -e "$stale_dataset_path" ]]; then
        echo "  Removing stale merged dataset: $stale_dataset_path"
        run_or_echo rm -rf "$stale_dataset_path"
        found_this_round=1
    fi
    if [[ -e "$stale_stats_path" ]]; then
        echo "  Removing stale merged stats:   $stale_stats_path"
        run_or_echo rm -rf "$stale_stats_path"
        found_this_round=1
    fi
    (( found_this_round )) && stale_found=1
done
(( stale_found )) || echo "  (none found)"

# If we're starting mid-pipeline at a dag round > 1 OR entering only the
# post-loop final-scratch phase (EFFECTIVE_START_ROUND > NUM_ROUNDS), the
# previous round's training output must exist so we can pull current_policy
# from it. Round 0 is optional (run train_sweep.sh) only when
# EFFECTIVE_START_ROUND == 1.
#
# CURRENT_POLICY isn't consumed by the final-scratch step itself (it trains
# from scratch), but it's reported in the final summary as "last finetune-round
# policy" — resolve it from round NUM_ROUNDS in that case so the summary is
# accurate when the orchestrator re-enters only to do the final-scratch step.
if (( EFFECTIVE_START_ROUND > 1 )); then
    PREV_R=$((EFFECTIVE_START_ROUND - 1))
    (( PREV_R > NUM_ROUNDS )) && PREV_R="$NUM_ROUNDS"
    PREV_FT="$(train_output_dir_for_round "$PREV_R")"
    PREV_SCRATCH="$PREV_FT"
    if training_exists "$PREV_FT"; then
        CURRENT_POLICY="$(resolve_latest_checkpoint "$PREV_FT")"
    else
        echo "ERROR: starting at round $EFFECTIVE_START_ROUND requires a trained policy from round $PREV_R," >&2
        echo "  but $PREV_FT doesn't exist." >&2
        exit 1
    fi
    if (( EFFECTIVE_START_ROUND > NUM_ROUNDS )); then
        echo "Post-loop final-scratch phase; last finetune-round policy: $CURRENT_POLICY"
    else
        echo "Round $EFFECTIVE_START_ROUND will resume from policy: $CURRENT_POLICY"
    fi
fi

# ── shared training helpers (used by Round 0, the per-round loop, and the
# post-loop final-scratch phase) ──────────────────────────────────────────────
# When the sim is orchestrator-managed, omit --env.external_port from training
# so lerobot-train falls through to its own in-process env factory. In
# --no_manage_splatsim mode, forward the port so it shares the user's sim.
TRAIN_EXT_PORT_SWEEP=()
TRAIN_EXT_PORT_RESUME=()
if [[ "$MANAGE_SPLATSIM" == false ]]; then
    TRAIN_EXT_PORT_SWEEP=( --env_external_port="$ENV_EXTERNAL_PORT" )
    TRAIN_EXT_PORT_RESUME=( --env.external_port="$ENV_EXTERNAL_PORT" )
fi
# --headless propagation into training (both scratch via train_sweep.sh and
# finetune via resume_training.sh). Two surfaces to gate:
#   * --env.headless=true → SplatSimEnv injects headless into its in-process
#     PybulletRobotServerBase, so the inline-eval sim connects via p.DIRECT.
#     Only meaningful in managed-splatsim mode (when MANAGE_SPLATSIM=false the
#     env uses external_port and never spawns a local pybullet client, so
#     the env-side toggle is a no-op — but suppressing it keeps the dry-run
#     output clean and the user's external sim's GUI choice unaffected).
#   * --policy.shared_autonomy_config.show_slider=false → defensive in case
#     the resumed policy's saved train_config.json has SA wrapper enabled.
#     Most finetune policies don't, but a few (e.g. those that resume from
#     intervention-recording dirs) carry it forward. The wrapper's own
#     fallback path tolerates the field even when SA is disabled (the cfg
#     dataclass field always exists), so this is safe to add unconditionally.
# Built ONCE here so the same array can be appended to all three training
# invocations (scratch, finetune, post-loop final-scratch).
HEADLESS_TRAIN_ARGS=()
if [[ "$HEADLESS" == true ]]; then
    HEADLESS_TRAIN_ARGS=( --policy.shared_autonomy_config.show_slider=false )
    if [[ "$MANAGE_SPLATSIM" == true ]]; then
        HEADLESS_TRAIN_ARGS+=( --env.headless=true )
    fi
fi
# Scratch-mode training goes through train_sweep.sh, which has its own
# strict arg parser. We added a `--headless` knob there that injects the
# same two SHARED_ARGS as the finetune path (--env.headless=true when
# splatsim is managed + --policy.shared_autonomy_config.show_slider=false
# defensively). One flag at the boundary, same downstream effect.
HEADLESS_TRAIN_SCRATCH_ARGS=()
[[ "$HEADLESS" == true ]] && HEADLESS_TRAIN_SCRATCH_ARGS=( --headless )
# Offline-mode flag: when --push_to_hub is NOT set on the orchestrator (the
# default), disable the trained policy's hub push.
OFFLINE_POLICY_ARG=()
if [[ "$PUSH_TO_HUB" == false ]]; then
    OFFLINE_POLICY_ARG+=( "--policy.push_to_hub=false" )
fi
# Runner that tolerates the well-known PyBullet/Tcl teardown abort.
# lerobot-train completes training and eval, saves the checkpoint, then
# SIGABRTs in env.close() from "Tcl_AsyncDelete: async handler deleted by
# the wrong thread". Exit code is 134 (SIGABRT) but the checkpoint is
# already on disk and complete. We verify training_exists() (which checks
# model.safetensors AND that the last-checkpoint step matches the planned
# target) and treat the run as successful when that's the case. Reads
# TRAIN_OUTPUT_DIR from the caller's scope.
run_training_step() {
    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY-RUN] $*"
        return 0
    fi
    echo "+ $*"
    local rc=0
    "$@" || rc=$?
    if (( rc == 0 )); then
        return 0
    fi
    if training_exists "$TRAIN_OUTPUT_DIR"; then
        echo "WARNING: training subprocess exited with status $rc, but" >&2
        echo "  $TRAIN_OUTPUT_DIR/checkpoints/last/pretrained_model" >&2
        echo "  is present with the expected step count — treating as success" >&2
        echo "  (likely a benign PyBullet/Tcl teardown abort)." >&2
        return 0
    fi
    echo "ERROR: training failed (exit $rc) and expected checkpoint at" >&2
    echo "  $TRAIN_OUTPUT_DIR/checkpoints/last/pretrained_model" >&2
    echo "  is missing or incomplete." >&2
    return "$rc"
}
# train_sweep.sh invokes lerobot-train without --resume; lerobot-train errors
# on any pre-existing output dir. An earlier orchestrator invocation (or an
# earlier version of this script that pre-mkdir'd the dagger/ sidecar subdir
# before training) may have left a partial output dir with nothing but
# `dagger/` inside. Detect that exact pattern and rm -rf it so the upcoming
# train_sweep.sh call doesn't trip the FileExistsError check. Anything else
# (checkpoints/, wandb/, config files, ...) means real training output is
# present — left alone.
cleanup_pre_train_partial() {
    local dir="$1"
    [[ -d "$dir" ]] || return 0
    local entries
    entries=$(ls -A "$dir" 2>/dev/null)
    if [[ "$entries" == "dagger" ]]; then
        echo "[cleanup] removing leftover partial dir $dir (only dagger/ subdir present — from a prior failed attempt)" >&2
        run_or_echo rm -rf "$dir"
    fi
}
# Print GPU state before training. Useful when finetune OOMs and you want
# to identify the actual memory consumers.
print_gpu_state() {
    if [[ "$DRY_RUN" != true ]] && command -v nvidia-smi >/dev/null 2>&1; then
        echo "GPU state before training:"
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1 \
            | sed 's/^/  /'
        nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader 2>&1 \
            | sed 's/^/  total: /'
        echo
    fi
}

# ── Round 0: train base if needed; always resolve CURRENT_POLICY for round 1 ──
# Split into two responsibilities:
#   (a) resolve CURRENT_POLICY for round 1's input (always needed when round 1
#       is in the loop, even if resuming mid-round means step 1 won't actually
#       run — the round header still prints it).
#   (b) actually invoke train_sweep.sh on the base dataset (only when starting
#       from step 1 fresh AND no prior base training exists AND no
#       --initial_policy_path was given).
if (( EFFECTIVE_START_ROUND == 1 )); then
    BASE_TRAINING_DIR="$LEROBOT_ROOT/outputs/training/${TRAIN_OUTPUT_MODEL_PREFIX}_${BASE_SHORT}_${TRAIN_OUTPUT_ACTION_TAG}_basewrist"

    if [[ -n "$INITIAL_POLICY_PATH" ]]; then
        # Resolve the user-supplied path to a pretrained_model dir.
        if [[ -d "$INITIAL_POLICY_PATH/checkpoints/last/pretrained_model" ]]; then
            CURRENT_POLICY="$INITIAL_POLICY_PATH/checkpoints/last/pretrained_model"
        elif [[ -d "$INITIAL_POLICY_PATH/pretrained_model" ]]; then
            CURRENT_POLICY="$INITIAL_POLICY_PATH/pretrained_model"
        elif [[ -d "$INITIAL_POLICY_PATH" && -f "$INITIAL_POLICY_PATH/train_config.json" ]]; then
            CURRENT_POLICY="$INITIAL_POLICY_PATH"
        else
            CURRENT_POLICY="$(resolve_latest_checkpoint "$INITIAL_POLICY_PATH")"
        fi
        echo "Round 0: skipped (using --initial_policy_path=$CURRENT_POLICY)."
    elif training_exists "$BASE_TRAINING_DIR"; then
        CURRENT_POLICY="$(resolve_latest_checkpoint "$BASE_TRAINING_DIR")"
        echo "Round 0: skipped (found existing base training at $BASE_TRAINING_DIR)."
    elif (( EFFECTIVE_START_STEP == 1 )); then
        # Starting fresh and no prior base — train it now. The orchestrator
        # has not started its managed sim yet (start_sim happens inside the
        # per-round loop), so in managed mode we omit --env_external_port and
        # let lerobot-train spawn its own in-process sim.
        echo "Round 0: training base policy on $BASE_REPO..."
        # train_sweep.sh pre-flights stats_rel{8,50}.json when training a
        # relative-action policy. Generate them up-front if missing so the
        # base-train step is self-contained. Skip in abs mode (--no_relative
        # below disables the pre-flight).
        if [[ "$ACTION_FORMAT" == "rel" ]] && ! stats_exists "$BASE_DATASET_SHORT"; then
            echo "Round 0: computing sidecar rel-action stats for base ($BASE_DATASET_SHORT)..."
            run_or_echo bash "$SCRIPT_DIR/compute_relative_stats.sh" --dataset_repo="$BASE_REPO"
        fi
        ROUND0_ABS_ACTION_ARG=()
        if [[ "$ACTION_FORMAT" == "abs" ]]; then
            ROUND0_ABS_ACTION_ARG=( --no_relative )
        fi
        # Use run_training_step (not run_or_echo) so the benign PyBullet/Tcl
        # teardown SIGABRT at the end of lerobot-train is tolerated the same
        # way it is in the per-round and post-loop training steps. Reads
        # TRAIN_OUTPUT_DIR from this scope.
        TRAIN_OUTPUT_DIR="$BASE_TRAINING_DIR"
        cleanup_pre_train_partial "$TRAIN_OUTPUT_DIR"
        run_training_step bash "$SCRIPT_DIR/train_sweep.sh" \
            --dataset_repo="$BASE_REPO" \
            --model="$TRAIN_OUTPUT_MODEL_PREFIX" \
            "${ROUND0_ABS_ACTION_ARG[@]}" \
            "${TRAIN_EXT_PORT_SWEEP[@]}" \
            "${HEADLESS_TRAIN_SCRATCH_ARGS[@]}" \
            "${OFFLINE_POLICY_ARG[@]}"
        if [[ "$DRY_RUN" != true ]]; then
            CURRENT_POLICY="$(resolve_latest_checkpoint "$BASE_TRAINING_DIR")"
        else
            CURRENT_POLICY="$BASE_TRAINING_DIR/checkpoints/last/pretrained_model"
        fi
    else
        # Resuming mid-round-1 with no initial_policy_path AND no base training
        # dir. We have no way to source a round-1 input policy. Abort.
        echo "ERROR: resuming at round 1, step $EFFECTIVE_START_STEP, but cannot resolve" >&2
        echo "  a round-1 input policy. Either:" >&2
        echo "    - pass --initial_policy_path=<existing checkpoint>, or" >&2
        echo "    - run from step 1 so train_sweep.sh trains the base policy." >&2
        exit 1
    fi
fi

# ── per-round loop ────────────────────────────────────────────────────────────
# (training helpers — TRAIN_EXT_PORT_*, OFFLINE_POLICY_ARG, run_training_step,
#  print_gpu_state — were hoisted above Round 0 so they're available to all
#  three training phases.)
#
# In retrain mode the loop runs ONLY round $RETRAIN_ROUND (chain breaks at the
# suffixed output dir — subsequent rounds would be ambiguous about which
# checkpoint to source as input). Otherwise it runs through round $NUM_ROUNDS.
EFFECTIVE_END_ROUND="$NUM_ROUNDS"
[[ -n "$RETRAIN_ROUND" ]] && EFFECTIVE_END_ROUND="$RETRAIN_ROUND"

# Track which subset the long-running SplatSim is currently configured for.
# Initialized ONCE here (before the round loop) so per-round comparisons
# in the round body correctly detect "subset changed across rounds" instead
# of re-resetting to empty every round (which would force a useless restart
# every time the user happens to want the same failed subset two rounds
# in a row). Updated at the actual start_sim call site (search
# EVAL_BENCHMARK_SUBSET_FOR_SIM below).
EVAL_BENCHMARK_SUBSET_CURRENT_SIM=""

for r in $(seq "$EFFECTIVE_START_ROUND" "$EFFECTIVE_END_ROUND"); do
    INT_SHORT="$(int_short_for_round "$r")"
    INT_REPO="$(int_repo_for_round "$r")"
    ALIAS_REPO="$(alias_repo_for_round "$r")"
    MERGED_SHORT="$(merged_short_for_round "$r")"
    MERGED_REPO="$(merged_repo_for_round "$r")"
    MERGED_STATS_DIR="$STATS_BASE/$MERGED_SHORT"

    # Every dag round uses INTERMEDIATE_MODE. FINAL_MODE controls a separate
    # post-loop phase below — not a round.
    MODE="$INTERMEDIATE_MODE"
    TRAIN_OUTPUT_DIR="$(train_output_dir_for_round "$r")"

    # Rerun-blends mode: at each round, OVERRIDE the carried-forward
    # CURRENT_POLICY with SOURCE's _ft_dag(r-1) (or --initial_policy_path at
    # r=1). This is the "branching from source at each round" semantic — each
    # rerun round is an independent "what would have happened at round r if
    # we'd had these blends starting from source's _ft_dag(r-1)" experiment,
    # NOT a continuation of the rerun's own previous round. The carried
    # CURRENT_POLICY from end-of-round is ignored in rerun mode.
    if [[ "$RERUN_MODE_ENABLED" == "true" ]]; then
        if (( r == 1 )); then
            # Round 1 starts from --initial_policy_path, same as the
            # non-rerun's round 1. We don't bother resolving _ft_dag0 since
            # there is no such thing; source's round-1 training also branched
            # off --initial_policy_path.
            if [[ -f "$INITIAL_POLICY_PATH/checkpoints/last/pretrained_model/config.json" ]]; then
                CURRENT_POLICY="$INITIAL_POLICY_PATH/checkpoints/last/pretrained_model"
            elif [[ -f "$INITIAL_POLICY_PATH/pretrained_model/config.json" ]]; then
                CURRENT_POLICY="$INITIAL_POLICY_PATH/pretrained_model"
            elif [[ -f "$INITIAL_POLICY_PATH/config.json" ]]; then
                CURRENT_POLICY="$INITIAL_POLICY_PATH"
            else
                CURRENT_POLICY="$(resolve_latest_checkpoint "$INITIAL_POLICY_PATH")"
            fi
        else
            _SRC_PREV_DIR="$LEROBOT_ROOT/outputs/training/${SOURCE_POLICY_BASENAME}_ft_dag$((r - 1))"
            CURRENT_POLICY="$(resolve_latest_checkpoint "$_SRC_PREV_DIR")"
        fi
        echo "[rerun] Round $r branching from source policy: $CURRENT_POLICY"
    fi

    # The starting step within this round.
    if (( r == EFFECTIVE_START_ROUND )); then STEP=$EFFECTIVE_START_STEP; else STEP=1; fi

    echo
    echo "════════════════════════════════════════════════════════════════"
    echo "ROUND $r / $NUM_ROUNDS   (mode=$MODE, starting at step $STEP)"
    echo "  Intervention dataset: $INT_REPO"
    echo "  Alias dataset:        $ALIAS_REPO"
    echo "  Merged dataset:       $MERGED_REPO"
    echo "  Training output:      $TRAIN_OUTPUT_DIR"
    echo "  Current policy:       $CURRENT_POLICY"
    echo "════════════════════════════════════════════════════════════════"

    # Ensure the external SplatSim is running before any sim-using step.
    # Step 1 (intervention recording) and step 2 (blending) both need it;
    # steps 3-5 are pure data ops (alias, merge, stats) that don't touch
    # the sim; step 6 (train) spawns its own in-process sim. We stop the
    # orchestrator-managed sim immediately after step 2 finishes (below)
    # so it doesn't hold GPU memory idle across merge/stats/training-load.
    # Idempotent: no-op if we already launched it or if --no_manage_splatsim.
    # Resuming at step 2 (skipping step 1 because the intervention dataset
    # already exists) still needs the sim brought up here — without this
    # gate including step 2, blending would race against a possibly-dead
    # sim from a prior run.
    if (( STEP <= 2 )); then
        # Compute the per-round failed-only subset BEFORE start_sim so the
        # initial launch uses the correct subset directly. (Putting this
        # AFTER start_sim caused a wasteful stop+restart on every round
        # that needed a non-default subset.) The block also sets the
        # downstream SUBSET_ARG and EFFECTIVE_N_EPISODES_FOR_INT used by
        # the step-1 lerobot-eval call below, so we need to run it
        # whether or not we end up launching/restarting the sim here.
        EVAL_BENCHMARK_SUBSET_FOR_SIM=""
        SUBSET_ARG=()
        if [[ -n "$INTERVENTION_SUBSET_JSON" ]]; then
            SUBSET_ARG+=( "--env.eval_benchmark_subset=$INTERVENTION_SUBSET_JSON" )
        fi
        EFFECTIVE_N_EPISODES_FOR_INT="$INTERVENTION_N_EPISODES"
        if [[ "$DAGGER_SKIP_SUCCEEDED_IN_PREV_EVAL" == "true" && "$r" -gt 1 ]]; then
            PREV_TRAIN_OUTPUT_DIR=$(train_output_dir_for_round "$((r - 1))")
            FAILED_JSON=""
            FAILED_RC=0
            FAILED_JSON=$(python3 "$SCRIPT_DIR/dagger_failed_scenarios.py" \
                --prev_train_dir="$PREV_TRAIN_OUTPUT_DIR" 2>&1) || FAILED_RC=$?
            if (( FAILED_RC == 0 )); then
                FAILED_CSV=$(echo "$FAILED_JSON" | python3 -c \
                    "import sys,json; d=json.load(sys.stdin); print(','.join(str(x) for x in d['failed']))")
                N_FAILED=$(echo "$FAILED_JSON" | python3 -c \
                    "import sys,json; print(json.load(sys.stdin)['n_failed'])")
                N_TOTAL_PRIOR=$(echo "$FAILED_JSON" | python3 -c \
                    "import sys,json; print(json.load(sys.stdin)['n_total'])")
                if (( N_FAILED > 0 )); then
                    SUBSET_ARG=( "--env.eval_benchmark_subset=[$FAILED_CSV]" )
                    EFFECTIVE_N_EPISODES_FOR_INT="$N_FAILED"
                    EVAL_BENCHMARK_SUBSET_FOR_SIM="$FAILED_CSV"
                    echo "  [skip_succeeded] round $r targeting $N_FAILED/$N_TOTAL_PRIOR scenarios "\
"that failed in round $((r - 1))'s training-time eval: [$FAILED_CSV]"
                else
                    echo "  [skip_succeeded] round $((r - 1)) had 100% success "\
"($N_TOTAL_PRIOR/$N_TOTAL_PRIOR); falling back to full subset for round $r"
                fi
            else
                echo "  [skip_succeeded] no usable prior eval data under "\
"$PREV_TRAIN_OUTPUT_DIR (rc=$FAILED_RC); falling back to full subset for round $r"
                echo "$FAILED_JSON" | sed 's/^/    /' | head -3
            fi
        elif [[ "$DAGGER_SKIP_SUCCEEDED_IN_PREV_EVAL" != "true" ]]; then
            echo "  [skip_succeeded] disabled — running interventions on full subset"
        fi
        # Restart-if-subset-changed: when SplatSim is already running with
        # a DIFFERENT subset from a prior round, tear it down first so the
        # next start_sim launches fresh with the correct one. start_sim
        # is a no-op when sim is still running, so without this stop_sim
        # the relaunch wouldn't happen.
        if [[ "$MANAGE_SPLATSIM" == "true" \
              && -n "${MANAGED_SIM_PID:-}" \
              && "${MANAGED_SIM_PID:-}" != "DRYRUN" ]] \
           && kill -0 "${MANAGED_SIM_PID:-}" 2>/dev/null \
           && [[ "$EVAL_BENCHMARK_SUBSET_FOR_SIM" != "$EVAL_BENCHMARK_SUBSET_CURRENT_SIM" ]]; then
            _from="${EVAL_BENCHMARK_SUBSET_CURRENT_SIM:-<default>}"
            _to="${EVAL_BENCHMARK_SUBSET_FOR_SIM:-<default>}"
            echo "  [splat-restart] subset changed: '$_from' → '$_to'; tearing down + relaunching SplatSim"
            stop_sim
        fi
        start_sim
        EVAL_BENCHMARK_SUBSET_CURRENT_SIM="$EVAL_BENCHMARK_SUBSET_FOR_SIM"
    fi

    # Step 1: Record interventions (or in rerun mode, validate source).
    if (( STEP <= 1 )); then
        if [[ "$RERUN_MODE_ENABLED" == "true" ]]; then
            # Rerun mode: validate source intervention is on disk, then skip
            # the lerobot-eval recording. The upfront validation block
            # already checked round-by-round; this is the per-round guard
            # against datasets disappearing mid-run (rare).
            if [[ ! -d "$LEROBOT_CACHE/$INT_REPO" ]]; then
                echo "ERROR: source intervention $INT_REPO missing on disk." >&2
                echo "  Expected at: $LEROBOT_CACHE/$INT_REPO" >&2
                exit 1
            fi
            echo "--- Round $r, Step 1: REUSING source intervention $INT_REPO ---"
            # Stats sidecar: source-intervention sidecars are stable across
            # reruns (the underlying dataset is read-only), so skip the
            # recompute if the sidecar for the active model's chunk size is
            # already on disk. The non-rerun branch below still runs the
            # recompute unconditionally because that path just wrote a fresh
            # intervention dataset whose stats need to be (re)derived.
            if [[ "$ACTION_FORMAT" == "rel" ]]; then
                if stats_exists "$INT_SHORT"; then
                    echo "--- Round $r, Step 1b: source sidecar stats already on disk for $INT_SHORT; skipping ---"
                else
                    echo "--- Round $r, Step 1b: compute sidecar stats for $INT_SHORT ---"
                    run_or_echo bash "$SCRIPT_DIR/compute_relative_stats.sh" --dataset_repo="$INT_REPO"
                fi
            fi
            # Done with step 1 in rerun mode — skip the recording path below.
            STEP_1_REUSED=true
        else
            STEP_1_REUSED=false
        fi
    fi
    if (( STEP <= 1 )) && [[ "${STEP_1_REUSED:-false}" != "true" ]]; then
        # Preserve-existing-dataset short-circuit: if the intervention dataset
        # for this round is already on disk (e.g. a prior run lost only its
        # stats sidecar, or the user preserved interventions while wiping
        # training), DON'T re-launch lerobot-eval — its teleop_dataset_repo_id
        # path appends to non-empty datasets, which would silently grow /
        # pollute the original recording. Just regen the stats sidecar (if
        # missing for rel mode) and treat step 1 as done. To force fresh
        # recording, the user can delete the dataset directory first or pass
        # --force_restart.
        if dataset_exists "$INT_REPO"; then
            echo "--- Round $r, Step 1: PRESERVING existing intervention dataset $INT_REPO ---"
            echo "  (dataset on disk; skipping lerobot-eval recording)"
            if [[ "$ACTION_FORMAT" == "rel" ]]; then
                if stats_exists "$INT_SHORT"; then
                    echo "--- Round $r, Step 1b: sidecar stats already on disk for $INT_SHORT; skipping ---"
                else
                    echo "--- Round $r, Step 1b: regen sidecar stats for existing $INT_SHORT ---"
                    run_or_echo bash "$SCRIPT_DIR/compute_relative_stats.sh" --dataset_repo="$INT_REPO"
                fi
            fi
            STEP_1_PRESERVED=true
        else
            STEP_1_PRESERVED=false
        fi
    else
        STEP_1_PRESERVED=false
    fi
    if (( STEP <= 1 )) && [[ "${STEP_1_REUSED:-false}" != "true" ]] && [[ "${STEP_1_PRESERVED:-false}" != "true" ]]; then
        echo "--- Round $r, Step 1: record interventions ---"
        # SUBSET_ARG, EFFECTIVE_N_EPISODES_FOR_INT, and SplatSim's launch
        # subset are all decided earlier in this iteration (before the
        # round's first start_sim) — see the EVAL_BENCHMARK_SUBSET_FOR_SIM
        # block above. By the time we reach here, SplatSim is already
        # running with the right subset for this round.
        #
        # Offline-mode flag: skip the TeleopRecording dataset push when
        # --push_to_hub is NOT set on the orchestrator (the default). When
        # the user opts in with --push_to_hub, omit this flag so the
        # wrapper's default (push enabled) wins.
        OFFLINE_DATASET_ARG=()
        if [[ "$PUSH_TO_HUB" == false ]]; then
            OFFLINE_DATASET_ARG+=( "--env.teleop_push_to_hub=false" )
        fi
        # Intervention recording is driven by `lerobot-eval` with the
        # `--intervention.*` config field (see AGENTS.md / CLAUDE.md).
        #
        # Three hard requirements that lerobot-eval validates at startup
        # when --intervention.method is set:
        #   1. --policy.shared_autonomy_config.enabled=true
        #      → both 'rrt' and 'oracle_goal' guidance sources live on the SA
        #         wrapper; without it there's no intervention path.
        #   2. --env.include_oracle_info=true
        #      → RRT/oracle_goal need target_ee_pos / q_goal_bias from
        #         oracle_env_config. Without it the planner refuses to plan.
        #   3. --seed=0
        #      → EvalPipelineConfig.seed defaults to 1000 (configs/eval.py:39).
        #         splatsim's seed-pinned reset uses this to pick scenario 0,
        #         1, 2, ... in order. With the default seed, the recorder
        #         starts at a random scenario in the benchmark and counts
        #         forward from there — which breaks DAgger's "keep training
        #         on the same hard scenarios round over round" semantics.
        # Other SA settings (forward_flow_ratio, blend_strategy, etc.) can
        # be added via --intervention_extra_args.
        # --headless mode: turn off the Tkinter ratio slider AND the SA
        # wrapper's per-policy pybullet GUI client (both gated by the same
        # show_slider field — see shared_autonomy_wrapper.py:240). The
        # external SplatSim was launched headless above; this kills the
        # last visualizer in the loop.
        HEADLESS_EVAL_ARG=()
        if [[ "$HEADLESS" == true ]]; then
            HEADLESS_EVAL_ARG=( --policy.shared_autonomy_config.show_slider=false )
        fi
        # Dedicated RRT-clearance args forwarded as SA-config fields. Kept
        # separate from --intervention_extra_args so they're individually
        # discoverable in --help, validated at parse time, and recorded in
        # the sidecar's `config` block as their own fields rather than
        # buried in a free-form string. Empty → omit, letting
        # RRTToGoalPlanner fall through to SplatSim's defaults
        # (_COLLISION_CLEARANCE = 0.01 m, self = 0.0 m).
        RRT_CLEARANCE_ARGS=()
        if [[ -n "$RRT_OBSTACLE_CLEARANCE" ]]; then
            RRT_CLEARANCE_ARGS+=( "--policy.shared_autonomy_config.rrt_obstacle_clearance=$RRT_OBSTACLE_CLEARANCE" )
        fi
        if [[ -n "$RRT_SELF_COLLISION_CLEARANCE" ]]; then
            RRT_CLEARANCE_ARGS+=( "--policy.shared_autonomy_config.rrt_self_collision_clearance=$RRT_SELF_COLLISION_CLEARANCE" )
        fi
        if [[ -n "$RRT_SELF_COLLISION_SKIP_PAIRS" ]]; then
            RRT_CLEARANCE_ARGS+=( "--policy.shared_autonomy_config.rrt_self_collision_skip_pairs=$RRT_SELF_COLLISION_SKIP_PAIRS" )
        fi
        # shellcheck disable=SC2086  # INTERVENTION_EXTRA_ARGS may contain multiple flags
        run_or_echo lerobot-eval \
            --policy.path="$CURRENT_POLICY" \
            --policy.shared_autonomy_config.enabled=true \
            --env.type=splatsim \
            --env.task=upright_small_engine_new \
            --env.fps=30 \
            --env.external_port="$ENV_EXTERNAL_PORT" \
            --env.external_host="$ENV_EXTERNAL_HOST" \
            --env.include_oracle_info=true \
            --env.episode_length="$INTERVENTION_MAX_EPISODE_LENGTH" \
            --env.eval_benchmark_repo_id="$EVAL_BENCHMARK_REPO_ID" \
            --env.teleop_dataset_repo_id="$INT_REPO" \
            --eval.n_episodes="$EFFECTIVE_N_EPISODES_FOR_INT" \
            --eval.batch_size=1 \
            --eval.use_async_envs=false \
            --seed=0 \
            --output_dir="$TRAIN_OUTPUT_DIR/dagger/interventions" \
            --intervention.method="$INTERVENTION_METHOD" \
            --intervention.oracle_goal_chunk_steps="$INTERVENTION_ORACLE_GOAL_CHUNK_STEPS" \
            "${SUBSET_ARG[@]}" \
            "${OFFLINE_DATASET_ARG[@]}" \
            "${HEADLESS_EVAL_ARG[@]}" \
            "${RRT_CLEARANCE_ARGS[@]}" \
            $INTERVENTION_EXTRA_ARGS

        # Stats on the intervention dataset (relative-action sidecar).
        # Folded into step 1 — completion of step 1 requires BOTH the recording
        # AND its stats sidecar (when ACTION_FORMAT=rel). For absolute-action
        # policies, lerobot-train uses the dataset's own meta/stats.json
        # directly and this sidecar would be incorrect to apply.
        if [[ "$ACTION_FORMAT" == "rel" ]]; then
            echo "--- Round $r, Step 1b: compute sidecar stats for $INT_SHORT ---"
            run_or_echo bash "$SCRIPT_DIR/compute_relative_stats.sh" --dataset_repo="$INT_REPO"
        else
            echo "--- Round $r, Step 1b: SKIPPED (--action_format=abs; rel-stats sidecar not used) ---"
        fi
    fi

    # Step 2: Per-ratio blended-intervention datasets (--blends). Each ratio
    # in $BLENDS triggers a closed-loop replay of the round's intervention
    # episodes through SplatSim with the SA wrapper's forward_flow_ratio set
    # to that ratio, producing a separate dataset stored alongside the raw
    # intervention. Both raw and each blend get merged in step 4 — see the
    # EXTRA_SOURCES build there. Empty $BLENDS → no-op.
    #
    # Reuses the existing orchestrator-managed SplatSim on $ENV_EXTERNAL_PORT
    # (no separate start_sim/stop_sim cycling between step 1 and step 2).
    if (( STEP <= 2 )) && (( ${#BLENDS[@]} > 0 )); then
        echo "--- Round $r, Step 2: blend intervention at ratios=${BLENDS[*]} ---"
        for R in "${BLENDS[@]}"; do
            BLEND_SHORT="$(blend_short_for_round "$r" "$R")"
            BLEND_REPO="$(blend_repo_for_round "$r" "$R")"
            # Per-ratio resume: skip blend creation when the dataset is already
            # on disk AND (action_format != rel OR its stats sidecar exists).
            # Mirrors step 1's sub-check semantics. NOTE: this gate covers ONLY
            # the augment_dataset_with_blending.py invocation — the collision-
            # filter sub-step below has its own independent resume check, so a
            # user who toggled --filter_blend_collisions on AFTER the blends
            # were already built still gets the `_nocoll` siblings generated
            # without redoing the blend.
            _blend_complete=false
            if dataset_exists "$BLEND_REPO"; then
                if [[ "$ACTION_FORMAT" != "rel" ]] || stats_exists "$BLEND_SHORT"; then
                    _blend_complete=true
                fi
            fi
            if [[ "$_blend_complete" == true ]]; then
                echo "  ratio=$R → $BLEND_REPO already on disk (with stats); skipping blend creation."
            else
                echo "  ratio=$R → producing $BLEND_REPO via augment_dataset_with_blending.py"
                BLEND_PUSH_ARG=()
                [[ "$PUSH_TO_HUB" == true ]] && BLEND_PUSH_ARG+=( "--push_to_hub" )
                # shellcheck disable=SC2086  # BLEND_EXTRA_ARGS may contain multiple flags
                # `--episode_index=all` explicitly says "process every episode in
                # the source dataset" — recognized by augment_dataset_with_blending.py.
                # The source intervention dataset already contains only the
                # per-round subset, so "all" == that subset. Override via
                # --blend_extra_args=--episode_index=N-M if you need finer control.
                run_or_echo python "$SCRIPT_DIR/augment_dataset_with_blending.py" \
                    --dataset_repo_id="$INT_REPO" \
                    --target_dataset_repo_id="$BLEND_REPO" \
                    --policy_path="$CURRENT_POLICY" \
                    --forward_flow_ratios="[$R]" \
                    --episode_index="all" \
                    --env_external_port="$ENV_EXTERNAL_PORT" \
                    --env_external_host="$ENV_EXTERNAL_HOST" \
                    --env_task="upright_small_engine_new" \
                    --eval_benchmark_repo_id="$EVAL_BENCHMARK_REPO_ID" \
                    --blend_strategy="denoise" \
                    --guidance_repr="absolute_pos" \
                    --blend_mode="once_per_chunk" \
                    --drain_chunk \
                    "${BLEND_PUSH_ARG[@]}" \
                    $BLEND_EXTRA_ARGS
                # Stats sidecar for the blended dataset (mirrors step 1b for raw int).
                if [[ "$ACTION_FORMAT" == "rel" ]]; then
                    run_or_echo bash "$SCRIPT_DIR/compute_relative_stats.sh" --dataset_repo="$BLEND_REPO"
                fi
            fi
            # Step 2b — collision filter (optional). Replays the blend dataset
            # through a headless splatsim and drops episodes that hit obstacles
            # (with a configurable pre-collision trim margin). Step 4's merge
            # then uses the resulting `_nocoll` sibling instead of the raw blend.
            # Idempotent: short-circuits if `_nocoll` (and its stats sidecar
            # in rel mode) is already on disk.
            if [[ "$FILTER_BLEND_COLLISIONS" == "true" ]]; then
                NOCOLL_SHORT="$(nocoll_short_for_round "$r" "$R")"
                NOCOLL_REPO="$(nocoll_repo_for_round "$r" "$R")"
                _nocoll_complete=false
                if dataset_exists "$NOCOLL_REPO"; then
                    if [[ "$ACTION_FORMAT" != "rel" ]] || stats_exists "$NOCOLL_SHORT"; then
                        _nocoll_complete=true
                    fi
                fi
                if [[ "$_nocoll_complete" == true ]]; then
                    echo "  ratio=$R → $NOCOLL_REPO already on disk (with stats); skipping filter."
                else
                    echo "  ratio=$R → producing $NOCOLL_REPO via filter_blend_collisions.py (headless sim)"
                    # Per-episode CSV with the filter's kept/dropped decisions
                    # lands under <train_dir>/dagger/blend_collision_filter/ so
                    # the audit trail travels with the lineage's other dagger
                    # sidecars. One CSV per ratio per round; filename is
                    # <NOCOLL_SHORT>.csv (matches the target dataset).
                    FILTER_OUTPUT_DIR="$TRAIN_OUTPUT_DIR/dagger/blend_collision_filter"
                    start_filter_sim
                    # shellcheck disable=SC2086  # FILTER_COLLISION_EXTRA_ARGS may contain multiple flags
                    run_or_echo python "$SCRIPT_DIR/filter_blend_collisions.py" \
                        --source_repo_id="$BLEND_REPO" \
                        --target_repo_id="$NOCOLL_REPO" \
                        --env_task="upright_small_engine_new" \
                        --env_robot_name="$SPLATSIM_ROBOT_NAME" \
                        --env_external_port="$FILTER_COLLISION_ENV_PORT_RESOLVED" \
                        --env_external_host="$ENV_EXTERNAL_HOST" \
                        --env_eval_benchmark_repo_id="$EVAL_BENCHMARK_REPO_ID" \
                        --output_dir="$FILTER_OUTPUT_DIR" \
                        $FILTER_COLLISION_EXTRA_ARGS
                    # Stats sidecar for the filtered dataset.
                    if [[ "$ACTION_FORMAT" == "rel" ]]; then
                        run_or_echo bash "$SCRIPT_DIR/compute_relative_stats.sh" --dataset_repo="$NOCOLL_REPO"
                    fi
                fi
            fi
        done
    elif (( STEP <= 2 )); then
        echo "--- Round $r, Step 2: SKIPPED (--blends is empty; no blended datasets to produce) ---"
    fi

    # Stop the orchestrator-managed external SplatSim now that the only
    # sim-using steps of the round (1 and 2) are done. Steps 3-5 are pure
    # dataset ops (alias / merge / stats); step 6 spawns its own in-process
    # sim for inline eval. Idempotent + gated to mirror the start_sim above,
    # so a resume at STEP>=3 (sim never started) is a clean no-op.
    # Also stop the auxiliary collision-filter sim. It runs with STRICTER
    # collision clearances than the main sim (forwarded from the SA wrapper's
    # rrt_*_clearance values via start_filter_sim) — and the only consumer is
    # step 2's `filter_blend_collisions.py` invocation, which has already
    # completed by here. Stopping it now (rather than waiting for the EXIT
    # trap) prevents the filter sim from sitting idle through steps 3-6 /
    # step 6b and rules out any chance of a later step accidentally
    # connecting to its port and inheriting the stricter semantics.
    # Idempotent — no-op when MANAGED_FILTER_SIM_PID is empty.
    if (( STEP <= 2 )); then
        stop_sim
        stop_filter_sim
    fi

    # Step 3: Hardlink-alias via augment_ratios_sweep.sh ratio=0 branch.
    # Skipped when --skip_alias_step is set; merge consumes raw intervention
    # datasets directly instead.
    if (( STEP <= 3 )) && [[ "$SKIP_ALIAS_STEP" == false ]]; then
        echo "--- Round $r, Step 3: hardlink-alias intervention dataset under $ALIAS_REPO ---"
        run_or_echo bash "$SCRIPT_DIR/augment_ratios_sweep.sh" \
            --dataset_short="$INT_SHORT" \
            --ratios="0.0" \
            --model="$MODEL" \
            --action_format="$ACTION_FORMAT"
    elif (( STEP <= 3 )); then
        echo "--- Round $r, Step 3: SKIPPED (--skip_alias_step) ---"
    fi

    # Step 4: Merge cumulatively (base + dag1 + ... + dag{r}, using either
    # the _pirel00 aliases or the raw dag{N} datasets depending on whether
    # the alias step is enabled). SKIPPED in weighted-sampling mode — step 6
    # feeds the per-source datasets straight into lerobot-train's
    # multi-dataset path (--dataset.repo_ids + sample_weights + stats_paths)
    # and the DataLoader's WeightedRandomSampler picks per-source frames at
    # the target ratio; no on-disk merged dataset is needed.
    if (( STEP <= 4 )) && [[ "$PURE_POLICY_MODE" == "true" ]]; then
        echo "--- Round $r, Step 4: SKIPPED (PURE_POLICY_MODE; training uses base dataset only, no merge needed) ---"
    elif (( STEP <= 4 )) && [[ "$USE_WEIGHTED_SAMPLING" == "true" ]]; then
        echo "--- Round $r, Step 4: SKIPPED (--use_weighted_sampling; no merged dataset needed) ---"
    elif (( STEP <= 4 )); then
        echo "--- Round $r, Step 4: cumulative merge → $MERGED_REPO ---"
        # The merge tool refuses to overwrite an existing target dir
        # (LeRobotDatasetMetadata.create uses mkdir(exist_ok=False)). If a
        # previous run created this round's merged dataset but then crashed
        # before stats were computed (step 5), resume detection brings us
        # back here — and the bare merge would fail with FileExistsError.
        # Nuke any pre-existing target so we start clean. Safe because the
        # only way we reach step 4 (instead of 5+) is if stats DON'T exist
        # for this merged dataset, which means it's stale/partial anyway.
        STALE_MERGED_DIR="$LEROBOT_CACHE/$MERGED_REPO"
        if [[ -e "$STALE_MERGED_DIR" ]]; then
            echo "  Removing stale merged dataset dir before re-merge: $STALE_MERGED_DIR"
            run_or_echo rm -rf "$STALE_MERGED_DIR"
        fi
        # Build the list of intervention sources with per-condition counts:
        #   raw × (N - n_blends) + each_blend × 1
        # where N = $TARGET_INTERVENTION_VOLUME. aggregate_datasets just
        # iterates the repo_ids list without dedup, so duplicate entries
        # concatenate the same data into the merged dataset. Total
        # intervention-related content per round = N × raw_intervention,
        # independent of how many blends — apples-to-apples across 0/1/2/…
        # blend conditions.
        #
        # INCREMENTAL MERGE: if round (r-1)'s merged dataset is still on disk
        # (it's only cleaned up at round r's pre-train-cleanup, AFTER step 4
        # has run), we use it as the merge base and only feed round r's NEW
        # sources to --extra_sources. The previous merged already contains
        # base + every round-1..(r-1) source at the configured oversample, so
        # this is mathematically equivalent to the full merge but processes
        # 3× fewer sources per round (and per-source overhead in the merge
        # tool — dataset open, metadata reads — dominates the wall time at
        # high source counts). Falls back to a full merge when the previous
        # merged is missing (round 1, or after a manual cleanup).
        PREV_MERGED_REPO=""
        if (( r >= 2 )); then
            _prev_candidate="$(merged_repo_for_round "$((r - 1))")"
            if [[ -d "$LEROBOT_CACHE/$_prev_candidate" ]]; then
                PREV_MERGED_REPO="$_prev_candidate"
            elif [[ "$DRY_RUN" == true ]]; then
                # In dry-run, the previous round's merged isn't actually on
                # disk (we skipped its real creation), so check whether the
                # current dry-run "wrote" it on a prior iteration.
                # DRY_RUN_MERGED_CREATED is populated below after each
                # round's merge step prints its dry-run command.
                if [[ " ${DRY_RUN_MERGED_CREATED:-} " == *" $_prev_candidate "* ]]; then
                    PREV_MERGED_REPO="$_prev_candidate"
                fi
            fi
        fi
        if [[ -n "$PREV_MERGED_REPO" ]]; then
            MERGE_BASE_REPO="$PREV_MERGED_REPO"
            ROUNDS_TO_ADD=( "$r" )
            echo "  incremental merge: base = $PREV_MERGED_REPO; adding only round $r's new sources"
        else
            MERGE_BASE_REPO="$BASE_REPO"
            ROUNDS_TO_ADD=( $(seq 1 "$r") )
            if (( r >= 2 )); then
                echo "  full merge (round $((r - 1))'s merged not on disk): $BASE_REPO + all rounds 1..$r"
            else
                echo "  full merge (round 1): $BASE_REPO + round 1's sources"
            fi
        fi
        # Per-round source counts. Each round contributes:
        #   raw_count copies of the raw intervention dataset
        #   1 copy of each blend dataset
        # totaling TARGET_INTERVENTION_VOLUME copies of intervention-sized data.
        _raw_count=$(( TARGET_INTERVENTION_VOLUME - ${#BLENDS[@]} ))
        EXTRA_SOURCES=()
        for prev in "${ROUNDS_TO_ADD[@]}"; do
            # Raw intervention dataset (or its alias when --skip_alias_step is
            # false). Repeated _raw_count times so the total intervention-
            # related content per round stays at N × raw regardless of blends.
            if [[ "$SKIP_ALIAS_STEP" == true ]]; then
                SRC_RAW="$(int_repo_for_round "$prev")"
            else
                SRC_RAW="$(alias_repo_for_round "$prev")"
            fi
            for ((dup=0; dup < _raw_count; dup++)); do
                EXTRA_SOURCES+=( "$SRC_RAW" )
            done
            # Blended-intervention datasets (one per ratio in $BLENDS), each
            # included exactly once. Empty $BLENDS → loop is a no-op and the
            # round is composed entirely of raw_count copies of raw.
            # When --filter_blend_collisions is on, swap in the `_nocoll`
            # sibling produced in step 2b. The raw blend stays on disk
            # (cross-rerun cacheable) but isn't part of THIS lineage's merge.
            for R in "${BLENDS[@]}"; do
                if [[ "$FILTER_BLEND_COLLISIONS" == "true" ]]; then
                    SRC_BLEND="$(nocoll_repo_for_round "$prev" "$R")"
                else
                    SRC_BLEND="$(blend_repo_for_round "$prev" "$R")"
                fi
                EXTRA_SOURCES+=( "$SRC_BLEND" )
            done
        done
        # Composition log. Always print so the merged dataset's makeup is
        # discoverable in orchestrator logs without having to recount EXTRA_SOURCES.
        if (( ${#BLENDS[@]} > 0 )); then
            if (( _raw_count == 0 )); then
                echo "  target intervention volume: N=${TARGET_INTERVENTION_VOLUME} → per round: blends only (raw dropped, N == n_blends) — ${#BLENDS[@]} blend(s) × 1 = ${TARGET_INTERVENTION_VOLUME}× raw_intervention worth"
            else
                echo "  target intervention volume: N=${TARGET_INTERVENTION_VOLUME} → per round: raw × ${_raw_count} + ${#BLENDS[@]} blend(s) × 1 = ${TARGET_INTERVENTION_VOLUME}× raw_intervention worth of intervention-related content"
            fi
            echo "  blend ratios: ${BLENDS[*]} (${#BLENDS[@]} extra source(s) per round at 1× each)"
        else
            echo "  target intervention volume: N=${TARGET_INTERVENTION_VOLUME} → per round: raw × ${_raw_count} (no blends) = ${TARGET_INTERVENTION_VOLUME}× raw_intervention"
        fi
        run_or_echo python "$SCRIPT_DIR/merge_augmented_datasets_for_training.py" \
            --base "$MERGE_BASE_REPO" \
            --extra_sources "${EXTRA_SOURCES[@]}" \
            --output_repo_id="$MERGED_REPO"
        if [[ "$DRY_RUN" == true ]]; then
            # Track virtual creation so later rounds' dry-run output reflects
            # the incremental-merge chain that would happen in a real run.
            DRY_RUN_MERGED_CREATED="${DRY_RUN_MERGED_CREATED:-} $MERGED_REPO"
        fi
    fi

    # Step 5: Stats on merged dataset (relative-action sidecar; only used when
    # ACTION_FORMAT=rel — see Step 2 comment). SKIPPED in weighted mode (no
    # merged dataset exists; per-source sidecars from step 1b/2b are what
    # step 6 consumes via --dataset.stats_paths).
    if (( STEP <= 5 )) && [[ "$PURE_POLICY_MODE" == "true" ]]; then
        echo "--- Round $r, Step 5: SKIPPED (PURE_POLICY_MODE; no merged dataset to compute stats for) ---"
    elif (( STEP <= 5 )) && [[ "$USE_WEIGHTED_SAMPLING" == "true" ]]; then
        echo "--- Round $r, Step 5: SKIPPED (--use_weighted_sampling; per-source stats sidecars already on disk from steps 1b/2b) ---"
    elif (( STEP <= 5 )); then
        if [[ "$ACTION_FORMAT" == "rel" ]]; then
            echo "--- Round $r, Step 5: compute sidecar stats for $MERGED_SHORT ---"
            run_or_echo bash "$SCRIPT_DIR/compute_relative_stats.sh" --dataset_repo="$MERGED_REPO"
        else
            echo "--- Round $r, Step 5: SKIPPED (--action_format=abs; rel-stats sidecar not used) ---"
        fi
    fi

    # Cleanup of round (r-1)'s merged dataset BEFORE this round's training
    # starts (used to be Step 7 after training). Both dag{r-1}_m and dag{r}_m
    # exist on disk briefly between the merge step and this cleanup; doing
    # the cleanup before training means at training time we only hold dag{r}_m.
    # Saves ~15 GB of disk during the longest-running step (training itself).
    # Safe because dag{r-1}_m was only needed for round (r-1)'s training,
    # which has already produced its checkpoint by the time we get here.
    # Originals (base + per-round dag/alias) stay on disk for future mix-and-match.
    if (( r > 1 )) && (( STEP <= 6 )); then
        echo "--- Round $r, Pre-train cleanup: remove round $((r-1))'s merged dataset ---"
        PREV_MERGED_REPO="$(merged_repo_for_round $((r-1)))"
        PREV_MERGED_SHORT="$(merged_short_for_round $((r-1)))"
        run_or_echo rm -rf "$LEROBOT_CACHE/$PREV_MERGED_REPO"
        run_or_echo rm -rf "$STATS_BASE/$PREV_MERGED_SHORT"
    fi

    # Step 6: Train.
    if (( STEP <= 6 )); then
        echo "--- Round $r, Step 6: train ($MODE) ---"
        # External SplatSim was stopped right after step 2; lerobot-train
        # spawns its OWN in-process sim for inline eval, which pools its
        # CUDA context with training (lower fragmentation, lower OOM risk).
        # In --no_manage_splatsim mode there's no orchestrator-managed sim
        # to stop; the user-launched external sim stays up and the user
        # takes responsibility for the memory tradeoff.
        print_gpu_state
        if [[ "$MODE" == "scratch" ]]; then
            # Pass --run_name so train_sweep.sh writes to ${BASE_POLICY_NAME}_dag${r}
            # instead of its default ${MODEL}_<dataset>_<action>_basewrist naming.
            # This keeps scratch-mode and finetune-mode rounds at the same path.
            # In retrain mode the same suffix used for TRAIN_OUTPUT_DIR is also
            # applied to the run_name so wandb/job_name don't collide.
            SCRATCH_RUN_NAME="${BASE_POLICY_NAME}_dag${r}"
            if [[ -n "$RETRAIN_ROUND" && "$r" == "$RETRAIN_ROUND" ]]; then
                SCRATCH_RUN_NAME="${SCRATCH_RUN_NAME}_${RETRAIN_SUFFIX}"
            fi
            # Forward the action-format choice. train_sweep.sh defaults to
            # USE_RELATIVE_ACTIONS=true and pre-flights stats_rel{N}.json — when
            # we're training an abs-action policy those sidecars don't exist
            # (steps 2/5 skipped them on purpose), so omit them with --no_relative.
            ABS_ACTION_ARG=()
            if [[ "$ACTION_FORMAT" == "abs" ]]; then
                ABS_ACTION_ARG=( --no_relative )
            fi
            cleanup_pre_train_partial "$TRAIN_OUTPUT_DIR"
            run_training_step bash "$SCRIPT_DIR/train_sweep.sh" \
                --dataset_repo="$MERGED_REPO" \
                --run_name="$SCRATCH_RUN_NAME" \
                --model="$TRAIN_OUTPUT_MODEL_PREFIX" \
                "${ABS_ACTION_ARG[@]}" \
                "${TRAIN_EXT_PORT_SWEEP[@]}" \
                "${HEADLESS_TRAIN_SCRATCH_ARGS[@]}" \
                "${OFFLINE_POLICY_ARG[@]}"
        else
            # Finetune mode. Two important details:
            #
            # 1. lerobot-train's `steps` is the TOTAL target step count, not
            #    additional. When resuming from step 26000 and user asks for
            #    --finetune_steps=200, we must pass --steps=26200 so training
            #    runs for 200 more steps from where it left off. Without this,
            #    `cfg.steps <= current_step` and the loop exits at step 0.
            #
            # 2. The new `_ft`-suffixed training dir is created by overriding
            #    --policy.repo_id / --output_dir / --job_name on the resumed
            #    run; otherwise lerobot-train writes back into the original
            #    training dir from train_config.json.
            FT_RUN_NAME="${BASE_POLICY_NAME}_ft_dag${r}"
            if [[ -n "$RETRAIN_ROUND" && "$r" == "$RETRAIN_ROUND" ]]; then
                FT_RUN_NAME="${FT_RUN_NAME}_${RETRAIN_SUFFIX}"
            fi
            # Determine the current checkpoint step. In real runs this comes
            # from the resolved path (.../checkpoints/{step}/pretrained_model/).
            # In dry-run, paths from previous loop iterations are synthetic and
            # contain "checkpoints/last/..." rather than a numeric step, so we
            # fall back to a cached running counter that's bumped each round.
            CURRENT_STEP=""
            RESOLVED_POLICY="$(readlink -f "$CURRENT_POLICY" 2>/dev/null || echo "$CURRENT_POLICY")"
            CURRENT_STEP="$(echo "$RESOLVED_POLICY" | grep -oE 'checkpoints/[0-9]+/' | head -1 | grep -oE '[0-9]+' || true)"
            if [[ -z "$CURRENT_STEP" ]]; then
                if [[ "$DRY_RUN" == true && -n "${DRYRUN_NEXT_STEP:-}" ]]; then
                    CURRENT_STEP="$DRYRUN_NEXT_STEP"
                else
                    echo "ERROR: could not extract current step from policy path: $CURRENT_POLICY" >&2
                    echo "  (resolved as: $RESOLVED_POLICY)" >&2
                    echo "  Expected path of the form .../checkpoints/{step}/pretrained_model/" >&2
                    exit 1
                fi
            fi
            # Force base-10 to avoid octal parsing of '026000'.
            TARGET_STEPS=$((10#${CURRENT_STEP} + FINETUNE_STEPS))

            # ── Extension mode for the last complete round ──────────────
            # When --extend_last_round demoted round r from 6/6 → 5/6, we want
            # lerobot-train to resume from r's EXISTING checkpoint (not the
            # carried-forward prev-round checkpoint), while the TARGET stays
            # what the orchestrator would have planned originally (=
            # starting_step + FINETUNE_STEPS — already computed above as
            # TARGET_STEPS using CURRENT_STEP = starting step). Without this
            # override, lerobot-train --resume reads checkpoint state from
            # `config_path/..`, which is the carried-forward checkpoint —
            # round r's existing weights would be overwritten by re-running
            # the same steps from scratch.
            if [[ "$EXTEND_LAST_ROUND" == "true" && "$r" == "${LAST_COMPLETE_ROUND:-0}" ]]; then
                _ext_ckpt="$(resolve_latest_checkpoint "$TRAIN_OUTPUT_DIR" 2>/dev/null || true)"
                if [[ -n "$_ext_ckpt" && -d "$_ext_ckpt" ]]; then
                    echo "Finetune: --extend_last_round active for round $r:"
                    echo "  Overriding resume checkpoint to round $r's existing $_ext_ckpt"
                    echo "  Target stays $TARGET_STEPS (= starting step ${CURRENT_STEP} + FINETUNE_STEPS=${FINETUNE_STEPS})"
                    CURRENT_POLICY="$_ext_ckpt"
                else
                    echo "WARNING: --extend_last_round set for round $r but resolve_latest_checkpoint returned empty." >&2
                    echo "  Falling back to the carried-forward CURRENT_POLICY (will re-run from scratch)." >&2
                fi
            fi
            # Cache the predicted next-round step for dry-run continuation.
            DRYRUN_NEXT_STEP="$TARGET_STEPS"
            echo "Finetune: resuming from step ${CURRENT_STEP} → target step ${TARGET_STEPS} (+${FINETUNE_STEPS})"
            # Sanity check: the orchestrator's --action_format must match the
            # resumed policy's use_relative_actions. Mismatch → wrong stats
            # normalization → silent policy corruption. Fail loudly here so the
            # user catches it before training kicks off rather than discovering
            # the policy got worse after the fact.
            #
            # Gate on the config file existing rather than DRY_RUN so dry-run
            # of round 1 (real --initial_policy_path) catches mismatches just
            # like a real run would; later-round dry-runs with synthetic paths
            # silently skip.
            # Guard `readlink -f` so the action-format sanity check is skipped
            # when the resumed policy dir doesn't exist on disk yet — which is
            # always true for round ≥ 2 in --dry-run (synthesized "what round
            # r-1 WOULD produce" path), and avoids `set -e` aborting the loop
            # before the [DRY-RUN] resume_training.sh line is printed.
            RESUME_CFG_PATH=""
            if [[ -e "$CURRENT_POLICY" ]]; then
                RESUME_CFG_PATH="$(readlink -f "$CURRENT_POLICY")/train_config.json"
            fi
            if [[ -f "$RESUME_CFG_PATH" ]]; then
                POLICY_REL="$(python3 -c "
import json,sys
c = json.load(open(sys.argv[1]))
print('true' if c.get('policy',{}).get('use_relative_actions') else 'false')
" "$RESUME_CFG_PATH")"
                EXPECTED_REL="false"
                [[ "$ACTION_FORMAT" == "rel" ]] && EXPECTED_REL="true"
                if [[ "$POLICY_REL" != "$EXPECTED_REL" ]]; then
                    echo "ERROR: --action_format=$ACTION_FORMAT but resumed policy has use_relative_actions=$POLICY_REL." >&2
                    echo "  Resumed policy:   $CURRENT_POLICY" >&2
                    echo "  Set --action_format=$([[ "$POLICY_REL" == "true" ]] && echo rel || echo abs) on the orchestrator." >&2
                    echo "  (Mismatch would re-normalize actions with the wrong sidecar and silently corrupt the policy.)" >&2
                    exit 1
                fi
            fi
            FT_BATCH_SIZE_ARG=()
            [[ -n "$FINETUNE_BATCH_SIZE" ]] && FT_BATCH_SIZE_ARG=(--batch_size="$FINETUNE_BATCH_SIZE")
            # Force a CONSTANT peak LR for the finetune. Mechanism depends on
            # the resumed policy's scheduler config:
            #
            # - cosine_decay_with_warmup (pi05/pi0): set decay_lr = peak_lr.
            #   The lambda blends toward decay_lr at end-of-decay; equating
            #   the two gives a flat peak. Default to auto-detected
            #   optimizer_lr from train_config.json; override with
            #   --finetune_decay_lr=<value>.
            # - diffuser/cosine (diffusion): the HF cosine scheduler decays
            #   to 0 by num_training_steps and is useless past the original
            #   end. Switch to scheduler.name=constant which returns a
            #   constant lambda=1, giving lr = base_lr (peak from the saved
            #   optimizer's initial_lr) every step.
            #
            # Both knobs default to "flatten at peak"; an explicit
            # --finetune_decay_lr=<value> still wins for cosine_decay_with_warmup.
            EFFECTIVE_DECAY_LR="$FINETUNE_DECAY_LR"
            FT_SCHEDULER_NAME_ARG=()
            # Gate detection on the train_config.json existing rather than on
            # DRY_RUN. This way the round-1 dry-run (which has a real
            # --initial_policy_path on disk) shows the auto-set scheduler
            # flag; later-round dry-runs with synthetic paths fall through
            # silently because the file isn't there yet.
            # Always recompute from the current CURRENT_POLICY — bash variables
            # persist across loop iterations and we'd otherwise reuse round
            # (r-1)'s config path in round r.
            # Guard `readlink -f` so the action-format sanity check is skipped
            # when the resumed policy dir doesn't exist on disk yet — which is
            # always true for round ≥ 2 in --dry-run (synthesized "what round
            # r-1 WOULD produce" path), and avoids `set -e` aborting the loop
            # before the [DRY-RUN] resume_training.sh line is printed.
            RESUME_CFG_PATH=""
            if [[ -e "$CURRENT_POLICY" ]]; then
                RESUME_CFG_PATH="$(readlink -f "$CURRENT_POLICY")/train_config.json"
            fi
            if [[ -f "$RESUME_CFG_PATH" ]]; then
                POLICY_SUPPORTS_DECAY_LR="$(python3 -c "
import json,sys
c = json.load(open(sys.argv[1]))
print('true' if 'scheduler_decay_lr' in c.get('policy',{}) else 'false')
" "$RESUME_CFG_PATH")"
                SCHEDULER_TYPE="$(python3 -c "
import json,sys
c = json.load(open(sys.argv[1]))
print((c.get('scheduler') or {}).get('type') or '')
" "$RESUME_CFG_PATH")"
                if [[ "$POLICY_SUPPORTS_DECAY_LR" != "true" ]]; then
                    if [[ -n "$EFFECTIVE_DECAY_LR" ]]; then
                        echo "WARNING: --finetune_decay_lr=$EFFECTIVE_DECAY_LR ignored — policy at $CURRENT_POLICY has no scheduler_decay_lr field (likely diffusion). Skipping override." >&2
                    fi
                    EFFECTIVE_DECAY_LR=""
                    # Diffuser scheduler (HF cosine that decays to 0 by end-of-training):
                    # switch to constant so the finetune runs at peak LR.
                    if [[ "$SCHEDULER_TYPE" == "diffuser" ]]; then
                        FT_SCHEDULER_NAME_ARG=(--scheduler.name=constant)
                        echo "Finetune: auto-set --scheduler.name=constant (diffuser scheduler decays to 0 at end-of-training; flat LR keeps the finetune effective)."
                    fi
                elif [[ -z "$EFFECTIVE_DECAY_LR" ]]; then
                    PEAK_LR="$(python3 -c "
import json,sys
c = json.load(open(sys.argv[1]))
print(c.get('policy',{}).get('optimizer_lr') or '')
" "$RESUME_CFG_PATH")"
                    if [[ -n "$PEAK_LR" ]]; then
                        EFFECTIVE_DECAY_LR="$PEAK_LR"
                        echo "Finetune: auto-set --scheduler_decay_lr=$EFFECTIVE_DECAY_LR (peak optimizer_lr from train_config.json — override with --finetune_decay_lr=<value>)."
                    fi
                fi
            elif [[ "$DRY_RUN" != true ]]; then
                # A real run with no train_config.json under CURRENT_POLICY is
                # a real problem — the resume itself will fail downstream. Warn
                # so the user notices it here rather than after the sim spins up.
                echo "WARNING: train_config.json not found at $RESUME_CFG_PATH; skipping scheduler/decay-lr auto-detection." >&2
            fi
            FT_DECAY_LR_ARG=()
            [[ -n "$EFFECTIVE_DECAY_LR" ]] && FT_DECAY_LR_ARG=(--scheduler_decay_lr="$EFFECTIVE_DECAY_LR")
            # The relative-action stats sidecar (stats_rel{N}.json, where N is
            # the policy's chunk size) is only correct for policies trained with
            # use_relative_actions=True. For absolute-action policies, overriding
            # stats_path to a relative sidecar re-normalizes absolute action
            # values against delta stats, which DESTROYS the action distribution
            # and corrupts the policy over even a few hundred finetune steps.
            # When ACTION_FORMAT=abs, omit the override so lerobot-train uses the
            # merged dataset's own meta/stats.json (computed at merge time with
            # absolute actions).
            #
            # Weighted mode also needs FT_CHUNK (to pick the right per-source
            # sidecar from $STATS_BASE/<short>/stats_rel${N}.json), so compute
            # it unconditionally when ACTION_FORMAT=rel.
            FT_STATS_PATH_ARG=()
            FT_CHUNK=""
            if [[ "$ACTION_FORMAT" == "rel" ]]; then
                # Read the chunk size straight off the resumed policy's
                # train_config.json (`policy.n_action_steps`) — that's the
                # authoritative source of truth for which sidecar to use, no
                # per-model hardcoding required.
                FT_CHUNK="$(n_action_steps_from_policy_path "$CURRENT_POLICY")"
                if [[ -z "$FT_CHUNK" ]]; then
                    if [[ "$DRY_RUN" == true ]]; then
                        # Dry-run for round >1 walks through synthetic policy
                        # paths whose train_config.json doesn't exist yet
                        # (the prior round wasn't actually trained). Use a
                        # placeholder so the dry-run can continue and the user
                        # sees the rest of the planned commands. The placeholder
                        # is wrapped in <> to make it obvious it's not a real
                        # path that would resolve.
                        FT_CHUNK="<unknown-in-dry-run>"
                        echo "  [dry-run] could not read policy.n_action_steps from $CURRENT_POLICY/train_config.json (round >1 dry-run); using <unknown-in-dry-run> placeholder in --dataset.stats_path."
                    else
                        echo "ERROR: could not read policy.n_action_steps from $CURRENT_POLICY/train_config.json" >&2
                        echo "  Needed to pick the right stats_rel{N}.json sidecar for finetune." >&2
                        exit 1
                    fi
                fi
            fi
            # ── Weighted-sampling mode: build the multi-dataset arg set ──
            # In weighted mode, lerobot-train consumes a list of sub-datasets
            # via --dataset.repo_ids plus parallel sample_weights / stats_paths.
            # We require rel actions for now — the stats_paths machinery in
            # lerobot.datasets.multi_source_normalizing_dataset reads
            # JSON sidecars at the paths we pass; for abs we'd need to point at
            # each repo's meta/stats.json instead, which works but isn't tested
            # for DAgger. Gate explicitly so the abs path doesn't silently
            # corrupt.
            FT_MULTI_DATASET_ARGS=()
            DAG_CFG_WEIGHTED_REPO_IDS_JSON="[]"
            DAG_CFG_WEIGHTED_WEIGHTS_JSON="[]"
            DAG_CFG_WEIGHTED_STATS_PATHS_JSON="[]"
            if [[ "$USE_WEIGHTED_SAMPLING" == "true" ]]; then
                if [[ "$ACTION_FORMAT" != "rel" ]]; then
                    echo "ERROR: --use_weighted_sampling currently requires --action_format=rel." >&2
                    echo "  Per-source stats sidecars (stats_rel{N}.json) drive normalization in" >&2
                    echo "  multi-dataset mode. Abs-action support would need to point stats_paths" >&2
                    echo "  at each repo's meta/stats.json instead, which isn't wired here yet." >&2
                    exit 1
                fi
                # Build parallel arrays: repo_ids[i], stats_paths[i], and the
                # final sample_weights[i]. Base is index 0; round 1..r contribute
                # 1 raw + n_blends entries each (in that order).
                W_REPO_IDS=( "$BASE_REPO" )
                W_STATS_PATHS=( "$STATS_BASE/$BASE_REPO_DATASET_SHORT/stats_rel${FT_CHUNK}.json" )
                # Step 6 always trains on RAW blends. Step 6b (the sibling
                # nocoll policy, gated on --filter_blend_collisions) builds
                # its own NC variants below.
                for _p in $(seq 1 "$r"); do
                    _p_int_repo="$(int_repo_for_round "$_p")"
                    _p_int_short="$(int_short_for_round "$_p")"
                    W_REPO_IDS+=( "$_p_int_repo" )
                    W_STATS_PATHS+=( "$STATS_BASE/$_p_int_short/stats_rel${FT_CHUNK}.json" )
                    for _R in "${BLENDS[@]}"; do
                        _b_repo="$(blend_repo_for_round "$_p" "$_R")"
                        _b_short="$(blend_short_for_round "$_p" "$_R")"
                        W_REPO_IDS+=( "$_b_repo" )
                        W_STATS_PATHS+=( "$STATS_BASE/$_b_short/stats_rel${FT_CHUNK}.json" )
                    done
                done
                # Two-level allotment:
                #   * Base gets a fixed (1 - f) share regardless of frame counts.
                #   * The remaining f is split EQUALLY across the R DAgger
                #     rounds (each round gets f / R).
                #   * Within each round, the round's allotment is split across
                #     its sub-datasets (raw intervention + each blend variant)
                #     PROPORTIONAL to their frame counts. This matters when
                #     --filter_blend_collisions trims some blend datasets so
                #     they're smaller than their raw sibling — without the
                #     proportional split, an empty/tiny blend would still get
                #     the same per-sub share as the much-larger raw intervention.
                #
                # Frame counts are read from the LeRobot dataset cache at
                # ~/.cache/huggingface/lerobot/<repo>/meta/info.json. By the
                # time this runs (orchestrator step 6), step 5 has already
                # computed stats over every sub-dataset, so they're all on
                # disk locally.
                _n_dag=$(( ${#W_REPO_IDS[@]} - 1 ))
                _n_blends=${#BLENDS[@]}
                read -ra W_WEIGHTS <<< "$(python3 -c "
import os, sys, json

f = float(sys.argv[1])
R = int(sys.argv[2])           # current round number (1..NUM_ROUNDS)
group_size = 1 + int(sys.argv[3])  # 1 raw int + n_blends sub-datasets per round
# In dry-run mode, datasets that step 2 would produce aren't actually on disk.
# Fall back to equal-share weights so the printed command preview is still
# usable. The real run re-derives proportional weights once step 2 has
# materialized every nocoll sibling on disk.
dry_run = sys.argv[4] == 'true'
repo_ids = sys.argv[5:]
# Layout: [base, r1_int, r1_blend_0, ..., r1_blend_(b-1), r2_int, ..., rR_blend_(b-1)]
assert len(repo_ids) == 1 + R * group_size, (
    f'repo_ids layout mismatch: got {len(repo_ids)}, expected 1 + {R}*{group_size}'
)

CACHE = os.path.expanduser('~/.cache/huggingface/lerobot')
def frames_for(repo):
    p = os.path.join(CACHE, repo, 'meta', 'info.json')
    if not os.path.isfile(p):
        if dry_run:
            return 1  # placeholder so dry-run can still print a sample_weights list
        raise FileNotFoundError(f'meta/info.json not found for {repo} at {p}')
    return int(json.load(open(p))['total_frames'])

per_round = f / R
weights = [1.0 - f]
for r_i in range(R):
    group = repo_ids[1 + r_i*group_size : 1 + (r_i+1)*group_size]
    counts = [frames_for(rep) for rep in group]
    total = sum(counts)
    if total <= 0:
        raise ValueError(f'round {r_i+1}: all sub-datasets are empty ({list(zip(group, counts))})')
    for c in counts:
        weights.append(per_round * c / total)
# Force exact sum=1 by absorbing FP error into the last non-zero weight
# (avoid the last position if it's a 0-frame sub-dataset — the validator
# would still accept it but the log would show a weird non-zero weight
# on what's logically an empty source).
for i in range(len(weights) - 1, -1, -1):
    if weights[i] > 0:
        weights[i] = 1.0 - sum(weights[:i]) - sum(weights[i+1:])
        break
print(' '.join(f'{w:.10f}' for w in weights))
" "$DAGGER_DATA_FRACTION" "$r" "$_n_blends" "$DRY_RUN" "${W_REPO_IDS[@]}")"
                # Format draccus list strings as JSON arrays — draccus accepts
                # both [a,b,c] and ['a','b','c']; we go with quoted/JSON so paths
                # with slashes (which draccus might otherwise mishandle) round-trip
                # cleanly. python3's json.dumps gets the escaping right.
                #
                # IMPORTANT: use separators=(",", ":") so the output contains NO
                # spaces. The downstream `resume_training.sh` concatenates these
                # into a flat $CMD string and runs it via `eval "$CMD"`, which
                # word-splits on whitespace. Default `json.dumps` emits `, ` (with
                # a space) which would split `[A,` `B]` apart and break draccus
                # parsing. Compact form (`[A,B]`) survives the eval intact.
                DAG_CFG_WEIGHTED_REPO_IDS_JSON=$(python3 -c "import json,sys; print(json.dumps(sys.argv[1:], separators=(',', ':')))" "${W_REPO_IDS[@]}")
                DAG_CFG_WEIGHTED_STATS_PATHS_JSON=$(python3 -c "import json,sys; print(json.dumps(sys.argv[1:], separators=(',', ':')))" "${W_STATS_PATHS[@]}")
                DAG_CFG_WEIGHTED_WEIGHTS_JSON=$(python3 -c "import json,sys; print(json.dumps([float(x) for x in sys.argv[1:]], separators=(',', ':')))" "${W_WEIGHTS[@]}")
                # Build the weighted-mode dataset args. The wrapper
                # (MultiSourceNormalizingDataset) handles all stats selection
                # internally via --dataset.norm_mode; we deliberately DO NOT
                # also pass --dataset.stats_path here.
                #
                # CONTEXT (was a silent footgun): before --norm_mode existed,
                # this code unconditionally passed --dataset.stats_path=<BASE
                # sidecar>, which lerobot_train.py:254-259 applies AFTER the
                # wrapper finishes — clobbering the wrapper's aggregated
                # stats with base-only. The surrounding comment claimed
                # per-source _normalizers were still applied per-frame inside
                # the wrapper, but the wrapper was refactored to single-pass
                # aggregation and the per-source normalizers were removed.
                # Net effect: effective behavior was always "base_only" by
                # accident regardless of intent. That made intervention rows
                # with raw range > base's normalize to values WAY outside
                # [-1, 1] (e.g. +7.5 on int_dag2 dim 2), almost certainly
                # destabilizing finetune. The fix: stop passing
                # --dataset.stats_path here, let the wrapper expose
                # mode-appropriate stats:
                #   * norm_mode=aggregated → wrapper exposes
                #     min-of-mins / max-of-maxes across all sources.
                #   * norm_mode=base_only → wrapper exposes source[0] (base)
                #     stats only. Equivalent to the prior accidental
                #     behavior, but now an explicit opt-in.
                # IMPORTANT: pass --dataset.stats_path= (EMPTY STRING) to
                # CLEAR the value baked into the resumed train_config.json.
                # The base policy was trained against a single dataset with
                # `dataset.stats_path` pointing at the base sidecar — that
                # value persists across `--config_path=...` loads and
                # triggers `lerobot_train.py:254-259`'s override of
                # `dataset.meta.stats` AFTER the wrapper finishes. Even
                # though we no longer EXPLICITLY pass --dataset.stats_path=...
                # here, draccus still uses the inherited config value if we
                # don't override it. The fix is the same idiom we use for
                # `--dataset.repo_id=` (cleared so it doesn't conflict with
                # `--dataset.repo_ids`): pass empty to disable the override
                # and let the wrapper's mode-appropriate stats stand.
                FT_MULTI_DATASET_ARGS=(
                    --dataset.repo_ids="$DAG_CFG_WEIGHTED_REPO_IDS_JSON"
                    --dataset.sample_weights="$DAG_CFG_WEIGHTED_WEIGHTS_JSON"
                    --dataset.stats_paths="$DAG_CFG_WEIGHTED_STATS_PATHS_JSON"
                    --dataset.norm_mode="$NORM_MODE"
                    --dataset.stats_path=
                )
                echo "Weighted sampling: ${#W_REPO_IDS[@]} sub-datasets (1 base + $_n_dag DAgger); per-round share=$(python3 -c "print(f'{float(\"$DAGGER_DATA_FRACTION\")/$r:.4f}')") (frame-proportional within round); weights=${W_WEIGHTS[*]}; norm_mode=$NORM_MODE"
            elif [[ "$ACTION_FORMAT" == "rel" ]]; then
                # Merge-mode rel: point at the merged-dataset sidecar (default
                # legacy behavior).
                FT_STATS_PATH_ARG=( --dataset.stats_path="$MERGED_STATS_DIR/stats_rel${FT_CHUNK}.json" )
            fi
            # Force inline eval onto the benchmark dataset + the same subset
            # the intervention recording uses, regardless of what the resumed
            # train_config.json had baked in. Some checkpoints (especially
            # older ones, or diffusion runs that predate the benchmark wiring
            # in train_sweep.sh) save these as None and would otherwise have
            # their inline eval fall back to random scenarios — which breaks
            # round-over-round DAgger progress comparison.
            FT_EVAL_BENCHMARK_ARG=( --env.eval_benchmark_repo_id="$EVAL_BENCHMARK_REPO_ID" )
            if [[ -n "$INTERVENTION_SUBSET_JSON" ]]; then
                FT_EVAL_BENCHMARK_ARG+=( --env.eval_benchmark_subset="$INTERVENTION_SUBSET_JSON" )
            fi
            # In weighted mode, --dataset.repo_id MUST NOT be set (mutually
            # exclusive with --dataset.repo_ids per TrainPipelineConfig.validate).
            # The CATCH: the resumed train_config.json has `dataset.repo_id` baked
            # in from when the base policy was trained, so we must override it to
            # an EMPTY STRING (`--dataset.repo_id=`) to clear the loaded value —
            # not just omit the flag. The validator's check is on truthiness:
            # empty string == "not set", so `repo_ids` and an empty `repo_id`
            # coexist legally.
            # In merge mode, --dataset.repo_id points at the merged dataset (and
            # overwrites the loaded value with the new one).
            FT_REPO_ID_ARG=()
            if [[ "$USE_WEIGHTED_SAMPLING" == "true" ]]; then
                FT_REPO_ID_ARG=( --dataset.repo_id= )
            else
                FT_REPO_ID_ARG=( --dataset.repo_id="$MERGED_REPO" )
            fi
            # PURE_POLICY_MODE override: trumps the merge / weighted-sampling
            # data setup above. Resume training on the BASE dataset ONLY
            # (no intervention/blend merge, no multi-dataset weighted sampling)
            # so the resulting checkpoint is the matched-compute base-only
            # baseline for round r. All other args (target_steps, eval
            # benchmark + subset, scheduler / batch / port wiring) stay
            # identical to a normal finetune round so the comparison against
            # the DAgger lineages is apples-to-apples in compute terms.
            if [[ "$PURE_POLICY_MODE" == "true" ]]; then
                FT_REPO_ID_ARG=( --dataset.repo_id="$BASE_REPO" )
                FT_MULTI_DATASET_ARGS=()
                if [[ "$ACTION_FORMAT" == "rel" ]]; then
                    FT_STATS_PATH_ARG=( --dataset.stats_path="$STATS_BASE/$BASE_REPO_DATASET_SHORT/stats_rel${FT_CHUNK}.json" )
                else
                    FT_STATS_PATH_ARG=()
                fi
                echo "  [pure_policy] training data: $BASE_REPO (base only)"
            fi
            # Skip-if-already-at-target check: if TRAIN_OUTPUT_DIR's latest
            # checkpoint step >= TARGET_STEPS, lerobot-train would no-op
            # anyway (cfg.steps <= current_step exits at iteration 0). We
            # still pay the orchestrator overhead (sim start/stop, sidecar
            # mkdir, wandb init, etc.). Detect and skip the whole step 6
            # body so we proceed straight to step 6b. Common when the user
            # flipped on --filter_blend_collisions for an already-trained
            # lineage — resume detection demoted step=6 → 5 to force step 6b,
            # but step 6 itself has nothing to do.
            _skip_step_6=false
            if [[ -d "$TRAIN_OUTPUT_DIR" ]]; then
                _existing_ckpt="$(resolve_latest_checkpoint "$TRAIN_OUTPUT_DIR" 2>/dev/null || true)"
                if [[ -n "$_existing_ckpt" ]]; then
                    _existing_step="$(readlink -f "$_existing_ckpt" 2>/dev/null \
                        | grep -oE 'checkpoints/[0-9]+/' | head -1 \
                        | grep -oE '[0-9]+' || true)"
                    if [[ -n "$_existing_step" ]] && (( 10#$_existing_step >= TARGET_STEPS )); then
                        echo "  [step 6 skip] $TRAIN_OUTPUT_DIR already at step $_existing_step >= target $TARGET_STEPS; skipping the resume_training.sh invocation (would no-op anyway)."
                        _skip_step_6=true
                    fi
                fi
            fi
            # shellcheck disable=SC2086  # FINETUNE_EXTRA_ARGS is word-split intentionally
            if [[ "$_skip_step_6" != true ]]; then
            run_training_step bash "$SCRIPT_DIR/resume_training.sh" "$CURRENT_POLICY" \
                "${FT_REPO_ID_ARG[@]}" \
                "${FT_MULTI_DATASET_ARGS[@]}" \
                "${FT_STATS_PATH_ARG[@]}" \
                --policy.repo_id="$FT_RUN_NAME" \
                --output_dir="$TRAIN_OUTPUT_DIR" \
                --job_name="$FT_RUN_NAME" \
                --steps="$TARGET_STEPS" \
                --eval_freq="$FINETUNE_EVAL_FREQ" \
                --save_freq="$FINETUNE_SAVE_FREQ" \
                "${TRAIN_EXT_PORT_RESUME[@]}" \
                "${FT_BATCH_SIZE_ARG[@]}" \
                "${FT_DECAY_LR_ARG[@]}" \
                "${FT_SCHEDULER_NAME_ARG[@]}" \
                "${FT_EVAL_BENCHMARK_ARG[@]}" \
                "${HEADLESS_TRAIN_ARGS[@]}" \
                "${OFFLINE_POLICY_ARG[@]}" \
                $FINETUNE_EXTRA_ARGS \
                --eval.n_episodes="$EVAL_N_EPISODES"
            fi  # end of _skip_step_6 != true
        fi
        # Write the per-round config sidecar AFTER training succeeds (or
        # was skipped because already at target). The sidecar reflects the
        # CURRENT orchestrator invocation, not training-time history, so
        # writing it on a skip is correct.
        write_dagger_config_sidecar "$r" "$TRAIN_OUTPUT_DIR/dagger/config.json" "$TRAIN_OUTPUT_DIR"
        # ── Step 6b: collision-filtered sibling policy ──────────────────
        # When --filter_blend_collisions is on, train a SECOND policy this
        # round using `_nocoll` siblings in place of raw blends. Output goes
        # to <TRAIN_OUTPUT_DIR>_nc so the raw policy stays untouched.
        # Both policies finetune from the SAME starting checkpoint
        # (CURRENT_POLICY at this point in the loop = round r's input —
        # CURRENT_POLICY isn't bumped until after this block), so the two
        # outputs are a clean per-round A/B test. Gated on:
        #   * --filter_blend_collisions   (the user opted in)
        #   * --use_weighted_sampling      (multi-dataset arg shape required)
        #   * non-empty BLENDS             (nothing to filter otherwise)
        #   * MODE=finetune                (we only wire the finetune path;
        #                                   scratch mode for the sibling
        #                                   would need train_sweep.sh
        #                                   plumbing that isn't needed yet)
        #   * non-PURE_POLICY_MODE         (no DAgger data in pure mode)
        if [[ "$FILTER_BLEND_COLLISIONS" == "true" ]] && \
           [[ "$USE_WEIGHTED_SAMPLING" == "true" ]] && \
           [[ "$MODE" == "finetune" ]] && \
           [[ "$PURE_POLICY_MODE" != "true" ]] && \
           (( ${#BLENDS[@]} > 0 )); then
            NOCOLL_TRAIN_OUTPUT_DIR="$(nocoll_train_output_dir_for_round "$r")"
            NOCOLL_RUN_NAME="$(nocoll_run_name_for_round "$r")"
            if training_exists "$NOCOLL_TRAIN_OUTPUT_DIR"; then
                echo "--- Round $r, Step 6b: nocoll sibling policy already exists at $NOCOLL_TRAIN_OUTPUT_DIR; skipping ---"
            else
                echo "--- Round $r, Step 6b: train collision-filtered sibling policy (finetune) → $NOCOLL_TRAIN_OUTPUT_DIR ---"
                # Build NC variants of the multi-dataset arrays. Same layout
                # as step 6's W_REPO_IDS, but raw blend repos are replaced by
                # their `_nocoll` siblings.
                NC_REPO_IDS=( "$BASE_REPO" )
                NC_STATS_PATHS=( "$STATS_BASE/$BASE_REPO_DATASET_SHORT/stats_rel${FT_CHUNK}.json" )
                for _p in $(seq 1 "$r"); do
                    _p_int_repo="$(int_repo_for_round "$_p")"
                    _p_int_short="$(int_short_for_round "$_p")"
                    NC_REPO_IDS+=( "$_p_int_repo" )
                    NC_STATS_PATHS+=( "$STATS_BASE/$_p_int_short/stats_rel${FT_CHUNK}.json" )
                    for _R in "${BLENDS[@]}"; do
                        _nc_repo="$(nocoll_repo_for_round "$_p" "$_R")"
                        _nc_short="$(nocoll_short_for_round "$_p" "$_R")"
                        NC_REPO_IDS+=( "$_nc_repo" )
                        NC_STATS_PATHS+=( "$STATS_BASE/$_nc_short/stats_rel${FT_CHUNK}.json" )
                    done
                done
                _nc_n_blends=${#BLENDS[@]}
                # Reuse step 6's proportional-weight algorithm — same
                # python script, just fed the NC arrays.
                read -ra NC_WEIGHTS <<< "$(python3 -c "
import os, sys, json
f = float(sys.argv[1])
R = int(sys.argv[2])
group_size = 1 + int(sys.argv[3])
dry_run = sys.argv[4] == 'true'
repo_ids = sys.argv[5:]
assert len(repo_ids) == 1 + R * group_size, (
    f'repo_ids layout mismatch: got {len(repo_ids)}, expected 1 + {R}*{group_size}'
)
CACHE = os.path.expanduser('~/.cache/huggingface/lerobot')
def frames_for(repo):
    p = os.path.join(CACHE, repo, 'meta', 'info.json')
    if not os.path.isfile(p):
        if dry_run:
            return 1
        raise FileNotFoundError(f'meta/info.json not found for {repo} at {p}')
    return int(json.load(open(p))['total_frames'])
per_round = f / R
weights = [1.0 - f]
for r_i in range(R):
    group = repo_ids[1 + r_i*group_size : 1 + (r_i+1)*group_size]
    counts = [frames_for(rep) for rep in group]
    total = sum(counts)
    if total <= 0:
        raise ValueError(f'round {r_i+1}: all sub-datasets are empty ({list(zip(group, counts))})')
    for c in counts:
        weights.append(per_round * c / total)
for i in range(len(weights) - 1, -1, -1):
    if weights[i] > 0:
        weights[i] = 1.0 - sum(weights[:i]) - sum(weights[i+1:])
        break
print(' '.join(f'{w:.10f}' for w in weights))
" "$DAGGER_DATA_FRACTION" "$r" "$_nc_n_blends" "$DRY_RUN" "${NC_REPO_IDS[@]}")"
                NC_REPO_IDS_JSON=$(python3 -c "import json,sys; print(json.dumps(sys.argv[1:], separators=(',', ':')))" "${NC_REPO_IDS[@]}")
                NC_STATS_PATHS_JSON=$(python3 -c "import json,sys; print(json.dumps(sys.argv[1:], separators=(',', ':')))" "${NC_STATS_PATHS[@]}")
                NC_WEIGHTS_JSON=$(python3 -c "import json,sys; print(json.dumps([float(x) for x in sys.argv[1:]], separators=(',', ':')))" "${NC_WEIGHTS[@]}")
                echo "  [step 6b] Weighted sampling (nocoll): ${#NC_REPO_IDS[@]} sub-datasets; per-round share=$(python3 -c "print(f'{float(\"$DAGGER_DATA_FRACTION\")/$r:.4f}')"); weights=${NC_WEIGHTS[*]}"
                # shellcheck disable=SC2086  # FINETUNE_EXTRA_ARGS is word-split intentionally
                run_training_step bash "$SCRIPT_DIR/resume_training.sh" "$CURRENT_POLICY" \
                    --dataset.repo_id= \
                    --dataset.repo_ids="$NC_REPO_IDS_JSON" \
                    --dataset.sample_weights="$NC_WEIGHTS_JSON" \
                    --dataset.stats_paths="$NC_STATS_PATHS_JSON" \
                    --dataset.norm_mode="$NORM_MODE" \
                    --dataset.stats_path= \
                    --policy.repo_id="$NOCOLL_RUN_NAME" \
                    --output_dir="$NOCOLL_TRAIN_OUTPUT_DIR" \
                    --job_name="$NOCOLL_RUN_NAME" \
                    --steps="$TARGET_STEPS" \
                    --eval_freq="$FINETUNE_EVAL_FREQ" \
                    --save_freq="$FINETUNE_SAVE_FREQ" \
                    "${TRAIN_EXT_PORT_RESUME[@]}" \
                    "${FT_BATCH_SIZE_ARG[@]}" \
                    "${FT_DECAY_LR_ARG[@]}" \
                    "${FT_SCHEDULER_NAME_ARG[@]}" \
                    "${FT_EVAL_BENCHMARK_ARG[@]}" \
                    "${HEADLESS_TRAIN_ARGS[@]}" \
                    "${OFFLINE_POLICY_ARG[@]}" \
                    $FINETUNE_EXTRA_ARGS \
                    --eval.n_episodes="$EVAL_N_EPISODES"
                write_dagger_config_sidecar "$r" "$NOCOLL_TRAIN_OUTPUT_DIR/dagger/config.json" "$NOCOLL_TRAIN_OUTPUT_DIR"
            fi
        fi
        # Relaunch the external sim now that training has freed the GPU,
        # so the next round's intervention recording can use it. Gated on
        # EFFECTIVE_END_ROUND (not NUM_ROUNDS) so retrain mode — where the
        # loop ends early at RETRAIN_ROUND — doesn't waste a sim start
        # after the final training in the loop.
        if (( r < EFFECTIVE_END_ROUND )); then
            start_sim
        fi
    fi

    # Roll the policy forward for the next round.
    if [[ "$DRY_RUN" == true ]]; then
        CURRENT_POLICY="$TRAIN_OUTPUT_DIR/checkpoints/last/pretrained_model"
    else
        CURRENT_POLICY="$(resolve_latest_checkpoint "$TRAIN_OUTPUT_DIR")"
    fi

    # wandb artifact cache grows unboundedly across runs (each finetune logs
    # a fresh policy checkpoint artifact). Cap it at 5GB at the end of every
    # round so a long DAgger run doesn't slowly fill the disk. Best-effort:
    # silently skipped if wandb CLI isn't installed.
    if [[ "$DRY_RUN" != true ]] && command -v wandb >/dev/null 2>&1; then
        echo "--- Round $r, Step 7: wandb artifact cache cleanup (cap 5GB) ---"
        wandb artifact cache cleanup 5GB 2>&1 | sed 's/^/  /' || true
    fi

    # HuggingFace datasets-library parquet build cache: lerobot-edit-dataset's
    # merge operation calls datasets.load_dataset under the hood, which caches
    # each merge's parquet view under ~/.cache/huggingface/datasets/parquet/
    # in a hash-keyed directory. Every round's merge creates a fresh ~30 GB
    # entry, and these are NEVER cleaned up by the merge tooling itself.
    # Across N rounds that's ~30N GB of pure waste in the global HF cache,
    # independent of the per-round merged-dataset rm we do under
    # ~/.cache/huggingface/lerobot/. Nuke them at the end of each round.
    # Safe — this is a build cache; rebuilt on demand by the next merge.
    HF_PARQUET_CACHE="$HOME/.cache/huggingface/datasets/parquet"
    if [[ "$DRY_RUN" != true && -d "$HF_PARQUET_CACHE" ]]; then
        BEFORE_SIZE="$(du -sh "$HF_PARQUET_CACHE" 2>/dev/null | awk '{print $1}')"
        echo "--- Round $r, Step 7b: clean HF datasets parquet cache (was $BEFORE_SIZE) ---"
        rm -rf "$HF_PARQUET_CACHE"/*
    fi

    # Refresh the round-over-round progress table + plot. Best-effort: never
    # fails the orchestrator. Restricted to this run's lineage so multi-lineage
    # machines don't slow the call down scanning unrelated dirs.
    if [[ "$DRY_RUN" != true ]]; then
        echo "--- Round $r, Step 7c: refresh dagger_progress table + plot ---"
        bash "$SCRIPT_DIR/dagger_progress.sh" \
            --filter="$PROGRESS_LINEAGE_FILTER" \
            --model="$TRAIN_OUTPUT_MODEL_PREFIX" 2>&1 | sed 's/^/  /' || true
    fi
done

# ── post-loop: optional final from-scratch train on round N's merged data ─────
# Motivation: when --intermediate_mode=finetune the N dag rounds give you a
# monotonic round-over-round curve (each step finetunes from the previous),
# but the final deployable policy is a finetune over a long sequence of
# weight updates. Training one extra model from scratch on the same final
# merged dataset gives a clean "deployable policy on full data" reference
# that's directly comparable to the finetune curve.
FINAL_POLICY_PATH="$CURRENT_POLICY"  # default: last finetune round
# In retrain mode skip the post-loop final-scratch phase. It would either
# re-train the existing final-scratch from scratch (clobbering it) or no-op
# (already complete) — neither matches user intent, which is "retune round N".
if [[ -n "$RETRAIN_ROUND" ]] && do_final_scratch; then
    echo
    echo "Post-loop final-scratch phase: SKIPPED (in --retrain_round mode)."
fi
if [[ -z "$RETRAIN_ROUND" ]] && do_final_scratch; then
    FINAL_SCRATCH_DIR="$(train_output_dir_final_scratch)"
    FINAL_SCRATCH_RUN_NAME="$(basename "$FINAL_SCRATCH_DIR")"
    LAST_MERGED_REPO="$(merged_repo_for_round "$NUM_ROUNDS")"

    if training_exists "$FINAL_SCRATCH_DIR"; then
        echo
        echo "════════════════════════════════════════════════════════════════"
        echo "POST-LOOP FINAL SCRATCH: already complete at $FINAL_SCRATCH_DIR"
        echo "════════════════════════════════════════════════════════════════"
        if [[ "$DRY_RUN" == true ]]; then
            FINAL_POLICY_PATH="$FINAL_SCRATCH_DIR/checkpoints/last/pretrained_model"
        else
            FINAL_POLICY_PATH="$(resolve_latest_checkpoint "$FINAL_SCRATCH_DIR")"
        fi
    else
        echo
        echo "════════════════════════════════════════════════════════════════"
        if [[ "$USE_WEIGHTED_SAMPLING" == "true" ]]; then
            echo "POST-LOOP FINAL SCRATCH: train from scratch on weighted union of {base + every round's intervention + every round's blends}"
            echo "  Training output: $FINAL_SCRATCH_DIR"
        else
            echo "POST-LOOP FINAL SCRATCH: train from scratch on round $NUM_ROUNDS's merged data"
            echo "  Dataset:         $LAST_MERGED_REPO"
            echo "  Training output: $FINAL_SCRATCH_DIR"
        fi
        echo "════════════════════════════════════════════════════════════════"

        # train_sweep.sh manages its own sim if --no_manage_splatsim is set;
        # otherwise stop the orchestrator-managed sim so lerobot-train can
        # spawn its own in-process eval sim.
        stop_sim
        print_gpu_state

        # Sanity: the merged dataset must exist. The stale-merged sweep above
        # keeps round NUM_ROUNDS's merged when EFFECTIVE_START_ROUND > NUM_ROUNDS,
        # but if the user manually deleted it the final-scratch step would silently
        # operate on a missing dataset — bail with a clear message instead.
        # Skipped entirely in weighted-sampling mode: there's no per-round
        # merged dataset (step 4 is unconditionally skipped); the final-scratch
        # train consumes the per-source union via --multi_dataset_* passthrough
        # args directly.
        if [[ "$USE_WEIGHTED_SAMPLING" != "true" ]]; then
            if [[ "$DRY_RUN" != true && ! -f "$LEROBOT_CACHE/$LAST_MERGED_REPO/meta/info.json" ]]; then
                echo "ERROR: final-scratch step needs $LAST_MERGED_REPO but it's not on disk." >&2
                echo "  Path checked: $LEROBOT_CACHE/$LAST_MERGED_REPO" >&2
                echo "  Re-run round $NUM_ROUNDS's step 4 (merge) to regenerate it." >&2
                exit 1
            fi
        fi

        ABS_ACTION_ARG=()
        if [[ "$ACTION_FORMAT" == "abs" ]]; then
            ABS_ACTION_ARG=( --no_relative )
        fi

        # Weighted mode: build the per-source W_* arrays (mirrors the per-round
        # step-6 finetune logic, but with round=NUM_ROUNDS so every round's
        # interventions + blends are included) and pass them to train_sweep.sh
        # via the --multi_dataset_* passthrough flags. The single-dataset
        # --dataset_repo flag is dropped in this branch (train_sweep.sh
        # ignores it when in multi-dataset mode and requires --run_name= to
        # be explicit).
        MULTI_DATASET_ARGS=()
        if [[ "$USE_WEIGHTED_SAMPLING" == "true" ]]; then
            if [[ "$ACTION_FORMAT" != "rel" ]]; then
                echo "ERROR: --use_weighted_sampling + --final_mode=scratch currently requires --action_format=rel" >&2
                echo "  (per-source sidecars are stats_rel{N}.json; abs path isn't wired)." >&2
                exit 1
            fi
            # FT_CHUNK source-of-truth for the from-scratch case: read off the
            # base policy's n_action_steps (the policy we're comparing against
            # was trained with this chunk; the from-scratch run will use the
            # same). For finetune rounds, FT_CHUNK is read from the resumed
            # policy — that's not available here, so go to the base directly.
            FS_CHUNK="$(n_action_steps_from_policy_path "$INITIAL_POLICY_PATH")"
            if [[ -z "$FS_CHUNK" ]]; then
                if [[ "$DRY_RUN" == true ]]; then
                    FS_CHUNK="<unknown-in-dry-run>"
                else
                    echo "ERROR: could not read policy.n_action_steps from $INITIAL_POLICY_PATH/train_config.json" >&2
                    echo "  Needed to pick the right stats_rel{N}.json sidecar for the multi-source training." >&2
                    exit 1
                fi
            fi
            FS_W_REPO_IDS=( "$BASE_REPO" )
            FS_W_STATS_PATHS=( "$STATS_BASE/$BASE_REPO_DATASET_SHORT/stats_rel${FS_CHUNK}.json" )
            for _p in $(seq 1 "$NUM_ROUNDS"); do
                _p_int_repo="$(int_repo_for_round "$_p")"
                _p_int_short="$(int_short_for_round "$_p")"
                FS_W_REPO_IDS+=( "$_p_int_repo" )
                FS_W_STATS_PATHS+=( "$STATS_BASE/$_p_int_short/stats_rel${FS_CHUNK}.json" )
                for _R in "${BLENDS[@]}"; do
                    if [[ "$FILTER_BLEND_COLLISIONS" == "true" ]]; then
                        _b_repo="$(nocoll_repo_for_round "$_p" "$_R")"
                        _b_short="$(nocoll_short_for_round "$_p" "$_R")"
                    else
                        _b_repo="$(blend_repo_for_round "$_p" "$_R")"
                        _b_short="$(blend_short_for_round "$_p" "$_R")"
                    fi
                    FS_W_REPO_IDS+=( "$_b_repo" )
                    FS_W_STATS_PATHS+=( "$STATS_BASE/$_b_short/stats_rel${FS_CHUNK}.json" )
                done
            done
            _fs_n_dag=$(( ${#FS_W_REPO_IDS[@]} - 1 ))
            read -ra FS_W_WEIGHTS <<< "$(python3 -c "
import sys
f = float(sys.argv[1])
n = int(sys.argv[2])
base = 1.0 - f
per = f / n
weights = [base] + [per] * n
weights[-1] = 1.0 - sum(weights[:-1])
print(' '.join(f'{w:.10f}' for w in weights))
" "$DAGGER_DATA_FRACTION" "$_fs_n_dag")"
            FS_REPO_IDS_JSON=$(python3 -c "import json,sys; print(json.dumps(sys.argv[1:], separators=(',', ':')))" "${FS_W_REPO_IDS[@]}")
            FS_STATS_PATHS_JSON=$(python3 -c "import json,sys; print(json.dumps(sys.argv[1:], separators=(',', ':')))" "${FS_W_STATS_PATHS[@]}")
            FS_WEIGHTS_JSON=$(python3 -c "import json,sys; print(json.dumps([float(x) for x in sys.argv[1:]], separators=(',', ':')))" "${FS_W_WEIGHTS[@]}")
            MULTI_DATASET_ARGS=(
                --multi_dataset_repo_ids="$FS_REPO_IDS_JSON"
                --multi_dataset_sample_weights="$FS_WEIGHTS_JSON"
                --multi_dataset_stats_paths="$FS_STATS_PATHS_JSON"
                --multi_dataset_norm_mode="$NORM_MODE"
            )
            echo "Final-scratch weighted sampling: ${#FS_W_REPO_IDS[@]} sub-datasets (1 base + $_fs_n_dag DAgger); weights=${FS_W_WEIGHTS[*]}; norm_mode=$NORM_MODE"
        fi

        TRAIN_OUTPUT_DIR="$FINAL_SCRATCH_DIR"   # consumed by run_training_step
        cleanup_pre_train_partial "$TRAIN_OUTPUT_DIR"
        # Forward eval scope + subset to the from-scratch step so its inline
        # eval covers the SAME scenarios at the SAME count as the per-round
        # finetune evals — previously hardcoded at 5 in train_sweep.sh, making
        # final-scratch + finetune rounds non-comparable in `dagger_progress`.
        FS_EVAL_ARGS=( --eval_n_episodes="$EVAL_N_EPISODES" )
        if [[ -n "$INTERVENTION_SUBSET_JSON" ]]; then
            FS_EVAL_ARGS+=( --eval_benchmark_subset="$INTERVENTION_SUBSET_JSON" )
        fi
        if [[ "$USE_WEIGHTED_SAMPLING" == "true" ]]; then
            # --dataset_repo intentionally OMITTED — train_sweep.sh defaults
            # it to the long-since-unused placeholder when multi-dataset args
            # are set, and the multi-mode validation in train_sweep.sh
            # ignores its value anyway.
            run_training_step bash "$SCRIPT_DIR/train_sweep.sh" \
                --run_name="$FINAL_SCRATCH_RUN_NAME" \
                --model="$TRAIN_OUTPUT_MODEL_PREFIX" \
                "${ABS_ACTION_ARG[@]}" \
                "${TRAIN_EXT_PORT_SWEEP[@]}" \
                "${HEADLESS_TRAIN_SCRATCH_ARGS[@]}" \
                "${OFFLINE_POLICY_ARG[@]}" \
                "${MULTI_DATASET_ARGS[@]}" \
                "${FS_EVAL_ARGS[@]}"
        else
            run_training_step bash "$SCRIPT_DIR/train_sweep.sh" \
                --dataset_repo="$LAST_MERGED_REPO" \
                --run_name="$FINAL_SCRATCH_RUN_NAME" \
                --model="$TRAIN_OUTPUT_MODEL_PREFIX" \
                "${ABS_ACTION_ARG[@]}" \
                "${TRAIN_EXT_PORT_SWEEP[@]}" \
                "${HEADLESS_TRAIN_SCRATCH_ARGS[@]}" \
                "${OFFLINE_POLICY_ARG[@]}" \
                "${FS_EVAL_ARGS[@]}"
        fi
        # Sidecar written AFTER training (not before) — train_sweep.sh
        # invokes lerobot-train without --resume, which errors on any
        # pre-existing output dir; pre-mkdir would induce the failure.
        write_dagger_config_sidecar "$NUM_ROUNDS" "$FINAL_SCRATCH_DIR/dagger/config.json" "$FINAL_SCRATCH_DIR"

        if [[ "$DRY_RUN" == true ]]; then
            FINAL_POLICY_PATH="$FINAL_SCRATCH_DIR/checkpoints/last/pretrained_model"
        else
            FINAL_POLICY_PATH="$(resolve_latest_checkpoint "$FINAL_SCRATCH_DIR")"
        fi

        # Post-final-scratch cleanup: round N's merged dataset is no longer
        # needed (final-scratch consumed it, no further training will). Mirrors
        # the per-round pre-train cleanup that deletes dag{r-1}_m.
        # No-op in weighted-sampling mode: there's no merged dataset to clean
        # (step 4 was skipped on every round; final-scratch consumed the
        # per-source union directly via --multi_dataset_* passthroughs).
        if [[ "$USE_WEIGHTED_SAMPLING" != "true" ]]; then
            echo "--- Post-loop cleanup: remove round $NUM_ROUNDS's merged dataset ---"
            run_or_echo rm -rf "$LEROBOT_CACHE/$LAST_MERGED_REPO"
            run_or_echo rm -rf "$STATS_BASE/$(merged_short_for_round "$NUM_ROUNDS")"
        fi

        if [[ "$DRY_RUN" != true ]] && command -v wandb >/dev/null 2>&1; then
            echo "--- Post-loop: wandb artifact cache cleanup (cap 5GB) ---"
            wandb artifact cache cleanup 5GB 2>&1 | sed 's/^/  /' || true
        fi

        if [[ "$DRY_RUN" != true ]]; then
            echo "--- Post-loop: refresh dagger_progress table + plot ---"
            bash "$SCRIPT_DIR/dagger_progress.sh" \
                --filter="$PROGRESS_LINEAGE_FILTER" \
                --model="$TRAIN_OUTPUT_MODEL_PREFIX" 2>&1 | sed 's/^/  /' || true
        fi
    fi
fi

# ── final summary ─────────────────────────────────────────────────────────────
echo
echo "════════════════════════════════════════════════════════════════"
echo "DAgger orchestration complete!"
echo "════════════════════════════════════════════════════════════════"
echo "Per-round intervention datasets:"
for r in $(seq 1 "$NUM_ROUNDS"); do
    echo "  Round $r: $(int_repo_for_round "$r")"
done
echo
echo "Final merged training dataset:  $(merged_repo_for_round "$NUM_ROUNDS")"
echo "Last finetune-round policy:     $CURRENT_POLICY"
if do_final_scratch && [[ -z "$RETRAIN_ROUND" ]]; then
    echo "Final-scratch policy:           $FINAL_POLICY_PATH"
fi
echo
echo "All per-round intervention datasets remain on disk for future"
echo "mix-and-match (their hardlink aliases under _${MODEL}${ACTION_FORMAT}00 also persist)."
