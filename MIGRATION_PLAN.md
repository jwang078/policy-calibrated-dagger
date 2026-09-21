# Moving the paper code out of the lerobot fork

Status: in progress since 2026-09-21; see the step list for what is done.

## Where things are today

| Place | What lives there | Size |
| --- | --- | --- |
| `lerobot` fork, `src/lerobot/` | 191 commits / 19.5k lines over upstream (branched 2026-08-07): DART relabelling, multi-source datasets, the shared-autonomy blend wrapper, guidance sources, the intervention controller, teleop recording, the SplatSim `EnvConfig`, plus hooks into `lerobot_train.py`, `lerobot_eval.py`, both factories, the diffusion model and the relative-action processor | 35 new files, 43 modified upstream files |
| `lerobot` fork, `my_scripts/` | the DAgger orchestrator and its helpers (bash), dataset augmentation (blend rollouts), calibration analyses, plotting, `paper_plots/` (figures + `tables_repro/`) | 56 scripts + 5 folders |
| `SplatSim` (yours, public) | the simulator; `splatsim/gym_env.py` (gymnasium wrapper), `utils/lerobot_utils.py` (dataset writing), `agents/lerobot_agent.py`, `utils/lerobot_splatsim_wrapper.py` | imports lerobot in 9 files |
| `policy-calibrated-dagger` (this repo) | LICENSE only | |

The problem: the method is invisible inside a fork of a 100k-line library, and the SplatSim environment is
wired into that fork's `envs/configs.py` instead of being something `lerobot-eval` can find on its own.

## Target layout

Same shape as SplatSim and lerobot: an importable package at the top, runnable scripts below it.

```
policy-calibrated-dagger/
├── pyproject.toml                  name = "pcdagger"; deps: lerobot (fork, pinned), splatsim, torch, scipy…
├── README.md                       the method, the two tables, how to reproduce
├── pcdagger/                       importable package
│   ├── dart/                       calibrated noise injection (the paper's method)
│   │   ├── relabel.py              ← lerobot/datasets/dart_relabel.py       (DartChunkDataset, chunk_labels, demo_geometry)
│   │   ├── schedules.py            ← paper_plots/tables_repro/analysis/build_*schedule*.py
│   │   ├── measure_sigma.py        ← analysis/measure_sigma_multi.py + measure_sigma_lever.py (one function, n_arm param)
│   │   └── measure_w.py            ← analysis/measure_w_lever.py + the planar W fit from fig_w_dial/plot.py
│   ├── blend/                      partial denoising as a learned interpolator
│   │   ├── wrapper.py              ← lerobot/policies/shared_autonomy_wrapper.py
│   │   ├── chunk_anchor.py         ← lerobot/policies/common/chunk_anchor.py
│   │   ├── rollout.py              ← my_scripts/lib_sa_rollout.py
│   │   ├── guidance/               ← lerobot/policies/guidance/*  (base, obs-teleop, RRT, oracle-goal)
│   │   └── augment.py              ← my_scripts/augment_dataset_with_blending.py (as a module + CLI)
│   ├── dagger/                     the interactive loop
│   │   ├── intervention.py         ← lerobot/scripts/intervention_controller.py + configs/intervention.py
│   │   ├── recording.py            ← lerobot/policies/teleop_recording.py
│   │   ├── naming.py               ← my_scripts/dagger_naming.py
│   │   ├── lineage.py              ← dagger_plot.py / dagger_progress.sh readers (sidecars, eval_info)
│   │   └── anomalies.py            ← dagger_detect_dataset_anomalies.py
│   ├── datasets/
│   │   ├── multi_source.py         ← lerobot/datasets/multi_source_normalizing_dataset.py
│   │   ├── stats.py                ← my_scripts/compute_relative_stats.sh (python)
│   │   └── episode_io.py           ← my_scripts/lib_dataset_episode_io.py
│   ├── lerobot_glue/               what lerobot needs to see: TrainPipelineConfig fields, dataset/policy wrapping
│   │   ├── config.py               ← the `dart_*` / multi-source fields from configs/default.py + train.py
│   │   ├── dataset_factory.py      ← the DART/multi-source branch of lerobot/datasets/factory.py
│   │   └── policy_factory.py       ← _wrap_with_shared_autonomy / temporal_ensemble / last_mile
│   ├── viz/                        ← lib_sa_plotting, dart_sim_video, visualize_* scripts (as modules)
│   └── stats.py                    the paired lineage tests (TOST etc.) from the paper
├── scripts/                        thin CLIs, one per verb; each `python -m pcdagger.<cli>` or `pcdagger-<verb>`
│   ├── dagger_orchestrate.sh       ← my_scripts/dagger_orchestrate.sh (+ _sweep, _repeat)   [bash, unchanged at first]
│   ├── train_sweep.sh, resume_training.sh, run_all_evals.sh, lib_*.sh, env_profiles/
│   ├── augment_blends.py           ← augment_dataset_with_blending.py CLI
│   ├── calibrate.py                measure sigma + W + build schedule for (lineage, round)
│   ├── visualize_*.py, dagger_plot*.py, summarize_evals.py, filter_blend_collisions.py …
├── paper/                          ← my_scripts/paper_plots/ verbatim (figures/, tables_repro/, data/, make_figures.sh)
├── sandbox/gridworld_dagger_sim/   ← my_scripts/gridworld_dagger_sim (lerobot-free, keeps its own README)
└── tests/                          ← the fork's 2 custom test files + the smoke tests
```

Rules that keep it modular:

- `pcdagger/` never imports from `scripts/` or `paper/`; `scripts/` are 20-line CLIs that call package functions.
- `pcdagger.dart` and `pcdagger.blend` depend on lerobot only through `lerobot.policies.pretrained`,
  the processor pipeline and `LeRobotDataset`; `pcdagger.lerobot_glue` is the only module that
  touches lerobot's factories/config classes, so a lerobot upgrade only ever breaks one folder.
- SplatSim is an optional dependency (`pcdagger[splatsim]`): only `pcdagger.blend.guidance.rrt_source`
  and `pcdagger.dagger.intervention` import it, lazily.

## The SplatSim environment: plugin package + hub repo

lerobot already has the two mechanisms you want, so nothing custom is needed:

1. **Plugin package (the real integration).** `lerobot-train` and `lerobot-eval` call
   `register_third_party_plugins()`, which imports every installed distribution named
   `lerobot_env_*`. So the 463-line `SplatSimEnv` `EnvConfig` (fork `envs/configs.py:649-1100`)
   moves into SplatSim as its own tiny distribution:

   ```
   SplatSim/
   └── lerobot_env_splatsim/           pip install -e SplatSim/lerobot_env_splatsim   (or a [lerobot] extra)
       ├── pyproject.toml              name = "lerobot_env_splatsim", depends on splatsim + lerobot
       └── lerobot_env_splatsim/__init__.py   @EnvConfig.register_subclass("splatsim") class SplatSimEnv …
   ```

   `--env.type=splatsim` then keeps working verbatim, on stock upstream lerobot, with zero edits to
   `envs/configs.py`. This is the same convention upstream documents for `lerobot_policy_*`.

2. **Hub repo (discoverability).** `JennyWWW/splatsim-env` on the Hub with `env.py` exposing
   `make_env(n_envs, use_async_envs, cfg)` that imports `lerobot_env_splatsim`, a `requirements.txt`
   (`splatsim`, `lerobot_env_splatsim`) and a README with the two tasks (`planar_3joint`,
   `upright_small_engine_new`) and the benchmark dataset ids. Loading: `make_env("JennyWWW/splatsim-env",
   trust_remote_code=True)`. The scene data itself (splats, `data/assets`, `data/stages`) is too large for the
   repo; the README points at a Hub *dataset* repo `JennyWWW/splatsim-scenes` that `install.sh` (or a
   `splatsim download-scene lever` command) fetches into `data/`.

Also to move into SplatSim now, since they belong to the simulator and not the paper:
`teleop_recording.py` (dataset writing during rollouts), `sim_seeding.py`, `robots/splatsim_lerobot/`
(the real-robot Robot adapter), and `lerobot_splatsim_wrapper.py` already there.

## What stays in the lerobot fork

After the move the fork is a *thin* branch of hooks, ideally < 800 lines, each a candidate upstream PR:

| Hook | Lines today | Why it can't be a plugin | Path to upstream |
| --- | --- | --- | --- |
| `train_utils.resume_after_prepare` re-applies `--optimizer.lr` | 31 | bug fix | PR as-is (it's a real upstream bug) |
| dataset factory: multi-dataset build (`repo_ids`, `exclude_features`, unconsumed-camera exclusion) | ~150 | dataset classes are lerobot-side | small PRs; the DART/normalizing wrap is in `pcdagger.lerobot_glue.dataset` |
| policy factory: `rename_map` / `obs_dim_slice` processors, relative-step reconnect | ~60 | processors are lerobot-side | small PR; the policy wrappers are in `pcdagger.lerobot_glue.policy` |
| `TrainPipelineConfig` / `DatasetConfig` extra fields (`dart_*`, `multi_source_*`, `shared_autonomy`) | ~200 | draccus needs the fields on the config class | generic `dataset.extra: dict` + `policy.wrappers: list` would remove this |
| ~~`lerobot_eval.py` intervention mode~~ | 0 | moved: `pcdagger.dagger.eval` (`pcdagger-eval`), the fork script is upstream's | done |
| `modeling_diffusion.py` (`generate_actions(noise=, sa_noise_ratio=)`, pending-chunk access) | 241 | partial denoising needs the scheduler hooks | PR: "expose `noise` and `start_timestep` in `generate_actions`" is a small, defensible upstream change |
| relative-action processor anchoring, select-obs-dims processor | 300 | processors are lerobot-side | small PRs |
| ~~multi-source dataset / normalizer stats~~ | 0 | in `pcdagger.datasets.multi_source`, applied by `pcdagger.lerobot_glue.dataset` | done |

Everything else in the 19.5k lines is paper code and moves.

## Order of work (each step ends green)

Green means `paper/tables_repro/smoke_test.sh` 13/13 and `paper/make_figures.sh` bit-identical, run from
the *new* location.

1. **DONE 2026-09-21 — Scaffold** this repo: `pyproject.toml`, `pcdagger/__init__.py`, `scripts/`, `paper/`. Move
   `my_scripts/paper_plots` → `paper/` and `gridworld_dagger_sim` → `sandbox/` with `git mv`-style history
   (`git filter-repo --subdirectory-filter my_scripts/paper_plots` into a branch, then merge with
   `--allow-unrelated-histories`, so `git log` still shows the figure history). Fix the handful of absolute
   paths (`~/code/lerobot/outputs`, `~/code/SplatSim`) into two env vars with defaults:
   `PCDAGGER_OUTPUTS`, `SPLATSIM_ROOT`. Smoke + figures green from the new path. *Half a day.*
   Done: `paper/` and `sandbox/` carry their fork history (13 commits merged); `pcdagger/paths.{py,sh}`
   hold the four locations; every hard-coded `~/code/...` in `paper/` now reads them. Lesson: the fork's
   `paper_plots/.gitignore` had never actually tracked `paper_plots/data/` (a `!data/**` negation under a
   `*.npz` rule does not re-include), so the snapshots were rebuilt from the sources here and the root
   `.gitignore` negates the directory explicitly.
2. **DONE 2026-09-21 — SplatSim plugin.** Create `lerobot_env_splatsim` inside SplatSim from the fork's `SplatSimEnv`;
   delete that class from the fork; `lerobot-eval --env.type=splatsim` still works because the plugin
   registers it. Move `teleop_recording.py`, `sim_seeding.py`, `robots/splatsim_lerobot` into SplatSim.
   Run one 2-episode planar and lever eval. *One day.*
   Done: `SplatSim/lerobot_env_splatsim/` (config, robot, recording, seeding); fork keeps two re-export
   shims for `teleop_recording` / `sim_seeding` until step 3. Both evals reproduce Sept-16 step for step.
   Gotcha: install the plugin with `--no-deps`; the fork's editable metadata was stale (0.4.3 pins) and a
   plain `pip install -e` downgraded huggingface_hub/draccus. Refreshed with `pip install --no-deps -e lerobot`.
3. **DONE 2026-09-21 — Move the pure-python method code** (`dart_relabel`, `chunk_anchor`, `guidance/`,
   `shared_autonomy_wrapper`, `multi_source_normalizing_dataset`, `intervention_controller`,
   `lib_sa_rollout`, `augment_dataset_with_blending`) into `pcdagger/`, leaving one-line re-export shims
   at the old lerobot paths (`from pcdagger.dart.relabel import *`) so every script keeps running
   unchanged. Smoke green. Then repoint the scripts and delete the shims. *Two days; the imports are the
   whole job.*
4. **DONE 2026-09-21 — Scripts.** Move `my_scripts/*.sh` and the remaining `.py` into `scripts/`; the bash orchestrator stays
   bash (7.5k lines, works, documented) but its `python -c` snippets call `pcdagger` functions. Delete
   `my_scripts/` from the fork. *One day.*
   Done (3 and 4 together, one filter-repo extraction of 159 commits): `pcdagger/{dart,blend,dagger,datasets,viz,extras}`
   and `scripts/` are populated, every import rewritten, `pip install --no-deps -e .` makes `pcdagger` importable.
   The fork keeps 23 one-line re-export shims (`lerobot.policies.shared_autonomy_wrapper` etc. → `pcdagger.*`)
   for its factory / eval hooks, and `my_scripts/` is gone from it. Kept in the fork on purpose (lerobot's own
   config classes embed them): `configs/{shared_autonomy,intervention,last_mile,temporal_ensemble}.py`,
   `policies/common/chunk_anchor.py`, `processor/select_observation_dims_processor.py`.
   Gotchas: the orchestrator, cleanup and lineage scripts derived the lerobot root from their own location
   (`$SCRIPT_DIR/..`); they now require `LEROBOT_ROOT` from `pcdagger/paths.sh`, which every moved shell script
   sources first. Verified: smoke 13/13, all figures, orchestrator dry-run resolves every round and calls
   `scripts/{resume_training,compute_relative_stats,dagger_orchestrate}.sh`.
5. **Shrink the fork — DONE 2026-09-21.** The fork's `lerobot_train.py` and
   `lerobot_eval.py` are back to upstream; their fork versions are `pcdagger/train.py` and
   `pcdagger/dagger/eval.py`, installed as the console scripts `pcdagger-train` / `pcdagger-eval`, which every
   script and `paper/tables_repro/lib_repro.sh` now call (`PCDAGGER_TRAIN` / `PCDAGGER_EVAL` override). The
   DART + multi-source wrapping left `datasets/factory.py` for `pcdagger/lerobot_glue/dataset.py`
   (`make_train_eval_datasets`), the `_wrap_with_*` policy wrappers left `policies/factory.py` for
   `pcdagger/lerobot_glue/policy.py`; the 23 re-export shims are deleted. Two modules went the other way
   because lerobot's own classes import them: `policies/temporal_ensembler.py` (ACT uses it) and
   `policies/pi05/{modeling,configuration}_shared_autonomy.py` (modeling_pi05 uses them). Fork over
   upstream: 5203/313 → 3597/207 lines, 73 → 49 files, no `pcdagger` import anywhere in it. Verified:
   smoke 13/13, figures, orchestrator dry-run.
   Upstream sync: branch `upstream-sync-2026-09-21` = fork main + merge of upstream/main (60 commits, 7
   conflicting files, all resolved by keeping both sides — see the merge commit). `main` stays frozen at the
   paper code (tag `paper-icra2027-frozen`). On the merged branch: smoke 13/13 (planar q5 episodes identical),
   figures, orchestrator dry-run. The checkout (and so the editable install) is on the sync branch now.
6. **Publish.** `lerobot_env_splatsim` + hub env repo; `JennyWWW/splatsim-scenes` dataset with the two
   scenes; the archive (`~/paper_archive/pcdagger_icra2027`, 77 GB) as Hub model + dataset repos, one per
   table (checkpoints as `JennyWWW/pcdagger-planar-table1`, evals as a dataset); README links. Open the
   two small upstream PRs (lr resume fix, `generate_actions(noise=, start_timestep=)`).

Steps 1–2 are independent of 3–5 and are worth doing first: they make the paper reproducible from a
public checkout, which is what a reviewer would try.

## Decisions (Jenny, 2026-09-21)

- Package name: `pcdagger`.
- The bash orchestrator stays bash.
- Non-paper code (`last_mile/`, `temporal_ensemble*`, `pi05/modeling_shared_autonomy.py`) is kept under
  `pcdagger/extras/`, not deleted.
- Fork history travels with the code: each move is a `git filter-repo` extraction of the moved paths
  (renamed to their new locations) merged into this repo with `--allow-unrelated-histories`, so `git log`
  and `git blame` on the moved files show the original commits. The fork is tagged
  `pre-migration-2026-09-21` before the first move.
