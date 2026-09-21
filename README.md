# Policy-Calibrated DAgger

Offline calibrated noise injection for interactive imitation learning (ICRA 2027 submission).

This repository is being assembled from the paper's working code in the `jwang078/lerobot` fork; see
[MIGRATION_PLAN.md](MIGRATION_PLAN.md) for what has moved and what is still on its way.

```
pcdagger/      the importable package
  train.py, dagger/eval.py      pcdagger-train / pcdagger-eval: lerobot-train / lerobot-eval plus the
                                paper's machinery (multi-source + DART datasets, policy wrappers,
                                intervention recording)
  dart/          train-time DART relabelling (DartChunkDataset, noise schedules)
  blend/         the blended-rollout shared-autonomy wrapper, RRT guidance sources, the GUI
  dagger/        intervention controller, lineage naming, the eval loop
  datasets/      multi-source normalizing dataset, dataset directory + episode helpers
  lerobot_glue/  make_train_eval_datasets and the post-factory policy wrappers used by train/eval
  viz/           plotting and kinematics helpers, the DART video
  extras/        non-paper experiments (last-mile help, temporal ensembling)
scripts/       the DAgger orchestrator (bash) and the analysis / visualization CLIs
paper/         everything behind the paper's figures and tables — `bash paper/make_figures.sh`,
               `paper/tables_repro/README.md` for Table I / II, `paper/tables_repro/smoke_test.sh`
sandbox/       gridworld_dagger_sim: a lerobot-free sandbox for blended-DAgger semantics
```

`pcdagger` imports `lerobot` (the fork) and `splatsim`; nothing imports `pcdagger` back. The fork
carries only the library-level hooks the package needs (config fields, the dataset/processor and
diffusion-model changes, see [MIGRATION_PLAN.md](MIGRATION_PLAN.md)); SplatSim's LeRobot
environment is the `lerobot_env_splatsim` plugin in the SplatSim repo.

## Setup

```bash
pip install -e ".[splatsim]"          # pcdagger + the lerobot fork + SplatSim
pcdagger-train --help                 # installed console scripts; every script here calls these
pcdagger-eval --help                  # (override with PCDAGGER_TRAIN / PCDAGGER_EVAL)
```

Locations are environment variables with the defaults of the machine the paper was run on
(`pcdagger/paths.py`, `pcdagger/paths.sh`): `LEROBOT_ROOT`, `SPLATSIM_ROOT`, `PCDAGGER_OUTPUTS`
(training runs, evals), `LEROBOT_CACHE_DIR` (datasets). Figures rebuild without any of them from the
snapshots in `paper/data/`.
