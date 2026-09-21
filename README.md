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

## Keeping up with lerobot and SplatSim

`pcdagger` sits on top of both; the boundary is deliberately narrow and checked:

- `pcdagger/compat.py` is the only place that imports private lerobot helpers. Everything else is
  public API, the fork's config dataclasses (`lerobot.configs.{shared_autonomy,intervention,last_mile,
  temporal_ensemble}`) and the fork's hooks listed in [MIGRATION_PLAN.md](MIGRATION_PLAN.md).
- `bash scripts/check_dependency_contract.sh` (no GPU, seconds) runs `tests/test_dependency_contract.py`:
  every lerobot / SplatSim name pcdagger imports (scanned from the source, so the list can't go stale),
  the signatures of the fork hooks pcdagger calls, and that the `splatsim` env plugin still registers.
  It then reports how far `pcdagger/train.py` and `pcdagger/dagger/eval.py` have drifted from the fork's
  own `lerobot_train.py` / `lerobot_eval.py` (they started as copies of them) and how far the fork is
  from huggingface/lerobot.
- `pyproject.toml` pins both dependencies to the `pcdagger-compat-<date>` tag the contract test and
  `paper/tables_repro/smoke_test.sh` were last green against. To move: update the fork
  (`scripts/sync_upstream.sh`), run the contract test, run the smoke test, re-tag, move the pin.

