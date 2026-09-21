# Policy-Calibrated DAgger

Offline calibrated noise injection for interactive imitation learning (ICRA 2027 submission).

This repository is being assembled from the paper's working code in the `jwang078/lerobot` fork; see
[MIGRATION_PLAN.md](MIGRATION_PLAN.md) for what has moved and what is still on its way.

```
pcdagger/      importable package (in progress: paths.py so far; dart/, blend/, dagger/ … follow)
paper/         everything behind the paper's figures and tables — `bash paper/make_figures.sh`,
               `paper/tables_repro/README.md` for Table I / II, `paper/tables_repro/smoke_test.sh`
sandbox/       gridworld_dagger_sim: a lerobot-free sandbox for blended-DAgger semantics
scripts/       (next) the DAgger orchestrator and the analysis / visualization CLIs
```

## Setup

```bash
pip install -e ".[splatsim]"          # pcdagger + the lerobot fork + SplatSim
```

Locations are environment variables with the defaults of the machine the paper was run on
(`pcdagger/paths.py`, `pcdagger/paths.sh`): `LEROBOT_ROOT`, `SPLATSIM_ROOT`, `PCDAGGER_OUTPUTS`
(training runs, evals), `LEROBOT_CACHE_DIR` (datasets). Figures rebuild without any of them from the
snapshots in `paper/data/`.
