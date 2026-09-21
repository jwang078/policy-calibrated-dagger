# What goes on the Hugging Face Hub (review before uploading)

`bash stage_hub_upload.sh` lays the whole upload out under `~/paper_archive/hub_upload/` as hard links of
the archive (`archive_tables.sh`), so you can look at every folder first. Nothing is uploaded by the script;
it prints the `hf upload` commands at the end. Staged 2026-09-21: 224 checkpoints, 130 datasets, 22 GB.

## Already on the Hub, verified identical to the archive (episodes, frames, features)

- `JennyWWW/planar_3joint_12` — the planar base demonstrations (500 episodes)
- `JennyWWW/eval_planar_3joint_benchmark` — the 100 planar evaluation scenarios
- `JennyWWW/eval_splatsim_approach_lever_13_benchmark` — the 100 lever evaluation scenarios

## 1. Datasets — 130 LeRobot dataset repos, 6.1 GB, same names as the local cache

Uploading under the existing `JennyWWW/<name>` ids means every script in this folder works from a fresh
machine with no edits: lerobot pulls a missing `JennyWWW/<name>` from the Hub into `LEROBOT_CACHE_DIR`.

| what | repos | size | needed for |
| --- | --- | --- | --- |
| `splatsim_approach_lever_13_smooth_r84` | 1 | 3.9 GB | lever base demos (Table II BC and every fine-tune) |
| `planar_12_{05,06,07,08,09}dag_diff_r_dag{1..5}` | 25 | ~5 MB each | the five planar lineages' interventions (every Table I cell) |
| `planar_12_10cl_diff_r_dag{1..5}` | 5 | ~5 MB each | the closed-loop lineage (Table I last row) |
| `planar_12_05dag_diff_r_dag{1..5}_blend{005..090}` | 70 | ~5 MB each | re-measuring the planar calibration (the shipped schedules are already in git: `analysis/noise_schedules.tar.gz`) — **optional** |
| `lever_d100_03dagcap_r84_diff_r_dag{1..5}` | 5 | 67–103 MB | the lever lineage's interventions (Table II) |
| `lever_d100_03dagcap_r84_diff_r_dag{1..5}_blend{010..090}` | 29 | ~40 MB each | re-measuring the lever calibration (W and the per-round schedules) — **optional** |

Skipping the two optional rows leaves 36 repos / ~4.4 GB; the calibration then reproduces from the committed
schedules only (that is what `smoke_test.sh`'s "from cached deltas" checks do).

## 2. Policies — two model repos, 16 GB

`checkpoints/<step>/pretrained_model` of every run the archive verified a table cell against, plus each run's
`train_config.json` and its training-time `eval/*.json`. Optimizer states (38 GB), wandb logs and eval videos
are left out. Layout mirrors `outputs/training/<group>/<run>/<step>/`, so a download into
`PCDAGGER_OUTPUTS/training` restores the paths the scripts expect.

- `JennyWWW/pcdagger-planar-table1` — 13 GB: `bc175k` (the BC checkpoint), `scarcity_study{,_s2..s5}` (the
  five lineages' 80 arms each), `scarcity_study_cl` (closed loop), and the `diffusion_planar_*_ft_dag*` collector
  policies of each DAgger round (needed only to re-collect interventions).
- `JennyWWW/pcdagger-lever-table2` — 2.9 GB: the lever BC (`diffusion_approach_lever_13_smooth_r84_delta_basewrist`
  + seeds 1–2), the per-round collectors (`..._ft_dag{1..5}`), `lever_r84_calib` (calibrated arms), `lever_r84_fb`
  (HG-DAgger and fixed-sigma arms).

Question for you: one repo per table as above, or one repo per group (7 + 10 smaller repos)? Per table is
simpler to cite; per group downloads faster when someone only wants the BC checkpoint.

## 3. Results — one dataset repo, 1.1 GB: `JennyWWW/pcdagger-icra2027-results`

- `eval300/` — every 300-episode evaluation behind the tables (eval json + per-episode telemetry, no videos)
- `dataset_stats/` — the normalization sidecars (`stats_path` inputs) every fine-tune used
- `tables_repro_analysis/` — the per-anchor deviation measurements (`sigma_deltas_*.npz`, 511 MB) and the
  schedules; with these the calibration scripts run with no policy and no simulator

## 4. Not on the Hub

- **SplatSim lever scene** (`robot_iphone_w_engine_curtain`, 370 MB) — already downloadable from the SplatSim
  README (Google Drive). Mirroring it as `JennyWWW/splatsim-stages` would be nicer than Drive, your call.
  The planar task has no splat assets; `small_engine_new` and the UR5 body ship in the repo.
- **EnvHub repo** (`JennyWWW/splatsim-env` with an `env.py`) — lerobot's EnvHub promises "no install", which
  SplatSim cannot keep (CUDA rasterizer, pybullet, gello). The `lerobot_env_splatsim` plugin in the SplatSim
  repo is the real integration; an EnvHub repo would be a 20-line pointer to it. Recommend skipping.
- **Code** — GitHub only: `jwang078/policy-calibrated-dagger`, `jwang078/lerobot` (tags
  `paper-icra2027-frozen`, `pcdagger-compat-2026-09-21`), `jwang078/SplatSim` (tag `pcdagger-compat-2026-09-21`).

## Before uploading

1. Look through `~/paper_archive/hub_upload/` (`du -sh */*` per folder).
2. Decide the two questions above (optional blend datasets; per-table vs per-group model repos).
3. Every repo needs a README model/dataset card; a one-paragraph card pointing at the paper repo and this
   file is enough. Datasets in LeRobot format get the dataset viewer for free.
4. Run the printed `hf upload` commands (they resume on re-run). Uploads are public by default; add
   `--private` if you want to flip them at camera-ready.
