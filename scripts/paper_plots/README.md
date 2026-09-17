# Paper figures and tables

Everything that produced the figures and tables of _Policy-Calibrated DAgger_ (ICRA 2027 submission),
in one place. `bash make_figures.sh` regenerates every figure from the input snapshots committed under
`data/`, with no dataset cache, checkpoints, simulator or `~/data` needed. The two results tables have
their own kit in [`tables_repro/`](tables_repro/README.md).

| Paper                                           | Folder                                                       | Command                                                 | Inputs it needs (all committed)                                                                                                                                                                 |
| ----------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Fig. 1 teaser (lever, splat background)         | `fig1_teaser_lever/`                                         | `python plot.py dag1_ep9_t4`                            | `geom_<tag>.json`, `bg_<tag>.png` (written by `render.py`, see below)                                                                                                                           |
| planar teaser (earlier draft)                   | `fig1_teaser/`                                               | `python plot.py`                                        | `data/planar_12_05dag_diff_r_dag3__ep24.npz`, `servo_states_ep24_f61.json`, the K=3 pooled schedule                                                                                             |
| Fig. 2 method overview                          | `fig_method_overview/`                                       | `bash make_figures.sh overview`                         | `calibration_panels.pdf` from `fig_calibration_steps/plot.py` (`TASK=lever_r84`, exact `CONFIG_JSON` in `make_figures.sh`) embedded by `method_overview_merged.tex` (TikZ, Times via newtxtext) |
| Fig. 3 partial denoising at one observation     | `fig_partial_denoise/`                                       | `python plot.py`                                        | `data/partial_denoise_ep10_a62.json` (the policy samples; drawing from it is bit-identical to a fresh run with the round-3 policy)                                                              |
| Fig. 4 blend dial / W fit                       | `fig_w_dial/`                                                | `python plot.py --cached`                               | `w_dial_data.json` (settling cells of all 5 rounds × 13 ratios), `data/planar_bc_policy_config.json` (DDPM schedule)                                                                            |
| Fig. 5 task frames                              | `lever_task_frames/`                                         | static PNGs                                             | rendered from the eval videos; the planar row was assembled in the paper source                                                                                                                 |
| Tables I, II                                    | `tables_repro/`                                              | `python gen_table.py`, `python analysis/lever_table.py` | `outputs/eval300/*` on this machine; recipes in its README                                                                                                                                      |
| per-joint sigma (cut from the paper)            | `fig_noise_levels/`                                          | `python plot.py`                                        | `data/noise_levels_covs.json` (pooled covariances per lineage and round)                                                                                                                        |
| standalone 4-panel calibration (planar / lever) | `fig_calibration_steps/`, `fig_calibration_steps_lever_r84/` | `python plot.py`                                        | episode and blend snapshots in `data/`, one episode of open-loop deltas, the pooled schedule                                                                                                    |
| DART-relabelled blend rollout (earlier draft)   | `dart_chunks_src47/`                                         | `python plot.py`                                        | its own `.npz`                                                                                                                                                                                  |

## How the snapshots work

Every script reads its raw inputs through [`figdata.py`](figdata.py): one episode's states and actions
from a LeRobot dataset, the dataset's state cloud for the PCA plane, a policy's `config.json`, one
episode of open-loop deltas. The first call reads the source (`~/.cache/huggingface/lerobot/JennyWWW`,
`outputs/training`, `tables_repro/analysis`) and writes exactly what the figure needs into `data/`
(small `.npz`/`.json`, ~1 MB in total); later calls read the snapshot. `FIGDATA_REFRESH=1` re-reads the
sources. `make_figures.sh` points the source paths at `/nonexistent` so it proves the snapshots suffice;
on 2026-09-17 every figure rebuilt bit-identically that way.

Two figures also depend on artefacts that are _not_ snapshotted because they are large or live in
SplatSim:

- `fig1_teaser_lever/render.py` runs the SplatSim engine scene in-process to render the background and
  compute the overlay geometry. Its outputs (`bg_*.png`, `geom_*.json`) are committed, so `plot.py` and
  the styling GUI work without it; re-rendering needs the splat scene under `SplatSim/data/output/` and the
  `~/.cache` datasets.
- The noise schedules come from `tables_repro/analysis` (`bash tables_repro/analysis/unpack_schedules.sh`
  after a fresh clone).

Each figure folder also has a `build_gui.py` + `gui_template.html`: a styling page whose CONFIG block is
pasted back into the script. `cands/` folders (candidate sheets) are git-ignored.
