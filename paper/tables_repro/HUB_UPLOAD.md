# Hub upload: the simulator scenes (review before uploading)

Decision 2026-09-21: policies and the intervention datasets stay in the local archive
(`~/paper_archive/pcdagger_icra2027`, built by `archive_tables.sh`); only the simulator data goes to the
Hugging Face Hub. The three datasets the reproduction scripts pull by name are already there and match the
archive: `JennyWWW/planar_3joint_12`, `JennyWWW/eval_planar_3joint_benchmark`,
`JennyWWW/eval_splatsim_approach_lever_13_benchmark`.

## `JennyWWW/splatsim-scenes` (dataset repo, 1.4 GB)

`bash stage_hub_upload.sh` stages it under `~/paper_archive/hub_upload/splatsim-scenes/` and prints the
upload command; nothing is uploaded by the script.

| file | size | what |
| --- | --- | --- |
| `robot_iphone_w_engine_curtain.tar.gz` | 390 MB | the UR5 + wrist camera scanned in the engine scene: `splat/` (Gaussian-splatting output, iteration 30000) and `sfm/` (COLMAP). The lever task renders from this. |
| `vine_scene.tar.gz` | 1.0 GB | the grape-vine prop: splat, SfM, and the `segmentations/vine_and_trellis/` build (collision body, cost fields) |
| `README.md` | | dataset card with the two `tar xzf ... -C data/stages` lines |

These are exactly the tarballs the SplatSim README currently links on Google Drive (`~/data/*.tar.gz`,
built 2026-09-18). Each unpacks to `data/stages/<scene>/` and carries only the scan bytes; every
`stage.yaml` and the segmentation labels are tracked in the SplatSim repo, so a fresh clone plus the
tarball is a working scene. The planar task has no scan (pybullet only), and the UR5 body ships in the repo.

After uploading, point the SplatSim README's "Download the example scenes" section at the Hub
(`hf download JennyWWW/splatsim-scenes --repo-type dataset --local-dir ...`) and keep or drop the Drive links.

## Not uploaded

- Policies (224 checkpoints, 16 GB) and the 130 intervention / blend datasets: in the local archive only.
- An EnvHub `env.py` repo: lerobot's EnvHub promises "no install", which SplatSim cannot keep (CUDA
  rasterizer, pybullet, gello); the `lerobot_env_splatsim` plugin in the SplatSim repo is the integration.
- Code is on GitHub: `jwang078/policy-calibrated-dagger`, `jwang078/lerobot` (tags `paper-icra2027-frozen`,
  `pcdagger-compat-2026-09-21`), `jwang078/SplatSim` (tag `pcdagger-compat-2026-09-21`).
