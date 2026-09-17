# Reproducing the two results tables

Everything that produced Table I (planar reaching, `tab:overall_success`) and Table II (engine lever,
`tab:overall_success_splatsim`) in the Policy-Calibrated DAgger paper. The scripts here are verbatim copies
of the ones that ran, with the volatile scratchpad path rewritten to this directory, so each one can be
run from here as-is. Outputs (checkpoints, evals) still go to `lerobot/outputs/`, datasets to
`~/.cache/huggingface/lerobot/JennyWWW/`.

Layout:

- `dagger_cmds/` — orchestrator commands that trained BC and collected the DAgger lineages (planar).
- `*.sh` — chain scripts, one per stage (see the two recipes below).
- `analysis/` — calibration code (W fit, sigma measurement, schedule builders), dataset shrinking, and
  the table generators. Also holds the exact noise schedules used by every calibrated arm
  (`noise_schedule_pooled_*`, `noise_schedule_sigma_alpha_*`, `noise_schedule_pooled_lever_r84_*`).
- `gen_table.py` — Table I LaTeX; `analysis/lever_table.py` — Table II LaTeX.
- `smoke_test.sh` — runs every command shape above in miniature (see _Smoke test_ at the end).

First time here: `bash analysis/unpack_schedules.sh` (the schedules are committed as one 13 MB tarball).

> **Learning-rate caveat (found 2026-09-16).** Every Table I arm passes `--optimizer.lr=1e-6`, but until the
> 2026-09-14 patch in `src/lerobot/common/train_utils.py` the resumed optimizer state silently overrode that
> flag, so the K=1..5 arms actually trained on the BC checkpoint's own cosine schedule (1e-5 peak, ~6e-6 at
> step 76k, decaying to 0 at 175k; see `lr:` in their logs). With current code the flag is honoured, so to
> reproduce those arms pass `--optimizer.lr=1e-5` (the K=6 scripts do; `LR_FT` env). Training K=6 at a true
> 1e-6 gave ~77% instead of ~87% for HG-DAgger on three lineages. Table II (lever) was trained entirely after
> the patch at a true constant 1e-6, so it is internally consistent.

Conventions: `PY=~/miniforge3/envs/splatsim/bin/python`, `LR=~/code/lerobot`. Every script defines
`S=<this directory>` and expects `analysis/` beside it. Long jobs were launched detached
(`setsid nohup bash script.sh > script.out 2>&1 < /dev/null &`) so editor crashes don't kill them.

---

## Table I — planar reaching (5 lineages, 300 episodes per cell)

One "lineage" = one independently collected DAgger run. Lineages are tagged `05dag`…`09dag`
(= table seeds s1…s5, intervention sample seeds 0…4); their training dirs are
`outputs/training/scarcity_study{,_s2,_s3,_s4,_s5}`. Every table cell is a policy fine-tuned **from the
same BC 75k checkpoint** on base + rounds 1..K of that lineage, to 175k steps total, then evaluated with
300 episodes (100 fixed scenarios × 3) by `reeval300*.sh`. Cells are mean ± SE across the 5 lineages.

### 1. BC and the DAgger lineage (rounds 1–5)

```
bash dagger_cmds/05dag_cmd.sh        # lineage s1; likewise 06dag…09dag for s2…s5
```

Round 0 of the orchestrator trains BC (`diffusion_planar_3joint_12_delta_stateng`, 75k steps, 500 demos).
Each round K then rolls out the previous lineage policy on the 100 scenarios, records HG-DAgger
interventions on the failures (`planar_12_<tag>_diff_r_dag<K>`), and fine-tunes the lineage +10k
(`…_<tag>_ft_dag<K>`, only used to collect the next round). Extend a lineage by one round with
`bash dagger_cmds/09dag_cmd.sh --num_rounds=6` (the scripts pass extra args through; `--resume` picks up
the completed rounds).

### 2. HG-DAgger and fixed-isotropic rows

```
bash seed_grid.sh 05dag s1           # arms q{K} (HG-DAgger) and q{K}_dn{2,4,8,12,16}swv for K=1..5
```

`q{K}` = from-base fine-tune on base + dag1..K (sample weights 0.7 base / 0.3 interventions, split
equally across rounds). `q{K}_dn{σ}swv` = the same data with DART state noise of fixed std σ
(`--dataset.dart_relabel=true --dataset.dart_state_noise_std=σ --dataset.dart_vel_noise_std=0.3` plus
the mask flags in `SW`).

### 3. Calibrated rows (pooled Σ̄^α and per-step Σ^α_t)

Calibration for lineage s, round K:

```
$PY analysis/measure_sigma_multi.py s1q{K} outputs/training/scarcity_study/q{K}/checkpoints/last/pretrained_model 05dag "1,...,K"
$PY analysis/build_sigma_schedule_k.py s1 05dag K
```

The first samples the round-K HG policy open loop along every intervention of rounds 1..K and stores the
per-anchor deltas (`sigma_deltas_s1q{K}_dag{R}.npz`; the ones behind the shipped schedules are kept in
`analysis/`, git-ignored, ~320 MB). The policy samples are not seeded, so a re-measurement reproduces the
pooled covariance to within a few percent, not bit-for-bit. The second scales them with W = 8.0 (the closed-loop
settling error from the blend dial, stable across rounds on planar) and writes
`noise_schedule_pooled_s1_K{K}.json` (pooled Σ̄^α) and `noise_schedule_sigma_alpha_s1_K{K}.json`
(per-step Σ^α_t). Training the arms:

- s1, s2: `pool_arm.sh`, `sigma_arm.sh` (K=2) and `calib_k_arms.sh` (K=1,3,4,5; includes the two
  calibration calls above). The K=2 schedules of s1/s2 were measured first, under the tag `s1`/`s2`
  instead of `s1q2`/`s2q2` (`measure_sigma_multi.py s1 … "1,2"`, then `analysis/build_sigma_schedule.py s1 05dag`
  for the per-step file). The deltas are also linked under the `s{1,2}q2` names, and
  `build_sigma_schedule_k.py s1 05dag 2` regenerates both K=2 files from them (pooled identical, per-step
  within 1e-3 from float32 vs float64 accumulation).
- s3–s5: `sample5c.sh` (same recipe, arms `q{K}_dnpool` and `q{K}_dnsig`).

### 4. Closed-loop pooled row (1 lineage, `10cl`)

The closed-loop lineage collects each round with the _calibrated_ policy instead of the HG policy:

```
bash closed_loop_10cl.sh             # rounds 1..5: collect -> measure sigma with ft_dag{K-1} -> schedule cl1_K{K} -> lineage finetune
bash cl_arms.sh                      # table arms q{K}_dnpoolcl from base on the 10cl datasets
bash reeval300_cl.sh
```

`10cl_cmd_k{K}.sh` are the per-round orchestrator commands `closed_loop_10cl.sh` generates (the
`--finetune_extra_args` carry that round's schedule). Round 1 reuses the s1 round-1 interventions.

### 5. Evaluation (300 episodes per checkpoint)

```
bash reeval300.sh / reeval300_s45.sh / reeval300_calib.sh / reeval300_calib345.sh / reeval300_cl.sh
```

Each starts a planar SplatSim node on its own port and runs `lerobot-eval` with
`--eval.n_episodes=300 --env.eval_benchmark_subset=[0..99] --seed=0`. Results land in
`outputs/eval300/<training dir>/<arm>/eval_info.json`. Never point two evals at one sim port.

### 6. K = 6 (added after the deadline extension)

Lineages were extended with `bash dagger_cmds/0Xdag_cmd.sh --num_rounds=6` (sequential, port 6001), then:

```
bash planar_k6_train2.sh "s1 s3 s5"      # lane A: q6 for each lineage -> sigma + K6 schedules -> q6_dnpool -> q6_dnsig
bash planar_k6_train2.sh "s2 s4"         # lane B (4 dataloader workers each; two trainers is the RAM limit)
bash planar_k6_eval.sh L 3 $((6027+L))   # three polling 300-ep eval lanes (L = 0,1,2)
bash planar_k6h.sh                       # after lane B: closed-loop K=6 arm (cl_arms_k6.sh + planar_cl_eval.sh),
                                         # then fixed sigma 4,2,8,12,16 via FIXED=1 planar_k6_train2.sh
```

`planar_k6_train2.sh` passes `--optimizer.lr=${LR_FT:-1e-5}` (see the learning-rate caveat above).
`closed_loop_10cl_k6.sh` collected the closed-loop round 6 and built `noise_schedule_pooled_cl1_K6.json`
with the round-5 closed-loop policy; its lineage fine-tune is only needed for a round 7.

### 7. Table

```
$PY gen_table.py                     # reads outputs/eval300/*, writes the colored LaTeX tabular
```

---

## Table II — engine lever, image observations (84 px)

All arms are fine-tuned **from the same BC 75k checkpoint** to 95k (+20k), on base + rounds 1..K.
HG-DAgger and calibrated arms of round K share the same interventions. Cells are mean ± SE over
3 training seeds, each seed averaged over 3 eval seeds × 100 scenarios (900 episodes).

### 1. Datasets and BC

```
$PY analysis/shrink_lever_dataset.py         # base demos -> 84 px copy: splatsim_approach_lever_13_smooth_r84
bash lever_r84_strip.sh                      # drop the letterbox image columns (keeps stretch)
bash lever_r84_chain.sh                      # BC 75k: diffusion_approach_lever_13_smooth_r84_delta_basewrist (policy resize 84, crop 1.0)
bash lever_r84_bceval.sh                     # BC 100-scenario eval (outputs/eval300/lever_cam/bc_r84)
```

The splat node always renders 224 px; the policy resizes internally. Intervention datasets are recorded
at 224 and re-encoded to 84 px with `analysis/shrink_video_dataset_r84.py`.

### 2. Round 1

```
bash lever_r84_stage2.sh     # orchestrator (lever_orchestrator_argv.txt): record dag1 with BC, HG r1 lineage finetune
bash lever_r84_blends.sh     # blend dial: partial-denoising rollouts at ratios 0.1..0.9 on dag1 (two sim lanes)
bash lever_r84_calib.sh      # W fit (analysis/measure_w_lever.py), sigma (analysis/measure_sigma_lever.py),
                             # pooled K1 schedule (analysis/build_pooled_schedule_lever.py), calibrated r1 from base, evals
bash lever_r84_evals_r1.sh   # 100-scenario evals of the +20k checkpoints
```

The HG r1 +20k number for seed 0 comes from the orchestrator's inline eval at step 95000
(`…_ft_dag1/eval/eval_info_step_095000.json`); `lever_table.py` picks it up.

### 3. Round 2 and rounds ≥ 3

```
bash lever_r84_round2.sh; bash lever_r84_blends2.sh; bash lever_r84_calib2.sh     # round 2
bash lever_r84_round.sh K            # generic round K>=3 (driver: lever_r84_rounds345.sh)
```

`lever_r84_round.sh K`: exposes HG r(K−1) as the orchestrator's round-(K−1) policy, records dag K on the
scenarios it failed, trains HG rK from base while the blend lanes run, fits W on rounds 1..K, measures
sigma with HG r(K−1) on dag1..K, builds `noise_schedule_pooled_lever_r84_K{K}.json`, trains calibrated
rK from base, evaluates both on two sim nodes.

### 4. Training seeds

```
SEEDS="1 2" ROUNDS="1 2" bash lever_r84_seeds.sh
```

lerobot's resume restores the checkpoint RNG _after_ applying `--seed`, so a plain seed flag replicates
seed 0. The script makes a per-seed copy of the BC checkpoint with a freshly seeded
`training_state/rng_state.safetensors` (`…_basewrist_seed{N}`) and trains HG + calibrated from it with
the round's schedule, then evaluates each with eval seed = training seed.

### 5. Eval seeds (300 episodes per checkpoint)

```
bash lever_r84_lane.sh L 3 PORT      # L = 0,1,2; KS="1 2" (default) selects rounds; lever_r84_lane_r2first.sh = round-2-first order
```

Eval dirs are `outputs/eval300/lever_cam/r84_{hg,cal}{K}_20k[_s{seed}]_e{evalseed}`. Evals are GPU-bound
(splat rendering): N concurrent lever evals give no throughput gain over one, so lanes only reorder work.

### 6. Table

```
$PY analysis/lever_table.py          # prints per-seed means, writes analysis/table_lever.tex
```

---

## Machine notes that bit us

- 30 GB RAM: one lever trainer (~13 GB with 8 dataloader workers) + two splat nodes + evals is the limit;
  a second trainer or three nodes OOM-kills VS Code and the desktop app. Planar trainers are light.
- Never put `pkill -f pattern` in a command that also contains the unbracketed pattern text (it kills the
  calling shell); use `pkill -f 'name[.]sh'`.
- Every eval/collection lane needs its own sim port. Ports used: planar lineages 6001, planar evals
  6023–6030, lever blends 6042/6043, lever evals 6045–6049.

### 8. Eval-seed error bars for single-checkpoint rows (added 2026-09-16)

```
bash planar_cl_seeds.sh L 4 $((6027+L))   # closed-loop pooled q{K}_dnpoolcl, eval seeds 1,2 (300 ep each) -> eval300/scarcity_study_cl/*_e{1,2}
bash planar_bc_seeds.sh L 2 $((6027+L))   # planar BC (outputs/training/bc175k), eval seeds 1,2
BC_ONLY=1 bash lever_r84_lane_bc.sh 3 4 6048   # lever BC, eval seeds 1,2 (100 ep each)
```

The ± on these rows is the SE over three evaluation seeds of one checkpoint, not over lineages.

## Overrides

The analysis scripts read and write `<this dir>/analysis`; set `TABLES_REPRO_DIR=/elsewhere` to redirect
them (what `smoke_test.sh` does, so the shipped schedules are never touched). Lever calibration scripts
take `INT_PREFIX` (intervention repo prefix, default the 224 px lineage) and `TAGPFX` (`r84_` for the
84 px files). `lever_r84_seeds.sh` takes `SEEDS` and `ROUNDS`, the eval lanes take `KS`.

## Smoke test (2026-09-17)

`bash smoke_test.sh` (~10 min, GPU) re-runs every command shape the tables depend on in miniature and
compares what it can against the shipped artifacts. Results of the last run, after the paper was submitted
and the machine was idle:

- training: planar `q2`, `q2_dn4swv`, `q2_dnpool`, `q2_dnsig` and lever `hg2`, `cal2`, each +20 steps from
  the real base checkpoint on the real datasets and schedules — all wrote a checkpoint (rc 0);
- eval: 2-episode `lerobot-eval` against a fresh planar node (port 6023) and a fresh lever splat node
  (6045) — both wrote `eval_info.json`;
- analysis: planar `build_sigma_schedule_k` rebuilt the K=2 schedules from the cached deltas
  (pooled identical to shipped); `measure_sigma_multi` re-measured s1 round 1 (pooled covariance within
  ~3 % of shipped, stochastic); lever `measure_w_lever` reproduced W = 7.94 for rounds 1,2 exactly and
  `build_pooled_schedule_lever` reproduced the K=2 schedule bit-for-bit; `measure_sigma_lever` re-measured
  round 1 with the BC policy;
- tables: `gen_table.py` regenerates a tabular identical to `table1_tabular.tex`; `lever_table.py` prints Table II.
- orchestrator: `dagger_cmds/05dag_cmd.sh --dry-run` resolves every round as complete on this machine, and with a
  fresh `--run_tag` prints the exact record → stats → fine-tune commands it would run (a real round takes hours).
