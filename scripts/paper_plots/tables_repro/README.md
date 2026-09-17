# Reproducing the two results tables

Everything behind Table I (planar reaching, `tab:overall_success`) and Table II (engine lever, image
observations, `tab:overall_success_splatsim`) of the Policy-Calibrated DAgger paper.

- `lib_repro.sh` — shared paths, dataset/weight builders, training and eval helpers; every script below sources it.
- `planar_arms.sh`, `planar_calibrate.sh`, `planar_eval.sh`, `planar_closed_loop.sh` — Table I.
- `lever_bc.sh`, `lever_round.sh`, `lever_seeds.sh`, `lever_eval.sh` — Table II.
- `dagger_cmds/` — the DAgger orchestrator commands that trained the planar BC and collected the five lineages.
- `analysis/` — calibration code (W fit, sigma measurement, schedule builders), dataset shrinking, and the
  Table II generator. The exact noise schedules used by every calibrated arm are committed as one tarball:
  **run `bash analysis/unpack_schedules.sh` once after cloning.** The per-anchor deltas they were built from
  (`sigma_deltas_*.npz`, ~320 MB) are kept here on the machine, git-ignored.
- `gen_table.py` — Table I LaTeX (`table1_tabular.tex` is its output); `analysis/lever_table.py` — Table II.
- `smoke_test.sh` — runs the scripts above in miniature (see the end).
- `history/` — the scripts exactly as they ran during the paper crunch (gated chains, per-round one-offs,
  K=6 attempts). The clean scripts are rewrites of them with the same commands; keep the history for
  provenance, run the clean ones.

Conventions: outputs go to `lerobot/outputs/training` (checkpoints) and `lerobot/outputs/eval300` (evals),
datasets live in `~/.cache/huggingface/lerobot/JennyWWW`. Long jobs were launched detached
(`setsid nohup bash script.sh > script.out 2>&1 < /dev/null &`) so editor crashes don't kill them.

> **Learning-rate caveat (found 2026-09-16).** Every Table I arm passed `--optimizer.lr=1e-6`, but until the
> 2026-09-14 patch in `src/lerobot/common/train_utils.py` the resumed optimizer state silently overrode that
> flag, so the K=1..5 arms actually trained on the BC checkpoint's own cosine schedule (1e-5 peak, ~6e-6 at
> step 76k, decaying to 0 at 175k; see `lr:` in their logs). With current code the flag is honoured, so the
> clean scripts pass `PLANAR_LR=1e-5` by default to reproduce those arms (training K=6 at a true 1e-6 gave
> ~77% instead of ~87% for HG-DAgger on three lineages). Table II (lever) was trained entirely after the patch
> at a true constant 1e-6 and is internally consistent.

---

## Table I — planar reaching (5 lineages, 300 episodes per cell)

One "lineage" = one independently collected DAgger run. Lineages `s1…s5` are orchestrator tags
`05dag…09dag` (intervention sample seeds 0…4); their training groups are `outputs/training/scarcity_study{,_s2,…,_s5}`.
Every table cell is a policy fine-tuned **from the same BC 75k checkpoint** on base + rounds 1..K of that
lineage, to 175k steps, then evaluated with 300 episodes (100 fixed scenarios × 3). Cells are mean ± SE across
the 5 lineages.

```
bash dagger_cmds/05dag_cmd.sh          # 1. BC (round 0, 75k steps on 500 demos) + the s1 lineage, rounds 1-5; 06dag…09dag = s2…s5
bash planar_arms.sh s1                 # 2. HG-DAgger (q{K}), fixed sigma 2/4/8/12/16 (q{K}_dn{s}swv), pooled (q{K}_dnpool), per-step (q{K}_dnsig), K=1..5
bash planar_eval.sh 6023 s1            # 3. 300-episode evals of every arm on a fresh sim node (one port per lane)
bash planar_closed_loop.sh             # 4. the closed-loop lineage (10cl), then: ARMS=pool bash planar_arms.sh cl; bash planar_eval.sh 6027 cl
bash planar_eval.sh 6025 bc            # 5. the BC row (outputs/training/bc175k = the BC schedule run to 175k)
SEED=1 bash planar_eval.sh 6026 cl     #    extra eval seeds for the single-checkpoint rows (closed-loop, BC): SEED=1, SEED=2
python gen_table.py                    # 6. Table I (reads outputs/eval300/*)
```

Details:

- The orchestrator (`my_scripts/dagger_orchestrate_sweep.sh`) rolls the previous lineage policy on the 100
  scenarios, records RRT-expert interventions on the failures (`planar_12_<tag>_diff_r_dag<K>`), and
  fine-tunes the lineage +10k (`…_<tag>_ft_dag<K>`, only used to collect the next round; the DART flag it
  carries is a no-op: zero noise, no pattern). `--resume` picks up completed rounds; `--num_rounds=6` extends.
- `planar_arms.sh`: interventions get an exact 0.3 share of every batch, split equally over rounds; noised
  arms self-relabel the intervention repos with DART (`dart_vel_noise_std=0.3`, hold-tail masking), either
  with a fixed isotropic sigma (med-step units) or with a schedule. `ARMS=…` and the K list select subsets.
- `planar_calibrate.sh s K` (called by `planar_arms.sh` when a schedule is missing): samples the round-K HG
  policy open loop along every intervention of rounds 1..K (`analysis/measure_sigma_multi.py`, deltas to
  `analysis/sigma_deltas_s{s}q{K}_dag{R}.npz`), then `analysis/build_sigma_schedule_k.py` scales them with
  W = 8.0 (the closed-loop settling error from the blend dial, stable across rounds on planar: 13 ratios in
  [0.05, 0.9] on lineage s1) into `noise_schedule_pooled_s{s}_K{K}.json` (Σ̄^α) and
  `noise_schedule_sigma_alpha_s{s}_K{K}.json` (Σ^α_t). The policy samples are not seeded, so a re-measurement
  reproduces the pooled covariance to within a few percent, not bit-for-bit. (Historical note: the s1/s2 K=2
  schedules were measured under the tags `s1`/`s2`; the deltas are linked under `s1q2`/`s2q2` too and the
  builder regenerates them from there, pooled identical, per-step within 1e-3.)
- Closed-loop row: `planar_closed_loop.sh` collects each round with the calibrated policy itself (round 1
  shared with s1), re-measuring the schedule on the collecting checkpoint each round; the table arm is then
  trained from base on those datasets like every other cell (`q{K}_dnpoolcl` in `scarcity_study_cl`).
- The ± on the closed-loop and BC rows is the SE over three evaluation seeds of one checkpoint, not over lineages.

## Table II — engine lever, image observations (84 px)

All arms are fine-tuned **from the same BC 75k checkpoint** to 95k (+20k, constant lr 1e-6), on base +
rounds 1..K; HG-DAgger and calibrated arms of round K share the same interventions. Cells are mean ± SE over
3 training seeds, each seed averaged over 3 eval seeds × 100 scenarios (900 episodes). The splat node always
renders 224 px; the policy resizes internally, and intervention videos are re-encoded to 84 px.

```
bash lever_bc.sh                       # 1. 84 px datasets, BC (75k), BC eval (outputs/eval300/lever_cam/bc_r84)
bash lever_round.sh 1                  # 2. round K: record dag K, HG rK from base, blend dial (6 ratios), W + sigma + pooled
bash lever_round.sh 2                  #    schedule (analysis/noise_schedule_pooled_lever_r84_K{K}.json), calibrated rK from base,
                                       #    100-scenario evals of both (r84_{hg,cal}{K}_20k); rounds 3-4 were run but are not in the paper
SEEDS="1 2" ROUNDS="1 2" bash lever_seeds.sh    # 3. training seeds 1, 2 (reseeded BC copies), evals with eval seed = training seed
bash lever_eval.sh 0 3 6047 &          # 4. the other two eval seeds of every checkpoint + BC seeds 1, 2 (lanes 1 and 2 on 6048, 6049)
python analysis/lever_table.py         # 5. Table II
```

Details:

- `lever_round.sh K`: for K=1 the orchestrator's own +20k fine-tune is the HG r1 arm (inline eval at 95k);
  for K≥2 the orchestrator only records (it is stopped after the stats sidecar) and HG rK is trained from
  base. The collecting policy (BC for round 1, HG r(K−1) after) is also the one blended and measured: W is
  refit on rounds 1..K each round (`analysis/measure_w_lever.py`; 8.10 / 7.94 / 8.07 / 8.06 for K=1..4),
  Σ̂ comes from `analysis/measure_sigma_lever.py`, the schedule from `analysis/build_pooled_schedule_lever.py`.
- `lever_seeds.sh`: lerobot's resume restores the checkpoint RNG _after_ applying `--seed`, so a plain seed
  flag replicates seed 0; each seed gets a hard-linked copy of the BC checkpoint with a freshly seeded
  `training_state/rng_state.safetensors`.
- Evals are GPU-bound (splat rendering): concurrent lever evals give no throughput gain over one, so lanes only reorder work.

## Environment knobs (see the top of `lib_repro.sh`)

`OUT_TRAIN`, `OUT_EVAL` (output roots), `TABLES_REPRO_DIR` (where `analysis/*.py` read and write deltas and
schedules), `PLANAR_STEPS` / `LEVER_STEPS` / `PLANAR_LR` / `*_WORKERS`, `N_EPISODES` (planar evals, 300),
`LEVER_N` (lever evals, 100), `SEED` (planar eval seed), `DRY=1` (print training commands only).
Lever calibration scripts take `INT_PREFIX` (intervention repo prefix) and `TAGPFX` (`r84_`).

## Machine notes that bit us

- 30 GB RAM: one lever trainer (~13 GB with 8 dataloader workers) + two splat nodes + evals is the limit;
  a second trainer or three nodes OOM-kills VS Code and the desktop app. Planar trainers are light.
- Never put `pkill -f pattern` in a command that also contains the unbracketed pattern text (it kills the
  calling shell); use `pkill -f 'name[.]sh'`.
- Every eval/collection lane needs its own sim port. Ports used: planar lineages 6001, planar evals
  6023–6031, lever blends 6042/6043, lever evals 6045–6049.

## Smoke test

`bash smoke_test.sh` (~10 min, GPU) runs the clean scripts in miniature — +20 training steps from the real
base checkpoints on the real datasets and schedules, 2-episode evals against fresh sim nodes, the calibration
scripts against the cached deltas — with every output redirected to `lerobot/outputs/smoke/`, and compares
what it can against the shipped artifacts. Last run 2026-09-17, after submission, all parts passed:

- dry: `planar_arms.sh` prints the full-size (175k-step) commands with the schedule flags;
- train: `planar_arms.sh s1 2` (q2, q2_dn4swv, q2_dnpool, q2_dnsig) and `lever_seeds.sh` (seed 7, round 2:
  reseeded BC copy, hg2_s7, q2_dnpool_s7, then their 2-episode evals on two lever nodes);
- eval: `planar_eval.sh 6023 s1 q2`;
- analysis: `planar_calibrate.sh s1 2` rebuilds the K=2 schedules from the cached deltas bit-for-bit;
  `planar_calibrate.sh s1 1` re-measures round 1 with the real q1 policy (pooled row within ~3 % of shipped);
  `measure_w_lever` reproduces W = 7.94 for rounds 1,2 exactly; `build_pooled_schedule_lever` reproduces the
  K=2 schedule bit-for-bit; `measure_sigma_lever` re-measures round 1 with the BC policy;
- table: `gen_table.py` regenerates a tabular identical to `table1_tabular.tex`; `lever_table.py` prints Table II.

The DAgger orchestrator itself is only dry-run (`bash dagger_cmds/05dag_cmd.sh --dry-run` resolves every
round as complete on this machine and, with a fresh `--run_tag`, prints the record → stats → fine-tune
commands it would run); a real round takes hours.
