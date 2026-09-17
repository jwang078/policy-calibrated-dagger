#!/bin/bash
# Smoke test for the reproduction scripts: runs the real scripts in miniature
# (+20 training steps, 2 eval episodes) against the real base checkpoints,
# datasets and cached calibration deltas, everything redirected to
# lerobot/outputs/smoke/ so nothing under this directory or outputs/training
# is touched. Needs the GPU. ~10 minutes.
#
#   bash smoke_test.sh                     # everything
#   PARTS="dry train" bash smoke_test.sh   # a subset of: dry train eval analysis table
#
#   dry       DRY=1 planar_arms.sh prints the full-size training commands (175k steps) without running
#   train     planar_arms.sh (hg, dn4, pool, sig arms of s1 at K=2) and lever_seeds.sh (hg2/cal2 with a
#             fresh training seed, incl. the reseeded BC copy) — each +20 steps, then 2-episode evals
#   eval      planar_eval.sh on the smoke q2 arm (fresh planar node)
#   analysis  planar_calibrate.sh: K=2 schedules rebuilt from cached deltas (must match the shipped ones
#             bit-for-bit), K=1 re-measured with the real q1 policy (stochastic, pooled row within a few %);
#             lever: measure_w_lever (must reproduce W = 7.94), build_pooled_schedule_lever K=2 (bit-for-bit),
#             measure_sigma_lever K=1 with the BC policy
#   table     gen_table.py (tabular identical to table1_tabular.tex) and analysis/lever_table.py
# Last run: see README (2026-09-17, all parts passed).
S=$(cd "$(dirname "$0")" && pwd); LR=/home/jennyw2/code/lerobot; PY=$HOME/miniforge3/envs/splatsim/bin/python
PARTS=${PARTS:-"dry train eval analysis table"}
SM=$LR/outputs/smoke; rm -rf "$SM"; mkdir -p "$SM/training" "$SM/eval300" "$SM/tables_repro/analysis"
ln -sf "$S"/analysis/*.npz "$S"/analysis/*.json "$S"/analysis/*.txt "$SM/tables_repro/analysis/" 2>/dev/null
# the environment every script call below runs under
export OUT_TRAIN=$SM/training OUT_EVAL=$SM/eval300 TABLES_REPRO_DIR=$SM/tables_repro
export PLANAR_BASE=$LR/outputs/training/diffusion_planar_3joint_12_delta_stateng LEVER_BASE=$LR/outputs/training/diffusion_approach_lever_13_smooth_r84_delta_basewrist
export PLANAR_STEPS=75020 LEVER_STEPS=75020 N_EPISODES=2 LEVER_N=2
FAIL=0; log() { echo "[smoke $(date +%H:%M:%S)] $*"; }
check() { if [ "$1" = 0 ]; then log "PASS $2"; else log "FAIL $2 (rc=$1)"; FAIL=1; fi; }
sched_diff() {  # NEW SHIPPED -> max |diff| over all rows
  $PY -c "
import json,numpy as np,sys
a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2])); md=0.0
for r in a:
    for e in a[r]:
        x=np.array(a[r][e]); y=np.array(b[r][e]); md=max(md, float(np.max(np.abs(x-y))) if x.shape==y.shape else float('inf'))
print(f'{md:.3g}')" "$1" "$2"; }

if [[ " $PARTS " == *" dry "* ]]; then
  OUT=$(PLANAR_STEPS=175000 DRY=1 ARMS="hg dn4 pool" bash $S/planar_arms.sh s1 1 2>&1); RC=$?
  echo "$OUT" | grep -q -- "--steps=175000" && echo "$OUT" | grep -q -- "--dataset.dart_state_noise_schedule=" || RC=$((RC+100))
  check $RC "dry-run planar_arms.sh s1 K=1 (hg, dn4, pool): full-size commands printed"
fi
if [[ " $PARTS " == *" train "* ]]; then
  ARMS="hg dn4 pool sig" bash $S/planar_arms.sh s1 2 > "$SM/planar_arms.log" 2>&1; RC=$?
  for A in q2 q2_dn4swv q2_dnpool q2_dnsig; do [ -d "$OUT_TRAIN/scarcity_study/$A/checkpoints/075020/pretrained_model" ] || { RC=$((RC+100)); log "missing $A"; }; done
  check $RC "train planar_arms.sh s1 K=2: q2, q2_dn4swv, q2_dnpool, q2_dnsig (+20 steps)"
  SEEDS=7 ROUNDS=2 bash $S/lever_seeds.sh > "$SM/lever_seeds.log" 2>&1; RC=$?
  for A in lever_r84_fb/hg2_s7 lever_r84_calib/q2_dnpool_s7; do [ -d "$OUT_TRAIN/$A/checkpoints/075020/pretrained_model" ] || { RC=$((RC+100)); log "missing $A"; }; done
  [ -f "$OUT_TRAIN/diffusion_approach_lever_13_smooth_r84_delta_basewrist_seed7/checkpoints/075000/training_state/rng_state.safetensors" ] || RC=$((RC+100))
  check $RC "train lever_seeds.sh seed 7 round 2: reseeded BC copy, hg2_s7, q2_dnpool_s7 (+20 steps)"
  for A in r84_hg2_20k_s7 r84_cal2_20k_s7; do [ -f "$OUT_EVAL/lever_cam/$A/eval_info.json" ]; check $? "eval lever $A (2 episodes, seed 7)"; done
fi
if [[ " $PARTS " == *" eval "* ]]; then
  bash $S/planar_eval.sh 6023 s1 q2 > "$SM/planar_eval.log" 2>&1; RC=$?
  [ -f "$OUT_EVAL/scarcity_study/q2/eval_info.json" ] || RC=$((RC+100)); check $RC "eval planar_eval.sh s1 q2 (2 episodes, port 6023)"
fi
if [[ " $PARTS " == *" analysis "* ]]; then
  A=$TABLES_REPRO_DIR/analysis
  rm -f $A/noise_schedule_pooled_s1_K2.json $A/noise_schedule_sigma_alpha_s1_K2.json     # rebuild from the cached deltas
  POLICY=$LR/outputs/training/scarcity_study/q2/checkpoints/last/pretrained_model bash $S/planar_calibrate.sh s1 2 > "$SM/calibrate_k2.log" 2>&1; RC=$?
  D=$(sched_diff $A/noise_schedule_pooled_s1_K2.json $S/analysis/noise_schedule_pooled_s1_K2.json); [ "$D" = 0 ] || RC=$((RC+100))
  check $RC "analysis planar_calibrate.sh s1 K=2 from cached deltas (pooled max|diff| vs shipped = $D)"
  rm -f $A/sigma_deltas_s1q1_dag1.npz $A/noise_schedule_pooled_s1_K1.json $A/noise_schedule_sigma_alpha_s1_K1.json
  POLICY=$LR/outputs/training/scarcity_study/q1/checkpoints/last/pretrained_model bash $S/planar_calibrate.sh s1 1 > "$SM/calibrate_k1.log" 2>&1; RC=$?
  [ -f $A/sigma_deltas_s1q1_dag1.npz ] || RC=$((RC+100))
  check $RC "analysis planar_calibrate.sh s1 K=1 re-measured with q1 (pooled max|diff| vs shipped = $(sched_diff $A/noise_schedule_pooled_s1_K1.json $S/analysis/noise_schedule_pooled_s1_K1.json), rows are O(10-30), stochastic)"
  PFX=lever_d100_03dagcap_r84_diff; cd $LR
  INT_PREFIX=$PFX $PY $S/analysis/measure_w_lever.py 1,2 > "$A/w_lever_r84_rounds1,2.smoke.txt" 2>&1; RC=$?
  W=$(grep -oE "W = [0-9.]+" "$A/w_lever_r84_rounds1,2.smoke.txt" | head -1 | awk '{print $3}'); WS=$(grep -oE "W = [0-9.]+" "$S/analysis/w_lever_r84_rounds1,2.txt" | head -1 | awk '{print $3}')
  [ "$W" = "$WS" ] || RC=$((RC+100)); check $RC "analysis lever measure_w_lever rounds 1,2 (W=$W, shipped $WS)"
  rm -f $A/noise_schedule_pooled_lever_r84_K2.json
  INT_PREFIX=$PFX TAGPFX=r84_ $PY $S/analysis/build_pooled_schedule_lever.py 2 $W > /dev/null; RC=$?
  D=$(sched_diff $A/noise_schedule_pooled_lever_r84_K2.json $S/analysis/noise_schedule_pooled_lever_r84_K2.json); [ "$D" = 0 ] || RC=$((RC+100))
  check $RC "analysis lever build_pooled_schedule_lever K=2 from cached deltas (max|diff| vs shipped = $D)"
  rm -f $A/sigma_deltas_lever_r84_q1_dag1.npz
  $PY $S/analysis/measure_sigma_lever.py r84_q1_dag1 $LEVER_BASE/checkpoints/last/pretrained_model ${PFX}_r_dag1 > "$SM/measure_sigma_lever.log" 2>&1; RC=$?
  [ -f $A/sigma_deltas_lever_r84_q1_dag1.npz ] || RC=$((RC+100)); check $RC "analysis lever measure_sigma_lever K=1 with the BC policy"
fi
if [[ " $PARTS " == *" table "* ]]; then
  cd $LR; $PY $S/gen_table.py > /dev/null; RC=$?
  diff -q <(sed -n '/begin{tabular}/,/end{tabular}/p' $S/table_full_benchmark.tex) <(sed -n '/begin{tabular}/,/end{tabular}/p' $S/table1_tabular.tex) > /dev/null || RC=$((RC+100))
  check $RC "table gen_table.py: tabular identical to table1_tabular.tex"
  $PY $S/analysis/lever_table.py > /dev/null; check $? "table analysis/lever_table.py"
fi
[ $FAIL = 0 ] && log "SMOKE OK" || log "SMOKE FAILED"; exit $FAIL
