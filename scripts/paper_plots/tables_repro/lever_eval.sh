#!/bin/bash
# Table II eval seeds: every (round, training seed, arm) checkpoint of rounds
# 1-2 evaluated with the two eval seeds it has not seen yet (100 scenarios
# each), so each training seed has 3 x 100 episodes; plus the BC checkpoint
# with eval seeds 1 and 2. One lane = one sim node; run several lanes on
# different ports to reorder the work (evals are GPU-bound, so concurrent lanes
# do not add throughput).
#
#   bash lever_eval.sh <lane> <nlanes> <port>          e.g. lanes 0,1,2 of 3 on ports 6047,6048,6049
#   KS="1 2" bash lever_eval.sh 0 1 6047               rounds to cover (default 1 2)
#   BC_ONLY=1 bash lever_eval.sh 0 1 6048              just the BC eval seeds
# Results: outputs/eval300/lever_cam/r84_{hg,cal}<K>_20k[_s<train seed>]_e<eval seed>, bc_r84_e<seed>.
S=$(cd "$(dirname "$0")" && pwd); source "$S/lib_repro.sh"; LOG_TAG="lever-eval$1"
LANE=$1; NL=$2; PORT=$3; CKD=$(printf %06d $LEVER_STEPS)
ckpt() {  # K TRAIN_SEED hg|cal
  local K=$1 TS=$2 ARM=$3 SFX; SFX=$([ $TS = 0 ] || echo _s$TS)
  if [ $ARM = cal ]; then echo "$LEVER_CAL_DIR/q${K}_dnpool$SFX/checkpoints/$CKD/pretrained_model"
  elif [ $TS = 0 ] && [ $K = 1 ]; then echo "$OUT_TRAIN/${LEVER_BASE_NAME}_${LEVER_RUN_TAG}_ft_dag1/checkpoints/$CKD/pretrained_model"
  else echo "$LEVER_HG_DIR/hg${K}$SFX/checkpoints/$CKD/pretrained_model"; fi; }
JOBS=()   # "name|checkpoint|eval seed"
for SD in 1 2; do JOBS+=("bc_r${RES}_e$SD|$LEVER_BASE/checkpoints/last/pretrained_model|$SD"); done
if [ "${BC_ONLY:-0}" != 1 ]; then
  # round 1 first; within a round the first missing eval seed of every checkpoint, then the second
  for K in ${KS:-1 2}; do for PASS in 1 2; do for TS in 0 1 2; do for ARM in cal hg; do
    ES=$(for s in 0 1 2; do [ $s != $TS ] && echo $s; done | sed -n ${PASS}p)
    JOBS+=("r${RES}_${ARM}${K}_20k$([ $TS = 0 ] || echo _s$TS)_e$ES|$(ckpt $K $TS $ARM)|$ES")
  done; done; done; done
fi
mkdir -p "$OUT_EVAL"; SIM=$(start_lever_node $PORT); trap 'stop_node $SIM' EXIT; wait_port $PORT || exit 1
n=0; for J in "${JOBS[@]}"; do
  if [ $((n % NL)) = $LANE ]; then IFS='|' read -r NAME CK ES <<< "$J"; eval_lever "$NAME" "$CK" $PORT $ES; fi; n=$((n+1))
done
log "LANE${LANE}_DONE"
