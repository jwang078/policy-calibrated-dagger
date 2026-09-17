#!/bin/bash
# 300-episode evaluation (3 passes over the 100 fixed scenarios) of planar arms
# on a fresh sim node. Results: outputs/eval300/<group>/<arm>[_e<seed>]/eval_info.json.
#
#   bash planar_eval.sh <port> <lineage s1..s5|cl> [arm ...]      default: every Table I arm of the lineage
#   SEED=1 bash planar_eval.sh 6024 cl q1_dnpoolcl q2_dnpoolcl    extra eval seeds (closed-loop row, BC row)
#   bash planar_eval.sh 6025 bc                                    the BC row (outputs/training/bc175k)
#
# Never point two evals at one port; ports 6023-6031 were used for planar evals.
S=$(cd "$(dirname "$0")" && pwd); source "$S/lib_repro.sh"; LOG_TAG=eval
PORT=$1; SFX=$2; shift 2; SEED=${SEED:-0}
if [ "$SFX" = bc ]; then GD=base; ARMS=${*:-bc175k}; else
  GD=$(lineage_group $SFX) || exit 1
  if [ $# -gt 0 ]; then ARMS=$*; else ARMS=""; for K in 1 2 3 4 5; do
    if [ $SFX = cl ]; then ARMS="$ARMS q${K}_dnpoolcl"; else ARMS="$ARMS q$K q${K}_dnpool q${K}_dnsig"; for SG in 2 4 8 12 16; do ARMS="$ARMS q${K}_dn${SG}swv"; done; fi; done; fi
fi
mkdir -p "$OUT_EVAL"; SIM=$(start_planar_node $PORT); trap 'stop_node $SIM' EXIT; wait_port $PORT || exit 1
for ARM in $ARMS; do CK=$OUT_TRAIN/$GD/$ARM; [ "$GD" = base ] && CK=$OUT_TRAIN/$ARM; eval_planar $GD $ARM "$CK/checkpoints/last/pretrained_model" $PORT $SEED; done
log "EVAL_DONE $GD on $PORT"
