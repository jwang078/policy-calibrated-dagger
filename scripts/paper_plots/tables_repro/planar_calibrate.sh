#!/bin/bash
# Calibration for one planar lineage and round: measure the round-K HG-DAgger
# policy's open-loop deviations along every intervention of rounds 1..K, then
# scale them to W = 8 and write the pooled (Sigma-bar^alpha) and per-step
# (Sigma^alpha_t) schedules.
#
#   bash planar_calibrate.sh <lineage s1..s5|cl> <K>
#
# Needs the lineage's HG-DAgger arm q{K} (planar_arms.sh). Writes
#   analysis/sigma_deltas_<lineage>q<K>_dag<R>.npz   R = 1..K   (skipped when present)
#   analysis/noise_schedule_pooled_<tag>_K<K>.json, noise_schedule_sigma_alpha_<tag>_K<K>.json
# For the closed-loop lineage (cl) the measuring policy is the closed-loop
# lineage's own round-(K-1) fine-tune, see planar_closed_loop.sh.
S=$(cd "$(dirname "$0")" && pwd); source "$S/lib_repro.sh"; LOG_TAG=calibrate
SFX=$1; K=$2; TAG=$(lineage_tag $SFX) || exit 1; GD=$(lineage_group $SFX); ST=$(schedule_tag $SFX)
POLICY=${POLICY:-$OUT_TRAIN/$GD/q$K/checkpoints/last/pretrained_model}
[ -d "$POLICY" ] || { log "no measuring policy at $POLICY"; exit 1; }
cd "$LR"
log "measuring $SFX K=$K with $(basename $(dirname $(dirname $(dirname $POLICY))))/$(basename $(dirname $(dirname $POLICY)))"
$PY $S/analysis/measure_sigma_multi.py ${ST}q${K} "$POLICY" $TAG "$(seq -s, 1 $K)" || exit 1
$PY $S/analysis/build_sigma_schedule_k.py $ST $TAG $K || exit 1
ls -la $A/noise_schedule_pooled_${ST}_K${K}.json $A/noise_schedule_sigma_alpha_${ST}_K${K}.json | awk '{print $5, $9}'
