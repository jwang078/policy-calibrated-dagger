#!/bin/bash
# Table I arms for one planar lineage: every cell is the BC checkpoint (75k)
# resumed to 175k on base + rounds 1..K of that lineage's interventions.
#
#   bash planar_arms.sh <lineage s1..s5|cl> [K ...]        default K = 1 2 3 4 5
#   ARMS="hg dn2 dn4 dn8 dn12 dn16 pool sig" bash planar_arms.sh s3
#
# Arms (training dir outputs/training/<group>/<arm>):
#   hg          q{K}            HG-DAgger: the aggregated data, no injected noise
#   dn<sigma>   q{K}_dn<s>swv   DART with a fixed isotropic sigma (med-step units)
#   pool        q{K}_dnpool     pooled Sigma-bar^alpha schedule   (Ours)
#   sig         q{K}_dnsig      per-step Sigma^alpha_t schedule   (Ours)
#   for lineage cl only:  pool -> q{K}_dnpoolcl on the closed-loop datasets
# pool/sig call planar_calibrate.sh when the schedule is missing (needs hg first).
# Datasets: the lineage's interventions must exist (dagger_cmds/<tag>_cmd.sh).
S=$(cd "$(dirname "$0")" && pwd); source "$S/lib_repro.sh"; LOG_TAG=arms
SFX=$1; shift; KS=${*:-"1 2 3 4 5"}; ARMS=${ARMS:-"hg dn2 dn4 dn8 dn12 dn16 pool sig"}
TAG=$(lineage_tag $SFX) || exit 1; GD=$(lineage_group $SFX); ST=$(schedule_tag $SFX); G=$OUT_TRAIN/$GD
[ -d "$PLANAR_BASE/checkpoints/last" ] || { log "no BC checkpoint at $PLANAR_BASE (dagger_cmds round 0 trains it)"; exit 1; }
for K in $KS; do
  planar_ensure_stats $TAG $K
  for ARM in $ARMS; do
    case $ARM in
      hg)   planar_train $G/q$K "${GD}_q$K" $TAG $K -- --dataset.dart_relabel=false ;;
      dn*)  SIG=${ARM#dn}; planar_train $G/q${K}_dn${SIG}swv "${GD}_q${K}_dn${SIG}swv" $TAG $K -- "${DART_COMMON[@]}" --dataset.dart_state_noise_std=$SIG ;;
      pool|sig)
        [ $ARM = pool ] && { NAME=q${K}_dnpool; SCHED=$A/noise_schedule_pooled_${ST}_K${K}.json; } || { NAME=q${K}_dnsig; SCHED=$A/noise_schedule_sigma_alpha_${ST}_K${K}.json; }
        [ $SFX = cl ] && NAME=${NAME}cl
        [ -f "$SCHED" ] || bash $S/planar_calibrate.sh $SFX $K || { log "no schedule for $SFX K=$K, skip $NAME"; continue; }
        planar_train $G/$NAME "${GD}_$NAME" $TAG $K -- "${DART_COMMON[@]}" "${DART_SCHED[@]}" --dataset.dart_state_noise_schedule=$SCHED ;;
      *) log "unknown arm $ARM"; exit 1 ;;
    esac
  done
done
log "ARMS_DONE $SFX K=$KS"
