#!/bin/bash
# Table II training seeds: re-train the HG-DAgger and calibrated arms of rounds
# 1-2 from the BC checkpoint with different seeds, same datasets and schedules
# as seed 0, then 100-scenario evals with eval seed = training seed.
#
#   SEEDS="1 2" ROUNDS="1 2" bash lever_seeds.sh
#
# lerobot's resume restores the checkpoint RNG *after* applying --seed, so a
# plain seed flag would replicate seed 0. Each seed therefore gets its own copy
# of the BC checkpoint (hard links) with a freshly seeded
# training_state/rng_state.safetensors: <BC>_seed<N>.
# Arms: lever_r84_fb/hg<K>_s<N>, lever_r84_calib/q<K>_dnpool_s<N>; evals r84_{hg,cal}<K>_20k_s<N>.
S=$(cd "$(dirname "$0")" && pwd); source "$S/lib_repro.sh"; LOG_TAG=lever-seeds
SEEDS=${SEEDS:-"1 2"}; ROUNDS=${ROUNDS:-"1 2"}; CKD=$(printf %06d $LEVER_STEPS); BC_CK=075000
make_seed_base() {  # SEED -> training dir on stdout
  local SD=$1 D=$OUT_TRAIN/${LEVER_BASE_NAME}_seed$SD
  if [ ! -f "$D/checkpoints/$BC_CK/training_state/rng_state.safetensors" ]; then
    mkdir -p "$D/checkpoints/$BC_CK/training_state"; cp -al "$LEVER_BASE/checkpoints/$BC_CK/pretrained_model" "$D/checkpoints/$BC_CK/pretrained_model"
    local f; for f in optimizer_param_groups.json optimizer_state.safetensors scheduler_state.json training_step.json; do ln "$LEVER_BASE/checkpoints/$BC_CK/training_state/$f" "$D/checkpoints/$BC_CK/training_state/$f"; done
    ln -sfn "$BC_CK" "$D/checkpoints/last"
    $PY - "$LR" "$SD" "$D/checkpoints/$BC_CK/training_state/rng_state.safetensors" <<'EOF' >&2
import sys; sys.path.insert(0, sys.argv[1] + "/src")
import random, numpy as np, torch
from safetensors.torch import save_file
from lerobot.utils.random_utils import serialize_rng_state
sd = int(sys.argv[2]); random.seed(sd); np.random.seed(sd); torch.manual_seed(sd); torch.cuda.manual_seed_all(sd)
save_file(serialize_rng_state(), sys.argv[3]); print("rng reseeded", sd)
EOF
  fi
  echo "$D"
}
mkdir -p "$OUT_EVAL"; SIM1=$(start_lever_node 6045); SIM2=$(start_lever_node 6046); trap 'stop_node $SIM1; stop_node $SIM2' EXIT; wait_port 6045 && wait_port 6046 || exit 1
E1=""; E2=""
for SD in $SEEDS; do
  LEVER_TRAIN_BASE=$(make_seed_base $SD); log "seed $SD base: $LEVER_TRAIN_BASE"; export LEVER_TRAIN_BASE
  for K in $ROUNDS; do
    SCHED=$A/noise_schedule_pooled_lever_r${RES}_K${K}.json; [ -f "$SCHED" ] || { log "no schedule $SCHED (lever_round.sh $K first)"; exit 1; }
    lever_train "$LEVER_HG_DIR/hg${K}_s$SD" lever_r${RES}_hg${K}_s$SD $K -- --seed=$SD
    lever_train "$LEVER_CAL_DIR/q${K}_dnpool_s$SD" lever_r${RES}_cal${K}_s$SD $K -- --seed=$SD "${DART_COMMON[@]}" "${DART_SCHED[@]}" --dataset.dart_state_noise_schedule=$SCHED
    [ -n "$E1" ] && wait $E1 $E2   # the previous pair's evals must finish before the nodes are reused
    ( eval_lever r${RES}_hg${K}_20k_s$SD "$LEVER_HG_DIR/hg${K}_s$SD/checkpoints/$CKD/pretrained_model" 6045 $SD ) & E1=$!
    ( eval_lever r${RES}_cal${K}_20k_s$SD "$LEVER_CAL_DIR/q${K}_dnpool_s$SD/checkpoints/$CKD/pretrained_model" 6046 $SD ) & E2=$!
  done
done
[ -n "$E1" ] && wait $E1 $E2; log "LEVER_SEEDS_DONE"
