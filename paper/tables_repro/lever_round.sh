#!/bin/bash
# Table II round K (from-base protocol): one paired HG-DAgger / calibrated cell.
#
#   bash lever_round.sh <K>          rounds in order: 1, 2, 3, ...
#
#   1. record: the DAgger orchestrator rolls the round-(K-1) HG policy (K=1: the BC) over the 100
#      scenarios, records RRT-expert interventions on the ones it fails (dataset ${LEVER_PFX}_r_dag<K>,
#      224 px), writes the stats sidecar. For K=1 its own +20k fine-tune IS the HG r1 arm (inline eval at
#      95k); for K>=2 it is stopped after the stats and the intervention videos are re-encoded to 84 px.
#   2. HG rK from the BC checkpoint on base + dag1..K (+20k, constant lr 1e-6): lever_r84_fb/hg<K>
#   3. blend dial: partial-denoising rollouts of the collecting policy at 6 ratios on dag K (two nodes)
#   4. calibration: W fit on rounds 1..K, Sigma-hat of the collecting policy along dag1..K, pooled schedule
#   5. calibrated rK from the BC checkpoint with that schedule: lever_r84_calib/q<K>_dnpool
#   6. 100-scenario evals of both arms (eval seed 0) on two nodes -> outputs/eval300/lever_cam/r84_{hg,cal}<K>_20k
# Ports: orchestrator 6001, blends 6042/6043, evals 6045/6046. Needs lever_bc.sh first.
S=$(cd "$(dirname "$0")" && pwd); source "$S/lib_repro.sh"; LOG_TAG="lever-r$1"
K=$1; P=$((K-1)); CKD=$(printf %06d $LEVER_STEPS); SUF=_20k
DAGK=${LEVER_PFX}_r_dag${K}; FT_DIR=$OUT_TRAIN/${LEVER_BASE_NAME}_${LEVER_RUN_TAG}_ft_dag   # orchestrator's per-round dirs
HG_CK() { echo "$LEVER_HG_DIR/hg$1/checkpoints/$CKD/pretrained_model"; }                       # HG r<K> from base
# the policy that collects round K, blends on it and is measured for its schedule
if [ $K = 1 ]; then COLLECTOR=$LEVER_BASE/checkpoints/last/pretrained_model
elif [ $K = 2 ]; then COLLECTOR=${FT_DIR}1/checkpoints/$CKD/pretrained_model
else COLLECTOR=$(HG_CK $P); fi
[ -d "$COLLECTOR" ] || { log "no collecting policy at $COLLECTOR (run round $P first)"; exit 1; }
cd "$LR"
# orchestrator argv: the lever lineage's recorded flags minus the ones set here
INT_X=$(grep -o "intervention_extra_args=.*" $S/lever_orchestrator_argv.txt | sed 's/^intervention_extra_args=//')
ORCH=(); while IFS= read -r a || [ -n "$a" ]; do [ -z "$a" ] && continue
  case "$a" in --resume|--num_rounds=*|--run_tag=*|--base_short=*|--finetune_extra_args=*|--intervention_extra_args=*|--initial_policy_path=*|--exclude_gripper_from_state*|--finetune_steps=*|--finetune_save_freq=*|--finetune_eval_freq=*) ;; *) ORCH+=("$a") ;; esac
done < $S/lever_orchestrator_argv.txt
ORCH+=(--resume --base_short=approach_lever_13_smooth_r${RES} --num_rounds=$K --run_tag=$LEVER_RUN_TAG --initial_policy_path="$LEVER_BASE" --exclude_gripper_from_state=false
  --finetune_steps=$LEVER_FT --finetune_save_freq=20000 --finetune_eval_freq=20000 --intervention_extra_args="$INT_X"
  --finetune_extra_args="--optimizer.lr=1e-6 --dataset.multi_source_feature_intersection=true --scheduler.name=constant --num_workers=4 --wandb.enable=false")

# ---- 0. shim (K>=3): expose HG r(K-1), trained from base, as the orchestrator's round-(K-1) policy dir
if [ $K -ge 3 ] && [ ! -d "${FT_DIR}$P/checkpoints/$CKD" ]; then
  EVP=$OUT_EVAL/$LEVER_EVAL_GROUP/r${RES}_hg${P}${SUF}/eval_info.json; [ -f "$EVP" ] || { log "no eval of hg$P at $EVP"; exit 1; }
  mkdir -p "${FT_DIR}$P/checkpoints" "${FT_DIR}$P/eval"; rm -rf "${FT_DIR}$P/checkpoints/last"
  cp -r "$LEVER_HG_DIR/hg$P/checkpoints/$CKD" "${FT_DIR}$P/checkpoints/$CKD"; ln -s "$CKD" "${FT_DIR}$P/checkpoints/last"
  cp "$EVP" "${FT_DIR}$P/eval/eval_info_step_$CKD.json"
  $PY - "${FT_DIR}$P/checkpoints/$CKD/pretrained_model/train_config.json" $LEVER_STEPS <<'EOF'
import json, sys
c = json.load(open(sys.argv[1])); c["env"]["eval_benchmark_subset"] = list(range(100)); c["steps"] = int(sys.argv[2])
json.dump(c, open(sys.argv[1], "w"), indent=4)
EOF
  log "shim: hg$P exposed as $(basename ${FT_DIR}$P)"
fi
# ---- 1. record round K
if [ ! -f "$LR/outputs/dataset_stats/$DAGK/stats_rel64.json" ]; then
  if [ $K -ge 2 ]; then   # the resume must plan to start at round K step 1 (recording), else abort loudly
    bash my_scripts/dagger_orchestrate.sh "${ORCH[@]}" --dry-run > "$OUT_TRAIN/lever_r${RES}_orchestrator_r${K}_dryrun.log" 2>&1
    grep -q "next is round $K, step 1" "$OUT_TRAIN/lever_r${RES}_orchestrator_r${K}_dryrun.log" || { log "orchestrator would not start at round $K step 1 — see the dry-run log"; exit 1; }
  fi
  log "orchestrator: recording round $K with $(basename $(dirname $(dirname $(dirname $COLLECTOR))))"
  if [ $K = 1 ]; then
    bash my_scripts/dagger_orchestrate.sh "${ORCH[@]}" > "$OUT_TRAIN/lever_r${RES}_orchestrator_r${K}.log" 2>&1; log "orchestrator rc=$?"
  else
    setsid bash my_scripts/dagger_orchestrate.sh "${ORCH[@]}" > "$OUT_TRAIN/lever_r${RES}_orchestrator_r${K}.log" 2>&1 & OPID=$!
    until [ -f "$LR/outputs/dataset_stats/$DAGK/stats_rel64.json" ] || ! kill -0 $OPID 2>/dev/null; do sleep 20; done
    sleep 15; kill -- -$OPID 2>/dev/null; sleep 5; kill -9 -- -$OPID 2>/dev/null
    rm -rf "${FT_DIR}$K/checkpoints"   # its lineage-style fine-tune is not wanted (HG rK is trained from base below)
    log "orchestrator stopped after the dag$K stats"
  fi
  [ -f "$LR/outputs/dataset_stats/$DAGK/stats_rel64.json" ] || { log "no dag$K stats"; exit 1; }
  [ $K = 1 ] || $PY $S/analysis/shrink_video_dataset_r${RES}.py $DAGK $RES > "$OUT_TRAIN/lever_r${RES}_shrink_dag${K}.log" 2>&1 || { log "dag$K shrink failed"; exit 1; }
fi
log "dag$K: $($PY -c "import json;i=json.load(open('$HF/$DAGK/meta/info.json'));print(i['total_episodes'],'episodes',i['total_frames'],'frames')")"
# ---- 2. HG rK from base (K=1: the orchestrator's fine-tune, exposed under the same name)
if [ $K = 1 ]; then mkdir -p "$LEVER_HG_DIR"; [ -e "$LEVER_HG_DIR/hg1" ] || ln -s "${FT_DIR}1" "$LEVER_HG_DIR/hg1"
else lever_train "$LEVER_HG_DIR/hg$K" lever_r${RES}_hg$K $K -- ; fi
# ---- 3. blend dial on dag K with the collecting policy (every other episode; two nodes)
blend_lane() {  # PORT ratios...
  local PORT=$1; shift; local EPS; EPS=$($PY -c "import json;n=json.load(open('$HF/$DAGK/meta/info.json'))['total_episodes'];print('['+','.join(str(i) for i in range(0,n,2))+']')")
  local SIM; SIM=$(start_lever_node $PORT --strict_goal_tolerances); wait_port $PORT || return 1
  local R TAG TGT; for R in "$@"; do TAG=$($PY -c "print(f'{int(round(float($R)*100)):03d}')"); TGT=JennyWWW/${DAGK}_blend$TAG
    [ -d "$HF/${DAGK}_blend$TAG/data" ] && { log "$TGT exists"; continue; }; log "blend r=$R on $PORT -> $TGT"
    nice -n 10 $PY my_scripts/augment_dataset_with_blending.py --dataset_repo_id=JennyWWW/$DAGK --target_dataset_repo_id="$TGT" --policy_path="$COLLECTOR" \
      "--forward_flow_ratios=[$R]" "--episode_indices=$EPS" --samples_per_episode=1 --relabel_actions=guidance --env_external_port=$PORT --env_external_host=127.0.0.1 \
      --env_task=upright_small_engine_new --env_robot_name=robot_iphone_w_engine_curtain --num_dofs=6 --eval_benchmark_repo_id=$LEVER_BENCH \
      --blend_strategy=denoise --guidance_repr=absolute_pos --blend_mode=every_step --blend_interval_frac=0.5 --fixed_base_noise=true --resample_noise_per_reblend=false --clip_sample=false --sample_seed=42 \
      --show_guidance_ghost=false --progress_guidance=true --guidance_from_dart_labels=true --anchor_suffix_steps=0 --anchor_prefix_steps=0 --anchor_every_denoise_step=false --blend_ratio_goal_taper=0 \
      --max_blend_end_gap_steps=100000 --max_blend_end_lag_indices=100000 --pad_after_success=false --rtc_prev_chunk=true --rtc_max_guidance_weight=3 --rtc_hard_prefix_xfade=8 \
      --blend_ratio_backoff=false --blend_tube_steps=0 --max_tube_breach_ratio=0 > "$OUT_TRAIN/lever_r${RES}_blend${K}_$TAG.log" 2>&1; log "$TGT rc=$?"
  done; stop_node $SIM
}
mkdir -p "$OUT_EVAL"; set -- $LEVER_BLEND_RATIOS
( blend_lane 6042 $1 $2 $3 ) & B1=$!; ( blend_lane 6043 $4 $5 $6 ) & B2=$!; wait $B1 $B2; log "blends of round $K done"
# ---- 4. W, Sigma-hat, pooled schedule
ROUNDS=$(seq -s, 1 $K)
INT_PREFIX=$LEVER_PFX $PY $S/analysis/measure_w_lever.py $ROUNDS > "$A/w_lever_r${RES}_rounds${ROUNDS}.txt" 2>&1
W=$(grep -oE "W = [0-9.]+" "$A/w_lever_r${RES}_rounds${ROUNDS}.txt" | head -1 | awk '{print $3}'); log "W(rounds $ROUNDS) = $W"
[ "$($PY -c "w=float('${W:-nan}'); print('yes' if 3<=w<=40 else 'no')")" = yes ] || { log "W out of range — see $A/w_lever_r${RES}_rounds${ROUNDS}.txt"; exit 1; }
for R in $(seq 1 $K); do $PY $S/analysis/measure_sigma_lever.py r${RES}_q${K}_dag$R "$COLLECTOR" "${LEVER_PFX}_r_dag$R" > "$A/sigma_r${RES}_q${K}_dag$R.log" 2>&1 || { log "sigma dag$R failed"; exit 1; }; done
INT_PREFIX=$LEVER_PFX TAGPFX=r${RES}_ $PY $S/analysis/build_pooled_schedule_lever.py $K $W | tee -a "$A/sigma_r${RES}_q${K}_dag${K}.log"
SCHED=$A/noise_schedule_pooled_lever_r${RES}_K${K}.json; [ -f "$SCHED" ] || { log "no K$K schedule"; exit 1; }
# ---- 5. calibrated rK from base
lever_train "$LEVER_CAL_DIR/q${K}_dnpool" lever_r${RES}_cal$K $K -- "${DART_COMMON[@]}" "${DART_SCHED[@]}" --dataset.dart_state_noise_schedule=$SCHED
# ---- 6. evals (eval seed 0) on two nodes
SIM1=$(start_lever_node 6045); SIM2=$(start_lever_node 6046); trap 'stop_node $SIM1; stop_node $SIM2' EXIT; wait_port 6045 && wait_port 6046 || exit 1
( eval_lever r${RES}_cal${K}${SUF} "$LEVER_CAL_DIR/q${K}_dnpool/checkpoints/$CKD/pretrained_model" 6045 0 ) & E1=$!
( eval_lever r${RES}_hg${K}${SUF} "$(HG_CK $K)" 6046 0 ) & E2=$!; wait $E1 $E2
log "LEVER_ROUND${K}_DONE"
