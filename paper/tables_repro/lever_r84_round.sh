#!/bin/bash
# Generic 84px paired round K>=3 (from-base protocol). usage: lever_r84_round.sh K
#  0. shim: expose HG r(K-1) (lever_r84_fb/hg{K-1}) as the orchestrator's round-(K-1)
#     policy dir (…_ft_dag{K-1}: checkpoints/last symlink + eval_info from its
#     standalone eval + train_config with the eval subset) so resume/skip logic works.
#  1. orchestrator --resume --num_rounds=K: record dag{K} with HG r(K-1) on the
#     scenarios it failed, sidecar stats; killed before its finetune; videos -> 84px.
#  2. HG rK FROM BASE (+40k, 6 workers) while blends K (6 ratios, 2 nodes) run with HG r(K-1).
#  3. W(rounds 1..K), Sigma-hat with HG r(K-1) on dag1..K, pooled K schedule, cal rK FROM BASE (+40k, 8 workers).
#  4. 100-ep evals of hg{K} and cal{K} (+40k) on two nodes -> LEVER_R84_ROUND{K}_DONE.
set -u
K=$1
FT=${FT:-20000}; CKSTEP=$((75000+FT)); SUF=$([ "$FT" = 20000 ] && echo _20k || echo "")   # rounds>=3: +20k first (Jenny 15:00); +40k later by resuming
S=/home/jennyw2/code/lerobot/my_scripts/paper_plots/tables_repro
LR=/home/jennyw2/code/lerobot; TR=$LR/outputs/training; PY=/home/jennyw2/miniforge3/envs/splatsim/bin/python
EV=$HOME/miniforge3/envs/splatsim/bin/lerobot-eval; RES=84; C=$HOME/.cache/huggingface/lerobot/JennyWWW
BASE=diffusion_approach_lever_13_smooth_r${RES}_delta_basewrist; BC=$TR/$BASE/checkpoints/last/pretrained_model
PFX=lever_d100_03dagcap_r${RES}_diff; P=$((K-1))
HGP=$TR/lever_r${RES}_fb/hg${P}; HGP_CK=$HGP/checkpoints/$(printf %06d $CKSTEP)/pretrained_model   # HG r(K-1) at +FT, from base
FT_P=$TR/${BASE}_d100_03dagcap_r${RES}_ft_dag${P}                                       # orchestrator's view of round K-1
SW='--dataset.dart_mask_hold_tail=true --dataset.dart_selective_mask=true --policy.do_mask_loss_for_padding=true'
log() { echo "[r84 K=$K $(date +%H:%M:%S)] $*"; }
cd "$LR"
# ── 0. shim ──
[ -d "$HGP_CK" ] || { log "ABORT: no HG r$P checkpoint at $HGP_CK"; exit 1; }
EVP=$LR/outputs/eval300/lever_cam/r${RES}_hg${P}${SUF}/eval_info.json; [ -f "$EVP" ] || { log "ABORT: no standalone eval for hg$P"; exit 1; }
mkdir -p "$FT_P/checkpoints" "$FT_P/eval"
# the orchestrator's completeness check: basename(readlink last) == train_config.steps -> COPY the +FT checkpoint and set steps=CKSTEP in the copy
CKD=$(printf %06d $CKSTEP); rm -rf "$FT_P/checkpoints/$CKD" "$FT_P/checkpoints/last"
cp -r "$HGP/checkpoints/$CKD" "$FT_P/checkpoints/$CKD"; ln -s "$CKD" "$FT_P/checkpoints/last"
cp "$EVP" "$FT_P/eval/eval_info_step_$CKD.json"
$PY - <<PYEOF
import json, glob
for f in glob.glob("$FT_P/checkpoints/$CKD/pretrained_model/train_config.json") + glob.glob("$HGP/checkpoints/*/pretrained_model/train_config.json"):
    c=json.load(open(f)); c["env"]["eval_benchmark_subset"]=list(range(100))
    if f.startswith("$FT_P"): c["steps"]=$CKSTEP
    json.dump(c, open(f,"w"), indent=4)
PYEOF
# dry-run guard: the orchestrator must plan to start at round K step 1 (recording), else abort loudly
INT_X=$(grep -o "intervention_extra_args=.*" $S/lever_orchestrator_argv.txt | sed 's/^intervention_extra_args=//')
ARGS=(); while IFS= read -r a || [ -n "$a" ]; do [ -z "$a" ] && continue; case "$a" in --resume|--num_rounds=*|--run_tag=*|--base_short=*|--finetune_extra_args=*|--intervention_extra_args=*|--initial_policy_path=*|--exclude_gripper_from_state|--exclude_gripper_from_state=*|--finetune_steps=*|--finetune_save_freq=*|--finetune_eval_freq=*) ;; *) ARGS+=("$a") ;; esac; done < $S/lever_orchestrator_argv.txt
bash my_scripts/dagger_orchestrate.sh "${ARGS[@]}" --dry-run --resume --base_short=approach_lever_13_smooth_r${RES} --num_rounds=$K --run_tag=d100_03dagcap_r${RES} \
  --initial_policy_path="$TR/$BASE" --exclude_gripper_from_state=false --finetune_steps=$FT --finetune_save_freq=20000 --finetune_eval_freq=20000 \
  --intervention_extra_args="$INT_X" --finetune_extra_args="--optimizer.lr=1e-6 --dataset.multi_source_feature_intersection=true --scheduler.name=constant --num_workers=4 --wandb.enable=false" \
  > "$S/r${RES}_orchestrator_r${K}_dryrun.log" 2>&1
grep -E "Round [0-9]+:.*steps complete|Pipeline detected|skip_succeeded" "$S/r${RES}_orchestrator_r${K}_dryrun.log" | cut -c1-160
grep -q "next is round $K, step 1" "$S/r${RES}_orchestrator_r${K}_dryrun.log" || { log "ABORT: orchestrator would not start at round $K step 1 (see dry-run log)"; exit 1; }
# ── 1. record round K ──
DAGK=${PFX}_r_dag${K}
if [ ! -f "$LR/outputs/dataset_stats/$DAGK/stats_rel64.json" ]; then
  INT_X=$(grep -o "intervention_extra_args=.*" $S/lever_orchestrator_argv.txt | sed 's/^intervention_extra_args=//')
  ARGS=(); while IFS= read -r a || [ -n "$a" ]; do [ -z "$a" ] && continue; case "$a" in --resume|--num_rounds=*|--run_tag=*|--base_short=*|--finetune_extra_args=*|--intervention_extra_args=*|--initial_policy_path=*|--exclude_gripper_from_state|--exclude_gripper_from_state=*|--finetune_steps=*|--finetune_save_freq=*|--finetune_eval_freq=*) ;; *) ARGS+=("$a") ;; esac; done < $S/lever_orchestrator_argv.txt
  log "orchestrator round $K (record + stats; killed before its finetune)"
  setsid bash my_scripts/dagger_orchestrate.sh "${ARGS[@]}" --resume \
    --base_short=approach_lever_13_smooth_r${RES} --num_rounds=$K --run_tag=d100_03dagcap_r${RES} \
    --initial_policy_path="$TR/$BASE" --exclude_gripper_from_state=false --finetune_steps=$FT --finetune_save_freq=20000 --finetune_eval_freq=20000 \
    --intervention_extra_args="$INT_X" \
    --finetune_extra_args="--optimizer.lr=1e-6 --dataset.multi_source_feature_intersection=true --scheduler.name=constant --num_workers=4 --wandb.enable=false" \
    > "$S/r${RES}_orchestrator_r${K}.log" 2>&1 &
  OPID=$!
  until [ -f "$LR/outputs/dataset_stats/$DAGK/stats_rel64.json" ] || ! kill -0 $OPID 2>/dev/null; do sleep 20; done
  sleep 15; kill -- -$OPID 2>/dev/null; sleep 5; kill -9 -- -$OPID 2>/dev/null
  log "orchestrator killed after dag$K stats ($([ -f $LR/outputs/dataset_stats/$DAGK/stats_rel64.json ] && echo present || echo MISSING))"
  rm -rf "$TR/${BASE}_d100_03dagcap_r${RES}_ft_dag${K}/checkpoints"   # keep dagger/interventions videos + csv for visualize_intervention_episode.py
  [ -f "$LR/outputs/dataset_stats/$DAGK/stats_rel64.json" ] || { log "ABORT: no dag$K stats"; exit 1; }
  $PY $S/analysis/shrink_video_dataset_r84.py $DAGK $RES > $S/r${RES}_shrink_dag${K}.log 2>&1 || { log "ABORT: dag$K shrink failed"; exit 1; }
  log "dag$K shrunk to ${RES}px: $($PY -c "import json;i=json.load(open('$C/$DAGK/meta/info.json'));print(i['total_episodes'],'eps',i['total_frames'],'frames')")"
fi
# weights / repos for base + dag1..K
FR=$($PY -c "import json;print(' '.join(str(json.load(open('$C/${PFX}_r_dag%d/meta/info.json'%i))['total_frames']) for i in range(1,$K+1)))")
WTS=$($PY -c "f=[$(echo $FR | tr ' ' ',')];s=sum(f);print('[0.7, '+', '.join(str(round(0.3*x/s,9)) for x in f)+']')")
REPOS=$($PY -c "import json;print(json.dumps(['JennyWWW/splatsim_approach_lever_13_smooth_r${RES}']+['JennyWWW/${PFX}_r_dag%d'%i for i in range(1,$K+1)]))")
STATS=$($PY -c "import json;print(json.dumps(['$LR/outputs/dataset_stats/approach_lever_13_smooth_r${RES}/stats_rel64.json']+['$LR/outputs/dataset_stats/${PFX}_r_dag%d/stats_rel64.json'%i for i in range(1,$K+1)]))")
log "weights $WTS"
# ── 2a. blends K in background (HG r(K-1) policy on dag K) ──
blend_lane() {  # PORT ratios...
  local PORT=$1; shift; local EPS; EPS=$($PY -c "import json;n=json.load(open('$C/$DAGK/meta/info.json'))['total_episodes'];print('['+','.join(str(i) for i in range(0,n,2))+']')")
  cd /home/jennyw2/code/SplatSim
  python -u scripts/launch_nodes.py --robot sim_ur_pybullet_small_engine_new_interactive --robot_port $PORT --hostname 127.0.0.1 \
    --eval_benchmark_repo_id JennyWWW/eval_splatsim_approach_lever_13_benchmark --headless --control_gui --render_mode splat --strict_goal_tolerances > "$S/r${RES}_blend${K}_sim_$PORT.log" 2>&1 &
  local SIM=$!; for i in $(seq 1 150); do (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null && break; sleep 2; done; sleep 5
  cd "$LR"
  for R in "$@"; do local TAG; TAG=$($PY -c "print(f'{int(round(float($R)*100)):03d}')"); local TGT=JennyWWW/${DAGK}_blend$TAG
    [ -d "$C/${DAGK}_blend$TAG/data" ] && { log "$TGT exists"; continue; }; log "collecting $TGT on $PORT"
    nice -n 10 python my_scripts/augment_dataset_with_blending.py --dataset_repo_id=JennyWWW/$DAGK --target_dataset_repo_id="$TGT" --policy_path="$HGP_CK" \
      "--forward_flow_ratios=[$R]" "--episode_indices=$EPS" --samples_per_episode=1 --relabel_actions=guidance --env_external_port=$PORT --env_external_host=127.0.0.1 \
      --env_task=upright_small_engine_new --env_robot_name=robot_iphone_w_engine_curtain --num_dofs=6 --eval_benchmark_repo_id=JennyWWW/eval_splatsim_approach_lever_13_benchmark \
      --blend_strategy=denoise --guidance_repr=absolute_pos --blend_mode=every_step --blend_interval_frac=0.5 --fixed_base_noise=true --resample_noise_per_reblend=false --clip_sample=false --sample_seed=42 \
      --show_guidance_ghost=false --progress_guidance=true --guidance_from_dart_labels=true --anchor_suffix_steps=0 --anchor_prefix_steps=0 --anchor_every_denoise_step=false --blend_ratio_goal_taper=0 \
      --max_blend_end_gap_steps=100000 --max_blend_end_lag_indices=100000 --pad_after_success=false --rtc_prev_chunk=true --rtc_max_guidance_weight=3 --rtc_hard_prefix_xfade=8 \
      --blend_ratio_backoff=false --blend_tube_steps=0 --max_tube_breach_ratio=0 > "$S/r${RES}_blend${K}_$TAG.log" 2>&1; log "$TGT rc=$?"
  done; kill $SIM 2>/dev/null
}
( blend_lane 6042 0.20 0.10 0.30; ) & B1=$!
( blend_lane 6043 0.50 0.75 0.90; ) & B2=$!
# ── 2b. HG rK from base ──
DIRH=$TR/lever_r${RES}_fb/hg${K}
if [ ! -d "$DIRH/checkpoints/$(printf %06d $CKSTEP)" ]; then rm -rf "$DIRH"; log "training HG r$K from base"
  bash my_scripts/resume_training.sh "$BC" --optimizer.lr=1e-6 --scheduler.name=constant --num_workers=6 \
    --dataset.multi_source_feature_intersection=true --dataset.repo_id= --dataset.repo_ids="$REPOS" --dataset.sample_weights="$WTS" --dataset.stats_paths="$STATS" \
    --dataset.norm_mode=aggregated --dataset.stats_path= --dataset.use_weighted_sampling=true --steps=$CKSTEP --eval_freq=0 --env_eval_freq=0 --save_freq=20000 \
    --output_dir="$DIRH" --job_name=lever_r${RES}_hg${K} --policy.repo_id=lever_r${RES}_hg${K} --policy.push_to_hub=false --wandb.enable=false > "$DIRH.log" 2>&1; log "hg$K rc=$?"
fi
wait $B1 $B2; log "blends $K done"
# ── 3. W, Sigma-hat, schedule, cal rK ──
ROUNDS=$(seq -s, 1 $K)
INT_PREFIX=$PFX $PY $S/analysis/measure_w_lever.py $ROUNDS > "$S/analysis/w_lever_r${RES}_rounds${ROUNDS}.txt" 2>&1
W=$(grep -oE "W = [0-9.]+" "$S/analysis/w_lever_r${RES}_rounds${ROUNDS}.txt" | head -1 | awk '{print $3}'); log "W(rounds $ROUNDS) = '$W'"; grep -E "W =|CI" "$S/analysis/w_lever_r${RES}_rounds${ROUNDS}.txt"
[ "$($PY -c "w=float('${W:-nan}'); print('yes' if 3<=w<=40 else 'no')")" = yes ] || { log "ABORT: W out of range"; exit 1; }
for R in $(seq 1 $K); do $PY $S/analysis/measure_sigma_lever.py r${RES}_q${K}_dag$R "$HGP_CK" "${PFX}_r_dag$R" > $S/analysis/sigma_r${RES}_q${K}_dag$R.log 2>&1; log "sigma dag$R rc=$?"; done
INT_PREFIX=$PFX TAGPFX=r${RES}_ $PY $S/analysis/build_pooled_schedule_lever.py $K $W | tee -a $S/analysis/sigma_r${RES}_q${K}_dag${K}.log
SCHED=$S/analysis/noise_schedule_pooled_lever_r${RES}_K${K}.json; [ -f "$SCHED" ] || { log "ABORT: no K$K schedule"; exit 1; }
DIRC=$TR/lever_r${RES}_calib/q${K}_dnpool
if [ ! -d "$DIRC/checkpoints/$(printf %06d $CKSTEP)" ]; then rm -rf "$DIRC"; log "training calibrated r$K from base"
  bash my_scripts/resume_training.sh "$BC" --optimizer.lr=1e-6 --scheduler.name=constant --num_workers=8 \
    --dataset.multi_source_feature_intersection=true --dataset.repo_id= --dataset.repo_ids="$REPOS" --dataset.sample_weights="$WTS" --dataset.stats_paths="$STATS" \
    --dataset.norm_mode=aggregated --dataset.stats_path= --dataset.use_weighted_sampling=true \
    --dataset.dart_relabel=true '--dataset.dart_self_relabel_pattern=_r_dag\d+$' --dataset.dart_state_noise_std=0 --dataset.dart_state_noise_schedule=$SCHED --dataset.dart_state_noise_scale=1.0 --dataset.dart_vel_noise_std=0.3 $SW \
    --steps=$CKSTEP --eval_freq=0 --env_eval_freq=0 --save_freq=20000 \
    --output_dir="$DIRC" --job_name=lever_r${RES}_cal${K} --policy.repo_id=lever_r${RES}_cal${K} --policy.push_to_hub=false --wandb.enable=false > "$DIRC.log" 2>&1; log "cal$K rc=$?"
fi
# ── 4. evals on two nodes ──
start_node() { cd /home/jennyw2/code/SplatSim; python -u scripts/launch_nodes.py --robot sim_ur_pybullet_small_engine_new_interactive --robot_port $1 --hostname 127.0.0.1 \
    --eval_benchmark_repo_id JennyWWW/eval_splatsim_approach_lever_13_benchmark --headless --control_gui --render_mode splat > "$S/r${RES}_eval${K}_sim_$1.log" 2>&1 & echo $!; }
SIM1=$(start_node 6045); SIM2=$(start_node 6046); trap 'kill ${SIM1:-} ${SIM2:-} 2>/dev/null' EXIT
for PT in 6045 6046; do for i in $(seq 1 150); do (exec 3<>/dev/tcp/127.0.0.1/$PT) 2>/dev/null && break; sleep 2; done; done; sleep 5
cd "$LR"; SUBSET100=$($PY -c "print('['+','.join(str(i) for i in range(100))+']')")
eval_one() { local NAME=$1 CK=$2 PORT=$3; local D="$LR/outputs/eval300/lever_cam/$NAME"
  [ -f "$D/eval_info.json" ] && { log "$NAME done"; return; }; [ -d "$CK" ] || { log "$NAME: no ckpt $CK"; return; }; mkdir -p "$D"; log "eval100 $NAME on $PORT"
  $EV --env.type=splatsim --env.task=upright_small_engine_new --env.camera_names='["base_rgb","wrist_rgb"]' --env.image_resize_modes='["stretch"]' \
    --env.fps=30 --env.eval_benchmark_repo_id=JennyWWW/eval_splatsim_approach_lever_13_benchmark --policy.path="$CK" --eval.n_episodes=100 --output_dir="$D" \
    --eval.batch_size=1 --eval.use_async_envs=false --seed=0 --rename_map='{}' --env.episode_length=1000 --env.eval_benchmark_subset="$SUBSET100" \
    --env.external_port=$PORT --env.max_parallel_tasks=1 --env.robot_name=robot_iphone_w_engine_curtain --env.cam_i=3 --env.use_gripper=true \
    --env.debug_mode=off --env.headless=true --env.control_gui=true --env.include_oracle_info=false --env.terminate_on_collision=true --env.wrist_cam_ver=2 \
    --env.teleop_pad_short_episodes=true --env.teleop_state_jump_split_threshold_rad=0.15 --env.num_dofs=6 --env.state_dim=7 --env.action_dim=7 --env.env_state_dim=0 > "$D/log.txt" 2>&1
  log "$NAME rc=$? succ=$($PY -c "
import json
try: print(json.load(open('$D/eval_info.json'))['overall']['pc_success'])
except Exception: print('n/a')")"; }
( eval_one r${RES}_cal${K}${SUF} "$DIRC/checkpoints/$(printf %06d $CKSTEP)/pretrained_model" 6045 ) & PA=$!
( eval_one r${RES}_hg${K}${SUF} "$DIRH/checkpoints/$(printf %06d $CKSTEP)/pretrained_model" 6046 ) & PB=$!
wait $PA $PB
touch "$S/LEVER_R84_ROUND${K}_DONE"; log "LEVER_R84_ROUND${K}_DONE"
