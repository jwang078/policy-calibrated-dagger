#!/bin/bash
# r84 round 2 (from-base protocol, Jenny 2026-09-15): the orchestrator records
# round-2 interventions with the round-1 HG policy (+40k ft_dag1, on the
# scenarios that failed its +40k inline eval) and computes the sidecar stats;
# its own lineage-style finetune is NOT wanted -> the orchestrator is killed as
# soon as the dag2 stats exist, then HG r2 is trained FROM BASE on
# base + dag1 + dag2 (+40k, weights 0.7 / 0.3 split by frames, constant lr).
set -u
S=/home/jennyw2/code/lerobot/my_scripts/paper_plots/tables_repro
LR=/home/jennyw2/code/lerobot; TR=$LR/outputs/training; RES=84; C=$HOME/.cache/huggingface/lerobot/JennyWWW
BASE=diffusion_approach_lever_13_smooth_r${RES}_delta_basewrist; BC=$TR/$BASE/checkpoints/last/pretrained_model
PFX=lever_d100_03dagcap_r${RES}_diff; DAG1=${PFX}_r_dag1; DAG2=${PFX}_r_dag2
log() { echo "[r84r2 $(date +%H:%M:%S)] $*"; }
until [ -f "$S/LEVER_R84_HG1_DONE" ]; do sleep 60; done
cd "$LR"
if [ ! -f "$LR/outputs/dataset_stats/$DAG2/stats_rel64.json" ]; then
  INT_X=$(grep -o "intervention_extra_args=.*" $S/lever_orchestrator_argv.txt | sed 's/^intervention_extra_args=//')
  ARGS=(); while IFS= read -r a || [ -n "$a" ]; do [ -z "$a" ] && continue; case "$a" in --resume|--num_rounds=*|--run_tag=*|--base_short=*|--finetune_extra_args=*|--intervention_extra_args=*|--initial_policy_path=*|--exclude_gripper_from_state|--exclude_gripper_from_state=*|--finetune_steps=*|--finetune_save_freq=*|--finetune_eval_freq=*) ;; *) ARGS+=("$a") ;; esac; done < $S/lever_orchestrator_argv.txt
  log "orchestrator round 2 (record + stats only; will be killed before its finetune)"
  setsid bash my_scripts/dagger_orchestrate.sh "${ARGS[@]}" --resume \
    --base_short=approach_lever_13_smooth_r${RES} --num_rounds=2 --run_tag=d100_03dagcap_r${RES} \
    --initial_policy_path="$TR/$BASE" --exclude_gripper_from_state=false --finetune_steps=40000 --finetune_save_freq=20000 --finetune_eval_freq=20000 \
    --intervention_extra_args="$INT_X" \
    --finetune_extra_args="--optimizer.lr=1e-6 --dataset.multi_source_feature_intersection=true --scheduler.name=constant --num_workers=4 --wandb.enable=false" \
    > "$S/r${RES}_orchestrator_r2.log" 2>&1 &
  OPID=$!
  until [ -f "$LR/outputs/dataset_stats/$DAG2/stats_rel64.json" ] || ! kill -0 $OPID 2>/dev/null; do sleep 20; done
  sleep 15   # let the stats step finish writing
  kill -- -$OPID 2>/dev/null; sleep 5; kill -9 -- -$OPID 2>/dev/null
  log "orchestrator killed after dag2 stats ($([ -f $LR/outputs/dataset_stats/$DAG2/stats_rel64.json ] && echo present || echo MISSING))"
  rm -rf "$TR/${BASE}_d100_03dagcap_r${RES}_ft_dag2"     # the unwanted lineage finetune dir, if it started
fi
[ -f "$LR/outputs/dataset_stats/$DAG2/stats_rel64.json" ] || { log "ABORT: no dag2 stats"; exit 1; }
# the node records at 224px -> re-encode the intervention videos to 84px so they batch with the 84px base
$HOME/miniforge3/envs/splatsim/bin/python $S/analysis/shrink_video_dataset_r84.py $DAG2 $RES > $S/r${RES}_shrink_dag2.log 2>&1 || { log "ABORT: dag2 shrink failed"; exit 1; }
log "dag2 shrunk to ${RES}px"
touch "$S/LEVER_R84_DAG2_DONE"
# ── HG r2 FROM BASE ──
F1=$(python3 -c "import json;print(json.load(open('$C/$DAG1/meta/info.json'))['total_frames'])"); F2=$(python3 -c "import json;print(json.load(open('$C/$DAG2/meta/info.json'))['total_frames'])")
WTS=$(python3 -c "f=[$F1,$F2];s=sum(f);print('[0.7, '+', '.join(str(round(0.3*x/s,9)) for x in f)+']')"); echo "$WTS" > $S/LEVER_R84_WTS2
REPOS="[\"JennyWWW/splatsim_approach_lever_13_smooth_r${RES}\", \"JennyWWW/$DAG1\", \"JennyWWW/$DAG2\"]"
STATS="[\"$LR/outputs/dataset_stats/approach_lever_13_smooth_r${RES}/stats_rel64.json\", \"$LR/outputs/dataset_stats/$DAG1/stats_rel64.json\", \"$LR/outputs/dataset_stats/$DAG2/stats_rel64.json\"]"
DIR=$TR/lever_r${RES}_fb/hg2; mkdir -p $TR/lever_r${RES}_fb
if [ ! -d "$DIR/checkpoints/last" ]; then
  rm -rf "$DIR"; log "training HG r2 from base: weights $WTS"
  bash my_scripts/resume_training.sh "$BC" --optimizer.lr=1e-6 --scheduler.name=constant --num_workers=6 \
    --dataset.multi_source_feature_intersection=true --dataset.repo_id= --dataset.repo_ids="$REPOS" --dataset.sample_weights="$WTS" --dataset.stats_paths="$STATS" \
    --dataset.norm_mode=aggregated --dataset.stats_path= --dataset.use_weighted_sampling=true \
    --steps=115000 --eval_freq=0 --env_eval_freq=0 --save_freq=20000 \
    --output_dir="$DIR" --job_name=lever_r${RES}_hg2 --policy.repo_id=lever_r${RES}_hg2 --policy.push_to_hub=false --wandb.enable=false > "$DIR.log" 2>&1
  log "hg2 rc=$?"
fi
touch "$S/LEVER_R84_HG2_DONE"; log "LEVER_R84_HG2_DONE"
