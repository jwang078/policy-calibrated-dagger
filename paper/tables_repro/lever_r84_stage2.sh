#!/bin/bash
# r84 lineage stage 2: after the BC eval, hand the new BC to the DAgger
# orchestrator for ONE round (record interventions on the scenarios the BC
# failed, sidecar stats, HG-DAgger finetune +40k FROM BASE at constant lr 1e-6 with its
# inline 100-ep evals at +20k and +40k; +40k = primary). The orchestrator manages its own splat node
# on port 6001. Recording/eval envs render at 84x84 (--env.observation_*).
set -u
S=/home/jennyw2/code/lerobot/my_scripts/paper_plots/tables_repro
LR=/home/jennyw2/code/lerobot; RES=84
RUN_TRAINED=diffusion_approach_lever_13_smooth_delta_basewristng_r${RES}          # what lever_r84_chain.sh wrote
RUN_DERIVED=diffusion_approach_lever_13_smooth_r${RES}_delta_basewrist          # what the orchestrator expects for --base_short=…_r84
log() { echo "[r84s2 $(date +%H:%M:%S)] $*"; }
until [ -f "$S/LEVER_R84_BC_DONE" ]; do sleep 30; done
cd "$LR"
if [ -d outputs/training/$RUN_TRAINED ] && [ ! -d outputs/training/$RUN_DERIVED ]; then
  mv outputs/training/$RUN_TRAINED outputs/training/$RUN_DERIVED; log "renamed BC dir -> $RUN_DERIVED"
fi
BASE_DIR=$LR/outputs/training/$RUN_DERIVED
mkdir -p "$BASE_DIR/eval"
cp outputs/eval300/lever_cam/bc_r${RES}/eval_info.json "$BASE_DIR/eval/eval_info_step_075000.json"   # skip-succeeded needs it here
INT_X=$(grep -o "intervention_extra_args=.*" $S/lever_orchestrator_argv.txt | sed 's/^intervention_extra_args=//')
ARGS=()
while IFS= read -r a || [ -n "$a" ]; do
  [ -z "$a" ] && continue; case "$a" in --resume|--num_rounds=*|--run_tag=*|--base_short=*|--finetune_extra_args=*|--intervention_extra_args=*|--initial_policy_path=*|--exclude_gripper_from_state|--exclude_gripper_from_state=*|--finetune_steps=*|--finetune_save_freq=*|--finetune_eval_freq=*) ;; *) ARGS+=("$a") ;; esac
done < $S/lever_orchestrator_argv.txt
log "orchestrator round 1 from $BASE_DIR"
bash my_scripts/dagger_orchestrate.sh "${ARGS[@]}" \
  --base_short=approach_lever_13_smooth_r${RES} --num_rounds=1 --run_tag=d100_03dagcap_r${RES} \
  --initial_policy_path="$BASE_DIR" --resume --exclude_gripper_from_state=false --finetune_steps=40000 --finetune_save_freq=20000 --finetune_eval_freq=20000 \
  --intervention_extra_args="$INT_X" \
  --finetune_extra_args="--optimizer.lr=1e-6 --dataset.multi_source_feature_intersection=true --scheduler.name=constant --num_workers=4 --wandb.enable=false" \
  > "$S/r${RES}_orchestrator.log" 2>&1
log "orchestrator rc=$?"
ls -d outputs/training/${RUN_DERIVED}_d100_03dagcap_r${RES}_ft_dag1/checkpoints/last 2>/dev/null && touch "$S/LEVER_R84_HG1_DONE"
log "LEVER_R84_HG1_DONE=$([ -f $S/LEVER_R84_HG1_DONE ] && echo yes || echo NO)"
