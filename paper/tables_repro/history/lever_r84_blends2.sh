#!/bin/bash
# r84 lineage: round-1 blends for W (6 ratios, two nodes 6042/6043), policy = the
# new 84px BC, on the round-1 intervention dataset once the orchestrator has
# finished recording it (its stats sidecar exists). Then LEVER_R84_BLENDS2_DONE.
set -u
S=/home/jennyw2/code/lerobot/my_scripts/paper_plots/tables_repro
LR=/home/jennyw2/code/lerobot; RES=84; C=$HOME/.cache/huggingface/lerobot/JennyWWW
CKPT=$LR/outputs/training/diffusion_approach_lever_13_smooth_r${RES}_delta_basewrist_d100_03dagcap_r${RES}_ft_dag1/checkpoints/last/pretrained_model   # HG r1 +40k
log() { echo "[r84blend2 $(date +%H:%M:%S)] $*"; }
until [ -f "$S/LEVER_R84_DAG2_DONE" ] && [ -f "$S/LEVER_R84_CAL1_DONE" ]; do sleep 60; done
INT=""; until [ -n "$INT" ]; do INT=$(ls -d $C/lever_d100_03dagcap_r${RES}*_r_dag2 2>/dev/null | head -1); [ -n "$INT" ] && [ -f "$LR/outputs/dataset_stats/$(basename $INT)/stats_rel64.json" ] || { INT=""; sleep 60; }; done
SHORT=$(basename $INT); log "intervention dataset: $SHORT"; echo "$SHORT" > $S/LEVER_R84_INT2_SHORT
EPS=$(python3 -c "
import json
n=json.load(open('$INT/meta/info.json'))['total_episodes']; print('['+','.join(str(i) for i in range(0,n,2))+']')")
lane() {  # PORT ratios...
  local PORT=$1; shift
  cd /home/jennyw2/code/SplatSim
  python -u scripts/launch_nodes.py --robot sim_ur_pybullet_small_engine_new_interactive --robot_port $PORT --hostname 127.0.0.1 \
    --eval_benchmark_repo_id JennyWWW/eval_splatsim_approach_lever_13_benchmark --headless --control_gui --render_mode splat \
    --strict_goal_tolerances > "$S/r${RES}_blend2_sim_$PORT.log" 2>&1 &
  local SIM=$!
  for i in $(seq 1 150); do (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null && break; sleep 2; done; sleep 5
  cd "$LR"
  for R in "$@"; do
    local TAG=$(python3 -c "print(f'{int(round(float($R)*100)):03d}')"); local TGT=JennyWWW/${SHORT}_blend$TAG
    [ -d "$C/${SHORT}_blend$TAG/data" ] && { log "$TGT exists, skip"; continue; }
    log "collecting $TGT on $PORT"
    nice -n 10 python my_scripts/augment_dataset_with_blending.py \
      --dataset_repo_id=JennyWWW/$SHORT --target_dataset_repo_id="$TGT" --policy_path="$CKPT" \
      "--forward_flow_ratios=[$R]" "--episode_indices=$EPS" --samples_per_episode=1 --relabel_actions=guidance \
      --env_external_port=$PORT --env_external_host=127.0.0.1 \
      --env_task=upright_small_engine_new --env_robot_name=robot_iphone_w_engine_curtain --num_dofs=6 \
      --eval_benchmark_repo_id=JennyWWW/eval_splatsim_approach_lever_13_benchmark \
      --blend_strategy=denoise --guidance_repr=absolute_pos --blend_mode=every_step --blend_interval_frac=0.5 \
      --fixed_base_noise=true --resample_noise_per_reblend=false --clip_sample=false --sample_seed=42 --show_guidance_ghost=false \
      --progress_guidance=true --guidance_from_dart_labels=true --anchor_suffix_steps=0 --anchor_prefix_steps=0 \
      --anchor_every_denoise_step=false --blend_ratio_goal_taper=0 --max_blend_end_gap_steps=100000 --max_blend_end_lag_indices=100000 \
      --pad_after_success=false --rtc_prev_chunk=true --rtc_max_guidance_weight=3 --rtc_hard_prefix_xfade=8 \
      --blend_ratio_backoff=false --blend_tube_steps=0 --max_tube_breach_ratio=0 > "$S/r${RES}_blend2_$TAG.log" 2>&1
    log "$TGT rc=$?"
  done
  kill $SIM 2>/dev/null
}
lane 6042 0.20 0.10 0.30 & P1=$!
lane 6043 0.50 0.75 0.90 & P2=$!
wait $P1 $P2
touch "$S/LEVER_R84_BLENDS2_DONE"; log "LEVER_R84_BLENDS2_DONE"
