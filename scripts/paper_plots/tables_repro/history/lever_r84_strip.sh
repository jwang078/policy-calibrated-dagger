#!/bin/bash
# Gate: run the letterbox strip once the r84 BC trainer has exited (its 75k
# checkpoint exists and no lerobot-train is reading the dataset). Finishes in
# ~2 min, well before the orchestrator's finetune (after eval + recording).
S=/home/jennyw2/code/lerobot/my_scripts/paper_plots/tables_repro
T=/home/jennyw2/code/lerobot/outputs/training/diffusion_approach_lever_13_smooth_delta_basewristng_r84
until [ -d $T/checkpoints/075000 ] && ! pgrep -f "lerobot-train.*_r84" >/dev/null; do sleep 20; done
echo "[strip $(date +%H:%M:%S)] BC trainer gone; stripping letterbox columns"
cd /home/jennyw2/code/lerobot && nice -n 5 ~/miniforge3/envs/splatsim/bin/python $S/analysis/strip_letterbox_lever_r84.py
echo "[strip $(date +%H:%M:%S)] rc=$?"; touch $S/LEVER_R84_STRIP_DONE
