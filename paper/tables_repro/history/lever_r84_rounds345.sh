#!/bin/bash
S=/home/jennyw2/code/lerobot/my_scripts/paper_plots/tables_repro
until [ -f "$S/LEVER_R84_CAL2_DONE" ]; do sleep 60; done     # round-2 evals (incl. r84_hg2) finished
for K in 3 4 5; do bash $S/lever_r84_round.sh $K; [ -f "$S/LEVER_R84_ROUND${K}_DONE" ] || { echo "round $K did not finish; stopping"; exit 1; }; done
echo "ROUNDS 3-5 DONE $(date +%H:%M)"
