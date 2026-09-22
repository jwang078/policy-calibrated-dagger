#!/bin/bash
# Stills and clips of the Fig. 1 teaser episode for the paper video, all 640x359
# (the figure's own splat camera). Needs the frames from render_episode_video.py.
#   bash make_video_assets.sh [TAG]        default dag1_ep9_t4 -> video/<TAG>/
set -eu
H=$(cd "$(dirname "$0")" && pwd); TAG=${1:-dag1_ep9_t4}; PY=$HOME/miniforge3/envs/splatsim/bin/python; FF=$HOME/miniforge3/envs/splatsim/bin/ffmpeg
V=$H/video/$TAG; F=$V/frames; T0=$($PY -c "import json;print(json.load(open('$H/geom_$TAG.json'))['T0'])"); N=$(ls $F | wc -l); FPS=30
cd "$H"
# ---- stills (exact 640x359, no title) ----
cp $F/f0000.png                                   $V/1_start_still.png
cp $F/$(printf f%04d.png $T0)                     $V/3_anchor_still_clean.png
TITLE=0 BAND=1 ARCS=0 ANCHOR=0 OUT=$V/4_anchor_still_band.png      $PY plot.py $TAG > /dev/null
TITLE=0 BAND=0 PATH=1 ARCS=0 ANCHOR=0 OUT=$V/4b_anchor_still_path_only.png $PY plot.py $TAG > /dev/null
TITLE=0 BAND=1 ARCS=1 ANCHOR=1 OUT=$V/5_anchor_still_band_arcs.png $PY plot.py $TAG > /dev/null
# ---- clips ----
enc() { $FF -y -loglevel error -framerate $2 -start_number $3 -i $F/f%04d.png -frames:v $4 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p" -c:v libx264 -crf 16 "$1"; }
enc $V/2_start_to_anchor.mp4        $FPS 0 $((T0 + 1))                 # real time: the anchor is only $T0 frames in
enc $V/2_start_to_anchor_slow.mp4   6    0 $((T0 + 1))                 # same frames at 6 fps so it reads as motion
enc $V/6_anchor_to_end.mp4          $FPS $T0 $((N - T0))
enc $V/0_full_episode.mp4           $FPS 0 $N
# ---- context: the orchestrator's own rollout video of this scenario (policy + expert takeovers, 224 px, 20 fps) ----
SCEN=$($PY -c "import json;print(json.load(open('$H/geom_$TAG.json'))['scenario'])")
D=$HOME/code/lerobot/outputs/training/diffusion_approach_lever_13_smooth_r84_delta_basewrist_d100_03dagcap_r84_ft_dag1/dagger/interventions
ROW=$(awk -F, -v s=$SCEN 'NR>1 && $1==s {print NR-2}' $D/intervention_per_scenario.csv)
[ -f "$D/videos/splatsim_0/eval_episode_$ROW.mp4" ] && $FF -y -loglevel error -i "$D/videos/splatsim_0/eval_episode_$ROW.mp4" -vf "crop=224:224:0:0" -c:v libx264 -crf 16 -pix_fmt yuv420p "$V/7_full_rollout_scenario${SCEN}_224px.mp4"
ls -la $V | grep -v frames
