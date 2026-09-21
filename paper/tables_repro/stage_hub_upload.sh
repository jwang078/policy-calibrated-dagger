#!/bin/bash
# Lay out everything the paper's Hub upload needs under $HUB_STAGE (default ~/paper_archive/hub_upload),
# as hard links of the archive built by archive_tables.sh — nothing is uploaded, no extra disk is used.
# Review the tree, then upload each top-level folder with `hf upload` (commands printed at the end).
#
#   bash stage_hub_upload.sh              # build the tree and print sizes
#   HUB_STAGE=/elsewhere bash stage_hub_upload.sh
#
# Layout (see HUB_UPLOAD.md for what each is and why):
#   datasets/<repo_id name>/           LeRobot datasets, one Hub dataset repo each, same names as today
#   models/pcdagger-planar-table1/     <group>/<run>/<step>/  (pretrained_model only: no optimizer state, wandb, videos)
#   models/pcdagger-lever-table2/
#   results/                           eval300/, dataset_stats/, tables_repro analysis inputs (npz + schedules)
set -euo pipefail
ARCHIVE=${ARCHIVE:-$HOME/paper_archive/pcdagger_icra2027}
HUB_STAGE=${HUB_STAGE:-$HOME/paper_archive/hub_upload}
[ -d "$ARCHIVE/training" ] || { echo "no archive at $ARCHIVE (run archive_tables.sh first)"; exit 1; }
rm -rf "$HUB_STAGE"; mkdir -p "$HUB_STAGE"/{datasets,models,results}

link_tree() {  # link_tree SRC DST [find-prune-args...]: hard-link every file of SRC under DST
    local src=$1 dst=$2; shift 2
    (cd "$src" && find . -type f "$@" -print0) | while IFS= read -r -d '' f; do
        mkdir -p "$dst/$(dirname "$f")"; ln "$src/$f" "$dst/$f"
    done
}

# 1. datasets — every dataset the archive verified the tables against, minus the ones already on the Hub
ON_HUB="eval_planar_3joint_benchmark eval_splatsim_approach_lever_13_benchmark planar_3joint_12"
for d in "$ARCHIVE"/datasets/*/; do
    n=$(basename "$d"); case " $ON_HUB " in *" $n "*) continue;; esac
    link_tree "$d" "$HUB_STAGE/datasets/$n" -not -path "*/_pre_es7_backup/*"
done

# 2. models — pretrained_model of every checkpoint step the archive kept, grouped by table
planar="bc175k scarcity_study scarcity_study_cl scarcity_study_s2 scarcity_study_s3 scarcity_study_s4 scarcity_study_s5"
for g in "$ARCHIVE"/training/*/; do
    group=$(basename "$g")
    case "$group" in
        diffusion_planar_*|scarcity_study*|bc175k) repo=pcdagger-planar-table1;;
        *) repo=pcdagger-lever-table2;;
    esac
    for run in "$g"*/; do
        r=$(basename "$run"); [ -d "$run/checkpoints" ] || continue
        for step in "$run"/checkpoints/*/; do
            s=$(basename "$step"); [ "$s" = last ] && continue
            [ -d "$step/pretrained_model" ] || continue
            link_tree "$step/pretrained_model" "$HUB_STAGE/models/$repo/$group/$r/$s"
        done
        [ -f "$run/train_config.json" ] && ln "$run/train_config.json" "$HUB_STAGE/models/$repo/$group/$r/train_config.json"
        [ -d "$run/eval" ] && link_tree "$run/eval" "$HUB_STAGE/models/$repo/$group/$r/eval" -name "*.json"
    done
done

# 3. results — the 300-episode evals (json + per-episode telemetry, no videos), the sidecar stats, and the
#    calibration inputs (per-anchor deltas .npz + the committed schedules) so the analysis scripts run offline
link_tree "$ARCHIVE/eval300" "$HUB_STAGE/results/eval300" -not -name "*.mp4" -not -name "*.log"
link_tree "$ARCHIVE/dataset_stats" "$HUB_STAGE/results/dataset_stats"
link_tree "$ARCHIVE/tables_repro/analysis" "$HUB_STAGE/results/tables_repro_analysis" \( -name "*.npz" -o -name "*.tar.gz" -o -name "*.json" \)

echo "== staged under $HUB_STAGE (apparent sizes; hard links, no extra disk):"
du -sh "$HUB_STAGE"/datasets "$HUB_STAGE"/models/* "$HUB_STAGE"/results/* | sed 's/^/  /'
echo "  datasets: $(ls "$HUB_STAGE/datasets" | wc -l) repos  | model checkpoints: $(find "$HUB_STAGE/models" -name model.safetensors | wc -l)"
cat <<CMDS

== upload commands (after review; each is idempotent, re-run to resume):
  for d in $HUB_STAGE/datasets/*/; do hf upload --repo-type dataset "JennyWWW/\$(basename \$d)" "\$d" . ; done
  hf upload --repo-type model JennyWWW/pcdagger-planar-table1 $HUB_STAGE/models/pcdagger-planar-table1 .
  hf upload --repo-type model JennyWWW/pcdagger-lever-table2  $HUB_STAGE/models/pcdagger-lever-table2 .
  hf upload --repo-type dataset JennyWWW/pcdagger-icra2027-results $HUB_STAGE/results .
CMDS
