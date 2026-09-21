#!/bin/bash
# Collect everything Table I and Table II depend on into one folder, as HARD LINKS
# (same filesystem, zero extra disk; the archive keeps the data alive if the
# originals are deleted later). Then verify every table cell against it.
#
#   bash archive_tables.sh [DEST]        default ~/paper_archive/pcdagger_icra2027
#
# Layout of DEST:
#   datasets/<repo>        ~/.cache/huggingface/lerobot/JennyWWW/<repo>
#   training/<dir>         lerobot/outputs/training/<dir>   (table arms only for the scarcity_study* groups)
#   eval300/, dataset_stats/                                 lerobot/outputs/{eval300,dataset_stats}
#   tables_repro/          this kit (git archive of HEAD) with the schedules unpacked and the delta caches
#   MANIFEST.txt           what was linked, sizes, and the per-cell verification
set -u
S=$(cd "$(dirname "$0")" && pwd); LR=/home/jennyw2/code/lerobot; HF=$HOME/.cache/huggingface/lerobot/JennyWWW
DEST=${1:-$HOME/paper_archive/pcdagger_icra2027}; PY=$HOME/miniforge3/envs/splatsim/bin/python
log() { echo "[archive $(date +%H:%M:%S)] $*"; }
mkdir -p "$DEST"/{datasets,training,eval300}; M=$DEST/MANIFEST.txt; : > "$M"
MISSING=0
link() {  # SRC DEST_PARENT  (cp -al; skipped if already there)
  local src=$1 dst=$2/$(basename "$1")
  [ -e "$src" ] || { echo "MISSING  $src" | tee -a "$M"; MISSING=$((MISSING+1)); return 1; }
  [ -e "$dst" ] || cp -al "$src" "$dst"; return 0
}

# ---- datasets ----------------------------------------------------------------
DS=(planar_3joint_12 eval_planar_3joint_benchmark)
for tag in 05dag 06dag 07dag 08dag 09dag 10cl; do for K in 1 2 3 4 5; do DS+=(planar_12_${tag}_diff_r_dag$K); done; done
for K in 1 2 3 4 5; do for r in 005 010 015 020 025 030 035 040 045 050 065 075 090; do DS+=(planar_12_05dag_diff_r_dag${K}_blend$r); done; done   # the W dial (13 ratios, lineage s1)
DS+=(splatsim_approach_lever_13_smooth_r84 eval_splatsim_approach_lever_13_benchmark)
for K in 1 2 3 4 5; do DS+=(lever_d100_03dagcap_r84_diff_r_dag$K); for r in 010 020 030 050 075 090; do [ -d "$HF/lever_d100_03dagcap_r84_diff_r_dag${K}_blend$r" ] && DS+=(lever_d100_03dagcap_r84_diff_r_dag${K}_blend$r); done; done
log "linking ${#DS[@]} datasets"; for d in "${DS[@]}"; do link "$HF/$d" "$DEST/datasets"; done

# ---- training dirs -----------------------------------------------------------
TR=$LR/outputs/training
link "$TR/diffusion_planar_3joint_12_delta_stateng" "$DEST/training"; link "$TR/bc175k" "$DEST/training"
for tag in 05dag 06dag 07dag 08dag 09dag 10cl; do for K in 1 2 3 4 5; do link "$TR/diffusion_planar_3joint_12_delta_stateng_${tag}_ft_dag$K" "$DEST/training"; done; done
for G in scarcity_study scarcity_study_s2 scarcity_study_s3 scarcity_study_s4 scarcity_study_s5; do
  mkdir -p "$DEST/training/$G"
  for K in 1 2 3 4 5; do for A in q$K q${K}_dn2swv q${K}_dn4swv q${K}_dn8swv q${K}_dn12swv q${K}_dn16swv q${K}_dnpool q${K}_dnsig; do
    link "$TR/$G/$A" "$DEST/training/$G"; [ -f "$TR/$G/$A.log" ] && link "$TR/$G/$A.log" "$DEST/training/$G"; done; done
done
mkdir -p "$DEST/training/scarcity_study_cl"; for K in 1 2 3 4 5; do link "$TR/scarcity_study_cl/q${K}_dnpoolcl" "$DEST/training/scarcity_study_cl"; done
for d in diffusion_approach_lever_13_smooth_r84_delta_basewrist diffusion_approach_lever_13_smooth_r84_delta_basewrist_seed1 diffusion_approach_lever_13_smooth_r84_delta_basewrist_seed2 \
         lever_r84_fb lever_r84_calib; do link "$TR/$d" "$DEST/training"; done
for K in 1 2 3 4 5; do link "$TR/diffusion_approach_lever_13_smooth_r84_delta_basewrist_d100_03dagcap_r84_ft_dag$K" "$DEST/training"; done
log "linking evals + stats"; for g in $(ls $LR/outputs/eval300); do link "$LR/outputs/eval300/$g" "$DEST/eval300"; done
link "$LR/outputs/dataset_stats" "$DEST"

# ---- this kit ----------------------------------------------------------------
rm -rf "$DEST/tables_repro"; mkdir -p "$DEST/tables_repro"
(cd "$LR" && git archive HEAD my_scripts/paper_plots/tables_repro | tar -x -C "$DEST/tables_repro" --strip-components=3)
(cd "$DEST/tables_repro" && bash analysis/unpack_schedules.sh > /dev/null)
for f in "$S"/analysis/*.npz; do link "$f" "$DEST/tables_repro/analysis" > /dev/null; done
echo "kit: lerobot $(cd $LR && git rev-parse --short HEAD), SplatSim $(cd $HOME/code/SplatSim && git rev-parse --short HEAD)" >> "$M"

# ---- verify every table cell -------------------------------------------------
log "verifying table cells against the archive"
$PY - "$DEST" <<'EOF' | tee -a "$M"
import glob, json, os, sys
D = sys.argv[1]; bad = 0
def ok(p, what):
    global bad
    if not os.path.exists(p): bad += 1; print(f"  MISSING {what}: {p}")
# Table I: 5 lineages x 5 rounds x 8 arms, checkpoint 175000 + 300-episode eval; closed-loop + BC rows
for G in ["scarcity_study", "scarcity_study_s2", "scarcity_study_s3", "scarcity_study_s4", "scarcity_study_s5"]:
    for K in range(1, 6):
        for A in [f"q{K}", *[f"q{K}_dn{s}swv" for s in (2, 4, 8, 12, 16)], f"q{K}_dnpool", f"q{K}_dnsig"]:
            ok(f"{D}/training/{G}/{A}/checkpoints/175000/pretrained_model/model.safetensors", f"ckpt {G}/{A}")
            ev = f"{D}/eval300/{G}/{A}/eval_info.json"; ok(ev, f"eval {G}/{A}")
            if os.path.exists(ev):
                n = len(json.load(open(ev))["per_task"][0]["metrics"]["successes"])
                if n != 300: print(f"  NOTE {G}/{A}: {n} episodes (expected 300)")
for K in range(1, 6):
    ok(f"{D}/training/scarcity_study_cl/q{K}_dnpoolcl/checkpoints/175000/pretrained_model/model.safetensors", f"ckpt cl q{K}")
    for sfx in ["", "_e1", "_e2"]: ok(f"{D}/eval300/scarcity_study_cl/q{K}_dnpoolcl{sfx}/eval_info.json", f"eval cl q{K}{sfx}")
for sfx in ["", "_e1", "_e2"]: ok(f"{D}/eval300/base/bc175k{sfx}/eval_info.json", f"eval BC{sfx}")
# Table II: rounds 1-2 x 3 training seeds x 2 arms, each with 3 eval seeds; BC with 3 eval seeds
def lever_ckpt(K, ts, arm):
    s = "" if ts == 0 else f"_s{ts}"
    if arm == "cal": return f"{D}/training/lever_r84_calib/q{K}_dnpool{s}/checkpoints/095000/pretrained_model/model.safetensors"
    if ts == 0 and K == 1: return f"{D}/training/diffusion_approach_lever_13_smooth_r84_delta_basewrist_d100_03dagcap_r84_ft_dag1/checkpoints/095000/pretrained_model/model.safetensors"
    return f"{D}/training/lever_r84_fb/hg{K}{s}/checkpoints/095000/pretrained_model/model.safetensors"
for K in (1, 2):
    for ts in (0, 1, 2):
        for arm in ("hg", "cal"):
            ok(lever_ckpt(K, ts, arm), f"ckpt lever {arm}{K} seed{ts}")
            s = "" if ts == 0 else f"_s{ts}"
            for es in (0, 1, 2):
                if es == ts and ts == 0 and arm == "hg" and K == 1:
                    ok(f"{D}/training/diffusion_approach_lever_13_smooth_r84_delta_basewrist_d100_03dagcap_r84_ft_dag1/eval/eval_info_step_095000.json", "inline eval hg1 seed0"); continue
                name = f"r84_{arm}{K}_20k{s}" + ("" if es == ts else f"_e{es}")
                ok(f"{D}/eval300/lever_cam/{name}/eval_info.json", f"eval lever {name}")
for sfx in ["", "_e1", "_e2"]: ok(f"{D}/eval300/lever_cam/bc_r84{sfx}/eval_info.json", f"eval lever BC{sfx}")
print(f"cell verification: {'OK, every checkpoint and eval present' if bad == 0 else str(bad) + ' missing'}")
EOF
# the table generators must reproduce the paper tables from the archived evals alone
(cd "$DEST/tables_repro" && OUTPUTS_ROOT=$DEST $PY gen_table.py 2>&1 | grep -i "warning" ; \
  diff -q <(sed -n '/begin{tabular}/,/end{tabular}/p' table_full_benchmark.tex) <(sed -n '/begin{tabular}/,/end{tabular}/p' table1_tabular_all300.tex) > /dev/null \
  && echo "Table I regenerated from the archive: identical to table1_tabular_all300.tex" || echo "Table I regenerated from the archive: DIFFERS from table1_tabular_all300.tex") | tee -a "$M"
(cd "$DEST/tables_repro" && OUTPUTS_ROOT=$DEST $PY analysis/lever_table.py | tail -8) >> "$M" 2>&1

# ---- sizes -------------------------------------------------------------------
{ echo; echo "sizes (apparent; the files are hard links of the originals, so no extra disk is used until the originals go):"
  du -sh "$DEST"/datasets "$DEST"/training "$DEST"/eval300 "$DEST"/dataset_stats "$DEST"/tables_repro "$DEST"; } | tee -a "$M"
log "done: $DEST ($MISSING missing sources)"
