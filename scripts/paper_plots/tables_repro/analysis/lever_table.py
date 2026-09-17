"""Lever 84px results table: per (round, arm) mean +- SE over TRAINING seeds, each seed's success averaged over its eval seeds.
BC (one checkpoint) reports mean +- SE over eval seeds. Reads outputs/eval300/lever_cam/*/eval_info.json.
"""

import glob
import json
import os
import re

import numpy as np

E = "/home/jennyw2/code/lerobot/outputs/eval300/lever_cam"


def succ(d):
    try:
        return float(json.load(open(f"{d}/eval_info.json"))["overall"]["pc_success"])
    except Exception:
        return None


# ---- collect: name -> (round, arm, train_seed, eval_seed, succ)
cells = {}  # (K, arm) -> {train_seed: [succ per eval seed]}
inline_hg1 = "/home/jennyw2/code/lerobot/outputs/training/diffusion_approach_lever_13_smooth_r84_delta_basewrist_d100_03dagcap_r84_ft_dag1/eval/eval_info_step_095000.json"
for d in sorted(glob.glob(f"{E}/r84_*_20k*")):
    m = re.match(r"r84_(hg|cal)(\d)_20k(?:_s(\d))?(?:_e(\d))?$", os.path.basename(d))
    if not m:
        continue
    arm, K, ts, es = m.group(1), int(m.group(2)), int(m.group(3) or 0), m.group(4)
    s = succ(d)
    if s is None:
        continue
    cells.setdefault((K, arm), {}).setdefault(ts, []).append(s)
# HG r1 seed 0 +20k came from the orchestrator's inline eval at step 95000 (same 100 scenarios)
if os.path.exists(inline_hg1) and not os.path.exists(
    f"{E}/r84_hg1_20k/eval_info.json"
):  # no standalone eval-seed-0 run for this ckpt
    j = json.load(open(inline_hg1))
    s = (
        float(j["overall"]["pc_success"])
        if "overall" in j
        else float(np.mean(j["per_task"][0]["metrics"]["successes"]) * 100)
    )
    cells.setdefault((1, "hg"), {}).setdefault(0, []).insert(0, s)
bc = [succ(d) for d in [f"{E}/bc_r84"] + sorted(glob.glob(f"{E}/bc_r84_e*"))]
bc = [b for b in bc if b is not None]


def fmt(vals):
    vals = np.array(vals, float)
    m = vals.mean()
    if len(vals) < 2:
        return f"{m:.1f}", m
    return f"{m:.1f} $\\pm$ {vals.std(ddof=1) / np.sqrt(len(vals)):.1f}", m


rounds = sorted({K for K, _ in cells})
print("round | arm | per-seed (mean over eval seeds; n evals) | cell")
rows = {"hg": [], "cal": []}
for K in rounds:
    for arm in ["hg", "cal"]:
        seeds = cells.get((K, arm), {})
        per_seed = [np.mean(v) for _, v in sorted(seeds.items())]
        desc = ", ".join(f"s{ts}:{np.mean(v):.0f}(n{len(v)})" for ts, v in sorted(seeds.items()))
        txt, m = fmt(per_seed) if per_seed else ("--", np.nan)
        rows[arm].append((txt, m, len(per_seed), sum(len(v) for v in seeds.values())))
        print(f"K={K} {arm:3s} | {desc} | {txt}")
bctxt, _ = fmt(bc)
print(f"BC | {bc} | {bctxt}")
# ---- LaTeX
L = []
L.append(r"\begin{tabular}{l|" + "c|" * (len(rounds) - 1) + "c}")
L.append(r"\hline")
L.append("Method & " + " & ".join(f"$K{{=}}{K}$" for K in rounds) + r" \\")
L.append(r"\hline")
L.append(
    f"BC (no aggregation) & \\multicolumn{{{len(rounds)}}}{{c}}{{{bctxt}\\textsuperscript{{(1s,{len(bc) * 100})}}}} \\\\"
)
L.append(r"\hline")


def row(name, arm):
    cellstr = []
    for txt, m, ns, ne in rows[arm]:
        sup = f"\\textsuperscript{{({ns}s,{ne * 100})}}" if ns else ""
        cellstr.append(f"{txt}{sup}")
    return name + " & " + " & ".join(cellstr) + r" \\"


L.append(row(r"HG-DAgger~\cite{kelly2019hg}", "hg"))
L.append(r"\noalign{\hrule height 0.9pt}")
L.append(row(r"Pooled $\bar{\Sigma}^{\alpha}$ (Ours)", "cal"))
L.append(r"\hline")
L.append(r"\end{tabular}")
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "table_lever.tex")
open(out, "w").write("\n".join(L) + "\n")
print("\nwrote", out)
print("\n".join(L))
