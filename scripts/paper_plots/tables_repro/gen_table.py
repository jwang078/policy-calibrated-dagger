"""Generate the full-benchmark paper table (table_full_benchmark.tex) with
per-cell background colors.

Color coding (Jenny's spec, 2026-09-13):
  - per COLUMN K, over all non-BC rows: lo = min(mean - SE), hi = max(mean + SE)
  - each cell's t = (mean - lo) / (hi - lo), clamped to [0, 1]
  - cell color = linear RGB interpolation  #B1D9F0 (light, low) -> #54B2E8 (dark, high)
  - BC row is uncolored and excluded from the range.

Values: best-available per (row, K) — 300-ep re-eval preferred, else the
inline 100-ep eval; mean +- SE over lineages with lineage-count superscript.

Requires in the paper preamble:  \\usepackage[table]{xcolor}   (the [table]
option loads colortbl and enables \\cellcolor).

Rerun after new arms/evals land:  python gen_table.py
"""

import json
import os
import re

import numpy as np

NK = 5  # number of DAgger rounds (columns); K=6 incomplete
DIRS = {
    "s1": "scarcity_study",
    "s2": "scarcity_study_s2",
    "s3": "scarcity_study_s3",
    "s4": "scarcity_study_s4",
    "s5": "scarcity_study_s5",
}
T = "/home/jennyw2/code/lerobot/outputs/training"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "table_full_benchmark.tex")

C_LO = (0xF0, 0xF3, 0xF5)  # off-white      = lowest mean in the column
C_HI = (0x54, 0xB2, 0xE8)  # darker sky blue = highest mean in the column

ROWS = [  # (label, arm pattern)
    (r"HG-DAgger~\cite{kelly2019hg}", "q%d"),
    (r"Fixed iso.\ noise ($\sigma{=}2$)~\cite{ke2021grasping}", "q%d_dn2swv"),
    (r"Fixed iso.\ noise ($\sigma{=}4$)", "q%d_dn4swv"),
    (r"Fixed iso.\ noise ($\sigma{=}8$)", "q%d_dn8swv"),
    (r"Fixed iso.\ noise ($\sigma{=}12$)", "q%d_dn12swv"),
    (r"Fixed iso.\ noise ($\sigma{=}16$)", "q%d_dn16swv"),
    (r"Pooled $\bar{\Sigma}^{\alpha}$ (Ours)", "q%d_dnpool"),
    (r"Per-step $\hat\Sigma^{\alpha}$ (Ours)", "q%d_dnsig"),
]


def outc(p):
    m = re.search(r"sum_rewards': \[([^\]]*)\]", open(p, errors="ignore").read())
    return np.array([float(x) for x in m.group(1).split(",")])


def vec(sd, arm):
    p = f"/home/jennyw2/code/lerobot/outputs/eval300/{DIRS[sd]}/{arm}/eval_info.json"
    try:
        s = np.array(json.load(open(p))["per_task"][0]["metrics"]["successes"], dtype=float)
        if len(s) == 300:
            return s.reshape(3, 100).mean(axis=0)
    except Exception:
        pass
    try:
        v = outc(f"{T}/{DIRS[sd]}/{arm}.log")
        return v if len(v) == 100 else None
    except Exception:
        return None


def cell_stats(pat, K):
    vals = [100 * v.mean() for sd in DIRS if (v := vec(sd, pat % K)) is not None]
    if not vals:
        return None
    m = float(np.mean(vals))
    se = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else None
    return m, se, len(vals)


def lerp_hex(t):
    t = min(max(t, 0.0), 1.0)
    return "".join(f"{round(a + t * (b - a)):02X}" for a, b in zip(C_LO, C_HI))


def cl_cell(K):
    """Closed-loop pooled arm (single lineage): mean and SE over 300-ep evals with seeds 0,1,2."""
    vals = []
    for suf in ["", "_e1", "_e2"]:
        p = f"/home/jennyw2/code/lerobot/outputs/eval300/scarcity_study_cl/q{K}_dnpoolcl{suf}/eval_info.json"
        try:
            s = np.array(json.load(open(p))["per_task"][0]["metrics"]["successes"], dtype=float)
            if len(s) == 300:
                vals.append(100 * s.mean())
        except Exception:
            pass
    if not vals:
        return None
    return (
        float(np.mean(vals)),
        (float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else None),
        len(vals),
    )


def bc_value():
    vals = []
    for suf in ["", "_e1", "_e2"]:
        try:
            s = np.array(
                json.load(
                    open(f"/home/jennyw2/code/lerobot/outputs/eval300/base/bc175k{suf}/eval_info.json")
                )["per_task"][0]["metrics"]["successes"],
                dtype=float,
            )
            vals.append(100 * s.mean())
        except Exception:
            pass
    if vals:
        return (
            float(np.mean(vals)),
            (float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else None),
            len(vals),
        )
    return 100 * outc(f"{T}/bc175k.log").mean(), None, 1


stats = {(lab, K): cell_stats(pat, K) for lab, pat in ROWS for K in range(1, NK + 1)}
ranges = {}
for K in range(1, NK + 1):
    col = [s for (lab, k), s in stats.items() if k == K and s is not None]
    lo = min(m - (se or 0) for m, se, _ in col)
    hi = max(m + (se or 0) for m, se, _ in col)
    ranges[K] = (lo, hi)
COLOR_SCOPE = os.environ.get(
    "COLOR_SCOPE", "global"
)  # "global": one shade scale for the whole table; "column": per DAgger round
if COLOR_SCOPE == "global":
    glo = min(v[0] for v in ranges.values())
    ghi = max(v[1] for v in ranges.values())
    ranges = dict.fromkeys(ranges, (glo, ghi))


def fmt_cell(lab, K):
    s = stats[(lab, K)]
    if s is None:
        return "--"
    m, se, n = s
    lo, hi = ranges[K]
    color = lerp_hex((m - lo) / max(hi - lo, 1e-9))
    body = f"{m:.1f}" + (f" $\\pm$ {se:.1f}" if se is not None else "")
    return f"\\cellcolor[HTML]{{{color}}}{body}\\textsuperscript{{({n})}}"


bc, bc_se, bc_n = bc_value()
lines = [
    "% AUTO-GENERATED by gen_table.py -- edit the script, not this file.",
    "% Preamble requirement: \\usepackage[table]{xcolor}",
    r"\begin{table*}[ht]",
    r"\centering",
    r"\begin{tabular}{l|" + "c|" * (NK - 1) + "c}",
    r"\hline",
    "Method & " + " & ".join(f"$K{{=}}{K}$" for K in range(1, NK + 1)) + r" \\",
    r"\hline",
    f"BC (no aggregation) & \\multicolumn{{{NK}}}{{c}}{{{bc:.1f}"
    + (f" $\\pm$ {bc_se:.1f}" if bc_se is not None else "")
    + "\\textsuperscript{(1)}} \\\\",
    r"\hline",
]
GROUP_RULE = r"\noalign{\hrule height 0.9pt}"  # heavier than \hline so it stays visible over cell shading
for i, (lab, _) in enumerate(ROWS):
    cells = " & ".join(fmt_cell(lab, K) for K in range(1, NK + 1))
    lines.append(f"{lab} & {cells} \\\\")
    if lab.startswith("HG-DAgger") or lab.startswith(r"Fixed iso.\ noise ($\sigma{=}16$"):
        lines.append(GROUP_RULE)
# closed-loop pooled row: single lineage that COLLECTED with the noise-trained
# policy (shares round 1 with lineage 1) — separated from the retrospective block
lines.append(GROUP_RULE)
cl_cells = []
for K in range(1, NK + 1):
    r = cl_cell(K)
    if r is None:
        cl_cells.append("--")
        continue
    m, se, n = r
    lo, hi = ranges[K]
    color = lerp_hex((m - lo) / max(hi - lo, 1e-9))
    body = f"{m:.1f}" + (f" $\\pm$ {se:.1f}" if se is not None else "")
    cl_cells.append(f"\\cellcolor[HTML]{{{color}}}{body}\\textsuperscript{{(1)}}")
lines.append(r"Closed-loop pooled $\bar{\Sigma}^{\alpha}$ (Ours) & " + " & ".join(cl_cells) + r" \\")
lines += [
    r"\hline",
    r"\end{tabular}",
    r"\caption{Success rate on our fixed 100-scenario planar reaching environment, "
    r"mean $\pm$ standard error across independently collected DAgger lineages "
    r"(superscript: number of lineages; 300-episode evaluation where available). "
    r"All noise arms are trained from the base policy on the pooled intervention data "
    r"of rounds $1..K$; HG-DAgger is the same data with no injected noise. "
    r"Calibrated methods use no noise-level sweep. "
    r"The closed-loop row is a single lineage whose rounds $2..5$ were collected "
    r"by the noise-trained policy itself, with the schedule re-estimated on that "
    r"policy each round; it shares round 1 with lineage 1.}",
    r"\label{tab:overall_success}",
    r"\end{table*}",
]
open(OUT, "w").write("\n".join(lines) + "\n")
print(f"wrote {OUT}  (BC {bc:.1f} over {bc_n} eval seeds)")
for K in range(1, NK + 1):
    print(f"  K{K}: color range [{ranges[K][0]:.1f}, {ranges[K][1]:.1f}]")
