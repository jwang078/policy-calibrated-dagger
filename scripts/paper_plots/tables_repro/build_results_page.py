import glob
import json
import os
import re

import numpy as np

DIRS = {
    "s1": "/home/jennyw2/code/lerobot/outputs/training/scarcity_study",
    "s2": "/home/jennyw2/code/lerobot/outputs/training/scarcity_study_s2",
    "s3": "/home/jennyw2/code/lerobot/outputs/training/scarcity_study_s3",
    "s4": "/home/jennyw2/code/lerobot/outputs/training/scarcity_study_s4",
    "s5": "/home/jennyw2/code/lerobot/outputs/training/scarcity_study_s5",
}
ROWS = [
    ("baseline", "q%d"),
    ("iso dn2", "q%d_dn2swv"),
    ("iso dn4", "q%d_dn4swv"),
    ("iso dn8", "q%d_dn8swv"),
    ("iso dn12", "q%d_dn12swv"),
    ("iso dn16", "q%d_dn16swv"),
    ("Σ^α sched", "q%d_dnsig"),
    ("pooled DART", "q%d_dnpool"),
]
SEEDED = {"baseline", "iso dn2", "iso dn4", "iso dn8", "iso dn12", "iso dn16"}


def outc(path):
    try:
        m = re.search(r"sum_rewards': \[([^\]]*)\]", open(path, errors="ignore").read())
        return np.array([float(x) for x in m.group(1).split(",")]) if m else None
    except Exception:
        return None


def outc300(sd_dir, arm):
    # 300-ep re-eval: 3 passes over the benchmark, episode i -> scenario i%100
    p = f"/home/jennyw2/code/lerobot/outputs/eval300/{os.path.basename(sd_dir)}/{arm}/eval_info.json"
    try:
        s = np.array(json.load(open(p))["per_task"][0]["metrics"]["successes"], dtype=float)
        return s.reshape(3, 100).mean(axis=0) if len(s) == 300 else None  # per-scenario rate of 3
    except Exception:
        return None


n300 = 0


def vec(sd_dir, arm):
    # best-available per-scenario success rates (length 100), 300-ep preferred
    global n300
    v3 = outc300(sd_dir, arm)
    if v3 is not None:
        n300 += 1
        return v3
    v = outc(f"{sd_dir}/{arm}.log")
    return v if (v is not None and len(v) == 100) else None


# cache all vectors once
V = {}
for name, pat in ROWS:
    for K in range(1, 6):
        for sd, base in DIRS.items():
            V[(name, K, sd)] = vec(base, pat % K)

# DART-responsive set: every scenario some noise arm has ever done strictly
# better on than the SAME seed/K baseline (union over all seeds, rounds, arms)
resp = set()
for K in range(1, 6):
    for sd in DIRS:
        b = V[("baseline", K, sd)]
        if b is None:
            continue
        for name, _ in ROWS:
            if name == "baseline":
                continue
            v = V[(name, K, sd)]
            if v is None:
                continue
            resp |= set(np.where(v > b + 1e-9)[0].tolist())
U = sorted(resp)

data = {}
for name, pat in ROWS:
    data[name] = {}
    for K in range(1, 6):
        full = {}
        d19 = {}
        fix = {}
        for sd in DIRS:
            v = V[(name, K, sd)]
            if v is None:
                continue
            full[sd] = round(float(v.mean() * 100), 1)
            d19[sd] = round(float(v[U].mean() * 100), 1)
            b = V[("baseline", K, sd)]
            if b is not None and name != "baseline":
                fails = b < 0.5
                if fails.sum():
                    fix[sd] = round(100 * float(((v >= 0.5) & fails).sum() / fails.sum()), 1)
        data[name][K] = {"full": full, "d19": d19, "fix": fix}


# BC (round 0) budget-matched arm: 300-ep eval preferred, else inline 100-ep
def bc_vec():
    p = "/home/jennyw2/code/lerobot/outputs/eval300/base/bc175k/eval_info.json"
    try:
        s = np.array(json.load(open(p))["per_task"][0]["metrics"]["successes"], dtype=float)
        if len(s) == 300:
            return s.reshape(3, 100).mean(axis=0), 300
    except Exception:
        pass
    v = outc("/home/jennyw2/code/lerobot/outputs/training/bc175k.log")
    return (v, 100) if (v is not None and len(v) == 100) else (None, 0)


import datetime

seeds_done = {}
for sd, p in DIRS.items():
    done = 0
    for f in glob.glob(p + "/q*.log"):
        if "dnlat" in f or "dnell" in f or "dn6" in f:
            continue
        if "sum_rewards" in open(f, errors="ignore").read():
            done += 1
    seeds_done[sd] = done
bv, beps = bc_vec()
# closed-loop pooled row (single lineage, shares round 1 with s1): raw rates,
# subset rates, and paired gain vs the s1 plain baseline
cl = {}
for K in range(1, 6):
    v = vec("/home/jennyw2/code/lerobot/outputs/training/scarcity_study_cl", f"q{K}_dnpoolcl")
    b = V[("baseline", K, "s1")]
    if v is None:
        continue
    cl[str(K)] = {
        "full": round(float(v.mean() * 100), 1),
        "d19": round(float(v[U].mean() * 100), 1),
        "gfull": (round(float((v.mean() - b.mean()) * 100), 1) if b is not None else None),
        "gd19": (round(float((v[U].mean() - b[U].mean()) * 100), 1) if b is not None else None),
    }
meta = {
    "updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    "seeds_done": seeds_done,
    "n300": n300,
    "resp_n": len(U),
    "resp_eps": U,
    "seeded_rows": sorted(SEEDED),
    "cl": cl,
    "bc": (
        {"full": round(float(bv.mean() * 100), 1), "d19": round(float(bv[U].mean() * 100), 1), "eps": beps}
        if bv is not None
        else None
    ),
}
open("/home/jennyw2/code/lerobot/my_scripts/paper_plots/tables_repro/results_data.json", "w").write(
    json.dumps({"data": data, "meta": meta})
)
print("data built:", {k: meta[k] for k in ("updated", "seeds_done", "n300", "resp_n")})
print("responsive set:", U)

# keep the paper table in lockstep with the artifact (Jenny 2026-09-13)
import subprocess
import sys

subprocess.run(
    [sys.executable, "/home/jennyw2/code/lerobot/my_scripts/paper_plots/tables_repro/paper/gen_table.py"],
    check=False,
)
