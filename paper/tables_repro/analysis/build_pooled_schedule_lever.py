"""Pooled alpha*Sigma-hat schedule for the lever image policy (6 joints).

usage: python build_pooled_schedule_lever.py K W
  reads  $S/analysis/sigma_deltas_lever_q{K}_dag{R}.npz for R=1..K
  writes $S/analysis/noise_schedule_pooled_lever_K{K}.json
         rows = 21 upper-triangle entries of the 6x6 covariance (med-step^2),
         IU order (i<=j row-major) = DartChunkDataset._IU6; one constant row
         per demo frame per intervention episode.
Sigma-hat = pooled UNCENTERED second moment of the policy-vs-expert chunk
deltas (frame-weighted over anchors and rounds). DART: Sigma^alpha = alpha/(T tr Sigma-hat) Sigma-hat
with alpha = T W^2 (the policy's cumulative squared settling error), i.e. the scale
s = W^2 / tr Sigma-hat and tr Sigma^alpha = W^2.
"""

import json
import os
import sys

import numpy as np

S = os.environ.get("TABLES_REPRO_DIR", "/home/jennyw2/code/lerobot/my_scripts/paper_plots/tables_repro")
K, W = int(sys.argv[1]), float(sys.argv[2])
INT_PREFIX = os.environ.get("INT_PREFIX", "lever_d100_03dagcap_cam_diff")
TAGPFX = os.environ.get(
    "TAGPFX", ""
)  # e.g. TAGPFX=r84_ -> sigma_deltas_lever_r84_q1_dag1.npz, noise_schedule_pooled_lever_r84_K1.json
IU6 = [(i, j) for i in range(6) for j in range(i, 6)]
acc = np.zeros((6, 6))
n_tot = 0
shapes = {}
for R in range(1, K + 1):
    z = np.load(f"{S}/analysis/sigma_deltas_lever_{TAGPFX}q{K}_dag{R}.npz")
    repo = f"JennyWWW/{INT_PREFIX}_r_dag{R}"
    shapes[repo] = {}
    for k in z.files:
        if not k.endswith("_d"):
            continue
        e = int(k[2:-2])
        d = z[k]
        t = z[f"ep{e}_t"]
        med = float(z[f"ep{e}_med"])
        if d.size == 0:
            continue
        X = d.reshape(-1, 6).astype(np.float64) / med
        acc += X.T @ X
        n_tot += len(X)
        shapes[repo][str(e)] = int(t.max()) + 1
C = acc / n_tot
scale = W * W / float(np.trace(C))  # DART scale s = alpha/(T tr Sigma-hat), alpha = T W^2
Cs = C * scale
row = [float(Cs[i, j]) for (i, j) in IU6]
sched = {r: {e: [row] * T for e, T in shapes[r].items()} for r in shapes}
out = f"{S}/analysis/noise_schedule_pooled_lever_{TAGPFX}K{K}.json"
json.dump(sched, open(out, "w"))
ev = np.linalg.eigvalsh(Cs)
print(
    f"K{K}: W={W:.2f} tr_open={np.trace(C):.1f} med^2  scale s={scale:.4f} (DART alpha/T = W^2 = {W * W:.1f} med^2)  "
    f"per-joint sigma (med) = {np.sqrt(np.diag(Cs)).round(2).tolist()}  eig ratio max/min={ev.max() / max(ev.min(), 1e-9):.1f}"
)
print("wrote", out)
