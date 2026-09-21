import json
import os
import sys

import numpy as np

S = os.environ.get("TABLES_REPRO_DIR", "/home/jennyw2/code/lerobot/my_scripts/paper_plots/tables_repro")
SFX, LIN, K = sys.argv[1], sys.argv[2], int(sys.argv[3])
W = 8.0
prof_all = {}
covs = []
shapes = {}
for R in range(1, K + 1):
    z = np.load(f"{S}/analysis/sigma_deltas_{SFX}q{K}_dag{R}.npz", allow_pickle=True)
    eps = sorted({int(k[2:].split("_")[0]) for k in z.files if k.startswith("ep") and k.endswith("_d")})
    repo = f"JennyWWW/planar_12_{LIN}_diff_r_dag{R}"
    prof_all[repo] = {}
    shapes[repo] = {}
    for e in eps:
        d = z[f"ep{e}_d"]
        t = z[f"ep{e}_t"]
        med = float(z[f"ep{e}_med"])
        if d.size == 0:
            continue
        cvs = np.empty((d.shape[0], 3, 3))
        X_all = d.reshape(-1, 3).astype(np.float64) / med
        covs.append((X_all.T @ X_all / len(X_all), len(X_all)))
        for i in range(d.shape[0]):
            X = d[i].reshape(-1, 3).astype(np.float64) / med
            cvs[i] = X.T @ X / len(X)
        T = int(t.max()) + 1
        prof = np.empty((T, 6))
        for f in range(T):
            C = cvs[int(np.argmin(np.abs(t - f)))]
            prof[f] = [C[0, 0], C[1, 1], C[2, 2], C[0, 1], C[0, 2], C[1, 2]]
        prof_all[repo][str(e)] = prof
        shapes[repo][str(e)] = T
tr = float(np.mean([np.mean(np.array(p)[:, :3].sum(1)) for r in prof_all for p in prof_all[r].values()]))
alpha = W * W / tr
sig_sched = {r: {e: (np.array(p) * alpha).tolist() for e, p in prof_all[r].items()} for r in prof_all}
json.dump(sig_sched, open(f"{S}/analysis/noise_schedule_sigma_alpha_{SFX}_K{K}.json", "w"))
Cbar = sum(c * n for c, n in covs) / sum(n for _, n in covs)
alpha_p = W * W / float(np.trace(Cbar))
Cs = Cbar * alpha_p
row = [float(Cs[0, 0]), float(Cs[1, 1]), float(Cs[2, 2]), float(Cs[0, 1]), float(Cs[0, 2]), float(Cs[1, 2])]
pool_sched = {r: {e: [row] * T for e, T in shapes[r].items()} for r in shapes}
json.dump(pool_sched, open(f"{S}/analysis/noise_schedule_pooled_{SFX}_K{K}.json", "w"))
print(f"{SFX} K{K}: alpha_sig={alpha:.3f} alpha_pool={alpha_p:.3f} tr_open={tr:.1f}")
