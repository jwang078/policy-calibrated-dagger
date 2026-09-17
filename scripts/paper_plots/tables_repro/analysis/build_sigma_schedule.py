import json
import os
import sys

import numpy as np

S = os.environ.get("TABLES_REPRO_DIR", "/home/jennyw2/code/lerobot/my_scripts/paper_plots/tables_repro")
TAG, LIN = sys.argv[1], sys.argv[2]  # s1 05dag | s2 06dag
W = 8.0  # dial-extrapolated deployed scale [med]
sched = {}
trs = []
for K in (1, 2):
    z = np.load(f"{S}/analysis/sigma_deltas_{TAG}_dag{K}.npz", allow_pickle=True)
    eps = sorted({int(k[2:].split("_")[0]) for k in z.files if k.startswith("ep") and k.endswith("_d")})
    repo = f"JennyWWW/planar_12_{LIN}_diff_r_dag{K}"
    sched[repo] = {}
    for e in eps:
        d = z[f"ep{e}_d"]
        t = z[f"ep{e}_t"]
        med = float(z[f"ep{e}_med"])
        if d.size == 0:
            continue
        covs = np.empty((d.shape[0], 3, 3))
        for i in range(d.shape[0]):
            X = d[i].reshape(-1, 3) / med
            covs[i] = X.T @ X / len(X)
        T = int(t.max()) + 1
        prof = np.empty((T, 6))
        for f in range(T):
            C = covs[int(np.argmin(np.abs(t - f)))]
            prof[f] = [C[0, 0], C[1, 1], C[2, 2], C[0, 1], C[0, 2], C[1, 2]]
        trs.append(prof[:, :3].sum(axis=1).mean())
        sched[repo][str(e)] = prof
tr = float(np.mean(trs))
alpha = W * W / tr
for repo in sched:
    for e in sched[repo]:
        sched[repo][e] = (sched[repo][e] * alpha).tolist()
out = f"{S}/analysis/noise_schedule_sigma_alpha_{TAG}_K2.json"
json.dump(sched, open(out, "w"))
C = np.array(
    sched[f"JennyWWW/planar_12_{LIN}_diff_r_dag1"][
        list(sched[f"JennyWWW/planar_12_{LIN}_diff_r_dag1"].keys())[0]
    ]
)[10]
M = np.array([[C[0], C[3], C[4]], [C[3], C[1], C[5]], [C[4], C[5], C[2]]])
print(
    f"{TAG}: tr(Sigma_open)={tr:.1f} med^2  alpha_hat={alpha:.3f}  example eigen-sigmas={np.sqrt(np.maximum(np.linalg.eigvalsh(M), 0)).round(2)}  -> {out}"
)
