"""W for the lever image policy from blend settling (same measurement and
dial fit as fig_w_dial_allrounds.py; 6 arm joints; source = lever r_dag
interventions). Prints per-cell RMS settle, the dial fit W (+bootstrap CI),
and the RMS of the highest available ratio as a fallback.

usage: python measure_w_lever.py [rounds=1,2]
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.expanduser("~/code/lerobot/my_scripts"))
from lib_sa_rollout import progress_guidance_index

ROUNDS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "1,2").split(",")]
TAGS = ["010", "020", "030", "050", "075", "090"]
NARM = 6
CACHE = os.path.expanduser("~/.cache/huggingface/lerobot/JennyWWW")
INT_PREFIX = os.environ.get(
    "INT_PREFIX", "lever_d100_03dagcap_cam_diff"
)  # intervention repo short prefix (…_r_dag{R}, …_r_dag{R}_blend{TTT})


def load(repo):
    fs = glob.glob(f"{CACHE}/{repo}/data/**/*.parquet", recursive=True)
    dfs = []
    for f in fs:
        try:
            dfs.append(pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.state"]))
        except Exception as e:  # file still being written by a running blend lane
            print(f"  skipping unreadable {f}: {type(e).__name__}")
    return pd.concat(dfs) if dfs else None


SRC, MEDS = {}, {}
for rd in ROUNDS:
    db = load(f"{INT_PREFIX}_r_dag{rd}")
    D = {
        int(e): np.stack(g.sort_values("frame_index")["observation.state"].to_numpy())[:, :NARM]
        for e, g in db.groupby("episode_index")
    }
    SRC[rd] = D
    MEDS[rd] = {}
    for e, v in D.items():
        st = np.linalg.norm(np.diff(v, axis=0), axis=1)
        st = st[st > 1e-6]
        MEDS[rd][e] = float(np.median(st))


def settles(repo, rd):
    db = load(repo)
    out = {}
    rj0 = rjs = 0
    if db is None:
        return out, rj0, rjs
    for e, g in db.groupby("episode_index"):
        B = np.stack(g.sort_values("frame_index")["observation.state"].to_numpy())[:, :NARM]
        d0 = {se: float(np.linalg.norm(B[0] - D[0])) for se, D in SRC[rd].items()}
        se = min(d0, key=d0.get)
        if d0[se] > 5 * MEDS[rd][se]:
            rj0 += 1
            continue
        D3 = SRC[rd][se]
        m = MEDS[rd][se]
        j = 0
        dv = []
        for t in range(len(B)):
            j = progress_guidance_index(D3, B[t], j, window=48)
            dv.append(np.linalg.norm(B[t] - D3[j]) / m)
        dv = np.array(dv)
        if len(dv) < 60:
            rjs += 1
            continue
        out[int(e)] = dict(
            settle=float(np.sqrt(np.mean(dv[40 : min(len(dv), 120)] ** 2))),
            esc=bool((dv >= 15).any()),
            src=se,
        )
    return out, rj0, rjs


cfg = json.load(
    open(
        os.path.expanduser(
            "~/code/lerobot/outputs/training/diffusion_approach_lever_13_smooth_delta_basewristng/checkpoints/075000/pretrained_model/config.json"
        )
    )
)
T = int(cfg["num_train_timesteps"])
assert cfg["beta_schedule"] == "squaredcos_cap_v2"


def _ab(t):
    return np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2


i = np.arange(T)
betas = np.minimum(1.0 - _ab((i + 1) / T) / _ab(i / T), 0.999)
alphabar = np.cumprod(1.0 - betas)


def g_of_r(r):
    return 1.0 - alphabar[int(round(r * (T - 1)))]


def dial(g, W, R):
    return W * g / (g + R * (1.0 - g))


cells = []
for rd in ROUNDS:
    for tag in TAGS:
        repo = f"{INT_PREFIX}_r_dag{rd}_blend{tag}"
        Sx, a, b = settles(repo, rd)
        if not Sx:
            continue
        stab = [v["settle"] for v in Sx.values() if not v["esc"]]
        rms = float(np.sqrt(np.mean(np.array(stab) ** 2))) if stab else np.nan
        cells.append(dict(rd=rd, r=int(tag) / 100, rms=rms, n=len(stab), esc=len(Sx) - len(stab)))
        print(
            f"dag{rd} r={int(tag) / 100:.2f}: stable={len(stab)} escaped={len(Sx) - len(stab)} rej_start={a} rej_short={b} "
            f"RMS={rms:.2f} med  (median {np.median(stab) if stab else float('nan'):.2f}, p95 {np.percentile(stab, 95) if stab else float('nan'):.2f})"
        )
fitc = [c for c in cells if c["n"] >= 3 and np.isfinite(c["rms"])]
if len(fitc) >= 3:
    g = np.array([g_of_r(c["r"]) for c in fitc])
    y = np.array([c["rms"] for c in fitc])
    (W, R), pcov = curve_fit(dial, g, y, p0=(10.0, 1.0), bounds=([1.0, 0.001], [60.0, 100.0]), maxfev=20000)
    resid = y - dial(g, W, R)
    print(
        f"\ndial fit over {len(fitc)} cells: W = {W:.2f} (±{np.sqrt(pcov[0, 0]):.2f}), R = {R:.3f}, RMSE {np.sqrt(np.mean(resid**2)):.2f}"
    )
    rng = np.random.default_rng(0)
    Wb = []
    for _ in range(1000):
        idx = rng.integers(0, len(fitc), len(fitc))
        try:
            (w, _), _ = curve_fit(
                dial, g[idx], y[idx], p0=(W, R), bounds=([1.0, 0.001], [60.0, 100.0]), maxfev=5000
            )
            Wb.append(w)
        except Exception:
            pass
    if Wb:
        print(f"bootstrap 95% CI for W: [{np.percentile(Wb, 2.5):.2f}, {np.percentile(Wb, 97.5):.2f}]")
else:
    print("\n<3 fittable cells; no dial fit yet")
