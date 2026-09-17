"""Per-joint calibrated noise (pooled Sigma-bar^alpha) per DAgger round, planar and lever, in the same
median-demo-step units as the fixed-isotropic sigma sweep. Filled = calibrated sigma_j = sqrt(diag(s*Sigma_bar));
hollow = raw open-loop spread sqrt(diag(Sigma_bar)) before calibration. Data: sigma_deltas_*.npz from the
open-loop measurements (planar: mean +- SE over 5 lineages; lever: single lineage).
Usage: python plot.py  [OUT=path.pdf]
"""

import glob
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
S = os.environ.get("DELTAS_DIR", os.path.join(os.path.dirname(HERE), "tables_repro", "analysis"))  # sigma_deltas_*.npz caches
OUT = os.environ.get("OUT", os.path.join(HERE, "fig_noise_levels.pdf"))
FIXED = [2, 4, 8, 12, 16]
PLANAR_W = 8.0
LEVER_W = {1: 8.10, 2: 7.94, 3: 8.07, 4: 8.06}
PLANAR_KS = [
    int(k) for k in os.environ.get("PLANAR_KS", "1,2,3,4,5").split(",")
]  # rounds with complete results
LEVER_KS = [int(k) for k in os.environ.get("LEVER_KS", "1,2").split(",")]
MED_PLANAR, MED_LEVER = 0.0176, 0.0166


def cbar(prefix, K, nj):
    covs = []
    for R in range(1, K + 1):
        p = f"{S}/{prefix}q{K}_dag{R}.npz"
        if not os.path.exists(p) and K == 2:
            p = f"{S}/{prefix.rstrip('q')}_dag{R}.npz"  # s1/s2 K=2 legacy names
        if not os.path.exists(p):
            return None
        z = np.load(p, allow_pickle=True)
        for e in sorted(
            {int(k[2:].split("_")[0]) for k in z.files if k.startswith("ep") and k.endswith("_d")}
        ):
            d = z[f"ep{e}_d"]
            med = float(z[f"ep{e}_med"])
            if d.size == 0:
                continue
            X = d.reshape(-1, nj).astype(np.float64) / med
            covs.append((X.T @ X / len(X), len(X)))
    return sum(c * n for c, n in covs) / sum(n for _, n in covs) if covs else None


def sig(C, W):
    raw = np.sqrt(np.diag(C))
    cal = np.sqrt(np.diag(C * W * W / np.trace(C)))
    return raw, cal


# ---- planar: lineages s1..s5, K=1..6
planar = {}
for K in PLANAR_KS:
    raws, cals = [], []
    for s in ["s1", "s2", "s3", "s4", "s5"]:
        C = cbar(f"sigma_deltas_{s}", K, 3)
        if C is None:
            continue
        r, c = sig(C, PLANAR_W)
        raws.append(r)
        cals.append(c)
    if cals:
        planar[K] = (np.array(raws), np.array(cals))
# ---- lever: single lineage, K=1..4
lever = {}
for K in LEVER_KS:
    C = cbar("sigma_deltas_lever_r84_", K, 6)
    if C is not None:
        lever[K] = sig(C, LEVER_W[K])

CFG = dict(
    cal_color="#2a5f9e",
    raw_color="#b8b8b8",
    markers=["o", "s", "^", "D", "v", "P"],
    planar_joints=["Joint 1 (base)", "Joint 2", "Joint 3"],
    lever_joints=["Shoulder pan", "Shoulder lift", "Elbow", "Wrist 1", "Wrist 2", "Wrist 3"],
    fixed_color="#b0b0b0",
    fig_size=(7.2, 2.7),
    ymax=18.5,
    show_raw=os.environ.get("SHOW_RAW", "1") == "1",
)
plt.rcParams.update({"font.size": 8, "font.family": os.environ.get("FONT", "DejaVu Sans")})
fig, axes = plt.subplots(1, 2, figsize=CFG["fig_size"], gridspec_kw=dict(width_ratios=[5, 3.9]))


def panel(ax, data, names, med, title, se=True):
    Ks = sorted(data)
    for s_ in FIXED:
        ax.axhline(s_, color=CFG["fixed_color"], ls=(0, (4, 3)), lw=0.8, zorder=1)
    for j, name in enumerate(names):
        m = CFG["markers"][j]
        raw = np.array([data[K][0][:, j].mean() if data[K][0].ndim == 2 else data[K][0][j] for K in Ks])
        cal = np.array([data[K][1][:, j].mean() if data[K][1].ndim == 2 else data[K][1][j] for K in Ks])
        err = np.array(
            [
                data[K][1][:, j].std(ddof=1) / np.sqrt(len(data[K][1]))
                if (se and data[K][1].ndim == 2 and len(data[K][1]) > 1)
                else 0
                for K in Ks
            ]
        )
        off = 0.0  # points sit exactly at round K
        if CFG["show_raw"]:
            ax.plot(np.array(Ks) + off, raw, marker=m, color=CFG["raw_color"], lw=0.9, ms=4.0, zorder=2)
        ax.errorbar(
            np.array(Ks) + off,
            cal,
            yerr=err if se else None,
            marker=m,
            color=CFG["cal_color"],
            lw=1.2,
            ms=4.4,
            capsize=2,
            zorder=3,
        )
    ax.set_xticks(Ks)
    ax.set_xlim(Ks[0] - 0.5, Ks[-1] + 0.5)
    ax.set_ylim(0, CFG["ymax"])
    ax.set_yticks(FIXED)
    ax.set_yticklabels([f"$\\sigma$={v}" for v in FIXED])
    ax.set_xlabel("DAgger round $K$")
    ax.set_title(title, fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax2 = ax.twinx()
    ax2.set_ylim(0, CFG["ymax"] * med)
    ax2.spines[["top", "left"]].set_visible(False)
    ax2.set_yticks([v * med for v in FIXED])
    ax2.set_yticklabels([f"{v * med:.2g}" + (" rad" if v == FIXED[-1] else "") for v in FIXED])
    ax2.tick_params(labelsize=6.5)
    from matplotlib.lines import Line2D

    hj = [
        Line2D([], [], marker=CFG["markers"][j], color="#555555", lw=0, ms=4.4, label=name)
        for j, name in enumerate(names)
    ]
    leg = ax.legend(
        handles=hj,
        fontsize=6.2,
        loc="upper center" if len(names) > 3 else "upper right",
        ncol=3 if len(names) > 3 else 1,
        frameon=True,
        handlelength=1.2,
        fancybox=False,
        framealpha=1.0,
        edgecolor="#d0d0d0",
        facecolor="white",
        borderpad=0.6,
        labelspacing=0.35,
        columnspacing=0.7,
        handletextpad=0.5,
    )
    leg.get_frame().set_linewidth(0.6)
    leg.set_zorder(10)


panel(axes[0], planar, CFG["planar_joints"], MED_PLANAR, "2D Planar Reacher")
panel(axes[1], lever, CFG["lever_joints"], MED_LEVER, "3D Engine-Lever Reaching", se=False)
axes[0].set_ylabel("Per-joint $\\sigma$", fontsize=8)
from matplotlib.lines import Line2D

hc = [
    Line2D(
        [],
        [],
        color=CFG["cal_color"],
        lw=1.4,
        marker="o",
        ms=4.4,
        label="Pooled calibrated $\\bar\\Sigma^{\\alpha}$",
    )
]
if CFG["show_raw"]:
    hc.append(
        Line2D(
            [],
            [],
            color=CFG["raw_color"],
            lw=1.0,
            marker="o",
            ms=4.0,
            label="Pooled open-loop spread $\\bar\\Sigma$ before rescaling",
        )
    )
fig.legend(handles=hc, loc="lower center", ncol=2, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.04))
plt.tight_layout(rect=(0, 0.05, 1, 1))
plt.savefig(OUT, bbox_inches="tight", dpi=200)
plt.savefig(OUT.rsplit(".", 1)[0] + ".png", bbox_inches="tight", dpi=200)
print("wrote", OUT)
for K, (r, c) in planar.items():
    print(f"planar K{K}: raw {np.round(r.mean(0), 2).tolist()} cal {np.round(c.mean(0), 2).tolist()}")
for K, (r, c) in lever.items():
    print(f"lever K{K}: raw {np.round(r, 2).tolist()} cal {np.round(c, 2).tolist()}")
