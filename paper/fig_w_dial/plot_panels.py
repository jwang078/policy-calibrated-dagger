"""Two-panel view of the equilibrium fit: (a) settling error vs blend ratio
(W = asymptote, R = half-saturation point); (b) normalized error x/W vs the
noise fraction g on a log axis, where the one-parameter shape g/(g+R(1-g))
makes R visible directly (x/W = 1/2 at g* = R/(1+R)).
Reads w_dial_data.json (written by fig_w_dial_allrounds.py).
"""

import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = os.path.dirname(os.path.abspath(__file__))
import sys  # noqa: E402

sys.path.insert(0, os.path.dirname(S))
import figdata  # noqa: E402

D = json.load(open(f"{S}/w_dial_data.json"))
W, R, Wlo, Whi = D["W"], D["R"], D["W_lo"], D["W_hi"]
cells = [c for c in D["cells"] if c["rms"] is not None and np.isfinite(c["rms"])]
rdcol = {1: "#2a78d6", 2: "#eda100", 3: "#008300", 4: "#e87ba4", 5: "#4a3aa7"}

# g(beta) from the trained model's DDPM schedule (squaredcos_cap_v2, T=100)
cfg = figdata.json_file(
    "planar_bc_policy_config.json",
    "~/code/lerobot/outputs/training/diffusion_planar_3joint_12_delta_stateng/checkpoints/last/pretrained_model/config.json",
)
T = int(cfg["num_train_timesteps"])
_ab = lambda t: np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2
i = np.arange(T)
betas = np.minimum(1.0 - _ab((i + 1) / T) / _ab(i / T), 0.999)
alphabar = np.cumprod(1.0 - betas)
g_of_r = lambda r: 1.0 - alphabar[int(round(r * (T - 1)))]
g_smooth = lambda r: 1.0 - np.interp(np.asarray(r) * (T - 1), np.arange(T), alphabar)
dial = lambda g, W, R: W * g / (g + R * (1.0 - g))
g_half = R / (1.0 + R)
# beta at which g(beta) = g_half
rr = np.linspace(0, 1, 2000)
beta_half = float(rr[np.argmin(np.abs(g_smooth(rr) - g_half))])

plt.rcParams.update({"font.size": 9.5, "font.family": "DejaVu Sans"})
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.3), gridspec_kw={"width_ratios": [1.15, 1]})

# (a) beta-space
a.axhspan(Wlo, Whi, color="#666666", alpha=0.10, zorder=1)
a.axhline(W, color="#666666", lw=1.1, ls="--", zorder=2)
a.plot(rr, dial(g_smooth(rr), W, R), "-", color="#333333", lw=1.8, zorder=3)
for c in cells:
    x = c["r"] + (c["rd"] - 3) * 0.008
    if c["in_fit"]:
        a.scatter(x, c["rms"], s=55, c=rdcol[c["rd"]], edgecolors="white", linewidths=0.9, zorder=6)
    else:
        a.scatter(x, c["rms"], s=55, facecolors="none", edgecolors=rdcol[c["rd"]], linewidths=1.3, zorder=6)
a.plot([beta_half], [W / 2], "o", ms=8, mfc="white", mec="#333333", mew=1.6, zorder=7)
a.annotate(f"W = {W:.1f}", xy=(0.005, W), xytext=(0.005, W + 0.3), ha="left", va="bottom", color="#444444")
a.annotate(
    f"W/2 at β = {beta_half:.2f}\n(g* = R/(1+R) = {g_half:.3f})",
    xy=(beta_half, W / 2),
    xytext=(beta_half + 0.12, W / 2 - 1.6),
    ha="left",
    va="top",
    color="#444444",
    fontsize=8.5,
    arrowprops=dict(arrowstyle="-", color="#888888", lw=0.8),
)
a.set_xlim(-0.02, 1.02)
a.set_ylim(0, max(W * 1.18, Whi * 1.08, max(c["rms"] for c in cells) * 1.08))
a.set_xlabel("blend ratio β  (policy authority →)")
a.set_ylabel("settling error  [median-step units]")
a.set_title("(a) equilibrium error vs blend ratio")
hs = [
    plt.Line2D([], [], marker="o", ls="", color=rdcol[rd], ms=6, markeredgecolor="white", label=f"round {rd}")
    for rd in range(1, 6)
]
hs.append(plt.Line2D([], [], ls="-", color="#333333", lw=1.8, label=f"fit  W={W:.2f}, R={R:.3f}"))
a.legend(handles=hs, fontsize=8, frameon=False, loc="lower right", ncol=2)

# (b) g-space, normalized
gg = np.logspace(-3, 0, 400)
b.plot(gg, dial(gg, 1.0, R), "-", color="#333333", lw=1.8, zorder=3)
for c in cells:
    g = g_of_r(c["r"])
    y = c["rms"] / W
    if c["in_fit"]:
        b.scatter(g, y, s=55, c=rdcol[c["rd"]], edgecolors="white", linewidths=0.9, zorder=6)
    else:
        b.scatter(g, y, s=55, facecolors="none", edgecolors=rdcol[c["rd"]], linewidths=1.3, zorder=6)
b.axhline(1.0, color="#666666", lw=1.1, ls="--", zorder=2)
b.axhline(0.5, color="#bbbbbb", lw=0.8, ls=":", zorder=2)
b.axvline(g_half, color="#bbbbbb", lw=0.8, ls=":", zorder=2)
b.plot([g_half], [0.5], "o", ms=8, mfc="white", mec="#333333", mew=1.6, zorder=7)
b.annotate(
    f"g* = R/(1+R) = {g_half:.3f}",
    xy=(g_half, 0.5),
    xytext=(g_half * 1.6, 0.36),
    ha="left",
    va="top",
    color="#444444",
    fontsize=8.5,
    arrowprops=dict(arrowstyle="-", color="#888888", lw=0.8),
)
b.set_xscale("log")
b.set_xlim(2e-3, 1.05)
b.set_ylim(0, 1.45)
b.set_xlabel("noise fraction g = 1 − ᾱ$_{k_{sw}}$  (log)")
b.set_ylabel("settling error / W")
b.set_title("(b) normalized: x/W = g / (g + R(1−g))")
for ax in (a, b):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e2e2e2", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
fig.tight_layout()
out = os.path.join(S, "fig_w_R_panels.png")
fig.savefig(out, dpi=150)
print(f"W={W:.2f} R={R:.4f} g*={g_half:.4f} beta_half={beta_half:.3f}  -> {out}")
