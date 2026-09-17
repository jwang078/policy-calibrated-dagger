"""Equilibrium-error ("W dial") figure: blend settling error vs blend ratio, all 5
planar rounds x 13 ratios, with the fit x(g) = W g / (g + R (1-g)).

Run:  python my_scripts/paper_plots/fig_w_dial/plot.py [--cached]
  --cached draws from the bundled w_dial_data.json (no blend re-measurement).
  Style knobs live in CONFIG; my_scripts/paper_plots/fig_w_dial/build_gui.py
  builds an interactive styler that emits a CONFIG block.

W-dial settling figure, ALL-ROUNDS blend family.

Same measurement as fig_w_dial.py's settles(), except the reference is each
round's SOURCE INTERVENTION dataset (planar_12_05dag_diff_r_dag{RD}) instead of
the shared benchmark, and the per-episode settling statistic is the RMS of the
progress-matched distance over ticks 40:120 (project convention: RMS everywhere).

Cells = per-(round, ratio) RMS over stable (non-escaped) episodes.  Fit the dial
x(g) = W*g/(g + R*(1-g)) on cells with >= 3 stable episodes; cells below that
are plotted hollow and excluded from the fit.
"""

import glob
import json
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.expanduser("~/code/lerobot/my_scripts"))
from lib_sa_rollout import progress_guidance_index

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import figdata  # noqa: E402  (snapshots the raw inputs into paper_plots/data)

DATA_JSON = os.path.join(HERE, "w_dial_data.json")
# --cached: skip the (slow) blend re-measurement and draw from w_dial_data.json
CACHED = "--cached" in sys.argv and os.path.exists(DATA_JSON)

# ── loading ──────────────────────────────────────────────────────────────────


def load(repo):
    root = os.path.expanduser(f"~/.cache/huggingface/lerobot/JennyWWW/{repo}")
    fs = glob.glob(root + "/data/**/*.parquet", recursive=True)
    dfs = []
    for f in fs:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:  # a lane killed mid-write leaves a truncated file
            print(f"  skipping unreadable {f}: {type(e).__name__}")
    return pd.concat(dfs) if dfs else None


ROUNDS = [1, 2, 3, 4, 5]
ALL13 = ["005", "010", "015", "020", "025", "030", "035", "040", "045", "050", "065", "075", "090"]
TAGS = {rd: list(ALL13) for rd in ROUNDS}

# Per-round source-intervention reference trajectories + med-steps.
# med-step = median per-tick displacement norm, EXCLUDING near-zero steps
# (intervention episodes can hold still around takeovers; a plain median
# would be dragged toward 0 and blow up the normalized distances).
SRC, MEDS = {}, {}
if not CACHED:
    for rd in ROUNDS:
        db = load(f"planar_12_05dag_diff_r_dag{rd}")
        D = {
            int(e): np.stack(g.sort_values("frame_index")["observation.state"].to_numpy())[:, :3]
            for e, g in db.groupby("episode_index")
        }
        SRC[rd] = D
        MEDS[rd] = {}
        for e, v in D.items():
            st = np.linalg.norm(np.diff(v, axis=0), axis=1)
            st = st[st > 1e-6]
            MEDS[rd][e] = float(np.median(st))

# ── settling measurement (mirrors fig_w_dial.py settles(), source-referenced) ─


def settles(repo, rd):
    """Per-blend-episode settling stat vs the round's source interventions.

    Match each blend episode to the source episode with the nearest start
    state (reject > 5 med-steps away); walk with the monotone progress-matched
    cursor; per-episode stat = RMS of dv[40:min(len,120)]; escape = any tick
    with dv >= 15.
    """
    db = load(repo)
    out = {}
    rej_start = rej_short = 0
    if db is None:
        return out, rej_start, rej_short
    D_all, m_all = SRC[rd], MEDS[rd]
    for e, g in db.groupby("episode_index"):
        B = np.stack(g.sort_values("frame_index")["observation.state"].to_numpy())[:, :3]
        d0 = {se: float(np.linalg.norm(B[0] - D[0])) for se, D in D_all.items()}
        se = min(d0, key=d0.get)
        if d0[se] > 5 * m_all[se]:
            rej_start += 1
            continue
        D3 = D_all[se]
        m = m_all[se]
        j = 0
        dv = []
        for t in range(len(B)):
            j = progress_guidance_index(D3, B[t], j, window=48)
            dv.append(np.linalg.norm(B[t] - D3[j]) / m)
        dv = np.array(dv)
        if len(dv) < 60:
            rej_short += 1
            continue
        out[int(e)] = dict(
            settle=float(np.sqrt(np.mean(dv[40 : min(len(dv), 120)] ** 2))),
            esc=bool((dv >= 15).any()),
            src=se,
        )
    return out, rej_start, rej_short


# ── g(r): policy authority from the trained policy's DDPM schedule ───────────

# the planar BC policy's DDPM schedule (snapshot of its config.json)
cfg = figdata.json_file(
    "planar_bc_policy_config.json",
    "~/code/lerobot/outputs/training/diffusion_planar_3joint_12_delta_stateng/checkpoints/last/pretrained_model/config.json",
)
T = int(cfg["num_train_timesteps"])
assert cfg["beta_schedule"] == "squaredcos_cap_v2", cfg["beta_schedule"]
assert cfg["noise_scheduler_type"] == "DDPM", cfg["noise_scheduler_type"]


def _alpha_bar_fn(t):  # diffusers squaredcos_cap_v2
    return np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2


i = np.arange(T)
betas = np.minimum(1.0 - _alpha_bar_fn((i + 1) / T) / _alpha_bar_fn(i / T), 0.999)
alphabar = np.cumprod(1.0 - betas)


def g_of_r(r):
    """Noise-variance fraction at the partial-denoise switching step."""
    k_sw = int(round(r * (T - 1)))
    return 1.0 - alphabar[k_sw]


def g_of_r_smooth(r):
    """Continuous version for drawing the curve (interpolates the index)."""
    k = np.asarray(r, dtype=float) * (T - 1)
    return 1.0 - np.interp(k, np.arange(T), alphabar)


def dial(g, W, R):
    return W * g / (g + R * (1.0 - g))


if CACHED:
    _d = json.load(open(DATA_JSON))
    cells = _d["cells"]
    eps = [tuple(e) for e in _d["eps"]]
    W_hat, R_hat, W_lo, W_hi = (float(_d[k]) for k in ("W", "R", "W_lo", "W_hi"))
    for c in cells:
        c["rms"] = np.nan if c["rms"] is None else c["rms"]
    print(f"(cached) W = {W_hat:.2f} [{W_lo:.2f}, {W_hi:.2f}]  R = {R_hat:.3f}  {len(cells)} cells")
else:
    # ── collect cells ────────────────────────────────────────────────────────────

    MIN_STABLE = 3
    cells = []  # dicts: rd, r, rms, n_stable, n_esc, n_meas, in_fit
    eps = []  # (rd, r, settle) per stable episode (context dots)
    missing = []
    for rd in ROUNDS:
        for tag in TAGS[rd]:
            r = int(tag) / 100.0
            repo = f"planar_12_05dag_diff_r_dag{rd}_blend{tag}"
            Sx, rj0, rjs = settles(repo, rd)
            if not Sx:
                missing.append(repo)
                continue
            stab = [v["settle"] for v in Sx.values() if not v["esc"]]
            n_esc = sum(v["esc"] for v in Sx.values())
            rms = float(np.sqrt(np.mean(np.array(stab) ** 2))) if stab else np.nan
            cells.append(
                dict(
                    rd=rd,
                    r=r,
                    rms=rms,
                    n_stable=len(stab),
                    n_esc=n_esc,
                    n_meas=len(Sx),
                    rej_start=rj0,
                    rej_short=rjs,
                    in_fit=len(stab) >= MIN_STABLE,
                )
            )
            eps.extend((rd, r, s) for s in stab)
            print(
                f"dag{rd} r={r:.2f}: stable={len(stab)} escaped={n_esc} "
                f"rej_start={rj0} rej_short={rjs} rms={rms:.3f} "
                f"{'FIT' if len(stab) >= MIN_STABLE else 'EXCLUDED(<3 stable)'}"
            )
    if missing:
        print("missing/empty repos skipped:", missing)

    # ── fit (W, R) on qualifying cells ───────────────────────────────────────────

    def dial(g, W, R):
        return W * g / (g + R * (1.0 - g))

    fitc = [c for c in cells if c["in_fit"] and np.isfinite(c["rms"])]
    g_fit = np.array([g_of_r(c["r"]) for c in fitc])
    y_fit = np.array([c["rms"] for c in fitc])
    (W_hat, R_hat), pcov = curve_fit(
        dial, g_fit, y_fit, p0=(10.0, 1.0), bounds=([1.0, 0.001], [30.0, 100.0]), maxfev=20000
    )
    perr = np.sqrt(np.diag(pcov))
    resid = y_fit - dial(g_fit, W_hat, R_hat)
    rmse = float(np.sqrt(np.mean(resid**2)))
    r2 = float(1.0 - np.sum(resid**2) / np.sum((y_fit - y_fit.mean()) ** 2))
    print(
        f"\nfitted W = {W_hat:.2f} med-steps (±{perr[0]:.2f}),  "
        f"R = {R_hat:.3f} (±{perr[1]:.3f})  [{len(fitc)} cells]"
    )
    print(f"R^2 = {r2:.4f}   RMSE = {rmse:.3f} med-steps")

    # bootstrap 95% CI for W, resampling cells
    rng = np.random.default_rng(0)
    Wb = []
    for _ in range(2000):
        idx = rng.integers(0, len(fitc), len(fitc))
        try:
            (w, _), _ = curve_fit(
                dial,
                g_fit[idx],
                y_fit[idx],
                p0=(W_hat, R_hat),
                bounds=([1.0, 0.001], [30.0, 100.0]),
                maxfev=5000,
            )
            Wb.append(w)
        except Exception:
            pass
    W_lo, W_hi = np.percentile(Wb, [2.5, 97.5])
    print(f"bootstrap 95% CI for W: [{W_lo:.2f}, {W_hi:.2f}]  ({len(Wb)}/2000 resamples converged)")

# ── per-cell table ───────────────────────────────────────────────────────────

all_tags = ALL13
print("\nper-cell table (RMS settle [med-steps], n_stable/n_total measured):")
hdr = "round " + "".join(f"{'r=' + str(int(t) / 100):>16}" for t in all_tags)
print(hdr)
for rd in ROUNDS:
    row = f"dag{rd}  "
    for tag in all_tags:
        c = next((c for c in cells if c["rd"] == rd and abs(c["r"] - int(tag) / 100) < 1e-9), None)
        if c is None:
            row += f"{'—':>16}"
        else:
            mark = "" if c["in_fit"] else "*"
            row += f"{c['rms']:>8.2f}{mark:<1} {c['n_stable']}/{c['n_meas']:<4}"
    print(row)
print("(* = <3 stable episodes: plotted hollow, excluded from the fit)")

# ── CONFIG (editable; the GUI in w_dial_gui.html emits this block) ──────────
CONFIG = {
    "title": "Blended Policy Error at Equilibrium vs Blend Ratio β",
    "xlabel": "Blend Ratio β",
    "ylabel": "Blended Policy Error at Equilibrium $x(\\sigma_{k_{sw}}^2)$ [med-steps]",
    "annot_text": "Policy Error at Equilibrium W = {W:.1f} med-steps",
    "annot_x": -0.017,
    "annot_y_offset": 0.35,
    "annot_ha": "left",
    "annot_va": "bottom",
    "annot_color": "#444444",
    "annot_size": 8,
    "round_colors": {1: "#f66151", 2: "#ffbe6f", 3: "#f9f06b", 4: "#8ff0a4", 5: "#99c1f1"},
    "fit_color": "#333333",
    "wline_color": "#666666",
    "ci_color": "#666666",
    "eps_color": "#8a949e",
    "legend": {
        "rounds": {"label": "Round {rd} (RMS)", "show": True},
        "excluded": {"label": "<3 stable (excluded)", "show": True},
        "eps": {"label": "Per-episode $x(\\sigma_{k_{sw}}^2)$", "show": True},
        "ci": {"label": "W 95% CI [{Wlo:.1f}, {Whi:.1f}]", "show": True},
        "fit": {"label": "Fitted $x(\\sigma_{k_{sw}}^2)$", "show": True},
    },
    "show_eps": True,
    "show_wline": True,
    "show_ci": True,
    "show_annot": True,
    "legend_loc": "lower right",
    "legend_ncol": 2,
    "title_size": 13.5,
    "label_size": 11.5,
    "tick_size": 10.5,
    "legend_size": 10,
    "top_axis_g": False,
    "top_axis_label": "noise-variance fraction $\\sigma_{k_{sw}}^2 = 1 - \\bar\\alpha_{k_{sw}}$",
    "top_axis_ticks": [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95],
}
_fmt = dict(W=W_hat, Wlo=W_lo, Whi=W_hi, R=R_hat)
import re as _re


def _sub(s, **kw):
    """Substitute only the known placeholders ({W} {Wlo} {Whi} {R} {rd}, with an
    optional :fmt spec) so LaTeX braces in labels survive untouched.
    """
    vals = {**_fmt, **kw}
    return _re.sub(
        r"\{(W|Wlo|Whi|R|rd)(?::([^}]*))?\}", lambda m: format(vals[m.group(1)], m.group(2) or ""), s
    )


json.dump(
    {
        "cells": cells,
        "eps": eps,
        "W": W_hat,
        "R": R_hat,
        "W_lo": W_lo,
        "W_hi": W_hi,
        "curve": [
            [float(r), float(v)]
            for r, v in zip(np.linspace(0, 1, 200), dial(g_of_r_smooth(np.linspace(0, 1, 200)), W_hat, R_hat))
        ],
        "config": CONFIG,
    },
    open(DATA_JSON, "w"),
    default=float,
)

# ── figure ───────────────────────────────────────────────────────────────────

rdcol = CONFIG["round_colors"]
plt.rcParams.update(
    {
        "font.size": CONFIG["tick_size"],
        "axes.labelsize": CONFIG["label_size"],
        "axes.titlesize": CONFIG["title_size"],
        "font.family": "DejaVu Sans",
    }
)
fig, ax = plt.subplots(figsize=(7, 5.2))

# fitted curve over r in [0, 1]
rr = np.linspace(0.0, 1.0, 400)
ax.plot(rr, dial(g_of_r_smooth(rr), W_hat, R_hat), "-", color=CONFIG["fit_color"], lw=1.8, zorder=8)

# asymptote at W
if CONFIG["show_wline"]:
    ax.axhline(W_hat, color=CONFIG["wline_color"], lw=1.1, ls="--", zorder=2)
if CONFIG["show_annot"]:
    ax.annotate(
        _sub(CONFIG["annot_text"]),
        xy=(0.0, W_hat),
        xytext=(CONFIG["annot_x"], W_hat + CONFIG["annot_y_offset"]),
        ha=CONFIG["annot_ha"],
        va=CONFIG["annot_va"],
        fontsize=CONFIG["annot_size"],
        color=CONFIG["annot_color"],
    )

# bootstrap 95% CI band on W
if CONFIG["show_ci"]:
    ax.axhspan(W_lo, W_hi, color=CONFIG["ci_color"], alpha=0.10, zorder=1)

# faint per-episode values (context only — NOT what the curve is fit to)
jrng = np.random.default_rng(0)
if CONFIG["show_eps"]:
    for rd, r, s in eps:
        ax.scatter(
            r + jrng.normal(0, 0.007), s, s=9, alpha=0.16, c=CONFIG["eps_color"], edgecolors="none", zorder=2
        )

# THE FIT TARGETS: per-(round, ratio) RMS cells, colored by round;
# hollow = excluded (<3 stable episodes)
for c in cells:
    if not np.isfinite(c["rms"]):
        continue
    x = c["r"] + (c["rd"] - 3) * 0.008
    if c["in_fit"]:
        ax.scatter(
            x, c["rms"], s=70, marker="o", c=rdcol[c["rd"]], edgecolors="white", linewidths=1.0, zorder=6
        )
    else:
        ax.scatter(
            x,
            c["rms"],
            s=70,
            marker="o",
            facecolors="none",
            edgecolors=rdcol[c["rd"]],
            linewidths=1.4,
            zorder=6,
        )

# legend
LG = CONFIG["legend"]
handles = []
if LG["rounds"]["show"]:
    handles += [
        plt.Line2D(
            [],
            [],
            marker="o",
            ls="",
            color=rdcol[rd],
            ms=6.5,
            markeredgecolor="white",
            label=_sub(LG["rounds"]["label"], rd=rd),
        )
        for rd in ROUNDS
    ]
if LG["excluded"]["show"] and any(not c["in_fit"] for c in cells):
    handles.append(
        plt.Line2D(
            [],
            [],
            marker="o",
            ls="",
            markerfacecolor="none",
            markeredgecolor="#666666",
            ms=6.5,
            label=_sub(LG["excluded"]["label"]),
        )
    )
if LG["eps"]["show"] and CONFIG["show_eps"]:
    handles.append(
        plt.Line2D(
            [],
            [],
            marker="o",
            ls="",
            color=CONFIG["eps_color"],
            ms=4,
            alpha=0.5,
            label=_sub(LG["eps"]["label"]),
        )
    )
if LG["ci"]["show"] and CONFIG["show_ci"]:
    handles.append(
        plt.Rectangle((0, 0), 1, 1, color=CONFIG["ci_color"], alpha=0.15, label=_sub(LG["ci"]["label"]))
    )
if LG["fit"]["show"]:
    handles.append(
        plt.Line2D([], [], ls="-", color=CONFIG["fit_color"], lw=1.8, label=_sub(LG["fit"]["label"]))
    )
if handles:
    ax.legend(
        handles=handles,
        fontsize=CONFIG["legend_size"],
        frameon=False,
        loc=CONFIG["legend_loc"],
        ncol=CONFIG["legend_ncol"],
        columnspacing=1.0,
        handletextpad=0.5,
    )

ax.set_xlim(-0.02, 1.02)
ymax_pts = max([s for _, _, s in eps] + [c["rms"] for c in cells if np.isfinite(c["rms"])])
ax.set_ylim(0, max(W_hat * 1.18, W_hi * 1.08, ymax_pts * 1.08))
ax.set_xlabel(CONFIG["xlabel"], fontsize=CONFIG["label_size"])
ax.set_ylabel(CONFIG["ylabel"], fontsize=CONFIG["label_size"])
ax.set_title(CONFIG["title"], fontsize=CONFIG["title_size"])
ax.tick_params(labelsize=CONFIG["tick_size"])
if CONFIG.get("top_axis_g"):
    # secondary axis in g: forward g(beta) via the schedule, inverse by interpolation
    _bb = np.linspace(0.0, 1.0, 2001)
    _gg = g_of_r_smooth(_bb)
    _fwd = lambda b: np.interp(b, _bb, _gg)
    _inv = lambda g: np.interp(g, _gg, _bb)
    secax = ax.secondary_xaxis("top", functions=(_fwd, _inv))
    secax.set_xticks([t for t in CONFIG["top_axis_ticks"] if 0 < t < 1])
    secax.set_xticklabels(
        [f"{t:g}" for t in CONFIG["top_axis_ticks"] if 0 < t < 1], fontsize=CONFIG["tick_size"] - 1
    )
    secax.set_xlabel(CONFIG["top_axis_label"], fontsize=CONFIG["label_size"] - 1)
    secax.spines["top"].set_visible(True)
    # title sits above the secondary axis
    ax.title.set_position((0.5, 1.0))
    ax.set_title(CONFIG["title"], fontsize=CONFIG["title_size"], pad=22)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#dddddd", lw=0.6, zorder=0)
ax.set_axisbelow(True)
fig.tight_layout()
out = os.path.join(HERE, "fig_w_dial.png")
fig.savefig(out, dpi=150)
print("saved", out)
