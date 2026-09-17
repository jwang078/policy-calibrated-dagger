#!/usr/bin/env python
"""Partial denoising at one observation: the recorded expert chunk (r = 0) and
policy samples obtained by forward-noising that chunk to the switching step of
ratio r and denoising back (r = 0.2 .. 1.0; r = 1 is a pure policy sample).
Planar 3-joint task, round-3 policy, drawn in end-effector space. Open-loop:
no simulator. Style knobs in CONFIG.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
import torch

# ── CONFIG ───────────────────────────────────────────────────────────────────
EPISODE, ANCHOR = (
    int(os.environ.get("EPISODE", 10)),
    int(os.environ.get("ANCHOR", 62)),
)  # planar_12_05dag_diff_r_dag3 episode / frame
RATIOS = tuple(float(x) for x in os.environ.get("RATIOS", "0.2,0.4,0.6,0.8,1.0").split(","))
N_SAMPLES = int(os.environ.get("N_SAMPLES", 1))  # samples per ratio
SEED = int(os.environ.get("SEED", 0))
RESEED = {
    float(k): int(v) for k, v in (kv.split(":") for kv in os.environ.get("RESEED", "1.0:1").split(",") if kv)
}  # e.g. RESEED="1.0:3" re-seeds the generator right before that ratio (later ratios are affected too)
POLICY = "outputs/training/scarcity_study/q3/checkpoints/last/pretrained_model"  # round-3 planar policy
SPACE = os.environ.get(
    "SPACE", "pca"
)  # "pca": dataset-PCA plane of joint space (rotated so the episode's motion is horizontal); "ee": end-effector x/z
Y_STRETCH = os.environ.get(
    "Y_STRETCH", "auto"
)  # pca only: "auto" = fill a square-ish panel (independent axis scales); a number = fixed across/along display ratio
Y_STRETCH = Y_STRETCH if Y_STRETCH == "auto" else float(Y_STRETCH)
OUT = os.environ.get(
    "OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_partial_denoise.png")
)
# ─────────────────────────────────────────────────────────────────────────────
LR = os.path.expanduser("~/code/lerobot")
sys.path.insert(0, f"{LR}/src")
sys.path.insert(0, f"{LR}/my_scripts")
os.chdir(LR)
import dart_sim_video as dsv
import matplotlib
import pybullet as pb
from safetensors.torch import load_file

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.processor import PolicyProcessorPipeline

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = f"JennyWWW/planar_12_05dag_diff_r_dag{3}"
policy = DiffusionPolicy.from_pretrained(POLICY).cuda().eval()
pre = PolicyProcessorPipeline.from_pretrained(POLICY, config_filename="policy_preprocessor.json")
stt = load_file(POLICY + "/policy_preprocessor_step_5_normalizer_processor.safetensors")
lo = stt["action.min"].cuda()
rng_ = (stt["action.max"].cuda() - lo).clamp(min=1e-8)
cfg = policy.config
fps = 30
N_ACT, start = cfg.n_action_steps, cfg.n_obs_steps - 1
dts = {
    "observation.state": [i / fps for i in range(1 - cfg.n_obs_steps, 1)],
    "observation.environment_state": [i / fps for i in range(1 - cfg.n_obs_steps, 1)],
    "action": [i / fps for i in range(1 - cfg.n_obs_steps, 1 - cfg.n_obs_steps + cfg.horizon)],
}
ds = LeRobotDataset(REPO, delta_timestamps=dts, episodes=[EPISODE])
epi = np.array(ds.hf_dataset["episode_index"])
fri = np.array(ds.hf_dataset["frame_index"])
idx = np.where(epi == EPISODE)[0]
idx = idx[np.argsort(fri[idx])]
item = ds[int(idx[ANCHOR])]
ns = policy.diffusion.noise_scheduler
T_train = ns.config.num_train_timesteps
raw = {k: item[k][None] for k in ("observation.state", "observation.environment_state", "action")}
proc = pre(raw)
nb = {k: proc[k].cuda() for k in ("observation.state", "observation.environment_state")}
ng = proc["action"].cuda()
torch.manual_seed(SEED)
# the policy predicts actions RELATIVE to the current observed state (rel stats): add the anchor state back
q_anchor = item["observation.state"][-1][:3].numpy().astype(float)


def unnorm(a):
    return ((a + 1) / 2 * rng_ + lo)[:, :, :3].cpu().numpy() + q_anchor


expert = unnorm(ng[:, start : start + N_ACT])[0]
_raw = item["action"][start : start + N_ACT, :3].numpy()
assert np.abs(expert - _raw).max() < 2e-2, (
    f"relative-action convention mismatch: {np.abs(expert - _raw).max()}"
)
samples = {}
with torch.no_grad():
    for r in RATIOS:
        if r in RESEED:
            torch.manual_seed(RESEED[r])
        nbk = {k: v.repeat_interleave(N_SAMPLES, 0) for k, v in nb.items()}
        ngk = ng.repeat_interleave(N_SAMPLES, 0)
        if r >= 1.0:
            x = torch.randn_like(ngk)
            pred = policy.diffusion.generate_actions(nbk, noise=x, sa_noise_ratio=1.0)
        else:
            t_sw = int(r * T_train)
            x = ns.add_noise(
                ngk,
                torch.randn_like(ngk),
                torch.full((ngk.shape[0],), t_sw - 1, dtype=torch.long, device="cuda"),
            )
            pred = policy.diffusion.generate_actions(nbk, noise=x, sa_noise_ratio=r)
        samples[r] = unnorm(pred[:, :N_ACT] if pred.shape[1] > N_ACT else pred)
# end-effector FK (finger-pad midpoint, same convention as the teaser)
cid = pb.connect(pb.DIRECT)
body = pb.loadURDF(dsv.PLANAR_URDF, useFixedBase=True, physicsClientId=cid)


def fk(q):
    dsv._pose(pb, cid, body, q)
    pl = np.array(pb.getLinkState(body, 8, computeForwardKinematics=True, physicsClientId=cid)[0])
    pr = np.array(pb.getLinkState(body, 13, computeForwardKinematics=True, physicsClientId=cid)[0])
    return ((pl + pr) / 2)[[0, 2]]


St = np.stack(
    [
        np.asarray(v, dtype=float)
        for v in pd.concat(
            [
                pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.state"])
                for f in glob.glob(
                    os.path.expanduser(f"~/.cache/huggingface/lerobot/{REPO}/data/**/*.parquet"),
                    recursive=True,
                )
            ]
        )
        .query(f"episode_index=={EPISODE}")
        .sort_values("frame_index")["observation.state"]
    ]
)[:, :3]
if SPACE == "pca":
    # same convention as fig_calibration_steps: PCA of ALL intervention states of this dataset, plane rotated so this episode's net motion is +x
    _all = pd.concat(
        [
            pd.read_parquet(f, columns=["observation.state"])
            for f in glob.glob(
                os.path.expanduser(f"~/.cache/huggingface/lerobot/{REPO}/data/**/*.parquet"), recursive=True
            )
        ]
    )
    _fit = np.stack([np.asarray(v, dtype=float)[:3] for v in _all["observation.state"]])
    _mu = _fit.mean(0)
    _u, _sv, _vt = np.linalg.svd(_fit - _mu, full_matrices=False)
    P2 = _vt[:2].T
    EXPL = _sv[:2] ** 2 / (_sv**2).sum()
    _d = (St[-1] - St[0]) @ P2
    _th = np.arctan2(_d[1], _d[0])
    _Rm = np.array([[np.cos(-_th), -np.sin(-_th)], [np.sin(-_th), np.cos(-_th)]])
    P2 = P2 @ _Rm.T
    fk = lambda q: (np.asarray(q, dtype=float) - _mu) @ P2
    AXL = (
        "PCA projection of joint space (dim 1) [rad]",
        "PCA projection of joint space (dim 2) [rad]"
        + (f"  (axis ×{Y_STRETCH:g})" if Y_STRETCH not in ("auto", 1) else ""),
    )
    print(f"dataset PCA plane explains {EXPL.sum() * 100:.0f}% of joint variance")
else:
    AXL = ("end-effector x [m]", "end-effector z [m]")
path = np.array([fk(q) for q in St])
ee_exp = np.array([fk(q) for q in expert])
ee_s = {r: np.array([[fk(q) for q in s] for s in samples[r]]) for r in RATIOS}
pb.disconnect(cid)
# ── CONFIG (style; the GUI built by build_gui.py emits this block) ──────────
CONFIG = {
    "title": "Partial denoising at one observation",
    "xlabel": "Joint-space PC1 [rad]",
    "ylabel": "Joint-space PC2 [rad]",
    "ratio_labels": ["β = 0.2", "β = 0.4", "β = 0.6", "β = 0.8", "β = 1.0"],
    "palette": "plasma",
    "palette_range": [0.1, 0.85],
    "col_ratios": ["#f66151", "#ffa348", "#f6d32d", "#2ec27e", "#1a5fb4"],
    "expert_label": "Sampled expert chunk (β = 0)",
    "col_expert": "#3d3846",
    "path_label": "Expert intervention",
    "col_path": "#c0bfbc",
    "path_style": "--",
    "show_path": True,
    "lw_ratio": 1.8,
    "lw_expert": 3,
    "lw_path": 1.8,
    "end_arrows": True,
    "arrow_size": 9,
    "step_ticks": 8,
    "tick_ms": 2.6,
    "end_labels": True,
    "end_label_sizes": [8.5, 8.5, 8.5, 8.5, 8.5],
    "end_label_offsets": [[6, -1], [3, -5], [-25, -10], [3, -3], [-17, -9]],
    "anchor_ms": 7.5,
    "expert_end_label": "β = 0.0",
    "show_expert_end_label": True,
    "expert_end_label_size": 8.5,
    "expert_end_label_offset": [3, 7],
    "annotations": [
        {
            "text": "current observation $o_t$",
            "target": "anchor",
            "xy_frac": [0.1, 0.92],
            "color": "#3d3846",
            "size": 8.5,
            "show": False,
        },
        {
            "text": "expert chunk (β = 0)",
            "target": "expert_end",
            "xy_frac": [0.62, 0.86],
            "color": "#3d3846",
            "size": 8.5,
            "show": False,
        },
        {
            "text": "β = 1: unguided policy\nheads back",
            "target": "ratio_end:1.0",
            "xy_frac": [0.06, 0.3],
            "color": "#3d3846",
            "size": 8.5,
            "show": False,
        },
    ],
    "legend_loc": "lower left",
    "show_legend": False,
    "size_title": 10.5,
    "size_label": 9.5,
    "size_tick": 8.5,
    "size_legend": 7.5,
    "grid_alpha": 0.5,
    "fig_size": [5, 3.3],
    "max_xticks": 7,
    "max_yticks": 7,
}
# ─────────────────────────────────────────────────────────────────────────────
if len(CONFIG["ratio_labels"]) != len(RATIOS):  # e.g. RATIOS=0.1,...,1.0: regenerate per-ratio lists
    CONFIG["ratio_labels"] = [f"β = {r:.1f}" for r in RATIOS]
    CONFIG["end_label_sizes"] = [CONFIG["end_label_sizes"][0]] * len(RATIOS)
    CONFIG["end_label_offsets"] = [[6, 0]] * len(RATIOS)
LABEL_EVERY = int(os.environ.get("LABEL_EVERY", 1))
plt.rcParams.update({"font.size": CONFIG["size_tick"], "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=tuple(CONFIG["fig_size"]))


def ratio_color(i):
    if CONFIG["palette"] == "custom":
        return CONFIG["col_ratios"][i % len(CONFIG["col_ratios"])]
    lo_, hi_ = CONFIG["palette_range"]
    return matplotlib.colors.to_hex(
        plt.get_cmap(CONFIG["palette"])(lo_ + (hi_ - lo_) * i / max(len(RATIOS) - 1, 1))
    )


def arrow_at_end(pts, col, lw):
    if not CONFIG["end_arrows"] or len(pts) < 3:
        return
    ax.annotate(
        "",
        xy=pts[-1],
        xytext=pts[-3],
        arrowprops=dict(
            arrowstyle="-|>",
            color=col,
            lw=lw * 0.8,
            mutation_scale=CONFIG["arrow_size"],
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=6,
    )


def ticks_along(pts, col):
    n = CONFIG["step_ticks"]
    if n:
        ax.plot(
            pts[::n, 0], pts[::n, 1], "o", ms=CONFIG["tick_ms"], color=col, mec="white", mew=0.5, zorder=5
        )


if CONFIG["show_path"]:
    ax.plot(
        path[:, 0],
        path[:, 1],
        CONFIG["path_style"],
        lw=CONFIG["lw_path"],
        color=CONFIG["col_path"],
        zorder=1,
        label=CONFIG["path_label"],
    )
ends = {}
for i, r in enumerate(RATIOS):
    col = ratio_color(i)
    for k, s in enumerate(ee_s[r]):
        ax.plot(
            s[:, 0],
            s[:, 1],
            "-",
            lw=CONFIG["lw_ratio"],
            color=col,
            alpha=0.95,
            zorder=3,
            solid_capstyle="round",
            label=CONFIG["ratio_labels"][i] if k == 0 else None,
        )
        arrow_at_end(s, col, CONFIG["lw_ratio"])
        ticks_along(s, col)
        if k == 0:
            ends[f"ratio_end:{r}"] = s[-1]
            if CONFIG["end_labels"] and (i % LABEL_EVERY == 0 or i == len(RATIOS) - 1):
                ax.annotate(
                    CONFIG["ratio_labels"][i],
                    xy=s[-1],
                    xytext=CONFIG["end_label_offsets"][i],
                    textcoords="offset points",
                    fontsize=CONFIG["end_label_sizes"][i],
                    color=col,
                    va="center",
                    zorder=7,
                )
ax.plot(
    ee_exp[:, 0],
    ee_exp[:, 1],
    "-",
    lw=CONFIG["lw_expert"],
    color=CONFIG["col_expert"],
    zorder=4,
    solid_capstyle="round",
    label=CONFIG["expert_label"],
)
arrow_at_end(ee_exp, CONFIG["col_expert"], CONFIG["lw_expert"])
ticks_along(ee_exp, CONFIG["col_expert"])
ax.plot(
    [ee_exp[0, 0]],
    [ee_exp[0, 1]],
    "o",
    ms=CONFIG["anchor_ms"],
    color=CONFIG["col_expert"],
    mec="white",
    mew=1.2,
    zorder=8,
)
ends["anchor"] = ee_exp[0]
ends["expert_end"] = ee_exp[-1]
if CONFIG["show_expert_end_label"]:
    ax.annotate(
        CONFIG["expert_end_label"],
        xy=ee_exp[-1],
        xytext=CONFIG["expert_end_label_offset"],
        textcoords="offset points",
        fontsize=CONFIG["expert_end_label_size"],
        color=CONFIG["col_expert"],
        va="center",
        zorder=7,
    )
ax.set_aspect(Y_STRETCH if SPACE == "pca" else "equal")  # "auto" -> independent axis scales
ax.set_xlabel(CONFIG["xlabel"], fontsize=CONFIG["size_label"])
ax.set_ylabel(CONFIG["ylabel"], fontsize=CONFIG["size_label"])
ax.set_title(CONFIG["title"], fontsize=CONFIG["size_title"])
if CONFIG["show_legend"]:  # legend order: intervention path, sampled expert chunk, then the blend ratios
    hs, ls = ax.get_legend_handles_labels()
    order = [ls.index(CONFIG["expert_label"])] + [i for i, l in enumerate(ls) if l != CONFIG["expert_label"]]
    if CONFIG["show_path"]:
        order.remove(ls.index(CONFIG["path_label"]))
        order.insert(0, ls.index(CONFIG["path_label"]))
    ax.legend(
        [hs[i] for i in order],
        [ls[i] for i in order],
        fontsize=CONFIG["size_legend"],
        frameon=False,
        loc=CONFIG["legend_loc"],
    )
ax.spines[["top", "right"]].set_visible(False)
ax.grid(color="#dddddd", lw=0.6, alpha=CONFIG["grid_alpha"])
ax.set_axisbelow(True)
from matplotlib.ticker import MaxNLocator

ax.xaxis.set_major_locator(MaxNLocator(CONFIG["max_xticks"]))
ax.yaxis.set_major_locator(MaxNLocator(CONFIG["max_yticks"]))
allp = np.concatenate([ee_exp] + [s.reshape(-1, 2) for s in ee_s.values()])
pad = 0.10 * np.ptp(allp, axis=0).max() + 0.02
ax.set_xlim(allp[:, 0].min() - pad, allp[:, 0].max() + pad)
ax.set_ylim(allp[:, 1].min() - pad, allp[:, 1].max() + pad)
for an in CONFIG["annotations"]:
    tgt = ends.get(an["target"])
    if tgt is None or not an.get("show", True):
        continue
    ax.annotate(
        an["text"],
        xy=tgt,
        xytext=an["xy_frac"],
        textcoords="axes fraction",
        fontsize=an["size"],
        color=an["color"],
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="-", color=an["color"], lw=0.8, shrinkA=2, shrinkB=4, alpha=0.8),
        zorder=9,
    )
fig.tight_layout()
fig.savefig(OUT, dpi=150)
# data dump for the GUI (build_gui.py)
import json

bb = ax.get_position()
Wpx, Hpx = fig.get_size_inches() * 100
json.dump(
    {
        "config": CONFIG,
        "fig_px": [Wpx, Hpx],
        "ratios": list(RATIOS),
        "axes": {
            "x0": bb.x0 * Wpx,
            "x1": bb.x1 * Wpx,
            "y0": (1 - bb.y1) * Hpx,
            "y1": (1 - bb.y0) * Hpx,
            "xlim": list(ax.get_xlim()),
            "ylim": list(ax.get_ylim()),
        },
        "path": path.tolist(),
        "expert": ee_exp.tolist(),
        "samples": [ee_s[r].tolist() for r in RATIOS],
        "episode": EPISODE,
        "anchor": ANCHOR,
        "ratio_colors": [ratio_color(i) for i in range(len(RATIOS))],
    },
    open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_data.json"), "w"),
)
for r in RATIOS:
    d = np.linalg.norm(ee_s[r] - ee_exp[None], axis=2).mean() * (100 if SPACE == "ee" else 1)
    print(
        f"r={r:.1f}: mean deviation from expert chunk {d:.2f} {'cm' if SPACE == 'ee' else 'rad (PCA plane)'}"
    )
print("wrote", OUT)
