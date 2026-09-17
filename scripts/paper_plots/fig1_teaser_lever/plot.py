#!/usr/bin/env python
"""Figure-1 teaser (lever): draw the expert intervention path, the calibrated
noise band and a few real recovery arcs on the photoreal render produced by
render.py (bg_<tag>.png + geom_<tag>.json). Same conventions as the planar
teaser (fig1_teaser/plot.py): band = stroked union of per-point k-sigma pixel
ellipses, arcs = real dart_relabel servo labels trimmed where they meet the path.

usage: python plot.py [TAG]        (TAG default dag1_ep9_t4)  ->  fig1_teaser_lever_<TAG>.png
"""

import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path

HERE = os.path.dirname(os.path.abspath(__file__))
TAG = sys.argv[1] if len(sys.argv) > 1 else "dag1_ep9_t4"
# ── CONFIG ───────────────────────────────────────────────────────────────────
CONFIG = {
    "title": [
        ["Policy-calibrated noise", "#62a0ea"],
        [" near ", "#333333"],
        ["recorded expert data", "#1a5fb4"],
        ["\nproduces ", "#333333"],
        ["targeted recovery trajectories", "#ff7800"],
    ],
    "path_t_range": [0, 119],
    "col_path": "#1a5fb4",
    "lw_path": 3.2,
    "arc_picks": [[4, 1], [4, 5], [64, 2]],
    "col_arc": "#ff7800",
    "lw_arc": 2.6,
    "arc_trim_px": 3,
    "arc_min_travel_px": 25,
    "arc_start_ms": 4.5,
    "col_band": "#3584e4",
    "band_strokes": [[3, 0.16], [2, 0.26], [1, 0.4]],
    "show_anchor": True,
    "col_anchor": "#1a5fb4",
    "anchor_ms": 7,
    "crop": [0, 0, 1, 1],
    "title_size": 13,
    "fig_dpi": 160,
    "fig_width": 5.2,
}
# ─────────────────────────────────────────────────────────────────────────────
G = json.load(open(f"{HERE}/geom_{TAG}.json"))
img = plt.imread(f"{HERE}/bg_{TAG}.png")
H, W = img.shape[:2]
PF = np.array(G["px_path"])
anchor = np.array(G["px_anchor"])
ta, tb = CONFIG["path_t_range"]
tb = min(tb, len(PF) - 1)
P = PF[ta : tb + 1]
BAND = [b for b in G["band"] if ta <= b["t"] <= tb]


def ellipse(mu, cov, k, n=40):
    w, v = np.linalg.eigh(np.asarray(cov))
    w = np.clip(w, 1e-6, None)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    circ = np.stack([np.cos(th), np.sin(th)], 1) * np.sqrt(w) * k
    return np.asarray(mu) + circ @ v.T


def union_patch(polys, color, alpha, z):
    verts, codes = [], []
    for poly in polys:
        verts += list(poly) + [poly[0]]
        codes += [Path.MOVETO] + [Path.LINETO] * (len(poly) - 1) + [Path.CLOSEPOLY]
    return PathPatch(Path(verts, codes), facecolor=color, edgecolor="none", alpha=alpha, zorder=z)


def trim(arc):  # keep the arc until it first comes within arc_trim_px of the path (after leaving its start)
    arc = np.asarray(arc)
    d = np.min(np.linalg.norm(arc[:, None, :] - PF[None, :, :], axis=2), axis=1)
    moved = np.linalg.norm(arc - arc[0], axis=1) > CONFIG["arc_min_travel_px"]
    hit = np.where(moved & (d < CONFIG["arc_trim_px"]))[0]
    return arc[: hit[0] + 1] if len(hit) else arc


x0, y0, x1, y1 = CONFIG["crop"]
X0, Y0, X1, Y1 = int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H)
fig_w = CONFIG["fig_width"]
fig_h = fig_w * (Y1 - Y0) / (X1 - X0) + 0.75
fig = plt.figure(figsize=(fig_w, fig_h), dpi=CONFIG["fig_dpi"])
ax = fig.add_axes([0, 0, 1, (fig_h - 0.75) / fig_h])
ax.imshow(img, zorder=0)
ax.set_xlim(X0, X1)
ax.set_ylim(Y1, Y0)
ax.axis("off")
for k, a in CONFIG["band_strokes"]:
    ax.add_patch(union_patch([ellipse(b["mu"], b["cov"], k) for b in BAND], CONFIG["col_band"], a, 2))
ax.plot(
    P[:, 0], P[:, 1], "-", color=CONFIG["col_path"], lw=CONFIG["lw_path"], solid_capstyle="round", zorder=4
)
by_key = {(a.get("t0", G["T0"]), a.get("draw", i)): a for i, a in enumerate(G["arcs"])}
arcs = [by_key[tuple(k)] for k in CONFIG["arc_picks"] if tuple(k) in by_key]
for a in arcs:
    t = trim(a["px"])
    ax.plot(
        t[:, 0], t[:, 1], "-", color=CONFIG["col_arc"], lw=CONFIG["lw_arc"], solid_capstyle="round", zorder=5
    )
    ax.plot([t[0, 0]], [t[0, 1]], "o", ms=CONFIG["arc_start_ms"], color=CONFIG["col_arc"], zorder=6)
if CONFIG["show_anchor"]:
    ax.plot(
        [anchor[0]],
        [anchor[1]],
        "o",
        ms=CONFIG["anchor_ms"],
        color=CONFIG["col_anchor"],
        mec="white",
        mew=1.2,
        zorder=7,
    )
# colored multi-part title (two lines, centered)
lines = [[]]
for txt, col in CONFIG["title"]:
    parts = txt.split("\n")
    for i, part in enumerate(parts):
        if i:
            lines.append([])
        if part:
            lines[-1].append((part, col))

for li, line in enumerate(lines):
    y = 1 - (0.3 + 0.42 * li) * 0.75 / fig_h
    r = fig.canvas.get_renderer()
    widths = []
    for txt, col in line:
        tt = fig.text(0, 0, txt, fontsize=CONFIG["title_size"], color=col)
        widths.append(tt.get_window_extent(r).width / fig.dpi / fig_w)
        tt.remove()
    x = 0.5 - sum(widths) / 2
    for (txt, col), w in zip(line, widths):
        fig.text(x, y, txt, fontsize=CONFIG["title_size"], color=col, ha="left", va="center")
        x += w
out = f"{HERE}/fig1_teaser_lever_{TAG}.png"
fig.savefig(out, dpi=CONFIG["fig_dpi"])
json.dump({"config": CONFIG, "tag": TAG}, open(f"{HERE}/fig_config_{TAG}.json", "w"))
print(
    "wrote",
    out,
    "| arcs drawn:",
    len(arcs),
    "band points:",
    len(G["band"]),
    "| sigma (med):",
    np.round(G["sigma_med"], 2).tolist(),
)
