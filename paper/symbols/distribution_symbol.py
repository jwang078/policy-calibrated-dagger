"""A soft concentric-ellipse 'distribution' glyph on a transparent background (no label; put your own
text in the middle). Writes distribution.png (2400 px wide), distribution.svg and distribution.pdf.
Knobs: COLOR, RADII (outer->inner, fraction of the outer ellipse), ALPHAS (per ring, stacked), ASPECT."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

COLOR = os.environ.get("COLOR", "#3d55cc")
RADII = [float(x) for x in os.environ.get("RADII", "1.0,0.76,0.52,0.28").split(",")]
ALPHAS = [float(x) for x in os.environ.get("ALPHAS", "0.045,0.075,0.11,0.16").split(",")]
ASPECT = float(os.environ.get("ASPECT", "0.62"))  # height / width of the ellipses
OUT = os.environ.get("OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "distribution"))

fig = plt.figure(figsize=(6, 6 * ASPECT), dpi=400)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(-1.02, 1.02)
ax.set_ylim(-1.02 * ASPECT, 1.02 * ASPECT)
ax.set_aspect("equal")
ax.axis("off")
for r, a in zip(RADII, ALPHAS):
    ax.add_patch(Ellipse((0, 0), 2 * r, 2 * r * ASPECT, facecolor=COLOR, edgecolor="none", alpha=a, antialiased=True))
for ext in ("png", "svg", "pdf"):
    fig.savefig(f"{OUT}.{ext}", transparent=True, dpi=400)
print("wrote", OUT + ".{png,svg,pdf}")
