"""Build gui.html / gui_artifact.html for one rendered tag: inlines geom_<tag>.json,
the background PNG (data URI) and the current CONFIG (from plot.py's dump).
usage: python build_gui.py [TAG]
"""

import base64
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TAG = sys.argv[1] if len(sys.argv) > 1 else "dag1_ep9_t4"
g = json.load(open(f"{HERE}/geom_{TAG}.json"))
cfg = json.load(open(f"{HERE}/fig_config_{TAG}.json"))["config"]


def rnd(x):
    if isinstance(x, float):
        return round(x, 3)
    if isinstance(x, list):
        return [rnd(v) for v in x]
    if isinstance(x, dict):
        return {k: rnd(v) for k, v in x.items()}
    return x


data = {
    "W": g["W"],
    "H": g["H"],
    "px_path": rnd(g["px_path"]),
    "px_anchor": rnd(g["px_anchor"]),
    "band": rnd(g["band"]),
    "arcs": [
        {"t0": a["t0"], "draw": a["draw"], "mahal": round(a["mahal"], 3), "px": rnd(a["px"])}
        for a in g["arcs"]
    ],
    "bg": "data:image/png;base64," + base64.b64encode(open(f"{HERE}/bg_{TAG}.png", "rb").read()).decode(),
    "config": cfg,
}
s = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
html = open(f"{HERE}/gui_template.html").read().replace("__DATA__", s).replace("__TAG__", TAG)
open(f"{HERE}/gui_artifact.html", "w").write(html)
full = (
    '<!doctype html>\n<html><head><meta charset="utf-8">\n'
    + html.replace("<h1>", "</head><body>\n<h1>", 1)
    + "\n</body></html>\n"
)
open(f"{HERE}/gui.html", "w").write(full)
print("wrote gui.html", len(full) // 1024, "KB")
