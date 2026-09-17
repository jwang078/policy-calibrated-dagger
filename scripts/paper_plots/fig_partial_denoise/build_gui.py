"""Build gui.html (interactive styler) from gui_template.html + fig_data.json
(written by plot.py). Open gui.html in a browser; Copy CONFIG -> paste over the
CONFIG block in plot.py and re-run for the matplotlib PNG.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "fig_data.json")))


def rnd(x):
    if isinstance(x, float):
        return round(x, 4)
    if isinstance(x, list):
        return [rnd(v) for v in x]
    if isinstance(x, dict):
        return {k: rnd(v) for k, v in x.items()}
    return x


s = json.dumps(rnd(d), separators=(",", ":")).replace("</", "<\\/")
html = (
    open(os.path.join(HERE, "gui_template.html"))
    .read()
    .replace("__DATA__", s)
    .replace("__EP__", str(d["episode"]))
    .replace("__T__", str(d["anchor"]))
)
open(os.path.join(HERE, "gui_artifact.html"), "w").write(
    html
)  # body-only copy for the claude.ai artifact host
html = (
    '<!doctype html>\n<html><head><meta charset="utf-8">\n'
    + html.replace("<h1>", "</head><body>\n<h1>", 1)
    + "\n</body></html>\n"
)
open(os.path.join(HERE, "gui.html"), "w").write(html)
print("wrote", os.path.join(HERE, "gui.html"), len(html) // 1024, "KB")
