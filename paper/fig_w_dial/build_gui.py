"""Build gui.html (interactive styler for fig_w_dial) from gui_template.html +
w_dial_data.json. Open gui.html in a browser; "Copy CONFIG" -> paste into plot.py.
"""

import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "w_dial_data.json")))
for c in d["cells"]:
    if c["rms"] is None or (isinstance(c["rms"], float) and math.isnan(c["rms"])):
        c["rms"] = None
for k in ("W", "R", "W_lo", "W_hi"):
    d[k] = float(d[k])
d["config"]["round_colors"] = {str(k): v for k, v in d["config"]["round_colors"].items()}
s = json.dumps(d).replace("</", "<\\/")
html = open(os.path.join(HERE, "gui_template.html")).read().replace("__DATA__", s)
# standalone page needs a document skeleton (the artifact host adds one)
html = '<!doctype html>\n<html><head><meta charset="utf-8">\n' + html.replace("<title>", "<title>", 1)
html = html.replace("<h1>", "</head><body>\n<h1>", 1) + "\n</body></html>\n"
open(os.path.join(HERE, "gui.html"), "w").write(html)
print("wrote", os.path.join(HERE, "gui.html"))
