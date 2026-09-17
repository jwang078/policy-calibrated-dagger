"""Pre-compute an 84x84 copy of the lever base dataset (LeRobot v3, images as
PNG bytes inside the data parquet files): every image column is decoded,
stretch-resized to RESxRES (LANCZOS), re-encoded as PNG. meta/ is copied with
the image feature shapes rewritten. Output repo: <src>_r<RES>.
"""

import glob
import io
import json
import os
import shutil
from multiprocessing import Pool

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

RES = int(os.environ.get("RES", 84))
NPROC = int(os.environ.get("NPROC", 6))
SRC = os.path.expanduser("~/.cache/huggingface/lerobot/JennyWWW/splatsim_approach_lever_13_smooth")
DST = SRC + f"_r{RES}"
IMG_COLS = None


def shrink_bytes(b):
    im = Image.open(io.BytesIO(b)).convert("RGB").resize((RES, RES), Image.LANCZOS)
    out = io.BytesIO()
    im.save(out, format="PNG", compress_level=1)
    return out.getvalue()


def do_file(rel):
    src, dst = os.path.join(SRC, rel), os.path.join(DST, rel)
    if os.path.exists(dst):
        return rel, 0
    t = pq.read_table(src)
    cols = {}
    for name in t.column_names:
        col = t.column(name)
        if name.startswith("observation.images."):
            recs = col.to_pylist()
            cols[name] = pa.array(
                [{"bytes": shrink_bytes(r["bytes"]), "path": r["path"]} for r in recs], type=col.type
            )
        else:
            cols[name] = col
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    pq.write_table(pa.table(cols, schema=t.schema), dst + ".tmp")
    os.replace(dst + ".tmp", dst)
    return rel, t.num_rows


if __name__ == "__main__":
    os.makedirs(DST, exist_ok=True)
    if not os.path.exists(os.path.join(DST, "meta")):
        shutil.copytree(os.path.join(SRC, "meta"), os.path.join(DST, "meta"))
        info = json.load(open(os.path.join(DST, "meta/info.json")))
        for k, v in info["features"].items():
            if k.startswith("observation.images."):
                v["shape"] = [3, RES, RES]
        json.dump(info, open(os.path.join(DST, "meta/info.json"), "w"), indent=4)
    files = sorted(os.path.relpath(f, SRC) for f in glob.glob(SRC + "/data/chunk-*/*.parquet"))
    done = 0
    with Pool(NPROC) as p:
        for i, (rel, n) in enumerate(p.imap_unordered(do_file, files)):
            done += n
            if i % 25 == 0 or i == len(files) - 1:
                print(f"{i + 1}/{len(files)} files, {done} frames", flush=True)
    print("DONE", DST)
