"""Rebuild the 84px lever base dataset with ONLY the *_stretch image columns
(letterbox columns dropped from every parquet file and from meta/info.json +
meta/stats.json). Built in a temp dir, validated by loading an item through
LeRobotDataset, then swapped in: <r84> -> <r84>_withletterbox, tmp -> <r84>.
"""

import glob
import json
import os
import shutil
import sys
from multiprocessing import Pool

import pyarrow.parquet as pq

SRC = os.path.expanduser("~/.cache/huggingface/lerobot/JennyWWW/splatsim_approach_lever_13_smooth_r84")
TMP = SRC + "_tmp"
OLD = SRC + "_withletterbox"
DROP = lambda name: name.startswith("observation.images.") and name.endswith("_letterbox")


def do_file(rel):
    t = pq.read_table(os.path.join(SRC, rel))
    keep = [c for c in t.column_names if not DROP(c)]
    dst = os.path.join(TMP, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    pq.write_table(t.select(keep), dst)
    return t.num_rows


if __name__ == "__main__":
    if os.path.exists(TMP):
        shutil.rmtree(TMP)
    os.makedirs(TMP)
    shutil.copytree(os.path.join(SRC, "meta"), os.path.join(TMP, "meta"))
    info = json.load(open(os.path.join(TMP, "meta/info.json")))
    info["features"] = {k: v for k, v in info["features"].items() if not DROP(k)}
    json.dump(info, open(os.path.join(TMP, "meta/info.json"), "w"), indent=4)
    sp = os.path.join(TMP, "meta/stats.json")
    if os.path.exists(sp):
        st = json.load(open(sp))
        json.dump({k: v for k, v in st.items() if not DROP(k)}, open(sp, "w"), indent=4)
    files = sorted(os.path.relpath(f, SRC) for f in glob.glob(SRC + "/data/chunk-*/*.parquet"))
    with Pool(6) as p:
        n = sum(p.map(do_file, files))
    print(f"rewrote {len(files)} files, {n} frames", flush=True)
    # validate before swapping
    sys.path.insert(0, "/home/jennyw2/code/lerobot/src")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("JennyWWW/splatsim_approach_lever_13_smooth_r84_tmp")
    it = ds[1000]
    keys = [k for k in it if k.startswith("observation.images.")]
    assert keys == ["observation.images.base_rgb_stretch", "observation.images.wrist_rgb_stretch"], keys
    assert tuple(it[keys[0]].shape) == (3, 84, 84), it[keys[0]].shape
    assert len(ds) == n, (len(ds), n)
    os.rename(SRC, OLD)
    os.rename(TMP, SRC)
    print("SWAPPED: stripped dataset at", SRC, "| original kept at", OLD)
