"""Re-encode every video of a LeRobot v3 video dataset to RESxRES in place
(same codec settings as recorded: av1 yuv420p crf30 preset12 g2), update the
feature shapes in meta/info.json, validate by loading an item. Originals are
kept in <repo>/videos_224 (first run only).
usage: python shrink_video_dataset_r84.py <repo_short> [RES]
"""

import glob
import json
import os
import shutil
import subprocess
import sys

short = sys.argv[1]
RES = int(sys.argv[2]) if len(sys.argv) > 2 else 84
ROOT = os.path.expanduser(f"~/.cache/huggingface/lerobot/JennyWWW/{short}")
FF = os.path.expanduser("~/miniforge3/envs/splatsim/bin/ffmpeg")
info = json.load(open(f"{ROOT}/meta/info.json"))
vids = {k: v for k, v in info["features"].items() if v["dtype"] == "video"}
if all(v["shape"][1] == RES for v in vids.values()):
    print("already", RES)
    sys.exit(0)
bak = f"{ROOT}/videos_224"
if not os.path.isdir(bak):
    shutil.copytree(f"{ROOT}/videos", bak)
for f in sorted(glob.glob(f"{ROOT}/videos/**/*.mp4", recursive=True)):
    tmp = f + ".tmp.mp4"
    cmd = [
        FF,
        "-y",
        "-loglevel",
        "error",
        "-i",
        os.path.join(bak, os.path.relpath(f, f"{ROOT}/videos")),
        "-vf",
        f"scale={RES}:{RES}:flags=lanczos",
        "-c:v",
        "libsvtav1",
        "-crf",
        "30",
        "-preset",
        "12",
        "-g",
        "2",
        "-pix_fmt",
        "yuv420p",
        "-an",
        tmp,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        print("ffmpeg failed", f, r.stderr[-400:])
        sys.exit(1)
    os.replace(tmp, f)
    print("re-encoded", os.path.relpath(f, ROOT), flush=True)
for k, v in vids.items():
    v["shape"] = [3, RES, RES]
    v.setdefault("info", {})
    v["info"]["video.height"] = RES
    v["info"]["video.width"] = RES
json.dump(info, open(f"{ROOT}/meta/info.json", "w"), indent=4)
sys.path.insert(0, "/home/jennyw2/code/lerobot/src")
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset(
    f"JennyWWW/{short}", delta_timestamps={"observation.images.base_rgb_stretch": [-1 / 30, 0]}
)
it = ds[len(ds) // 2]
print("validated item:", tuple(it["observation.images.base_rgb_stretch"].shape), "frames", len(ds))
assert tuple(it["observation.images.base_rgb_stretch"].shape) == (2, 3, RES, RES)
print("DONE", ROOT)
