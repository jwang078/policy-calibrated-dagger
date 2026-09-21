"""Open-loop discrepancy (Sigma-hat) for the LEVER image policy: pure-policy
chunk draws (K=4, stride 2) at every anchor of an intervention dataset, minus
the recorded expert chunk -> per-anchor deltas (6 arm joints, rad).

usage: python measure_sigma_lever.py TAG CKPT REPO_SHORT
  -> $S/analysis/sigma_deltas_lever_{TAG}.npz  (same layout as planar:
     ep{e}_t (n_starts,), ep{e}_d (n_starts, KD, N_ACT, 6), ep{e}_med)
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/home/jennyw2/code/lerobot/src")
from safetensors.torch import load_file

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.processor import PolicyProcessorPipeline

S = os.environ.get("TABLES_REPRO_DIR", "/home/jennyw2/code/lerobot/my_scripts/paper_plots/tables_repro")
TAG, CK, SHORT = sys.argv[1], sys.argv[2], sys.argv[3]
REPO = f"JennyWWW/{SHORT}"
OUT = f"{S}/analysis/sigma_deltas_lever_{TAG}.npz"
KD, STRIDE, BATCH, NARM, fps = 4, 2, 8, 6, 30
if os.path.exists(OUT):
    print("exists", OUT)
    sys.exit(0)

policy = DiffusionPolicy.from_pretrained(CK).cuda().eval()
pre = PolicyProcessorPipeline.from_pretrained(CK, config_filename="policy_preprocessor.json")
stt = load_file(sorted(glob.glob(CK + "/policy_preprocessor_step_*_normalizer_processor.safetensors"))[0])
lo = stt["action.min"].cuda()
rng = (stt["action.max"].cuda() - lo).clamp(min=1e-8)
cfg = policy.config
N_ACT, start = cfg.n_action_steps, cfg.n_obs_steps - 1
IMG_KEYS = ["observation.images.base_rgb_stretch", "observation.images.wrist_rgb_stretch"]
obs_ts = [i / fps for i in range(1 - cfg.n_obs_steps, 1)]
dts = {
    "observation.state": obs_ts,
    "action": [i / fps for i in range(1 - cfg.n_obs_steps, 1 - cfg.n_obs_steps + cfg.horizon)],
}
for k in IMG_KEYS:
    dts[k] = obs_ts
ds = LeRobotDataset(REPO, delta_timestamps=dts)
epi = np.array(ds.hf_dataset["episode_index"])
fri = np.array(ds.hf_dataset["frame_index"])
root = os.path.expanduser(f"~/.cache/huggingface/lerobot/{REPO}")
di = pd.concat(
    [
        pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.state"])
        for f in glob.glob(root + "/data/**/*.parquet", recursive=True)
    ]
)
arrs = {}
with torch.no_grad():
    for EP in sorted(np.unique(epi)):
        D = np.stack(di[di.episode_index == EP].sort_values("frame_index")["observation.state"].to_numpy())[
            :, :NARM
        ]
        st = np.linalg.norm(np.diff(D, axis=0), axis=1)
        st = st[st > 1e-6]
        med = float(np.median(st)) if len(st) else 1e-3
        T = len(D)
        idxs = np.where(epi == EP)[0]
        idxs = idxs[np.argsort(fri[idxs])]
        starts = list(range(0, T, STRIDE))
        dl = []
        for b0 in range(0, len(starts), BATCH):
            frames = starts[b0 : b0 + BATCH]
            items = [ds[int(idxs[min(f, len(idxs) - 1)])] for f in frames]
            raw = {
                k: torch.stack([it[k] for it in items]) for k in ["observation.state", "action"] + IMG_KEYS
            }
            proc = pre(raw)
            nb = {"observation.state": proc["observation.state"].cuda().repeat_interleave(KD, dim=0)}
            nb["observation.images"] = torch.stack(
                [proc[k].cuda().repeat_interleave(KD, dim=0) for k in cfg.image_features], dim=-4
            )
            ng = proc["action"].cuda().repeat_interleave(KD, dim=0)
            x = torch.randn_like(ng)
            pred = policy.diffusion.generate_actions(nb, noise=x, sa_noise_ratio=1.0)
            relP = ((pred + 1) / 2 * rng + lo)[:, :, :NARM]
            relE = ((ng[:, start : start + N_ACT] + 1) / 2 * rng + lo)[:, :, :NARM]
            dl.append((relP - relE).view(len(frames), KD, N_ACT, NARM).cpu().numpy().astype(np.float32))
        arrs[f"ep{EP}_t"] = np.array(starts, dtype=np.int32)
        arrs[f"ep{EP}_d"] = np.concatenate(dl, axis=0)
        arrs[f"ep{EP}_med"] = np.float32(med)
        print(
            f"ep {EP}: T={T} anchors={len(starts)} med={med:.4f} "
            f"rms/med={np.sqrt((arrs[f'ep{EP}_d'].reshape(-1, NARM) ** 2).sum(1).mean()) / med:.2f}",
            flush=True,
        )
np.savez_compressed(OUT, **arrs)
print("saved", OUT)
