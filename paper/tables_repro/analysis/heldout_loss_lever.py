"""Diffusion training loss of lever checkpoints on (a) the round-1 training
interventions (dag1), (b) held-out interventions never seen by round-1 arms
(dag2), (c) a fixed slice of the base demos. Same sampled frames, same
diffusion timesteps/noise (seeded) for every checkpoint -> overfitting check.
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/jennyw2/code/lerobot/src")
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.processor import PolicyProcessorPipeline

TR = "/home/jennyw2/code/lerobot/outputs/training"
B0 = f"{TR}/diffusion_approach_lever_13_smooth_delta_basewristng"
CKPTS = {
    "bc_075k": f"{B0}/checkpoints/075000/pretrained_model",
    "hg1_lineage_095k": f"{B0}_d100_03dagcap_cam_ft_dag1/checkpoints/095000/pretrained_model",
    "cal1_lineage_095k": f"{TR}/lever_cam_calib/q1_dnpool/checkpoints/last/pretrained_model",
    "hg1_frombase_175k": f"{TR}/lever_cam_frombase/q1/checkpoints/last/pretrained_model",
}
SETS = {
    "dag1_train": "JennyWWW/lever_d100_03dagcap_cam_diff_r_dag1",
    "dag2_heldout": "JennyWWW/lever_d100_03dagcap_cam_diff_r_dag2",
    "base_demos": "JennyWWW/splatsim_approach_lever_13_smooth",
}
N, BS, fps = 1024, 32, 30
IMG_KEYS = ["observation.images.base_rgb_stretch", "observation.images.wrist_rgb_stretch"]
res = {}
for name, ck in CKPTS.items():
    if not os.path.isdir(ck):
        print("skip", name)
        continue
    policy = DiffusionPolicy.from_pretrained(ck).cuda().eval()
    pre = PolicyProcessorPipeline.from_pretrained(ck, config_filename="policy_preprocessor.json")
    cfg = policy.config
    obs_ts = [i / fps for i in range(1 - cfg.n_obs_steps, 1)]
    dts = {
        "observation.state": obs_ts,
        "action": [i / fps for i in range(1 - cfg.n_obs_steps, 1 - cfg.n_obs_steps + cfg.horizon)],
    }
    for k in IMG_KEYS:
        dts[k] = obs_ts
    res[name] = {}
    for sname, repo in SETS.items():
        ds = LeRobotDataset(repo, delta_timestamps=dts)
        idx = np.random.RandomState(0).choice(len(ds), size=min(N, len(ds)), replace=False)
        torch.manual_seed(0)
        losses = []
        with torch.no_grad():
            for b in range(0, len(idx), BS):
                items = [ds[int(i)] for i in idx[b : b + BS]]
                raw = {
                    k: torch.stack([it[k] for it in items])
                    for k in ["observation.state", "action"] + IMG_KEYS
                }
                proc = pre(raw)
                batch = {
                    k: proc[k].cuda() for k in ["observation.state", "action"] + list(cfg.image_features)
                }
                batch["action_is_pad"] = torch.stack([it["action_is_pad"] for it in items]).cuda()
                loss, _ = policy.forward(batch)
                losses.append(float(loss))
        res[name][sname] = float(np.mean(losses))
        print(f"{name:20s} {sname:14s} loss={res[name][sname]:.5f}  (n={len(idx)})", flush=True)
    del policy
    torch.cuda.empty_cache()
json.dump(res, open(os.path.dirname(os.path.abspath(__file__)) + "/heldout_loss_lever.json", "w"), indent=1)
