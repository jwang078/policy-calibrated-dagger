"""Open-loop discrepancy measurement, lineage-correct, MULTI-RATIO, with raw
delta persistence.

pass1: r=0.5 for rounds 1-5 -> saves deltas npz per round AND builds the
       v5 MIXTURE schedule (per-chunk-age v4 components, no moment collapse)
       -> $S/noise_schedule_dnak2_b5_mix.json
pass2: r=0.25 and r=0.75 for rounds 1-5 -> saves deltas npz, then writes
       openloop_multiratio_summary.json + error-vs-sigma plot.

Noise-level conversion (the piece we skipped at r=0.5 because it was ~x1):
add_noise gives x = sqrt(abar)*x0 + sqrt(1-abar)*eps, so the equivalent
x0-space noise sigma is sqrt((1-abar)/abar), taken from the model's own
alphas_cumprod at t_sw = int(r * num_train_timesteps).
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
CKPT = "/home/jennyw2/code/lerobot/outputs/training/diffusion_planar_3joint_12_delta_stateng/checkpoints/last/pretrained_model"
LINEAGE = "/home/jennyw2/code/lerobot/outputs/training/diffusion_planar_3joint_12_delta_stateng_03dag_ft_dag{K}/checkpoints/last/pretrained_model"
FLOOR, CAP = 0.5**2, 3 * 8**2
fps = 30
TAG = sys.argv[1]  # s1 | s2
POLICY_CK = sys.argv[2]  # pretrained_model dir of the K=2 grid baseline
LINEAGE_TAG = sys.argv[3]  # 05dag | 06dag


def load_policy(rd):
    ck = POLICY_CK
    pol = DiffusionPolicy.from_pretrained(ck).cuda().eval()
    pr = PolicyProcessorPipeline.from_pretrained(ck, config_filename="policy_preprocessor.json")
    stt = load_file(ck + "/policy_preprocessor_step_5_normalizer_processor.safetensors")
    lo = stt["action.min"].cuda()
    rng = (stt["action.max"].cuda() - lo).clamp(min=1e-8)
    return pol, pr, lo, rng


def sample_round(RD, RATIO, KD, STRIDE):
    """Returns (deltas: {ep: {t: (KD,N_ACT,3)}}, meds: {ep: med}, sigma_eff_norm, rng3)."""
    REPO = f"JennyWWW/planar_12_{LINEAGE_TAG}_diff_r_dag{RD}"
    policy, pre, lo, rng = load_policy(RD)
    cfg = policy.config
    N_ACT = cfg.n_action_steps
    start = cfg.n_obs_steps - 1
    dts = {
        "observation.state": [i / fps for i in range(1 - cfg.n_obs_steps, 1)],
        "observation.environment_state": [i / fps for i in range(1 - cfg.n_obs_steps, 1)],
        "action": [i / fps for i in range(1 - cfg.n_obs_steps, 1 - cfg.n_obs_steps + cfg.horizon)],
    }
    ds = LeRobotDataset(REPO, delta_timestamps=dts)
    ns = policy.diffusion.noise_scheduler
    T_train = ns.config.num_train_timesteps
    if RATIO >= 1.0:
        # PURE POLICY: initial latent is plain N(0,1) — the expert chunk never
        # enters the sampler; guide weight exactly zero.
        t_sw = T_train
        sigma_eff_norm = float("inf")
    else:
        t_sw = int(RATIO * T_train)
        abar = float(ns.alphas_cumprod[t_sw - 1])
        sigma_eff_norm = float(np.sqrt((1 - abar) / abar))
    epi = np.array(ds.hf_dataset["episode_index"])
    fri = np.array(ds.hf_dataset["frame_index"])
    root = os.path.expanduser(f"~/.cache/huggingface/lerobot/JennyWWW/planar_12_{LINEAGE_TAG}_diff_r_dag{RD}")
    di = pd.concat([pd.read_parquet(f) for f in glob.glob(root + "/data/**/*.parquet", recursive=True)])
    deltas_all, meds = {}, {}
    with torch.no_grad():
        for EP in sorted(np.unique(epi)):
            D = np.stack(
                di[di.episode_index == EP].sort_values("frame_index")["observation.state"].to_numpy()
            )[:, :3]
            med = float(np.median(np.linalg.norm(np.diff(D, axis=0), axis=1)))
            T = len(D)
            idxs = np.where(epi == EP)[0]
            idxs = idxs[np.argsort(fri[idxs])]
            deltas = {}
            starts = list(range(0, T, STRIDE))
            for b0 in range(0, len(starts), 16):
                frames = starts[b0 : b0 + 16]
                items = [ds[int(idxs[min(f, len(idxs) - 1)])] for f in frames]
                raw = {
                    "observation.state": torch.stack([it["observation.state"] for it in items]),
                    "observation.environment_state": torch.stack(
                        [it["observation.environment_state"] for it in items]
                    ),
                    "action": torch.stack([it["action"] for it in items]),
                }
                proc = pre(raw)
                nb = {
                    k: proc[k].cuda().repeat_interleave(KD, dim=0)
                    for k in ("observation.state", "observation.environment_state")
                }
                ng = proc["action"].cuda().repeat_interleave(KD, dim=0)
                if RATIO >= 1.0:
                    x = torch.randn_like(ng)
                    pred = policy.diffusion.generate_actions(nb, noise=x, sa_noise_ratio=1.0)
                else:
                    x = ns.add_noise(
                        ng,
                        torch.randn_like(ng),
                        torch.full((ng.shape[0],), t_sw - 1, dtype=torch.long, device="cuda"),
                    )
                    pred = policy.diffusion.generate_actions(nb, noise=x, sa_noise_ratio=RATIO)
                relP = ((pred + 1) / 2 * rng + lo)[:, :, :3]
                relE = ((ng[:, start : start + N_ACT] + 1) / 2 * rng + lo)[:, :, :3]
                dd = (relP - relE).view(len(frames), KD, N_ACT, 3).cpu().numpy().astype(np.float32)
                for bi, f in enumerate(frames):
                    deltas[f] = dd[bi]
            deltas_all[int(EP)] = deltas
            meds[int(EP)] = med
    rng3 = rng[:3].cpu().numpy().tolist()
    del policy
    torch.cuda.empty_cache()
    return deltas_all, meds, sigma_eff_norm, rng3, N_ACT


def save_npz(RD, RATIO, deltas_all, meds, sigma_eff_norm, rng3):
    arrs = {}
    for ep, dl in deltas_all.items():
        ts = sorted(dl)
        arrs[f"ep{ep}_t"] = np.array(ts, dtype=np.int32)
        arrs[f"ep{ep}_d"] = np.stack([dl[t] for t in ts])  # (n_starts, KD, N_ACT, 3)
        arrs[f"ep{ep}_med"] = np.float32(meds[ep])
    arrs["sigma_eff_norm"] = np.float32(sigma_eff_norm)
    arrs["rng3"] = np.array(rng3, dtype=np.float32)
    np.savez_compressed(f"{S}/analysis/sigma_deltas_{TAG}_dag{RD}.npz", **arrs)


if __name__ == "__main__":
    ROUNDS = [int(x) for x in sys.argv[4].split(",")]
    import os as _os

    for RD in ROUNDS:
        out = f"{S}/analysis/sigma_deltas_{TAG}_dag{RD}.npz"
        if _os.path.exists(out):
            print("skip existing", out, flush=True)
            continue
        deltas, meds, sig, rng3, NA = sample_round(RD, 1.0, 4, 2)
        save_npz(RD, 1.0, deltas, meds, sig, rng3)
        print("saved", TAG, "dag", RD, flush=True)
