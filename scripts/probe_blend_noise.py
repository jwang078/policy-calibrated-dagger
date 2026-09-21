"""Open-loop estimate of the closed-loop blend noise level — no sim, no rollouts.

The blend rollout's stationary deviation from the expert corridor decomposes
into (a) the per-chunk INNOVATION (how far one noise+denoise step lands from
the expert, conditioned on an on-corridor state) and (b) COMPOUNDING (how much
of an existing deviation survives the next blend step). Both are pure policy
forward passes, so the stationary deviation distribution D* is computable by
fixed-point iteration over dataset anchors:

    D_0 = {0};  D_{k+1} = executed-prefix deviations of one blend step at
                anchors perturbed by delta ~ D_k  (tangent-velocity histories)

Validation: run against the anc8 blend rounds' intervention datasets with the
round's generating policy and compare D* to the MEASURED rollout deviations
(dev med 2.9 / 1.8 / 1.7 ... med-steps for dag1 / dag2 / dag3).

The probe drives the real SharedAutonomyPolicyWrapper blend path (same code
as blend generation): per anchor it makes two select_action calls (the first
primes the 2-frame obs history along the demo tangent and seeds the RTC prev-
chunk state, mirroring a mid-episode rebuild), then reads the blended chunk
and decodes it against the anchor state.

Usage:
    python my_scripts/probe_blend_noise.py \
        --policy_path outputs/training/.../pretrained_model \
        --dataset_repo_id JennyWWW/planar_12_03dag_diff_r_dag1 \
        --ratio 0.5 --anchor_suffix_steps 8 --iters 4 --anchors_per_iter 48
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dart_labels import chunk_labels, demo_geometry  # noqa: E402
from lib_sa_policy_loading import apply_clip_sample_override, load_wrapped_policy  # noqa: E402
from lib_sa_rollout import _build_sim_batch  # noqa: E402


def _load_episodes(repo_id: str) -> dict[int, dict]:
    import pandas as pd

    from lerobot.utils.constants import HF_LEROBOT_HOME

    root = str(HF_LEROBOT_HOME / repo_id)
    files = sorted(glob.glob(os.path.join(root, "data/**/*.parquet"), recursive=True))
    df = pd.concat([pd.read_parquet(f) for f in files]).sort_values(["episode_index", "frame_index"])
    out = {}
    for ep, g in df.groupby("episode_index"):
        out[int(ep)] = {
            "state": np.stack(g["observation.state"].to_numpy()).astype(np.float64),
            "action": np.stack(g["action"].to_numpy()).astype(np.float64),
            "env": (
                np.stack(g["observation.environment_state"].to_numpy()).astype(np.float64)
                if "observation.environment_state" in g.columns
                else None
            ),
        }
    return out


def _project_local(q: np.ndarray, geom, center: float, window: float) -> float:
    lo = max(0, int(np.floor(center - window)))
    hi = min(len(geom.seg_len), int(np.ceil(center + window)) + 1)
    best_i, best_d = float(np.clip(center, 0, len(geom.seg_len))), np.inf
    for i in range(lo, hi):
        if geom.seg_len[i] < 1e-9:
            continue
        u = float(np.clip(np.dot(q - geom.P[i], geom.seg[i]) / (geom.seg_len[i] ** 2), 0.0, 1.0))
        d = float(np.linalg.norm(q - (geom.P[i] + u * geom.seg[i])))
        if d < best_d:
            best_d, best_i = d, i + u
    return best_i


def _interp(mat: np.ndarray, i: float) -> np.ndarray:
    lo = int(np.clip(np.floor(i), 0, len(mat) - 1))
    hi = min(lo + 1, len(mat) - 1)
    f = float(np.clip(i - lo, 0, 1))
    return (1 - f) * mat[lo] + f * mat[hi]


def one_blend_step(
    wrapper,
    obs_preprocessor,
    ep: dict,
    geom,
    t: int,
    delta: np.ndarray,
    ratio: float,
    horizon: int,
    n_arm: int,
    device: str,
    rng: np.random.Generator,
) -> np.ndarray | None:
    """One primed blend rebuild at frame t perturbed by delta.

    Returns the decoded absolute blended chunk (chunk_len, n_arm), or None
    when the anchor is unusable.
    """
    q_true = ep["state"][t, :n_arm]
    q = q_true + delta
    di = _project_local(q, geom, float(t), 3.0 * (np.linalg.norm(delta) / geom.med_step) + 6.0)
    # tangent velocity at the projected index (per tick)
    hi_i, lo_i = min(di + 1.0, len(geom.P) - 1.0), max(di - 1.0, 0.0)
    v_tan = (_interp(geom.P, hi_i) - _interp(geom.P, lo_i)) / max(hi_i - lo_i, 1e-9)

    # DART-track guidance from the perturbed state (mirrors
    # guidance_from_dart_labels): the track the rollout would have fed.
    track = np.asarray(
        chunk_labels(
            q,
            di,
            geom,
            horizon=horizon,
            prev_state=q - v_tan,
            velocity=v_tan,
        ),
        dtype=np.float32,
    )
    if track.shape[1] < ep["action"].shape[1]:
        pad = np.stack(
            [ep["action"][min(int(round(di)) + k, len(ep["action"]) - 1)] for k in range(len(track))]
        ).astype(np.float32)
        full = pad.copy()
        full[:, : track.shape[1]] = track
        track = full

    wrapper.reset()
    wrapper.forward_flow_ratio = float(ratio)
    wrapper.sample_seed = int(rng.integers(1, 2**31 - 1))
    env_state = ep["env"][t] if ep["env"] is not None else None

    def _batch(q_arm: np.ndarray, guidance: np.ndarray) -> dict:
        ap = ep["state"][t].copy()
        ap[:n_arm] = q_arm
        env_obs = {"agent_pos": ap[None].astype(np.float32)}
        if env_state is not None:
            env_obs["environment_state"] = env_state[None].astype(np.float32)
        return _build_sim_batch(
            env_obs,
            env_preprocessor=None,
            obs_preprocessor=obs_preprocessor,
            rename_map={},
            device=device,
            task_description=None,
            guidance_chunk=guidance,
        )

    base_noise = None
    with torch.no_grad():
        # priming call: previous tick's state along the tangent (fills the
        # 2-frame obs history + RTC prev-chunk state like a mid-episode tick)
        wrapper.select_action(_batch(q - v_tan, track), base_noise=base_noise)
        # the probed rebuild
        wrapper.select_action(_batch(q, track), base_noise=base_noise)
    chunk = wrapper._guided_chunk
    if chunk is None:
        return None
    # decode model-space chunk -> absolute joints (anchor = current state,
    # which EVERY_STEP refreshed at this build)
    rows = []
    with torch.no_grad():
        for k in range(chunk.shape[1]):
            rows.append(wrapper.postprocessor(chunk[:, k, :]).cpu().numpy().reshape(-1)[:n_arm])
    return np.stack(rows)


def main() -> None:
    """Parse args, load the wrapped policy, and run the fixed-point probe."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--policy_path", required=True)
    ap.add_argument("--dataset_repo_id", required=True)
    ap.add_argument("--ratio", type=float, default=0.5)
    ap.add_argument("--anchor_suffix_steps", type=int, default=8)
    ap.add_argument("--anchor_suffix_to_goal", type=lambda s: s.lower() == "true", default=False)
    ap.add_argument("--rtc_hard_prefix_xfade", type=int, default=8)
    ap.add_argument("--rtc_max_guidance_weight", type=float, default=3.0)
    ap.add_argument("--exec_prefix", type=int, default=16, help="blend_interval ticks per rebuild")
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--anchors_per_iter", type=int, default=48)
    ap.add_argument("--num_dofs", type=int, default=3)
    ap.add_argument("--robot_name", default="planar_3joint")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--dump_csv", default=None, help="append per-anchor rows: iter,episode,frame,dev_med_steps"
    )
    args = ap.parse_args()

    wrapper, obs_pre = load_wrapped_policy(
        args.policy_path,
        forward_flow_ratio=args.ratio,
        robot_name=args.robot_name,
        num_dofs=args.num_dofs,
        device=args.device,
    )
    apply_clip_sample_override(wrapper, False)
    from lerobot.policies.shared_autonomy_wrapper import BlendMode, GuidanceBlendStrategy

    wrapper.guidance_blend_strategy = GuidanceBlendStrategy.DENOISE
    wrapper.blend_mode = BlendMode.EVERY_STEP
    wrapper.anchor_prefix_steps = 0
    wrapper.anchor_suffix_steps = args.anchor_suffix_steps
    wrapper.anchor_suffix_to_goal = args.anchor_suffix_to_goal
    wrapper.anchor_every_denoise_step = True
    wrapper.rtc_prev_chunk_guidance = True
    wrapper.rtc_max_guidance_weight = args.rtc_max_guidance_weight
    wrapper.rtc_hard_prefix_xfade = args.rtc_hard_prefix_xfade
    wrapper.resample_noise_per_reblend = False

    eps = _load_episodes(args.dataset_repo_id)
    n_arm = args.num_dofs
    geoms = {e: demo_geometry(d["state"], d["action"], n_arm) for e, d in eps.items() if len(d["state"]) > 48}
    rng = np.random.default_rng(args.seed)
    horizon = int(getattr(wrapper.config, "horizon", 64) or 64)

    pool: list[np.ndarray] = [np.zeros(n_arm)]
    print(f"probe: {len(geoms)} episodes | ratio={args.ratio} | exec_prefix={args.exec_prefix}")
    for it in range(args.iters):
        devs, vecs, frame_devs = [], [], []
        ep_ids = rng.choice(sorted(geoms), size=args.anchors_per_iter, replace=True)
        for e in ep_ids:
            d, geom = eps[int(e)], geoms[int(e)]
            t = int(rng.integers(8, len(d["state"]) - 40))
            delta = pool[int(rng.integers(0, len(pool)))]
            chunk = one_blend_step(
                wrapper, obs_pre, d, geom, t, delta, args.ratio, horizon, n_arm, args.device, rng
            )
            if chunk is None:
                continue
            npfx = min(args.exec_prefix, len(chunk))
            # endpoint (state at the NEXT rebuild -> injected as next iter's
            # perturbation) + per-position deviations over the whole executed
            # prefix (the rollout's frame-median samples the interval, where
            # the path bows out before the suffix anchor pulls it back).
            end = chunk[npfx - 1]
            di_end = _project_local(end, geom, float(t + args.exec_prefix), 12.0)
            vec = end - _interp(geom.P, di_end)
            vecs.append(vec)
            devs.append(np.linalg.norm(vec) / geom.med_step)
            if args.dump_csv:
                with open(args.dump_csv, "a") as f:
                    f.write(f"{it},{int(e)},{t},{np.linalg.norm(vec) / geom.med_step:.4f}\n")
            for k in range(npfx):
                di_k = _project_local(chunk[k], geom, float(t + k), 12.0)
                frame_devs.append(np.linalg.norm(chunk[k] - _interp(geom.P, di_k)) / geom.med_step)
        devs = np.array(devs)
        fd = np.array(frame_devs)
        print(
            f"iter {it}: endpoint med {np.median(devs):.2f} p90 {np.percentile(devs, 90):.2f} | "
            f"frame med {np.median(fd):.2f} p90 {np.percentile(fd, 90):.2f} med-steps (n={len(devs)})"
        )
        pool = vecs if vecs else pool
    print("\nfixed-point (last iter) = predicted stationary blend deviation")


if __name__ == "__main__":
    main()
