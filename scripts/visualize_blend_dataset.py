"""Visualize a SAVED blend dataset against its source intervention.

The dataset-side sibling of visualize_shared_autonomy_sim.py: same reading
(per-joint traces, EE trajectory, deviation), but from data already on disk —
no sim, no policy. Resolves the source intervention from the blend episode's
own provenance metadata (``source_dataset_repo_id`` + ``source_episode_idx``),
falling back to stripping the ``_blend...`` suffix off the repo name
(dagger_naming's grammar).

Usage:
    python my_scripts/visualize_blend_dataset.py \
        --blend_repo_id JennyWWW/planar_12_03dag_diff_r_dag1_blend050anc8 \
        --episode_index 0 [--out out.png]

    # first/middle/last episodes, one figure each:
    python my_scripts/visualize_blend_dataset.py --blend_repo_id ... --episodes fml
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dart_labels import demo_geometry  # noqa: E402


def _root(repo_id: str) -> str:
    from lerobot.utils.constants import HF_LEROBOT_HOME

    return str(HF_LEROBOT_HOME / repo_id)


def _load_episode(repo_id: str, ep: int) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(_root(repo_id), "data/**/*.parquet"), recursive=True))
    df = pd.concat([pd.read_parquet(f) for f in files])
    out = df[df.episode_index == ep].sort_values("frame_index")
    if out.empty:
        raise SystemExit(f"{repo_id}: episode {ep} not found")
    return out


def _episodes_meta(repo_id: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(_root(repo_id), "meta/episodes/**/*.parquet"), recursive=True))
    return pd.concat([pd.read_parquet(f) for f in files])


def _resolve_source(meta_row: pd.Series, blend_repo: str) -> tuple[str, int]:
    src_repo = meta_row.get("source_dataset_repo_id")
    if not isinstance(src_repo, str) or not src_repo:
        # dagger_naming grammar fallback: strip the _blend<NNN><variant> tail.
        src_repo = re.sub(r"_blend\d{3}[a-z0-9]*(_nc|_tfc|_nocoll)?$", "", blend_repo)
    return src_repo, int(meta_row["source_episode_idx"])


def _ee(df: pd.DataFrame) -> np.ndarray | None:
    if "observation.environment_state" not in df.columns:
        return None
    env = np.stack(df["observation.environment_state"].to_numpy())
    # planar convention: trailing two env dims are the EE position. Use the
    # per-episode moving-dims heuristic to stay honest.
    ptp = np.ptp(env, axis=0)
    moving = np.where(ptp > 0.005)[0]
    if len(moving) < 2:
        return None
    return env[:, moving[-2:]]


def plot_episode(blend_repo: str, ep: int, out: str | None, n_arm: int = 3) -> str:
    """Plot one blend episode against its source intervention (joints, actions, EE path); returns the PNG path."""
    meta = _episodes_meta(blend_repo)
    row = meta[meta.episode_index == ep].iloc[0]
    src_repo, src_ep = _resolve_source(row, blend_repo)
    b = _load_episode(blend_repo, ep)
    s = _load_episode(src_repo, src_ep)
    bq = np.stack(b["observation.state"].to_numpy())[:, :n_arm]
    blend_act = np.stack(b["action"].to_numpy())[:, :n_arm]
    sq = np.stack(s["observation.state"].to_numpy())[:, :n_arm]
    geom = demo_geometry(sq, np.stack(s["action"].to_numpy()), n_arm)
    d2 = np.sqrt(((bq[:, None, :] - sq[None, :, :]) ** 2).sum(-1))
    dev = d2.min(1) / geom.med_step
    proj_idx = d2.argmin(1)
    step_b = np.linalg.norm(np.diff(bq, axis=0), axis=1) / geom.med_step
    step_s = np.linalg.norm(np.diff(sq, axis=0), axis=1) / geom.med_step
    ratio = row.get("blend_ratio", float("nan"))
    scen = row.get("source_scenario_idx", "?")

    fig, axes = plt.subplots(2, 3, figsize=(19, 9))
    for j in range(n_arm):
        ax = axes[0, j]
        ax.plot(np.arange(len(sq)), sq[:, j], color="0.6", lw=5, alpha=0.7, label="intervention (source)")
        ax.plot(np.arange(len(bq)), bq[:, j], color="tab:blue", lw=1.6, label="blend state")
        ax.plot(
            np.arange(len(blend_act)),
            blend_act[:, j],
            color="tab:red",
            lw=0.9,
            ls="--",
            alpha=0.8,
            label="blend action",
        )
        ax.set_title(f"Joint {j + 1}")
        ax.set_xlabel("tick")
        if j == 0:
            ax.legend(fontsize=8)
    ax = axes[1, 0]
    ee_b, ee_s = _ee(b), _ee(s)
    if ee_b is not None and ee_s is not None:
        ax.plot(ee_s[:, 0], ee_s[:, 1], color="0.6", lw=5, alpha=0.7, label="intervention EE")
        pts = ax.scatter(
            ee_b[:, 0],
            ee_b[:, 1],
            c=np.arange(len(ee_b)),
            cmap="viridis",
            s=6,
            label="blend EE (time-colored)",
        )
        fig.colorbar(pts, ax=ax, label="tick")
        ax.set_title("EE trajectory")
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no EE dims in env_state", ha="center")
    ax = axes[1, 1]
    ax.plot(
        dev, color="tab:blue", label=f"deviation (med {np.median(dev):.1f}, p90 {np.percentile(dev, 90):.1f})"
    )
    ax2 = ax.twinx()
    ax2.plot(np.arange(len(step_b)), step_b, color="tab:orange", lw=0.8, alpha=0.7, label="blend step")
    ax2.axhline(np.median(step_s[step_s > 1e-9]), color="0.4", ls=":", label="source med step")
    ax.set_title("Deviation to source (med-steps) / step size")
    ax.set_xlabel("tick")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax = axes[1, 2]
    ax.plot([0, len(sq) - 1], [0, len(sq) - 1], color="0.6", lw=3, alpha=0.7, label="source pace (y=x)")
    ax.plot(
        np.arange(len(bq)) * (len(sq) - 1) / max(len(bq) - 1, 1),
        proj_idx,
        color="tab:blue",
        lw=1.4,
        label="blend progress (projected)",
    )
    ax.set_title("Progress alignment (lead/lag vs source)")
    ax.set_xlabel("blend tick (rescaled to source length)")
    ax.set_ylabel("nearest source index")
    ax.legend(fontsize=8)
    fig.suptitle(
        f"{blend_repo.split('/')[-1]} ep {ep} (sample {int(row.get('blend_sample_idx', 0))})  ←  "
        f"{src_repo.split('/')[-1]} ep {src_ep}  |  scenario {scen}  |  ratio {ratio}  |  "
        f"{len(bq)} vs {len(sq)} ticks"
    )
    fig.tight_layout()
    out = out or f"blend_vs_int_{blend_repo.split('/')[-1]}_ep{ep}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"saved → {out}")
    return out


def main() -> None:
    """Parse args and render one figure per requested episode."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--blend_repo_id", required=True)
    ap.add_argument("--episode_index", type=int, default=None)
    ap.add_argument("--episodes", default=None, help="'fml' = first/middle/last")
    ap.add_argument("--out", default=None)
    ap.add_argument("--num_arm_joints", type=int, default=3)
    args = ap.parse_args()
    if args.episodes == "fml":
        meta = _episodes_meta(args.blend_repo_id)
        n = len(meta)
        eps = sorted({0, n // 2, n - 1})
    else:
        eps = [args.episode_index or 0]
    for ep in eps:
        out = args.out if len(eps) == 1 else None
        plot_episode(args.blend_repo_id, int(ep), out, n_arm=args.num_arm_joints)


if __name__ == "__main__":
    main()
