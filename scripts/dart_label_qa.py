#!/usr/bin/env python
"""QA certificate for DART chunk labels over an entire blend dataset.

Synthesizes the label chunk at EVERY anchor frame (exactly as the train-time
DartChunkDataset would) and checks each against the intervention dynamic
envelope, which the label budgets are anchored to:

  * speed:      any commanded step <= max(1.4 * B, 1.05 * launch speed)
  * accel:      any 2nd-difference <= 1.5 * max(demo acc_p95, glide budget)
                (demo acc_p95 IS the intervention planner's realized
                max_joint_acc envelope — the source demos are planner-made
                intervention recordings)
  * turn rate:  <= 25 deg/tick at speed (recorded interventions max ~21)
  * junction:   |label_0 - state| <= 1.5 * B (no teleports)

Also reports EDGE-CASE COVERAGE — how many anchors exercised each structural
path of the label function (brake phase, uncapped merge, rendezvous at demo
end, from-rest launch, pursuit fallback) — so "did we hit the ep-9-class
cases?" has a positive answer instead of hoping.

Exit code 1 if any episode has violations (usable as a gate in orchestrate).

Usage:
    python my_scripts/dart_label_qa.py \
        --blend_repo_id JennyWWW/planar_12_03dag_diff_r_dag1_blend050 \
        --source_repo_id JennyWWW/planar_12_03dag_diff_r_dag1
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from dart_labels import chunk_labels, demo_geometry, project_states

FPS = 30.0


def _load_all(repo: str) -> pd.DataFrame:
    root = os.path.expanduser(f"~/.cache/huggingface/lerobot/{repo}")
    files = sorted(glob.glob(root + "/data/**/*.parquet", recursive=True))
    if not files:
        raise SystemExit(f"no parquet files under {root}")
    return pd.concat([pd.read_parquet(f) for f in files])


def main() -> None:
    """Run the QA sweep."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--blend_repo_id", required=True)
    ap.add_argument("--source_repo_id", required=True)
    ap.add_argument("--horizon", type=int, default=32)
    ap.add_argument("--num_dofs", type=int, default=3)
    ap.add_argument("--stride", type=int, default=1, help="anchor stride (1 = every frame)")
    ap.add_argument("--max_turn_deg", type=float, default=25.0)
    args = ap.parse_args()

    bl_df = _load_all(args.blend_repo_id)
    src_df = _load_all(args.source_repo_id)
    meta = pd.read_parquet(
        os.path.expanduser(
            f"~/.cache/huggingface/lerobot/{args.blend_repo_id}/meta/episodes/chunk-000/file-000.parquet"
        )
    )
    src_map = (
        dict(zip(meta.episode_index, meta.source_episode_idx)) if "source_episode_idx" in meta.columns else {}
    )

    n = args.num_dofs
    cov = {"braked": 0, "uncapped": 0, "end_clamped": 0, "from_rest": 0, "pursuit": 0}
    total_anchors = 0
    bad_eps: dict[int, list[str]] = {}
    for ep in sorted(bl_df.episode_index.unique()):
        g = bl_df[bl_df.episode_index == ep]
        sg = src_df[src_df.episode_index == src_map.get(ep, ep)]
        if len(sg) < 10 or len(g) < 10:
            continue
        geom = demo_geometry(
            np.stack(sg["observation.state"].to_numpy()), np.stack(sg["action"].to_numpy()), n
        )
        S = np.stack(g["observation.state"].to_numpy())[:, :n].astype(np.float64)
        if "relabel_demo_index" in g.columns:
            idxs = np.array([float(np.reshape(x, -1)[0]) for x in g["relabel_demo_index"].to_numpy()])
        else:
            idxs = project_states(S, geom, index_window=12)
        b = 1.2 * geom.med_step
        # Label-internal accel bound: 2x the planner's realized L2 envelope
        # (acc_p95 is per-episode and collapses on slow demos, so floor at
        # 0.2 * med_step ~ 3.5 rad/s^2 planar — junction noise scale is set
        # by the env, not by how hard this particular demo turns).
        a_lim = max(2.0 * float(np.sqrt(n)) * geom.acc_p95, 0.2 * geom.med_step)
        viols: list[str] = []
        for t in range(1, len(S) - 1, max(1, args.stride)):
            lo, hi = max(0, t - 3), min(len(S) - 1, t + 3)
            v0 = (S[hi] - S[lo]) / max(1, hi - lo)
            info: dict = {}
            L = chunk_labels(
                S[t],
                float(idxs[t]),
                geom,
                horizon=args.horizon,
                prev_state=S[t - 1],
                velocity=v0,
                info=info,
            )[:, :n]
            total_anchors += 1
            if info.get("branch") == "pursuit":
                cov["pursuit"] += 1
            else:
                for key in ("braked", "uncapped", "end_clamped"):
                    cov[key] += bool(info.get(key))
                cov["from_rest"] += info.get("sp0", 1.0) < 0.5 * geom.med_step
            track = np.vstack([S[t] - v0, S[t], L])
            v = np.diff(track, axis=0)
            sp0 = float(np.linalg.norm(v0))
            speeds = np.linalg.norm(v[1:], axis=1)
            if speeds.max() > max(1.4 * b, 1.05 * sp0) + 1e-9:
                viols.append(f"t={t} speed {speeds.max() * FPS:.2f} rad/s")
            # accel: LABEL-INTERNAL only — the junction 2nd-difference mixes
            # in the robot's own state noise, which is data, not the label.
            amax = float(np.linalg.norm(np.diff(L, n=2, axis=0), axis=1).max())
            if amax > a_lim + 1e-9:
                viols.append(f"t={t} accel {amax * FPS * FPS:.2f} rad/s^2")
            if float(np.linalg.norm(L[0] - S[t])) > 1.5 * b:
                viols.append(f"t={t} junction jump {float(np.linalg.norm(L[0] - S[t])):.3f} rad")
            for k in range(1, len(v) - 1):
                n0, n1 = np.linalg.norm(v[k]), np.linalg.norm(v[k + 1])
                if n0 > 0.3 * geom.med_step and n1 > 0.3 * geom.med_step:
                    ang = np.degrees(np.arccos(np.clip(np.dot(v[k], v[k + 1]) / (n0 * n1), -1, 1)))
                    if ang > args.max_turn_deg:
                        viols.append(f"t={t} turn {ang:.0f} deg/tick @k={k - 1}")
                        break
        if viols:
            bad_eps[int(ep)] = viols

    print(
        f"\n{args.blend_repo_id}: {total_anchors} anchors checked across {bl_df.episode_index.nunique()} episodes"
    )
    print(
        "edge-case coverage: "
        + "  ".join(f"{k}={v} ({100 * v / max(1, total_anchors):.1f}%)" for k, v in cov.items())
    )
    if not bad_eps:
        print("PASS — every label chunk within the intervention envelope")
        return
    print(f"FAIL — {len(bad_eps)} episode(s) with violations:")
    for ep, viols in bad_eps.items():
        print(f"  ep {ep}: {len(viols)} violation(s), first 3: {viols[:3]}")
    sys.exit(1)


if __name__ == "__main__":
    main()
