#!/usr/bin/env python
"""Score a visualize_shared_autonomy_sim.py rollout_data.npz for blend fidelity.

Fast iteration loop for the blend pipeline:

    python my_scripts/visualize_shared_autonomy_sim.py ... --no_show
    python my_scripts/check_blend_fidelity.py <run_dir>/rollout_data.npz

Everything is computed against the npz's own ``guidance_actions_raw`` (the
source episode's expert actions), so no dataset access is needed, and EVERY
threshold is expressed in the demo's own units — ``med_step`` (its median
per-tick joint step) and its median speed — so the same gates apply
unchanged across episodes (slow launches, short paths) and environments
(planar, small-engine UR5e, ...). Pass ``--n_arm``/``--fps`` for non-planar
data; defaults assume a trailing gripper dim and 30 fps.

Checks (dimensionless forms of the measured failure modes they guard):

  ratio 0.00 (pure guidance — must be a faithful replay):
    * frozen-command fraction < 5% of live ticks (frozen = speed below 5%
      of the demo's own median speed; the self-throttle froze 55%)
    * mean PATH deviation < 3 x med_step (nearest-guidance-point — pace
      insensitive; the pre-clock bug measured 44 x)
    * pace (demo indices per wall tick over the live span) in [0.95, 1.05]
    * endpoint: within 6 x med_step of the demo's endpoint. Ending EARLY
      but ON the path (final deviation <= 3 x med_step) is a WARN, not a
      FAIL — that is the env's loose eval-success hold truncating the
      replay (ep4 2026-08-19; use --strict_goal_tolerances to replay
      through it), not a divergence. Ending OFF-path FAILs.
  every ratio:
    * launch speed within [0.5x, 2.0x] of the guidance's own launch
      (relative — rest-seeding lurched 2.7x) — WARN at policy ratios.
"""

from __future__ import annotations

import argparse

import numpy as np


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npz", help="rollout_data.npz from visualize_shared_autonomy_sim.py")
    ap.add_argument(
        "--n_arm",
        type=int,
        default=0,
        help="arm joint count (0 = infer as action dims - 1, assuming a trailing gripper dim)",
    )
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    g_full = np.asarray(z["guidance_actions_raw"], dtype=np.float64)
    n_arm = args.n_arm if args.n_arm > 0 else max(1, g_full.shape[1] - 1)
    fps = float(args.fps)
    g = g_full[:, :n_arm]

    g_steps = np.linalg.norm(np.diff(g, axis=0), axis=1)
    med_step = float(np.median(g_steps[g_steps > 1e-9])) if (g_steps > 1e-9).any() else 1e-3
    g_sp = g_steps * fps
    g_med = float(np.median(g_sp))
    g_launch = float(np.mean(g_sp[:3]))
    frozen_sp = 0.05 * g_med  # rad/s; scale-free "not moving" threshold
    print(
        f"guidance: {len(g)} frames, n_arm={n_arm}, med_step {med_step:.4f} rad "
        f"({g_med:.3f} rad/s cruise), launch {g_launch:.3f} rad/s"
    )
    print(
        f"gates (demo units): frozen<{frozen_sp:.3f} rad/s, path_dev<{3 * med_step:.3f}, "
        f"end<{6 * med_step:.3f} (on-path early = WARN), pace [0.95, 1.05]"
    )

    fails: list[str] = []
    warns: list[str] = []
    ratio_keys = sorted(k for k in z.files if k.startswith("ratio_"))
    print(
        f"\n{'ratio':>10s} {'live':>5s} {'frozen%':>8s} {'launch':>7s} {'sp_med':>7s} "
        f"{'path_dev':>9s} {'pace':>6s} {'end_gap':>8s}"
    )
    for k in ratio_keys:
        a = np.asarray(z[k], dtype=np.float64)[:, :n_arm]
        a = a[~np.isnan(a).any(axis=1)]
        # trim trailing hold (post-success / end-of-budget) at the same
        # scale-free threshold used for the frozen gate.
        sp = np.linalg.norm(np.diff(a, axis=0), axis=1) * fps
        end = len(sp)
        while end > 10 and sp[end - 1] < frozen_sp:
            end -= 1
        a = a[: end + 1]
        sp = sp[:end]
        frozen = float((sp < frozen_sp).mean()) if len(sp) else 0.0
        launch = float(np.mean(sp[:3])) if len(sp) >= 3 else 0.0
        # PATH-aligned deviation: distance to the nearest guidance point —
        # insensitive to pace differences.
        d2 = np.linalg.norm(a[:, None, :] - g[None, :, :], axis=2)
        path_dev = d2.min(axis=1)
        nearest = d2.argmin(axis=1)
        # PACE: demo-index progress per wall tick over the live span.
        pace = float(np.polyfit(np.arange(len(nearest)), nearest, 1)[0]) if len(nearest) > 10 else 1.0
        end_gap = float(np.linalg.norm(a[-1] - g[-1]))
        print(
            f"{k:>10s} {len(a):5d} {100 * frozen:7.1f}% {launch:7.3f} "
            f"{np.median(sp):7.3f} {path_dev.mean():9.3f} {pace:6.2f} {end_gap:8.3f}"
        )
        ratio = float(k.split("_")[1])
        if ratio == 0.0:
            if frozen > 0.05:
                fails.append(f"{k}: frozen-command fraction {100 * frozen:.0f}% (> 5% of live ticks)")
            if path_dev.mean() > 3 * med_step:
                fails.append(
                    f"{k}: mean PATH deviation {path_dev.mean():.3f} rad (> 3 x med_step {3 * med_step:.3f})"
                )
            if not (0.95 <= pace <= 1.05):
                fails.append(f"{k}: pace {pace:.2f}x demo (outside [0.95, 1.05])")
            if end_gap > 6 * med_step:
                if path_dev[-1] <= 3 * med_step:
                    warns.append(
                        f"{k}: ends early ON-path at demo index {int(nearest[-1])}/{len(g) - 1} "
                        f"(loose eval-success hold or budget truncation — consider "
                        f"--strict_goal_tolerances), {end_gap:.3f} rad short of the endpoint"
                    )
                else:
                    fails.append(
                        f"{k}: ends {end_gap:.3f} rad from the demo endpoint AND "
                        f"{path_dev[-1]:.3f} off-path (> 3 x med_step) — diverged"
                    )
            if g_launch > 0.1 * g_med and not (0.5 * g_launch <= launch <= 2.0 * g_launch):
                fails.append(
                    f"{k}: launch {launch:.3f} outside [0.5x, 2.0x] of guidance launch {g_launch:.3f}"
                )
        elif g_launch > 0.1 * g_med and not (0.5 * g_launch <= launch <= 2.0 * g_launch):
            # Non-zero ratios carry a policy contribution — a launch lurch
            # here reflects the POLICY's velocity discontinuity at this
            # state, informative but not a blend-machinery failure.
            warns.append(
                f"{k}: launch {launch:.3f} vs guidance launch {g_launch:.3f} "
                f"(policy-side lurch enters the blend data)"
            )

    print()
    for w in warns:
        print(f"WARN {w}")
    if fails:
        for f in fails:
            print(f"FAIL {f}")
        raise SystemExit(1)
    print("ALL PASS" + (" (with warnings)" if warns else ""))


if __name__ == "__main__":
    main()
