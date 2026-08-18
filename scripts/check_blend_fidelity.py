#!/usr/bin/env python
"""Score a visualize_shared_autonomy_sim.py rollout_data.npz for blend fidelity.

Fast iteration loop for the blend pipeline (see 2026-08-18 forensics):

    python my_scripts/visualize_shared_autonomy_sim.py ... --no_show
    python my_scripts/check_blend_fidelity.py <run_dir>/rollout_data.npz

Everything is computed against the npz's own ``guidance_actions_raw`` (the
source episode's expert actions), so no dataset access is needed. The arrays
per ratio are the COMMANDED actions of the closed-loop rollout.

Checks (thresholds encode the measured failure modes they guard against):

  ratio 0.00 (pure guidance — must be a faithful replay):
    * frozen-command fraction < 5%      (self-throttle froze 55% of ticks)
    * mean dev vs time-aligned guidance < 0.05 rad (was 0.81, end 1.48)
  every ratio:
    * first-step commanded speed within [0.5x, 1.6x] of the guidance's own
      launch speed (rest-seeding lurched 0.67 rad/s vs carried 0.24-0.30)
  reported, not gated: median speed ratio vs guidance, dev mean/end.
"""

from __future__ import annotations

import sys

import numpy as np

FPS = 30.0


def main() -> None:
    """CLI entry point."""
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <rollout_data.npz>")
    z = np.load(sys.argv[1], allow_pickle=True)
    g = np.asarray(z["guidance_actions_raw"], dtype=np.float64)[:, :3]
    g_sp = np.linalg.norm(np.diff(g, axis=0), axis=1) * FPS
    g_launch = float(np.mean(g_sp[:3]))
    g_med = float(np.median(g_sp))
    print(f"guidance: {len(g)} frames, launch {g_launch:.3f} rad/s, median speed {g_med:.3f}")

    fails: list[str] = []
    ratio_keys = sorted(k for k in z.files if k.startswith("ratio_"))
    print(
        f"\n{'ratio':>10s} {'live':>5s} {'frozen%':>8s} {'launch':>7s} {'sp_med':>7s} {'dev_mean':>9s} {'dev_end':>8s}"
    )
    for k in ratio_keys:
        a = np.asarray(z[k], dtype=np.float64)[:, :3]
        ok = ~np.isnan(a).any(axis=1)
        a = a[ok]
        # trim trailing post-success hold
        sp = np.linalg.norm(np.diff(a, axis=0), axis=1) * FPS
        end = len(sp)
        while end > 10 and sp[end - 1] < 0.02:
            end -= 1
        a = a[: end + 1]
        sp = sp[:end]
        frozen = float((sp < 0.02).mean())
        launch = float(np.mean(sp[:3]))
        n = min(len(a), len(g))
        dev = np.linalg.norm(a[:n] - g[:n], axis=1)
        print(
            f"{k:>10s} {len(a):5d} {100 * frozen:7.1f}% {launch:7.3f} "
            f"{np.median(sp):7.3f} {dev.mean():9.3f} {dev[n - 1]:8.3f}"
        )
        ratio = float(k.split("_")[1])
        if ratio == 0.0:
            if frozen > 0.05:
                fails.append(f"{k}: frozen-command fraction {100 * frozen:.0f}% (> 5%)")
            if dev.mean() > 0.05:
                fails.append(f"{k}: mean dev vs guidance {dev.mean():.3f} rad (> 0.05)")
        if g_launch > 0.05 and not (0.5 * g_launch <= launch <= 1.6 * g_launch):
            fails.append(f"{k}: launch {launch:.3f} outside [0.5x, 1.6x] of guidance launch {g_launch:.3f}")

    print()
    if fails:
        for f in fails:
            print(f"FAIL {f}")
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
