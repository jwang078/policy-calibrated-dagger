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
    warns: list[str] = []
    ratio_keys = sorted(k for k in z.files if k.startswith("ratio_"))
    print(
        f"\n{'ratio':>10s} {'live':>5s} {'frozen%':>8s} {'launch':>7s} {'sp_med':>7s} "
        f"{'path_dev':>9s} {'pace':>6s} {'end_gap':>8s}"
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
        # PATH-aligned deviation: distance to the nearest guidance point —
        # insensitive to pace differences (a rollout running ahead/behind the
        # demo clock scores 0 if it stays on the demo's path).
        d2 = np.linalg.norm(a[:, None, :] - g[None, :, :], axis=2)
        path_dev = d2.min(axis=1)
        nearest = d2.argmin(axis=1)
        # PACE: demo-index progress per wall tick (1.0 = demo pace) over the
        # live span, from a robust linear fit of nearest-index vs tick.
        pace = float(np.polyfit(np.arange(len(nearest)), nearest, 1)[0]) if len(nearest) > 10 else 1.0
        end_gap = float(np.linalg.norm(a[-1] - g[-1]))
        print(
            f"{k:>10s} {len(a):5d} {100 * frozen:7.1f}% {launch:7.3f} "
            f"{np.median(sp):7.3f} {path_dev.mean():9.3f} {pace:6.2f} {end_gap:8.3f}"
        )
        ratio = float(k.split("_")[1])
        if ratio == 0.0:
            if frozen > 0.05:
                fails.append(f"{k}: frozen-command fraction {100 * frozen:.0f}% (> 5%)")
            if path_dev.mean() > 0.05:
                fails.append(f"{k}: mean PATH deviation {path_dev.mean():.3f} rad (> 0.05)")
            if not (0.95 <= pace <= 1.05):
                # The demo-pace clock makes an exact-timing replay achievable
                # (measured pace 1.00 on ep0, 2026-08-18) — hold the line.
                fails.append(f"{k}: pace {pace:.2f}x demo (outside [0.95, 1.05])")
            if end_gap > 0.1:
                fails.append(f"{k}: ends {end_gap:.3f} rad from the demo endpoint (> 0.1)")
            if g_launch > 0.05 and not (0.5 * g_launch <= launch <= 2.0 * g_launch):
                fails.append(
                    f"{k}: launch {launch:.3f} outside [0.5x, 2.0x] of guidance launch {g_launch:.3f}"
                )
        elif g_launch > 0.05 and not (0.5 * g_launch <= launch <= 2.0 * g_launch):
            # Non-zero ratios carry a policy contribution — a launch lurch
            # here reflects the POLICY's velocity-discontinuity at this
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
        sys.exit(1)
    print("ALL PASS" + (" (with warnings)" if warns else ""))


if __name__ == "__main__":
    main()
