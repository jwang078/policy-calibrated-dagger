#!/usr/bin/env python
"""Render DART label chunks ON THE ROBOT — offline PyBullet, no sim server.

Companion to ``visualize_dart_chunks.py``: the plots show the corrections in
joint space, this shows what they LOOK like on the planar 3-joint arm.

Fully self-contained — it loads
``SplatSim/splatsim/robot_definitions/urdf/planar_3joint.urdf`` into a DIRECT
(headless) PyBullet client and re-creates the scene cuboids from the recorded
``observation.environment_state`` ([block xz, obstacle_i xz..., ee xz]). No
``launch_nodes.py`` server, no ZMQ, no splats. The camera matches the planar
env's ``PYBULLET_CAMERA_*`` (side view along +Y, the plane normal), rendered
larger than the 224² policy view for legibility.

Three robots share the world:

  * EXECUTED (opaque, URDF colors) — the blend rollout's ``observation.state``,
    i.e. the policy-augmented trajectory the labels correct FROM.
  * EXPERT (translucent grey) — the source demo pose at the executed frame's
    projected demo index; the gap to it IS the corridor deviation. DURING a
    chunk it advances along that chunk's OWN demo clock (``info["clock"]``,
    ~1 demo index per label tick), so the ghost is where the expert would be
    when the chunk ends, not where it was when the chunk started. Freezing it
    at the anchor index instead made every chunk look like it missed by
    0.45-0.64 rad — that is just 30-odd demo steps of expert progress.
  * CHUNK (translucent, per-anchor plasma color) — during a correction
    interlude, the arm flies through the H label poses the dataloader would
    serve at that anchor, leaving an end-effector trail.

Timeline: play the rollout at ``fps``; at each selected anchor the executed arm
holds while the chunk ghost runs through k=0..H-1 (starting immediately — see
``anchor_pause_frames``), then a short hold on the rejoin, clear, resume.
"""

from __future__ import annotations

import os

import numpy as np

# Robot / scene constants mirrored from
# SplatSim/splatsim/robots/sim_robot_pybullet_planar.py (PlanarPybulletRobotServer).
SPLATSIM_ROOT = os.path.expanduser("~/code/SplatSim")
PLANAR_URDF = f"{SPLATSIM_ROOT}/splatsim/robot_definitions/urdf/planar_3joint.urdf"
CAMERA_EYE = (0.0, -1.1, 0.2)
CAMERA_TARGET = (0.0, 0.0, 0.2)
CAMERA_FOV = 60.0
ARM_JOINTS = (1, 2, 3)  # joint index 0 is the fixed world_joint
BLOCK_SIZE, BLOCK_RGB = 0.05, (80 / 255, 120 / 255, 200 / 255)
OBSTACLE_SIZE, OBSTACLE_RGB = 0.07, (200 / 255, 90 / 255, 70 / 255)

# The three arms overlap constantly, so each gets ONE flat color — the URDF's
# own per-link palette (yellow/green/grey) made the executed and expert arms
# indistinguishable where they cross. Alpha < 1 renders as a stipple in
# TinyRenderer, which reads as "ghost" and is why the expert keeps it.
# Teal, deliberately: the scene's target block is blue and the obstacles red,
# so an exec arm in either color hides the object it is reaching for.
EXEC_RGBA = (0.00, 0.50, 0.45, 1.0)
EXPERT_RGBA = (0.55, 0.55, 0.55, 0.45)
TRAIL_EXEC_RGB = (0.0, 0.28, 0.26)  # darker than the arm so the trail reads on top of it


def _pose(pb, cid, body, q) -> None:
    """Teleport the arm joints of `body` to `q` (gripper joints stay at 0)."""
    for j, qi in zip(ARM_JOINTS, np.asarray(q, dtype=float)[: len(ARM_JOINTS)]):
        pb.resetJointState(body, j, float(qi), physicsClientId=cid)


def _tint(pb, cid, body, rgba) -> None:
    """Recolor every link of `body` (alpha < 1 renders translucent)."""
    for link in range(-1, pb.getNumJoints(body, physicsClientId=cid)):
        pb.changeVisualShape(body, link, rgbaColor=list(rgba), physicsClientId=cid)


def _ee_link_index(pb, cid, body) -> int:
    """`wrist_camera_link` is the EE reference frame (see the URDF header)."""
    for j in range(pb.getNumJoints(body, physicsClientId=cid)):
        if pb.getJointInfo(body, j, physicsClientId=cid)[12].decode() == "wrist_camera_link":
            return j
    return pb.getNumJoints(body, physicsClientId=cid) - 1


def _ee_pos(pb, cid, body, link) -> np.ndarray:
    return np.asarray(pb.getLinkState(body, link, computeForwardKinematics=True, physicsClientId=cid)[0])


def _overlay(img: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> np.ndarray:
    """Draw colored text lines top-left. No-op (returns img) without PIL."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:  # pragma: no cover - PIL ships with the env
        return img
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    y = 8
    for text, rgb in lines:
        d.text((10, y), text, fill=rgb, font=font, stroke_width=2, stroke_fill=(255, 255, 255))
        y += 24
    return np.asarray(im)


def make_planar_jacobian(size_free_client: bool = True):
    """``jacobian_fn(q) -> (2, 3)`` EE (x, z) Jacobian for the planar arm.

    Central finite differences on the URDF's own FK in a private DIRECT
    client, so it stays exact if the URDF's link lengths change. Feed it to
    ``chunk_labels(..., jacobian_fn=...)`` for the Cartesian-aware gain.
    """
    import pybullet as pb

    cid = pb.connect(pb.DIRECT)
    body = pb.loadURDF(PLANAR_URDF, useFixedBase=True, physicsClientId=cid)
    link = _ee_link_index(pb, cid, body)

    def ee(q):
        _pose(pb, cid, body, q)
        return _ee_pos(pb, cid, body, link)[[0, 2]]

    def jacobian_fn(q, h: float = 1e-5):
        q = np.asarray(q, dtype=float)
        J = np.zeros((2, len(ARM_JOINTS)))
        for i in range(len(ARM_JOINTS)):
            qp, qm = q.copy(), q.copy()
            qp[i] += h
            qm[i] -= h
            J[:, i] = (ee(qp) - ee(qm)) / (2 * h)
        return J

    return jacobian_fn


def _c255(rgb) -> tuple[int, int, int]:
    return tuple(int(255 * c) for c in rgb[:3])


def _frame_camera(pb, cid, body, ee_link, S, geom, env_state, autofit: bool):
    """Env viewpoint, optionally re-centered/tightened onto this episode.

    Fits the X-Z bounding box of everything that matters (executed + expert EE
    paths and the scene objects), keeps the env's eye distance along -Y, and
    picks the smallest FOV (capped at the env's 60°) that contains it with a
    15% margin — so the arm is not a small object in a mostly empty frame.
    """
    if not autofit:
        return CAMERA_EYE, CAMERA_TARGET, CAMERA_FOV
    pts = []
    for q in np.vstack([np.asarray(S, dtype=float)[:, : geom.n_arm], geom.P]):
        _pose(pb, cid, body, q)
        pts.append(_ee_pos(pb, cid, body, ee_link)[[0, 2]])
    pts.append([0.0, 0.2])  # the base pivot is always in frame
    if env_state is not None and len(env_state) >= 4:
        pts.extend(np.asarray(env_state, dtype=float).reshape(-1)[:-2].reshape(-1, 2))
    pts = np.asarray(pts, dtype=float)
    lo, hi = pts.min(axis=0) - 0.12, pts.max(axis=0) + 0.12  # link/gripper bulk
    cx, cz = 0.5 * (lo + hi)
    half = 1.15 * max(hi - lo).item() / 2
    dist = abs(CAMERA_EYE[1])
    fov = min(CAMERA_FOV, 2 * np.degrees(np.arctan2(half, dist)))
    return (cx, CAMERA_EYE[1], cz), (cx, 0.0, cz), float(fov)


def _scene_objects(pb, cid, env_state: np.ndarray | None) -> None:
    """Re-create the recorded scene cuboids at their (x, z) coords (y = 0).

    Layout (PlanarPybulletRobotServer.ORACLE_STATE_*): the first pair is the
    target block, the middle pairs the obstacles, the LAST pair the EE — which
    is proprioception, not an object, so it is dropped.
    """
    if env_state is None or len(env_state) < 4:
        return
    coords = np.asarray(env_state, dtype=float).reshape(-1)[:-2].reshape(-1, 2)
    for i, (x, z) in enumerate(coords):
        size = BLOCK_SIZE if i == 0 else OBSTACLE_SIZE
        rgb = BLOCK_RGB if i == 0 else OBSTACLE_RGB
        half = [size / 2] * 3
        vis = pb.createVisualShape(pb.GEOM_BOX, halfExtents=half, rgbaColor=[*rgb, 1.0], physicsClientId=cid)
        pb.createMultiBody(
            baseMass=0, baseVisualShapeIndex=vis, basePosition=[x, 0.0, z], physicsClientId=cid
        )


def render_dart_sim_video(
    S: np.ndarray,
    idxs: np.ndarray,
    geom,
    chunks: dict[int, np.ndarray],
    anchors_to_show: list[int],
    colors: dict[int, tuple],
    out: str,
    infos: dict | None = None,
    env_state: np.ndarray | None = None,
    fps: float = 30.0,
    size: int = 720,
    stride: int = 1,
    autofit: bool = True,
    chunks_only: bool = False,
    lead_in_frames: int = 20,
    anchor_pause_frames: int = 0,
    end_hold_frames: int = 16,
    chunk_frame_repeat: int = 2,
    title: str = "",
) -> str:
    """Write an mp4 of the rollout with a chunk-ghost interlude at each anchor.

    ``autofit`` keeps the env camera's viewpoint but re-centers and tightens the
    FOV onto what this episode actually uses (the env's 224² policy framing
    covers the whole placement annulus, so a single episode fills little of it).

    ``anchor_pause_frames`` / ``end_hold_frames`` bracket each interlude;
    the pause before the chunk defaults to 0 so there is no stall between the
    executed rollout and the correction.

    ``chunks_only`` drops the rollout playback and the other two arms: the
    video is then ONLY the label chunks, back to back — the expert's response
    on its own, with nothing overlaid on it. Each chunk is preceded by
    ``lead_in_frames`` of the executed blend trajectory running into its
    anchor, played on the SAME body (teal, then recolored at k=0), so the
    state -> label_0 seam is visible as motion: continuous means no visible
    stop, jump, or direction reversal at the color change.

    ``chunk_frame_repeat`` slows the ghost (each label held that many video
    frames) — at 30 fps a 32-step chunk otherwise flashes past in 1 s.
    """
    import imageio.v2 as imageio
    import pybullet as pb
    from dart_labels import _interp_rows

    cid = pb.connect(pb.DIRECT)
    try:
        pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 0, physicsClientId=cid)
        expert = pb.loadURDF(PLANAR_URDF, useFixedBase=True, physicsClientId=cid)
        chunk_bot = pb.loadURDF(PLANAR_URDF, useFixedBase=True, physicsClientId=cid)
        exec_bot = pb.loadURDF(PLANAR_URDF, useFixedBase=True, physicsClientId=cid)
        _tint(pb, cid, expert, EXPERT_RGBA)
        _tint(pb, cid, exec_bot, EXEC_RGBA)
        _scene_objects(pb, cid, env_state)
        ee_link = _ee_link_index(pb, cid, exec_bot)
        # Park the chunk ghost far below the floor until an interlude needs it
        # (a body can't be hidden in PyBullet; alpha 0 still z-fights).
        pb.resetBasePositionAndOrientation(chunk_bot, [0, 0, -50], [0, 0, 0, 1], physicsClientId=cid)

        # Framed from the same episode extents in both modes, so the
        # chunks-only video is directly comparable to the overlaid one.
        eye, target, fov = _frame_camera(pb, cid, exec_bot, ee_link, S, geom, env_state, autofit)
        if chunks_only:
            for body in (exec_bot, expert):
                pb.resetBasePositionAndOrientation(body, [0, 0, -50], [0, 0, 0, 1], physicsClientId=cid)
        view = pb.computeViewMatrix(list(eye), list(target), [0, 0, 1])
        proj = pb.computeProjectionMatrixFOV(fov, 1.0, 0.01, 20.0)

        def grab(lines):
            _, _, rgb, _, _ = pb.getCameraImage(
                size, size, view, proj, renderer=pb.ER_TINY_RENDERER, physicsClientId=cid
            )
            img = np.reshape(rgb, (size, size, 4))[:, :, :3].astype(np.uint8)
            return _overlay(img, lines)

        def show(t: int) -> None:
            """Pose executed + expert robots for rollout frame t."""
            _pose(pb, cid, exec_bot, S[t])
            _pose(pb, cid, expert, _interp_rows(geom.P, float(idxs[t])))

        # Trails are SPHERE BODIES, not addUserDebugLine — TinyRenderer's
        # getCameraImage does not draw debug items (verified: they are absent
        # from the rendered frame), so a debug-line trail would be invisible.
        trail_ids: list[int] = []

        def dot(pos, rgb, radius):
            vis = pb.createVisualShape(
                pb.GEOM_SPHERE, radius=radius, rgbaColor=[*rgb, 1.0], physicsClientId=cid
            )
            trail_ids.append(
                pb.createMultiBody(
                    baseMass=0,
                    baseVisualShapeIndex=vis,
                    basePosition=list(pos),
                    physicsClientId=cid,
                )
            )

        writer = imageio.get_writer(out, fps=fps, macro_block_size=1, quality=8)
        anchor_set = set(anchors_to_show)
        # 720 px fits ~55 chars at this font size — a full repo-id title would
        # run off the right edge.
        header = [(title[:55], (20, 20, 20))] if title else []
        ticks = sorted(anchor_set) if chunks_only else range(0, len(S), max(1, stride))
        for t in ticks:
            base = header + [
                (f"frame {t}/{len(S) - 1}   demo index {idxs[t]:.1f}", (20, 20, 20)),
            ]
            if not chunks_only:
                show(t)
                dot(_ee_pos(pb, cid, exec_bot, ee_link), TRAIL_EXEC_RGB, 0.006)
                base += [
                    ("policy-augmented rollout (executed)", _c255(EXEC_RGBA)),
                    ("expert demo @ projected index", (110, 110, 110)),
                ]
                writer.append_data(grab(base))
            else:
                base += [("DART label chunk only", (110, 110, 110))]

            if t not in anchor_set:
                continue
            # ── correction interlude: the label chunk served at this anchor ──
            C = np.asarray(chunks[t], dtype=float)
            col = tuple(float(c) for c in colors[t][:3])
            col255 = tuple(int(255 * c) for c in col)
            inf = (infos or {}).get(t, {})
            clock = np.asarray(inf.get("clock", []), dtype=float)
            # Everything drawn from here on (lead-in dots included) is cleared
            # when this anchor is done.
            clear_from = len(trail_ids)
            if chunks_only and lead_in_frames > 0:
                # Run the executed trajectory into the anchor on the chunk body
                # itself, so the chunk is a CONTINUATION of a moving arm rather
                # than a fresh pose appearing at rest.
                _tint(pb, cid, chunk_bot, EXEC_RGBA)
                pb.resetBasePositionAndOrientation(chunk_bot, [0, 0, 0], [0, 0, 0, 1], physicsClientId=cid)
                for tt in range(max(0, t - lead_in_frames), t):
                    _pose(pb, cid, chunk_bot, S[tt])
                    dot(_ee_pos(pb, cid, chunk_bot, ee_link), TRAIL_EXEC_RGB, 0.006)
                    writer.append_data(
                        grab(
                            header
                            + [
                                (f"frame {tt}/{len(S) - 1}   demo index {idxs[tt]:.1f}", (20, 20, 20)),
                                ("executed rollout — lead-in to the anchor", _c255(EXEC_RGBA)),
                            ]
                        )
                    )
            # Ghost-stippled only when it has to be read THROUGH the other two
            # arms; alone in the frame it may as well be solid.
            _tint(pb, cid, chunk_bot, (*col, 1.0 if chunks_only else 0.55))
            pb.resetBasePositionAndOrientation(chunk_bot, [0, 0, 0], [0, 0, 0, 1], physicsClientId=cid)
            _pose(pb, cid, chunk_bot, S[t])
            ghost_prev = _ee_pos(pb, cid, chunk_bot, ee_link)
            n_anchor_trail = clear_from
            # No pause by default: the ghost peels off the executed pose in
            # the very next frame, so the rollout flows straight into the
            # correction instead of stalling on a freeze-frame.
            for _ in range(anchor_pause_frames):
                writer.append_data(grab(base + [(f"DART anchor @ frame {t}", col255)]))
            end_i = float(len(geom.A) - 1)
            for k in range(len(C)):
                _pose(pb, cid, chunk_bot, C[k])
                p_k = _ee_pos(pb, cid, chunk_bot, ee_link)
                dot(p_k, col, 0.010)
                ghost_prev = p_k
                # Where the expert is when this label fires. The labels are
                # built from the demo ACTION row at their own clock (see
                # dart_relabel.chunk_labels), so the ghost is posed from
                # geom.A — posing it from geom.P leaves the demo's own
                # state-vs-action tracking gap (~0.014 rad here) as a
                # permanent residual the chunk can never close.
                clk = float(clock[k]) if len(clock) > k else min(float(idxs[t]) + k, end_i)
                dev = float(np.linalg.norm(C[k][: geom.n_arm] - geom.P, axis=1).min())
                clk_line = (
                    f"demo clock {clk:.1f}"
                    if chunks_only
                    else f"expert command @ demo index {clk:.1f}  (chunk clock)"
                )
                lines = base[:-1] + [
                    (clk_line, (110, 110, 110)),
                    (f"DART label chunk @ anchor {t}   k = {k}/{len(C) - 1}", col255),
                ]
                if chunks_only:
                    lines.append((f"corridor deviation {dev:.3f} rad", col255))
                else:
                    _pose(pb, cid, expert, _interp_rows(geom.A, clk)[: geom.n_arm])
                    gap = float(np.linalg.norm(p_k - _ee_pos(pb, cid, expert, ee_link)) * 1000)
                    lines.append(
                        (
                            f"corridor deviation {dev:.3f} rad   ·   EE gap to expert {gap:.0f} mm",
                            col255,
                        )
                    )
                frame = grab(lines)
                for _ in range(max(1, chunk_frame_repeat)):
                    writer.append_data(frame)
            for _ in range(end_hold_frames):
                writer.append_data(
                    grab(
                        base[:-1]
                        + [
                            (clk_line, (110, 110, 110)),
                            ("chunk complete — rejoined the expert", col255),
                        ]
                    )
                )
            # Drop the ghost + its trail; keep the executed trail accumulating.
            for lid in trail_ids[n_anchor_trail:]:
                pb.removeBody(lid, physicsClientId=cid)
            del trail_ids[n_anchor_trail:]
            pb.resetBasePositionAndOrientation(chunk_bot, [0, 0, -50], [0, 0, 0, 1], physicsClientId=cid)
        writer.close()
    finally:
        pb.disconnect(cid)
    print(f"saved → {out}")
    return out
