"""Shared autonomy plotting helpers — joint-angle grids and 3D EE trajectories
overlaid by ``forward_flow_ratio``. Used by the SA visualization scripts.

Extracted from ``visualize_shared_autonomy_DEPRECATED`` so downstream callers
don't have to depend on a deprecated module.
"""

from __future__ import annotations

from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


def _ratio_colors(ratios: list[float]):
    """Return a list of colors, one per ratio, using the plasma colormap."""
    cmap = plt.colormaps["plasma"].resampled(max(len(ratios), 2))
    return [cmap(i / max(len(ratios) - 1, 1)) for i in range(len(ratios))]


def plot_joint_angles(
    action_chunks_by_ratio: dict[float, np.ndarray],
    joint_names: list[str],
    episode_index: int,
    frame_index: int,
    obs_states_raw: np.ndarray | None = None,
    guidance_actions_raw: np.ndarray | None = None,
    decoded_guidance_raw: np.ndarray | None = None,
    output_path: Path | None = None,
    no_show: bool = False,
):
    """Grid of joint angle subplots, one colored line per forward_flow_ratio.

    ``obs_states_raw``: optional ``[n_obs_steps, action_dim]`` raw joint states
                        shown in gray at timesteps ``[-n_obs_steps, …, -1]``
                        before the predicted chunk.
    ``guidance_actions_raw``: optional ``[n_action_steps, action_dim]`` GT
                              actions shown in green at timesteps
                              ``[0, …, n_action_steps-1]`` for reference.
    """
    ratios = sorted(action_chunks_by_ratio.keys())
    colors = _ratio_colors(ratios)
    n_dims = len(joint_names)
    n_cols = 3
    n_rows = ceil(n_dims / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))
    axes = np.array(axes).reshape(-1)

    n_obs = obs_states_raw.shape[0] if obs_states_raw is not None else 0

    for dim_idx in range(n_dims):
        ax = axes[dim_idx]

        # Observation context: gradient from light gray (far from t=0) to dark gray (t=-1).
        if obs_states_raw is not None and obs_states_raw.shape[0] > 0:
            obs_ts = np.arange(-n_obs, 0)
            n = len(obs_ts)
            light, dark = 0.82, 0.38
            # Per-point gray level: 0=earliest (lightest), n-1=latest (darkest)
            grays = [light + (dark - light) * (i / max(n - 1, 1)) for i in range(n)]
            # Draw line segments between consecutive points
            for i in range(n - 1):
                seg_gray = (grays[i] + grays[i + 1]) / 2
                ax.plot(
                    obs_ts[i : i + 2],
                    obs_states_raw[i : i + 2, dim_idx],
                    color=str(seg_gray),
                    linewidth=1.8,
                    linestyle="--",
                    zorder=1,
                )
            # Draw markers separately so each gets its own color
            for i in range(n):
                label = "observation" if (i == 0 and dim_idx == 0) else "_nolegend_"
                ax.plot(
                    [obs_ts[i]],
                    [obs_states_raw[i, dim_idx]],
                    color=str(grays[i]),
                    marker="o",
                    markersize=4,
                    markerfacecolor=str(grays[i]),
                    linewidth=0,
                    zorder=2,
                    label=label,
                )

        # Guidance (GT) actions in green.
        if guidance_actions_raw is not None and guidance_actions_raw.shape[0] > 0:
            g_ts = np.arange(guidance_actions_raw.shape[0])
            ax.plot(
                g_ts,
                guidance_actions_raw[:, dim_idx],
                color="green",
                linewidth=1.8,
                linestyle=":",
                marker="o",
                markersize=4,
                label="guidance" if dim_idx == 0 else "_nolegend_",
                zorder=3,
            )

        # Decoded guidance_chunk (orange dashed): the raw-joint decode of what the
        # wrapper actually fed into the blend. If this overlaps the green guidance,
        # the blend's input is correct and any deviation is the blend math's fault.
        if decoded_guidance_raw is not None and decoded_guidance_raw.shape[0] > 0:
            dg_ts = np.arange(decoded_guidance_raw.shape[0])
            ax.plot(
                dg_ts,
                decoded_guidance_raw[:, dim_idx],
                color="orange",
                linewidth=1.8,
                linestyle="--",
                marker="s",
                markersize=4,
                label="decoded guidance_chunk" if dim_idx == 0 else "_nolegend_",
                zorder=3,
            )

        for ratio, color in zip(ratios, colors, strict=True):
            chunk = action_chunks_by_ratio[ratio]
            timesteps = np.arange(chunk.shape[0])
            ax.plot(
                timesteps, chunk[:, dim_idx], color=color, label=f"ratio={ratio:.2f}", linewidth=1.8, zorder=2
            )

        if n_obs > 0:
            ax.axvline(0, color="gray", linewidth=0.8, linestyle=":", alpha=0.6)
        ax.set_title(joint_names[dim_idx], fontsize=9)
        ax.set_xlabel("timestep")
        ax.set_ylabel("joint angle (rad)")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots.
    for idx in range(n_dims, len(axes)):
        axes[idx].set_visible(False)

    # Shared legend — collect from all axes to include the "observation" entry.
    all_handles, all_labels = [], []
    for ax in axes[:n_dims]:
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
            if label not in all_labels:
                all_handles.append(handle)
                all_labels.append(label)
    fig.legend(all_handles, all_labels, loc="lower right", fontsize=9, ncol=2)
    fig.suptitle(
        f"Joint angles — episode {episode_index}, frame {frame_index}",
        fontsize=11,
        y=1.01,
    )
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), bbox_inches="tight", dpi=150)
        print(f"Saved joint angles plot → {output_path}")
    if not no_show:
        plt.show()
    plt.close(fig)


def plot_ee_trajectories_3d(
    ee_trajectories_by_ratio: dict[float, np.ndarray],
    episode_index: int,
    frame_index: int,
    obs_ee_positions: np.ndarray | None = None,
    guidance_ee_positions: np.ndarray | None = None,
    decoded_guidance_ee_positions: np.ndarray | None = None,
    output_path: Path | None = None,
    no_show: bool = False,
):
    """Interactive 3D plotly figure with one EE trajectory per forward_flow_ratio.

    ``obs_ee_positions``: optional ``[n_obs_steps, 3]`` EE positions shown in
                          gray before ``t=0``.
    ``guidance_ee_positions``: optional ``[n_action_steps, 3]`` GT EE positions
                               shown in green.
    """
    ratios = sorted(ee_trajectories_by_ratio.keys())
    cmap = plt.colormaps["plasma"].resampled(max(len(ratios), 2))

    def to_hex(c):
        r, g, b, _ = c
        return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"

    fig = go.Figure()

    # Observation context: gradient from light gray (far from t=0) to dark gray (t=-1).
    if obs_ee_positions is not None and len(obs_ee_positions) > 0:
        n_obs_pts = len(obs_ee_positions)
        light, dark = 0.82, 0.38
        grays = [light + (dark - light) * (i / max(n_obs_pts - 1, 1)) for i in range(n_obs_pts)]

        def _gray_rgb(g: float) -> str:
            v = int(round(g * 255))
            return f"rgb({v},{v},{v})"

        # Line segments (one trace per segment so each can have its own color)
        for i in range(n_obs_pts - 1):
            seg_color = _gray_rgb((grays[i] + grays[i + 1]) / 2)
            fig.add_trace(
                go.Scatter3d(
                    x=obs_ee_positions[i : i + 2, 0],
                    y=obs_ee_positions[i : i + 2, 1],
                    z=obs_ee_positions[i : i + 2, 2],
                    mode="lines",
                    line={"color": seg_color, "width": 3, "dash": "dash"},
                    showlegend=False,
                )
            )
        # Markers (one trace with per-point colors; first gets the legend entry)
        fig.add_trace(
            go.Scatter3d(
                x=obs_ee_positions[:, 0],
                y=obs_ee_positions[:, 1],
                z=obs_ee_positions[:, 2],
                mode="markers",
                name="observation",
                marker={
                    "color": [_gray_rgb(g) for g in grays],
                    "size": [8] * (n_obs_pts - 1) + [10],
                    "symbol": "circle",
                    "opacity": 0.9,
                },
            )
        )

    # Guidance (GT) EE trajectory in green.
    if guidance_ee_positions is not None and len(guidance_ee_positions) > 0:
        n_g = len(guidance_ee_positions)
        sizes_g = [12] + [5] * (n_g - 2) + [8]
        symbols_g = ["diamond"] + ["circle"] * (n_g - 2) + ["x"]
        fig.add_trace(
            go.Scatter3d(
                x=guidance_ee_positions[:, 0],
                y=guidance_ee_positions[:, 1],
                z=guidance_ee_positions[:, 2],
                mode="lines+markers",
                name="guidance",
                line={"color": "green", "width": 3, "dash": "dot"},
                marker={"color": "green", "size": sizes_g, "symbol": symbols_g, "opacity": 0.9},
            )
        )

    # Decoded guidance_chunk (orange dashed): raw-joint decode of what the wrapper fed
    # into the blend. Overlap with the green trajectory ⇒ blend input is correct.
    if decoded_guidance_ee_positions is not None and len(decoded_guidance_ee_positions) > 0:
        n_dg = len(decoded_guidance_ee_positions)
        sizes_dg = [12] + [5] * (n_dg - 2) + [8]
        symbols_dg = ["diamond"] + ["circle"] * (n_dg - 2) + ["x"]
        fig.add_trace(
            go.Scatter3d(
                x=decoded_guidance_ee_positions[:, 0],
                y=decoded_guidance_ee_positions[:, 1],
                z=decoded_guidance_ee_positions[:, 2],
                mode="lines+markers",
                name="decoded guidance_chunk",
                line={"color": "orange", "width": 3, "dash": "dash"},
                marker={"color": "orange", "size": sizes_dg, "symbol": symbols_dg, "opacity": 0.9},
            )
        )

    for i, ratio in enumerate(ratios):
        traj = ee_trajectories_by_ratio[ratio]  # [n_action_steps+1, 3]
        color = to_hex(cmap(i / max(len(ratios) - 1, 1)))
        n = traj.shape[0]
        sizes = [12] + [5] * (n - 2) + [8]
        # ['circle', 'circle-open', 'cross', 'diamond',
        # 'diamond-open', 'square', 'square-open', 'x']
        symbols = ["diamond"] + ["circle"] * (n - 2) + ["x"]
        fig.add_trace(
            go.Scatter3d(
                x=traj[:, 0],
                y=traj[:, 1],
                z=traj[:, 2],
                mode="lines+markers",
                name=f"ratio={ratio:.2f}",
                line={"color": color, "width": 4},
                marker={"color": color, "size": sizes, "symbol": symbols, "opacity": 0.9},
            )
        )

    fig.update_layout(
        title={
            "text": f"EE trajectory — episode {episode_index}, frame {frame_index}",
            "font": {"size": 14},
        },
        scene={
            "xaxis_title": "X (m)",
            "yaxis_title": "Y (m)",
            "zaxis_title": "Z (m)",
            "aspectmode": "data",
        },
        legend={"font": {"size": 12}},
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        print(f"Saved EE trajectory → {output_path}")
    if not no_show:
        fig.show()
