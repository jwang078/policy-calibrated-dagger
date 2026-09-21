"""Tkinter GUI for the SharedAutonomyPolicyWrapper (ratio slider + controls)."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lerobot.policies.shared_autonomy_wrapper import SharedAutonomyPolicyWrapper


def launch_ratio_slider(wrapper: SharedAutonomyPolicyWrapper) -> None:
    """Launch a Tkinter slider in a background daemon thread to live-edit forward_flow_ratio."""
    import tkinter as tk

    def _run():
        root = tk.Tk()
        root.title("Shared Autonomy")
        root.resizable(False, False)

        tk.Label(root, text="forward_flow_ratio", font=("Helvetica", 12)).pack(padx=16, pady=(12, 0))

        var = tk.DoubleVar(value=wrapper.forward_flow_ratio)
        label = tk.Label(root, text=f"{wrapper.forward_flow_ratio:.2f}", font=("Courier", 14, "bold"))
        label.pack(pady=(0, 4))

        def on_change(val):
            ratio = round(float(val), 2)
            wrapper.forward_flow_ratio = ratio
            label.config(text=f"{ratio:.2f}")

        slider = tk.Scale(
            root,
            from_=0.0,
            to=1.0,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            length=300,
            variable=var,
            command=on_change,
            showvalue=False,
        )
        slider.pack(padx=16, pady=(0, 16))

        tk.Label(root, text="0 = pure guidance    1 = pure policy", font=("Helvetica", 9), fg="gray").pack(
            pady=(0, 4)
        )

        # --- Full control toggle ---
        # Track the last nonzero ratio so we can restore it when toggling back.
        init_ratio = wrapper.forward_flow_ratio
        last_policy_ratio = [init_ratio if init_ratio > 0.0 else 1.0]

        def _toggle_full_control():
            current = wrapper.forward_flow_ratio
            if current == 0.0:
                # Restore last policy ratio
                new_ratio = last_policy_ratio[0]
            else:
                # Save current nonzero ratio, switch to full teleop
                last_policy_ratio[0] = current
                new_ratio = 0.0
            wrapper.forward_flow_ratio = new_ratio
            var.set(new_ratio)
            label.config(text=f"{new_ratio:.2f}")
            _update_toggle_text()

        toggle_btn = tk.Button(
            root,
            text="",
            font=("Helvetica", 11, "bold"),
            width=24,
            command=_toggle_full_control,
        )
        toggle_btn.pack(padx=16, pady=(4, 12))

        def _update_toggle_text():
            if wrapper.forward_flow_ratio == 0.0:
                toggle_btn.config(text="Full Control: ON", fg="green")
            else:
                toggle_btn.config(text="Full Control: OFF", fg="black")

        _update_toggle_text()

        # Track the ratio the slider had when the user first clicked down.
        ratio_at_press = [wrapper.forward_flow_ratio]

        def _on_slider_press(event):
            ratio_at_press[0] = wrapper.forward_flow_ratio

        def _on_slider_release(event):
            released_ratio = round(float(var.get()), 2)
            if released_ratio > 0.0:
                # Released at a nonzero value — save it directly.
                last_policy_ratio[0] = released_ratio
            else:
                # Released at 0.0 — save the value from when they clicked down,
                # unless that was also 0.0 (then keep whatever was saved before).
                if ratio_at_press[0] > 0.0:
                    last_policy_ratio[0] = ratio_at_press[0]
            _update_toggle_text()

        slider.bind("<ButtonPress-1>", _on_slider_press)
        slider.bind("<ButtonRelease-1>", _on_slider_release)

        # Update toggle text on every slider tick (but don't save last_policy_ratio here)
        _orig_on_change = on_change

        def on_change_with_toggle(val):
            _orig_on_change(val)
            _update_toggle_text()

        slider.config(command=on_change_with_toggle)

        # --- Pause / Resume button ---
        def _toggle_pause():
            if wrapper._run_event.is_set():
                wrapper._run_event.clear()  # pause
            else:
                wrapper._run_event.set()  # unpause
            _update_pause_text()

        pause_btn = tk.Button(
            root,
            text="",
            font=("Helvetica", 11, "bold"),
            width=24,
            command=_toggle_pause,
        )
        pause_btn.pack(padx=16, pady=(0, 12))

        pause_warning = tk.Label(
            root, text="", font=("Helvetica", 9), fg="red", wraplength=280, justify=tk.CENTER
        )
        pause_warning.pack(padx=16, pady=(0, 4))

        def _update_pause_text():
            if wrapper._run_event.is_set():
                pause_btn.config(text="Pause", fg="black")
                pause_warning.config(text="")
            else:
                pause_btn.config(text="Resume", fg="orange")
                pause_warning.config(text="Do not move or reset the robot or environment while paused")

        _update_pause_text()

        # --- RRT to Goal toggle ---
        # Single button: first click plans + executes; click again while busy cancels.
        from lerobot.policies.rrt_to_goal import RRTMode

        rrt_btn = tk.Button(
            root,
            text="RRT to Goal",
            font=("Helvetica", 11, "bold"),
            width=24,
            command=lambda: wrapper.trigger_rrt_to_goal(),
        )
        rrt_btn.pack(padx=16, pady=(0, 4))

        rrt_status = tk.Label(root, text="RRT: idle", font=("Helvetica", 9), fg="gray")
        rrt_status.pack(padx=16, pady=(0, 12))

        rrt_colors = {
            RRTMode.IDLE: "gray",
            RRTMode.PLANNING: "orange",
            RRTMode.EXECUTING: "green",
        }

        def _poll_rrt():
            mode = wrapper._rrt.mode
            rrt_status.config(text=f"RRT: {mode.value}", fg=rrt_colors[mode])
            rrt_btn.config(text=("Cancel RRT" if mode != RRTMode.IDLE else "RRT to Goal"))
            root.after(150, _poll_rrt)

        _poll_rrt()

        # --- Teleop recording status ---
        from lerobot.policies.teleop_recording import TeleopRecordingContext

        ctx = TeleopRecordingContext.get_instance()

        sep = tk.Frame(root, height=1, bg="gray")
        sep.pack(fill=tk.X, padx=16, pady=(0, 8))

        rec_label = tk.Label(root, text="", font=("Courier", 11), anchor="w", justify=tk.LEFT)
        rec_label.pack(padx=16, pady=(0, 12), fill=tk.X)

        def _update_recording_status():
            if ctx.recording:
                frames = ctx.episode_frame_count
                threshold = ctx.min_episode_length
                if ctx.padding:
                    color = "orange"
                    status = f"PAD  {frames}/{threshold} frames"
                elif frames >= threshold:
                    color = "green"
                    status = f"REC  {frames} frames (will save)"
                else:
                    color = "red"
                    status = f"REC  {frames}/{threshold} frames (too short)"
            else:
                status = "Not recording"
                color = "gray"
            saved = ctx.total_saved_episodes
            rec_label.config(text=f"{status}\nSaved episodes: {saved}", fg=color)
            root.after(200, _update_recording_status)

        _update_recording_status()

        # --- Discard episode button ---
        def _discard_episode():
            # Auto-pause so no new frames are added after discard
            wrapper._run_event.clear()
            _update_pause_text()
            ctx.discard_requested = True
            # Update display immediately so user sees the discard
            ctx.recording = False
            ctx.episode_frame_count = 0

        discard_btn = tk.Button(
            root,
            text="Discard Episode",
            font=("Helvetica", 11, "bold"),
            width=24,
            fg="red",
            command=_discard_episode,
        )
        discard_btn.pack(padx=16, pady=(0, 12))

        root.mainloop()

    t = threading.Thread(target=_run, daemon=True, name="sa-ratio-slider")
    t.start()
