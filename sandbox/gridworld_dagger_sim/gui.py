"""Tkinter GUI for the gridworld DAgger simulator.

Single window, four tabs (Edit Grid / Edit Interventions / Compose Policy /
Rollouts) sharing one Canvas. All math lives in core.py — this file is just
view + interaction.
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Any

import numpy as np

from my_scripts.gridworld_dagger_sim.core import (
    Action,
    Grid,
    NodeType,
    Policy,
    SimState,
    build_policy_catalog,
    default_state,
    evaluate,
    load_state,
    rollout,
    save_state,
)

# ── Visual constants ────────────────────────────────────────────────────────

CELL_PX = 110
NODE_RADIUS = 28
CANVAS_PAD = 50

NODE_COLOR = {
    NodeType.REGULAR: "#cccccc",
    NodeType.SUCCESS: "#34a853",
    NodeType.FAILURE: "#ea4335",
}
NODE_OUTLINE = "#444"
EDGE_DEFAULT = "#888"
EDGE_POLICY = "#1a73e8"

TRAJ_COLORS = ["#f59e0b", "#8b5cf6", "#06b6d4", "#ef4444", "#22c55e", "#ec4899", "#3b82f6", "#a855f7"]


# ─────────────────────────────── Main App ───────────────────────────────────


class GridworldSimApp:
    def __init__(self, root: tk.Tk, state: SimState):
        self.root = root
        root.title("Gridworld DAgger Simulator")

        self.state = state
        self._catalog_cache: dict[str, Policy] | None = None
        self._show_visitation = tk.BooleanVar(value=False)

        # Top: notebook with 4 tabs.
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=8)
        self.notebook.bind("<<NotebookTabChanged>>", lambda _e: self._render_for_tab())

        # Center: a frame per tab, each containing the shared canvas region
        # + tab-specific side panel.
        self.tab_grid = ttk.Frame(self.notebook)
        self.tab_policy = ttk.Frame(self.notebook)
        self.tab_int = ttk.Frame(self.notebook)
        self.tab_comp = ttk.Frame(self.notebook)
        self.tab_roll = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_grid, text="Edit Grid")
        self.notebook.add(self.tab_policy, text="Edit Policy")
        self.notebook.add(self.tab_int, text="Edit Interventions")
        self.notebook.add(self.tab_comp, text="Compose Policy")
        self.notebook.add(self.tab_roll, text="Rollouts")

        # Each tab gets a left-side canvas + right-side controls.
        self.canvas_grid = self._make_canvas(self.tab_grid)
        self.canvas_policy = self._make_canvas(self.tab_policy)
        self.canvas_int = self._make_canvas(self.tab_int)
        self.canvas_comp = self._make_canvas(self.tab_comp)
        self.canvas_roll = self._make_canvas(self.tab_roll)

        self._build_tab_grid()
        self._build_tab_policy()
        self._build_tab_interventions()
        self._build_tab_compose()
        self._build_tab_rollouts()

        self._invalidate_catalog()

        # Bottom: status bar.
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(root, textvariable=self.status_var, anchor="w", relief="sunken").pack(
            fill="x", side="bottom"
        )

        self._render_for_tab()

    # ── Canvas helpers ─────────────────────────────────────────────────────

    def _make_canvas(self, parent: ttk.Frame) -> tk.Canvas:
        H, W = self.state.grid.height, self.state.grid.width
        w = W * CELL_PX + 2 * CANVAS_PAD
        h = H * CELL_PX + 2 * CANVAS_PAD
        wrapper = ttk.Frame(parent)
        wrapper.pack(side="left", fill="both", expand=True, padx=4, pady=4)
        c = tk.Canvas(
            wrapper, width=w, height=h, bg="white", highlightthickness=1, highlightbackground="#bbb"
        )
        c.pack(fill="both", expand=True)
        return c

    def _cell_center(self, r: int, c: int) -> tuple[int, int]:
        return CANVAS_PAD + c * CELL_PX + CELL_PX // 2, CANVAS_PAD + r * CELL_PX + CELL_PX // 2

    def _canvas_to_cell(self, x: int, y: int) -> tuple[int, int] | None:
        gx = x - CANVAS_PAD
        gy = y - CANVAS_PAD
        if gx < 0 or gy < 0:
            return None
        c = gx // CELL_PX
        r = gy // CELL_PX
        if r >= self.state.grid.height or c >= self.state.grid.width:
            return None
        return int(r), int(c)

    def _resize_all_canvases(self) -> None:
        H, W = self.state.grid.height, self.state.grid.width
        w = W * CELL_PX + 2 * CANVAS_PAD
        h = H * CELL_PX + 2 * CANVAS_PAD
        for canv in (
            self.canvas_grid,
            self.canvas_policy,
            self.canvas_int,
            self.canvas_comp,
            self.canvas_roll,
        ):
            canv.config(width=w, height=h)

    # ── Tab 1: Edit Grid ───────────────────────────────────────────────────

    def _build_tab_grid(self) -> None:
        side = ttk.Frame(self.tab_grid)
        side.pack(side="right", fill="y", padx=8, pady=8)

        ttk.Label(side, text="Edit Grid", font=("Helvetica", 13, "bold")).pack(anchor="w", pady=(0, 6))
        ttk.Label(
            side,
            text="Left-click a cell:\nREGULAR → FAILURE → SUCCESS\nBottom-right toggles SUCCESS ↔ FAILURE.",
            justify="left",
        ).pack(anchor="w", pady=(0, 12))

        ttk.Button(side, text="Resize Grid…", command=self._resize_grid_dialog).pack(fill="x", pady=2)
        ttk.Button(side, text="Reset to Default Grid", command=self._reset_default_grid).pack(
            fill="x", pady=2
        )

        ttk.Separator(side, orient="horizontal").pack(fill="x", pady=8)
        ttk.Button(side, text="Save to examples…", command=self._save_to_examples_dialog).pack(
            fill="x", pady=2
        )
        ttk.Button(side, text="Save State…", command=self._save_state_dialog).pack(fill="x", pady=2)
        ttk.Button(side, text="Load State…", command=self._load_state_dialog).pack(fill="x", pady=2)

        ttk.Separator(side, orient="horizontal").pack(fill="x", pady=8)
        ttk.Checkbutton(
            side, text="Show visitation overlay", variable=self._show_visitation, command=self._render_for_tab
        ).pack(anchor="w")

        self.canvas_grid.bind("<Button-1>", self._on_click_grid_edit)

    def _on_click_grid_edit(self, event) -> None:
        cell = self._canvas_to_cell(event.x, event.y)
        if cell is None:
            return
        r, c = cell
        is_br = (r == self.state.grid.height - 1) and (c == self.state.grid.width - 1)
        nt = NodeType(int(self.state.grid.node_types[r, c]))
        if is_br:
            # Only SUCCESS / FAILURE allowed
            nt = NodeType.FAILURE if nt == NodeType.SUCCESS else NodeType.SUCCESS
        else:
            order = [NodeType.REGULAR, NodeType.FAILURE, NodeType.SUCCESS]
            nt = order[(order.index(nt) + 1) % 3]
        self.state.grid.node_types[r, c] = int(nt)
        self._invalidate_catalog()
        self._render_for_tab()
        self.status_var.set(f"Cell ({r},{c}) → {nt.name}")

    def _resize_grid_dialog(self) -> None:
        h = simpledialog.askinteger(
            "Resize Grid", "Height:", initialvalue=self.state.grid.height, minvalue=2, maxvalue=12
        )
        if h is None:
            return
        w = simpledialog.askinteger(
            "Resize Grid", "Width:", initialvalue=self.state.grid.width, minvalue=2, maxvalue=12
        )
        if w is None:
            return
        new_nt = np.zeros((h, w), dtype=np.int8)
        # Copy overlap.
        hh = min(h, self.state.grid.height)
        ww = min(w, self.state.grid.width)
        new_nt[:hh, :ww] = self.state.grid.node_types[:hh, :ww]
        # Bottom-right MUST be SUCCESS/FAILURE; if it's REGULAR after copy, set to SUCCESS.
        if NodeType(int(new_nt[h - 1, w - 1])) == NodeType.REGULAR:
            new_nt[h - 1, w - 1] = int(NodeType.SUCCESS)
        self.state.grid = Grid(height=h, width=w, node_types=new_nt)
        # Drop interventions/compositions/overrides that may have referenced
        # out-of-bounds cells.
        self.state.interventions = [iv for iv in self.state.interventions if self._traj_in_bounds(iv["path"])]
        self.state.base_trajectories = [
            t for t in self.state.base_trajectories if self._traj_in_bounds(t["path"])
        ]
        self.state.compositions = []  # safest reset
        self._invalidate_catalog()
        self._resize_all_canvases()
        self._refresh_trajectory_listbox("base")
        self._refresh_trajectory_listbox("interventions")
        self._refresh_comp_dropdowns()
        self._refresh_rollout_dropdown()
        self._render_for_tab()
        self.status_var.set(f"Grid resized to {h}x{w}")

    def _reset_default_grid(self) -> None:
        if not messagebox.askyesno(
            "Reset Grid", "Reset to default 4×4 grid? (Base demos / interventions / compositions wiped.)"
        ):
            return
        self.state = default_state()
        self._invalidate_catalog()
        self._resize_all_canvases()
        self._refresh_trajectory_listbox("base")
        self._refresh_trajectory_listbox("interventions")
        self._refresh_comp_dropdowns()
        self._refresh_rollout_dropdown()
        self._render_for_tab()
        self.status_var.set("Grid reset.")

    def _traj_in_bounds(self, path: list[tuple[int, int]]) -> bool:
        return all(0 <= r < self.state.grid.height and 0 <= c < self.state.grid.width for r, c in path)

    @staticmethod
    def _examples_dir() -> Path:
        return Path(__file__).parent / "examples"

    def _save_state_dialog(self) -> None:
        p = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
            initialdir=str(self._examples_dir()),
        )
        if not p:
            return
        save_state(p, self.state)
        self.status_var.set(f"Saved → {p}")

    def _save_to_examples_dialog(self) -> None:
        """Quick save to `examples/<name>.json`. Just asks for a name —
        no path browsing, no extension required. Overwrites with confirmation
        if the file already exists."""
        name = simpledialog.askstring(
            "Save to examples",
            "File name (no extension):",
            initialvalue="default_4x4",
        )
        if not name:
            return
        name = name.strip()
        if name.endswith(".json"):
            name = name[:-5]
        if not name:
            self.status_var.set("Save cancelled — empty name.")
            return
        examples_dir = self._examples_dir()
        examples_dir.mkdir(parents=True, exist_ok=True)
        path = examples_dir / f"{name}.json"
        if path.exists():
            if not messagebox.askyesno(
                "Overwrite?",
                f"{path.name} already exists in examples/. Overwrite?",
            ):
                self.status_var.set("Save cancelled.")
                return
        try:
            save_state(path, self.state)
        except Exception as e:
            messagebox.showerror("Save failed", str(e))
            return
        self.status_var.set(f"Saved → {path}")

    def _load_state_dialog(self) -> None:
        p = filedialog.askopenfilename(
            filetypes=[("JSON", "*.json")],
            initialdir=str(self._examples_dir()),
        )
        if not p:
            return
        try:
            self.state = load_state(p)
        except Exception as e:
            messagebox.showerror("Load failed", str(e))
            return
        self._invalidate_catalog()
        self._resize_all_canvases()
        self._refresh_trajectory_listbox("base")
        self._refresh_trajectory_listbox("interventions")
        self._refresh_comp_dropdowns()
        self._refresh_rollout_dropdown()
        self._render_for_tab()
        self.status_var.set(f"Loaded ← {p}")

    # ── Tab 2: Edit Base Trajectories ──────────────────────────────────────

    def _build_tab_policy(self) -> None:
        side = ttk.Frame(self.tab_policy)
        side.pack(side="right", fill="y", padx=8, pady=8)

        ttk.Label(side, text="Base Policy from Demos", font=("Helvetica", 13, "bold")).pack(
            anchor="w", pady=(0, 6)
        )
        ttk.Label(
            side,
            text="Draw a few demos (ideally reaching success).\n"
            "The base policy = empirical action distribution\n"
            "from these demos at visited cells, falling back\n"
            "to uniform-over-valid actions at unvisited cells.",
            justify="left",
        ).pack(anchor="w", pady=(0, 8))

        ttk.Button(side, text="New base demo", command=lambda: self._start_new_trajectory("base")).pack(
            fill="x", pady=2
        )
        self._base_finish_btn = ttk.Button(
            side, text="Finish demo", command=self._finish_trajectory, state="disabled"
        )
        self._base_finish_btn.pack(fill="x", pady=2)
        ttk.Button(side, text="Cancel draw", command=self._cancel_draw).pack(fill="x", pady=2)

        ttk.Separator(side, orient="horizontal").pack(fill="x", pady=8)
        ttk.Label(side, text="Base demos:").pack(anchor="w")
        self.base_listbox = tk.Listbox(side, width=28, height=10)
        self.base_listbox.pack(fill="x", pady=2)
        ttk.Button(
            side, text="Delete selected", command=lambda: self._delete_selected_trajectory("base")
        ).pack(fill="x", pady=2)

        self.canvas_policy.bind("<Button-1>", self._on_click_trajectory)
        self.canvas_policy.bind("<Button-3>", lambda _e: self._cancel_draw())

        self._refresh_trajectory_listbox("base")

    def _render_policy_tab(self) -> None:
        # Build the current base policy (from base_trajectories) and render it.
        try:
            base = self._catalog().get("base")
        except Exception as e:
            self.status_var.set(f"Base policy build failed: {e}")
            base = None
        trajs: list[tuple[list[tuple[int, int]], str]] = []
        for i, t in enumerate(self.state.base_trajectories):
            color = TRAJ_COLORS[i % len(TRAJ_COLORS)]
            trajs.append((t["path"], color))
        # Show in-progress draft only if it's for THIS collection.
        if self._draw_state.get("collection") == "base" and self._draw_state.get("path"):
            trajs.append((self._draw_state["path"], self._draw_state.get("color") or "#888"))
        self._draw_base_grid(self.canvas_policy, policy=base, trajectories=trajs)

    # ── Tab 3: Edit Interventions ──────────────────────────────────────────

    def _build_tab_interventions(self) -> None:
        side = ttk.Frame(self.tab_int)
        side.pack(side="right", fill="y", padx=8, pady=8)

        ttk.Label(side, text="Interventions", font=("Helvetica", 13, "bold")).pack(anchor="w", pady=(0, 6))
        ttk.Label(
            side,
            text="Expert demos to add on top of the base policy.\n"
            "Intervention policy = empirical at visited cells,\n"
            "falls back to base at unvisited cells.",
            justify="left",
        ).pack(anchor="w", pady=(0, 8))

        # Shared draw state across both trajectory tabs.
        self._draw_state: dict[str, Any] = {"path": None, "color": None, "collection": None}

        ttk.Button(
            side, text="New trajectory", command=lambda: self._start_new_trajectory("interventions")
        ).pack(fill="x", pady=2)
        self._finish_btn = ttk.Button(
            side, text="Finish trajectory", command=self._finish_trajectory, state="disabled"
        )
        self._finish_btn.pack(fill="x", pady=2)
        ttk.Button(side, text="Cancel draw", command=self._cancel_draw).pack(fill="x", pady=2)

        ttk.Separator(side, orient="horizontal").pack(fill="x", pady=8)
        ttk.Label(side, text="Committed trajectories:").pack(anchor="w")
        self.int_listbox = tk.Listbox(side, width=28, height=10)
        self.int_listbox.pack(fill="x", pady=2)
        ttk.Button(
            side, text="Delete selected", command=lambda: self._delete_selected_trajectory("interventions")
        ).pack(fill="x", pady=2)

        self.canvas_int.bind("<Button-1>", self._on_click_trajectory)
        self.canvas_int.bind("<Button-3>", lambda _e: self._cancel_draw())

        self._refresh_trajectory_listbox("interventions")

    # ── Shared trajectory editing (used by Tab 2 + Tab 3) ──────────────────

    def _get_collection(self, collection: str) -> list[dict]:
        if collection == "base":
            return self.state.base_trajectories
        if collection == "interventions":
            return self.state.interventions
        raise ValueError(f"Unknown trajectory collection: {collection!r}")

    def _get_finish_btn(self, collection: str):
        return self._base_finish_btn if collection == "base" else self._finish_btn

    def _start_new_trajectory(self, collection: str) -> None:
        coll_list = self._get_collection(collection)
        idx = len(coll_list)
        # Start with an EMPTY path — first click picks the start cell. This
        # lets the user draw HG-DAgger rescue interventions that begin where
        # the human took over (not necessarily (0,0)).
        self._draw_state["path"] = []
        self._draw_state["color"] = TRAJ_COLORS[idx % len(TRAJ_COLORS)]
        self._draw_state["collection"] = collection
        self._get_finish_btn(collection).config(state="disabled")
        label = "base demo" if collection == "base" else "intervention trajectory"
        self.status_var.set(
            f"Drawing {label}: first click picks the START cell (any non-terminal), "
            f"then click adjacent cells (right/down) to extend. Right-click to cancel."
        )
        self._render_for_tab()

    def _cancel_draw(self) -> None:
        coll = self._draw_state.get("collection")
        self._draw_state["path"] = None
        self._draw_state["color"] = None
        self._draw_state["collection"] = None
        if coll:
            self._get_finish_btn(coll).config(state="disabled")
        self.status_var.set("Draw cancelled.")
        self._render_for_tab()

    def _on_click_trajectory(self, event) -> None:
        path: list[tuple[int, int]] | None = self._draw_state.get("path")
        coll: str | None = self._draw_state.get("collection")
        if path is None or coll is None:
            self.status_var.set("Click 'New trajectory' / 'New base demo' first.")
            return
        cell = self._canvas_to_cell(event.x, event.y)
        if cell is None:
            return
        # First click: set the START cell (any non-terminal cell).
        if not path:
            r, c = cell
            if self.state.grid.is_terminal(r, c):
                self.status_var.set(f"Can't start at terminal {(r, c)}. Pick a non-terminal cell.")
                return
            path.append((r, c))
            label = "base demo" if coll == "base" else "intervention"
            if (r, c) != (0, 0):
                self.status_var.set(
                    f"{label} starts at ({r},{c}) — HG-DAgger style. "
                    f"Click adjacent cells (right/down) to extend."
                )
            else:
                self.status_var.set(f"{label} starts at (0,0). Click adjacent cells (right/down) to extend.")
            self._render_for_tab()
            return
        # Subsequent clicks: extend.
        cur_r, cur_c = path[-1]
        if self.state.grid.is_terminal(cur_r, cur_c):
            self.status_var.set("Already at terminal — click Finish.")
            return
        r, c = cell
        if (
            (r, c) == (cur_r, cur_c + 1)
            and Action.RIGHT in self.state.grid.outgoing(cur_r, cur_c)
            or (r, c) == (cur_r + 1, cur_c)
            and Action.DOWN in self.state.grid.outgoing(cur_r, cur_c)
        ):
            pass
        else:
            self.status_var.set(f"Invalid move {(cur_r, cur_c)} → ({r},{c}). Right or Down only.")
            return
        path.append((r, c))
        # Finish enables once the path has at least one forward step (≥2 cells).
        if len(path) >= 2:
            self._get_finish_btn(coll).config(state="normal")
        if self.state.grid.is_terminal(r, c):
            self.status_var.set("Reached terminal — click Finish to commit.")
        else:
            self.status_var.set(
                f"Drew {(cur_r, cur_c)} → ({r},{c}). Click Finish anytime to commit a partial path."
            )
        self._render_for_tab()

    def _finish_trajectory(self) -> None:
        path: list[tuple[int, int]] | None = self._draw_state.get("path")
        coll: str | None = self._draw_state.get("collection")
        if not path or len(path) < 2 or coll is None:
            self.status_var.set("Draw at least one step before finishing.")
            return
        ended_terminal = self.state.grid.is_terminal(*path[-1])
        coll_list = self._get_collection(coll)
        n = len(coll_list) + 1
        prefix = "base_demo" if coll == "base" else "traj"
        name = f"{prefix}_{n}" if ended_terminal else f"{prefix}_{n}_partial"
        coll_list.append({"name": name, "path": list(path)})
        self._draw_state["path"] = None
        self._draw_state["color"] = None
        self._draw_state["collection"] = None
        self._get_finish_btn(coll).config(state="disabled")
        self._invalidate_catalog()
        self._refresh_trajectory_listbox(coll)
        self._render_for_tab()
        self.status_var.set(f"Committed {name} ({len(path)} cells, ends at {path[-1]}).")

    def _delete_selected_trajectory(self, collection: str) -> None:
        listbox = self.base_listbox if collection == "base" else self.int_listbox
        sel = listbox.curselection()
        if not sel:
            return
        i = sel[0]
        coll_list = self._get_collection(collection)
        removed = coll_list.pop(i)
        self._invalidate_catalog()
        self._refresh_trajectory_listbox(collection)
        self._render_for_tab()
        self.status_var.set(f"Removed {removed['name']}.")

    def _refresh_trajectory_listbox(self, collection: str) -> None:
        listbox = self.base_listbox if collection == "base" else self.int_listbox
        coll_list = self._get_collection(collection)
        listbox.delete(0, "end")
        for i, t in enumerate(coll_list):
            color = TRAJ_COLORS[i % len(TRAJ_COLORS)]
            listbox.insert("end", f"● {t['name']}  ({len(t['path'])} cells)  [{color}]")

    # ── Tab 3: Compose Policy ──────────────────────────────────────────────

    def _build_tab_compose(self) -> None:
        side = ttk.Frame(self.tab_comp)
        side.pack(side="right", fill="y", padx=8, pady=8)

        ttk.Label(side, text="Compose Policy", font=("Helvetica", 13, "bold")).pack(anchor="w", pady=(0, 6))

        ttk.Label(
            side, text="Rows: (policy, weight). BC-correct\nvisitation-weighted mix.", justify="left"
        ).pack(anchor="w", pady=(0, 6))

        # Builder rows
        self._comp_rows: list[dict[str, Any]] = []
        self._comp_rows_frame = ttk.Frame(side)
        self._comp_rows_frame.pack(fill="x")

        ttk.Button(side, text="Add row", command=lambda: self._add_comp_row()).pack(fill="x", pady=2)
        ttk.Button(side, text="Add blend (slider)", command=self._add_blend_slider).pack(fill="x", pady=2)
        ttk.Button(side, text="Clear rows", command=self._clear_comp_rows).pack(fill="x", pady=2)

        ttk.Separator(side, orient="horizontal").pack(fill="x", pady=8)
        ttk.Label(side, text="Name:").pack(anchor="w")
        self._comp_name_var = tk.StringVar(value="blend_0.4")
        ttk.Entry(side, textvariable=self._comp_name_var).pack(fill="x")
        ttk.Button(side, text="Save composition", command=self._save_composition).pack(fill="x", pady=4)

        ttk.Separator(side, orient="horizontal").pack(fill="x", pady=8)
        ttk.Label(side, text="Saved compositions:").pack(anchor="w")
        self.comp_listbox = tk.Listbox(side, width=28, height=6)
        self.comp_listbox.pack(fill="x", pady=2)
        ttk.Button(side, text="Delete selected", command=self._delete_selected_composition).pack(
            fill="x", pady=2
        )

        self._comp_success_var = tk.StringVar(value="(no preview yet)")
        ttk.Label(side, textvariable=self._comp_success_var, foreground="#1a73e8").pack(anchor="w", pady=4)

        self._refresh_comp_dropdowns()

    def _add_comp_row(self, policy: str = "base", weight: float = 0.5) -> None:
        row_frame = ttk.Frame(self._comp_rows_frame)
        row_frame.pack(fill="x", pady=1)
        pol_var = tk.StringVar(value=policy)
        wt_var = tk.DoubleVar(value=weight)
        names = self._available_policy_names()
        ddl = ttk.Combobox(row_frame, textvariable=pol_var, values=names, width=14, state="readonly")
        ddl.pack(side="left")
        slider = ttk.Scale(row_frame, from_=0.0, to=1.0, variable=wt_var, orient="horizontal", length=120)
        slider.pack(side="left", padx=4)
        wt_label = ttk.Label(row_frame, text=f"{weight:.2f}", width=5)
        wt_label.pack(side="left")
        row = {"frame": row_frame, "policy": pol_var, "weight": wt_var, "label": wt_label, "combobox": ddl}

        def on_change(*_a):
            wt_label.config(text=f"{wt_var.get():.2f}")
            self._render_for_tab()

        pol_var.trace_add("write", on_change)
        wt_var.trace_add("write", on_change)

        def remove():
            row_frame.destroy()
            self._comp_rows.remove(row)
            self._render_for_tab()

        ttk.Button(row_frame, text="✕", width=2, command=remove).pack(side="right")
        self._comp_rows.append(row)
        self._render_for_tab()

    def _add_blend_slider(self) -> None:
        # Convenience: clear rows and add `base` + `intervention` coupled by
        # one slider. Row 1 is `base` so the slider value reads as the blend
        # ratio in the DAgger convention: 0.0 = pure intervention, 1.0 = pure
        # base.
        #
        # NOTE: this composition is resolved via `Policy.weighted_sum`
        # (BC-correct visitation-weighted), not `Policy.blend` (simple per-cell
        # mix). The two operations model different things — see README's
        # "Math: two distinct mixing operations" section. For a simple per-cell
        # blend, use the CLI: `--sweep_blend` or `Policy.blend(...)` directly.
        self._clear_comp_rows()
        self._add_comp_row("base", 0.4)
        self._add_comp_row("intervention", 0.6)
        # Bind the two weights so they sum to 1.
        a, b = self._comp_rows[0], self._comp_rows[1]
        a_lock = {"v": False}

        def couple_a(*_):
            if a_lock["v"]:
                return
            a_lock["v"] = True
            try:
                b["weight"].set(1.0 - a["weight"].get())
            finally:
                a_lock["v"] = False

        def couple_b(*_):
            if a_lock["v"]:
                return
            a_lock["v"] = True
            try:
                a["weight"].set(1.0 - b["weight"].get())
            finally:
                a_lock["v"] = False

        a["weight"].trace_add("write", couple_a)
        b["weight"].trace_add("write", couple_b)

    def _clear_comp_rows(self) -> None:
        for row in list(self._comp_rows):
            row["frame"].destroy()
        self._comp_rows.clear()
        self._render_for_tab()

    def _available_policy_names(self) -> list[str]:
        names = ["base", "intervention"]
        names.extend(c["name"] for c in self.state.compositions)
        return names

    def _refresh_comp_dropdowns(self) -> None:
        names = self._available_policy_names()
        for row in self._comp_rows:
            cb = row["combobox"]
            cb.config(values=names)
        # Listbox
        self.comp_listbox.delete(0, "end")
        for c in self.state.compositions:
            summary = " + ".join(f"{r['weight']:.2f}·{r['policy']}" for r in c["rows"])
            self.comp_listbox.insert("end", f"{c['name']}:  {summary}")

    def _save_composition(self) -> None:
        if not self._comp_rows:
            messagebox.showinfo("No rows", "Add at least one row.")
            return
        name = self._comp_name_var.get().strip()
        if not name:
            messagebox.showinfo("Name", "Give the composition a name.")
            return
        if name in {"base", "intervention"}:
            messagebox.showerror("Reserved name", f"'{name}' is reserved.")
            return
        comp = {
            "name": name,
            "rows": [
                {"policy": r["policy"].get(), "weight": float(r["weight"].get())} for r in self._comp_rows
            ],
        }
        # Overwrite existing by same name.
        self.state.compositions = [c for c in self.state.compositions if c["name"] != name]
        self.state.compositions.append(comp)
        self._invalidate_catalog()
        self._refresh_comp_dropdowns()
        self._refresh_rollout_dropdown()
        self.status_var.set(f"Saved composition '{name}'.")

    def _delete_selected_composition(self) -> None:
        sel = self.comp_listbox.curselection()
        if not sel:
            return
        i = sel[0]
        removed = self.state.compositions.pop(i)
        self._invalidate_catalog()
        self._refresh_comp_dropdowns()
        self._refresh_rollout_dropdown()
        self.status_var.set(f"Removed composition '{removed['name']}'.")

    def _current_compose_policy(self) -> Policy | None:
        if not self._comp_rows:
            return None
        try:
            cat = self._catalog()
        except Exception as e:
            self.status_var.set(f"Composition error: {e}")
            return None
        weighted = []
        for row in self._comp_rows:
            name = row["policy"].get()
            w = float(row["weight"].get())
            if name not in cat or w <= 0:
                continue
            weighted.append((w, cat[name]))
        if not weighted:
            return None
        try:
            return Policy.weighted_sum(self.state.grid, weighted)
        except Exception as e:
            self.status_var.set(f"Composition error: {e}")
            return None

    # ── Tab 4: Rollouts ────────────────────────────────────────────────────

    def _build_tab_rollouts(self) -> None:
        side = ttk.Frame(self.tab_roll)
        side.pack(side="right", fill="y", padx=8, pady=8)

        ttk.Label(side, text="Rollouts", font=("Helvetica", 13, "bold")).pack(anchor="w", pady=(0, 6))

        ttk.Label(side, text="Policy:").pack(anchor="w", pady=(6, 0))
        self._roll_policy_var = tk.StringVar(value="base")
        self.roll_dropdown = ttk.Combobox(
            side,
            textvariable=self._roll_policy_var,
            values=self._available_policy_names(),
            state="readonly",
            width=22,
        )
        self.roll_dropdown.pack(fill="x")
        self._roll_policy_var.trace_add("write", lambda *_a: self._reset_rollout_stats())

        ttk.Label(side, text="Seed (optional):").pack(anchor="w", pady=(6, 0))
        self._roll_seed_var = tk.StringVar(value="")
        ttk.Entry(side, textvariable=self._roll_seed_var, width=10).pack(anchor="w")

        # Policy noise: softens action probs toward uniform-over-valid-actions
        # before rollout. Exposes the compounding-error / DAgger-motivation story.
        ttk.Label(side, text="Policy noise ε:").pack(anchor="w", pady=(6, 0))
        self._policy_noise_var = tk.DoubleVar(value=0.0)
        ttk.Scale(
            side, from_=0.0, to=1.0, variable=self._policy_noise_var, orient="horizontal", length=180
        ).pack(fill="x")
        self._policy_noise_label = ttk.Label(side, text="ε = 0.00")
        self._policy_noise_label.pack(anchor="w")
        self._policy_noise_var.trace_add(
            "write",
            lambda *_: self._policy_noise_label.config(text=f"ε = {float(self._policy_noise_var.get()):.2f}"),
        )

        ttk.Separator(side, orient="horizontal").pack(fill="x", pady=8)
        ttk.Label(side, text="Step-through", font=("Helvetica", 11, "bold")).pack(anchor="w")
        ttk.Button(side, text="Next rollout", command=self._do_one_rollout).pack(fill="x", pady=2)
        self._auto_play_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            side, text="Auto-play", variable=self._auto_play_var, command=self._toggle_autoplay
        ).pack(anchor="w")
        ttk.Label(side, text="Speed (s/rollout):").pack(anchor="w")
        self._auto_speed = tk.DoubleVar(value=0.5)
        ttk.Scale(side, from_=0.05, to=2.0, variable=self._auto_speed, orient="horizontal").pack(fill="x")

        self._roll_tally_var = tk.StringVar(value="N=0 | succ=0 | fail=0 | succ_rate=0.000")
        ttk.Label(
            side, textvariable=self._roll_tally_var, foreground="#1a73e8", font=("Helvetica", 11, "bold")
        ).pack(anchor="w", pady=6)
        ttk.Button(side, text="Reset stats", command=self._reset_rollout_stats).pack(fill="x")

        ttk.Separator(side, orient="horizontal").pack(fill="x", pady=8)
        ttk.Label(side, text="Batch evaluation", font=("Helvetica", 11, "bold")).pack(anchor="w")
        ttk.Label(side, text="N rollouts:").pack(anchor="w")
        self._batch_n_var = tk.IntVar(value=10000)
        ttk.Entry(side, textvariable=self._batch_n_var, width=12).pack(anchor="w")
        ttk.Button(side, text="Run batch", command=self._do_batch).pack(fill="x", pady=2)

        self._batch_result_var = tk.StringVar(value="(no batch yet)")
        ttk.Label(side, textvariable=self._batch_result_var, justify="left").pack(anchor="w", pady=4)

        # State for stepping
        self._roll_stats = {"n": 0, "succ": 0, "fail": 0}
        self._last_traj: list[tuple[int, int]] = []
        self._last_outcome: NodeType | None = None
        self._autoplay_after: str | None = None

    def _refresh_rollout_dropdown(self) -> None:
        names = self._available_policy_names()
        self.roll_dropdown.config(values=names)
        if self._roll_policy_var.get() not in names:
            self._roll_policy_var.set(names[0])

    def _reset_rollout_stats(self) -> None:
        self._roll_stats = {"n": 0, "succ": 0, "fail": 0}
        self._last_traj = []
        self._last_outcome = None
        self._roll_tally_var.set("N=0 | succ=0 | fail=0 | succ_rate=0.000")
        self._render_for_tab()

    def _rng_from_seed(self) -> np.random.Generator:
        s = self._roll_seed_var.get().strip()
        if not s:
            return np.random.default_rng()
        try:
            return np.random.default_rng(int(s))
        except ValueError:
            return np.random.default_rng()

    def _do_one_rollout(self) -> None:
        cat = self._catalog()
        name = self._roll_policy_var.get()
        if name not in cat:
            self.status_var.set(f"No policy '{name}'.")
            return
        rng = self._rng_from_seed()
        pol = self._maybe_soften(cat[name])
        path, outcome = rollout(self.state.grid, pol, rng)
        self._last_traj = path
        self._last_outcome = outcome
        self._roll_stats["n"] += 1
        if outcome == NodeType.SUCCESS:
            self._roll_stats["succ"] += 1
        elif outcome == NodeType.FAILURE:
            self._roll_stats["fail"] += 1
        n = max(self._roll_stats["n"], 1)
        self._roll_tally_var.set(
            f"N={self._roll_stats['n']} | succ={self._roll_stats['succ']} "
            f"| fail={self._roll_stats['fail']} | succ_rate={self._roll_stats['succ'] / n:.3f}"
        )
        self._render_for_tab()

    def _toggle_autoplay(self) -> None:
        if self._auto_play_var.get():
            self._schedule_autoplay()
        else:
            if self._autoplay_after is not None:
                self.root.after_cancel(self._autoplay_after)
                self._autoplay_after = None

    def _schedule_autoplay(self) -> None:
        if not self._auto_play_var.get():
            return
        self._do_one_rollout()
        ms = int(max(self._auto_speed.get(), 0.05) * 1000)
        self._autoplay_after = self.root.after(ms, self._schedule_autoplay)

    def _do_batch(self) -> None:
        cat = self._catalog()
        name = self._roll_policy_var.get()
        if name not in cat:
            return
        n = int(self._batch_n_var.get())
        rng = self._rng_from_seed()
        pol = self._maybe_soften(cat[name])
        result = evaluate(self.state.grid, pol, n, rng)
        eps = float(self._policy_noise_var.get())
        noise_tag = f" (ε={eps:.2f})" if eps > 0 else ""
        self._batch_result_var.set(
            f"N={result['n']}{noise_tag}\nsucc={result['succ']} ({result['succ_rate']:.3f})\n"
            f"fail={result['fail']} ({result['fail_rate']:.3f})"
        )
        self.status_var.set(f"Batch: {name}{noise_tag} → succ_rate={result['succ_rate']:.3f}")

    def _maybe_soften(self, pol: Policy) -> Policy:
        """Apply policy_noise softening if the slider is non-zero."""
        eps = float(self._policy_noise_var.get())
        return pol.softened(self.state.grid, eps) if eps > 0 else pol

    # ── Catalog ────────────────────────────────────────────────────────────

    def _catalog(self) -> dict[str, Policy]:
        if self._catalog_cache is None:
            self._catalog_cache = build_policy_catalog(self.state)
        return self._catalog_cache

    def _invalidate_catalog(self) -> None:
        self._catalog_cache = None

    # ── Render dispatch ────────────────────────────────────────────────────

    def _render_for_tab(self) -> None:
        i = self.notebook.index(self.notebook.select())
        if i == 0:
            self._render_grid_tab()
        elif i == 1:
            self._render_policy_tab()
        elif i == 2:
            self._render_intervention_tab()
        elif i == 3:
            self._render_compose_tab()
        elif i == 4:
            self._render_rollout_tab()

    def _draw_base_grid(
        self,
        canvas: tk.Canvas,
        policy: Policy | None = None,
        trajectories: list[tuple[list[tuple[int, int]], str]] | None = None,
        highlight: tuple[int, int] | None = None,
    ) -> None:
        canvas.delete("all")
        # Visitation overlay (under nodes/edges).
        if policy is not None and self._show_visitation.get():
            for r in range(self.state.grid.height):
                for c in range(self.state.grid.width):
                    v = float(policy.visitation[r, c])
                    if v <= 0:
                        continue
                    x, y = self._cell_center(r, c)
                    halo = NODE_RADIUS + 14
                    # opacity simulated by gray-level (no real alpha in tk)
                    g_byte = max(180, int(255 - 75 * v))
                    color = f"#{g_byte:02x}{g_byte:02x}ff"
                    canvas.create_oval(x - halo, y - halo, x + halo, y + halo, outline="", fill=color)

        # Edges (right/down).
        for r in range(self.state.grid.height):
            for c in range(self.state.grid.width):
                if self.state.grid.is_terminal(r, c):
                    continue
                for a in self.state.grid.outgoing(r, c):
                    self._draw_edge(canvas, r, c, a, policy)

        # Nodes.
        for r in range(self.state.grid.height):
            for c in range(self.state.grid.width):
                self._draw_node(canvas, r, c)

        # Trajectories.
        if trajectories:
            for path, color in trajectories:
                self._draw_trajectory(canvas, path, color)

        # Highlight.
        if highlight is not None:
            r, c = highlight
            x, y = self._cell_center(r, c)
            canvas.create_oval(
                x - NODE_RADIUS - 6,
                y - NODE_RADIUS - 6,
                x + NODE_RADIUS + 6,
                y + NODE_RADIUS + 6,
                outline="#fbbc05",
                width=3,
            )

    def _draw_node(self, canvas: tk.Canvas, r: int, c: int) -> None:
        x, y = self._cell_center(r, c)
        nt = NodeType(int(self.state.grid.node_types[r, c]))
        canvas.create_oval(
            x - NODE_RADIUS,
            y - NODE_RADIUS,
            x + NODE_RADIUS,
            y + NODE_RADIUS,
            fill=NODE_COLOR[nt],
            outline=NODE_OUTLINE,
            width=2,
        )
        if r == 0 and c == 0:
            canvas.create_text(
                x - NODE_RADIUS - 8,
                y - NODE_RADIUS - 8,
                text="S",
                fill="#1a73e8",
                font=("Helvetica", 12, "bold"),
            )
        # Cell coords label (faint)
        canvas.create_text(x, y + NODE_RADIUS + 12, text=f"({r},{c})", fill="#888", font=("Helvetica", 8))

    def _draw_edge(self, canvas: tk.Canvas, r: int, c: int, a: Action, policy: Policy | None) -> None:
        x0, y0 = self._cell_center(r, c)
        if a == Action.RIGHT:
            x1, y1 = self._cell_center(r, c + 1)
        else:
            x1, y1 = self._cell_center(r + 1, c)
        # Shorten endpoints so arrow doesn't overlap node circles.
        dx, dy = x1 - x0, y1 - y0
        length = (dx * dx + dy * dy) ** 0.5
        ux, uy = dx / length, dy / length
        sx0 = x0 + NODE_RADIUS * ux
        sy0 = y0 + NODE_RADIUS * uy
        sx1 = x1 - NODE_RADIUS * ux
        sy1 = y1 - NODE_RADIUS * uy
        # Width / color
        if policy is not None:
            p = float(policy.probs[r, c, int(a)])
            width = 2 + 8 * p
            color = EDGE_POLICY
            stipple = "" if p > 0.05 else "gray50"
        else:
            width = 2
            color = EDGE_DEFAULT
            stipple = ""
        canvas.create_line(
            sx0, sy0, sx1, sy1, arrow="last", arrowshape=(12, 14, 5), width=width, fill=color, stipple=stipple
        )
        if policy is not None:
            # Show probability mid-edge.
            mx, my = (sx0 + sx1) / 2, (sy0 + sy1) / 2
            canvas.create_text(
                mx,
                my - 8,
                text=f"{policy.probs[r, c, int(a)]:.2f}",
                fill=color,
                font=("Helvetica", 9, "bold"),
            )

    def _draw_trajectory(self, canvas: tk.Canvas, path: list[tuple[int, int]], color: str) -> None:
        if len(path) < 2:
            return
        for i in range(len(path) - 1):
            x0, y0 = self._cell_center(*path[i])
            x1, y1 = self._cell_center(*path[i + 1])
            canvas.create_line(x0, y0, x1, y1, width=4, fill=color, capstyle="round")

    # ── Per-tab render ─────────────────────────────────────────────────────

    def _render_grid_tab(self) -> None:
        self._draw_base_grid(self.canvas_grid)

    def _render_intervention_tab(self) -> None:
        trajs: list[tuple[list[tuple[int, int]], str]] = []
        for i, iv in enumerate(self.state.interventions):
            color = TRAJ_COLORS[i % len(TRAJ_COLORS)]
            trajs.append((iv["path"], color))
        # Show in-progress draft only when it belongs to THIS collection.
        if self._draw_state.get("collection") == "interventions" and self._draw_state.get("path"):
            trajs.append((self._draw_state["path"], self._draw_state.get("color") or "#888"))
        self._draw_base_grid(self.canvas_int, trajectories=trajs)

    def _render_compose_tab(self) -> None:
        pol = self._current_compose_policy()
        self._draw_base_grid(self.canvas_comp, policy=pol)
        # Live success rate estimate (1000 rollouts) — fast for small grids.
        if pol is not None:
            rng = np.random.default_rng(0)
            r = evaluate(self.state.grid, pol, 1000, rng)
            self._comp_success_var.set(f"succ_rate ≈ {r['succ_rate']:.3f}  (1000 rollouts, seed=0)")
        else:
            self._comp_success_var.set("(add rows to preview)")

    def _render_rollout_tab(self) -> None:
        cat = self._catalog()
        name = self._roll_policy_var.get()
        pol = cat.get(name)
        last_traj_pair = []
        if self._last_traj:
            color = "#34a853" if self._last_outcome == NodeType.SUCCESS else "#ea4335"
            last_traj_pair.append((self._last_traj, color))
        highlight = self._last_traj[-1] if self._last_traj else None
        self._draw_base_grid(self.canvas_roll, policy=pol, trajectories=last_traj_pair, highlight=highlight)


def launch(state: SimState | None = None) -> None:
    """Open the GUI. Blocks until the window closes."""
    if state is None:
        state = default_state()
    root = tk.Tk()
    app = GridworldSimApp(root, state)
    root.mainloop()


if __name__ == "__main__":
    launch()
