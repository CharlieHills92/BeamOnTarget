"""Particles tab — multi-beam source configuration with row-based pairing."""
import os
import glob
import tkinter as tk
from tkinter import ttk

from gui.gui_widgets import (make_card, _SCRIPT_DIR,
                              resolve_path as _resolve_path, browse_directory,
                              get_project_folder as _get_project_folder)

# Default beam definitions used when BEAM_SOURCES is absent from config.
_DEFAULT_BEAM_SOURCES = [
    {"label": "DNB",  "directory": "..\\BEAM_CONFIGS\\DNB",
     "transform": {"translation_m": [-11.410436, -26.617882, -0.920], "rotation_z_deg": 116.0}},
    {"label": "HNB1", "directory": "..\\BEAM_CONFIGS\\HNB1",
     "transform": {"translation_m": [0.57723, -32.385248, -1.453462], "rotation_z_deg": 79.543}},
    {"label": "HNB2", "directory": "..\\BEAM_CONFIGS\\HNB2",
     "transform": {"translation_m": [0.0, 0.0, 0.0], "rotation_z_deg": 0.0}},
]


class ParticlesTab(ttk.Frame):
    """Multi-beam source parameters and per-beam .bl file browsers.

    Files are paired by row index across columns:
      row 0 of DNB + row 0 of HNB1 + row 0 of HNB2 → simulation run 0
      row 1 of DNB + row 1 of HNB1                  → simulation run 1  (if HNB2 is empty)
    Columns shorter than the longest column simply contribute no file for the
    extra rows.  Use ▲/▼ to reorder files within a column.
    """

    def __init__(self, parent, cfg, colours, *, view_sources_o3d_fn):
        super().__init__(parent)
        self.cfg = cfg
        self._colours = colours
        self._view_sources_o3d = view_sources_o3d_fn
        self._build()

    # ------------------------------------------------------------------
    #  Build UI
    # ------------------------------------------------------------------
    def _build(self):
        # Outer scrollable canvas so everything fits when window is small
        outer_canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0,
                                 bg=self._colours["bg"])
        vscroll = ttk.Scrollbar(self, orient="vertical", command=outer_canvas.yview)
        outer_canvas.configure(yscrollcommand=vscroll.set)
        outer_canvas.pack(side="left", fill="both", expand=True)
        vscroll.pack(side="right", fill="y")

        inner = ttk.Frame(outer_canvas)
        inner.bind("<Configure>",
                   lambda e: outer_canvas.configure(
                       scrollregion=outer_canvas.bbox("all")))
        outer_canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_mousewheel(event):
            outer_canvas.yview_scroll(
                -1 * (event.delta // 120) if event.delta else
                (-1 if event.num == 4 else 1), "units")
        outer_canvas.bind("<MouseWheel>", _on_mousewheel)
        outer_canvas.bind("<Button-4>", _on_mousewheel)
        outer_canvas.bind("<Button-5>", _on_mousewheel)

        # ── Global simulation params ─────────────────────────────────
        param_card = make_card(inner, "Simulation Parameters", pady=(12, 8))

        ttk.Label(param_card, text="Particles per beamlet:", style="Card.TLabel").grid(
            row=0, column=0, sticky="w", pady=4)
        self.var_npb = tk.IntVar(value=self.cfg.get("NUM_PARTICLES_PER_BEAMLET", 10001))
        ttk.Entry(param_card, textvariable=self.var_npb, width=12).grid(
            row=0, column=1, sticky="w", padx=(8, 0))

        ttk.Label(param_card, text="Beamlet grid radius (m):", style="Card.TLabel").grid(
            row=1, column=0, sticky="w", pady=4)
        self.var_radius = tk.DoubleVar(value=self.cfg.get("BEAMLET_RADIUS_M", 0.007))
        ttk.Entry(param_card, textvariable=self.var_radius, width=12).grid(
            row=1, column=1, sticky="w", padx=(8, 0))

        ttk.Label(param_card, text="Particle batch size:", style="Card.TLabel").grid(
            row=2, column=0, sticky="w", pady=4)
        self.var_batch = tk.IntVar(value=self.cfg.get("PARTICLE_BATCH_SIZE", 2_500_000))
        ttk.Entry(param_card, textvariable=self.var_batch, width=12).grid(
            row=2, column=1, sticky="w", padx=(8, 0))

        param_card.columnconfigure(1, weight=1)

        # ── Beam source columns ──────────────────────────────────────
        beam_sources = self.cfg.get("BEAM_SOURCES", _DEFAULT_BEAM_SOURCES)
        self._beam_sources = [dict(bs) for bs in beam_sources]

        sources_card = make_card(inner, "Beam Source Directories & Files", pady=(8, 8))

        # Container that holds the three columns side-by-side
        cols_frame = ttk.Frame(sources_card, style="Card.TFrame")
        cols_frame.pack(fill="both", expand=True)

        self._col_vars = {}   # label → {dir_var, listbox, bl_order}
        for col_idx, bs in enumerate(self._beam_sources):
            self._build_beam_column(cols_frame, col_idx, bs)

        # Refresh / view buttons row
        btn_row = ttk.Frame(sources_card, style="Card.TFrame")
        btn_row.pack(fill="x", pady=(6, 0))
        ttk.Button(btn_row, text="↻ Refresh All", style="Secondary.TButton",
                   command=self._refresh_all_columns).pack(side="left", padx=(0, 6))
        ttk.Button(btn_row, text="👁 View Sources (Open3D)", style="Secondary.TButton",
                   command=self._view_sources_o3d).pack(side="right")

        # ── Beam Combinations ────────────────────────────────────────
        # Info label explaining the row-pairing logic
        info_card = make_card(inner, "How runs are generated", pady=(4, 12))
        ttk.Label(
            info_card,
            text=(
                "Row 0 of each column runs together \u2192 one simulation.\n"
                "Row 1 of each column \u2192 next simulation, and so on.\n"
                "Use \u25b2\u25bc to reorder files.  "
                "Columns with no file at a given row are skipped."
            ),
            style="Card.TLabel", foreground=self._colours["dim"],
            justify="left").pack(anchor="w")

    # ------------------------------------------------------------------
    #  Build one beam-source column
    # ------------------------------------------------------------------
    def _build_beam_column(self, parent, col_idx, beam_source):
        label = beam_source["label"]
        frame = ttk.LabelFrame(parent, text=label, style="Card.TFrame", padding=6)
        frame.grid(row=0, column=col_idx, sticky="nsew", padx=(0 if col_idx == 0 else 6, 0))
        parent.columnconfigure(col_idx, weight=1)

        # Directory row
        dir_var = tk.StringVar(value=beam_source.get("directory", ""))
        dir_entry = ttk.Entry(frame, textvariable=dir_var, width=22,
                              font=("Segoe UI", 9))
        dir_entry.pack(fill="x", pady=(0, 2))
        ttk.Button(frame, text="Browse…", style="Secondary.TButton",
                   command=lambda v=dir_var: browse_directory(v)).pack(
                       fill="x", pady=(0, 4))

        # .bl file listbox
        lb = tk.Listbox(frame, height=10, font=("Segoe UI", 9),
                        bg="white", fg=self._colours["fg"],
                        selectbackground=self._colours["accent"],
                        selectforeground="white",
                        highlightthickness=0, bd=1, relief="solid",
                        exportselection=False)
        lb.pack(fill="both", expand=True, pady=(0, 4))

        # Order buttons
        order_frm = ttk.Frame(frame, style="Card.TFrame")
        order_frm.pack(fill="x")
        ttk.Button(order_frm, text="▲", width=3, style="Secondary.TButton",
                   command=lambda l=lb: self._move_bl_item(l, -1)).pack(
                       side="left", padx=(0, 2))
        ttk.Button(order_frm, text="▼", width=3, style="Secondary.TButton",
                   command=lambda l=lb: self._move_bl_item(l, 1)).pack(side="left")

        self._col_vars[label] = {
            "dir_var": dir_var,
            "listbox": lb,
            "bl_order": [],      # ordered list of basenames
        }

        # Initial fill
        dir_var.trace_add("write", lambda *_a, lbl=label: self._refresh_column(lbl))
        self._refresh_column(label)

    # ------------------------------------------------------------------
    #  Column helpers
    # ------------------------------------------------------------------
    def _refresh_column(self, label):
        col = self._col_vars.get(label)
        if not col:
            return
        dir_val = col["dir_var"].get()
        dir_abs = _resolve_path(dir_val) if not os.path.isabs(dir_val) else dir_val
        bl_files = sorted(glob.glob(os.path.join(dir_abs, "*.bl")))
        basenames = [os.path.basename(f) for f in bl_files]

        # Preserve current custom order; append newly found files at the end
        current_order = col["bl_order"]
        new_order = [b for b in current_order if b in basenames]
        for b in basenames:
            if b not in new_order:
                new_order.append(b)
        col["bl_order"] = new_order

        lb = col["listbox"]
        lb.delete(0, "end")
        for b in new_order:
            lb.insert("end", b)
        if not new_order:
            lb.insert("end", "(no .bl files)")

    def _refresh_all_columns(self):
        for label in self._col_vars:
            self._refresh_column(label)

    def _move_bl_item(self, listbox, direction):
        sel = listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        new_idx = idx + direction
        count = listbox.size()
        if new_idx < 0 or new_idx >= count:
            return
        # Swap in listbox
        item = listbox.get(idx)
        other = listbox.get(new_idx)
        listbox.delete(idx)
        listbox.insert(idx, other)
        listbox.delete(new_idx)
        listbox.insert(new_idx, item)
        listbox.selection_clear(0, "end")
        listbox.selection_set(new_idx)
        listbox.see(new_idx)
        # Keep bl_order in sync
        for label, col in self._col_vars.items():
            if col["listbox"] is listbox:
                order = col["bl_order"]
                order[idx], order[new_idx] = order[new_idx], order[idx]
                break

    def _get_selected_file(self, label):
        """Return the selected .bl basename for a given beam label, or None."""
        col = self._col_vars.get(label)
        if not col:
            return None
        sel = col["listbox"].curselection()
        if not sel:
            return None
        val = col["listbox"].get(sel[0])
        if val.startswith("("):
            return None
        return val

    # ------------------------------------------------------------------
    #  collect / refresh
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    #  collect / refresh
    # ------------------------------------------------------------------
    def collect(self, d):
        """Write particle keys into *d*.

        BEAM_COMBINATIONS is auto-generated by pairing column rows:
          row 0 of DNB + row 0 of HNB1 + row 0 of HNB2 -> run 0
          row 1 of DNB + row 1 of HNB1 (HNB2 empty)    -> run 1
        """
        d["NUM_PARTICLES_PER_BEAMLET"] = self.var_npb.get()
        d["BEAMLET_RADIUS_M"] = self.var_radius.get()
        d["PARTICLE_BATCH_SIZE"] = self.var_batch.get()

        updated_sources = []
        for bs in self._beam_sources:
            label = bs["label"]
            col = self._col_vars.get(label, {})
            new_dir = col["dir_var"].get() if col else bs.get("directory", "")
            updated = dict(bs)
            updated["directory"] = new_dir
            updated_sources.append(updated)
        d["BEAM_SOURCES"] = updated_sources
        if updated_sources:
            d["PARTICLE_SOURCE_DIR"] = updated_sources[0]["directory"]

        # Auto-generate BEAM_COMBINATIONS by zipping column rows
        labels = [bs["label"] for bs in self._beam_sources]
        col_files = {}
        for lbl in labels:
            col = self._col_vars.get(lbl)
            col_files[lbl] = (
                [f for f in col["bl_order"] if not f.startswith("(")]
                if col else []
            )

        max_rows = max((len(col_files[lbl]) for lbl in labels), default=0)
        combinations = []
        for row_idx in range(max_rows):
            sources, name_parts = [], []
            for lbl in labels:
                files = col_files[lbl]
                if row_idx < len(files):
                    fname = files[row_idx]
                    sources.append({"label": lbl, "file": fname})
                    name_parts.append(os.path.splitext(fname)[0])
            if sources:
                combinations.append({"name": "__".join(name_parts), "sources": sources})
        d["BEAM_COMBINATIONS"] = combinations

    def refresh(self, c):
        """Push config *c* values back into the widgets."""
        self.cfg = c
        self.var_npb.set(c.get("NUM_PARTICLES_PER_BEAMLET", 10001))
        self.var_radius.set(c.get("BEAMLET_RADIUS_M", 0.007))
        self.var_batch.set(c.get("PARTICLE_BATCH_SIZE", 2_500_000))

        beam_sources = c.get("BEAM_SOURCES", _DEFAULT_BEAM_SOURCES)
        self._beam_sources = [dict(bs) for bs in beam_sources]
        for bs in self._beam_sources:
            label = bs["label"]
            col = self._col_vars.get(label)
            if col:
                col["dir_var"].set(bs.get("directory", ""))
                self._refresh_column(label)

    # Legacy property so sim_gui.py's var_src_dir references keep working
    @property
    def var_src_dir(self):
        """Return the directory StringVar for the first beam source (DNB)."""
        first_label = self._beam_sources[0]["label"] if self._beam_sources else None
        if first_label and first_label in self._col_vars:
            return self._col_vars[first_label]["dir_var"]
        return tk.StringVar(value=self.cfg.get("PARTICLE_SOURCE_DIR", ""))
