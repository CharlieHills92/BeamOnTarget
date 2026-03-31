#!/usr/bin/env python3
# sim_gui.py
"""
Tkinter GUI for managing the BeamOnTarget simulation.

Reads / writes config.json through the config module.
Launches run_simulation.py as a subprocess (preserving CLI compatibility).
Launches ParaView externally for geometry and results viewing.
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import json
import os
import sys
import subprocess
import glob
import threading

import viewer  # built-in Open3D viewer

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_JSON = os.path.join(_SCRIPT_DIR, "config.json")
_RUN_SIMULATION = os.path.join(_SCRIPT_DIR, "run_simulation.py")
_PYTHON = sys.executable  # the same Python that launched the GUI


# ===================================================================
#  Helper: load / save JSON directly (no import config, to stay clean)
# ===================================================================
def load_config():
    with open(_CONFIG_JSON, "r") as f:
        return json.load(f)

def save_config(data):
    with open(_CONFIG_JSON, "w") as f:
        json.dump(data, f, indent=4)


# ===================================================================
#  ParaView launcher helpers
# ===================================================================

def _pv_geometry_script(cfg):
    """Return a Python script string that ParaView will execute to show STLs."""
    folders = cfg.get("GEOMETRY_FOLDERS", {})
    colors = [
        (0.8, 0.2, 0.2), (0.2, 0.6, 0.8), (0.2, 0.8, 0.3),
        (0.9, 0.7, 0.1), (0.6, 0.3, 0.7), (0.9, 0.4, 0.1),
        (0.4, 0.8, 0.8), (0.8, 0.4, 0.6),
    ]
    lines = [
        "from paraview.simple import *",
        "paraview.simple._DisableFirstRenderCameraReset()",
        "rv = GetActiveViewOrCreate('RenderView')",
        "rv.ResetCamera()",
    ]
    ci = 0
    for folder, settings in folders.items():
        scale = settings.get("scale", 1)
        folder_abs = os.path.join(_SCRIPT_DIR, folder)
        if not os.path.isdir(folder_abs):
            continue
        stl_files = sorted(glob.glob(os.path.join(folder_abs, "*.stl")) +
                           glob.glob(os.path.join(folder_abs, "*.STL")))
        r, g, b = colors[ci % len(colors)]
        ci += 1
        for stl in stl_files:
            name = os.path.splitext(os.path.basename(stl))[0]
            lines.append(f"reader = STLReader(FileNames=[r'{stl}'])")
            if scale != 1:
                lines.append(f"t = Transform(Input=reader)")
                lines.append(f"t.Transform.Scale = [{scale}, {scale}, {scale}]")
                lines.append(f"dp = Show(t, rv)")
            else:
                lines.append(f"dp = Show(reader, rv)")
            lines.append(f"dp.DiffuseColor = [{r}, {g}, {b}]")
            lines.append(f"dp.Opacity = 0.85")
            lines.append(f"RenameSource('{folder}/{name}', reader)")
    lines.append("rv.ResetCamera()")
    lines.append("Render()")
    return "\n".join(lines)


def _pv_results_script(results_dir):
    """Return a Python script that opens all .vtp files in a results dir."""
    vtp_files = sorted(glob.glob(os.path.join(results_dir, "**", "*.vtp"), recursive=True))
    if not vtp_files:
        return None
    lines = [
        "from paraview.simple import *",
        "paraview.simple._DisableFirstRenderCameraReset()",
        "rv = GetActiveViewOrCreate('RenderView')",
    ]
    for vtp in vtp_files:
        name = os.path.splitext(os.path.basename(vtp))[0]
        lines.append(f"reader = XMLPolyDataReader(FileName=[r'{vtp}'])")
        lines.append(f"dp = Show(reader, rv)")
        lines.append(f"dp.SetRepresentationType('Surface')")
        lines.append(f"ColorBy(dp, ('CELLS', 'Power_Density_W_m2'))")
        lines.append(f"dp.RescaleTransferFunctionToDataRange(True, False)")
        lines.append(f"RenameSource('{name}', reader)")
    lines.append("rv.ResetCamera()")
    lines.append("Render()")
    lines.append("# Show color bar")
    lines.append("dp.SetScalarBarVisibility(rv, True)")
    return "\n".join(lines)


def launch_paraview(script_content, pv_path, pv_module="ParaView"):
    """Write a temp script and launch ParaView via a shell that loads the
    required EasyBuild module first (so that LD_LIBRARY_PATH etc. are set)."""
    tmp_script = os.path.join(_SCRIPT_DIR, ".pv_temp_script.py")
    with open(tmp_script, "w") as f:
        f.write(script_content)

    # Build a shell command that loads the module, then runs ParaView.
    # Using 'bash -lc' so that the module system is initialised.
    shell_cmd = f'ml {pv_module} 2>/dev/null; "{pv_path}" --script="{tmp_script}"'
    try:
        proc = subprocess.Popen(
            ["bash", "-l", "-c", shell_cmd],
            cwd=_SCRIPT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True)
        # Start a thread to watch for early failure (give it a few seconds)
        def _watch():
            try:
                proc.wait(timeout=5)
                if proc.returncode and proc.returncode != 0:
                    err = proc.stderr.read().strip() if proc.stderr else ""
                    msg = f"ParaView exited with code {proc.returncode}."
                    if err:
                        msg += f"\n\n{err[:600]}"
                    # Schedule messagebox on main thread
                    try:
                        import tkinter as _tk
                        # If we still have a running Tk instance
                        messagebox.showerror("ParaView Error", msg)
                    except Exception:
                        pass
            except subprocess.TimeoutExpired:
                pass  # still running — good
        threading.Thread(target=_watch, daemon=True).start()
    except FileNotFoundError:
        messagebox.showerror("ParaView not found",
                             f"Cannot find ParaView at:\n{pv_path}\n\n"
                             "Update the path in the General tab.")


# ===================================================================
#  Main GUI Application
# ===================================================================

class SimGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("BeamOnTarget — Simulation Manager")
        self.geometry("1000x760")
        self.minsize(860, 640)
        self.cfg = load_config()
        self._apply_theme()
        self._build_ui()
        self._build_statusbar()

    # ------------------------------------------------------------------
    #  Modern theme & styling
    # ------------------------------------------------------------------
    def _apply_theme(self):
        style = ttk.Style(self)
        # Use clam as the base — it supports most colour overrides
        style.theme_use("clam")

        # --- Colour palette ---
        BG       = "#f0f2f5"   # main background
        CARD_BG  = "#ffffff"   # card / frame background
        ACCENT   = "#2563eb"   # blue accent (buttons, active tab)
        ACCENT2  = "#1d4ed8"   # darker accent (pressed)
        FG       = "#1e293b"   # primary text
        FG_DIM   = "#64748b"   # secondary text
        BORDER   = "#cbd5e1"   # subtle borders
        SUCCESS  = "#16a34a"
        DANGER   = "#dc2626"

        self.configure(bg=BG)

        # Notebook (tabs)
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab",
                         background=BG, foreground=FG, padding=[14, 6],
                         font=("Segoe UI", 10))
        style.map("TNotebook.Tab",
                   background=[("selected", CARD_BG)],
                   foreground=[("selected", ACCENT)],
                   expand=[("selected", [0, 0, 0, 2])])

        # Frames
        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=CARD_BG, relief="flat")

        # Labels
        style.configure("TLabel", background=BG, foreground=FG,
                         font=("Segoe UI", 10))
        style.configure("Card.TLabel", background=CARD_BG, foreground=FG,
                         font=("Segoe UI", 10))
        style.configure("Header.TLabel", background=BG, foreground=ACCENT,
                         font=("Segoe UI", 12, "bold"))
        style.configure("CardHeader.TLabel", background=CARD_BG,
                         foreground=ACCENT, font=("Segoe UI", 11, "bold"))
        style.configure("Dim.TLabel", background=BG, foreground=FG_DIM,
                         font=("Segoe UI", 9))
        style.configure("Status.TLabel", background=BORDER, foreground=FG,
                         font=("Segoe UI", 9), padding=[8, 4])

        # Buttons
        style.configure("TButton", font=("Segoe UI", 10), padding=[10, 5],
                         background=ACCENT, foreground="white", borderwidth=0)
        style.map("TButton",
                   background=[("active", ACCENT2), ("pressed", ACCENT2)],
                   foreground=[("disabled", FG_DIM)])

        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"),
                         padding=[14, 6], background=ACCENT, foreground="white")
        style.map("Accent.TButton",
                   background=[("active", ACCENT2)])

        style.configure("Danger.TButton", font=("Segoe UI", 10),
                         padding=[10, 5], background=DANGER, foreground="white")
        style.map("Danger.TButton",
                   background=[("active", "#b91c1c")])

        style.configure("Success.TButton", font=("Segoe UI", 10),
                         padding=[10, 5], background=SUCCESS, foreground="white")
        style.map("Success.TButton",
                   background=[("active", "#15803d")])

        style.configure("Secondary.TButton", font=("Segoe UI", 10),
                         padding=[10, 5], background="#e2e8f0", foreground=FG,
                         borderwidth=0)
        style.map("Secondary.TButton",
                   background=[("active", "#cbd5e1")])

        # Entries
        style.configure("TEntry", fieldbackground="white", foreground=FG,
                         borderwidth=1, padding=[6, 4],
                         font=("Segoe UI", 10))

        # Spinbox
        style.configure("TSpinbox", fieldbackground="white", foreground=FG,
                         padding=[6, 4], font=("Segoe UI", 10))

        # Checkbutton
        style.configure("TCheckbutton", background=BG, foreground=FG,
                         font=("Segoe UI", 10))
        style.configure("Card.TCheckbutton", background=CARD_BG,
                         foreground=FG, font=("Segoe UI", 10))

        # Treeview
        style.configure("Treeview", background="white", foreground=FG,
                         fieldbackground="white", rowheight=26,
                         font=("Segoe UI", 10), borderwidth=0)
        style.configure("Treeview.Heading", background=BG, foreground=FG,
                         font=("Segoe UI", 10, "bold"), padding=[4, 4])
        style.map("Treeview",
                   background=[("selected", "#dbeafe")],
                   foreground=[("selected", ACCENT)])

        # Separator
        style.configure("TSeparator", background=BORDER)

        # Scrollbar
        style.configure("Vertical.TScrollbar", background=BG,
                         troughcolor=CARD_BG, borderwidth=0)

        # Store colours for use in widgets that don't support style
        self._colours = {
            "bg": BG, "card": CARD_BG, "accent": ACCENT, "fg": FG,
            "dim": FG_DIM, "border": BORDER, "success": SUCCESS,
            "danger": DANGER,
        }

    # ------------------------------------------------------------------
    #  Helper: card-like frame with optional title
    # ------------------------------------------------------------------
    def _make_card(self, parent, title=None, padx=12, pady=(0, 10)):
        """Return a content Frame inside a white card with rounded-look padding.

        A wrapper frame is packed into *parent*; inside it an optional title
        label is packed, then a content frame is packed and returned.  The
        caller can freely use **either** ``pack`` or ``grid`` inside the
        returned content frame without conflicting with the title label's
        geometry manager.
        """
        wrapper = ttk.Frame(parent, style="Card.TFrame", padding=14)
        wrapper.pack(fill="x", padx=padx, pady=pady)
        if title:
            ttk.Label(wrapper, text=title, style="CardHeader.TLabel").pack(
                anchor="w", pady=(0, 8))
        content = ttk.Frame(wrapper, style="Card.TFrame")
        content.pack(fill="both", expand=True)
        return content

    # ------------------------------------------------------------------
    #  Status bar
    # ------------------------------------------------------------------
    def _build_statusbar(self):
        self._statusbar = ttk.Label(self, text="  Ready", style="Status.TLabel")
        self._statusbar.pack(side="bottom", fill="x")

    def _set_status(self, text):
        self._statusbar.config(text=f"  {text}")

    # ------------------------------------------------------------------
    #  Build the tabbed interface
    # ------------------------------------------------------------------
    def _build_ui(self):
        self._build_menubar()

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=8, pady=(8, 0))

        self._build_general_tab(notebook)
        self._build_geometry_tab(notebook)
        self._build_particles_tab(notebook)
        self._build_output_tab(notebook)
        self._build_smoothing_tab(notebook)
        self._build_run_tab(notebook)

    # ------------------------------------------------------------------
    #  Menu bar
    # ------------------------------------------------------------------
    def _build_menubar(self):
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Save Config", accelerator="Ctrl+S",
                              command=self._save)
        file_menu.add_command(label="Save Config As…", accelerator="Ctrl+Shift+S",
                              command=self._save_as)
        file_menu.add_command(label="Load Config…", accelerator="Ctrl+O",
                              command=self._load_config_file)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self.destroy)
        menubar.add_cascade(label="File", menu=file_menu)
        self.config(menu=menubar)

        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_command(label="View All (Open3D)…",
                              command=self._view_all_o3d)
        view_menu.add_command(label="View Geometry (Open3D)…",
                              command=self._view_geometry_o3d)
        view_menu.add_command(label="View Results (Open3D)…",
                              command=self._view_results_o3d)
        view_menu.add_command(label="View Sources (Open3D)…",
                              command=self._view_sources_o3d)
        view_menu.add_separator()
        view_menu.add_command(label="View Geometry (ParaView)…",
                              command=self._view_geometry)
        view_menu.add_command(label="View Results (ParaView)…",
                              command=self._view_results)
        menubar.add_cascade(label="View", menu=view_menu)

        self.bind_all("<Control-s>", lambda e: self._save())
        self.bind_all("<Control-Shift-S>", lambda e: self._save_as())
        self.bind_all("<Control-o>", lambda e: self._load_config_file())

    # ------------------------------------------------------------------
    #  GENERAL tab
    # ------------------------------------------------------------------
    def _build_general_tab(self, nb):
        outer = ttk.Frame(nb)
        nb.add(outer, text="  ⚙  General  ")

        # --- Engine card ---
        card = self._make_card(outer, "Engine Settings", pady=(12, 10))

        row = 0
        ttk.Label(card, text="CPU cores (-1 = all):", style="Card.TLabel").grid(
            row=row, column=0, sticky="w", pady=5)
        self.var_cpu = tk.IntVar(value=self.cfg.get("NUM_CPU_CORES", 1))
        ttk.Spinbox(card, from_=-1, to=256, textvariable=self.var_cpu,
                     width=8).grid(row=row, column=1, sticky="w", padx=(8, 0))

        row += 1
        ttk.Label(card, text="Deposition fraction (0–1):", style="Card.TLabel").grid(
            row=row, column=0, sticky="w", pady=5)
        self.var_dep_frac = tk.DoubleVar(value=self.cfg.get("DEPOSITION_FRACTION", 1.0))
        ttk.Entry(card, textvariable=self.var_dep_frac, width=10).grid(
            row=row, column=1, sticky="w", padx=(8, 0))

        row += 1
        self.var_diag = tk.BooleanVar(value=self.cfg.get("ENABLE_DIAGNOSTIC_SURFACES", False))
        ttk.Checkbutton(card, text="Enable diagnostic (transparent) surfaces",
                         variable=self.var_diag,
                         style="Card.TCheckbutton").grid(
            row=row, column=0, columnspan=2, sticky="w", pady=5)

        row += 1
        ttk.Label(card, text="Geometry cache dir:", style="Card.TLabel").grid(
            row=row, column=0, sticky="w", pady=5)
        self.var_cache = tk.StringVar(value=self.cfg.get("GEOMETRY_CACHE_DIR", "geometry_cache"))
        ttk.Entry(card, textvariable=self.var_cache, width=30).grid(
            row=row, column=1, sticky="we", padx=(8, 0))

        card.columnconfigure(1, weight=1)

        # --- ParaView card ---
        pv_card = self._make_card(outer, "ParaView Integration")

        ttk.Label(pv_card, text="ParaView path:", style="Card.TLabel").grid(
            row=0, column=0, sticky="w", pady=5)
        self.var_pv_path = tk.StringVar(value=self.cfg.get("PARAVIEW_PATH", "paraview"))
        ttk.Entry(pv_card, textvariable=self.var_pv_path, width=50).grid(
            row=0, column=1, sticky="we", padx=(8, 4))
        ttk.Button(pv_card, text="Browse…", style="Secondary.TButton",
                    command=self._browse_pv).grid(row=0, column=2)

        ttk.Label(pv_card, text="ParaView module (ml):", style="Card.TLabel").grid(
            row=1, column=0, sticky="w", pady=5)
        self.var_pv_module = tk.StringVar(value=self.cfg.get("PARAVIEW_MODULE", "ParaView"))
        ttk.Entry(pv_card, textvariable=self.var_pv_module, width=30).grid(
            row=1, column=1, sticky="w", padx=(8, 0))

        pv_card.columnconfigure(1, weight=1)

    def _browse_pv(self):
        p = filedialog.askopenfilename(title="Select ParaView executable",
                                       filetypes=[("All files", "*")])
        if p:
            self.var_pv_path.set(p)

    # ------------------------------------------------------------------
    #  GEOMETRY tab (editable table)
    # ------------------------------------------------------------------
    def _build_geometry_tab(self, nb):
        outer = ttk.Frame(nb)
        nb.add(outer, text="  📐  Geometry  ")

        card = self._make_card(outer, "Geometry Folders", pady=(12, 10))

        # Treeview
        cols = ("folder", "scale", "target_length", "save_details",
                "is_diagnostic", "save_impact_data", "max_impact_records")
        tree_frame = ttk.Frame(card, style="Card.TFrame")
        tree_frame.pack(fill="both", expand=True)

        self.geo_tree = ttk.Treeview(tree_frame, columns=cols, show="headings", height=8)
        headers = {"folder": "Folder", "scale": "Scale", "target_length": "Target Len",
                   "save_details": "Details", "is_diagnostic": "Diagnostic",
                   "save_impact_data": "Impacts", "max_impact_records": "Max Records"}
        widths = {"folder": 140, "scale": 60, "target_length": 90, "save_details": 65,
                  "is_diagnostic": 80, "save_impact_data": 65, "max_impact_records": 100}
        for c in cols:
            self.geo_tree.heading(c, text=headers[c])
            self.geo_tree.column(c, width=widths[c], anchor="center")
        self.geo_tree.column("folder", anchor="w")

        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.geo_tree.yview)
        self.geo_tree.configure(yscrollcommand=vsb.set)
        self.geo_tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        self._populate_geo_tree()

        # Buttons
        btn_frm = ttk.Frame(card, style="Card.TFrame")
        btn_frm.pack(fill="x", pady=(10, 0))
        ttk.Button(btn_frm, text="＋ Add Folder…", style="Secondary.TButton",
                    command=self._add_geo_folder).pack(side="left", padx=(0, 4))
        ttk.Button(btn_frm, text="✎ Edit Selected…", style="Secondary.TButton",
                    command=self._edit_geo_folder).pack(side="left", padx=4)
        ttk.Button(btn_frm, text="✕ Remove", style="Secondary.TButton",
                    command=self._remove_geo_folder).pack(side="left", padx=4)
        ttk.Button(btn_frm, text="🔍 ParaView", style="Secondary.TButton",
                    command=self._view_geometry).pack(side="right", padx=(4, 0))
        ttk.Button(btn_frm, text="� Open3D", style="Secondary.TButton",
                    command=self._view_geometry_o3d).pack(side="right", padx=4)

    def _populate_geo_tree(self):
        for item in self.geo_tree.get_children():
            self.geo_tree.delete(item)
        for folder, s in self.cfg.get("GEOMETRY_FOLDERS", {}).items():
            self.geo_tree.insert("", "end", values=(
                folder, s.get("scale", 1), s.get("target_length", 1.0),
                "✓" if s.get("save_details") else "", "✓" if s.get("is_diagnostic") else "",
                "✓" if s.get("save_impact_data") else "",
                s.get("max_impact_records") if s.get("max_impact_records") is not None else "—"))

    def _add_geo_folder(self):
        self._geo_dialog(None)

    def _edit_geo_folder(self):
        sel = self.geo_tree.selection()
        if not sel:
            return
        folder = self.geo_tree.item(sel[0])["values"][0]
        self._geo_dialog(str(folder))

    def _remove_geo_folder(self):
        sel = self.geo_tree.selection()
        if not sel:
            return
        folder = str(self.geo_tree.item(sel[0])["values"][0])
        if messagebox.askyesno("Remove", f"Remove geometry folder '{folder}'?"):
            self.cfg["GEOMETRY_FOLDERS"].pop(folder, None)
            self._populate_geo_tree()

    def _geo_dialog(self, existing_folder):
        """Pop up a dialog to add/edit a geometry folder entry."""
        dlg = tk.Toplevel(self)
        dlg.title("Edit Geometry Folder" if existing_folder else "Add Geometry Folder")
        dlg.geometry("400x360")
        dlg.transient(self)
        dlg.grab_set()

        s = self.cfg.get("GEOMETRY_FOLDERS", {}).get(existing_folder, {}) if existing_folder else {}

        entries = {}
        row = 0

        ttk.Label(dlg, text="Folder path:").grid(row=row, column=0, sticky="w", padx=8, pady=4)
        v_folder = tk.StringVar(value=existing_folder or "")
        e = ttk.Entry(dlg, textvariable=v_folder, width=30)
        e.grid(row=row, column=1, sticky="we", padx=8)
        if existing_folder:
            e.config(state="disabled")
        entries["folder"] = v_folder

        fields = [
            ("scale", "Scale:", float, 1),
            ("target_length", "Target length:", float, 1.0),
        ]
        for key, label, typ, default in fields:
            row += 1
            ttk.Label(dlg, text=label).grid(row=row, column=0, sticky="w", padx=8, pady=4)
            v = tk.StringVar(value=str(s.get(key, default)))
            ttk.Entry(dlg, textvariable=v, width=15).grid(row=row, column=1, sticky="w", padx=8)
            entries[key] = (v, typ)

        bools = [
            ("save_details", "Save detailed reports", False),
            ("is_diagnostic", "Is diagnostic (transparent)", False),
            ("save_impact_data", "Save impact data", False),
            ("show_in_plot", "Show in plot", False),
        ]
        for key, label, default in bools:
            row += 1
            v = tk.BooleanVar(value=s.get(key, default))
            ttk.Checkbutton(dlg, text=label, variable=v).grid(row=row, column=0, columnspan=2, sticky="w", padx=8, pady=2)
            entries[key] = (v, bool)

        row += 1
        ttk.Label(dlg, text="Max impact records:").grid(row=row, column=0, sticky="w", padx=8, pady=4)
        mir = s.get("max_impact_records")
        v_mir = tk.StringVar(value=str(mir) if mir is not None else "")
        ttk.Entry(dlg, textvariable=v_mir, width=15).grid(row=row, column=1, sticky="w", padx=8)
        entries["max_impact_records"] = v_mir

        def _ok():
            folder_name = entries["folder"].get().strip()
            if not folder_name:
                messagebox.showwarning("Missing", "Folder path is required."); return
            new_s = {}
            for key, (var, typ) in [(k, v) for k, v in entries.items() if k not in ("folder", "max_impact_records")]:
                try:
                    new_s[key] = typ(var.get())
                except (ValueError, tk.TclError):
                    new_s[key] = var.get()
            mir_val = entries["max_impact_records"].get().strip()
            new_s["max_impact_records"] = int(mir_val) if mir_val else None
            if "GEOMETRY_FOLDERS" not in self.cfg:
                self.cfg["GEOMETRY_FOLDERS"] = {}
            self.cfg["GEOMETRY_FOLDERS"][folder_name] = new_s
            self._populate_geo_tree()
            dlg.destroy()

        row += 1
        ttk.Button(dlg, text="OK", command=_ok).grid(row=row, column=0, columnspan=2, pady=10)
        dlg.columnconfigure(1, weight=1)

    def _view_geometry(self):
        script = _pv_geometry_script(self.cfg)
        pv = self.var_pv_path.get()
        pv_mod = self.var_pv_module.get()
        launch_paraview(script, pv, pv_mod)

    def _view_geometry_o3d(self):
        """Open the built-in Open3D geometry viewer with folder selection."""
        src_dir = self.var_src_dir.get()
        viewer.view_geometry(self, _SCRIPT_DIR,
                             self.cfg.get("GEOMETRY_FOLDERS", {}),
                             source_dir=src_dir)

    def _view_results_o3d(self):
        """Open the built-in Open3D results viewer with heatmap colouring."""
        outdir = self.var_outdir.get()
        src_dir = self.var_src_dir.get()
        viewer.view_results(self, _SCRIPT_DIR, outdir,
                            geometry_folders=self.cfg.get("GEOMETRY_FOLDERS", {}),
                            source_dir=src_dir)

    def _view_all_o3d(self):
        """Open the built-in Open3D viewer showing geometry + results."""
        outdir = self.var_outdir.get()
        src_dir = self.var_src_dir.get()
        viewer.view_all(self, _SCRIPT_DIR, outdir,
                        self.cfg.get("GEOMETRY_FOLDERS", {}),
                        source_dir=src_dir)

    def _view_sources_o3d(self):
        """Open the built-in Open3D source viewer showing beamlet positions."""
        src_dir = self.var_src_dir.get()
        viewer.view_sources(self, _SCRIPT_DIR, src_dir,
                            geometry_folders=self.cfg.get("GEOMETRY_FOLDERS", {}))

    # ------------------------------------------------------------------
    #  PARTICLES tab
    # ------------------------------------------------------------------
    def _build_particles_tab(self, nb):
        outer = ttk.Frame(nb)
        nb.add(outer, text="  🔬  Particles  ")

        # --- Beam source card ---
        card = self._make_card(outer, "Beam Source", pady=(12, 10))

        row = 0
        ttk.Label(card, text="Beam config directory:", style="Card.TLabel").grid(
            row=row, column=0, sticky="w", pady=5)
        self.var_src_dir = tk.StringVar(value=self.cfg.get("PARTICLE_SOURCE_DIR", "BEAM_CONFIGS"))
        ttk.Entry(card, textvariable=self.var_src_dir, width=30).grid(
            row=row, column=1, sticky="we", padx=(8, 4))
        ttk.Button(card, text="Browse…", style="Secondary.TButton",
                    command=lambda: self._browse_dir(self.var_src_dir)).grid(
            row=row, column=2)

        row += 1
        ttk.Label(card, text="Particles per beamlet:", style="Card.TLabel").grid(
            row=row, column=0, sticky="w", pady=5)
        self.var_npb = tk.IntVar(value=self.cfg.get("NUM_PARTICLES_PER_BEAMLET", 10001))
        ttk.Entry(card, textvariable=self.var_npb, width=12).grid(
            row=row, column=1, sticky="w", padx=(8, 0))

        row += 1
        ttk.Label(card, text="Beamlet radius (m):", style="Card.TLabel").grid(
            row=row, column=0, sticky="w", pady=5)
        self.var_radius = tk.DoubleVar(value=self.cfg.get("BEAMLET_RADIUS_M", 0.007))
        ttk.Entry(card, textvariable=self.var_radius, width=12).grid(
            row=row, column=1, sticky="w", padx=(8, 0))

        row += 1
        ttk.Label(card, text="Particle batch size:", style="Card.TLabel").grid(
            row=row, column=0, sticky="w", pady=5)
        self.var_batch = tk.IntVar(value=self.cfg.get("PARTICLE_BATCH_SIZE", 2_500_000))
        ttk.Entry(card, textvariable=self.var_batch, width=12).grid(
            row=row, column=1, sticky="w", padx=(8, 0))

        row += 1
        ttk.Label(card, text="Sources per worker (empty=auto):", style="Card.TLabel").grid(
            row=row, column=0, sticky="w", pady=5)
        spw = self.cfg.get("SOURCES_PER_WORKER")
        self.var_spw = tk.StringVar(value=str(spw) if spw is not None else "")
        ttk.Entry(card, textvariable=self.var_spw, width=12).grid(
            row=row, column=1, sticky="w", padx=(8, 0))

        card.columnconfigure(1, weight=1)

        # --- Beam files card ---
        bl_card = self._make_card(outer, "Beam Configuration Files")

        self.bl_listbox = tk.Listbox(bl_card, height=8, width=50,
                                      font=("Segoe UI", 10),
                                      bg="white", fg=self._colours["fg"],
                                      selectbackground=self._colours["accent"],
                                      selectforeground="white",
                                      highlightthickness=0, bd=1,
                                      relief="solid")
        self.bl_listbox.pack(fill="both", expand=True, pady=(0, 6))
        self._refresh_bl_list()

        bl_btn_frm = ttk.Frame(bl_card, style="Card.TFrame")
        bl_btn_frm.pack(fill="x")
        ttk.Button(bl_btn_frm, text="↻ Refresh", style="Secondary.TButton",
                    command=self._refresh_bl_list).pack(side="left")
        ttk.Button(bl_btn_frm, text="👁 View Sources (Open3D)",
                    style="Secondary.TButton",
                    command=self._view_sources_o3d).pack(side="right")

    def _refresh_bl_list(self):
        self.bl_listbox.delete(0, "end")
        src = self.var_src_dir.get()
        src_abs = os.path.join(_SCRIPT_DIR, src) if not os.path.isabs(src) else src
        bl_files = sorted(glob.glob(os.path.join(src_abs, "*.bl")))
        for f in bl_files:
            self.bl_listbox.insert("end", os.path.basename(f))
        if not bl_files:
            self.bl_listbox.insert("end", "(no .bl files found)")

    def _browse_dir(self, var):
        d = filedialog.askdirectory(initialdir=_SCRIPT_DIR, title="Select directory")
        if d:
            # Store relative path if inside project dir
            try:
                rel = os.path.relpath(d, _SCRIPT_DIR)
                var.set(rel)
            except ValueError:
                var.set(d)

    # ------------------------------------------------------------------
    #  OUTPUT tab
    # ------------------------------------------------------------------
    def _build_output_tab(self, nb):
        outer = ttk.Frame(nb)
        nb.add(outer, text="  📁  Output  ")

        # --- Paths card ---
        card = self._make_card(outer, "Output Settings", pady=(12, 10))

        ttk.Label(card, text="Output directory:", style="Card.TLabel").grid(
            row=0, column=0, sticky="w", pady=5)
        self.var_outdir = tk.StringVar(value=self.cfg.get("DETAILED_OUTPUT_DIR", "OUTPUT"))
        ttk.Entry(card, textvariable=self.var_outdir, width=30).grid(
            row=0, column=1, sticky="we", padx=(8, 4))
        ttk.Button(card, text="Browse…", style="Secondary.TButton",
                    command=lambda: self._browse_dir(self.var_outdir)).grid(
            row=0, column=2)

        ttk.Label(card, text="Summary CSV filename:", style="Card.TLabel").grid(
            row=1, column=0, sticky="w", pady=5)
        self.var_summary = tk.StringVar(value=self.cfg.get("SUMMARY_CSV_FILENAME", "power_summary_by_object.csv"))
        ttk.Entry(card, textvariable=self.var_summary, width=40).grid(
            row=1, column=1, sticky="we", padx=(8, 0))

        ttk.Label(card, text="Rays to show in plot:", style="Card.TLabel").grid(
            row=2, column=0, sticky="w", pady=5)
        self.var_nrays = tk.IntVar(value=self.cfg.get("NUM_RAYS_TO_SHOW_IN_PLOT", 0))
        ttk.Entry(card, textvariable=self.var_nrays, width=10).grid(
            row=2, column=1, sticky="w", padx=(8, 0))

        card.columnconfigure(1, weight=1)

        # --- File options card ---
        opts_card = self._make_card(outer, "Save Options")

        checkboxes = [
            ("SAVE_PARAVIEW_FILES", "Save ParaView (.vtp) files"),
            ("SAVE_BINARY_POWERLOADS", "Save binary (.npy) power loads"),
            ("SAVE_CSV_REPORTS", "Save CSV reports"),
        ]
        for key, label in checkboxes:
            v = tk.BooleanVar(value=self.cfg.get(key, False))
            ttk.Checkbutton(opts_card, text=label, variable=v,
                             style="Card.TCheckbutton").pack(anchor="w", pady=2)
            setattr(self, f"var_{key}", v)

        # --- Visualisation card ---
        vis_card = self._make_card(outer, "Visualisation")

        vis_checks = [
            ("RUN_VISUALIZATION_AFTER_SIM", "Run visualisation after simulation"),
            ("VISUALIZE_ALL_RAYS", "Visualise all rays (including misses)"),
            ("ENABLE_VISUALIZATION", "Enable visualisation (master switch)"),
        ]
        for key, label in vis_checks:
            v = tk.BooleanVar(value=self.cfg.get(key, False))
            ttk.Checkbutton(vis_card, text=label, variable=v,
                             style="Card.TCheckbutton").pack(anchor="w", pady=2)
            setattr(self, f"var_{key}", v)

        btn_frm = ttk.Frame(vis_card, style="Card.TFrame")
        btn_frm.pack(fill="x", pady=(8, 0))
        ttk.Button(btn_frm, text="👁 Results (Open3D)", style="Secondary.TButton",
                    command=self._view_results_o3d).pack(side="left", padx=(0, 8))
        ttk.Button(btn_frm, text="�🔍 Results (ParaView)", style="Secondary.TButton",
                    command=self._view_results).pack(side="left")

    def _view_results(self):
        outdir = self.var_outdir.get()
        outdir_abs = os.path.join(_SCRIPT_DIR, outdir) if not os.path.isabs(outdir) else outdir
        if not os.path.isdir(outdir_abs):
            messagebox.showwarning("No results", f"Output directory not found:\n{outdir_abs}")
            return
        # Let user pick a subfolder
        subdirs = sorted([d for d in os.listdir(outdir_abs)
                          if os.path.isdir(os.path.join(outdir_abs, d))])
        if not subdirs:
            messagebox.showinfo("No results", "No result subfolders found.")
            return

        pick = _PickDialog(self, "Select result set", subdirs)
        self.wait_window(pick)
        if pick.result is None:
            return
        chosen_dir = os.path.join(outdir_abs, pick.result)
        script = _pv_results_script(chosen_dir)
        if script is None:
            messagebox.showinfo("No VTP", f"No .vtp files in {chosen_dir}")
            return
        launch_paraview(script, self.var_pv_path.get(), self.var_pv_module.get())

    # ------------------------------------------------------------------
    #  SMOOTHING tab
    # ------------------------------------------------------------------
    def _build_smoothing_tab(self, nb):
        outer = ttk.Frame(nb)
        nb.add(outer, text="  🔄  Smoothing  ")

        card = self._make_card(outer, "Post-Processing Smoother", pady=(12, 10))

        self.var_smoother = tk.BooleanVar(value=self.cfg.get("RUN_SMOOTHER_AFTER_SIM", False))
        ttk.Checkbutton(card, text="Run batch smoother after simulation",
                         variable=self.var_smoother,
                         style="Card.TCheckbutton").pack(anchor="w", pady=(0, 8))

        grid = ttk.Frame(card, style="Card.TFrame")
        grid.pack(fill="x")

        ttk.Label(grid, text="Smoothing radius (m):", style="Card.TLabel").grid(
            row=0, column=0, sticky="w", pady=5)
        self.var_sm_radius = tk.DoubleVar(value=self.cfg.get("SMOOTHING_RADIUS", 0.02))
        ttk.Entry(grid, textvariable=self.var_sm_radius, width=12).grid(
            row=0, column=1, sticky="w", padx=(8, 0))

        ttk.Label(grid, text="Max cell area (m², empty=None):", style="Card.TLabel").grid(
            row=1, column=0, sticky="w", pady=5)
        mca = self.cfg.get("SMOOTHING_MAX_CELL_AREA")
        self.var_sm_mca = tk.StringVar(value=str(mca) if mca is not None else "")
        ttk.Entry(grid, textvariable=self.var_sm_mca, width=12).grid(
            row=1, column=1, sticky="w", padx=(8, 0))

    # ------------------------------------------------------------------
    #  RUN tab (save, run, log)
    # ------------------------------------------------------------------
    def _build_run_tab(self, nb):
        outer = ttk.Frame(nb)
        nb.add(outer, text="  ▶  Run  ")

        # --- Action buttons ---
        btn_card = self._make_card(outer, pady=(12, 6))
        btn_row = ttk.Frame(btn_card, style="Card.TFrame")
        btn_row.pack(fill="x")

        ttk.Button(btn_row, text="💾 Save Config", style="Secondary.TButton",
                    command=self._save).pack(side="left", padx=(0, 4))
        ttk.Button(btn_row, text="💾 Save As…", style="Secondary.TButton",
                    command=self._save_as).pack(side="left", padx=4)
        ttk.Button(btn_row, text="📂 Load Config…", style="Secondary.TButton",
                    command=self._load_config_file).pack(side="left", padx=4)

        ttk.Separator(btn_row, orient="vertical").pack(side="left", fill="y",
                                                         padx=12, pady=2)

        ttk.Button(btn_row, text="▶  Run Simulation", style="Accent.TButton",
                    command=self._run_sim).pack(side="left", padx=4)
        ttk.Button(btn_row, text="⏹  Stop", style="Danger.TButton",
                    command=self._stop_sim).pack(side="left", padx=4)

        # SDCC checkbox
        self.var_sdcc = tk.BooleanVar(value=False)
        ttk.Checkbutton(btn_card, text="Run on SLURM server (srun --exclusive)",
                         variable=self.var_sdcc,
                         style="Card.TCheckbutton").pack(anchor="w", pady=(8, 0))

        # --- Log output ---
        log_card = self._make_card(outer, "Console Output", pady=(6, 10))

        self.log_text = scrolledtext.ScrolledText(
            log_card, height=24, state="disabled",
            font=("Consolas", 10), wrap="word",
            bg="#1e293b", fg="#e2e8f0", insertbackground="#e2e8f0",
            selectbackground=self._colours["accent"],
            highlightthickness=0, bd=0, padx=10, pady=8)
        self.log_text.pack(fill="both", expand=True)

        self._sim_process = None

    # ------------------------------------------------------------------
    #  Collect GUI → dict
    # ------------------------------------------------------------------
    def _collect(self):
        d = dict(self.cfg)  # start from current (preserves unknown keys)
        d["NUM_CPU_CORES"] = self.var_cpu.get()
        d["ENABLE_DIAGNOSTIC_SURFACES"] = self.var_diag.get()
        d["GEOMETRY_CACHE_DIR"] = self.var_cache.get()
        d["DEPOSITION_FRACTION"] = self.var_dep_frac.get()
        d["PARAVIEW_PATH"] = self.var_pv_path.get()
        d["PARAVIEW_MODULE"] = self.var_pv_module.get()
        # Geometry table is already updated in self.cfg via dialog
        d["GEOMETRY_FOLDERS"] = self.cfg.get("GEOMETRY_FOLDERS", {})
        d["PARTICLE_SOURCE_DIR"] = self.var_src_dir.get()
        d["NUM_PARTICLES_PER_BEAMLET"] = self.var_npb.get()
        d["BEAMLET_RADIUS_M"] = self.var_radius.get()
        d["PARTICLE_BATCH_SIZE"] = self.var_batch.get()
        spw = self.var_spw.get().strip()
        d["SOURCES_PER_WORKER"] = int(spw) if spw else None
        d["DETAILED_OUTPUT_DIR"] = self.var_outdir.get()
        d["SAVE_PARAVIEW_FILES"] = self.var_SAVE_PARAVIEW_FILES.get()
        d["SAVE_BINARY_POWERLOADS"] = self.var_SAVE_BINARY_POWERLOADS.get()
        d["SAVE_CSV_REPORTS"] = self.var_SAVE_CSV_REPORTS.get()
        d["RUN_VISUALIZATION_AFTER_SIM"] = self.var_RUN_VISUALIZATION_AFTER_SIM.get()
        d["VISUALIZE_ALL_RAYS"] = self.var_VISUALIZE_ALL_RAYS.get()
        d["ENABLE_VISUALIZATION"] = self.var_ENABLE_VISUALIZATION.get()
        d["SUMMARY_CSV_FILENAME"] = self.var_summary.get()
        d["NUM_RAYS_TO_SHOW_IN_PLOT"] = self.var_nrays.get()
        d["RUN_SMOOTHER_AFTER_SIM"] = self.var_smoother.get()
        d["SMOOTHING_RADIUS"] = self.var_sm_radius.get()
        mca = self.var_sm_mca.get().strip()
        d["SMOOTHING_MAX_CELL_AREA"] = float(mca) if mca else None
        return d

    # ------------------------------------------------------------------
    #  Save
    # ------------------------------------------------------------------
    def _save(self):
        try:
            self.cfg = self._collect()
            save_config(self.cfg)
            self._log("✔ Configuration saved to config.json\n")
            self._set_status("Configuration saved")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    def _save_as(self):
        """Save the current config to a user-chosen JSON file."""
        path = filedialog.asksaveasfilename(
            initialdir=_SCRIPT_DIR,
            title="Save Config As",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*")])
        if not path:
            return
        try:
            self.cfg = self._collect()
            with open(path, "w") as f:
                json.dump(self.cfg, f, indent=4)
            self._log(f"✔ Configuration saved to {os.path.basename(path)}\n")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    def _load_config_file(self):
        """Load a config from a user-chosen JSON file and refresh the GUI."""
        path = filedialog.askopenfilename(
            initialdir=_SCRIPT_DIR,
            title="Load Config",
            filetypes=[("JSON files", "*.json"), ("All files", "*")])
        if not path:
            return
        try:
            with open(path, "r") as f:
                new_cfg = json.load(f)
            self.cfg = new_cfg
            self._refresh_all_from_cfg()
            self._log(f"✔ Configuration loaded from {os.path.basename(path)}\n")
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def _refresh_all_from_cfg(self):
        """Push self.cfg values back into every GUI widget."""
        c = self.cfg
        # General
        self.var_cpu.set(c.get("NUM_CPU_CORES", 1))
        self.var_diag.set(c.get("ENABLE_DIAGNOSTIC_SURFACES", False))
        self.var_cache.set(c.get("GEOMETRY_CACHE_DIR", "geometry_cache"))
        self.var_dep_frac.set(c.get("DEPOSITION_FRACTION", 1.0))
        self.var_pv_path.set(c.get("PARAVIEW_PATH", "paraview"))
        self.var_pv_module.set(c.get("PARAVIEW_MODULE", "ParaView"))
        # Geometry table
        self._populate_geo_tree()
        # Particles
        self.var_src_dir.set(c.get("PARTICLE_SOURCE_DIR", "BEAM_CONFIGS"))
        self.var_npb.set(c.get("NUM_PARTICLES_PER_BEAMLET", 10001))
        self.var_radius.set(c.get("BEAMLET_RADIUS_M", 0.007))
        self.var_batch.set(c.get("PARTICLE_BATCH_SIZE", 2_500_000))
        spw = c.get("SOURCES_PER_WORKER")
        self.var_spw.set(str(spw) if spw is not None else "")
        self._refresh_bl_list()
        # Output
        self.var_outdir.set(c.get("DETAILED_OUTPUT_DIR", "OUTPUT"))
        self.var_SAVE_PARAVIEW_FILES.set(c.get("SAVE_PARAVIEW_FILES", True))
        self.var_SAVE_BINARY_POWERLOADS.set(c.get("SAVE_BINARY_POWERLOADS", False))
        self.var_SAVE_CSV_REPORTS.set(c.get("SAVE_CSV_REPORTS", False))
        self.var_RUN_VISUALIZATION_AFTER_SIM.set(c.get("RUN_VISUALIZATION_AFTER_SIM", False))
        self.var_VISUALIZE_ALL_RAYS.set(c.get("VISUALIZE_ALL_RAYS", False))
        self.var_ENABLE_VISUALIZATION.set(c.get("ENABLE_VISUALIZATION", True))
        self.var_summary.set(c.get("SUMMARY_CSV_FILENAME", "power_summary_by_object.csv"))
        self.var_nrays.set(c.get("NUM_RAYS_TO_SHOW_IN_PLOT", 0))
        # Smoothing
        self.var_smoother.set(c.get("RUN_SMOOTHER_AFTER_SIM", False))
        self.var_sm_radius.set(c.get("SMOOTHING_RADIUS", 0.02))
        mca = c.get("SMOOTHING_MAX_CELL_AREA")
        self.var_sm_mca.set(str(mca) if mca is not None else "")

    # ------------------------------------------------------------------
    #  Run simulation in background thread
    # ------------------------------------------------------------------
    def _run_sim(self):
        if self._sim_process and self._sim_process.poll() is None:
            messagebox.showinfo("Running", "A simulation is already running.")
            return

        # Save first
        self._save()

        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")
        self._log("▶ Starting simulation…\n\n")
        self._set_status("Simulation running…")

        def _worker():
            try:
                if self.var_sdcc.get():
                    # Wrap in srun: allocate an exclusive compute node,
                    # run the simulation, then exit the srun shell.
                    shell_cmd = (
                        f'srun --exclusive --pty /bin/bash -c '
                        f'"{_PYTHON} {_RUN_SIMULATION}"'
                    )
                    self._sim_process = subprocess.Popen(
                        ["bash", "-l", "-c", shell_cmd],
                        cwd=_SCRIPT_DIR,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1)
                else:
                    self._sim_process = subprocess.Popen(
                        [_PYTHON, _RUN_SIMULATION],
                        cwd=_SCRIPT_DIR,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1)
                for line in self._sim_process.stdout:
                    self._log(line)
                self._sim_process.wait()
                rc = self._sim_process.returncode
                if rc == 0:
                    self._log(f"\n✔ Simulation finished.\n")
                    self.after(0, lambda: self._set_status("Simulation completed successfully"))
                else:
                    self._log(f"\n✖ Simulation exited with code {rc}.\n")
                    self.after(0, lambda: self._set_status(f"Simulation failed (code {rc})"))
            except Exception as e:
                self._log(f"\n✖ Error: {e}\n")
                self.after(0, lambda: self._set_status("Simulation error"))

        threading.Thread(target=_worker, daemon=True).start()

    def _stop_sim(self):
        if self._sim_process and self._sim_process.poll() is None:
            self._sim_process.terminate()
            self._log("\n⏹ Simulation terminated by user.\n")
            self._set_status("Simulation stopped")

    def _log(self, text):
        """Thread-safe append to the log widget."""
        def _append():
            self.log_text.config(state="normal")
            self.log_text.insert("end", text)
            self.log_text.see("end")
            self.log_text.config(state="disabled")
        self.after(0, _append)


# ===================================================================
#  Small pick-list dialog
# ===================================================================
class _PickDialog(tk.Toplevel):
    def __init__(self, parent, title, items):
        super().__init__(parent)
        self.title(title)
        self.geometry("380x340")
        self.transient(parent)
        self.grab_set()
        self.configure(bg="#f0f2f5")
        self.result = None

        ttk.Label(self, text="Select a result set:",
                   font=("Segoe UI", 11, "bold"),
                   background="#f0f2f5", foreground="#2563eb").pack(
            anchor="w", padx=12, pady=(12, 6))

        lb = tk.Listbox(self, height=12, font=("Segoe UI", 10),
                          bg="white", fg="#1e293b",
                          selectbackground="#2563eb",
                          selectforeground="white",
                          highlightthickness=0, bd=1, relief="solid")
        lb.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        for item in items:
            lb.insert("end", item)
        lb.selection_set(0)

        def _ok():
            sel = lb.curselection()
            if sel:
                self.result = lb.get(sel[0])
            self.destroy()

        ttk.Button(self, text="Open in ParaView", command=_ok).pack(pady=(0, 12))
        lb.bind("<Double-1>", lambda e: _ok())


# ===================================================================
#  Entry point
# ===================================================================
if __name__ == "__main__":
    app = SimGUI()
    app.mainloop()
