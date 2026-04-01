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

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; we blit to Tk canvases
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import viewer  # built-in Open3D viewer

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
_IS_FROZEN = getattr(sys, 'frozen', False)  # True when running from PyInstaller exe
_SCRIPT_DIR = (os.path.dirname(sys.executable) if _IS_FROZEN
               else os.path.dirname(os.path.abspath(__file__)))
_CONFIG_JSON = os.path.join(_SCRIPT_DIR, "config.json")
_RUN_SIMULATION = os.path.join(_SCRIPT_DIR, "run_simulation.py")
_PYTHON = sys.executable  # the same Python that launched the GUI


# ===================================================================
#  Helper: load / save JSON directly (no import config, to stay clean)
# ===================================================================
def _resolve_path(relative_path):
    """Resolve a simulation-file path relative to the main application folder."""
    return os.path.join(_SCRIPT_DIR, relative_path)


def load_config():
    with open(_CONFIG_JSON, "r") as f:
        return json.load(f)

def save_config(data):
    with open(_CONFIG_JSON, "w") as f:
        json.dump(data, f, indent=4)


# ===================================================================
#  Simulation runner for frozen (PyInstaller) mode
# ===================================================================
class _SimulationStream:
    """Capture stdout/stderr and yield lines for the GUI logger."""
    def __init__(self):
        self.lines = []

    def write(self, msg):
        if msg:
            self.lines.append(msg)

    def flush(self):
        pass


def _run_simulation_frozen(log_fn=None):
    """Run the bundled run_simulation module directly when frozen."""
    class _LiveStream:
        def write(self, text):
            if text and log_fn:
                log_fn(text)
        def flush(self):
            pass

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    old_cwd = os.getcwd()

    sys.stdout = _LiveStream()
    sys.stderr = _LiveStream()

    try:
        os.chdir(_SCRIPT_DIR)
        import run_simulation
        run_simulation.main()
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        os.chdir(old_cwd)


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
        folder_abs = _resolve_path(folder)
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
    """Write a temp script and launch ParaView.

    On Linux the EasyBuild module *pv_module* is loaded first (via
    ``bash -lc 'ml …'``) so that LD_LIBRARY_PATH etc. are set.

    On Windows ParaView is invoked directly — no module system needed.
    """
    tmp_script = os.path.join(_SCRIPT_DIR, ".pv_temp_script.py")
    with open(tmp_script, "w") as f:
        f.write(script_content)

    try:
        if sys.platform == "win32":
            # Windows: call ParaView executable directly
            proc = subprocess.Popen(
                [pv_path, "--script=" + tmp_script],
                cwd=_SCRIPT_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True)
        else:
            # Linux / macOS: load EasyBuild module, then launch
            shell_cmd = (f'ml {pv_module} 2>/dev/null; '
                         f'"{pv_path}" --script="{tmp_script}"')
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
                    try:
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
        self._build_run_tab(notebook)
        self._build_results_tab(notebook)

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

        # ParaView module (EasyBuild) — only shown on Linux
        self.var_pv_module = tk.StringVar(value=self.cfg.get("PARAVIEW_MODULE", "ParaView"))
        if sys.platform != "win32":
            ttk.Label(pv_card, text="ParaView module (ml):", style="Card.TLabel").grid(
                row=1, column=0, sticky="w", pady=5)
            ttk.Entry(pv_card, textvariable=self.var_pv_module, width=30).grid(
                row=1, column=1, sticky="w", padx=(8, 0))

        pv_card.columnconfigure(1, weight=1)

    def _browse_pv(self):
        if sys.platform == "win32":
            ftypes = [("Executable files", "*.exe"), ("All files", "*")]
        else:
            ftypes = [("All files", "*")]
        p = filedialog.askopenfilename(title="Select ParaView executable",
                                       filetypes=ftypes)
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
        folder_frm = ttk.Frame(dlg)
        folder_frm.grid(row=row, column=1, sticky="we", padx=8)
        e = ttk.Entry(folder_frm, textvariable=v_folder, width=24)
        e.pack(side="left", fill="x", expand=True)

        def _browse_geo_folder():
            d = filedialog.askdirectory(
                initialdir=_SCRIPT_DIR,
                title="Select geometry folder",
                parent=dlg,
            )
            if d:
                try:
                    rel = os.path.relpath(d, _SCRIPT_DIR)
                except ValueError:
                    rel = d  # different drive on Windows
                if existing_folder:
                    e.config(state="normal")
                v_folder.set(rel)
                if existing_folder:
                    e.config(state="disabled")

        ttk.Button(folder_frm, text="Browse…", command=_browse_geo_folder).pack(
            side="left", padx=(4, 0)
        )
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
            # If the folder was renamed via Browse, remove the old key
            if existing_folder and folder_name != existing_folder:
                self.cfg["GEOMETRY_FOLDERS"].pop(existing_folder, None)
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
        ttk.Label(card, text="Beamlet grid radius (m):", style="Card.TLabel").grid(
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
        src_abs = _resolve_path(src) if not os.path.isabs(src) else src
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

        card.columnconfigure(1, weight=1)

        # --- File options card ---
        opts_card = self._make_card(outer, "Save Options")

        checkboxes = [
            ("SAVE_PARAVIEW_FILES", "Save ParaView (.vtp) files"),
            ("SAVE_CSV_REPORTS", "Save CSV reports"),
        ]
        for key, label in checkboxes:
            v = tk.BooleanVar(value=self.cfg.get(key, False))
            ttk.Checkbutton(opts_card, text=label, variable=v,
                             style="Card.TCheckbutton").pack(anchor="w", pady=2)
            setattr(self, f"var_{key}", v)

        ttk.Button(opts_card, text="📤 Extract results data…",
                    style="Secondary.TButton",
                    command=self._open_extract_dialog).pack(
                        anchor="w", pady=(8, 0))

        # --- Post-Processing Smoother card ---
        sm_card = self._make_card(outer, "Post-Processing Smoother")

        self.var_smoother = tk.BooleanVar(value=self.cfg.get("RUN_SMOOTHER_AFTER_SIM", False))
        ttk.Checkbutton(sm_card, text="Run batch smoother after simulation",
                         variable=self.var_smoother,
                         style="Card.TCheckbutton").pack(anchor="w", pady=(0, 8))

        sm_grid = ttk.Frame(sm_card, style="Card.TFrame")
        sm_grid.pack(fill="x")

        ttk.Label(sm_grid, text="Smoothing radius (m):", style="Card.TLabel").grid(
            row=0, column=0, sticky="w", pady=5)
        self.var_sm_radius = tk.DoubleVar(value=self.cfg.get("SMOOTHING_RADIUS", 0.02))
        ttk.Entry(sm_grid, textvariable=self.var_sm_radius, width=12).grid(
            row=0, column=1, sticky="w", padx=(8, 0))

        ttk.Label(sm_grid, text="Max cell area (m², empty=None):", style="Card.TLabel").grid(
            row=1, column=0, sticky="w", pady=5)
        mca = self.cfg.get("SMOOTHING_MAX_CELL_AREA")
        self.var_sm_mca = tk.StringVar(value=str(mca) if mca is not None else "")
        ttk.Entry(sm_grid, textvariable=self.var_sm_mca, width=12).grid(
            row=1, column=1, sticky="w", padx=(8, 0))

    def _refresh_csv_result_sets(self):
        """Scan the output directory and populate the simulation check-list."""
        outdir = self.var_outdir.get()
        outdir_abs = (_resolve_path(outdir)
                      if not os.path.isabs(outdir) else outdir)
        sets = []
        if os.path.isdir(outdir_abs):
            for d in sorted(os.listdir(outdir_abs)):
                sub = os.path.join(outdir_abs, d)
                if os.path.isdir(sub):
                    sets.append(d)
        # Rebuild checkbox list
        for w in self._csv_sim_inner.winfo_children():
            w.destroy()
        self._csv_sim_vars.clear()
        for name in sets:
            var = tk.BooleanVar(value=True)
            cb = ttk.Checkbutton(self._csv_sim_inner, text=name,
                                  variable=var, style="Card.TCheckbutton")
            cb.pack(anchor="w", padx=2, pady=1)
            self._csv_sim_vars[name] = var

    def _csv_select_all(self):
        for v in self._csv_sim_vars.values():
            v.set(True)

    def _csv_select_none(self):
        for v in self._csv_sim_vars.values():
            v.set(False)

    def _chart_comp_select_all(self):
        for d in self._chart_comp_vars.values():
            d["var"].set(True)

    def _chart_comp_select_none(self):
        for d in self._chart_comp_vars.values():
            d["var"].set(False)

    def _refresh_chart_comp_list(self):
        """Rebuild the component checklist from loaded plot data.

        Each component gets a checkbox, an editable label (defaults to
        the STL / object name), and a multiplier entry (defaults to 1.0).
        Previous values are preserved when the list is rebuilt.
        """
        # Collect union of all object names in display order
        all_objects = []
        for entries in self._csv_plot_data.values():
            for e in entries:
                if e["name"] not in all_objects:
                    all_objects.append(e["name"])

        # Preserve previous settings where possible
        prev = {}
        for k, d in self._chart_comp_vars.items():
            prev[k] = {
                "checked": d["var"].get(),
                "label": d["label"].get(),
                "mult": d["mult"].get(),
            }

        for w in self._chart_comp_inner.winfo_children():
            w.destroy()
        self._chart_comp_vars.clear()

        # Header row
        hdr = ttk.Frame(self._chart_comp_inner, style="Card.TFrame")
        hdr.pack(fill="x", padx=2, pady=(0, 2))
        ttk.Label(hdr, text="✓", style="Card.TLabel",
                  width=2).pack(side="left")
        ttk.Label(hdr, text="Label", style="Card.TLabel",
                  width=14).pack(side="left", padx=(2, 0))
        ttk.Label(hdr, text="Mult", style="Card.TLabel",
                  width=5).pack(side="left", padx=(2, 0))

        for name in all_objects:
            p = prev.get(name, {})
            var = tk.BooleanVar(value=p.get("checked", True))
            label_var = tk.StringVar(value=p.get("label", name))
            mult_var = tk.StringVar(value=p.get("mult", "1.0"))

            row_frm = ttk.Frame(self._chart_comp_inner, style="Card.TFrame")
            row_frm.pack(fill="x", padx=2, pady=1)

            ttk.Checkbutton(row_frm, variable=var,
                             style="Card.TCheckbutton").pack(
                                 side="left")
            lbl_entry = ttk.Entry(row_frm, textvariable=label_var, width=14,
                                   font=("Segoe UI", 9))
            lbl_entry.pack(side="left", padx=(2, 0))
            mult_entry = ttk.Entry(row_frm, textvariable=mult_var, width=5,
                                    font=("Segoe UI", 9))
            mult_entry.pack(side="left", padx=(2, 0))

            self._chart_comp_vars[name] = {
                "var": var,
                "label": label_var,
                "mult": mult_var,
            }

    def _load_summary_csv(self):
        """Load summary CSV data for all selected simulations.

        For each selected simulation, always loads the raw CSV first,
        then — if the 'Use smoothed' checkbox is checked — overlays
        smoothed values where available.

        The table shows the first selected simulation; the bar plots
        compare all selected simulations side-by-side.
        """
        import csv as csv_mod

        selected = [name for name, var in self._csv_sim_vars.items()
                     if var.get()]
        if not selected:
            self.var_csv_status.set("No simulations selected.")
            return

        outdir = self.var_outdir.get()
        outdir_abs = (_resolve_path(outdir)
                      if not os.path.isabs(outdir) else outdir)

        # --- helper: find first existing CSV from a list of candidates ---
        def _find_csv(search_dir, extra_candidates=None):
            candidates = list(extra_candidates or [])
            if os.path.isdir(search_dir):
                for f in sorted(glob.glob(os.path.join(search_dir, "*.csv"))):
                    if f not in candidates:
                        candidates.append(f)
            for c in candidates:
                if os.path.isfile(c):
                    return c
            return None

        # --- helper: read CSV → list[dict] ---
        def _read_csv(path):
            with open(path, "r", newline="") as fh:
                return list(csv_mod.DictReader(fh))

        # --- helper: detect column keys ---
        def _detect_keys(rows):
            name_key = power_key = density_key = None
            if not rows:
                return None, None, None
            keys = list(rows[0].keys())
            for k in keys:
                kl = k.lower()
                if "name" in kl or "file" in kl:
                    name_key = k
                elif "total" in kl and "power" in kl:
                    power_key = k
                elif "peak" in kl or "density" in kl:
                    density_key = k
            if name_key is None and len(keys) >= 1:
                name_key = keys[0]
            if power_key is None and len(keys) >= 2:
                power_key = keys[1]
            if density_key is None and len(keys) >= 3:
                density_key = keys[2]
            return name_key, power_key, density_key

        # --- helper: normalise object name for matching ---
        def _norm(name):
            base = os.path.splitext(os.path.basename(str(name)))[0]
            for prefix in ("results_", "smoothed_"):
                if base.startswith(prefix):
                    base = base[len(prefix):]
                    break
            return base

        # --- Load data for every selected simulation ---
        self._csv_plot_data.clear()
        first_display_order = None
        first_merged = None
        first_raw_path = None
        n_loaded = 0
        n_errors = 0

        for result_set in selected:
            base_dir = os.path.join(outdir_abs, result_set)
            raw_candidates = [
                os.path.join(base_dir, f"summary_{result_set}.csv"),
                os.path.join(base_dir, "summary_report.csv"),
                os.path.join(base_dir, "power_summary_by_object.csv"),
            ]
            raw_path = _find_csv(base_dir, raw_candidates)
            if raw_path is None:
                n_errors += 1
                continue

            try:
                raw_rows = _read_csv(raw_path)
            except Exception:
                n_errors += 1
                continue

            raw_nk, raw_pk, raw_dk = _detect_keys(raw_rows)

            merged = {}
            display_order = []
            for row in raw_rows:
                obj_raw = row.get(raw_nk, "?") if raw_nk else "?"
                obj = _norm(obj_raw)
                tp = row.get(raw_pk, "N/A") if raw_pk else "N/A"
                pd_ = row.get(raw_dk, "N/A") if raw_dk else "N/A"
                merged[obj] = {"name": obj, "tp": tp, "pd": pd_,
                               "source": "raw"}
                if obj not in display_order:
                    display_order.append(obj)

            # Overlay smoothed if requested
            if self.var_csv_use_smoothed.get():
                sm_dir = os.path.join(base_dir, "SMOOTHED")
                sm_candidates = [
                    os.path.join(sm_dir, "smoothed_summary.csv"),
                    os.path.join(sm_dir, "summary_report.csv"),
                ]
                smoothed_path = _find_csv(sm_dir, sm_candidates)
                if smoothed_path:
                    try:
                        sm_rows = _read_csv(smoothed_path)
                        sm_nk, sm_pk, sm_dk = _detect_keys(sm_rows)
                        for row in sm_rows:
                            obj_raw = row.get(sm_nk, "?") if sm_nk else "?"
                            obj = _norm(obj_raw)
                            tp = row.get(sm_pk, "N/A") if sm_pk else "N/A"
                            pd_ = row.get(sm_dk, "N/A") if sm_dk else "N/A"
                            if obj in merged:
                                merged[obj] = {
                                    "name": merged[obj]["name"],
                                    "tp": tp, "pd": pd_,
                                    "source": "smoothed"}
                            else:
                                merged[obj] = {
                                    "name": obj, "tp": tp, "pd": pd_,
                                    "source": "smoothed"}
                                display_order.append(obj)
                    except Exception:
                        pass

            # Store
            entries = [merged[o] for o in display_order]
            self._csv_plot_data[result_set] = entries
            n_loaded += 1

            if first_merged is None:
                first_merged = merged
                first_display_order = display_order
                first_raw_path = raw_path

        # --- Populate treeview with first simulation ---
        for item in self.csv_tree.get_children():
            self.csv_tree.delete(item)

        total_power_sum = 0.0
        peak_max = 0.0
        if first_merged and first_display_order:
            for obj in first_display_order:
                d = first_merged[obj]
                self.csv_tree.insert("", "end",
                                     values=(d["name"], d["tp"], d["pd"],
                                             d["source"]))
                try:
                    total_power_sum += float(d["tp"])
                except (ValueError, TypeError):
                    pass
                try:
                    peak_max = max(peak_max, float(d["pd"]))
                except (ValueError, TypeError):
                    pass

        if total_power_sum > 0:
            self.var_csv_total_power.set(
                f"Total deposited power:  {total_power_sum:.4e} W")
        else:
            self.var_csv_total_power.set("")

        info_parts = []
        if first_raw_path:
            info_parts.append(os.path.basename(first_raw_path))
        info_parts.append(f"{n_loaded} sim(s) loaded")
        if n_errors:
            info_parts.append(f"{n_errors} failed")
        if peak_max > 0:
            info_parts.append(f"Max peak = {peak_max:.4e} W/m²")
        self.var_csv_status.set("  |  ".join(info_parts))

        # --- Refresh component checklist & update bar plots ---
        self._refresh_chart_comp_list()
        self._update_csv_bar_plots()

    # ------------------------------------------------------------------
    #  Bar-plot rendering
    # ------------------------------------------------------------------
    def _update_csv_bar_plots(self):
        """Redraw the two bar charts from self._csv_plot_data."""
        import numpy as np

        ax_peak = self._csv_ax_peak
        ax_power = self._csv_ax_power
        ax_peak.clear()
        ax_power.clear()

        data = self._csv_plot_data  # { sim_name: [{ name, tp, pd, source }] }
        if not data:
            ax_peak.set_title("Peak Heat Load [W/m²]", fontsize=10)
            ax_power.set_title("Total Power [W]", fontsize=10)
            self._csv_fig.tight_layout()
            self._csv_canvas_mpl.draw()
            return

        sim_names = list(data.keys())

        # Collect union of all object names, filtered by component checklist
        all_objects = []
        for sim in sim_names:
            for entry in data[sim]:
                if entry["name"] not in all_objects:
                    all_objects.append(entry["name"])
        # Apply component filter (if checklist is populated)
        if self._chart_comp_vars:
            all_objects = [o for o in all_objects
                           if self._chart_comp_vars.get(
                               o, {"var": tk.BooleanVar(value=True)}
                           )["var"].get()]

        n_objects = len(all_objects)
        n_sims = len(sim_names)

        if n_objects == 0:
            self._csv_fig.tight_layout()
            self._csv_canvas_mpl.draw()
            return

        # Colour palette
        cmap = plt.get_cmap("tab10")
        colours = [cmap(i % 10) for i in range(n_sims)]

        # Build per-object multipliers from the component checklist
        obj_mult = {}
        for obj in all_objects:
            d = self._chart_comp_vars.get(obj)
            if d:
                try:
                    obj_mult[obj] = float(d["mult"].get())
                except (ValueError, TypeError):
                    obj_mult[obj] = 1.0
            else:
                obj_mult[obj] = 1.0

        x = np.arange(n_objects)
        total_bar_width = 0.8
        bar_width = total_bar_width / max(n_sims, 1)

        for si, sim in enumerate(sim_names):
            # Build lookup for this sim
            lookup = {e["name"]: e for e in data[sim]}
            peaks = []
            powers = []
            for obj in all_objects:
                m = obj_mult[obj]
                e = lookup.get(obj)
                if e:
                    try:
                        peaks.append(float(e["pd"]) * m)
                    except (ValueError, TypeError):
                        peaks.append(0.0)
                    try:
                        powers.append(float(e["tp"]) * m)
                    except (ValueError, TypeError):
                        powers.append(0.0)
                else:
                    peaks.append(0.0)
                    powers.append(0.0)

            offset = (si - (n_sims - 1) / 2) * bar_width
            label = sim if len(sim) <= 30 else sim[:27] + "…"
            ax_peak.bar(x + offset, peaks, bar_width * 0.9,
                        label=label, color=colours[si], edgecolor="white",
                        linewidth=0.5)
            ax_power.bar(x + offset, powers, bar_width * 0.9,
                         label=label, color=colours[si], edgecolor="white",
                         linewidth=0.5)

        # Build display labels from the component checklist
        display_labels = []
        for obj in all_objects:
            d = self._chart_comp_vars.get(obj)
            if d:
                display_labels.append(d["label"].get())
            else:
                display_labels.append(obj)

        # Formatting
        for ax, title, unit, log_var in [
            (ax_peak, "Peak Heat Load", "W/m²", self.var_chart_log_peak),
            (ax_power, "Total Deposited Power", "W", self.var_chart_log_power),
        ]:
            ax.set_title(f"{title} [{unit}]", fontsize=10, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(display_labels, rotation=45, ha="right",
                               fontsize=8)
            ax.set_ylabel(unit, fontsize=9)
            if log_var.get():
                ax.set_yscale("log")
            else:
                ax.set_yscale("linear")
                ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax.grid(axis="y", alpha=0.3, linewidth=0.5)
            ax.set_axisbelow(True)
            if n_sims > 1:
                ax.legend(fontsize=7, loc="upper right", framealpha=0.8)

        self._csv_fig.tight_layout()
        self._csv_canvas_mpl.draw()

    def _view_results(self):
        outdir = self.var_outdir.get()
        outdir_abs = _resolve_path(outdir) if not os.path.isabs(outdir) else outdir
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

    def _open_extract_dialog(self):
        """Open the VTP data extraction dialog."""
        outdir = self.var_outdir.get()
        outdir_abs = (_resolve_path(outdir)
                      if not os.path.isabs(outdir) else outdir)
        dlg = _ExtractDialog(self, outdir_abs)
        self.wait_window(dlg)

    # ------------------------------------------------------------------
    #  RESULTS tab (visualisation, summary reports, comparison charts)
    # ------------------------------------------------------------------
    def _build_results_tab(self, nb):
        outer_wrapper = ttk.Frame(nb)
        nb.add(outer_wrapper, text="  �  Results  ")

        top_pw = ttk.PanedWindow(outer_wrapper, orient="horizontal")
        top_pw.pack(fill="both", expand=True)

        # ============================================================
        # LEFT side — scrollable cards
        # ============================================================
        left_wrapper = ttk.Frame(top_pw)
        top_pw.add(left_wrapper, weight=1)

        res_canvas = tk.Canvas(left_wrapper, borderwidth=0,
                               highlightthickness=0,
                               bg=self._colours["bg"])
        res_vscroll = ttk.Scrollbar(left_wrapper, orient="vertical",
                                     command=res_canvas.yview)
        outer = ttk.Frame(res_canvas)
        outer.bind("<Configure>",
                   lambda e: res_canvas.configure(
                       scrollregion=res_canvas.bbox("all")))
        res_canvas.create_window((0, 0), window=outer, anchor="nw")
        res_canvas.configure(yscrollcommand=res_vscroll.set)
        res_canvas.pack(side="left", fill="both", expand=True)
        res_vscroll.pack(side="right", fill="y")
        # Mouse-wheel scrolling
        def _on_mousewheel(event):
            res_canvas.yview_scroll(-1 if event.num == 4 else 1, "units")
        res_canvas.bind("<Button-4>", _on_mousewheel)
        res_canvas.bind("<Button-5>", _on_mousewheel)
        outer.bind("<Button-4>", _on_mousewheel)
        outer.bind("<Button-5>", _on_mousewheel)

        # --- Visualisation card ---
        vis_card = self._make_card(outer, "Visualisation", pady=(12, 10))

        vis_checks = [
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
        ttk.Button(btn_frm, text="🔍 Results (ParaView)", style="Secondary.TButton",
                    command=self._view_results).pack(side="left")

        # --- Summary Reports card ---
        sum_card = self._make_card(outer, "Summary Reports")

        # Top row: smoothed checkbox + refresh + load buttons
        sel_frm = ttk.Frame(sum_card, style="Card.TFrame")
        sel_frm.pack(fill="x", pady=(0, 6))

        self.var_csv_use_smoothed = tk.BooleanVar(value=True)
        ttk.Checkbutton(sel_frm, text="Use smoothed if available",
                         variable=self.var_csv_use_smoothed,
                         style="Card.TCheckbutton").pack(side="left", padx=(0, 12))

        ttk.Button(sel_frm, text="↻ Refresh", style="Secondary.TButton",
                    command=self._refresh_csv_result_sets).pack(side="left", padx=(0, 4))
        ttk.Button(sel_frm, text="📊 Load Selected", style="Secondary.TButton",
                    command=self._load_summary_csv).pack(side="left", padx=4)

        # Simulation selector (checkable listbox)
        sim_lbl_frm = ttk.Frame(sum_card, style="Card.TFrame")
        sim_lbl_frm.pack(fill="x")
        ttk.Label(sim_lbl_frm, text="Simulations:",
                  style="Card.TLabel").pack(side="left")
        ttk.Button(sim_lbl_frm, text="All", style="Secondary.TButton",
                    command=self._csv_select_all).pack(side="right", padx=2)
        ttk.Button(sim_lbl_frm, text="None", style="Secondary.TButton",
                    command=self._csv_select_none).pack(side="right", padx=2)

        sim_list_frm = ttk.Frame(sum_card, style="Card.TFrame")
        sim_list_frm.pack(fill="x", pady=(2, 6))
        self._csv_sim_vars = {}  # name → BooleanVar
        sim_canvas = tk.Canvas(sim_list_frm, height=80, borderwidth=0,
                               highlightthickness=0, bg="white")
        sim_sb = ttk.Scrollbar(sim_list_frm, orient="vertical",
                                command=sim_canvas.yview)
        self._csv_sim_inner = ttk.Frame(sim_canvas, style="Card.TFrame")
        self._csv_sim_inner.bind(
            "<Configure>",
            lambda e: sim_canvas.configure(
                scrollregion=sim_canvas.bbox("all")))
        sim_canvas.create_window((0, 0), window=self._csv_sim_inner,
                                  anchor="nw")
        sim_canvas.configure(yscrollcommand=sim_sb.set)
        sim_canvas.pack(side="left", fill="both", expand=True)
        sim_sb.pack(side="right", fill="y")
        self._csv_sim_canvas = sim_canvas

        # Treeview for CSV data (shows first selected sim)
        ttk.Label(sum_card, text="Table (first selected sim):",
                  style="Card.TLabel").pack(anchor="w", pady=(2, 2))
        csv_tree_frm = ttk.Frame(sum_card, style="Card.TFrame")
        csv_tree_frm.pack(fill="both", expand=True)

        csv_cols = ("object", "total_power", "peak_density", "source")
        self.csv_tree = ttk.Treeview(csv_tree_frm, columns=csv_cols,
                                      show="headings", height=8)
        self.csv_tree.heading("object", text="Object")
        self.csv_tree.heading("total_power", text="Total Power [W]")
        self.csv_tree.heading("peak_density", text="Peak Density [W/m²]")
        self.csv_tree.heading("source", text="Source")
        self.csv_tree.column("object", width=160, anchor="w")
        self.csv_tree.column("total_power", width=130, anchor="e")
        self.csv_tree.column("peak_density", width=130, anchor="e")
        self.csv_tree.column("source", width=70, anchor="center")

        csv_vsb = ttk.Scrollbar(csv_tree_frm, orient="vertical",
                                 command=self.csv_tree.yview)
        self.csv_tree.configure(yscrollcommand=csv_vsb.set)
        self.csv_tree.pack(side="left", fill="both", expand=True)
        csv_vsb.pack(side="right", fill="y")

        # Total power row
        self.var_csv_total_power = tk.StringVar(value="")
        ttk.Label(sum_card, textvariable=self.var_csv_total_power,
                  style="Card.TLabel",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(6, 0))

        # Status / totals row
        self.var_csv_status = tk.StringVar(
            value="Refresh, select simulations, then click Load.")
        ttk.Label(sum_card, textvariable=self.var_csv_status,
                  style="Card.TLabel",
                  foreground=self._colours["dim"]).pack(anchor="w", pady=(4, 0))

        # ============================================================
        # RIGHT side — component filter + matplotlib bar charts
        # ============================================================
        right_frm = ttk.Frame(top_pw, style="Card.TFrame", padding=8)
        top_pw.add(right_frm, weight=1)

        ttk.Label(right_frm, text="Comparison Charts",
                  style="CardHeader.TLabel").pack(anchor="w", pady=(0, 4))

        # Horizontal split: component list | plots
        chart_pw = ttk.PanedWindow(right_frm, orient="horizontal")
        chart_pw.pack(fill="both", expand=True)

        # ---- Component filter (left of plots) ----
        comp_frm = ttk.Frame(chart_pw, style="Card.TFrame")
        chart_pw.add(comp_frm, weight=0)

        comp_hdr = ttk.Frame(comp_frm, style="Card.TFrame")
        comp_hdr.pack(fill="x", pady=(0, 2))
        ttk.Label(comp_hdr, text="Components:",
                  style="Card.TLabel").pack(anchor="w")
        btn_row_comp = ttk.Frame(comp_frm, style="Card.TFrame")
        btn_row_comp.pack(fill="x", pady=(0, 2))
        ttk.Button(btn_row_comp, text="All", style="Secondary.TButton",
                    command=self._chart_comp_select_all).pack(
                        side="left", padx=(0, 2))
        ttk.Button(btn_row_comp, text="None", style="Secondary.TButton",
                    command=self._chart_comp_select_none).pack(
                        side="left", padx=2)
        ttk.Button(btn_row_comp, text="↻ Plot", style="Secondary.TButton",
                    command=self._update_csv_bar_plots).pack(
                        side="left", padx=2)

        # Log-scale checkboxes
        log_frm = ttk.Frame(comp_frm, style="Card.TFrame")
        log_frm.pack(fill="x", pady=(2, 2))
        self.var_chart_log_peak = tk.BooleanVar(value=False)
        ttk.Checkbutton(log_frm, text="Log scale (peak)",
                         variable=self.var_chart_log_peak,
                         style="Card.TCheckbutton").pack(anchor="w")
        self.var_chart_log_power = tk.BooleanVar(value=False)
        ttk.Checkbutton(log_frm, text="Log scale (power)",
                         variable=self.var_chart_log_power,
                         style="Card.TCheckbutton").pack(anchor="w")

        comp_list_frm = ttk.Frame(comp_frm, style="Card.TFrame")
        comp_list_frm.pack(fill="both", expand=True, pady=(2, 0))
        self._chart_comp_vars = {}  # object_name → BooleanVar
        comp_canvas = tk.Canvas(comp_list_frm, width=220, borderwidth=0,
                                highlightthickness=0, bg="white")
        comp_sb = ttk.Scrollbar(comp_list_frm, orient="vertical",
                                 command=comp_canvas.yview)
        self._chart_comp_inner = ttk.Frame(comp_canvas, style="Card.TFrame")
        self._chart_comp_inner.bind(
            "<Configure>",
            lambda e: comp_canvas.configure(
                scrollregion=comp_canvas.bbox("all")))
        comp_canvas.create_window((0, 0), window=self._chart_comp_inner,
                                   anchor="nw")
        comp_canvas.configure(yscrollcommand=comp_sb.set)
        comp_canvas.pack(side="left", fill="both", expand=True)
        comp_sb.pack(side="right", fill="y")

        # ---- Plots (right of component list) ----
        plot_frm = ttk.Frame(chart_pw, style="Card.TFrame")
        chart_pw.add(plot_frm, weight=1)

        self._csv_fig = Figure(figsize=(5, 6), dpi=100,
                                facecolor="white", tight_layout=True)
        self._csv_ax_peak = self._csv_fig.add_subplot(2, 1, 1)
        self._csv_ax_power = self._csv_fig.add_subplot(2, 1, 2)
        self._csv_canvas_mpl = FigureCanvasTkAgg(self._csv_fig, plot_frm)
        self._csv_canvas_mpl.get_tk_widget().pack(fill="both", expand=True)

        # Initialise empty chart labels
        self._csv_ax_peak.set_title("Peak Heat Load [W/m²]", fontsize=10)
        self._csv_ax_power.set_title("Total Power [W]", fontsize=10)
        self._csv_fig.tight_layout()
        self._csv_canvas_mpl.draw()

        # Internal data store:  { sim_name: [{ name, tp, pd, source }, …] }
        self._csv_plot_data = {}

        # Populate the simulation checklist on build
        self._refresh_csv_result_sets()

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

        # SDCC checkbox (Linux HPC only — hidden on Windows)
        self.var_sdcc = tk.BooleanVar(value=False)
        if sys.platform != "win32":
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
        d["SAVE_CSV_REPORTS"] = self.var_SAVE_CSV_REPORTS.get()
        d["ENABLE_VISUALIZATION"] = self.var_ENABLE_VISUALIZATION.get()
        d["SUMMARY_CSV_FILENAME"] = self.var_summary.get()
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
        self.var_SAVE_CSV_REPORTS.set(c.get("SAVE_CSV_REPORTS", False))
        self.var_ENABLE_VISUALIZATION.set(c.get("ENABLE_VISUALIZATION", True))
        self.var_summary.set(c.get("SUMMARY_CSV_FILENAME", "power_summary_by_object.csv"))
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
                if _IS_FROZEN:
                    # Running from PyInstaller exe: stream output live via log_fn
                    _run_simulation_frozen(log_fn=self._log)
                    self._sim_process = None
                    rc = 0
                elif self.var_sdcc.get():
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
                    for line in self._sim_process.stdout:
                        self._log(line)
                    self._sim_process.wait()
                    rc = self._sim_process.returncode
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
#  VTP data extraction dialog
# ===================================================================
class _ExtractDialog(tk.Toplevel):
    """Dialog for extracting mesh cell data from a VTP file to CSV."""

    def __init__(self, parent, initial_dir):
        super().__init__(parent)
        self.title("Extract Results Data")
        self.geometry("560x440")
        self.transient(parent)
        self.grab_set()
        self.configure(bg="#f0f2f5")
        self._parent = parent

        pad = dict(padx=12, pady=4)

        # --- Input VTP file ---
        ttk.Label(self, text="Input VTP file:",
                  font=("Segoe UI", 10, "bold"),
                  background="#f0f2f5").grid(
            row=0, column=0, sticky="w", **pad)

        self.var_input = tk.StringVar()
        ttk.Entry(self, textvariable=self.var_input, width=48).grid(
            row=0, column=1, sticky="we", padx=(0, 4), pady=4)
        ttk.Button(self, text="Browse…",
                    command=lambda: self._browse_vtp(initial_dir)).grid(
            row=0, column=2, padx=(0, 12), pady=4)

        # --- Output CSV file ---
        ttk.Label(self, text="Output CSV file:",
                  font=("Segoe UI", 10, "bold"),
                  background="#f0f2f5").grid(
            row=1, column=0, sticky="w", **pad)

        self.var_output = tk.StringVar()
        ttk.Entry(self, textvariable=self.var_output, width=48).grid(
            row=1, column=1, sticky="we", padx=(0, 4), pady=4)
        ttk.Button(self, text="Browse…",
                    command=self._browse_output).grid(
            row=1, column=2, padx=(0, 12), pady=4)

        # --- Properties to export ---
        ttk.Separator(self, orient="horizontal").grid(
            row=2, column=0, columnspan=3, sticky="we", padx=12, pady=8)

        prop_lbl = ttk.Label(self, text="Properties to export:",
                              font=("Segoe UI", 10, "bold"),
                              background="#f0f2f5")
        prop_lbl.grid(row=3, column=0, sticky="w", **pad)

        prop_frm = ttk.Frame(self)
        prop_frm.grid(row=4, column=0, columnspan=3, sticky="w", padx=16)

        self.var_geometry = tk.BooleanVar(value=True)
        ttk.Checkbutton(prop_frm, text="Export geometry (X, Y, Z)",
                         variable=self.var_geometry).pack(anchor="w", pady=1)

        self.var_area = tk.BooleanVar(value=True)
        ttk.Checkbutton(prop_frm, text="Export cell area",
                         variable=self.var_area).pack(anchor="w", pady=1)

        self.var_power = tk.BooleanVar(value=True)
        ttk.Checkbutton(prop_frm, text="Export power (Deposited_Power_W)",
                         variable=self.var_power).pack(anchor="w", pady=1)

        self.var_powerload = tk.BooleanVar(value=True)
        ttk.Checkbutton(prop_frm, text="Export power load (Power_Density_W_m2)",
                         variable=self.var_powerload).pack(anchor="w", pady=1)

        # --- Multiplier ---
        ttk.Separator(self, orient="horizontal").grid(
            row=5, column=0, columnspan=3, sticky="we", padx=12, pady=8)

        mult_frm = ttk.Frame(self)
        mult_frm.grid(row=6, column=0, columnspan=3, sticky="w", padx=12)

        ttk.Label(mult_frm, text="Multiplication factor:",
                  font=("Segoe UI", 10, "bold"),
                  background="#f0f2f5").pack(side="left")
        self.var_mult = tk.StringVar(value="1.0")
        ttk.Entry(mult_frm, textvariable=self.var_mult, width=10).pack(
            side="left", padx=(8, 0))
        ttk.Label(mult_frm, text="(applied to power & power load)",
                  background="#f0f2f5",
                  foreground="#64748b").pack(side="left", padx=(8, 0))

        # --- Ignore zeros ---
        self.var_ignore_zeros = tk.BooleanVar(value=False)
        ttk.Checkbutton(self, text="Ignore zero-valued rows",
                         variable=self.var_ignore_zeros).grid(
            row=7, column=0, columnspan=3, sticky="w", padx=16, pady=(8, 4))

        # --- Status label ---
        self.var_status = tk.StringVar(value="")
        ttk.Label(self, textvariable=self.var_status,
                  background="#f0f2f5", foreground="#64748b",
                  font=("Segoe UI", 9)).grid(
            row=8, column=0, columnspan=3, sticky="w", padx=12, pady=(4, 0))

        # --- Buttons ---
        btn_frm = ttk.Frame(self)
        btn_frm.grid(row=9, column=0, columnspan=3, pady=12)

        ttk.Button(btn_frm, text="💾  Save CSV",
                    command=self._do_extract).pack(side="left", padx=4)
        ttk.Button(btn_frm, text="Cancel",
                    command=self.destroy).pack(side="left", padx=4)

        self.columnconfigure(1, weight=1)

    def _browse_vtp(self, initial_dir):
        p = filedialog.askopenfilename(
            parent=self,
            initialdir=initial_dir,
            title="Select VTP file",
            filetypes=[("VTP files", "*.vtp"), ("All files", "*")])
        if p:
            self.var_input.set(p)
            # Auto-fill output name
            if not self.var_output.get():
                base = os.path.splitext(p)[0]
                self.var_output.set(base + "_extracted.csv")

    def _browse_output(self):
        init_dir = os.path.dirname(self.var_input.get()) or _SCRIPT_DIR
        p = filedialog.asksaveasfilename(
            parent=self,
            initialdir=init_dir,
            title="Save CSV As",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*")])
        if p:
            self.var_output.set(p)

    def _do_extract(self):
        """Run the extraction in a background thread."""
        input_vtp = self.var_input.get().strip()
        output_csv = self.var_output.get().strip()
        if not input_vtp:
            messagebox.showwarning("Missing", "Please select an input VTP file.",
                                    parent=self)
            return
        if not output_csv:
            messagebox.showwarning("Missing", "Please specify an output file.",
                                    parent=self)
            return
        if not os.path.isfile(input_vtp):
            messagebox.showerror("Not found",
                                  f"Input file not found:\n{input_vtp}",
                                  parent=self)
            return

        try:
            mult = float(self.var_mult.get())
        except ValueError:
            messagebox.showwarning("Invalid", "Multiplication factor must be a number.",
                                    parent=self)
            return

        export_geom = self.var_geometry.get()
        export_area = self.var_area.get()
        export_power = self.var_power.get()
        export_pload = self.var_powerload.get()
        ignore_zeros = self.var_ignore_zeros.get()

        if not any([export_geom, export_area, export_power, export_pload]):
            messagebox.showwarning("Nothing selected",
                                    "Select at least one property to export.",
                                    parent=self)
            return

        self.var_status.set("Extracting…")
        self.update_idletasks()

        def _worker():
            try:
                import pyvista as pv
                import pandas as pd
                import numpy as np

                dataset = pv.read(input_vtp)
                if isinstance(dataset, pv.MultiBlock):
                    if dataset.n_blocks == 0:
                        self.after(0, lambda: self.var_status.set(
                            "ERROR: MultiBlock dataset is empty."))
                        return
                    mesh = dataset.combine()
                elif isinstance(dataset, pv.PolyData):
                    mesh = dataset
                else:
                    self.after(0, lambda: self.var_status.set(
                        f"ERROR: Unsupported type {type(dataset)}"))
                    return

                mesh.clean(inplace=True)
                if mesh.n_cells == 0:
                    self.after(0, lambda: self.var_status.set(
                        "ERROR: Mesh has 0 cells."))
                    return

                face_centers = mesh.cell_centers().points
                valid = np.isfinite(face_centers).all(axis=1)
                n_valid = int(np.sum(valid))
                if n_valid == 0:
                    self.after(0, lambda: self.var_status.set(
                        "ERROR: No cells with finite coordinates."))
                    return

                # Build DataFrame
                if export_geom:
                    df = pd.DataFrame(face_centers[valid],
                                       columns=["X", "Y", "Z"])
                else:
                    df = pd.DataFrame(index=range(n_valid))

                if export_area:
                    try:
                        mesh = mesh.compute_cell_sizes()
                        df["Area"] = mesh.cell_data["Area"][valid]
                    except Exception:
                        pass

                power_cols = []
                if export_power:
                    key = "Deposited_Power_W"
                    if key in mesh.cell_data:
                        df[key] = mesh.cell_data[key][valid] * mult
                        power_cols.append(key)

                if export_pload:
                    key = "Power_Density_W_m2"
                    if key in mesh.cell_data:
                        df[key] = mesh.cell_data[key][valid] * mult
                        power_cols.append(key)

                # Filter zeros
                if ignore_zeros and power_cols:
                    mask = pd.Series([True] * len(df))
                    for col in power_cols:
                        if col in df.columns:
                            mask &= (np.abs(df[col]) > 1e-12)
                    df = df[mask].reset_index(drop=True)

                n_rows = len(df)
                df.to_csv(output_csv, index=False, float_format="%.6e")

                self.after(0, lambda: self.var_status.set(
                    f"✔ Saved {n_rows} rows to {os.path.basename(output_csv)}"))
                self.after(0, lambda: messagebox.showinfo(
                    "Done",
                    f"Extracted {n_rows} rows to:\n{output_csv}",
                    parent=self))

            except Exception as e:
                self.after(0, lambda: self.var_status.set(f"ERROR: {e}"))

        threading.Thread(target=_worker, daemon=True).start()


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
def main():
    app = SimGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
