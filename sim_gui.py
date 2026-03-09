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
        self.geometry("920x720")
        self.minsize(800, 600)
        self.cfg = load_config()
        self._build_ui()

    # ------------------------------------------------------------------
    #  Build the tabbed interface
    # ------------------------------------------------------------------
    def _build_ui(self):
        self._build_menubar()

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=6, pady=6)

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

        self.bind_all("<Control-s>", lambda e: self._save())
        self.bind_all("<Control-Shift-S>", lambda e: self._save_as())
        self.bind_all("<Control-o>", lambda e: self._load_config_file())

    # ------------------------------------------------------------------
    #  GENERAL tab
    # ------------------------------------------------------------------
    def _build_general_tab(self, nb):
        frm = ttk.Frame(nb, padding=10)
        nb.add(frm, text=" General ")

        row = 0
        ttk.Label(frm, text="CPU cores (-1 = all):").grid(row=row, column=0, sticky="w", pady=4)
        self.var_cpu = tk.IntVar(value=self.cfg.get("NUM_CPU_CORES", 1))
        ttk.Spinbox(frm, from_=-1, to=256, textvariable=self.var_cpu, width=8).grid(row=row, column=1, sticky="w")

        row += 1
        self.var_diag = tk.BooleanVar(value=self.cfg.get("ENABLE_DIAGNOSTIC_SURFACES", False))
        ttk.Checkbutton(frm, text="Enable diagnostic (transparent) surfaces", variable=self.var_diag).grid(row=row, column=0, columnspan=2, sticky="w", pady=4)

        row += 1
        ttk.Label(frm, text="Geometry cache dir:").grid(row=row, column=0, sticky="w", pady=4)
        self.var_cache = tk.StringVar(value=self.cfg.get("GEOMETRY_CACHE_DIR", "geometry_cache"))
        ttk.Entry(frm, textvariable=self.var_cache, width=30).grid(row=row, column=1, sticky="w")

        row += 1
        ttk.Label(frm, text="Deposition fraction (0–1):").grid(row=row, column=0, sticky="w", pady=4)
        self.var_dep_frac = tk.DoubleVar(value=self.cfg.get("DEPOSITION_FRACTION", 1.0))
        ttk.Entry(frm, textvariable=self.var_dep_frac, width=10).grid(row=row, column=1, sticky="w")

        row += 1
        ttk.Label(frm, text="ParaView path:").grid(row=row, column=0, sticky="w", pady=4)
        self.var_pv_path = tk.StringVar(value=self.cfg.get("PARAVIEW_PATH", "paraview"))
        pv_entry = ttk.Entry(frm, textvariable=self.var_pv_path, width=60)
        pv_entry.grid(row=row, column=1, sticky="we", padx=(0, 4))
        ttk.Button(frm, text="Browse…", command=self._browse_pv).grid(row=row, column=2)

        row += 1
        ttk.Label(frm, text="ParaView module (ml):").grid(row=row, column=0, sticky="w", pady=4)
        self.var_pv_module = tk.StringVar(value=self.cfg.get("PARAVIEW_MODULE", "ParaView"))
        ttk.Entry(frm, textvariable=self.var_pv_module, width=30).grid(row=row, column=1, sticky="w")

        frm.columnconfigure(1, weight=1)

    def _browse_pv(self):
        p = filedialog.askopenfilename(title="Select ParaView executable",
                                       filetypes=[("All files", "*")])
        if p:
            self.var_pv_path.set(p)

    # ------------------------------------------------------------------
    #  GEOMETRY tab (editable table)
    # ------------------------------------------------------------------
    def _build_geometry_tab(self, nb):
        frm = ttk.Frame(nb, padding=10)
        nb.add(frm, text=" Geometry ")

        # Treeview
        cols = ("folder", "scale", "target_length", "save_details",
                "is_diagnostic", "save_impact_data", "max_impact_records")
        self.geo_tree = ttk.Treeview(frm, columns=cols, show="headings", height=8)
        headers = {"folder": "Folder", "scale": "Scale", "target_length": "Target Len",
                   "save_details": "Details", "is_diagnostic": "Diagnostic",
                   "save_impact_data": "Impacts", "max_impact_records": "Max Records"}
        widths = {"folder": 120, "scale": 60, "target_length": 90, "save_details": 60,
                  "is_diagnostic": 80, "save_impact_data": 60, "max_impact_records": 100}
        for c in cols:
            self.geo_tree.heading(c, text=headers[c])
            self.geo_tree.column(c, width=widths[c], anchor="center")
        self.geo_tree.column("folder", anchor="w")
        self.geo_tree.pack(fill="both", expand=True)

        self._populate_geo_tree()

        # Buttons
        btn_frm = ttk.Frame(frm)
        btn_frm.pack(fill="x", pady=(6, 0))
        ttk.Button(btn_frm, text="Add Folder…", command=self._add_geo_folder).pack(side="left", padx=2)
        ttk.Button(btn_frm, text="Edit Selected…", command=self._edit_geo_folder).pack(side="left", padx=2)
        ttk.Button(btn_frm, text="Remove Selected", command=self._remove_geo_folder).pack(side="left", padx=2)
        ttk.Button(btn_frm, text="🔍 View Geometry in ParaView", command=self._view_geometry).pack(side="right", padx=2)

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

    # ------------------------------------------------------------------
    #  PARTICLES tab
    # ------------------------------------------------------------------
    def _build_particles_tab(self, nb):
        frm = ttk.Frame(nb, padding=10)
        nb.add(frm, text=" Particles ")

        row = 0
        ttk.Label(frm, text="Beam config directory:").grid(row=row, column=0, sticky="w", pady=4)
        self.var_src_dir = tk.StringVar(value=self.cfg.get("PARTICLE_SOURCE_DIR", "BEAM_CONFIGS"))
        ttk.Entry(frm, textvariable=self.var_src_dir, width=30).grid(row=row, column=1, sticky="we")
        ttk.Button(frm, text="Browse…", command=lambda: self._browse_dir(self.var_src_dir)).grid(row=row, column=2, padx=4)

        row += 1
        ttk.Label(frm, text="Particles per beamlet:").grid(row=row, column=0, sticky="w", pady=4)
        self.var_npb = tk.IntVar(value=self.cfg.get("NUM_PARTICLES_PER_BEAMLET", 10001))
        ttk.Entry(frm, textvariable=self.var_npb, width=12).grid(row=row, column=1, sticky="w")

        row += 1
        ttk.Label(frm, text="Beamlet radius (m):").grid(row=row, column=0, sticky="w", pady=4)
        self.var_radius = tk.DoubleVar(value=self.cfg.get("BEAMLET_RADIUS_M", 0.007))
        ttk.Entry(frm, textvariable=self.var_radius, width=12).grid(row=row, column=1, sticky="w")

        row += 1
        ttk.Label(frm, text="Particle batch size:").grid(row=row, column=0, sticky="w", pady=4)
        self.var_batch = tk.IntVar(value=self.cfg.get("PARTICLE_BATCH_SIZE", 2_500_000))
        ttk.Entry(frm, textvariable=self.var_batch, width=12).grid(row=row, column=1, sticky="w")

        row += 1
        ttk.Label(frm, text="Sources per worker (empty=auto):").grid(row=row, column=0, sticky="w", pady=4)
        spw = self.cfg.get("SOURCES_PER_WORKER")
        self.var_spw = tk.StringVar(value=str(spw) if spw is not None else "")
        ttk.Entry(frm, textvariable=self.var_spw, width=12).grid(row=row, column=1, sticky="w")

        # List .bl files found
        row += 1
        ttk.Label(frm, text="Beam config files found:").grid(row=row, column=0, sticky="nw", pady=4)
        self.bl_listbox = tk.Listbox(frm, height=8, width=50)
        self.bl_listbox.grid(row=row, column=1, columnspan=2, sticky="we", pady=4)
        self._refresh_bl_list()
        ttk.Button(frm, text="Refresh", command=self._refresh_bl_list).grid(row=row+1, column=1, sticky="w")

        frm.columnconfigure(1, weight=1)

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
        frm = ttk.Frame(nb, padding=10)
        nb.add(frm, text=" Output ")

        row = 0
        ttk.Label(frm, text="Output directory:").grid(row=row, column=0, sticky="w", pady=4)
        self.var_outdir = tk.StringVar(value=self.cfg.get("DETAILED_OUTPUT_DIR", "OUTPUT"))
        ttk.Entry(frm, textvariable=self.var_outdir, width=30).grid(row=row, column=1, sticky="we")
        ttk.Button(frm, text="Browse…", command=lambda: self._browse_dir(self.var_outdir)).grid(row=row, column=2, padx=4)

        checkboxes = [
            ("SAVE_PARAVIEW_FILES", "Save ParaView (.vtp) files"),
            ("SAVE_BINARY_POWERLOADS", "Save binary (.npy) power loads"),
            ("SAVE_CSV_REPORTS", "Save CSV reports"),
            ("RUN_VISUALIZATION_AFTER_SIM", "Run visualisation after simulation"),
            ("VISUALIZE_ALL_RAYS", "Visualise all rays (including misses)"),
            ("ENABLE_VISUALIZATION", "Enable visualisation (master switch)"),
        ]
        for key, label in checkboxes:
            row += 1
            v = tk.BooleanVar(value=self.cfg.get(key, False))
            ttk.Checkbutton(frm, text=label, variable=v).grid(row=row, column=0, columnspan=3, sticky="w", pady=2)
            setattr(self, f"var_{key}", v)

        row += 1
        ttk.Label(frm, text="Summary CSV filename:").grid(row=row, column=0, sticky="w", pady=4)
        self.var_summary = tk.StringVar(value=self.cfg.get("SUMMARY_CSV_FILENAME", "power_summary_by_object.csv"))
        ttk.Entry(frm, textvariable=self.var_summary, width=40).grid(row=row, column=1, sticky="we")

        row += 1
        ttk.Label(frm, text="Rays to show in plot:").grid(row=row, column=0, sticky="w", pady=4)
        self.var_nrays = tk.IntVar(value=self.cfg.get("NUM_RAYS_TO_SHOW_IN_PLOT", 0))
        ttk.Entry(frm, textvariable=self.var_nrays, width=10).grid(row=row, column=1, sticky="w")

        # View results button
        row += 1
        ttk.Separator(frm, orient="horizontal").grid(row=row, column=0, columnspan=3, sticky="we", pady=8)
        row += 1
        ttk.Button(frm, text="🔍 View Results in ParaView…", command=self._view_results).pack() if False else \
        ttk.Button(frm, text="🔍 View Results in ParaView…", command=self._view_results).grid(row=row, column=0, columnspan=3, sticky="w", pady=4)

        frm.columnconfigure(1, weight=1)

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
        frm = ttk.Frame(nb, padding=10)
        nb.add(frm, text=" Smoothing ")

        row = 0
        self.var_smoother = tk.BooleanVar(value=self.cfg.get("RUN_SMOOTHER_AFTER_SIM", False))
        ttk.Checkbutton(frm, text="Run batch smoother after simulation", variable=self.var_smoother).grid(row=row, column=0, columnspan=2, sticky="w", pady=4)

        row += 1
        ttk.Label(frm, text="Smoothing radius (m):").grid(row=row, column=0, sticky="w", pady=4)
        self.var_sm_radius = tk.DoubleVar(value=self.cfg.get("SMOOTHING_RADIUS", 0.02))
        ttk.Entry(frm, textvariable=self.var_sm_radius, width=12).grid(row=row, column=1, sticky="w")

        row += 1
        ttk.Label(frm, text="Max cell area (m², empty=None):").grid(row=row, column=0, sticky="w", pady=4)
        mca = self.cfg.get("SMOOTHING_MAX_CELL_AREA")
        self.var_sm_mca = tk.StringVar(value=str(mca) if mca is not None else "")
        ttk.Entry(frm, textvariable=self.var_sm_mca, width=12).grid(row=row, column=1, sticky="w")

    # ------------------------------------------------------------------
    #  RUN tab (save, run, log)
    # ------------------------------------------------------------------
    def _build_run_tab(self, nb):
        frm = ttk.Frame(nb, padding=10)
        nb.add(frm, text=" Run ")

        btn_frm = ttk.Frame(frm)
        btn_frm.pack(fill="x")
        ttk.Button(btn_frm, text="💾 Save Config", command=self._save).pack(side="left", padx=4)
        ttk.Button(btn_frm, text="💾 Save As…", command=self._save_as).pack(side="left", padx=4)
        ttk.Button(btn_frm, text="📂 Load Config…", command=self._load_config_file).pack(side="left", padx=4)
        ttk.Separator(btn_frm, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Button(btn_frm, text="▶ Run Simulation", command=self._run_sim).pack(side="left", padx=4)
        ttk.Button(btn_frm, text="⏹ Stop", command=self._stop_sim).pack(side="left", padx=4)

        ttk.Separator(frm, orient="horizontal").pack(fill="x", pady=6)

        self.log_text = scrolledtext.ScrolledText(frm, height=28, state="disabled",
                                                   font=("Courier", 9), wrap="word")
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

        def _worker():
            try:
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
                self._log(f"\n{'✔ Simulation finished.' if rc == 0 else f'✖ Simulation exited with code {rc}.'}\n")
            except Exception as e:
                self._log(f"\n✖ Error: {e}\n")

        threading.Thread(target=_worker, daemon=True).start()

    def _stop_sim(self):
        if self._sim_process and self._sim_process.poll() is None:
            self._sim_process.terminate()
            self._log("\n⏹ Simulation terminated by user.\n")

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
        self.geometry("340x300")
        self.transient(parent)
        self.grab_set()
        self.result = None

        lb = tk.Listbox(self, height=12)
        lb.pack(fill="both", expand=True, padx=8, pady=8)
        for item in items:
            lb.insert("end", item)
        lb.selection_set(0)

        def _ok():
            sel = lb.curselection()
            if sel:
                self.result = lb.get(sel[0])
            self.destroy()

        ttk.Button(self, text="Open in ParaView", command=_ok).pack(pady=(0, 8))
        lb.bind("<Double-1>", lambda e: _ok())


# ===================================================================
#  Entry point
# ===================================================================
if __name__ == "__main__":
    app = SimGUI()
    app.mainloop()
