#!/usr/bin/env python3
# sim_gui.py
"""
Tkinter GUI for managing the BeamOnTarget simulation.

Reads / writes config.json through the config module.
Launches run_simulation.py as a subprocess (preserving CLI compatibility).
Launches ParaView externally for geometry and results viewing.
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import sys
import subprocess
import glob
import threading

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; sub-tabs import their own
from PIL import Image, ImageTk

from gui import viewer  # built-in Open3D viewer
from config import load_config, save_config
from gui.gui_widgets import (
    make_card,
    parse_vec3,
    resolve_path as _resolve_path,
    set_project_folder as _set_project_folder,
    get_project_folder as _get_project_folder,
    to_relative_path as _to_relative_path,
)
from gui.gui_fields import FieldsTab
from gui.gui_reactions import ReactionsTab
from gui.gui_results import ResultsTab
from gui.gui_geometry import GeometryTab
from gui.gui_particles import ParticlesTab
from gui.gui_output import OutputTab
from gui.gui_run import RunTab

# ---------------------------------------------------------------------------
# Resolve paths  (canonical copies live in gui_widgets; keep local for
# the handful of module-level constants that still need _SCRIPT_DIR).
# ---------------------------------------------------------------------------
_IS_FROZEN = getattr(sys, 'frozen', False)  # True when running from PyInstaller exe
_SCRIPT_DIR = (os.path.dirname(sys.executable) if _IS_FROZEN
               else os.path.dirname(os.path.abspath(__file__)))
_CONFIG_JSON = os.path.join(_SCRIPT_DIR, "config", "config.json")
_RUN_SIMULATION = os.path.join(_SCRIPT_DIR, "run_simulation.py")
_RUN_SMOOTHING = os.path.join(_SCRIPT_DIR, "postprocessing", "smooth_results.py")
_PYTHON = sys.executable  # the same Python that launched the GUI
_SPLASH_LOGO = os.path.join(_SCRIPT_DIR, "BOT_logo.png")
_APP_ICON_BMP = os.path.join(_SCRIPT_DIR, "BOT_icon.bmp")


from gui.paraview_utils import _pv_geometry_script, _pv_results_script, launch_paraview
from gui.gui_extract import _ExtractDialog, _PickDialog

class SimGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self._icon_photo = None
        self._splash_logo_photo = None
        self._splash = None
        self._active_config_path = _CONFIG_JSON
        self._queued_config_paths = []
        self._stop_requested = False

        self.withdraw()
        self._set_app_icon()
        self._show_startup_logo()

        self.title("BeamOnTarget — Simulation Manager")
        self.geometry("1000x760")
        self.minsize(860, 640)
        self.cfg = load_config(self._active_config_path)
        self._sync_project_folder(default_to_active=True)
        self._apply_theme()
        self._build_statusbar()
        self._build_ui()

        self.deiconify()
        self.lift()
        self.after(250, self._close_startup_logo)

    def _set_app_icon(self):
        """Set the runtime app icon for title bar/taskbar on Windows."""
        try:
            if os.path.exists(_APP_ICON_BMP):
                self._icon_photo = ImageTk.PhotoImage(Image.open(_APP_ICON_BMP))
                self.iconphoto(True, self._icon_photo)
        except Exception:
            pass

    def _show_startup_logo(self):
        """Show a centered splash logo while the GUI is initializing."""
        if not os.path.exists(_SPLASH_LOGO):
            return
        try:
            img = Image.open(_SPLASH_LOGO)
            self._splash_logo_photo = ImageTk.PhotoImage(img)
            splash = tk.Toplevel(self)
            splash.overrideredirect(True)
            splash.attributes("-topmost", True)

            label = tk.Label(splash, image=self._splash_logo_photo, borderwidth=0, highlightthickness=0)
            label.pack()

            splash.update_idletasks()
            width = img.width
            height = img.height
            x = (self.winfo_screenwidth() - width) // 2
            y = (self.winfo_screenheight() - height) // 2
            splash.geometry(f"{width}x{height}+{x}+{y}")
            splash.update()
            self._splash = splash
        except Exception:
            self._splash = None

    def _close_startup_logo(self):
        if self._splash is not None and self._splash.winfo_exists():
            self._splash.destroy()
            self._splash = None

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
        """Delegate to the shared module-level ``make_card``."""
        return make_card(parent, title=title, padx=padx, pady=pady)

    # ------------------------------------------------------------------
    #  Status bar
    # ------------------------------------------------------------------
    def _build_statusbar(self):
        bottom = ttk.Frame(self, style="TFrame")
        bottom.pack(side="bottom", fill="x")

        cfg_row = ttk.Frame(bottom, style="TFrame")
        cfg_row.pack(fill="x", padx=8, pady=(4, 0))
        ttk.Label(cfg_row, text="Configuration file:", style="TLabel").pack(side="left")
        self.var_config_file = tk.StringVar(value=self._active_config_path)
        ttk.Entry(cfg_row, textvariable=self.var_config_file, state="readonly").pack(
            side="left", fill="x", expand=True, padx=(8, 4))
        ttk.Button(cfg_row, text="Browse...", style="Secondary.TButton",
                   command=self._browse_active_config).pack(side="left")

        self._statusbar = ttk.Label(bottom, text="  Ready", style="Status.TLabel")
        self._statusbar.pack(fill="x", pady=(4, 0))

    def _set_status(self, text):
        self._statusbar.config(text=f"  {text}")

    def _set_active_config_path(self, path):
        self._active_config_path = os.path.abspath(path)
        self.var_config_file.set(self._active_config_path)

    def _sync_project_folder(self, default_to_active=False):
        """Update global project-folder resolver from current cfg/active config."""
        project_folder = (self.cfg.get("PROJECT_FOLDER") if isinstance(self.cfg, dict) else None) or ""
        if hasattr(self, "var_project_folder"):
            widget_value = self.var_project_folder.get().strip()
            if widget_value:
                project_folder = widget_value
        project_folder = project_folder.strip() if isinstance(project_folder, str) else ""
        if not project_folder and default_to_active:
            project_folder = os.path.dirname(self._active_config_path)
            self.cfg["PROJECT_FOLDER"] = project_folder
        elif project_folder and not os.path.isabs(project_folder):
            # Relative PROJECT_FOLDER is interpreted from the active config location.
            project_folder = os.path.abspath(os.path.join(os.path.dirname(self._active_config_path), project_folder))
        _set_project_folder(project_folder or os.path.dirname(self._active_config_path))

    def _browse_active_config(self):
        path = filedialog.askopenfilename(
            initialdir=os.path.dirname(self._active_config_path),
            title="Select configuration file",
            filetypes=[("JSON files", "*.json"), ("All files", "*")])
        if path:
            self._load_config_from_path(path)

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
        self._build_fields_tab(notebook)
        self._build_reactions_tab(notebook)
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

        # --- Tracking Method card ---
        method_card = self._make_card(outer, "Tracking Method", pady=(12, 10))

        row = 0
        ttk.Label(method_card, text="Method:", style="Card.TLabel").grid(
            row=row, column=0, sticky="w", pady=5)
        raw_mode = self.cfg.get("TRACKING_MODE", "ray")
        initial_mode = "EM Tracing" if raw_mode == "em_track_then_bvh" else "Ray Tracing"
        self.var_tracking_mode = tk.StringVar(value=initial_mode)
        mode_frame = ttk.Frame(method_card, style="Card.TFrame")
        mode_combo = ttk.Combobox(mode_frame, textvariable=self.var_tracking_mode,
                                   values=["Ray Tracing", "EM Tracing"],
                                   state="readonly", width=20)
        mode_combo.pack(side="left")

        rm = self.cfg.get("REACTION_MODEL", {})
        reactions_on = rm.get("type", "none") not in ("none", "off", "null")
        self.var_reactions_enabled = tk.BooleanVar(value=reactions_on and raw_mode == "em_track_then_bvh")
        self._reactions_check = ttk.Checkbutton(
            mode_frame, text="Enable reactions",
            variable=self.var_reactions_enabled, style="Card.TCheckbutton")
        self._reactions_check.pack(side="left", padx=(16, 0))

        mode_frame.grid(row=row, column=1, sticky="w", padx=(8, 0))

        def _on_mode_change(*_):
            is_em = self.var_tracking_mode.get() == "EM Tracing"
            state = "normal" if is_em else "disabled"
            self._reactions_check.configure(state=state)
            if not is_em:
                self.var_reactions_enabled.set(False)
            for child in self._em_card.winfo_children():
                try:
                    child.configure(state=state)
                except tk.TclError:
                    pass

        self.var_tracking_mode.trace_add("write", _on_mode_change)

        row += 1
        ttk.Label(method_card, text="Bounding box min (m):", style="Card.TLabel").grid(
            row=row, column=0, sticky="w", pady=4)
        bbox_min = self.cfg.get("EM_BOUNDING_BOX_MIN_CORNER_M") or [0.0, -0.5, -1.3]
        self.var_bbox_min = tk.StringVar(value=f"{bbox_min[0]}, {bbox_min[1]}, {bbox_min[2]}")
        ttk.Entry(method_card, textvariable=self.var_bbox_min, width=24).grid(
            row=row, column=1, sticky="w", padx=(8, 0))

        row += 1
        ttk.Label(method_card, text="Bounding box max (m):", style="Card.TLabel").grid(
            row=row, column=0, sticky="w", pady=4)
        bbox_max = self.cfg.get("EM_BOUNDING_BOX_MAX_CORNER_M") or [13.0, 0.5, 0.8]
        self.var_bbox_max = tk.StringVar(value=f"{bbox_max[0]}, {bbox_max[1]}, {bbox_max[2]}")
        ttk.Entry(method_card, textvariable=self.var_bbox_max, width=24).grid(
            row=row, column=1, sticky="w", padx=(8, 0))

        method_card.columnconfigure(1, weight=1)

        # --- EM Tracking card (near mode selector) ---
        self._em_card = self._make_card(outer, "EM Tracking", pady=(12, 10))
        em_card = self._em_card

        r = 0
        ttk.Label(em_card, text="Step length (m):", style="Card.TLabel").grid(
            row=r, column=0, sticky="w", pady=4)
        self.var_em_step = tk.DoubleVar(value=self.cfg.get("EM_STEP_LENGTH_M", 0.02))
        ttk.Entry(em_card, textvariable=self.var_em_step, width=12).grid(
            row=r, column=1, sticky="w", padx=(8, 0))

        ttk.Label(em_card, text="Max steps:", style="Card.TLabel").grid(
            row=r, column=2, sticky="w", padx=(24, 0), pady=4)
        self.var_em_max_steps = tk.IntVar(value=self.cfg.get("EM_MAX_STEPS", 500))
        ttk.Entry(em_card, textvariable=self.var_em_max_steps, width=10).grid(
            row=r, column=3, sticky="w", padx=(8, 0))

        r += 1
        ttk.Label(em_card, text="Min energy (eV):", style="Card.TLabel").grid(
            row=r, column=0, sticky="w", pady=4)
        val = self.cfg.get("EM_MIN_ENERGY_EV")
        self.var_em_min_energy = tk.StringVar(value=str(val) if val is not None else "")
        ttk.Entry(em_card, textvariable=self.var_em_min_energy, width=12).grid(
            row=r, column=1, sticky="w", padx=(8, 0))

        ttk.Label(em_card, text="BVH checkpoint (m):", style="Card.TLabel").grid(
            row=r, column=2, sticky="w", padx=(24, 0), pady=4)
        self.var_em_checkpoint = tk.DoubleVar(value=self.cfg.get("EM_BVH_CHECKPOINT_DISTANCE_M", 1.0))
        ttk.Entry(em_card, textvariable=self.var_em_checkpoint, width=12).grid(
            row=r, column=3, sticky="w", padx=(8, 0))

        em_card.columnconfigure(1, weight=1)
        em_card.columnconfigure(3, weight=1)

        _on_mode_change()  # apply initial state

        # --- Project card ---
        card = self._make_card(outer, "Project Settings", pady=(12, 10))
        card.master.pack_configure(before=method_card.master)

        row = 0
        ttk.Label(card, text="CPU cores (-1 = all):", style="Card.TLabel").grid(
            row=row, column=0, sticky="w", pady=5)
        self.var_cpu = tk.IntVar(value=self.cfg.get("NUM_CPU_CORES", 1))
        ttk.Spinbox(card, from_=-1, to=256, textvariable=self.var_cpu,
                     width=8).grid(row=row, column=1, sticky="w", padx=(8, 0))

        row += 1
        ttk.Label(card, text="Project folder (if blank resolves to the .json file folder):", style="Card.TLabel").grid(
            row=row, column=0, sticky="w", pady=5)
        self.var_project_folder = tk.StringVar(value=self.cfg.get("PROJECT_FOLDER", os.path.dirname(self._active_config_path)))
        ttk.Entry(card, textvariable=self.var_project_folder, width=30).grid(
            row=row, column=1, sticky="we", padx=(8, 4))

        def _on_project_folder_change(*_):
            self.cfg["PROJECT_FOLDER"] = self.var_project_folder.get().strip()
            self._sync_project_folder(default_to_active=True)

        self.var_project_folder.trace_add("write", _on_project_folder_change)
        ttk.Button(card, text="Browse…", style="Secondary.TButton",
                    command=self._browse_project_folder).grid(row=row, column=2)

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

    def _browse_project_folder(self):
        initial = _get_project_folder()
        p = filedialog.askdirectory(
            title="Select project folder",
            initialdir=initial)
        if p:
            self.var_project_folder.set(_to_relative_path(p, os.path.dirname(self._active_config_path)))
            self.cfg["PROJECT_FOLDER"] = self.var_project_folder.get().strip()
            self._sync_project_folder(default_to_active=True)

    # ------------------------------------------------------------------
    #  GEOMETRY tab (delegated to gui_geometry.py)
    # ------------------------------------------------------------------
    def _build_geometry_tab(self, nb):
        self._geometry_tab = GeometryTab(
            nb, self.cfg, self._colours,
            view_geometry_fn=self._view_geometry,
            view_geometry_o3d_fn=self._view_geometry_o3d,
        )
        nb.add(self._geometry_tab, text="  📐  Geometry  ")

    # ------------------------------------------------------------------
    #  PARTICLES tab (delegated to gui_particles.py)
    # ------------------------------------------------------------------
    def _build_particles_tab(self, nb):
        self._particles_tab = ParticlesTab(
            nb, self.cfg, self._colours,
            view_sources_o3d_fn=self._view_sources_o3d,
        )
        nb.add(self._particles_tab, text="  🔬  Particles  ")

    @property
    def var_src_dir(self):
        return self._particles_tab.var_src_dir

    # ------------------------------------------------------------------
    #  FIELDS tab (delegated to gui_fields.py)
    # ------------------------------------------------------------------
    def _build_fields_tab(self, nb):
        self._fields_tab = FieldsTab(nb, self.cfg,
                                     get_collect_fn=self._collect)
        nb.add(self._fields_tab, text="  🧲  Fields  ")

    # ------------------------------------------------------------------
    #  REACTIONS tab (delegated to gui_reactions.py)
    # ------------------------------------------------------------------
    def _build_reactions_tab(self, nb):
        self._reactions_tab = ReactionsTab(
            nb, self.cfg, self._colours,
            get_bbox_min=lambda: self._parse_vec3(self.var_bbox_min.get()),
            get_bbox_max=lambda: self._parse_vec3(self.var_bbox_max.get()),
            get_em_step=lambda: self.var_em_step.get(),
            get_collect_fn=self._collect,
        )
        nb.add(self._reactions_tab, text="  ⚛  Reactions  ")

    # ------------------------------------------------------------------
    #  OUTPUT tab (delegated to gui_output.py)
    # ------------------------------------------------------------------
    def _build_output_tab(self, nb):
        self._output_tab = OutputTab(
            nb, self.cfg, self._colours,
            open_extract_fn=self._open_extract_dialog,
        )
        nb.add(self._output_tab, text="  📁  Output  ")

    @property
    def var_outdir(self):
        return self._output_tab.var_outdir

    def _view_results(self):
        outdir = self.var_outdir.get()
        outdir_abs = _resolve_path(outdir) if not os.path.isabs(outdir) else outdir
        if not os.path.isdir(outdir_abs):
            messagebox.showwarning("No results", f"Output directory not found:\n{outdir_abs}")
            return
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
        outdir = self.var_outdir.get()
        outdir_abs = (_resolve_path(outdir)
                      if not os.path.isabs(outdir) else outdir)
        dlg = _ExtractDialog(self, outdir_abs)
        self.wait_window(dlg)

    # ------------------------------------------------------------------
    #  Viewer helpers (shared by menu bar + tab buttons)
    # ------------------------------------------------------------------
    def _view_geometry(self):
        script = _pv_geometry_script(self.cfg)
        pv = self.var_pv_path.get()
        pv_mod = self.var_pv_module.get()
        launch_paraview(script, pv, pv_mod)

    def _view_geometry_o3d(self):
        src_dir = self.var_src_dir.get()
        viewer.view_geometry(self, _get_project_folder(),
                             self.cfg.get("GEOMETRY_FOLDERS", {}),
                             source_dir=src_dir)

    def _view_results_o3d(self):
        outdir = self.var_outdir.get()
        src_dir = self.var_src_dir.get()
        viewer.view_results(self, _get_project_folder(), outdir,
                            geometry_folders=self.cfg.get("GEOMETRY_FOLDERS", {}),
                            source_dir=src_dir)

    def _view_all_o3d(self):
        outdir = self.var_outdir.get()
        src_dir = self.var_src_dir.get()
        viewer.view_all(self, _get_project_folder(), outdir,
                        self.cfg.get("GEOMETRY_FOLDERS", {}),
                        source_dir=src_dir)

    def _view_sources_o3d(self):
        src_dir = self.var_src_dir.get()
        viewer.view_sources(self, _get_project_folder(), src_dir,
                            geometry_folders=self.cfg.get("GEOMETRY_FOLDERS", {}))

    # ------------------------------------------------------------------
    #  RESULTS tab
    # ------------------------------------------------------------------
    def _build_results_tab(self, nb):
        self._results_tab = ResultsTab(
            nb, self.cfg, self._colours,
            var_outdir=self.var_outdir,
            var_pv_path=self.var_pv_path,
            var_pv_module=self.var_pv_module,
            view_results_fn=self._view_results,
            open_extract_fn=self._open_extract_dialog,
            view_results_o3d_fn=self._view_results_o3d,
        )
        nb.add(self._results_tab, text="  📊  Results  ")


    # ------------------------------------------------------------------
    #  RUN tab (delegated to gui_run.py)
    # ------------------------------------------------------------------
    def _build_run_tab(self, nb):
        self._run_tab = RunTab(
            nb, self.cfg, self._colours,
            save_fn=self._save,
            save_as_fn=self._save_as,
            load_config_fn=self._load_config_file,
            set_status_fn=self._set_status,
            get_active_config_path=lambda: self._active_config_path,
            set_active_config_path=self._set_active_config_path,
            get_smoothing_params=lambda: (
                self._output_tab.var_sm_radius.get(),
                self._output_tab.var_sm_mca.get().strip(),
            ),
        )
        nb.add(self._run_tab, text="  ▶  Run  ")

    # ------------------------------------------------------------------
    #  Shared helpers
    # ------------------------------------------------------------------
    _parse_vec3 = staticmethod(parse_vec3)

    def _load_config_from_path(self, path):
        try:
            path = os.path.abspath(path)
            new_cfg = load_config(path)
            if not str(new_cfg.get("PROJECT_FOLDER", "")).strip():
                new_cfg["PROJECT_FOLDER"] = os.path.dirname(path)
            self.cfg = new_cfg
            self._set_active_config_path(path)
            self._sync_project_folder(default_to_active=True)
            self._refresh_all_from_cfg()
            self._log(f"✔ Configuration loaded from {os.path.basename(path)}\n")
            self._set_status(f"Loaded config: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def _sync_cfg_to_tabs(self):
        """Ensure every tab references the same cfg dict."""
        for tab in (self._geometry_tab, self._particles_tab,
                    self._output_tab, self._run_tab):
            tab.cfg = self.cfg

    def _log(self, text):
        """Delegate logging to the Run tab."""
        self._run_tab._log(text)

    def _collect(self):
        d = dict(self.cfg)  # start from current (preserves unknown keys)
        # Tracking method
        is_em = self.var_tracking_mode.get() == "EM Tracing"
        d["TRACKING_MODE"] = "em_track_then_bvh" if is_em else "ray"
        # EM settings
        d["EM_STEP_LENGTH_M"] = self.var_em_step.get()
        d["EM_MAX_STEPS"] = self.var_em_max_steps.get()
        v = self.var_em_min_energy.get().strip()
        d["EM_MIN_ENERGY_EV"] = float(v) if v else None
        d["EM_BVH_CHECKPOINT_DISTANCE_M"] = self.var_em_checkpoint.get()
        d["EM_BOUNDING_BOX_MIN_CORNER_M"] = self._parse_vec3(self.var_bbox_min.get())
        d["EM_BOUNDING_BOX_MAX_CORNER_M"] = self._parse_vec3(self.var_bbox_max.get())
        # External field — delegate to FieldsTab
        self._fields_tab.collect(d)
        # Reaction model — delegate to ReactionsTab
        self._reactions_tab.collect(d, is_em and self.var_reactions_enabled.get())
        project_folder = self.var_project_folder.get().strip() or os.path.dirname(self._active_config_path)
        d["PROJECT_FOLDER"] = project_folder
        d["NUM_CPU_CORES"] = self.var_cpu.get()
        d["GEOMETRY_CACHE_DIR"] = self.var_cache.get()
        d["PARAVIEW_PATH"] = self.var_pv_path.get()
        d["PARAVIEW_MODULE"] = self.var_pv_module.get()
        # Geometry / Particles / Output — delegated
        self._geometry_tab.collect(d)
        self._particles_tab.collect(d)
        self._output_tab.collect(d)
        d["ENABLE_VISUALIZATION"] = self._results_tab.var_ENABLE_VISUALIZATION.get()
        return d

    # ------------------------------------------------------------------
    #  Save
    # ------------------------------------------------------------------
    def _save(self):
        try:
            self.cfg = self._collect()
            self._sync_project_folder(default_to_active=True)
            self._sync_cfg_to_tabs()
            save_config(self.cfg, self._active_config_path)
            self._log(f"✔ Configuration saved to {os.path.basename(self._active_config_path)}\n")
            self._set_status("Configuration saved")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    def _save_as(self):
        """Save the current config to a user-chosen JSON file."""
        path = filedialog.asksaveasfilename(
            initialdir=os.path.dirname(self._active_config_path),
            title="Save Config As",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*")])
        if not path:
            return
        try:
            self.cfg = self._collect()
            self._sync_cfg_to_tabs()
            path = os.path.abspath(path)
            if not str(self.cfg.get("PROJECT_FOLDER", "")).strip():
                self.cfg["PROJECT_FOLDER"] = os.path.dirname(path)
            save_config(self.cfg, path)
            self._set_active_config_path(path)
            self._sync_project_folder(default_to_active=True)
            self._log(f"✔ Configuration saved to {os.path.basename(path)}\n")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    def _load_config_file(self):
        """Load a config from a user-chosen JSON file and refresh the GUI."""
        path = filedialog.askopenfilename(
            initialdir=os.path.dirname(self._active_config_path),
            title="Load Config",
            filetypes=[("JSON files", "*.json"), ("All files", "*")])
        if not path:
            return
        self._load_config_from_path(path)

    def _refresh_all_from_cfg(self):
        """Push self.cfg values back into every GUI widget."""
        c = self.cfg
        # General — tracking method
        raw_mode = c.get("TRACKING_MODE", "ray")
        self.var_tracking_mode.set("EM Tracing" if raw_mode == "em_track_then_bvh" else "Ray Tracing")
        rm = c.get("REACTION_MODEL", {})
        reactions_on = rm.get("type", "none") not in ("none", "off", "null")
        self.var_reactions_enabled.set(reactions_on and raw_mode == "em_track_then_bvh")
        # General — engine
        self.var_project_folder.set(c.get("PROJECT_FOLDER", os.path.dirname(self._active_config_path)))
        self.var_cpu.set(c.get("NUM_CPU_CORES", 1))
        self.var_cache.set(c.get("GEOMETRY_CACHE_DIR", "geometry_cache"))
        self.var_pv_path.set(c.get("PARAVIEW_PATH", "paraview"))
        self.var_pv_module.set(c.get("PARAVIEW_MODULE", "ParaView"))
        self._sync_project_folder(default_to_active=True)
        # Geometry / Particles / Output — delegated
        self._geometry_tab.refresh(c)
        self._particles_tab.refresh(c)
        self._output_tab.refresh(c)
        self._results_tab.var_ENABLE_VISUALIZATION.set(c.get("ENABLE_VISUALIZATION", True))
        # Fields — EM settings
        self.var_em_step.set(c.get("EM_STEP_LENGTH_M", 0.02))
        self.var_em_max_steps.set(c.get("EM_MAX_STEPS", 500))
        v = c.get("EM_MIN_ENERGY_EV")
        self.var_em_min_energy.set(str(v) if v is not None else "")
        self.var_em_checkpoint.set(c.get("EM_BVH_CHECKPOINT_DISTANCE_M", 1.0))
        bbox_min = c.get("EM_BOUNDING_BOX_MIN_CORNER_M") or [0.0, -0.5, -1.3]
        self.var_bbox_min.set(f"{bbox_min[0]}, {bbox_min[1]}, {bbox_min[2]}")
        bbox_max = c.get("EM_BOUNDING_BOX_MAX_CORNER_M") or [13.0, 0.5, 0.8]
        self.var_bbox_max.set(f"{bbox_max[0]}, {bbox_max[1]}, {bbox_max[2]}")
        # Fields — per-component (delegated)
        self._fields_tab.refresh(c)
        # Reactions (delegated)
        self._reactions_tab.refresh(c)


def main():
    app = SimGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
