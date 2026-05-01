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
import datetime
import os
import sys
import subprocess
import glob
import threading
import shlex

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; we blit to Tk canvases
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from PIL import Image, ImageTk
import numpy as np

import viewer  # built-in Open3D viewer
from config import load_config, save_config

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
_IS_FROZEN = getattr(sys, 'frozen', False)  # True when running from PyInstaller exe
_SCRIPT_DIR = (os.path.dirname(sys.executable) if _IS_FROZEN
               else os.path.dirname(os.path.abspath(__file__)))
_CONFIG_JSON = os.path.join(_SCRIPT_DIR, "config.json")
_RUN_SIMULATION = os.path.join(_SCRIPT_DIR, "run_simulation.py")
_RUN_SMOOTHING = os.path.join(_SCRIPT_DIR, "smooth_results.py")
_PYTHON = sys.executable  # the same Python that launched the GUI
_SPLASH_LOGO = os.path.join(_SCRIPT_DIR, "BOT_logo.png")
_APP_ICON_BMP = os.path.join(_SCRIPT_DIR, "BOT_icon.bmp")


# ===================================================================
#  Helper: load / save JSON directly (no import config, to stay clean)
# ===================================================================
def _resolve_path(relative_path):
    """Resolve a simulation-file path relative to the main application folder."""
    return os.path.join(_SCRIPT_DIR, relative_path)


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


def _run_module_frozen(module_name, argv=None, log_fn=None):
    """Run a bundled module main(argv) directly when frozen."""
    argv = argv or []

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
        module = __import__(module_name)
        module.main(argv)
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

        # --- Engine card ---
        card = self._make_card(outer, "Engine Settings", pady=(12, 10))

        row = 0
        ttk.Label(card, text="CPU cores (-1 = all):", style="Card.TLabel").grid(
            row=row, column=0, sticky="w", pady=5)
        self.var_cpu = tk.IntVar(value=self.cfg.get("NUM_CPU_CORES", 1))
        ttk.Spinbox(card, from_=-1, to=256, textvariable=self.var_cpu,
                     width=8).grid(row=row, column=1, sticky="w", padx=(8, 0))

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
            ("save_details", "Save detailed reports (VTP files)", False),
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
    #  FIELDS tab (EM settings + external field)
    # ------------------------------------------------------------------
    def _build_fields_tab(self, nb):
        outer = ttk.Frame(nb)
        nb.add(outer, text="  ⚡  Fields  ")

        ef = self.cfg.get("EXTERNAL_FIELD", {})
        ef_type = ef.get("type", "zero")

        # --- Magnetic Field card ---
        b_card = self._make_card(outer, "Magnetic Field", pady=(12, 10))

        # Determine initial B-field mode from config
        if ef_type in ("uniform", "rid_segment_y", "rid_piecewise"):
            bvec = ef.get("magnetic_field_t", [0.0, 0.0, 0.0])
            b_init = "Fixed field (T)" if any(v != 0 for v in bvec) else "No field"
        else:
            bvec = [0.0, 0.0, 0.0]
            b_init = "No field"

        row = 0
        ttk.Label(b_card, text="Source:", style="Card.TLabel").grid(
            row=row, column=0, sticky="w", pady=4)
        self.var_bfield_mode = tk.StringVar(value=b_init)
        b_combo = ttk.Combobox(b_card, textvariable=self.var_bfield_mode,
                                values=["No field", "Fixed field (T)", "External B field"],
                                state="readonly", width=20)
        b_combo.grid(row=row, column=1, sticky="w", padx=(8, 0))

        row += 1
        self._b_fixed_frame = ttk.Frame(b_card, style="Card.TFrame")
        self._b_fixed_frame.grid(row=row, column=0, columnspan=2, sticky="w", pady=4)

        ttk.Label(self._b_fixed_frame, text="Bx:", style="Card.TLabel").pack(side="left")
        self.var_bx = tk.DoubleVar(value=bvec[0])
        ttk.Entry(self._b_fixed_frame, textvariable=self.var_bx, width=10).pack(side="left", padx=(4, 12))

        ttk.Label(self._b_fixed_frame, text="By:", style="Card.TLabel").pack(side="left")
        self.var_by = tk.DoubleVar(value=bvec[1])
        ttk.Entry(self._b_fixed_frame, textvariable=self.var_by, width=10).pack(side="left", padx=(4, 12))

        ttk.Label(self._b_fixed_frame, text="Bz:", style="Card.TLabel").pack(side="left")
        self.var_bz = tk.DoubleVar(value=bvec[2])
        ttk.Entry(self._b_fixed_frame, textvariable=self.var_bz, width=10).pack(side="left", padx=(4, 0))

        def _on_bfield_mode(*_):
            show = self.var_bfield_mode.get() == "Fixed field (T)"
            self._b_fixed_frame.grid() if show else self._b_fixed_frame.grid_remove()

        self.var_bfield_mode.trace_add("write", _on_bfield_mode)
        _on_bfield_mode()

        b_card.columnconfigure(1, weight=1)

        # --- Electric Field card ---
        e_card = self._make_card(outer, "Electric Field")

        # Determine initial E-field mode from config
        if ef_type in ("rid_segment_y", "rid_piecewise"):
            e_init = "ERID field (simplified)"
            evec = [0.0, 0.0, 0.0]
        elif ef_type == "uniform":
            evec = ef.get("electric_field_vpm", [0.0, 0.0, 0.0])
            e_init = "Fixed field (V/m)" if any(v != 0 for v in evec) else "No field"
        else:
            evec = [0.0, 0.0, 0.0]
            e_init = "No field"

        row = 0
        ttk.Label(e_card, text="Source:", style="Card.TLabel").grid(
            row=row, column=0, sticky="w", pady=4)
        self.var_efield_mode = tk.StringVar(value=e_init)
        e_combo = ttk.Combobox(e_card, textvariable=self.var_efield_mode,
                                values=["No field", "Fixed field (V/m)",
                                        "External E field", "ERID field (simplified)"],
                                state="readonly", width=20)
        e_combo.grid(row=row, column=1, sticky="w", padx=(8, 0))

        # -- Fixed E-field entries --
        row += 1
        self._e_fixed_frame = ttk.Frame(e_card, style="Card.TFrame")
        self._e_fixed_frame.grid(row=row, column=0, columnspan=2, sticky="w", pady=4)

        ttk.Label(self._e_fixed_frame, text="Ex:", style="Card.TLabel").pack(side="left")
        self.var_ex = tk.DoubleVar(value=evec[0])
        ttk.Entry(self._e_fixed_frame, textvariable=self.var_ex, width=10).pack(side="left", padx=(4, 12))

        ttk.Label(self._e_fixed_frame, text="Ey:", style="Card.TLabel").pack(side="left")
        self.var_ey = tk.DoubleVar(value=evec[1])
        ttk.Entry(self._e_fixed_frame, textvariable=self.var_ey, width=10).pack(side="left", padx=(4, 12))

        ttk.Label(self._e_fixed_frame, text="Ez:", style="Card.TLabel").pack(side="left")
        self.var_ez = tk.DoubleVar(value=evec[2])
        ttk.Entry(self._e_fixed_frame, textvariable=self.var_ez, width=10).pack(side="left", padx=(4, 0))

        # -- RID field entries --
        row += 1
        self._rid_frame = ttk.Frame(e_card, style="Card.TFrame")
        self._rid_frame.grid(row=row, column=0, columnspan=2, sticky="w", pady=4)

        ttk.Label(self._rid_frame, text="ERID panel voltage (V):", style="Card.TLabel").pack(side="left")
        self.var_v_rid = tk.DoubleVar(value=ef.get("v_rid_v", self.cfg.get("V_RID_V", 20000.0)))
        ttk.Entry(self._rid_frame, textvariable=self.var_v_rid, width=12).pack(side="left", padx=(4, 16))

        ttk.Label(self._rid_frame, text="x_min (m):", style="Card.TLabel").pack(side="left")
        self.var_rid_xmin = tk.DoubleVar(value=ef.get("x_min_m", 5.4))
        ttk.Entry(self._rid_frame, textvariable=self.var_rid_xmin, width=10).pack(side="left", padx=(4, 16))

        ttk.Label(self._rid_frame, text="x_max (m):", style="Card.TLabel").pack(side="left")
        self.var_rid_xmax = tk.DoubleVar(value=ef.get("x_max_m", 7.2))
        ttk.Entry(self._rid_frame, textvariable=self.var_rid_xmax, width=10).pack(side="left", padx=(4, 0))

        def _on_efield_mode(*_):
            mode = self.var_efield_mode.get()
            self._e_fixed_frame.grid() if mode == "Fixed field (V/m)" else self._e_fixed_frame.grid_remove()
            self._rid_frame.grid() if mode == "ERID field (simplified)" else self._rid_frame.grid_remove()

        self.var_efield_mode.trace_add("write", _on_efield_mode)
        _on_efield_mode()

        e_card.columnconfigure(1, weight=1)

    # ------------------------------------------------------------------
    #  REACTIONS tab
    # ------------------------------------------------------------------
    def _build_reactions_tab(self, nb):
        outer_wrapper = ttk.Frame(nb)
        nb.add(outer_wrapper, text="  🧪  Reactions  ")

        top_pw = ttk.PanedWindow(outer_wrapper, orient="horizontal")
        top_pw.pack(fill="both", expand=True)

        # ============================================================
        # LEFT side — scrollable cards
        # ============================================================
        left_wrapper = ttk.Frame(top_pw)
        top_pw.add(left_wrapper, weight=1)

        rxn_canvas = tk.Canvas(left_wrapper, borderwidth=0,
                                highlightthickness=0,
                                bg=self._colours["bg"])
        rxn_vscroll = ttk.Scrollbar(left_wrapper, orient="vertical",
                                     command=rxn_canvas.yview)
        outer = ttk.Frame(rxn_canvas)
        outer.bind("<Configure>",
                   lambda e: rxn_canvas.configure(
                       scrollregion=rxn_canvas.bbox("all")))
        rxn_canvas.create_window((0, 0), window=outer, anchor="nw")
        rxn_canvas.configure(yscrollcommand=rxn_vscroll.set)
        rxn_canvas.pack(side="left", fill="both", expand=True)
        rxn_vscroll.pack(side="right", fill="y")

        rm = self.cfg.get("REACTION_MODEL", {})

        # --- Background Gas Density card ---
        dens_card = self._make_card(outer, "Background Gas Density", pady=(12, 10))

        # Determine initial mode
        has_profile = bool(rm.get("density_profile_file", ""))
        dens_init = "Density profile file (.dens)" if has_profile else "Uniform density"

        row = 0
        ttk.Label(dens_card, text="Source:", style="Card.TLabel").grid(
            row=row, column=0, sticky="w", pady=4)
        self.var_density_mode = tk.StringVar(value=dens_init)
        ttk.Combobox(dens_card, textvariable=self.var_density_mode,
                      values=["Uniform density", "Density profile file (.dens)"],
                      state="readonly", width=28).grid(
            row=row, column=1, sticky="w", padx=(8, 0))

        # -- Uniform density widgets --
        row += 1
        self._dens_uniform_frame = ttk.Frame(dens_card, style="Card.TFrame")
        self._dens_uniform_frame.grid(row=row, column=0, columnspan=3, sticky="we", pady=2)
        ttk.Label(self._dens_uniform_frame, text="Density (m⁻³):", style="Card.TLabel").pack(side="left")
        self.var_bg_density = tk.DoubleVar(value=rm.get("background_density_m3", 0.0))
        ttk.Entry(self._dens_uniform_frame, textvariable=self.var_bg_density, width=14).pack(
            side="left", padx=(8, 0))

        # -- Profile file widgets --
        row += 1
        self._dens_profile_frame = ttk.Frame(dens_card, style="Card.TFrame")
        self._dens_profile_frame.grid(row=row, column=0, columnspan=3, sticky="we", pady=2)

        ttk.Label(self._dens_profile_frame, text="File:", style="Card.TLabel").pack(side="left")
        self.var_density_file = tk.StringVar(value=rm.get("density_profile_file", ""))
        ttk.Entry(self._dens_profile_frame, textvariable=self.var_density_file, width=30).pack(
            side="left", fill="x", expand=True, padx=(8, 4))
        ttk.Button(self._dens_profile_frame, text="Browse…", style="Secondary.TButton",
                    command=self._browse_density_file).pack(side="left")

        row += 1
        dens_dir_frame = ttk.Frame(dens_card, style="Card.TFrame")
        dens_dir_frame.grid(row=row, column=0, columnspan=3, sticky="we", pady=2)
        ttk.Label(dens_dir_frame, text="Profile direction:", style="Card.TLabel").pack(side="left")
        dd = rm.get("density_profile_direction",
                     self.cfg.get("DENSITY_DIRECTION", [1.0, 0.0, 0.0]))
        self.var_density_dir = tk.StringVar(value=f"{dd[0]}, {dd[1]}, {dd[2]}")
        ttk.Entry(dens_dir_frame, textvariable=self.var_density_dir, width=24).pack(
            side="left", padx=(8, 0))

        def _on_density_mode(*_):
            is_uniform = self.var_density_mode.get() == "Uniform density"
            if is_uniform:
                self._dens_uniform_frame.grid()
                self._dens_profile_frame.grid_remove()
            else:
                self._dens_uniform_frame.grid_remove()
                self._dens_profile_frame.grid()

        self.var_density_mode.trace_add("write", _on_density_mode)
        _on_density_mode()

        dens_card.columnconfigure(1, weight=1)

        # --- Reaction Definitions (Cross Sections) card ---
        cs_card = self._make_card(outer, "Reaction Definitions (Cross Sections)")

        # Channel definitions: (label, reaction_desc, channel_key)
        self._cs_channels = [
            ("H⁻/D⁻ → H⁰/D⁰", "Single stripping",  "single_strip_neg_to_neutral"),
            ("H⁻/D⁻ → H⁺/D⁺", "Double stripping",   "double_strip_neg_to_positive"),
            ("H⁰/D⁰ → H⁺/D⁺", "Neutral stripping",  "strip_neutral_to_positive"),
            ("H⁺/D⁺ → H⁰/D⁰", "Charge exchange",    "charge_exchange_pos_to_neutral"),
        ]

        manual_cs = rm.get("manual_cross_sections") or {}
        has_manual = bool(manual_cs)
        cs_mode_init = "Manual (m²)" if has_manual else "Built-in polynomial fit"

        row = 0
        ttk.Label(cs_card, text="Source:", style="Card.TLabel").grid(
            row=row, column=0, sticky="w", pady=4)
        self.var_cs_mode = tk.StringVar(value=cs_mode_init)
        ttk.Combobox(cs_card, textvariable=self.var_cs_mode,
                      values=["Built-in polynomial fit", "Manual (m²)"],
                      state="readonly", width=24).grid(
            row=row, column=1, columnspan=2, sticky="w", padx=(8, 0))

        # -- Freeze checkbox (only for built-in mode) --
        row += 1
        self.var_fixed_cs = tk.BooleanVar(value=rm.get("fixed_cs", False))
        self._cs_freeze_check = ttk.Checkbutton(
            cs_card, text="Freeze at initial energy",
            variable=self.var_fixed_cs, style="Card.TCheckbutton")
        self._cs_freeze_check.grid(row=row, column=0, columnspan=3, sticky="w", pady=(2, 6))

        # -- Channel rows: always visible, col 2 switches between label and entry --
        self._cs_manual_vars = {}
        self._cs_source_labels = {}
        self._cs_entry_widgets = {}
        self._cs_unit_labels = {}

        for i, (label, desc, key) in enumerate(self._cs_channels):
            r = row + 1 + i
            ttk.Label(cs_card, text=f"{label}", style="Card.TLabel").grid(
                row=r, column=0, sticky="w", pady=2)
            ttk.Label(cs_card, text=f"{desc}", style="Card.TLabel").grid(
                row=r, column=1, sticky="w", padx=(8, 0), pady=2)

            # Built-in label
            lbl = ttk.Label(cs_card, text="(built-in polynomial fit)", style="Card.TLabel")
            lbl.grid(row=r, column=2, sticky="w", padx=(8, 0), pady=2)
            self._cs_source_labels[key] = lbl

            # Manual entry + unit
            var = tk.StringVar(value=f"{manual_cs.get(key, 0.0):.3e}")
            self._cs_manual_vars[key] = var
            ent = ttk.Entry(cs_card, textvariable=var, width=12)
            ent.grid(row=r, column=2, sticky="w", padx=(8, 0), pady=2)
            self._cs_entry_widgets[key] = ent
            unit = ttk.Label(cs_card, text="m²", style="Card.TLabel")
            unit.grid(row=r, column=3, sticky="w", padx=(4, 0), pady=2)
            self._cs_unit_labels[key] = unit

        def _on_cs_mode(*_):
            is_manual = self.var_cs_mode.get() == "Manual (m²)"
            for key in self._cs_manual_vars:
                if is_manual:
                    self._cs_source_labels[key].grid_remove()
                    self._cs_entry_widgets[key].grid()
                    self._cs_unit_labels[key].grid()
                else:
                    self._cs_entry_widgets[key].grid_remove()
                    self._cs_unit_labels[key].grid_remove()
                    self._cs_source_labels[key].grid()
            if is_manual:
                self._cs_freeze_check.grid_remove()
            else:
                self._cs_freeze_check.grid()

        self.var_cs_mode.trace_add("write", _on_cs_mode)
        _on_cs_mode()

        cs_card.columnconfigure(1, weight=0)

        # ============================================================
        # RIGHT side — embedded diagnostic plots
        # ============================================================
        right_frm = ttk.Frame(top_pw, style="Card.TFrame", padding=8)
        top_pw.add(right_frm, weight=1)

        ttk.Label(right_frm, text="Diagnostic Plots",
                  style="CardHeader.TLabel").pack(anchor="w", pady=(0, 4))

        # --- plot parameter row ---
        param_row = ttk.Frame(right_frm, style="Card.TFrame")
        param_row.pack(fill="x", pady=(0, 4))
        ttk.Label(param_row, text="Species:",
                  style="Card.TLabel").pack(side="left", padx=(0, 4))
        self.var_plot_species = tk.StringVar(value="H")
        cb_species = ttk.Combobox(param_row, textvariable=self.var_plot_species,
                                  values=["H", "D"], state="readonly", width=4)
        cb_species.pack(side="left", padx=(0, 12))
        ttk.Label(param_row, text="Energy [eV]:",
                  style="Card.TLabel").pack(side="left", padx=(0, 4))
        self.var_plot_energy = tk.DoubleVar(value=870e3)
        ttk.Entry(param_row, textvariable=self.var_plot_energy, width=12).pack(side="left")

        # --- plot buttons ---
        btn_row = ttk.Frame(right_frm, style="Card.TFrame")
        btn_row.pack(fill="x", pady=(0, 6))
        ttk.Button(btn_row, text="Cross Sections", style="Secondary.TButton",
                    command=self._plot_cross_sections).pack(side="left", padx=(0, 6))
        ttk.Button(btn_row, text="Gas Density", style="Secondary.TButton",
                    command=self._plot_gas_density).pack(side="left", padx=(0, 6))
        ttk.Button(btn_row, text="Species Evolution", style="Secondary.TButton",
                    command=self._plot_species_evolution).pack(side="left")

        self._rxn_fig = Figure(figsize=(5.5, 4.5), dpi=100)
        self._rxn_canvas = FigureCanvasTkAgg(self._rxn_fig, master=right_frm)
        self._rxn_toolbar = NavigationToolbar2Tk(self._rxn_canvas, right_frm)
        self._rxn_toolbar.update()
        self._rxn_canvas.get_tk_widget().pack(fill="both", expand=True)
        self._rxn_cursor_artists = []
        self._rxn_canvas.mpl_connect("motion_notify_event", self._on_rxn_mouse_move)

    # ------------------------------------------------------------------
    def _on_rxn_mouse_move(self, event):
        """Show value labels on every plotted line at the cursor x-coordinate."""
        for a in self._rxn_cursor_artists:
            a.remove()
        self._rxn_cursor_artists.clear()

        if not self._rxn_fig.axes:
            return
        ax = self._rxn_fig.axes[0]
        if event.inaxes != ax or event.xdata is None:
            self._rxn_canvas.draw_idle()
            return

        x = event.xdata
        is_logx = ax.get_xscale() == "log"
        is_logy = ax.get_yscale() == "log"

        vl = ax.axvline(x, color="#888888", ls=":", lw=0.8, alpha=0.5)
        self._rxn_cursor_artists.append(vl)

        # Collect (y_value, label, color) for each data line
        items = []
        for line in ax.get_lines():
            lbl = line.get_label()
            if not lbl or lbl.startswith("_"):
                continue
            xd = np.asarray(line.get_xdata(), dtype=float)
            yd = np.asarray(line.get_ydata(), dtype=float)
            if len(xd) <= 2:          # skip axvline / cursor markers
                continue
            if x < xd.min() or x > xd.max():
                continue
            if is_logx and is_logy:
                yi = 10 ** np.interp(np.log10(x),
                                     np.log10(xd),
                                     np.log10(np.maximum(yd, 1e-300)))
            elif is_logx:
                yi = np.interp(np.log10(x), np.log10(xd), yd)
            else:
                yi = np.interp(x, xd, yd)
            items.append((yi, lbl, line.get_color()))

        # Draw dots and labels close to points
        for idx, (yi, lbl, col) in enumerate(items):
            dot, = ax.plot(x, yi, "o", color=col, ms=5, zorder=10)
            self._rxn_cursor_artists.append(dot)
            y_off = 6 * (1 if idx % 2 == 0 else -1)
            txt = ax.annotate(
                f"{lbl}: {yi:.3e}", xy=(x, yi),
                xytext=(8, y_off), textcoords="offset points",
                fontsize=7.5, color=col, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=col,
                          alpha=0.85, lw=0.6),
                zorder=11,
            )
            self._rxn_cursor_artists.append(txt)

        self._rxn_canvas.draw_idle()

    def _plot_cross_sections(self):
        """Show cross sections vs particle energy (log-log) in the embedded canvas."""
        import cross_sections as cs
        iso = self.var_plot_species.get()
        energy = np.logspace(1, 7, 500)  # 10 eV to 10 MeV
        sigma_ss = np.maximum(cs.cs_hm_single_strip(energy, isotope=iso), 1e-25)
        sigma_ds = np.maximum(cs.cs_hm_double_strip(energy, isotope=iso), 1e-25)
        sigma_ns = np.maximum(cs.cs_proj_ionization_h0(energy, isotope=iso), 1e-25)
        sigma_cx = np.maximum(cs.cs_cx_hp(energy, isotope=iso), 1e-25)

        neg = "H⁻" if iso == "H" else "D⁻"
        neu = "H⁰" if iso == "H" else "D⁰"
        pos = "H⁺" if iso == "H" else "D⁺"

        self._rxn_fig.clear()
        ax = self._rxn_fig.add_subplot(111)
        ax.loglog(energy, sigma_ss, label=f"{neg}→{neu} (single strip)", linewidth=1.8)
        ax.loglog(energy, sigma_ds, label=f"{neg}→{pos} (double strip)", linewidth=1.8)
        ax.loglog(energy, sigma_ns, label=f"{neu}→{pos} (neutral strip)", linewidth=1.8)
        ax.loglog(energy, sigma_cx, label=f"{pos}→{neu} (charge exchange)", linewidth=1.8)
        ax.set_xlabel("Particle energy [eV]")
        ax.set_ylabel("Cross section [m²]")
        ax.set_ylim(bottom=1e-25)
        ax.set_title(f"Beam–Gas Cross Sections ({iso})")
        ax.axvline(self.var_plot_energy.get(), color="gray", linestyle="--",
                    linewidth=1.0, alpha=0.7, label=f"E = {self.var_plot_energy.get():.0f} eV")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=9)
        self._rxn_fig.tight_layout()
        self._rxn_canvas.draw()

    def _plot_gas_density(self):
        """Show the gas density profile along the beam direction through the bounding box."""
        from reactions import BeamCrossSectionReaction

        density_dir = self._parse_vec3(self.var_density_dir.get())
        bbox_min = self._parse_vec3(self.var_bbox_min.get())
        bbox_max = self._parse_vec3(self.var_bbox_max.get())

        dir_arr = np.array(density_dir, dtype=np.float64)
        dir_norm = np.linalg.norm(dir_arr)
        if dir_norm <= 0:
            messagebox.showerror("Gas Density", "Invalid density direction.")
            return
        dir_arr /= dir_norm

        from prerun_analysis import _ray_exit_distance_from_box
        start = np.array(bbox_min, dtype=np.float64)
        line_len = _ray_exit_distance_from_box(start, dir_arr, bbox_min, bbox_max)
        if line_len is None or line_len <= 0:
            messagebox.showerror("Gas Density",
                                 "Density direction does not traverse the bounding box.")
            return

        is_uniform = self.var_density_mode.get() == "Uniform density"
        density_file = None if is_uniform else self.var_density_file.get()
        bg_density = self.var_bg_density.get()

        model = BeamCrossSectionReaction(
            background_density_m3=bg_density,
            density_profile_file=density_file,
            density_profile_direction=density_dir,
            verbose=False,
        )

        steps = max(2, int(np.ceil(line_len / 0.01)) + 1)
        distance = np.linspace(0.0, line_len, steps)
        positions = start[np.newaxis, :] + distance[:, np.newaxis] * dir_arr[np.newaxis, :]
        dens = np.maximum(np.asarray(model._density_at_positions(positions),
                                      dtype=np.float64), 0.0)

        self._rxn_fig.clear()
        ax = self._rxn_fig.add_subplot(111)
        ax.plot(distance, dens, color="tab:green", linewidth=2.0)
        ax.set_xlabel("Distance along beam direction [m]")
        ax.set_ylabel("Gas density [m⁻³]")
        ax.set_title("Background Gas Density Profile")
        ax.grid(True, alpha=0.3)
        self._rxn_fig.tight_layout()
        self._rxn_canvas.draw()

    def _plot_species_evolution(self):
        """Show analytical species evolution using prerun_analysis logic."""
        import cross_sections as cs
        from reactions import BeamCrossSectionReaction

        # Read current config values
        bbox_min = self._parse_vec3(self.var_bbox_min.get())
        bbox_max = self._parse_vec3(self.var_bbox_max.get())
        step_len = self.var_em_step.get()
        density_dir = self._parse_vec3(self.var_density_dir.get())

        # Build a temporary reaction model from current GUI state
        density_file = self.var_density_file.get() \
            if self.var_density_mode.get() != "Uniform density" else None
        bg_density = self.var_bg_density.get()
        model = BeamCrossSectionReaction(
            background_density_m3=bg_density,
            density_profile_file=density_file,
            density_profile_direction=density_dir,
            verbose=False,
        )

        # Use beam line along density direction from bbox min to max
        dir_arr = np.array(density_dir, dtype=np.float64)
        dir_norm = np.linalg.norm(dir_arr)
        if dir_norm <= 0:
            messagebox.showerror("Species Evolution", "Invalid density direction.")
            return
        dir_arr /= dir_norm

        from prerun_analysis import _ray_exit_distance_from_box
        start = np.array(bbox_min, dtype=np.float64)
        line_len = _ray_exit_distance_from_box(start, dir_arr, bbox_min, bbox_max)
        if line_len is None or line_len <= 0:
            messagebox.showerror("Species Evolution",
                                 "Beam direction does not traverse the bounding box.")
            return

        steps = max(2, int(np.ceil(line_len / step_len)) + 1)
        distance = np.linspace(0.0, line_len, steps)
        positions = start[np.newaxis, :] + distance[:, np.newaxis] * dir_arr[np.newaxis, :]

        density_sampler = model._density_at_positions
        density_m3 = np.maximum(np.asarray(density_sampler(positions), dtype=np.float64), 0.0)

        iso = self.var_plot_species.get()
        avg_energy_ev = self.var_plot_energy.get()
        from constants import HYDROGEN_MASS_KG, DEUTERIUM_MASS_KG, ELEMENTARY_CHARGE_C
        avg_mass_kg = HYDROGEN_MASS_KG if iso == "H" else DEUTERIUM_MASS_KG
        avg_speed = np.sqrt(2.0 * avg_energy_ev * ELEMENTARY_CHARGE_C / avg_mass_kg)

        if self.var_cs_mode.get() == "Manual (m²)":
            s_ss = float(self._cs_manual_vars["single_strip_neg_to_neutral"].get())
            s_ds = float(self._cs_manual_vars["double_strip_neg_to_positive"].get())
            s_ns = float(self._cs_manual_vars["strip_neutral_to_positive"].get())
            s_cx = float(self._cs_manual_vars["charge_exchange_pos_to_neutral"].get())
        else:
            sigma = cs.channel_cross_sections(avg_energy_ev, isotope=iso)
            s_ss = float(np.asarray(sigma[cs.CH_SINGLE_STRIP]))
            s_ds = float(np.asarray(sigma[cs.CH_DOUBLE_STRIP]))
            s_ns = float(np.asarray(sigma[cs.CH_NEUTRAL_STRIP]))
            s_cx = float(np.asarray(sigma[cs.CH_CHARGE_EXCHANGE]))

        fractions = np.zeros((steps, 3), dtype=np.float64)
        fractions[0] = [1.0, 0.0, 0.0]  # start as H-

        for i in range(steps - 1):
            ds = distance[i + 1] - distance[i]
            dt = ds / avg_speed
            f_neg, f_neu, f_pos = fractions[i]
            n = density_m3[i]
            dn = -(s_ss + s_ds) * n * avg_speed * f_neg
            d0 = s_ss * n * avg_speed * f_neg - s_ns * n * avg_speed * f_neu + s_cx * n * avg_speed * f_pos
            dp = s_ds * n * avg_speed * f_neg + s_ns * n * avg_speed * f_neu - s_cx * n * avg_speed * f_pos
            nxt = fractions[i] + dt * np.array([dn, d0, dp])
            nxt = np.maximum(nxt, 0.0)
            s = nxt.sum()
            if s > 0:
                nxt /= s
            fractions[i + 1] = nxt

        self._rxn_fig.clear()
        ax = self._rxn_fig.add_subplot(111)
        ax.plot(distance, fractions[:, 0], label="H⁻/D⁻", linewidth=2.0)
        ax.plot(distance, fractions[:, 1], label="H⁰/D⁰", linewidth=2.0)
        ax.plot(distance, fractions[:, 2], label="H⁺/D⁺", linewidth=2.0)
        ax.set_xlabel("Distance along beam direction [m]")
        ax.set_ylabel("Species fraction")
        ax.set_title(f"Analytical Species Evolution (E={avg_energy_ev/1e3:.0f} keV)")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend()
        self._rxn_fig.tight_layout()
        self._rxn_canvas.draw()

    def _browse_density_file(self):
        p = filedialog.askopenfilename(
            initialdir=_SCRIPT_DIR,
            title="Select density profile file",
            filetypes=[("Density files", "*.dens"), ("All files", "*")])
        if p:
            try:
                rel = os.path.relpath(p, _SCRIPT_DIR)
                self.var_density_file.set(rel)
            except ValueError:
                self.var_density_file.set(p)

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
            species_power = {}   # suffix → column name
            species_density = {}  # suffix → column name
            if not rows:
                return None, None, None, {}, {}
            keys = list(rows[0].keys())
            for k in keys:
                kl = k.lower()
                if "name" in kl or "file" in kl:
                    name_key = k
                elif "total" in kl and "power" in kl:
                    # Check for species suffix (e.g. _H-, _H0, _H+)
                    for suffix in ("_H-", "_H0", "_H+", "_D-", "_D0", "_D+"):
                        if k.endswith(suffix):
                            species_power[suffix[1:]] = k
                            break
                    else:
                        if power_key is None:
                            power_key = k
                elif "peak" in kl or "density" in kl:
                    for suffix in ("_H-", "_H0", "_H+", "_D-", "_D0", "_D+"):
                        if k.endswith(suffix):
                            species_density[suffix[1:]] = k
                            break
                    else:
                        if density_key is None:
                            density_key = k
            if name_key is None and len(keys) >= 1:
                name_key = keys[0]
            if power_key is None and len(keys) >= 2:
                power_key = keys[1]
            if density_key is None and len(keys) >= 3:
                density_key = keys[2]
            return name_key, power_key, density_key, species_power, species_density

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

            raw_nk, raw_pk, raw_dk, raw_sp, raw_sd = _detect_keys(raw_rows)

            merged = {}
            display_order = []
            for row in raw_rows:
                obj_raw = row.get(raw_nk, "?") if raw_nk else "?"
                obj = _norm(obj_raw)
                tp = row.get(raw_pk, "N/A") if raw_pk else "N/A"
                pd_ = row.get(raw_dk, "N/A") if raw_dk else "N/A"
                entry = {"name": obj, "tp": tp, "pd": pd_, "source": "raw"}
                for sp_key, col in raw_sp.items():
                    entry[f"tp_{sp_key}"] = row.get(col, "N/A")
                for sp_key, col in raw_sd.items():
                    entry[f"pd_{sp_key}"] = row.get(col, "N/A")
                merged[obj] = entry
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
                        sm_nk, sm_pk, sm_dk, sm_sp, sm_sd = _detect_keys(sm_rows)
                        for row in sm_rows:
                            obj_raw = row.get(sm_nk, "?") if sm_nk else "?"
                            obj = _norm(obj_raw)
                            tp = row.get(sm_pk, "N/A") if sm_pk else "N/A"
                            pd_ = row.get(sm_dk, "N/A") if sm_dk else "N/A"
                            # Start from existing entry to preserve species data
                            prev = merged.get(obj, {"name": obj})
                            entry = dict(prev)
                            entry.update({"tp": tp, "pd": pd_, "source": "smoothed"})
                            for sp_key, col in sm_sp.items():
                                entry[f"tp_{sp_key}"] = row.get(col, "N/A")
                            for sp_key, col in sm_sd.items():
                                entry[f"pd_{sp_key}"] = row.get(col, "N/A")
                            merged[obj] = entry
                            if obj not in display_order:
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

        data = self._csv_plot_data  # { sim_name: [{ name, tp, pd, source, tp_H-, ... }] }
        if not data:
            ax_peak.set_title("Peak Heat Load [W/m²]", fontsize=10)
            ax_power.set_title("Total Power [W]", fontsize=10)
            self._csv_fig.tight_layout()
            self._csv_canvas_mpl.draw()
            return

        # Determine which keys to use based on species selector
        sp_sel = self.var_chart_species.get()
        _species_map = {
            "H⁻/D⁻ (negative)": "H-",
            "H⁰/D⁰ (neutrals)": "H0",
            "H⁺/D⁺ (positive)": "H+",
        }
        sp_suffix = _species_map.get(sp_sel)
        tp_key = f"tp_{sp_suffix}" if sp_suffix else "tp"
        pd_key = f"pd_{sp_suffix}" if sp_suffix else "pd"
        species_label = f" ({sp_suffix})" if sp_suffix else ""

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
                        peaks.append(float(e.get(pd_key, "N/A")) * m)
                    except (ValueError, TypeError):
                        peaks.append(0.0)
                    try:
                        powers.append(float(e.get(tp_key, "N/A")) * m)
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
            ax.set_title(f"{title}{species_label} [{unit}]", fontsize=10, fontweight="bold")
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

        # Species filter
        sp_frm = ttk.Frame(comp_frm, style="Card.TFrame")
        sp_frm.pack(fill="x", pady=(2, 2))
        ttk.Label(sp_frm, text="Species:", style="Card.TLabel").pack(side="left")
        self.var_chart_species = tk.StringVar(value="Total")
        ttk.Combobox(sp_frm, textvariable=self.var_chart_species,
                      values=["Total", "H⁻/D⁻ (negative)", "H⁰/D⁰ (neutrals)",
                              "H⁺/D⁺ (positive)"],
                      state="readonly", width=18).pack(side="left", padx=(4, 0))

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
        ttk.Button(btn_row, text="≈  Run Smoothing Only", style="Secondary.TButton",
                command=self._run_smoothing_only).pack(side="left", padx=4)
        ttk.Button(btn_row, text="⏹  Stop", style="Danger.TButton",
                    command=self._stop_sim).pack(side="left", padx=4)

        # SDCC checkbox (Linux HPC only — hidden on Windows)
        self.var_sdcc = tk.BooleanVar(value=False)
        if sys.platform != "win32":
            ttk.Checkbutton(btn_card, text="Run on SLURM server (srun --exclusive)",
                             variable=self.var_sdcc,
                             style="Card.TCheckbutton").pack(anchor="w", pady=(8, 0))

        queue_card = self._make_card(outer, "Configuration Queue", pady=(6, 6))
        self.run_cfg_listbox = tk.Listbox(
            queue_card, height=6, font=("Segoe UI", 10),
            bg="white", fg=self._colours["fg"],
            selectbackground=self._colours["accent"],
            selectforeground="white", highlightthickness=0, bd=1, relief="solid")
        self.run_cfg_listbox.pack(fill="both", expand=False, pady=(0, 6))

        queue_btns = ttk.Frame(queue_card, style="Card.TFrame")
        queue_btns.pack(fill="x")
        ttk.Button(queue_btns, text="Add...", style="Secondary.TButton",
                    command=self._add_run_config).pack(side="left")
        ttk.Button(queue_btns, text="Delete", style="Secondary.TButton",
                    command=self._remove_run_config).pack(side="left", padx=(4, 0))
        ttk.Button(queue_btns, text="Use Active Config", style="Secondary.TButton",
                    command=self._add_active_config_to_queue).pack(side="left", padx=(8, 0))

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
    def _add_run_config(self):
        path = filedialog.askopenfilename(
            initialdir=os.path.dirname(self._active_config_path),
            title="Add configuration file",
            filetypes=[("JSON files", "*.json"), ("All files", "*")])
        if not path:
            return
        path = os.path.abspath(path)
        if path not in self._queued_config_paths:
            self._queued_config_paths.append(path)
            self._refresh_run_config_listbox()

    def _add_active_config_to_queue(self):
        path = os.path.abspath(self._active_config_path)
        if path not in self._queued_config_paths:
            self._queued_config_paths.append(path)
            self._refresh_run_config_listbox()

    def _remove_run_config(self):
        sel = list(self.run_cfg_listbox.curselection())
        if not sel:
            return
        for idx in reversed(sel):
            del self._queued_config_paths[idx]
        self._refresh_run_config_listbox()

    def _refresh_run_config_listbox(self):
        self.run_cfg_listbox.delete(0, "end")
        for p in self._queued_config_paths:
            self.run_cfg_listbox.insert("end", p)

    def _get_execution_config_paths(self):
        if self._queued_config_paths:
            return list(self._queued_config_paths)
        return [os.path.abspath(self._active_config_path)]

    def _load_config_from_path(self, path):
        try:
            path = os.path.abspath(path)
            new_cfg = load_config(path)
            self.cfg = new_cfg
            self._set_active_config_path(path)
            self._refresh_all_from_cfg()
            self._log(f"✔ Configuration loaded from {os.path.basename(path)}\n")
            self._set_status(f"Loaded config: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    @staticmethod
    def _parse_vec3(text):
        """Parse a comma-separated string into a list of 3 floats."""
        parts = [s.strip() for s in text.split(",")]
        return [float(parts[i]) if i < len(parts) else 0.0 for i in range(3)]

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
        # External field — compose from separate B / E selections
        b_mode = self.var_bfield_mode.get()
        e_mode = self.var_efield_mode.get()
        bvec = [self.var_bx.get(), self.var_by.get(), self.var_bz.get()] \
            if b_mode == "Fixed field (T)" else [0.0, 0.0, 0.0]

        if e_mode == "ERID field (simplified)":
            ef = {"type": "rid_segment_y",
                  "v_rid_v": self.var_v_rid.get(),
                  "x_min_m": self.var_rid_xmin.get(),
                  "x_max_m": self.var_rid_xmax.get(),
                  "magnetic_field_t": bvec}
        elif e_mode == "Fixed field (V/m)" or b_mode == "Fixed field (T)":
            evec = [self.var_ex.get(), self.var_ey.get(), self.var_ez.get()] \
                if e_mode == "Fixed field (V/m)" else [0.0, 0.0, 0.0]
            ef = {"type": "uniform",
                  "electric_field_vpm": evec,
                  "magnetic_field_t": bvec}
        else:
            ef = {"type": "zero"}
        d["EXTERNAL_FIELD"] = ef
        d["V_RID_V"] = self.var_v_rid.get()
        # Reaction model
        if is_em and self.var_reactions_enabled.get():
            dd_vec = self._parse_vec3(self.var_density_dir.get())
            rm_dict = {
                "type": "beam_gas_cross_sections",
                "background_density_m3": self.var_bg_density.get(),
                "density_profile_file": self.var_density_file.get(),
                "density_profile_direction": dd_vec,
            }
            if self.var_cs_mode.get() == "Manual (m²)":
                rm_dict["manual_cross_sections"] = {
                    k: float(v.get()) for k, v in self._cs_manual_vars.items()
                }
            else:
                rm_dict["fixed_cs"] = self.var_fixed_cs.get()
            d["REACTION_MODEL"] = rm_dict
            d["DENSITY_DIRECTION"] = dd_vec
        else:
            d["REACTION_MODEL"] = {"type": "none"}
        d["NUM_CPU_CORES"] = self.var_cpu.get()
        d["GEOMETRY_CACHE_DIR"] = self.var_cache.get()
        d["PARAVIEW_PATH"] = self.var_pv_path.get()
        d["PARAVIEW_MODULE"] = self.var_pv_module.get()
        # Geometry table is already updated in self.cfg via dialog
        d["GEOMETRY_FOLDERS"] = self.cfg.get("GEOMETRY_FOLDERS", {})
        d["PARTICLE_SOURCE_DIR"] = self.var_src_dir.get()
        d["NUM_PARTICLES_PER_BEAMLET"] = self.var_npb.get()
        d["BEAMLET_RADIUS_M"] = self.var_radius.get()
        d["PARTICLE_BATCH_SIZE"] = self.var_batch.get()
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
            path = os.path.abspath(path)
            save_config(self.cfg, path)
            self._set_active_config_path(path)
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
        self.var_cpu.set(c.get("NUM_CPU_CORES", 1))
        self.var_cache.set(c.get("GEOMETRY_CACHE_DIR", "geometry_cache"))
        self.var_pv_path.set(c.get("PARAVIEW_PATH", "paraview"))
        self.var_pv_module.set(c.get("PARAVIEW_MODULE", "ParaView"))
        # Geometry table
        self._populate_geo_tree()
        # Particles
        self.var_src_dir.set(c.get("PARTICLE_SOURCE_DIR", "BEAM_CONFIGS"))
        self.var_npb.set(c.get("NUM_PARTICLES_PER_BEAMLET", 10001))
        self.var_radius.set(c.get("BEAMLET_RADIUS_M", 0.007))
        self.var_batch.set(c.get("PARTICLE_BATCH_SIZE", 2_500_000))
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
        # Fields — B and E field cards
        ef = c.get("EXTERNAL_FIELD", {})
        ef_type = ef.get("type", "zero")
        # Magnetic field
        bvec = ef.get("magnetic_field_t", [0.0, 0.0, 0.0])
        if ef_type in ("uniform", "rid_segment_y", "rid_piecewise") and any(v != 0 for v in bvec):
            self.var_bfield_mode.set("Fixed field (T)")
        else:
            self.var_bfield_mode.set("No field")
        self.var_bx.set(bvec[0]); self.var_by.set(bvec[1]); self.var_bz.set(bvec[2])
        # Electric field
        if ef_type in ("rid_segment_y", "rid_piecewise"):
            self.var_efield_mode.set("ERID field (simplified)")
        elif ef_type == "uniform":
            evec = ef.get("electric_field_vpm", [0.0, 0.0, 0.0])
            self.var_efield_mode.set("Fixed field (V/m)" if any(v != 0 for v in evec) else "No field")
            self.var_ex.set(evec[0]); self.var_ey.set(evec[1]); self.var_ez.set(evec[2])
        else:
            self.var_efield_mode.set("No field")
        self.var_v_rid.set(ef.get("v_rid_v", c.get("V_RID_V", 20000.0)))
        self.var_rid_xmin.set(ef.get("x_min_m", 5.4))
        self.var_rid_xmax.set(ef.get("x_max_m", 7.2))
        # Reactions
        rm = c.get("REACTION_MODEL", {})
        self.var_bg_density.set(rm.get("background_density_m3", 0.0))
        self.var_density_file.set(rm.get("density_profile_file", ""))
        dd = rm.get("density_profile_direction",
                     c.get("DENSITY_DIRECTION", [1.0, 0.0, 0.0]))
        self.var_density_dir.set(f"{dd[0]}, {dd[1]}, {dd[2]}")
        # Cross-section mode
        manual_cs = rm.get("manual_cross_sections") or {}
        if manual_cs:
            self.var_cs_mode.set("Manual (m²)")
            for key, var in self._cs_manual_vars.items():
                var.set(f"{manual_cs.get(key, 0.0):.3e}")
        else:
            self.var_cs_mode.set("Built-in polynomial fit")
            self.var_fixed_cs.set(rm.get("fixed_cs", False))

    # ------------------------------------------------------------------
    #  Run simulation in background thread
    # ------------------------------------------------------------------
    def _run_sim(self):
        self._run_jobs(mode="simulation")

    def _run_smoothing_only(self):
        self._run_jobs(mode="smoothing")

    def _run_jobs(self, mode="simulation"):
        if self._sim_process and self._sim_process.poll() is None:
            messagebox.showinfo("Running", "A simulation is already running.")
            return

        # Save first
        self._save()
        self._stop_requested = False
        cfg_paths = self._get_execution_config_paths()

        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")
        mode_label = "simulation" if mode == "simulation" else "smoothing"
        self._log(f"▶ Starting {mode_label} queue with {len(cfg_paths)} configuration file(s)…\n\n")
        self._set_status(f"{mode_label.capitalize()} queue running…")

        def _worker():
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = os.path.join(_SCRIPT_DIR, "logs")
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, f"{mode_label}_{timestamp}.log")

            with open(log_path, "w", encoding="utf-8") as _lf:
                def _tee(text):
                    self._log(text)
                    _lf.write(text)
                    _lf.flush()

                _tee(f"Log file: {log_path}\n\n")
                successes = 0
                failures = 0
                try:
                    for idx, cfg_path in enumerate(cfg_paths, 1):
                        if self._stop_requested:
                            break
                        _tee(f"\n=== [{idx}/{len(cfg_paths)}] {mode_label.capitalize()} for {cfg_path} ===\n")
                        rc = self._run_single_job(mode, cfg_path, _tee)
                        if rc == 0:
                            successes += 1
                            _tee("✔ Completed successfully.\n")
                        else:
                            failures += 1
                            _tee(f"✖ Failed with return code {rc}.\n")
                except Exception as e:
                    _tee(f"\n✖ Error: {e}\n")
                    self.after(0, lambda: self._set_status(f"{mode_label.capitalize()} error"))
                    return

                if self._stop_requested:
                    self.after(0, lambda: self._set_status(f"{mode_label.capitalize()} queue stopped"))
                    _tee("\n⏹ Queue stopped by user.\n")
                elif failures == 0:
                    self.after(0, lambda: self._set_status(f"{mode_label.capitalize()} queue completed successfully"))
                    _tee(f"\n✔ Queue complete: {successes} succeeded, {failures} failed.\n")
                else:
                    self.after(0, lambda: self._set_status(f"{mode_label.capitalize()} queue completed with failures"))
                    _tee(f"\n✖ Queue complete: {successes} succeeded, {failures} failed.\n")

        threading.Thread(target=_worker, daemon=True).start()

    def _run_single_job(self, mode, cfg_path, log_fn):
        cfg_path = os.path.abspath(cfg_path)
        if mode == "simulation":
            script = _RUN_SIMULATION
            argv = ["-i", cfg_path]
        else:
            script = _RUN_SMOOTHING
            argv = ["-i", cfg_path, "-r", str(self.var_sm_radius.get())]
            max_area_txt = self.var_sm_mca.get().strip()
            if max_area_txt:
                argv += ["-a", max_area_txt]

        if _IS_FROZEN:
            module_name = "run_simulation" if mode == "simulation" else "smooth_results"
            _run_module_frozen(module_name, argv=argv, log_fn=log_fn)
            self._sim_process = None
            return 0

        if mode == "simulation" and sys.platform != "win32" and self.var_sdcc.get():
            py_cmd = f"{shlex.quote(_PYTHON)} {shlex.quote(script)} " + " ".join(shlex.quote(a) for a in argv)
            shell_cmd = f'srun --exclusive --pty /bin/bash -c "{py_cmd}"'
            cmd = ["bash", "-l", "-c", shell_cmd]
        else:
            cmd = [_PYTHON, script] + argv

        self._sim_process = subprocess.Popen(
            cmd,
            cwd=_SCRIPT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1)
        for line in self._sim_process.stdout:
            log_fn(line)
            if self._stop_requested:
                break
        if self._stop_requested and self._sim_process.poll() is None:
            self._sim_process.terminate()
        self._sim_process.wait()
        rc = self._sim_process.returncode
        self._sim_process = None
        return rc

    def _stop_sim(self):
        self._stop_requested = True
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
