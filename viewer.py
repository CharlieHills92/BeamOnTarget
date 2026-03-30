#!/usr/bin/env python3
# viewer.py
"""
Built-in 3D viewer for BeamOnTarget.

Renders the 3D scene with Open3D (visible=False, GPU-accelerated) and
displays the result inside the tkinter selection dialog — no separate
window needed.  Mouse drag rotates the view; scroll-wheel zooms.

Entry points called from sim_gui.py:
  - ``view_geometry(parent, script_dir, geometry_folders)``
  - ``view_results(parent, script_dir, output_dir, geometry_folders)``
  - ``view_all(parent, script_dir, output_dir, geometry_folders)``
"""
import os
import glob
import threading
import math

import numpy as np
import open3d as o3d
import pyvista as pv

import tkinter as tk
from PIL import Image, ImageTk

# ---------------------------------------------------------------------------
#  Colour palette for geometry folders
# ---------------------------------------------------------------------------
_FOLDER_COLORS = [
    (0.8, 0.2, 0.2), (0.2, 0.6, 0.8), (0.2, 0.8, 0.3),
    (0.9, 0.7, 0.1), (0.6, 0.3, 0.7), (0.9, 0.4, 0.1),
    (0.4, 0.8, 0.8), (0.8, 0.4, 0.6),
]

# Jet-like colour-map for power density
_CMAP_STOPS = np.array([
    [0.0, 0.0, 0.5],   # dark blue
    [0.0, 0.0, 1.0],   # blue
    [0.0, 0.5, 1.0],
    [0.0, 1.0, 1.0],   # cyan
    [0.5, 1.0, 0.5],
    [1.0, 1.0, 0.0],   # yellow
    [1.0, 0.5, 0.0],
    [1.0, 0.0, 0.0],   # red
    [0.5, 0.0, 0.0],   # dark red
])


def _apply_jet_colormap(values):
    """Map *values* (0-1 normalised) to RGB via a jet-like colour map."""
    n_stops = len(_CMAP_STOPS)
    t = np.clip(values, 0.0, 1.0) * (n_stops - 1)
    idx = np.clip(t.astype(int), 0, n_stops - 2)
    frac = (t - idx)[:, None]
    return _CMAP_STOPS[idx] * (1 - frac) + _CMAP_STOPS[idx + 1] * frac


# ===================================================================
#  Mesh loading helpers
# ===================================================================

def _load_stl(stl_path, scale, color):
    """Read an STL, optionally scale, paint uniform colour."""
    mesh = o3d.io.read_triangle_mesh(stl_path)
    if mesh.is_empty():
        return None
    if scale != 1:
        mesh.scale(scale, center=(0, 0, 0))
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    return mesh


def _read_vtp_as_o3d(vtp_path):
    """Read a VTP via pyvista, convert to Open3D TriangleMesh + power array."""
    pv_mesh = pv.read(vtp_path)
    if pv_mesh.n_cells == 0:
        return None, None

    verts = np.asarray(pv_mesh.points, dtype=np.float64)
    if hasattr(pv_mesh, "faces"):
        faces_raw = np.asarray(pv_mesh.faces)
    else:
        faces_raw = np.asarray(pv_mesh.cells)
    try:
        faces = faces_raw.reshape(-1, 4)[:, 1:4].astype(np.int32)
    except ValueError:
        return None, None

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(verts)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
    o3d_mesh.compute_vertex_normals()

    power = None
    if "Power_Density_W_m2" in pv_mesh.cell_data:
        power = np.asarray(pv_mesh.cell_data["Power_Density_W_m2"], dtype=np.float64)
    elif "Deposited_Power_W" in pv_mesh.cell_data:
        power = np.asarray(pv_mesh.cell_data["Deposited_Power_W"], dtype=np.float64)
    return o3d_mesh, power


def _colour_mesh_by_power(mesh, power, global_vmax=None):
    """Apply a jet colour-map to *mesh* based on per-cell *power*.

    If *global_vmax* is given the normalisation uses that value so that
    several meshes share the same colour scale.
    """
    triangles = np.asarray(mesh.triangles)
    n_verts = len(mesh.vertices)

    if power is None or len(power) == 0:
        mesh.paint_uniform_color([0.7, 0.7, 0.7])
        return 0.0

    vmax = power.max()
    if vmax <= 0:
        mesh.paint_uniform_color([0.7, 0.7, 0.7])
        return 0.0

    norm_max = global_vmax if (global_vmax and global_vmax > 0) else vmax
    normed = np.clip(power / norm_max, 0, 1)
    cell_colors = _apply_jet_colormap(normed)

    vert_colors = np.zeros((n_verts, 3), dtype=np.float64)
    vert_counts = np.zeros(n_verts, dtype=np.float64)
    for vi in range(3):
        np.add.at(vert_colors, triangles[:, vi], cell_colors)
        np.add.at(vert_counts, triangles[:, vi], 1)
    mask = vert_counts > 0
    vert_colors[mask] /= vert_counts[mask, None]
    mesh.vertex_colors = o3d.utility.Vector3dVector(vert_colors)
    return vmax


# ===================================================================
#  Colorbar helper — draws into an existing Tk Canvas widget
# ===================================================================

def _draw_colorbar_on_canvas(canvas, vmin, vmax, scale_factor,
                              unit_label="W/m²"):
    """Draw a vertical colour bar with tick labels onto *canvas*.

    Clears any previous content first.  Can be called repeatedly
    (e.g. after each View click with a new vmax / scale).
    """
    canvas.delete("all")

    cw = int(canvas.cget("width"))
    ch = int(canvas.cget("height"))

    bar_width = 24
    margin_top = 30
    margin_bot = 10
    margin_left = 6
    bar_height = ch - margin_top - margin_bot

    if bar_height < 20 or vmax <= 0:
        canvas.create_text(cw // 2, ch // 2, text="No data",
                           fill="#888", font=("Segoe UI", 9))
        return

    display_vmin = vmin * scale_factor
    display_vmax = vmax * scale_factor

    # Title
    if scale_factor != 1.0:
        title = f"[× {scale_factor:.1e}] W/m²"
    else:
        title = "W/m²"
    canvas.create_text(cw // 2, 12, text=title, fill="#1e293b",
                       font=("Segoe UI", 8, "bold"), anchor="center")

    # Gradient
    n_stops = len(_CMAP_STOPS)
    for i in range(bar_height):
        t = 1.0 - i / max(bar_height - 1, 1)
        pos = t * (n_stops - 1)
        idx = min(int(pos), n_stops - 2)
        frac = pos - idx
        r, g, b = (_CMAP_STOPS[idx] * (1 - frac) + _CMAP_STOPS[idx + 1] * frac)
        hex_c = "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))
        y = margin_top + i
        canvas.create_line(margin_left, y, margin_left + bar_width, y,
                           fill=hex_c, width=1)

    # Border
    canvas.create_rectangle(margin_left, margin_top,
                            margin_left + bar_width,
                            margin_top + bar_height - 1,
                            outline="#333", width=1)

    # Ticks
    tick_count = 6
    x0 = margin_left + bar_width
    for j in range(tick_count + 1):
        frac_j = j / tick_count
        y = margin_top + int((1.0 - frac_j) * (bar_height - 1))
        val = display_vmin + frac_j * (display_vmax - display_vmin)
        canvas.create_line(x0, y, x0 + 4, y, fill="#333", width=1)
        if abs(val) >= 1e4 or (abs(val) > 0 and abs(val) < 0.01):
            lbl = f"{val:.1e}"
        else:
            lbl = f"{val:.2f}"
        canvas.create_text(x0 + 6, y, text=lbl, anchor="w",
                           font=("Segoe UI", 7), fill="#1e293b")


def _show_in_open3d(geometries, title="BeamOnTarget Viewer"):
    """Open an Open3D visualisation window (blocking).  Legacy fallback."""
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1280, height=800)
    for geom in geometries:
        vis.add_geometry(geom)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.15, 0.15, 0.18])
    opt.mesh_show_back_face = True
    opt.light_on = True
    vis.reset_view_point(True)
    vis.run()
    vis.destroy_window()


def _scan_vtp_vmax(vtp_paths):
    """Return the global max power-density across *vtp_paths* (lightweight)."""
    vmax = 0.0
    for vtp in vtp_paths:
        try:
            pv_mesh = pv.read(vtp)
        except Exception:
            continue
        for key in ("Power_Density_W_m2", "Deposited_Power_W"):
            if key in pv_mesh.cell_data:
                arr = np.asarray(pv_mesh.cell_data[key])
                if len(arr) > 0:
                    vmax = max(vmax, float(arr.max()))
                break
    return vmax


# ===================================================================
#  Embedded 3D viewport — renders with Open3D, displays in Tk Canvas
# ===================================================================

class _EmbeddedViewer:
    """Renders meshes off-screen with Open3D (visible=False) and shows
    the image in a Tk Canvas, with mouse-drag rotation and scroll-wheel
    zoom.

    The Visualizer is created once and kept alive — only the camera is
    updated on each interaction, giving GPU-accelerated re-renders.
    """

    _BG = np.array([0.15, 0.15, 0.18])
    _RENDER_W = 720
    _RENDER_H = 520

    def __init__(self, parent_frame):
        self._frame = parent_frame
        self._canvas = tk.Canvas(parent_frame, bg="#262630",
                                 highlightthickness=0,
                                 width=self._RENDER_W,
                                 height=self._RENDER_H)
        self._canvas.pack(fill="both", expand=True)
        self._photo = None          # keep reference to avoid GC
        self._vis = None            # persistent Open3D Visualizer
        self._resize_after_id = None  # debounce id for resize

        # Camera spherical coordinates (azimuth, elevation, distance)
        self._azimuth = 45.0
        self._elevation = 30.0
        self._distance = None       # auto-computed on first render
        self._focus = np.zeros(3)   # look-at point

        # Mouse state
        self._drag_start = None
        self._canvas.bind("<ButtonPress-1>", self._on_press)
        self._canvas.bind("<B1-Motion>", self._on_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_release)
        self._canvas.bind("<MouseWheel>", self._on_scroll)        # Windows/macOS
        self._canvas.bind("<Button-4>", self._on_scroll_up)       # Linux
        self._canvas.bind("<Button-5>", self._on_scroll_down)     # Linux
        self._canvas.bind("<ButtonPress-3>", self._on_press_right)
        self._canvas.bind("<B3-Motion>", self._on_drag_right)
        self._canvas.bind("<ButtonRelease-3>", self._on_release_right)
        self._drag_right_start = None

        # Re-render on canvas resize (debounced)
        self._canvas.bind("<Configure>", self._on_canvas_resize)

        # Placeholder text
        self._canvas.create_text(
            self._RENDER_W // 2, self._RENDER_H // 2,
            text="Click  ▶ View  to render",
            fill="#666", font=("Segoe UI", 12))

    # ------ public API ------

    def set_meshes(self, o3d_meshes):
        """Build a persistent Open3D Visualizer with the given meshes."""
        # Close any previous visualizer
        if self._vis is not None:
            try:
                self._vis.destroy_window()
            except Exception:
                pass

        # Use current canvas size for the render target
        self._canvas.update_idletasks()
        w = max(self._canvas.winfo_width(), 64)
        h = max(self._canvas.winfo_height(), 64)
        self._RENDER_W = w
        self._RENDER_H = h

        # Keep meshes so we can rebuild on resize
        self._o3d_meshes = list(o3d_meshes)

        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False,
                          width=self._RENDER_W, height=self._RENDER_H)

        for mesh in o3d_meshes:
            vis.add_geometry(mesh)

        opt = vis.get_render_option()
        opt.background_color = self._BG
        opt.mesh_show_back_face = True
        opt.light_on = True

        # Compute scene bounds for camera setup
        all_pts = []
        for mesh in o3d_meshes:
            pts = np.asarray(mesh.vertices)
            if len(pts) > 0:
                all_pts.append(pts)
        if not all_pts:
            return
        all_pts = np.vstack(all_pts)
        bmin = all_pts.min(axis=0)
        bmax = all_pts.max(axis=0)
        self._focus = (bmin + bmax) / 2.0
        diag = np.linalg.norm(bmax - bmin)
        self._distance = diag * 1.5

        self._vis = vis
        self._render()

    def _render(self):
        """Update camera, render, and display the frame in the Tk canvas."""
        if self._vis is None:
            return

        # Compute camera position from spherical coordinates
        az = math.radians(self._azimuth)
        el = math.radians(self._elevation)
        d = self._distance or 1.0
        eye = np.array([
            self._focus[0] + d * math.cos(el) * math.cos(az),
            self._focus[1] + d * math.cos(el) * math.sin(az),
            self._focus[2] + d * math.sin(el),
        ])

        # Set the camera via ViewControl
        ctr = self._vis.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()

        # Build extrinsic matrix (world-to-camera)
        forward = self._focus - eye
        forward /= np.linalg.norm(forward)
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, world_up)
        rn = np.linalg.norm(right)
        if rn < 1e-6:
            world_up = np.array([0.0, 1.0, 0.0])
            right = np.cross(forward, world_up)
            rn = np.linalg.norm(right)
        right /= rn
        up = np.cross(right, forward)

        # Open3D extrinsic: [R|t] where R columns are right, -up, forward
        # and t = -R @ eye
        R = np.array([right, -up, forward])  # 3x3
        t = -R @ eye
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = t

        param.extrinsic = extrinsic
        ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)

        # Render and capture
        self._vis.poll_events()
        self._vis.update_renderer()
        img_buf = self._vis.capture_screen_float_buffer(do_render=True)
        img_arr = (np.asarray(img_buf) * 255).astype(np.uint8)

        # Display in Tk canvas
        pil_img = Image.fromarray(img_arr)
        self._photo = ImageTk.PhotoImage(pil_img)
        self._canvas.delete("all")
        self._canvas.create_image(0, 0, anchor="nw", image=self._photo)

    # ------ mouse interaction ------

    def _on_press(self, event):
        self._drag_start = (event.x, event.y, self._azimuth, self._elevation)

    def _on_drag(self, event):
        if self._drag_start is None:
            return
        x0, y0, az0, el0 = self._drag_start
        dx = event.x - x0
        dy = event.y - y0
        self._azimuth = az0 - dx * 0.4
        self._elevation = max(-89, min(89, el0 + dy * 0.4))
        self._render()

    def _on_release(self, event):
        self._drag_start = None

    def _on_press_right(self, event):
        self._drag_right_start = (event.x, event.y,
                                  self._focus.copy())

    def _on_drag_right(self, event):
        if self._drag_right_start is None:
            return
        x0, y0, foc0 = self._drag_right_start
        dx = event.x - x0
        dy = event.y - y0
        scale = (self._distance or 1.0) * 0.002
        az = math.radians(self._azimuth)
        rx, ry = math.sin(az), -math.cos(az)
        self._focus = foc0.copy()
        self._focus[0] += dx * scale * rx
        self._focus[1] += dx * scale * ry
        self._focus[2] -= dy * scale
        self._render()

    def _on_release_right(self, event):
        self._drag_right_start = None

    def _on_scroll(self, event):
        factor = 0.9 if event.delta > 0 else 1.1
        if self._distance is not None:
            self._distance *= factor
            self._render()

    def _on_scroll_up(self, event):
        if self._distance is not None:
            self._distance *= 0.9
            self._render()

    def _on_scroll_down(self, event):
        if self._distance is not None:
            self._distance *= 1.1
            self._render()

    def _on_canvas_resize(self, event):
        """Debounced handler: rebuild the Open3D window at the new size."""
        new_w = max(event.width, 64)
        new_h = max(event.height, 64)
        if new_w == self._RENDER_W and new_h == self._RENDER_H:
            return
        # Debounce — only act 200 ms after the last resize event
        if self._resize_after_id is not None:
            self._canvas.after_cancel(self._resize_after_id)
        self._resize_after_id = self._canvas.after(
            200, lambda: self._do_resize(new_w, new_h))

    def _do_resize(self, new_w, new_h):
        """Recreate the Open3D visualizer at the new canvas size."""
        self._resize_after_id = None
        if self._vis is None:
            return  # nothing rendered yet
        self._RENDER_W = new_w
        self._RENDER_H = new_h
        # Rebuild with the stored meshes
        meshes = getattr(self, '_o3d_meshes', None)
        if meshes:
            self.set_meshes(meshes)


# ===================================================================
#  Unified tkinter selection dialog
# ===================================================================

def _pick_items_dialog(parent, geometry_folders, results_dir,
                       geo_checked=True, res_checked=True,
                       load_and_show_fn=None):
    """Persistent selection dialog with geometry folders AND result files.

    The dialog stays open so the user can change the selection and click
    **View** repeatedly.  Each click launches a new Open3D window via
    *load_and_show_fn(sel_geo, sel_res)* in a daemon thread.
    The dialog closes only when the user clicks **Close** or the ✕.

    If *load_and_show_fn* is ``None`` the dialog falls back to the old
    one-shot behaviour (returns selection and closes).

    Returns ``(None, None)`` — the callback-based flow makes the return
    value unused when *load_and_show_fn* is supplied.
    """
    from tkinter import ttk

    dlg = tk.Toplevel(parent)
    dlg.title("Select items to display")
    dlg.geometry("1100x650")
    dlg.minsize(700, 400)
    dlg.result_geo = None
    dlg.result_res = None

    # Use update + deiconify to avoid flicker
    dlg.withdraw()
    dlg.update_idletasks()
    dlg.deiconify()

    # ====== BOTTOM BAR (scale + status + buttons) — packed FIRST so it
    #        never gets pushed off-screen by the expanding main pane. ======
    bottom_frm = ttk.Frame(dlg)
    bottom_frm.pack(side="bottom", fill="x", padx=8, pady=(0, 4))

    # --- Power density scale factor ---
    scale_frm = ttk.Frame(bottom_frm)
    scale_frm.pack(fill="x", pady=(4, 0))
    ttk.Label(scale_frm, text="Power density scale factor:",
              font=("", 10)).pack(side="left")
    scale_var = tk.StringVar(value="1.0")
    scale_entry = ttk.Entry(scale_frm, textvariable=scale_var, width=14)
    scale_entry.pack(side="left", padx=(6, 0))
    ttk.Label(scale_frm, text="(e.g. 1e-6 for MW/m²)",
              foreground="grey", font=("", 9)).pack(side="left", padx=(6, 0))

    # --- status label ---
    status_var = tk.StringVar(value="")
    status_lbl = ttk.Label(bottom_frm, textvariable=status_var,
                           foreground="grey")
    status_lbl.pack(fill="x", pady=(2, 0))

    # --- buttons ---
    btn_frm = ttk.Frame(bottom_frm)
    btn_frm.pack(fill="x", pady=(4, 0))

    # ====== MAIN AREA — horizontal PanedWindow ======
    main_pane = tk.PanedWindow(dlg, orient="horizontal", sashwidth=6,
                                sashrelief="raised", bg="#cccccc")
    main_pane.pack(fill="both", expand=True, padx=4, pady=4)

    # -- LEFT: scrollable checkbox list (resizable via sash) --
    left_frm = ttk.Frame(main_pane)

    # --- scrollable frame with BOTH vertical and horizontal scrollbars ---
    canvas = tk.Canvas(left_frm, borderwidth=0, highlightthickness=0)
    v_scroll = ttk.Scrollbar(left_frm, orient="vertical",
                              command=canvas.yview)
    h_scroll = ttk.Scrollbar(left_frm, orient="horizontal",
                              command=canvas.xview)
    inner = ttk.Frame(canvas)
    inner.bind("<Configure>",
               lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=inner, anchor="nw")
    canvas.configure(yscrollcommand=v_scroll.set,
                     xscrollcommand=h_scroll.set)

    # Grid layout: canvas fills, scrollbars on edges
    canvas.grid(row=0, column=0, sticky="nsew")
    v_scroll.grid(row=0, column=1, sticky="ns")
    h_scroll.grid(row=1, column=0, sticky="ew")
    left_frm.rowconfigure(0, weight=1)
    left_frm.columnconfigure(0, weight=1)

    main_pane.add(left_frm, minsize=180, width=260)

    # -- CENTRE: embedded 3D viewport --
    view_frm = ttk.LabelFrame(main_pane, text="3D View")
    viewer = _EmbeddedViewer(view_frm)
    main_pane.add(view_frm, minsize=300)

    # -- RIGHT: colorbar --
    cbar_frm = ttk.LabelFrame(main_pane, text="Colour Bar")
    cbar_canvas = tk.Canvas(cbar_frm, width=120, height=500,
                            bg="#f0f2f5", highlightthickness=0)
    cbar_canvas.pack(fill="both", expand=True, padx=4, pady=4)
    cbar_canvas.create_text(60, 250, text="No data yet",
                            fill="#999", font=("Segoe UI", 9))
    main_pane.add(cbar_frm, minsize=130, width=140)

    dlg._cbar_canvas = cbar_canvas
    dlg._viewer = viewer

    all_geo_vars = {}   # folder_name → BooleanVar
    all_res_vars = {}   # vtp_abs_path → BooleanVar

    # --- Geometry folders ---
    if geometry_folders:
        ttk.Label(inner, text="Geometry",
                  font=("", 11, "bold")).pack(anchor="w", padx=4, pady=(8, 2))
        ci = 0
        for folder in geometry_folders:
            color = _FOLDER_COLORS[ci % len(_FOLDER_COLORS)]
            ci += 1
            var = tk.BooleanVar(value=geo_checked)
            frm = ttk.Frame(inner)
            frm.pack(fill="x", pady=1, padx=8)
            sw = tk.Canvas(frm, width=14, height=14, highlightthickness=0)
            hex_c = "#%02x%02x%02x" % tuple(int(c * 255) for c in color)
            sw.create_rectangle(0, 0, 14, 14, fill=hex_c, outline=hex_c)
            sw.pack(side="left", padx=(0, 6))
            ttk.Checkbutton(frm, text=folder, variable=var).pack(side="left")
            all_geo_vars[folder] = var

    # --- Result files ---
    results_abs = results_dir or ""
    result_sets = {}  # subdir_name → [vtp_abs_paths]
    if os.path.isdir(results_abs):
        for d in sorted(os.listdir(results_abs)):
            sub = os.path.join(results_abs, d)
            if not os.path.isdir(sub):
                continue
            vtps = sorted(glob.glob(os.path.join(sub, "**", "*.vtp"),
                                    recursive=True))
            if vtps:
                result_sets[d] = vtps

    if result_sets:
        ttk.Label(inner, text="Results",
                  font=("", 11, "bold")).pack(anchor="w", padx=4, pady=(12, 2))
        for set_name, vtps in result_sets.items():
            ttk.Label(inner, text=f"  {set_name}",
                      font=("", 10, "italic")).pack(anchor="w", padx=8, pady=(4, 0))
            for vtp in vtps:
                bn = os.path.splitext(os.path.basename(vtp))[0]
                var = tk.BooleanVar(value=res_checked)
                ttk.Checkbutton(inner, text=f"    {bn}",
                                variable=var).pack(anchor="w", padx=12, pady=0)
                all_res_vars[vtp] = var

    # --- buttons (in the bottom bar) ---
    def _all():
        for v in list(all_geo_vars.values()) + list(all_res_vars.values()):
            v.set(True)

    def _none():
        for v in list(all_geo_vars.values()) + list(all_res_vars.values()):
            v.set(False)

    ttk.Button(btn_frm, text="All", width=6, command=_all).pack(side="left", padx=2)
    ttk.Button(btn_frm, text="None", width=6, command=_none).pack(side="left", padx=2)

    def _get_selection():
        sel_g = [f for f, v in all_geo_vars.items() if v.get()]
        sel_r = [p for p, v in all_res_vars.items() if v.get()]
        try:
            sf = float(scale_var.get())
        except (ValueError, TypeError):
            sf = 1.0
        return sel_g, sel_r, sf

    if load_and_show_fn is not None:
        # ---- Persistent mode: View button renders into embedded viewer ----
        def _view():
            sel_g, sel_r, sf = _get_selection()
            if not sel_g and not sel_r:
                status_var.set("Nothing selected.")
                return
            n = len(sel_g) + len(sel_r)
            status_var.set(f"Loading {n} item(s)…")

            # Draw colour-bar immediately (fast VTP scan)
            if sel_r:
                try:
                    vmax_quick = _scan_vtp_vmax(sel_r)
                except Exception:
                    vmax_quick = 0.0
                if vmax_quick > 0 and hasattr(dlg, '_cbar_canvas'):
                    _draw_colorbar_on_canvas(dlg._cbar_canvas,
                                             0.0, vmax_quick, sf)
            dlg.update_idletasks()

            # Load meshes in worker thread, then render
            threading.Thread(
                target=lambda: _view_worker(sel_g, sel_r, sf),
                daemon=True,
            ).start()

        def _view_worker(sel_g, sel_r, sf):
            geoms, vmax, title = load_and_show_fn(sel_g, sel_r, sf)
            if not geoms:
                return
            # Schedule the render on the main Tk thread
            def _do_render():
                try:
                    dlg._viewer.set_meshes(geoms)
                    status_var.set(f"{title}  —  drag to rotate, scroll to zoom")
                except Exception as exc:
                    status_var.set(f"Render error: {exc}")
            try:
                dlg.after(0, _do_render)
            except tk.TclError:
                pass

        def _close():
            # Destroy the Open3D visualizer before closing
            if hasattr(dlg, '_viewer') and dlg._viewer._vis is not None:
                try:
                    dlg._viewer._vis.destroy_window()
                    dlg._viewer._vis = None
                except Exception:
                    pass
            dlg.destroy()

        view_btn = ttk.Button(btn_frm, text="▶ View", command=_view)
        view_btn.pack(side="right", padx=2)
        ttk.Button(btn_frm, text="Close", command=_close).pack(side="right", padx=2)

        # Don't block with wait_window — the dialog is non-modal while
        # the viewer is open.  The caller returns immediately.
    else:
        # ---- One-shot fallback (legacy) ----
        def _ok():
            dlg.result_geo, dlg.result_res, _ = _get_selection()
            dlg.destroy()

        def _cancel():
            dlg.destroy()

        ttk.Button(btn_frm, text="View", command=_ok).pack(side="right", padx=2)
        ttk.Button(btn_frm, text="Cancel", command=_cancel).pack(side="right", padx=2)

        parent.wait_window(dlg)

    return dlg.result_geo, dlg.result_res, dlg


# ===================================================================
#  Loaders
# ===================================================================

def _load_selected_geometry(script_dir, geometry_folders, selected_folders):
    """Load STL meshes for *selected_folders*."""
    meshes = []
    ci = 0
    for folder in geometry_folders:
        color = _FOLDER_COLORS[ci % len(_FOLDER_COLORS)]
        ci += 1
        if folder not in selected_folders:
            continue
        settings = geometry_folders[folder]
        scale = settings.get("scale", 1)
        folder_abs = os.path.join(script_dir, folder)
        if not os.path.isdir(folder_abs):
            continue
        for stl_path in sorted(
            glob.glob(os.path.join(folder_abs, "*.stl"))
            + glob.glob(os.path.join(folder_abs, "*.STL"))
        ):
            mesh = _load_stl(stl_path, scale, color)
            if mesh is not None:
                meshes.append(mesh)
    return meshes


def _load_selected_results(selected_vtp_paths, scale_factor=1.0):
    """Load VTP result files and return coloured meshes + max power.

    Power density values are multiplied by *scale_factor* before
    colouring so that the user can express them in convenient units
    (e.g. 1e-6 → MW/m²).  The returned *vmax_all* is the raw
    (un-scaled) maximum so the colour-bar can display scaled ticks.
    """
    meshes = []
    vmax_all = 0.0
    # First pass: find global vmax across all VTPs
    all_data = []
    for vtp in selected_vtp_paths:
        mesh, power = _read_vtp_as_o3d(vtp)
        if mesh is None:
            continue
        if power is not None and len(power) > 0:
            vmax_all = max(vmax_all, power.max())
        all_data.append((mesh, power))

    # Second pass: colour with consistent scale
    for mesh, power in all_data:
        _colour_mesh_by_power(mesh, power, global_vmax=vmax_all if vmax_all > 0 else None)
        meshes.append(mesh)

    return meshes, vmax_all


# ===================================================================
#  High-level entry points (called from sim_gui.py)
# ===================================================================

def _status_msg(parent, msg):
    if hasattr(parent, "_log"):
        parent._log(f"🔍 {msg}\n")


def view_geometry(parent, script_dir, geometry_folders):
    """Geometry viewer — persistent selection dialog → Open3D window."""

    def _load_and_show(sel_geo, sel_res, sf):
        geoms = _load_selected_geometry(script_dir, geometry_folders, sel_geo)
        vmax = 0.0
        if sel_res:
            res_meshes, vmax = _load_selected_results(sel_res, sf)
            geoms += res_meshes
        if not geoms:
            parent.after(0, lambda: _status_msg(parent, "No meshes found."))
            return [], 0.0, ""
        parent.after(0, lambda: _status_msg(parent,
                     f"Loading {len(geoms)} meshes…"))
        return geoms, vmax, f"Geometry ({len(geoms)} meshes)"

    _pick_items_dialog(
        parent, geometry_folders, results_dir=None,
        geo_checked=True, res_checked=False,
        load_and_show_fn=_load_and_show,
    )


def view_results(parent, script_dir, output_dir, geometry_folders=None):
    """Results viewer — persistent selection dialog → Open3D heatmap."""
    outdir_abs = (os.path.join(script_dir, output_dir)
                  if not os.path.isabs(output_dir) else output_dir)
    geometry_folders = geometry_folders or {}

    def _load_and_show(sel_geo, sel_res, sf):
        geoms = []
        if sel_geo:
            geoms += _load_selected_geometry(script_dir, geometry_folders,
                                             sel_geo)
        vmax = 0.0
        if sel_res:
            res_meshes, vmax = _load_selected_results(sel_res, sf)
            geoms += res_meshes
        if not geoms:
            parent.after(0, lambda: _status_msg(parent, "No meshes found."))
            return [], 0.0, ""
        title = f"Results ({len(geoms)} meshes"
        if vmax > 0:
            title += f", max {vmax * sf:.2e}"
        title += ")"
        parent.after(0, lambda: _status_msg(parent,
                     f"Loading {len(geoms)} meshes…"))
        return geoms, vmax, title

    _pick_items_dialog(
        parent, geometry_folders, outdir_abs,
        geo_checked=False, res_checked=True,
        load_and_show_fn=_load_and_show,
    )


def view_all(parent, script_dir, output_dir, geometry_folders):
    """Show everything — persistent selection dialog → Open3D window."""
    outdir_abs = (os.path.join(script_dir, output_dir)
                  if not os.path.isabs(output_dir) else output_dir)

    def _load_and_show(sel_geo, sel_res, sf):
        geoms = []
        if sel_geo:
            geoms += _load_selected_geometry(script_dir, geometry_folders,
                                             sel_geo)
        vmax = 0.0
        if sel_res:
            res_meshes, vmax = _load_selected_results(sel_res, sf)
            geoms += res_meshes
        if not geoms:
            parent.after(0, lambda: _status_msg(parent, "No meshes found."))
            return [], 0.0, ""
        title = f"Viewer ({len(geoms)} meshes)"
        parent.after(0, lambda: _status_msg(parent,
                     f"Loading {len(geoms)} meshes…"))
        return geoms, vmax, title

    _pick_items_dialog(
        parent, geometry_folders, outdir_abs,
        geo_checked=True, res_checked=True,
        load_and_show_fn=_load_and_show,
    )


# ===================================================================
#  Standalone test
# ===================================================================
if __name__ == "__main__":
    import json

    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(script_dir, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)

    gf = cfg.get("GEOMETRY_FOLDERS", {})
    folders = list(gf.keys())
    print(f"Available folders: {folders}")

    items = _load_selected_geometry(script_dir, gf, folders)
    print(f"Loaded {len(items)} geometry meshes")
    if items:
        _show_in_open3d(items, "Geometry Test")
