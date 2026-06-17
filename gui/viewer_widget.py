#!/usr/bin/env python3
# viewer_widget.py
"""
Colourmap utilities and the embedded Open3D viewport widget for BeamOnTarget.

Contents:
  - _CMAP_STOPS / _apply_jet_colormap  -- jet-like colour mapping
  - _draw_colorbar_on_canvas           -- Tk Canvas colour bar
  - _EmbeddedViewer                    -- GPU-accelerated Open3D viewport
"""
import math
import threading

import numpy as np
import open3d as o3d

import tkinter as tk
from PIL import Image, ImageTk, ImageDraw

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
        title = f"[× {scale_factor:.1e}] {unit_label}"
    else:
        title = unit_label
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
        self._render_pending = None   # throttle id for rendering
        self._rendering = False       # True while _render is executing

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
        # Cancel any pending throttled render / resize callbacks
        if self._render_pending is not None:
            self._canvas.after_cancel(self._render_pending)
            self._render_pending = None
        if self._resize_after_id is not None:
            self._canvas.after_cancel(self._resize_after_id)
            self._resize_after_id = None

        # Close any previous visualizer
        if self._vis is not None:
            try:
                self._vis.destroy_window()
            except Exception:
                pass
            self._vis = None

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
        opt.point_size = 5.0  # visible size for PointCloud sources

        # Compute scene bounds for camera setup
        all_pts = []
        for geom in o3d_meshes:
            if isinstance(geom, o3d.geometry.PointCloud):
                pts = np.asarray(geom.points)
            else:
                pts = np.asarray(geom.vertices)
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

    def _request_render(self):
        """Schedule a render if one is not already pending (throttled)."""
        if self._vis is None:
            return
        if self._render_pending is not None:
            return  # already scheduled
        if self._rendering:
            # A render is in progress — schedule one for after it finishes
            self._render_pending = self._canvas.after(30, self._do_throttled_render)
            return
        self._render_pending = self._canvas.after(16, self._do_throttled_render)

    def _do_throttled_render(self):
        """Execute the throttled render."""
        self._render_pending = None
        self._render()

    def _render(self):
        """Update camera, render, and display the frame in the Tk canvas."""
        if self._vis is None or self._rendering:
            return
        self._rendering = True
        try:
            self._render_impl()
        finally:
            self._rendering = False

    def _render_impl(self):
        """Internal: compute camera, render, blit to canvas."""
        vis = self._vis
        if vis is None:
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
        try:
            ctr = vis.get_view_control()
            param = ctr.convert_to_pinhole_camera_parameters()
        except Exception:
            return  # visualizer was destroyed between check and use

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
        try:
            ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)

            # Render and capture
            vis.poll_events()
            vis.update_renderer()
            img_buf = vis.capture_screen_float_buffer(do_render=True)
        except Exception:
            return  # visualizer was destroyed during rendering
        img_arr = (np.asarray(img_buf) * 255).astype(np.uint8)

        # Display in Tk canvas
        pil_img = Image.fromarray(img_arr)

        # Draw orientation axes overlay (bottom-left, ParaView-style)
        self._draw_axis_overlay(pil_img, R)

        self._photo = ImageTk.PhotoImage(pil_img)
        self._canvas.delete("all")
        self._canvas.create_image(0, 0, anchor="nw", image=self._photo)

    # ------ axis overlay ------

    def _draw_axis_overlay(self, pil_img, R):
        """Draw an orientation axis widget in the bottom-left corner.

        *R* is the 3×3 camera rotation matrix (world-to-camera) used for
        the current frame.  The axes are projected with the same
        rotation but drawn at a fixed screen position so they behave
        like ParaView's orientation widget.
        """
        draw = ImageDraw.Draw(pil_img)
        w, h = pil_img.size

        # Centre of the axis widget (bottom-left with some margin)
        cx, cy = 52, h - 52
        axis_len = 40  # pixels

        # World axes projected to screen via camera rotation R
        # R rows are: right, -up, forward  →  screen x = right, y = up
        axes = [
            (np.array([1.0, 0.0, 0.0]), (220, 60, 60),   "X"),
            (np.array([0.0, 1.0, 0.0]), (60, 180, 60),   "Y"),
            (np.array([0.0, 0.0, 1.0]), (70, 120, 220),  "Z"),
        ]

        # Sort by depth (forward component) so farther axes draw first
        projected = []
        for world_dir, color, label in axes:
            cam = R @ world_dir        # [right, down, forward]
            sx = cam[0] * axis_len     # screen x (right)
            sy = cam[1] * axis_len     # screen y (down — matches PIL coords)
            depth = cam[2]             # forward (into screen)
            projected.append((depth, sx, sy, color, label))
        projected.sort(key=lambda t: t[0])  # draw far-to-near

        a_size = 7   # arrowhead length in pixels
        a_half = 3   # arrowhead half-width

        for depth, sx, sy, color, label in projected:
            tip_len = math.hypot(sx, sy)
            if tip_len < 1e-3:
                continue

            # Unit direction from centre to tip
            dx = sx / tip_len
            dy = sy / tip_len
            # Perpendicular
            px = -dy
            py = dx

            # End-point of the shaft (stop short of arrowhead)
            shaft_ex = cx + sx - dx * a_size
            shaft_ey = cy + sy - dy * a_size
            tip_x = cx + sx
            tip_y = cy + sy

            # Draw shaft line (from centre to base of arrowhead)
            draw.line([(cx, cy), (shaft_ex, shaft_ey)],
                      fill=color, width=2)

            # Arrowhead — filled triangle
            base1 = (shaft_ex + px * a_half,
                     shaft_ey + py * a_half)
            base2 = (shaft_ex - px * a_half,
                     shaft_ey - py * a_half)
            draw.polygon([(tip_x, tip_y), base1, base2], fill=color)

            # Label — offset outward from the tip
            lx = cx + sx + dx * 10
            ly = cy + sy + dy * 10
            # Centre the text on (lx, ly); approximate glyph size ~6×10 px
            draw.text((lx - 4, ly - 6), label, fill=color)

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
        self._request_render()

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
        self._request_render()

    def _on_release_right(self, event):
        self._drag_right_start = None

    def _on_scroll(self, event):
        factor = 0.9 if event.delta > 0 else 1.1
        if self._distance is not None:
            self._distance *= factor
            self._request_render()

    def _on_scroll_up(self, event):
        if self._distance is not None:
            self._distance *= 0.9
            self._request_render()

    def _on_scroll_down(self, event):
        if self._distance is not None:
            self._distance *= 1.1
            self._request_render()

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
        if self._rendering:
            # A render is in progress — retry after a short delay
            self._resize_after_id = self._canvas.after(
                100, lambda: self._do_resize(new_w, new_h))
            return
        self._RENDER_W = new_w
        self._RENDER_H = new_h
        # Rebuild with the stored meshes
        meshes = getattr(self, '_o3d_meshes', None)
        if meshes:
            self.set_meshes(meshes)


# ===================================================================
#  Unified tkinter selection dialog
