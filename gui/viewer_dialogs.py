#!/usr/bin/env python3
# viewer_dialogs.py
"""
Selection dialogs for the BeamOnTarget embedded 3D viewer.

  - _pick_items_dialog   -- geometry + results + optional sources checklist
  - _pick_sources_dialog -- dedicated particle-source viewer with geometry overlay
"""
import os
import glob
import threading

import tkinter as tk
from tkinter import ttk

from gui.viewer_widget import _EmbeddedViewer, _draw_colorbar_on_canvas
from gui.viewer_loaders import (
    _FOLDER_COLORS, _scan_vtp_vmax,
    _load_selected_geometry, _load_selected_results,
    _build_source_geometries,
)

def _pick_items_dialog(parent, geometry_folders, results_dir,
                       geo_checked=True, res_checked=True,
                       load_and_show_fn=None, source_dir=None,
                       source_dirs=None):
    """Persistent selection dialog with geometry folders, result files,
    and optionally particle-source (.bl) files.

    *source_dirs* overrides *source_dir* when provided: it is a list of
    ``{label, dir_abs, transform}`` dicts (one per beam source) which are
    shown as separate labelled sections in the Sources checkbox list.

    The dialog stays open so the user can change the selection and click
    **View** repeatedly.  Each click launches a new Open3D window via
    *load_and_show_fn(sel_geo, sel_res, sf, sel_bl, show_dir, arrow_len)*
    in a daemon thread.
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

    # --- Source options row (shown when any source dirs are available) ---
    show_dir_var = tk.BooleanVar(value=True)
    arrow_len_var = tk.StringVar(value="0")
    # Resolve effective source dirs: prefer source_dirs list, fall back to source_dir
    _src_dirs = source_dirs or []
    if not _src_dirs and source_dir and os.path.isdir(source_dir):
        _src_dirs = [{"label": "Sources", "dir_abs": source_dir, "transform": None}]
    if _src_dirs:
        src_opts_frm = ttk.Frame(bottom_frm)
        src_opts_frm.pack(fill="x", pady=(4, 0))
        ttk.Checkbutton(src_opts_frm, text="Plot source direction",
                         variable=show_dir_var).pack(side="left")
        ttk.Label(src_opts_frm, text="    Arrow length (m):",
                  font=("", 10)).pack(side="left", padx=(12, 0))
        ttk.Entry(src_opts_frm, textvariable=arrow_len_var,
                  width=8).pack(side="left", padx=(6, 0))
        ttk.Label(src_opts_frm, text="(0 = auto)",
                  foreground="grey", font=("", 9)).pack(side="left", padx=(4, 0))

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

    # --- Source (.bl) files — one section per beam label ---
    all_bl_vars = {}        # bl_abs_path → BooleanVar
    all_bl_transforms = {}  # bl_abs_path → transform dict or None

    if _src_dirs:
        ttk.Label(inner, text="Sources",
                  font=("", 11, "bold")).pack(anchor="w", padx=4, pady=(12, 2))
        for src_entry in _src_dirs:
            lbl = src_entry["label"]
            d_abs = src_entry["dir_abs"]
            tfm = src_entry["transform"]
            bl_files = sorted(glob.glob(os.path.join(d_abs, "*.bl")))
            if not bl_files:
                continue
            # Sub-heading per beam
            ttk.Label(inner, text=f"  {lbl}",
                      font=("", 10, "italic")).pack(anchor="w", padx=8, pady=(4, 0))
            for bl in bl_files:
                bn = os.path.splitext(os.path.basename(bl))[0]
                var = tk.BooleanVar(value=False)
                ttk.Checkbutton(inner, text=f"    {bn}",
                                variable=var).pack(anchor="w", padx=12, pady=1)
                all_bl_vars[bl] = var
                all_bl_transforms[bl] = tfm

    # --- buttons (in the bottom bar) ---
    def _all():
        for v in (list(all_geo_vars.values()) + list(all_res_vars.values())
                  + list(all_bl_vars.values())):
            v.set(True)

    def _none():
        for v in (list(all_geo_vars.values()) + list(all_res_vars.values())
                  + list(all_bl_vars.values())):
            v.set(False)

    ttk.Button(btn_frm, text="All", width=6, command=_all).pack(side="left", padx=2)
    ttk.Button(btn_frm, text="None", width=6, command=_none).pack(side="left", padx=2)

    def _get_selection():
        sel_g = [f for f, v in all_geo_vars.items() if v.get()]
        sel_r = [p for p, v in all_res_vars.items() if v.get()]
        sel_bl = [p for p, v in all_bl_vars.items() if v.get()]
        try:
            sf = float(scale_var.get())
        except (ValueError, TypeError):
            sf = 1.0
        sd = show_dir_var.get()
        try:
            al = float(arrow_len_var.get())
        except (ValueError, TypeError):
            al = 0.0
        return sel_g, sel_r, sf, sel_bl, sd, al

    if load_and_show_fn is not None:
        # ---- Persistent mode: View button renders into embedded viewer ----
        _busy = threading.Lock()

        def _view():
            if not _busy.acquire(blocking=False):
                status_var.set("Still loading… please wait.")
                return
            sel_g, sel_r, sf, sel_bl, sd, al = _get_selection()
            if not sel_g and not sel_r and not sel_bl:
                status_var.set("Nothing selected.")
                _busy.release()
                return
            n = len(sel_g) + len(sel_r) + len(sel_bl)
            status_var.set(f"Loading {n} item(s)…")
            view_btn.configure(state="disabled")

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
                target=lambda: _view_worker(sel_g, sel_r, sf,
                                            sel_bl, sd, al),
                daemon=True,
            ).start()

        def _view_worker(sel_g, sel_r, sf, sel_bl, sd, al):
            try:
                geoms, vmax, title = load_and_show_fn(
                    sel_g, sel_r, sf, sel_bl, sd, al)
            except Exception as exc:
                try:
                    dlg.after(0, lambda: status_var.set(f"Error: {exc}"))
                except tk.TclError:
                    pass
                return
            finally:
                _busy.release()
                try:
                    dlg.after(0, lambda: view_btn.configure(state="normal"))
                except tk.TclError:
                    pass
            if not geoms:
                try:
                    dlg.after(0, lambda: status_var.set("No meshes to display."))
                except tk.TclError:
                    pass
                return
            # Schedule the render on the main Tk thread
            def _do_render():
                try:
                    dlg._viewer.set_meshes(geoms)
                    info = title
                    info += "  —  drag to rotate, scroll to zoom"
                    status_var.set(info)
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
            dlg.result_geo, dlg.result_res, _, _, _, _ = _get_selection()
            dlg.destroy()

        def _cancel():
            dlg.destroy()

        ttk.Button(btn_frm, text="View", command=_ok).pack(side="right", padx=2)
        ttk.Button(btn_frm, text="Cancel", command=_cancel).pack(side="right", padx=2)

        parent.wait_window(dlg)

    return dlg.result_geo, dlg.result_res, dlg


def _pick_sources_dialog(parent, script_dir, source_dir, geometry_folders,
                         beam_sources=None):
    """Selection dialog for particle sources with embedded 3D viewer.

    Shows .bl files as checkboxes (grouped by beam label when *beam_sources*
    is supplied), plus options for direction arrows and arrow length.
    Geometry folders can optionally be overlaid.

    When *beam_sources* is a list of ``{label, directory, transform}`` dicts
    each beam's files are shown under its own heading and the stored transform
    is applied before rendering so all sources appear in the Tokamak frame.
    """
    from tkinter import ttk

    # Build effective source list: either multi-beam or single directory
    # Each entry: {label, dir_abs, transform}
    _effective_sources = []
    if beam_sources:
        for bs in beam_sources:
            d = bs.get("directory", "")
            if os.path.isabs(d):
                d_abs = d
            else:
                # Directories in config are relative to the config/ subfolder
                # (e.g. "..\\BEAM_CONFIGS\\DNB" from config/config.json).
                d_abs = os.path.normpath(
                    os.path.join(script_dir, "config", d))
            if os.path.isdir(d_abs):
                _effective_sources.append({
                    "label": bs.get("label", os.path.basename(d_abs)),
                    "dir_abs": d_abs,
                    "transform": bs.get("transform", None),
                })
    if not _effective_sources:
        # Fallback: single directory (legacy)
        src_abs = (_resolve_viewer_path(script_dir, source_dir)
                   if not os.path.isabs(source_dir) else source_dir)
        _effective_sources.append({"label": "Sources", "dir_abs": src_abs,
                                    "transform": None})

    dlg = tk.Toplevel(parent)
    dlg.title("View Particle Sources")
    dlg.geometry("1100x650")
    dlg.minsize(700, 400)

    dlg.withdraw()
    dlg.update_idletasks()
    dlg.deiconify()

    # ====== BOTTOM BAR ======
    bottom_frm = ttk.Frame(dlg)
    bottom_frm.pack(side="bottom", fill="x", padx=8, pady=(0, 4))

    # --- Options row ---
    opts_frm = ttk.Frame(bottom_frm)
    opts_frm.pack(fill="x", pady=(4, 0))

    show_dir_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(opts_frm, text="Plot source direction",
                     variable=show_dir_var).pack(side="left")

    ttk.Label(opts_frm, text="    Arrow length (m):",
              font=("", 10)).pack(side="left", padx=(12, 0))
    arrow_len_var = tk.StringVar(value="0")
    ttk.Entry(opts_frm, textvariable=arrow_len_var, width=8).pack(
        side="left", padx=(6, 0))
    ttk.Label(opts_frm, text="(0 = auto)",
              foreground="grey", font=("", 9)).pack(side="left", padx=(4, 0))

    # --- Status ---
    status_var = tk.StringVar(value="")
    ttk.Label(bottom_frm, textvariable=status_var,
              foreground="grey").pack(fill="x", pady=(2, 0))

    # --- Buttons ---
    btn_frm = ttk.Frame(bottom_frm)
    btn_frm.pack(fill="x", pady=(4, 0))

    # ====== MAIN AREA ======
    main_pane = tk.PanedWindow(dlg, orient="horizontal", sashwidth=6,
                                sashrelief="raised", bg="#cccccc")
    main_pane.pack(fill="both", expand=True, padx=4, pady=4)

    # -- LEFT: checkboxes --
    left_frm = ttk.Frame(main_pane)
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
    canvas.grid(row=0, column=0, sticky="nsew")
    v_scroll.grid(row=0, column=1, sticky="ns")
    h_scroll.grid(row=1, column=0, sticky="ew")
    left_frm.rowconfigure(0, weight=1)
    left_frm.columnconfigure(0, weight=1)
    main_pane.add(left_frm, minsize=180, width=260)

    # -- CENTRE: 3D viewport --
    view_frm = ttk.LabelFrame(main_pane, text="3D View")
    viewer = _EmbeddedViewer(view_frm)
    main_pane.add(view_frm, minsize=300)

    # -- RIGHT: colorbar --
    cbar_frm = ttk.LabelFrame(main_pane, text="Current Density")
    cbar_canvas = tk.Canvas(cbar_frm, width=120, height=500,
                            bg="#f0f2f5", highlightthickness=0)
    cbar_canvas.pack(fill="both", expand=True, padx=4, pady=4)
    cbar_canvas.create_text(60, 250, text="No data yet",
                            fill="#999", font=("Segoe UI", 9))
    main_pane.add(cbar_frm, minsize=130, width=140)

    dlg._cbar_canvas = cbar_canvas
    dlg._viewer = viewer

    # --- Populate checkboxes ---
    all_bl_vars = {}   # bl_abs_path → BooleanVar
    all_bl_transforms = {}  # bl_abs_path → transform dict or None
    all_geo_vars = {}  # folder_name → BooleanVar

    # Source files — grouped by beam label
    any_bl_found = False
    for src_entry in _effective_sources:
        lbl = src_entry["label"]
        d_abs = src_entry["dir_abs"]
        tfm = src_entry["transform"]
        bl_files = sorted(glob.glob(os.path.join(d_abs, "*.bl")))
        if not bl_files:
            continue
        any_bl_found = True
        ttk.Label(inner, text=lbl,
                  font=("", 11, "bold")).pack(anchor="w", padx=4, pady=(8, 2))
        for bl in bl_files:
            bn = os.path.splitext(os.path.basename(bl))[0]
            var = tk.BooleanVar(value=False)
            ttk.Checkbutton(inner, text=f"  {bn}",
                            variable=var).pack(anchor="w", padx=12, pady=1)
            all_bl_vars[bl] = var
            all_bl_transforms[bl] = tfm

    if not any_bl_found:
        ttk.Label(inner, text="(no .bl files found)",
                  foreground="grey").pack(padx=8, pady=8)

    # Geometry overlays
    if geometry_folders:
        ttk.Label(inner, text="Geometry Overlay",
                  font=("", 11, "bold")).pack(anchor="w", padx=4, pady=(12, 2))
        ci = 0
        for folder in geometry_folders:
            color = _FOLDER_COLORS[ci % len(_FOLDER_COLORS)]
            ci += 1
            var = tk.BooleanVar(value=False)
            frm = ttk.Frame(inner)
            frm.pack(fill="x", pady=1, padx=8)
            sw = tk.Canvas(frm, width=14, height=14, highlightthickness=0)
            hex_c = "#%02x%02x%02x" % tuple(int(c * 255) for c in color)
            sw.create_rectangle(0, 0, 14, 14, fill=hex_c, outline=hex_c)
            sw.pack(side="left", padx=(0, 6))
            ttk.Checkbutton(frm, text=folder, variable=var).pack(side="left")
            all_geo_vars[folder] = var

    # --- Button actions ---
    def _all():
        for v in list(all_bl_vars.values()) + list(all_geo_vars.values()):
            v.set(True)

    def _none():
        for v in list(all_bl_vars.values()) + list(all_geo_vars.values()):
            v.set(False)

    ttk.Button(btn_frm, text="All", width=6, command=_all).pack(
        side="left", padx=2)
    ttk.Button(btn_frm, text="None", width=6, command=_none).pack(
        side="left", padx=2)

    _busy = threading.Lock()

    def _view():
        if not _busy.acquire(blocking=False):
            status_var.set("Still loading… please wait.")
            return
        sel_bl = [p for p, v in all_bl_vars.items() if v.get()]
        sel_geo = [f for f, v in all_geo_vars.items() if v.get()]
        if not sel_bl and not sel_geo:
            status_var.set("Nothing selected.")
            _busy.release()
            return

        try:
            al = float(arrow_len_var.get())
        except (ValueError, TypeError):
            al = 0.0
        sd = show_dir_var.get()

        n = len(sel_bl) + len(sel_geo)
        status_var.set(f"Loading {n} item(s)…")
        view_btn.configure(state="disabled")
        dlg.update_idletasks()

        threading.Thread(
            target=lambda: _view_worker(sel_bl, sel_geo, al, sd),
            daemon=True).start()

    def _view_worker(sel_bl, sel_geo, al, sd):
        try:
            geoms = []
            vmax = 0.0

            # Load sources with per-file transforms
            if sel_bl:
                transforms = {bl: all_bl_transforms[bl]
                              for bl in sel_bl
                              if all_bl_transforms.get(bl) is not None}
                src_geoms, vmax = _build_source_geometries(
                    sel_bl, arrow_length=al, show_direction=sd,
                    transforms=transforms if transforms else None)
                geoms += src_geoms

            # Load geometry overlays
            if sel_geo:
                geoms += _load_selected_geometry(script_dir, geometry_folders,
                                                 sel_geo)
        except Exception as exc:
            try:
                dlg.after(0, lambda: status_var.set(f"Error: {exc}"))
            except tk.TclError:
                pass
            return
        finally:
            _busy.release()
            try:
                dlg.after(0, lambda: view_btn.configure(state="normal"))
            except tk.TclError:
                pass

        if not geoms:
            try:
                dlg.after(0, lambda: status_var.set("No meshes to display."))
            except tk.TclError:
                pass
            return

        def _do_render():
            try:
                dlg._viewer.set_meshes(geoms)
                if vmax > 0:
                    _draw_colorbar_on_canvas(
                        dlg._cbar_canvas, 0.0, vmax, 1.0,
                        unit_label="A/m²")
                info = f"Sources ({len(geoms)} objects)"
                if vmax > 0:
                    info += f"  —  max j = {vmax:.2e} A/m²"
                info += "  —  drag to rotate, scroll to zoom"
                status_var.set(info)
            except Exception as exc:
                status_var.set(f"Render error: {exc}")
        try:
            dlg.after(0, _do_render)
        except tk.TclError:
            pass

    def _close():
        if hasattr(dlg, '_viewer') and dlg._viewer._vis is not None:
            try:
                dlg._viewer._vis.destroy_window()
                dlg._viewer._vis = None
            except Exception:
                pass
        dlg.destroy()

    view_btn = ttk.Button(btn_frm, text="▶ View", command=_view)
    view_btn.pack(side="right", padx=2)
    ttk.Button(btn_frm, text="Close", command=_close).pack(
        side="right", padx=2)


