#!/usr/bin/env python3
# viewer.py
"""
Built-in 3D viewer for BeamOnTarget -- public entry points.

Imports from:
  - viewer_widget  : _EmbeddedViewer, colormap and colorbar utilities
  - viewer_loaders : mesh / VTP / particle-source loading helpers
  - viewer_dialogs : _pick_items_dialog, _pick_sources_dialog

Entry points called from sim_gui.py:
  - view_geometry(parent, script_dir, geometry_folders)
  - view_results(parent, script_dir, output_dir, geometry_folders)
  - view_all(parent, script_dir, output_dir, geometry_folders)
  - view_sources(parent, script_dir, source_dir, geometry_folders)
"""
import os
import glob

from gui.viewer_loaders import (
    _resolve_viewer_path, _load_selected_geometry, _build_source_geometries,
    _read_vtp_as_o3d, _colour_mesh_by_power, _show_in_open3d,
)
from gui.viewer_dialogs import _pick_items_dialog, _pick_sources_dialog

# ===================================================================
#  High-level entry points (called from sim_gui.py)
# ===================================================================

class _MeshCache:
    """Caches loaded geometry / results / sources across View clicks.

    Only the items whose selection or parameters changed are reloaded;
    everything else is reused from the previous call.
    """

    def __init__(self, script_dir, geometry_folders):
        self._script_dir = script_dir
        self._geometry_folders = geometry_folders
        # geometry:  folder_name → list[mesh]
        self._geo_cache = {}
        # results:   vtp_path → (mesh, power_array)
        self._res_cache = {}
        self._res_vmax = 0.0
        self._prev_res_sel = []
        # sources:   separate caches for dots (PointCloud) and arrows
        self._src_files_key = None   # tuple(sorted bl paths)
        self._src_pc = None          # PointCloud (dots)
        self._src_vmax = 0.0
        self._src_arrow_key = None   # (files_key, arrow_len, show_dir)
        self._src_arrow_mesh = None  # merged arrow TriangleMesh or None

    # -- geometry --
    def get_geometry(self, sel_folders):
        """Return meshes for *sel_folders*, loading only new ones."""
        meshes = []
        for folder in sel_folders:
            if folder not in self._geo_cache:
                self._geo_cache[folder] = _load_selected_geometry(
                    self._script_dir, self._geometry_folders, [folder])
            meshes += self._geo_cache[folder]
        return meshes

    # -- results --
    def get_results(self, sel_vtps, sf):
        """Return coloured meshes + vmax, reloading only new VTPs.

        If the selection changed we must re-colour everything because
        the global vmax may have changed.
        """
        sel_set = tuple(sorted(sel_vtps))
        if sel_set == tuple(sorted(self._prev_res_sel)):
            # Exact same selection — return cached
            return [m for m, _ in
                    [self._res_cache[v] for v in sel_vtps
                     if v in self._res_cache]], self._res_vmax

        # Load any VTPs we haven't seen
        for vtp in sel_vtps:
            if vtp not in self._res_cache:
                mesh, power = _read_vtp_as_o3d(vtp)
                if mesh is not None:
                    self._res_cache[vtp] = (mesh, power)

        # Compute global vmax across selection
        vmax_all = 0.0
        pairs = []
        for vtp in sel_vtps:
            if vtp not in self._res_cache:
                continue
            mesh, power = self._res_cache[vtp]
            if power is not None and len(power) > 0:
                vmax_all = max(vmax_all, float(power.max()))
            pairs.append((mesh, power))

        # Re-colour with consistent scale
        for mesh, power in pairs:
            _colour_mesh_by_power(mesh, power,
                                  global_vmax=vmax_all if vmax_all > 0 else None)

        self._res_vmax = vmax_all
        self._prev_res_sel = list(sel_vtps)
        return [m for m, _ in pairs], vmax_all

    # -- sources --
    def get_sources(self, sel_bl, arrow_len, show_dir):
        """Return source geometries + vmax.

        The PointCloud (dots) is cached by file selection alone —
        changing arrow length or direction toggle does NOT rebuild it.
        The arrow mesh is cached separately by (files, arrow_len,
        show_dir) so only the arrows are rebuilt when those change.
        """
        files_key = tuple(sorted(sel_bl))

        # Rebuild dots only when file selection changes
        if files_key != self._src_files_key:
            # Need full rebuild (dots + arrows)
            all_geoms, vmax = _build_source_geometries(
                sel_bl, arrow_length=arrow_len, show_direction=show_dir)
            # First element is always the PointCloud
            self._src_pc = all_geoms[0] if all_geoms else None
            self._src_vmax = vmax
            self._src_files_key = files_key
            arrow_key = (files_key, arrow_len, show_dir)
            self._src_arrow_key = arrow_key
            self._src_arrow_mesh = all_geoms[1] if len(all_geoms) > 1 else None
            return list(all_geoms), vmax

        # Dots are cached — check if arrows need rebuilding
        arrow_key = (files_key, arrow_len, show_dir)
        if arrow_key != self._src_arrow_key:
            # Rebuild only arrows (dots will be identical, discard them)
            all_geoms, _ = _build_source_geometries(
                sel_bl, arrow_length=arrow_len, show_direction=show_dir)
            self._src_arrow_key = arrow_key
            self._src_arrow_mesh = all_geoms[1] if len(all_geoms) > 1 else None

        # Assemble result
        geoms = []
        if self._src_pc is not None:
            geoms.append(self._src_pc)
        if self._src_arrow_mesh is not None:
            geoms.append(self._src_arrow_mesh)
        return geoms, self._src_vmax


def _status_msg(parent, msg):
    if hasattr(parent, "_log"):
        parent._log(f"🔍 {msg}\n")


def view_geometry(parent, script_dir, geometry_folders, source_dir=None):
    """Geometry viewer — persistent selection dialog → Open3D window."""
    src_abs = None
    if source_dir:
        src_abs = (_resolve_viewer_path(script_dir, source_dir)
                   if not os.path.isabs(source_dir) else source_dir)
    cache = _MeshCache(script_dir, geometry_folders)

    def _load_and_show(sel_geo, sel_res, sf, sel_bl, sd, al):
        geoms = cache.get_geometry(sel_geo) if sel_geo else []
        vmax = 0.0
        if sel_res:
            res_meshes, vmax = cache.get_results(sel_res, sf)
            geoms += res_meshes
        if sel_bl:
            src_geoms, _ = cache.get_sources(sel_bl, al, sd)
            geoms += src_geoms
        if not geoms:
            parent.after(0, lambda: _status_msg(parent, "No meshes found."))
            return [], 0.0, ""
        parent.after(0, lambda: _status_msg(parent,
                     f"Loading {len(geoms)} meshes…"))
        return geoms, vmax, f"Geometry ({len(geoms)} meshes)"

    _pick_items_dialog(
        parent, geometry_folders, results_dir=None,
        geo_checked=False, res_checked=False,
        load_and_show_fn=_load_and_show,
        source_dir=src_abs,
    )


def view_results(parent, script_dir, output_dir, geometry_folders=None,
                 source_dir=None):
    """Results viewer — persistent selection dialog → Open3D heatmap."""
    outdir_abs = (os.path.join(script_dir, output_dir)
                  if not os.path.isabs(output_dir) else output_dir)
    geometry_folders = geometry_folders or {}
    src_abs = None
    if source_dir:
        src_abs = (_resolve_viewer_path(script_dir, source_dir)
                   if not os.path.isabs(source_dir) else source_dir)
    cache = _MeshCache(script_dir, geometry_folders)

    def _load_and_show(sel_geo, sel_res, sf, sel_bl, sd, al):
        geoms = cache.get_geometry(sel_geo) if sel_geo else []
        vmax = 0.0
        if sel_res:
            res_meshes, vmax = cache.get_results(sel_res, sf)
            geoms += res_meshes
        if sel_bl:
            src_geoms, _ = cache.get_sources(sel_bl, al, sd)
            geoms += src_geoms
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
        geo_checked=False, res_checked=False,
        load_and_show_fn=_load_and_show,
        source_dir=src_abs,
    )


def view_all(parent, script_dir, output_dir, geometry_folders,
             source_dir=None):
    """Show everything — persistent selection dialog → Open3D window."""
    outdir_abs = (os.path.join(script_dir, output_dir)
                  if not os.path.isabs(output_dir) else output_dir)
    src_abs = None
    if source_dir:
        src_abs = (_resolve_viewer_path(script_dir, source_dir)
                   if not os.path.isabs(source_dir) else source_dir)
    cache = _MeshCache(script_dir, geometry_folders)

    def _load_and_show(sel_geo, sel_res, sf, sel_bl, sd, al):
        geoms = cache.get_geometry(sel_geo) if sel_geo else []
        vmax = 0.0
        if sel_res:
            res_meshes, vmax = cache.get_results(sel_res, sf)
            geoms += res_meshes
        if sel_bl:
            src_geoms, _ = cache.get_sources(sel_bl, al, sd)
            geoms += src_geoms
        if not geoms:
            parent.after(0, lambda: _status_msg(parent, "No meshes found."))
            return [], 0.0, ""
        title = f"Viewer ({len(geoms)} meshes)"
        parent.after(0, lambda: _status_msg(parent,
                     f"Loading {len(geoms)} meshes…"))
        return geoms, vmax, title

    _pick_items_dialog(
        parent, geometry_folders, outdir_abs,
        geo_checked=False, res_checked=False,
        load_and_show_fn=_load_and_show,
        source_dir=src_abs,
    )


def view_sources(parent, script_dir, source_dir, geometry_folders=None):
    """Particle-source viewer — shows beamlet locations + optional arrows.

    Colour encodes current density (A/m²).
    """
    geometry_folders = geometry_folders or {}
    _pick_sources_dialog(parent, script_dir, source_dir, geometry_folders)


# ===================================================================
#  Standalone test
# ===================================================================
if __name__ == "__main__":
    import json

    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(os.path.dirname(script_dir), "config", "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)

    gf = cfg.get("GEOMETRY_FOLDERS", {})
    folders = list(gf.keys())
    print(f"Available folders: {folders}")

    items = _load_selected_geometry(script_dir, gf, folders)
    print(f"Loaded {len(items)} geometry meshes")
    if items:
        _show_in_open3d(items, "Geometry Test")
