#!/usr/bin/env python3
# viewer_loaders.py
"""
Mesh, VTP, and particle-source loading helpers for the BeamOnTarget viewer.

Functions:
  - _resolve_viewer_path, _FOLDER_COLORS -- path helpers and colour palette
  - _load_stl, _read_vtp_as_o3d, _colour_mesh_by_power -- mesh I/O
  - _show_in_open3d, _scan_vtp_vmax -- utility helpers
  - _load_selected_geometry/results -- high-level loaders
  - _load_bl_file, _build_source_geometries -- particle source helpers
"""
import os
import glob

import numpy as np
import open3d as o3d
import pyvista as pv
import pandas as pd

from gui.viewer_widget import _apply_jet_colormap

def _resolve_viewer_path(script_dir, relative_path):
    """Resolve a simulation-file path relative to the main application folder."""
    return os.path.join(script_dir, relative_path)


# ---------------------------------------------------------------------------
#  Colour palette for geometry folders
# ---------------------------------------------------------------------------
_FOLDER_COLORS = [
    (0.8, 0.2, 0.2), (0.2, 0.6, 0.8), (0.2, 0.8, 0.3),
    (0.9, 0.7, 0.1), (0.6, 0.3, 0.7), (0.9, 0.4, 0.1),
    (0.4, 0.8, 0.8), (0.8, 0.4, 0.6),
]

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
        folder_abs = _resolve_viewer_path(script_dir, folder)
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
#  Particle source helpers
# ===================================================================

def _load_bl_file(bl_path):
    """Read a .bl file and return a DataFrame with source data."""
    try:
        df = pd.read_csv(bl_path, comment='#', sep=r'\s+')
    except Exception:
        return None
    return df


def _build_source_geometries(bl_paths, arrow_length=0.0,
                              show_direction=False):
    """Build Open3D geometries for particle sources.

    Source positions are rendered as a **PointCloud** (zero triangle
    overhead — just coloured dots).  Direction arrows, when requested,
    are built as a single merged TriangleMesh (cylinder + cone per
    source).

    Each source is coloured by its ``CurrentDensity_A_m2``.

    Returns
    -------
    geoms : list[o3d.geometry.Geometry]
        A PointCloud and, optionally, a merged arrow TriangleMesh.
    vmax : float
        Maximum current density across all sources.
    """
    # Collect all sources across files (vectorised — no iterrows)
    dfs = []
    for bl in bl_paths:
        df = _load_bl_file(bl)
        if df is not None and not df.empty:
            dfs.append(df)

    if not dfs:
        return [], 0.0

    combined = pd.concat(dfs, ignore_index=True)

    # Extract positions, directions, current densities as numpy arrays
    positions = combined[['CenterX', 'CenterY', 'CenterZ']].to_numpy(
        dtype=np.float64)
    directions = combined[['DirX', 'DirY', 'DirZ']].to_numpy(
        dtype=np.float64)
    currents = combined['CurrentDensity_A_m2'].to_numpy(dtype=np.float64)

    # Normalise for colouring
    vmax = currents.max() if len(currents) > 0 else 0.0
    if vmax > 0:
        normed = np.clip(currents / vmax, 0, 1)
    else:
        normed = np.zeros(len(currents))

    colors = _apply_jet_colormap(normed)

    # --- Source dots as PointCloud (zero triangles) ---
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(positions)
    pc.colors = o3d.utility.Vector3dVector(colors)

    geoms = [pc]

    # --- Direction arrows (only when requested) ---
    if show_direction:
        # Auto-scale arrow length from scene size
        if len(positions) > 1:
            extent = positions.max(axis=0) - positions.min(axis=0)
            scene_size = np.linalg.norm(extent)
        else:
            scene_size = 1.0
        if arrow_length <= 0:
            arrow_length = scene_size * 0.08
        shaft_r = scene_size * 0.003
        shaft_len = arrow_length * 0.75
        cone_len = arrow_length * 0.25
        cone_r = shaft_r * 2.5

        unit_cyl = o3d.geometry.TriangleMesh.create_cylinder(
            radius=shaft_r, height=shaft_len, resolution=6)
        unit_cyl.translate([0, 0, shaft_len / 2])  # base at origin
        unit_cyl.compute_vertex_normals()
        cyl_verts_np = np.asarray(unit_cyl.vertices)
        cyl_tris_np = np.asarray(unit_cyl.triangles)
        n_cv = len(cyl_verts_np)

        unit_cone = o3d.geometry.TriangleMesh.create_cone(
            radius=cone_r, height=cone_len, resolution=6)
        unit_cone.translate([0, 0, shaft_len])  # base at top of shaft
        unit_cone.compute_vertex_normals()
        cone_verts_np = np.asarray(unit_cone.vertices)
        cone_tris_np = np.asarray(unit_cone.triangles)
        n_kov = len(cone_verts_np)

        all_verts = []
        all_tris = []
        all_colors_v = []
        vert_offset = 0
        z_axis = np.array([0.0, 0.0, 1.0])

        for i in range(len(positions)):
            pos = positions[i]
            col = colors[i]
            d = directions[i]
            d_norm = d / (np.linalg.norm(d) + 1e-12)

            # Compute rotation matrix from Z to direction
            rot_axis = np.cross(z_axis, d_norm)
            rot_norm = np.linalg.norm(rot_axis)
            if rot_norm > 1e-6:
                rot_axis /= rot_norm
                angle = np.arccos(np.clip(np.dot(z_axis, d_norm), -1, 1))
                R = o3d.geometry.get_rotation_matrix_from_axis_angle(
                    rot_axis * angle)
            elif np.dot(z_axis, d_norm) < 0:
                R = o3d.geometry.get_rotation_matrix_from_axis_angle(
                    np.array([1.0, 0.0, 0.0]) * np.pi)
            else:
                R = np.eye(3)

            # Cylinder
            cv = (cyl_verts_np @ R.T) + pos
            all_verts.append(cv)
            all_tris.append(cyl_tris_np + vert_offset)
            all_colors_v.append(np.tile(col, (n_cv, 1)))
            vert_offset += n_cv

            # Cone
            kov = (cone_verts_np @ R.T) + pos
            all_verts.append(kov)
            all_tris.append(cone_tris_np + vert_offset)
            all_colors_v.append(np.tile(col, (n_kov, 1)))
            vert_offset += n_kov

        # Build single merged arrow mesh
        arrow_mesh = o3d.geometry.TriangleMesh()
        arrow_mesh.vertices = o3d.utility.Vector3dVector(
            np.vstack(all_verts))
        arrow_mesh.triangles = o3d.utility.Vector3iVector(
            np.vstack(all_tris))
        arrow_mesh.vertex_colors = o3d.utility.Vector3dVector(
            np.vstack(all_colors_v))
        arrow_mesh.compute_vertex_normals()
        geoms.append(arrow_mesh)

    return geoms, vmax


# ===================================================================
#  Source-viewer dialog
# ===================================================================

