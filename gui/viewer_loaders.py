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
import trimesh as _trimesh

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

# Directory used to cache pre-decimated viewer meshes.  Set at startup by
# _load_selected_geometry from the geometry-folder settings.
_VIEWER_CACHE_DIR = None


def _lod_cache_path(stl_path, max_faces, scale):
    """Return the path of the cached LOD file for *stl_path*."""
    global _VIEWER_CACHE_DIR
    if not _VIEWER_CACHE_DIR:
        return None
    stem = os.path.splitext(os.path.basename(stl_path))[0]
    # Encode scale in the filename so a different scale gets its own cache
    scale_tag = f"_s{scale:.4g}".replace(".", "p") if scale != 1 else ""
    cache_name = f"{stem}_lod{max_faces}{scale_tag}.stl"
    return os.path.join(_VIEWER_CACHE_DIR, cache_name)


def _voxel_decimate(verts, faces, target_faces):
    """Fast numpy voxel-grid mesh decimation.

    Clusters vertices into a regular voxel grid and reindexes faces.
    Much faster than Open3D\'s vertex_clustering for very large meshes
    (handles 26M faces in ~20s vs 60s).

    Returns (new_verts, new_faces) as float64 / int32 arrays.
    """
    origin = verts.min(axis=0)
    extent = float(np.linalg.norm(verts.max(axis=0) - origin))
    if extent == 0:
        return verts, faces
    voxel_size = extent / (max(target_faces, 1) ** 0.5) * 1.5

    # Map each vertex to a voxel index encoded as a single int64
    vi = np.floor((verts - origin) / voxel_size).astype(np.int64)
    dims = vi.max(axis=0) + 2
    voxel_id = (vi[:, 0] * dims[1] + vi[:, 1]) * dims[2] + vi[:, 2]

    # Sort voxel IDs to group vertices; build inverse mapping via cumsum
    sort_idx = np.argsort(voxel_id, kind='stable')
    vs = voxel_id[sort_idx]
    changes = np.empty(len(vs), dtype=bool)
    changes[0] = True
    changes[1:] = vs[1:] != vs[:-1]
    cid = np.cumsum(changes) - 1
    inv = np.empty_like(cid, dtype=np.int32)
    inv[sort_idx] = cid
    n = int(cid[-1]) + 1

    # Compute centroid per voxel cluster
    counts = np.bincount(inv, minlength=n).astype(np.float64)
    nx = np.bincount(inv, weights=verts[:, 0], minlength=n) / counts
    ny = np.bincount(inv, weights=verts[:, 1], minlength=n) / counts
    nz = np.bincount(inv, weights=verts[:, 2], minlength=n) / counts
    new_verts = np.column_stack([nx, ny, nz])

    # Remap faces and discard degenerate triangles
    nf = inv[faces]
    valid = (nf[:, 0] != nf[:, 1]) & (nf[:, 1] != nf[:, 2]) & (nf[:, 0] != nf[:, 2])
    return new_verts, nf[valid].astype(np.int32)


def _load_stl(stl_path, scale, color, viewer_max_faces=None):
    """Read an STL, optionally scale, decimate for viewer speed, paint colour.

    Uses trimesh for fast binary STL reading (~30x faster than Open3D on
    large files), then applies voxel-grid decimation if needed.  Decimated
    meshes are cached to disk so subsequent opens are near-instant.

    Args:
        stl_path: path to the .stl file.
        scale: uniform scale factor.
        color: RGB tuple for uniform colouring.
        viewer_max_faces: target face count for LOD decimation.  The cached
            result is reused on all subsequent opens.
    """
    cache_file = _lod_cache_path(stl_path, viewer_max_faces, scale) if viewer_max_faces else None

    if cache_file and os.path.isfile(cache_file):
        # Fast path: small pre-decimated cache file
        mesh = o3d.io.read_triangle_mesh(cache_file)
        if not mesh.is_empty():
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color(color)
            return mesh

    # Load with trimesh — fast binary STL reader
    try:
        tm = _trimesh.load(stl_path, process=False)
        verts = np.asarray(tm.vertices, dtype=np.float64)
        faces = np.asarray(tm.faces, dtype=np.int32)
    except Exception:
        # Fallback to Open3D reader
        mesh = o3d.io.read_triangle_mesh(stl_path)
        if mesh.is_empty():
            return None
        verts = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)

    if scale != 1:
        verts = verts * scale

    needs_decimate = viewer_max_faces and len(faces) > viewer_max_faces
    if needs_decimate:
        verts, faces = _voxel_decimate(verts, faces, viewer_max_faces)

    # Build Open3D mesh from numpy arrays
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    if mesh.is_empty():
        return None

    # Save decimated LOD to disk for next open
    if needs_decimate and cache_file:
        try:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            # Use trimesh to write — handles normals automatically
            tm_out = _trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            tm_out.export(cache_file)
        except Exception:
            pass  # Cache write failure is non-fatal

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
    """Load STL meshes for *selected_folders*.

    Respects ``viewer_max_faces`` from each folder's settings to keep the
    viewer fast on high-polygon meshes.  Decimated meshes are cached on disk
    in the viewer LOD cache directory (``geometry_cache/viewer_lod`` by
    default) so only the very first open is slow.
    """
    global _VIEWER_CACHE_DIR
    # Use a fixed viewer LOD cache directory next to the script
    if _VIEWER_CACHE_DIR is None:
        _VIEWER_CACHE_DIR = os.path.join(script_dir, "config", "geometry_cache", "viewer_lod")

    meshes = []
    ci = 0
    for folder in geometry_folders:
        color = _FOLDER_COLORS[ci % len(_FOLDER_COLORS)]
        ci += 1
        if folder not in selected_folders:
            continue
        settings = geometry_folders[folder]
        scale = settings.get("scale", 1)
        max_faces_raw = settings.get("viewer_max_faces", 50_000)
        max_faces = int(max_faces_raw) if max_faces_raw else None
        folder_abs = _resolve_viewer_path(script_dir, folder)
        if not os.path.isdir(folder_abs):
            continue
        for stl_path in sorted(
            glob.glob(os.path.join(folder_abs, "*.stl"))
            + glob.glob(os.path.join(folder_abs, "*.STL"))
        ):
            mesh = _load_stl(stl_path, scale, color, viewer_max_faces=max_faces)
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


def _apply_transform_to_df(df, translation_m, rotation_z_deg):
    """Apply Rz(theta) rotation + translation to positions and directions in *df*.

    Modifies CenterX/Y/Z and DirX/Y/Z columns in-place to convert from
    beam-local coordinates to the Tokamak global frame.
    """
    t = np.asarray(translation_m, dtype=np.float64)
    theta = np.deg2rad(rotation_z_deg)
    c, s = np.cos(theta), np.sin(theta)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    pos = df[['CenterX', 'CenterY', 'CenterZ']].to_numpy(dtype=np.float64)
    dirs = df[['DirX', 'DirY', 'DirZ']].to_numpy(dtype=np.float64)

    df[['CenterX', 'CenterY', 'CenterZ']] = pos @ Rz.T + t
    dir_rot = dirs @ Rz.T
    norms = np.linalg.norm(dir_rot, axis=1, keepdims=True)
    df[['DirX', 'DirY', 'DirZ']] = np.where(norms > 0, dir_rot / norms, dir_rot)


def _build_source_geometries(bl_paths, arrow_length=0.0,
                              show_direction=False, transforms=None):
    """Build Open3D geometries for particle sources.

    Source positions are rendered as a **PointCloud** (zero triangle
    overhead — just coloured dots).  Direction arrows, when requested,
    are built as a single merged TriangleMesh (cylinder + cone per
    source).

    Each source is coloured by its ``CurrentDensity_A_m2``.

    Args:
        bl_paths: list of .bl file paths.
        arrow_length: override arrow length in metres (0 = auto).
        show_direction: whether to render direction arrows.
        transforms: optional dict ``{bl_abs_path: {"translation_m": [...],
            "rotation_z_deg": float}}`` used to convert each file's
            beamlet positions/directions from beam-local coordinates to
            the Tokamak global frame before rendering.

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
        if df is None or df.empty:
            continue
        # Apply per-file coordinate transform if provided
        if transforms and bl in transforms:
            t_cfg = transforms[bl]
            t = t_cfg.get("translation_m", [0.0, 0.0, 0.0])
            r = t_cfg.get("rotation_z_deg", 0.0)
            if any(v != 0.0 for v in t) or r != 0.0:
                _apply_transform_to_df(df, t, r)
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

