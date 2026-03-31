# geometry.py
"""
Handles loading, scaling, and refining of mesh geometry from folder-based
definitions.  Includes a caching system for refined meshes.
"""
import trimesh
import numpy as np
import os
import glob
from tqdm import tqdm


def load_scene(geometry_folders, cache_dir=None):
    """
    Load STL geometry from the specified folder dict, apply per-group scaling
    and optional mesh refinement, and return grouped meshes.

    Args:
        geometry_folders: dict  {folder_path: {scale, target_length, ...}}
        cache_dir: optional path for storing refined-mesh caches.

    Returns:
        dict  {folder_name: [trimesh objects]}
    """
    grouped_meshes = {}
    print("\nLoading and processing geometry from folders...")

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        print(f"  Using geometry cache: '{cache_dir}'")

    for folder_path, settings in geometry_folders.items():
        if not os.path.isdir(folder_path):
            print(f"  WARNING: folder '{folder_path}' not found — skipping.")
            continue

        scale = settings.get("scale", 1.0)
        target_length = settings.get("target_length", None)

        stl_files = sorted(
            glob.glob(os.path.join(folder_path, "*.stl")) +
            glob.glob(os.path.join(folder_path, "*.STL")))

        if not stl_files:
            print(f"  INFO: no STL files in '{folder_path}'.")
            continue

        print(f"  Processing {len(stl_files)} files from '{folder_path}'...")
        meshes = []
        for f in tqdm(stl_files, desc=f"  '{folder_path}'"):
            try:
                mesh = None
                basename = os.path.basename(f)

                # -- cache lookup --
                if cache_dir and target_length:
                    cache_fn = (f"{os.path.splitext(basename)[0]}"
                                f"_L{target_length}_S{scale}.stl")
                    cache_path = os.path.join(cache_dir, cache_fn)
                    if os.path.exists(cache_path):
                        mesh = trimesh.load_mesh(cache_path)

                # -- full processing --
                if mesh is None:
                    mesh = trimesh.load_mesh(f)
                    if scale != 1.0:
                        mesh.apply_scale(scale)
                    if target_length and target_length > 0:
                        mesh = mesh.subdivide_to_size(max_edge=target_length)
                    if cache_dir and target_length:
                        mesh.export(cache_path)

                mesh.metadata['name'] = basename
                meshes.append(mesh)
            except Exception as exc:
                print(f"\n  Error processing '{f}': {exc} — skipping.")

        if meshes:
            grouped_meshes[folder_path] = meshes

    if not grouped_meshes:
        print("\nFATAL: no valid geometry loaded. Exiting.")
        raise SystemExit(1)

    n_obj = sum(len(v) for v in grouped_meshes.values())
    print(f"\nScene loaded: {n_obj} objects in {len(grouped_meshes)} groups.")
    return grouped_meshes
