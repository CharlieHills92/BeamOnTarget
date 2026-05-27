# geometry.py
"""
Handles loading, scaling, and refining of mesh geometry from folder-based definitions.
Includes a caching system to store refined meshes for faster startup.
"""
import trimesh
import os
import glob
from tqdm import tqdm


def load_scene(geometry_folders, cache_dir=None):
    grouped_meshes = {}

    print("\nLoading and processing geometry from folders...")

    # Single cache folder inside models
    if cache_dir is None:
        first_folder = next(iter(geometry_folders))
        cache_dir = os.path.join(os.path.dirname(first_folder), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Using geometry cache directory: '{cache_dir}'")

    for folder_path, settings in geometry_folders.items():
        if not os.path.isdir(folder_path):
            print(f"WARNING: Geometry folder not found: '{folder_path}'. Skipping.")
            continue

        scale = settings.get("scale", 1.0)
        target_length = settings.get("target_length", None)

        # Cache lives inside the geometry folder itself
        effective_cache_dir = os.path.join(folder_path, "cache") if cache_dir is None else cache_dir
        os.makedirs(effective_cache_dir, exist_ok=True)

        stl_files_in_folder = glob.glob(os.path.join(folder_path, '*.stl'))

        if not stl_files_in_folder:
            print(f"INFO: No .stl files found in folder '{folder_path}'.")
            continue

        print(f"Processing {len(stl_files_in_folder)} files from '{folder_path}'... (cache: '{effective_cache_dir}')")

        meshes_in_folder = []
        for f in tqdm(stl_files_in_folder, desc=f"Folder '{folder_path}'"):
            try:
                mesh = None
                basename = os.path.basename(f)

                # --- Caching Logic ---
                if target_length:
                    cache_filename = f"{os.path.splitext(basename)[0]}_L{target_length}_S{scale}.stl"
                    cache_path = os.path.join(effective_cache_dir, cache_filename)
                    if os.path.exists(cache_path):
                        mesh = trimesh.load_mesh(cache_path)

                # If mesh was not loaded from cache, do the full processing
                if mesh is None:
                    mesh = trimesh.load_mesh(f)

                    if scale != 1.0:
                        mesh.apply_scale(scale)

                    if target_length and target_length > 0:
                        mesh = mesh.subdivide_to_size(max_edge=target_length)

                    if target_length:
                        mesh.export(cache_path)

                mesh.metadata['name'] = basename
                meshes_in_folder.append(mesh)

            except Exception as e:
                print(f"\nError processing mesh '{f}': {e}. Skipping.")

        if meshes_in_folder:
            grouped_meshes[folder_path] = meshes_in_folder

    if not grouped_meshes:
        print("\nFATAL ERROR: No valid geometry was loaded from any folder. Exiting.")
        exit()

    num_total_objects = sum(len(v) for v in grouped_meshes.values())
    print(f"\nScene loaded: {num_total_objects} objects found in {len(grouped_meshes)} geometry groups.")

    return grouped_meshes