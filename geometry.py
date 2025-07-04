# geometry.py
"""
Handles loading, scaling, and refining of mesh geometry from folder-based definitions.
Includes a caching system to store refined meshes for faster startup.
"""
import trimesh
import numpy as np
import os
import glob
from tqdm import tqdm

def create_dummy_meshes():
    """Creates coarse dummy meshes in the 'DUMMY' subfolder if they don't exist."""
    dummy_dir = "DUMMY"
    if not os.path.exists(dummy_dir):
        os.makedirs(dummy_dir)
        
    cube_path = os.path.join(dummy_dir, "cube.stl")
    if not os.path.exists(cube_path):
        print("Creating coarse dummy mesh 'DUMMY/cube.stl'...")
        mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        mesh.apply_translation([0, 0, 1.5])
        mesh.export(cube_path)

    sphere_path = os.path.join(dummy_dir, "sphere.stl")
    if not os.path.exists(sphere_path):
        print("Creating coarse dummy mesh 'DUMMY/sphere.stl'...")
        mesh = trimesh.creation.icosphere(subdivisions=2)
        mesh.apply_translation([0, 0, -1.5])
        mesh.export(sphere_path)

def load_scene(geometry_folders, cache_dir=None):
    """
    Loads geometry from specified folders, applies group settings for scaling
    and refinement, and uses a cache to speed up loading of already-processed meshes.
    Returns the meshes grouped by folder name.
    
    Args:
        geometry_folders (dict): Configuration dictionary for geometry folders.
        cache_dir (str, optional): Path to the directory for storing/loading cached
                                   refined meshes. Defaults to None (caching disabled).
    
    Returns:
        dict: A dictionary of {folder_name: [list_of_trimesh_objects]}.
    """
    create_dummy_meshes() # Ensures dummy files/folder exist if needed

    # The main data structure is a dictionary, e.g.:
    # {'NEU': [mesh1, mesh2], 'RID': [mesh3, mesh4]}
    grouped_meshes = {}

    print("\nLoading and processing geometry from folders...")
    
    # Create the cache directory if it doesn't exist and caching is enabled
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        print(f"Using geometry cache directory: '{cache_dir}'")

    # Iterate through the high-level folder definitions from the config
    for folder_path, settings in geometry_folders.items():
        if not os.path.isdir(folder_path):
            print(f"WARNING: Geometry folder not found: '{folder_path}'. Skipping.")
            continue
            
        scale = settings.get("scale", 1.0)
        target_length = settings.get("target_length", None)
        
        # Find all .stl files inside this specific folder
        search_path = os.path.join(folder_path, '*.stl')
        stl_files_in_folder = glob.glob(search_path)
        
        if not stl_files_in_folder:
            print(f"INFO: No .stl files found in folder '{folder_path}'.")
            continue

        print(f"Processing {len(stl_files_in_folder)} files from '{folder_path}'...")
        
        meshes_in_folder = []
        for f in tqdm(stl_files_in_folder, desc=f"Folder '{folder_path}'"):
            try:
                mesh = None
                basename = os.path.basename(f)
                
                # --- Caching Logic ---
                if cache_dir:
                    # Create a unique filename based on the original name, scale, and target length.
                    # This ensures that if you change parameters, a new cache file is generated.
                    # Note: We only cache refined meshes, as unrefined meshes load fast anyway.
                    cache_filename = ""
                    if target_length:
                        cache_filename = f"{os.path.splitext(basename)[0]}_L{target_length}_S{scale}.stl"
                    
                    if cache_filename:
                        cache_path = os.path.join(cache_dir, cache_filename)
                        if os.path.exists(cache_path):
                            # Load the already-refined mesh directly from the cache
                            mesh = trimesh.load_mesh(cache_path)
                
                # If mesh was not loaded from cache, do the full processing
                if mesh is None:
                    mesh = trimesh.load_mesh(f)
                    
                    # Apply scaling before refinement
                    if scale != 1.0:
                        mesh.apply_scale(scale)
                    
                    # Apply global refinement if specified
                    if target_length and target_length > 0:
                        mesh = mesh.subdivide_to_size(max_edge=target_length)
                    
                    # If caching is enabled and we refined the mesh, save it to the cache
                    if cache_dir and target_length:
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