# output.py
"""Handles all output from the simulation, including visualization and data export."""
import pyvista as pv
import numpy as np
import pandas as pd
import os
import glob
import trimesh
from scipy.spatial import KDTree

# --- Configuration for Visualization Performance ---
# Target number of faces for simplified visualization meshes.
# Set to None to disable decimation and use the full-resolution mesh.
VISUALIZATION_DECIMATION_TARGET_FACES = 5000

def save_paraview_reports(original_meshes, deposited_power, object_names, save_flags, output_directory):
    """
    Saves the results for each object as a VTK PolyData file (.vtp), which
    contains both the mesh and the associated data for easy loading in ParaView.
    This is the recommended method for saving detailed results.
    """
    if not any(save_flags):
        print("\nNo objects configured for detailed saving.")
        return
        
    print(f"\nSaving ParaView (.vtp) reports to folder: '{output_directory}'...")
    os.makedirs(output_directory, exist_ok=True)
    
    for i, mesh in enumerate(original_meshes):
        if not save_flags[i]:
            continue
            
        object_name = object_names[i]
        power_data = deposited_power[i]
        
        # 1. Convert the trimesh object to a PyVista mesh
        pv_mesh = pv.wrap(mesh)
        
        # --- NEW: Store the original filename in the VTP file's field data ---
        pv_mesh.field_data['source_filename'] = np.array([object_name])
        
        # 2. Calculate power density and attach BOTH power and density as data
        face_areas = mesh.area_faces
        power_density = np.divide(power_data, face_areas, out=np.zeros_like(power_data), where=face_areas > 0)
        
        # Use names compatible with ParaView (no spaces or special chars)
        pv_mesh.cell_data['Deposited_Power_W'] = power_data
        pv_mesh.cell_data['Power_Density_W_m2'] = power_density

        # --- NEW: Set the active scalar for visualization ---
        # This tells ParaView which data array to show by default.
        if 'Power_Density_W_m2' in pv_mesh.cell_data:
            pv_mesh.cell_data.active_scalars_name = 'Power_Density_W_m2'
        
        # 3. Save the PyVista mesh to a .vtp file
        sanitized_name = os.path.splitext(object_name)[0]
        output_filename = f"results_{sanitized_name}.vtp"
        full_output_path = os.path.join(output_directory, output_filename)
        
        pv_mesh.save(full_output_path, binary=True)
        print(f"  - Saved ParaView report for '{object_name}' to '{full_output_path}'")
            
    print("ParaView report generation complete.")


# def save_paraview_reports(original_meshes, deposited_power, object_names, save_flags, output_directory):
#     """
#     Saves the final results to .vtp files, ensuring data is correctly mapped
#     to a cleaned version of the geometry to prevent corruption.
#     """
#     if not any(save_flags):
#         print("\nNo objects configured for detailed saving.")
#         return
        
#     print(f"\nSaving ParaView (.vtp) reports to folder: '{output_directory}'...")
#     os.makedirs(output_directory, exist_ok=True)
    
#     for i, original_mesh_trimesh in enumerate(original_meshes):
#         if not save_flags[i]:
#             continue
            
#         object_name = object_names[i]
#         power_data_original = deposited_power[i]
        
#         # --- NEW, ROBUST DATA MAPPING AND SAVING PROCESS ---
        
#         # 1. Create a cleaned version of the geometry.
#         #    We start from the trimesh object for consistency.
#         cleaned_mesh_trimesh = original_mesh_trimesh.copy()
#         # You can use trimesh's processing tools for cleaning if needed
#         # e.g., cleaned_mesh_trimesh.remove_degenerate_faces()
#         # For now, we'll convert to pyvista and clean there.
#         pv_mesh_cleaned = pv.wrap(cleaned_mesh_trimesh)
#         pv_mesh_cleaned.clean(inplace=True)
        
#         # 2. Check if the mesh is still valid.
#         if pv_mesh_cleaned.n_cells == 0:
#             print(f"  - WARNING: Mesh for '{object_name}' became empty after cleaning. Skipping file save.")
#             continue

#         # 3. MAP the original data to the cleaned mesh using a KD-Tree.
#         #    This finds the closest original face for each face in the new, cleaned mesh.
#         print(f"  - Mapping data from original ({len(original_mesh_trimesh.faces)} faces) to cleaned ({pv_mesh_cleaned.n_cells} faces) mesh for '{object_name}'...")
#         tree = KDTree(original_mesh_trimesh.triangles_center)
#         _, closest_face_indices = tree.query(pv_mesh_cleaned.cell_centers().points)
        
#         # 4. Create new data arrays by sampling from the original data.
#         power_data_mapped = power_data_original[closest_face_indices]
        
#         # 5. Attach the NEW, correctly-sized data arrays to the CLEANED mesh.
#         face_areas_cleaned = pv_mesh_cleaned.area
#         power_density_mapped = np.divide(power_data_mapped, face_areas_cleaned, out=np.zeros_like(power_data_mapped), where=face_areas_cleaned > 0)
        
#         pv_mesh_cleaned.cell_data['Deposited_Power_W'] = power_data_mapped
#         pv_mesh_cleaned.cell_data['Power_Density_W_m2'] = power_density_mapped
#         pv_mesh_cleaned.field_data['source_filename'] = np.array([object_name])
        
#         # 6. Save the clean mesh with its correctly mapped data.
#         sanitized_name = os.path.splitext(object_name)[0]
#         output_filename = f"results_{sanitized_name}.vtp"
#         full_output_path = os.path.join(output_directory, output_filename)
        
#         pv_mesh_cleaned.save(full_output_path, binary=True)
#         print(f"  - Saved clean ParaView report for '{object_name}' to '{full_output_path}'")
            
#     print("ParaView report generation complete.")


def visualize_setup(grouped_meshes, particle_sources, geometry_folders_config, show_sources=True):
    """
    Creates a 3D plot of the initial setup, only showing geometry groups
    that are flagged for visualization.
    """
    print("\nGenerating setup visualization...")
    plotter = pv.Plotter(window_size=[1200, 900], notebook=False)

    print(f"  - Plotting geometry groups marked for visualization...")
    colors = pv.Color("blue"), pv.Color("red"), pv.Color("green"), pv.Color("purple"), pv.Color("orange")
    
    color_index = 0
    for folder_name, mesh_list in grouped_meshes.items():
        settings = geometry_folders_config.get(folder_name, {})
        if not settings.get("show_in_plot", True):
            print(f"    - Skipping visualization of group '{folder_name}' as configured.")
            continue
        is_diagnostic = settings.get("is_diagnostic", False)
        opacity = 0.2 if is_diagnostic else 0.6
        simplified_mesh_list = []
        for mesh in mesh_list:
            if VISUALIZATION_DECIMATION_TARGET_FACES and len(mesh.faces) > VISUALIZATION_DECIMATION_TARGET_FACES:
                print(f"    - Simplifying '{mesh.metadata['name']}' for preview: {len(mesh.faces)} -> {VISUALIZATION_DECIMATION_TARGET_FACES} faces")
                simplified_mesh = mesh.simplify_quadric_decimation(face_count=VISUALIZATION_DECIMATION_TARGET_FACES)
                simplified_mesh_list.append(simplified_mesh)
            else:
                simplified_mesh_list.append(mesh)
        folder_block = pv.MultiBlock([pv.wrap(m) for m in simplified_mesh_list])
        plotter.add_mesh(folder_block, color=colors[color_index % len(colors)], opacity=opacity, label=folder_name)
        color_index += 1
    if show_sources and particle_sources:
        print(f"  - Plotting {len(particle_sources)} particle sources...")
        for source in particle_sources:
            center, direction = source.get_visualization_repr()
            arrow = pv.Arrow(start=center, direction=direction, scale='auto')
            plotter.add_mesh(arrow, color='yellow', line_width=5)
    plotter.add_legend(); plotter.add_axes(); plotter.enable_parallel_projection()
    plotter.show_bounds(grid='front', location='outer', all_edges=True)
    print("Showing interactive setup plot. Close the window to continue.")
    plotter.show()

def visualize_scene(original_meshes, deposited_power, object_names, geometry_folders_config):
    """
    Creates an interactive plot of results. DOES NOT PLOT RAYS.
    """
    print("\nGenerating results visualization...")
    plotter = pv.Plotter(window_size=[1200, 900], notebook=False)
    
    scene_for_plotting, all_power_densities = pv.MultiBlock(), []
    file_settings_map = {}
    for folder, settings in geometry_folders_config.items():
        search_path = os.path.join(folder, '*.stl')
        for f in glob.glob(search_path): file_settings_map[os.path.basename(f)] = settings
    for i, full_mesh in enumerate(original_meshes):
        settings = file_settings_map.get(object_names[i], {})
        if not settings.get("show_in_plot", True): continue
        vis_mesh_trimesh = full_mesh
        if VISUALIZATION_DECIMATION_TARGET_FACES and len(full_mesh.faces) > VISUALIZATION_DECIMATION_TARGET_FACES:
            vis_mesh_trimesh = full_mesh.simplify_quadric_decimation(face_count=VISUALIZATION_DECIMATION_TARGET_FACES)
        pv_mesh = pv.wrap(vis_mesh_trimesh)
        tree = KDTree(full_mesh.triangles_center); _, closest_face_indices = tree.query(vis_mesh_trimesh.triangles_center)
        resampled_power, resampled_areas = deposited_power[i][closest_face_indices], vis_mesh_trimesh.area_faces
        power_density = np.divide(resampled_power, resampled_areas, out=np.zeros(len(resampled_areas)), where=resampled_areas > 0)
        all_power_densities.append(power_density); pv_mesh.cell_data['Power_Density_W_m2'] = power_density
        scene_for_plotting.append(pv_mesh)
    if all_power_densities: global_max = np.max([np.max(d) for d in all_power_densities if d.size > 0] or [1.0])
    else: global_max = 1.0
    if global_max == 0: global_max = 1.0
    if scene_for_plotting.n_blocks > 0:
        plotter.add_mesh(scene_for_plotting, scalars='Power_Density_W_m2', cmap='inferno', clim=[0, global_max], scalar_bar_args={'title': 'Power Density (W/m^2)'})

    rays_were_plotted = False
    if visualize_all_rays:
        if len(all_ray_origins) > 0:
            rays_were_plotted = True
            print("  - Visualizing a sample of ALL generated rays (hits and misses)...")
            scene_mesh_for_viz = trimesh.util.concatenate(original_meshes)
            intersector_viz = trimesh.ray.ray_pyembree.RayMeshIntersector(scene_mesh_for_viz)
            locations_viz, index_ray_viz, _ = intersector_viz.intersects_location(ray_origins=all_ray_origins, ray_directions=all_ray_directions, multiple_hits=False)
            ray_length = np.linalg.norm(scene_mesh_for_viz.bounding_box.extents); endpoints = all_ray_origins + all_ray_directions * ray_length; endpoints[index_ray_viz] = locations_viz
            hit_mask = np.zeros(len(all_ray_origins), dtype=bool); hit_mask[index_ray_viz] = True
            if np.any(~hit_mask):
                points = np.c_[all_ray_origins[~hit_mask], endpoints[~hit_mask]].reshape(-1, 3); n_lines = len(points)//2; lines = np.c_[np.full(n_lines, 2), np.arange(n_lines*2).reshape(-1, 2)]; plotter.add_mesh(pv.PolyData(points, lines=lines), color='red', line_width=1, label='Missed Rays')
            if np.any(hit_mask):
                points = np.c_[all_ray_origins[hit_mask], endpoints[hit_mask]].reshape(-1, 3); n_lines = len(points)//2; lines = np.c_[np.full(n_lines, 2), np.arange(n_lines*2).reshape(-1, 2)]; plotter.add_mesh(pv.PolyData(points, lines=lines), color='cyan', line_width=1, label='Hitting Rays')
    else:
        if len(hit_ray_origins) > 0:
            rays_were_plotted = True
            print("  - Visualizing a random sample of ONLY hitting rays...");
            points = np.c_[hit_ray_origins, hit_ray_endpoints].reshape(-1, 3); n_lines = len(hit_ray_origins); lines_array = np.c_[np.full(n_lines, 2), np.arange(n_lines * 2).reshape(-1, 2)]
            plotter.add_mesh(pv.PolyData(points, lines=lines_array), color='cyan', line_width=1, label='Hitting Rays')
    if rays_were_plotted:
        plotter.add_legend()
    plotter.enable_parallel_projection(); plotter.add_axes(); plotter.camera_position = 'xy'
    plotter.show_bounds(grid='front', location='outer', all_edges=True);
    print("Showing interactive plot..."); plotter.show()

def save_summary_to_csv(original_meshes, deposited_power, object_names, filename, outputdirectory="."):
    """
    Saves a summary of total power and peak power density for each object.
    This version filters invalid data to ensure consistency with analysis scripts.
    """
    print(f"\nSaving object power summary to '{filename}'...")
    summary_data = []
    
    for i, mesh in enumerate(original_meshes):
        power_array = deposited_power[i]
        
        # --- THIS IS THE KEY FIX ---
        # Calculate power density on the full, original mesh data
        face_areas = mesh.area_faces
        power_density = np.divide(power_array, face_areas, out=np.full_like(power_array, np.nan), where=face_areas > 0)
        
        # Now, create a mask of valid, finite density values
        valid_mask = np.isfinite(power_density)
        
        # Calculate stats using ONLY the valid data
        if np.any(valid_mask):
            peak_density = np.max(power_density[valid_mask])
        else:
            peak_density = 0.0 # No valid data to find a peak from
            
        # Total power is the sum of all power, regardless of density calculation
        total_power = np.sum(power_array)
        
        summary_data.append({
            'object_name': object_names[i],
            'total_deposited_power_W': total_power,
            'peak_power_density_W_m2': peak_density
        })
        
    df = pd.DataFrame(summary_data)
    df['total_deposited_power_W'] = df['total_deposited_power_W'].apply(lambda x: f'{x:.4e}')
    df['peak_power_density_W_m2'] = df['peak_power_density_W_m2'].apply(lambda x: f'{x:.4e}')
    outfile = os.path.join(outputdirectory, filename)
    df.to_csv(outfile, index=False)
    print("Summary save complete.")
# def save_summary_to_csv(original_meshes, deposited_power, object_names, filename):
#     """Saves a high-level summary of total power and peak power density."""
#     print(f"\nSaving object power summary to '{filename}'...")
#     summary_data = []
#     for i, mesh in enumerate(original_meshes):
#         power_array, total_power = deposited_power[i], np.sum(deposited_power[i])
#         face_areas = mesh.area_faces
#         power_density = np.divide(power_array, face_areas, out=np.zeros_like(power_array), where=face_areas > 0)
#         peak_density = np.max(power_density)
#         summary_data.append({'object_name': object_names[i], 'total_deposited_power_W': total_power, 'peak_power_density_W_m2': peak_density})
#     df = pd.DataFrame(summary_data)
#     df['total_deposited_power_W'] = df['total_deposited_power_W'].apply(lambda x: f'{x:.3e}')
#     df['peak_power_density_W_m2'] = df['peak_power_density_W_m2'].apply(lambda x: f'{x:.3e}')
#     df.to_csv(filename, index=False)
#     print("Summary save complete.")


def save_detailed_reports(original_meshes, deposited_power, object_names, save_flags, 
                          output_directory, save_binary=True, save_csv=True):
    """Saves detailed reports as CSV and/or binary files."""
    if not any(save_flags):
        print("\nNo detailed reports configured for saving.")
        return
    print(f"\nSaving detailed reports to folder: '{output_directory}'...")
    os.makedirs(output_directory, exist_ok=True)
    for i, mesh in enumerate(original_meshes):
        if not save_flags[i]: continue
        object_name, power_data = object_names[i], deposited_power[i]
        sanitized_name = os.path.splitext(object_name)[0]
        if save_binary:
            binary_filename = f"powerload_{sanitized_name}.npy"
            full_binary_path = os.path.join(output_directory, binary_filename)
            np.save(full_binary_path, power_data)
        if save_csv and np.sum(power_data) > 0:
            csv_filename = f"power_density_{sanitized_name}.csv"
            full_csv_path = os.path.join(output_directory, csv_filename)
            object_face_data, face_areas, face_centers = [], mesh.area_faces, mesh.triangles_center
            hit_indices = np.where(power_data > 0)[0]
            for face_idx in hit_indices:
                area, power = face_areas[face_idx], power_data[face_idx]
                density = power / area if area > 0 else 0.0
                center = face_centers[face_idx]
                object_face_data.append({'face_id': face_idx, 'deposited_power_W': power, 'deposited_power_density_W_m2': density, 'center_x': center[0], 'center_y': center[1], 'center_z': center[2]})
            if object_face_data:
                df = pd.DataFrame(object_face_data)
                df['deposited_power_W'] = df['deposited_power_W'].apply(lambda x: f'{x:.3e}')
                df['deposited_power_density_W_m2'] = df['deposited_power_density_W_m2'].apply(lambda x: f'{x:.3e}')
                df.to_csv(full_csv_path, index=False)
    print("Detailed report generation complete.")