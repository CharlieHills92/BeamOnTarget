# output.py
"""Handles all output from the simulation, including visualization and data export."""
import pyvista as pv
import numpy as np
import pandas as pd
import os

# --- Configuration for Visualization Performance ---
# Target number of faces for simplified visualization meshes.
# Set to None to disable decimation and use the full-resolution mesh.
VISUALIZATION_DECIMATION_TARGET_FACES = 5000

def save_paraview_reports(original_meshes, deposited_power, object_names, save_flags,
                         output_directory, per_species_power=None):
    """
    Saves the results for each object as a VTK PolyData file (.vtp), which
    contains both the mesh and the associated data for easy loading in ParaView.
    This is the recommended method for saving detailed results.

    When per_species_power is provided (dict {charge_state: [array_per_object]}),
    additional cell data arrays are added for each species, e.g.
    Deposited_Power_W_H-, Power_Density_W_m2_H-.
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
        
        pv_mesh.field_data['source_filename'] = np.array([object_name])
        
        # 2. Calculate power density and attach BOTH power and density as data
        face_areas = mesh.area_faces
        power_density = np.divide(power_data, face_areas, out=np.zeros_like(power_data), where=face_areas > 0)
        
        pv_mesh.cell_data['Deposited_Power_W'] = power_data
        pv_mesh.cell_data['Power_Density_W_m2'] = power_density

        # 3. Add per-species arrays if available
        if per_species_power:
            for cs in sorted(per_species_power.keys()):
                label = _SPECIES_LABELS.get(cs, f"q{cs}")
                sp_power = per_species_power[cs][i]
                sp_density = np.divide(sp_power, face_areas,
                                       out=np.zeros_like(sp_power), where=face_areas > 0)
                pv_mesh.cell_data[f'Deposited_Power_W_{label}'] = sp_power
                pv_mesh.cell_data[f'Power_Density_W_m2_{label}'] = sp_density

        # Set the active scalar for visualization
        if 'Power_Density_W_m2' in pv_mesh.cell_data:
            pv_mesh.cell_data.active_scalars_name = 'Power_Density_W_m2'
        
        # 4. Save the PyVista mesh to a .vtp file
        sanitized_name = os.path.splitext(object_name)[0]
        output_filename = f"results_{sanitized_name}.vtp"
        full_output_path = os.path.join(output_directory, output_filename)
        
        pv_mesh.save(full_output_path, binary=True)
        print(f"  - Saved ParaView report for '{object_name}' to '{full_output_path}'")
            
    print("ParaView report generation complete.")



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

def save_summary_to_csv(original_meshes, deposited_power, object_names, outfile):
    """Write per-object summary to a CSV file."""
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    print(f"\nSaving object power summary to '{outfile}'...")
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
    df.to_csv(outfile, index=False)
    print("Summary save complete.")


def save_detailed_reports(original_meshes, deposited_power, object_names, save_flags, 
                          output_directory, save_binary=True, save_csv=True,
                          per_species_power=None):
    """Saves detailed reports as CSV and/or binary files.

    When per_species_power is provided, the CSV includes additional columns
    for each species: deposited_power_W_<label>, deposited_power_density_W_m2_<label>.
    """
    if not any(save_flags):
        print("\nNo detailed reports configured for saving.")
        return
    print(f"\nSaving detailed reports to folder: '{output_directory}'...")
    os.makedirs(output_directory, exist_ok=True)

    # Pre-sort species keys so column order is deterministic
    species_keys = sorted(per_species_power.keys()) if per_species_power else []

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
            face_areas, face_centers = mesh.area_faces, mesh.triangles_center
            hit_indices = np.where(power_data > 0)[0]
            object_face_data = []
            for face_idx in hit_indices:
                area, power = face_areas[face_idx], power_data[face_idx]
                density = power / area if area > 0 else 0.0
                center = face_centers[face_idx]
                row = {
                    'face_id': face_idx,
                    'deposited_power_W': f'{power:.3e}',
                    'deposited_power_density_W_m2': f'{density:.3e}',
                    'center_x': center[0], 'center_y': center[1], 'center_z': center[2],
                }
                for cs in species_keys:
                    label = _SPECIES_LABELS.get(cs, f"q{cs}")
                    sp_pwr = float(per_species_power[cs][i][face_idx])
                    sp_dens = sp_pwr / area if area > 0 else 0.0
                    row[f'deposited_power_W_{label}'] = f'{sp_pwr:.3e}'
                    row[f'deposited_power_density_W_m2_{label}'] = f'{sp_dens:.3e}'
                object_face_data.append(row)
            if object_face_data:
                df = pd.DataFrame(object_face_data)
                df.to_csv(full_csv_path, index=False)
    print("Detailed report generation complete.")


def save_impact_data_csv(impact_data, object_names, save_impact_flags, output_directory):
    """
    Saves per-particle impact data as CSV files for objects flagged with save_impact_data=True.

    Each CSV contains columns: source_index, mass_kg, charge_state, pos_x, pos_y, pos_z,
                                dir_x, dir_y, dir_z, kinetic_energy_eV, current_A.
    A header comment records the total number of impacts and how many were stored.

    Args:
        impact_data: list of dicts from the engine, one per object.
        object_names: list of object name strings.
        save_impact_flags: list of bools, True if impact data should be saved.
        output_directory: path to the output folder.
    """
    if not any(save_impact_flags):
        return

    os.makedirs(output_directory, exist_ok=True)
    print(f"\nSaving impact data CSV files to '{output_directory}'...")

    for i, name in enumerate(object_names):
        if not save_impact_flags[i]:
            continue

        d = impact_data[i]
        total_hits = d['total_hits']
        stored_hits = d['stored_hits']
        records = d['records']

        sanitized_name = os.path.splitext(name)[0]
        csv_path = os.path.join(output_directory, f"impact_data_{sanitized_name}.csv")

        columns = ['source_index', 'mass_kg', 'charge_state', 'pos_x', 'pos_y', 'pos_z',
                    'dir_x', 'dir_y', 'dir_z', 'kinetic_energy_eV', 'current_A']

        with open(csv_path, 'w') as f:
            f.write(f"# Impact data for: {name}\n")
            f.write(f"# Total particle impacts: {total_hits}\n")
            f.write(f"# Stored impact records:  {stored_hits}\n")
            if total_hits > 0:
                f.write(f"# Fraction stored: {stored_hits / total_hits:.4f}\n")
            else:
                f.write(f"# Fraction stored: N/A (no impacts)\n")

        if records:
            df = pd.DataFrame(records, columns=columns)
            df['source_index'] = df['source_index'].astype(int)
            df['charge_state'] = df['charge_state'].astype(int)
            df.to_csv(csv_path, index=False, mode='a')
        else:
            # Write header-only CSV so downstream tools see the columns
            pd.DataFrame(columns=columns).to_csv(csv_path, index=False, mode='a')

        print(f"  - {name}: {stored_hits}/{total_hits} impacts saved to '{csv_path}'")

    print("Impact data export complete.")


_SPECIES_LABELS = {-1: "H-", 0: "H0", 1: "H+"}