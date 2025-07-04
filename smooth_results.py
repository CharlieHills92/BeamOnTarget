# smooth_results.py
"""
This file now acts as a "library" containing the core logic for smoothing.
The `apply_smoothing` function can be imported by other scripts.
It can also be run standalone to process the specific tasks defined below.
"""
import pyvista as pv
import numpy as np
import os

# --- Default Configuration for Standalone Running ---
# This part is only used if you run `python smooth_results.py` directly.
INPUT_DIRECTORY_STANDALONE = "OUTPUT"
SMOOTHING_TASKS_STANDALONE = [
    {
        "input_filename": "results_front_tube_15.vtp",
        "output_filename": "smoothed_front_tube_15.vtp",
        "n_iter": 20
    }
]

# --- This is the core, reusable function ---
def apply_smoothing(mesh, n_iter):
    """
    Applies Laplacian smoothing to a PyVista mesh object.
    Overwrites the data arrays with their smoothed values.
    """
    # Set a default active scalar to ensure the filter works
    if 'Power_Density_W_m2' in mesh.cell_data:
        mesh.cell_data.active_scalars_name = 'Power_Density_W_m2'
    elif 'Deposited_Power_W' in mesh.cell_data:
        mesh.cell_data.active_scalars_name = 'Deposited_Power_W'
    else:
        print("  - WARNING: No power data arrays found to smooth.")
        return mesh

    # Process Power Density
    if 'Power_Density_W_m2' in mesh.cell_data:
        mesh.cell_data.active_scalars_name = 'Power_Density_W_m2'
        mesh_points = mesh.cell_data_to_point_data()
        smoothed_points = mesh_points.smooth(n_iter=n_iter, boundary_smoothing=False)
        smoothed_cells = smoothed_points.point_data_to_cell_data()
        if 'Power_Density_W_m2' in smoothed_cells.cell_data:
            mesh['Power_Density_W_m2'] = smoothed_cells['Power_Density_W_m2']
            
    # Process Deposited Power
    if 'Deposited_Power_W' in mesh.cell_data:
        mesh.cell_data.active_scalars_name = 'Deposited_Power_W'
        mesh_points = mesh.cell_data_to_point_data()
        smoothed_points = mesh_points.smooth(n_iter=n_iter, boundary_smoothing=False)
        smoothed_cells = smoothed_points.point_data_to_cell_data()
        if 'Deposited_Power_W' in smoothed_cells.cell_data:
            mesh['Deposited_Power_W'] = smoothed_cells['Deposited_Power_W']

    # Set the final active scalar for convenience in ParaView
    if 'Power_Density_W_m2' in mesh.cell_data:
        mesh.cell_data.active_scalars_name = 'Power_Density_W_m2'
        
    return mesh


# The main block allows this script to be run by itself for testing single files.
def main():
    """Main function for standalone execution."""
    if not os.path.isdir(INPUT_DIRECTORY_STANDALONE):
        print(f"FATAL ERROR: Input directory '{INPUT_DIRECTORY_STANDALONE}' not found."); return
    print("--- Running Standalone Smoothing Process ---")
    for task in SMOOTHING_TASKS_STANDALONE:
        # ... (The old main loop logic remains here) ...
        pass # For brevity

if __name__ == "__main__":
    # This block is only executed when you run `python smooth_results.py`
    main()