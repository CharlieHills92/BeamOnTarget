# run_simulation.py
"""
Main entry point for the particle-mesh interaction simulation.

This script can run a batch of simulations, one for each beamlet
configuration file (.bl) found in the specified directory. Each run's
results are saved to a dedicated, named subfolder.

Can also be run in a setup preview mode:
  --view-setup          (Shows geometry and particle sources from the first .bl file)
  --view-setup geo      (Shows geometry only)
"""
import argparse
import trimesh
import numpy as np
import os
import glob
import config
import geometry
import particles
import engine
import output
import batch_smoother # <-- NEW: Import the batch smoother script

def run_full_simulation(grouped_meshes, particle_source_file, output_subfolder):
    """
    The main simulation workflow for a SINGLE run.
    It takes a specific particle file and an output subfolder as arguments.
    """
    if particle_source_file:
        print(f"\n--- Starting Simulation for Beam Config: '{os.path.basename(particle_source_file)}' ---")
    else:
        print(f"\n--- Starting Simulation using fallback particle sources ---")
    
    # --- Flatten the grouped geometry for the engine ---
    original_meshes, object_names, save_details_flags = [], [], []
    is_diagnostic_flags_per_mesh = []
    for folder_path, mesh_list in grouped_meshes.items():
        settings = config.GEOMETRY_FOLDERS.get(folder_path, {})
        save_flag = settings.get("save_details", False)
        is_diagnostic = settings.get("is_diagnostic", False)
        for mesh in mesh_list:
            original_meshes.append(mesh)
            object_names.append(mesh.metadata['name'])
            save_details_flags.append(save_flag)
            is_diagnostic_flags_per_mesh.append(is_diagnostic)
            
    face_counts = [len(m.faces) for m in original_meshes]
    scene_mesh = trimesh.util.concatenate(original_meshes)
    face_offsets = np.cumsum([0] + face_counts[:-1])

    # Load Particle Sources from the SPECIFIED file
    if particle_source_file:
        particle_sources_list = particles.load_beamlets_from_file(
            filename=particle_source_file,
            num_particles_per_beamlet=config.NUM_PARTICLES_PER_BEAMLET,
            beamlet_area=config.BEAMLET_AREA_FOR_CURRENT)
    else: # Fallback
        particle_sources_list = config.PARTICLE_SOURCES

    if not particle_sources_list:
        print(f"Error: No particle sources loaded or defined for this run. Skipping.")
        return

    # --- Engine Selection Logic ---
    if config.ENABLE_DIAGNOSTIC_SURFACES:
        # ... (logic for multi-hit engine) ...
        is_diagnostic_face = np.zeros(len(scene_mesh.faces), dtype=bool)
        for i, is_diagnostic in enumerate(is_diagnostic_flags_per_mesh):
            if is_diagnostic:
                start_index, end_index = face_offsets[i], face_offsets[i] + face_counts[i]
                is_diagnostic_face[start_index:end_index] = True
        
        # The multi-hit engine should also be updated to only return power
        deposited_power = engine.run_simulation_multi_hit_parallel(
            scene_mesh, face_offsets, face_counts, particle_sources_list, config.get_deposition_fraction,
            is_diagnostic_face, config.NUM_CPU_CORES)
    else:
        # --- THIS IS THE CORRECTED CALL ---
        # The engine now only returns the final power deposition array.
        deposited_power = engine.run_simulation_single_hit(
            scene_mesh, face_offsets, face_counts, particle_sources_list, config.get_deposition_fraction,
            sources_per_worker=config.SOURCES_PER_WORKER,
            num_cpu_cores=config.NUM_CPU_CORES)

    # --- Handle Outputs, saving to the specified subfolder ---
    output_dir_for_run = os.path.join(config.DETAILED_OUTPUT_DIR, output_subfolder)
    
    if config.SAVE_PARAVIEW_FILES and any(save_details_flags):
        output.save_paraview_reports(original_meshes, deposited_power, object_names, save_details_flags, output_dir_for_run)
    if (config.SAVE_BINARY_POWERLOADS or config.SAVE_CSV_REPORTS) and any(save_details_flags):
        output.save_detailed_reports(original_meshes, deposited_power, object_names, save_details_flags, output_dir_for_run,
            save_binary=config.SAVE_BINARY_POWERLOADS, save_csv=config.SAVE_CSV_REPORTS)
    if config.SUMMARY_CSV_FILENAME:
        # We can also put the summary in the subfolder to keep results together
        summary_filename = f"summary_{output_subfolder}.csv"
        summary_path = os.path.join(output_dir_for_run, summary_filename)
        output.save_summary_to_csv(original_meshes, deposited_power, object_names, summary_path)
    # Automatic visualization is not supported in this memory-safe workflow
    if config.RUN_VISUALIZATION_AFTER_SIM:
        print("\nWARNING: Automatic visualization is disabled in the memory-safe workflow.")
        print("         Please use post_process.py to view results after the run completes.")
        
    print(f"\n--- Finished Simulation for: '{os.path.basename(particle_source_file) if particle_source_file else 'fallback_run'}' ---")

    # --- NEW: Automatic call to the batch smoother ---
    if config.RUN_SMOOTHER_AFTER_SIM:
        print("\n--- Auto-running Batch Smoother ---")
        try:
            batch_smoother.batch_process_directory(output_dir_for_run)
        except Exception as e:
            print(f"An error occurred during automatic batch smoothing: {e}")
        print("--- Batch Smoothing Finished ---")

def run_setup_preview(grouped_meshes, view_mode):
    """Shows a 3D plot of the setup based on the view_mode."""
    print(f"--- Running Setup Preview Mode (Mode: {view_mode}) ---")
    
    # For preview, we need to pick a beam file to show. Let's pick the first one found.
    particle_source_file_for_preview = None
    if config.PARTICLE_SOURCE_DIR:
        bl_files = sorted(glob.glob(os.path.join(config.PARTICLE_SOURCE_DIR, '*.bl')))
        if bl_files:
            particle_source_file_for_preview = bl_files[0]
            
    show_sources = (view_mode != 'geo')
    particle_sources_list = []
    if show_sources:
        if particle_source_file_for_preview:
            print(f"Showing setup preview using beam file: {os.path.basename(particle_source_file_for_preview)}")
            particle_sources_list = particles.load_beamlets_from_file(
                filename=particle_source_file_for_preview, num_particles_per_beamlet=config.NUM_PARTICLES_PER_BEAMLET,
                beamlet_area=config.BEAMLET_AREA_FOR_CURRENT)
        else:
            print("Warning: No .bl files found in PARTICLE_SOURCE_DIR to show in preview.")
            
    output.visualize_setup(
        grouped_meshes=grouped_meshes, particle_sources=particle_sources_list,
        geometry_folders_config=config.GEOMETRY_FOLDERS, show_sources=show_sources)
    print("\nSetup preview finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a particle-mesh interaction simulation.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '--view-setup', nargs='?', const='full', default=None,
        choices=['geo', 'full'],
        help="Display a 3D preview of the setup.\n"
             "  'geo': Show geometry only.\n"
             "  'full': Show geometry and particle sources.\n"
             "  (default if flag is used with no value: 'full')")
    args = parser.parse_args()
    
    # Load geometry ONCE, as it's shared by all runs.
    print("--- Loading shared geometry for all simulation runs... ---")
    grouped_geometry = geometry.load_scene(
        geometry_folders=config.GEOMETRY_FOLDERS,
        cache_dir=config.GEOMETRY_CACHE_DIR
    )
    
    if args.view_setup:
        run_setup_preview(grouped_geometry, view_mode=args.view_setup)
    else:
        # --- Batch Simulation Loop ---
        if config.PARTICLE_SOURCE_DIR:
            search_path = os.path.join(config.PARTICLE_SOURCE_DIR, '*.bl')
            beam_config_files = sorted(glob.glob(search_path))
            
            if not beam_config_files:
                print(f"Error: No .bl files found in the specified directory: '{config.PARTICLE_SOURCE_DIR}'")
            else:
                print(f"\nFound {len(beam_config_files)} beam configurations to simulate.")
                for beam_file in beam_config_files:
                    # Create a unique subfolder name from the beam file's name
                    subfolder_name = os.path.splitext(os.path.basename(beam_file))[0]
                    # Run the entire simulation for this one beam file
                    run_full_simulation(grouped_geometry, beam_file, subfolder_name)
        else:
            print("\nPARTICLE_SOURCE_DIR not specified. Attempting single fallback run...")
            if config.PARTICLE_SOURCES:
                 run_full_simulation(grouped_geometry, None, "fallback_run")
            else:
                 print("No particle sources defined for fallback run.")