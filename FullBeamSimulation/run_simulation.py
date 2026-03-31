#!/usr/bin/env python3
# run_simulation.py
"""
Main entry point for FullBeamSimulation.

Supports two tracking engines selectable via config.json:
  - "raytrace"   : infinite-ray Embree cast (fastest, no EM fields)
  - "boris"      : stepped Boris integrator in E + B fields

Boris engine options (all in config.json):
  - BORIS_RELATIVISTIC          : use relativistic Boris push
  - BORIS_NULLCOLL_ENABLED      : run null-collision MC gas interactions
                                  before the Boris stepper
  - BORIS_NULLCOLL_TRACK_ELECTRONS : generate & track secondary electrons

Batch mode: processes every .bl file in PARTICLE_SOURCE_DIR.
"""
import argparse
import trimesh
import numpy as np
import os
import glob
import time

import config
import geometry
import particles
import deposition
import output

# Engine modules (imported on demand to avoid import-time overhead)
import engine_raytrace
import engine_boris

# Optional modules
try:
    from fields import ElectricField, MagneticField
except ImportError:
    ElectricField = MagneticField = None

try:
    from background import GasDensityProfile
except ImportError:
    GasDensityProfile = None


def run_full_simulation(grouped_meshes, particle_source_file, output_subfolder):
    """
    Complete simulation workflow for a single beam configuration file.
    """
    basename = os.path.basename(particle_source_file) if particle_source_file else "fallback"
    print(f"\n{'='*70}")
    print(f"  Starting simulation: {basename}")
    print(f"  Engine: {config.TRACKING_ENGINE}")
    if config.TRACKING_ENGINE.lower() == "boris":
        print(f"  Relativistic: {config.BORIS_RELATIVISTIC}")
        if config.BORIS_NULLCOLL_ENABLED:
            print(f"  Null-collision MC: ON  "
                  f"(track_electrons={config.BORIS_NULLCOLL_TRACK_ELECTRONS})")
    print(f"{'='*70}")

    t0 = time.time()

    # -- Flatten geometry --
    original_meshes, object_names, save_flags = [], [], []
    for folder_path, mesh_list in grouped_meshes.items():
        settings = config.GEOMETRY_FOLDERS.get(folder_path, {})
        save_flag = settings.get("save_details", False)
        for mesh in mesh_list:
            original_meshes.append(mesh)
            object_names.append(mesh.metadata['name'])
            save_flags.append(save_flag)

    face_counts = [len(m.faces) for m in original_meshes]
    scene_mesh = trimesh.util.concatenate(original_meshes)
    face_offsets = np.cumsum([0] + face_counts[:-1])

    # -- Load particle sources --
    if particle_source_file:
        particle_sources = particles.load_beamlets_from_file(
            filename=particle_source_file,
            num_particles_per_beamlet=config.NUM_PARTICLES_PER_BEAMLET,
            beamlet_area=config.BEAMLET_AREA_FOR_CURRENT)
    else:
        particle_sources = config.PARTICLE_SOURCES

    if not particle_sources:
        print("ERROR: no particle sources loaded. Skipping.")
        return

    # -- Create depositor --
    dep = deposition.Depositor(
        face_counts, face_offsets,
        deposition_fraction_fn=config.get_deposition_fraction)

    # -- Select and run engine --
    engine_name = config.TRACKING_ENGINE.lower()

    if engine_name == "raytrace":
        engine_raytrace.run(
            scene_mesh, face_offsets, face_counts, particle_sources,
            dep, config.SOURCES_PER_WORKER, config.NUM_CPU_CORES)

    elif engine_name == "boris":
        # Load fields
        if ElectricField is None:
            from fields import ElectricField as _EF, MagneticField as _BF
        else:
            _EF, _BF = ElectricField, MagneticField

        E_field = _EF(config.E_FIELD_FILE) if config.E_FIELD_FILE else _EF()
        B_field = _BF(config.B_FIELD_FILE) if config.B_FIELD_FILE else _BF()

        # Optionally load gas profile for null-collision MC
        gas = None
        if config.BORIS_NULLCOLL_ENABLED:
            if GasDensityProfile is None:
                from background import GasDensityProfile as _GDP
            else:
                _GDP = GasDensityProfile
            if config.GAS_PROFILE_FILE:
                gas = _GDP(filepath=config.GAS_PROFILE_FILE)
            elif config.GAS_DENSITY_UNIFORM is not None:
                gas = _GDP(uniform_value=config.GAS_DENSITY_UNIFORM)
            else:
                print("WARNING: BORIS_NULLCOLL_ENABLED=true but no gas "
                      "profile or uniform density set. Null-collision skipped.")

        engine_boris.run(
            scene_mesh, face_offsets, face_counts, particle_sources,
            dep, config.SOURCES_PER_WORKER, config.NUM_CPU_CORES,
            E_field=E_field, B_field=B_field,
            step_length=config.BORIS_STEP_LENGTH_M,
            max_steps=config.BORIS_MAX_STEPS,
            relativistic=config.BORIS_RELATIVISTIC,
            gas_profile=gas,
            track_electrons=config.BORIS_NULLCOLL_TRACK_ELECTRONS)

    else:
        raise ValueError(f"Unknown TRACKING_ENGINE: '{config.TRACKING_ENGINE}'. "
                         f"Use 'raytrace' or 'boris'.")

    elapsed = time.time() - t0
    print(f"\nSimulation completed in {elapsed:.1f} s")

    # -- Save outputs --
    output_dir = os.path.join(config.DETAILED_OUTPUT_DIR, output_subfolder)

    if config.SAVE_PARAVIEW_FILES and any(save_flags):
        output.save_paraview_reports(
            original_meshes, dep, object_names, save_flags, output_dir)

    if (config.SAVE_BINARY_POWERLOADS or config.SAVE_CSV_REPORTS) and any(save_flags):
        output.save_detailed_reports(
            original_meshes, dep, object_names, save_flags, output_dir,
            save_binary=config.SAVE_BINARY_POWERLOADS,
            save_csv=config.SAVE_CSV_REPORTS)

    if config.SUMMARY_CSV_FILENAME:
        output.save_summary_csv(
            original_meshes, dep, object_names, output_dir,
            filename=f"summary_{output_subfolder}.csv")

    print(f"\n--- Finished: {basename} ---")


# ===================================================================
#  Entry point
# ===================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FullBeamSimulation — particle tracking & deposition",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '--engine', choices=['raytrace', 'boris'],
        default=None,
        help="Override TRACKING_ENGINE from config.json")
    parser.add_argument(
        '--relativistic', action='store_true', default=None,
        help="Enable relativistic Boris push (override BORIS_RELATIVISTIC)")
    parser.add_argument(
        '--nullcoll', action='store_true', default=None,
        help="Enable null-collision MC in Boris engine "
             "(override BORIS_NULLCOLL_ENABLED)")
    parser.add_argument(
        '--track-electrons', action='store_true', default=None,
        help="Track secondary electrons from null-collision ionisation events")
    args = parser.parse_args()

    # Apply CLI overrides
    if args.engine:
        config.TRACKING_ENGINE = args.engine
    if args.relativistic:
        config.BORIS_RELATIVISTIC = True
    if args.nullcoll:
        config.BORIS_NULLCOLL_ENABLED = True
    if args.track_electrons:
        config.BORIS_NULLCOLL_TRACK_ELECTRONS = True

    # -- Load geometry once --
    print("--- Loading shared geometry ---")
    grouped_geometry = geometry.load_scene(
        geometry_folders=config.GEOMETRY_FOLDERS,
        cache_dir=config.GEOMETRY_CACHE_DIR)

    # -- Batch simulation loop --
    if config.PARTICLE_SOURCE_DIR:
        bl_files = sorted(glob.glob(
            os.path.join(config.PARTICLE_SOURCE_DIR, '*.bl')))

        if not bl_files:
            print(f"ERROR: no .bl files in '{config.PARTICLE_SOURCE_DIR}'")
        else:
            print(f"\nFound {len(bl_files)} beam configuration(s).")
            for bf in bl_files:
                subfolder = os.path.splitext(os.path.basename(bf))[0]
                run_full_simulation(grouped_geometry, bf, subfolder)
    else:
        print("\nPARTICLE_SOURCE_DIR not set. Attempting fallback run...")
        if config.PARTICLE_SOURCES:
            run_full_simulation(grouped_geometry, None, "fallback_run")
        else:
            print("No particle sources defined.")
