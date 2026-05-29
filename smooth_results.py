# smooth_results.py
"""
This file now acts as a "library" containing the core logic for smoothing.
The `apply_smoothing` function can be imported by other scripts.
It can also be run standalone to process the specific tasks defined below.

Smoothing method: Area-weighted moving average.
For each cell, we find all neighbours whose centroid lies within a given
radius (using a fast KDTree spatial lookup).  The smoothed power density
for that cell is then computed as:

    Power_Density_smoothed = Sum(Deposited_Power) / Sum(Area)

over the cell itself and all its neighbours within the radius.  This approach
is physically consistent: it preserves the total deposited power and computes
a meaningful averaged power density.
"""
import pyvista as pv
import numpy as np
from scipy.spatial import cKDTree
import os
import time
import glob
import argparse

import config

# --- Core reusable function ---
def apply_smoothing(mesh, radius=None, n_iter=None, max_cell_area=None):
    """
    Applies area-weighted moving average smoothing to a PyVista mesh,
    now respecting surface orientation (normals) to avoid internal geometry bleeding.
    """
    # --- Validate inputs -----------------------------------------------------------
    if 'Deposited_Power_W' not in mesh.cell_data:
        print("  - WARNING: 'Deposited_Power_W' not found in cell data.")
        return mesh, None

    if radius is None or radius <= 0:
        print("  - WARNING: No valid smoothing radius provided.")
        return mesh, None

    t0 = time.time()
    deposited_power = np.array(mesh.cell_data['Deposited_Power_W'], dtype=np.float64)
    n_cells = mesh.n_cells

    # --- Compute cell areas and centroids ------------------------------------------
    cell_sizes = mesh.compute_cell_sizes(length=False, area=True, volume=False)
    areas = np.array(cell_sizes.cell_data['Area'], dtype=np.float64)
    centroids = mesh.cell_centers().points 

    # --- Compute/Get Normals -------------------------------------------------------
    # Ensure mesh has cell normals. compute_normals creates 'Normals' in cell_data.
    if 'Normals' not in mesh.cell_data:
        mesh = mesh.compute_normals(cell_normals=True, point_normals=False, inplace=False)
    
    cell_normals = np.array(mesh.cell_data['Normals'], dtype=np.float64)
    
    # Angle threshold: 7 degrees. 
    # Dot product of two unit vectors = cos(theta).
    cos_threshold = np.cos(np.deg2rad(7.0))
    print("looking at face normals deviating 7")

    # --- Initialize Result Array ---------------------------------------------------
    if 'Power_Density_W_m2' in mesh.cell_data:
        smoothed_density = np.array(mesh.cell_data['Power_Density_W_m2'], dtype=np.float64)
    else:
        smoothed_density = np.divide(deposited_power, areas,
                                     out=np.zeros(n_cells, dtype=np.float64),
                                     where=areas > 0)

    peak_density_before = float(smoothed_density.max())

    # --- Determine which cells need smoothing --------------------------------------
    if max_cell_area is not None and max_cell_area > 0:
        small_mask = (areas < max_cell_area) & (deposited_power > 0)
    else:
        small_mask = deposited_power > 0
    
    small_indices = np.where(small_mask)[0]
    n_small = len(small_indices)

    if n_small == 0:
        return mesh, None

    # --- Build KDTrees -------------------------------------------------------------
    print(f"  - Building KDTree for {n_cells:,} centroids...")
    tree = cKDTree(centroids)
    small_centroids = centroids[small_indices]
    small_tree = cKDTree(small_centroids)

    print(f"  - Querying neighbours (Radius: {radius}m, Normal Tol: 7°)...")
    # Initial spatial query
    raw_neighbours_list = small_tree.query_ball_tree(tree, r=radius)

    # --- Weighted Average with Normal Filtering ------------------------------------
    # We iterate through the "small" cells (those targeted for smoothing)
    for j in range(n_small):
        target_idx = small_indices[j]
        neighbor_indices = np.array(raw_neighbours_list[j])
        
        if len(neighbor_indices) == 0:
            continue

        # Get the normal of the target cell
        target_normal = cell_normals[target_idx]
        
        # Get normals of all spatial neighbors
        neigh_normals = cell_normals[neighbor_indices]
        
        # Vectorized dot product: (N, 3) dot (3,) -> (N,)
        dot_products = np.dot(neigh_normals, target_normal)
        
        # Filter: Only keep neighbors facing roughly the same way (0-7 degrees)
        # Note: dot product of unit vectors is 1.0 if perfectly aligned.
        valid_mask = dot_products >= cos_threshold
        final_indices = neighbor_indices[valid_mask]

        if final_indices.size > 0:
            total_p = np.sum(deposited_power[final_indices])
            total_a = np.sum(areas[final_indices])
            smoothed_density[target_idx] = total_p / total_a if total_a > 0 else 0.0

    # Write back main array
    mesh.cell_data['Power_Density_W_m2'] = smoothed_density

    # --- Smooth per-species arrays (using the same logic) --------------------------
    species_stats = {}
    for key in list(mesh.cell_data.keys()):
        if key.startswith('Deposited_Power_W_') and key != 'Deposited_Power_W':
            suffix = key[len('Deposited_Power_W_'):]
            sp_power = np.array(mesh.cell_data[key], dtype=np.float64)
            density_key = f'Power_Density_W_m2_{suffix}'
            
            # Initial sp_density
            sp_density = np.zeros(n_cells, dtype=np.float64)
            np.divide(sp_power, areas, out=sp_density, where=areas > 0)
            sp_peak_before = float(sp_density.max())

            # Apply filtered smoothing
            for j in range(n_small):
                target_idx = small_indices[j]
                neighbor_indices = np.array(raw_neighbours_list[j])
                
                # Filter indices by normal again (could be cached, but simple enough to redo)
                dot_products = np.dot(cell_normals[neighbor_indices], cell_normals[target_idx])
                final_indices = neighbor_indices[dot_products >= cos_threshold]

                if final_indices.size > 0:
                    sp_total = np.sum(sp_power[final_indices])
                    sp_area = np.sum(areas[final_indices])
                    sp_density[target_idx] = sp_total / sp_area if sp_area > 0 else 0.0
            
            mesh.cell_data[density_key] = sp_density
            species_stats[suffix] = {
                "total_power_W": float(np.sum(sp_power)),
                "peak_density_before": sp_peak_before,
                "peak_density_after": float(sp_density.max()),
            }

    mesh.cell_data.active_scalars_name = 'Power_Density_W_m2'
    elapsed = time.time() - t0
    
    stats = {
        "n_cells": n_cells,
        "n_smoothed": n_small,
        "radius": radius,
        "max_cell_area": max_cell_area,
        "elapsed_s": round(elapsed, 2),
        "total_power_W": float(np.sum(deposited_power)),
        "peak_density_before": peak_density_before,
        "peak_density_after": float(smoothed_density.max()),
        "min_area_m2": float(areas.min()),
        "max_area_m2": float(areas.max()),
        "species": species_stats,
    }
    print(f"  - Normal-aware smoothing complete ({elapsed:.1f}s).")
    return mesh, stats


def _resolve_config_relative_paths(config_path):
    """Resolve path-like config entries relative to selected config file."""
    base_dir = os.path.dirname(os.path.abspath(config_path))

    def _abs_if_relative(path_value):
        if not path_value:
            return path_value
        if os.path.isabs(path_value):
            return path_value
        return os.path.abspath(os.path.join(base_dir, path_value))

    config.DETAILED_OUTPUT_DIR = _abs_if_relative(config.DETAILED_OUTPUT_DIR)


def _find_subdirs_with_results(parent_dir):
    subdirs = []
    for entry in sorted(os.listdir(parent_dir)):
        full_path = os.path.join(parent_dir, entry)
        if not os.path.isdir(full_path):
            continue
        if entry.upper() == "SMOOTHED":
            continue
        has_results = (glob.glob(os.path.join(full_path, "*.vtp")) or
                       glob.glob(os.path.join(full_path, "*.vtm")))
        if has_results:
            subdirs.append(full_path)
    return subdirs


def main(argv=None):
    """Smooth existing simulation outputs using a selected JSON config."""
    import batch_smoother
    import generate_report

    parser = argparse.ArgumentParser(description="Apply smoothing to completed simulation outputs.")
    parser.add_argument(
        '-i', '--input-config',
        default=None,
        help="Path to a JSON configuration file. Defaults to config.json.")
    parser.add_argument(
        '-r', '--radius',
        type=float,
        default=None,
        help="Override smoothing radius in metres.")
    parser.add_argument(
        '-a', '--max-cell-area',
        type=float,
        default=None,
        help="Override maximum smoothed cell area in m^2. Use 0 to smooth all powered cells.")
    args = parser.parse_args(argv)

    if args.input_config:
        cfg_path = os.path.abspath(args.input_config)
        if not os.path.isfile(cfg_path):
            print(f"FATAL ERROR: Config file not found: '{cfg_path}'")
            return
        config.apply_config(path=cfg_path)
        _resolve_config_relative_paths(cfg_path)
        print(f"Using configuration file: {cfg_path}")

    output_root = config.DETAILED_OUTPUT_DIR
    if not os.path.isdir(output_root):
        print(f"FATAL ERROR: Output directory '{output_root}' not found.")
        return

    radius = args.radius if args.radius is not None else config.SMOOTHING_RADIUS
    if args.max_cell_area is None:
        max_cell_area = config.SMOOTHING_MAX_CELL_AREA
    else:
        max_cell_area = args.max_cell_area if args.max_cell_area > 0 else None

    has_direct_results = (glob.glob(os.path.join(output_root, "*.vtp")) or
                          glob.glob(os.path.join(output_root, "*.vtm")))
    subdirs = _find_subdirs_with_results(output_root)

    if has_direct_results and not subdirs:
        dirs_to_process = [output_root]
    elif subdirs:
        dirs_to_process = subdirs
    else:
        print(f"No .vtp/.vtm files or result subfolders found in '{output_root}'. Nothing to do.")
        return

    print(f"\n=== Smoothing Configuration ===")
    print(f"  Radius        : {radius}")
    print(f"  Max cell area : {max_cell_area}")
    print(f"  Directories   : {len(dirs_to_process)}")
    print(f"===============================")

    for idx, result_dir in enumerate(dirs_to_process, 1):
        print(f"\n[{idx}/{len(dirs_to_process)}] Processing: {result_dir}")
        try:
            batch_smoother.batch_process_directory(
                result_dir,
                radius=radius,
                max_cell_area=max_cell_area)
        except Exception as e:
            print(f"  ERROR while smoothing '{result_dir}': {e}")
            continue

        smoothed_dir = os.path.join(result_dir, "SMOOTHED")
        if os.path.isdir(smoothed_dir):
            try:
                generate_report.generate_summary_csv(smoothed_dir)
            except Exception as e:
                print(f"  ERROR during smoothed report generation: {e}")

    print("\n=== Smoothing Complete ===")

if __name__ == "__main__":
    # This block is only executed when you run `python smooth_results.py`
    main()