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
    Applies area-weighted moving average smoothing to a PyVista mesh.

    For each cell the algorithm:
      1. Builds a KDTree of cell centroids for fast spatial queries.
      2. For every cell, finds all cells whose centroid is within *radius*.
      3. Computes Power_Density_W_m2 = Sum(Power) / Sum(Area) over that
         neighbourhood.

    The Deposited_Power_W array is left unchanged (power is conserved per cell).
    Only the Power_Density_W_m2 array is recomputed.

    Parameters
    ----------
    mesh : pyvista.PolyData
        The mesh with cell data arrays 'Deposited_Power_W' and
        'Power_Density_W_m2'.
    radius : float
        The smoothing radius (same units as the mesh coordinates, typically
        metres).  If *None*, no smoothing is applied.
    n_iter : int, optional
        Accepted for backward-compatibility with the old Laplacian interface
        but **ignored**.  Use *radius* instead.
    max_cell_area : float, optional
        If provided, only cells with area **below** this threshold (in m²)
        are smoothed.  Larger cells keep their original power density.
        This dramatically speeds up smoothing on meshes that mix very fine
        and very coarse regions (e.g. set to 4e-6 for 4 × 10⁻⁶ m²).

    Returns
    -------
    mesh : pyvista.PolyData
        The mesh with updated 'Power_Density_W_m2'.
    stats : dict
        Dictionary with smoothing statistics (n_cells, n_smoothed,
        elapsed_s, total_power_W, peak_density_before, peak_density_after,
        etc.).  Returns *None* when no smoothing was performed.
    """
    # --- Validate inputs -----------------------------------------------------------
    if 'Deposited_Power_W' not in mesh.cell_data:
        print("  - WARNING: 'Deposited_Power_W' not found in cell data. "
              "Cannot perform area-weighted smoothing.")
        return mesh, None

    if radius is None or radius <= 0:
        print("  - WARNING: No valid smoothing radius provided. Returning mesh unchanged.")
        return mesh, None

    t0 = time.time()
    deposited_power = np.array(mesh.cell_data['Deposited_Power_W'], dtype=np.float64)
    n_cells = mesh.n_cells

    # --- Compute cell areas from the mesh ------------------------------------------
    cell_areas = mesh.compute_cell_sizes(length=False, area=True, volume=False)
    areas = np.array(cell_areas.cell_data['Area'], dtype=np.float64)

    # --- Compute cell centroids ----------------------------------------------------
    centroids = mesh.cell_centers().points  # (n_cells, 3)

    # --- Determine which cells need smoothing --------------------------------------
    # Start with the original (un-smoothed) power density as the baseline.
    if 'Power_Density_W_m2' in mesh.cell_data:
        smoothed_density = np.array(mesh.cell_data['Power_Density_W_m2'], dtype=np.float64)
    else:
        # Fallback: compute from deposited power and area
        smoothed_density = np.divide(deposited_power, areas,
                                     out=np.zeros(n_cells, dtype=np.float64),
                                     where=areas > 0)

    peak_density_before = float(smoothed_density.max())

    if max_cell_area is not None and max_cell_area > 0:
        small_mask = (areas < max_cell_area) & (deposited_power > 0)
        small_indices = np.where(small_mask)[0]
        n_small = len(small_indices)
        n_area_only = int(np.sum(areas < max_cell_area))
        print(f"  - max_cell_area filter: {n_area_only:,} / {n_cells:,} cells "
              f"below {max_cell_area:.2e} m²")
        print(f"  - After deposited power > 0 filter: {n_small:,} cells will be smoothed.")
    else:
        small_mask = deposited_power > 0
        small_indices = np.where(small_mask)[0]
        n_small = len(small_indices)
        print(f"  - No area filter; deposited power > 0 filter: "
              f"{n_small:,} / {n_cells:,} cells will be smoothed.")

    if n_small == 0:
        print("  - WARNING: No cells match the filtering criteria. Returning mesh unchanged.")
        # Still compute per-species stats even when no smoothing is needed
        early_species_stats = {}
        for key in list(mesh.cell_data.keys()):
            if key.startswith('Deposited_Power_W_') and key != 'Deposited_Power_W':
                suffix = key[len('Deposited_Power_W_'):]
                sp_power = np.array(mesh.cell_data[key], dtype=np.float64)
                density_key = f'Power_Density_W_m2_{suffix}'
                if density_key in mesh.cell_data:
                    sp_dens = np.array(mesh.cell_data[density_key], dtype=np.float64)
                else:
                    sp_dens = np.divide(sp_power, areas,
                                        out=np.zeros(n_cells, dtype=np.float64),
                                        where=areas > 0)
                sp_peak = float(sp_dens.max()) if n_cells > 0 else 0.0
                early_species_stats[suffix] = {
                    "total_power_W": float(np.sum(sp_power)),
                    "peak_density_before": sp_peak,
                    "peak_density_after": sp_peak,
                }
        stats = {
            "n_cells": n_cells,
            "n_smoothed": 0,
            "radius": radius,
            "max_cell_area": max_cell_area,
            "elapsed_s": 0.0,
            "total_power_W": float(np.sum(deposited_power)),
            "peak_density_before": peak_density_before,
            "peak_density_after": peak_density_before,
            "min_area_m2": float(areas.min()),
            "max_area_m2": float(areas.max()),
            "species": early_species_stats,
        }
        return mesh, stats

    # --- Build KDTree on ALL centroids (neighbours can be any cell) ----------------
    print(f"  - Building KDTree for {n_cells:,} cell centroids ...")
    tree = cKDTree(centroids)

    # --- Build a second tree only for the small cells to do the query efficiently --
    small_centroids = centroids[small_indices]
    small_tree = cKDTree(small_centroids)

    print(f"  - Querying neighbours within radius = {radius} m "
          f"for {n_small:,} small cells ...")
    # For each small cell, find all cells (including large ones) within radius
    neighbours_list = small_tree.query_ball_tree(tree, r=radius)

    print(f"  - Computing area-weighted power density ...")
    for j in range(n_small):
        idx = neighbours_list[j]
        total_power = np.sum(deposited_power[idx])
        total_area = np.sum(areas[idx])
        if total_area > 0:
            smoothed_density[small_indices[j]] = total_power / total_area
        else:
            smoothed_density[small_indices[j]] = 0.0

    # --- Write back ----------------------------------------------------------------
    mesh.cell_data['Power_Density_W_m2'] = smoothed_density
    # Deposited_Power_W is intentionally left unchanged (conserved).

    # --- Smooth per-species arrays if present --------------------------------------
    species_stats = {}
    for key in list(mesh.cell_data.keys()):
        if key.startswith('Deposited_Power_W_') and key != 'Deposited_Power_W':
            suffix = key[len('Deposited_Power_W_'):]
            sp_power = np.array(mesh.cell_data[key], dtype=np.float64)
            density_key = f'Power_Density_W_m2_{suffix}'
            sp_density = np.zeros(n_cells, dtype=np.float64)
            if density_key in mesh.cell_data:
                sp_density = np.array(mesh.cell_data[density_key], dtype=np.float64)
            else:
                np.divide(sp_power, areas, out=sp_density, where=areas > 0)
            sp_peak_before = float(sp_density.max()) if n_cells > 0 else 0.0
            for j in range(n_small):
                idx = neighbours_list[j]
                sp_total = np.sum(sp_power[idx])
                sp_area = np.sum(areas[idx])
                if sp_area > 0:
                    sp_density[small_indices[j]] = sp_total / sp_area
                else:
                    sp_density[small_indices[j]] = 0.0
            mesh.cell_data[density_key] = sp_density
            species_stats[suffix] = {
                "total_power_W": float(np.sum(sp_power)),
                "peak_density_before": sp_peak_before,
                "peak_density_after": float(sp_density.max()),
            }

    # Set the active scalar for ParaView convenience
    mesh.cell_data.active_scalars_name = 'Power_Density_W_m2'

    elapsed = time.time() - t0
    print(f"  - Smoothing complete in {elapsed:.1f}s. Total power conserved: "
          f"{np.sum(deposited_power):.6g} W")

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