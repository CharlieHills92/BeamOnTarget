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
    }

    return mesh, stats


# The main block allows this script to be run by itself for testing single files.
def main():
    """Main function for standalone execution."""
    # --- Standalone configuration (only used when run directly) ---
    INPUT_DIRECTORY_STANDALONE = "OUTPUT"
    SMOOTHING_TASKS_STANDALONE = [
        {
            "input_filename": "results_front_tube_15.vtp",
            "output_filename": "smoothed_front_tube_15.vtp",
            "radius": 0.02,         # smoothing radius in metres
            "max_cell_area": 4e-6   # only smooth cells smaller than 4e-6 m²
        }
    ]

    if not os.path.isdir(INPUT_DIRECTORY_STANDALONE):
        print(f"FATAL ERROR: Input directory '{INPUT_DIRECTORY_STANDALONE}' not found.")
        return
    print("--- Running Standalone Smoothing Process ---")
    for task in SMOOTHING_TASKS_STANDALONE:
        input_path = os.path.join(INPUT_DIRECTORY_STANDALONE, task["input_filename"])
        output_path = os.path.join(INPUT_DIRECTORY_STANDALONE, task["output_filename"])
        radius = task.get("radius", 0.02)
        max_cell_area = task.get("max_cell_area", None)

        if not os.path.isfile(input_path):
            print(f"  - File not found: {input_path}. Skipping.")
            continue

        print(f"\n  Processing: {task['input_filename']} (radius={radius}, max_cell_area={max_cell_area})")
        mesh = pv.read(input_path)
        mesh_copy = mesh.copy(deep=True)
        smoothed, _stats = apply_smoothing(mesh_copy, radius=radius, max_cell_area=max_cell_area)
        smoothed.save(output_path, binary=True)
        print(f"  Saved: {output_path}")

    print("\n--- Standalone Smoothing Process Complete ---")

if __name__ == "__main__":
    # This block is only executed when you run `python smooth_results.py`
    main()