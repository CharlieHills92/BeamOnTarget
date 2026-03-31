# output.py
"""
Output writers for FullBeamSimulation.

Saves:
  * VTP files (ParaView) with per-species power density + current density.
  * CSV summary of total power / current per object and species.
  * Binary (.npy) power-load arrays.
  * Detailed per-face CSV reports.
"""
import pyvista as pv
import numpy as np
import pandas as pd
import os

from deposition import SPECIES_LABELS


# ===================================================================
#  VTP (ParaView) output — per species
# ===================================================================
def save_paraview_reports(original_meshes, depositor, object_names,
                          save_flags, output_directory):
    """
    Save per-object VTP files containing per-species power density and
    current density as cell data arrays.
    """
    if not any(save_flags):
        print("\nNo objects configured for VTP saving.")
        return

    print(f"\nSaving ParaView (.vtp) reports to '{output_directory}'...")
    os.makedirs(output_directory, exist_ok=True)

    power_by_species, current_by_species = depositor.get_results()
    total_power = depositor.get_total_power()
    total_current = depositor.get_total_current()

    for i, mesh in enumerate(original_meshes):
        if not save_flags[i]:
            continue

        pv_mesh = pv.wrap(mesh)
        face_areas = mesh.area_faces
        name = object_names[i]
        pv_mesh.field_data['source_filename'] = np.array([name])

        # Total power density
        pw = total_power[i]
        pd_arr = np.divide(pw, face_areas,
                           out=np.zeros_like(pw), where=face_areas > 0)
        pv_mesh.cell_data['Total_Power_W'] = pw
        pv_mesh.cell_data['Total_Power_Density_W_m2'] = pd_arr

        # Total current density
        cur = total_current[i]
        cd_arr = np.divide(cur, face_areas,
                           out=np.zeros_like(cur), where=face_areas > 0)
        pv_mesh.cell_data['Total_Current_A'] = cur
        pv_mesh.cell_data['Total_Current_Density_A_m2'] = cd_arr

        # Per-species arrays
        for q in sorted(power_by_species.keys()):
            label = SPECIES_LABELS.get(q, f"q{q}")
            p_arr = power_by_species[q][i]
            c_arr = current_by_species[q][i]
            p_dens = np.divide(p_arr, face_areas,
                               out=np.zeros_like(p_arr), where=face_areas > 0)
            c_dens = np.divide(c_arr, face_areas,
                               out=np.zeros_like(c_arr), where=face_areas > 0)
            pv_mesh.cell_data[f'Power_W_{label}'] = p_arr
            pv_mesh.cell_data[f'Power_Density_W_m2_{label}'] = p_dens
            pv_mesh.cell_data[f'Current_A_{label}'] = c_arr
            pv_mesh.cell_data[f'Current_Density_A_m2_{label}'] = c_dens

        # Default scalar for ParaView colour map
        if 'Total_Power_Density_W_m2' in pv_mesh.cell_data:
            pv_mesh.cell_data.active_scalars_name = 'Total_Power_Density_W_m2'

        sanitized = os.path.splitext(name)[0]
        out_path = os.path.join(output_directory, f"results_{sanitized}.vtp")
        pv_mesh.save(out_path, binary=True)
        print(f"  - Saved '{name}' → '{out_path}'")

    print("VTP report generation complete.")


# ===================================================================
#  Summary CSV
# ===================================================================
def save_summary_csv(original_meshes, depositor, object_names,
                     output_directory, filename="power_summary_by_object.csv"):
    """
    One-row-per-object CSV with total power, peak power density,
    total current, and breakdowns by species.
    """
    print(f"\nSaving summary CSV to '{output_directory}/{filename}'...")
    os.makedirs(output_directory, exist_ok=True)

    power_by_sp, current_by_sp = depositor.get_results()
    total_power = depositor.get_total_power()
    total_current = depositor.get_total_current()

    rows = []
    for i, mesh in enumerate(original_meshes):
        face_areas = mesh.area_faces
        pw = total_power[i]
        pd_arr = np.divide(pw, face_areas,
                           out=np.full_like(pw, np.nan), where=face_areas > 0)
        valid = np.isfinite(pd_arr)
        peak_pd = float(np.max(pd_arr[valid])) if valid.any() else 0.0

        row = {
            'object_name': object_names[i],
            'total_power_W': float(pw.sum()),
            'peak_power_density_W_m2': peak_pd,
            'total_current_A': float(total_current[i].sum()),
        }

        # Per-species columns
        for q in sorted(power_by_sp.keys()):
            label = SPECIES_LABELS.get(q, f"q{q}")
            row[f'power_W_{label}'] = float(power_by_sp[q][i].sum())
            row[f'current_A_{label}'] = float(current_by_sp[q][i].sum())

        rows.append(row)

    df = pd.DataFrame(rows)
    # Format floats
    for col in df.columns:
        if col == 'object_name':
            continue
        df[col] = df[col].apply(lambda x: f'{x:.4e}')

    out_path = os.path.join(output_directory, filename)
    df.to_csv(out_path, index=False)
    print(f"  Saved summary to '{out_path}'")


# ===================================================================
#  Detailed per-face reports
# ===================================================================
def save_detailed_reports(original_meshes, depositor, object_names,
                          save_flags, output_directory,
                          save_binary=True, save_csv=True):
    """Save per-face binary (.npy) and/or CSV reports for flagged objects."""
    if not any(save_flags):
        return
    print(f"\nSaving detailed reports to '{output_directory}'...")
    os.makedirs(output_directory, exist_ok=True)

    total_power = depositor.get_total_power()

    for i, mesh in enumerate(original_meshes):
        if not save_flags[i]:
            continue
        name = object_names[i]
        pw = total_power[i]
        sanitized = os.path.splitext(name)[0]

        if save_binary:
            np.save(os.path.join(output_directory,
                                 f"powerload_{sanitized}.npy"), pw)

        if save_csv and pw.sum() > 0:
            face_areas = mesh.area_faces
            centers = mesh.triangles_center
            hit_idx = np.where(pw > 0)[0]
            rows = []
            for fi in hit_idx:
                area = face_areas[fi]
                density = pw[fi] / area if area > 0 else 0.0
                c = centers[fi]
                rows.append({
                    'face_id': fi,
                    'deposited_power_W': f'{pw[fi]:.3e}',
                    'power_density_W_m2': f'{density:.3e}',
                    'center_x': c[0], 'center_y': c[1], 'center_z': c[2],
                })
            if rows:
                df = pd.DataFrame(rows)
                df.to_csv(os.path.join(output_directory,
                                       f"power_density_{sanitized}.csv"),
                          index=False)

    print("Detailed report generation complete.")
