# engine_ray.py
"""Ray-casting simulation engine (single-hit BVH intersection)."""
import trimesh
import numpy as np
import os
import math
from tqdm import tqdm

from simulation.engine_helpers import (
    _iter_particle_batches, _empty_impact_data, _merge_impact_records,
)

def _process_particle_batch_ray(particle_batch, intersector, face_offsets, face_counts,
                                deposition_model, save_impact_flags=None, max_impact_records=None):
    """
        WORKER FUNCTION: Processes one particle batch and returns sparse updates per object.
    Returns: (sparse_updates, impact_records)
      sparse_updates: list of tuples (indices_array, values_array) per object.
      impact_records: list of dicts per object (only for objects with save_impact_flags=True).
        Each dict has keys: 'hit_count', 'data' (list of row-tuples, capped at max_impact_records).
        If an object has no hits in the batch, returns (None, None) for that object in sparse_updates,
    and {'hit_count': 0, 'data': []} in impact_records.
    """
    num_objects = len(face_counts)
    sparse_updates = [(None, None) for _ in face_counts]

    # Initialise per-object impact record containers
    if save_impact_flags is None:
        save_impact_flags = [False] * num_objects
    if max_impact_records is None:
        max_impact_records = [None] * num_objects
    impact_records = [{'hit_count': 0, 'data': []} for _ in range(num_objects)]
    
    if particle_batch["origins"].size == 0:
        return sparse_updates, impact_records  # No particles; no updates

    ray_origins = particle_batch["origins"]
    ray_directions = particle_batch["directions"]
    particle_powers = particle_batch["powers"]
    particle_energies_eV = particle_batch["energies_ev"]
    particle_masses = particle_batch["masses_kg"]
    particle_charges = particle_batch["charge_states"]
    particle_source_indices = particle_batch["source_indices"]
    particle_currents = particle_batch["currents_a"]

    # Sub-chunk the ray intersection to avoid memory allocation failures inside trimesh
    _RAY_SUB_CHUNK = 500_000
    n_rays = ray_origins.shape[0]
    if n_rays <= _RAY_SUB_CHUNK:
        locations, index_ray, index_tri_global = intersector.intersects_location(
            ray_origins=ray_origins, ray_directions=ray_directions, multiple_hits=False)
    else:
        loc_parts, iray_parts, itri_parts = [], [], []
        for start in range(0, n_rays, _RAY_SUB_CHUNK):
            end = min(start + _RAY_SUB_CHUNK, n_rays)
            locs_c, iray_c, itri_c = intersector.intersects_location(
                ray_origins=ray_origins[start:end],
                ray_directions=ray_directions[start:end],
                multiple_hits=False)
            if len(locs_c) > 0:
                loc_parts.append(locs_c)
                iray_parts.append(iray_c + start)  # offset ray indices
                itri_parts.append(itri_c)
        if loc_parts:
            locations = np.concatenate(loc_parts, axis=0)
            index_ray = np.concatenate(iray_parts, axis=0)
            index_tri_global = np.concatenate(itri_parts, axis=0)
        else:
            locations = np.empty((0, 3), dtype=np.float64)
            index_ray = np.empty(0, dtype=np.intp)
            index_tri_global = np.empty(0, dtype=np.intp)
    
    if len(locations) > 0:
        colliding_particle_power = particle_powers[index_ray]
        colliding_particle_energy_eV = particle_energies_eV[index_ray]
        fraction = deposition_model(colliding_particle_energy_eV)
        power_to_deposit = colliding_particle_power * fraction

        object_indices = np.searchsorted(face_offsets, index_tri_global, side='right') - 1
        local_tri_indices = index_tri_global - face_offsets[object_indices]

        # Build sparse updates per object — fully vectorized
        for obj_idx in np.unique(object_indices):
            mask = (object_indices == obj_idx)
            idxs = local_tri_indices[mask]
            vals = power_to_deposit[mask]
            if idxs.size > 0:
                # Vectorized duplicate summation via bincount
                n_faces = face_counts[obj_idx]
                summed = np.bincount(idxs, weights=vals, minlength=n_faces)
                nonzero = summed.nonzero()[0]
                sparse_updates[obj_idx] = (nonzero.astype(np.int32),
                                           summed[nonzero].astype(np.float32))

            # --- Collect impact data for flagged objects (vectorized) ---
            if save_impact_flags[obj_idx]:
                hit_rays = index_ray[mask]
                hit_count = hit_rays.size
                impact_records[obj_idx]['hit_count'] += hit_count

                hit_locs = locations[mask]
                hit_dirs = ray_directions[hit_rays]
                hit_energies = particle_energies_eV[hit_rays]
                hit_masses = particle_masses[hit_rays]
                hit_charges = particle_charges[hit_rays]
                hit_src_indices = particle_source_indices[hit_rays]
                hit_currents = particle_currents[hit_rays]

                # Build records as a structured numpy array, then convert to list of tuples
                records_array = np.column_stack([
                    hit_src_indices.astype(np.float64),
                    hit_masses,
                    hit_charges.astype(np.float64),
                    hit_locs,
                    hit_dirs,
                    hit_energies,
                    hit_currents,
                ])
                impact_records[obj_idx]['data'].extend(
                    [tuple(row) for row in records_array])

    return sparse_updates, impact_records


def run_simulation_single_hit(scene_mesh, face_offsets, face_counts, particle_sources_list,
                              deposition_model, particle_batch_size, num_cpu_cores,
                              save_impact_flags=None, max_impact_records=None):
    """
    MANAGER FUNCTION: Dispatches chunks, combines power results and (optionally) impact data.
    Returns: (final_deposited_power, impact_data)
      impact_data: list of dicts per object. Each dict has:
        'total_hits': int — total number of particle impacts on this object
        'stored_hits': int — number of impact records actually stored (≤ max_impact_records)
        'records': list of tuples (source_index, mass_kg, charge_state, pos_x, pos_y, pos_z,
                                   dir_x, dir_y, dir_z, kinetic_energy_eV, current_A)
      Only populated for objects where save_impact_flags[i] is True.
    """
    num_objects = len(face_counts)
    if save_impact_flags is None:
        save_impact_flags = [False] * num_objects
    if max_impact_records is None:
        max_impact_records = [None] * num_objects

    # Resolve cores: -1 means all available
    available_cores = os.cpu_count() or 1
    n_jobs = available_cores if (num_cpu_cores == -1) else max(1, int(num_cpu_cores))
    print(f"\nInitializing FAST, Memory-Safe Parallel simulation engine...")
    print(f"  - Using {n_jobs} threads (available cores: {available_cores}).")
    print(f"  - Target particle batch size: {int(particle_batch_size)}")
    
    intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(scene_mesh)

    # Count total batches for progress bar without materializing them all in memory
    total_particles = sum(int(s.num_particles) for s in particle_sources_list)
    total_batches = math.ceil(total_particles / particle_batch_size) if total_particles > 0 else 0

    # Progressive accumulation: avoid keeping all partial results in memory
    final_deposited_power = [np.zeros(count, dtype=np.float32) for count in face_counts]

    # Accumulators for impact data (reservoir sampling for unbiased selection)
    impact_data = _empty_impact_data(num_objects)

    print(f"Processing ~{total_batches} Particle Batches (sequential)...")
    batch_iter = _iter_particle_batches(particle_sources_list, particle_batch_size)
    for batch in tqdm(batch_iter, total=total_batches, desc="Processing Particle Batches"):
        sparse_chunk, chunk_impacts = _process_particle_batch_ray(
            batch, intersector, face_offsets, face_counts,
            deposition_model, save_impact_flags, max_impact_records)
        # Apply sparse updates to the accumulator immediately
        for obj_idx, (idxs, vals) in enumerate(sparse_chunk):
            if idxs is None:
                continue
            np.add.at(final_deposited_power[obj_idx], idxs, vals)
        _merge_impact_records(impact_data, chunk_impacts, save_impact_flags, max_impact_records)

    total_deposited = sum(arr.sum() for arr in final_deposited_power)
    print(f"Total power deposited: {total_deposited:.2f} W")

    # Print impact data summary
    for obj_idx in range(num_objects):
        if save_impact_flags[obj_idx] and impact_data[obj_idx]['total_hits'] > 0:
            d = impact_data[obj_idx]
            print(f"  Impact data for object {obj_idx}: {d['stored_hits']} records stored out of {d['total_hits']} total hits.")
    
    return final_deposited_power, impact_data


