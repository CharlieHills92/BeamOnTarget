# engine.py
"""
The definitive high-performance, parallel, memory-safe simulation engine.
Its ONLY job is to compute the final power deposition. It does not handle
visualization sampling to ensure minimal memory footprint.
"""
import trimesh
import numpy as np
from joblib import Parallel, delayed
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def _process_source_chunk(source_chunk, intersector, face_offsets, face_counts,
                          deposition_model, seed, save_impact_flags=None, max_impact_records=None):
    """
    WORKER FUNCTION: Processes a chunk of sources and returns sparse updates per object.
    Returns: (sparse_updates, impact_records)
      sparse_updates: list of tuples (indices_array, values_array) per object.
      impact_records: list of dicts per object (only for objects with save_impact_flags=True).
        Each dict has keys: 'hit_count', 'data' (list of row-tuples, capped at max_impact_records).
    If an object has no hits in the chunk, returns (None, None) for that object in sparse_updates,
    and {'hit_count': 0, 'data': []} in impact_records.
    """
    np.random.seed(seed)
    num_objects = len(face_counts)
    sparse_updates = [(None, None) for _ in face_counts]

    # Initialise per-object impact record containers
    if save_impact_flags is None:
        save_impact_flags = [False] * num_objects
    if max_impact_records is None:
        max_impact_records = [None] * num_objects
    impact_records = [{'hit_count': 0, 'data': []} for _ in range(num_objects)]
    
    # Generate all particles for the sources in this chunk
    chunk_origins_list, chunk_dirs_list, chunk_powers_list, chunk_energies_list = [], [], [], []
    chunk_masses_list, chunk_charges_list, chunk_source_indices_list, chunk_currents_list = [], [], [], []
    for source in source_chunk:
        if source.num_particles > 0:
            origins, dirs, powers, energies, currents, charge_states = source.generate()
            chunk_origins_list.append(origins); chunk_dirs_list.append(dirs)
            chunk_powers_list.append(powers); chunk_energies_list.append(energies)
            chunk_currents_list.append(currents)
            # Mass is a scalar per source; expand to per-particle array
            chunk_masses_list.append(np.full(len(origins), source.mass, dtype=np.float64))
            chunk_charges_list.append(charge_states)
            # Source index: scalar per source, expand to per-particle array
            chunk_source_indices_list.append(np.full(len(origins), source.source_index, dtype=np.int32))
            
    if not chunk_origins_list:
        return sparse_updates, impact_records  # No particles; no updates

    ray_origins = np.concatenate(chunk_origins_list)
    ray_directions = np.concatenate(chunk_dirs_list)
    particle_powers = np.concatenate(chunk_powers_list)
    particle_energies_eV = np.concatenate(chunk_energies_list)
    particle_masses = np.concatenate(chunk_masses_list)
    particle_charges = np.concatenate(chunk_charges_list)
    particle_source_indices = np.concatenate(chunk_source_indices_list)
    particle_currents = np.concatenate(chunk_currents_list)

    locations, index_ray, index_tri_global = intersector.intersects_location(
        ray_origins=ray_origins, ray_directions=ray_directions, multiple_hits=False)
    
    if len(locations) > 0:
        colliding_particle_power = particle_powers[index_ray]
        colliding_particle_energy_eV = particle_energies_eV[index_ray]
        fraction = deposition_model(colliding_particle_energy_eV)
        power_to_deposit = colliding_particle_power * fraction

        object_indices = np.searchsorted(face_offsets, index_tri_global, side='right') - 1
        local_tri_indices = index_tri_global - face_offsets[object_indices]

        # Build sparse updates per object
        for obj_idx in np.unique(object_indices):
            mask = (object_indices == obj_idx)
            idxs = local_tri_indices[mask]
            vals = power_to_deposit[mask]
            # Combine duplicates within this chunk for the same object to reduce work
            if idxs.size > 0:
                # Sort by index, then sum duplicates
                order = np.argsort(idxs)
                idxs_sorted = idxs[order]
                vals_sorted = vals[order]
                # Run-length encode sum
                unique_idxs = []
                unique_vals = []
                prev = None
                acc = 0.0
                for j in range(idxs_sorted.size):
                    iidx = int(idxs_sorted[j])
                    v = float(vals_sorted[j])
                    if prev is None:
                        prev = iidx; acc = v
                    elif iidx == prev:
                        acc += v
                    else:
                        unique_idxs.append(prev); unique_vals.append(acc)
                        prev = iidx; acc = v
                if prev is not None:
                    unique_idxs.append(prev); unique_vals.append(acc)
                sparse_updates[obj_idx] = (np.asarray(unique_idxs, dtype=np.int32), np.asarray(unique_vals, dtype=np.float32))

            # --- Collect impact data for flagged objects ---
            if save_impact_flags[obj_idx]:
                hit_count = int(mask.sum())
                impact_records[obj_idx]['hit_count'] += hit_count

                hit_rays = index_ray[mask]
                hit_locs = locations[mask]
                hit_dirs = ray_directions[hit_rays]
                hit_energies = particle_energies_eV[hit_rays]
                hit_masses = particle_masses[hit_rays]
                hit_charges = particle_charges[hit_rays]
                hit_src_indices = particle_source_indices[hit_rays]
                hit_currents = particle_currents[hit_rays]

                # Send ALL records from this chunk to the manager;
                # the manager applies reservoir sampling for unbiased selection.
                for k in range(hit_count):
                    impact_records[obj_idx]['data'].append((
                        int(hit_src_indices[k]),  # source_index
                        hit_masses[k],            # mass_kg
                        int(hit_charges[k]),       # charge_state
                        hit_locs[k, 0],            # pos_x
                        hit_locs[k, 1],            # pos_y
                        hit_locs[k, 2],            # pos_z
                        hit_dirs[k, 0],            # dir_x
                        hit_dirs[k, 1],            # dir_y
                        hit_dirs[k, 2],            # dir_z
                        hit_energies[k],           # kinetic_energy_eV
                        hit_currents[k],           # current_A
                    ))

    return sparse_updates, impact_records


def run_simulation_single_hit(scene_mesh, face_offsets, face_counts, particle_sources_list, 
                              deposition_model, sources_per_worker, num_cpu_cores,
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
    print(f"  - Grouping {sources_per_worker} sources per worker.")
    
    intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(scene_mesh)
    source_chunks = [particle_sources_list[i:i + sources_per_worker] for i in range(0, len(particle_sources_list), sources_per_worker)]
    
    # Progressive accumulation: avoid keeping all partial results in memory
    final_deposited_power = [np.zeros(count, dtype=np.float32) for count in face_counts]

    # Accumulators for impact data (reservoir sampling for unbiased selection)
    impact_data = [{'total_hits': 0, 'stored_hits': 0, 'records': []} for _ in range(num_objects)]

    print("Processing Source Chunks (progressive combine)...")
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(_process_source_chunk, chunk, intersector, face_offsets, face_counts,
                                   deposition_model, i, save_impact_flags, max_impact_records)
                   for i, chunk in enumerate(source_chunks)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Source Chunks"):
            sparse_chunk, chunk_impacts = future.result()
            # Apply sparse updates to the accumulator immediately
            for obj_idx, (idxs, vals) in enumerate(sparse_chunk):
                if idxs is None:
                    continue
                np.add.at(final_deposited_power[obj_idx], idxs, vals)

            # Merge impact data using reservoir sampling for unbiased selection.
            # Algorithm: for each new record, if the reservoir isn't full, append.
            # Otherwise, with probability cap/total_seen, replace a random entry.
            # This guarantees every impact has equal probability of being stored,
            # regardless of which chunk or source it came from.
            for obj_idx in range(num_objects):
                if not save_impact_flags[obj_idx]:
                    continue
                cap = max_impact_records[obj_idx]  # None means store everything
                new_records = chunk_impacts[obj_idx]['data']
                reservoir = impact_data[obj_idx]['records']
                total_seen = impact_data[obj_idx]['total_hits']

                for record in new_records:
                    total_seen += 1
                    if cap is None or len(reservoir) < cap:
                        reservoir.append(record)
                    else:
                        # Reservoir is full: replace a random element with
                        # probability cap/total_seen (Vitter's Algorithm R)
                        j = np.random.randint(0, total_seen)
                        if j < cap:
                            reservoir[j] = record

                impact_data[obj_idx]['total_hits'] = total_seen
                impact_data[obj_idx]['stored_hits'] = len(reservoir)
            
    total_deposited = sum(arr.sum() for arr in final_deposited_power)
    print(f"Total power deposited: {total_deposited:.2f} W")

    # Print impact data summary
    for obj_idx in range(num_objects):
        if save_impact_flags[obj_idx] and impact_data[obj_idx]['total_hits'] > 0:
            d = impact_data[obj_idx]
            print(f"  Impact data for object {obj_idx}: {d['stored_hits']} records stored out of {d['total_hits']} total hits.")
    
    return final_deposited_power, impact_data
