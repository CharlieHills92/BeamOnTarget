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

def _process_source_chunk(source_chunk, intersector, face_offsets, face_counts, deposition_model, seed):
    """
    WORKER FUNCTION: Processes a chunk of sources and returns sparse updates per object.
    Returns: list of tuples (indices_array, values_array) per object.
    If an object has no hits in the chunk, returns (None, None) for that object.
    """
    np.random.seed(seed)
    sparse_updates = [(None, None) for _ in face_counts]
    
    # Generate all particles for the sources in this chunk
    chunk_origins_list, chunk_dirs_list, chunk_powers_list, chunk_energies_list = [], [], [], []
    for source in source_chunk:
        if source.num_particles > 0:
            origins, dirs, powers, energies, _, _ = source.generate()
            chunk_origins_list.append(origins); chunk_dirs_list.append(dirs)
            chunk_powers_list.append(powers); chunk_energies_list.append(energies)
            
    if not chunk_origins_list:
        return sparse_updates # No particles; no updates

    ray_origins = np.concatenate(chunk_origins_list)
    ray_directions = np.concatenate(chunk_dirs_list)
    particle_powers = np.concatenate(chunk_powers_list)
    particle_energies_eV = np.concatenate(chunk_energies_list)

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

    return sparse_updates


def run_simulation_single_hit(scene_mesh, face_offsets, face_counts, particle_sources_list, 
                              deposition_model, sources_per_worker, num_cpu_cores):
    """
    MANAGER FUNCTION: Dispatches chunks, combines power results. Does not handle visualization.
    """
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

    print("Processing Source Chunks (progressive combine)...")
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(_process_source_chunk, chunk, intersector, face_offsets, face_counts, deposition_model, i)
                   for i, chunk in enumerate(source_chunks)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Source Chunks"):
            sparse_chunk = future.result()
            # Apply sparse updates to the accumulator immediately
            for obj_idx, (idxs, vals) in enumerate(sparse_chunk):
                if idxs is None:
                    continue
                np.add.at(final_deposited_power[obj_idx], idxs, vals)
            
    total_deposited = sum(arr.sum() for arr in final_deposited_power)
    print(f"Total power deposited: {total_deposited:.2f} W")
    
    return final_deposited_power # Return ONLY the final power data
