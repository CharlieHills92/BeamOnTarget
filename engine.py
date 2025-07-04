# engine.py
"""
The definitive high-performance, parallel, memory-safe simulation engine.
Its ONLY job is to compute the final power deposition. It does not handle
visualization sampling to ensure minimal memory footprint.
"""
import trimesh
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

def _process_source_chunk(source_chunk, intersector, face_offsets, face_counts, deposition_model, seed):
    """
    WORKER FUNCTION: Processes a chunk of sources and returns ONLY the power it deposited.
    """
    np.random.seed(seed)
    deposited_power = [np.zeros(count) for count in face_counts]
    
    # Generate all particles for the sources in this chunk
    chunk_origins_list, chunk_dirs_list, chunk_powers_list, chunk_energies_list = [], [], [], []
    for source in source_chunk:
        if source.num_particles > 0:
            origins, dirs, powers, energies, _, _ = source.generate()
            chunk_origins_list.append(origins); chunk_dirs_list.append(dirs)
            chunk_powers_list.append(powers); chunk_energies_list.append(energies)
            
    if not chunk_origins_list:
        return deposited_power # Return just the empty power array

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
        
        for obj_idx in np.unique(object_indices):
            mask = (object_indices == obj_idx)
            np.add.at(deposited_power[obj_idx], local_tri_indices[mask], power_to_deposit[mask])
            
    return deposited_power # Return ONLY the power data


def run_simulation_single_hit(scene_mesh, face_offsets, face_counts, particle_sources_list, 
                              deposition_model, sources_per_worker, num_cpu_cores):
    """
    MANAGER FUNCTION: Dispatches chunks, combines power results. Does not handle visualization.
    """
    print(f"\nInitializing FAST, Memory-Safe Parallel simulation engine...")
    print(f"  - Using {num_cpu_cores} CPU cores.")
    print(f"  - Grouping {sources_per_worker} sources per worker.")
    
    intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(scene_mesh)
    source_chunks = [particle_sources_list[i:i + sources_per_worker] for i in range(0, len(particle_sources_list), sources_per_worker)]
    
    with Parallel(n_jobs=num_cpu_cores) as parallel:
        tasks = (delayed(_process_source_chunk)(
            chunk, intersector, face_offsets, face_counts, deposition_model, seed=i
        ) for i, chunk in enumerate(source_chunks))
        
        # results is now just a list of deposited_power arrays
        results = parallel(tqdm(tasks, desc="Processing Source Chunks", total=len(source_chunks)))

    print("\nCombining results from parallel workers...")
    final_deposited_power = [np.zeros(count) for count in face_counts]
    for partial_power in results:
        for i in range(len(final_deposited_power)):
            final_deposited_power[i] += partial_power[i]
            
    total_deposited = sum(arr.sum() for arr in final_deposited_power)
    print(f"Total power deposited: {total_deposited:.2f} W")
    
    return final_deposited_power # Return ONLY the final power data