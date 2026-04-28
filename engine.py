# engine.py
"""
The definitive high-performance, parallel, memory-safe simulation engine.
Its ONLY job is to compute the final power deposition. It does not handle
visualization sampling to ensure minimal memory footprint.
"""
import trimesh
import numpy as np
import os
import math
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from field_provider import create_field_provider
from reactions import create_reaction_model
from em_tracker_v2 import trace_particle_batch_em_only
from trajectory_intersector import intersect_trajectory_segments_bvh

def _empty_particle_batch():
    return {
        "origins": np.empty((0, 3), dtype=np.float64),
        "directions": np.empty((0, 3), dtype=np.float64),
        "powers": np.empty(0, dtype=np.float64),
        "energies_ev": np.empty(0, dtype=np.float64),
        "currents_a": np.empty(0, dtype=np.float64),
        "masses_kg": np.empty(0, dtype=np.float64),
        "charge_states": np.empty(0, dtype=np.int32),
        "source_indices": np.empty(0, dtype=np.int32),
    }


def _generate_source_particle_data(source, count):
    origins, dirs, powers, energies, currents, charge_states = source.generate(num_particles=count)
    count = len(origins)
    return {
        "origins": np.asarray(origins, dtype=np.float64),
        "directions": np.asarray(dirs, dtype=np.float64),
        "powers": np.asarray(powers, dtype=np.float64),
        "energies_ev": np.asarray(energies, dtype=np.float64),
        "currents_a": np.asarray(currents, dtype=np.float64),
        "masses_kg": np.full(count, source.mass, dtype=np.float64),
        "charge_states": np.asarray(charge_states, dtype=np.int32),
        "source_indices": np.full(count, source.source_index, dtype=np.int32),
    }


def _iter_particle_batches(particle_sources_list, particle_batch_size):
    particle_batch_size = max(1, int(particle_batch_size))

    batch_parts = {key: [] for key in _empty_particle_batch().keys()}
    batch_count = 0

    for source in particle_sources_list:
        remaining = int(source.num_particles)
        if remaining <= 0:
            continue

        while remaining > 0:
            space_left = particle_batch_size - batch_count
            take_count = min(space_left, remaining)
            generated = _generate_source_particle_data(source, take_count)
            if generated["origins"].size == 0:
                break
            for key, value in generated.items():
                batch_parts[key].append(value)
            produced = int(generated["origins"].shape[0])
            batch_count += produced
            remaining -= produced

            if batch_count >= particle_batch_size:
                yield {
                    key: np.concatenate(parts, axis=0)
                    for key, parts in batch_parts.items()
                }
                batch_parts = {key: [] for key in _empty_particle_batch().keys()}
                batch_count = 0

    if batch_count > 0:
        yield {
            key: np.concatenate(parts, axis=0)
            for key, parts in batch_parts.items()
        }


def _empty_impact_data(num_objects):
    return [{"total_hits": 0, "stored_hits": 0, "records": []} for _ in range(num_objects)]


def _merge_impact_records(impact_data, chunk_impacts, save_impact_flags, max_impact_records):
    num_objects = len(impact_data)
    for obj_idx in range(num_objects):
        if not save_impact_flags[obj_idx]:
            continue
        cap = max_impact_records[obj_idx]
        new_records = chunk_impacts[obj_idx]["data"]
        reservoir = impact_data[obj_idx]["records"]
        total_seen = impact_data[obj_idx]["total_hits"]

        for record in new_records:
            total_seen += 1
            if cap is None or len(reservoir) < cap:
                reservoir.append(record)
            else:
                j = np.random.randint(0, total_seen)
                if j < cap:
                    reservoir[j] = record

        impact_data[obj_idx]["total_hits"] = total_seen
        impact_data[obj_idx]["stored_hits"] = len(reservoir)


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
    particle_batches = list(_iter_particle_batches(particle_sources_list, particle_batch_size))
    
    # Progressive accumulation: avoid keeping all partial results in memory
    final_deposited_power = [np.zeros(count, dtype=np.float32) for count in face_counts]

    # Accumulators for impact data (reservoir sampling for unbiased selection)
    impact_data = _empty_impact_data(num_objects)

    print("Processing Particle Batches (progressive combine)...")
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(_process_particle_batch_ray, batch, intersector, face_offsets, face_counts,
                                   deposition_model, save_impact_flags, max_impact_records)
                   for batch in particle_batches]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Particle Batches"):
            sparse_chunk, chunk_impacts = future.result()
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


def run_simulation_em_track_then_bvh(
    scene_mesh,
    face_offsets,
    face_counts,
    particle_sources_list,
    deposition_model,
    particle_batch_size,
    num_cpu_cores,
    em_step_length_m,
    em_max_steps,
    em_max_distance_m=None,
    em_min_energy_ev=None,
    external_field_cfg=None,
    reaction_model_cfg=None,
    bounding_box_min_corner_m=None,
    bounding_box_max_corner_m=None,
    save_impact_flags=None,
    max_impact_records=None,
    em_bvh_checkpoint_distance_m=None,
):
    """
    Two-phase EM particle tracking:
    Phase 1: EM integration inside bounding box (no geometry checks).
    Phase 2: Deferred BVH intersection check on all trajectory segments.

    Optional intermediate BVH checkpoints (em_bvh_checkpoint_distance_m):
    If set, a BVH check is performed every time a particle has travelled that
    distance. Particles confirmed to have hit geometry are deactivated immediately
    and not traced further, reducing total computation for early-hitting particles.
    """
    num_objects = len(face_counts)
    if save_impact_flags is None:
        save_impact_flags = [False] * num_objects
    if max_impact_records is None:
        max_impact_records = [None] * num_objects

    available_cores = os.cpu_count() or 1
    n_jobs = available_cores if (num_cpu_cores == -1) else max(1, int(num_cpu_cores))
    print("\nInitializing Two-Phase EM BVH simulation engine...")
    print(f"  - Using {n_jobs} threads (available cores: {available_cores}).")
    print(f"  - Target particle batch size: {int(particle_batch_size)}")
    print(f"  - Step length: {em_step_length_m:.3e} m, max steps: {int(em_max_steps)}")
    if bounding_box_min_corner_m is not None and bounding_box_max_corner_m is not None:
        print(f"  - Bounding box min corner: {bounding_box_min_corner_m}")
        print(f"  - Bounding box max corner: {bounding_box_max_corner_m}")

    # Compute checkpoint steps (None = disabled)
    bvh_checkpoint_steps = None
    if em_bvh_checkpoint_distance_m is not None and em_bvh_checkpoint_distance_m > 0:
        # Convert user distance (m) into a fixed integer number of EM steps.
        bvh_checkpoint_steps = max(1, int(math.ceil(em_bvh_checkpoint_distance_m / em_step_length_m)))
        effective_checkpoint_distance_m = bvh_checkpoint_steps * em_step_length_m
        print(
            f"  - BVH checkpoint every {em_bvh_checkpoint_distance_m:.3g} m "
            f"(~{effective_checkpoint_distance_m:.3g} m, {bvh_checkpoint_steps} steps)"
        )

    intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(scene_mesh)
    particle_batches = list(_iter_particle_batches(particle_sources_list, particle_batch_size))

    final_deposited_power = [np.zeros(count, dtype=np.float32) for count in face_counts]
    impact_data = _empty_impact_data(num_objects)

    print("Processing particle batches (Phase 1 EM tracing + Phase 2 BVH)...")

    def _process_particle_batch_em(batch, seed):
        field_provider = create_field_provider(external_field_cfg)
        reaction_model = create_reaction_model(reaction_model_cfg)

        # Accumulated sparse updates and impact records across all checkpoint passes
        all_sparse = [(None, None) for _ in face_counts]
        all_impacts = [{'hit_count': 0, 'data': []} for _ in range(num_objects)]

        def _merge_into_all(sparse_chunk, imp_chunk):
            for obj_idx, (idxs, vals) in enumerate(sparse_chunk):
                if idxs is None:
                    continue
                prev_idxs, prev_vals = all_sparse[obj_idx]
                if prev_idxs is None:
                    all_sparse[obj_idx] = (idxs, vals)
                else:
                    all_sparse[obj_idx] = (
                        np.concatenate([prev_idxs, idxs]),
                        np.concatenate([prev_vals, vals]),
                    )
            for obj_idx in range(num_objects):
                c = imp_chunk[obj_idx]
                all_impacts[obj_idx]['hit_count'] += c['hit_count']
                all_impacts[obj_idx]['data'].extend(c['data'])

        def _bvh_checkpoint_callback(segments_chunk):
            """Run BVH on a checkpoint segment chunk; accumulate results; return hit PIDs."""
            sparse_chunk, imp_chunk, hit_pids = intersect_trajectory_segments_bvh(
                segments_chunk, intersector, face_offsets, face_counts, deposition_model,
                save_impact_flags=save_impact_flags,
            )
            _merge_into_all(sparse_chunk, imp_chunk)
            return hit_pids

        trajectory_segments = trace_particle_batch_em_only(
            batch,
            field_provider,
            reaction_model,
            em_step_length_m,
            em_max_steps,
            em_max_distance_m,
            em_min_energy_ev,
            bounding_box_min_corner_m,
            bounding_box_max_corner_m,
            seed=seed,
            bvh_checkpoint_steps=bvh_checkpoint_steps,
            bvh_hit_callback=_bvh_checkpoint_callback if bvh_checkpoint_steps is not None else None,
        )

        # Final BVH pass on the surviving segments (last checkpoint interval or all if no checkpoints)
        final_sparse, final_impacts, _ = intersect_trajectory_segments_bvh(
            trajectory_segments, intersector, face_offsets, face_counts, deposition_model,
            save_impact_flags=save_impact_flags,
        )
        _merge_into_all(final_sparse, final_impacts)

        # Consolidate duplicate triangle indices within each object
        consolidated = []
        for obj_idx in range(num_objects):
            idxs, vals = all_sparse[obj_idx]
            if idxs is None or idxs.size == 0:
                consolidated.append((None, None))
                continue
            order = np.argsort(idxs)
            idxs_s = idxs[order]
            vals_s = vals[order]
            u_idxs, inv = np.unique(idxs_s, return_inverse=True)
            u_vals = np.zeros(u_idxs.size, dtype=np.float32)
            np.add.at(u_vals, inv, vals_s)
            consolidated.append((u_idxs, u_vals))

        return consolidated, all_impacts

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(_process_particle_batch_em, batch, i)
                   for i, batch in enumerate(particle_batches)]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Particle Batches"):
            sparse_updates, chunk_impacts = future.result()
            for obj_idx, (idxs, vals) in enumerate(sparse_updates):
                if idxs is None:
                    continue
                np.add.at(final_deposited_power[obj_idx], idxs, vals)
            _merge_impact_records(impact_data, chunk_impacts, save_impact_flags, max_impact_records)

    total_deposited = sum(arr.sum() for arr in final_deposited_power)
    print(f"Total power deposited: {total_deposited:.2f} W")

    for obj_idx in range(num_objects):
        if save_impact_flags[obj_idx] and impact_data[obj_idx]['total_hits'] > 0:
            d = impact_data[obj_idx]
            print(f"  Impact data for object {obj_idx}: {d['stored_hits']} records stored out of {d['total_hits']} total hits.")

    return final_deposited_power, impact_data



