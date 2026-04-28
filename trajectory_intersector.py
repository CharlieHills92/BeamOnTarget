"""
Phase 2: Deferred trajectory-to-BVH intersection checking.
Takes trajectory segments from Phase 1 and finds first hits against geometry.
"""

import numpy as np


def intersect_trajectory_segments_bvh(
    trajectory_segments,
    intersector,
    face_offsets,
    face_counts,
    deposition_model,
    save_impact_flags=None,
):
    """
    Check trajectory segments against BVH and compute power deposition.
    
    Args:
        trajectory_segments: structured array from Phase 1 with fields:
            - start_pos, end_pos, velocity, mass_kg, charge_state, 
              kinetic_energy_ev, current_a, source_index
        intersector: trimesh BVH intersector
        face_offsets: cumulative face counts per object (for unmapping triangles to objects)
        face_counts: face count per object
        deposition_model: function(energy_eV) -> fraction_deposited
    
    Returns:
        (sparse_updates, impact_records)
            sparse_updates: list of (index_array, value_array) tuples per object
            impact_records: list of dicts per object with 'hit_count' and 'data'
    """
    num_objects = len(face_counts)
    if save_impact_flags is None:
        save_impact_flags = [False] * num_objects
    sparse_updates = [(None, None) for _ in face_counts]
    impact_records = [{'hit_count': 0, 'data': []} for _ in range(num_objects)]

    if len(trajectory_segments) == 0:
        return sparse_updates, impact_records

    # Keep only the earliest collision per particle after BVH testing.

    # Extract segments for BVH query
    ray_origins = trajectory_segments['start_pos'].astype(np.float64)
    ray_ends = trajectory_segments['end_pos'].astype(np.float64)
    segment_vectors = ray_ends - ray_origins
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)

    # Normalize to ray directions
    ray_directions = np.zeros_like(segment_vectors)
    valid_segs = segment_lengths > 1e-14
    ray_directions[valid_segs] = segment_vectors[valid_segs] / segment_lengths[valid_segs][:, np.newaxis]

    # BVH query
    locations, index_ray, index_tri_global = intersector.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        multiple_hits=False,
    )

    if len(locations) == 0:
        return sparse_updates, impact_records, set()

    # Validate hits are within segment length
    impact_vec = locations - ray_origins[index_ray]
    impact_dist = np.linalg.norm(impact_vec, axis=1)
    allowed = impact_dist <= (segment_lengths[index_ray] + 1e-9)

    if not np.any(allowed):
        return sparse_updates, impact_records, set()

    # Process allowed hits
    hit_ray_indices = index_ray[allowed]
    hit_tri_global = index_tri_global[allowed]
    hit_positions = locations[allowed]

    hit_particle_ids = trajectory_segments['particle_id'][hit_ray_indices]
    hit_step_indices = trajectory_segments['step_index'][hit_ray_indices]

    keep_mask = np.zeros(hit_ray_indices.shape[0], dtype=bool)
    earliest_hits = {}
    for idx, (particle_id, step_index, distance) in enumerate(zip(hit_particle_ids, hit_step_indices, impact_dist[allowed])):
        current = earliest_hits.get(int(particle_id))
        candidate = (int(step_index), float(distance), idx)
        if current is None or candidate[:2] < current[:2]:
            earliest_hits[int(particle_id)] = candidate
    for _, (_, _, idx) in earliest_hits.items():
        keep_mask[idx] = True

    hit_ray_indices = hit_ray_indices[keep_mask]
    hit_tri_global = hit_tri_global[keep_mask]
    hit_positions = hit_positions[keep_mask]

    # Extract particle properties at hit
    hit_segments = trajectory_segments[hit_ray_indices]
    hit_velocities = hit_segments['velocity']
    hit_masses = hit_segments['mass_kg']
    hit_charges = hit_segments['charge_state']
    hit_kinetic_eV = hit_segments['kinetic_energy_ev']
    hit_currents = hit_segments['current_a']
    hit_source_indices = hit_segments['source_index']

    # Compute kinetic energy from velocity
    v_mag_sq = np.einsum("ij,ij->i", hit_velocities, hit_velocities)
    e_kinetic_eV = 0.5 * hit_masses * v_mag_sq / 1.602176634e-19

    # Deposition
    charge_div = np.maximum(np.abs(hit_charges), 1)
    power = e_kinetic_eV * hit_currents / charge_div
    frac = deposition_model(e_kinetic_eV)
    deposit = power * frac

    # Map to objects
    object_indices = np.searchsorted(face_offsets, hit_tri_global, side='right') - 1
    local_tri_indices = hit_tri_global - face_offsets[object_indices]

    # Build sparse updates per object
    for obj_idx in np.unique(object_indices):
        mask = object_indices == obj_idx
        idxs = local_tri_indices[mask]
        vals = deposit[mask]

        if idxs.size > 0:
            # Sort and combine duplicates
            order = np.argsort(idxs)
            idxs_sorted = idxs[order]
            vals_sorted = vals[order]

            unique_idxs = []
            unique_vals = []
            prev = None
            acc = 0.0
            for i in range(idxs_sorted.size):
                ii = int(idxs_sorted[i])
                vv = float(vals_sorted[i])
                if prev is None:
                    prev = ii
                    acc = vv
                elif ii == prev:
                    acc += vv
                else:
                    unique_idxs.append(prev)
                    unique_vals.append(acc)
                    prev = ii
                    acc = vv
            if prev is not None:
                unique_idxs.append(prev)
                unique_vals.append(acc)

            sparse_updates[obj_idx] = (
                np.asarray(unique_idxs, dtype=np.int32),
                np.asarray(unique_vals, dtype=np.float32),
            )

        # Collect impact records only for flagged objects
        hit_count = int(mask.sum())
        impact_records[obj_idx]['hit_count'] += hit_count

        if not save_impact_flags[obj_idx]:
            continue

        local_hits = np.flatnonzero(mask)
        for h in local_hits:
            hv = hit_velocities[h]
            hv_norm = np.linalg.norm(hv)
            hv_dir = hv / hv_norm if hv_norm > 1e-12 else np.array([0.0, 0.0, 0.0], dtype=np.float32)
            impact_records[obj_idx]['data'].append((
                int(hit_source_indices[h]),
                float(hit_masses[h]),
                int(hit_charges[h]),
                float(hit_positions[h, 0]),
                float(hit_positions[h, 1]),
                float(hit_positions[h, 2]),
                float(hv_dir[0]),
                float(hv_dir[1]),
                float(hv_dir[2]),
                float(e_kinetic_eV[h]),
                float(hit_currents[h]),
            ))

    # hit_ray_indices is already filtered to earliest-hit-per-particle at this point
    hit_pids = set(int(p) for p in trajectory_segments['particle_id'][hit_ray_indices])
    return sparse_updates, impact_records, hit_pids
