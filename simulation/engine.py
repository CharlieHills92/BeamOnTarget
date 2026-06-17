# engine.py
"""
Simulation engine entry points for BeamOnTarget.

  run_simulation_single_hit        -- ray-based engine (re-exported from engine_ray)
  run_simulation_em_track_then_bvh -- two-phase EM + BVH engine
"""
import trimesh
import numpy as np
import os
import math
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from core.field_provider import create_field_provider
from core.reactions import create_reaction_model
from core.em_tracker_v2 import trace_particle_batch_em_only
from core.trajectory_intersector import intersect_trajectory_segments_bvh

from simulation.engine_helpers import (
    _iter_particle_batches, _empty_impact_data, _merge_impact_records,
)
from simulation.engine_ray import run_simulation_single_hit  # noqa: F401

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

    perf_enabled = os.environ.get("BEAMONTARGET_PROFILE_TIMING", "").strip().lower() in {"1", "true", "yes", "on"}

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

    # Count total batches for progress bar without materializing all in memory
    total_particles = sum(int(s.num_particles) for s in particle_sources_list)
    total_batches = math.ceil(total_particles / particle_batch_size) if total_particles > 0 else 0

    final_deposited_power = [np.zeros(count, dtype=np.float32) for count in face_counts]
    per_species_power = {}  # {charge_state: [np.zeros per object]}
    impact_data = _empty_impact_data(num_objects)

    print(f"Processing ~{total_batches} particle batches (Phase 1 EM tracing + Phase 2 BVH)...")

    def _process_particle_batch_em(batch, seed):
        perf_stats = {
            "em_pure_s": 0.0,
            "reaction_apply_s": 0.0,
            "checkpoint_bvh_s": 0.0,
            "final_bvh_s": 0.0,
        }
        field_provider = create_field_provider(external_field_cfg)
        reaction_model = create_reaction_model(reaction_model_cfg)

        # Accumulated sparse updates and impact records across all checkpoint passes
        all_sparse = [(None, None) for _ in face_counts]
        all_impacts = [{'hit_count': 0, 'data': []} for _ in range(num_objects)]
        # Per-species sparse updates: {charge_state: [(idxs, vals) per object]}
        all_sparse_by_species = {}

        def _merge_sparse_list(target, source):
            for obj_idx, (idxs, vals) in enumerate(source):
                if idxs is None:
                    continue
                prev_idxs, prev_vals = target[obj_idx]
                if prev_idxs is None:
                    target[obj_idx] = (idxs, vals)
                else:
                    target[obj_idx] = (
                        np.concatenate([prev_idxs, idxs]),
                        np.concatenate([prev_vals, vals]),
                    )

        def _merge_into_all(sparse_chunk, imp_chunk, species_chunk):
            _merge_sparse_list(all_sparse, sparse_chunk)
            for obj_idx in range(num_objects):
                c = imp_chunk[obj_idx]
                all_impacts[obj_idx]['hit_count'] += c['hit_count']
                all_impacts[obj_idx]['data'].extend(c['data'])
            for cs, sp_list in species_chunk.items():
                if cs not in all_sparse_by_species:
                    all_sparse_by_species[cs] = [(None, None) for _ in face_counts]
                _merge_sparse_list(all_sparse_by_species[cs], sp_list)

        def _bvh_checkpoint_callback(segments_chunk):
            """Run BVH on a checkpoint segment chunk; accumulate results; return hit PIDs."""
            sparse_chunk, imp_chunk, hit_pids, species_chunk = intersect_trajectory_segments_bvh(
                segments_chunk, intersector, face_offsets, face_counts, deposition_model,
                save_impact_flags=save_impact_flags,
            )
            _merge_into_all(sparse_chunk, imp_chunk, species_chunk)
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
            perf_stats=perf_stats if perf_enabled else None,
        )

        # Final BVH pass on the surviving segments (last checkpoint interval or all if no checkpoints)
        final_bvh_started_at = time.perf_counter()
        final_sparse, final_impacts, _, final_species = intersect_trajectory_segments_bvh(
            trajectory_segments, intersector, face_offsets, face_counts, deposition_model,
            save_impact_flags=save_impact_flags,
        )
        if perf_enabled:
            perf_stats["final_bvh_s"] += time.perf_counter() - final_bvh_started_at
        _merge_into_all(final_sparse, final_impacts, final_species)

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

        # Consolidate per-species sparse updates
        consolidated_by_species = {}
        for cs, sp_list in all_sparse_by_species.items():
            cons = []
            for obj_idx in range(num_objects):
                idxs, vals = sp_list[obj_idx]
                if idxs is None or idxs.size == 0:
                    cons.append((None, None))
                    continue
                order = np.argsort(idxs)
                idxs_s = idxs[order]
                vals_s = vals[order]
                u_idxs, inv = np.unique(idxs_s, return_inverse=True)
                u_vals = np.zeros(u_idxs.size, dtype=np.float32)
                np.add.at(u_vals, inv, vals_s)
                cons.append((u_idxs, u_vals))
            consolidated_by_species[cs] = cons

        return consolidated, all_impacts, perf_stats, consolidated_by_species

    perf_totals = {
        "em_pure_s": 0.0,
        "reaction_apply_s": 0.0,
        "checkpoint_bvh_s": 0.0,
        "final_bvh_s": 0.0,
    }

    # Cap concurrency: EM workers are memory-heavy (trajectory storage)
    effective_workers = min(n_jobs, max(total_batches, 1), 4)
    # Feed batches lazily: only keep 'effective_workers' batches in flight at once
    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        futures = {}
        batch_iter = _iter_particle_batches(particle_sources_list, particle_batch_size)
        batches_submitted = 0

        # Prime the pool with initial batches
        for batch in batch_iter:
            futures[executor.submit(_process_particle_batch_em, batch, batches_submitted)] = None
            batches_submitted += 1
            if batches_submitted >= effective_workers * 2:
                break

        with tqdm(total=total_batches, desc="Processing Particle Batches") as pbar:
            while futures:
                done_futures = []
                for future in as_completed(futures):
                    done_futures.append(future)
                    break  # process one at a time to keep memory bounded

                for future in done_futures:
                    del futures[future]
                    sparse_updates, chunk_impacts, chunk_perf, chunk_species = future.result()
                    for obj_idx, (idxs, vals) in enumerate(sparse_updates):
                        if idxs is None:
                            continue
                        np.add.at(final_deposited_power[obj_idx], idxs, vals)
                    # Accumulate per-species power
                    for cs, sp_list in chunk_species.items():
                        if cs not in per_species_power:
                            per_species_power[cs] = [np.zeros(count, dtype=np.float32) for count in face_counts]
                        for obj_idx, (idxs, vals) in enumerate(sp_list):
                            if idxs is None:
                                continue
                            np.add.at(per_species_power[cs][obj_idx], idxs, vals)
                    _merge_impact_records(impact_data, chunk_impacts, save_impact_flags, max_impact_records)
                    if perf_enabled:
                        for key in perf_totals:
                            perf_totals[key] += chunk_perf.get(key, 0.0)
                    pbar.update(1)

                    # Submit next batch from the generator to keep pipeline full
                    try:
                        next_batch = next(batch_iter)
                        futures[executor.submit(_process_particle_batch_em, next_batch, batches_submitted)] = None
                        batches_submitted += 1
                    except StopIteration:
                        pass

    total_deposited = sum(arr.sum() for arr in final_deposited_power)
    print(f"Total power deposited: {total_deposited:.2f} W")
    if per_species_power:
        for cs in sorted(per_species_power.keys()):
            sp_total = sum(arr.sum() for arr in per_species_power[cs])
            label = {-1: "H-", 0: "H0", 1: "H+"}.get(cs, f"q={cs}")
            if total_deposited > 0:
                print(f"  Species {label}: {sp_total:.2f} W ({100*sp_total/total_deposited:.1f}%)")
            else:
                print(f"  Species {label}: {sp_total:.2f} W")

    for obj_idx in range(num_objects):
        if save_impact_flags[obj_idx] and impact_data[obj_idx]['total_hits'] > 0:
            d = impact_data[obj_idx]
            print(f"  Impact data for object {obj_idx}: {d['stored_hits']} records stored out of {d['total_hits']} total hits.")

    if perf_enabled:
        total_traced = (perf_totals['em_pure_s'] + perf_totals['reaction_apply_s']
                        + perf_totals['checkpoint_bvh_s'] + perf_totals['final_bvh_s'])
        print("Performance timing summary (aggregated across worker batches):")
        print(f"  - EM pure (Boris + bookkeeping): {perf_totals['em_pure_s']:.2f} s")
        print(f"  - Reaction apply: {perf_totals['reaction_apply_s']:.2f} s")
        print(f"  - BVH checkpoint callbacks: {perf_totals['checkpoint_bvh_s']:.2f} s")
        print(f"  - Final BVH passes: {perf_totals['final_bvh_s']:.2f} s")
        print(f"  - Total traced: {total_traced:.2f} s")

    return final_deposited_power, impact_data, per_species_power



