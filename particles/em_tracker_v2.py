"""
EM particle tracing (Phase 1) — Boris integration without geometry intersection.
Outputs trajectory segments for deferred BVH checking in Phase 2.
"""

import time

import numpy as np
from particles.constants import ELEMENTARY_CHARGE_C
from particles.species import SpeciesFrame, build_species_frame


SEGMENT_DTYPE = [
    ('particle_id', np.int32),
    ('step_index', np.int32),
    ('start_pos', np.float32, (3,)),
    ('end_pos', np.float32, (3,)),
    ('velocity', np.float32, (3,)),
    ('mass_kg', np.float64),
    ('charge_state', np.int32),
    ('kinetic_energy_ev', np.float64),
    ('current_a', np.float64),
    ('source_index', np.int32),
]


def boris_push(velocities_mps, q_over_m, dt_s, electric_field_vpm, magnetic_field_t):
    """Non-relativistic Boris pusher for vectorized particle updates."""
    half_dt = 0.5 * dt_s
    qm_half_dt = q_over_m * half_dt

    v_minus = velocities_mps + electric_field_vpm * qm_half_dt[:, np.newaxis]

    t = magnetic_field_t * qm_half_dt[:, np.newaxis]
    t_mag2 = np.einsum("ij,ij->i", t, t)
    s = (2.0 * t.T / (1.0 + t_mag2)).T

    v_prime = v_minus + np.cross(v_minus, t)
    v_plus = v_minus + np.cross(v_prime, s)

    return v_plus + electric_field_vpm * qm_half_dt[:, np.newaxis]


def trace_particle_batch_em_only(
    particle_batch,
    field_provider,
    reaction_model,
    em_step_length_m,
    em_max_steps,
    em_max_distance_m=None,
    em_min_energy_ev=None,
    bounding_box_min_corner_m=None,
    bounding_box_max_corner_m=None,
    seed=None,
    bvh_checkpoint_steps=None,
    bvh_hit_callback=None,
    perf_stats=None,
):
    """
    Trace particles using EM integration ONLY (no mesh checks).

    Optional checkpointing: if bvh_checkpoint_steps is set, every that many steps
    the accumulated segments are flushed to bvh_hit_callback(segments_chunk), which
    must return a set/array of global particle IDs that have hit geometry. Those
    particles are deactivated immediately. This avoids tracing them for the remaining
    distance. Segments for particles already handled by the callback are NOT returned;
    only surviving segments (last checkpoint interval) are returned for the final BVH
    pass in the caller.

    Returns:
        trajectory_segments: structured array with fields:
            - particle_id (int32): global particle index
            - start_pos (float32, shape (3,)): segment start position
            - end_pos (float32, shape (3,)): segment end position
            - velocity (float32, shape (3,)): velocity at end of step
            - mass_kg (float64): particle mass
            - charge_state (int32): charge state
            - kinetic_energy_ev (float64): kinetic energy in eV
            - current_a (float64): particle beam current
            - source_index (int32): source that generated particle
    """
    trace_started_at = time.perf_counter()
    rng = np.random.default_rng(seed)

    if particle_batch["origins"].size == 0:
        return np.array([], dtype=SEGMENT_DTYPE)

    positions = particle_batch["origins"].astype(np.float64, copy=True)
    directions = particle_batch["directions"].astype(np.float64, copy=True)
    energies_ev = particle_batch["energies_ev"].astype(np.float64, copy=True)
    currents_a = particle_batch["currents_a"].astype(np.float64, copy=True)
    source_indices = particle_batch["source_indices"].astype(np.int32, copy=True)
    masses = particle_batch["masses_kg"].astype(np.float64, copy=True)
    charge_states = particle_batch["charge_states"].astype(np.int32, copy=True)

    species = build_species_frame(mass_kg=masses, charge_state_e=charge_states)
    reaction_species_frame = SpeciesFrame(
        mass_kg=np.empty(0, dtype=np.float64),
        charge_state_e=np.empty(0, dtype=np.int32),
    )

    speeds = np.sqrt((2.0 * np.maximum(energies_ev, 0.0) * ELEMENTARY_CHARGE_C) / np.maximum(species.mass_kg, 1e-30))
    velocities = directions * speeds[:, np.newaxis]

    particle_time = np.zeros(len(positions), dtype=np.float64)
    travel_distance = np.zeros(len(positions), dtype=np.float64)
    active = np.ones(len(positions), dtype=bool)

    # Per-field accumulators — one list.append per field per step (no Python per-particle loop)
    seg_pid  = []
    seg_sidx = []
    seg_spos = []
    seg_epos = []
    seg_vel  = []
    seg_mass = []
    seg_cs   = []
    seg_eev  = []
    seg_cur  = []
    seg_src  = []

    use_checkpoints = (bvh_checkpoint_steps is not None and bvh_hit_callback is not None
                       and bvh_checkpoint_steps > 0)

    def _flush_checkpoint():
        """Assemble current accumulators into a structured array and pass to callback.
        Returns the set of hit global particle IDs; clears accumulators in place."""
        if not seg_pid:
            return set()
        chunk = np.empty(np.concatenate(seg_pid).size, dtype=SEGMENT_DTYPE)
        chunk['particle_id']       = np.concatenate(seg_pid)
        chunk['step_index']        = np.concatenate(seg_sidx)
        chunk['start_pos']         = np.concatenate(seg_spos)
        chunk['end_pos']           = np.concatenate(seg_epos)
        chunk['velocity']          = np.concatenate(seg_vel)
        chunk['mass_kg']           = np.concatenate(seg_mass)
        chunk['charge_state']      = np.concatenate(seg_cs)
        chunk['kinetic_energy_ev'] = np.concatenate(seg_eev)
        chunk['current_a']         = np.concatenate(seg_cur)
        chunk['source_index']      = np.concatenate(seg_src)
        callback_started_at = time.perf_counter()
        hit_pids = bvh_hit_callback(chunk)
        if perf_stats is not None:
            perf_stats["checkpoint_bvh_s"] += time.perf_counter() - callback_started_at
        # clear accumulators
        for lst in (seg_pid, seg_sidx, seg_spos, seg_epos, seg_vel,
                    seg_mass, seg_cs, seg_eev, seg_cur, seg_src):
            lst.clear()
        return set(hit_pids) if hit_pids is not None else set()

    if bounding_box_min_corner_m is not None and bounding_box_max_corner_m is not None:
        bbox_min = np.minimum(
            np.asarray(bounding_box_min_corner_m, dtype=np.float64),
            np.asarray(bounding_box_max_corner_m, dtype=np.float64),
        )
        bbox_max = np.maximum(
            np.asarray(bounding_box_min_corner_m, dtype=np.float64),
            np.asarray(bounding_box_max_corner_m, dtype=np.float64),
        )
    else:
        bbox_min = None
        bbox_max = None

    for step_idx in range(int(em_max_steps)):
        active_idx = np.flatnonzero(active)
        if active_idx.size == 0:
            break

        p = positions[active_idx]
        v = velocities[active_idx]
        m = species.mass_kg[active_idx]
        q_state = species.charge_state_e[active_idx]

        # Speed via einsum — avoids np.linalg.norm overhead
        speed = np.sqrt(np.maximum(np.einsum('ij,ij->i', v, v), 0.0))
        valid = speed > 1e-12
        speed_safe = np.where(valid, speed, 1.0)
        dt = em_step_length_m / speed_safe

        e_field, b_field = field_provider.sample(p, particle_time[active_idx])
        q_over_m = (q_state * ELEMENTARY_CHARGE_C) / np.maximum(m, 1e-30)
        v_next = boris_push(v, q_over_m, dt, e_field, b_field)

        # Boris conserves speed — seg_len = em_step_length_m; no second norm needed
        p_next = p + v_next * dt[:, np.newaxis]
        if not np.all(valid):
            p_next[~valid] = p[~valid]  # zero-speed particles stay in place

        if bbox_min is not None:
            out_of_bounds = np.any((p_next < bbox_min) | (p_next > bbox_max), axis=1)
        else:
            out_of_bounds = np.zeros(active_idx.size, dtype=bool)

        # Vectorized segment accumulation — one list.append per field per step
        valid_gidx = active_idx[valid]
        if valid_gidx.size > 0:
            seg_pid.append(valid_gidx)
            seg_sidx.append(np.full(valid_gidx.size, step_idx, dtype=np.int32))
            seg_spos.append(p[valid].astype(np.float32))
            seg_epos.append(p_next[valid].astype(np.float32))
            seg_vel.append(v_next[valid].astype(np.float32))
            seg_mass.append(m[valid])
            seg_cs.append(q_state[valid])
            seg_eev.append(energies_ev[valid_gidx])
            seg_cur.append(currents_a[valid_gidx])
            seg_src.append(source_indices[valid_gidx])

        # Update state
        positions[valid_gidx] = p_next[valid]
        velocities[valid_gidx] = v_next[valid]
        particle_time[valid_gidx] += dt[valid]
        # Segment length = |v_next| * dt; equals em_step_length_m only when E=0
        seg_len_valid = np.sqrt(np.maximum(np.einsum('ij,ij->i', v_next[valid], v_next[valid]), 0.0)) * dt[valid]
        travel_distance[valid_gidx] += seg_len_valid

        # Deactivate particles that left bounds or did not move
        live_mask = valid & (~out_of_bounds)
        active[active_idx[~live_mask]] = False

        # Apply reactions and check stopping conditions on remaining active
        still_active_idx = active_idx[live_mask]
        if still_active_idx.size > 0:
            # Reuse dt[live_mask] — Boris conserves speed so dt is unchanged after push
            dt_live = dt[live_mask]
            reaction_species_frame.mass_kg = species.mass_kg[still_active_idx]
            reaction_species_frame.charge_state_e = species.charge_state_e[still_active_idx].copy()
            reaction_started_at = time.perf_counter()
            reaction_model.apply(
                species_frame=reaction_species_frame,
                positions_m=positions[still_active_idx],
                velocities_mps=velocities[still_active_idx],
                dt_s=dt_live,
                rng=rng,
            )
            if perf_stats is not None:
                perf_stats["reaction_apply_s"] += time.perf_counter() - reaction_started_at
            species.charge_state_e[still_active_idx] = reaction_species_frame.charge_state_e

            if em_max_distance_m is not None:
                active[still_active_idx[travel_distance[still_active_idx] >= em_max_distance_m]] = False

            if em_min_energy_ev is not None:
                vel_live = velocities[still_active_idx]
                e_live = 0.5 * species.mass_kg[still_active_idx] * np.einsum("ij,ij->i", vel_live, vel_live) / ELEMENTARY_CHARGE_C
                active[still_active_idx[e_live <= em_min_energy_ev]] = False

        # BVH checkpoint: flush accumulated segments, deactivate hit particles
        if use_checkpoints and (step_idx + 1) % bvh_checkpoint_steps == 0:
            hit_pids = _flush_checkpoint()
            if hit_pids:
                for pid in hit_pids:
                    active[pid] = False

    # Assemble structured output array from per-field lists
    if not seg_pid:
        return np.array([], dtype=SEGMENT_DTYPE)

    all_pid = np.concatenate(seg_pid)
    out = np.empty(all_pid.size, dtype=SEGMENT_DTYPE)
    out['particle_id']       = all_pid
    out['step_index']        = np.concatenate(seg_sidx)
    out['start_pos']         = np.concatenate(seg_spos)
    out['end_pos']           = np.concatenate(seg_epos)
    out['velocity']          = np.concatenate(seg_vel)
    out['mass_kg']           = np.concatenate(seg_mass)
    out['charge_state']      = np.concatenate(seg_cs)
    out['kinetic_energy_ev'] = np.concatenate(seg_eev)
    out['current_a']         = np.concatenate(seg_cur)
    out['source_index']      = np.concatenate(seg_src)
    if perf_stats is not None:
        elapsed = time.perf_counter() - trace_started_at
        # Pure EM time = total elapsed minus reaction and checkpoint time
        # (which are already accumulated separately inside this function)
        perf_stats["em_pure_s"] += elapsed - perf_stats["reaction_apply_s"] - perf_stats["checkpoint_bvh_s"]
    return out
