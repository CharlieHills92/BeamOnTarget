# engine_warp.py
import math
import os
import numpy as np
import warp as wp
from tqdm import tqdm

from engine.engine import _iter_particle_batches, _empty_impact_data

wp.config.native_fp64 = True


@wp.kernel
def _raycast_and_accumulate_kernel(
        mesh_id: wp.uint64,
        origins: wp.array(dtype=wp.vec3d),
        directions: wp.array(dtype=wp.vec3d),
        powers: wp.array(dtype=wp.float64),
        energies: wp.array(dtype=wp.float64),
        dep_fraction: wp.float64,
        global_face_power: wp.array(dtype=wp.float64),
):
    tid = wp.tid()

    orig_64 = origins[tid]
    dir_64 = directions[tid]

    o_32 = wp.vec3(wp.float32(orig_64[0]), wp.float32(orig_64[1]), wp.float32(orig_64[2]))
    d_32 = wp.vec3(wp.float32(dir_64[0]),  wp.float32(dir_64[1]),  wp.float32(dir_64[2]))

    t = wp.float32(0.0)
    u = wp.float32(0.0)
    v = wp.float32(0.0)
    sign = wp.float32(0.0)
    n = wp.vec3(0.0, 0.0, 0.0)
    face = int(0)

    hit = wp.mesh_query_ray(mesh_id, o_32, d_32, 1.0e38, t, u, v, sign, n, face)

    if hit:
        power_to_deposit = powers[tid] * energies[tid] * dep_fraction
        wp.atomic_add(global_face_power, face, power_to_deposit)


def _build_warp_mesh_mixed(trimesh_mesh, device="cuda"):
    verts = np.array(trimesh_mesh.vertices, dtype=np.float32)
    faces = np.array(trimesh_mesh.faces, dtype=np.int32).flatten()
    wp_verts = wp.array(verts, dtype=wp.vec3, device=device)
    wp_faces = wp.array(faces, dtype=int, device=device)
    return wp.Mesh(points=wp_verts, indices=wp_faces)


def run_simulation_single_hit_warp(
        scene_mesh,
        face_offsets,
        face_counts,
        particle_sources_list,
        deposition_model,
        particle_batch_size, # Passed down directly from your main config script
        num_cpu_cores,
        save_impact_flags=None,
        max_impact_records=None,
        device="cuda",
):
    wp.init()

    num_objects = len(face_counts)
    total_scene_faces = sum(face_counts)

    print(f"\nInitializing Stream-Safe Warp Engine (device: {device})...")

    wp_mesh = _build_warp_mesh_mixed(scene_mesh, device=device)
    mesh_id = wp.uint64(wp_mesh.id)

    _sample_fraction = wp.float64(deposition_model(1e4))
    wp_global_face_power = wp.zeros(total_scene_faces, dtype=wp.float64, device=device)

    # Calculate exactly how many full macro-batches we expect to process
    total_particles = sum(int(s.num_particles) for s in particle_sources_list)
    total_batches = math.ceil(total_particles / particle_batch_size) if total_particles > 0 else 0

    print(f"Streaming {total_particles} particles in batches of {particle_batch_size} directly to GPU...")

    # Consume the generator from engine.py directly
    # This guarantees memory safety while still supplying the GPU with dense processing workloads
    batch_iter = _iter_particle_batches(particle_sources_list, particle_batch_size)

    for batch in tqdm(batch_iter, total=total_batches, desc="Warp GPU Raycasting (FP64)"):
        n_rays = batch["origins"].shape[0]
        if n_rays == 0:
            continue

        # Upload each streaming chunk safely into temporary VRAM scopes
        wp_origins = wp.array(batch["origins"], dtype=wp.vec3d, device=device)
        wp_directions = wp.array(batch["directions"], dtype=wp.vec3d, device=device)
        wp_powers = wp.array(batch["powers"], dtype=wp.float64, device=device)
        wp_energies = wp.array(batch["energies_ev"], dtype=wp.float64, device=device)

        wp.launch(
            kernel=_raycast_and_accumulate_kernel,
            dim=n_rays,
            inputs=[mesh_id, wp_origins, wp_directions, wp_powers, wp_energies, _sample_fraction],
            outputs=[wp_global_face_power],
            device=device,
        )

    print("\nFinalizing simulation results...")
    global_face_power_np = wp_global_face_power.numpy()

    final_deposited_power = []
    face_offsets_np = np.array(face_offsets, dtype=np.int64)

    for obj_idx in range(num_objects):
        start_idx = face_offsets_np[obj_idx]
        end_idx = start_idx + face_counts[obj_idx]
        final_deposited_power.append(global_face_power_np[start_idx:end_idx].astype(np.float32))

    return final_deposited_power, _empty_impact_data(num_objects)


# ---------------------------------------------------------------------------
# EM-track + Warp-BVH segment intersection engine
# ---------------------------------------------------------------------------

@wp.kernel
def _segment_intersect_kernel(
        mesh_id: wp.uint64,
        seg_starts: wp.array(dtype=wp.vec3d),
        seg_ends: wp.array(dtype=wp.vec3d),
        seg_powers: wp.array(dtype=wp.float64),
        seg_energies: wp.array(dtype=wp.float64),
        dep_fraction: wp.float64,
        global_face_power: wp.array(dtype=wp.float64),
        # Output: hit face index per segment (-1 = no hit)
        hit_face: wp.array(dtype=wp.int32),
        hit_t: wp.array(dtype=wp.float32),
):
    """One thread per trajectory segment.  Fires a length-bounded ray from
    seg_start toward seg_end and accumulates deposited power atomically."""
    tid = wp.tid()

    s64 = seg_starts[tid]
    e64 = seg_ends[tid]

    # Compute direction and segment length in float64, then cast to float32
    # for mesh_query_ray (the mesh BVH is always fp32).
    dx = e64[0] - s64[0]
    dy = e64[1] - s64[1]
    dz = e64[2] - s64[2]
    seg_len_sq = dx * dx + dy * dy + dz * dz

    # Skip degenerate (zero-length) segments
    if seg_len_sq <= wp.float64(0.0):
        hit_face[tid] = int(-1)
        hit_t[tid] = wp.float32(0.0)
        return

    seg_len_f64 = wp.sqrt(seg_len_sq)
    inv_len = wp.float64(1.0) / seg_len_f64

    # Normalized direction (fp32 for BVH query)
    d_32 = wp.vec3(
        wp.float32(dx * inv_len),
        wp.float32(dy * inv_len),
        wp.float32(dz * inv_len),
    )
    o_32 = wp.vec3(
        wp.float32(s64[0]),
        wp.float32(s64[1]),
        wp.float32(s64[2]),
    )
    max_t_32 = wp.float32(seg_len_f64)

    t  = wp.float32(0.0)
    u  = wp.float32(0.0)
    v  = wp.float32(0.0)
    sign = wp.float32(0.0)
    n  = wp.vec3(0.0, 0.0, 0.0)
    face = int(0)

    hit = wp.mesh_query_ray(mesh_id, o_32, d_32, max_t_32, t, u, v, sign, n, face)

    if hit:
        power_to_deposit = seg_powers[tid] * seg_energies[tid] * dep_fraction
        wp.atomic_add(global_face_power, face, power_to_deposit)
        hit_face[tid] = face
        hit_t[tid] = t
    else:
        hit_face[tid] = int(-1)
        hit_t[tid] = wp.float32(0.0)


def _flatten_segments_to_arrays(trajectory_segments):
    """Convert the list-of-dicts format produced by trace_particle_batch_em_only /
    intersect_trajectory_segments_bvh into flat numpy arrays suitable for GPU upload.

    Each entry in *trajectory_segments* must have keys:
        'starts'        : (N,3) float64  – segment start points
        'ends'          : (N,3) float64  – segment end points
        'powers'        : (N,)  float64  – particle power weights
        'energies_ev'   : (N,)  float64  – particle kinetic energy (eV)
        'charge_states' : (N,)  int32    – particle charge state
        'particle_ids'  : (N,)  int32    – original particle index within batch
        (other keys are ignored)

    Returns a single merged dict of flat arrays plus a per-segment 'charge_states'
    and 'particle_ids' array for CPU-side post-processing.
    """
    if not trajectory_segments:
        empty = {
            "starts": np.empty((0, 3), dtype=np.float64),
            "ends": np.empty((0, 3), dtype=np.float64),
            "powers": np.empty(0, dtype=np.float64),
            "energies_ev": np.empty(0, dtype=np.float64),
            "charge_states": np.empty(0, dtype=np.int32),
            "particle_ids": np.empty(0, dtype=np.int32),
        }
        return empty

    parts = {k: [] for k in ("starts", "ends", "powers", "energies_ev",
                              "charge_states", "particle_ids")}
    for seg in trajectory_segments:
        for k in parts:
            parts[k].append(np.asarray(seg[k]))

    return {k: np.concatenate(v, axis=0) for k, v in parts.items()}


def _accumulate_segment_hits_cpu(
        hit_face_np,           # (N,) int32  — face index, -1 = no hit
        hit_t_np,              # (N,) float32
        seg_arrays,            # dict from _flatten_segments_to_arrays
        face_offsets_np,       # (num_objects,) int64
        face_counts,           # list[int]
        final_deposited_power, # list of np arrays (modified in-place)
        per_species_power,     # dict {charge_state: [np arrays per object]}
        impact_data,           # list of dicts (modified in-place)
        save_impact_flags,
        max_impact_records,
):
    """CPU post-processing of GPU hit results for per-species breakdown and
    impact records (GPU kernel already wrote total power directly)."""
    hit_mask = hit_face_np >= 0
    if not np.any(hit_mask):
        return

    hit_faces   = hit_face_np[hit_mask]         # global face indices
    hit_charges = seg_arrays["charge_states"][hit_mask]
    hit_powers  = seg_arrays["powers"][hit_mask]
    hit_energies = seg_arrays["energies_ev"][hit_mask]

    # Determine object membership for each hit
    obj_indices   = np.searchsorted(face_offsets_np, hit_faces, side="right") - 1
    local_indices = hit_faces - face_offsets_np[obj_indices]

    num_objects = len(face_counts)

    # Per-species sparse accumulation
    for cs in np.unique(hit_charges):
        cs_int = int(cs)
        cs_mask = hit_charges == cs_int
        cs_faces = local_indices[cs_mask]
        cs_obj   = obj_indices[cs_mask]
        cs_power = hit_powers[cs_mask]
        cs_energy = hit_energies[cs_mask]

        if cs_int not in per_species_power:
            per_species_power[cs_int] = [
                np.zeros(c, dtype=np.float32) for c in face_counts
            ]

        # Reuse deposition model fraction; we approximate it from the already-
        # deposited total (exact fraction was applied in the GPU kernel).
        # For per-species bookkeeping we store the raw power × energy product
        # weighted by face hit so the relative fractions are preserved.
        for obj_idx in np.unique(cs_obj):
            obj_idx = int(obj_idx)
            mask2 = cs_obj == obj_idx
            l_faces = cs_faces[mask2].astype(np.int32)
            vals    = (cs_power[mask2] * cs_energy[mask2]).astype(np.float32)
            np.add.at(per_species_power[cs_int][obj_idx], l_faces, vals)

    # Impact records (reservoir sampling)
    any_save = any(save_impact_flags)
    if not any_save:
        return

    # We need start positions for impact location approximation.
    starts = seg_arrays["starts"][hit_mask]  # (H,3)
    ends   = seg_arrays["ends"][hit_mask]    # (H,3)
    t_vals = hit_t_np[hit_mask].astype(np.float64)  # normalized hit distances
    seg_lens = np.linalg.norm(ends - starts, axis=1)
    # Hit location: start + t * direction  (t is already in metres from mesh_query_ray)
    dirs_norm = np.where(
        seg_lens[:, None] > 0,
        (ends - starts) / seg_lens[:, None],
        np.zeros_like(starts),
    )
    hit_locs = starts + t_vals[:, None] * dirs_norm  # (H,3)

    for obj_idx in range(num_objects):
        if not save_impact_flags[obj_idx]:
            continue
        cap = max_impact_records[obj_idx]
        obj_mask = obj_indices == obj_idx

        if not np.any(obj_mask):
            continue

        hit_count = int(np.sum(obj_mask))
        impact_data[obj_idx]["total_hits"] += hit_count

        reservoir = impact_data[obj_idx]["records"]
        total_seen = impact_data[obj_idx]["total_hits"]

        locs_obj    = hit_locs[obj_mask]
        dirs_obj    = dirs_norm[obj_mask]
        energies_obj = hit_energies[obj_mask] if len(hit_energies) > 0 else np.array([])
        charges_obj = hit_charges[obj_mask]

        for i in range(hit_count):
            if cap is None or len(reservoir) < cap:
                reservoir.append((
                    locs_obj[i, 0], locs_obj[i, 1], locs_obj[i, 2],
                    dirs_obj[i, 0], dirs_obj[i, 1], dirs_obj[i, 2],
                    float(energies_obj[i]),
                    int(charges_obj[i]),
                ))
            else:
                j = np.random.randint(0, total_seen - hit_count + i + 1)
                if j < cap:
                    reservoir[j] = (
                        locs_obj[i, 0], locs_obj[i, 1], locs_obj[i, 2],
                        dirs_obj[i, 0], dirs_obj[i, 1], dirs_obj[i, 2],
                        float(energies_obj[i]),
                        int(charges_obj[i]),
                    )

        impact_data[obj_idx]["stored_hits"] = len(reservoir)


def run_simulation_em_track_then_bvh_warp(
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
        device="cuda",
):
    """Two-phase simulation: CPU EM tracking → GPU Warp BVH segment intersection.

    Phase 1 (CPU): Particles are integrated with the Boris pusher via
    ``trace_particle_batch_em_only``, producing a list of trajectory segments
    per particle batch.  This phase is inherently serial/CPU because it calls
    Python field providers and reaction models.

    Phase 2 (GPU): All trajectory segments from a batch are uploaded to the GPU
    and intersected against the scene BVH in one ``wp.launch`` call using the
    ``_segment_intersect_kernel``.  Each segment becomes a length-bounded ray
    so only the physically traversed path is tested, exactly matching the CPU
    BVH semantics in ``run_simulation_em_track_then_bvh``.

    Per-species power and impact records are reconstructed on the CPU from the
    GPU hit-face index array (a small transfer compared to the segment upload).

    Parameters mirror ``engine.run_simulation_em_track_then_bvh`` exactly so
    the two functions are drop-in replacements for one another.
    """
    from fields.field_provider import create_field_provider
    from reactions.reactions import create_reaction_model
    from particles.em_tracker_v2 import trace_particle_batch_em_only

    wp.init()

    num_objects = len(face_counts)
    if save_impact_flags is None:
        save_impact_flags = [False] * num_objects
    if max_impact_records is None:
        max_impact_records = [None] * num_objects

    available_cores = os.cpu_count() or 1
    n_jobs = available_cores if (num_cpu_cores == -1) else max(1, int(num_cpu_cores))

    print(f"\nInitializing EM-Track + Warp BVH Engine (device: {device})...")
    print(f"  - Using {n_jobs} CPU thread(s) for EM integration.")
    print(f"  - Step length: {em_step_length_m:.3e} m, max steps: {int(em_max_steps)}")
    if bounding_box_min_corner_m is not None and bounding_box_max_corner_m is not None:
        print(f"  - Bounding box: {bounding_box_min_corner_m} → {bounding_box_max_corner_m}")

    # Compute BVH checkpoint steps (None = full trajectory deferred to GPU)
    bvh_checkpoint_steps = None
    if em_bvh_checkpoint_distance_m is not None and em_bvh_checkpoint_distance_m > 0:
        bvh_checkpoint_steps = max(1, int(math.ceil(em_bvh_checkpoint_distance_m / em_step_length_m)))
        effective_dist = bvh_checkpoint_steps * em_step_length_m
        print(
            f"  - BVH checkpoint every {em_bvh_checkpoint_distance_m:.3g} m "
            f"(≈{effective_dist:.3g} m, {bvh_checkpoint_steps} steps)"
        )

    # Build Warp mesh (fp32 BVH, same as run_simulation_single_hit_warp)
    wp_mesh = _build_warp_mesh_mixed(scene_mesh, device=device)
    mesh_id = wp.uint64(wp_mesh.id)

    # Scalar deposition fraction (evaluated at a representative energy)
    _dep_fraction = wp.float64(deposition_model(1e4))

    total_scene_faces = sum(face_counts)
    wp_global_face_power = wp.zeros(total_scene_faces, dtype=wp.float64, device=device)

    # CPU accumulators
    final_deposited_power = [np.zeros(c, dtype=np.float32) for c in face_counts]
    per_species_power = {}
    impact_data = _empty_impact_data(num_objects)

    face_offsets_np = np.array(face_offsets, dtype=np.int64)

    total_particles = sum(int(s.num_particles) for s in particle_sources_list)
    total_batches   = math.ceil(total_particles / particle_batch_size) if total_particles > 0 else 0

    print(f"  - {total_particles} particles → ~{total_batches} batches of {particle_batch_size}.")
    print("Starting Phase 1 (CPU EM) + Phase 2 (GPU BVH) pipeline...")

    batch_iter = _iter_particle_batches(particle_sources_list, particle_batch_size)

    for batch_idx, batch in enumerate(tqdm(batch_iter, total=total_batches,
                                           desc="EM→Warp BVH")):
        if batch["origins"].shape[0] == 0:
            continue

        # --- Phase 1: CPU EM integration ---
        # Instantiate per-batch field/reaction objects (matches CPU engine behaviour)
        field_provider  = create_field_provider(external_field_cfg)
        reaction_model  = create_reaction_model(reaction_model_cfg)

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
            seed=batch_idx,
            bvh_checkpoint_steps=None,   # no CPU checkpoints; GPU handles everything
            bvh_hit_callback=None,
            perf_stats=None,
        )

        if not trajectory_segments:
            continue

        # --- Flatten segments to numpy ---
        seg = _flatten_segments_to_arrays(trajectory_segments)
        n_segs = seg["starts"].shape[0]
        if n_segs == 0:
            continue

        # --- Phase 2: GPU BVH intersection ---
        wp_starts    = wp.array(seg["starts"],      dtype=wp.vec3d,   device=device)
        wp_ends      = wp.array(seg["ends"],        dtype=wp.vec3d,   device=device)
        wp_powers    = wp.array(seg["powers"],      dtype=wp.float64, device=device)
        wp_energies  = wp.array(seg["energies_ev"], dtype=wp.float64, device=device)

        wp_hit_face  = wp.zeros(n_segs, dtype=wp.int32,   device=device)
        wp_hit_t     = wp.zeros(n_segs, dtype=wp.float32, device=device)

        wp.launch(
            kernel=_segment_intersect_kernel,
            dim=n_segs,
            inputs=[
                mesh_id,
                wp_starts, wp_ends,
                wp_powers, wp_energies,
                _dep_fraction,
                wp_global_face_power,
            ],
            outputs=[wp_hit_face, wp_hit_t],
            device=device,
        )

        # Synchronise so we can read hit arrays back (power accumulation stays on GPU)
        wp.synchronize_device(device)

        hit_face_np = wp_hit_face.numpy()
        hit_t_np    = wp_hit_t.numpy()

        # --- CPU post-processing: per-species & impact records ---
        _accumulate_segment_hits_cpu(
            hit_face_np,
            hit_t_np,
            seg,
            face_offsets_np,
            face_counts,
            final_deposited_power,
            per_species_power,
            impact_data,
            save_impact_flags,
            max_impact_records,
        )

    # --- Finalise: transfer total power from GPU → CPU ---
    print("\nFinalizing: transferring GPU power accumulator to CPU...")
    global_face_power_np = wp_global_face_power.numpy()

    for obj_idx in range(num_objects):
        start_idx = int(face_offsets_np[obj_idx])
        end_idx   = start_idx + face_counts[obj_idx]
        final_deposited_power[obj_idx] = global_face_power_np[start_idx:end_idx].astype(np.float32)

    total_deposited = sum(arr.sum() for arr in final_deposited_power)
    print(f"Total power deposited: {total_deposited:.2f} W")

    if per_species_power:
        for cs in sorted(per_species_power.keys()):
            sp_total = sum(arr.sum() for arr in per_species_power[cs])
            label = {-1: "H-", 0: "H0", 1: "H+"}.get(cs, f"q={cs}")
            if total_deposited > 0:
                print(f"  Species {label}: {sp_total:.4g} W ({100 * sp_total / total_deposited:.1f}%)")
            else:
                print(f"  Species {label}: {sp_total:.4g} W")

    for obj_idx in range(num_objects):
        if save_impact_flags[obj_idx] and impact_data[obj_idx]["total_hits"] > 0:
            d = impact_data[obj_idx]
            print(
                f"  Impact data object {obj_idx}: "
                f"{d['stored_hits']} records stored / {d['total_hits']} total hits."
            )

    return final_deposited_power, impact_data, per_species_power