# engine_warp.py
import math
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