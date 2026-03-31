# engine_raytrace.py
"""
Infinite-ray Embree engine  (the original BeamOnTarget approach).

Each particle is cast as an infinite ray from its origin along its direction.
The first intersection with the scene mesh determines where power/current is
deposited.  This is the fastest engine (~ns per ray via Embree BVH).
"""
import numpy as np
import trimesh
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def _process_chunk(source_chunk, intersector, face_offsets, seed):
    """
    Worker: generate particles from sources, cast rays, return hit info.

    Returns:
        global_face_indices (M,)  int
        hit_energies        (M,)  float64  [eV]
        hit_currents        (M,)  float64  [A]
        hit_charges         (M,)  int
    """
    np.random.seed(seed)
    all_origins, all_dirs = [], []
    all_energies, all_currents, all_masses, all_charges = [], [], [], []

    for src in source_chunk:
        if src.num_particles <= 0:
            continue
        origins, dirs, energies, currents, masses, charges = src.generate()
        all_origins.append(origins)
        all_dirs.append(dirs)
        all_energies.append(energies)
        all_currents.append(currents)
        all_masses.append(masses)
        all_charges.append(charges)

    if not all_origins:
        return np.empty(0, int), np.empty(0), np.empty(0), np.empty(0, int)

    ray_origins = np.concatenate(all_origins)
    ray_dirs = np.concatenate(all_dirs)
    energies = np.concatenate(all_energies)
    currents = np.concatenate(all_currents)
    charges = np.concatenate(all_charges)

    _locs, idx_ray, idx_tri = intersector.intersects_location(
        ray_origins=ray_origins, ray_directions=ray_dirs, multiple_hits=False)

    if len(idx_ray) == 0:
        return np.empty(0, int), np.empty(0), np.empty(0), np.empty(0, int)

    return (idx_tri.astype(np.int64),
            energies[idx_ray],
            currents[idx_ray],
            charges[idx_ray])


def run(scene_mesh, face_offsets, face_counts, particle_sources,
        depositor, sources_per_worker, num_cpu_cores):
    """
    Run the infinite-ray engine.

    Args:
        scene_mesh:       concatenated trimesh of all objects.
        face_offsets:     (num_objects,) cumulative face offsets.
        face_counts:      list of face counts per object.
        particle_sources: list of ParticleSource.
        depositor:        deposition.Depositor instance.
        sources_per_worker: int — sources per parallel chunk.
        num_cpu_cores:    int — threads (-1 = all).
    """
    import os
    available = os.cpu_count() or 1
    n_jobs = available if num_cpu_cores == -1 else max(1, int(num_cpu_cores))

    print(f"\n[engine_raytrace] Initialising — {n_jobs} threads, "
          f"{sources_per_worker} sources/worker")

    intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(scene_mesh)
    chunks = [particle_sources[i:i + sources_per_worker]
              for i in range(0, len(particle_sources), sources_per_worker)]

    print(f"  Casting rays for {len(chunks)} chunks...")
    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        futures = [pool.submit(_process_chunk, ch, intersector, face_offsets, i)
                   for i, ch in enumerate(chunks)]
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="  Ray-trace chunks"):
            gfi, e, c, q = fut.result()
            if len(gfi) > 0:
                depositor.deposit(gfi, q, e, c)

    depositor.summary()
