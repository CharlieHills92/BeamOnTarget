# engine_boris.py
"""
Stepped Boris integrator with short-ray Embree collision detection.

Tracks charged particles through static E and B fields.  At each time step
the velocity is updated via the Boris push, then a short ray is cast from
the old position to the new position to check for mesh intersections.

Supports:
  * Non-relativistic and relativistic Boris push (user-selectable).
  * User-defined step length  (dl) and maximum number of steps.
  * Optional null-collision MC gas interactions (via nullcoll module).
    When enabled, particles are first transported ballistically through
    the gas region with species-changing collisions, then the Boris
    stepper takes over for all surviving charged particles.
    Neutrals (D⁰) produced by the gas phase are cast as infinite rays.
  * Vectorised over all alive particles per step (NumPy batched).
"""
import numpy as np
import trimesh
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os

# Physical constants
_E_CHARGE = 1.602176634e-19   # C
_AMU      = 1.66053906660e-27  # kg
_C        = 2.99792458e8       # m/s


def _boris_push_nonrel(v, E, B, q, m, dt):
    """
    Non-relativistic Boris push for N particles simultaneously.

    Args:
        v: (N, 3) velocities [m/s]
        E: (N, 3) electric field [V/m]
        B: (N, 3) magnetic field [T]
        q: (N,) charge [C]
        m: (N,) mass [kg]
        dt: float — time step [s]

    Returns:
        v_new: (N, 3) updated velocities
    """
    qdt_over_2m = (q * dt / (2.0 * m))[:, None]   # (N, 1)

    # half-step E acceleration
    v_minus = v + qdt_over_2m * E

    # rotation in B
    t = qdt_over_2m * B                        # (N, 3)
    t_mag2 = np.sum(t * t, axis=1, keepdims=True)
    s = 2.0 * t / (1.0 + t_mag2)               # (N, 3)

    v_prime = v_minus + np.cross(v_minus, t)
    v_plus = v_minus + np.cross(v_prime, s)

    # second half-step E acceleration
    v_new = v_plus + qdt_over_2m * E
    return v_new


def _boris_push_rel(v, E, B, q, m, dt):
    """
    Relativistic Boris push (Vay 2008 variant).

    Args: same as non-relativistic.
    Returns: v_new (N, 3).
    """
    qdt_over_2m = (q * dt / (2.0 * m))[:, None]

    # Lorentz factor at half step
    gamma_n = 1.0 / np.sqrt(1.0 - np.sum(v * v, axis=1, keepdims=True) / _C**2)
    gamma_n = np.clip(gamma_n, 1.0, 1e6)

    u_n = gamma_n * v  # 4-velocity

    # half acceleration
    u_minus = u_n + qdt_over_2m * E

    # rotation
    gamma_minus = np.sqrt(1.0 + np.sum(u_minus * u_minus, axis=1, keepdims=True) / _C**2)
    t = qdt_over_2m * B / gamma_minus
    t_mag2 = np.sum(t * t, axis=1, keepdims=True)
    s = 2.0 * t / (1.0 + t_mag2)

    u_prime = u_minus + np.cross(u_minus, t)
    u_plus = u_minus + np.cross(u_prime, s)

    # second half acceleration
    u_new = u_plus + qdt_over_2m * E

    gamma_new = np.sqrt(1.0 + np.sum(u_new * u_new, axis=1, keepdims=True) / _C**2)
    v_new = u_new / gamma_new
    return v_new


def _process_chunk(source_chunk, intersector, face_offsets,
                   E_field, B_field, step_length, max_steps,
                   relativistic, gas_profile, track_electrons, seed):
    """
    Worker: generate particles, optionally apply null-collision MC gas
    interactions, then step charged particles through E/B fields.

    When gas_profile is provided and has non-zero density:
      1. All particles go through null-collision MC (ballistic in gas).
      2. Surviving neutrals (charge_e == 0) are cast as infinite rays.
      3. Surviving charged particles (D⁺, D⁻) enter the Boris stepper.
      4. Secondary electrons (if track_electrons) are cast as infinite rays.

    Returns:
        global_face_indices (M,)  int
        hit_energies_eV     (M,)  float64
        hit_currents_A      (M,)  float64
        hit_charges         (M,)  int
    """
    np.random.seed(seed)

    # Gather all particles from this chunk of sources
    all_pos, all_dir = [], []
    all_e, all_cur, all_m, all_q = [], [], [], []
    for src in source_chunk:
        if src.num_particles <= 0:
            continue
        origins, dirs, energies, currents, masses, charges = src.generate()
        all_pos.append(origins)
        all_dir.append(dirs)
        all_e.append(energies)
        all_cur.append(currents)
        all_m.append(masses)
        all_q.append(charges)

    if not all_pos:
        return np.empty(0, int), np.empty(0), np.empty(0), np.empty(0, int)

    pos = np.concatenate(all_pos)          # (N, 3)
    dirs = np.concatenate(all_dir)         # (N, 3)  unit vectors
    energy_eV = np.concatenate(all_e)      # (N,)
    current = np.concatenate(all_cur)      # (N,)
    mass_kg = np.concatenate(all_m)        # (N,)
    charge_e = np.concatenate(all_q)       # (N,)  in units of e

    # Accumulators for ALL hits (Boris + neutrals + electrons)
    hit_faces, hit_energies, hit_currents, hit_charges = [], [], [], []

    # ------------------------------------------------------------------
    # Phase 1: Null-collision MC through background gas (optional)
    # ------------------------------------------------------------------
    if gas_profile is not None and gas_profile.max_value > 0:
        from nullcoll import apply_gas_interactions

        nc = apply_gas_interactions(
            pos, dirs, energy_eV, current, mass_kg, charge_e,
            gas_profile, track_electrons=track_electrons, seed=seed)

        # Update particle arrays with post-gas state
        pos       = nc.pos
        dirs      = nc.dirs
        energy_eV = nc.energy_eV
        current   = nc.current
        mass_kg   = nc.mass_kg
        charge_e  = nc.charge_e

        # --- Neutrals (q == 0): cast infinite rays (no EM steering) ---
        neutral_mask = (charge_e == 0) & nc.alive
        if neutral_mask.sum() > 0:
            n_pos = pos[neutral_mask]
            n_dir = dirs[neutral_mask]
            locs, idx_ray, idx_tri = intersector.intersects_location(
                ray_origins=n_pos, ray_directions=n_dir,
                multiple_hits=False)
            if len(idx_ray) > 0:
                hit_faces.append(idx_tri.astype(np.int64))
                hit_energies.append(energy_eV[neutral_mask][idx_ray])
                hit_currents.append(current[neutral_mask][idx_ray])
                hit_charges.append(charge_e[neutral_mask][idx_ray])

        # --- Secondary electrons: cast infinite rays ---
        if len(nc.sec_origins) > 0:
            locs, idx_ray, idx_tri = intersector.intersects_location(
                ray_origins=nc.sec_origins, ray_directions=nc.sec_dirs,
                multiple_hits=False)
            if len(idx_ray) > 0:
                hit_faces.append(idx_tri.astype(np.int64))
                hit_energies.append(nc.sec_energies[idx_ray])
                hit_currents.append(nc.sec_currents[idx_ray])
                hit_charges.append(nc.sec_charges[idx_ray])

        # --- Keep only alive charged particles for Boris ---
        charged_mask = (charge_e != 0) & nc.alive
        if charged_mask.sum() == 0:
            # No charged survivors → return what we have
            if not hit_faces:
                return np.empty(0, int), np.empty(0), np.empty(0), np.empty(0, int)
            return (np.concatenate(hit_faces),
                    np.concatenate(hit_energies),
                    np.concatenate(hit_currents),
                    np.concatenate(hit_charges).astype(int))

        # Filter down to charged survivors
        pos       = pos[charged_mask]
        dirs      = dirs[charged_mask]
        energy_eV = energy_eV[charged_mask]
        current   = current[charged_mask]
        mass_kg   = mass_kg[charged_mask]
        charge_e  = charge_e[charged_mask]

    # ------------------------------------------------------------------
    # Phase 2: Boris EM stepping for charged particles
    # ------------------------------------------------------------------
    N = len(pos)
    charge_C = charge_e.astype(np.float64) * _E_CHARGE  # Coulombs

    # Convert energy → speed (non-rel approximation for initial velocity)
    KE_J = energy_eV * _E_CHARGE
    speed = np.sqrt(2.0 * KE_J / mass_kg)   # m/s
    vel = dirs * speed[:, None]              # (N, 3)

    # Time step from step length and speed
    # dt = dl / |v|  — different per particle; we use a uniform dl
    dl = step_length

    alive = np.ones(N, dtype=bool)
    push_fn = _boris_push_rel if relativistic else _boris_push_nonrel

    for _step in range(max_steps):
        n_alive = alive.sum()
        if n_alive == 0:
            break

        idx = np.where(alive)[0]
        p = pos[idx]                  # (n, 3)
        v = vel[idx]                  # (n, 3)
        m = mass_kg[idx]
        q = charge_C[idx]

        # Evaluate fields at current positions
        E_vals = E_field(p)           # (n, 3)
        B_vals = B_field(p)           # (n, 3)

        # dt per particle = dl / |v|
        v_mag = np.linalg.norm(v, axis=1)
        v_mag = np.clip(v_mag, 1.0, None)  # avoid /0
        dt = dl / v_mag                     # (n,)

        # Boris push
        v_new = push_fn(v, E_vals, B_vals, q, m, dt)

        # New position
        p_new = p + v_new * dt[:, None]

        # Short-ray collision check: ray from p → p_new
        ray_dirs = p_new - p
        ray_lens = np.linalg.norm(ray_dirs, axis=1)
        safe = ray_lens > 1e-12
        ray_unit = np.zeros_like(ray_dirs)
        ray_unit[safe] = ray_dirs[safe] / ray_lens[safe, None]

        # Only cast rays for particles that actually moved
        if safe.sum() > 0:
            locs, idx_ray, idx_tri = intersector.intersects_location(
                ray_origins=p[safe],
                ray_directions=ray_unit[safe],
                multiple_hits=False)

            if len(idx_ray) > 0:
                # Check that intersection is within the step length
                hit_dist = np.linalg.norm(locs - p[safe][idx_ray], axis=1)
                within = hit_dist <= ray_lens[safe][idx_ray] * 1.001

                if within.sum() > 0:
                    # Map back to global particle indices:
                    # safe_idx[k] gives the local-within-alive index for
                    # the k-th ray that was actually cast (safe==True).
                    # idx_ray gives which of those cast rays hit.
                    safe_idx = np.where(safe)[0]
                    hit_local = safe_idx[idx_ray[within]]   # indices into alive subset
                    global_particle_idx = idx[hit_local]    # indices into full arrays

                    hit_faces.append(idx_tri[within].astype(np.int64))

                    # Kinetic energy at impact from the NEW velocity
                    v_at_hit = v_new[hit_local]
                    KE_hit = 0.5 * mass_kg[global_particle_idx] * \
                             np.sum(v_at_hit**2, axis=1) / _E_CHARGE  # eV
                    hit_energies.append(KE_hit)
                    hit_currents.append(current[global_particle_idx])
                    hit_charges.append(charge_e[global_particle_idx])

                    # Kill hit particles
                    alive[global_particle_idx] = False

        # Update survivors: v_new and p_new are (n_alive,3) arrays
        # indexed the same as idx.  Mask to only update those still alive.
        still_alive = alive[idx]
        pos[idx[still_alive]] = p_new[still_alive]
        vel[idx[still_alive]] = v_new[still_alive]

    if not hit_faces:
        return np.empty(0, int), np.empty(0), np.empty(0), np.empty(0, int)

    return (np.concatenate(hit_faces),
            np.concatenate(hit_energies),
            np.concatenate(hit_currents),
            np.concatenate(hit_charges).astype(int))


def run(scene_mesh, face_offsets, face_counts, particle_sources,
        depositor, sources_per_worker, num_cpu_cores,
        E_field, B_field, step_length, max_steps, relativistic,
        gas_profile=None, track_electrons=False):
    """
    Run the Boris EM stepped engine.

    Args:
        scene_mesh, face_offsets, face_counts: geometry (concatenated).
        particle_sources: list of ParticleSource.
        depositor:        deposition.Depositor instance.
        sources_per_worker, num_cpu_cores: parallelism settings.
        E_field, B_field: fields.ElectricField / MagneticField callables.
        step_length: float — step size in metres.
        max_steps: int — maximum steps before particle is considered lost.
        relativistic: bool — use relativistic Boris push.
        gas_profile: background.GasDensityProfile or None.
        track_electrons: bool — generate and track secondary electrons
                         from null-collision ionisation events.
    """
    available = os.cpu_count() or 1
    n_jobs = available if num_cpu_cores == -1 else max(1, int(num_cpu_cores))

    nullcoll_on = gas_profile is not None and gas_profile.max_value > 0

    print(f"\n[engine_boris] Initialising — {n_jobs} threads, "
          f"dl={step_length:.4f} m, max_steps={max_steps}, "
          f"relativistic={relativistic}, "
          f"nullcoll={nullcoll_on}, track_electrons={track_electrons}")

    intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(scene_mesh)
    chunks = [particle_sources[i:i + sources_per_worker]
              for i in range(0, len(particle_sources), sources_per_worker)]

    print(f"  Stepping {len(chunks)} chunks...")
    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        futures = [
            pool.submit(_process_chunk, ch, intersector, face_offsets,
                        E_field, B_field, step_length, max_steps,
                        relativistic, gas_profile, track_electrons, i)
            for i, ch in enumerate(chunks)]

        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="  Boris chunks"):
            gfi, e, c, q = fut.result()
            if len(gfi) > 0:
                depositor.deposit(gfi, q, e, c)

    depositor.summary()
