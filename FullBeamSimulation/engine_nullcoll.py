# engine_nullcoll.py
"""
Null-collision Monte-Carlo engine for particle–background-gas interactions.

Particles travel in straight lines (ballistic, no EM fields) but may undergo
species-changing collisions with the background gas:

  D⁻ → D⁰  (single stripping)
  D⁻ → D⁺  (double stripping)
  D⁰ → D⁺  (ionisation)
  D⁰ → D⁻  (electron capture)
  D⁺ → D⁰  (charge exchange)

The null-collision method adds a fictitious "null" reaction so that the total
collision frequency is constant (= σ_max × n_gas_max × v).  Between real
collisions the particle flies freely; at each candidate event we roll to decide
real vs null.

Optionally, secondary electrons produced by ionisation / stripping events can
be generated and tracked (as straight-line rays toward the geometry).

After all gas interactions, surviving particles are cast as infinite rays
(like engine_raytrace) to find final wall intersections.
"""
import numpy as np
import trimesh
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os

import cross_sections as xsec
from cross_sections import REACTIONS

_E_CHARGE = 1.602176634e-19  # C
_ME_KG    = 9.1093837015e-31  # electron mass


def _process_chunk(source_chunk, intersector, face_offsets,
                   gas_profile, sigma_max_per_species, track_electrons,
                   seed):
    """
    Worker: generate particles, apply null-collision MC through gas,
    then cast final rays for survivors.

    Returns:
        global_face_indices (M,) int
        hit_energies_eV     (M,) float64
        hit_currents_A      (M,) float64
        hit_charges         (M,) int
    """
    np.random.seed(seed)

    # -- gather particles --
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

    pos = np.concatenate(all_pos)
    dirs = np.concatenate(all_dir)
    energy_eV = np.concatenate(all_e)
    current = np.concatenate(all_cur)
    mass_kg = np.concatenate(all_m)
    charge_e = np.concatenate(all_q).astype(int)

    N = len(pos)

    # -- compute speed from energy --
    speed = np.sqrt(2.0 * energy_eV * _E_CHARGE / mass_kg)  # m/s

    # -- secondary electron collector --
    sec_origins, sec_dirs, sec_energies, sec_currents, sec_charges = \
        [], [], [], [], []

    # -- null-collision loop for each particle --
    # Vectorised: process all particles simultaneously
    alive = np.ones(N, dtype=bool)
    max_events = 100  # safety cap per particle

    n_gas_max = gas_profile.max_value
    if n_gas_max <= 0:
        # No gas → skip MC, go straight to ray casting
        pass
    else:
        for _evt in range(max_events):
            n_alive = alive.sum()
            if n_alive == 0:
                break

            idx = np.where(alive)[0]

            # For each alive particle, find σ_max for its current charge state
            sig_max = np.array([sigma_max_per_species.get(int(charge_e[i]), 0.0)
                                for i in idx])

            # Null collision frequency: ν_null = n_gas_max * σ_max * v
            v = speed[idx]
            nu_null = n_gas_max * sig_max * v
            nu_null = np.clip(nu_null, 1e-20, None)

            # Time to next candidate event: exponential distribution
            dt = -np.log(np.random.random(n_alive)) / nu_null

            # Advance position
            pos[idx] += dirs[idx] * (v * dt)[:, None]

            # Check if still inside gas (n_gas > 0 at new position)
            n_local = gas_profile(pos[idx])
            outside = n_local <= 0
            # Particles that left the gas region are done with MC
            # (they continue as ballistic to wall)

            # For particles still in gas, decide: real or null collision
            in_gas = ~outside
            if in_gas.sum() > 0:
                idx_in = idx[in_gas]
                n_loc = n_local[in_gas]

                # Real collision probability = σ_real * n_local / (σ_max * n_gas_max)
                # σ_real = total cross section at current energy and charge state
                sigma_real = np.array([
                    xsec.total_cross_section(int(charge_e[i]), energy_eV[i])
                    for i in idx_in])
                prob_real = sigma_real * n_loc / (sig_max[in_gas] * n_gas_max)
                prob_real = np.clip(prob_real, 0, 1)

                roll = np.random.random(len(idx_in))
                real_event = roll < prob_real

                if real_event.sum() > 0:
                    # Determine WHICH reaction occurs
                    for j_loc in np.where(real_event)[0]:
                        i_global = idx_in[j_loc]
                        q_old = int(charge_e[i_global])
                        E_part = energy_eV[i_global]

                        # Build probability vector for outgoing channels
                        channels = []
                        probs_ch = []
                        for (qf, qt), sig_fn in REACTIONS.items():
                            if qf == q_old:
                                channels.append(qt)
                                probs_ch.append(float(sig_fn(E_part)))
                        if not channels:
                            continue
                        total_s = sum(probs_ch)
                        if total_s <= 0:
                            continue
                        probs_ch = [p / total_s for p in probs_ch]

                        # Roll for which channel
                        r = np.random.random()
                        cum = 0.0
                        new_q = q_old
                        for ch, pr in zip(channels, probs_ch):
                            cum += pr
                            if r < cum:
                                new_q = ch
                                break

                        charge_e[i_global] = new_q

                        # If ionisation or stripping released an electron,
                        # optionally create a secondary
                        if track_electrons and new_q > q_old:
                            n_electrons = new_q - q_old
                            for _ne in range(n_electrons):
                                # Electron born at collision point, isotropic
                                phi = np.random.uniform(0, 2 * np.pi)
                                cos_th = np.random.uniform(-1, 1)
                                sin_th = np.sqrt(1 - cos_th**2)
                                e_dir = np.array([sin_th * np.cos(phi),
                                                  sin_th * np.sin(phi),
                                                  cos_th])
                                # Electron energy ~ few eV (thermal)
                                e_energy = np.random.exponential(5.0)  # eV
                                sec_origins.append(pos[i_global].copy())
                                sec_dirs.append(e_dir)
                                sec_energies.append(e_energy)
                                sec_currents.append(current[i_global])
                                sec_charges.append(-2)  # marker for electron

    # -- Final ray cast for all survivors to find wall impact --
    hit_faces_all, hit_e_all, hit_c_all, hit_q_all = [], [], [], []

    alive_idx = np.where(alive)[0]
    if len(alive_idx) > 0:
        locs, idx_ray, idx_tri = intersector.intersects_location(
            ray_origins=pos[alive_idx],
            ray_directions=dirs[alive_idx],
            multiple_hits=False)
        if len(idx_ray) > 0:
            global_p = alive_idx[idx_ray]
            hit_faces_all.append(idx_tri.astype(np.int64))
            hit_e_all.append(energy_eV[global_p])
            hit_c_all.append(current[global_p])
            hit_q_all.append(charge_e[global_p])

    # -- Cast secondary electrons if any --
    if sec_origins:
        sec_o = np.array(sec_origins)
        sec_d = np.array(sec_dirs)
        sec_e = np.array(sec_energies)
        sec_c = np.array(sec_currents)
        sec_q = np.array(sec_charges)

        locs, idx_ray, idx_tri = intersector.intersects_location(
            ray_origins=sec_o, ray_directions=sec_d, multiple_hits=False)
        if len(idx_ray) > 0:
            hit_faces_all.append(idx_tri.astype(np.int64))
            hit_e_all.append(sec_e[idx_ray])
            hit_c_all.append(sec_c[idx_ray])
            hit_q_all.append(sec_q[idx_ray])

    if not hit_faces_all:
        return np.empty(0, int), np.empty(0), np.empty(0), np.empty(0, int)

    return (np.concatenate(hit_faces_all),
            np.concatenate(hit_e_all),
            np.concatenate(hit_c_all),
            np.concatenate(hit_q_all).astype(int))


def run(scene_mesh, face_offsets, face_counts, particle_sources,
        depositor, sources_per_worker, num_cpu_cores,
        gas_profile, track_electrons=False):
    """
    Run the null-collision Monte-Carlo engine.

    Args:
        scene_mesh, face_offsets, face_counts: geometry.
        particle_sources: list of ParticleSource.
        depositor: deposition.Depositor.
        sources_per_worker, num_cpu_cores: parallelism.
        gas_profile: background.GasDensityProfile.
        track_electrons: bool — generate and track secondary electrons.
    """
    available = os.cpu_count() or 1
    n_jobs = available if num_cpu_cores == -1 else max(1, int(num_cpu_cores))

    print(f"\n[engine_nullcoll] Initialising — {n_jobs} threads, "
          f"track_electrons={track_electrons}")

    intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(scene_mesh)

    # Pre-compute σ_max for each charge state over the energy range
    # (assume 100 eV – 10 MeV covers everything)
    sigma_max_per_species = {}
    for q in [-1, 0, +1]:
        sm = xsec.max_total_cross_section(q, (100, 1e7))
        sigma_max_per_species[q] = sm
        if sm > 0:
            print(f"  σ_max(q={q:+d}) = {sm:.3e} m²")

    chunks = [particle_sources[i:i + sources_per_worker]
              for i in range(0, len(particle_sources), sources_per_worker)]

    print(f"  Processing {len(chunks)} chunks through gas...")
    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        futures = [
            pool.submit(_process_chunk, ch, intersector, face_offsets,
                        gas_profile, sigma_max_per_species, track_electrons, i)
            for i, ch in enumerate(chunks)]

        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="  Null-collision chunks"):
            gfi, e, c, q = fut.result()
            if len(gfi) > 0:
                depositor.deposit(gfi, q, e, c)

    depositor.summary()
