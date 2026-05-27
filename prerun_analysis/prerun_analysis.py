"""Pre-run diagnostics for EM tracking and reactions."""

import os

import numpy as np

import particles.cross_sections as cross_sections
from particles.constants import ELEMENTARY_CHARGE_C
from fields.field_provider import create_field_provider
from reactions.reactions import create_reaction_model



def _estimate_em_larmor_radius(particle_sources_list, external_field_cfg, reaction_model_cfg):
    """Print a lightweight EM/reaction pre-run analysis with Larmor radius estimate."""
    if not particle_sources_list:
        return

    weighted_speed = 0.0
    weighted_energy_ev = 0.0
    weighted_mass = 0.0
    weighted_abs_charge_e = 0.0
    total_weight = 0.0
    positions = []

    for src in particle_sources_list:
        weight = float(max(int(getattr(src, "num_particles", 0)), 0))
        if weight <= 0.0:
            continue

        e_min, e_max = getattr(src, "energy_range", (0.0, 0.0))
        e_avg_ev = 0.5 * (float(e_min) + float(e_max))
        mass_kg = float(getattr(src, "mass", 0.0))
        abs_charge_e = abs(float(getattr(src, "charge_state", 0.0)))

        if mass_kg > 0.0 and e_avg_ev >= 0.0:
            speed_mps = np.sqrt(2.0 * e_avg_ev * ELEMENTARY_CHARGE_C / mass_kg)
        else:
            speed_mps = 0.0

        weighted_speed += weight * speed_mps
        weighted_energy_ev += weight * max(e_avg_ev, 0.0)
        weighted_mass += weight * max(mass_kg, 0.0)
        weighted_abs_charge_e += weight * abs_charge_e
        total_weight += weight

        try:
            center, _ = src.get_visualization_repr()
            positions.append(np.asarray(center, dtype=np.float64))
        except Exception:
            continue

    if total_weight <= 0.0:
        return

    avg_abs_speed_mps = weighted_speed / total_weight
    avg_energy_ev = weighted_energy_ev / total_weight
    avg_mass_kg = weighted_mass / total_weight
    avg_abs_charge_e = weighted_abs_charge_e / total_weight

    if positions:
        sample_positions = np.vstack(positions)
    else:
        sample_positions = np.zeros((1, 3), dtype=np.float64)

    field_provider = create_field_provider(external_field_cfg)
    _, b_field_t = field_provider.sample(sample_positions, np.zeros(len(sample_positions), dtype=np.float64))
    b_norm_t = np.linalg.norm(np.asarray(b_field_t, dtype=np.float64), axis=1)
    max_b_t = float(np.max(b_norm_t)) if b_norm_t.size > 0 else 0.0

    print("\n--- EM/Reactions Pre-Run Analysis ---")
    print(f"  - Reaction model type: {str((reaction_model_cfg or {}).get('type', 'none')).strip()}")
    print(f"  - Average energy: {avg_energy_ev:.6e} eV")
    print(f"  - Average |v|: {avg_abs_speed_mps:.6e} m/s")
    print(f"  - Average mass: {avg_mass_kg:.6e} kg")
    print(f"  - Average |q|: {avg_abs_charge_e:.6e} e")
    print(f"  - Max |B|: {max_b_t:.6e} T")

    q_abs_c = avg_abs_charge_e * ELEMENTARY_CHARGE_C
    if q_abs_c <= 0.0:
        print("  - Larmor radius estimate: undefined (average |q| is zero).")
    elif max_b_t <= 0.0:
        print("  - Larmor radius estimate: infinite (max |B| is zero).")
    else:
        larmor_radius_m = avg_mass_kg * avg_abs_speed_mps / (q_abs_c * max_b_t)
        print(f"  - Larmor radius estimate: {larmor_radius_m:.6e} m")


def _ray_exit_distance_from_box(start_pos_m, direction, bbox_min, bbox_max):
    """Return forward distance from start to box exit along direction, or None."""
    p = np.asarray(start_pos_m, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    bmin = np.asarray(bbox_min, dtype=np.float64)
    bmax = np.asarray(bbox_max, dtype=np.float64)

    t_min = 0.0
    t_max = np.inf
    eps = 1e-15

    for axis in range(3):
        if abs(d[axis]) <= eps:
            if p[axis] < bmin[axis] or p[axis] > bmax[axis]:
                return None
            continue

        t1 = (bmin[axis] - p[axis]) / d[axis]
        t2 = (bmax[axis] - p[axis]) / d[axis]
        t_near = min(t1, t2)
        t_far = max(t1, t2)
        t_min = max(t_min, t_near)
        t_max = min(t_max, t_far)
        if t_max < t_min:
            return None

    if not np.isfinite(t_max) or t_max <= 0.0:
        return None
    return float(t_max)


def _analyze_reaction_species_evolution(
    particle_sources_list,
    reaction_model_cfg,
    bbox_min_corner_m,
    bbox_max_corner_m,
    output_dir_for_run,
    em_step_length_m,
):
    """Estimate species evolution along average beam line and save a diagnostic plot."""
    if not particle_sources_list:
        return

    model_type = str((reaction_model_cfg or {}).get("type", "none")).strip().lower()
    if model_type in ("none", "off", "null"):
        print("\n--- Reaction Evolution Analysis ---")
        print("  - Skipped: REACTION_MODEL is disabled.")
        return

    if bbox_min_corner_m is None or bbox_max_corner_m is None:
        print("\n--- Reaction Evolution Analysis ---")
        print("  - Skipped: EM bounding box is not defined.")
        return

    weights = []
    starts = []
    directions = []
    energies_ev = []
    masses_kg = []
    init_charges = []

    for src in particle_sources_list:
        weight = float(max(int(getattr(src, "num_particles", 0)), 0))
        if weight <= 0.0:
            continue
        weights.append(weight)

        try:
            center, direction = src.get_visualization_repr()
            starts.append(np.asarray(center, dtype=np.float64))
            directions.append(np.asarray(direction, dtype=np.float64))
        except Exception:
            starts.append(np.zeros(3, dtype=np.float64))
            directions.append(np.array([1.0, 0.0, 0.0], dtype=np.float64))

        e_min, e_max = getattr(src, "energy_range", (0.0, 0.0))
        energies_ev.append(0.5 * (float(e_min) + float(e_max)))
        masses_kg.append(float(getattr(src, "mass", 0.0)))
        init_charges.append(int(getattr(src, "charge_state", 0)))

    if not weights:
        return

    w = np.asarray(weights, dtype=np.float64)
    w /= np.sum(w)
    start_avg = np.sum(np.asarray(starts, dtype=np.float64) * w[:, np.newaxis], axis=0)

    dir_raw = np.sum(np.asarray(directions, dtype=np.float64) * w[:, np.newaxis], axis=0)
    dir_norm = np.linalg.norm(dir_raw)
    if dir_norm <= 0.0 or not np.isfinite(dir_norm):
        avg_dir = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        avg_dir = dir_raw / dir_norm

    line_length_m = _ray_exit_distance_from_box(start_avg, avg_dir, bbox_min_corner_m, bbox_max_corner_m)
    if line_length_m is None or line_length_m <= 0.0:
        print("\n--- Reaction Evolution Analysis ---")
        print("  - Skipped: average source line does not intersect bounding box forward.")
        return

    avg_energy_ev = float(np.sum(np.maximum(np.asarray(energies_ev, dtype=np.float64), 0.0) * w))
    avg_mass_kg = float(np.sum(np.maximum(np.asarray(masses_kg, dtype=np.float64), 0.0) * w))
    if avg_mass_kg <= 0.0 or avg_energy_ev < 0.0:
        print("\n--- Reaction Evolution Analysis ---")
        print("  - Skipped: invalid average mass/energy for reaction analysis.")
        return

    avg_speed_mps = np.sqrt(2.0 * avg_energy_ev * ELEMENTARY_CHARGE_C / avg_mass_kg)
    if avg_speed_mps <= 0.0 or not np.isfinite(avg_speed_mps):
        print("\n--- Reaction Evolution Analysis ---")
        print("  - Skipped: non-positive average speed.")
        return

    charge_weights = {-1: 0.0, 0: 0.0, 1: 0.0}
    for charge, weight in zip(init_charges, w):
        if charge in charge_weights:
            charge_weights[charge] += float(weight)

    init_vec = np.array(
        [charge_weights[-1], charge_weights[0], charge_weights[1]],
        dtype=np.float64,
    )
    total_known = float(np.sum(init_vec))
    if total_known <= 0.0:
        print("\n--- Reaction Evolution Analysis ---")
        print("  - Skipped: no initial particles with charge in {-1,0,+1}.")
        return
    init_vec /= total_known

    reaction_model = create_reaction_model(reaction_model_cfg)
    density_sampler = getattr(reaction_model, "_density_at_positions", None)
    if not callable(density_sampler):
        print("\n--- Reaction Evolution Analysis ---")
        print("  - Skipped: selected reaction model does not provide cross-section evolution.")
        return

    isotope = getattr(reaction_model, "isotope", reaction_model_cfg.get("isotope", "H"))
    sigma = cross_sections.channel_cross_sections(avg_energy_ev, isotope=isotope)
    sigma_single = float(np.asarray(sigma[cross_sections.CH_SINGLE_STRIP]))
    sigma_double = float(np.asarray(sigma[cross_sections.CH_DOUBLE_STRIP]))
    sigma_n_to_p = float(np.asarray(sigma[cross_sections.CH_NEUTRAL_STRIP]))
    sigma_p_to_n = float(np.asarray(sigma[cross_sections.CH_CHARGE_EXCHANGE]))

    # Use the same spatial step used by the EM tracker.
    ds_target = float(em_step_length_m)
    if not np.isfinite(ds_target) or ds_target <= 0.0:
        ds_target = line_length_m
    ds_target = min(ds_target, line_length_m)
    steps = max(2, int(np.ceil(line_length_m / ds_target)) + 1)

    distance_m = np.linspace(0.0, line_length_m, steps)
    positions = start_avg[np.newaxis, :] + distance_m[:, np.newaxis] * avg_dir[np.newaxis, :]
    density_m3 = np.asarray(density_sampler(positions), dtype=np.float64)
    density_m3 = np.maximum(density_m3, 0.0)

    r_single = density_m3 * sigma_single * avg_speed_mps
    r_double = density_m3 * sigma_double * avg_speed_mps
    r_n_to_p = density_m3 * sigma_n_to_p * avg_speed_mps
    r_p_to_n = density_m3 * sigma_p_to_n * avg_speed_mps

    # Mean free path lambda = 1 / (n * sigma), reported for species-level totals.
    sigma_neg_total = sigma_single + sigma_double
    sigma_neu_total = sigma_n_to_p
    sigma_pos_total = sigma_p_to_n

    mfp_neg = np.where(density_m3 * sigma_neg_total > 0.0, 1.0 / (density_m3 * sigma_neg_total), np.inf)
    mfp_neu = np.where(density_m3 * sigma_neu_total > 0.0, 1.0 / (density_m3 * sigma_neu_total), np.inf)
    mfp_pos = np.where(density_m3 * sigma_pos_total > 0.0, 1.0 / (density_m3 * sigma_pos_total), np.inf)
    mfp_all = np.concatenate([mfp_neg, mfp_neu, mfp_pos])
    finite_mfp = mfp_all[np.isfinite(mfp_all) & (mfp_all > 0.0)]

    if finite_mfp.size > 0:
        mfp_min_m = float(np.min(finite_mfp))
        mfp_max_m = float(np.max(finite_mfp))
    else:
        mfp_min_m = float("inf")
        mfp_max_m = float("inf")

    fractions = np.zeros((steps, 3), dtype=np.float64)
    fractions[0] = init_vec

    for i in range(steps - 1):
        ds = distance_m[i + 1] - distance_m[i]
        dt = ds / avg_speed_mps
        f_neg, f_neu, f_pos = fractions[i]
        dn = -(r_single[i] + r_double[i]) * f_neg
        d0 = r_single[i] * f_neg - r_n_to_p[i] * f_neu + r_p_to_n[i] * f_pos
        dp = r_double[i] * f_neg + r_n_to_p[i] * f_neu - r_p_to_n[i] * f_pos
        next_vec = fractions[i] + dt * np.array([dn, d0, dp], dtype=np.float64)
        next_vec = np.maximum(next_vec, 0.0)
        ssum = float(np.sum(next_vec))
        if ssum > 0.0:
            next_vec /= ssum
        fractions[i + 1] = next_vec

    print("\n--- Reaction Evolution Analysis ---")
    print(f"  - Average start position: {np.array2string(start_avg, precision=6)} m")
    print(f"  - Average beam direction: {np.array2string(avg_dir, precision=6)}")
    print(f"  - Analysis line length in bbox: {line_length_m:.6e} m")
    print(f"  - Species-evolution step length (EM): {ds_target:.6e} m")
    print(f"  - Average energy for cross sections: {avg_energy_ev:.6e} eV")
    print(f"  - sigma(H-/D- -> H0/D0): {sigma_single:.6e} m^2")
    print(f"  - sigma(H-/D- -> H+/D+): {sigma_double:.6e} m^2")
    print(f"  - sigma(H0/D0 -> H+/D+): {sigma_n_to_p:.6e} m^2")
    print(f"  - sigma(H+/D+ -> H0/D0): {sigma_p_to_n:.6e} m^2")
    if np.isfinite(mfp_min_m):
        print(f"  - Mean free path range (all species transitions): [{mfp_min_m:.6e}, {mfp_max_m:.6e}] m")
        print(f"  - Step length / min mean free path: {ds_target / mfp_min_m:.6e}")
    else:
        print("  - Mean free path range (all species transitions): infinite (zero density and/or zero sigma).")
    print(
        "  - Initial fractions [H-/D-, H0/D0, H+/D+]: "
        f"[{fractions[0, 0]:.6f}, {fractions[0, 1]:.6f}, {fractions[0, 2]:.6f}]"
    )
    print(
        "  - Final fractions at bbox exit [H-/D-, H0/D0, H+/D+]: "
        f"[{fractions[-1, 0]:.6f}, {fractions[-1, 1]:.6f}, {fractions[-1, 2]:.6f}]"
    )

    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        os.makedirs(output_dir_for_run, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=120)
        ax.plot(distance_m, fractions[:, 0], label="H-/D-", linewidth=2.0)
        ax.plot(distance_m, fractions[:, 1], label="H0/D0", linewidth=2.0)
        ax.plot(distance_m, fractions[:, 2], label="H+/D+", linewidth=2.0)
        ax.set_xlabel("Distance along average beam line [m]")
        ax.set_ylabel("Species fraction")
        ax.set_title("Analytical Species Evolution at Average Source Energy")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()

        plot_path = os.path.join(output_dir_for_run, "species_evolution_average_line.png")
        fig.savefig(plot_path)
        plt.close(fig)
        print(f"  - Saved species evolution plot: {plot_path}")

        fig2, ax2 = plt.subplots(figsize=(8.0, 4.5), dpi=120)
        ax2.plot(distance_m, density_m3, color="tab:green", linewidth=2.0)
        ax2.set_xlabel("Distance along average beam line [m]")
        ax2.set_ylabel("Gas density [m^-3]")
        ax2.set_title("Gas Density Along Average Beam Line")
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()

        density_plot_path = os.path.join(output_dir_for_run, "gas_density_average_line.png")
        fig2.savefig(density_plot_path)
        plt.close(fig2)
        print(f"  - Saved gas density plot: {density_plot_path}")
    except Exception as exc:
        print(f"  - Plot generation skipped due to error: {exc}")


def run_em_prerun_analysis(
    particle_sources_list,
    external_field_cfg,
    reaction_model_cfg,
    bbox_min_corner_m,
    bbox_max_corner_m,
    output_dir_for_run,
    em_step_length_m,
):
    """Run all EM pre-run diagnostics."""
    _estimate_em_larmor_radius(
        particle_sources_list=particle_sources_list,
        external_field_cfg=external_field_cfg,
        reaction_model_cfg=reaction_model_cfg,
    )
    _analyze_reaction_species_evolution(
        particle_sources_list=particle_sources_list,
        reaction_model_cfg=reaction_model_cfg,
        bbox_min_corner_m=bbox_min_corner_m,
        bbox_max_corner_m=bbox_max_corner_m,
        output_dir_for_run=output_dir_for_run,
        em_step_length_m=em_step_length_m,
    )
