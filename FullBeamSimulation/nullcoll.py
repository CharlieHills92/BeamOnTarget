# nullcoll.py
"""
Null-collision Monte-Carlo module for particle–background-gas interactions.

This is NOT a standalone engine.  It is called by engine_boris to apply
gas-phase species-changing collisions BEFORE (or interleaved with) the
Boris EM stepping.

Supported reactions (D in D₂ background):

  D⁻ → D⁰  (single stripping)
  D⁻ → D⁺  (double stripping)
  D⁰ → D⁺  (ionisation)
  D⁰ → D⁻  (electron capture)
  D⁺ → D⁰  (charge exchange)

The null-collision method adds a fictitious "null" reaction so that the
total collision frequency is constant (= σ_max × n_gas_max × v).  Between
real collisions the particle flies freely; at each candidate event we roll
to decide real vs null.

Optionally, secondary electrons produced by ionisation / stripping events
can be generated and returned for downstream tracking.

Usage
-----
    from nullcoll import apply_gas_interactions
    result = apply_gas_interactions(pos, dirs, energy_eV, current,
                                   mass_kg, charge_e, gas_profile,
                                   track_electrons=False, seed=0)
    # result is a NullCollResult namedtuple
"""
import numpy as np
from collections import namedtuple

import cross_sections as xsec
from cross_sections import REACTIONS

_E_CHARGE = 1.602176634e-19  # C
_ME_KG    = 9.1093837015e-31  # electron mass

# Return type ----------------------------------------------------------------
NullCollResult = namedtuple("NullCollResult", [
    "pos",          # (N, 3) updated positions
    "dirs",         # (N, 3) directions (unchanged for ballistic transport)
    "energy_eV",    # (N,)   energies (unchanged — elastic approx)
    "current",      # (N,)   currents
    "mass_kg",      # (N,)   masses
    "charge_e",     # (N,)   updated integer charge states
    "alive",        # (N,)   bool — particles still alive after gas region
    # Secondary electrons (empty arrays when track_electrons=False)
    "sec_origins",  # (K, 3)
    "sec_dirs",     # (K, 3)
    "sec_energies", # (K,)
    "sec_currents", # (K,)
    "sec_charges",  # (K,)   all −2 by convention
])


def _precompute_sigma_max(energy_range=(100, 1e7)):
    """Return dict {charge_state: σ_max} over the given energy range."""
    sigma_max = {}
    for q in [-1, 0, +1]:
        sm = xsec.max_total_cross_section(q, energy_range)
        sigma_max[q] = sm
        if sm > 0:
            print(f"  [nullcoll] σ_max(q={q:+d}) = {sm:.3e} m²")
    return sigma_max


def apply_gas_interactions(pos, dirs, energy_eV, current, mass_kg, charge_e,
                           gas_profile, *, track_electrons=False, seed=0,
                           max_events=100):
    """
    Apply null-collision MC gas interactions to a batch of particles.

    Particles are advanced ballistically (straight-line, no fields) through
    the gas region.  Their charge state may change due to stripping /
    ionisation / capture reactions.

    Parameters
    ----------
    pos : (N, 3) float64       — starting positions [m]
    dirs : (N, 3) float64      — unit direction vectors
    energy_eV : (N,) float64   — kinetic energies [eV]
    current : (N,) float64     — per-particle currents [A]
    mass_kg : (N,) float64     — particle masses [kg]
    charge_e : (N,) int        — charge states in units of e
    gas_profile : GasDensityProfile callable
    track_electrons : bool      — generate secondary electrons?
    seed : int                  — RNG seed for reproducibility
    max_events : int            — safety cap on MC events per particle

    Returns
    -------
    NullCollResult  (namedtuple — see module docstring)
    """
    np.random.seed(seed)

    _AMU = 1.66053906660e-27  # kg

    N = len(pos)
    pos = pos.copy()
    charge_e = charge_e.copy().astype(int)

    # Energy per amu for cross-section lookup
    # (tables are parameterised as energy/amu; for H at 1 amu this = total E)
    mass_amu = mass_kg / _AMU
    energy_per_amu_eV = energy_eV / mass_amu

    # Speed from kinetic energy
    speed = np.sqrt(2.0 * energy_eV * _E_CHARGE / mass_kg)  # m/s

    # Pre-compute σ_max per species
    sigma_max_per_species = _precompute_sigma_max()

    # Secondary electron collectors
    sec_origins, sec_dirs, sec_energies, sec_currents, sec_charges = \
        [], [], [], [], []

    alive = np.ones(N, dtype=bool)
    in_mc = np.ones(N, dtype=bool)   # still inside gas region (MC active)
    n_gas_max = gas_profile.max_value

    if n_gas_max > 0:
        for _evt in range(max_events):
            n_active = in_mc.sum()
            if n_active == 0:
                break

            idx = np.where(in_mc)[0]

            # σ_max for each active particle's current charge state (vectorized)
            sig_max = np.zeros(n_active)
            for q_val, sm in sigma_max_per_species.items():
                qmask = charge_e[idx] == q_val
                sig_max[qmask] = sm

            v = speed[idx]
            nu_null = n_gas_max * sig_max * v
            nu_null = np.clip(nu_null, 1e-20, None)

            # Exponential time to next candidate event
            dt = -np.log(np.random.random(n_active)) / nu_null

            # Advance position ballistically
            pos[idx] += dirs[idx] * (v * dt)[:, None]

            # Local gas density at new position
            n_local = gas_profile(pos[idx])
            outside = n_local <= 0

            # Particles that exited the gas are done with MC
            # (they remain alive for Boris, but leave the MC loop)
            if outside.sum() > 0:
                in_mc[idx[outside]] = False

            # For particles still in gas: real or null collision?
            in_gas = ~outside
            if in_gas.sum() > 0:
                idx_in = idx[in_gas]
                n_loc = n_local[in_gas]

                # Vectorized σ_real: group by charge state
                sigma_real = np.zeros(len(idx_in))
                for q_val in [-1, 0, +1]:
                    qmask = charge_e[idx_in] == q_val
                    if qmask.sum() > 0:
                        sigma_real[qmask] = xsec.total_cross_section(
                            q_val, energy_per_amu_eV[idx_in[qmask]])

                prob_real = sigma_real * n_loc / (sig_max[in_gas] * n_gas_max)
                prob_real = np.clip(prob_real, 0, 1)

                roll = np.random.random(len(idx_in))
                real_event = roll < prob_real

                if real_event.sum() > 0:
                    # Vectorized channel selection per charge state
                    re_idx = idx_in[real_event]
                    re_E_per_amu = energy_per_amu_eV[re_idx]
                    re_charges = charge_e[re_idx]

                    for q_val in [-1, 0, +1]:
                        qmask = re_charges == q_val
                        if qmask.sum() == 0:
                            continue

                        # Get channels for this from-charge
                        ch_list = [(qt, sig_fn) for (qf, qt), sig_fn
                                   in REACTIONS.items() if qf == q_val]
                        if not ch_list:
                            continue

                        sub_idx = re_idx[qmask]
                        sub_E = re_E_per_amu[qmask]
                        n_sub = len(sub_idx)

                        # Evaluate all channel cross-sections (vectorized)
                        ch_sigmas = np.zeros((len(ch_list), n_sub))
                        ch_targets = []
                        for k, (qt, sig_fn) in enumerate(ch_list):
                            ch_sigmas[k] = sig_fn(sub_E)
                            ch_targets.append(qt)

                        totals = ch_sigmas.sum(axis=0)
                        totals = np.clip(totals, 1e-30, None)
                        ch_probs = ch_sigmas / totals  # (n_channels, n_sub)

                        # Cumulative probabilities for vectorized selection
                        cum_probs = np.cumsum(ch_probs, axis=0)  # (n_ch, n_sub)
                        rolls = np.random.random(n_sub)

                        # Select channel: first cumulative bin that exceeds roll
                        for k, qt in enumerate(ch_targets):
                            selected = (rolls < cum_probs[k])
                            if k > 0:
                                selected &= (rolls >= cum_probs[k - 1])
                            if selected.sum() > 0:
                                old_q = charge_e[sub_idx[selected]]
                                charge_e[sub_idx[selected]] = qt

                                # Secondary electrons
                                if track_electrons and qt > q_val:
                                    n_e = qt - q_val
                                    for ii in sub_idx[selected]:
                                        for _ne in range(n_e):
                                            phi = np.random.uniform(0, 2*np.pi)
                                            cos_th = np.random.uniform(-1, 1)
                                            sin_th = np.sqrt(1 - cos_th**2)
                                            e_dir = np.array([
                                                sin_th * np.cos(phi),
                                                sin_th * np.sin(phi),
                                                cos_th])
                                            sec_origins.append(pos[ii].copy())
                                            sec_dirs.append(e_dir)
                                            sec_energies.append(
                                                np.random.exponential(5.0))
                                            sec_currents.append(current[ii])
                                            sec_charges.append(-2)

    # Build secondary arrays (may be empty)
    if sec_origins:
        s_o = np.array(sec_origins)
        s_d = np.array(sec_dirs)
        s_e = np.array(sec_energies)
        s_c = np.array(sec_currents)
        s_q = np.array(sec_charges, dtype=int)
    else:
        s_o = np.empty((0, 3))
        s_d = np.empty((0, 3))
        s_e = np.empty(0)
        s_c = np.empty(0)
        s_q = np.empty(0, dtype=int)

    return NullCollResult(
        pos=pos, dirs=dirs, energy_eV=energy_eV, current=current,
        mass_kg=mass_kg, charge_e=charge_e, alive=alive,
        sec_origins=s_o, sec_dirs=s_d, sec_energies=s_e,
        sec_currents=s_c, sec_charges=s_q)
