"""Cross section utilities for H/D beam-gas interactions.

This module mirrors the polynomial fits currently used in the C++
implementation (cross_sections.cpp) and provides a Python API that is easy to
call from the EM tracker once particle energy and background gas density are
known.

All energies are in eV and cross sections are in m^2.
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np


# Canonical channel keys used by the tracker layer.
CH_SINGLE_STRIP = "single_strip_neg_to_neutral"   # H-/D- -> H0/D0
CH_DOUBLE_STRIP = "double_strip_neg_to_positive"  # H-/D- -> H+/D+
CH_NEUTRAL_STRIP = "strip_neutral_to_positive"    # H0/D0 -> H+/D+
CH_CHARGE_EXCHANGE = "charge_exchange_pos_to_neutral"  # H+/D+ -> H0/D0


def _to_mass_factor(isotope: str) -> float:
    """Return the mass scaling used by legacy fits (H=1, D=2)."""
    iso = str(isotope).strip().upper()
    if iso == "H":
        return 1.0
    if iso == "D":
        return 2.0
    raise ValueError(f"Unsupported isotope '{isotope}'. Use 'H' or 'D'.")


def _energy_scaled(energy_ev: np.ndarray, mass_factor: float) -> np.ndarray:
    """Apply the energy scaling used by the legacy fit tables."""
    return energy_ev / float(mass_factor)


def _safe_log10(x: np.ndarray) -> np.ndarray:
    """log10 with invalid values masked to NaN."""
    out = np.full_like(x, np.nan, dtype=np.float64)
    valid = np.isfinite(x) & (x > 0.0)
    out[valid] = np.log10(x[valid])
    return out


def _clean_sigma(values: np.ndarray) -> np.ndarray:
    """Clamp invalid/negative fitted values to zero."""
    out = np.asarray(values, dtype=np.float64)
    out[~np.isfinite(out)] = 0.0
    out[out < 0.0] = 0.0
    return out


def cs_hm_single_strip(energy_ev, isotope: str = "H"):
    """H-/D- single stripping cross section: H-/D- -> H0/D0."""
    e = np.asarray(energy_ev, dtype=np.float64)
    ms = _to_mass_factor(isotope)
    es = _energy_scaled(e, ms)
    lg10e = _safe_log10(es)

    sigma = np.zeros_like(es, dtype=np.float64)
    mask = np.isfinite(lg10e) & (es > 2.4)
    l = lg10e[mask]
    sigma[mask] = np.power(
        10.0,
        (
            -0.0013 * l**6
            + 0.0313 * l**5
            - 0.2912 * l**4
            + 1.2932 * l**3
            - 2.8823 * l**2
            + 3.276 * l
            - 20.892
        ),
    )
    sigma = _clean_sigma(sigma)
    return sigma.item() if np.ndim(energy_ev) == 0 else sigma


def cs_hm_double_strip(energy_ev, isotope: str = "H"):
    """H-/D- double stripping cross section: H-/D- -> H+/D+."""
    e = np.asarray(energy_ev, dtype=np.float64)
    ms = _to_mass_factor(isotope)
    es = _energy_scaled(e, ms)
    lg10e = _safe_log10(es)

    sigma = np.zeros_like(es, dtype=np.float64)
    mask = np.isfinite(lg10e) & (es > 1.0e3)
    l = lg10e[mask]
    sigma[mask] = np.power(
        10.0,
        (
            -0.010114 * l**6
            + 0.303523 * l**5
            - 3.711695 * l**4
            + 23.674607 * l**3
            - 83.406121 * l**2
            + 154.867111 * l
            - 138.733509
            - 1.0
        ),
    )
    sigma = _clean_sigma(sigma)
    return sigma.item() if np.ndim(energy_ev) == 0 else sigma


def cs_proj_ionization_h0(energy_ev, isotope: str = "H"):
    """Neutral stripping cross section: H0/D0 -> H+/D+."""
    e = np.asarray(energy_ev, dtype=np.float64)
    ms = _to_mass_factor(isotope)
    es = _energy_scaled(e, ms)
    lg10e = _safe_log10(es)

    sigma = np.zeros_like(es, dtype=np.float64)
    mask = np.isfinite(lg10e) & (es > 0.0)
    l = lg10e[mask]
    sigma[mask] = np.power(
        10.0,
        (
            0.000186 * l**6
            - 0.0004 * l**5
            - 0.055208 * l**4
            + 0.73408 * l**3
            - 4.288931 * l**2
            + 12.812774 * l
            - 34.752437
            - 1.0
        ),
    )
    sigma = _clean_sigma(sigma)
    return sigma.item() if np.ndim(energy_ev) == 0 else sigma


def cs_cx_hp(energy_ev, isotope: str = "H"):
    """Charge exchange cross section: H+/D+ -> H0/D0."""
    e = np.asarray(energy_ev, dtype=np.float64)
    ms = _to_mass_factor(isotope)
    es = _energy_scaled(e, ms)
    lg10e = _safe_log10(es)

    sigma = np.zeros_like(es, dtype=np.float64)
    mask = np.isfinite(lg10e) & (es > 0.0)
    l = lg10e[mask]
    sigma[mask] = np.power(
        10.0,
        (
            1.26307e-02 * l**6
            - 2.57671e-01 * l**5
            + 2.04941e00 * l**4
            - 8.16387e00 * l**3
            + 1.67493e01 * l**2
            - 1.47594e01 * l
            - 1.80339e01
        ),
    )
    sigma = _clean_sigma(sigma)
    return sigma.item() if np.ndim(energy_ev) == 0 else sigma


def channel_cross_sections(energy_ev, isotope: str = "H") -> Dict[str, np.ndarray]:
    """Return all requested beam-gas channels for H/D isotopes."""
    return {
        CH_SINGLE_STRIP: cs_hm_single_strip(energy_ev, isotope=isotope),
        CH_DOUBLE_STRIP: cs_hm_double_strip(energy_ev, isotope=isotope),
        CH_NEUTRAL_STRIP: cs_proj_ionization_h0(energy_ev, isotope=isotope),
        CH_CHARGE_EXCHANGE: cs_cx_hp(energy_ev, isotope=isotope),
    }

def channel_rates_s(energy_ev, speed_mps, background_density_m3, isotope: str = "H"):
    """Return per-channel reaction rates [1/s] as n*sigma*v.

    Args:
        energy_ev: Particle kinetic energy in eV (scalar or numpy array).
        speed_mps: Particle speed in m/s (scalar or numpy array).
        background_density_m3: Background gas density in m^-3.
        isotope: "H" or "D".
    """
    density = np.asarray(background_density_m3, dtype=np.float64)
    if np.any(density < 0.0) or not np.all(np.isfinite(density)):
        raise ValueError("background_density_m3 must be finite and >= 0")

    speed = np.asarray(speed_mps, dtype=np.float64)
    speed = np.maximum(speed, 0.0)

    sigmas = channel_cross_sections(energy_ev, isotope=isotope)
    return {k: np.asarray(v, dtype=np.float64) * speed * density for k, v in sigmas.items()}


def channel_probabilities(energy_ev, speed_mps, dt_s, background_density_m3, isotope: str = "H"):
    """Return per-step channel probabilities from rates.

    Probability model: p = 1 - exp(-rate * dt).
    dt_s may be a scalar or an array with one value per particle.
    """
    dt = np.asarray(dt_s, dtype=np.float64)
    if np.any(dt < 0.0) or not np.all(np.isfinite(dt)):
        raise ValueError("dt_s must be finite and >= 0")

    rates = channel_rates_s(
        energy_ev=energy_ev,
        speed_mps=speed_mps,
        background_density_m3=background_density_m3,
        isotope=isotope,
    )
    return {k: 1.0 - np.exp(-np.asarray(v, dtype=np.float64) * dt) for k, v in rates.items()}
