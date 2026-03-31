# cross_sections.py
"""
Reaction cross-section database for null-collision Monte-Carlo.

Each reaction is represented by a callable  σ(E_eV) → cross-section [m²]
where E_eV is the kinetic energy of the projectile in eV.

Built-in reactions (log-linear interpolation of ORNL / Barnett / ADAS data):

  D⁻ + D₂  →  D⁰  + e⁻ + D₂     single stripping   σ_m10
  D⁻ + D₂  →  D⁺  + 2e⁻ + D₂    double stripping   σ_m1p1
  D⁰ + D₂  →  D⁺  + e⁻ + D₂     ionisation         σ_0p1
  D⁰ + D₂  →  D⁻  + D₂⁺          electron capture   σ_0m1
  D⁺ + D₂  →  D⁰  + D₂⁺          charge exchange    σ_p10

Users can add custom reactions by sub-classing CrossSection.
"""
import numpy as np
from scipy.interpolate import interp1d


class CrossSection:
    """Base cross-section: log-linear interpolation of tabulated data."""

    def __init__(self, energy_eV, sigma_m2, label="custom"):
        """
        Args:
            energy_eV: 1-D array of energies [eV], must be monotonically increasing.
            sigma_m2:  1-D array of cross-sections [m²] at those energies.
            label:     human-readable name.
        """
        self.label = label
        self._log_E = np.log10(np.asarray(energy_eV, dtype=np.float64))
        self._log_s = np.log10(np.clip(np.asarray(sigma_m2, dtype=np.float64),
                                       1e-30, None))
        self._interp = interp1d(self._log_E, self._log_s,
                                kind='linear', bounds_error=False,
                                fill_value=(self._log_s[0], self._log_s[-1]))

    def __call__(self, energy_eV):
        """Evaluate σ(E) for scalar or array input.  Returns m²."""
        E = np.asarray(energy_eV, dtype=np.float64)
        return 10.0 ** self._interp(np.log10(np.clip(E, 1.0, None)))

    def __repr__(self):
        return f"CrossSection('{self.label}')"


# ---------------------------------------------------------------------------
#  Built-in H/H₂ cross-sections  (approximate fits from ORNL-6086 / Barnett)
#  Energy range: 1 keV – 10 MeV  (total kinetic energy for H projectiles,
#  i.e. 1 amu — equivalent to energy-per-amu for these tables).
#
#  For projectiles heavier than 1 amu (e.g. D at 2 amu), the caller must
#  convert total energy → energy/amu before looking up cross-sections.
#  The nullcoll module handles this via the mass_kg array.
# ---------------------------------------------------------------------------

# Tabulated energies in eV (= total energy for 1 amu projectile = energy/amu)
_E_tab = np.array([
    1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 2e5,
    5e5, 8.7e5, 1e6, 2e6, 5e6, 1e7
])

# D⁻ + D₂ → D⁰ + e⁻ + D₂   (single electron detachment / stripping)
_sig_m10 = np.array([
    2.0e-20, 5.0e-20, 1.5e-19, 2.5e-19, 3.0e-19, 3.2e-19, 3.0e-19, 2.5e-19,
    1.5e-19, 1.0e-19, 8.0e-20, 4.0e-20, 1.0e-20, 3.0e-21
])

# D⁻ + D₂ → D⁺ + 2e⁻ + D₂  (double stripping)
_sig_m1p1 = np.array([
    1.0e-21, 3.0e-21, 1.5e-20, 4.0e-20, 7.0e-20, 1.0e-19, 1.1e-19, 1.0e-19,
    6.0e-20, 3.5e-20, 2.8e-20, 1.2e-20, 3.0e-21, 8.0e-22
])

# D⁰ + D₂ → D⁺ + e⁻ + D₂   (electron-loss ionisation of neutral)
_sig_0p1 = np.array([
    1.0e-22, 5.0e-22, 5.0e-21, 2.0e-20, 4.0e-20, 6.0e-20, 5.5e-20, 4.5e-20,
    2.5e-20, 1.5e-20, 1.2e-20, 5.0e-21, 1.5e-21, 5.0e-22
])

# D⁰ + D₂ → D⁻ + D₂⁺       (electron capture — negligible above ~100 keV)
_sig_0m1 = np.array([
    5.0e-21, 3.0e-21, 1.0e-21, 2.0e-22, 2.0e-23, 5.0e-25, 1.0e-26, 1.0e-28,
    1.0e-30, 1.0e-30, 1.0e-30, 1.0e-30, 1.0e-30, 1.0e-30
])

# D⁺ + D₂ → D⁰ + D₂⁺       (charge exchange / neutralisation)
_sig_p10 = np.array([
    3.0e-19, 2.0e-19, 8.0e-20, 3.0e-20, 1.0e-20, 2.0e-21, 5.0e-22, 1.0e-22,
    1.0e-23, 3.0e-24, 1.5e-24, 3.0e-25, 3.0e-26, 5.0e-27
])


# Pre-built cross-section objects
sigma_m10  = CrossSection(_E_tab, _sig_m10,  "D⁻→D⁰  single strip")
sigma_m1p1 = CrossSection(_E_tab, _sig_m1p1, "D⁻→D⁺  double strip")
sigma_0p1  = CrossSection(_E_tab, _sig_0p1,  "D⁰→D⁺  ionisation")
sigma_0m1  = CrossSection(_E_tab, _sig_0m1,  "D⁰→D⁻  e⁻ capture")
sigma_p10  = CrossSection(_E_tab, _sig_p10,  "D⁺→D⁰  CX")


# ---------------------------------------------------------------------------
#  Reaction registry  (maps  (from_charge, to_charge) → CrossSection)
# ---------------------------------------------------------------------------
# Charge states: -1 = D⁻,  0 = D⁰,  +1 = D⁺

REACTIONS = {
    (-1,  0): sigma_m10,    # single stripping
    (-1, +1): sigma_m1p1,   # double stripping
    ( 0, +1): sigma_0p1,    # ionisation
    ( 0, -1): sigma_0m1,    # electron capture
    (+1,  0): sigma_p10,    # charge exchange
}


def total_cross_section(charge_state, energy_eV):
    """Sum of all outgoing cross-sections for a given incoming charge state."""
    total = 0.0
    for (q_from, _q_to), sigma in REACTIONS.items():
        if q_from == charge_state:
            total += sigma(energy_eV)
    return total


def max_total_cross_section(charge_state, energy_range_eV, n_samples=200):
    """
    Estimate the maximum total cross-section for a charge state over an
    energy range.  Used by the null-collision method to set σ_max.
    """
    energies = np.geomspace(max(energy_range_eV[0], 1.0),
                            energy_range_eV[1], n_samples)
    sigma_tot = np.zeros_like(energies)
    for (q_from, _), sig in REACTIONS.items():
        if q_from == charge_state:
            sigma_tot += sig(energies)
    return float(np.max(sigma_tot))
