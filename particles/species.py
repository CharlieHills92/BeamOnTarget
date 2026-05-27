"""Species state container used by EM particle tracking."""

from dataclasses import dataclass
import numpy as np
from particles.constants import ELEMENTARY_CHARGE_C



@dataclass
class SpeciesFrame:
    """Per-particle species properties."""

    mass_kg: np.ndarray
    charge_state_e: np.ndarray

    def q_over_m(self):
        """Charge-to-mass ratio in SI units (C/kg)."""
        return (self.charge_state_e * ELEMENTARY_CHARGE_C) / np.maximum(self.mass_kg, 1e-30)


def build_species_frame(mass_kg, charge_state_e):
    """Create a SpeciesFrame from arrays."""
    return SpeciesFrame(
        mass_kg=np.asarray(mass_kg, dtype=np.float64),
        charge_state_e=np.asarray(charge_state_e, dtype=np.int32),
    )
