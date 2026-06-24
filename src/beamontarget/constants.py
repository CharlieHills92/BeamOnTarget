"""Shared physical constants used across BeamOnTarget."""

ATOMIC_MASS_UNIT_KG = 1.66053906660e-27
ELEMENTARY_CHARGE_C = 1.602176634e-19

# Atomic masses for the isotopes used by this codebase.
HYDROGEN_MASS_KG = 1.00784 * ATOMIC_MASS_UNIT_KG
DEUTERIUM_MASS_KG = 2.01410177812 * ATOMIC_MASS_UNIT_KG

# Common aliases for the negative-ion beam species names used in tests/configs.
H_MINUS_MASS_KG = HYDROGEN_MASS_KG
D_MINUS_MASS_KG = DEUTERIUM_MASS_KG
