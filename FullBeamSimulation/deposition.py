# deposition.py
"""
Per-species power and current accumulator on mesh faces.

Maintains separate arrays for each (species / charge_state) and provides
combined totals for output.  Thread-safe accumulation via numpy add.at().
"""
import numpy as np


# ---------------------------------------------------------------------------
# Species labels
# ---------------------------------------------------------------------------
SPECIES_LABELS = {
    -1: "D_minus",
     0: "D_zero",
    +1: "D_plus",
    -2: "electron",   # secondary electrons if tracked
}


class Depositor:
    """
    Accumulates deposited power [W] and current [A] per mesh face,
    separately for each charge state.

    Usage:
        dep = Depositor(face_counts, face_offsets)
        dep.deposit(global_face_indices, charge_states, powers_W, currents_A)
        ...
        power_by_species, current_by_species = dep.get_results()
    """

    def __init__(self, face_counts, face_offsets, deposition_fraction_fn=None):
        """
        Args:
            face_counts:  list of int — number of faces per object.
            face_offsets: 1-D int array — cumulative face offset per object.
            deposition_fraction_fn: callable(energy_eV) → fraction [0,1].
                If None, all kinetic energy is deposited.
        """
        self.face_counts = list(face_counts)
        self.face_offsets = np.asarray(face_offsets)
        self.num_objects = len(face_counts)
        self._dep_frac = deposition_fraction_fn

        # species we track:  keyed by charge_state
        self._species = {}

    # ------------------------------------------------------------------
    def _ensure_species(self, charge_state):
        if charge_state not in self._species:
            self._species[charge_state] = {
                'power': [np.zeros(c, dtype=np.float64) for c in self.face_counts],
                'current': [np.zeros(c, dtype=np.float64) for c in self.face_counts],
                'hit_count': [0] * self.num_objects,
            }

    # ------------------------------------------------------------------
    def deposit(self, global_face_indices, charge_states,
                energies_eV, currents_A):
        """
        Accumulate power and current for a batch of particle hits.

        Args:
            global_face_indices: (M,) int — face index in the concatenated scene mesh.
            charge_states:       (M,) int — charge state of each hitting particle.
            energies_eV:         (M,) float — kinetic energy [eV].
            currents_A:          (M,) float — current per macro-particle [A].
        """
        if len(global_face_indices) == 0:
            return

        global_face_indices = np.asarray(global_face_indices, dtype=np.int64)
        charge_states = np.asarray(charge_states, dtype=int)
        energies_eV = np.asarray(energies_eV, dtype=np.float64)
        currents_A = np.asarray(currents_A, dtype=np.float64)

        # Convert energy → deposited power
        if self._dep_frac is not None:
            frac = self._dep_frac(energies_eV)
        else:
            frac = 1.0
        powers_W = energies_eV * currents_A * frac

        # Determine which object each face belongs to
        obj_indices = np.searchsorted(self.face_offsets, global_face_indices,
                                      side='right') - 1
        local_faces = global_face_indices - self.face_offsets[obj_indices]

        # Group by species
        for q in np.unique(charge_states):
            self._ensure_species(int(q))
            mask = charge_states == q
            obj_idx_q = obj_indices[mask]
            loc_face_q = local_faces[mask]
            power_q = powers_W[mask]
            current_q = currents_A[mask]

            for oi in np.unique(obj_idx_q):
                omask = obj_idx_q == oi
                fi = loc_face_q[omask]
                np.add.at(self._species[int(q)]['power'][oi], fi, power_q[omask])
                np.add.at(self._species[int(q)]['current'][oi], fi, current_q[omask])
                self._species[int(q)]['hit_count'][oi] += int(omask.sum())

    # ------------------------------------------------------------------
    def get_results(self):
        """
        Returns:
            power_by_species:  dict {charge_state: [np.array per object]}
            current_by_species: dict {charge_state: [np.array per object]}
        """
        power_out, current_out = {}, {}
        for q, data in self._species.items():
            power_out[q] = data['power']
            current_out[q] = data['current']
        return power_out, current_out

    def get_total_power(self):
        """Return list of total (all-species) deposited power per object."""
        totals = [np.zeros(c, dtype=np.float64) for c in self.face_counts]
        for _q, data in self._species.items():
            for oi in range(self.num_objects):
                totals[oi] += data['power'][oi]
        return totals

    def get_total_current(self):
        """Return list of total (all-species) deposited current per object."""
        totals = [np.zeros(c, dtype=np.float64) for c in self.face_counts]
        for _q, data in self._species.items():
            for oi in range(self.num_objects):
                totals[oi] += data['current'][oi]
        return totals

    def summary(self):
        """Print a brief summary of deposited totals."""
        print("\n--- Deposition Summary ---")
        for q in sorted(self._species.keys()):
            label = SPECIES_LABELS.get(q, f"q={q}")
            total_p = sum(arr.sum() for arr in self._species[q]['power'])
            total_c = sum(arr.sum() for arr in self._species[q]['current'])
            total_h = sum(self._species[q]['hit_count'])
            print(f"  {label:12s}: {total_p:12.2f} W   {total_c:12.4e} A   "
                  f"{total_h:10d} hits")
        # Grand total
        grand_p = sum(arr.sum() for arr in self.get_total_power())
        grand_c = sum(arr.sum() for arr in self.get_total_current())
        print(f"  {'TOTAL':12s}: {grand_p:12.2f} W   {grand_c:12.4e} A")
        print()
