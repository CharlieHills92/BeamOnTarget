# background.py
"""
Background gas density and plasma profile interpolators for the
null-collision Monte-Carlo engine.

Profiles can be loaded from:
  * CSV files with columns  x, y, z, value       (3-D profile)
  * CSV files with columns  x, value              (1-D profile along x)
  * NPY files of shape (N, 4) or (N, 2)
  * Or constructed analytically (uniform constant)

If no file is provided, the profile returns zero everywhere (= no interactions).

For 1-D profiles the density depends only on the x-coordinate of each
particle; y and z are ignored.  This is the typical setup for a beam-line
gas density that varies along the beam axis.
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator


class ScalarProfile:
    """Interpolated scalar field  f(x, y, z)."""

    def __init__(self, filepath=None, uniform_value=None, label="profile"):
        """
        Args:
            filepath: CSV or .npy with columns (x, y, z, value) or (x, value).
            uniform_value: if set, ignore file and return this constant.
            label: human-readable name for logging.
        """
        self.label = label
        self._uniform = uniform_value
        self._interp = None
        self._interp_1d = None
        self._kind = None
        self._max_value = 0.0

        if uniform_value is not None:
            self._max_value = float(uniform_value)
        elif filepath is not None:
            self._load(filepath)

    # ------------------------------------------------------------------
    def _load(self, filepath):
        if filepath.endswith('.npy'):
            data = np.load(filepath)
        else:
            data = np.loadtxt(filepath, delimiter=',', skiprows=1)

        ncols = data.shape[1]

        if ncols == 2:
            # ---- 1-D profile: (x, value) ----
            from scipy.interpolate import interp1d

            x = data[:, 0]
            vals = data[:, 1]
            self._max_value = float(np.max(vals))

            # Linear interpolation; return 0 outside the data range
            self._interp_1d = interp1d(
                x, vals, kind='linear', bounds_error=False, fill_value=0.0)
            self._kind = '1d'
            print(f"  Loaded 1-D {self.label} ({len(x)} points, "
                  f"x=[{x.min():.2f}..{x.max():.2f}] m) from '{filepath}'")

        elif ncols >= 4:
            # ---- 3-D profile: (x, y, z, value) ----
            pts = data[:, :3]
            vals = data[:, 3]
            self._max_value = float(np.max(vals))

            ux = np.unique(pts[:, 0])
            uy = np.unique(pts[:, 1])
            uz = np.unique(pts[:, 2])

            if len(ux) * len(uy) * len(uz) == len(pts):
                nx, ny, nz = len(ux), len(uy), len(uz)
                grid_vals = vals.reshape(nx, ny, nz)
                self._interp = RegularGridInterpolator(
                    (ux, uy, uz), grid_vals,
                    bounds_error=False, fill_value=0.0)
                self._kind = 'regular'
                print(f"  Loaded regular-grid {self.label} "
                      f"({nx}×{ny}×{nz}) from '{filepath}'")
            else:
                self._interp = LinearNDInterpolator(pts, vals, fill_value=0.0)
                self._kind = 'scattered'
                print(f"  Loaded scattered {self.label} "
                      f"({len(pts)} pts) from '{filepath}'")
        else:
            raise ValueError(
                f"Gas profile '{filepath}' has {ncols} columns; "
                f"expected 2 (x, value) or 4+ (x, y, z, value).")

    # ------------------------------------------------------------------
    def __call__(self, positions):
        """
        Evaluate profile at (N, 3) positions.  Returns (N,) array.
        """
        positions = np.asarray(positions, dtype=np.float64)
        if positions.ndim == 1:
            positions = positions[np.newaxis, :]
        n = positions.shape[0]

        if self._uniform is not None:
            return np.full(n, self._uniform)

        if self._kind == '1d':
            # Interpolate along x only (column 0)
            return self._interp_1d(positions[:, 0])

        if self._interp is None:
            return np.zeros(n)

        result = self._interp(positions)
        return np.nan_to_num(result, nan=0.0)

    @property
    def max_value(self):
        return self._max_value


class GasDensityProfile(ScalarProfile):
    """
    Gas number density n_gas(x, y, z) in m⁻³.
    Used by the null-collision engine to determine interaction probability.
    """
    def __init__(self, filepath=None, uniform_value=None):
        super().__init__(filepath=filepath, uniform_value=uniform_value,
                         label="gas density [m⁻³]")
