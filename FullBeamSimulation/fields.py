# fields.py
"""
Static electric and magnetic field interpolators.

Field data is loaded from CSV or NPY files with columns:
    x, y, z, Fx, Fy, Fz
and interpolated onto arbitrary query points using
scipy.interpolate.RegularGridInterpolator (fast) when the grid is regular, or
scipy.interpolate.LinearNDInterpolator (slower) for scattered data.

If no file is provided, the field returns zero everywhere.
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator


class VectorField:
    """Interpolated 3-D vector field  F(x, y, z) → (Fx, Fy, Fz)."""

    def __init__(self, filepath=None, scale=1.0):
        """
        Args:
            filepath: path to CSV or .npy file.
                CSV must have columns x, y, z, Fx, Fy, Fz.
                NPY must be shape (N, 6) in the same order.
            scale:  multiplicative factor applied to the field values.
        """
        self._interp = None
        self._scale = scale
        if filepath is not None:
            self._load(filepath)

    # ------------------------------------------------------------------
    def _load(self, filepath):
        if filepath.endswith('.npy'):
            data = np.load(filepath)
        else:
            data = np.loadtxt(filepath, delimiter=',', skiprows=1)

        pts = data[:, :3]
        vals = data[:, 3:6] * self._scale

        # Try to detect a regular grid
        ux = np.unique(pts[:, 0])
        uy = np.unique(pts[:, 1])
        uz = np.unique(pts[:, 2])
        if len(ux) * len(uy) * len(uz) == len(pts):
            # Regular grid — much faster
            nx, ny, nz = len(ux), len(uy), len(uz)
            Fx = vals[:, 0].reshape(nx, ny, nz)
            Fy = vals[:, 1].reshape(nx, ny, nz)
            Fz = vals[:, 2].reshape(nx, ny, nz)
            self._interp = [
                RegularGridInterpolator((ux, uy, uz), Fx,
                                        bounds_error=False, fill_value=0.0),
                RegularGridInterpolator((ux, uy, uz), Fy,
                                        bounds_error=False, fill_value=0.0),
                RegularGridInterpolator((ux, uy, uz), Fz,
                                        bounds_error=False, fill_value=0.0),
            ]
            self._kind = 'regular'
            print(f"  Loaded regular-grid field ({nx}×{ny}×{nz}) from '{filepath}'")
        else:
            # Scattered data
            self._interp = LinearNDInterpolator(pts, vals, fill_value=0.0)
            self._kind = 'scattered'
            print(f"  Loaded scattered field ({len(pts)} points) from '{filepath}'")

    # ------------------------------------------------------------------
    def __call__(self, positions):
        """
        Evaluate the field at an array of positions.

        Args:
            positions: (N, 3) array of (x, y, z) coordinates.

        Returns:
            (N, 3) array of (Fx, Fy, Fz).
        """
        positions = np.asarray(positions, dtype=np.float64)
        if positions.ndim == 1:
            positions = positions[np.newaxis, :]

        if self._interp is None:
            return np.zeros_like(positions)

        if self._kind == 'regular':
            Fx = self._interp[0](positions)
            Fy = self._interp[1](positions)
            Fz = self._interp[2](positions)
            return np.column_stack([Fx, Fy, Fz])
        else:
            result = self._interp(positions)
            return np.nan_to_num(result, nan=0.0)


class ElectricField(VectorField):
    """E(x, y, z) in V/m."""
    pass


class MagneticField(VectorField):
    """B(x, y, z) in Tesla."""
    pass
