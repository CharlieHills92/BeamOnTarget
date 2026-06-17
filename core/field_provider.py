"""External electromagnetic field providers for particle tracking."""

import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d


class ExternalFieldProvider:
    """Base interface for external E/B field providers."""

    def sample(self, positions_m, time_s):
        """Return electric and magnetic fields for particle positions.

        Args:
            positions_m: Particle positions, shape (N, 3).
            time_s: Particle times, shape (N,) or scalar.

        Returns:
            (electric_field, magnetic_field): two arrays of shape (N, 3).
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
#  Grid field component — loads a single scalar field from a .fld CSV
# ---------------------------------------------------------------------------

class GridFieldComponent:
    """Single scalar field loaded from a 4-column CSV (.fld) file.

    File format:  x, y, z, value  (one row per grid point, comma-separated).
    The grid must be rectilinear (structured) but spacing need not be uniform.
    Points outside the grid are extrapolated to zero.
    """

    def __init__(self, filepath):
        resolved = filepath
        if not os.path.isabs(filepath):
            resolved = os.path.join(os.path.dirname(__file__), filepath)
        # Skip header rows that cannot be parsed as floats
        skip = 0
        with open(resolved, 'r') as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith('#'):
                    skip += 1
                    continue
                try:
                    float(line.split(',')[0])
                except ValueError:
                    skip += 1
                    continue
                break
        data = np.loadtxt(resolved, delimiter=',', dtype=np.float64,
                          skiprows=skip)
        if data.ndim != 2 or data.shape[1] < 4:
            raise ValueError(f"Expected 4 columns (x,y,z,value) in {filepath}, "
                             f"got shape {data.shape}")

        xs = np.sort(np.unique(data[:, 0]))
        ys = np.sort(np.unique(data[:, 1]))
        zs = np.sort(np.unique(data[:, 2]))
        expected = len(xs) * len(ys) * len(zs)
        if data.shape[0] != expected:
            raise ValueError(
                f"Grid in {filepath} is not rectilinear: "
                f"{len(xs)}×{len(ys)}×{len(zs)} = {expected} expected, "
                f"got {data.shape[0]} rows")

        # Build 3-D value array in (nx, ny, nz) order
        xi = np.searchsorted(xs, data[:, 0])
        yi = np.searchsorted(ys, data[:, 1])
        zi = np.searchsorted(zs, data[:, 2])
        values = np.empty((len(xs), len(ys), len(zs)), dtype=np.float64)
        values[xi, yi, zi] = data[:, 3]

        self._interp = RegularGridInterpolator(
            (xs, ys, zs), values,
            method='linear',
            bounds_error=False,
            fill_value=0.0,  # extrapolate to zero outside the grid 
        )
        self._path = resolved
        print(f"  GridFieldComponent: loaded {filepath} — "
              f"grid {len(xs)}×{len(ys)}×{len(zs)}, "
              f"range [{data[:,3].min():.4e}, {data[:,3].max():.4e}]")

    def evaluate(self, positions):
        """Return field values at positions (N, 3) -> (N,)."""
        return self._interp(positions)


class LineFieldComponent:
    """Single scalar field loaded from a 2-column profile file.

    File format: position, value (comma or whitespace separated).
    Header/comment rows are tolerated. The profile is linearly interpolated
    and clamped to edge values outside the sampled range.
    """

    def __init__(self, filepath):
        resolved = filepath
        if not os.path.isabs(filepath):
            resolved = os.path.join(os.path.dirname(__file__), filepath)

        try:
            data = np.loadtxt(resolved, dtype=np.float64, delimiter=',')
        except Exception:
            # Fall back to a tolerant reader so files may include headers.
            data = np.genfromtxt(resolved, dtype=np.float64, invalid_raise=False)

        data = np.atleast_2d(data)
        if data.shape[1] < 2:
            raise ValueError(
                f"Expected at least 2 columns (position,value) in {filepath}, "
                f"got shape {data.shape}"
            )

        pos = np.asarray(data[:, 0], dtype=np.float64)
        val = np.asarray(data[:, 1], dtype=np.float64)
        valid = np.isfinite(pos) & np.isfinite(val)
        pos = pos[valid]
        val = val[valid]

        if pos.size < 2:
            raise ValueError(f"Profile in {filepath} must contain at least 2 valid rows")

        order = np.argsort(pos)
        pos = pos[order]
        val = val[order]

        # Deduplicate repeated coordinates by averaging values.
        uniq_pos, inv = np.unique(pos, return_inverse=True)
        uniq_val = np.zeros_like(uniq_pos)
        counts = np.zeros_like(uniq_pos)
        np.add.at(uniq_val, inv, val)
        np.add.at(counts, inv, 1.0)
        uniq_val /= np.maximum(counts, 1.0)

        if uniq_pos.size < 2:
            raise ValueError(f"Profile in {filepath} must contain at least 2 distinct positions")

        self._interp = interp1d(
            uniq_pos,
            uniq_val,
            kind='linear',
            bounds_error=False,
            fill_value=(uniq_val[0], uniq_val[-1]),
        )
        self._path = resolved
        print(
            f"  LineFieldComponent: loaded {filepath} — "
            f"points {uniq_pos.size}, "
            f"x-range [{uniq_pos[0]:.4e}, {uniq_pos[-1]:.4e}], "
            f"range [{uniq_val.min():.4e}, {uniq_val.max():.4e}]"
        )

    def evaluate(self, coordinates):
        """Return field values at coordinates (N,) -> (N,)."""
        return np.asarray(self._interp(coordinates), dtype=np.float64)


# ---------------------------------------------------------------------------
#  Composite field provider — each of the 6 components is independent
# ---------------------------------------------------------------------------

class CompositeFieldProvider(ExternalFieldProvider):
    """Field provider where each component (Bx,By,Bz,Ex,Ey,Ez) is independent.

    Each component is described by a dict:
        {"mode": "zero"}
        {"mode": "fixed", "value": float}
        {"mode": "file", "file": "path/to/Bx.fld"}
        {"mode": "profile_x", "file": "path/to/Bx_profile.csv"}
        {"mode": "rid_ey", "v_rid_v": 20e3, "x_min_m": 5.4, "x_max_m": 7.2}
    """

    def __init__(self, components):
        self._sources = {}  # key -> ("zero",) | ("fixed", val) | ("grid", GridFieldComponent) | ("rid_ey", params)
        self._scales = {}   # key -> float  (multiplier applied after evaluation)
        for key in ("Bx", "By", "Bz", "Ex", "Ey", "Ez"):
            cfg = components.get(key, {"mode": "zero"})
            mode = cfg.get("mode", "zero")
            self._scales[key] = float(cfg.get("scale", 1.0))
            if mode == "fixed":
                self._sources[key] = ("fixed", float(cfg.get("value", 0.0)))
            elif mode == "file":
                self._sources[key] = ("grid", GridFieldComponent(cfg["file"]))
            elif mode in ("profile_x", "line"):
                self._sources[key] = ("line", LineFieldComponent(cfg["file"]))
            elif mode == "rid_ey" and key == "Ey":
                self._sources[key] = ("rid_ey", {
                    "v_rid_v": float(cfg.get("v_rid_v", 20e3)),
                    "x_min_m": float(cfg.get("x_min_m", 5.4)),
                    "x_max_m": float(cfg.get("x_max_m", 7.2)),
                })
            else:
                self._sources[key] = ("zero",)

    def sample(self, positions_m, time_s):
        n = len(positions_m)
        e = np.zeros((n, 3), dtype=np.float64)
        b = np.zeros((n, 3), dtype=np.float64)

        comp_map = {"Bx": (b, 0), "By": (b, 1), "Bz": (b, 2),
                     "Ex": (e, 0), "Ey": (e, 1), "Ez": (e, 2)}

        for key, (arr, col) in comp_map.items():
            src = self._sources[key]
            scale = self._scales[key]
            if src[0] == "zero":
                continue
            elif src[0] == "fixed":
                arr[:, col] = src[1] * scale
            elif src[0] == "grid":
                arr[:, col] = src[1].evaluate(positions_m) * scale
            elif src[0] == "line":
                arr[:, col] = src[1].evaluate(positions_m[:, 0]) * scale
            elif src[0] == "rid_ey":
                params = src[1]
                field_mag = params["v_rid_v"] / 0.108
                x = positions_m[:, 0]
                y = positions_m[:, 1]
                in_x = (x > params["x_min_m"]) & (x < params["x_max_m"])
                if np.any(in_x):
                    ey = np.zeros(n, dtype=np.float64)
                    ey[in_x & (y > 0.118) & (y < 0.24)] = -field_mag
                    ey[in_x & (y > 0.0) & (y < 0.118)] = +field_mag
                    ey[in_x & (y > -0.118) & (y < 0.0)] = -field_mag
                    ey[in_x & (y > -0.24) & (y < -0.118)] = +field_mag
                    arr[:, col] = ey * scale

        return e, b


class ZeroFieldProvider(ExternalFieldProvider):
    """No external EM field."""

    def sample(self, positions_m, time_s):
        n = len(positions_m)
        zeros = np.zeros((n, 3), dtype=np.float64)
        return zeros, zeros


class UniformFieldProvider(ExternalFieldProvider):
    """Spatially uniform, time-independent E and B fields."""

    def __init__(self, electric_field_vpm=None, magnetic_field_t=None):
        self._electric = np.asarray(electric_field_vpm or [0.0, 0.0, 0.0], dtype=np.float64)
        self._magnetic = np.asarray(magnetic_field_t or [0.0, 0.0, 0.0], dtype=np.float64)

    def sample(self, positions_m, time_s):
        n = len(positions_m)
        e = np.repeat(self._electric[np.newaxis, :], n, axis=0)
        b = np.repeat(self._magnetic[np.newaxis, :], n, axis=0)
        return e, b


class RIDSegmentFieldProvider(ExternalFieldProvider):
    """Piecewise y-directed E field in a finite x-region, with uniform B."""

    def __init__(
        self,
        v_rid_v=20e3,
        x_min_m=5.4,
        x_max_m=7.2,
        magnetic_field_t=None,
    ):
        self._v_rid_v = float(v_rid_v)
        self._x_min_m = float(x_min_m)
        self._x_max_m = float(x_max_m)
        self._magnetic = np.asarray(magnetic_field_t or [0.0, 0.0, 0.0], dtype=np.float64)
        self._field_mag_vpm = self._v_rid_v / 0.108

    def sample(self, positions_m, time_s):
        n = len(positions_m)
        e = np.zeros((n, 3), dtype=np.float64)
        b = np.repeat(self._magnetic[np.newaxis, :], n, axis=0)

        x = positions_m[:, 0]
        y = positions_m[:, 1]

        in_x = (x > self._x_min_m) & (x < self._x_max_m)
        if not np.any(in_x):
            return e, b

        ey = np.zeros(n, dtype=np.float64)

        ey[in_x & (y > 0.118) & (y < 0.24)] = -self._field_mag_vpm
        ey[in_x & (y > 0.0) & (y < 0.118)] = +self._field_mag_vpm
        ey[in_x & (y > -0.118) & (y < 0.0)] = -self._field_mag_vpm
        ey[in_x & (y > -0.24) & (y < -0.118)] = +self._field_mag_vpm

        e[:, 1] = ey
        return e, b


def create_field_provider(config_dict=None):
    """Build a field provider from a config dictionary."""
    cfg = config_dict or {}
    provider_type = str(cfg.get("type", "zero")).strip().lower()

    if provider_type in ("none", "zero", "off"):
        return ZeroFieldProvider()

    if provider_type == "composite":
        return CompositeFieldProvider(cfg.get("components", {}))

    if provider_type == "uniform":
        return UniformFieldProvider(
            electric_field_vpm=cfg.get("electric_field_vpm", [0.0, 0.0, 0.0]),
            magnetic_field_t=cfg.get("magnetic_field_t", [0.0, 0.0, 0.0]),
        )

    if provider_type in ("rid_segment_y", "rid_piecewise"):
        return RIDSegmentFieldProvider(
            v_rid_v=cfg.get("v_rid_v", 20e3),
            x_min_m=cfg.get("x_min_m", 5.4),
            x_max_m=cfg.get("x_max_m", 7.2),
            magnetic_field_t=cfg.get("magnetic_field_t", [0.0, 0.0, 0.0]),
        )

    raise ValueError(f"Unknown EXTERNAL_FIELD provider type: {provider_type}")
