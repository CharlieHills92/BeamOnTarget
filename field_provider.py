"""External electromagnetic field providers for particle tracking."""

import numpy as np


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
