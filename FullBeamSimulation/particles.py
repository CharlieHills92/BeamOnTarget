# particles.py
"""
Defines classes for various particle sources (beams) and functions to
load them from configuration files.

Every source.generate() returns:
    origins      (N, 3)  float64   starting positions [m]
    directions   (N, 3)  float64   unit direction vectors
    energies_eV  (N,)    float64   kinetic energy per macro-particle [eV]
    currents     (N,)    float64   electrical current per macro-particle [A]
    masses       (N,)    float64   rest mass per particle [kg]
    charges      (N,)    int       charge state in units of e
"""
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
#  Base class
# ---------------------------------------------------------------------------
class ParticleSource:
    """Base class for all particle sources."""

    def __init__(self, num_particles, energy_range=(100, 800),
                 mass=0.0, total_current=0.0, charge_state=0):
        self.num_particles = int(num_particles)
        self.energy_range = energy_range
        self.mass = mass
        self.total_current = total_current
        self.charge_state = charge_state

    def generate(self):
        raise NotImplementedError

    def get_visualization_repr(self):
        raise NotImplementedError

    # -- helpers ----------------------------------------------------------
    def _generate_energy(self):
        return np.random.uniform(self.energy_range[0], self.energy_range[1],
                                 self.num_particles)

    def _generate_current(self):
        if self.num_particles > 0:
            return np.full(self.num_particles,
                           self.total_current / self.num_particles)
        return np.zeros(0)

    def _generate_mass(self):
        return np.full(self.num_particles, self.mass, dtype=np.float64)

    def _generate_charge_state(self):
        return np.full(self.num_particles, self.charge_state, dtype=int)

    def _pack(self, origins, directions, energies, currents):
        """Return the standard 6-tuple."""
        return (origins, directions, energies, currents,
                self._generate_mass(), self._generate_charge_state())


# ---------------------------------------------------------------------------
#  Concrete beam types
# ---------------------------------------------------------------------------
class PlanarBeam(ParticleSource):
    """Rectangular beam of parallel particles."""

    def __init__(self, num_particles, center_point, size, direction, **kw):
        super().__init__(num_particles, **kw)
        self.center = np.asarray(center_point, dtype=np.float64)
        self.size = size
        self.direction = np.asarray(direction, dtype=np.float64)
        self.direction /= np.linalg.norm(self.direction)

    def get_visualization_repr(self):
        return self.center, self.direction

    def generate(self):
        d = self.direction
        u = np.array([0., 1., 0.]) if not np.allclose(d, [0, 1, 0]) \
            else np.array([1., 0., 0.])
        v = np.cross(d, u); v /= np.linalg.norm(v)
        u = np.cross(v, d)
        ru = np.random.uniform(-self.size[0]/2, self.size[0]/2, self.num_particles)
        rv = np.random.uniform(-self.size[1]/2, self.size[1]/2, self.num_particles)
        origins = self.center + ru[:, None]*u + rv[:, None]*v
        dirs = np.tile(d, (self.num_particles, 1))
        e = self._generate_energy(); c = self._generate_current()
        return self._pack(origins, dirs, e, c)


class ConicalBeam(ParticleSource):
    """Particles from a single point in a cone."""

    def __init__(self, num_particles, origin_point, central_axis,
                 cone_angle_deg, **kw):
        super().__init__(num_particles, **kw)
        self.origin = np.asarray(origin_point, dtype=np.float64)
        self.axis = np.asarray(central_axis, dtype=np.float64)
        self.axis /= np.linalg.norm(self.axis)
        self.cone_angle_rad = np.deg2rad(cone_angle_deg)

    def get_visualization_repr(self):
        return self.origin, self.axis

    def generate(self):
        n = self.num_particles
        z = np.random.uniform(np.cos(self.cone_angle_rad/2), 1, n)
        theta = np.random.uniform(0, 2*np.pi, n)
        phi = np.arccos(z)
        x, y = np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta)

        up = np.array([0., 0., 1.])
        rot_axis = np.cross(up, self.axis)
        if np.linalg.norm(rot_axis) < 1e-6:
            R = np.eye(3)
        else:
            rot_axis /= np.linalg.norm(rot_axis)
            ang = np.arccos(np.clip(np.dot(up, self.axis), -1, 1))
            c, s = np.cos(ang), np.sin(ang)
            K = np.array([[0, -rot_axis[2], rot_axis[1]],
                          [rot_axis[2], 0, -rot_axis[0]],
                          [-rot_axis[1], rot_axis[0], 0]])
            R = np.eye(3) + s*K + (1 - c)*(K @ K)
        dirs = np.vstack([x, y, z]).T @ R.T
        origins = np.tile(self.origin, (n, 1))
        e = self._generate_energy(); cur = self._generate_current()
        return self._pack(origins, dirs, e, cur)


class GaussianBeam(ParticleSource):
    """Parallel beam with Gaussian spatial distribution."""

    def __init__(self, num_particles, center_point, direction, sigma, **kw):
        super().__init__(num_particles, **kw)
        self.center = np.asarray(center_point, dtype=np.float64)
        self.direction = np.asarray(direction, dtype=np.float64)
        self.direction /= np.linalg.norm(self.direction)
        self.sigma = sigma if not isinstance(sigma, (int, float)) else (sigma, sigma)

    def get_visualization_repr(self):
        return self.center, self.direction

    def generate(self):
        d = self.direction
        u = np.array([0., 1., 0.]) if not np.allclose(d, [0, 1, 0]) \
            else np.array([1., 0., 0.])
        v = np.cross(d, u); v /= np.linalg.norm(v)
        u = np.cross(v, d)
        ru = np.random.normal(0, self.sigma[0], self.num_particles)
        rv = np.random.normal(0, self.sigma[1], self.num_particles)
        origins = self.center + ru[:, None]*u + rv[:, None]*v
        dirs = np.tile(d, (self.num_particles, 1))
        e = self._generate_energy(); c = self._generate_current()
        return self._pack(origins, dirs, e, c)


class GaussianTwissBeam(ParticleSource):
    """Gaussian phase-space beam defined by Twiss parameters."""

    def __init__(self, num_particles, center_point, direction,
                 alpha_x, beta_x, emittance_x_mm_mrad,
                 alpha_y, beta_y, emittance_y_mm_mrad, **kw):
        super().__init__(num_particles, **kw)
        self.center = np.asarray(center_point, dtype=np.float64)
        self.main_direction = np.asarray(direction, dtype=np.float64)
        self.main_direction /= np.linalg.norm(self.main_direction)
        self.alpha_x, self.beta_x = alpha_x, beta_x
        self.emit_x = emittance_x_mm_mrad * 1e-6
        self.alpha_y, self.beta_y = alpha_y, beta_y
        self.emit_y = emittance_y_mm_mrad * 1e-6

    def get_visualization_repr(self):
        return self.center, self.main_direction

    def generate(self):
        d = self.main_direction
        u = np.array([0., 1., 0.]) if not np.allclose(d, [0, 1, 0]) \
            else np.array([1., 0., 0.])
        v = np.cross(d, u); v /= np.linalg.norm(v)
        u = np.cross(v, d)
        n = self.num_particles
        u1x, u2x = np.random.normal(0, 1, n), np.random.normal(0, 1, n)
        u1y, u2y = np.random.normal(0, 1, n), np.random.normal(0, 1, n)

        x_pos = np.sqrt(self.beta_x * self.emit_x) * u1x
        x_prime = np.sqrt(self.emit_x / self.beta_x) * (
            -self.alpha_x * u1x + u2x)
        y_pos = np.sqrt(self.beta_y * self.emit_y) * u1y
        y_prime = np.sqrt(self.emit_y / self.beta_y) * (
            -self.alpha_y * u1y + u2y)

        origins = self.center + x_pos[:, None]*u + y_pos[:, None]*v
        dirs = d + x_prime[:, None]*u + y_prime[:, None]*v
        dirs /= np.linalg.norm(dirs, axis=1)[:, None]

        e = self._generate_energy(); c = self._generate_current()
        return self._pack(origins, dirs, e, c)


# ---------------------------------------------------------------------------
#  .bl file loader (ITER NBI beamlet format)
# ---------------------------------------------------------------------------
def load_beamlets_from_file(filename, num_particles_per_beamlet, beamlet_area):
    """Parse a .bl beamlet file and return a list of ParticleSource objects."""
    try:
        df = pd.read_csv(filename, comment='#', sep=r'\s+')
    except FileNotFoundError:
        print(f"Error: file not found: '{filename}'")
        return []
    except Exception as exc:
        print(f"Error reading '{filename}': {exc}")
        return []

    sources = []
    print(f"\nLoading {len(df)} beamlets from '{filename}'...")

    for _, row in df.iterrows():
        center = np.array([row['CenterX'], row['CenterY'], row['CenterZ']])
        direction = np.array([row['DirX'], row['DirY'], row['DirZ']])
        mass = row['Mass_kg']
        charge = int(row['Charge_e'])
        halo_frac = row['HaloFraction']
        energy_range = (row.get('MinEnergy_eV', 100),
                        row.get('MaxEnergy_eV', 800))
        total_current = row['CurrentDensity_A_m2'] * beamlet_area

        # --- core ---
        num_core = int(num_particles_per_beamlet * (1.0 - halo_frac))
        current_core = total_current * (1.0 - halo_frac)
        sigma_x = row['SigmaY_m']
        delta_x = row['DeltaY_rad'] / np.sqrt(2)
        emit_x = sigma_x * delta_x
        beta_x = sigma_x / delta_x if delta_x > 0 else 0
        sigma_y = row['SigmaZ_m']
        delta_y = row['DeltaZ_rad'] / np.sqrt(2)
        emit_y = sigma_y * delta_y
        beta_y = sigma_y / delta_y if delta_y > 0 else 0

        if num_core > 0:
            sources.append(GaussianTwissBeam(
                num_particles=num_core, center_point=center,
                direction=direction,
                alpha_x=0., beta_x=beta_x,
                emittance_x_mm_mrad=emit_x * 1e6,
                alpha_y=0., beta_y=beta_y,
                emittance_y_mm_mrad=emit_y * 1e6,
                total_current=current_core, mass=mass,
                charge_state=charge, energy_range=energy_range))

        # --- halo ---
        if halo_frac > 0:
            num_halo = int(num_particles_per_beamlet * halo_frac)
            current_halo = total_current * halo_frac
            delta_hx = row['DeltaHY_rad'] / np.sqrt(2)
            delta_hy = row['DeltaHZ_rad'] / np.sqrt(2)
            emit_hx = sigma_x * delta_hx
            beta_hx = sigma_x / delta_hx if delta_hx > 0 else 0
            emit_hy = sigma_y * delta_hy
            beta_hy = sigma_y / delta_hy if delta_hy > 0 else 0
            if num_halo > 0:
                sources.append(GaussianTwissBeam(
                    num_particles=num_halo, center_point=center,
                    direction=direction,
                    alpha_x=0., beta_x=beta_hx,
                    emittance_x_mm_mrad=emit_hx * 1e6,
                    alpha_y=0., beta_y=beta_hy,
                    emittance_y_mm_mrad=emit_hy * 1e6,
                    total_current=current_halo, mass=mass,
                    charge_state=charge, energy_range=energy_range))

    print(f"  Created {len(sources)} particle sources.")
    return sources
