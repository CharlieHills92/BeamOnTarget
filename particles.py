# particles.py
"""
Defines classes for various particle sources (beams) and functions to
load them from configuration files.
"""
import numpy as np
import pandas as pd

class ParticleSource:
    """Base class for all particle sources."""
    def __init__(self, num_particles, energy_range=(100, 800), mass=0.0, total_current=0.0, charge_state=0):
        """
        Initializes a particle source.

        Args:
            num_particles (int): The number of particles to generate.
            energy_range (tuple): The min and max energy for each particle in electron-Volts (eV).
            mass (float): The mass of a single particle in kg.
            total_current (float): The total electrical current of the beamlet in Amperes.
            charge_state (int): The charge of the particle in elementary charge units (e.g., +1 for a proton).
        """
        self.num_particles = int(num_particles)
        self.energy_range = energy_range
        self.mass = mass
        self.total_current = total_current
        self.charge_state = charge_state

    def generate(self):
        """Generates and returns all particle property arrays. Must be implemented by subclasses."""
        raise NotImplementedError("Each particle source must have a generate() method.")

    def get_visualization_repr(self):
        """Returns the center point and primary direction vector for visualization."""
        raise NotImplementedError("Subclass must implement get_visualization_repr.")

    def _generate_energy(self):
        """Generates random energy values for the particles in eV."""
        return np.random.uniform(self.energy_range[0], self.energy_range[1], self.num_particles)

    def _generate_current(self):
        """Calculates the current carried by each individual macro-particle."""
        if self.num_particles > 0:
            return np.full(self.num_particles, self.total_current / self.num_particles)
        return np.full(self.num_particles, 0.0)

    def _generate_charge_state(self):
        """Generates the charge state for each particle."""
        return np.full(self.num_particles, self.charge_state, dtype=int)


class PlanarBeam(ParticleSource):
    """A rectangular beam of parallel particles."""
    def __init__(self, num_particles, center_point, size, direction, **kwargs):
        super().__init__(num_particles, **kwargs)
        self.center = np.array(center_point)
        self.size = size
        self.direction = np.array(direction) / np.linalg.norm(direction)

    def get_visualization_repr(self):
        return self.center, self.direction

    def generate(self):
        u_vec = np.array([0.0, 1.0, 0.0]) if not np.allclose(self.direction, [0,1,0]) else np.array([1.0, 0.0, 0.0])
        v_vec = np.cross(self.direction, u_vec); v_vec /= np.linalg.norm(v_vec)
        u_vec = np.cross(v_vec, self.direction)
        rand_u = np.random.uniform(-self.size[0] / 2, self.size[0] / 2, self.num_particles)
        rand_v = np.random.uniform(-self.size[1] / 2, self.size[1] / 2, self.num_particles)
        ray_origins = self.center + rand_u[:, np.newaxis] * u_vec + rand_v[:, np.newaxis] * v_vec
        ray_directions = np.tile(self.direction, (self.num_particles, 1))
        particle_energies_eV = self._generate_energy(); particle_currents = self._generate_current()
        particle_charge_states = self._generate_charge_state()
        particle_powers = particle_energies_eV * particle_currents
        return ray_origins, ray_directions, particle_powers, particle_energies_eV, particle_currents, particle_charge_states


class ConicalBeam(ParticleSource):
    """A beam of particles originating from a single point in a cone."""
    def __init__(self, num_particles, origin_point, central_axis, cone_angle_deg, **kwargs):
        super().__init__(num_particles, **kwargs)
        self.origin = np.array(origin_point)
        self.axis = np.array(central_axis) / np.linalg.norm(central_axis)
        self.cone_angle_rad = np.deg2rad(cone_angle_deg)

    def get_visualization_repr(self):
        return self.origin, self.axis

    def generate(self):
        z = np.random.uniform(np.cos(self.cone_angle_rad / 2), 1, self.num_particles)
        theta = np.random.uniform(0, 2 * np.pi, self.num_particles)
        phi = np.arccos(z)
        x, y = np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta)
        up = np.array([0.0, 0.0, 1.0]); rot_axis = np.cross(up, self.axis)
        if np.linalg.norm(rot_axis) < 1e-6: rotation_matrix = np.eye(3)
        else:
            rot_axis /= np.linalg.norm(rot_axis)
            angle = np.arccos(np.dot(up, self.axis)); c, s = np.cos(angle), np.sin(angle)
            K = np.array([[0, -rot_axis[2], rot_axis[1]], [rot_axis[2], 0, -rot_axis[0]], [-rot_axis[1], rot_axis[0], 0]])
            rotation_matrix = np.eye(3) + s * K + (1 - c) * np.dot(K, K)
        local_dirs = np.vstack([x, y, z]).T
        ray_directions = np.dot(local_dirs, rotation_matrix.T)
        ray_origins = np.tile(self.origin, (self.num_particles, 1))
        particle_energies_eV = self._generate_energy(); particle_currents = self._generate_current()
        particle_charge_states = self._generate_charge_state()
        particle_powers = particle_energies_eV * particle_currents
        return ray_origins, ray_directions, particle_powers, particle_energies_eV, particle_currents, particle_charge_states


class GaussianBeam(ParticleSource):
    """A beam of parallel particles with a Gaussian spatial distribution."""
    def __init__(self, num_particles, center_point, direction, sigma, **kwargs):
        super().__init__(num_particles, **kwargs)
        self.center = np.array(center_point)
        self.direction = np.array(direction) / np.linalg.norm(direction)
        self.sigma = sigma if not isinstance(sigma, (int, float)) else (sigma, sigma)

    def get_visualization_repr(self):
        return self.center, self.direction

    def generate(self):
        u_vec = np.array([0.0, 1.0, 0.0]) if not np.allclose(self.direction, [0, 1, 0]) else np.array([1.0, 0.0, 0.0])
        v_vec = np.cross(self.direction, u_vec); v_vec /= np.linalg.norm(v_vec)
        u_vec = np.cross(v_vec, self.direction)
        rand_u = np.random.normal(loc=0.0, scale=self.sigma[0], size=self.num_particles)
        rand_v = np.random.normal(loc=0.0, scale=self.sigma[1], size=self.num_particles)
        ray_origins = self.center + rand_u[:, np.newaxis] * u_vec + rand_v[:, np.newaxis] * v_vec
        ray_directions = np.tile(self.direction, (self.num_particles, 1))
        particle_energies_eV = self._generate_energy(); particle_currents = self._generate_current()
        particle_charge_states = self._generate_charge_state()
        particle_powers = particle_energies_eV * particle_currents
        return ray_origins, ray_directions, particle_powers, particle_energies_eV, particle_currents, particle_charge_states


class TwissBeam(ParticleSource):
    """A hard-edged beam defined by Twiss parameters (K-V distribution)."""
    def __init__(self, num_particles, center_point, direction, alpha_x, beta_x, emittance_x_mm_mrad, alpha_y, beta_y, emittance_y_mm_mrad, **kwargs):
        super().__init__(num_particles, **kwargs)
        self.center, self.main_direction = np.array(center_point), np.array(direction) / np.linalg.norm(direction)
        self.alpha_x, self.beta_x, self.emit_x = alpha_x, beta_x, emittance_x_mm_mrad * 1e-6
        self.alpha_y, self.beta_y, self.emit_y = alpha_y, beta_y, emittance_y_mm_mrad * 1e-6

    def get_visualization_repr(self):
        return self.center, self.main_direction

    def generate(self):
        u_vec = np.array([0.0, 1.0, 0.0]) if not np.allclose(self.main_direction, [0, 1, 0]) else np.array([1.0, 0.0, 0.0])
        v_vec = np.cross(self.main_direction, u_vec); v_vec /= np.linalg.norm(v_vec)
        u_vec = np.cross(v_vec, self.main_direction)
        r_x, theta_x = np.sqrt(np.random.uniform(0, 1, self.num_particles)), np.random.uniform(0, 2 * np.pi, self.num_particles)
        u1x, u2x = r_x * np.cos(theta_x), r_x * np.sin(theta_x)
        r_y, theta_y = np.sqrt(np.random.uniform(0, 1, self.num_particles)), np.random.uniform(0, 2 * np.pi, self.num_particles)
        u1y, u2y = r_y * np.cos(theta_y), r_y * np.sin(theta_y)
        x_pos = np.sqrt(self.beta_x * self.emit_x) * u1x; x_prime = np.sqrt(self.emit_x / self.beta_x) * (-self.alpha_x * u1x + u2x)
        y_pos = np.sqrt(self.beta_y * self.emit_y) * u1y; y_prime = np.sqrt(self.emit_y / self.beta_y) * (-self.alpha_y * u1y + u2y)
        ray_origins = self.center + x_pos[:, np.newaxis] * u_vec + y_pos[:, np.newaxis] * v_vec
        ray_directions = (self.main_direction + x_prime[:, np.newaxis] * u_vec + y_prime[:, np.newaxis] * v_vec)
        ray_directions /= np.linalg.norm(ray_directions, axis=1)[:, np.newaxis]
        particle_energies_eV = self._generate_energy(); particle_currents = self._generate_current()
        particle_charge_states = self._generate_charge_state()
        particle_powers = particle_energies_eV * particle_currents
        return ray_origins, ray_directions, particle_powers, particle_energies_eV, particle_currents, particle_charge_states


class GaussianTwissBeam(ParticleSource):
    """A realistic beam defined by Twiss parameters with a Gaussian phase space distribution."""
    def __init__(self, num_particles, center_point, direction, alpha_x, beta_x, emittance_x_mm_mrad, alpha_y, beta_y, emittance_y_mm_mrad, **kwargs):
        super().__init__(num_particles, **kwargs)
        self.center, self.main_direction = np.array(center_point), np.array(direction) / np.linalg.norm(direction)
        self.alpha_x, self.beta_x, self.emit_x = alpha_x, beta_x, emittance_x_mm_mrad * 1e-6
        self.alpha_y, self.beta_y, self.emit_y = alpha_y, beta_y, emittance_y_mm_mrad * 1e-6

    def get_visualization_repr(self):
        return self.center, self.main_direction

    def generate(self):
        u_vec = np.array([0.0, 1.0, 0.0]) if not np.allclose(self.main_direction, [0, 1, 0]) else np.array([1.0, 0.0, 0.0])
        v_vec = np.cross(self.main_direction, u_vec); v_vec /= np.linalg.norm(v_vec)
        u_vec = np.cross(v_vec, self.main_direction)
        u1x, u2x = np.random.normal(0, 1, self.num_particles), np.random.normal(0, 1, self.num_particles)
        u1y, u2y = np.random.normal(0, 1, self.num_particles), np.random.normal(0, 1, self.num_particles)
        x_pos = np.sqrt(self.beta_x * self.emit_x) * u1x; x_prime = np.sqrt(self.emit_x / self.beta_x) * (-self.alpha_x * u1x + u2x)
        y_pos = np.sqrt(self.beta_y * self.emit_y) * u1y; y_prime = np.sqrt(self.emit_y / self.beta_y) * (-self.alpha_y * u1y + u2y)
        ray_origins = self.center + x_pos[:, np.newaxis] * u_vec + y_pos[:, np.newaxis] * v_vec
        ray_directions = (self.main_direction + x_prime[:, np.newaxis] * u_vec + y_prime[:, np.newaxis] * v_vec)
        ray_directions /= np.linalg.norm(ray_directions, axis=1)[:, np.newaxis]
        particle_energies_eV = self._generate_energy(); particle_currents = self._generate_current()
        particle_charge_states = self._generate_charge_state()
        particle_powers = particle_energies_eV * particle_currents
        return ray_origins, ray_directions, particle_powers, particle_energies_eV, particle_currents, particle_charge_states


def load_beamlets_from_file(filename, num_particles_per_beamlet, beamlet_area):
    """
    Parses a text file to create a list of particle sources.
    Each line in the file defines a beamlet with a core and an optional halo.
    """
    try:
        df = pd.read_csv(filename, comment='#', delim_whitespace=True)
    except FileNotFoundError:
        print(f"Error: Particle source file not found at '{filename}'"); return []
    except Exception as e:
        print(f"Error reading particle source file '{filename}': {e}"); return []
        
    all_sources = []
    print(f"\nLoading {len(df)} beamlet definitions from '{filename}'...")

    for index, row in df.iterrows():
        center_point = np.array([row['CenterX'], row['CenterY'], row['CenterZ']])
        direction = np.array([row['DirX'], row['DirY'], row['DirZ']])
        mass, charge, halo_frac = row['Mass_kg'], row['Charge_e'], row['HaloFraction']
        min_energy, max_energy = row.get('MinEnergy_eV', 100), row.get('MaxEnergy_eV', 800)
        energy_range = (min_energy, max_energy)
        total_current = row['CurrentDensity_A_m2'] * beamlet_area

        # Create the CORE beam
        num_core = int(num_particles_per_beamlet * (1.0 - halo_frac))
        current_core = total_current * (1.0 - halo_frac)
        sigma_x, delta_x = row['SigmaY_m'], row['DeltaY_rad']
        delta_x = delta_x / np.sqrt(2) # convert e-fold to sigma
        emit_x_m_rad, beta_x = sigma_x * delta_x, sigma_x / delta_x if delta_x > 0 else 0
        sigma_y, delta_y = row['SigmaZ_m'], row['DeltaZ_rad']
        delta_y = delta_y / np.sqrt(2) # convert e-fold to sigma
        emit_y_m_rad, beta_y = sigma_y * delta_y, sigma_y / delta_y if delta_y > 0 else 0
        if num_core > 0:
            all_sources.append(GaussianTwissBeam(
                num_particles=num_core, center_point=center_point, direction=direction,
                alpha_x=0.0, beta_x=beta_x, emittance_x_mm_mrad=emit_x_m_rad * 1e6,
                alpha_y=0.0, beta_y=beta_y, emittance_y_mm_mrad=emit_y_m_rad * 1e6,
                total_current=current_core, mass=mass, charge_state=charge, energy_range=energy_range))

        # Create the HALO beam
        if halo_frac > 0:
            num_halo, current_halo = int(num_particles_per_beamlet * halo_frac), total_current * halo_frac
            delta_hx, delta_hy = row['DeltaHY_rad'], row['DeltaHZ_rad']
            delta_hx = delta_hx / np.sqrt(2) # convert e-fold to sigma
            delta_hy = delta_hy / np.sqrt(2) # convert e-fold to sigma
            emit_hx_m_rad, beta_hx = sigma_x * delta_hx, sigma_x / delta_hx if delta_hx > 0 else 0
            emit_hy_m_rad, beta_hy = sigma_y * delta_hy, sigma_y / delta_hy if delta_hy > 0 else 0
            if num_halo > 0:
                all_sources.append(GaussianTwissBeam(
                    num_particles=num_halo, center_point=center_point, direction=direction,
                    alpha_x=0.0, beta_x=beta_hx, emittance_x_mm_mrad=emit_hx_m_rad * 1e6,
                    alpha_y=0.0, beta_y=beta_hy, emittance_y_mm_mrad=emit_hy_m_rad * 1e6,
                    total_current=current_halo, mass=mass, charge_state=charge, energy_range=energy_range))

    return all_sources