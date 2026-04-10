import numpy as np
from numpy.typing import NDArray
from typing import TYPE_CHECKING
from utils.constants import G, C
import os

if TYPE_CHECKING:
    from solar_system import SolarSystem
    from bodies import LargeBody


class PhysicalModel():
    """Abstract base class for physics models.
    
    Provide __init__ and function to transfer data between body objects and numpy arrays.
    """
    def __init__(self, solar_system: "SolarSystem", integrator: str = "leapfrog", general_relativity: bool = False):
        """Initialize with a reference to the solar system.
        
        :param solar_system: The SolarSystem instance to simulate
        :param integrator: Integration method ('euler', 'rk4' or 'leapfrog')
        :param general_relativity: Whether to include general relativity effects
        """
        self.solar_system = solar_system
        self.positions: NDArray[np.float64] = np.zeros((self.solar_system.nbodies, 3))  # shape (N, 3)
        self.velocities: NDArray[np.float64] = np.zeros((self.solar_system.nbodies, 3))  # shape (N, 3)
        self.accelerations: NDArray[np.float64] = np.zeros((self.solar_system.nbodies, 3))  # shape (N, 3)
        self.GMM: NDArray[np.float64] = np.zeros((0, 0))  # Will be initialized in do_nstep
        self.mass: NDArray[np.float64] = np.array([body.mass for body in self.solar_system.bodies], dtype=np.float64)  # shape (N,)
        self.integrator = integrator
        self.general_relativity = general_relativity
        print(f"Physical model initialized with integrator: {self.integrator}")

    def do_nstep(self, nstep: int, dt: np.float64) -> None:
        """Perform nstep integration steps.
        
        Integration happens in an inertial frame. Frame transformations
        should only be applied after integration for output/comparison.
        """
        # Debug: print integrator being used (only for ModelNumpyArray)
        #if hasattr(self, 'integrator'):
        #    print(f"Running {nstep} steps with {self.integrator} integrator, dt={dt}s")
        
        self.positions = self._collect_positions()
        self.velocities = self._collect_velocities()
        self.GMM = self._calculate_G_m2()  # Calculate once, it's constant
        
        # Calculate initial accelerations
        self._calculate_gravitational_accelerations(self.GMM)

        for _ in range(nstep):
            self._update_positions_and_velocities(dt)
            
        self._update_objects_states()

    def _collect_positions(self) -> NDArray[np.float64]:
        """Collect positions of all bodies into a numpy array."""
        return np.array([body.position for body in self.solar_system.bodies])
    
    def _collect_velocities(self) -> NDArray[np.float64]:
        """Collect velocities of all bodies into a numpy array."""
        return np.array([body.velocity for body in self.solar_system.bodies])
    
    def _update_objects_states(self) -> None:
        """Update the position and velocity of each body in the solar system."""
        for i, body in enumerate(self.solar_system.bodies):
            body.position = self.positions[i]
            body.velocity = self.velocities[i]

    def _update_positions_and_velocities(self, dt: np.float64) -> None:
        """Update positions and velocities using selected integration method.
        
        :param dt: Time step in seconds
        """
        if self.integrator == "rk4":
            self._integrate_rk4(dt)
        elif self.integrator == "leapfrog":
            self._integrate_leapfrog(dt)
        elif self.integrator == "euler":
            self._integrate_euler(dt)
        else:
            raise ValueError(f"Unknown integrator: {self.integrator}")

class ModelObjectOriented(PhysicalModel):
    def do_nstep(self, nstep, dt) -> None:
        """Perform nstep integration steps using an object-oriented approach."""
        # Calculate initial accelerations
        self._calculate_gravitational_accelerations()

        for _ in range(nstep):
            self._update_positions_and_velocities(dt)
 
    def _calculate_gravitational_accelerations(self) -> None:
        """Calculate accelerations using an object-oriented approach."""
        for i in range(self.solar_system.nbodies):
            for j in range(i+1,self.solar_system.nbodies):
                body_i = self.solar_system.bodies[i]
                body_j = self.solar_system.bodies[j]
                
                # Vector from i to j
                r_vec = body_j.position - body_i.position
                r_mag = np.linalg.norm(r_vec) + 1e-20  # Avoid division by zero
                r_hat = r_vec / r_mag
                
                # Gravitational force magnitude
                F_mag = G * body_i.mass * body_j.mass / r_mag**2
                
                # Force vector
                F_vec = F_mag * r_hat
                
                # Update accelerations (Newton's third law)
                body_i.acceleration += F_vec / body_i.mass
                body_j.acceleration -= F_vec / body_j.mass

                if i == 0 and self.general_relativity:
                    self._calculate_general_relativity_correction(body_i, body_j)
        
    def _integrate_euler(self, dt: np.float64) -> None:
        """Simple Euler integrator (not recommended for orbital dynamics)."""
        for i in range(self.solar_system.nbodies):
            body = self.solar_system.bodies[i]
            body.velocity += body.acceleration * dt
            body.position += body.velocity * dt
            body.acceleration = np.zeros(3,np.float64)  # Reset acceleration for next step

        # update accelerations for the next step
        self._calculate_gravitational_accelerations()
    
    def _integrate_rk4(self, dt: np.float64) -> None:
        """Runge-Kutta 4th order integrator (object-oriented, side-effect free)."""
        dt_half = dt * 0.5
        dt_sixth = dt / 6.0

        # Save initial positions, velocities, accelerations
        r0 = np.array([body.position.copy() for body in self.solar_system.bodies])
        v0 = np.array([body.velocity.copy() for body in self.solar_system.bodies])

        # k1: at (r0, v0)
        self._calculate_gravitational_accelerations()
        k1v = np.array([body.acceleration.copy() for body in self.solar_system.bodies])
        k1r = v0.copy()

        # k2: at (r0 + k1r*dt/2, v0 + k1v*dt/2)
        temp_r = r0 + k1r * dt_half
        temp_v = v0 + k1v * dt_half
        # Temporarily set body states
        for i, body in enumerate(self.solar_system.bodies):
            body.position = temp_r[i]
            body.velocity = temp_v[i]
            body.acceleration = np.zeros(3, np.float64)  # Reset acceleration for next calculation
        self._calculate_gravitational_accelerations()
        k2v = np.array([body.acceleration.copy() for body in self.solar_system.bodies])
        k2r = temp_v.copy()

        # k3: at (r0 + k2r*dt/2, v0 + k2v*dt/2)
        temp_r = r0 + k2r * dt_half
        temp_v = v0 + k2v * dt_half
        for i, body in enumerate(self.solar_system.bodies):
            body.position = temp_r[i]
            body.velocity = temp_v[i]
            body.acceleration = np.zeros(3, np.float64)  # Reset acceleration for next calculation
        self._calculate_gravitational_accelerations()
        k3v = np.array([body.acceleration.copy() for body in self.solar_system.bodies])
        k3r = temp_v.copy()

        # k4: at (r0 + k3r*dt, v0 + k3v*dt)
        temp_r = r0 + k3r * dt
        temp_v = v0 + k3v * dt
        for i, body in enumerate(self.solar_system.bodies):
            body.position = temp_r[i]
            body.velocity = temp_v[i]
            body.acceleration = np.zeros(3, np.float64)  # Reset acceleration for next calculation
        self._calculate_gravitational_accelerations()
        k4v = np.array([body.acceleration.copy() for body in self.solar_system.bodies])
        k4r = temp_v.copy()

        # Final update: restore r0, v0, then apply RK4 formula
        for i, body in enumerate(self.solar_system.bodies):
            body.position = r0[i] + dt_sixth * (k1r[i] + 2*k2r[i] + 2*k3r[i] + k4r[i])
            body.velocity = v0[i] + dt_sixth * (k1v[i] + 2*k2v[i] + 2*k3v[i] + k4v[i])
            body.acceleration = np.zeros(3, np.float64)  # Will be recalculated next step

    def _integrate_leapfrog(self, dt: np.float64) -> None:
        """Leapfrog (Verlet) symplectic integrator - conserves energy perfectly."""
        v_half = np.zeros((self.solar_system.nbodies, 3), dtype=np.float64)

        for i in range(self.solar_system.nbodies):
            body = self.solar_system.bodies[i]
            # Half-step velocity
            v_half[i] = body.velocity + 0.5 * body.acceleration * dt
            # Full-step position
            body.position += v_half[i] * dt
            body.acceleration = np.zeros(3,np.float64)  # Reset acceleration for next step
        
        # Recalculate accelerations at new positions
        self._calculate_gravitational_accelerations()
        
        for i in range(self.solar_system.nbodies):
            body = self.solar_system.bodies[i]
            # Complete velocity step using new acceleration
            body.velocity = v_half[i] + 0.5 * body.acceleration * dt

    def _calculate_general_relativity_correction(self, body_i: "LargeBody", body_j: "LargeBody") -> None:
        """Calculate general relativity correction for a body (1PN approximation).
        
        This is a simplified correction that only applies to the Sun's influence on other bodies.
        """        # Relative velocity and position to the Sun (body_i is the Sun)
        V = body_j.velocity - body_i.velocity  # Relative velocity
        R = body_j.position - body_i.position  # Relative position

        v = np.linalg.norm(V)  # Speed relative to Sun
        r = np.linalg.norm(R)  # Distance from Sun

        GM_sun = G * body_i.mass  # G * M_sun

        # GR correction terms
        term1 = GM_sun / (C**2 * r**3)
        term2 = 4 - GM_sun / r - v**2
        term3 = 4 * np.dot(R, V)

        a_GR = term1 * (term2 * R - term3 * V)
        
        body_j.acceleration += a_GR
        body_i.acceleration -= a_GR * body_j.mass / body_i.mass

class ModelNumpyArray(PhysicalModel):
    """Numpy array-based model for efficient medium N-body simulations.
        """
    def _calculate_G_m2(self) -> NDArray[np.float64]:
        """Calculate the gravitational force magnitude between two bodies using vectorized operations."""        
        # Broadcast to create (N, N) matrix where GMM[i,j] = G * mass[j]
        # This gives the "pulling mass" for each pair
        GMM = G * self.mass[np.newaxis, :]
        
        return GMM

    def _calculate_gravitational_accelerations(self, GMM: NDArray[np.float64]) -> None:
        """Calculate accelerations using fully vectorized operations.
        
        Computes gravitational acceleration: a_i = sum_j G*m_j*(r_j - r_i)/|r_j - r_i|³
        """
        # Position differences (j - i): vector FROM i TO j (attraction direction)
        pos_diff = self.positions[np.newaxis, :, :] - self.positions[:, np.newaxis, :]
        
        # Squared distances (avoiding sqrt for efficiency)
        dist_sq = np.sum(pos_diff**2, axis=2) + 1e-20  # Small value to avoid /0
        
        # Distance for normalization (compute sqrt only once)
        dist = np.sqrt(dist_sq)
        
        # Acceleration magnitude: a = G*m_j / r²
        # GMM[i,j] = G * mass[j], so a_mag[i,j] = GMM[i,j] / dist_sq[i,j]
        a_mag = GMM / dist_sq
        
        # Acceleration vectors: a_mag * unit_vector
        # unit_vector = pos_diff / dist = (r_j - r_i) / |r_j - r_i|
        # Mask diagonal to avoid self-interaction
        mask = dist > 1e-10
        accel_vectors = np.where(mask[:, :, np.newaxis], 
                                 a_mag[:, :, np.newaxis] * pos_diff / dist[:, :, np.newaxis],
                                 0.0)
        
        # Sum all accelerations acting on each body
        self.accelerations[:] = np.sum(accel_vectors, axis=1)
        if self.general_relativity:
            self._calculate_general_relativity_corrections()

    def _calculate_general_relativity_corrections(self) -> None:
        """Calculate general relativity corrections to the accelerations (1PN approximation).
        
        All internal calculations use simulation units: km, km/s, kg
        Fully vectorized implementation for efficiency.
        """
        # Relative velocity and position to the Sun (index 0)
        V = self.velocities - self.velocities[0]  # shape (N, 3)
        R = self.positions - self.positions[0]    # shape (N, 3)

        v = np.linalg.norm(V, axis=1)  # Speed relative to Sun (km/s), shape (N,)
        r = np.linalg.norm(R, axis=1)  # Distance from Sun (km), shape (N,)

        GM_sun = G * self.mass[0]  # G * M_sun (km^3/s^2)

        # Mask to skip Sun itself and avoid division by zero
        mask = r > 1e-10  # shape (N,)
        
        # Initialize terms (will be zero where masked)
        term1 = np.zeros_like(r)
        term2 = np.zeros_like(r)
        
        # Compute only for valid bodies (not the Sun)
        term1[mask] = GM_sun / (C**2 * r[mask]**3)
        term2[mask] = 4 - GM_sun / r[mask] - v[mask]**2
        
        # Dot product R·V for each body (vectorized), shape (N,)
        term3 = 4 * np.sum(R * V, axis=1)
        
        # Compute GR acceleration correction (fully vectorized)
        # Broadcasting: (N,1) * ((N,1) * (N,3) - (N,1) * (N,3))
        a_GR = term1[:, np.newaxis] * (term2[:, np.newaxis] * R - term3[:, np.newaxis] * V)
        
        self.accelerations += a_GR
    
    def _integrate_leapfrog(self, dt: np.float64) -> None:
        """Leapfrog (Verlet) symplectic integrator - conserves energy perfectly.
        
        This is the best integrator for long-term orbital dynamics because it's
        symplectic (preserves phase space volume) and time-reversible.
        
        :param dt: Time step in seconds
        """
        # Half-step velocity using current acceleration
        v_half = self.velocities + 0.5 * dt * self.accelerations
        
        # Full-step position using half-step velocity
        self.positions[:] = self.positions + dt * v_half
        
        # Compute acceleration at new position
        self._calculate_gravitational_accelerations(self.GMM)
        
        # Complete velocity step using new acceleration
        self.velocities[:] = v_half + 0.5 * dt * self.accelerations
    
    def _integrate_rk4(self, dt: np.float64) -> None:
        """Runge-Kutta 4th order integrator.
        
        More accurate than Leapfrog for short-term, but doesn't conserve energy
        as well over many orbits. Use for high-precision short simulations.
        :param dt: Time step in seconds
        """
        dt_half = dt * 0.5
        dt_sixth = dt / 6.0
        
        r0 = self.positions.copy()
        v0 = self.velocities.copy()
        
        # k1 - evaluate at current position (calculate fresh, don't reuse from previous step)
        self._calculate_gravitational_accelerations(self.GMM)
        k1a = self.accelerations.copy()
        k1v = v0.copy()
        
        # k2 - evaluate at midpoint using k1
        self.positions[:] = r0 + k1v * dt_half
        self._calculate_gravitational_accelerations(self.GMM)
        k2a = self.accelerations.copy()
        k2v = v0 + k1a * dt_half
        
        # k3 - evaluate at midpoint using k2
        self.positions[:] = r0 + k2v * dt_half
        self._calculate_gravitational_accelerations(self.GMM)
        k3a = self.accelerations.copy()
        k3v = v0 + k2a * dt_half
        
        # k4 - evaluate at endpoint using k3
        self.positions[:] = r0 + k3v * dt
        self._calculate_gravitational_accelerations(self.GMM)
        k4a = self.accelerations.copy()
        k4v = v0 + k3a * dt
        
        # Combine using RK4 formula (in-place to avoid allocation)
        self.velocities[:] = v0 + dt_sixth * (k1a + 2*k2a + 2*k3a + k4a)
        self.positions[:] = r0 + dt_sixth * (k1v + 2*k2v + 2*k3v + k4v)

    def _integrate_euler(self, dt: np.float64) -> None:
        """Simple Euler integrator (not recommended for orbital dynamics).
            
        Only for testing or very short simulations due to poor energy conservation.
        :param dt: Time step in seconds
        """
        self.velocities = self.velocities + self.accelerations * dt
        self.positions = self.positions + self.velocities * dt

            # update accelerations for the next step
        self._calculate_gravitational_accelerations(self.GMM)

class ModelNumba(PhysicalModel):
    """Numba-optimized model for large N-body simulations"""
    def __init__(self, solar_system: "SolarSystem", integrator: str = "leapfrog", general_relativity: bool = False, n_core: int = 1) -> None:
        super().__init__(solar_system, integrator, general_relativity)
        # import prercompiled function here to avoid numba compilation overhead during the first step
        from utils.numba_functions import (
        calculate_G_m2_numba,
        calculate_gravitational_accelerations_numba,
        integrate_euler_numba,
        integrate_leapfrog_numba_1,
        integrate_leapfrog_numba_2,
        rk4_calc_intermediate_pos_vel_numba,
        rk4_calc_intermediate_pos_numba,
        rk4_calc_final_pos_vel_numba,
        calculate_general_relativity_corrections_numba,
        )
        self._calculate_G_m2_numba = calculate_G_m2_numba
        self._calculate_gravitational_accelerations_numba = calculate_gravitational_accelerations_numba
        self._integrate_euler_numba = integrate_euler_numba
        self._integrate_leapfrog_numba_1 = integrate_leapfrog_numba_1
        self._integrate_leapfrog_numba_2 = integrate_leapfrog_numba_2
        self._rk4_calc_intermediate_pos_vel_numba = rk4_calc_intermediate_pos_vel_numba
        self._rk4_calc_intermediate_pos_numba = rk4_calc_intermediate_pos_numba
        self._rk4_calc_final_pos_vel_numba = rk4_calc_final_pos_vel_numba
        self._calculate_general_relativity_corrections_numba = calculate_general_relativity_corrections_numba

    def _calculate_G_m2(self) -> NDArray[np.float64]:
        """wrapper for numba function to calculate GM matrix"""
        mass_contig = np.ascontiguousarray(self.mass, dtype=np.float64)
        GMM = self._calculate_G_m2_numba(mass_contig, G)
        return GMM

    def _calculate_gravitational_accelerations(self, GMM: NDArray[np.float64]) -> None:
        """wrapper for numba function to calculate accelerations"""
        pos_contig = np.ascontiguousarray(self.positions, dtype=np.float64)
        GMM_contig = np.ascontiguousarray(GMM, dtype=np.float64)
        self.accelerations = self._calculate_gravitational_accelerations_numba(pos_contig, GMM_contig)
        if self.general_relativity:
            vel_contig = np.ascontiguousarray(self.velocities, dtype=np.float64)
            mass_contig = np.ascontiguousarray(self.mass, dtype=np.float64)
            self.accelerations += self._calculate_general_relativity_corrections_numba(pos_contig, vel_contig, mass_contig, G, C)

    def _integrate_euler(self, dt: np.float64) -> None:
        """wrapper for numba function to perform Euler integration"""
        vel_contig = np.ascontiguousarray(self.velocities, dtype=np.float64)
        pos_contig = np.ascontiguousarray(self.positions, dtype=np.float64)
        acc_contig = np.ascontiguousarray(self.accelerations, dtype=np.float64)
        self.velocities, self.positions = self._integrate_euler_numba(vel_contig, pos_contig, acc_contig, dt)
        # Recalculate accelerations for next step
        self._calculate_gravitational_accelerations(self.GMM)

    def _integrate_leapfrog(self, dt: np.float64) -> None:
        """wrapper for numba function to perform Leapfrog integration (kick-drift-kick, symplectic)"""
        vel_contig = np.ascontiguousarray(self.velocities, dtype=np.float64)
        pos_contig = np.ascontiguousarray(self.positions, dtype=np.float64)
        acc_contig = np.ascontiguousarray(self.accelerations, dtype=np.float64)
        # First half-kick and drift
        v_half, new_positions = self._integrate_leapfrog_numba_1(vel_contig, pos_contig, acc_contig, dt)
        # Update positions
        self.positions = new_positions
        # Recalculate accelerations at new positions
        self._calculate_gravitational_accelerations(self.GMM)
        acc_new = np.ascontiguousarray(self.accelerations, dtype=np.float64)
        # Second half-kick with new accelerations
        self.velocities = self._integrate_leapfrog_numba_2(v_half, v_half, acc_new, dt)

    def _integrate_rk4(self, dt: np.float64) -> None:
        """wrapper for numba function to perform RK4 integration (side-effect free, only update at end)"""
        dt_half = dt * 0.5
        dt_sixth = dt / 6.0

        r0 = self.positions.copy(order='C')
        v0 = self.velocities.copy(order='C')

        # k1
        self._calculate_gravitational_accelerations(self.GMM)
        k1a = self.accelerations.copy(order='C')
        k1v = v0.copy(order='C')

        # k2
        temp_pos2 = self._rk4_calc_intermediate_pos_numba(r0, k1v, dt_half)
        temp_vel2 = v0 + k1a * dt_half
        self.positions = temp_pos2
        self.velocities = temp_vel2
        self._calculate_gravitational_accelerations(self.GMM)
        k2a = self.accelerations.copy(order='C')
        k2v = temp_vel2.copy(order='C')

        # k3
        temp_pos3 = self._rk4_calc_intermediate_pos_numba(r0, k2v, dt_half)
        temp_vel3 = v0 + k2a * dt_half
        self.positions = temp_pos3
        self.velocities = temp_vel3
        self._calculate_gravitational_accelerations(self.GMM)
        k3a = self.accelerations.copy(order='C')
        k3v = temp_vel3.copy(order='C')

        # k4
        temp_pos4 = self._rk4_calc_intermediate_pos_numba(r0, k3v, dt)
        temp_vel4 = v0 + k3a * dt
        self.positions = temp_pos4
        self.velocities = temp_vel4
        self._calculate_gravitational_accelerations(self.GMM)
        k4a = self.accelerations.copy(order='C')
        k4v = temp_vel4.copy(order='C')

        # Restore original state and apply RK4 update
        self.positions = r0 + dt_sixth * (k1v + 2*k2v + 2*k3v + k4v)
        self.velocities = v0 + dt_sixth * (k1a + 2*k2a + 2*k3a + k4a)
