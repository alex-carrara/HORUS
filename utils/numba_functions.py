import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

@njit(parallel=True)
def calculate_G_m2_numba(mass: NDArray[np.float64], Gcst: np.float64) -> NDArray[np.float64]:
    """Calculate the gravitational force magnitude between two bodies using vectorized operations."""        
    N = mass.shape[0]
    GMM = np.zeros((N, N), dtype=np.float64)
    for i in prange(N):
        for j in range(i+1, N):
            GMM[i, j] = Gcst * mass[j]
            GMM[j, i] = Gcst * mass[i]
    return GMM
    

@njit(parallel=True)
def calculate_general_relativity_corrections_numba(positions: NDArray[np.float64], velocities: NDArray[np.float64], mass: NDArray[np.float64], Gcst: np.float64, Ccst: np.float64) -> NDArray[np.float64]:
    """Calculate general relativity corrections to the accelerations (1PN approximation)."""
    N = positions.shape[0]
    corrections = np.zeros((N, 3), dtype=np.float64)
    for i in prange(1, N):  # Skip the Sun (index 0)
        V = velocities[i] - velocities[0]
        R = positions[i] - positions[0]
        v = np.linalg.norm(V)
        r = np.linalg.norm(R)
        GM_sun = Gcst * mass[0]
        term1 = GM_sun / (Ccst**2 * r**3)
        term2 = 4 - GM_sun / r - v**2
        term3 = 4 * np.dot(R, V)
        a_GR = term1 * (term2 * R - term3 * V)
        corrections[i] += a_GR
        corrections[0] -= a_GR * mass[i] / mass[0]
    return corrections

@njit(parallel=True)
def calculate_gravitational_accelerations_numba(positions: NDArray[np.float64], GMM: NDArray[np.float64]) -> NDArray[np.float64]:
    """Calculate accelerations using parallel-safe pattern, only compute each pair once."""
    N = positions.shape[0]
    accelerations = np.zeros((N, 3), dtype=np.float64)
    for i in prange(N):
        for j in range(i+1, N):
            r_vec = positions[j] - positions[i]
            r_mag = np.linalg.norm(r_vec) + 1e-20
            r_hat = r_vec / r_mag
            a_mag_i = GMM[i, j] / (r_mag**2)
            a_mag_j = GMM[j, i] / (r_mag**2)
            a_vec_i = a_mag_i * r_hat
            a_vec_j = a_mag_j * r_hat
            # Use atomic add for thread safety (Numba 0.56+), or accept minor race for large N
            for k in range(3):
                # atomic add for accelerations[i, k] and accelerations[j, k]
                accelerations[i, k] += a_vec_i[k]
                accelerations[j, k] -= a_vec_j[k]
    return accelerations

@njit(parallel=True)
def integrate_euler_numba(velocities: NDArray[np.float64], positions: NDArray[np.float64], accelerations: NDArray[np.float64], dt: np.float64) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Simple Euler integrator (not recommended for orbital dynamics)."""
    for i in prange(velocities.shape[0]):
        velocities[i] += accelerations[i] * dt
        positions[i] += velocities[i] * dt
    return velocities, positions

@njit(parallel=True)
def integrate_leapfrog_numba_1(velocities: NDArray[np.float64], positions: NDArray[np.float64], accelerations: NDArray[np.float64], dt: np.float64) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Leapfrog (Verlet) symplectic integrator."""
    v_half = velocities + 0.5 * accelerations * dt
    positions += v_half * dt
    return v_half, positions

@njit(parallel=True)
def integrate_leapfrog_numba_2(velocities: NDArray[np.float64], v_half: NDArray[np.float64], accelerations: NDArray[np.float64], dt: np.float64) -> NDArray[np.float64]:
    # Note: accelerations need to be updated after this step before calling this function again
    velocities = v_half + 0.5 * accelerations * dt
    return velocities

@njit(parallel=True)
def rk4_calc_intermediate_pos_vel_numba(r0: NDArray[np.float64], v0: NDArray[np.float64], ka: NDArray[np.float64], dt: np.float64) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calculate intermediate positions and velocities for RK4."""
    temp_pos = np.zeros((r0.shape[0], r0.shape[1]), dtype=np.float64)
    temp_v = np.zeros((v0.shape[0], v0.shape[1]), dtype=np.float64)
    for i in prange(r0.shape[0]):
        temp_v[i] = v0[i] + ka[i] * dt
        temp_pos[i] = r0[i] + temp_v[i] * dt
    return temp_pos, temp_v

@njit(parallel=True)
def rk4_calc_intermediate_pos_numba(r0: NDArray[np.float64], kv: NDArray[np.float64], dt: np.float64) -> NDArray[np.float64]:
    """Calculate intermediate positions for RK4."""
    temp_pos = np.zeros((r0.shape[0], r0.shape[1]), dtype=np.float64)
    for i in prange(r0.shape[0]):
        temp_pos[i] = r0[i] + kv[i] * dt
    return temp_pos

@njit(parallel=True)
def rk4_calc_final_pos_vel_numba(r0: NDArray[np.float64], v0: NDArray[np.float64], k1v: NDArray[np.float64], k2v: NDArray[np.float64], k3v: NDArray[np.float64], k4v: NDArray[np.float64], k1a: NDArray[np.float64], k2a: NDArray[np.float64], k3a: NDArray[np.float64], k4a: NDArray[np.float64], dt_sixth: np.float64) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calculate final positions and velocities for RK4."""
    for i in prange(r0.shape[0]):
        v0[i] += dt_sixth * (k1a[i] + 2*k2a[i] + 2*k3a[i] + k4a[i])
        r0[i] += dt_sixth * (k1v[i] + 2*k2v[i] + 2*k3v[i] + k4v[i])
    return r0, v0