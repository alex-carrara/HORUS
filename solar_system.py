import numpy as np
from numpy.typing import NDArray
from bodies import LargeBody
from utils.universetime import Time
from utils.data_real_solar_system import get_real_solar_system_data
from utils.constants import G
# Import base class and specific model classes from engine
from engine import PhysicalModel,  ModelNumpyArray, ModelObjectOriented, ModelNumba

class SolarSystem:
    """A class to represent a solar system and managing the physics.
    """
    def __init__(self) -> None:
        self.bodies = []
        self.time: Time | None = None  # setup in create_time_manager
        self.model: PhysicalModel | None = None  # setup in initialize_physical_model
    
    @property
    def nbodies(self) -> int:
        """Number of bodies in the solar system."""
        return len(self.bodies)

    def add_body(self, name: str, mass: np.float64, radius: np.float64, position: NDArray[np.float64], velocity: NDArray[np.float64]) -> None:
        """Add a body to the solar system.

        :param name: The name of the body.
        :param mass: The mass of the body (kg).
        :param radius: The radius of the body (km).
        :param position: The initial position of the body (km).
        :param velocity: The initial velocity of the body (km/s).
        """
        self.bodies.append(LargeBody(name, mass, radius, position, velocity))
        
    def create_real_solar_system(self, date: str = None, include_minor_bodies: bool = False) -> None:
        """Create a solar system with real data at a specific date.

        :param date: Date in YYYY-MM-DD format. Defaults to today.
        :param include_minor_bodies: If True, include Jupiter's moons and other minor bodies
        """
        data = get_real_solar_system_data(date, include_minor_bodies=include_minor_bodies)
        for body_name, body_data in data.items():
            self.add_body(
                name=body_name,
                mass=body_data["mass"],
                radius=body_data["radius"],
                position=body_data["position"],
                velocity=body_data["velocity"]
            )

    def create_random_solar_system(self, n_bodies: int) -> None:
        """Create a solar system with a specified number of random bodies.

        :param n_bodies: Number of bodies to create.
        """
        self.add_body('star', mass=np.float64(1.989e30), radius=np.float64(696340), position=np.zeros(3, dtype=np.float64), velocity=np.zeros(3, dtype=np.float64))
        M_star = 1.989e30  # kg (mass of the star)
        for i in range(n_bodies-1):
            name = f"Body{i+1}"
            mass = np.random.uniform(1e24, 1e28)  # Random mass between 1e24 and 1e28 kg
            radius = np.random.uniform(1e3, 1e6)  # Random radius between 1e3 and 1e6 km
            # Random distance from star (log-uniform for realism)
            r = 10**np.random.uniform(7, 9)  # 1e7 to 1e9 km
            theta = np.random.uniform(0, 2*np.pi)
            # Small inclination for some bodies
            if np.random.rand() < 0.7:
                phi = np.random.normal(0, np.deg2rad(2))  # Most in ecliptic, a few slightly inclined
            else:
                phi = np.random.uniform(0, np.pi)
            # Spherical to Cartesian
            x = r * np.cos(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.cos(phi)
            z = r * np.sin(phi)
            position = np.array([x, y, z], dtype=np.float64)
            # Circular velocity magnitude
            v_circ = np.sqrt(G * M_star / r)
            # Perpendicular direction in orbital plane
            vx = -v_circ * np.sin(theta)
            vy = v_circ * np.cos(theta)
            vz = 0.0
            # Rotate velocity vector by inclination phi
            v_vec = np.array([vx, vy, vz], dtype=np.float64)
            if phi != 0.0:
                # Rotation matrix around axis perpendicular to (x, y, 0)
                axis = np.cross([0, 0, 1], position)
                if np.linalg.norm(axis) > 1e-10:
                    axis = axis / np.linalg.norm(axis)
                    c, s = np.cos(phi), np.sin(phi)
                    C = 1 - c
                    x_, y_, z_ = axis
                    R = np.array([
                        [c + x_**2*C, x_*y_*C - z_*s, x_*z_*C + y_*s],
                        [y_*x_*C + z_*s, c + y_**2*C, y_*z_*C - x_*s],
                        [z_*x_*C - y_*s, z_*y_*C + x_*s, c + z_**2*C]
                    ])
                    v_vec = R @ v_vec
            self.add_body(name, np.float64(mass), np.float64(radius), position, v_vec)

    def initialize_physical_model(self, model: str, integrator: str = "leapfrog", general_relativity: bool = False) -> None:
        """Initialize the physical model for the solar system.
        
        :param model: Model type ('objects_only', 'vectorized_numpy', 'loop_numba')
        :param integrator: Integration method for vectorized models ('rk4' or 'leapfrog')
        :param general_relativity: Whether to include general relativity corrections

        :raises ValueError: If model is not one of the valid options
        """
        if model == "objects_only":
            self.model = ModelObjectOriented(self, integrator=integrator, general_relativity=general_relativity)
        elif model == "vectorized_numpy":
            self.model = ModelNumpyArray(self, integrator=integrator, general_relativity=general_relativity)
        elif model == "loop_numba":
            self.model = ModelNumba(self, integrator=integrator, general_relativity=general_relativity)
        else:
            raise ValueError(f"Unknown model: {model}")
