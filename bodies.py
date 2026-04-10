import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field


@dataclass
class LargeBody:
    """A class to store data of a large celestial body.
    """
    name: str
    mass: np.float64
    radius: np.float64
    position: NDArray[np.float64]
    velocity: NDArray[np.float64]
    acceleration: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    def __post_init__(self) -> None:
        if not isinstance(self.position, np.ndarray):
            raise TypeError("Position must be a numpy array.")
        if self.position.shape[0] != 3:
            raise ValueError("Position must be a 3D vector.")
        if self.position.dtype != np.float64:
            raise ValueError("Position must be of type np.float64.")
        
        if not isinstance(self.velocity, np.ndarray):
            raise TypeError("Velocity must be a numpy array.")
        if self.velocity.shape[0] != 3:
            raise ValueError("Velocity must be a 3D vector.")
        if self.velocity.dtype != np.float64:
            raise ValueError("Velocity must be of type np.float64.")
        
        if not isinstance(self.radius, np.float64):
            raise TypeError("Radius must be of type np.float64.")
        if self.radius <= 0:
            raise ValueError("Radius must be positive.")
        
        if not isinstance(self.mass, np.float64):
            raise TypeError("Mass must be of type np.float64.")
        if self.mass <= 0:
            raise ValueError("Mass must be positive.")
    
    def update_position(self, position: NDArray[np.float64]) -> None:
        """Update the position of the body based on its velocity and the time step.

        :param position: The new position of the body.
        """
        if not isinstance(position, np.ndarray):
            raise TypeError("Position must be a numpy array.")
        if position.shape[0] != 3:
            raise ValueError("Position must be a 3D vector.")
        if position.dtype != np.float64:
            raise ValueError("Position must be of type np.float64.")
        self.position = position

    def update_velocity(self, velocity: NDArray[np.float64]) -> None:
        """Update the velocity of the body based on the forces acting on it and the time step.

        :param velocity: The new velocity of the body.
        """
        if not isinstance(velocity, np.ndarray):
            raise TypeError("Velocity must be a numpy array.")
        if velocity.shape[0] != 3:
            raise ValueError("Velocity must be a 3D vector.")
        if velocity.dtype != np.float64:
            raise ValueError("Velocity must be of type np.float64.")
        self.velocity = velocity