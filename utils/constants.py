import numpy as np
from typing import Final

# Gravitational constant in km^3 kg^-1 s^-2 (converted from m^3)
# Standard G = 6.67430e-11 m^3/(kg·s^2)
# G in km units = 6.67430e-11 × (1 km/1000 m)^3 = 6.67430e-20 km^3/(kg·s^2)

G: Final[np.float64] = np.float64(6.67430e-20)  # gravitational constant in km^3 kg^-1 s^-2
C: Final[np.float64] = np.float64(299792.458)  # speed of light in km/s

def large_body_mass(name: str) -> np.float64:
    """Get the mass of a large celestial body by name.

    :param name: The name of the celestial body.
    :return: The mass of the celestial body in kg.
    """
    # mass in kg
    mass = {
        "sun": 1.9885e30,
        "mercury": 3.3011e23,
        "venus": 4.8675e24,
        "earth": 5.97237e24,
        "moon": 7.342e22,
        "mars": 6.4171e23,
        "ceres": 9.384e20,
        "jupiter": 1.8982e27,
        "io": 8.932e22,
        "europa": 4.800e22,
        "ganymede": 1.482e23,
        "callisto": 1.076e23,
        "saturn": 5.6834e26,
        "uranus": 8.6810e25,
        "neptune": 1.02413e26,
        "pluto": 1.303e22
    }
    return np.float64(mass.get(name, 0.0))

def large_body_radius(name: str) -> np.float64:
    """Get the radius of a large celestial body by name.

    :param name: The name of the celestial body.
    :return: The radius of the celestial body in km.
    """
    # radius in km
    radius = {
        "sun": 696340,
        "mercury": 2439.7,
        "venus": 6051.8,
        "earth": 6371.0,
        "moon": 1737.4,
        "mars": 3389.5,
        "ceres": 469.7,
        "jupiter": 69911,
        "io": 1821.6,
        "europa": 1560.8,
        "ganymede": 2634.1,
        "callisto": 2410.3,
        "saturn": 58232,
        "uranus": 25362,
        "neptune": 24622,
        "pluto": 1188.3
    }
    return np.float64(radius.get(name, 0.0))