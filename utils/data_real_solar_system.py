import numpy as np
from numpy.typing import NDArray
from skyfield.api import load
from .constants import large_body_mass, large_body_radius
from datetime import datetime
from typing import Union


def get_real_solar_system_data(date: Union[str, datetime] = None, include_minor_bodies: bool = False) -> dict[str, dict[str, np.float64, np.float64, NDArray[np.float64], NDArray[np.float64]]]:
    """Load real solar system data at a specific date/time in barycentric ecliptic coordinates.
    
    Positions and velocities are given relative to the solar system barycenter
    (center of mass) in the ecliptic coordinate system, ensuring proper conservation 
    of momentum in N-body simulation.
    
    Bodies included: Sun, Mercury, Venus, Earth, Moon, Mars, Jupiter, Saturn, 
    Uranus, Neptune, Pluto (11 bodies total).
    
    Optional bodies (if include_minor_bodies=True and ephemeris available):
    - Jupiter's Galilean moons: Io, Europa, Ganymede, Callisto (requires jup365.bsp)
    - Ceres (requires asteroid ephemeris)
    
    Coordinate System: Ecliptic frame (J2000.0)
        - xy-plane: Earth's orbital plane (ecliptic plane)
        - z-axis: Perpendicular to ecliptic, toward ecliptic north pole
        - x-axis: Points toward vernal equinox
    
    :param date: Date/time as string 'YYYY-MM-DD' or datetime object. Defaults to now.
    :param include_minor_bodies: If True, attempt to load minor bodies (moons, asteroids)
    :return: Dictionary with body names as keys, each containing:
        - 'name': str
        - 'mass': np.float64 (kg)
        - 'radius': np.float64 (km)
        - 'position': NDArray[np.float64] (km, ecliptic barycentric frame, shape (3))
        - 'velocity': NDArray[np.float64] (km/s, ecliptic barycentric frame, shape (3))
    
    :raise TypeError: If date is not a string or datetime object.
    :raise ValueError: If date string is not in YYYY-MM-DD format

    Usage:
        # With date string (midnight UTC)
        data = get_real_solar_system_data('2000-01-01')
        
        # With datetime object (full precision)
        from datetime import datetime
        dt = datetime(2000, 1, 1, 12, 30, 45)
        data = get_real_solar_system_data(dt)
        
        earth_pos = data['earth']['position']  # Relative to barycenter, ecliptic coords
    """
    # Load ephemeris data from JPL database
    planets = load("de421.bsp")
    
    # Parse date/time
    ts = load.timescale()
    if date is None:
        t = ts.now()
    elif isinstance(date, datetime):
        # Use full datetime precision
        t = ts.utc(date.year, date.month, date.day, 
                   date.hour, date.minute, date.second + date.microsecond / 1e6)
    elif isinstance(date, str):
        # Parse YYYY-MM-DD string (assumes midnight UTC)
        parts = date.split("-")
        if len(parts) != 3:
            raise ValueError("Date string must be in YYYY-MM-DD format.")
        try:
            year, month, day = map(int, parts)
            t = ts.utc(year, month, day)
        except ValueError:
            raise ValueError("Date string must be in YYYY-MM-DD format with valid integers.")
    else:
        raise TypeError("Date must be a string (YYYY-MM-DD) or datetime object.")
    
    # Use solar system barycenter as reference point (center of mass)
    # We'll compute the actual barycenter from all loaded bodies to ensure
    # perfect momentum conservation: Σ(m_i * r_i) = 0 and Σ(m_i * v_i) = 0
    
    body_mapping = {
        "sun": "sun",
        "mercury": "mercury",
        "venus": "venus",
        "earth": "earth",
        "moon": "moon",
        "mars": "mars",
        "jupiter": "jupiter barycenter",
        "saturn": "saturn barycenter",
        "uranus": "uranus barycenter",
        "neptune": "neptune barycenter",
        "pluto": "pluto barycenter"
    }
    
    # Obliquity of the ecliptic at J2000.0 epoch (angle between equator and ecliptic)
    # This is the tilt of Earth's rotation axis relative to its orbital plane
    OBLIQUITY_J2000 = np.radians(23.439281)  # degrees -> radians
    
    # Rotation matrix to transform from equatorial (ICRS) to ecliptic coordinates
    # Rotation around x-axis by obliquity angle
    cos_eps = np.cos(OBLIQUITY_J2000)
    sin_eps = np.sin(OBLIQUITY_J2000)
    equatorial_to_ecliptic = np.array([
        [1.0,      0.0,       0.0],
        [0.0,  cos_eps,  sin_eps],
        [0.0, -sin_eps,  cos_eps]
    ], dtype=np.float64)
    
    # First pass: collect all raw positions and velocities from Skyfield
    # These are in ICRS equatorial frame (solar system barycentric)
    raw_data = {}
    for body_key, body_name in body_mapping.items():
        body = planets[body_name]
        body_state = body.at(t)
        
        simple_name = body_name.split()[0]
        mass = large_body_mass(simple_name)
        radius = large_body_radius(simple_name)
        
        raw_data[body_key] = {
            'name': simple_name,
            'mass': mass,
            'radius': radius,
            'position': body_state.position.km.astype(np.float64),
            'velocity': body_state.velocity.km_per_s.astype(np.float64)
        }
    
    # Optionally load minor bodies (moons, asteroids)
    if include_minor_bodies:
        # Try to load Jupiter's moons (requires jup365.bsp)
        jupiter_moons = {
            "io": "io",
            "europa": "europa",
            "ganymede": "ganymede",
            "callisto": "callisto"
        }
        
        try:
            # Attempt to load Jupiter satellite ephemeris
            jup_moons = load('jup365.bsp')
            for moon_key, moon_name in jupiter_moons.items():
                try:
                    moon_body = jup_moons[moon_name]
                    moon_state = moon_body.at(t)
                    mass = large_body_mass(moon_name)
                    radius = large_body_radius(moon_name)
                    
                    raw_data[moon_key] = {
                        'name': moon_name,
                        'mass': mass,
                        'radius': radius,
                        'position': moon_state.position.km.astype(np.float64),
                        'velocity': moon_state.velocity.km_per_s.astype(np.float64)
                    }
                    print(f"  Loaded {moon_name}")
                except Exception as e:
                    print(f"  Warning: Could not load {moon_name}: {e}")
        except Exception as e:
            print(f"  Warning: Jupiter moon ephemeris (jup365.bsp) not available: {e}")
        
        # Note: Ceres requires separate asteroid ephemeris files
        # You can download from: https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/asteroids/
        # Example: codes_300ast_20100725.bsp
        print("  Note: Ceres and other asteroids require separate ephemeris files")
    
    # Compute barycenter of ONLY the bodies we're simulating
    # This ensures our N-body system has zero total momentum
    # (Different from true solar system barycenter which includes asteroids/moons/etc)
    total_mass = sum(body['mass'] for body in raw_data.values())
    barycenter_pos = sum(body['mass'] * body['position'] for body in raw_data.values()) / total_mass
    barycenter_vel = sum(body['mass'] * body['velocity'] for body in raw_data.values()) / total_mass
    
    # Second pass: transform to barycentric frame and convert to ecliptic coordinates
    bodies_data = {}
    for body_key, body_info in raw_data.items():
        # First shift to barycentric frame
        pos_barycentric = body_info['position'] - barycenter_pos
        vel_barycentric = body_info['velocity'] - barycenter_vel
        
        # Then rotate from equatorial to ecliptic coordinates
        pos_ecliptic = equatorial_to_ecliptic @ pos_barycentric
        vel_ecliptic = equatorial_to_ecliptic @ vel_barycentric
        
        bodies_data[body_key] = {
            'name': body_info['name'],
            'mass': body_info['mass'],
            'radius': body_info['radius'],
            'position': pos_ecliptic,
            'velocity': vel_ecliptic
        }

    return bodies_data