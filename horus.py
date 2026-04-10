""" Hamiltonian Orbit Resolver for Unstable Systems (HORUS)

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

HORUS is a Python package to simulate gravitational interactions between celestial bodies using the discrete element method (DEM).

"""
from solar_system import SolarSystem
from utils.universetime import Time
from utils.vtk_exporter import VTKExporter


class Horus:
    """Provide the commands to run the HORUS simulation.
    """
    def __init__(self) -> None:
        self.solar_system = SolarSystem()
        self.time: Time | None = None
        
    def forge_cosmos(self, method: str, **kwargs) -> None:
        """Initialize the simulation by creating the solar system.
        
        :param method: Creation method ('real' for JPL ephemeris data)
        :param kwargs: Additional parameters:
            - date: Start date in YYYY-MM-DD format (for method='real')
            - include_minor_bodies: Include Jupiter's moons and asteroids (default: False)
        """
        if method == "real":
            self.solar_system.create_real_solar_system(
                kwargs.get("date"), 
                include_minor_bodies=kwargs.get("include_minor_bodies", False)
            )
        elif method == "random":
            n_bodies = kwargs.get("n_bodies")
            if n_bodies is None:
                raise ValueError("For method='random', you must provide n_bodies in kwargs.")
            self.solar_system.create_random_solar_system(n_bodies)
        elif method == "manual":
            print("Manual solar system creation, please add bodies using horus.solar_system.add_bodies()")

    def forge_chronos(self, start_date: str, end_date: str, time_step: float) -> None:
        """Initialize the simulation by creating the time steps.

        :param start_date: Starting date of the simulation in YYYY-MM-DD format.
        :param end_date: Ending date of the simulation in YYYY-MM-DD format.
        :param time_step: Time step of the simulation in seconds.
        """
        self.time = Time(start_date=start_date, end_date=end_date, time_step=time_step)

    def forge_gravitas(self, model: str, integrator: str = "leapfrog", general_relativity: bool = False) -> None:
        """Initialize the physical model to compute physical interactions.
        
        :param model: Model type ('objects_only', 'vectorized_numpy', 'loop_numba')
        :param integrator: Integration method ('rk4' or 'leapfrog'). Default 'leapfrog' for better long-term accuracy.
        :param general_relativity: Whether to include general relativity corrections
        :raises ValueError: If model is not one of the valid options
        """
        valid_models = {"objects_only", "vectorized_numpy", "loop_numba"}
        
        if model not in valid_models:
            raise ValueError(f"Invalid model '{model}'. Choose from: {', '.join(sorted(valid_models))}")
        
        self.solar_system.initialize_physical_model(model, integrator=integrator, general_relativity=general_relativity)

    def unleash_chronos_nstep(self, nstep: int) -> None:
        """Shift the simulation time by a number of steps.

        :param nstep: Number of steps to shift.
        """
        if self.time is None:
            raise ValueError("Time has not been initialized.")
        if self.solar_system is None:
            raise ValueError("Solar system has not been initialized.")
        if self.solar_system.model is None:
            raise ValueError("Physical model has not been initialized.")
        self.solar_system.model.do_nstep(nstep, self.time.timestep)
        self.time.step += nstep

    def charter_aeon(self, output_dir: str = "vtk_output") -> None:
        """Setup the exporter for the simulation.
        """ 
        if self.time is None:
            raise ValueError("Time has not been initialized.")
        if self.solar_system is None:
            raise ValueError("Solar system has not been initialized.")
        if not isinstance(output_dir, str):        
            raise ValueError("Output directory must be a string.")
        self.exporter = VTKExporter(self.solar_system, output_dir=output_dir)

    def engrave_aeon(self) -> None:
        """Write the current state of the simulation to disk.
        """
        if self.time is None:
            raise ValueError("Time has not been initialized.")
        if self.solar_system is None:
            raise ValueError("Solar system has not been initialized.")
        self.exporter.export_timestep(self.time)
