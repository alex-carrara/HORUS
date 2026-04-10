from functools import cached_property
import numpy as np
from datetime import datetime


class SimulationTime:
    """Represents a simulation time with multiple unit accessors."""
    
    def __init__(self, seconds: float):
        """Initialize with time in seconds."""
        self._seconds = seconds
    
    @property
    def second(self) -> float:
        """Time in seconds."""
        return self._seconds
    
    @property
    def minute(self) -> float:
        """Time in minutes."""
        return np.floor(self._seconds / 60.0)
    
    @property
    def hour(self) -> float:
        """Time in hours."""
        return np.floor(self._seconds / 3600.0)
    
    @property
    def day(self) -> float:
        """Time in days."""
        return np.floor(self._seconds / 86400.0)
    
    @property
    def year(self) -> float:
        """Time in years (365.25 days)."""
        return np.floor(self._seconds / 31557600.0)
    
    @property
    def year_decimal(self) -> float:
        """Time in years (365.25 days)."""
        return self._seconds / 31557600.0
    
    def __repr__(self) -> str:
        return f"Simulation time: {self.year} years, {self.day % 365} days, {self.hour % 24} hours, {self.minute % 60} minutes, {self.second % 60} seconds"


class Time:
    """A class to manage the time steps of the simulation."""
    
    def __init__(self, start_date: str, end_date: str, time_step: float) -> None:
        """Initialize time manager.
        
        :param start_date: Starting date in YYYY-MM-DD format
        :param end_date: Ending date in YYYY-MM-DD format
        :param time_step: Time step in seconds
        """
        self._validate_time_params(start_date, end_date, time_step)
        start = np.datetime64(start_date)
        end = np.datetime64(end_date)
        self.time = np.arange(start, end, np.timedelta64(int(time_step), 's'))
        self.timestep :np.float64 = time_step
        self.step :int = 0

    @cached_property
    def nstep(self) -> int:
        """Number of time steps in the simulation."""
        return len(self.time)
    
    @property
    def simulation_time(self) -> SimulationTime:
        """Current simulation time."""
        return SimulationTime(self.step * self.timestep)
    
    @property
    def current_datetime(self) -> np.datetime64:
        """Current date in the simulation."""
        if self.step < len(self.time):
            return self.time[self.step]
        else: 
            return self.time[-1] + np.timedelta64(int((self.step - self.nstep) * self.timestep), 's')

    @staticmethod
    def _validate_time_params(start_date: str, end_date: str, time_step: float) -> None:
        """Validate the parameters for time management.
        
        :param start_date: Starting date of the simulation in YYYY-MM-DD format.
        :param end_date: Ending date of the simulation in YYYY-MM-DD format.
        :param time_step: Time step of the simulation in seconds.

        :raise ValueError: If any of the parameters are invalid.
        :raise TypeError: If any of the parameters are of incorrect type.
        """
        if start_date is None or end_date is None:
            raise ValueError("Start date and end date must be provided.")
        
        if not isinstance(start_date, str) or not isinstance(end_date, str):
            raise TypeError("Start date and end date must be strings in YYYY-MM-DD format.")
        
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Dates must be in YYYY-MM-DD format.")
        
        if start >= end:
            raise ValueError("Start date must be before end date.")
        
        if not isinstance(time_step, (int, float, np.float64)):
            raise TypeError("Time step must be a number.")
        
        if time_step <= 0:
            raise ValueError("Time step must be positive.")
