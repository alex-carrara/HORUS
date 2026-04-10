"""VTK Exporter for N-body simulation data.

Export solar system simulation data to VTK format for visualization in ParaView.
"""

import numpy as np
from pathlib import Path
import shutil
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .universetime import Time
    from ..solar_system import SolarSystem
    from ..bodies import LargeBody
    
try:
    from pyevtk.hl import pointsToVTK
    HAS_PYEVTK = True
except ImportError:
    HAS_PYEVTK = False
    print("Warning: pyevtk not installed. Install with: pip install pyevtk")


class VTKExporter:
    """Export N-body simulation data to VTK format for ParaView visualization."""
    
    def __init__(self, solar_system: "SolarSystem", output_dir: str = "vtk_output", export_names: bool = True) -> None:
        """Initialize VTK exporter.
        
        :param solar_system: The solar system instance containing bodies
        :param output_dir: Directory to save VTK files (relative or absolute path)
        :param export_names: Whether to export body names to a JSON file for reference
        """
        if not HAS_PYEVTK:
            raise ImportError("pyevtk is required. Install with: pip install pyevtk")

        self.solar_system = solar_system
        self.output_dir = Path(output_dir)

        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

        self.output_dir.mkdir(parents=True)
        self.pvd_path = self.output_dir / "solar_system.pvd"
        self.init_pvd_file()
        self.timestep_counter = 0  # To keep track of exported timesteps

        if export_names:
            self._export_body_names(self.solar_system.bodies)


    def export_timestep(self, time: "Time") -> None:
        """Export current simulation state to VTK file.
        
        :param horus: Horus simulation instance
        :param timestep: Optional timestep number (uses internal counter if not provided)
        :return: Path to created VTK file
        """
        bodies = self.solar_system.bodies
        n_bodies = len(bodies)
        
        # Extract positions (N, 3) -> separate x, y, z arrays
        positions = np.array([body.position for body in bodies])
        x = np.ascontiguousarray(positions[:, 0])
        y = np.ascontiguousarray(positions[:, 1])
        z = np.ascontiguousarray(positions[:, 2])
        
        # Extract velocities
        velocities = np.array([body.velocity for body in bodies])
        vel_x = np.ascontiguousarray(velocities[:, 0])
        vel_y = np.ascontiguousarray(velocities[:, 1])
        vel_z = np.ascontiguousarray(velocities[:, 2])
        vel_magnitude = np.ascontiguousarray(np.linalg.norm(velocities, axis=1))
        
        # Extract other properties
        masses = np.array([body.mass for body in bodies], dtype=np.float64)
        radii = np.array([body.radius for body in bodies], dtype=np.float64)
        
        # Create body IDs (useful for coloring by body type)
        body_ids = np.arange(n_bodies, dtype=np.int32)
        
        # Prepare data dictionary (only numeric types - pyevtk doesn't support strings)
        data = {
            "velocity_magnitude": vel_magnitude,
            "velocity_x": vel_x,
            "velocity_y": vel_y,
            "velocity_z": vel_z,
            "mass": masses,
            "radius": radii,
            "body_id": body_ids
        }
        
        # Save to VTK
        filename = str(self.output_dir / f"solar_system_{self.timestep_counter:06d}")
        pointsToVTK(filename, x, y, z, data=data)
        self.timestep_counter += 1
        self.update_pvd_file(filename + ".vtu", time)

        if time.step >= time.nstep:
            self.end_pvd_file() 
    
    def _export_body_names(self, bodies: list["LargeBody"]) -> None:
        """Export body names to JSON file for reference.
            
        Creates body_names.json with mapping of body_id to body properties.
        
        :param bodies: List of body objects
        """
        body_info = {}
        for i, body in enumerate(bodies):
            body_info[i] = {
                "name": body.name
            }
        
        json_path = self.output_dir / "body_names.json"
        with open(json_path, 'w') as f:
            json.dump(body_info, f, indent=2)
        
        print(f"  Exported body names to: {json_path}")

    def init_pvd_file(self) -> None:
        """Create a PVD file for time series animation in ParaView.
        """
        with open(self.pvd_path, 'w') as f:
            f.write('<?xml version="1.0"?>\n')
            f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
            f.write('  <Collection>\n')


    def update_pvd_file(self, filename: str, time: "Time") -> None:
        """Update the PVD file with a new timestep entry.
        
        :param filename: The VTK file name for the current timestep
        :param time: The current simulation time
        """
        with open(self.pvd_path, 'a') as f:
            f.write(f'    <DataSet timestep="{time.current_datetime}" file="{filename}"/>\n')


    def end_pvd_file(self) -> None:
        """Finalize the PVD file if needed (not required in this implementation).
        """
        with open(self.pvd_path, 'a') as f:
            f.write('  </Collection>\n')
            f.write('</VTKFile>\n')
        