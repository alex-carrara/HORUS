"""VTK Reader for N-body simulation data.

Read VTK files and reconstruct body objects from simulation output.
"""

import numpy as np
from pathlib import Path
import json
from typing import List, Optional, Tuple
import glob
import xml.etree.ElementTree as ET

try:
    from vtk import vtkXMLUnstructuredGridReader
    from vtk.util.numpy_support import vtk_to_numpy
    HAS_VTK = True
except ImportError:
    HAS_VTK = False
    print("Warning: VTK not available. Install with: pip install vtk")

from bodies import LargeBody


class VTKReader:
    """Read VTK simulation output and reconstruct body objects."""
    
    def __init__(self, vtk_dir: str = "vtk_output"):
        """Initialize VTK reader.
        
        :param vtk_dir: Directory containing VTK files and body_names.json
        """
        if not HAS_VTK:
            raise ImportError("VTK is required. Install with: pip install vtk")
        
        self.vtk_dir = Path(vtk_dir)
        self.body_names = self._load_body_names()
        self.timestep_map = self._load_pvd_timesteps()  # Maps file index -> time in years
        
    def _load_body_names(self) -> dict:
        """Load body names mapping from JSON file.
        
        :return: Dictionary mapping body_id to body information
        """
        json_path = self.vtk_dir / "body_names.json"
        
        if not json_path.exists():
            print(f"Warning: {json_path} not found. Body names will be numbered.")
            return {}
        
        with open(json_path, 'r') as f:
            body_map = json.load(f)
        
        # Convert string keys to integers
        return {int(k): v for k, v in body_map.items()}
    
    def _load_pvd_timesteps(self) -> dict:
        """Load timestep mapping from PVD file.
        
        Parses the solar_system.pvd file to extract actual time values.
        
        :return: Dictionary mapping file index to time in years
        """
        pvd_path = self.vtk_dir / "solar_system.pvd"
        
        if not pvd_path.exists():
            print(f"Warning: {pvd_path} not found. Timesteps will be file indices.")
            return {}
        
        try:
            tree = ET.parse(pvd_path)
            root = tree.getroot()
            
            timestep_map = {}
            for dataset in root.findall('.//DataSet'):
                timestep = str(dataset.get('timestep'))
                filename = dataset.get('file')
                
                # Extract file index from filename (e.g., "vtk_output/solar_system_000042.vtu" -> 42)
                file_stem = Path(filename).stem
                file_index = int(file_stem.split('_')[-1])
                
                timestep_map[file_index] = timestep
            
            return timestep_map
        except Exception as e:
            print(f"Warning: Failed to parse PVD file: {e}")
            return {}
    
    def load_timestep(self, timestep: int) -> List[LargeBody]:
        """Load bodies from a specific timestep.
        
        :param timestep: Timestep number (matches file naming: solar_system_XXXXXX.vtu)
        :return: List of LargeBody objects
        """
        # Find the VTK file
        vtu_file = self.vtk_dir / f"solar_system_{timestep:06d}.vtu"
        
        if not vtu_file.exists():
            raise FileNotFoundError(f"VTK file not found: {vtu_file}")
        
        # Read VTK file
        reader = vtkXMLUnstructuredGridReader()
        reader.SetFileName(str(vtu_file))
        reader.Update()
        
        output = reader.GetOutput()
        points = output.GetPoints()
        point_data = output.GetPointData()
        
        # Extract positions
        positions = vtk_to_numpy(points.GetData())
        
        # Extract data arrays
        body_ids = vtk_to_numpy(point_data.GetArray('body_id'))
        masses = vtk_to_numpy(point_data.GetArray('mass'))
        radii = vtk_to_numpy(point_data.GetArray('radius'))
        vel_x = vtk_to_numpy(point_data.GetArray('velocity_x'))
        vel_y = vtk_to_numpy(point_data.GetArray('velocity_y'))
        vel_z = vtk_to_numpy(point_data.GetArray('velocity_z'))
        
        # Reconstruct velocities
        velocities = np.column_stack([vel_x, vel_y, vel_z])
        
        # Create LargeBody objects with body_id for sorting
        body_list = []
        for i in range(len(body_ids)):
            body_id = int(body_ids[i])
            
            # Get name from mapping or use default
            if body_id in self.body_names:
                name = self.body_names[body_id]['name']
            else:
                name = f"body_{body_id}"
            
            body = LargeBody(
                name=name,
                mass=np.float64(masses[i]),
                radius=np.float64(radii[i]),
                position=positions[i].astype(np.float64),
                velocity=velocities[i].astype(np.float64)
            )
            body_list.append((body_id, body))
        
        # Sort by body_id to ensure consistent ordering across timesteps
        body_list.sort(key=lambda x: x[0])
        bodies = [body for _, body in body_list]
        
        return bodies
    
    def load_all_timesteps(self) -> List[List[LargeBody]]:
        """Load all available timesteps.
        
        :return: List of timesteps, where each timestep is a list of LargeBody objects
        """
        vtu_files = sorted(glob.glob(str(self.vtk_dir / "solar_system_*.vtu")))
        
        if not vtu_files:
            raise FileNotFoundError(f"No VTK files found in {self.vtk_dir}")
        
        all_timesteps = []
        for vtu_file in vtu_files:
            # Extract timestep number from filename
            filename = Path(vtu_file).stem
            timestep = int(filename.split('_')[-1])
            
            bodies = self.load_timestep(timestep)
            all_timesteps.append(bodies)
        
        return all_timesteps
    
    def get_available_timesteps(self) -> List[float]:
        """Get list of available timesteps in years.
        
        Returns the actual time values from the PVD file, not file indices.
        
        :return: Sorted list of timestep values in years
        """
        if self.timestep_map:
            # Return actual time values from PVD
            indices = sorted(self.timestep_map.keys())
            return [self.timestep_map[idx] for idx in indices]
        else:
            # Fallback: return file indices if PVD not available
            vtu_files = glob.glob(str(self.vtk_dir / "solar_system_*.vtu"))
            
            timesteps = []
            for vtu_file in vtu_files:
                filename = Path(vtu_file).stem
                timestep = int(filename.split('_')[-1])
                timesteps.append(timestep)
            
            return sorted(timesteps)
    
    def get_file_indices(self) -> List[int]:
        """Get list of available file indices.
        
        :return: Sorted list of file index numbers (0, 1, 2, ...)
        """
        vtu_files = glob.glob(str(self.vtk_dir / "solar_system_*.vtu"))
        
        indices = []
        for vtu_file in vtu_files:
            filename = Path(vtu_file).stem
            index = int(filename.split('_')[-1])
            indices.append(index)
        
        return sorted(indices)
    
    def get_body_names(self) -> List[str]:
        """Get list of body names in order.
        
        :return: List of body names
        """
        if not self.body_names:
            return []
        
        sorted_ids = sorted(self.body_names.keys())
        return [self.body_names[i]['name'] for i in sorted_ids]
