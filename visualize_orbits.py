#!/usr/bin/env python
"""Orbit visualization for HORSU simulations

This script reads VTK files and creates an interactive matplotlib visualization
showing planet orbits and positions over time.

Usage:
    python visualize_orbits.py [options]
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from pathlib import Path
import argparse
from utils.vtk_reader import VTKReader
from typing import Final, TYPE_CHECKING
from bodies import LargeBody

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.artist import Artist

MAX_TRAIL_STATIC: Final[int] = 100 # Maximum number of trail segments to display for static trails
MIN_ALPHA: Final[float] = 0.05 # Minimum transparency for oldest trail segments
MAX_ALPHA: Final[float] = 0.9 # Maximum transparency for newest trail segments (current position)
MIN_LW: Final[float] = 0.3 # Minimum line width for oldest trail segments
MAX_LW: Final[float] = 2.5 # Maximum line width for newest trail segments (current position)
GAMMA: Final[float] = 1.5 # Exponential decay factor for trail fading (1.0=linear, >1.0=more emphasis on recent segments)

class OrbitVisualizer:
    """Visualize planetary orbits from simulation data."""

    def __init__(self, vtk_dir: str ="vtk_output", max_bodies: int | None = None, body_filter: list[str] | None =None, reference_body : str | None =None, frame_coordinate: tuple[str] | None = None):
        """Initialize orbit visualizer.
        
        :param vtk_dir: Directory containing VTK files
        :param max_bodies: Maximum number of bodies to show (0=Sun, 1=Mercury, etc.)
        :param body_filter: List of body names to include (e.g., ['sun', 'mercury', 'venus'])
                           If None, shows all bodies up to max_bodies
        :param reference_body: Name of body to use as reference frame (e.g., 'earth')
                              If None, uses barycentric frame
        """
        self.vtk_dir = Path(vtk_dir)
        self.max_bodies = max_bodies
        self.body_filter = [name.lower() for name in body_filter] if body_filter else None
        self.reference_body = reference_body.lower() if reference_body else None
        self.reader = VTKReader(str(vtk_dir))
        self.data: list[list[LargeBody]] = []  # Will store list of body lists
        self.body_names: list[str] = []
        self.body_indices: list[int] = []  # Maps filtered bodies to original indices
        self.colors: list[str] | None = None
        self.reference_body_index: int | None = None  # Index of reference body in loaded data
        self.frame_coordinate: tuple[str] | None = frame_coordinate  # Optionally set by user
    
    def estimate_revolution_times(self) -> list[int]:
        """Estimate revolution (orbital) time for each body in time steps.
        If the body does not complete a full revolution in the data, returns total length of data as fallback.
        
        :return: List of estimated revolution times in number of steps for each body
        """
        revolution_times = []

        for body_id in range(len(self.body_names)):
            positions = self.get_positions_in_frame(self.data, body_id)

            if positions is None or len(positions) < 2:
                revolution_times.append(len(self.data))
                continue

            # Use angle from center to detect full revolution
            center = np.mean(positions, axis=0)
            rel_pos = positions - center
            angles = np.arctan2(rel_pos[:, 1], rel_pos[:, 0])
            unwrapped = np.unwrap(angles)
            start_angle = unwrapped[0]
            revolved = np.where(np.abs(unwrapped - start_angle) >= 2 * np.pi)[0]
            if len(revolved) > 0:
                revolution_times.append(revolved[0])
            else:
                revolution_times.append(len(self.data))
        return revolution_times

    def load_data(self) -> None:
        """Load all VTK files using VTKReader."""
        print(f"Loading VTK files from {self.vtk_dir}...")
        
        # Load all timesteps
        all_data = self.reader.load_all_timesteps()
        
        # Get body names
        all_body_names = self.reader.get_body_names()
        if not all_body_names:
            # Fallback to default names if JSON not found
            n_bodies = len(all_data[0]) if all_data else 0
            all_body_names = [f"Body {i}" for i in range(n_bodies)]
        
        # Apply body filter if specified
        if self.body_filter:
            # Find indices of bodies to include
            self.body_indices = []
            self.body_names = []
            for i, name in enumerate(all_body_names):
                if name.lower() in self.body_filter:
                    self.body_indices.append(i)
                    self.body_names.append(name)
            
            # Filter data to only include selected bodies
            self.data = []
            for timestep_bodies in all_data:
                filtered_bodies = [timestep_bodies[i] for i in self.body_indices]
                self.data.append(filtered_bodies)
        else:
            if self.max_bodies is None:
                self.max_bodies = int(len(all_body_names))
                print("true")
            # Use all bodies up to max_bodies
            self.body_indices = list(range(min(self.max_bodies, len(all_body_names))))
            self.body_names = all_body_names[:self.max_bodies]
            self.data = [[timestep_bodies[i] for i in self.body_indices] 
                        for timestep_bodies in all_data]
        
        # Set up colors
        n_bodies = len(self.body_names)
        self.colors = plt.cm.tab20(np.linspace(0, 1, n_bodies))
        
        # Find reference body index if specified
        if self.reference_body:
            try:
                self.reference_body_index = self.body_names.index(self.reference_body)
                print(f"  Reference frame: {self.reference_body}")
            except ValueError:
                print(f"  Warning: Reference body '{self.reference_body}' not found in loaded bodies.")
                print(f"  Available bodies: {', '.join(self.body_names)}")
                print("  Using barycentric frame instead.")
                self.reference_body_index = None
        
        print(f"✓ Loaded {len(self.data)} timesteps with {n_bodies} bodies")
        print(f"  Bodies: {', '.join(self.body_names)}")
    
    def get_simulation_times(self) -> list[str]:
        """Return list of simulation times in years for each frame, using VTKReader's timestep_map or fallback to indices.
        
        return: List of time values in years corresponding to each timestep
        """
        if hasattr(self.reader, 'timestep_map') and self.reader.timestep_map:
            indices = sorted(self.reader.timestep_map.keys())
            return [self.reader.timestep_map[idx] for idx in indices]
        else:
            # fallback: 0, 1, 2, ...
            return [str(i) for i in range(len(self.data))]

    def get_positions_in_frame(self, timestep_bodies: list[list[LargeBody]], body_id: int) -> np.ndarray | None:
        """Get positions for a body across all time in the chosen reference frame.
        
        :param timestep_bodies: List of bodies at each timestep
        :param body_id: Index of the body to get positions

        :return: Array of positions in the reference frame
        """
        positions = []
        for timestep_data in timestep_bodies:
            if body_id < len(timestep_data):
                pos = timestep_data[body_id].position.copy()
                
                # Transform to reference frame if needed
                if self.reference_body_index is not None:
                    ref_pos = timestep_data[self.reference_body_index].position
                    pos -= ref_pos
                
                positions.append(pos)
        
        return np.array(positions) if positions else None
    
    def plot_orbits_static(self, output_file: str ="orbits.png") -> None:
        """Create a static plot showing all orbital paths.
        
        :param output_file: Output image file path
        """
        print("\nCreating static orbit plot...")
        
        # Setup figure
        fig, ax = plt.subplots(figsize=(12, 12))
        fig.patch.set_facecolor('black')
        
        if self.frame_coordinate is None:
            ax.set_aspect('equal')
        
        # Get simulation data
        revolution_times = self.estimate_revolution_times()
        sim_times = self.get_simulation_times()

        # Plot orbital trails and current positions for each body
        for body_id in range(len(self.body_names)):
            positions = self.get_positions_in_frame(self.data, body_id)
            if positions is not None and len(positions) > 0:
                positions_2d = positions[:, :2]
                n_trail = len(positions_2d)
                rev_time = revolution_times[body_id]
                
                # Draw full trail with fading effect
                for i in range(n_trail - 1):
                    age = n_trail - 2 - i
                    decay = min(1.0, (age / rev_time) ** GAMMA)
                    alpha = MAX_ALPHA - (MAX_ALPHA - MIN_ALPHA) * decay
                    lw = MAX_LW - (MAX_LW - MIN_LW) * decay
                    ax.plot(positions_2d[i:i+2, 0], positions_2d[i:i+2, 1],
                            color=self.colors[body_id], alpha=alpha, linewidth=lw, zorder=3)
                
                # Plot current position (glow + marker)
                ax.scatter(positions_2d[-1, 0], positions_2d[-1, 1],
                           color=self.colors[body_id], s=300, alpha=0.3, zorder=4)
                ax.scatter(positions_2d[-1, 0], positions_2d[-1, 1],
                           color=self.colors[body_id], s=100, 
                           edgecolors='white', linewidths=1, zorder=5, label=self.body_names[body_id])
        
        # Apply styling and title
        title = self._generate_plot_title(sim_times, plot_type="")
        self._setup_plot_style(ax, title)
        
        # Set manual axis limits if provided
        if self.frame_coordinate is not None:
            ax.set_aspect('auto')
            ax.set_xlim(self.frame_coordinate[0], self.frame_coordinate[1])
            ax.set_ylim(self.frame_coordinate[2], self.frame_coordinate[3])
            ax.autoscale(enable=False)
        
        plt.savefig(output_file, dpi=150, facecolor='black')
        print(f"✓ Saved static plot: {output_file}")
        plt.close()
    
    def plot_orbits_3d(self, output_file : str ="orbits_3d.png", elevation: int =20, azimuth: int =45) -> None:
        """Create a 3D plot showing orbital paths with perspective view.
        
        :param output_file: Output image file path
        :param elevation: Viewing angle elevation in degrees (default: 20)
        :param azimuth: Viewing angle azimuth in degrees (default: 45)
        """
        print("\nCreating 3D orbit plot...")
        
        # Setup 3D figure
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')
        
        # Get simulation data
        revolution_times = self.estimate_revolution_times()
        sim_times = self.get_simulation_times()

        # Track extents for axis limits
        xmin = ymin = zmin = 0
        xmax = ymax = zmax = 0

        # Plot bodies and orbital trails
        for body_id in range(len(self.body_names)):
            positions = self.get_positions_in_frame(self.data, body_id)
            if positions is not None and len(positions) > 0:
                positions_3d = positions[:, :]
                xmax = max(xmax,np.max(np.abs(positions_3d[:, 0])))
                ymax = max(ymax,np.max(np.abs(positions_3d[:, 1])))
                zmax = max(zmax,np.max(np.abs(positions_3d[:, 2])))
                xmin = min(xmin,np.min(np.abs(positions_3d[:, 0])))
                ymin = min(ymin,np.min(np.abs(positions_3d[:, 1])))
                zmin = min(zmin,np.min(np.abs(positions_3d[:, 2])))
                n_trail = len(positions_3d)
                rev_time = revolution_times[body_id]
                
                # Draw full trail with fading effect
                for i in range(n_trail - 1):
                    age = n_trail - 2 - i
                    decay = min(1.0, (age / rev_time) ** GAMMA)
                    alpha = MAX_ALPHA - (MAX_ALPHA - MIN_ALPHA) * decay
                    lw = MAX_LW - (MAX_LW - MIN_LW) * decay
                    ax.plot(positions_3d[i:i+2, 0], positions_3d[i:i+2, 1], positions_3d[i:i+2, 2],
                            color=self.colors[body_id], alpha=alpha, linewidth=lw, zorder=3)
                
                # Plot current position
                ax.scatter(positions_3d[-1, 0], positions_3d[-1, 1], positions_3d[-1, 2],
                           color=self.colors[body_id], s=300, 
                           alpha=0.3, zorder=4)
                ax.scatter(positions_3d[-1, 0], positions_3d[-1, 1], positions_3d[-1, 2],
                           color=self.colors[body_id], s=100, 
                           edgecolors='white', linewidths=1, zorder=5, label=self.body_names[body_id])

        # Apply title and styling
        title = self._generate_plot_title(sim_times, plot_type="3D")
        self._setup_plot_style(ax, title, elevation=elevation, azimuth=azimuth, 
                              box_aspect=[xmax-xmin, ymax-ymin, zmax-zmin])
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, facecolor='black')
        print(f"✓ Saved 3D plot: {output_file}")
        plt.close()
    
    def animate_orbits(self, output_file: str="orbits_animation.avi", fps: int=10, frame_skip: int=1) -> None:
        """Create an animated visualization of the orbits.
        
        :param output_file: Output video file path (.avi recommended)
        :param fps: Frames per second for output video
        :param frame_skip: Skip every N frames for faster rendering (1=all frames, 2=every other, etc.)
        """
        print("\nCreating animated visualization...")
        
        # === 1. Data Preparation ===
        data_frames = self.data[::frame_skip]
        print(f"  Using {len(data_frames)}/{len(self.data)} frames (skip={frame_skip})")
        
        n_bodies = len(self.body_names)
        revolution_times_full = self.estimate_revolution_times()
        # Scale revolution times to account for frame skipping
        revolution_times = [rt // frame_skip for rt in revolution_times_full]
        sim_times = self.get_simulation_times()
        
        # Pre-compute all trail positions for all frames
        trail_history = self._precompute_trail_history(data_frames, n_bodies)
        
        # === 2. Figure Setup ===
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_aspect('equal')
        fig.patch.set_facecolor('black')
        
        # Set plot limits
        self._setup_animation_limits(ax, trail_history)
        
        # Apply basic styling
        ax.set_facecolor('black')
        ax.set_xlabel('X (km)', color='white')
        ax.set_ylabel('Y (km)', color='white')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.2, color='white')
        
        # === 3. Initialize Plot Elements ===
        trail_collections, bodies_glow, bodies, time_text = self._setup_animation_elements(ax, n_bodies)
        
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.,
                  facecolor='black', edgecolor='white', labelcolor='white')
        
        # === 4. Pre-compute Animation Data ===
        base_colors = [self.colors[i][:3] for i in range(n_bodies)]
        time_texts = self._precompute_time_texts(data_frames, sim_times)

        # === 5. Define Animation Functions ===
        def init() -> list["Artist"]:
            """Initialize animation - clear all plot elements."""
            for lc in trail_collections:
                lc.set_segments([])
            for body in bodies_glow + bodies:
                body.set_offsets(np.empty((0, 2)))
            time_text.set_text('')
            return trail_collections + bodies_glow + bodies + [time_text]

        def animate(frame_idx: int) -> list["Artist"]:
            """Update animation for the given frame."""
            time_text.set_text(time_texts[frame_idx])
            
            for body_id in range(n_bodies):
                trail_data = trail_history[body_id][:frame_idx + 1]
                n_trail = len(trail_data)
                
                if n_trail > 1:
                    # Create line segments
                    segments = np.stack([trail_data[:-1], trail_data[1:]], axis=1)
                    
                    # Compute fade parameters efficiently
                    alphas, linewidths = self._compute_fade_parameters(n_trail, revolution_times[body_id])
                    
                    # Create RGBA colors
                    colors = np.zeros((n_trail - 1, 4))
                    colors[:, :3] = base_colors[body_id]
                    colors[:, 3] = alphas
                    
                    # Update trail
                    trail_collections[body_id].set_segments(segments)
                    trail_collections[body_id].set_colors(colors)
                    trail_collections[body_id].set_linewidths(linewidths)
                    
                    # Update body position
                    pos = trail_data[-1]
                    bodies_glow[body_id].set_offsets([pos])
                    bodies[body_id].set_offsets([pos])

            return trail_collections + bodies_glow + bodies + [time_text]
        
        # === 6. Create and Save Animation ===
        print(f"  Rendering animation ({len(data_frames)} frames at {fps} fps)...")
        anim = FuncAnimation(fig, animate, init_func=init,
                           frames=len(data_frames), interval=1000/fps, blit=True)
        
        self._save_animation(anim, output_file, fps)
        plt.close()

    def _setup_plot_style(self, ax: "Axes", title: str, elevation: int | None = None, azimuth: int | None = None, box_aspect: list[float] | None = None) -> None:
        """Apply common styling to 2D or 3D plots.
        
        :param ax: Matplotlib axes object (2D or 3D)
        :param title: Plot title
        :param elevation: Viewing angle elevation (3D only)
        :param azimuth: Viewing angle azimuth (3D only)
        :param box_aspect: Box aspect ratio for 3D plots [x, y, z]
        """
        # Detect if this is a 3D axis
        is_3d = hasattr(ax, 'zaxis')
        
        # Common styling
        ax.set_facecolor('black')
        ax.set_title(title, color='white', fontsize=14, pad=20)
        ax.set_xlabel('X (km)', color='white', labelpad=12 if is_3d else None)
        ax.set_ylabel('Y (km)', color='white', labelpad=12 if is_3d else None)
        
        if is_3d:
            # 3D-specific styling
            ax.set_zlabel('Z (km)', color='white', labelpad=12)
            if elevation is not None and azimuth is not None:
                ax.view_init(elev=elevation, azim=azimuth)
            ax.tick_params(colors='white', labelsize=9)
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('gray')
            ax.yaxis.pane.set_edgecolor('gray')
            ax.zaxis.pane.set_edgecolor('gray')
            ax.grid(True, alpha=0.2, color='white', linewidth=0.5)
            ax.xaxis._axinfo['grid']['color'] = (1, 1, 1, 0.2)
            ax.yaxis._axinfo['grid']['color'] = (1, 1, 1, 0.2)
            ax.zaxis._axinfo['grid']['color'] = (1, 1, 1, 0.2)
            if box_aspect is not None:
                ax.set_box_aspect(box_aspect)
            ax.legend(loc='upper left', facecolor='black',
                     edgecolor='white', labelcolor='white', framealpha=0.8)
        else:
            # 2D-specific styling
            ax.tick_params(colors='white')
            ax.grid(True, alpha=0.2, color='white')
            ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), borderaxespad=0.,
                      facecolor='black', edgecolor='white', labelcolor='white',
                      bbox_transform=ax.transAxes)
    
    def _compute_fade_parameters(self, n_trail: int, rev_time: int) -> tuple[np.ndarray, np.ndarray]:
        """Compute alpha and linewidth arrays for trail fading effect.
        
        :param n_trail: Number of trail points
        :param rev_time: Estimated revolution time for the body
        :return: Tuple of (alpha_array, linewidth_array)
        """
        ages = np.arange(n_trail - 2, -1, -1)
        decay = np.minimum(1.0, (ages / rev_time) ** GAMMA)
        alphas = MAX_ALPHA - (MAX_ALPHA - MIN_ALPHA) * decay
        linewidths = MAX_LW - (MAX_LW - MIN_LW) * decay
        return alphas, linewidths
    
    def _generate_plot_title(self, sim_times: list, plot_type: str = "2D") -> str:
        """Generate plot title based on simulation data and reference frame.
        
        :param sim_times: List of simulation times
        :param plot_type: Type of plot ("2D", "3D", or "Animation")
        :return: Formatted title string
        """
        start_date = sim_times[0].split("T")[0]
        end_date = sim_times[-1].split("T")[0]
        
        # Determine if we have dates or timesteps
        is_date = len(start_date) > 1
        
        # Build title based on reference frame
        if self.reference_body:
            base_title = f'Planetary Orbits {plot_type} - {self.reference_body.title()} Reference Frame'
        else:
            base_title = f'Planetary Orbits {plot_type}' if plot_type == "3D" else f'Planetary Orbits - Top View (Ecliptic Plane)'
        
        # Add time information
        if is_date:
            time_info = f'\nStart date: {start_date}, End date: {end_date}'
        else:
            time_info = f'\nStart: {start_date} timesteps, End: {sim_times[-1]} timesteps'
        
        return base_title + time_info

    def _precompute_trail_history(self, data_frames: list, n_bodies: int) -> list[np.ndarray]:
        """Pre-compute trail positions for all frames and bodies.
        
        :param data_frames: List of data frames to process
        :param n_bodies: Number of bodies to track
        :return: List of numpy arrays containing trail positions for each body
        """
        trail_history = [[] for _ in range(n_bodies)]
        
        for frame_idx in range(len(data_frames)):
            timestep_bodies = data_frames[frame_idx]
            ref_pos = None
            
            if self.reference_body_index is not None and self.reference_body_index < len(timestep_bodies):
                ref_pos = timestep_bodies[self.reference_body_index].position[:2]
            
            for body_id in range(n_bodies):
                if body_id < len(timestep_bodies):
                    pos = timestep_bodies[body_id].position[:2].copy()
                    if ref_pos is not None:
                        pos -= ref_pos
                    trail_history[body_id].append(pos)
        
        return [np.array(trail) for trail in trail_history]
    
    def _setup_animation_limits(self, ax: "Axes", trail_history: list[np.ndarray]):
        """Set up axis limits for animation based on trail data.
        
        :param ax: Matplotlib axes object
        :param trail_history: Pre-computed trail positions
        """
        all_positions = np.vstack(trail_history)
        max_extent = np.max(np.abs(all_positions)) * 1.1
        
        if self.frame_coordinate is not None:
            ax.set_xlim(self.frame_coordinate[0], self.frame_coordinate[1])
            ax.set_ylim(self.frame_coordinate[2], self.frame_coordinate[3])
        else:
            ax.set_xlim(-max_extent, max_extent)
            ax.set_ylim(-max_extent, max_extent)
    
    def _setup_animation_elements(self, ax: "Axes", n_bodies: int) -> tuple[list[LineCollection], list["Artist"], list["Artist"], "Artist"]:
        """Initialize plot elements for animation.
        
        :param ax: Matplotlib axes object
        :param n_bodies: Number of bodies
        :return: Tuple of (trail_collections, bodies_glow, bodies, time_text)
        """
        # LineCollections for efficient trail rendering
        trail_collections = []
        for i in range(n_bodies):
            lc = LineCollection([], linewidths=1.0, colors=self.colors[i], zorder=3)
            ax.add_collection(lc)
            trail_collections.append(lc)
        
        # Scatter plots for bodies (glow + marker)
        bodies_glow = [ax.scatter([], [], color=self.colors[i], s=400, alpha=0.3, zorder=4)
                       for i in range(n_bodies)]
        bodies = [ax.scatter([], [], color=self.colors[i], s=250, 
                            edgecolors='white', linewidths=2, zorder=5,
                            label=self.body_names[i])
                 for i in range(n_bodies)]
        
        # Time text overlay
        time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                           color='white', fontsize=12, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        return trail_collections, bodies_glow, bodies, time_text
    
    def _precompute_time_texts(self, data_frames: list[float], sim_times: list[str]) -> list[str]:
        """Pre-compute time text strings for all animation frames.
        
        :param data_frames: List of data frames
        :param sim_times: List of simulation times
        :return: List of formatted time strings
        """
        time_texts = []
        has_start_date = hasattr(self, 'sim_start_date') and self.sim_start_date is not None
        
        if has_start_date:
            start_date = self.sim_start_date if hasattr(self, 'sim_start_date') else None
            start_date_str = start_date.strftime('%Y-%m-%d') if start_date else ''
        
        for frame_idx in range(len(data_frames)):
            if frame_idx < len(sim_times):
                sim_time = sim_times[frame_idx]
                if has_start_date:
                    time_texts.append(f'Start date: {start_date_str}\nTime: {sim_time} yr' if isinstance(sim_time, str) else f'Start date: {start_date_str}\nTime: {sim_time:.2f} yr')
                else:
                    time_texts.append(f'Time: {sim_time} yr' if isinstance(sim_time, str) else f'Time: {sim_time:.2f} yr')
            else:
                time_texts.append(f'Frame: {frame_idx}')
        
        return time_texts
    
    def _save_animation(self, anim: FuncAnimation, output_file: str, fps: int) -> None:
        """Save animation to file using ffmpeg or pillow fallback.
        
        :param anim: FuncAnimation object
        :param output_file: Output file path
        :param fps: Frames per second
        """
        def progress_callback(current_frame: int, total_frames: int) -> None:
            """Callback function to report animation saving progress.
            
            :param current_frame: Current frame being saved
            :param total_frames: Total number of frames to save
            """
            if current_frame % 10 == 0 or current_frame == total_frames:
                print(f"  Progress: {current_frame}/{total_frames} frames ({100*current_frame/total_frames:.1f}%)", end='\r')
        
        try:
            from matplotlib.animation import FFMpegWriter
            writer = FFMpegWriter(fps=fps, bitrate=5000, codec='mpeg4')
            anim.save(output_file, writer=writer, dpi=80, progress_callback=progress_callback)
            print(f"\n✓ Saved animation: {output_file}")
        except Exception:
            print("  FFmpeg not available, trying pillow fallback...")
            output_gif = output_file.replace('.avi', '.gif')
            anim.save(output_gif, writer='pillow', fps=fps, dpi=80, progress_callback=progress_callback)
            print(f"\n✓ Saved animation as GIF: {output_gif}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Visualize planetary orbits from simulation")
    parser.add_argument('--input-dir', default='vtk_output',
                   help='Input directory with VTK files (default: vtk_output)')
    parser.add_argument('--max-bodies', type=int, default=None,
                   help='Max body ID to show (4 = up to Mars, default: all)')
    parser.add_argument('--bodies', nargs='+', default=None,
                   help='Specific bodies to show (e.g., --bodies sun mercury venus earth mars)')
    parser.add_argument('--reference-frame', type=str, default=None,
                   help='Body to use as reference frame (e.g., earth). Default is barycentric.')
    parser.add_argument('--output', default='orbits.png',
                   help='Output filename for plot (default: orbits.png)')
    parser.add_argument('--3d', dest='view_3d', action='store_true',
                   help='Generate 3D perspective view instead of 2D top view')
    parser.add_argument('--elevation', type=float, default=20,
                   help='3D view elevation angle in degrees (default: 20)')
    parser.add_argument('--azimuth', type=float, default=45,
                   help='3D view azimuth angle in degrees (default: 45)')
    parser.add_argument('--animate', action='store_true',
                   help='Generate animated AVI video instead of static plot')
    parser.add_argument('--fps', type=int, default=10,
                   help='Animation frame rate (default: 10)')
    parser.add_argument('--frame-skip', type=int, default=1,
                   help='Skip every N frames for faster rendering (default: 1, no skip)')
    parser.add_argument('--frame-coordinate', nargs=4, type=float, default=None,
                   metavar=('XMIN', 'XMAX', 'YMIN', 'YMAX'),
                   help='Set axis limits for 2D plots: xmin xmax ymin ymax (default: auto)')

    args = parser.parse_args()

    print("=" * 70)
    print("Planetary Orbit Visualizer")
    print("=" * 70)

    if args.bodies and args.max_bodies:
        print("Warning: Both --bodies and --max-bodies specified. --bodies will take precedence.")

    viz = OrbitVisualizer(vtk_dir=args.input_dir, max_bodies=args.max_bodies, body_filter=args.bodies, reference_body=args.reference_frame, frame_coordinate=args.frame_coordinate)
    
    #lod the data
    viz.load_data()

    if args.animate:
        output_name = args.output if args.output != 'orbits.png' else 'orbits_animation.avi'
        viz.animate_orbits(output_file=output_name, fps=args.fps, frame_skip=args.frame_skip)
    elif args.view_3d:
        output_name = args.output if args.output != 'orbits.png' else 'orbits_3d.png'
        viz.plot_orbits_3d(output_file=output_name, elevation=args.elevation, azimuth=args.azimuth)
    else:
        # Default: static 2D plot
        viz.plot_orbits_static(output_file=args.output)


if __name__ == "__main__":
    main()