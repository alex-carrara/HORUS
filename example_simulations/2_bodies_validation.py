from horus import Horus
import numpy as np
from utils.constants import G
import matplotlib.pyplot as plt
import time

def setup_initial_conditions_2_bodies(horus: Horus, orbit="circular"):
    """Manually set up a simple two-body system for testing."""
    m1 = np.float64(1e30)  # Mass of body 1
    m2 = np.float64(1e24)  # Mass of body 2
    M = np.float64(m1 + m2)

    if orbit == "circular":
        r = np.array([1e8, 0, 0], dtype=np.float64)  # initial separation (km)
        r1 = np.float64(m2/M) * r  # position of body 1
        r2 = -np.float64(m1/M) * r  # position of body 2
        v = np.array([0, np.sqrt(np.float64(G) * M / np.linalg.norm(r)), 0], dtype=np.float64)  # circular velocity
        v1 = np.float64(m2/M) * v  # velocity of body 1
        v2 = -np.float64(m1/M) * v  # velocity of body 2

    horus.solar_system.add_body(name="Body1", mass=m1, radius=np.float64(1e6), position=r1, velocity=v1)
    horus.solar_system.add_body(name="Body2", mass=m2, radius=np.float64(1e6), position=r2, velocity=v2)

    #elif orbit == "elliptical":

##########################################
##              parameters              ##
##########################################

start_date = "1900-01-01"
end_date = "2000-01-01"
time_step = 3600 # timestep in seconds
nstep_output = 31*24 # ouputs every months

###########################################
##             setup simulation          ##
###########################################

horus = Horus() # Initialize Simulation
horus.forge_cosmos(method="manual") # Setup solar system
setup_initial_conditions_2_bodies(horus, orbit="circular")  # Setup initial conditions for 2-body system
horus.forge_chronos(start_date=start_date, end_date=end_date, time_step=time_step) # Setup time steps
horus.forge_gravitas(model="loop_numba", integrator="rk4", general_relativity=False) # Setup physical model and integrator
horus.charter_aeon(output_dir="vtk_output")  # Setup VTK exporter

###########################################
##             run simulation            ##
########################################### 

separation_0 = np.linalg.norm(horus.solar_system.bodies[0].position - horus.solar_system.bodies[1].position)
separation = []
L0 = np.linalg.norm(horus.solar_system.bodies[0].mass * np.cross(horus.solar_system.bodies[0].position, horus.solar_system.bodies[0].velocity) + horus.solar_system.bodies[1].mass * np.cross(horus.solar_system.bodies[1].position, horus.solar_system.bodies[1].velocity))
L = []

start_time = time.perf_counter()
while horus.time.step < horus.time.nstep:
    horus.unleash_chronos_nstep(nstep_output)
    horus.engrave_aeon()  # Write current state to disk

    # get current separation between the two bodies
    separation.append(np.linalg.norm(horus.solar_system.bodies[0].position - horus.solar_system.bodies[1].position))
    
    #calculate angular momentum
    l = horus.solar_system.bodies[0].mass * np.cross(horus.solar_system.bodies[0].position, horus.solar_system.bodies[0].velocity) + horus.solar_system.bodies[1].mass * np.cross(horus.solar_system.bodies[1].position, horus.solar_system.bodies[1].velocity)
    L.append(np.linalg.norm(l))

end_time = time.perf_counter()

print(f"\nSimulation completed in {end_time - start_time:.2f} seconds.")

orbits = np.linspace(0, 100, len(separation))

# Plot the separation between the two bodies over time
plt.figure(figsize=(10, 6))
plt.plot(orbits,(np.array(separation)-separation_0)/separation_0, label="delta(r)/r0")
plt.xlabel("Time step")
plt.ylabel("Separation (km)")
plt.title("Separation between two bodies over time")
plt.grid(True)
plt.legend()
plt.savefig('separation_plot.png')

# Plot the angular momentum over time
L = np.array(L)
plt.figure(figsize=(10, 6))
plt.plot(orbits, (L-L0)/L0, label="|L-L0|/|L0|")
plt.xlabel("Time step")
plt.ylabel("Normalized Angular Momentum")
plt.title("Angular Momentum of the System over time")
plt.grid(True)
plt.legend()
plt.savefig('angular_momentum_plot.png')