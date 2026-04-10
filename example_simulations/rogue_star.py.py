from horus import Horus
import numpy as np

##########################################
##              parameters              ##
##########################################

start_date = "2026-04-11"
end_date = "2100-01-01"
time_step = 3*3600 # timestep in seconds
nstep_output = 2*7*4 # ouputs every weeks

###########################################
##             setup simulation          ##
###########################################

horus = Horus() # Initialize Simulation
horus.forge_cosmos(method="real", date=start_date, include_minor_bodies=False) # Setup solar system
horus.solar_system.add_body("Rogue star", mass=np.float64(1e30), radius=np.float64(1e6), position=np.array([9e9,7.5e8,3e8], dtype=np.float64), velocity=np.array([-6.5, 0.5, -0.2], dtype=np.float64)) # Add a test body in manual mode
horus.forge_chronos(start_date=start_date, end_date=end_date, time_step=time_step) # Setup time steps
horus.forge_gravitas(model="vectorized_numpy", integrator="leapfrog", general_relativity=True) # Setup physical model and integrator
horus.charter_aeon(output_dir="vtk_output")  # Setup VTK exporter

###########################################
##             run simulation            ##
########################################### 

while horus.time.step < horus.time.nstep:
    horus.unleash_chronos_nstep(nstep_output)
    horus.engrave_aeon()  # Write current state to disk
    if horus.time.step % 5000 < nstep_output:
        print(f"Step {horus.time.step}/{horus.time.nstep} - Simulation time: {horus.time.simulation_time}")

print("\nSimulation completed.")