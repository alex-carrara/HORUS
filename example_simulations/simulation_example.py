from horus import Horus

##########################################
##              parameters              ##
##########################################

start_date = "1945-01-01"
end_date = "1994-01-01"
time_step = 6*3600 # timestep in seconds
nstep_output = 7*4 # ouputs every weeks

###########################################
##             setup simulation          ##
###########################################

horus = Horus() # Initialize Simulation
horus.forge_cosmos(method="real", date=start_date, include_minor_bodies=True) # Setup solar system
horus.forge_chronos(start_date=start_date, end_date=end_date, time_step=time_step) # Setup time steps
horus.forge_gravitas(model="vectorized_numpy", integrator="euler", general_relativity=True) # Setup physical model and integrator
horus.charter_aeon(output_dir="vtk_output")  # Setup VTK exporter

###########################################
##             run simulation            ##
########################################### 

while horus.time.step < horus.time.nstep:
    horus.unleash_chronos_nstep(nstep_output)
    horus.engrave_aeon()  # Write current state to disk
    print(f"Step {horus.time.step}/{horus.time.nstep} - Simulation time: {horus.time.simulation_time}")

print("\nSimulation completed.")