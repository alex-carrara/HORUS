import os
n_core = 1
os.environ["OMP_NUM_THREADS"] = str(n_core)
os.environ["MKL_NUM_THREADS"] = str(n_core)
os.environ["OPENBLAS_NUM_THREADS"] = str(n_core)
os.environ["NUMBA_NUM_THREADS"] = str(n_core)
import time
from horus import Horus
import numpy as np

##########################################
##              parameters              ##
##########################################

start_date = "2026-04-11"
end_date = "2100-01-01"
time_step = 3600 # timestep in seconds
n_bodies = 2500
###########################################
##             setup simulation          ##
###########################################

horus = Horus() # Initialize Simulation
horus.forge_cosmos(method="random", n_bodies=n_bodies) # Setup random solar system
horus.forge_chronos(start_date=start_date, end_date=end_date, time_step=time_step) # Setup time steps
horus.forge_gravitas(model="loop_numba", integrator="leapfrog", general_relativity=True) # Setup physical model and integrator

###########################################
##             run simulation            ##
########################################### 

start_time = time.perf_counter()
horus.unleash_chronos_nstep(int(1e3))
end_time = time.perf_counter()
print(f"\nSimulation completed in {end_time - start_time:.2f} seconds.")