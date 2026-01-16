import subprocess
import sys
import numpy as np
import time

# Configuration
n_processes = 18  # Number of parallel processes
if len(sys.argv) > 1:
    base_seed = int(sys.argv[1])
else:
    base_seed = 100 

processes = []

start = time.time()
# Start all processes
for i in range(n_processes):
    seed = base_seed + i
    proc = subprocess.Popen([sys.executable, "main_with_obs_gathering.py", str(seed)]
                            ,stdout=subprocess.DEVNULL)
    processes.append(proc)

# Wait for all to finish
for proc in processes:
    proc.wait()

end = time.time()

print(f'Time elapsed: {(end - start)/60} minutes\n')

print(f"All processes completed, full traj array saved")