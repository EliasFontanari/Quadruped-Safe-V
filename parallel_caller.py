import subprocess
import sys
import numpy as np
import time

# Configuration
n_processes = 10  # Number of parallel processes
base_seed = 100     # Starting seed

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

base_folder = 'observation_datasets/'

dataset_list = []
for i in range(n_processes):
    dataset_list.append(np.load(base_folder+f'pairs_dataset_{base_seed + i}.npy'))

end = time.time()

print(f'Time elapsed: {(end - start)/60} minutes\n')
pairs = np.vstack(dataset_list)
np.save(base_folder + 'pairs_dataset_parallel.npy',pairs)

print(f"All processes completed, full pairs array saved, shape {pairs.shape}")