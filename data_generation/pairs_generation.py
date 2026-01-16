from function_utils.utils import generate_pairs
import numpy as np
import configuration.params as params

base_seed = 100
n_seed = 18

list_traj = []
for i in range(n_seed):
    list_traj.append(
        np.load(params.saving_path_obs +f"traj_dataset_{base_seed + i}.npy", mmap_mode="r")
    )

traj_complete = np.vstack(list_traj)

print(f"Trajectory complete shape {traj_complete.shape}")

pairs = generate_pairs(traj_complete)

print(f"Pairs shape {pairs.shape}")

np.save(params.saving_path_obs +f"pairs_aggregated.npy", pairs)
