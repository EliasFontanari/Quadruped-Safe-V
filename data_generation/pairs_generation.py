from function_utils.utils import generate_pairs
import numpy as np
import configuration.params as params
from tqdm import tqdm

base_seed = 150
n_seed = 18
list_traj_files = np.hstack([np.arange(150, 150 + n_seed),np.arange(200, 200 + n_seed)]).tolist()
list_traj = []
for element in list_traj_files:
    list_traj.append(
        np.load(params.saving_path_obs +f"traj_dataset_{element}.npy", mmap_mode="r")
    )

# print(f"Trajectory complete shape {traj_complete.shape}")

pairs_list = []
for traj in tqdm(list_traj):
    pairs_list.append(generate_pairs(traj))

pairs = np.vstack(pairs_list)

print(f"Pairs shape {pairs.shape}")

np.save(params.saving_path_obs +f"pairs_aggregated.npy", pairs)

indices_reduced = [3,4,5,6,19,20,22,23,24] + list(range(36,59)) 
indices_reduced = indices_reduced + list(np.array(indices_reduced) + 59) 


np.save(params.saving_path_obs +f"pairs_aggregated_reduced.npy", pairs[:,indices_reduced])
