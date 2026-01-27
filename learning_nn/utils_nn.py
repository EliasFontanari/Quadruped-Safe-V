import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
import mujoco
from function_utils.utils import lidar_scan

def generate_pairs(data):
    """
    Execute on n_episodes x horizon x (states + reached + violated) data
    """
    pairs = []
    for i in range(0,data.shape[0]):
        # if i % 1000 == 0:
        #     print(f'Progress: {100*i/data.shape[0]} %')
        for j in range(0,data.shape[1] - 1,1):
            if (data[i, j, :] != np.zeros(data.shape[2])).any() and (
                data[i, j + 1, :] != np.zeros(data.shape[2])
            ).any():
                pairs.append(np.hstack((data[i, j, :], data[i, j + 1, :])))
    return np.array(pairs)

def augment_dataset(data_pairs, region_start, region_end, number_artificial_samples, indexes):
    state_size = int(data_pairs.shape[1]/2 -2)
    samples_list = []
    for i in range(data_pairs.shape[0]):
        if (region_start <= data_pairs[i,:state_size][indexes]).all() and (region_end >= data_pairs[i,:state_size][indexes]).all():
            samples_list.append(data_pairs[i])
    counter = 0
    artificial_array = np.zeros((number_artificial_samples,data_pairs.shape[1]))
    while counter < number_artificial_samples:
        random_indx = np.random.randint(0,len(samples_list))
        artificial_array[counter] = samples_list[random_indx]
        counter += 1
    return np.vstack((data_pairs,artificial_array))

def remove_data_out_of_box(pairs, box_min,box_max):
    indexes_to_remove = []
    len_state = box_min.shape[0]
    for i in range(pairs.shape[0]):
        if (((pairs[i][:len_state] < box_min)+(pairs[i][:len_state] > box_max)).any()) or (((pairs[i][len_state+2:-2] < box_min)+(pairs[i][len_state+2:-2] > box_max)).any()):
            indexes_to_remove.append(i)
    pairs = np.delete(pairs,indexes_to_remove,axis=0)
    return pairs
# ====================================
#         State Normalization
# ====================================
def normalize_states(data_pairs):
    state_size = int(data_pairs.shape[1]/2 -2)
    mean_x = np.mean(data_pairs[:,:state_size], axis=0)
    std_x = np.std(data_pairs[:,:state_size], axis=0)

    mean_x_next = np.mean(data_pairs[:,-(state_size+2):-2], axis=0)
    std_x_next = np.std(data_pairs[:,-(state_size+2):-2], axis=0) 

    mean_tot = (mean_x + mean_x_next)/2
    std_tot = (std_x + std_x_next)/2

    data_pairs[:,:state_size] = (data_pairs[:,:state_size] - mean_tot) / (std_tot + 1e-8)
    data_pairs[:,-(state_size+2):-2] = (data_pairs[:,-(state_size+2):-2] - mean_tot) / (std_tot + 1e-8)
    return mean_tot,std_tot

# @jax.jit
def get_batches(pairs, batch_size, full_batches_num, partial_batch_length, key):
    # Get the dataset length
    dataset_len = pairs.shape[0]

    # Generate a permutation of indices
    rand_indx = jax.random.permutation(key, dataset_len)

    # Shuffle the dataset using the permuted indices
    shuffled_pairs = pairs[rand_indx]

    # Split the shuffled data into full batches
    full_batches = jnp.array_split(shuffled_pairs[:full_batches_num * batch_size], full_batches_num)

    
    last_batch = shuffled_pairs[-partial_batch_length:]
    full_batches.append(last_batch)

    return full_batches

import pickle
import jax
import jax.numpy as jnp
import flax.linen as nn
from params_learning import log_learning
if not(log_learning):
    class ValueNetwork(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Dense(512)(x)
            x = nn.LayerNorm()(x)
            x = nn.relu(x)
            x = nn.Dense(512)(x)
            x = nn.LayerNorm()(x)
            x = nn.relu(x)
            x = nn.Dense(512)(x)
            x = nn.LayerNorm()(x)
            x = nn.relu(x)
            x = nn.Dense(512)(x)
            x = nn.LayerNorm()(x)
            x = nn.relu(x)
            x = nn.Dense(512)(x)
            x = nn.LayerNorm()(x)
            x = nn.relu(x)
            # Output initialized to 1 (probability of survival)
            x = nn.Dense(1, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros)(x)
            x = nn.sigmoid(x) 
            # x = nn.Dense(1)(x)
            return x.squeeze(-1)
else:
    class ValueNetwork(nn.Module):
        @nn.compact
        def __call__(self, x):
            # x = nn.Dense(512)(x)
            # x = nn.LayerNorm()(x)
            # x = nn.relu(x)
            x = nn.Dense(512)(x)
            x = nn.LayerNorm()(x)
            x = nn.relu(x)
            x = nn.Dense(512)(x)
            x = nn.LayerNorm()(x)
            x = nn.relu(x)
            x = nn.Dense(512)(x)
            x = nn.LayerNorm()(x)
            x = nn.relu(x)
            x = nn.Dense(512)(x)
            x = nn.LayerNorm()(x)
            x = nn.relu(x)
            x = nn.Dense(512)(x)
            x = nn.LayerNorm()(x)
            x = nn.relu(x)
            # x = nn.Dense(1)(x)
            x = nn.Dense(1,kernel_init=nn.initializers.zeros,bias_init=nn.initializers.constant(0))(x)
            x = jnp.minimum(x, 0.0)

            # x = nn.leaky_relu(x)
            # x = nn.Dense(1)(x)
            return x.squeeze(-1)
        
from flax.training import train_state
class TrainState(train_state.TrainState):   # class equal to parent class
    pass

class FlaxCritic:
    def __init__(self, path):

        self._load_model(path)

    def _load_model(self, model_path: str):
        with open(model_path, 'rb') as f:
            data = pickle.load(f)

        self.params = data['model_params']
        self.mean = jnp.array(data['mean'])
        self.std = jnp.array(data['std'])
        self.model = ValueNetwork()
        self._inference_fn = jax.jit(self._evaluate)

    def _evaluate(self, params, obs):
        normalized_obs = (obs - self.mean) / (self.std)
        return self.model.apply(params, normalized_obs)

    def evaluate(self, obs):
        obs_jnp = jnp.array(obs)
        return self._inference_fn(self.params, obs_jnp)

    def is_safe(self, obs, threshold=0.0):
        return self.evaluate(obs) >= threshold

from torch.utils.data import Dataset, DataLoader

# Example custom dataset
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]
    
def custom_lr_schedule(epoch,lr):
    if epoch < 150:
        return lr
    else:
        return lr * 1e-2
    