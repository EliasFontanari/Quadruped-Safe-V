import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
import configuration.params as params

pairs = np.load("observation_datasets/pairs_dataset_parallel.npy")
from utils_nn import FlaxCritic

path_policy = f"data_learning/plots/2026-01-15_18-03_alpha_0.4_lr_0.0001_batch_2048/NN_JAX_log_learning_True_epoch_1000"
V_net = FlaxCritic(path_policy)

n_sector = V_net.mean.shape[0] - 37
joints = jnp.array([0.0, 0.9, -1.7, 0.0, 0.9, -1.7, 0.0, 0.9, -1.7, 0.0, 0.9, -1.7])

query_array = jnp.hstack(
    [jnp.array([0, 0, 0.38, 1, 0, 0, 0]), joints, jnp.zeros(18), jnp.ones(n_sector) * 2]
)

from data_generation.main_with_obs_gathering import run_single_simulation
from configuration.policy_loader import load_actor_network

if "log_learning_True" in path_policy:
    print("Log Learning")
    print(f"V_net prediction = {1 - 10 **V_net.evaluate(query_array)}")
else:
    print("No Log Learning")
    print(f"V_net prediction = {V_net.evaluate(query_array)}")

actor_network = load_actor_network(params.policy_path, params.device)

n_sim = 100

summed_outcome = 0
for i in tqdm(range(n_sim)):
    summed_outcome += run_single_simulation(
        actor_network,
        10,
        20000,
        0.15,
        inital_q=query_array[:19],
        initial_vel=query_array[19:37],
    )[1]

print(f"Prob_ reaching without fails = {summed_outcome/n_sim}")
