import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
import configuration.params as params
import mujoco


pairs = np.load("observation_datasets/pairs_dataset_parallel.npy")

from learning_nn.utils_nn import FlaxCritic
from learning_nn.utils_stat_computations import generate_query_grid, plot_V_XY
from function_utils.utils import lidar_scan
from learning_nn.params_learning import log_learning
import matplotlib.pyplot as plt

path_policy = f"data_learning/plots/2026-01-26_18-30_alpha_0.1_lr_5e-05_batch_4096/NN_JAX_log_learning_False_epoch_1900"
V_net = FlaxCritic(path_policy)

n_sector = V_net.mean.shape[0] - 37
joints = jnp.array([0.0, 0.9, -1.7, 0.0, 0.9, -1.7, 0.0, 0.9, -1.7, 0.0, 0.9, -1.7])

query_array = jnp.hstack(
    [jnp.array([0, 0, 0.38, 1, 0, 0, 0]), joints, jnp.zeros(18), jnp.ones(n_sector) * 2]
)

# query_array = jnp.hstack([jnp.array([1,0,0,0,0,0,0,0,0,0]),jnp.ones(n_sector) * 2])

from data_generation.main_with_obs_gathering import run_single_simulation
from configuration.policy_loader import load_actor_network

if "log_learning_True" in path_policy:
    print("Log Learning")
    print(f"V_net prediction = {1 - 10 **V_net.evaluate(query_array)}")
else:
    print("No Log Learning")
    print(f"V_net prediction = {V_net.evaluate(query_array)}")

actor_network = load_actor_network(params.policy_path, params.device)

model = mujoco.MjModel.from_xml_path(params.scene_path)
model.opt.timestep = params.timestep

data = mujoco.MjData(model)

n_sim = 1

summed_outcome = 0
for i in tqdm(range(n_sim)):
    summed_outcome += run_single_simulation(
        model,
        actor_network,
        10,
        20000,
        0.15,
        initial_q=query_array[:19],
        initial_vel=query_array[19:37],
    )[1]

print(f"Prob reaching without fails = {summed_outcome/n_sim}")

query_grid, grid_shape = generate_query_grid(params=params,model=model,data=data)
V = V_net.evaluate(query_grid)
V = V.reshape(grid_shape)

plot_V_XY(params,V,False)
plt.show()
plt.close()

