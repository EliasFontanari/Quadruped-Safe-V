import numpy as np
import jax
import jax.numpy as jnp


pairs = np.load('observation_datasets/pairs_dataset_parallel.npy')
from learning_nn.utils_nn import FlaxCritic

V_net = FlaxCritic(f'plots/2025-12-11_17-57_alpha_0.4_lr_0.0001_batch_2048/NN_JAX_log_learning_{True}_epoch_{1000}')

query_array = jnp.array()

# from main_with_obs import