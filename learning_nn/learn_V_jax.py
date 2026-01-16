# ====================================
#         TD-Learning for Safety-V(x)
# ====================================

import os
import time
import copy
import pickle
import numpy as np
from tqdm import tqdm

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

import matplotlib.pyplot as plt
from matplotlib import cm
from datetime import datetime
import sys

now = datetime.now().strftime("%Y-%m-%d_%H-%M")

from utils_nn import (
    generate_pairs,
    augment_dataset,
    normalize_states,
    get_batches,
    custom_lr_schedule,
)

import seaborn as sns

from params_learning import (
    alpha,
    log_training_target,
    log_learning,
    learning_rate,
    tau,
    batch_size,
    epochs,
)


import math
import gc

# ====================================
#      Environment Configuration
# ====================================
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_gpu_triton_gemm_any=True"
)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "True"

data_pairs = np.load(f"observation_datasets/pairs_aggregated.npy")  # 3110

state_size = int(data_pairs.shape[1] / 2 - 2)

# np.random.shuffle(data_pairs)

errors, errors_log = [], [[], []]

mean, std = normalize_states(data_pairs)
# mean,std=np.zeros(state_size),np.ones(state_size)
# _, _ = normalize_states(data_pairs[:,-(state_size+2):-2])

# ====================================
#         Convert to JAX tensors
# ====================================
np.random.shuffle(data_pairs)
# data_pairs = data_pairs[:20_000_000]
pairs_len = data_pairs.shape[0]

pairs_partition = []
sub_batch_len = 500_000
n_sub_batch = math.ceil(pairs_len / sub_batch_len)
for i in range(n_sub_batch):
    pairs_partition.append(
        jnp.array(
            data_pairs[i * sub_batch_len : min(pairs_len, (i + 1) * sub_batch_len)],
            dtype=jnp.float32,
        )
    )

print("Observation dataset shape:", data_pairs.shape)

# pairs = torch.Tensor(data_pairs)

# next_states = jnp.array(next_states, dtype=jnp.float32)
# failed = jnp.array(failed, dtype=jnp.float32)
# succeded = jnp.array(succeded, dtype=jnp.float32)


# ====================================
#    Initialize / Load Model Params
# ====================================
from utils_nn import ValueNetwork, TrainState

key = jax.random.PRNGKey(42)
model = ValueNetwork()
params = model.init(
    key, jnp.ones((1, state_size))
)  # network weights and shapes initialized
target_params = copy.deepcopy(params)

# ====================================
#       Optimizer & Train State
# ====================================

if len(sys.argv) >= 2:
    alpha = float(sys.argv[1])

epoch_change_lr = 200
full_sub_batch = data_pairs.shape[0] // sub_batch_len
sub_batch_lens = [sub_batch_len for _ in range(full_sub_batch)]
if data_pairs.shape[0] % sub_batch_len > 0:
    sub_batch_lens.append(data_pairs.shape[0] % sub_batch_len)

steps_for_epoch = 0
for sub_batch in sub_batch_lens:
    steps_for_epoch += sub_batch // batch_size
    if sub_batch_len % batch_size > 0:
        steps_for_epoch += 1

print(f"Gradient descent steps for epoch: {steps_for_epoch}")

sched = optax.piecewise_constant_schedule(
    init_value=learning_rate,
    boundaries_and_scales={200 * steps_for_epoch: 1e-1, 1000 * steps_for_epoch: 1e-1},
)

optimizer = optax.adam(learning_rate=sched)

state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)


# ====================================
#           Loss Function
# ====================================

if not (log_learning):

    @jax.jit
    def loss_fn(params, target_params, pairs):
        V_s = model.apply(params, pairs[:, :state_size])
        V_next = jax.lax.stop_gradient(
            model.apply(target_params, pairs[:, -(state_size + 2) : -2])
        )
        indicators = 1.0 - pairs[:, -1]
        target = indicators * ((1 - pairs[:, -2]) * V_next + pairs[:, -2])

        return jnp.mean(jnp.square(V_s - target))

else:

    @jax.jit
    def log_sum(l1, l2):
        # compute log(x1+x2) given: l1=log(x1) and l2=log(x2)
        return jax.lax.stop_gradient(
            jnp.maximum(l1, l2) + jnp.log10(1 + jnp.pow(10, -jnp.abs(l1 - l2)))
        )

    log_1_m_alpha = jnp.log10(1 - alpha)
    log_alpha = jnp.log10(alpha)

    target_fail = 0
    target_reached = log_training_target

    @jax.jit
    def loss_fn(params, target_params, pairs):
        V_s = model.apply(params, pairs[:, :state_size])
        V_next = jax.lax.stop_gradient(
            model.apply(target_params, pairs[:, -(state_size + 2) : -2])
        )
        indicators = 1.0 - pairs[:, -1]
        target = jnp.minimum(
            0.0,
            indicators
            * ((1 - pairs[:, -2]) * V_next / target_reached + pairs[:, -2])
            * target_reached
            + (1 - indicators) * target_fail,
        )
        l1 = V_s + log_1_m_alpha
        l2 = target + log_alpha
        target_alpha = log_sum(l1, l2)

        return jnp.mean((jnp.square(V_s - target_alpha)))


@jax.jit
def train_step(state, target_params, pairs):
    loss, grads = jax.value_and_grad(loss_fn)(state.params, target_params, pairs)
    state = state.apply_gradients(grads=grads)
    return state, loss


@jax.jit
def soft_update_target_params(target_params, params, tau):
    """
    Soft update of the target network
    """
    return jax.tree_util.tree_map(
        lambda t, p: tau * p + (1 - tau) * t, target_params, params
    )


@jax.jit
def hard_update_target_params(target_params, params):
    return soft_update_target_params(target_params, params, 0)


# @jax.jit
def get_batches(pairs, batch_size, full_batches_num, partial_batch_length, key):
    # Get the dataset length
    dataset_len = pairs.shape[0]

    # Generate a permutation of indices
    rand_indx = jax.random.permutation(key, dataset_len)

    # Shuffle the dataset using the permuted indices
    shuffled_pairs = pairs[rand_indx]

    # Split the shuffled data into full batches
    full_batches = jnp.array_split(
        shuffled_pairs[: full_batches_num * batch_size], full_batches_num
    )

    last_batch = shuffled_pairs[-partial_batch_length:]
    full_batches.append(last_batch)

    return full_batches


def plot_losses():
    # plot_error(errors,err,errors_log,step)
    plt.figure()
    plt.grid(True)
    plt.yscale("log")  # log scale on y-axis
    plt.plot(
        np.arange(len(losses)),
        losses,
        "blue",
        label="loss",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.legend()
    plt.title(f"loss")
    plt.savefig(f"{saving_path}/loss.png")
    # plt.show()
    plt.close()


# ====================================
#           Training Loop
# ====================================
rng = jax.random.PRNGKey(int(time.time()))
losses, min_losses, max_losses = [], [], []

saving_path = (
    f"data_learning/plots/{now}_alpha_{alpha}_lr_{learning_rate}_batch_{batch_size}"
)
plot_folder = os.mkdir(saving_path)

losses, errors_log_space, errors_normal_space = [], [], []
with tqdm(range(epochs), desc="Epoch") as pbar:
    for epoch in pbar:
        # if epoch == 10:
        #     new_optimizer = optax.adam(1e-4)
        #     state = TrainState.create(apply_fn=model.apply, params=state.params, tx=optimizer)
        print(f"Step: {state.opt_state[0].count}")

        epoch_loss = 0
        batch_losses = []
        for iii in range(len(pairs_partition)):
            pairs = pairs_partition[iii]
            # pairs = jnp.array(data_pairs[pairs_bounds[0]:pairs_bounds[1]])
            # Calculate the number of full batches and partial batch size
            full_batches_num = pairs.shape[0] // batch_size
            partial_batch_length = pairs.shape[0] % batch_size
            # print(f'epoch {pbar}')
            start_time = time.time()
            rng, subkey = jax.random.split(rng)
            batches = get_batches(
                pairs, batch_size, full_batches_num, partial_batch_length, rng
            )

            for batch in batches:
                # for batch in data_loader:
                batch = jnp.array(batch)
                state, loss = train_step(state, target_params, batch)
                target_params = soft_update_target_params(
                    target_params, state.params, tau
                )
                # print(f'loss: {loss}')
                epoch_loss += loss
                batch_losses.append(loss)
            # target_params = soft_update_target_params(target_params, state.params, tau)

            # if epoch % 1 == 0:
            #     target_params = hard_update_target_params(target_params,state.params)
            # print(f'sub_epoch {iii}')
            # del pairs
            # gc.collect()
            # jax.clear_caches()

        epoch_avg = epoch_loss / len(batches)
        # print(f'Mean loss: {epoch_avg}')

        losses.append(epoch_avg)
        min_losses.append(np.min(batch_losses))
        max_losses.append(np.max(batch_losses))
        pbar.set_postfix(
            {"Loss": f"{epoch_avg:.2e}", "Hz": f"{1 / (time.time() - start_time):.2f}"}
        )
        plot_losses()

        rng = subkey

        if epoch % 100 == 0:
            # ====================================
            #           Save Model
            # ====================================
            model_path = (
                f"{saving_path}/NN_JAX_log_learning_{log_learning}_epoch_{epoch}"
            )
            with open(model_path, "wb") as f:
                pickle.dump({"model_params": state.params, "mean": mean, "std": std}, f)
            print("Model saved at", model_path)
