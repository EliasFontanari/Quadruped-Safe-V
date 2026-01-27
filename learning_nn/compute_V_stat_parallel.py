import tqdm
from multiprocessing import Pool
import numpy as np
from tqdm import tqdm

# load policy
from data_generation.main_with_obs_gathering import run_single_simulation
from configuration.policy_loader import load_actor_network
from data_generation.main_with_obs_gathering import run_single_simulation
from configuration import params

import copy
import os
import multiprocessing
from multiprocessing import Pool, Value, Lock
import mujoco

from learning_nn.utils_stat_computations import generate_query_grid, plot_V_XY
import matplotlib.pyplot as plt

# Global variables for multiprocessing
counter = None
lock = None
total_tasks = None

def init_worker(shared_counter, shared_lock, shared_total):
    """Initialize worker process with shared counter and lock"""
    global counter, lock, total_tasks
    counter = shared_counter
    lock = shared_lock
    total_tasks = shared_total

def rollout_worker(n_sim, init_state):
    """
    Each process runs this function.
    It loads the policy and runs rollouts in its own environment.
    """
    global counter, lock, total_tasks
    # n_sim = arguments[0]
    # init_state = arguments[1]

    # policy = copy.deepcopy(policy)
    policy = load_actor_network(params.policy_path, params.device)

    model = model = mujoco.MjModel.from_xml_path(params.scene_path)
    model.opt.timestep = params.timestep

    successes = 0
    for _ in range(n_sim):
        successes += run_single_simulation(
            model,
            policy,
            params.decimation,
            params.ep_duration,
            params.noise_std,
            initial_q=init_state[: model.nq],
            initial_vel=init_state[model.nq : (model.nq + model.nv)],
        )[1]
    print(f"Initial_state {init_state[:2]}, successes: {successes}")
    # Update counter with lock
    # with lock:
    #     counter.value += 1
    #     progress_pct = 100 * counter.value / total_tasks.value
    #     print(f"PID {os.getpid()} - Completed {counter.value}/{total_tasks.value} ({progress_pct:.1f}%) - State: {init_state[:2]}, Success rate: {successes/n_sim:.3f}")
    
    return successes / n_sim

def parallel_rollouts(init_states, n_sim=10000, n_workers=8):
    """
    Run many rollouts in parallel over a list of initial states.
    - env_fn: function that returns a new environment instance
    - policy_fn: function that returns a new policy instance
    - init_states: list of initial states for each rollout
    """
    multiprocessing.set_start_method("spawn", force=True)

    # Create shared variables
    shared_counter = Value('i', 0)
    shared_lock = Lock()
    shared_total = Value('i', len(init_states))

    tasks = [(n_sim, s) for s in init_states]

    print(f"Starting parallel rollouts: {len(init_states)} tasks with {n_workers} workers")
    with Pool(processes=n_workers) as pool:
        results = pool.starmap(rollout_worker, tasks)
    print(f"\nCompleted all {len(init_states)} tasks!")
    return np.array(results)

if __name__ == "__main__":
    stat_computation = False
    model = mujoco.MjModel.from_xml_path(params.scene_path)
    model.opt.timestep = params.timestep
    
    data = mujoco.MjData(model)
    initial_states,grid_shape = generate_query_grid(params,model,data)

    print(f'Number of initial states to test: {np.prod(grid_shape)} \n')

    if stat_computation:
        res = parallel_rollouts(initial_states, n_sim=100, n_workers=10)

        print(f"shape{res.shape}")

        np.save("learning_nn/data_V/stat_test_parallel_t_prova.npy", res)
    V = np.load("learning_nn/data_V/stat_test_parallel_t_prova.npy")
    V = V.reshape(grid_shape)

    plot_V_XY(params,V,False)
    # plt.show()
    plt.savefig("learning_nn/data_V/stat_level_curve_prova.png")
    plt.show()
