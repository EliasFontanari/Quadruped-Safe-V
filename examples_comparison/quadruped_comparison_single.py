"""
MuJoCo JAX (MJX) Control Example
Demonstrates physics simulation with JAX acceleration
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
import mujoco
from mujoco import mjx
import numpy as np
import matplotlib.pyplot as plt
import time
from functools import partial
import jax.profiler

import os

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_GPU_TRITON_GEMM_ANY'] = 'true'

# Optional: Force CPU if you have CUDA issues
jax.config.update("jax_enable_x64", True)
# jax.config.update('jax_platform_name', 'cpu')

# Model Definition - Double Pendulum
XML_MODEL_PATH = '../aliengo/aliengo.xml'

def step_CPU(model, data, control):
    data.ctrl[:] = control
    mujoco.mj_step(model, data)

def simulate_trajectory_CPU(model, data, n_steps,controls):
    # Convert data to CPU numpy for storage during simulation
    trajectory = np.zeros((n_steps+1,model.nq+model.nv))
    
    # Reset data to ensure clean state
    # data = mujoco.make_data(model)
    trajectory[0,:] = np.hstack((data.qpos,data.qvel))
    
    for k in range(n_steps):
        # Convert to numpy on CPU to avoid CUDA memory issues
        step_CPU(model, data,controls[k])
        trajectory[k+1,:] = np.hstack((data.qpos,data.qvel))
    
    return trajectory, data

@jax.jit
def step_JAX(model, data, control):
   data = data.replace(ctrl=control)
   data = mjx.step(model, data)
   return data

def main():
    print("=" * 70)
    print("MuJoCo CPU vs MuJoCo JAX Control Example")
    print("=" * 70)
    
    controls = np.load('../control_still.npy')
    # Inital position
    init_q = np.array([0., 0., 0.38, 1., 0., 0., 0. , 0.0, 0.9, -1.7, 0.0, 0.9, -1.7, 0.0, 0.9, -1.7, 0.0, 0.9, -1.7])
    # Load model
    print("\n1. Loading MuJoCo model...")
    mj_model = mujoco.MjModel.from_xml_path(XML_MODEL_PATH)
    mjx_model = mjx.put_model(mj_model)
    print(f"   - DoF: {mj_model.nq}")
    print(f"   - Actuators: {mj_model.nu}")
    
    # Create initial data with proper reset
    mj_data = mujoco.MjData(mj_model)



    mj_data.qpos = init_q
    mujoco.mj_forward(mj_model, mj_data)  # Initialize properly
    mjx_data = mjx.put_data(mj_model, mj_data)
    mjx_data = mjx.forward(mjx_model, mjx_data)  # This normalizes internal states

    print(f'Initial vel {mjx_data.qvel}')    

    # Simulation parameters
    n_steps = 1000
    dt = 0.002

    controls = controls[:n_steps]
    
    print(f"\n2. Running single trajectory simulation...")
    print(f"   - Steps: {n_steps}")
    print(f"   - Duration: {n_steps * dt:.2f}s")
    
    print(f'CPU execution')
    start = time.time()
    trajectory, final_data = simulate_trajectory_CPU(
        mj_model, mj_data, n_steps,controls
    )

    np.save('trajectory_comparison_CPU.npy', trajectory)
    end = time.time()
    elapsed_time = end - start
    print(f'CPU execution: time elapsed {elapsed_time} seconds, steps per second {n_steps/elapsed_time}')
    
    original_traj = np.load('../traj_still.npy')    
    
    print("\n" + "=" * 70)
    print("Simulation complete!")
    print("=" * 70)

    with jax.profiler.trace("/tmp/jax-trace", create_perfetto_trace=True):
            
        # batched comparison
        n_envs = 32
        rng = jax.random.PRNGKey(0)
        rng, batch_rng = jax.random.split(rng,2)
        batch_rng = jax.random.split(batch_rng, n_envs)
        batch_state = jax.vmap(lambda rng: mjx_data.replace(qpos=mjx_data.qpos.at[:2].set(jax.random.uniform(rng, (2,),minval=-5,maxval=5))))(batch_rng)


        jit_batch_sim = jax.jit(jax.vmap(step_JAX, in_axes=(None, 0, None)))
        
        print(f'Jax execution batched: jit compilation')
        start = time.time()

        batch_state = jit_batch_sim(mjx_model,batch_state,controls[0])
        end = time.time()
        elapsed_time = end - start
        print(f'Elapsed time: {elapsed_time} seconds\n\n')

        batch_state = jax.vmap(lambda rng: mjx_data.replace(qpos=mjx_data.qpos.at[:2].set(jax.random.uniform(rng, (2,),minval=-5,maxval=5))))(batch_rng)
        
        traj_batch = jnp.zeros((n_envs,controls.shape[0] + 1, mj_model.nq + mj_model.nv))
        traj_batch = traj_batch.at[:,0].set(jnp.hstack((batch_state.qpos,batch_state.qvel)))

        print(f'Jax execution batched: for loop')
        start = time.time()

        for i in range(controls.shape[0]):
            batch_state = jit_batch_sim(mjx_model,batch_state,controls[i])
            traj_batch = traj_batch.at[:,i].set(jnp.hstack((batch_state.qpos,batch_state.qvel)))
        
        end = time.time()
        elapsed_time = end - start
        print(f'Jax execution batched without compilation in for loop: time elapsed {elapsed_time} seconds, steps per second {n_steps/(elapsed_time+1e-6)}')


        res = np.array(traj_batch)
    jax.profiler.save_device_memory_profile("memory_x32.prof")

    np.save('batched_sim.npy', res)

if __name__ == "__main__":
    main()
