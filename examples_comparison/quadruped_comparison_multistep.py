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
import os


# Optional: Force CPU if you have CUDA issues
# jax.config.update("jax_enable_x64", True)
# jax.config.update('jax_platform_name', 'cpu')

# Model Definition - Double Pendulum
XML_MODEL_PATH = '../aliengo/aliengo.xml'

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_GPU_TRITON_GEMM_ANY'] = 'true'

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

def simulate_trajectory_JAX(model, data, n_steps,controls):
    # Convert data to CPU numpy for storage during simulation
    trajectory = jnp.zeros((n_steps+1,model.nq+model.nv))
    trajectory = trajectory.at[0,:].set(jnp.hstack((data.qpos,data.qvel)))
    
    for k in range(n_steps):
        # Convert to numpy on CPU to avoid CUDA memory issues
        trajectory = trajectory.at[k+1,:].set(jnp.hstack((data.qpos,data.qvel)))
        # print(jnp.hstack((data.qpos,data.qvel)))
        data = step_JAX(model, data,controls[k])
      
    return trajectory, data

# @partial(jax.jit, static_argnames=['n_steps'])
@jax.jit
def simulate_trajectory_scan(model, data, controls):
    """
    Scan version
    
    Args:
        model: MJX model
        data: Initial MJX data
        controls: Array of shape (n_steps, n_ctrl) with controls for each step
    
    Returns:
        trajectory array, with qpos and qvel
    """

    initial_state = jnp.hstack((data.qpos, data.qvel))
    
    # @jax.jit
    def step_fn(data, control):
        
        # Step simulation
        data = data.replace(ctrl=control)
        data = mjx.step(model, data)
        
        return data, jnp.hstack((data.qpos,data.qvel))
    
    # Run scan over all controls
    final_data, trajectory = jax.lax.scan(step_fn, data, controls)
    trajectory = jnp.vstack((initial_state, trajectory))
    
    return trajectory, final_data


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

    mj_model.opt.iterations = 2
    mj_model.opt.ls_iterations = 2

    # mj_model.opt.maxhullvert = 64

    # Disable ALL implicit collisions
    for i in range(mj_model.ngeom):
        mj_model.geom_contype[i] = 0
        mj_model.geom_conaffinity[i] = 0
    
    # jax.profiler.start_trace("/tmp/tensorboard_logs")
    
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

    
    # MJX
    print(f'Jax execution')
    start = time.time()
    trajectory_jax, final_data_jax = simulate_trajectory_JAX(
        mjx_model, mjx_data, n_steps, controls
    )
    # trajectory_jax, final_data_jax = simulate_trajectory_scan(
    #     mjx_model, mjx_data, n_steps,controls
    # )
    end = time.time()
    elapsed_time = end - start
    print(f'Jax execution with compilation: time elapsed {elapsed_time} seconds, steps per second {n_steps/(elapsed_time+1e-6)}')

    print(f'Jax execution')
    start = time.time()
    trajectory_jax, final_data_jax = simulate_trajectory_JAX(
        mjx_model, mjx_data, n_steps, controls
    )
    # trajectory_jax, final_data_jax = simulate_trajectory_scan(
    #     mjx_model, mjx_data, n_steps,controls
    # )
    end = time.time()
    elapsed_time = end - start
    print(f'Jax execution without compilation: time elapsed {elapsed_time} seconds, steps per second {n_steps/(elapsed_time+1e-6)}')
    
    # trajectory_jax = np.zeros(trajectory.shape)
    np.save('trajectory_comparison_JAX.npy', trajectory_jax)

    np.save('trajectory_comparison_both.npy', np.stack((trajectory,trajectory_jax)))

    original_traj = np.load('../traj_still.npy')    
    
    print("\n" + "=" * 70)
    print("Simulation complete!")
    print("=" * 70)

    # batched comparison
    n_envs = 8
    rng = jax.random.PRNGKey(0)
    rng, batch_rng = jax.random.split(rng,2)
    batch_rng = jax.random.split(batch_rng, n_envs)
    batch_state = jax.vmap(lambda rng: mjx_data.replace(qpos=mjx_data.qpos.at[:2].set(jax.random.uniform(rng, (2,),minval=-5,maxval=5))))(batch_rng)

    jit_batch_sim = jax.jit(jax.vmap(simulate_trajectory_scan, in_axes=(None, 0, None)))
    
    print(f'Jax execution batched')
    start = time.time()
    traj_batch, _ = jit_batch_sim(mjx_model,batch_state,controls)
    end = time.time()
    elapsed_time = end - start
    print(f'Jax execution batched with compilation: time elapsed {elapsed_time} seconds, steps per second {n_steps/(elapsed_time+1e-6)}')

    print(f'Jax execution batched')
    start = time.time()
    traj_batch, _ = jit_batch_sim(mjx_model,batch_state,controls)
    end = time.time()
    elapsed_time = end - start
    print(f'Jax execution batched without compilation: time elapsed {elapsed_time} seconds, steps per second {n_steps/(elapsed_time+1e-6)}')

    res = np.array(traj_batch)
    np.save('batched_sim.npy', res)

    # jax.profiler.stop_trace()

    # fig, axes = plt.subplots(3,1)
    # time_sim = np.arange(n_steps+1) * dt
    
    # # Joint positions
    # axes[0].plot(time_sim, trajectory[:, 0], 'b-', linewidth=2, label='X')
    # axes[1].plot(time_sim, trajectory[:, 1], 'b-', linewidth=2, label='Y')
    # axes[2].plot(time_sim, trajectory[:, 2], 'b-', linewidth=2, label='Z')

    # axes[0].plot(time_sim, trajectory_jax[:, 0], 'r--', linewidth=2, label='X_JAX')
    # axes[1].plot(time_sim, trajectory_jax[:, 1], 'r--', linewidth=2, label='Y_JAX')
    # axes[2].plot(time_sim, trajectory_jax[:, 2], 'r--', linewidth=2, label='Z_JAX')

    # axes[0].plot(time_sim, original_traj[:trajectory.shape[0], 0], 'g.', linewidth=2, label='X_Original')
    # axes[1].plot(time_sim, original_traj[:trajectory.shape[0], 1], 'g.', linewidth=2, label='Y_Original')
    # axes[2].plot(time_sim, original_traj[:trajectory.shape[0], 2], 'g.', linewidth=2, label='Z_Original')
    
    # # plt.plot(time, trajectory_jax['qpos'][:, 1], 'r--', linewidth=2, label='Joint 2 JAX')
    # axes[0].set_ylabel('X [m]', fontsize=12)
    # axes[1].set_ylabel('Y [m]', fontsize=12)
    # axes[2].set_ylabel('Z [m]', fontsize=12)

    # axes[0].set_title('QUADRUPED X Y Z', fontsize=14, fontweight='bold')
    # axes[0].legend(loc='upper right')
    # axes[1].legend(loc='upper right')
    # axes[2].legend(loc='upper right')

    # axes[0].grid(True, alpha=0.3)
    # axes[1].grid(True, alpha=0.3)
    # axes[2].grid(True, alpha=0.3)

    # plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    main()
