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

# Optional: Force CPU if you have CUDA issues
# jax.config.update('jax_platform_name', 'cpu')

# Model Definition - Double Pendulum
XML_MODEL ="""
    <mujoco>
      <worldbody>
        <!-- Define the ball -->
        <body pos="5.425 0 1" name="freeball">
            <joint type="free" name="j"/>
            <geom type="sphere" name="g" size="0.12" mass="0.6" pos="0 0 0" />
        </body>
        <!-- Define the field  -->
        <geom type="plane" name="field" size="7 7.5 0.1" pos="5.425 0 0" solref="-1000 0"/>
      </worldbody>
      <!-- Enable gravity for the simulation -->
      <option gravity="0 0 -9.81" integrator="Euler" timestep="0.002" density="1.2" viscosity="0.00002" />
    </mujoco>
"""


def step_CPU(model, data):
    mujoco.mj_step(model, data)
    return data

def simulate_trajectory_CPU(model, data, n_steps):
    # Convert data to CPU numpy for storage during simulation
    trajectory = {'qpos': [], 'qvel': []}
    
    # Reset data to ensure clean state
    # data = mujoco.make_data(model)
    
    for _ in range(n_steps):
        # Convert to numpy on CPU to avoid CUDA memory issues
        trajectory['qpos'].append(np.array(data.qpos, copy=True))
        trajectory['qvel'].append(np.array(data.qvel, copy=True))
        data = step_CPU(model, data)
    
    for key in trajectory:
        trajectory[key] = np.array(trajectory[key])
    
    return trajectory, data

@jax.jit
def step_JAX(model, data):
   data = mjx.step(model, data)
   return data

def simulate_trajectory_JAX(model, data, n_steps):
    # Convert data to CPU numpy for storage during simulation
    trajectory = {'qpos': [], 'qvel': []}
    
    # Reset data to ensure clean state
    # data = mujoco.make_data(model)
    
    for _ in range(n_steps):
        # Convert to numpy on CPU to avoid CUDA memory issues
        trajectory['qpos'].append(np.array(data.qpos, copy=True))
        trajectory['qvel'].append(np.array(data.qvel, copy=True))
        data = step_JAX(model, data)
    
    for key in trajectory:
        trajectory[key] = np.array(trajectory[key])
    
    return trajectory, data


def main():
    print("=" * 70)
    print("MuJoCo CPU vs MuJoCo JAX Control Example")
    print("=" * 70)
    
    # Inital position
    init_q = np.array([0,0,10,1,0,0,0])
    # Load model
    print("\n1. Loading MuJoCo model...")
    mj_model = mujoco.MjModel.from_xml_string(XML_MODEL)
    mjx_model = mjx.put_model(mj_model)
    print(f"   - DoF: {mj_model.nq}")
    print(f"   - Actuators: {mj_model.nu}")
    
    # Create initial data with proper reset
    mj_data = mujoco.MjData(mj_model)

    mj_data.qpos = init_q
    mujoco.mj_forward(mj_model, mj_data)  # Initialize properly
    mjx_data = mjx.put_data(mj_model, mj_data)
    
    # Simulation parameters
    n_steps = 1000
    dt = 0.002
    
    print(f"\n2. Running single trajectory simulation...")
    print(f"   - Steps: {n_steps}")
    print(f"   - Duration: {n_steps * dt:.2f}s")
    
    trajectory, final_data = simulate_trajectory_CPU(
        mj_model, mj_data, n_steps
    )
    
    # MJX
    print(f'Jax execution')
    trajectory_jax, final_data_jax = simulate_trajectory_JAX(
        mjx_model, mjx_data, n_steps
    )
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    time = np.arange(n_steps) * dt
    
    # Joint positions
    axes[0].plot(time, trajectory['qpos'][:, 2], 'b-', linewidth=2, label='Z')
    # axes[0].plot(time, trajectory['qpos'][:, 1], 'g-', linewidth=2, label='Joint 2')
    axes[0].plot(time, trajectory_jax['qpos'][:, 2], 'y--', linewidth=2, label='Joint 1 JAX')
    # axes[0].plot(time, trajectory_jax['qpos'][:, 1], 'r--', linewidth=2, label='Joint 2 JAX')
    axes[0].set_ylabel('Position z', fontsize=12)
    axes[0].set_title('Ball bouncing', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # Joint velocities
    axes[1].plot(time, trajectory['qvel'][:, 2], 'b-', linewidth=2, label='V_Z')
    # axes[1].plot(time, trajectory['qvel'][:, 1], 'g-', linewidth=2, label='Joint 2')
    axes[1].plot(time, trajectory_jax['qvel'][:, 2], 'y--', linewidth=2, label='Joint 1 JAX')
    # axes[1].plot(time, trajectory_jax['qvel'][:, 1], 'r--', linewidth=2, label='Joint 2 JAX')
    axes[1].set_ylabel('Velocity', fontsize=12)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plt.show()
    
    print("\n" + "=" * 70)
    print("Simulation complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
