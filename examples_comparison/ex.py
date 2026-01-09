"""
MuJoCo JAX (MJX) Control Example
Demonstrates physics simulation with JAX acceleration and PD control
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
XML_MODEL = """
<mujoco>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  
  <worldbody>
    <light pos="0 0 3"/>
    <geom type="plane" size="2 2 0.1" rgba=".9 .9 .9 1"/>
    
    <body name="link1" pos="0 0 1">
      <joint name="joint1" type="hinge" axis="0 1 0" limited="false"/>
      <geom type="capsule" size="0.05" fromto="0 0 0 0 0 -0.5" rgba="1 0 0 1"/>
      <body name="link2" pos="0 0 -0.5">
        <joint name="joint2" type="hinge" axis="0 1 0" limited="false"/>
        <geom type="capsule" size="0.05" fromto="0 0 0 0 0 -0.5" rgba="0 1 0 1"/>
      </body>
    </body>
  </worldbody>
  
  <actuator>
    <motor name="motor1" joint="joint1" gear="1" ctrllimited="true" ctrlrange="-5 5"/>
    <motor name="motor2" joint="joint2" gear="1" ctrllimited="true" ctrlrange="-5 5"/>
  </actuator>
</mujoco>
"""

# @jit
def pd_controller(qpos, qvel, target_pos, kp=20.0, kd=5.0):
    """PD controller for position tracking"""
    pos_error = target_pos - qpos
    vel_error = -qvel
    ctrl = kp * pos_error + kd * vel_error
    return np.clip(ctrl, -5.0, 5.0)

# @jit
def step_with_control(model, data, target_pos, kp, kd):
    """Single simulation step with PD control"""
    ctrl = pd_controller(data.qpos, data.qvel, target_pos, kp, kd)
    data.ctrl[:] = ctrl
    mujoco.mj_step(model, data)
    return data

def simulate_trajectory(model, data, target_pos, n_steps, kp=20.0, kd=5.0):
    """Simulate a trajectory with PD control"""
    # Convert data to CPU numpy for storage during simulation
    trajectory = {'qpos': [], 'qvel': [], 'ctrl': []}
    
    # Reset data to ensure clean state
    # data = mujoco.make_data(model)
    
    for _ in range(n_steps):
        # Convert to numpy on CPU to avoid CUDA memory issues
        trajectory['qpos'].append(np.array(data.qpos, copy=True))
        trajectory['qvel'].append(np.array(data.qvel, copy=True))
        trajectory['ctrl'].append(np.array(data.ctrl, copy=True))
        data = step_with_control(model, data, target_pos, kp, kd)
    
    for key in trajectory:
        trajectory[key] = np.array(trajectory[key])
    
    return trajectory, data

@jit
def pd_controller_jax(qpos, qvel, target_pos, kp=20.0, kd=5.0):
    """PD controller for position tracking"""
    pos_error = target_pos - qpos
    vel_error = -qvel
    ctrl = kp * pos_error + kd * vel_error
    return jnp.clip(ctrl, -5.0, 5.0)

@jit
def step_with_control_jax(model, data, target_pos, kp, kd):
    """Single simulation step with PD control"""
    ctrl = pd_controller_jax(data.qpos, data.qvel, target_pos, kp, kd)
    data = data.replace(ctrl=ctrl)
    data = mjx.step(model, data)
    return data

def simulate_trajectory_jax(model, data, target_pos, n_steps, kp=20.0, kd=5.0):
    """Simulate a trajectory with PD control"""
    # Convert data to CPU numpy for storage during simulation
    trajectory = {'qpos': [], 'qvel': [], 'ctrl': []}
    
    # Reset data to ensure clean state
    data = mjx.make_data(model)
    
    for _ in range(n_steps):
        # Convert to numpy on CPU to avoid CUDA memory issues
        trajectory['qpos'].append(np.array(data.qpos, copy=True))
        trajectory['qvel'].append(np.array(data.qvel, copy=True))
        trajectory['ctrl'].append(np.array(data.ctrl, copy=True))
        data = step_with_control_jax(model, data, target_pos, kp, kd)
    
    for key in trajectory:
        trajectory[key] = np.array(trajectory[key])
    
    return trajectory, data

# @jit
# def batched_rollout(model, initial_qpos_batch, target_pos_batch, n_steps, kp, kd):
#     """Simulate multiple trajectories in parallel using JAX vectorization"""
#     def single_rollout(initial_qpos, target_pos):
#         data = mjx.make_data(model)
#         data = data.replace(qpos=initial_qpos)
        
#         def scan_fn(carry_data, _):
#             new_data = step_with_control(model, carry_data, target_pos, kp, kd)
#             return new_data, new_data.qpos
        
#         _, trajectory = jax.lax.scan(scan_fn, data, None, length=n_steps)
#         return trajectory
    
#     trajectories = vmap(single_rollout)(initial_qpos_batch, target_pos_batch)
#     return trajectories

def main():
    print("=" * 70)
    print("MuJoCo CPU vs MuJoCo JAX Control Example")
    print("=" * 70)
    
    # CPU simulation

    # Load model
    print("\n1. Loading MuJoCo model...")
    mj_model = mujoco.MjModel.from_xml_string(XML_MODEL)
    mjx_model = mjx.put_model(mj_model)
    print(f"   - DoF: {mj_model.nv}")
    print(f"   - Actuators: {mj_model.nu}")
    
    # Create initial data with proper reset
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)  # Initialize properly
    mjx_data = mjx.put_data(mj_model, mj_data)
    
    # Simulation parameters
    n_steps = 1000
    dt = 0.002
    target_pos = jnp.array([jnp.pi/4, -jnp.pi/3])  # Target: 45° and -60°
    kp, kd = 20.0, 5.0
    
    print(f"\n2. Running single trajectory simulation...")
    print(f"   - Steps: {n_steps}")
    print(f"   - Duration: {n_steps * dt:.2f}s")
    print(f"   - Target: [{target_pos[0]:.3f}, {target_pos[1]:.3f}] rad")
    
    trajectory, final_data = simulate_trajectory(
        mj_model, mj_data, target_pos, n_steps, kp, kd
    )
    
    # MJX
    print(f'Jax execution')
    trajectory_jax, final_data_jax = simulate_trajectory_jax(
        mjx_model, mjx_data, target_pos, n_steps, kp, kd
    )
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    time = np.arange(n_steps) * dt
    
    # Joint positions
    axes[0].plot(time, trajectory['qpos'][:, 0], 'b-', linewidth=2, label='Joint 1')
    axes[0].plot(time, trajectory['qpos'][:, 1], 'g-', linewidth=2, label='Joint 2')
    axes[0].plot(time, trajectory_jax['qpos'][:, 0], 'b--', linewidth=2, label='Joint 1 JAX')
    axes[0].plot(time, trajectory_jax['qpos'][:, 1], 'g--', linewidth=2, label='Joint 2 JAX')
    axes[0].axhline(target_pos[0], color='b', linestyle='--', alpha=0.5, label='Target 1')
    axes[0].axhline(target_pos[1], color='g', linestyle='--', alpha=0.5, label='Target 2')
    axes[0].set_ylabel('Position (rad)', fontsize=12)
    axes[0].set_title('PD Control of Double Pendulum', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # Joint velocities
    axes[1].plot(time, trajectory['qvel'][:, 0], 'b-', linewidth=2, label='Joint 1')
    axes[1].plot(time, trajectory['qvel'][:, 1], 'g-', linewidth=2, label='Joint 2')
    axes[1].plot(time, trajectory_jax['qvel'][:, 0], 'b--', linewidth=2, label='Joint 1 JAX')
    axes[1].plot(time, trajectory_jax['qvel'][:, 1], 'g--', linewidth=2, label='Joint 2 JAX')
    axes[1].set_ylabel('Velocity (rad/s)', fontsize=12)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    # Control signals
    axes[2].plot(time, trajectory['ctrl'][:, 0], 'b-', linewidth=2, label='Motor 1')
    axes[2].plot(time, trajectory['ctrl'][:, 1], 'g-', linewidth=2, label='Motor 2')
    axes[2].plot(time, trajectory_jax['ctrl'][:, 0], 'b--', linewidth=2, label='Joint 1 JAX')
    axes[2].plot(time, trajectory_jax['ctrl'][:, 1], 'g--', linewidth=2, label='Joint 2 JAX')
    axes[2].set_ylabel('Control Torque (Nm)', fontsize=12)
    axes[2].set_xlabel('Time (s)', fontsize=12)
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plt.show()
    
    print("\n" + "=" * 70)
    print("Simulation complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
