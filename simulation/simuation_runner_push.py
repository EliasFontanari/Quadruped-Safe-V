import mujoco
from mujoco import viewer
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from config_loader.policy_loader import load_config, load_actor_network
from config_loader.safety_vf_loader import FlaxCritic
import pygame

# Utility functions for control and quaternion math
from utils import scale_axis, swap_legs, clip_torques_in_groups, quat_rotate_inverse

import flax.linen as nn
import pickle
import jax
from jax import numpy as jnp
import os

# -------------------------------
# Environment and GPU setup
# -------------------------------
os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_triton_gemm_any=True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
os.environ["JAX_PLATFORM_NAME"] = "cpu"

device = 'cpu'
# -------------------------------
# Initialize joystick input
# -------------------------------
pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("No joystick connected")
else:
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"Detected joystick: {joystick.get_name()}")

# -------------------------------
# Parameters and thresholds
# -------------------------------
INCLINATION_THRESHOLD = 45  # degrees before torque disable

# -------------------------------
# Main simulation function
# -------------------------------
def run_simulation(config_path='config.yaml', decimation=10):

    # -------------------------------
    # Load configuration and networks
    # -------------------------------
    config = load_config(config_path)
    actor_network = load_actor_network(config)
    critic = FlaxCritic(config)

    # -------------------------------
    # Extract parameters from config
    # -------------------------------
    timestep = config['simulation']['timestep_simulation']
    default_joint_angles = np.array(config['robot']['default_joint_angles'])
    kp_custom = np.array(config['robot']['kp_custom'])
    kd_custom = np.array(config['robot']['kd_custom'])
    scaling_factors = config['scaling']

    actions = np.zeros(12)
    torques = np.zeros(12)

    # -------------------------------
    # Load MuJoCo model and initialize state
    # -------------------------------
    xml = "aliengo/aliengo.xml"
    model = mujoco.MjModel.from_xml_path(xml)
    model.opt.timestep = timestep
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    data.qpos = np.array([0., 0., 0.38, 1., 0., 0., 0.] + list(default_joint_angles))
    renderer = mujoco.Renderer(model)

    grav_tens = torch.tensor([[0., 0., -1.]], device='cpu', dtype=torch.double)

    timecounter = 0
    step_counter = 0
    current_actions = np.copy(actions)

    disable_torques = False

    # -------------------------------
    # Start MuJoCo viewer loop
    # -------------------------------
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():

            # -------------------------------
            # Periodic external disturbance (push) - Test purposes
            # -------------------------------
            if timecounter % 300 == 0 and timecounter > 10:
                data.qvel[1] += 2.0
                print("-------------------------------- ROBOT PUSHED -------------------------------- ")

            timecounter += 1
            step_start = time.time()

            # -------------------------------
            # Compute body inclination from quaternion
            # -------------------------------
            body_quat = data.qpos[3:7].copy()
            inclination = 2 * np.arcsin(np.sqrt(body_quat[1]**2 + body_quat[2]**2)) * (180 / np.pi)

            if inclination > INCLINATION_THRESHOLD:
                disable_torques = True

            # -------------------------------
            # Torque disabling logic
            # -------------------------------
            if disable_torques:
                torques = np.zeros_like(data.ctrl)
            else:
                # -------------------------------
                # Joystick input handling
                # -------------------------------
                if pygame.joystick.get_count() == 1:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            viewer.close()

                    axes = [joystick.get_axis(i) for i in [0, 1, 2, 5]]
                    summed_axes = (1 + axes[2]) - (1 + axes[3])
                    axes = np.array([axes[0], axes[1], summed_axes])

                    scaled_axes = [scale_axis(i, axes[i]) for i in range(len(axes))]
                    scaled_axes[0], scaled_axes[1] = scaled_axes[1], scaled_axes[0]
                    commands = np.array(scaled_axes)

                    threshold = 0.05
                    commands = np.array([x if abs(x) >= threshold else 0 for x in scaled_axes])
                else:
                    commands = np.zeros(3)
                    commands[0] = 1.0  # Default command forward

                # -------------------------------
                # Legs swap to match network order (see documentation)
                # -------------------------------
                joint_angles = swap_legs(data.qpos[7:].copy())
                joint_velocities = swap_legs(data.qvel[6:].copy())

                # -------------------------------
                # Compute observation vector and do inference
                # -------------------------------
                if step_counter % decimation == 0:
                    body_vel = data.qvel[3:6].copy()
                    body_quat_reordered = np.array([body_quat[1], body_quat[2], body_quat[3], body_quat[0]])
                    tensor_quat = torch.tensor(body_quat_reordered, device='cpu', dtype=torch.double).unsqueeze(0)
                    gravity_body = quat_rotate_inverse(tensor_quat, grav_tens)

                    scaled_body_vel = body_vel * scaling_factors['body_ang_vel']
                    scaled_commands = commands[:2] * scaling_factors['commands']
                    scaled_commands = np.append(scaled_commands, commands[2] * scaling_factors['body_ang_vel'])
                    scaled_gravity_body = gravity_body[0].cpu() * scaling_factors['gravity_body']
                    scaled_joint_angles = joint_angles * scaling_factors['joint_angles']
                    scaled_joint_velocities = joint_velocities * scaling_factors['joint_velocities']
                    scaled_actions = current_actions * scaling_factors['actions']

                    input_data = np.concatenate((
                        scaled_body_vel, scaled_commands, scaled_gravity_body,
                        scaled_joint_angles, scaled_joint_velocities, scaled_actions
                    ))
                    obs = torch.tensor(input_data, dtype=torch.float32).to(device)
                    obs = actor_network.norm_obs(obs)

                    with torch.no_grad():
                        current_actions = actor_network(obs).numpy()

                # -------------------------------
                # PD control for torque calculation
                # -------------------------------
                qDes = 0.5 * current_actions + default_joint_angles
                torques1 = (kp_custom * (qDes - joint_angles) - kd_custom * joint_velocities)

                # -------------------------------
                # Legs swap to match MuJoCo order (see documentation)
                # -------------------------------
                if not disable_torques:
                    torques = swap_legs(torques1)
                    torques = clip_torques_in_groups(torques)

            # -------------------------------
            # Add noise and apply torques
            # -------------------------------
            noise_std = 0.0
            noise = np.random.normal(0, noise_std, size=torques.shape)
            noisy_torques = torques + noise

            data.ctrl = noisy_torques
            mujoco.mj_step(model, data)

            # -------------------------------
            # Safety VF evaluation
            # -------------------------------
            body_quat_reordered = np.array([data.qpos[4], data.qpos[5], data.qpos[6], data.qpos[3]])
            tensor_quat = torch.tensor(body_quat_reordered, device='cpu', dtype=torch.double).unsqueeze(0)
            gravity_body = quat_rotate_inverse(tensor_quat, grav_tens)[0].cpu().numpy()

            body_ang_vel = data.qvel[3:6].copy()
            
            # -------------------------------
            # Legs swap to match network order (see documentation)
            # -------------------------------
            joint_pos = swap_legs(data.qpos[7:].copy())
            joint_vel = swap_legs(data.qvel[6:].copy())

            obs_flax_np = np.concatenate((
                gravity_body.astype(np.float32),
                body_ang_vel,
                joint_pos.astype(np.float32),
                joint_vel.astype(np.float32)
            ))

            obs_flax = jnp.array(obs_flax_np)

            V_safe = critic.evaluate(obs_flax)

            if V_safe > 0.3:
                print(f"\033[92mV_safe: {V_safe:.4f}\033[0m")
            else:
                print(f"\033[91mV_safe: {V_safe:.4f}\033[0m")

            # -------------------------------
            # Update viewer camera and visuals
            # -------------------------------
            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)
                viewer.cam.lookat[:] = data.qpos[:3]
            viewer.sync()

            step_counter += 1

            # -------------------------------
            # Real-time pacing of simulation loop
            # -------------------------------
            time_until_next_step = timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    renderer.close()

if __name__ == '__main__':
    run_simulation()
