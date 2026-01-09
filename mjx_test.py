import mujoco
from mujoco import viewer
import numpy as np
import jax
import jax.numpy as jnp
import torch
import time
import os
from tqdm import tqdm
from mujoco import mjx
from functools import partial
from itertools import combinations
import matplotlib.pyplot as plt


from config_loader.policy_loader import load_config, load_actor_network
from utils import scale_axis, swap_legs, clip_torques_in_groups
from jax_functions import quat_rotate_inverse, quat_to_yaw, potential_field_planner
import random

from torch_to_flax import flax_actor, flax_variables, norm_obs_jax

from PIL import Image

import os

# os.environ['XLA_FLAGS'] = '--xla_gpu_autotune_level=0'
# jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
# jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
# jax.config.update("jax_enable_x64", True)
# === Thresholds and constants ===
INCLINATION_THRESHOLD = 45.0  # degrees
FALL_HEIGHT_THRESHOLD = 0.2   # meters
CP_SAFE_RADIUS = 0.05         # meters
G = 9.81
robot_rad = 0.5
reached_target_rad = 0.05
vel_target_reached = 0.1

from params_quad_obs import vx_bound,vy_bound,yaw_bound,obstacles_list,target,gain_attraction,gain_repulsion,yaw_gain, n_moving_obstacle, n_obs, n_sector
from utils_obs import obstacle_circe, lidar_scan, get_pairs_collision,list_geoms,list_bodies

config_path = "config.yaml"
config = load_config(config_path)
actor_network = load_actor_network(config)

timestep = 0.002 # 500Hz  # config['simulation']['timestep_simulation']
default_joint_angles = np.array(config['robot']['default_joint_angles'])
kp_custom = np.array(config['robot']['kp_custom'])
kd_custom = np.array(config['robot']['kd_custom'])
scaling_factors = config['scaling']

# Init model and data
model = mujoco.MjModel.from_xml_path("aliengo/scene.xml")
model.opt.timestep = timestep
data = mujoco.MjData(model)
q0 = np.array([0., 0., 0.38, 1., 0., 0., 0.] + list(default_joint_angles))
data.qpos[:]=q0
mujoco.mj_forward(model, data)

@partial(jax.jit, static_argnames=['decimation'])
def run_single_step(mjx_my_model,mjx_my_data,q_init, decimation=16, noise_std=1.0, warmup_time=1.0, seed_key = 0):
    
    mjx_my_data.qpos.at[:].set(q_init[:data.qpos.shape[0]])
    mjx_my_data.qvel.at[:].set(q_init[:data.qvel.shape[0]])
    mjx_my_data = mjx.forward(mjx_model, mjx_my_data)
    
    @jax.jit
    def potential_field_planner(position,target):
        F_nav = gain_attraction*(target - position)
        return F_nav

    @jax.jit
    def quat_rotate_inverse(q, v):
        # Extract the shape and quaternion components
        q_w = q[0]  # Scalar part of the quaternion
        q_vec = q[1:]  # Vector part of the quaternion

        # Compute each term in the rotation
        a = v * (2.0 * q_w ** 2 - 1.0)  # Ensure proper shape for broadcasting
        b = jnp.cross(q_vec, v) * q_w * 2.0
        c = jnp.dot(q_vec,(jnp.dot(q_vec,v)))*2

        # Return the result of quaternion rotation
        return a - b + c

    @jax.jit
    def quat_to_yaw(quat):
        w,x, y, z = quat
        yaw = jnp.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        return yaw

    key = jax.random.key(seed_key)  # Use any seed for reproducibility
    new_key, subkey = jax.random.split(key)
        
    grav_tens = jnp.array([0., 0., -1.])
    current_actions = jnp.zeros(12)
    
    @jax.jit
    def swap_legs(array):
        """
        Swap the front and rear legs of the array based on predefined indices.
        
        The swap logic is fixed:
        - Swap front legs (indices 3:6) with (0:3)
        - Swap rear legs (indices 9:12) with (6:9)
        
        Parameters:
        - array: np.array
            The input array to modify.
        
        Returns:
        - np.array: The modified array with swapped segments.
        """
        a = array  # no copying needed, we'll build a new one

        # Extract segments
        front_1 = a[..., 0:3]
        front_2 = a[..., 3:6]
        rear_1  = a[..., 6:9]
        rear_2  = a[..., 9:12]

        # Reassemble with swapped segments
        return jnp.concatenate([front_2, front_1, rear_2, rear_1], axis=-1)

    @jax.jit
    def clip_torques_in_groups(torques):
        """
        Clip the elements of the `torques` array in groups of 3 with different ranges for each element.
        - The first and second elements in the group are clipped to [-35.0, 35.0]
        - The third element in the group is clipped to [-45.0, 45.0]
        
        Parameters:
        - torques: np.array
            The array of torques to modify.
            
        Returns:
        - np.array: The modified array with clipped values.
        """
        group_1_mask = jnp.clip(torques[::3], -35.0, 35.0)
        # Second group: clip in range (-35, 35)
        group_2_mask = jnp.clip(torques[1::3], -35.0, 35.0)
        # Third group: clip in range (-45, 45)
        group_3_mask = jnp.clip(torques[2::3], -45.0, 45.0)

        # Reconstruct the clipped torques
        clipped_torques = jnp.empty_like(torques)
        clipped_torques = clipped_torques.at[::3].set(group_1_mask)   # Apply first group clipping
        clipped_torques = clipped_torques.at[1::3].set(group_2_mask)   # Apply second group clipping
        clipped_torques = clipped_torques.at[2::3].set(group_3_mask)   # Apply third group clipping

        return clipped_torques
    
    @jax.jit
    def compute_action(current_actions):
        body_vel = qvel[3:6].copy()
        gravity_body = quat_rotate_inverse(body_quat, grav_tens)
        scaled_body_vel = body_vel * scaling_factors['body_ang_vel']
        scaled_commands = commands[:2] * scaling_factors['commands']
        scaled_commands = jnp.append(scaled_commands, commands[2] * scaling_factors['body_ang_vel'])
        scaled_gravity_body = gravity_body * scaling_factors['gravity_body']
        scaled_joint_angles = joint_angles * scaling_factors['joint_angles']
        scaled_joint_velocities = joint_velocities * scaling_factors['joint_velocities']
        scaled_actions = current_actions * scaling_factors['actions']

        input_data = jnp.concatenate((scaled_body_vel, scaled_commands, scaled_gravity_body,
                                    scaled_joint_angles, scaled_joint_velocities, scaled_actions))
    
        obs_jax = norm_obs_jax(input_data)

        # print(f' Difference obs_torch-jax {np.linalg.norm(obs.numpy() - obs_jax)}')
        return jax.lax.stop_gradient(flax_actor.apply(flax_variables,obs_jax))
    
    def step_fn(data, _):
        data = mjx.step(mjx_model, data)
        return data, None
    
    robot_pos = mjx_my_data.qpos[:3]
    robot_or = mjx_my_data.qpos[3:7]
    yaw = quat_to_yaw(robot_or)
    field_vel = potential_field_planner(robot_pos[:2],target[:2])


    commands = jnp.array([
        jnp.clip(field_vel[0],vx_bound[0],vx_bound[1]) ,   # vx
        jnp.clip(field_vel[1],vy_bound[0],vy_bound[1]) ,   # vy
        (jnp.arctan2((target[1] - robot_pos[1]),(target[0] - robot_pos[0])) - yaw)*yaw_gain     # yaw_rate
        ])
    
    qpos = mjx_my_data.qpos.copy()
    body_quat = qpos[3:7]
    
    
    qvel = mjx_my_data.qvel.copy()
    joint_angles = swap_legs(qpos[7:].copy())
    joint_velocities = swap_legs(qvel[6:])

    # current_actions = jax.lax.cond(step == 0,compute_action,recover_action,current_actions)
    current_actions = compute_action(current_actions)

    qDes = 0.5 * current_actions + default_joint_angles
    torques = kp_custom * (qDes - joint_angles) - kd_custom * joint_velocities
    torques = swap_legs(torques)

    ctrl = mjx_my_data.ctrl.at[:].set(torques)

    mjx_my_data = mjx_my_data.replace(ctrl=ctrl)

    mjx_my_data, _ = jax.lax.scan(step_fn, mjx_data, None, length=decimation) 

    qpos_after = mjx_my_data.qpos.copy()
    qvel_after = mjx_my_data.qvel.copy()

    return mjx_data, qpos_after

@partial(jax.jit, static_argnames=['decimation'])
def step_with_torques(mjx_model, mjx_data, torques, n_step, decimation=-1):
    """Apply torques and step simulation"""

    @jax.jit
    def swap_legs(array):
        """
        Swap the front and rear legs of the array based on predefined indices.
        
        The swap logic is fixed:
        - Swap front legs (indices 3:6) with (0:3)
        - Swap rear legs (indices 9:12) with (6:9)
        
        Parameters:
        - array: np.array
            The input array to modify.
        
        Returns:
        - np.array: The modified array with swapped segments.
        """
        a = array  # no copying needed, we'll build a new one

        # Extract segments
        front_1 = a[..., 0:3]
        front_2 = a[..., 3:6]
        rear_1  = a[..., 6:9]
        rear_2  = a[..., 9:12]

        # Reassemble with swapped segments
        return jnp.concatenate([front_2, front_1, rear_2, rear_1], axis=-1)
    
    current_actions = jnp.ones(12)
    grav_tens = jnp.array([0., 0., -1.])
    qpos = mjx_data.qpos.copy()
    body_quat = qpos[3:7]
    qvel = mjx_data.qvel.copy()
    joint_angles = swap_legs(qpos[7:].copy())
    joint_velocities = swap_legs(qvel[6:])

    @jax.jit
    def compute_action_true(dummy):
        commands = jnp.array([0.3,0,0])
        body_vel = qvel[3:6].copy()
        gravity_body = quat_rotate_inverse(body_quat, grav_tens)
        scaled_body_vel = body_vel * scaling_factors['body_ang_vel']
        scaled_commands = commands[:2] * scaling_factors['commands']
        scaled_commands = jnp.append(scaled_commands, commands[2] * scaling_factors['body_ang_vel'])
        scaled_gravity_body = gravity_body * scaling_factors['gravity_body']
        scaled_joint_angles = joint_angles * scaling_factors['joint_angles']
        scaled_joint_velocities = joint_velocities * scaling_factors['joint_velocities']
        scaled_actions = current_actions * scaling_factors['actions']

        input_data = jnp.concatenate((scaled_body_vel, scaled_commands, scaled_gravity_body,
                                    scaled_joint_angles, scaled_joint_velocities, scaled_actions))

        obs_jax = norm_obs_jax(input_data)

        # print(f' Difference obs_torch-jax {np.linalg.norm(obs.numpy() - obs_jax)}')
        ctrl = jax.lax.stop_gradient(flax_actor.apply(flax_variables,obs_jax))
        qDes = 0.5 * ctrl + default_joint_angles
        torques = kp_custom * (qDes - joint_angles) - kd_custom * joint_velocities
        torques = swap_legs(torques)
        # print(f'Torques shape {torques.shape}')
        return torques

    @jax.jit
    def compute_action_false(dummy):
        return torques

    # def step_fn(data, _):
    #     data = mjx.step(mjx_model, data)
    #     return data, None
    # Set control inputs
    torques=jax.lax.cond(n_step > decimation , compute_action_true, compute_action_false,None)

    
    mjx_data = mjx_data.replace(ctrl=torques)

    # print(ctrl)

    # mjx_data, _ = jax.lax.scan(step_fn, mjx_data, None, length=decimation) 
    # for _ in range(decimation):
        # Step simulation (returns new data)
    mjx_data = mjx.step(mjx_model, mjx_data)
    
    return mjx_data, mjx_data.qpos.copy(), mjx_data.qvel, torques

@jax.jit
def step_with_torques_2(mjx_model, mjx_data):
    """Apply torques and step simulation"""

    @jax.jit
    def compute_action_true():

        @jax.jit
        def swap_legs(array):
            """
            Swap the front and rear legs of the array based on predefined indices.
            
            The swap logic is fixed:
            - Swap front legs (indices 3:6) with (0:3)
            - Swap rear legs (indices 9:12) with (6:9)
            
            Parameters:
            - array: np.array
                The input array to modify.
            
            Returns:
            - np.array: The modified array with swapped segments.
            """
            a = array  # no copying needed, we'll build a new one

            # Extract segments
            front_1 = a[..., 0:3]
            front_2 = a[..., 3:6]
            rear_1  = a[..., 6:9]
            rear_2  = a[..., 9:12]

            # Reassemble with swapped segments
            return jnp.concatenate([front_2, front_1, rear_2, rear_1], axis=-1)
        
        @jax.jit
        def clip_torques_in_groups(torques):
            """
            Clip the elements of the `torques` array in groups of 3 with different ranges for each element.
            - The first and second elements in the group are clipped to [-35.0, 35.0]
            - The third element in the group is clipped to [-45.0, 45.0]
            
            Parameters:
            - torques: np.array
                The array of torques to modify.
                
            Returns:
            - np.array: The modified array with clipped values.
            """
            group_1_mask = jnp.clip(torques[::3], -35.0, 35.0)
            # Second group: clip in range (-35, 35)
            group_2_mask = jnp.clip(torques[1::3], -35.0, 35.0)
            # Third group: clip in range (-45, 45)
            group_3_mask = jnp.clip(torques[2::3], -45.0, 45.0)

            # Reconstruct the clipped torques
            clipped_torques = jnp.empty_like(torques)
            clipped_torques = clipped_torques.at[::3].set(group_1_mask)   # Apply first group clipping
            clipped_torques = clipped_torques.at[1::3].set(group_2_mask)   # Apply second group clipping
            clipped_torques = clipped_torques.at[2::3].set(group_3_mask)   # Apply third group clipping

            return clipped_torques
            
        current_actions = jnp.zeros(12)
        grav_tens = jnp.array([0., 0., -1.])
        qpos = mjx_data.qpos.copy()
        body_quat = qpos[3:7]
        qvel = mjx_data.qvel.copy()
        joint_angles = swap_legs(qpos[7:].copy())
        joint_velocities = swap_legs(qvel[6:])

        
        commands = jnp.array([0.25,0,0])
        body_vel = mjx_data.qvel[3:6].copy()
        gravity_body = quat_rotate_inverse(body_quat, grav_tens)
        scaled_body_vel = body_vel * scaling_factors['body_ang_vel']
        scaled_commands = commands[:2] * scaling_factors['commands']
        scaled_commands = jnp.append(scaled_commands, commands[2] * scaling_factors['body_ang_vel'])
        scaled_gravity_body = gravity_body * scaling_factors['gravity_body']
        scaled_joint_angles = joint_angles * scaling_factors['joint_angles']
        scaled_joint_velocities = joint_velocities * scaling_factors['joint_velocities']
        scaled_actions = current_actions * scaling_factors['actions']

        input_data = jnp.concatenate((scaled_body_vel, scaled_commands, scaled_gravity_body,
                                    scaled_joint_angles, scaled_joint_velocities, scaled_actions))

        obs_jax = norm_obs_jax(input_data)

        # print(f' Difference obs_torch-jax {np.linalg.norm(obs.numpy() - obs_jax)}')
        ctrl = jax.lax.stop_gradient(flax_actor.apply(flax_variables,obs_jax))
        qDes = 0.5 * ctrl + default_joint_angles
        torques = kp_custom * (qDes - joint_angles) - kd_custom * joint_velocities
        torques = swap_legs(torques)
        # print(f'Torques shape {torques.shape}')
        return clip_torques_in_groups(torques)
    
    torques = compute_action_true()

    mjx_data = mjx_data.replace(ctrl=torques)

    # print(ctrl)

    # mjx_data, _ = jax.lax.scan(step_fn, mjx_data, None, length=decimation) 
    # for _ in range(decimation):
        # Step simulation (returns new data)
    mjx_data = mjx.step(mjx_model, mjx_data)
    
    return mjx_data

if __name__ == "__main__":
    RENDERING = False

    n_envs = 32
    mjx_model = mjx.put_model(model)
    mjx_data = mjx.put_data(model,data)
    mjx_data = mjx.forward(mjx_model, mjx_data)

    rng = jax.random.PRNGKey(0)
    rng, batch_rng = jax.random.split(rng,2)
    batch_rng = jax.random.split(batch_rng, n_envs)
    batch_state = jax.vmap(lambda rng: mjx_data.replace(qpos=mjx_data.qpos.at[:2].set(jax.random.uniform(rng, (2,),minval=-5,maxval=5))))(batch_rng)

    # torque = jnp.ones((32,12)) * 1

    controls = np.load('controls_x.npy')
    # jit_step = jax.jit(jax.vmap(step_with_torques, in_axes=(None, 0, 0, None)))
    jit_step = jax.jit(jax.vmap(step_with_torques_2, in_axes=(None, 0)))

    n_step = 5000
    states = []
    states.append(jnp.hstack((batch_state.qpos.copy(),batch_state.qvel.copy())))
    print(f'Loop step')
    # batch_state, qpos, _ = step_with_torques(mjx_model, batch_state, torque)
    # states.append(qpos)
    now = time.time()

    if not(RENDERING):
        for i in tqdm(range(0,n_step)):        
            # mjx_data, state = run_single_step(mjx_model,mjx_data,state)
            # mjx_data, qpos, _ = step_with_torques(mjx_model, mjx_data, torque)
                # torques = compute_action_true(mjx_data)

            # mjx_data, q, v = step_with_torques_2(mjx_model, mjx_data)
            batch_state= jit_step(mjx_model, batch_state)


            # Usage
            # mjx_data, qpos, _ = step_with_torques(mjx_model, mjx_data, torque)
            # states.append(state)
            states.append(jnp.hstack((batch_state.qpos.copy(),batch_state.qvel.copy())))
            # print(f'Progress: {i}/{n_step}')
            # if i % 20 == 0:
            #     print(f"Step {i}: pos {q}, vel {v}")
    else:      
        
        xml_path = 'aliengo/scene_rendering.xml' 
        model_rendering = mujoco.MjModel.from_xml_path(xml_path)
        data_robots = []
        q0 = np.array([0., 0., 0.38, 1., 0., 0., 0.] + list(default_joint_angles))

        for _ in range(n_envs):
            data_robots.append(mujoco.MjData(model_rendering))
            data_robots[-1].qpos[:]=q0
            mujoco.mj_forward(model, data_robots[-1])

        # Visual options for ghost
        vopt2 = mujoco.MjvOption()
        vopt2.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
        pert = mujoco.MjvPerturb()
        catmask = mujoco.mjtCatBit.mjCAT_DYNAMIC
        
        # Create passive viewer
        with mujoco.viewer.launch_passive(
            model_rendering, data_robots[0],
        ) as viewer:
            while viewer.is_running():
                for i in tqdm(range(0,n_step)):        
        
                    batch_state, q, v = jit_step(mjx_model, batch_state)
                    states.append(batch_state.qpos.copy())

                    if i % 10 == 0:                 
                        viewer.user_scn.ngeom = 0
                        for j in range(1,len(data_robots)):
                            # Add ghost robot
                            data_robots[j].qpos = states[-1][j]
                            mujoco.mj_forward(model_rendering, data_robots[j])
                            
                            # Add ghost to scene
                            mujoco.mjv_addGeoms(
                                model_rendering, data_robots[j], vopt2, pert, catmask, viewer.user_scn
                            )

                        # Update main robot
                        viewer.sync()

    end = time.time()
    states = np.array(states)
    states = jnp.swapaxes(states, 0, 1)
    np.save('traj_mjx_test.npy',states)
    print(f'Traj shape : {states.shape}')
    elapsed_time = end - now
    print(f"Elapsed time: {elapsed_time:.4f} seconds")
