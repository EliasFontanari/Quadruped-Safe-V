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
q0 = np.array([0., 0., 0.5, 0., 1., 0., 0.] + list(default_joint_angles))
data.qpos[:]=q0
mujoco.mj_forward(model, data)

mjx_model = mjx.put_model(model)
mjx_data = mjx.put_data(model,data)
# mjx_data.qpos.at[:].set(q0)
mjx_data = mjx.forward(mjx_model, mjx_data)

obstacles_list_id = []
for obst in obstacles_list:
    obstacles_list_id.append(mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,obst))
obstacles_id = jnp.array(obstacles_list_id)

# compute id collisions and relevant pairs

bodies = list_bodies(model)
collision_pairs = list(combinations(bodies,2))
index_to_remove = []
collision_pairs_corrected = []
for i in range(len(collision_pairs)):
    if not(any("calf" in s for s in collision_pairs[i]) and any("world" in s for s in collision_pairs[i])):
        if not(any("obstacle" in s for s in collision_pairs[i]) and any("world" in s for s in collision_pairs[i])):
            if not("obstacle" in collision_pairs[i][0] and "obstacle" in collision_pairs[i][1]):
                collision_pairs_corrected.append(collision_pairs[i])
collision_pairs = collision_pairs_corrected
collision_pairs= jnp.array(np.vstack(([[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,pair[0]),mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,pair[1])] for pair in collision_pairs])))

idx_coll = []
for j in range(mjx_data.contact.geom.shape[0]):
    for i in range(collision_pairs.shape[0]):
        if (collision_pairs[i] == mjx_data.contact.geom[j]).all():
            idx_coll.append(i)

idx_coll =jnp.array(idx_coll)

# renderer = mujoco.Renderer(model)

# scene_option = mujoco.MjvOption()
# scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

# camera = mujoco.MjvCamera()

# # Set the camera position (top-down view) and the look-at point
# camera.lookat = np.array([0.0, 0.0, 0.0])  # Point camera looks at (center of scene)
# camera.distance = 10.0  # Distance from lookat point
# camera.azimuth = 0.0   # Rotation around vertical axis (degrees)
# camera.elevation = -90.0  # Look straight down (negative = down, positive = up)

# ok base che funziona
# @partial(jax.jit, static_argnames=['max_steps','decimation','noise_std','warmup_time','seed_key'])
# @jax.jit
@partial(jax.jit, static_argnames=['max_steps'])
def run_single_simulation( decimation=16, max_steps=10000, noise_std=1.0, warmup_time=1.0, seed_key = 0):
    state_dim = data.qpos.shape[0] + data.qvel.shape[0] + n_sector
    traj = jnp.zeros((state_dim,max_steps))
    collisions = jnp.zeros((idx_coll.shape[0],max_steps))

    @jax.jit
    def potential_field_planner(position,target,obstacles):
        F_nav = gain_attraction*(target - position)
        for obstacle in obstacles:
            diff_robot_obs = position - obstacle
            dist = jnp.maximum(0,jnp.linalg.norm(diff_robot_obs)) 
            F_nav += (gain_repulsion/(dist**4))*(diff_robot_obs)
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

    @partial(jax.jit, static_argnames=['view_angle','n_rays','n_sector'])
    def lidar_scan(view_angle,n_rays,n_sector):
        """
        Lidar scansion

        Parameters:
        model: mujoco model instance
        data: mujoco data instance
        view_angle: angle of view, centered at the CoM of the robot, in its body frame
        n_rays: number of distance lectures

        Returns:
        distances to obstacles
        """
        # yaw = quat_to_yaw(data.body(body_name).xquat)
        angles_plus_yaw = jnp.linspace(-view_angle/2,view_angle/2,n_rays)
        distances = jnp.zeros(angles_plus_yaw.shape[0])

        # @jax.jit
        def get_dist(angle):
            return mjx.ray(mjx_model,mjx_data,  mjx_data.qpos[:3] + jnp.array([0,0,0.1]),jnp.array([jnp.cos(angle),jnp.sin(angle),0]))[0]

        distances = jax.vmap(get_dist)(angles_plus_yaw)
        distances = jnp.where(distances==-1,1e6,distances)
        
        sectors = jnp.linspace(-view_angle/2,view_angle/2,n_sector)

        # @jax.jit
        def process_sector(i):
            # Create boolean mask for current sector
            mask = (angles_plus_yaw >= sectors[i]) & (angles_plus_yaw < sectors[i+1])

            return jnp.min(jnp.where(mask,distances,1e6))
    
        # Use vmap to vectorize over sectors
        scan = jax.vmap(process_sector)(jnp.arange(n_sector))

        distances = jnp.where(distances == -1, jnp.nan, distances)

        # plt.figure()
        # plt.grid(True)
        # plt.xlim(-10, 10)
        # plt.ylim(-10, 10)
        # plt.plot(distances * np.cos(angles_plus_yaw),distances * np.sin(angles_plus_yaw),'o')
        # plt.show()
    
        return scan.T 

    @jax.jit
    def get_obstacles_pos(id):
        return mjx_model.body_pos[id][:2]
    # Obstacles
    obstacles_pos = jax.vmap(get_obstacles_pos)(obstacles_id)
    # obstacle_pos = obstacles_pos_fun(obstacles_id)

    key = jax.random.key(seed_key)  # Use any seed for reproducibility
    new_key, subkey = jax.random.split(key)

    # sample moving_obstacles
    moving_obstacles_id = jax.random.choice(subkey, obstacles_id, shape=(n_moving_obstacle,), replace=False)
    
    @jax.jit
    def get_obstacles_vel(id):
        return target[:2] - mjx_model.body_pos[id][:2]
    
    moving_vel = jax.vmap(get_obstacles_vel)(moving_obstacles_id)

    # # @jax.jit
    # def update_moving_obs_mjx(mjx_model, moving_obstacles_id, moving_vel, key):
    #     """
    #     MuJoCo MJX - IMMUTABLE, must create new model with updated positions.
    #     Everything in JAX must be immutable for JIT compilation.
    #     """
    #     n_obstacles = moving_obstacles_id.shape[0]
        
    #     # Generate all keys at once (vectorized)
    #     keys = random.split(key, n_obstacles + 1)
    #     noise_keys = keys[:-1]
    #     new_key = keys[-1]
        
    #     # Vectorized noise generation
    #     noise = jax.vmap(lambda k: random.uniform(k, shape=(2,), minval=-0.005, maxval=0.005))(noise_keys)
        
    #     # Calculate position deltas
    #     deltas = moving_vel.T * 0.002 + noise  # Shape: (n_obstacles, 2)
        
    #     # Get current body positions from model
    #     current_body_pos = mjx_model.body_pos  # Shape: (n_bodies, 3)
        
    #     # Update positions using JAX indexing
    #     updated_body_pos = current_body_pos.at[moving_obstacles_id, :2].add(deltas)
        
    #     # Create new model with updated positions (immutable update)
    #     updated_model = mjx_model.replace(body_pos=updated_body_pos)
        
    #     return updated_model, new_key

    # def update_moving_obs():
    #     for i in range(moving_obstacles_id.shape[0]):
    #         new_key, sub_key = random.split(sub_key)
    #         mjx_model.body(moving_obstacles_id[i]).pos[:2] += moving_vel[:,i]*0.002 + jax.random.uniform(sub_key,shape=(2,),minval=-0.005,maxval=0.005)
    #         sub_key = new_key 
    
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

    @jax.jit
    def recover_action(current_actions):
        return current_actions
        
    for step in range(max_steps):
        robot_pos = mjx_data.qpos[:3]
        robot_or = mjx_data.qpos[3:7]
        yaw = quat_to_yaw(robot_or)
        field_vel = potential_field_planner(robot_pos[:2],target[:2],obstacles_pos)

        obstacles_scan = lidar_scan(2*jnp.pi,360,n_sector)

        # update_moving_obs()

        commands = jnp.array([
            jnp.clip(field_vel[0],vx_bound[0],vx_bound[1]) ,   # vx
            jnp.clip(field_vel[1],vy_bound[0],vy_bound[1]) ,   # vy
            (jnp.arctan2((target[1] - robot_pos[1]),(target[0] - robot_pos[0])) - yaw)*yaw_gain     # yaw_rate
            ])
    
        qpos = mjx_data.qpos.copy()
        body_quat = qpos[3:7]
        
        qvel = mjx_data.qvel.copy()
        joint_angles = swap_legs(qpos[7:].copy())
        joint_velocities = swap_legs(qvel[6:])

        current_actions = jax.lax.cond(step % decimation == 0,compute_action,recover_action,current_actions)
            # body_vel = qvel[3:6].copy()
            # gravity_body = quat_rotate_inverse(body_quat, grav_tens)
            # scaled_body_vel = body_vel * scaling_factors['body_ang_vel']
            # scaled_commands = commands[:2] * scaling_factors['commands']
            # scaled_commands = jnp.append(scaled_commands, commands[2] * scaling_factors['body_ang_vel'])
            # scaled_gravity_body = gravity_body * scaling_factors['gravity_body']
            # scaled_joint_angles = joint_angles * scaling_factors['joint_angles']
            # scaled_joint_velocities = joint_velocities * scaling_factors['joint_velocities']
            # scaled_actions = current_actions * scaling_factors['actions']

            # input_data = jnp.concatenate((scaled_body_vel, scaled_commands, scaled_gravity_body,
            #                             scaled_joint_angles, scaled_joint_velocities, scaled_actions))
        
            # obs_jax = norm_obs_jax(input_data)

            # # print(f' Difference obs_torch-jax {np.linalg.norm(obs.numpy() - obs_jax)}')
            # current_actions = jax.lax.stop_gradient(flax_actor.apply(flax_variables,obs_jax))

        qDes = 0.5 * current_actions + default_joint_angles
        torques = kp_custom * (qDes - joint_angles) - kd_custom * joint_velocities
        torques = swap_legs(torques)

        # # Apply noise only after warmup
        # if step >= warmup_steps-1:
        #     noise = np.random.normal(0, noise_std, size=torques.shape)
        #     torques += noise

        # === Step sim ===
        torques_noisy = clip_torques_in_groups(torques)
        mjx_data.ctrl.at[:].set(torques_noisy)
        mjx.step(mjx_model, mjx_data)        

        # mj_data = mjx.get_data(model, mjx_data)
        # renderer.update_scene(mj_data,camera=camera, scene_option=scene_option)
        # pixels = renderer.render()

        # img = Image.fromarray(pixels)
        # img.save(f'pixels/frame_{step}.png') 

        qpos_after = mjx_data.qpos.copy()
        qvel_after = mjx_data.qvel.copy()
                        
        obstacles_pos = jax.vmap(get_obstacles_pos)(obstacles_id)
        traj.at[:,step].set(jnp.hstack((qpos_after,qvel_after,jnp.zeros(n_sector))))
        collisions.at[:,step].set(mjx_data.contact.dist[idx_coll])
            
    return traj,collisions

# @partial(jax.jit, static_argnames=['decimation'])
def run_single_step(q_init, decimation=16, noise_std=1.0, warmup_time=1.0, seed_key = 0):
    
    collisions = jnp.zeros((idx_coll.shape[0],decimation))

    mjx_data.qpos.at[:].set(q_init[:data.qpos.shape[0]])
    mjx_data.qvel.at[:].set(q_init[:data.qvel.shape[0]])
    mjx_data_sim = mjx.forward(mjx_model, mjx_data)
    
    @jax.jit
    def potential_field_planner(position,target,obstacles):
        F_nav = gain_attraction*(target - position)
        for obstacle in obstacles:
            diff_robot_obs = position - obstacle
            dist = jnp.maximum(0,jnp.linalg.norm(diff_robot_obs)) 
            F_nav += (gain_repulsion/(dist**4))*(diff_robot_obs)
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

    @partial(jax.jit, static_argnames=['view_angle','n_rays','n_sector'])
    def lidar_scan(view_angle,n_rays,n_sector):
        """
        Lidar scansion

        Parameters:
        model: mujoco model instance
        data: mujoco data instance
        view_angle: angle of view, centered at the CoM of the robot, in its body frame
        n_rays: number of distance lectures

        Returns:
        distances to obstacles
        """
        # yaw = quat_to_yaw(data.body(body_name).xquat)
        angles_plus_yaw = jnp.linspace(-view_angle/2,view_angle/2,n_rays)
        distances = jnp.zeros(angles_plus_yaw.shape[0])

        # @jax.jit
        def get_dist(angle):
            return mjx.ray(mjx_model,mjx_data_sim,  mjx_data_sim.qpos[:3] + jnp.array([0,0,0.1]),jnp.array([jnp.cos(angle),jnp.sin(angle),0]))[0]

        distances = jax.vmap(get_dist)(angles_plus_yaw)
        distances = jnp.where(distances==-1,1e6,distances)
        
        sectors = jnp.linspace(-view_angle/2,view_angle/2,n_sector)

        # @jax.jit
        def process_sector(i):
            # Create boolean mask for current sector
            mask = (angles_plus_yaw >= sectors[i]) & (angles_plus_yaw < sectors[i+1])

            return jnp.min(jnp.where(mask,distances,1e6))
    
        # Use vmap to vectorize over sectors
        scan = jax.vmap(process_sector)(jnp.arange(n_sector))

        distances = jnp.where(distances == -1, jnp.nan, distances)
    
        return scan.T 

    @jax.jit
    def get_obstacles_pos(id):
        return mjx_data_sim.xpos[id][:2]
    # Obstacles
    obstacles_pos = jax.vmap(get_obstacles_pos)(obstacles_id)
    # obstacle_pos = obstacles_pos_fun(obstacles_id)

    key = jax.random.key(seed_key)  # Use any seed for reproducibility
    new_key, subkey = jax.random.split(key)

    # sample moving_obstacles
    moving_obstacles_id = jax.random.choice(subkey, obstacles_id, shape=(n_moving_obstacle,), replace=False)
    
    @jax.jit
    def get_obstacles_vel(id):
        return target[:2] - mjx_data_sim.xpos[id][:2]
    
    moving_vel = jax.vmap(get_obstacles_vel)(moving_obstacles_id)
    
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

    @jax.jit
    def recover_action(current_actions):
        return current_actions
    
    @jax.jit
    def find_first_collision_step(collisions):
        """
        collisions: array of shape [n_sims, n_steps]
        returns: array of shape [n_sims] with first collision step index
                or n_steps if no collision occurred
        """
        # Create mask where collisions are negative
        is_collision = collisions < 0
        
        # Find first True along step axis
        # argmax returns first True index (or 0 if all False)
        first_collision_idx = jnp.argmax(is_collision, axis=1)
        
        # Handle case where no collision occurred (all False)
        # If no collision, argmax returns 0, so we need to check
        has_collision = jnp.any(is_collision, axis=1)
        
        # Set to n_steps if no collision occurred
        n_steps = collisions.shape[1]
        first_collision_idx = jnp.where(has_collision, first_collision_idx, n_steps)
        
        return first_collision_idx
        
    for step in range(decimation):
        robot_pos = mjx_data_sim.qpos[:3]
        robot_or = mjx_data_sim.qpos[3:7]
        yaw = quat_to_yaw(robot_or)
        field_vel = potential_field_planner(robot_pos[:2],target[:2],obstacles_pos)

        obstacles_scan = lidar_scan(2*jnp.pi,360,n_sector)

        # update_moving_obs()

        commands = jnp.array([
            jnp.clip(field_vel[0],vx_bound[0],vx_bound[1]) ,   # vx
            jnp.clip(field_vel[1],vy_bound[0],vy_bound[1]) ,   # vy
            (jnp.arctan2((target[1] - robot_pos[1]),(target[0] - robot_pos[0])) - yaw)*yaw_gain     # yaw_rate
            ])
    
        qpos = mjx_data_sim.qpos.copy()
        body_quat = qpos[3:7]
        
        qvel = mjx_data_sim.qvel.copy()
        joint_angles = swap_legs(qpos[7:].copy())
        joint_velocities = swap_legs(qvel[6:])

        current_actions = jax.lax.cond(step == 0,compute_action,recover_action,current_actions)

        qDes = 0.5 * current_actions + default_joint_angles
        torques = kp_custom * (qDes - joint_angles) - kd_custom * joint_velocities
        torques = swap_legs(torques)

        torques = jnp.ones(12) *1000

        # # Apply noise only after warmup
        # if step >= warmup_steps-1:
        #     noise = np.random.normal(0, noise_std, size=torques.shape)
        #     torques += noise

        # === Step sim ===
        # torques_noisy = clip_torques_in_groups(torques)
        ctrl = mjx_data_sim.ctrl.at[:].set(torques)

        mjx_data_sim = mjx_data_sim.replace(ctrl=ctrl)

        mjx.step(mjx_model, mjx_data_sim)        

        qpos_after = mjx_data_sim.qpos.copy()
        qvel_after = mjx_data_sim.qvel.copy()
                        
        obstacles_pos = jax.vmap(get_obstacles_pos)(obstacles_id)
        collisions.at[:,step].set(mjx_data_sim.contact.dist[idx_coll])

    return jnp.hstack((qpos_after,qvel_after, obstacles_scan)), find_first_collision_step(collisions)

def run_batch_simulations(n_episodes=100, save_path="results", config_path="config.yaml", noise_std=1.0):
    start = time.time()
    os.makedirs(save_path, exist_ok=True)

    all_obs = []
    stats = []
    max_len = 0

    print("Running batch simulations...")
    # qpos_batch = jnp.array(np.tile(q_init,(n_episodes,1)))

    # batch = jax.vmap(lambda qpos: mjx_data.replace(qpos=qpos))(qpos_batch)

    data_traj, coll_traj = jax.vmap(run_single_simulation,in_axes=(None, None, None, None, 0))(10, 100, 1.0, 1.0,jnp.arange(n_episodes))
    np.save('data_mjx',data_traj)
    np.save('coll_mjx',coll_traj)
    print(f'Traj shape {data_traj.shape}')

    end = time.time()
    elapsed_time = end - start
    print(f"Elapsed time: {elapsed_time:.4f} seconds")

# def run_batch_simulations(n_episodes=100, max_steps=1000, save_path="results", config_path="config.yaml", noise_std=1.0):
#     start = time.time()
#     os.makedirs(save_path, exist_ok=True)


#     # rng = jax.random.PRNGKey(0)
#     # rng = jax.random.split(rng, 4096)
#     # batch = jax.vmap(lambda rng: mjx_data.replace(qpos=jax.random.uniform(rng, (1,))))(rng) 
#     q_init = np.hstack((q0,np.zeros(data.qvel.shape[0])))
#     print("Running batch simulations...")
#     qpos_batch = np.tile(q_init,(n_episodes,1))

#     for i in range(qpos_batch.shape[0]):
#         qpos_batch[i,:3] += np.random.uniform(1,5,(3))

#     qpos_batch = jnp.array(qpos_batch)


#     traj_batch = np.zeros((n_episodes,data.qpos.shape[0] + data.qvel.shape[0] + n_sector, max_steps))
#     coll_batch = np.zeros((n_episodes,idx_coll.shape[0],max_steps))
#     coll_batch[:,:,0] = np.ones(idx_coll.shape[0])
#     traj_batch[:,:qpos_batch.shape[1],0] = qpos_batch
#     traj_batch[:,qpos_batch.shape[0]:,0] = 0
#     # batch = jax.vmap(lambda qpos: mjx_data.replace(qpos=qpos))(qpos_batch)
#     q_init = qpos_batch

#     print(f'q_init shape {q_init}')
#     for i in range(max_steps):
#         next_state, next_coll = jax.vmap(run_single_step,in_axes=(0, None, None, None, 0))(q_init, 10, 1.0, 1.0, jnp.arange(n_episodes))
#         coll_batch[:,:,i] = next_coll
#         traj_batch[:,:,i] = next_state
#         q_init = next_state

#         if i % 100 == 0:
#             np.save('data_mjx',traj_batch)
#             np.save('coll_mjx',coll_batch)
#     print(f'Traj shape {traj_batch.shape}')

#     end = time.time()
#     elapsed_time = end - start
#     print(f"Elapsed time: {elapsed_time:.4f} seconds")



if __name__ == "__main__":
    run_batch_simulations(n_episodes=100,   save_path="observation_datasets", noise_std=10.0)
