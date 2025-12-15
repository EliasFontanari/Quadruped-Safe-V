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


from config_loader.policy_loader import load_config, load_actor_network
from utils import scale_axis, swap_legs, clip_torques_in_groups
from jax_functions import quat_rotate_inverse, quat_to_yaw, potential_field_planner
import random

# === Thresholds and constants ===
INCLINATION_THRESHOLD = 45.0  # degrees
FALL_HEIGHT_THRESHOLD = 0.2   # meters
CP_SAFE_RADIUS = 0.05         # meters
G = 9.81
robot_rad = 0.5
reached_target_rad = 0.05
vel_target_reached = 0.1

from params_quad_obs import vx_bound,vy_bound,yaw_bound,obstacles_list,target,gain_attraction,gain_repulsion,yaw_gain, n_moving_obstacle, n_obs
from utils_obs import obstacle_circe, lidar_scan, get_pairs_collision

config_path = "config.yaml"
config = load_config(config_path)
actor_network = load_actor_network(config)

timestep = 0.002 # 500Hz  # config['simulation']['timestep_simulation']
default_joint_angles = np.array(config['robot']['default_joint_angles'])
kp_custom = np.array(config['robot']['kp_custom'])
kd_custom = np.array(config['robot']['kd_custom'])
scaling_factors = config['scaling']

# Init model and data
model = mujoco.MjModel.from_xml_path("aliengo/random_scene.xml")
model.opt.timestep = timestep
data = mujoco.MjData(model)
data.qpos[:] = np.array([0., 0., 0.38, 1., 0., 0., 0.] + list(default_joint_angles))
mujoco.mj_forward(model, data)

mjx_model = mjx.put_model(model)
mjx_data = mjx.put_data(model,data)
obstacles_list_id = []
for obst in obstacles_list:
    obstacles_list_id.append(mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,obst))
obstacles_id = jnp.array(obstacles_list_id)


# ok base che funziona
def run_single_simulation(config, actor_network, decimation=16, max_steps=50000, noise_std=1.0, warmup_time=1.0, seed=None, seed_key = 0):
    
    def check_fallen(qpos, inclination_deg):
        return inclination_deg > 0 or qpos[2] < 0

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

    @partial(jax.jit, static_argnames=['view_angle','n_rays','body_name'])
    def lidar_scan(view_angle,n_rays,body_name):
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
        yaw = quat_to_yaw(data.body(body_name).xquat)
        angles_plus_yaw = yaw + jnp.linspace(-view_angle/2,view_angle/2,n_rays)
        distances = jnp.zeros(angles_plus_yaw.shape[0])

        @jax.jit
        def get_dist(angle):
            return mjx.ray(mjx_model,mjx_data,mjx_data.qpos[:3]+jnp.array([0,0,0.1]),jnp.array([jnp.cos(angle),jnp.sin(angle),0],dtype=np.float64),(),True,-1)[0]

        distances = jax.vmap(get_dist)(angles_plus_yaw)
          
        return distances

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

    def update_moving_obs():
        for i in range(moving_obstacles_id.shape[0]):
            new_key, sub_key = random.split(sub_key)
            mjx_model.body(moving_obstacles_id[i]).pos[:2] += moving_vel[:,i]*0.002 + jax.random.uniform(sub_key,shape=(2,),minval=-0.005,maxval=0.005)
            sub_key = new_key 
    
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
    
    observations = []

    obs_t_ = []
    obs_tp1_ = []
    for step in range(max_steps):
        robot_pos = mjx_data.qpos[:3]
        robot_or = mjx_data.qpos[3:7]
        yaw = quat_to_yaw(robot_or)
        field_vel = potential_field_planner(robot_pos[:2],target[:2],obstacles_pos)

        obstacles_scan = lidar_scan(jnp.pi,50,'trunk')

        update_moving_obs()

        commands = jnp.array([
            jnp.clip(field_vel[0],vx_bound[0],vx_bound[1]) ,   # vx
            jnp.clip(field_vel[1],vy_bound[0],vy_bound[1]) ,   # vy
            (jnp.arctan2((target[1] - robot_pos[1]),(target[0] - robot_pos[0])) - yaw)*yaw_gain     # yaw_rate
            ])
    
        qpos = mjx_data.qpos.copy()
        body_quat = qpos[3:7]

        joint_angles = swap_legs(qpos[7:].copy())
        joint_velocities = swap_legs(data.qvel[6:].copy())

        if step % decimation == 0:
            body_vel = data.qvel[3:6].copy()
            gravity_body = quat_rotate_inverse(body_quat, grav_tens)
            scaled_body_vel = body_vel * scaling_factors['body_ang_vel']
            scaled_commands = commands[:2] * scaling_factors['commands']
            scaled_commands = np.append(scaled_commands, commands[2] * scaling_factors['body_ang_vel'])
            scaled_gravity_body = gravity_body * scaling_factors['gravity_body']
            scaled_joint_angles = joint_angles * scaling_factors['joint_angles']
            scaled_joint_velocities = joint_velocities * scaling_factors['joint_velocities']
            scaled_actions = current_actions * scaling_factors['actions']

            input_data = jnp.concatenate((scaled_body_vel, scaled_commands, scaled_gravity_body,
                                        scaled_joint_angles, scaled_joint_velocities, scaled_actions))
            obs = torch.tensor(np.array(input_data), dtype=torch.float32)
            obs = actor_network.norm_obs(obs)

            with torch.no_grad():
                current_actions = jnp.array([actor_network(obs).cpu().numpy()])

        qDes = 0.5 * current_actions + default_joint_angles
        torques = kp_custom * (qDes - joint_angles) - kd_custom * joint_velocities
        torques = swap_legs(torques)

        # # Apply noise only after warmup
        # if step >= warmup_steps-1:
        #     noise = np.random.normal(0, noise_std, size=torques.shape)
        #     torques += noise

        # === Step sim ===
        torques_noisy = clip_torques_in_groups(torques)
        data.ctrl[:] = torques_noisy
        mjx.step(mjx_model, mjx_data)        

        # === Update fall and CP ===
        qpos_after = data.qpos.copy()
        body_quat_after = qpos_after[3:7]
        inclination_after = 2 * jnp.arcsin(jnp.sqrt(body_quat_after[1]**2 + body_quat_after[2]**2)) * (180 / jnp.pi)
        collisions = get_pairs_collision(data,model)

        if check_fallen(qpos_after, inclination_after) or collisions != None:
            fallen_flag = 1
            if collisions != None:
                print(collisions)
        else:
            fallen_flag = 0

        base_pos = qpos_after[:2].copy()
        base_vel = data.qvel[:2].copy()
        z_after = qpos_after[2]
        # cp = compute_capture_point(base_pos, base_vel, z_after)
        # cp_local = cp - base_pos

        if jnp.linalg.norm(base_pos - target[:2]) < reached_target_rad and fallen_flag == 0 and jnp.linalg.norm(base_vel) < vel_target_reached:
            reached_flag = 1
        else:
            reached_flag = 0

        if step % decimation == 0:
            gravity_proj_tp1 = quat_rotate_inverse(body_quat_after,grav_tens)

            body_ang_vel = data.qvel[3:6].copy()

            joint_pos_tp1 = swap_legs(qpos_after[7:].copy())
            joint_vel_tp1 = swap_legs(data.qvel[6:].copy())

            obs_tp1_ = np.concatenate((
                gravity_proj_tp1.astype(np.float32),
                body_ang_vel.astype(np.float32),
                joint_pos_tp1.astype(np.float32),
                joint_vel_tp1.astype(np.float32)
            ))

            done_flag = float(fallen_flag)
            cp_flag = float(reached_flag)

            full_obs = np.concatenate([obs_t_, obs_tp1_, [done_flag, cp_flag]])
            
            observations.append(full_obs)

            # If capture point is OK, then terminate the episode
            if cp_flag == 1:
                break

            obs_t_ = obs_tp1_
            
            obstacles_pos = jax.vmap(get_obstacles_pos)(obstacles_id)

            
            return np.array(observations), fallen_flag, reached_flag


def run_batch_simulations(n_episodes=100, save_path="results", config_path="config.yaml", noise_std=1.0):
    os.makedirs(save_path, exist_ok=True)

    config = load_config(config_path)
    actor_network = load_actor_network(config)

    all_obs = []
    stats = []
    max_len = 0

    print("Running batch simulations...")

    for i in tqdm(range(n_episodes)):
        actor_network = load_actor_network(config)
        obs, fallen, captured = run_single_simulation(config, actor_network, noise_std=noise_std, seed=int(time.time()))
        all_obs.append(obs)
        stats.append((fallen, captured, len(obs)))
        max_len = max(max_len, len(obs))

    # Pad episodes to max_len
    obs_dim = all_obs[0].shape[1]
    padded_obs = np.zeros((n_episodes, max_len, obs_dim), dtype=np.float32)

    for i, episode in enumerate(all_obs):
        padded_obs[i, :len(episode), :] = episode

    stats = np.array(stats, dtype=int)

    np.save(os.path.join(save_path, "observations_dataset.npy"), padded_obs)
    # np.save(os.path.join(save_path, "episode_stats.npy"), stats)

    print(f"Episodi completati: {n_episodes}")
    print(f"Caduti: {np.sum(stats[:, 0])}, CP raggiunto: {np.sum(stats[:, 1])}")
    print(f"Dati salvati in: {save_path}")
    print(f"Shape of observations: {padded_obs.shape}")

    return all_obs, stats # padded_obs, stats


if __name__ == "__main__":
    run_batch_simulations(n_episodes=1,  save_path="observation_datasets", noise_std=10.0)
