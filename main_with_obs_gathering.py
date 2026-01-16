import mujoco
from mujoco import viewer
import numpy as np
import torch
import time
import os
from tqdm import tqdm

from config_loader.policy_loader import load_config, load_actor_network
from utils import scale_axis, swap_legs, clip_torques_in_groups, quat_rotate_inverse, quat_to_yaw
import random

import copy 

import sys
import gc

if len(sys.argv) > 1:
    seed = int(sys.argv[1])
else:
    seed = 0 

np.random.seed(seed)


# === Thresholds and constants ===
INCLINATION_THRESHOLD = 45.0  # degrees
FALL_HEIGHT_THRESHOLD = 0.2   # meters
CP_SAFE_RADIUS = 0.05         # meters
G = 9.81
robot_rad = 0.1
reached_target_rad = 0.3
vel_target_reached = 0.8
device = 'cpu'

from params_quad_obs import vx_bound,vy_bound,yaw_bound,obstacles_list,target,gain_attraction,gain_repulsion,yaw_gain, n_moving_obstacle, n_obs, x_lim,y_lim,n_sector,n_rays, ep_duration, noise_period, vel_max_obs, vel_min_obs, body_attached, view_radius
from utils_obs import obstacle_circe, lidar_scan, get_pairs_collision, generate_obstacles_xml, quat_rotate_inverse_array,generate_pairs

# def potential_field_planner(position,orientation,target,obstacles,velocity):
#     F_nav = gain_attraction*(target - position)
#     for obstacle in obstacles:
#         diff_robot_obs = position - obstacle
#         dist = max(0,np.linalg.norm(diff_robot_obs) - robot_rad) 
#         F_nav += ((gain_repulsion*velocity)/(dist**4))*(diff_robot_obs)
#         # print(f'diff {diff_robot_obs}, dist {dist} , F_nav {F_nav}')
#     F_nav = np.hstack([F_nav,0])
#     F_nav = quat_rotate_inverse_array(orientation,F_nav)[:2]
#     return F_nav

def potential_field_planner(position,orientation,target,scan,velocity):
    F_nav = gain_attraction*(target - position)    
    for i in range(scan.shape[1]):
        dist = (np.array([np.cos(scan[1,i]), np.sin(scan[1,i])])) * scan[0,i] 
        dist_norm = max(1e-3,np.linalg.norm(dist - robot_rad)) 
        F_nav += (gain_repulsion*velocity/(dist_norm**4))*(-dist)  
    
    F_nav = np.hstack([F_nav,0])
    F_nav = quat_rotate_inverse_array(orientation,F_nav)[:2]
    return F_nav

def compute_capture_point(pos, vel, height):
    tc = np.sqrt(height / G)
    return pos + vel * tc

def check_fallen(qpos, inclination_deg):
    return inclination_deg > INCLINATION_THRESHOLD or qpos[2] < FALL_HEIGHT_THRESHOLD

def sample_target(x_lim,y_lim,enlargement):
    low_b = -np.array([x_lim,y_lim])*(1+enlargement)
    high_b = np.array([x_lim,y_lim])*(1+enlargement)
    target = np.random.uniform(low_b,high_b)
    while (low_b/(1+enlargement) <= target).any() and (target <= high_b/(1+enlargement)).any():
        target = np.random.uniform(low_b,high_b)
    return target

def run_single_simulation(config, actor_network, decimation=10, max_steps=10000, noise_amp_unif=1.0, seed=None):
    # target = sample_target(x_lim,y_lim,0.2)
    # generate_obstacles_xml(num_obstacles=n_obs,x_lim=x_lim,y_lim=y_lim,seed=seed)
    # print(f'\n Seed {seed} \n')
    # # print(f'Target: {target}\n')


    timestep = 0.002 # 500Hz  # config['simulation']['timestep_simulation']
    default_joint_angles = np.array(config['robot']['default_joint_angles'])
    kp_custom = np.array(config['robot']['kp_custom'])
    kd_custom = np.array(config['robot']['kd_custom'])
    scaling_factors = config['scaling']

    # Init model and data
    model = mujoco.MjModel.from_xml_path(f"aliengo/random_scene.xml")
    model.opt.timestep = timestep

    data = mujoco.MjData(model)
    
    initially_in_collision = True
    while(initially_in_collision):
        x_y_init = np.random.uniform(-np.array([x_lim,y_lim]),np.array([x_lim,y_lim]),2)
        yaw_init = np.random.uniform(0, 2*np.pi)   # scalar angle

        # print(f'XY {x_y_init}')
        half_yaw = yaw_init / 2
        quat_init = np.array([
            np.cos(half_yaw),  # w
            0.0,               # x
            0.0,               # y
            np.sin(half_yaw)   # z
        ])
        data.qpos[:] = np.array([x_y_init[0], x_y_init[1], 0.38, quat_init[0], quat_init[1], quat_init[2], quat_init[3]] + list(default_joint_angles))
        mujoco.mj_forward(model, data)
        collisions = get_pairs_collision(data,model)
        if collisions == None:
            initially_in_collision = False

    obs_dim = data.qpos.shape[0] + data.qvel.shape[0] + n_sector
    traj = np.zeros((obs_dim + 2,max_steps))
    
    # Obstacles
    obstacles_pos = []
    # obstacle_circe(obstacles_list,3,model)
    for obst in obstacles_list:
        obstacles_pos.append(model.body(obst).pos[:2])
    
    # sample moving_obstacles
    moving_obstacles = np.random.choice(np.arange(len(obstacles_list)),n_moving_obstacle,False)
    
    # compute trajectories of moving obstacles
    moving_vel = np.random.uniform(vel_min_obs,vel_max_obs,(n_moving_obstacle,1))
    obst_pos = data.geom_xpos[-len(obstacles_list):]
    directions = np.array([target[0] - obst_pos[moving_obstacles,0],target[1] -obst_pos[moving_obstacles,1]])/np.linalg.norm( np.array([target[0] - obst_pos[moving_obstacles,0],target[1] -obst_pos[moving_obstacles,1]]))
    directions = directions.T
    traj_obs = np.zeros((n_moving_obstacle,2,max_steps))
    for i in range(n_moving_obstacle):
        # traj_angle =  np.arange(max_steps)*timestep*moving_vel[i]
        # rot_mat = np.array([[np.cos(directions[i]), -np.sin(directions[i])],
        #                     [np.sin(directions[i]), np.cos(directions[i])]])
        traj_obs[i,0] = obst_pos[moving_obstacles[i]][0] + directions[i][0] * np.arange(max_steps) * timestep * moving_vel[i] 
        traj_obs[i,1] = obst_pos[moving_obstacles[i]][1] + directions[i][1] * np.arange(max_steps) * timestep * moving_vel[i] 
        traj_obs[i,0,-int(max_steps/2):] = traj_obs[i,0,:int(max_steps/2)][::-1]
        traj_obs[i,1,-int(max_steps/2):] = traj_obs[i,1,:int(max_steps/2)][::-1]


    grav_tens = torch.tensor([[0., 0., -1.]], device=device, dtype=torch.double)
    
    commands = np.zeros(3)

    current_actions = np.zeros(12)
    fallen_flag = 0
    reached_flag = 0

    for step in range(max_steps):
        robot_pos = data.qpos[:3]
        robot_or = data.qpos[3:7]

        data.mocap_pos[moving_obstacles,:2] = traj_obs[:,:,step]


        body_quat = data.qpos[3:7]

        joint_angles = swap_legs(data.qpos[7:])
        joint_velocities = swap_legs(data.qvel[6:])

        if step % decimation == 0:
            obstacles_scan = lidar_scan(model,data,2*np.pi,n_rays,'trunk',view_radius,n_sector)

            yaw = quat_to_yaw(robot_or)
            field_vel = potential_field_planner(robot_pos[:2],robot_or,target[:2],obstacles_scan,np.linalg.norm(data.qvel[:2]))

            commands = np.array([
            np.clip(field_vel[0],vx_bound[0],vx_bound[1]) ,   # vx
            np.clip(field_vel[1],vy_bound[0],vy_bound[1]) ,   # vy
            (np.arctan2((target[1] - robot_pos[1]),(target[0] - robot_pos[0])) - yaw)*yaw_gain     # yaw_rate
            ])

            body_vel = data.qvel[3:6]
            body_quat_reordered = np.array([body_quat[1], body_quat[2], body_quat[3], body_quat[0]])
            tensor_quat = torch.tensor(body_quat_reordered, device=device, dtype=torch.double).unsqueeze(0)
            gravity_body = quat_rotate_inverse(tensor_quat, grav_tens)

            scaled_body_vel = body_vel * scaling_factors['body_ang_vel']
            scaled_commands = commands[:2] * scaling_factors['commands']
            scaled_commands = np.append(scaled_commands, commands[2] * scaling_factors['body_ang_vel'])
            scaled_gravity_body = gravity_body[0].cpu() * scaling_factors['gravity_body']
            scaled_joint_angles = joint_angles * scaling_factors['joint_angles']
            scaled_joint_velocities = joint_velocities * scaling_factors['joint_velocities']
            scaled_actions = current_actions * scaling_factors['actions']

            input_data = np.concatenate((scaled_body_vel, scaled_commands, scaled_gravity_body,
                                        scaled_joint_angles, scaled_joint_velocities, scaled_actions))
            # obs_jax = norm_obs_jax(input_data)

            # current_actions = jax.lax.stop_gradient(flax_actor.apply(flax_variables,obs_jax))
            obs = torch.tensor(input_data, dtype=torch.float32)
            obs = actor_network.norm_obs(obs)

            with torch.no_grad():
                current_actions = actor_network(obs).numpy()

        current_actions = np.array(current_actions)
        qDes = 0.5 * current_actions + default_joint_angles
        torques = kp_custom * (qDes - joint_angles) - kd_custom * joint_velocities
        torques = swap_legs(torques)

        if step % noise_period ==0:
            noise_x = np.random.uniform(-noise_amp_unif,noise_amp_unif)
            noise_y = np.random.uniform(-noise_amp_unif,noise_amp_unif)
            data.qvel[0] += noise_x  # Push in x
            data.qvel[1] += noise_y  # Push in y

        # === Step sim ===
        torques_noisy = clip_torques_in_groups(torques)
        data.ctrl[:] = torques_noisy
        mujoco.mj_step(model, data)

        # # print(f'Step {step},\n lidar scan: {obstacles_scan} \n commands {commands} \n control {torques} \n state {data.qpos[:3]} \n noise {(noise_x,noise_y)}')

        # === Update fall and CP ===
        body_quat_after = data.qpos[3:7]
        inclination_after = 2 * np.arcsin(np.sqrt(body_quat_after[1]**2 + body_quat_after[2]**2)) * (180 / np.pi)
        collisions = get_pairs_collision(data,model)

        if check_fallen(data.qpos, inclination_after) or collisions != None:
            fallen_flag = 1
            # if collisions != None:
                # print(collisions)


        if np.linalg.norm(data.qpos[:2] - target[:2]) < reached_target_rad and fallen_flag == 0 and np.linalg.norm(data.qvel[:2]) < vel_target_reached:
            reached_flag = 1
            
        traj[:obs_dim,step] = np.hstack((data.qpos,data.qvel,obstacles_scan[0,:]))
        if fallen_flag:
            traj[-2:,step] = np.array([0,1])
            # print('\nCollision\n')
            break
        elif reached_flag and not(fallen_flag):
            traj[-2:,step] = np.array([1,0])
            # print('\nReached\n')
            break
    
    # if not(fallen_flag) and not(reached_flag):
        # print('\nNot reached not collided\n')
        
    return traj

def run_batch_simulations(n_episodes=1000, save_path="results", config_path="config.yaml", noise_std=1.0):
    os.makedirs(save_path, exist_ok=True)
    config = load_config(config_path)

    full_state = 37 + n_sector + 2
    traj_dataset = np.zeros((n_episodes,ep_duration,full_state))
    traj_dataset += 0
    # print("Running batch simulations...")

    times = []
    actor_network = load_actor_network(config)

    for i in tqdm(range(n_episodes)):
        start = time.time()
       
        traj_dataset[i] = run_single_simulation(config,actor_network=actor_network,noise_amp_unif=noise_std, seed=seed,max_steps=ep_duration, decimation=10).T
            
        end = time.time()
        times.append(end-start)
        # if i % 20 == 0 and i > 0:
            # print(f'Medium time for an episode: {np.sum(np.array(times))/i}')
        if i % 5 == 0:
            # print(f'')
            pairs = generate_pairs(traj_dataset)
            np.save(os.path.join(save_path, f"pairs_dataset_{seed}.npy"), pairs)
            del pairs
            gc.collect() 


    # pairs = generate_pairs(traj_dataset)
    # np.save(os.path.join(save_path, f"pairs_dataset_{seed}.npy"), pairs)


if __name__ == "__main__":
    run_batch_simulations(n_episodes=100,  save_path="observation_datasets", noise_std=0.15)
