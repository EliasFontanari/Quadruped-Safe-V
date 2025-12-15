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

import jax

from torch_to_flax import flax_actor, flax_variables, norm_obs_jax
# === Thresholds and constants ===
INCLINATION_THRESHOLD = 45.0  # degrees
FALL_HEIGHT_THRESHOLD = 0.2   # meters
CP_SAFE_RADIUS = 0.05         # meters
G = 9.81
robot_rad = 0.5
reached_target_rad = 0.05
vel_target_reached = 0.1
device = 'cpu' # 'cuda_0'

from params_quad_obs import vx_bound,vy_bound,yaw_bound,obstacles_list,target,gain_attraction,gain_repulsion,yaw_gain, n_moving_obstacle, n_obs, x_lim,y_lim,n_sector,n_rays, ep_duration
from utils_obs import obstacle_circe, lidar_scan, get_pairs_collision, generate_obstacles_xml, quat_rotate_inverse_array,generate_pairs

def potential_field_planner(position,orientation,target,obstacles):
    F_nav = gain_attraction*(target - position)
    for obstacle in obstacles:
        diff_robot_obs = position - obstacle
        dist = max(0,np.linalg.norm(diff_robot_obs)) 
        F_nav += (gain_repulsion/(dist**4))*(diff_robot_obs)
    F_nav = np.hstack([F_nav,0])
    F_nav = quat_rotate_inverse_array(orientation,F_nav)[:2]
    return F_nav

# def potential_field_planner(position,yaw,orientation,target,scan):
#     F_nav = gain_attraction*(target - position)
#     for i in range(scan.shape[1]):
#         diff_robot_obs = position - (yaw + np.array([np.cos(scan[0,i]), yaw + np.sin(scan[0,i])])) * scan[1,i] - robot_rad
#         dist = max(0,np.linalg.norm(diff_robot_obs)) 
#         F_nav += (gain_repulsion/(dist**4))*(diff_robot_obs)
#     F_nav = np.hstack([F_nav,0])
#     F_nav = quat_rotate_inverse_array(orientation,F_nav)[:2]
#     return F_nav

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

def run_single_simulation(config, actor_network, decimation=16, max_steps=50000, noise_std=1.0, warmup_time=1.0, seed=None):
    target = sample_target(x_lim,y_lim,0.2)
    # target=np.array([10,0,0])
    generate_obstacles_xml(num_obstacles=n_obs,x_lim=x_lim,y_lim=y_lim)

    timestep = 0.002 # 500Hz  # config['simulation']['timestep_simulation']
    default_joint_angles = np.array(config['robot']['default_joint_angles'])
    kp_custom = np.array(config['robot']['kp_custom'])
    kd_custom = np.array(config['robot']['kd_custom'])
    scaling_factors = config['scaling']

    # Init model and data
    model = mujoco.MjModel.from_xml_path("aliengo/random_scene.xml")
    model.opt.timestep = timestep
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    data.qpos[:] = np.array([0., 0., 0.38, 1., 0., 0., 0.] + list(default_joint_angles))
    
    mujoco.mj_forward(model, data)

    state_dim = data.qpos.shape[0] + data.qvel.shape[0] + n_sector
    traj = np.zeros((state_dim + 2,max_steps))
    

    # Obstacles
    obstacles_pos = []
    # obstacle_circe(obstacles_list,3,model)
    for obst in obstacles_list:
        obstacles_pos.append(model.body(obst).pos[:2])

    # sample moving_obstacles
    moving_obstacles = random.sample(obstacles_list,n_moving_obstacle)
    moving_vel = []
    for obst in moving_obstacles:
        direction = target[:2] - model.body(obst).pos[:2]
        moving_vel.append(direction/np.linalg.norm(direction))
    dir_vel = np.ones((2,len(moving_obstacles)))
    dir_vel[1,:] = 0

    grav_tens = torch.tensor([[0., 0., -1.]], device=device, dtype=torch.double)
    if seed is not None:
        np.random.seed(seed)

    warmup_steps = int(warmup_time / timestep)
    # commands = np.random.uniform(low=-1.0, high=1.0, size=(3,))
    commands = np.array([
        np.random.uniform(-0.5, 1.0),   # vx
        np.random.uniform(-0.3, 0.3),   # vy
        np.random.uniform(-0.7, 0.7)    # yaw_rate
    ])

    commands = np.zeros(3)

    current_actions = np.zeros(12)
    fallen_flag = 0
    reached_flag = 0
    observations = []

    obs_t_ = []
    obs_tp1_ = []
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            for step in range(max_steps):
                step_start = time.time()                  

                robot_pos = data.qpos[:3]
                robot_or = data.qpos[3:7]
                

                # move obstacles (it modifies also obstacle_pos because it contains same arrays)
                for (ii,obstacle) in enumerate(moving_obstacles):
                    if np.linalg.norm(model.body(obstacle).pos[:2] - target) < 0.2 and dir_vel[1,ii]==0:
                        dir_vel[0,ii] = -1
                        dir_vel[1,ii ]= 1
                    if np.linalg.norm(model.body(obstacle).pos[:2]) < 0.2 and  dir_vel[1,ii]==1:
                        dir_vel[0,ii] = 1
                        dir_vel[1,ii ]= 0
                    model.body(obstacle).pos[:2] += dir_vel[0,ii]*moving_vel[ii]*0.002 + np.random.uniform(-0.005,0.005,2)

                
                # commands[2] = np.clip((np.arctan2(commands[1] , commands[0])-yaw)*yaw_gain,yaw_bound[0],yaw_bound[1])
                # commands[2] = 0*np.pi/2


                qpos = data.qpos.copy()
                body_quat = qpos[3:7]

                joint_angles = swap_legs(qpos[7:].copy())
                joint_velocities = swap_legs(data.qvel[6:].copy())

                if step % decimation == 0:
                    obstacles_scan = lidar_scan(model,data,2*np.pi,360,'trunk',7,n_sector)

                    yaw = quat_to_yaw(robot_or)
                    # field_vel = potential_field_planner(robot_pos[:2],yaw, robot_or,target[:2],obstacles_scan)
                    field_vel = potential_field_planner(robot_pos[:2],robot_or,target[:2],obstacles_pos)

                    commands = np.array([
                    np.clip(field_vel[0],vx_bound[0],vx_bound[1]) ,   # vx
                    np.clip(field_vel[1],vy_bound[0],vy_bound[1]) ,   # vy
                    (np.arctan2((target[1] - robot_pos[1]),(target[0] - robot_pos[0])) - yaw)*yaw_gain     # yaw_rate
                    ])

                    body_vel = data.qvel[3:6].copy()
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
                    obs = torch.tensor(input_data, dtype=torch.float32)
                    obs = actor_network.norm_obs(obs)

                    obs_jax = norm_obs_jax(input_data)

                    # print(f' Difference obs_torch-jax {np.linalg.norm(obs.numpy() - obs_jax)}')
                    current_actions = jax.lax.stop_gradient(flax_actor.apply(flax_variables,obs_jax))
                    # with torch.no_grad():
                    #     current_actions = actor_network(obs).cpu().numpy()

                current_actions = np.array(current_actions)
                qDes = 0.5 * current_actions + default_joint_angles
                torques = kp_custom * (qDes - joint_angles) - kd_custom * joint_velocities
                torques = swap_legs(torques)

                # # Apply noise only after warmup
                if step >= warmup_steps and step % 100==0:
                    data.qvel[0] += np.random.uniform(-0.2,0.2)  # Push in x
                    data.qvel[1] += np.random.uniform(-0.2,0.2)  # Push in y

                # === Step sim ===
                torques_noisy = clip_torques_in_groups(torques)
                data.ctrl[:] = torques_noisy
                mujoco.mj_step(model, data)
                viewer.sync()
                

                # === Update fall and CP ===
                qpos_after = data.qpos.copy()
                q_vel = data.qvel.copy()
                body_quat_after = qpos_after[3:7]
                inclination_after = 2 * np.arcsin(np.sqrt(body_quat_after[1]**2 + body_quat_after[2]**2)) * (180 / np.pi)
                collisions = get_pairs_collision(data,model)

                if check_fallen(qpos_after, inclination_after) or collisions != None:
                    fallen_flag = 1
                    if collisions != None:
                        print(collisions)

                base_pos = qpos_after[:2].copy()
                base_vel = data.qvel[:2].copy()
                z_after = qpos_after[2]
                # cp = compute_capture_point(base_pos, base_vel, z_after)
                # cp_local = cp - base_pos

                if np.linalg.norm(base_pos - target[:2]) < reached_target_rad and fallen_flag == 0 and np.linalg.norm(base_vel) < 0.3: # vel_target_reached:
                    reached_flag = 1
                    print('Target reached')

                # if step % decimation == 0:
                print(f'Position: {data.qpos[:2]} ,Velocity: {data.qvel[:2]} , Target: {target[:2]}')
                gravity_proj_tp1 = quat_rotate_inverse(
                    torch.tensor([body_quat_after[1], body_quat_after[2], body_quat_after[3], body_quat_after[0]], device=device, dtype=torch.double).unsqueeze(0),
                    grav_tens
                )[0].cpu().numpy()

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

                full_obs = np.concatenate([obs_t_, [0,0], obs_tp1_, [done_flag, cp_flag]])
                
                if step >= warmup_steps:
                    observations.append(full_obs)

                # If capture point is OK, then terminate the episode
                # if cp_flag == 1:
                #     break

                obs_t_ = obs_tp1_
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
                    
                traj[:state_dim,step] = np.hstack((qpos_after,q_vel,obstacles_scan[0,:]))
                if fallen_flag:
                    traj[-2:,step] = np.array([0,1])
                    break
                elif reached_flag and not(fallen_flag):
                    traj[-2:,step] = np.array([1,0])
                    break
            
            # viewer.close()  
            return traj

# def run_single_simulation_no_viewer(config, actor_network, decimation=16, max_steps=5000, noise_std=1.0, warmup_time=1.0, seed=None):
    
#     timestep = 0.002 # 500Hz  # config['simulation']['timestep_simulation']
#     default_joint_angles = np.array(config['robot']['default_joint_angles'])
#     kp_custom = np.array(config['robot']['kp_custom'])
#     kd_custom = np.array(config['robot']['kd_custom'])
#     scaling_factors = config['scaling']

#     # Init model and data
#     model = mujoco.MjModel.from_xml_path("aliengo/random_scene.xml")
#     model.opt.timestep = timestep
#     data = mujoco.MjData(model)
#     mujoco.mj_forward(model, data)
#     data.qpos[:] = np.array([0., 0., 0.38, 1., 0., 0., 0.] + list(default_joint_angles))
    
#     mujoco.mj_forward(model, data)

#     # Obstacles
#     obstacles_pos = []
#     # obstacle_circe(obstacles_list,3,model)
#     for obst in obstacles_list:
#         obstacles_pos.append(model.body(obst).pos[:2])

#     # sample moving_obstacles
#     moving_obstacles = random.sample(obstacles_list,n_moving_obstacle)
#     moving_vel = []
#     for obst in moving_obstacles:
#         direction = target[:2] - model.body(obst).pos[:2]
#         moving_vel.append(direction/np.linalg.norm(direction))

#     grav_tens = torch.tensor([[0., 0., -1.]], device=device, dtype=torch.double)
#     if seed is not None:
#         np.random.seed(seed)

#     warmup_steps = int(warmup_time / timestep)
#     # commands = np.random.uniform(low=-1.0, high=1.0, size=(3,))
#     commands = np.array([
#         np.random.uniform(-0.5, 1.0),   # vx
#         np.random.uniform(-0.3, 0.3),   # vy
#         np.random.uniform(-0.7, 0.7)    # yaw_rate
#     ])

#     commands = np.zeros(3)

#     current_actions = np.zeros(12)
#     fallen_flag = 0
#     reached_flag = 0
#     observations = []

#     obs_t_ = []
#     obs_tp1_ = []
#     for step in range(max_steps):

#         robot_pos = data.qpos[:3]
#         robot_or = data.qpos[3:7]
#         yaw = quat_to_yaw(robot_or)
#         field_vel = potential_field_planner(robot_pos[:2],target[:2],obstacles_pos)

#         obstacles_scan = lidar_scan(model,data,np.pi,50,'trunk')

#         # move obstacles
#         for (ii,obstacle) in enumerate(moving_obstacles):
#             model.body(obstacle).pos[:2] += moving_vel[ii]*0.008 + np.random.uniform(-0.005,0.005,2)

#         commands = np.array([
#             np.clip(field_vel[0],vx_bound[0],vx_bound[1]) ,   # vx
#             np.clip(field_vel[1],vy_bound[0],vy_bound[1]) ,   # vy
#             (np.arctan2((target[1] - robot_pos[1]),(target[0] - robot_pos[0])) - yaw)*yaw_gain     # yaw_rate
#             ])
#         commands[2] = np.clip((np.arctan2(commands[1] , commands[0])-yaw)*yaw_gain,yaw_bound[0],yaw_bound[1])
#         # commands[2] = 0*np.pi/2

#         qpos = data.qpos.copy()
#         body_quat = qpos[3:7]

#         joint_angles = swap_legs(qpos[7:].copy())
#         joint_velocities = swap_legs(data.qvel[6:].copy())

#         if step % decimation == 0:
#             body_vel = data.qvel[3:6].copy()
#             body_quat_reordered = np.array([body_quat[1], body_quat[2], body_quat[3], body_quat[0]])
#             tensor_quat = torch.tensor(body_quat_reordered, device=device, dtype=torch.double).unsqueeze(0)
#             gravity_body = quat_rotate_inverse(tensor_quat, grav_tens)

#             scaled_body_vel = body_vel * scaling_factors['body_ang_vel']
#             scaled_commands = commands[:2] * scaling_factors['commands']
#             scaled_commands = np.append(scaled_commands, commands[2] * scaling_factors['body_ang_vel'])
#             scaled_gravity_body = gravity_body[0].cpu() * scaling_factors['gravity_body']
#             scaled_joint_angles = joint_angles * scaling_factors['joint_angles']
#             scaled_joint_velocities = joint_velocities * scaling_factors['joint_velocities']
#             scaled_actions = current_actions * scaling_factors['actions']

#             input_data = np.concatenate((scaled_body_vel, scaled_commands, scaled_gravity_body,
#                                         scaled_joint_angles, scaled_joint_velocities, scaled_actions))
#             obs = torch.tensor(input_data, dtype=torch.float32)
#             obs = actor_network.norm_obs(obs)

#             current_actions = jax.lax.stop_gradient(flax_actor.apply(flax_variables,obs))
#             with torch.no_grad():
#                 current_actions = actor_network(obs).cpu().numpy()

#         qDes = 0.5 * current_actions + default_joint_angles
#         torques = kp_custom * (qDes - joint_angles) - kd_custom * joint_velocities
#         torques = swap_legs(torques)

#         # # Apply noise only after warmup
#         if step >= warmup_steps and step % 40==0:
#             data.qvel[0] += np.random.uniform(-0.2,0.2)*0  # Push in x
#             data.qvel[1] += np.random.uniform(-0.2,0.2)*0  # Push in y

#         # === Step sim ===
#         torques_noisy = clip_torques_in_groups(torques)
#         data.ctrl[:] = torques_noisy
#         mujoco.mj_step(model, data)        

#         # === Update fall and CP ===
#         qpos_after = data.qpos.copy()
#         body_quat_after = qpos_after[3:7]
#         inclination_after = 2 * np.arcsin(np.sqrt(body_quat_after[1]**2 + body_quat_after[2]**2)) * (180 / np.pi)
#         collisions = get_pairs_collision(data,model)

#         if check_fallen(qpos_after, inclination_after) or collisions != None:
#             fallen_flag = 1
#             if collisions != None:
#                 print(collisions)

#         base_pos = qpos_after[:2].copy()
#         base_vel = data.qvel[:2].copy()
#         z_after = qpos_after[2]
#         # cp = compute_capture_point(base_pos, base_vel, z_after)
#         # cp_local = cp - base_pos

#         if np.linalg.norm(base_pos - target[:2]) < reached_target_rad and fallen_flag == 0 and np.linalg.norm(base_vel) < vel_target_reached:
#             reached_flag = 1

#         if step % decimation == 0:
#             gravity_proj_tp1 = quat_rotate_inverse(
#                 torch.tensor([body_quat_after[1], body_quat_after[2], body_quat_after[3], body_quat_after[0]], device=device, dtype=torch.double).unsqueeze(0),
#                 grav_tens
#             )[0].cpu().numpy()

#             body_ang_vel = data.qvel[3:6].copy()

#             joint_pos_tp1 = swap_legs(qpos_after[7:].copy())
#             joint_vel_tp1 = swap_legs(data.qvel[6:].copy())

#             obs_tp1_ = np.concatenate((
#                 gravity_proj_tp1.astype(np.float32),
#                 body_ang_vel.astype(np.float32),
#                 joint_pos_tp1.astype(np.float32),
#                 joint_vel_tp1.astype(np.float32)
#             ))

#             done_flag = float(fallen_flag)
#             cp_flag = float(reached_flag)

#             full_obs = np.concatenate([obs_t_, [0,0], obs_tp1_, [done_flag, cp_flag]])
            
#             if step >= warmup_steps:
#                 observations.append(full_obs)

#             obs_t_ = obs_tp1_
            
#             for obst in moving_obstacles:
#                 obst = model.body(obst).pos[:2]
            
#             # if fallen_flag or reached_flag: break
    
#     return np.array(observations), fallen_flag, reached_flag


def run_batch_simulations(n_episodes=10000, save_path="results", config_path="config.yaml", noise_std=1.0):
    os.makedirs(save_path, exist_ok=True)
    config = load_config(config_path)
    actor_network = load_actor_network(config)

    full_state = 37 + n_sector + 2
    traj_dataset = np.zeros((n_episodes,ep_duration,full_state))
    print("Running batch simulations...")

    for i in tqdm(range(n_episodes)):
        actor_network = load_actor_network(config)
        traj_dataset[i] = run_single_simulation(config, actor_network, noise_std=noise_std, seed=int(time.time()),max_steps=ep_duration).T
        if i % 100 == 0:
            pairs = generate_pairs(traj_dataset)
            np.save(os.path.join(save_path, "pairs_dataset.npy"), pairs)

    pairs = generate_pairs(traj_dataset)
    np.save(os.path.join(save_path, "pairs_dataset.npy"), pairs)

    print(f'Pairs_shape {pairs.shape}')
    # np.save(os.path.join(save_path, "episode_stats.npy"), stats)


if __name__ == "__main__":
    run_batch_simulations(n_episodes=10,  save_path="observation_datasets", noise_std=10.0)
