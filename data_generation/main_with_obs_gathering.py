import mujoco
from mujoco import viewer
import numpy as np
import torch
import time
import os
from tqdm import tqdm
import configuration.params as params

from configuration.policy_loader import load_actor_network
from function_utils.utils import (
    scale_axis,
    swap_legs,
    clip_torques_in_groups,
    quat_rotate_inverse,
    quat_to_yaw,
    obstacle_circe,
    lidar_scan,
    get_pairs_collision,
    generate_obstacles_xml,
    quat_rotate_inverse_array,
    generate_pairs,
    potential_field_planner,
    check_fallen,
)

import sys

if len(sys.argv) > 1:
    seed = int(sys.argv[1])
else:
    seed = 0

np.random.seed(seed)


def run_single_simulation(
    model,
    actor_network,
    decimation=10,
    max_steps=10000,
    noise_amp_unif=1.0,
    seed=None,
    inital_q=None,
    initial_vel=None
):
    # Init data
    data = mujoco.MjData(model)

    if inital_q == None:
        initially_in_collision = True
        while initially_in_collision:
            x_y_init = np.random.uniform(
                -np.array([params.x_lim, params.y_lim]),
                np.array([params.x_lim, params.y_lim]),
                2,
            )
            yaw_init = np.random.uniform(0, 2 * np.pi)  # scalar angle
            half_yaw = yaw_init / 2
            quat_init = np.array(
                [np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)]  # w  # x  # y  # z
            )
            data.qpos[:] = np.array(
                [
                    x_y_init[0],
                    x_y_init[1],
                    0.38,
                    quat_init[0],
                    quat_init[1],
                    quat_init[2],
                    quat_init[3],
                ]
                + list(params.default_joint_angles)
            )
            mujoco.mj_forward(model, data)
            collisions = get_pairs_collision(data, model)
            if collisions == None:
                initially_in_collision = False
    else:
        data.qpos[:] = inital_q
        data.qvel[:] = initial_vel
        mujoco.mj_forward(model, data)

    obs_dim = data.qpos.shape[0] + data.qvel.shape[0] + params.n_sector
    traj = np.zeros((obs_dim + 2, int(max_steps / decimation)))

    # sample moving_obstacles
    moving_obstacles = np.random.choice(
        np.arange(len(params.obstacles_list)), params.n_moving_obstacle, False
    )

    # compute trajectories of moving obstacles
    moving_vel = np.random.uniform(
        params.vel_min_obs, params.vel_max_obs, (params.n_moving_obstacle, 1)
    )
    obst_pos = data.geom_xpos[-len(params.obstacles_list) :]
    directions = np.array(
        [
            params.target[0] - obst_pos[moving_obstacles, 0],
            params.target[1] - obst_pos[moving_obstacles, 1],
        ]
    ) / np.linalg.norm(
        np.array(
            [
                params.target[0] - obst_pos[moving_obstacles, 0],
                params.target[1] - obst_pos[moving_obstacles, 1],
            ]
        )
    )
    directions = directions.T
    traj_obs = np.zeros((params.n_moving_obstacle, 2, max_steps))
    for i in range(params.n_moving_obstacle):
        # traj_angle =  np.arange(max_steps)*timestep*moving_vel[i]
        # rot_mat = np.array([[np.cos(directions[i]), -np.sin(directions[i])],
        #                     [np.sin(directions[i]), np.cos(directions[i])]])
        traj_obs[i, 0] = (
            obst_pos[moving_obstacles[i]][0]
            + directions[i][0] * np.arange(max_steps) * params.timestep * moving_vel[i]
        )
        traj_obs[i, 1] = (
            obst_pos[moving_obstacles[i]][1]
            + directions[i][1] * np.arange(max_steps) * params.timestep * moving_vel[i]
        )
        traj_obs[i, 0, -int(max_steps / 2) :] = traj_obs[i, 0, : int(max_steps / 2)][
            ::-1
        ]
        traj_obs[i, 1, -int(max_steps / 2) :] = traj_obs[i, 1, : int(max_steps / 2)][
            ::-1
        ]

    grav_tens = torch.tensor(
        [[0.0, 0.0, -1.0]], device=params.device, dtype=torch.double
    )

    commands = np.zeros(3)

    current_actions = np.zeros(12)
    fallen_flag = 0
    reached_flag = 0

    for step in range(max_steps):
        robot_pos = data.qpos[:3]
        robot_or = data.qpos[3:7]

        data.mocap_pos[moving_obstacles, :2] = traj_obs[:, :, step]

        body_quat = data.qpos[3:7]

        joint_angles = swap_legs(data.qpos[7:])
        joint_velocities = swap_legs(data.qvel[6:])

        if step % decimation == 0:
            obstacles_scan = lidar_scan(
                model,
                data,
                2 * np.pi,
                params.n_rays,
                params.view_radius,
                params.n_sector,
            )

            yaw = quat_to_yaw(robot_or)
            field_vel = potential_field_planner(
                robot_pos[:2],
                robot_or,
                params.target[:2],
                obstacles_scan,
                np.linalg.norm(data.qvel[:2]),
            )

            commands = np.array(
                [
                    np.clip(field_vel[0], params.vx_bound[0], params.vx_bound[1]),  # vx
                    np.clip(field_vel[1], params.vy_bound[0], params.vy_bound[1]),  # vy
                    0,  # yaw_rate
                ]
            )

            angle_error = np.arctan2(data.qvel[1], (data.qvel[0] + 1e-6)) - yaw
            if not (-np.pi <= angle_error <= np.pi):
                angle_error *= -1
            commands[2] = np.clip(angle_error, params.yaw_bound[0], params.yaw_bound[1])

            body_vel = data.qvel[3:6]
            body_quat_reordered = np.array(
                [body_quat[1], body_quat[2], body_quat[3], body_quat[0]]
            )
            tensor_quat = torch.tensor(
                body_quat_reordered, device=params.device, dtype=torch.double
            ).unsqueeze(0)
            gravity_body = quat_rotate_inverse(tensor_quat, grav_tens)

            scaled_body_vel = body_vel * params.body_ang_vel
            scaled_commands = commands[:2] * params.commands
            scaled_commands = np.append(
                scaled_commands, commands[2] * params.body_ang_vel
            )
            scaled_gravity_body = gravity_body[0].cpu() * params.gravity_body
            scaled_joint_angles = joint_angles * params.joint_angles
            scaled_joint_velocities = joint_velocities * params.joint_velocities
            scaled_actions = current_actions * params.actions

            input_data = np.concatenate(
                (
                    scaled_body_vel,
                    scaled_commands,
                    scaled_gravity_body,
                    scaled_joint_angles,
                    scaled_joint_velocities,
                    scaled_actions,
                )
            )
            # obs_jax = norm_obs_jax(input_data)

            # current_actions = jax.lax.stop_gradient(flax_actor.apply(flax_variables,obs_jax))
            obs = torch.tensor(input_data, dtype=torch.float32)
            obs = actor_network.norm_obs(obs)

            with torch.no_grad():
                current_actions = actor_network(obs).numpy()

        current_actions = np.array(current_actions)
        qDes = 0.5 * current_actions + params.default_joint_angles
        torques = (
            params.kp_custom * (qDes - joint_angles)
            - params.kd_custom * joint_velocities
        )
        torques = swap_legs(torques)

        if step % params.noise_period == 0:
            noise_x = np.random.uniform(-noise_amp_unif, noise_amp_unif)
            noise_y = np.random.uniform(-noise_amp_unif, noise_amp_unif)
            data.qvel[0] += noise_x  # Push in x
            data.qvel[1] += noise_y  # Push in y

        # === Step sim ===
        torques_noisy = clip_torques_in_groups(torques)
        data.ctrl[:] = torques_noisy
        mujoco.mj_step(model, data)

        # # print(f'Step {step},\n lidar scan: {obstacles_scan} \n commands {commands} \n control {torques} \n state {data.qpos[:3]} \n noise {(noise_x,noise_y)}')

        # === Update fall and CP ===
        body_quat_after = data.qpos[3:7]
        inclination_after = (
            2
            * np.arcsin(np.sqrt(body_quat_after[1] ** 2 + body_quat_after[2] ** 2))
            * (180 / np.pi)
        )
        collisions = get_pairs_collision(data, model)

        if check_fallen(data.qpos, inclination_after) or collisions != None:
            fallen_flag = 1
            # if collisions != None:
            # print(collisions)

        if (
            np.linalg.norm(data.qpos[:2] - params.target[:2])
            < params.reached_target_rad
            and fallen_flag == 0
            and np.linalg.norm(data.qvel[:2]) < params.vel_target_reached
        ):
            reached_flag = 1

        if step % decimation == 0:
            traj[:obs_dim, int(step / decimation)] = np.hstack(
                (data.qpos, data.qvel, obstacles_scan[0, :])
            )
        if fallen_flag:
            traj[-2:, int(step / decimation) + 1 * np.sign(step // decimation)] = (
                np.array([0, 1])
            )
            # print('\nCollision\n')
            break
        elif reached_flag and not (fallen_flag):
            traj[-2:, int(step / decimation) + 1 * np.sign(step // decimation)] = (
                np.array([1, 0])
            )
            # print('\nReached\n')
            break

    # if not(fallen_flag) and not(reached_flag):
    # print('\nNot reached not collided\n')

    return traj, (not(fallen_flag) and reached_flag)


def run_batch_simulations(
    n_episodes=1000,
):
    model = mujoco.MjModel.from_xml_path(params.scene_path)
    model.opt.timestep = params.timestep

    full_state = model.nq + model.nv + params.n_sector + 2
    traj_dataset = np.zeros(
        (n_episodes, int(params.ep_duration / params.decimation), full_state)
    )
    # allocate necessary memory in RAM
    traj_dataset += 0
    # print("Running batch simulations...")

    times = []
    actor_network = load_actor_network(params.policy_path, params.device)
    for i in tqdm(range(n_episodes)):
        start = time.time()

        traj_dataset[i], _ = run_single_simulation(
            model,
            actor_network=actor_network,
            noise_amp_unif=params.noise_std,
            seed=seed,
            max_steps=params.ep_duration,
            decimation=params.decimation,
        ).T

        end = time.time()
        times.append(end - start)
        # if i % 20 == 0 and i > 0:
        # print(f'Medium time for an episode: {np.sum(np.array(times))/i}')
        # if i % 50 == 0:
        #     # print(f'')
        #     pairs = generate_pairs(traj_dataset)
        #     np.save(os.path.join(save_path, f"pairs_dataset_{seed}.npy"), pairs)
        #     del pairs
        #     gc.collect()
        np.save(os.path.join(params.saving_path_obs, f"traj_dataset_{seed}.npy"), traj_dataset)

    # pairs = generate_pairs(traj_dataset)
    # np.save(os.path.join(save_path, f"pairs_dataset_{seed}.npy"), pairs)


if __name__ == "__main__":
    run_batch_simulations(
        n_episodes=750, save_path="observation_datasets", noise_std=0.2
    )
