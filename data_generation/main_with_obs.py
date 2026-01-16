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
    decimation=16,
    max_steps=10000,
    noise_std=1.0,
    warmup_time=1.0,
    seed=None,
    inital_q=None,
    initial_vel=None,
):
    # target = sample_target(x_lim,y_lim,0.2)
    # target = np.array([4,0])
    # generate_obstacles_xml(num_obstacles=n_obs, seed=None,x_lim=x_lim, y_lim=y_lim)

    # target=np.array([10,0,0])
    # generate_obstacles_xml(num_obstacles=n_obs,x_lim=x_lim,y_lim=y_lim)

    print(f"Target: {params.target}\n")

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

    # mujoco.mj_forward(model, data)

    state_dim = data.qpos.shape[0] + data.qvel.shape[0] + params.n_sector
    traj = np.zeros((state_dim + 2, max_steps))

    # # Obstacles
    # obstacles_pos = []
    # # obstacle_circe(obstacles_list,3,model)
    # for obst in obstacles_list:
    #     obstacles_pos.append(model.body(obst).pos[:2])

    # sample moving_obstacles
    moving_obstacles = np.random.choice(
        np.arange(len(params.obstacles_list)), params.n_moving_obstacle, False
    )
    moving_vel = np.random.uniform(1, 4, (params.n_moving_obstacle, 1))
    obst_pos = data.geom_xpos[-len(params.obstacles_list) :]

    # # compute sinusoidal trajectories of moving obstacles
    # amplitudes = np.random.uniform(2,3,(n_moving_obstacle,1))
    # obst_pos = data.geom_xpos[-len(obstacles_list):]
    # directions = np.arctan((target[1] -obst_pos[moving_obstacles,1])/(target[0] - obst_pos[moving_obstacles,0]))

    # traj_angles = np.arange(max_steps).reshape(1,max_steps)*timestep*moving_vel
    # # traj_obs = np.vstack((np.cos(traj_angles)*amplitudes,np.sin(traj_angles)*amplitudes))
    # traj_obs = np.zeros((n_moving_obstacle,2,max_steps))
    # for i in range(n_moving_obstacle):
    #     traj_angle =  np.arange(max_steps)*timestep*moving_vel[i]
    #     rot_mat = np.array([[np.cos(directions[i]), -np.sin(directions[i])],
    #                         [np.sin(directions[i]), np.cos(directions[i])]])
    #     traj_obs[i,:] = rot_mat.T @ -np.vstack((np.cos(traj_angle)*amplitudes[i] + obstacles_pos[i][0],traj_angle/moving_vel[i]+obstacles_pos[i][1]))

    # compute trajectories of moving obstacles (straight)

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
    angles = np.arctan(
        (params.target[1] - obst_pos[moving_obstacles, 1])
        / (params.target[0] - obst_pos[moving_obstacles, 0])
    )
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

        # indx_reached = np.where(np.linalg.norm(target-traj_obs[i,:]<1e-3))[0][0]

    # plt.figure()
    # for i in range(n_moving_obstacle):
    #     plt.plot(traj_obs[i,0,:],traj_obs[i,1,:])
    # plt.show()
    # plt.close()
    # dir_vel = np.ones((2,len(moving_obstacles)))
    # dir_vel[1,:] = 0

    grav_tens = torch.tensor(
        [[0.0, 0.0, -1.0]], device=params.device, dtype=torch.double
    )
    if seed is not None:
        np.random.seed(seed)

    warmup_steps = int(warmup_time / params.timestep)

    commands = np.zeros(3)

    current_actions = np.zeros(12)
    fallen_flag = 0
    reached_flag = 0
    observations = []

    obs_t_ = []
    obs_tp1_ = []

    controls = []
    traj_viz = []
    traj_viz.append(np.hstack((data.qpos, data.qvel)))
    controls = []

    start = time.time()

    with mujoco.viewer.launch_passive(
        model, data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        for step in range(max_steps):
            step_start = time.time()

            robot_pos = data.qpos[:3]
            robot_or = data.qpos[3:7]

            # move obstacles (it modifies also obstacle_pos because it contains same arrays)
            # for (ii,obstacle) in enumerate(moving_obstacles):
            #     if np.linalg.norm(model.body(obstacle).pos[:2] - target) < 0.2 and dir_vel[1,ii]==0:
            #         dir_vel[0,ii] = -1
            #         dir_vel[1,ii ]= 1
            #     if np.linalg.norm(model.body(obstacle).pos[:2]) < 0.2 and  dir_vel[1,ii]==1:
            #         dir_vel[0,ii] = 1
            #         dir_vel[1,ii ]= 0
            #     model.body(obstacle).pos[:2] += dir_vel[0,ii]*moving_vel[ii]*0.002 + np.random.uniform(-0.005,0.005,2)
            # data.geom_xpos[moving_obstacles,:2] = traj_obs[:,:,step]
            data.mocap_pos[moving_obstacles, :2] = traj_obs[:, :, step]

            qpos = data.qpos
            body_quat = qpos[3:7]

            joint_angles = swap_legs(qpos[7:])
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
                # field_vel = potential_field_planner(robot_pos[:2],robot_or,target[:2],obstacles_pos,np.linalg.norm(data.qvel[:2]))

                commands = np.array(
                    [
                        np.clip(
                            field_vel[0], params.vx_bound[0], params.vx_bound[1]
                        ),  # vx
                        np.clip(
                            field_vel[1], params.vy_bound[0], params.vy_bound[1]
                        ),  # vy
                        0,  # yaw_rate
                    ]
                )

                angle_error = np.arctan2(data.qvel[1], (data.qvel[0] + 1e-6)) - yaw
                if not (-np.pi <= angle_error <= np.pi):
                    angle_error *= -1
                commands[2] = np.clip(
                    angle_error, params.yaw_bound[0], params.yaw_bound[1]
                )  # (-3*np.pi/4-yaw)*yaw_gain #
                print(
                    f"Commanded x {commands[0]} , commanded y {commands[1]}, commanded yaw rate {commands[2]}"
                )

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
                # obs = torch.tensor(input_data, dtype=torch.float32)
                # obs = actor_network.norm_obs(obs)

                # obs_jax = norm_obs_jax(input_data)

                # # print(f' Difference obs_torch-jax {np.linalg.norm(obs.numpy() - obs_jax)}')
                # current_actions = jax.lax.stop_gradient(flax_actor.apply(flax_variables,obs_jax))
                # with torch.no_grad():
                #     current_actions = actor_network(obs).cpu().numpy()
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
                noise_x = np.random.uniform(-noise_std, noise_std)
                noise_y = np.random.uniform(-noise_std, noise_std)
                data.qvel[0] += noise_x  # Push in x
                data.qvel[1] += noise_y  # Push in y

            # === Step sim ===
            torques_noisy = clip_torques_in_groups(torques)
            data.ctrl[:] = torques_noisy
            mujoco.mj_step(model, data)
            # print(f'Step {step},\n lidar scan: {obstacles_scan} \n commands {commands} \n control {torques} \n state {data.qpos[:3]} \n noise {(noise_x,noise_y)}')

            viewer.sync()

            controls.append(torques_noisy)

            traj_viz.append(np.hstack((data.qpos, data.qvel)))

            # === Update fall and CP ===
            qpos_after = data.qpos
            q_vel = data.qvel
            body_quat_after = qpos_after[3:7]
            inclination_after = (
                2
                * np.arcsin(np.sqrt(body_quat_after[1] ** 2 + body_quat_after[2] ** 2))
                * (180 / np.pi)
            )
            collisions = get_pairs_collision(data, model)

            if check_fallen(qpos_after, inclination_after) or collisions != None:
                fallen_flag = 1
                if collisions != None:
                    print(collisions)

            base_pos = qpos_after[:2]
            base_vel = data.qvel[:2]
            z_after = qpos_after[2]
            # cp = compute_capture_point(base_pos, base_vel, z_after)
            # cp_local = cp - base_pos

            if (
                np.linalg.norm(base_pos - params.target[:2]) < params.reached_target_rad
                and fallen_flag == 0
                and np.linalg.norm(base_vel) < params.vel_target_reached
            ):  # vel_target_reached:
                reached_flag = 1
                print("Target reached")

            # if step % decimation == 0:
            # print(f'Position: {data.qpos[:2]} ,Velocity: {data.qvel[:2]} , Target: {target[:2]}')
            gravity_proj_tp1 = (
                quat_rotate_inverse(
                    torch.tensor(
                        [
                            body_quat_after[1],
                            body_quat_after[2],
                            body_quat_after[3],
                            body_quat_after[0],
                        ],
                        device=params.device,
                        dtype=torch.double,
                    ).unsqueeze(0),
                    grav_tens,
                )[0]
                .cpu()
                .numpy()
            )

            body_ang_vel = data.qvel[3:6]

            joint_pos_tp1 = swap_legs(qpos_after[7:])
            joint_vel_tp1 = swap_legs(data.qvel[6:])

            obs_tp1_ = np.concatenate(
                (
                    gravity_proj_tp1.astype(np.float32),
                    body_ang_vel.astype(np.float32),
                    joint_pos_tp1.astype(np.float32),
                    joint_vel_tp1.astype(np.float32),
                )
            )

            done_flag = float(fallen_flag)
            cp_flag = float(reached_flag)

            full_obs = np.concatenate([obs_t_, [0, 0], obs_tp1_, [done_flag, cp_flag]])

            if step >= warmup_steps:
                observations.append(full_obs)

            # If capture point is OK, then terminate the episode
            # if cp_flag == 1:
            #     break

            obs_t_ = obs_tp1_
            # time_until_next_step = model.opt.timestep - (time.time() - step_start)
            # if time_until_next_step > 0:
            #     time.sleep(time_until_next_step)

            traj[:state_dim, step] = np.hstack(
                (qpos_after, q_vel, obstacles_scan[0, :])
            )
            if fallen_flag:
                traj[-2:, step] = np.array([0, 1])
                print("Broken")
                break
            elif reached_flag and not (fallen_flag):
                traj[-2:, step] = np.array([1, 0])
                print("Reached")
                break
        np.save("control_still.npy", controls)
        np.save("traj_still.npy", traj_viz)

        end = time.time()
        elapsed = end - start

        print(
            f"Elapsed time: {elapsed} seconds, {step/elapsed} steps/second, trajectory length {traj.shape[1]} "
        )

        # plt.figure()
        # plt.plot(traj[0,:])
        # plt.show()
        # plt.close()

        return traj


def run_batch_simulations(
    n_episodes=1000
):
    actor_network = load_actor_network(params.policy_path, params.device)

    model = mujoco.MjModel.from_xml_path(params.scene_path)
    model.opt.timestep = params.timestep

    full_state = model.nq + model.nv + params.n_sector + 2
    traj_dataset = np.zeros((n_episodes, params.ep_duration, full_state))
    print("Running batch simulations...")

    for i in tqdm(range(n_episodes)):
        traj_dataset[i] = run_single_simulation(
            model,
            actor_network,
            noise_std=params.noise_std,
            seed=int(time.time()),
            max_steps=params.ep_duration,
            decimation=params.decimation,
        ).T
        if i % 100 == 0:
            pairs = generate_pairs(traj_dataset)
            np.save(os.path.join(params.saving_path_obs, "pairs_dataset.npy"), pairs)

    pairs = generate_pairs(traj_dataset)
    np.save(os.path.join(params.saving_path_obs, "pairs_dataset.npy"), pairs)

    print(f"Pairs_shape {pairs.shape}")
    # np.save(os.path.join(save_path, "episode_stats.npy"), stats)


if __name__ == "__main__":
    run_batch_simulations(
        n_episodes=5
    )
