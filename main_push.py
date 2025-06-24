import mujoco
import numpy as np
import torch
import time
import os
from tqdm import tqdm

from config_loader.policy_loader import load_config, load_actor_network
from utils import scale_axis, swap_legs, clip_torques_in_groups, quat_rotate_inverse

# === Thresholds and constants ===
INCLINATION_THRESHOLD = 45.0  # degrees
FALL_HEIGHT_THRESHOLD = 0.2   # meters
CP_SAFE_RADIUS = 0.05         # meters
G = 9.81

def compute_capture_point(pos, vel, height):
    tc = np.sqrt(height / G)
    return pos + vel * tc

def check_fallen(qpos, inclination_deg):
    return inclination_deg > INCLINATION_THRESHOLD or qpos[2] < FALL_HEIGHT_THRESHOLD

def run_single_simulation(config, actor_network, decimation=16, max_steps=2000, noise_std=1.0, warmup_time=1.0, seed=None):
    timestep = 0.002 # 500Hz  # config['simulation']['timestep_simulation']
    default_joint_angles = np.array(config['robot']['default_joint_angles'])
    kp_custom = np.array(config['robot']['kp_custom'])
    kd_custom = np.array(config['robot']['kd_custom'])
    scaling_factors = config['scaling']

    # Init model and data
    model = mujoco.MjModel.from_xml_path("aliengo/aliengo.xml")
    model.opt.timestep = timestep
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    data.qpos[:] = np.array([0., 0., 0.38, 1., 0., 0., 0.] + list(default_joint_angles))
    mujoco.mj_forward(model, data)

    grav_tens = torch.tensor([[0., 0., -1.]], device='cuda:0', dtype=torch.double)
    if seed is not None:
        np.random.seed(seed)

    warmup_steps = int(warmup_time / timestep)
    # commands = np.random.uniform(low=-1.0, high=1.0, size=(3,))
    commands = np.array([
        np.random.uniform(-0.5, 1.0),   # vx
        np.random.uniform(-0.3, 0.3),   # vy
        np.random.uniform(-0.7, 0.7)    # yaw_rate
    ])

    current_actions = np.zeros(12)
    fallen_flag = 0
    capture_flag = 0
    observations = []

    obs_t_ = []
    obs_tp1_ = []

    for step in range(max_steps):

        if step == warmup_steps - 5:
            data.qvel[0] += np.random.uniform(-2.0, 3.0)  # Push in x
            data.qvel[1] += np.random.uniform(-1.5,1.5)  # Push in y

        if step == warmup_steps:
            commands = np.zeros(3)

        qpos = data.qpos.copy()
        body_quat = qpos[3:7]

        joint_angles = swap_legs(qpos[7:].copy())
        joint_velocities = swap_legs(data.qvel[6:].copy())

        if step % decimation == 0:
            body_vel = data.qvel[3:6].copy()
            body_quat_reordered = np.array([body_quat[1], body_quat[2], body_quat[3], body_quat[0]])
            tensor_quat = torch.tensor(body_quat_reordered, device='cuda:0', dtype=torch.double).unsqueeze(0)
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

            with torch.no_grad():
                current_actions = actor_network(obs).cpu().numpy()

        qDes = 0.5 * current_actions + default_joint_angles
        torques = kp_custom * (qDes - joint_angles) - kd_custom * joint_velocities
        torques = swap_legs(torques)

        # Apply noise only after warmup
        if step >= warmup_steps-1:
            noise = np.random.normal(0, noise_std, size=torques.shape)
            torques += noise

        # === Step sim ===
        torques_noisy = clip_torques_in_groups(torques)
        data.ctrl[:] = torques_noisy
        mujoco.mj_step(model, data)

        # === Update fall and CP ===
        qpos_after = data.qpos.copy()
        body_quat_after = qpos_after[3:7]
        inclination_after = 2 * np.arcsin(np.sqrt(body_quat_after[1]**2 + body_quat_after[2]**2)) * (180 / np.pi)

        if check_fallen(qpos_after, inclination_after):
            fallen_flag = 1

        base_pos = qpos_after[:2].copy()
        base_vel = data.qvel[:2].copy()
        z_after = qpos_after[2]
        cp = compute_capture_point(base_pos, base_vel, z_after)
        cp_local = cp - base_pos

        if np.linalg.norm(cp_local) < CP_SAFE_RADIUS and step >= warmup_steps and fallen_flag == 0: # and inclination_after < INCLINATION_THRESHOLD/2:
            capture_flag = 1

        if step % decimation == 0:
            gravity_proj_tp1 = quat_rotate_inverse(
                torch.tensor([body_quat_after[1], body_quat_after[2], body_quat_after[3], body_quat_after[0]], device='cuda:0', dtype=torch.double).unsqueeze(0),
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
            cp_flag = float(capture_flag)

            full_obs = np.concatenate([obs_t_, obs_tp1_, [done_flag, cp_flag]])
            
            if step >= warmup_steps:
                observations.append(full_obs)

            # If capture point is OK, then terminate the episode
            if cp_flag == 1:
                break

            obs_t_ = obs_tp1_

    return np.array(observations), fallen_flag, capture_flag


def run_batch_simulations(n_episodes=100, save_path="results", config_path="config.yaml", noise_std=1.0):
    os.makedirs(save_path, exist_ok=True)

    config = load_config(config_path)
    actor_network = load_actor_network(config)

    all_obs = []
    stats = []
    max_len = 0

    print("Running batch simulations...")

    for i in tqdm(range(n_episodes)):
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
    run_batch_simulations(n_episodes=100, save_path="observation_datasets", noise_std=10.0)
