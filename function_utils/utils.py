import numpy as np
import torch
import configuration.params as params


# Quaternion rotation helper
def quat_rotate_inverse(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = (
        q_vec
        * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1)
        * 2.0
    )
    return a - b + c


def quat_to_yaw(quat):
    w, x, y, z = quat
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return yaw


# Function to scale axis values - Joystick 1
def scale_axis(index, value, threshold=0.05):
    """
    Scale and threshold joystick axis values for Joystick 1.
    If the absolute value of the input is below the threshold, the output is set to 0.
    """
    if abs(value) < threshold:  # Apply threshold condition
        return 0.0

    if index == 0:  # Axis 1 (Left Stick Y) -- Flip the sign
        return -value * 0.5  # Flip the sign and map to [-0.5, 0.5]
    elif index == 1:  # Axis 0 (Left Stick X)
        value *= -1
        if value > 0:
            return value * 1.0  # Positive direction: [0, 1]
        else:
            return value * 0.6  # Negative direction: [0, -0.3]
    elif index == 2:  # Axis 3 (Right Trigger)
        return value * 0.78  # Symmetric between [-0.78, 0.78]
    else:
        return value  # Default case, no scaling for other axes


# Function to scale axis values - Joystick 2
def scale_axis2(index, value, threshold=0.05):
    """
    Scale and threshold joystick axis values for Joystick 2.
    If the absolute value of the input is below the threshold, the output is set to 0.
    """
    if abs(value) < threshold:  # Apply threshold condition
        return 0.0

    if index == 0:  # Axis 1 (Left Stick Y) -- Flip the sign
        return -value  # Flip the sign and map to [-0.5, 0.5]
    elif index == 1:  # Axis 0 (Left Stick X)
        value *= -1
        if value > 0:
            return value * 2.0  # Positive direction: [0, 1]
        else:
            return value * 1.0  # Negative direction: [0, -0.5]
    elif index == 2:  # Axis 3 (Right Trigger)
        return value  # Symmetric between [-0.5, 0.5]
    else:
        return value  # Default case, no scaling for other axes


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
    array_copy = array.copy()  # Make a copy to avoid modifying the original array

    # Swap front legs (3:6) with (0:3)
    array_copy[0:3] = array[3:6]
    array_copy[3:6] = array[0:3]

    # Swap rear legs (9:12) with (6:9)
    array_copy[6:9] = array[9:12]
    array_copy[9:12] = array[6:9]

    return array_copy


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
    torques_copy = torques.copy()  # Make a copy to avoid modifying the original array

    # Iterate over the array in groups of 3
    for i in range(0, len(torques), 3):  # Step by 3 to handle each group
        torques_copy[i] = np.clip(
            torques_copy[i], -35.0, 35.0
        )  # First element in group
        torques_copy[i + 1] = np.clip(
            torques_copy[i + 1], -35.0, 35.0
        )  # Second element in group
        torques_copy[i + 2] = np.clip(
            torques_copy[i + 2], -45.0, 45.0
        )  # Third element in group

    return torques_copy


import mujoco
import matplotlib.pyplot as plt


def obstacle_circe(obstacles_list, radius, mj_model):
    theta = np.linspace(0, 2 * np.pi, len(obstacles_list))
    for i in range(theta.shape[0]):
        mj_model.body(obstacles_list[i]).pos[:2] = radius * np.array(
            [np.cos(theta[i]), np.sin(theta[i])]
        )


def lidar_scan(model, data, view_angle, n_rays, ray_amp, n_sector):
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
    angles = np.linspace(-view_angle / 2, view_angle / 2, n_rays)
    distances = np.zeros(angles.shape[0])
    for i in range(angles.shape[0]):
        geom_id = np.array([1], dtype=np.int32)
        distances[i] = mujoco.mj_ray(
            model,
            data,
            data.qpos[:3] + np.array([0, 0, 0.10]),
            np.array([np.cos(angles[i]), np.sin(angles[i]), 0], dtype=np.float64),
            None,
            1,
            -1,
            geom_id,
        )
    distances = np.where(distances == -1, 1e6, distances)
    distances = np.where(distances <= ray_amp, distances, 1e6)

    angles_sector = np.linspace(angles[0], angles[-1], n_sector + 1)
    scan = np.zeros((2, n_sector))
    for i in range(n_sector):
        indexes = np.where(
            (angles >= angles_sector[i]) & (angles <= angles_sector[i + 1])
        )[0]
        scan[0, i] = np.min(distances[indexes])
        scan[1, i] = (angles_sector[i] + angles_sector[i + 1]) / 2

    return scan


# Python script to generate MuJoCo XML with 100 cylindrical obstacles
def generate_obstacles_xml(
    file_name="aliengo/random_scene.xml",
    num_obstacles=200,
    x_lim=10,
    y_lim=10,
    radius_lim=[0.05, 0.2],
    height_lim=[0.2, 1],
    seed=0,
):
    if seed == None:
        file_name = f"aliengo/random_scene.xml"
    else:
        file_name = f"aliengo/random_scene_{seed}.xml"
    # Start the worldbody XML
    xml_content = """<mujoco>
    <include file="aliengo.xml"/>
    <worldbody>
    """
    # Add 100 obstacles (cylinders) with random positions
    for i in range(0, num_obstacles):
        x_pos = np.random.uniform(-x_lim, x_lim)
        y_pos = np.random.uniform(-y_lim, y_lim)
        while (-0.6 <= x_pos <= 0.6) and (-0.6 <= y_pos <= 0.6):
            x_pos = np.random.uniform(-x_lim, x_lim)
            y_pos = np.random.uniform(-y_lim, y_lim)
        heigth = np.random.uniform(height_lim[0], height_lim[1])
        z_pos = heigth

        # Define the obstacle's XML entry
        obstacle_xml = f"""
        <body name="obstacle_{i}" pos="{x_pos} {y_pos} {z_pos}" mocap="true">
            <geom type="capsule" size="{np.random.uniform(radius_lim[0],radius_lim[1])} {heigth}" rgba="0 0 1 1"/>
        </body>
        """
        xml_content += obstacle_xml
    # Close the worldbody and mujoco tags
    xml_content += """
    </worldbody>
    </mujoco>
    """

    # Write to file
    with open(file_name, "w") as f:
        f.write(xml_content)


def get_children(model, body_name):
    id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    return [i for i in range(model.nbody) if model.body_parentid[i] == id]


def get_pairs_collision(data, model):
    if len(data.contact) > 0:
        collisions = []
        for i in range(len(data.contact)):
            pair = (
                mujoco.mj_id2name(
                    model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    model.geom_bodyid[data.contact[i].geom[0]],
                ),
                mujoco.mj_id2name(
                    model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    model.geom_bodyid[data.contact[i].geom[1]],
                ),
            )
            if not (any("calf" in s for s in pair) and any("world" in s for s in pair)):
                collisions.append(
                    (
                        mujoco.mj_id2name(
                            model,
                            mujoco.mjtObj.mjOBJ_BODY,
                            model.geom_bodyid[data.contact[i].geom[0]],
                        ),
                        mujoco.mj_id2name(
                            model,
                            mujoco.mjtObj.mjOBJ_BODY,
                            model.geom_bodyid[data.contact[i].geom[1]],
                        ),
                    )
                )
        if len(collisions) == 0:
            return None
        return list(set(collisions))
    else:
        return None


def quat_rotate_inverse_array(q, v):
    # Extract the shape and quaternion components
    q_w = q[0]  # Scalar part of the quaternion
    q_vec = q[1:]  # Vector part of the quaternion

    # Compute each term in the rotation
    a = v * (2.0 * q_w**2 - 1.0)  # Ensure proper shape for broadcasting
    b = np.cross(q_vec, v) * q_w * 2.0
    c = np.dot(q_vec, (np.dot(q_vec, v))) * 2

    # Return the result of quaternion rotation
    return a - b + c


def generate_pairs(data):
    """
    Execute on n_episodes x horizon x (states + reached + violated) data
    """
    pairs = []
    for i in range(0, data.shape[0]):
        # if i % 1000 == 0:
        #     print(f'Progress: {100*i/data.shape[0]} %')
        for j in range(0, data.shape[1] - 1, 1):
            if (data[i, j, :] != np.zeros(data.shape[2])).any() and (
                data[i, j + 1, :] != np.zeros(data.shape[2])
            ).any():
                pairs.append(np.hstack((data[i, j, :], data[i, j + 1, :])))
    return np.array(pairs)


def list_bodies(model):
    body_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        for i in range(model.nbody)
    ]
    return body_names


def list_geoms(model):
    geom_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        for i in range(model.ngeom)
    ]
    return geom_names


def potential_field_planner(position, orientation, target, scan, velocity):
    F_nav = params.gain_attraction * (target - position)
    for i in range(scan.shape[1]):
        dist = (np.array([np.cos(scan[1, i]), np.sin(scan[1, i])])) * scan[0, i]
        dist_norm = max(1e-3, np.linalg.norm(dist - params.robot_rad))
        F_nav += (params.gain_repulsion * velocity / (dist_norm**4)) * (-dist)

    F_nav = np.hstack([F_nav, 0])
    F_nav = quat_rotate_inverse_array(orientation, F_nav)[:2]
    return F_nav


def compute_capture_point(pos, vel, height):
    tc = np.sqrt(height / params.G)
    return pos + vel * tc


def check_fallen(qpos, inclination_deg):
    return (
        inclination_deg > params.INCLINATION_THRESHOLD
        or qpos[2] < params.FALL_HEIGHT_THRESHOLD
    )


def sample_target(x_lim, y_lim, enlargement):
    low_b = -np.array([x_lim, y_lim]) * (1 + enlargement)
    high_b = np.array([x_lim, y_lim]) * (1 + enlargement)
    target = np.random.uniform(low_b, high_b)
    while (low_b / (1 + enlargement) <= target).any() and (
        target <= high_b / (1 + enlargement)
    ).any():
        target = np.random.uniform(low_b, high_b)
    return target
