import numpy as np

# simulation
timestep = 0.002  # 0.00125
framerate = 60
duration = 7
framerate = 60
decimation = 10

default_joint_angles = np.array(
    [0.0, 0.9, -1.7, 0.0, 0.9, -1.7, 0.0, 0.9, -1.7, 0.0, 0.9, -1.7]
)
default_joint_angles.flags.writeable = False

kp_custom = np.array([100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100])
kp_custom.flags.writeable = False

kd_custom = np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
kd_custom.flags.writeable = False

mu_vRange = np.array([0.0, 0.3])
mu_vRange.flags.writeable = False

FsRange = np.array([0.0, 2.5])
FsRange.flags.writeable = False

default_mid = np.array([0.0, 0.9, -1.7])
default_mid.flags.writeable = False

INCLINATION_THRESHOLD = 45.0  # degrees
FALL_HEIGHT_THRESHOLD = 0.2  # meters
CP_SAFE_RADIUS = 0.05  # meters
G = 9.81
robot_rad = 0.0
reached_target_rad = 0.3
vel_target_reached = 0.8


# policy scaling
body_ang_vel = 0.25
commands = 2.0
gravity_body = 1.0
joint_angles = 1.0
joint_velocities = 0.05
actions = 1.0

vx_bound = np.array([-0.5, 2.0])  # vx
vx_bound.flags.writeable = False

vy_bound = np.array([-0.4, 0.4])  # vy
vy_bound.flags.writeable = False

yaw_bound = np.array([-0.7, 0.7])
yaw_bound.flags.writeable = False

yaw_gain = 5
n_obs = 40
obstacles_list = tuple([f"obstacle_{i}" for i in range(n_obs)])
gain_attraction = 1
gain_repulsion = 5

percentage_moving_obstacle = 0.0
n_moving_obstacle = int(percentage_moving_obstacle * n_obs)
vel_min_obs = 0.2
vel_max_obs = 0.5

x_lim = 6
y_lim = 6

ep_duration = 20000

# Lidar
n_sector = 20
n_rays = 1000
view_radius = 7
body_attached = "trunk"

# Applied noise
noise_period = 100
noise_std = 0.2

# target
target = np.array([-5.5, 6.1])
target.flags.writeable = False

# paths
policy_path = "nn_params/nominal_policy.pth"
saving_path_obs = "observation_datasets/"
scene_path = "aliengo/random_scene.xml"

# device
device = "cpu"
