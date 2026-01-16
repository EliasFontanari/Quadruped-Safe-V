import numpy as np

vx_bound =  np.array([-0.5, 2.0])   # vx
vy_bound =  np.array([-0.3, 0.3])   # vy
yaw_bound = np.array([-0.7,0.7])
yaw_gain = 0.3
n_obs = 40
obstacles_list = [f'obstacle_{i}' for i in range(n_obs)]
gain_attraction = 1
gain_repulsion = 4

percentage_moving_obstacle = 0.
n_moving_obstacle = int(percentage_moving_obstacle*n_obs)
vel_min_obs = 0.2
vel_max_obs = 0.5

x_lim = 6
y_lim = 6

ep_duration = 20000

#Lidar
n_sector = 20
n_rays = 1000
view_radius = 7
body_attached = 'trunk'

# Applied noise
noise_period = 100

# target
target = np.array([-5.5,6.1])