import numpy as np

vx_bound =  np.array([-0.5, 1.0])   # vx
vy_bound =  np.array([-0.3, 0.3])   # vy
yaw_bound = np.array([-0.7,0.7])
yaw_gain = 0.5
n_obs = 15
obstacles_list = [f'obstacle_{i}' for i in range(n_obs)]
target = np.array([10,0,0])
gain_attraction = 2
gain_repulsion = 10

percentage_moving_obstacle = 0.
n_moving_obstacle = int(percentage_moving_obstacle*n_obs)

x_lim = 6
y_lim = 6

ep_duration = 10000

#Lidar
n_sector = 12
n_rays = 360