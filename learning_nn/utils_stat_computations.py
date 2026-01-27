import numpy as np
from tqdm import tqdm
import mujoco
from function_utils.utils import lidar_scan

def generate_query_grid(params,model,data):
    # Create x and y ranges
    step_size_x_y = 0.2
    x = np.arange(-params.x_lim, params.x_lim, step_size_x_y)
    y = np.arange(-params.y_lim, params.y_lim, step_size_x_y)

    # Create a meshgrid
    X, Y = np.meshgrid(x, y)

    grid_shape = X.shape

    # Flatten and stack to create (x, y) pairs
    xy_grid = np.stack([X.ravel(), Y.ravel()], axis=1)
    standard_conf_vector = np.hstack(
        [np.array([0.38, 1, 0, 0, 0]), params.default_joint_angles, np.zeros(model.nv)]
    )

    query_grid = np.zeros((xy_grid.shape[0],model.nq + model.nv + params.n_sector))


    for i in tqdm(range(xy_grid.shape[0])):
        query_grid[i] = np.hstack([xy_grid[i],standard_conf_vector,np.zeros(params.n_sector)])
        data.qpos = query_grid[i,:model.nq]
        data.qvel = query_grid[i, model.nq:(model.nq + model.nv)]
        mujoco.mj_forward(model, data)
        scansion = lidar_scan(
            model,
            data,
            params.view_angle,
            params.n_rays,
            params.view_radius,
            params.n_sector,
        )
        query_grid[i,-params.n_sector:] = scansion[0]
    return query_grid, grid_shape


import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
def plot_V_XY(params,query_grid, log_learning):
    fig, ax = plt.subplots()

    plt.grid(True)

    V_flipped = np.flipud(query_grid)
    sns.heatmap(V_flipped, annot=False, cmap=cm.coolwarm_r, ax=ax, vmin= query_grid.min(), vmax=query_grid.max(),
                cbar=True,
                )

    num_x_points = query_grid.shape[1]  # Number of columns
    num_y_points = query_grid.shape[0]  # Number of rows
    
    # Create tick positions at regular intervals
    # Positions are indices in the heatmap (0 to num_points)
    num_ticks = 5
    x_tick_positions = np.linspace(0, num_x_points - 1, num_ticks)
    y_tick_positions = np.linspace(0, num_y_points - 1, num_ticks)
    
    # Create tick labels corresponding to actual x, y values
    # Your grid goes from -lim to +lim
    x_tick_labels = np.linspace(-params.x_lim, params.x_lim, num_ticks)
    y_tick_labels = np.linspace(params.y_lim, -params.y_lim, num_ticks)  # Reversed because of flipud
    
    ax.set_xticks(x_tick_positions)
    ax.set_yticks(y_tick_positions)
    ax.set_xticklabels(np.round(x_tick_labels, 2))
    ax.set_yticklabels(np.round(y_tick_labels, 2))

    # ax.set_xticklabels(np.round(x[::x_interval], 2))
    # ax.set_yticklabels(np.round(y[::-y_interval]+0.02, 1))

    level = -3 if log_learning else 0.99
    x_coords = np.linspace(0, num_x_points - 1, num_x_points)
    y_coords = np.linspace(0, num_y_points - 1, num_y_points)
    X_contour, Y_contour = np.meshgrid(x_coords, y_coords)
    
    contours = ax.contour(X_contour, Y_contour, V_flipped, levels=[level], 
                         colors="black", linestyles='dashed')
    ax.clabel(contours, inline=True, fontsize=8, fmt="%.2f")
    plt.xlabel("x")
    plt.ylabel("y")

    ax.legend()
    plt.title(
        f"V"
    )
