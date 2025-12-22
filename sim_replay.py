import mujoco
import mujoco.viewer
import numpy as np
import time

def visualize_simulation(model_path, qpos_trajectory, qvel_trajectory=None, 
                         decimation = 10, dt_single = 0.002, loop=False, sim_idx=0):
    """
    Visualize a recorded MuJoCo simulation
    
    Args:
        model_path: Path to MuJoCo XML model file
        qpos_trajectory: Array of shape [n_sims, n_steps, nq] or [n_steps, nq]
        qvel_trajectory: Array of shape [n_sims, n_steps, nv] or [n_steps, nv] (optional)
        fps: Frames per second for playback
        loop: Whether to loop the visualization
        sim_idx: Which simulation to visualize if batch data provided
    """
    # Load model
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    # Handle batch dimension
    if qpos_trajectory.ndim == 3:
        qpos_traj = qpos_trajectory[:,sim_idx]
    else:
        qpos_traj = qpos_trajectory
    
    n_steps = qpos_traj.shape[0]
    dt = dt_single * decimation
    
    print(f"Visualizing trajectory with {n_steps} steps at {1/dt} frequency")
    print("Press ESC to exit, SPACE to pause")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        paused = False
        
        while viewer.is_running():
            step_start = time.time()
            
            if not paused:
                # Set state from trajectory
                data.qpos[:] = qpos_traj[step,:]
                
                # Forward kinematics to update visualizations
                mujoco.mj_forward(model, data)
                
                # Update viewer
                viewer.sync()
                
                # Advance step
                step += 1
                if step >= n_steps:
                    if loop:
                        step = 0
                        print("Looping...")
                    else:
                        print("Simulation complete")
                        break
            
            # Maintain framerate
            elapsed = time.time() - step_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

data_q = np.load('traj_mjx_test.npy')
# data_q = data_q[9,:19,:]

visualize_simulation(
    model_path="aliengo/aliengo.xml",
    qpos_trajectory=data_q,  # shape: [n_steps, nq]
    qvel_trajectory=None,  # shape: [n_steps, nv]
    loop=True
)