"""
Render multiple robots (ghost copies) using mujoco.viewer instead of Renderer
"""

import mujoco
import numpy as np
import mediapy as media
from pathlib import Path
import mujoco.viewer

# ============================================================================
# METHOD 1: Using passive viewer with manual rendering
# ============================================================================

def multiple_robots_passive_viewer(n_robots=2):
    """
    Use passive viewer to render multiple robot copies
    """
    
    # Get humanoid model
    print('Loading quadruped model...')
    xml_path = 'aliengo/random_scene.xml'
    
    model = mujoco.MjModel.from_xml_path(xml_path)
    data_robots = []
    for _ in range(n_robots):
        data_robots.append(mujoco.MjData(model))
        
    
    # Episode parameters
    duration = 3
    framerate = 60
    for data_robot in data_robots:
        data_robot.qpos[0:2] = [-.5, -.5]
        data_robot.qvel[2] = 4
    ctrl_phase = 2 * np.pi * np.random.rand(model.nu)
    ctrl_freq = 1
    
    # Visual options for ghost
    vopt2 = mujoco.MjvOption()
    vopt2.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    pert = mujoco.MjvPerturb()
    catmask = mujoco.mjtCatBit.mjCAT_DYNAMIC
    
    frames = []
    
    # Create passive viewer
    with mujoco.viewer.launch_passive(
        model, data_robots[0],
    ) as viewer:
        i=0
        # Simulation loop
        while viewer.is_running():
            # Control signal
            data_robots[0].ctrl = np.sin(ctrl_phase + 2 * np.pi * data_robots[0].time * ctrl_freq)
            
            # Step simulation
            mujoco.mj_step(model, data_robots[0])
            
            # Render at specified framerate
            if len(frames) < data_robots[0].time * framerate:
                
                viewer.user_scn.ngeom = 0
                for i in range(1,len(data_robots)):
                    # Add ghost robot
                    data_robots[i].qpos = data_robots[0].qpos
                    data_robots[i].qpos[0] += i  # Offset x
                    data_robots[i].qpos[1] += i  # Offset y
                    mujoco.mj_forward(model, data_robots[i])
                    
                    # Add ghost to scene
                    mujoco.mjv_addGeoms(
                        model, data_robots[i], vopt2, pert, catmask, viewer.user_scn
                    )

                # Update main robot
                viewer.sync()
                
                # Capture frame from viewer
                # pixels = viewer.read_pixels()
                # frames.append(pixels)
    
    return np.array(frames), framerate


if __name__ == "__main__":
    print("="*70)
    print("Multiple Robots with mujoco.viewer")
    print("="*70)
    
    frames, framerate = multiple_robots_passive_viewer(n_robots=5)

# ============================================================================
# USAGE SUMMARY
# ============================================================================

