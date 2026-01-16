"""
Render multiple robots (ghost copies) using mujoco.viewer instead of Renderer
"""

import mujoco
import numpy as np
import mediapy as media
from pathlib import Path
import mujoco.viewer
from tqdm import tqdm

if __name__ == "__main__":
    data_q = np.load('examples_comparison/batched_sim.npy')
    print(data_q.shape)

    n_envs = data_q.shape[0]
    n_step = data_q.shape[1]

    xml_path = 'aliengo/scene_rendering.xml' 
    model_rendering = mujoco.MjModel.from_xml_path(xml_path)
    data_robots = []

    for l in range(n_envs):
        data_robots.append(mujoco.MjData(model_rendering))
        data_robots[-1].qpos[:] = data_q[l,0,:model_rendering.nq]
        data_robots[-1].qvel[:] = data_q[l,0,model_rendering.nq:]

        mujoco.mj_forward(model_rendering, data_robots[-1])

    # Visual options for ghost
    vopt2 = mujoco.MjvOption()
    vopt2.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    pert = mujoco.MjvPerturb()
    catmask = mujoco.mjtCatBit.mjCAT_DYNAMIC
    
    # Create passive viewer
    with mujoco.viewer.launch_passive(
        model_rendering, data_robots[0],
    ) as viewer:
        while viewer.is_running():
            for i in tqdm(range(0,n_step)):        
                 
                viewer.user_scn.ngeom = 0
                for j in range(len(data_robots)):
                    # Add ghost robot
                    data_robots[j].qpos = data_q[j,i,:model_rendering.nq]
                    data_robots[j].qvel = data_q[j,i,model_rendering.nq:]
                    mujoco.mj_forward(model_rendering, data_robots[j])
                    
                    # Add ghost to scene
                    mujoco.mjv_addGeoms(
                        model_rendering, data_robots[j], vopt2, pert, catmask, viewer.user_scn
                    )

                # Update main robot
                viewer.sync()
