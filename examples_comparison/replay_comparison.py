"""
Render multiple robots (ghost copies) using mujoco.viewer instead of Renderer
"""

import mujoco
import numpy as np
import mediapy as media
from pathlib import Path
import mujoco.viewer
from tqdm import tqdm
import time
if __name__ == "__main__":
    data_q = np.load('trajectory_comparison_both.npy')
    print(data_q.shape)

    n_envs = data_q.shape[0]
    n_step = data_q.shape[1]

    # data_original = np.load('../traj_still.npy')

    # print(data_original[:,:data_q.shape[2]].shape)

    # data_q[1] = data_original[:data_q.shape[1],:data_q.shape[2]]


    xml_path = '../aliengo/aliengo.xml' 
    model_rendering = mujoco.MjModel.from_xml_path(xml_path)
    data_robots = []

    for l in range(n_envs):
        data_robots.append(mujoco.MjData(model_rendering))
        data_robots[-1].qpos[:] = data_q[l,0,:model_rendering.nq]
        data_robots[-1].qvel[:] = data_q[l,0,model_rendering.nq:]

        mujoco.mj_forward(model_rendering, data_robots[-1])

    # Visual options for ghost
    vopt2 = mujoco.MjvOption()
    vopt2.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    pert = mujoco.MjvPerturb()
    catmask = mujoco.mjtCatBit.mjCAT_DYNAMIC

    print('-'*70 +'\n' + ' '*22 + 'JAX robot transparent' +' '*22 +'\n' + '-'*70 +'\n' )
    
    # Create passive viewer
    with mujoco.viewer.launch_passive(
        model_rendering, data_robots[0],
    ) as viewer:
        while viewer.is_running():
            for i in tqdm(range(0,n_step)):        
                step_start = time.time() 
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
                time_until_next_step = model_rendering.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
