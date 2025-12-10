import numpy as np
from params_quad_obs import obstacles_list
from utils import quat_to_yaw
import mujoco
from params_quad_obs import n_obs
import matplotlib.pyplot as plt

def obstacle_circe(obstacles_list,radius,mj_model):
    theta = np.linspace(0,2*np.pi,len(obstacles_list))
    for i in range(theta.shape[0]):
        mj_model.body(obstacles_list[i]).pos[:2] = radius*np.array([np.cos(theta[i]),np.sin(theta[i])])

def lidar_scan(model,data,view_angle,n_rays,body_name):
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
    yaw = quat_to_yaw(data.body(body_name).xquat)
    angles_plus_yaw = yaw + np.linspace(-view_angle/2,view_angle/2,n_rays)
    distances = np.zeros(angles_plus_yaw.shape[0])
    # excluded_bodies = get_children(model,'trunk')
    # excluded_bodies.append(mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'world'))
    # excluded_bodies.append(mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'trunk'))
    body_intersected_list = []
    for i in range(angles_plus_yaw.shape[0]):
        geom_id = np.array([1], dtype=np.int32)
        distance = mujoco.mj_ray(model,data,data.body(body_name).xpos+np.array([0,0,0.1]),np.array([np.cos(angles_plus_yaw[i]),np.sin(angles_plus_yaw[i]),0],dtype=np.float64),None,1,-1,geom_id)
        if distance != -1: 
            distances[i] = distance 
        else: distances[i] = None
        body_intersected = mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,model.geom_bodyid[geom_id[0]])
        body_intersected_list.append(body_intersected)
    
    # plt.figure()
    # plt.grid(True)
    # plt.xlim(-10, 10)
    # plt.ylim(-10, 10)
    # plt.plot(distances * np.cos(angles_plus_yaw),distances * np.sin(angles_plus_yaw),'o')
    # plt.show()
    return distances

# Python script to generate MuJoCo XML with 100 cylindrical obstacles

def generate_obstacles_xml(file_name="aliengo/random_scene.xml", num_obstacles=200,x_lim=10,y_lim=10,radius_lim=[0.05,0.2],height_lim=[0.2,1]):
    # Start the worldbody XML
    xml_content = '''<mujoco>
    <include file="aliengo.xml"/>
    <worldbody>
    '''
    # Add 100 obstacles (cylinders) with random positions
    for i in range(0, num_obstacles):
        x_pos = np.random.uniform(-x_lim,x_lim)
        y_pos = np.random.uniform(-y_lim,y_lim)
        while (-0.6<=x_pos<=0.6) and (-0.6<=y_pos<=0.6):
            x_pos = np.random.uniform(-x_lim,x_lim)
            y_pos = np.random.uniform(-y_lim,y_lim)
        heigth = np.random.uniform(height_lim[0],height_lim[1])
        z_pos = heigth
        
        # Define the obstacle's XML entry
        obstacle_xml = f'''
        <body name="obstacle_{i}" pos="{x_pos} {y_pos} {z_pos}">
            <geom type="cylinder" size="{np.random.uniform(radius_lim[0],radius_lim[1])} {heigth}" rgba="0 0 1 1"/>
        </body>
        '''
        
        xml_content += obstacle_xml

    # Close the worldbody and mujoco tags
    xml_content += '''
    </worldbody>
    </mujoco>
    '''
    
    # Write to file
    with open(file_name, "w") as f:
        f.write(xml_content)

def get_children(model, body_name):
    id = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'trunk')
    return [
        i for i in range(model.nbody)
        if model.body_parentid[i] == id
    ]

def get_pairs_collision(data,model):
    if len(data.contact) > 0:
        collisions = []
        for i in range(len(data.contact)):
            pair = (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,model.geom_bodyid[data.contact[i].geom[0]]), \
                    mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,model.geom_bodyid[data.contact[i].geom[1]]))
            if not(any("calf" in s for s in pair) and any("world" in s for s in pair)):
                collisions.append((mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,model.geom_bodyid[data.contact[i].geom[0]]), \
                        mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,model.geom_bodyid[data.contact[i].geom[1]])))
        if len(collisions) == 0:
            return None
        return list(set(collisions))
    else: return None
generate_obstacles_xml(num_obstacles=n_obs)


