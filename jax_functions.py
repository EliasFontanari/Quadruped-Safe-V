import jax 
import jax.numpy as jnp
from params_quad_obs import vx_bound,vy_bound,yaw_bound,obstacles_list,target,gain_attraction,gain_repulsion,yaw_gain, n_moving_obstacle

@jax.jit
def check_fallen(qpos, inclination_deg):
    return inclination_deg > 0 or qpos[2] < 0

@jax.jit
def potential_field_planner(position,target,obstacles):
    F_nav = gain_attraction*(target - position)
    for obstacle in obstacles:
        diff_robot_obs = position - obstacle
        dist = jnp.maximum(0,jnp.linalg.norm(diff_robot_obs)) 
        F_nav += (gain_repulsion/(dist**4))*(diff_robot_obs)
    return F_nav

@jax.jit
def quat_rotate_inverse(q, v):
    # Extract the shape and quaternion components
    q_w = q[0]  # Scalar part of the quaternion
    q_vec = q[1:]  # Vector part of the quaternion

    # Compute each term in the rotation
    a = v * (2.0 * q_w ** 2 - 1.0)  # Ensure proper shape for broadcasting
    b = jnp.cross(q_vec, v) * q_w * 2.0
    c = jnp.dot(q_vec,(jnp.dot(q_vec,v)))*2

    # Return the result of quaternion rotation
    return a - b + c

@jax.jit
def quat_to_yaw(quat):
    w,x, y, z = quat
    yaw = jnp.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return yaw

print(quat_to_yaw([1,0,0,0]))
print(quat_rotate_inverse(jnp.array([1,0,0,0]),jnp.array([0,0,1])))
