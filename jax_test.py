import numpy as np
import jax 
import jax.numpy as jnp

# Define two sets of points
points_a = jnp.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])  # shape (3, 2)
points_b = jnp.array([[4.0, 5.0], [5.0, 6.0], [6.0, 7.0]])  # shape (3, 2)

# Function to compute squared distance between two points
@jax.jit
def squared_distance(p1, p2):
    return jnp.sum((p1 - p2) ** 2)

# Vectorize the squared_distance function over both points_a and points_b
vectorized_squared_distance = jax.vmap(squared_distance, in_axes=(0, 0))

print('Here')
# Apply the vectorized function to the arrays
distances = vectorized_squared_distance(points_a, points_b)
print(distances)