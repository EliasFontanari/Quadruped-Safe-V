import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import yaml
import numpy as np

# Function to load YAML configuration
def load_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

# # Running Mean Std class (similar to your PyTorch version)
# class RunningMeanStd(nn.Module):
#     shape: tuple
    
#     def setup(self):
#         # Initialize mean and variance with zeros
#         self.mean = self.param('mean', nn.initializers.zeros, self.shape)
#         self.var = self.param('var', nn.initializers.ones, self.shape)
#         self.count = self.param('count', nn.initializers.zeros, ())
    
#     def __call__(self, x):
#         # Update running statistics
#         batch_size = x.shape[0]
#         delta = x - self.mean
#         self.mean = self.mean + delta / (self.count + 1)
#         self.var = self.var + delta * (x - self.mean)
#         self.count += batch_size
#         return (x - self.mean) / jnp.sqrt(self.var + 1e-6)

class RunningMeanStd:
    """Running mean and std normalization using pure JAX."""
    
    def __init__(self, shape):
        self.shape = shape
        self.mean = jnp.zeros(shape)
        self.var = jnp.ones(shape)
        self.count = 0.0
    
    def update(self, x):
        """Update running statistics with new batch of data."""
        batch_mean = jnp.mean(x, axis=0)
        batch_var = jnp.var(x, axis=0)
        batch_count = x.shape[0]
        
        # Welford's online algorithm for combining statistics
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = M2 / total_count
        
        # Update internal state
        self.mean = new_mean
        self.var = new_var
        self.count = total_count
    
    def normalize(self, x):
        """Normalize input using current statistics."""
        return (x - self.mean) / jnp.sqrt(self.var + 1e-8)
    
    def __call__(self, x, update_stats=True):
        """Normalize input, optionally updating statistics."""
        if update_stats:
            self.update(x)
        return self.normalize(x)
    
    def reset(self):
        """Reset statistics to initial values."""
        self.mean = jnp.zeros(self.shape)
        self.var = jnp.ones(self.shape)
        self.count = 0.0

# Actor Network Class
class ActorNetwork(nn.Module):
    action_dim: int
    input_dim: int

    @nn.compact
    def __call__(self, x):
        # x=self.norm_obs(x)
        x = nn.Dense(512)(x)
        x = nn.elu(x)
        x = nn.Dense(256)(x)
        x = nn.elu(x)
        x = nn.Dense(128)(x)
        x = nn.elu(x)
        x=nn.Dense(self.action_dim)(x)
        return x
    
class ActorModel():
    def __init__(self,input_dim,action_dim):
        self.net = ActorNetwork(action_dim,input_dim)
        key = jax.random.PRNGKey(42)
        self.params = self.net.init(key,jnp.ones((1,self.net.input_dim)))

        self.running_mean_std = RunningMeanStd(input_dim)
    
    def norm_input(self,x):
        return self.running_mean_std(x)
    
    def forward(self,x):
        x = self.norm_input(x)
        return self.net.apply(self.params,x)

# net = ActorNetwork(12,45)
# key = jax.random.PRNGKey(42)
# params = net.init(key,jnp.ones((1,45)))
model = ActorModel(45,12)

# Optimizer (Optax)
def create_optimizer(learning_rate=1e-3):
    return optax.adam(learning_rate)

# Initialize parameters
def initialize_model(key, input_dim, action_dim, mlp_units=[512, 256, 128]):
    model = create_model(input_dim, action_dim, mlp_units)
    params = model.init(key, jnp.ones((1, input_dim)))  # Pass a dummy input for initialization
    return model, params

# Example usage
key = jax.random.PRNGKey(0)
input_dim = 4  # Example input dimension
action_dim = 2  # Example action dimension

# Initialize model and optimizer
model, params = initialize_model(key, input_dim, action_dim)
optimizer = create_optimizer()

# Forward pass example
x = jnp.ones((1, input_dim))  # Dummy input
mu = model.apply({'params': params}, x)
print(mu)