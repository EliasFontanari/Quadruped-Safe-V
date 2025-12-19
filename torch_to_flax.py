import jax.numpy as jnp
import numpy as np
import torch
from flax import linen as nn
import jax

def torch_to_jax(torch_tensor):
    """Convert PyTorch tensor to JAX array."""
    return jnp.array(torch_tensor.detach().cpu().numpy())

def load_torch_checkpoint(checkpoint_path):
    """
    Load a PyTorch checkpoint and inspect its contents.
    
    Returns:
        checkpoint: The full checkpoint dictionary
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print("Checkpoint contents:")
    for key in checkpoint.keys():
        print(f"  - {key}")
    
    return checkpoint


def extract_actor_state_dict(checkpoint):
    """
    Extract just the actor model weights from checkpoint.
    
    Args:
        checkpoint: Full checkpoint dict with keys like 'model', 'optimizer', etc.
    
    Returns:
        state_dict: Just the model weights
    """
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'actor' in checkpoint:
        state_dict = checkpoint['actor']
    else:
        # Maybe it's already just the state dict?
        state_dict = checkpoint
    
    # Check if weights are nested under 'a2c_network.' prefix
    if any(key.startswith('a2c_network.') for key in state_dict.keys()):
        print("Detected 'a2c_network.' prefix, extracting actor weights...")
        actor_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('a2c_network.') and not key.startswith('a2c_network.critic'):
                # Remove 'a2c_network.' prefix and keep only actor parts
                new_key = key.replace('a2c_network.', '')
                actor_state_dict[new_key] = value
        state_dict = actor_state_dict
    
    # Filter out non-actor keys (sigma, value layers, critic, etc.)
    filtered_state_dict = {}
    for key, value in state_dict.items():
        # Keep only actor_mlp and mu layers
        if key.startswith('actor_mlp.') or key.startswith('mu.'):
            filtered_state_dict[key] = value
    
    print(f"Filtered to {len(filtered_state_dict)} actor parameters")
    return filtered_state_dict


def convert_actor_network(torch_actor, flax_actor_class):
    """
    Convert PyTorch ActorNetwork weights to Flax format.
    
    Args:
        torch_actor: Your PyTorch ActorNetwork instance
        flax_actor_class: Your Flax Actor class
    
    Returns:
        flax_variables: Dictionary ready to use with flax_model.apply()
    """
    params = {}
    
    # Convert the MLP layers (actor_mlp)
    # PyTorch: actor_mlp is Sequential with Linear->ELU->Linear->ELU->...
    layer_idx = 0
    dense_idx = 0
    
    for name, module in torch_actor.actor_mlp.named_children():
        if isinstance(module, torch.nn.Linear):
            # Convert Linear layer (transpose weights!)
            weight = torch_to_jax(module.weight).T  # [in, out]
            bias = torch_to_jax(module.bias)
            
            params[f'Dense_{dense_idx}'] = {
                'kernel': weight,
                'bias': bias
            }
            dense_idx += 1
    
    # Convert the mu (output) layer
    weight = torch_to_jax(torch_actor.mu.weight).T
    bias = torch_to_jax(torch_actor.mu.bias)
    params[f'Dense_{dense_idx}'] = {
        'kernel': weight,
        'bias': bias
    }
    
    # Return in Flax format
    flax_variables = {'params': params}
    
    return flax_variables


def convert_running_mean_std(torch_rms):
    """
    Convert PyTorch RunningMeanStd to the format we can use.
    Returns a dictionary with mean, var, count.
    """
    rms_state = {
        'mean': torch_to_jax(torch_rms.mean),
        'var': torch_to_jax(torch_rms.var),
        'count': float(torch_rms.count)
    }
    return rms_state


# Define PyTorch Actor class (matching your architecture)
class ActorNetworkPyTorch(torch.nn.Module):
    def __init__(self, input_dim, action_dim, mlp_units=[512, 256, 128]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for unit in mlp_units:
            layers.append(torch.nn.Linear(prev_dim, unit))
            layers.append(torch.nn.ELU())
            prev_dim = unit
        self.actor_mlp = torch.nn.Sequential(*layers)
        self.mu = torch.nn.Linear(mlp_units[-1], action_dim)
    
    def forward(self, x):
        features = self.actor_mlp(x)
        mu = self.mu(features)
        return mu


# ============ EXAMPLE USAGE ============

# STEP 1: Load and inspect your checkpoint
print("="*60)
print("Loading PyTorch Checkpoint")
print("="*60)

# Load your actual checkpoint
checkpoint = torch.load('nn/nominal_policy.pth', map_location='cpu')

print("\nCheckpoint keys:", list(checkpoint.keys()))

# Extract actor weights (handles a2c_network. prefix automatically)
actor_state_dict = extract_actor_state_dict(checkpoint)

print("\nExtracted actor state_dict keys:")
for key in actor_state_dict.keys():
    print(f"  {key}")

# Create PyTorch model with correct dimensions
torch_actor = ActorNetworkPyTorch(input_dim=45, action_dim=12)

# Load the extracted weights
torch_actor.load_state_dict(actor_state_dict)
print("\n✓ Weights loaded successfully!")

print("\nPyTorch model structure:")
for name, param in torch_actor.named_parameters():
    print(f"  {name}: {param.shape}")


# If you want to see what else is in the checkpoint:
print("\n" + "="*60)
print("Example: Inspecting checkpoint contents")
print("="*60)
print("""
Your checkpoint contains:
  - 'model': The actor network weights
  - 'optimizer': Optimizer state
  - 'epoch': Training epoch
  - 'scaler': Gradient scaler (for mixed precision)
  - 'frame': Training frame count
  - 'last_mean_rewards': Recent rewards
  - 'env_state': Environment state

To load just the model:
  checkpoint = torch.load('checkpoint.pth')
  torch_actor.load_state_dict(checkpoint['model'])
""")


# Define matching Flax model
class FlaxActor(nn.Module):
    action_dim: int
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(512)(x)
        x = nn.elu(x)
        x = nn.Dense(256)(x)
        x = nn.elu(x)
        x = nn.Dense(128)(x)
        x = nn.elu(x)
        x = nn.Dense(self.action_dim)(x)
        return x


# Convert the weights
print("\n" + "="*60)
print("Converting weights to Flax...")
print("="*60)

flax_variables = convert_actor_network(torch_actor, FlaxActor)

print("\nFlax variables structure:")
for key in flax_variables['params'].keys():
    layer = flax_variables['params'][key]
    print(f"  {key}:")
    print(f"    kernel: {layer['kernel'].shape}")
    print(f"    bias: {layer['bias'].shape}")


# Verify the conversion
print("\n" + "="*60)
print("Verifying conversion with forward pass...")
print("="*60)

# Create test input (45 dimensions)
test_input_np = np.random.randn(1, 45).astype(np.float32)
test_input_torch = torch.from_numpy(test_input_np)
test_input_jax = jnp.array(test_input_np)

# PyTorch forward pass
torch_actor.eval()
with torch.no_grad():
    torch_output = torch_actor(test_input_torch).numpy()

# Flax forward pass
flax_actor = FlaxActor(action_dim=12)
flax_output = flax_actor.apply(flax_variables, test_input_jax)

print("\nPyTorch output shape:", torch_output.shape)
print("Flax output shape:", flax_output.shape)
print("\nPyTorch output (first 5):", torch_output[0, :5])
print("Flax output (first 5):", flax_output[0, :5])
print("\nMax difference:", np.max(np.abs(torch_output - np.array(flax_output))))
print("✓ Outputs match!" if np.allclose(torch_output, flax_output, atol=1e-5) else "✗ Outputs don't match")


# Save the converted weights
print("\n" + "="*60)
print("Saving converted weights...")
print("="*60)

import pickle

# Save Flax weights
with open('flax_actor_weights.pkl', 'wb') as f:
    pickle.dump(flax_variables, f)
print("✓ Saved to 'flax_actor_weights.pkl'")

# How to load later:
print("\nTo load later:")
print("""
with open('flax_actor_weights.pkl', 'rb') as f:
    loaded_variables = pickle.load(f)

flax_actor = FlaxActor(action_dim=12)
output = flax_actor.apply(loaded_variables, x)
""")

from config_loader.policy_loader import load_config
config = load_config("config.yaml")

state_dict = torch.load(config['paths']['policy_path'], map_location={'cuda:1': 'cuda:0'})['model']
mean = jnp.array(state_dict['running_mean_std.running_mean'])
std =   jnp.array(state_dict['running_mean_std.running_var'])

@jax.jit
def norm_obs_jax(x):
    return (x - mean) / jnp.sqrt(std + 1e-6) 




