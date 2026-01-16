import flax.linen as nn
import pickle
import jax
import jax.numpy as jnp
import yaml
from pathlib import Path


class CriticNetwork(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(512)(x)
        x = nn.LayerNorm()(x)
        x = nn.elu(x)
        x = nn.Dense(256)(x)
        x = nn.LayerNorm()(x)
        x = nn.elu(x)
        x = nn.Dense(128)(x)
        x = nn.LayerNorm()(x)
        x = nn.elu(x)
        x = nn.Dense(1, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.ones)(x)
        return x.squeeze(-1)


class FlaxCritic:
    def __init__(self, config):

        checkpoint_path = config["paths"]["safety_vf_path"]
        self._load_model(checkpoint_path)

    def _load_model(self, model_path: str):
        with open(model_path, 'rb') as f:
            data = pickle.load(f)

        self.params = data['model_params']
        self.mean = jnp.array(data['mean'])
        self.std = jnp.array(data['std'])
        self.model = CriticNetwork()
        self._inference_fn = jax.jit(self._evaluate)

    def _evaluate(self, params, obs):
        normalized_obs = (obs - self.mean) / (self.std + 1e-8)
        return self.model.apply(params, normalized_obs)

    def evaluate(self, obs):
        obs_jnp = jnp.array(obs)
        return self._inference_fn(self.params, obs_jnp)

    def is_safe(self, obs, threshold=0.0):
        return self.evaluate(obs) >= threshold
