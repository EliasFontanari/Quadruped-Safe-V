import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
import copy
from functools import partial

state_size = 6
log_learning = False


class ValueNetwork(nn.Module):
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
        # Output initialized to 1 (probability of survival)
        x = nn.Dense(1, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros)(x)
        # x = nn.sigmoid(x) 
        # x = nn.Dense(1)(x)
        return x.squeeze(-1)
        
class ValueNetworkLog(nn.Module):
    @nn.compact
    def __call__(self, x):
        # x = nn.Dense(512)(x)
        # x = nn.LayerNorm()(x)
        # x = nn.relu(x)
        x = nn.Dense(512)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        # x = nn.Dense(1,kernel_init=nn.initializers.zeros,bias_init=nn.initializers.constant(0))(x)

        # x = nn.leaky_relu(x)
        # x = nn.Dense(1)(x)
        return x.squeeze(-1)
  
def train_step(self, model, state_params, target_params, pairs, input_size):
    return _train_step_trad(model, state_params, target_params, pairs, input_size)

    @jax.jit
    def soft_update_target_params(params,target_params,tau):
        """
        Soft update of the target network
        """
        return jax.tree_util.tree_map(lambda t, p: tau * p + (1 - tau) * t, target_params, params)

    @jax.jit
    def hard_update_target_params(params,target_params):
        return jax.tree_util.tree_map(lambda t, p: 0 * p + (1 - 0) * t, target_params, params)
    
class TrainingModelLOG(TrainingModel):
    def __init__(self, net_structure,input_size,learning_rate,tau_update_target,alpha):
        super().__init__( net_structure,input_size,learning_rate,tau_update_target)
        self.alpha = alpha
        self.log_1_m_alpha = jnp.log10(1-alpha)
        self.log_alpha = jnp.log10(alpha)
        self.target_fail = 0
        self.target_reached = -6

    @partial(jax.jit, static_argnums=(0,))
    def log_sum(self, l1, l2):
        # compute log(x1+x2) given: l1=log(x1) and l2=log(x2)
        return jax.lax.stop_gradient(jnp.maximum(l1, l2) + jnp.log10(1+jnp.pow(10,-jnp.abs(l1-l2))))
    
    @partial(jax.jit, static_argnums=(0,))
    def loss_fn(self, pairs):
        V_s = self.net_model.apply(self.params, pairs[:,:self.input_size])
        V_next = jax.lax.stop_gradient(self.net_model.apply(self.target_params, pairs[:,-(self.input_size+2):-2]))
        indicators = 1.0 - pairs[:,-1]
        target = jnp.minimum(-1e-5,indicators * ((1-pairs[:,-2])*V_next/self.target_reached + pairs[:,-2]) * self.target_reached + (1-indicators) * self.target_fail)
        l1 = V_s + self.log_1_m_alpha
        l2 = target + self.log_alpha
        target_alpha = self.log_sum(l1,l2) 

        return jnp.mean((jnp.square(V_s - target_alpha)))
    
class TrainState(train_state.TrainState):   # class equal to parent class
    pass
@partial(jax.jit, static_argnums=((0,)))
def _train_step_trad(model,state_params,target_params,pairs,input_size):
    def loss_fn(model,params,target_params,pairs,input_size):
        V_s = model.apply(params, pairs[:,:input_size])
        V_next = jax.lax.stop_gradient(model.apply(target_params, pairs[:,-(input_size+2):-2]))
        indicators = 1.0 - pairs[:,-1]
        target = indicators * ((1-pairs[:,-2])*V_next + pairs[:,-2])
        return jnp.mean(jnp.square(V_s - target))
    
    loss, grads = jax.value_and_grad(loss_fn)(model,state_params.params,target_params,pairs,input_size)
    state = state_params.apply_gradients(grads=grads)
    return state, loss