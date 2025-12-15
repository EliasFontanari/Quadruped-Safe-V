import yaml
import torch
import torch.nn as nn
from rl_games.algos_torch.running_mean_std import RunningMeanStd
# from stable_baselines3.common.running_mean_std import RunningMeanStd


# Function to load YAML configuration
def load_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

# Actor Network Class
class ActorNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, mlp_units=[512, 256, 128], activation=nn.ELU):
        super(ActorNetwork, self).__init__()
        layers = []
        prev_dim = input_dim
        for unit in mlp_units:
            layers.append(nn.Linear(prev_dim, unit))
            layers.append(activation())
            prev_dim = unit
        self.actor_mlp = nn.Sequential(*layers)
        self.mu = nn.Linear(mlp_units[-1], action_dim)
        self.running_mean_std = RunningMeanStd((input_dim,))

    def forward(self, x):
        features = self.actor_mlp(x)
        mu = self.mu(features)
        return mu

    def norm_obs(self, observation):
        with torch.no_grad():
            return self.running_mean_std(observation)
        
class CriticNetwork(nn.Module):
    def __init__(self, input_dim, mlp_units=[512, 256, 128], activation=nn.ELU):
        super(CriticNetwork, self).__init__()
        layers = []
        prev_dim = input_dim
        for unit in mlp_units:
            layers.append(nn.Linear(prev_dim, unit))
            layers.append(activation())
            prev_dim = unit
        self.critic_mlp = nn.Sequential(*layers)
        self.value = nn.Linear(mlp_units[-1], 1)  # Output scalare V(s)
        self.running_mean_std = RunningMeanStd((input_dim,))  # Normalizzazione input

    def forward(self, x):
        features = self.critic_mlp(x)
        value = self.value(features)
        return value

    def norm_obs(self, observation):
        with torch.no_grad():
            return self.running_mean_std(observation)


# Function to load the actor network
def load_actor_network(config):
    input_dim = 45
    action_dim = 12
    actor_network = ActorNetwork(input_dim=input_dim, action_dim=action_dim)
    state_dict = torch.load(config['paths']['policy_path'], map_location={'cuda:1': 'cuda:0'})['model']
    actor_state_dict = {k.replace('a2c_network.', ''): v for k, v in state_dict.items()
                        if k.startswith('a2c_network.actor_mlp') or k.startswith('a2c_network.mu')or k.startswith('running_mean_std.running_mean') or k.startswith('running_mean_std.running_var') or k.startswith('running_mean_std.count')}
    actor_network.load_state_dict(actor_state_dict)
    return actor_network

def load_critic_network(config):
    input_dim = 45
    critic_network = CriticNetwork(input_dim=input_dim)
    state_dict = torch.load(config['paths']['policy_path'], map_location={'cuda:1': 'cuda:0'})['model']
    critic_state_dict = {k.replace('a2c_network.', ''): v for k, v in state_dict.items()
                         if k.startswith('a2c_network.critic_mlp') or k.startswith('a2c_network.value')
                         or k.startswith('value_mean_std')}
    
    print(state_dict.keys())
    critic_network.load_state_dict(critic_state_dict, strict=False)
    return critic_network

