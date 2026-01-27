import yaml
import torch
import torch.nn as nn
from rl_games.algos_torch.running_mean_std import RunningMeanStd
import configuration.params as params
 

# Actor Network Class
class ActorNetwork(nn.Module):
    def __init__(
        self, input_dim, action_dim, mlp_units=[512, 256, 128], activation=nn.ELU
    ):
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


# Function to load the actor network
def load_actor_network(path, device):
    input_dim = 45
    action_dim = 12
    actor_network = ActorNetwork(input_dim=input_dim, action_dim=action_dim)
    state_dict = torch.load(path, map_location=device)["model"]
    actor_state_dict = {
        k.replace("a2c_network.", ""): v
        for k, v in state_dict.items()
        if k.startswith("a2c_network.actor_mlp")
        or k.startswith("a2c_network.mu")
        or k.startswith("running_mean_std.running_mean")
        or k.startswith("running_mean_std.running_var")
        or k.startswith("running_mean_std.count")
    }
    actor_network.load_state_dict(actor_state_dict)
    return actor_network
