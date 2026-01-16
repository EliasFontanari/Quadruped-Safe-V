from tqdm import tqdm
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader, Dataset
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import random
import numpy as np
import torch
import torch.nn as nn

torch.set_printoptions(profile="full")

class SupervisedDataset(Dataset):
    def __init__(self, x, device='cpu'):
        # convert into PyTorch tensors and remember them
        self.X = torch.tensor(x, dtype=torch.float32)
        self.device = device

    def __len__(self):
        # this should return the size of the dataset
        return self.X.shape[0]
 
    def __getitem__(self, idx):
        # this should return one sample from the dataset
        features = self.X[idx]
        return features

class NeuralNetwork(nn.Module):
    """A simple feedforward neural network."""

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        number_hidden,
        activation_hidden=nn.ELU(),
        activation_output=nn.Sigmoid(),
        ub=None,
        log_representation=False,
        mean=0,
        std=1,
    ):
        super().__init__()
        layers = []

        # Input layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.LayerNorm(hidden_size))
        layers.append(activation_hidden)

        # net_sizes = []
        # for k  in range(number_hidden):
        #     net_sizes.append(int(hidden_size/(2**k)))
        # net_sizes.append(int(net_sizes[-1]/2**k))
        # net_sizes.append(output_size)

        size_out = hidden_size
        # Hidden layers
        # for _ in range(number_hidden):
        #     size_in = int(size_out)
        #     size_out = int(size_out / 2)
        #     layers.append(nn.Linear(size_in, size_out))
        #     layers.append(activation)

        # # Output layer
        # layers.append(nn.Linear(size_out, output_size))
        # layers.append(nn.Sigmoid())

        # Hidden layers
        for _ in range(number_hidden):
            # size_in = int(size_out)
            # size_out = int(size_out / 2)
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(activation_hidden)

        # Output layer
        layers.append(nn.Linear(hidden_size, output_size))
        layers.append(activation_output)

        # Hidden layers
        # for kk in range(len(net_sizes)-1):
        #     layers.append(nn.Linear(net_sizes[kk], net_sizes[kk+1]))
        #     if (kk +1) != number_hidden:
        #         layers.append(activation)
        #     else:
        #         layers.append(nn.Sigmoid())

        self.linear_stack = nn.Sequential(*layers)

        self.ub = ub if ub is not None else 1

        # self.initialize_weights()

        self.input_size = input_size

        self.log_repr = log_representation
        self.mean = mean
        self.std = std

        if self.log_repr:
            self.ub = 1
        print(self)

    def forward(self, x):
        # out = self.linear_stack(x[:,:self.input_size])* self.ub
        out = self.linear_stack(x)

        # if self.log_repr:
        #     out = torch.clamp(out,max=0.001)
        return out * self.ub  # (out + 1) * self.ub / 2

    def initialize_weights(self):
        for layer in self.linear_stack:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

class RegressionNN:
    """Class that compute training and test of a neural network."""

    def __init__(self, batch_size, model, model_target, loss_fn, optimizer, settings):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.model_target = model_target
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.beta = 0.95
        self.batch_size = batch_size
        self.plot_train = True
        self.data_dir = ""
        self.settings = settings
        self.tau = 0.01

        self.error_log = []
        self.error = []

    def compute_target(self, x, successes, failures):
        targets = torch.zeros((x.shape[0], 1), dtype=torch.float32).to("cuda")

        # not ended states
        mask = ~(successes.bool() | failures.bool())

        if self.model.log_repr:
            targets[successes.bool()] = -10.0
            targets[failures.bool()] = 0.0
            # V_n = torch.clamp(self.model.forward(x),max=0.).detach()
            # indicators = 1.0 - failures
            # targets = indicators * ((1-successes)*V_n/(-10) + successes) * (-10)
            targets[mask] = self.model.forward(x[mask]).detach()


        else:
            targets[successes.bool()] = 1.0
            targets[failures.bool()] = 0.0
            targets[mask] = self.model.forward(x[mask]).detach()

        return targets

    def training(self, x_train: torch.Tensor, epochs):
        """Training of the neural network."""

        progress_bar = tqdm(total=epochs, desc="Training")

        # initially all targets 0
        val_targets = torch.zeros((x_train.shape[0], 1), device=self.device)

        loss_evol_train = []
        loss_evol_val = []
        loss_lp = 1

        n = len(x_train)
        for ep in range(epochs):

            self.model.train()
            # Shuffle the data
            idx = torch.randperm(n)
            x_perm = x_train[idx]
            # Split in batches
            x_batches = torch.split(x_perm, self.batch_size)

            for x in x_batches:

                # print(x[:10, -3])
                # print(x[:10, self.model.input_size + 2 : -2])
                # compute targets with network
                val_targets = self.compute_target(x[:, self.model.input_size + 2 : -2], x[:, -2], x[:, -1])
                # Forward pass
                V_pred = self.model(x[:, : self.model.input_size])

                # NAN CHECK
                if torch.isnan(V_pred).any():
                    print(f"NaN detected in outputs at epoch {ep}")
                    break
                # Compute the loss
                loss = self.loss_fn(V_pred, val_targets)

                if torch.isnan(loss).any():
                    print(f"NaN detected in loss at epoch {ep}")
                    break
                # Backward and optimize
                # print(f'loop : {ep}')
                self.optimizer.zero_grad()
                loss.backward()

                for name, param in self.model.named_parameters():
                    if torch.isnan(param.grad).any():
                        print(f"NaN detected in gradients of {name} at epoch {ep}")
                        break

                self.optimizer.step()

                self.soft_update()

                loss_lp = self.beta * loss_lp + (1 - self.beta) * loss.item()

            if ep % 1 == 0:
                print(f"Loss training: {np.mean(loss_lp)}")
                self.plot_learning(ep)

            # x.detach().cpu()

            loss_evol_train.append(loss_lp)
            # # Validation
            # loss_val = self.validation(x_val, y_val)
            # if ep % 100 == 0:
            #     print(f'Loss training: {loss_lp}')
            #     print(f'Loss validation: {loss_val}')
            # loss_evol_val.append(loss_val)
            progress_bar.update(1)

        progress_bar.close()
        return loss_evol_train

    def validation(self, x_val, y_val):
        """Compute the loss wrt to validation data."""
        x_batches = torch.split(x_val, self.batch_size)
        y_batches = torch.split(y_val, self.batch_size)
        self.model.eval()
        tot_loss = 0
        y_out = []
        with torch.no_grad():
            for x, y in zip(x_batches, y_batches):
                y_pred = self.model(x)
                y_out.append(y_pred)
                loss = self.loss_fn(y_pred, y)
                tot_loss += loss.item()
            y_out = torch.cat(y_out, dim=0)
        return tot_loss / len(x_batches)

    def testing(self, x_test, y_test):
        """Compute the RMSE wrt to training or test data."""
        x_batches = torch.split(x_test, self.batch_size)
        y_batches = torch.split(y_test, self.batch_size)
        self.model.eval()
        y_pred = []
        with torch.no_grad():
            for x, y in zip(x_batches, y_batches):
                y_pred.append(self.model(x))
            y_pred = torch.cat(y_pred, dim=0)
            rmse = torch.sqrt(mse_loss(y_pred, y_test)).item()
            rel_err = (
                y_pred - y_test
            ) / y_test  # torch.maximum(y_test, torch.Tensor([1.]).to(self.device))
        return rmse, rel_err

    def save_model(self):
        # Save the model
        torch.save(
            {
                "mean": self.model.mean,
                "std": self.model.std,
                "model": self.model.state_dict(),
            },
            f"nn_V_safe_log_{self.model.log_repr}.pt",
        )

    def plot_learning(self, ep=0):
        grid_for_net = np.tile(self.settings.state_grid,(self.settings.state_size,1)).T

        if ep == 0:
            # check correct size of array
            print(f'Shape of net grid {grid_for_net.shape} grid_for_net[0] {grid_for_net[0]}')

        V_net = (
            self.model.forward(
                torch.from_numpy(
                    (grid_for_net)
                )
                .float()
                .clone()
                .to(self.device)
            )
            .detach()
            .cpu()
            .numpy()
        )


        print(
            f"output for 1.8: {self.model.forward(torch.from_numpy((np.array([1.8]*self.settings.state_size)-self.settings.mean)/self.settings.std).float().to(self.device)).detach().cpu().numpy()}"
        )

        V_prob = np.copy(self.settings.V_prob)
        V_prob = np.power(V_prob,self.settings.state_size)
        # V_prob[V_prob==0.] = 1e-10
        

        # self.error_log.append(np.linalg.norm(np.log10(1-V_prob) - np.minimum(V_net.squeeze(),0)))
        # self.error.append(np.linalg.norm(V_prob - np.maximum(0,1-np.power(10,V_net.squeeze()))))

        plt.figure()
        plt.grid(True)
        if self.model.log_repr:

            self.error_log.append(np.linalg.norm(np.log10(1-V_prob) - np.minimum(V_net.squeeze(),0)))
            self.error.append(np.linalg.norm(V_prob - np.maximum(0,1-np.power(10,V_net.squeeze()))))

            # plt.yscale('log')
            plt.plot(
                self.settings.state_grid,
                np.log10(V_prob),
                "blue",
                label="V_prob",
            )
            plt.plot(self.settings.state_grid, np.minimum(V_net,0), "red", label="V_net")
            plt.xlabel("State Grid")
            plt.ylabel("V")

            plt.legend()
            plt.title(
                f"Epoch {ep} log10(V(x))  x_th: {self.settings.x_th} u_max: {self.settings.u_max} d_max: {self.settings.d_max} noise: {self.settings.distr}"
            )
            plt.savefig(
                f"plots/V_comparison{self.__class__.__name__}_x_th:{self.settings.x_th}_u_max:_{self.settings.u_max}_d_max:_{self.settings.d_max}_distr_{self.settings.distr}_state_size{self.settings.state_size}_log_repr{self.settings.log_repr}_ep{ep}.png"
            )
            # plt.show()
            plt.close()
            
            plt.figure()
            plt.grid(True)
            plt.plot(
                self.settings.state_grid,
                np.log10(1-V_prob),
                "blue",
                label="V_prob",
            )
            plt.plot(self.settings.state_grid, np.minimum(V_net,0), "red", label="V_net")
            plt.xlabel("State Grid")
            plt.ylabel("V")

            plt.legend()
            plt.title(
                f"Epoch {ep} log10(1-V(x))  x_th: {self.settings.x_th} u_max: {self.settings.u_max} d_max: {self.settings.d_max} noise: {self.settings.distr}"
            )
            plt.savefig(
                f"plots/V_comparison{self.__class__.__name__}_x_th:{self.settings.x_th}_u_max:_{self.settings.u_max}_d_max:_{self.settings.d_max}_distr_{self.settings.distr}_state_size{self.settings.state_size}_log_repr_1-V(X)_{self.settings.log_repr}_ep{ep}.png"
            )
            # plt.show()
            plt.close()

            plt.figure()
            plt.grid(True)
            plt.plot(
                self.settings.state_grid,
                np.log10(1-V_prob),
                "blue",
                label="V_prob",
            )
            plt.plot(self.settings.state_grid, np.minimum(V_net,0), "red", label="V_net")
            plt.xlabel("State Grid")
            plt.ylabel("V")

            plt.legend()
            plt.title(
                f"Epoch {ep} log10(1-V(x))  x_th: {self.settings.x_th} u_max: {self.settings.u_max} d_max: {self.settings.d_max} noise: {self.settings.distr}"
            )
            plt.savefig(
                f"plots/V_comparison{self.__class__.__name__}_x_th:{self.settings.x_th}_u_max:_{self.settings.u_max}_d_max:_{self.settings.d_max}_distr_{self.settings.distr}_state_size_log_repr_1-V(X)_{self.settings.log_repr}.png"
            )
            # plt.show()
            plt.close()


            plt.figure()
            plt.grid(True)
            plt.plot(
                self.settings.state_grid, V_prob, "blue", label="V_prob"
            )
            plt.gca().set_aspect("equal")
            plt.plot(self.settings.state_grid, np.maximum(0,1-np.power(10,V_net)), "red", label="V_net")
            plt.xlabel("State Grid")
            plt.ylabel("V")

            plt.legend()
            plt.title(
                f"Epoch {ep} V(x)  x_th: {self.settings.x_th} u_max: {self.settings.u_max} d_max: {self.settings.d_max} noise: {self.settings.distr}"
            )
            plt.savefig(
                f"plots/V_comparison{self.__class__.__name__}_x_th:{self.settings.x_th}_u_max:_{self.settings.u_max}_d_max:_{self.settings.d_max}_distr_{self.settings.distr}_state_size{self.settings.state_size}_log_repr{self.settings.log_repr}_ep{ep}.png"
            )
            plt.savefig(
                f"plots/V_comparison{self.__class__.__name__}_x_th:{self.settings.x_th}_u_max:_{self.settings.u_max}_d_max:_{self.settings.d_max}_distr_{self.settings.distr}_state_size{self.settings.state_size}_log_reprFalse_ep{ep}.png"
            )
            # plt.show()
            plt.close()          

        else:

            # print(f'V_net: {V_net}')

            self.error_log.append(np.linalg.norm(np.log10(1-V_prob) - np.log10(1-V_net.squeeze())))
            self.error.append(np.linalg.norm(V_prob - V_net.squeeze()))

            plt.plot(
                self.settings.state_grid, V_prob, "blue", label="V_prob"
            )
            plt.gca().set_aspect("equal")
            plt.plot(self.settings.state_grid, V_net, "red", label="V_net")
            plt.xlabel("State Grid")
            plt.ylabel("V")

            plt.legend()
            plt.title(
                f"Epoch {ep} V(x)  x_th: {self.settings.x_th} u_max: {self.settings.u_max} d_max: {self.settings.d_max} noise: {self.settings.distr}"
            )
            plt.savefig(
                f"plots/V_comparison{self.__class__.__name__}_x_th:{self.settings.x_th}_u_max:_{self.settings.u_max}_d_max:_{self.settings.d_max}_distr_{self.settings.distr}_state_size{self.settings.state_size}_log_repr{self.settings.log_repr}_ep{ep}.png"
            )
            plt.savefig(
                f"plots/V_comparison{self.__class__.__name__}_x_th:{self.settings.x_th}_u_max:_{self.settings.u_max}_d_max:_{self.settings.d_max}_distr_{self.settings.distr}_state_size{self.settings.state_size}_log_repr{self.settings.log_repr}.png"
            )
            # plt.show()
            plt.close()

            plt.figure()
            plt.grid(True)
            plt.plot(
                self.settings.state_grid,
                np.log10(1 - V_prob),
                "blue",
                label="V_prob",
            )
            plt.plot(
                self.settings.state_grid, np.log10(1 - V_net), "red", label="V_net"
            )
            # plt.xticks(np.arange(np.min(self.settings.state_grid), np.max(
            #     self.settings.state_grid)+0.2, 0.2), fontsize=10)
            plt.xlabel("State Grid")
            plt.ylabel("V")

            plt.legend()
            plt.title(
                f"Epoch {ep} V(x)  x_th: {self.settings.x_th} u_max: {self.settings.u_max} d_max: {self.settings.d_max} noise: {self.settings.distr} log_10(1-V(x))"
            )
            plt.savefig(
                f"plots/V_comparison{self.__class__.__name__}_x_th:{self.settings.x_th}_u_max:_{self.settings.u_max}_d_max:_{self.settings.d_max}_distr_{self.settings.distr}_state_size{self.settings.state_size}_log_repr_ep{ep}.png"
            )
            # plt.show()
            plt.close()

        plt.figure()
        plt.grid(True)
        plt.plot(
            self.error, "blue", label="error - traditional space",linewidth=2, linestyle='-', marker='o'
        )
        plt.xlabel("Epoch")
        plt.ylabel("Error")

        plt.legend()
        plt.title(
            f"ERROR Epoch {ep} V(x)  x_th: {self.settings.x_th} u_max: {self.settings.u_max} d_max: {self.settings.d_max} noise: {self.settings.distr}"
        )
        plt.savefig(
            f"plots/ERROR{self.__class__.__name__}_x_th:{self.settings.x_th}_u_max:_{self.settings.u_max}_d_max:_{self.settings.d_max}_distr_{self.settings.distr}_state_size{self.settings.state_size}.png"
        )
        plt.close()

        plt.figure()
        plt.grid(True)
        plt.plot(
            self.error_log, "blue", label="error - log space",linewidth=2, linestyle='-', marker='o'
        )
        plt.xlabel("Epoch")
        plt.ylabel("Error")

        plt.legend()
        plt.title(
            f"ERROR_LOG Epoch {ep} V(x)  x_th: {self.settings.x_th} u_max: {self.settings.u_max} d_max: {self.settings.d_max} noise: {self.settings.distr}"
        )
        plt.savefig(
            f"plots/ERROR_LOG{self.__class__.__name__}_x_th:{self.settings.x_th}_u_max:_{self.settings.u_max}_d_max:_{self.settings.d_max}_distr_{self.settings.distr}_state_size{self.settings.state_size}.png"
        )
        plt.close()


        # print(self.error)
    def soft_update(self):
        for target_param, source_param in zip(
            self.model_target.parameters(), self.model.parameters()
        ):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )

class RegressionSupervisedTraditionalSpace(RegressionNN):
    def __init__(self, batch_size, model, model_target, loss_fn, optimizer, settings):
        super().__init__( batch_size, model, model_target, loss_fn, optimizer, settings)

    def compute_target(self, x, successes, failures):
        pass
    
    def training(self, x_train, epochs):
        """Training of the neural network."""


        progress_bar = tqdm(total=epochs, desc="Training")

        loss_evol_train = []
        loss_evol_val = []
        loss_lp = 1

        n = len(x_train)

        # loader = DataLoader(x_train,shuffle=True,batch_size=self.batch_size,num_workers=32)
        for ep in range(epochs):

            V_ground_truth = torch.tensor(np.power(self.settings.V_prob,self.settings.state_size)).to(self.device).unsqueeze(-1).detach()
            states_grid =   torch.from_numpy((np.tile(self.settings.state_grid,(self.settings.state_size,1)).T)).float().clone().to(self.device).detach()

            self.model.train()
            # Shuffle the data
            idx = torch.randperm(n)
            x_perm = x_train[idx]
            # Split in batches
            x_batches = torch.split(x_perm, self.batch_size)

            for x in x_batches:
                # x = x.to(self.device)
                x_input, val_targets = self.compute_x_y_loss(x)

                # NAN CHECK
                # if torch.isnan(x_input).any():
                #     print(f"NaN detected in outputs at epoch {ep}")
                #     break
                # Compute the loss
                loss = self.loss_fn(x_input, val_targets)

                # if torch.isnan(loss).any():
                #     print(f"NaN detected in loss at epoch {ep}")
                #     break
                # Backward and optimize
                # print(f'loop : {ep}')
                self.optimizer.zero_grad()
                loss.backward()

                # for name, param in self.model.named_parameters():
                #     if torch.isnan(param.grad).any():
                #         print(f"NaN detected in gradients of {name} at epoch {ep}")
                #         break

                self.optimizer.step()

                loss_lp = self.beta * loss_lp + (1 - self.beta) * loss.item()

            if ep % 10 == 0:
                print(f"Loss training: {np.mean(loss_lp)}")
                with torch.no_grad():
                    print(f"Loss respect ground_truth: {self.loss_fn(self.model.forward(states_grid),V_ground_truth)}")
                self.plot_learning(ep)

            # x.detach().cpu()

            loss_evol_train.append(loss_lp)
           
            progress_bar.update(1)

        progress_bar.close()
        return loss_evol_train
    
    def compute_x_y_loss(self,x):
        val_targets = x[:,-1]
        x_input = x[:,:self.model.input_size]
        x_input = self.model(x_input)
        return x_input, val_targets.unsqueeze(-1)
    
class RegressionSupervisedLogTarget(RegressionSupervisedTraditionalSpace):
    def __init__(self, batch_size, model, model_target, loss_fn, optimizer, settings):
        super().__init__( batch_size, model, model_target, loss_fn, optimizer, settings)

    def compute_x_y_loss(self, x):
        val_targets = x[:,-1]
        val_targets = torch.maximum(torch.log10(1-val_targets),torch.tensor(-10)).detach()
        x_input = x[:,:self.model.input_size]
        x_input = torch.clamp(self.model(x_input),max=1-1e-5)
        x_input = torch.log10(1-x_input)
        return x_input, val_targets.unsqueeze(-1)
    
class RegressionSupervisedLogSpace(RegressionSupervisedTraditionalSpace):
    def __init__(self, batch_size, model, model_target, loss_fn, optimizer, settings):
        super().__init__(batch_size, model, model_target, loss_fn, optimizer, settings)
        self.model.ub = 1
        import types
        self.model.forward = types.MethodType(forward_clamp, self.model)
        
    def compute_x_y_loss(self, x):
        val_targets = x[:,-1]
        val_targets = torch.maximum(torch.log10(1-val_targets),torch.tensor(-10)).detach()
        val_target_max = torch.max(val_targets)
        if val_target_max > 0:
            print(f'Max val {val_target_max}')
        x_input = x[:,:self.model.input_size]
        x_input = self.model(x_input)
        # x_input = torch.clamp(x_input,max=0.)
        return x_input, val_targets.unsqueeze(-1)
    
    def plot_learning(self, ep=0):
        V_net = (
            self.model.forward(
                torch.from_numpy(
                    (np.tile(self.settings.state_grid,(self.settings.state_size,1)).T)
                )
                .float()
                .clone()
                .to(self.device)
            )
            .detach()
            .cpu()
            .numpy()
        )

        if ep == 0:
            # check correct size of a array
            grid_for_net = np.tile(self.settings.state_grid,(self.settings.state_size,1)).T
            print(f'Shape of net grid {grid_for_net.shape[0]}  grid_for_net[0]   {grid_for_net[0]}')

        print(
            f"output for 0.3: {self.model.forward(torch.from_numpy((np.array([0.3]*self.settings.state_size)-self.settings.mean)/self.settings.std).float().to(self.device)).detach().cpu().numpy()}"
        )

        # figure 1

        plt.figure()
        plt.grid(True)
    
        V_prob = np.copy(self.settings.V_prob)
        V_prob = np.power(V_prob,self.settings.state_size)
        # plt.yscale('log')
        plt.plot(
            self.settings.state_grid,
            np.log10(1-V_prob),
            "blue",
            label="V_prob",
        )
        plt.plot(self.settings.state_grid, V_net, "red", label="V_net")
        plt.xlabel("State Grid")
        plt.ylabel("V")

        plt.legend()
        plt.title(
            f"Epoch {ep} log10(1-V(x))  x_th: {self.settings.x_th} u_max: {self.settings.u_max} d_max: {self.settings.d_max} noise: {self.settings.distr}"
        )
        plt.savefig(
            f"plots/V_comparison{self.__class__.__name__}_x_th:{self.settings.x_th}_u_max:_{self.settings.u_max}_d_max:_{self.settings.d_max}_distr_{self.settings.distr}_state_size{self.settings.state_size}_log_repr_ep{ep}.png"
        )
        # plt.show()
        plt.close()

        # figure 2

        plt.figure()
        plt.grid(True)

        plt.plot(
            self.settings.state_grid,
            np.log10(1-V_prob),
            "blue",
            label="V_prob",
        )
        plt.plot(self.settings.state_grid, V_net, "red", label="V_net")
        plt.xlabel("State Grid")
        plt.ylabel("V")

        plt.legend()
        plt.title(
            f"Epoch {ep} log10(1-V(x))  x_th: {self.settings.x_th} u_max: {self.settings.u_max} d_max: {self.settings.d_max} noise: {self.settings.distr}"
        )
        plt.savefig(
            f"plots/V_comparison{self.__class__.__name__}_x_th:{self.settings.x_th}_u_max:_{self.settings.u_max}_d_max:_{self.settings.d_max}_distr_{self.settings.distr}_state_size{self.settings.state_size}_log_repr.png"
        )
        # plt.show()
        plt.close()

        # figure 3

        plt.figure()
        plt.grid(True)

        plt.plot(
            self.settings.state_grid,
            V_prob,
            "blue",
            label="V_prob",
        )
        plt.plot(self.settings.state_grid, 1-np.power(10,V_net), "red", label="V_net")
        plt.xlabel("State Grid")
        plt.ylabel("V")

        plt.legend()
        plt.title(
            f"Epoch {ep} V(x)  x_th: {self.settings.x_th} u_max: {self.settings.u_max} d_max: {self.settings.d_max} noise: {self.settings.distr}"
        )
        plt.savefig(
            f"plots/V_comparison{self.__class__.__name__}_x_th:{self.settings.x_th}_u_max:_{self.settings.u_max}_d_max:_{self.settings.d_max}_distr_{self.settings.distr}_state_size{self.settings.state_size}_log_reprFalse.png"
        )
        # plt.show()
        plt.close()
    
def forward_clamp(self, x):
    # out = self.linear_stack(x[:,:self.input_size])* self.ub
    out = self.linear_stack(x)
    out = torch.clamp(out,max=0.)

    return out * self.ub  # (out + 1) * self.ub / 2

