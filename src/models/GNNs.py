
import torch
from torch_geometric.nn import GCNConv

class VGAE(torch.nn.Module):
    def __init__(self, n_input, n_hidden_units, n_output):
        super().__init__()
        self.conv1 = GCNConv(n_input, n_hidden_units)
        self.conv2 = GCNConv(n_hidden_units, n_output)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

class VGAEWrapper():
    def __init__(self, input_size, trial):
        self.model_name = "vgae"
        self.params = {
              'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
              'n_hidden_units': trial.suggest_int("n_hidden_units", 10, 100),
              'n_output': trial.suggest_int("n_output", 10, 20),
              'optimizer': trial.suggest_categorical("optimizer", ["Adam"]),
              'n_input': trial.suggest_int("n_input", input_size, input_size),
              }
        self.n_epochs = 100

        self.ModelClass = VGAE(n_input=self.params["n_input"],
                               n_hidden_units=self.params["n_hidden_units"],
                               n_output=self.params["n_output"])