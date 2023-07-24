import torch
import torch.nn.functional as F
from torch import nn

class MLP(torch.nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self,
                 n_in,
                 n_hid,
                 n_out,
                 do_prob=0.,
                 use_batch_norm=True,
                 final_layer=False,
                 activation='elu',
                 initialise_weights=True):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(n_in, n_hid)
        self.bn = torch.nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob
        self.use_batch_norm = use_batch_norm
        self.final_layer = final_layer
        if self.final_layer:
            self.fc2 = torch.nn.Linear(n_hid, n_hid)
            self.final = torch.nn.Linear(n_hid, n_out)
        else:
            self.fc2 = torch.nn.Linear(n_hid, n_out)
        if activation=='relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.ELU()
        if initialise_weights:
            self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, torch.nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        x = self.activation(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = self.activation(self.fc2(x))
        if self.final_layer:
            x = self.final(x)
        if self.use_batch_norm:
            return self.batch_norm(x)
        else:
            return x