import sys

import torch
from models.MLP import MLP
from torch import nn
from LRU_pytorch import LRU

class GRNNEncoder(nn.Module):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""

    def __init__(
        self, n_in, n_hid, n_edge_types, rnn='lstm', do_prob=0.0, factor=False
    ):
        super(GRNNEncoder, self).__init__()

        self.n_edge_types = n_edge_types

        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
        self.mlp2 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
        self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
        self.factor = factor
        if self.factor:
            self.mlp4 = MLP(n_hid * 3, n_hid, n_hid, do_prob)
        else:
            self.mlp4 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
        if rnn == 'lstm':
            self.rnn = nn.LSTM(n_hid, n_hid, num_layers=3, batch_first=True)
        elif rnn == 'gru':
            self.rnn = nn.GRU(n_hid, n_hid, num_layers=3, batch_first=True)
        elif rnn == 'lru':
            self.rnn = LRU(n_hid, n_hid, n_hid)
        else:
            raise NotImplementedError(rnn + ' is not implemented as a RNN block.')
        self.fc_out = nn.Linear(n_hid, n_edge_types)
        
        print("Using factor graph GRNN encoder.")

        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0.1)

    def edge2node(self, x, rel_rec):
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        # filter the hidden representation of each timestep to consider only the information of the receiver/sender node
        # rel_rec[num_features * num_atoms, num_atoms] * x[num_samples, num_atoms, num_timesteps]
        # receivers, senders: (num_samples, num_atoms,  num_timesteps*num_dims)
        receivers = torch.matmul(rel_rec, x) 
        senders = torch.matmul(rel_send, x)

        # concatecate filtered reciever/sender hidden representation
        edges = torch.cat([senders, receivers], dim=2)
        return edges

    def gnn_step(self):

        return 1

    def forward(self, inputs, rel_rec, rel_send):

        # Input has shape: [num_sims, num_atoms, num_timesteps, num_dims]
        B, N, T, D = inputs.size()
        x = inputs.transpose(1,2).reshape(B*T, N, D)
        # New shape: [num_sims*num_timesteps, num_atoms, num_dims]
        x = self.mlp1(x)  # 2-layer ELU net per node
        x = self.node2edge(x, rel_rec, rel_send)
        x = self.mlp2(x)
        x_skip = x

        if self.factor:
            x = self.edge2node(x, rel_rec, rel_send)
            x = self.mlp3(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)
        #else:
            #x = self.mlp3(x)
            #x = torch.cat((x, x_skip), dim=2)  # Skip connection
            #x = self.mlp4(x)
        R = x.size(1)
        x = x.reshape(B,T, R, -1).transpose(1,2).reshape(B*R,T,-1)
        # New shape: [num_sims*num_edges, num_timesteps, num_hid]
        x, _ = self.rnn(x)
        x = x[:,-1,:].reshape(B,R,-1)
        return self.fc_out(x).reshape(B, R, self.n_edge_types)
