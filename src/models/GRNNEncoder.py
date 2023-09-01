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

        self.hidden_dim = n_hid
        self.n_edge_types = n_edge_types
        self.num_rec_layers = 3

        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
        self.mlp2 = MLP(n_hid*2, n_hid, n_hid, do_prob)
        self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
        self.mlp4 = MLP(n_hid*2, n_hid, n_hid, do_prob)
        

        if rnn == 'lstm':
            num_rec_layers = 6
            self.hidden_decoder = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
            if num_rec_layers!= 1:
                self.hidden_decoder = nn.ModuleList([nn.LSTMCell(self.hidden_dim, self.hidden_dim)])
                for _ in range(1, num_rec_layers):
                    self.hidden_decoder.append(nn.LSTMCell(self.hidden_dim*2, self.hidden_dim))
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
        R = N*(N-1)
        x = inputs.transpose(1,2).reshape(B*T, N, D)
        # New shape: [num_sims*num_timesteps, num_atoms, num_dims]
        x = self.mlp1(x)  # 2-layer ELU net per node
        x = self.node2edge(x, rel_rec, rel_send)
        x = self.mlp2(x).reshape(B,T,R,self.hidden_dim)
        h_prev = [torch.zeros((B*R, self.hidden_dim)).to(inputs.device) for _ in range(self.num_rec_layers)]
        c_prev = [torch.zeros((B*R, self.hidden_dim)).to(inputs.device) for _ in range(self.num_rec_layers)]
        for t in range(T):
            prev_input = x[:,t,:,:].reshape(B*R,-1)
            for i in range(self.num_rec_layers):
                h_prev[i], c_prev[i] = self.hidden_decoder[i](prev_input, (h_prev[i], c_prev[i]))
                h_prev[i] = self.edge2node(h_prev[i].reshape(B,R,-1), rel_rec)
                h_prev[i] = self.mlp3(h_prev[i])
                h_prev[i] = self.node2edge(h_prev[i], rel_rec, rel_send)
                h_prev[i] = self.mlp4(h_prev[i]).reshape(B*R,-1)
                if i==0:
                    prev_input = torch.cat([x[:,t,:,:].reshape(B*R,-1), h_prev[i]], dim=-1)
                else:
                    prev_input = torch.cat([h_prev[i-1], h_prev[i]], dim=-1)
        x = h_prev[-1].reshape(B,R,-1)
        return self.fc_out(x).reshape(B, -1, self.n_edge_types)
