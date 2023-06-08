
import torch

from models.MLP import MLP
        
class MLPEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_edges, do_prob=0., factor=True):
        super(MLPEncoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_edges = num_edges
        self.factor = factor

        self.mlp1 = MLP(input_dim, hidden_dim, hidden_dim, do_prob)
        self.mlp2 = MLP(hidden_dim * 2, hidden_dim, hidden_dim, do_prob)
        self.mlp3 = MLP(hidden_dim, hidden_dim, hidden_dim, do_prob)
        if self.factor:
            self.mlp4 = MLP(hidden_dim * 3, hidden_dim, hidden_dim, do_prob)
            print("Using factor graph MLP encoder.")
        else:
            self.mlp4 = MLP(hidden_dim * 2, hidden_dim, hidden_dim, do_prob)
            print("Using MLP encoder.")
        self.fc_out = torch.nn.Linear(hidden_dim, num_edges)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0.1)

    def edge2node(self, x, rel_rec):
        # assumes that we have the same graph across all samples.
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        # assumes that we have the same graph across all samples.
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([senders, receivers], dim=2)
        return edges

    def forward(self, inputs, rel_rec, rel_send):
        if len(inputs.shape) > 3:
            # input shape: [num_samples (batch_size), num_objects, num_timesteps, num_dims]
            x = inputs.view(inputs.size(0), inputs.size(1), -1)
            # new shape: [num_sims, num_atoms, num_timesteps*num_dims]
        else:
            x = inputs.view(1, inputs.shape[0], inputs.shape[1])

        # NOTE: num_timesteps*num_dims timeseps => num_timesteps*num_dims parameters on the MLP
        # NOTE: why do we need to build the mlp parameters associated to the num_timesteps*num_dims instead of num_atoms ?
        x = self.mlp1(x)
        x = self.node2edge(x, rel_rec, rel_send)
        x = self.mlp2(x)
        x_skip = x

        if self.factor:
            x = self.edge2node(x, rel_rec)
            x = self.mlp3(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat((x, x_skip), dim=2)
            x = self.mlp4(x)
        else:
            x = self.mlp3(x)
            x = torch.cat((x, x_skip), dim=2)
            x = self.mlp4(x)

        return self.fc_out(x)
    
class MLPEncoderWrapper():
    def __init__(self, input_dim, hidden_dim=256, num_edges=2):
        self.model_name = "nrimlp"
        self.n_epochs = 100

        self.ModelClass = MLPEncoder(input_dim=input_dim,
                                     hidden_dim=hidden_dim,
                                     num_edges=num_edges)