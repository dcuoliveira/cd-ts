
import torch

from models.Transformer import Transformer
        
class TransformerEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_edges, do_prob=0.):
        super(TransformerEncoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_edges = num_edges

        self.transformer1 = Transformer(input_dim, hidden_dim, hidden_dim, do_prob)
        self.transformer2 = Transformer(hidden_dim * 2, hidden_dim, hidden_dim, do_prob)
        self.transformer3 = Transformer(hidden_dim, hidden_dim, hidden_dim, do_prob)

        self.fc_out = torch.nn.Linear(hidden_dim, num_edges)
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

    def forward(self, inputs, rel_rec, rel_send):
        if len(inputs.shape) > 3:
            # input shape: [num_samples (batch_size), num_objects, num_timesteps, num_feature_per_obj]
            x = inputs.view(inputs.size(0), inputs.size(1), -1)
            # new shape: [num_samples, num_atoms, num_timesteps*num_feature_per_obj]
        else:
            x = inputs.view(1, inputs.shape[0], inputs.shape[1])

        # NOTE: num_timesteps*num_feature_per_obj timeseps => num_timesteps*num_dims parameters on the MLP
        # NOTE: why do we need to build the mlp parameters associated to the num_timesteps*num_feature_per_obj instead of num_atoms ?
        
        # node hidden representation
        x = self.transformer1(x)
        # from nodes to edges hidden representation
        x = self.node2edge(x, rel_rec, rel_send)
        x = self.transformer2(x)
        # keep edges first interaction hidden representation
        x_skip = x

        if self.factor:
            # aggregate edge represantations back to nodes (now we have more than one neighbor interaction for each node)
            x = self.edge2node(x, rel_rec)
            x = self.transformer3(x)
            # from nodes to edges hidden representation
            x = self.node2edge(x, rel_rec, rel_send)
            # add edges edges first interaction hidden representation to the edges final interation
            x = torch.cat((x, x_skip), dim=2)
            x = self.transformer4(x)
        else:
            x = self.mlp3(x)
            x = torch.cat((x, x_skip), dim=2)
            x = self.transformer4(x)

        return self.fc_out(x)