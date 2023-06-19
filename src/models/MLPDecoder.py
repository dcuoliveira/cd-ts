import torch

from torch import nn
import numpy as np

from models.MLP import MLP

class MLPDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_edges):
        super(MLPDecoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_edges = num_edges

        self.nets = nn.ModuleList([
            MLP(2*self.input_dim, self.hidden_dim, self.hidden_dim, use_batch_norm=False, final_layer=True) for _ in range(self.num_edges)]
        )
        self.out_fc = MLP(self.hidden_dim + self.input_dim, self.hidden_dim, self.input_dim, use_batch_norm=False, final_layer=True)

    def forward(self, x, edges, rel_rec, rel_send, teacher_forcing=10):
        x = x.transpose(1, 2)
        B, T, N, D = x.size()
        predictions = torch.zeros(B, T, N, D).to(x.device)
        # only take n-th timesteps as starting points (n: teacher_forcing)
        last_x = x[:, 0::teacher_forcing, :, :]

        for t in range(teacher_forcing):
            receivers = torch.matmul(rel_rec, last_x)
            senders = torch.matmul(rel_send, last_x)
            input = torch.cat([senders, receivers], dim=-1)
            all_msg = torch.zeros(
                input.size(0), input.size(1), input.size(2), self.hidden_dim
            ).to(x.device)

            # h_{ij, t} = \sum_{e>0} z_{ij, e} * f_e(x_{i, t}, x_{j, t})
            for i in range(1, self.num_edges):   
                msg = self.nets[i](input)
                all_msg += edges[:, None, :, i, None] * msg
            
            # aggregation: f_v([\sum_{i \= j} h_ij, x_j])
            aggr_hid = torch.matmul(rel_rec.t(), all_msg)
            
            # prediction: x_{j, t+1} = x_{j, t} + f_v([\sum_{i \= j} h_{ij, t}, x_{j, t}])
            aggr_in = torch.cat([aggr_hid, last_x], dim=3)
            last_x = last_x + self.out_fc(aggr_in)
            predictions[:, t::teacher_forcing, :, :] = last_x
        
        predictions = predictions.transpose(1, 2)[:, :, :-1, :]
        return predictions

if __name__ == '__main__':

    x = torch.randn(4, 5, 20, 2)

    print(x.shape)

    # Some code from ACD for debugging
    def encode_onehot(labels):
        """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
        return labels_onehot
            
    def create_rel_rec_send(num_atoms):
        """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
        # Generate off-diagonal interactio  n graph
        off_diag = np.ones([num_atoms, num_atoms]) - np.eye(num_atoms)

        rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
        rel_rec = torch.FloatTensor(rel_rec)
        rel_send = torch.FloatTensor(rel_send)

        return rel_rec, rel_send

    rel_rec, rel_send = create_rel_rec_send(5)
    edges = torch.zeros(4,5*4,3)
    net = MLPDecoder(2,8,3)

    print(net(x, edges, rel_rec, rel_send).shape)