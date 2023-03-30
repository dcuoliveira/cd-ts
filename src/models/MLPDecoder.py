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
            MLP(2*self.input_dim, self.hidden_dim, self.hidden_dim) for i in range(self.num_edges)]
        )
        self.out_fc = nn.Linear(self.hidden_dim, self.input_dim)

    def forward(self, x, edges, rel_rec, rel_send):
        x = x.transpose(1, 2)
        B, T, N, D = x.size()
        predictions = torch.zeros(B,T-1,N,D)
        for t in range(1,T):

            current_x = x[:,t-1,:,:]
            receivers = torch.matmul(rel_rec, current_x)
            senders = torch.matmul(rel_send, current_x)
            input = torch.cat([senders, receivers], dim=-1)
            all_msg = torch.zeros(
                input.size(0), input.size(1), self.hidden_dim
            )
            # h_ij = f(x_i, x_j)
            for i in range(1,self.num_edges):   
                msg = self.nets[i](input)
                all_msg += msg*edges[:,:,i,None]
            # aggregation: \sum_{i\=j}h_ij
            aggr_hid = torch.matmul(rel_rec.t(), all_msg)
            # prediction: x_j = f(\sum_{i\=j}h_ij) + x_j,t-1
            predictions[:,t-1,:,:] = self.out_fc(aggr_hid) + x[:,t-1,:,:]
        predictions = predictions.transpose(1, 2)
        return predictions

if __name__ == '__main__':

    x = torch.randn(4, 20, 5, 2)

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
        # Generate off-diagonal interaction graph
        off_diag = np.ones([num_atoms, num_atoms]) - np.eye(num_atoms)

        rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
        rel_rec = torch.FloatTensor(rel_rec)
        rel_send = torch.FloatTensor(rel_send)

        return rel_rec, rel_send

    rel_rec, rel_send = create_rel_rec_send(5)
    edges = torch.zeros(4,5*4,3)
    net = MLPDecoder(2,8,2,3)

    print(net(x, edges, rel_rec, rel_send).shape)