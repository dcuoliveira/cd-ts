import torch

from torch import nn
import numpy as np

from torch.nn import functional as F

from models.MLP import MLP

class MLPDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_edges):
        super(MLPDecoder, self).__init__()
        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2 * input_dim, hidden_dim) for _ in range(num_edges)]
        )
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_edges)]
        )
        self.msg_out_shape = hidden_dim
        self.skip_first_edge_type = True

        self.out_fc1 = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.out_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out_fc3 = nn.Linear(hidden_dim, input_dim)

        print("Using learned interaction net decoder.")


    def single_step_forward(
        self, single_timestep_inputs, rel_rec, rel_send, single_timestep_rel_type
    ):

        # single_timestep_inputs has shape
        # [batch_size, num_timesteps, num_atoms, num_dims]

        # single_timestep_rel_type has shape:
        # [batch_size, num_timesteps, num_atoms*(num_atoms-1), num_edge_types]

        # Node2edge
        receivers = torch.matmul(rel_rec, single_timestep_inputs)
        senders = torch.matmul(rel_send, single_timestep_inputs)
        pre_msg = torch.cat([senders, receivers], dim=-1)

        all_msgs = torch.zeros(
            pre_msg.size(0), pre_msg.size(1), pre_msg.size(2), self.msg_out_shape
        )

        if single_timestep_inputs.is_cuda:
            all_msgs = all_msgs.cuda()

        if self.skip_first_edge_type:
            start_idx = 1
        else:
            start_idx = 0

        # Run separate MLP for every edge type
        # NOTE: To exclude one edge type, simply offset range by 1
        for i in range(start_idx, len(self.msg_fc2)):
            msg = F.relu(self.msg_fc1[i](pre_msg))
            msg = F.relu(self.msg_fc2[i](msg))
            msg = msg * single_timestep_rel_type[:, :, :, i : i + 1]
            all_msgs += msg

        # Aggregate all msgs to receiver
        agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous()

        # Skip connection
        aug_inputs = torch.cat([single_timestep_inputs, agg_msgs], dim=-1)

        # Output MLP
        pred = F.relu(self.out_fc1(aug_inputs))
        pred = F.relu(self.out_fc2(pred))
        pred = self.out_fc3(pred)

        # Predict position/velocity difference
        return single_timestep_inputs + pred

    def forward(self, inputs, rel_type, rel_rec, rel_send, pred_steps=1):
        # NOTE: Assumes that we have the same graph across all samples.

        inputs = inputs.transpose(1, 2).contiguous()

        sizes = [
            rel_type.size(0),
            inputs.size(1),
            rel_type.size(1),
            rel_type.size(2),
        ]  # batch, sequence length, interactions between particles, interaction types
        rel_type = rel_type.unsqueeze(1).expand(
            sizes
        )  # copy relations over sequence length

        time_steps = inputs.size(1)
        assert pred_steps <= time_steps
        preds = []

        # Only take n-th timesteps as starting points (n: pred_steps)
        last_pred = inputs[:, 0::pred_steps, :, :]
        curr_rel_type = rel_type[:, 0::pred_steps, :, :]
        # NOTE: Assumes rel_type is constant (i.e. same across all time steps).

        # Run n prediction steps
        for step in range(0, pred_steps):
            last_pred = self.single_step_forward(
                last_pred, rel_rec, rel_send, curr_rel_type
            )
            preds.append(last_pred)

        sizes = [
            preds[0].size(0),
            preds[0].size(1) * pred_steps,
            preds[0].size(2),
            preds[0].size(3),
        ]

        output = torch.zeros(sizes)
        if inputs.is_cuda:
            output = output.cuda()

        # Re-assemble correct timeline
        for i in range(len(preds)):
            output[:, i::pred_steps, :, :] = preds[i]

        pred_all = output[:, : (inputs.size(1) - 1), :, :]

        return pred_all.transpose(1, 2).contiguous()

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