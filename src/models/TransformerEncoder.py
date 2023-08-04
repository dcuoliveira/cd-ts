
import torch

from models.Transformer import Transformer
        
class TransformerEncoder(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 num_edges,
                 n_encoder_layers,
                 n_decoder_layers,
                 n_heads,
                 dim_feedforward_encoder,
                 dim_feedforward_decoder,
                 batch_first=True,
                 factor=True):
        super(TransformerEncoder, self).__init__()

        # parameters for the linear layer of the trasformer
        self.n_encoder_input_layer_in = input_dim
        self.n_encoder_input_layer_out = hidden_dim

        # parameters for the trasnformer encoder
        self.n_encoder_layers = n_encoder_layers
        self.n_decoder_layers = n_decoder_layers
        self.n_heads = n_heads
        self.dim_feedforward_encoder = dim_feedforward_encoder
        self.dim_feedforward_decoder = dim_feedforward_decoder

        # "graph" part of the transformer
        self.num_edges = num_edges
        self.factor = factor

        self.batch_first = batch_first

        self.transformer1 = Transformer(n_encoder_input_layer_in=self.n_encoder_input_layer_in,
                                        n_encoder_input_layer_out=self.n_encoder_input_layer_out,
                                        n_encoder_layers=self.n_encoder_layers,
                                        n_heads=self.n_heads,
                                        num_predicted_features=self.n_encoder_input_layer_out,
                                        dim_feedforward_encoder=self.dim_feedforward_encoder,
                                        batch_first=self.batch_first)
        
        self.transformer2 = Transformer(n_encoder_input_layer_in=self.n_encoder_input_layer_out * 2,
                                        n_encoder_input_layer_out=self.n_encoder_input_layer_out,
                                        n_encoder_layers=self.n_encoder_layers,
                                        n_heads=self.n_heads,
                                        num_predicted_features=self.n_encoder_input_layer_out,
                                        dim_feedforward_encoder=self.dim_feedforward_encoder,
                                        batch_first=self.batch_first)
        
        self.transformer3 = Transformer(n_encoder_input_layer_in=self.n_encoder_input_layer_out,
                                        n_encoder_input_layer_out=self.n_encoder_input_layer_out,
                                        n_encoder_layers=self.n_encoder_layers,
                                        n_heads=self.n_heads,
                                        num_predicted_features=self.n_encoder_input_layer_out,
                                        dim_feedforward_encoder=self.dim_feedforward_encoder,
                                        batch_first=self.batch_first)
        
        if self.factor:
            self.transformer4 = Transformer(n_encoder_input_layer_in=self.n_encoder_input_layer_out * 3,
                                            n_encoder_input_layer_out=self.n_encoder_input_layer_out,
                                            n_encoder_layers=self.n_encoder_layers,
                                            n_heads=self.n_heads,
                                            num_predicted_features=self.n_encoder_input_layer_out,
                                            dim_feedforward_encoder=self.dim_feedforward_encoder,
                                            batch_first=self.batch_first)
            print("Using factor graph Transformer encoder.")
        else:
            self.transformer4 = Transformer(n_encoder_input_layer_in=self.n_encoder_input_layer_out * 3,
                                            n_encoder_input_layer_out=self.n_encoder_input_layer_out,
                                            n_encoder_layers=self.n_encoder_layers,
                                            n_heads=self.n_heads,
                                            num_predicted_features=self.n_encoder_input_layer_out,
                                            dim_feedforward_encoder=self.dim_feedforward_encoder,
                                            batch_first=self.batch_first)
            print("Using Transformer encoder.")

        self.fc_out = torch.nn.Linear(self.n_encoder_input_layer_out, num_edges)
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
    
    def permute_dims(self, x):
        return x.permute(1, 0, 2) if self.batch_first else x

    def forward(self, inputs, rel_rec, rel_send):
        if len(inputs.shape) > 3:
            # input shape: [num_samples (batch_size), num_objects, num_timesteps, num_feature_per_obj]
            x = inputs.view(inputs.size(0), inputs.size(1), -1)
            # new shape: [num_samples, num_atoms, num_timesteps*num_feature_per_obj]
        else:
            x = inputs.view(1, inputs.shape[0], inputs.shape[1])

        if self.batch_first:
            x = x.permute(1, 0, 2)

        # NOTE: num_timesteps*num_feature_per_obj timeseps => num_timesteps*num_dims parameters on the MLP
        # NOTE: why do we need to build the mlp parameters associated to the num_timesteps*num_feature_per_obj instead of num_atoms ?
        
        # node hidden representation
        x = self.transformer1(x)
        x = self.permute_dims(x)

        # from nodes to edges hidden representation
        x = self.node2edge(x, rel_rec, rel_send)
        x = self.permute_dims(x)
        x = self.transformer2(x)

        x = self.permute_dims(x)

        # keep edges first interaction hidden representation
        x_skip = x

        if self.factor:
            # aggregate edge represantations back to nodes (now we have more than one neighbor interaction for each node)
            x = self.edge2node(x, rel_rec)
            x = self.permute_dims(x)
            x = self.transformer3(x)
            x = self.permute_dims(x)

            # from nodes to edges hidden representation
            x = self.node2edge(x, rel_rec, rel_send)

            # add edges edges first interaction hidden representation to the edges final interation
            x = torch.cat((x, x_skip), dim=2)
            x = self.permute_dims(x)
            x = self.transformer4(x)
        else:
            x = self.permute_dims(x)
            x = self.transformer3(x)
            x = self.permute_dims(x)
            x = torch.cat((x, x_skip), dim=2)
            x = self.permute_dims(x)
            x = self.transformer4(x)

        x = self.permute_dims(x)
        
        return self.fc_out(x)