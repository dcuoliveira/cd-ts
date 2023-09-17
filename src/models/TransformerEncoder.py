
import torch

from models.Transformer import Transformer
        
class TransformerEncoder(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 
                 embedd_hidden_dim,

                 pos_enc_dropout,

                 n_encoder_layers,
                 n_ffnn_encoder_hidden,
                 n_encoder_heads,
                 encoder_dropout,

                 n_decoder_layers,
                 n_ffnn_decoder_hidden,
                 n_decoder_heads,
                 decoder_dropout,

                 transformer_out_dim,

                 n_linear_input_out,

                 num_edges,
                 batch_first=False,
                 factor=True):
        super(TransformerEncoder, self).__init__()

        self.input_dim = input_dim
        self.factor = factor

        self.transformer1 = Transformer(input_dim=input_dim,
                                        
                                        embedd_hidden_dim=embedd_hidden_dim,

                                        pos_enc_dropout=pos_enc_dropout,

                                        encoder_input_dim=embedd_hidden_dim,
                                        n_encoder_layers=n_encoder_layers,
                                        n_ffnn_encoder_hidden=n_ffnn_encoder_hidden,
                                        n_encoder_heads=n_encoder_heads,
                                        encoder_dropout=encoder_dropout,
                                        
                                        decoder_input_dim=embedd_hidden_dim,
                                        n_decoder_layers=n_decoder_layers,
                                        n_ffnn_decoder_hidden=n_ffnn_decoder_hidden,
                                        n_decoder_heads=n_decoder_heads,
                                        decoder_dropout=decoder_dropout,

                                        out_dim=transformer_out_dim,

                                        batch_first=batch_first)
        
        self.transformer2 = Transformer(input_dim=input_dim * 4,
                                        
                                        embedd_hidden_dim=embedd_hidden_dim,

                                        pos_enc_dropout=pos_enc_dropout,

                                        encoder_input_dim=embedd_hidden_dim,
                                        n_encoder_layers=n_encoder_layers,
                                        n_ffnn_encoder_hidden=n_ffnn_encoder_hidden,
                                        n_encoder_heads=n_encoder_heads,
                                        encoder_dropout=encoder_dropout,
                                        
                                        decoder_input_dim=embedd_hidden_dim,
                                        n_decoder_layers=n_decoder_layers,
                                        n_ffnn_decoder_hidden=n_ffnn_decoder_hidden,
                                        n_decoder_heads=n_decoder_heads,
                                        decoder_dropout=decoder_dropout,

                                        out_dim=input_dim * 4,

                                        batch_first=batch_first)
        
        self.transformer3 = Transformer(input_dim=input_dim,
                                        
                                        embedd_hidden_dim=embedd_hidden_dim,

                                        pos_enc_dropout=pos_enc_dropout,

                                        encoder_input_dim=embedd_hidden_dim,
                                        n_encoder_layers=n_encoder_layers,
                                        n_ffnn_encoder_hidden=n_ffnn_encoder_hidden,
                                        n_encoder_heads=n_encoder_heads,
                                        encoder_dropout=encoder_dropout,
                                        
                                        decoder_input_dim=embedd_hidden_dim,
                                        n_decoder_layers=n_decoder_layers,
                                        n_ffnn_decoder_hidden=n_ffnn_decoder_hidden,
                                        n_decoder_heads=n_decoder_heads,
                                        decoder_dropout=decoder_dropout,

                                        out_dim=input_dim,

                                        batch_first=batch_first)
        
        if self.factor:
            self.transformer4 = Transformer(input_dim=input_dim * 4,
                                        
                                            embedd_hidden_dim=embedd_hidden_dim,

                                            pos_enc_dropout=pos_enc_dropout,

                                            encoder_input_dim=embedd_hidden_dim,
                                            n_encoder_layers=n_encoder_layers,
                                            n_ffnn_encoder_hidden=n_ffnn_encoder_hidden,
                                            n_encoder_heads=n_encoder_heads,
                                            encoder_dropout=encoder_dropout,
                                            
                                            decoder_input_dim=embedd_hidden_dim,
                                            n_decoder_layers=n_decoder_layers,
                                            n_ffnn_decoder_hidden=n_ffnn_decoder_hidden,
                                            n_decoder_heads=n_decoder_heads,
                                            decoder_dropout=decoder_dropout,

                                            out_dim=input_dim * 4,

                                            batch_first=batch_first)
        else:
            self.transformer4 = Transformer(input_dim=input_dim * 4,
                                        
                                            embedd_hidden_dim=embedd_hidden_dim,

                                            pos_enc_dropout=pos_enc_dropout,

                                            encoder_input_dim=embedd_hidden_dim,
                                            n_encoder_layers=n_encoder_layers,
                                            n_ffnn_encoder_hidden=n_ffnn_encoder_hidden,
                                            n_encoder_heads=n_encoder_heads,
                                            encoder_dropout=encoder_dropout,
                                            
                                            decoder_input_dim=embedd_hidden_dim,
                                            n_decoder_layers=n_decoder_layers,
                                            n_ffnn_decoder_hidden=n_ffnn_decoder_hidden,
                                            n_decoder_heads=n_decoder_heads,
                                            decoder_dropout=decoder_dropout,

                                            out_dim=input_dim * 4,

                                            batch_first=batch_first)
            print("Using Transformer encoder.")

        self.fc_out = torch.nn.Linear(n_linear_input_out, num_edges)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0.1)

    def edge2node(self, x, rel_rec):
        B, TD, N = x.shape
        x = x.reshape(B, N, TD)

        incoming = torch.matmul(rel_rec.t(), x)
        B, N, TD = incoming.shape

        return (incoming / incoming.size(1)).reshape(B, TD, N)

    def node2edge(self, x, rel_rec, rel_send):
        # filter the hidden representation of each timestep to consider only the information of the receiver/sender node
        ## rel_rec[num_features * num_atoms, num_atoms] * x[num_samples, num_atoms, num_timesteps]
        ## receivers, senders: (num_samples, num_atoms,  num_timesteps*num_dims)

        B, TD, N = x.shape
        x = x.reshape(B, N, TD)

        receivers = torch.matmul(rel_rec, x) 
        senders = torch.matmul(rel_send, x)

        # concatecate filtered reciever/sender hidden representation
        edges = torch.cat([senders, receivers], dim=2)
        B, N, TD = edges.shape

        return edges.reshape(B, TD, N)
    
    def permute_dims(self, x):
        return x.permute(1, 0, 2) if self.batch_first else x

    def forward(self, inputs, rel_rec, rel_send, mask=None):
        if len(inputs.shape) > 3:
            # input shape: [num_samples (batch_size), num_objects, num_timesteps, num_feature_per_obj]
            B, N, T, D = inputs.shape
            x = inputs.view(B, T*D, N)
            # new shape: [num_samples, num_atoms, num_timesteps*num_feature_per_obj]
        else:
            x = inputs.view(1, inputs.shape[0], inputs.shape[1])
        
        # node hidden representation
        ## NOTE: Attention is applied to the num_samples (batch_size) dimension
        ## NOTE - The decoder of each self.transformer can use an attention mask, but it is not. Do we need this here?
        x = self.transformer1.forward(src=x, input_mask=mask)

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
            x = torch.cat((x, x_skip), dim=1)
            x = self.transformer4(x)
        else:
            x = self.transformer3(x)
            x = torch.cat((x, x_skip), dim=2)
            x = self.transformer4(x)
        
        B, TD, N = x.shape
        x = x.reshape(B, N, TD)

        return self.fc_out(x)