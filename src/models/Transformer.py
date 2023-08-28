import math
import torch
import torch.nn as nn 
from torch import nn, Tensor

class PositionalEncoder(nn.Module):

    def __init__(
        self, 
        dropout: float=0.1, 
        max_seq_len: int=5000, 
        d_model: int=512,
        batch_first: bool=False
        ):

        """
        Args:
            dropout: the dropout rate
            max_seq_len: the maximum length of the input sequences
            d_model: The dimension of the output of sub-layers in the model (Vaswani et al, 2017)
        """

        super().__init__()

        self.d_model = d_model
        self.batch_first = batch_first
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        self.dropout = nn.Dropout(p=dropout)  

        if self.batch_first:
            pe = torch.zeros(1, max_seq_len, d_model)
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term)
        else:
            pe = torch.zeros(max_seq_len, 1, d_model)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: Tensor) -> Tensor:
        
        """
        Args:
            x: Tensor, shape [batch_size, enc_seq_len, d_model] or [enc_seq_len, batch_size, d_model]
        """

        if self.batch_first:
            x = x + self.pe[:,:x.size(1)]
        else:
            x = x + self.pe[:x.size(0)]

        return self.dropout(x)

class Transformer(nn.Module):

    def __init__(self, 
        n_encoder_input_layer_in: int,
        n_encoder_input_layer_out: int,
        n_encoder_layers: int=4,
        n_decoder_layers: int=4,
        n_heads: int=8,
        dropout_encoder: float=0.2, 
        dropout_decoder: float=0.2, 
        dropout_pos_enc: float=0.1,
        dim_feedforward_encoder: int=2048,
        dim_feedforward_decoder: int=2048,
        num_predicted_features: int=1,
        batch_first: bool=True,
        ): 

        """
        Args:

            n_in: int, number of input variables. 1 if univariate.
            n_out: int, aka d_model. All sub-layers in the model produce 
                     outputs of dimension n_out
            n_encoder_layers: int, number of stacked encoder layers in the encoder
            n_heads: int, the number of attention heads (aka parallel attention layers)
            dropout_encoder: float, the dropout rate of the encoder
            dropout_pos_enc: float, the dropout rate of the positional encoder
            dim_feedforward_encoder: int, number of neurons in the linear layer 
                                     of the encoder
            num_predicted_features: int, the number of features you want to predict.
        """

        super().__init__() 

        self.encoder_input_layer = nn.Linear(
            in_features=n_encoder_input_layer_in, 
            out_features=n_encoder_input_layer_out 
            )
        
        self.decoder_input_layer = nn.Linear(
            in_features=num_predicted_features,
            out_features=n_encoder_input_layer_out
            )  
        
        self.linear_mapping = nn.Linear(
            in_features=n_encoder_input_layer_out, 
            out_features=num_predicted_features
            )

        self.positional_encoding_layer = PositionalEncoder(
            d_model=n_encoder_input_layer_out,
            dropout=dropout_pos_enc
            )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_encoder_input_layer_out, 
            nhead=n_heads,
            dim_feedforward=dim_feedforward_encoder,
            dropout=dropout_encoder,
            batch_first=batch_first
            )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_encoder_layers, 
            norm=None
            )
        
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=n_encoder_input_layer_out,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_decoder,
            dropout=dropout_decoder,
            batch_first=batch_first
            )

        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_decoder_layers, 
            norm=None
            )

    def forward(self,
                inputs: Tensor,
                input_mask: Tensor=None) -> Tensor:
   
        # linear layer
        ## output shape: [src_seq_length, batch_size, hidden_dim]
        x = self.encoder_input_layer(inputs)
        # positional encoding
        x = self.positional_encoding_layer(x)

        # transformer encoder: muti-head self attention -> add & norm -> feed forward -> add & norm
        ## src shape: [src_seq_length, batch_size, hidden_dim]
        encoder_output = self.encoder(src=x)

        # transformer decoder I: masked muti-head self attention -> add & norm
        ## output shape: [tgt_seq_length, batch_size, hidden_dim]
        decoder_output = self.decoder_input_layer(encoder_output)

        # transformer decoder II: muti-head attention -> add & norm -> feed forward -> add & norm
        ## output shape: [tgt_seq_length, batch_size, hidden_dim]
        decoder_output = self.decoder(
            tgt=decoder_output,
            memory=x,
            tgt_mask=input_mask,
            memory_mask=input_mask # src and tgt masks may be different
            )

        # linear mapping
        ## output shape [tgt_seq_length, batch_size, num_features]
        decoder_output = self.linear_mapping(encoder_output)

        return decoder_output
