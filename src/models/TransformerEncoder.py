import math
import torch
import torch.nn as nn 
from torch import nn, Tensor


class PositionalEncoder(nn.Module):
    """
    The authors of the original transformer paper describe very succinctly what 
    the positional encoding layer does and why it is needed:
    
    "Since our model contains no recurrence and no convolution, in order for the 
    model to make use of the order of the sequence, we must inject some 
    information about the relative or absolute position of the tokens in the 
    sequence." (Vaswani et al, 2017)
    Adapted from: 
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(
        self, 
        dropout: float=0.1, 
        max_seq_len: int=5000, 
        d_model: int=512,
        batch_first: bool=False
        ):

        """
        Parameters:
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
            x: Tensor, shape [batch_size, enc_seq_len, dim_val] or [enc_seq_len, batch_size, dim_val]
        """
        if self.batch_first:
            x = x + self.pe[:,:x.size(1)]
        else:
            x = x + self.pe[:x.size(0)]

        return self.dropout(x)

class TransformerEncoder(nn.Module):

    def __init__(self, 
        input_size: int,
        dim_val: int=512,  
        n_encoder_layers: int=4,
        n_heads: int=8,
        dropout_encoder: float=0.2, 
        dropout_pos_enc: float=0.1,
        dim_feedforward_encoder: int=2048,
        num_predicted_features: int=1
        ): 

        """
        Args:

            input_size: int, number of input variables. 1 if univariate.
            dim_val: int, aka d_model. All sub-layers in the model produce 
                     outputs of dimension dim_val
            n_encoder_layers: int, number of stacked encoder layers in the encoder
            n_heads: int, the number of attention heads (aka parallel attention layers)
            dropout_encoder: float, the dropout rate of the encoder
            dropout_pos_enc: float, the dropout rate of the positional encoder
            dim_feedforward_encoder: int, number of neurons in the linear layer 
                                     of the encoder
            num_predicted_features: int, the number of features you want to predict.
        """

        super().__init__() 

        # Creating the three linear layers needed for the model
        self.encoder_input_layer = nn.Linear(
            in_features=input_size, 
            out_features=dim_val 
            )
        
        self.linear_mapping = nn.Linear(
            in_features=dim_val, 
            out_features=num_predicted_features
            )

        # Create positional encoder
        self.positional_encoding_layer = PositionalEncoder(
            d_model=dim_val,
            dropout=dropout_pos_enc
            )

        # The encoder layer used in the paper is identical to the one used by
        # Vaswani et al (2017) on which the PyTorch module is based.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_val, 
            nhead=n_heads,
            dim_feedforward=dim_feedforward_encoder,
            dropout=dropout_encoder,
            )

        # Stack the encoder layers in nn.TransformerDecoder
        # It seems the option of passing a normalization instance is redundant
        # in my case, because nn.TransformerEncoderLayer per default normalizes
        # after each sub-layer
        # (https://github.com/pytorch/pytorch/issues/24930).
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_encoder_layers, 
            norm=None
            )

    def forward(self,
                src: Tensor) -> Tensor:
   
        # linear layer
        # output shape: [src_seq_length, batch_size, hidden_dim]
        src = self.encoder_input_layer(src)
        # positional encoding
        src = self.positional_encoding_layer(src)

        # transformer encoder: muti-head self attention -> add & norm -> feed forward -> add & norm
        # each trasnformer encode consists of a multi-head self attention layer, a feed forward layer, and 2 residual sublayers
        # avici encoder consists of 4 transformer encoders
        # src shape: [src_seq_length, batch_size, hidden_dim]
        # NOTE - need attention mask ?
        encoder_output = self.encoder(src=src)
        encoder_output = self.encoder(src=encoder_output)
        encoder_output = self.encoder(src=encoder_output)
        encoder_output = self.encoder(src=encoder_output)

        # linear mapping
        # output shape [tgt_seq_length, batch_size, num_features]
        encoder_output = self.linear_mapping(encoder_output)

        return encoder_output
