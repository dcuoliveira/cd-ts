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
                 input_dim: int,

                 embedd_hidden_dim: int,
                 
                 pos_enc_dropout: float,

                 encoder_input_dim: int,
                 n_encoder_layers: int,
                 n_ffnn_encoder_hidden: int,
                 n_encoder_heads: int,
                 encoder_dropout: float, 
  
                 decoder_input_dim: int,
                 n_decoder_layers: int,
                 n_ffnn_decoder_hidden: int,
                 n_decoder_heads: int,
                 decoder_dropout: float, 

                 out_dim: int,
                 
                 batch_first: bool): 

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

        self.linear_encoder_embedding = nn.Linear(in_features=input_dim, 
                                                  out_features=embedd_hidden_dim)
        
        self.positional_encoding_layer = PositionalEncoder(d_model=embedd_hidden_dim,
                                                           dropout=pos_enc_dropout) 

        encoder_layer = nn.TransformerEncoderLayer(d_model=encoder_input_dim, 
                                                   nhead=n_encoder_heads,
                                                   dim_feedforward=n_ffnn_encoder_hidden,
                                                   dropout=encoder_dropout,
                                                   batch_first=batch_first)

        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer,
                                             num_layers=n_encoder_layers,
                                             norm=None)
        
        self.linear_decoder_embedding = nn.Linear(in_features=input_dim,
                                                  out_features=embedd_hidden_dim) 
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=decoder_input_dim,
                                                   nhead=n_decoder_heads,
                                                   dim_feedforward=n_ffnn_decoder_hidden,
                                                   dropout=decoder_dropout,
                                                   batch_first=batch_first)

        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer,
                                             num_layers=n_decoder_layers, 
                                             norm=None)
        
        self.linear_mapping = nn.Linear(in_features=n_ffnn_decoder_hidden,
                                        out_features=out_dim)

    def forward(self,
                src: Tensor=None,
                tgt: Tensor=None,
                input_mask: Tensor=None) -> Tensor:
        
        # NOTE: src and tgt are the same for all inputs we have. Need to understand the implications of it.
        # NOTE: We are always considering input_mask=None. This means that we are ignoring the time ordering of the input on the decoder.

        if tgt is None:
            tgt = src

        # create embeddings on the feature dimension for the src input
        ## output shape: [batch_size, src_seq_length, embedding_dim]
        src_embedd = self.linear_encoder_embedding(src)

        # add positional encoding
        src_embedd_pos = self.positional_encoding_layer(src_embedd)

        # transformer encoder: embeddings -> {mhsa -> add & norm -> ffnn -> add & norm} * n -> encoder out
        ## src shape: [src_seq_length, batch_size, hidden_dim]
        src_embedd_pos_encoder = self.encoder(src=src_embedd_pos)

        # create embeddings on the feature dimension for the tgt input
        ## output shape: [batch_size, tgt_seq_length, hidden_dim]
        tgt_embedd = self.linear_decoder_embedding(tgt)

        # transformer decoder: embeddings -> {masked mhsa -> add & norm -> (-> encoder out) mhsa -> add & norm -> ffnn -> add & norm} * n -> decoder out
        ## output shape: [tgt_seq_length, batch_size, hidden_dim]
        tgt_embedd_decoder = self.decoder(tgt=tgt_embedd,
                                          memory=src_embedd_pos_encoder,
                                          tgt_mask=input_mask,
                                          memory_mask=input_mask)

        # linear mapping
        ## output shape [tgt_seq_length, batch_size, num_features]
        out = self.linear_mapping(tgt_embedd_decoder)

        return out
