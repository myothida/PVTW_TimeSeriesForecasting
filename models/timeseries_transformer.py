import torch
import torch.nn as nn
from layers.attention import MultiHeadAttention
from layers.embedding import TimeSeriesEmbedding
from layers.decomposition import magnitude_based_decomp
from layers.transformer_encoder import TransformerEncoder
from layers.transformer_decoder import TransformerDecoder

"""
Customized Time Series Transformer for PVTW
"""

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_encoder_layers, num_decoder_layers, ff_dim, output_dim, threshold, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()

        # Use the embedding class
        self.mag_decomp = magnitude_based_decomp(threshold)
        self.embedding = TimeSeriesEmbedding(input_dim, model_dim)
        
        # Custom Transformer Encoder
        self.encoder = TransformerEncoder(
            model_dim=model_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_layers=num_encoder_layers,
            dropout=dropout
        )

        # Custom Transformer Decoder
        self.decoder = TransformerDecoder(
            model_dim=model_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_layers=num_decoder_layers,
            dropout=dropout
        )
        
        # Output projection
        self.output_layer = nn.Linear(model_dim, output_dim)

    def forward(self, src, tgt):

        # Magnitude decomposition
        src_smaller, src_larger = self.mag_decomp(src)
        tgt_smaller, tgt_larger = self.mag_decomp(tgt)

        # Apply embeddings separately for decomposed components
        src_smaller_embedded = self.embedding(src_smaller)  # Smaller waveforms
        src_larger_embedded = self.embedding(src_larger)    # Larger waveforms
        
        tgt_smaller_embedded = self.embedding(tgt_smaller)  # Smaller waveforms
        tgt_larger_embedded = self.embedding(tgt_larger)    # Larger waveforms

        # Combine after embedding (and potentially positional encoding)
        src_embedded = src_smaller_embedded + src_larger_embedded
        tgt_embedded = tgt_smaller_embedded + tgt_larger_embedded

        # Use the embedding class for input and target
        #src_embedded = self.embedding(src)  # [input_seq_length, batch_size, model_dim]
        #tgt_embedded = self.embedding(tgt)  # [output_seq_length, batch_size, model_dim]

        # Pass through encoder
        memory = self.encoder(src_embedded)        
        output = self.decoder(tgt_embedded, memory)

        # Final output projection
        return self.output_layer(output)


    
