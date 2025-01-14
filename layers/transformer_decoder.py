import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.attention import MultiHeadAttention, PositionwiseFeedForward



class TransformerDecoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(model_dim)
        self.cross_attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(model_dim)
        self.ffn = PositionwiseFeedForward(model_dim, ff_dim, dropout)
        self.norm3 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        Args:
            tgt: Target sequence [batch_size, tgt_seq_length, model_dim]
            memory: Encoder output [batch_size, src_seq_length, model_dim]
            tgt_mask: Target sequence mask [tgt_seq_length, tgt_seq_length]
            memory_mask: Source sequence mask [tgt_seq_length, src_seq_length]

        Returns:
            Updated target sequence [batch_size, tgt_seq_length, model_dim]
        """
        # Self-attention (target attending to itself)
        self_attention_output = self.self_attention(tgt, tgt, tgt, tgt_mask)
        tgt = self.norm1(tgt + self.dropout(self_attention_output))  # Add & Norm

        # Cross-attention (target attending to encoder output)
        cross_attention_output = self.cross_attention(tgt, memory, memory, memory_mask)
        tgt = self.norm2(tgt + self.dropout(cross_attention_output))  # Add & Norm

        # Feed-forward network
        ffn_output = self.ffn(tgt)
        tgt = self.norm3(tgt + self.dropout(ffn_output))  # Add & Norm
        return tgt

class TransformerDecoder(nn.Module):
    def __init__(self, model_dim, num_heads, ff_dim, num_layers, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(model_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)
        ])

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        Args:
            tgt: Target sequence [batch_size, tgt_seq_length, model_dim]
            memory: Encoder output [batch_size, src_seq_length, model_dim]
            tgt_mask: Target sequence mask [tgt_seq_length, tgt_seq_length]
            memory_mask: Source sequence mask [tgt_seq_length, src_seq_length]

        Returns:
            Final target sequence [batch_size, tgt_seq_length, model_dim]
        """
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, memory_mask)
        return tgt
