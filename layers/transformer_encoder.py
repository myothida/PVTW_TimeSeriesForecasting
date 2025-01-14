import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.attention import MultiHeadAttention, PositionwiseFeedForward


class TransformerEncoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(model_dim)
        self.ffn = PositionwiseFeedForward(model_dim, ff_dim, dropout)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Multi-head attention
        attention_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention_output))  # Add & Norm
        
        # Feed-forward network
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))  # Add & Norm
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, model_dim, num_heads, ff_dim, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(model_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x
