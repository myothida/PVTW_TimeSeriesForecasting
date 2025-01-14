import torch
import torch.nn as nn
import math

class TimeSeriesEmbedding(nn.Module):
    def __init__(self, input_dim, model_dim, max_len=5000):
        """
        Embeds the input time-series data into a higher-dimensional space and adds positional encoding.
        Args:
            input_dim (int): Number of features in the input data.
            model_dim (int): Dimensionality of the embedding space.
            max_len (int): Maximum sequence length for positional encoding.
        """
        super(TimeSeriesEmbedding, self).__init__()
        self.input_embedding = nn.Linear(input_dim, model_dim)
        self.positional_encoding = self._generate_positional_encoding(max_len, model_dim)
        self.model_dim = model_dim

    def forward(self, x):
        """
        Forward pass for embedding.
        Args:
            x (Tensor): Input tensor of shape [seq_length, batch_size, input_dim].
        Returns:
            Tensor: Embedded tensor of shape [seq_length, batch_size, model_dim].
        """
        x_embedded = self.input_embedding(x) * torch.sqrt(torch.tensor(self.model_dim, dtype=torch.float32))
        x_embedded += self.positional_encoding[: x.size(0), :].to(x.device)
        return x_embedded

    def _generate_positional_encoding(self, max_len, model_dim):
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2) * -(torch.log(torch.tensor(10000.0)) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(1)  # [max_len, 1, model_dim]






