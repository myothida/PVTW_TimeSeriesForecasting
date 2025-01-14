import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.head_dim = model_dim // num_heads

        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"

        self.query = nn.Linear(model_dim, model_dim)
        self.key = nn.Linear(model_dim, model_dim)
        self.value = nn.Linear(model_dim, model_dim)
        self.fc_out = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, mask=None):
        batch_size = queries.size(0)

        # Linear projections and split into heads
        Q = self.query(queries).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, heads, seq, head_dim]
        K = self.key(keys).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(values).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))  # [batch, heads, seq, seq]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # Combine heads
        out = torch.matmul(attention, V)  # [batch, heads, seq, head_dim]
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.model_dim)  # [batch, seq, model_dim]
        return self.fc_out(out)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, model_dim, ff_dim, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(model_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))