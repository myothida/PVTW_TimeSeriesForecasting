import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return trend, seasonal
    

class magnitude_based_decomp(nn.Module):
    """
    Decompose time series based on magnitude into smaller and larger waveforms
    """
    def __init__(self, threshold):
        super(magnitude_based_decomp, self).__init__()
        self.threshold = threshold

    def forward(self, x):
        """
        Decomposes the series into smaller and larger waveforms.
        Args:
            x (torch.Tensor): Input time series of shape (batch_size, sequence_length, num_features)
        Returns:
            smaller (torch.Tensor): Smaller waveform components
            larger (torch.Tensor): Larger waveform components
        """
        magnitude = torch.abs(x)
        smaller = torch.where(magnitude < self.threshold, x, torch.zeros_like(x))
        larger = torch.where(magnitude >= self.threshold, x, torch.zeros_like(x))
        return smaller, larger