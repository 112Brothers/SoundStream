import torch
from torch import nn

class ResidualUnit(nn.Module):
    def __init__(self, channels, d=1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=d, dilation=d)
        self.act = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        residual = x
        x = self.conv2(self.act(self.conv1(x)))
        return x + residual
