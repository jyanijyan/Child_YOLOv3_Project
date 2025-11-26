
## Conv + BN + LeakyReLU

import torch
import torch.nn as nn


class ConvBNLeaky(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2  # same padding

        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


## Residual Block

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # channels -> channels/2 -> channels
        hidden = channels // 2
        self.conv1 = ConvBNLeaky(channels, hidden, kernel_size=1)
        self.conv2 = ConvBNLeaky(hidden, channels, kernel_size=3)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))
