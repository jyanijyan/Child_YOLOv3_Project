

import torch
import torch.nn as nn


class ConvBNLeaky(nn.Module):
    """
    Conv2d + BatchNorm2d + LeakyReLU(0.1)

    Args:
        in_channels (int)
        out_channels (int)
        kernel_size (int)
        stride (int, optional): default 1
        padding (int or None): None이면 same padding ((k-1)//2)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2  # same padding

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class ResidualBlock(nn.Module):
    """
    Darknet-style Residual Block
    - 입력:  B x C x H x W
    - 구조: C -> C/2 -> C, 마지막에 skip connection
    """
    def __init__(self, channels: int):
        super().__init__()
        hidden = channels // 2

        self.conv1 = ConvBNLeaky(channels, hidden, kernel_size=1)
        self.conv2 = ConvBNLeaky(hidden, channels, kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        return residual + out
