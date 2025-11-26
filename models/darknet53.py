
import torch
import torch.nn as nn
from .layers import ConvBNLeaky, ResidualBlock

class Darknet53(nn.Module):
    def __init__(self):
        super().__init__()

        # 입력: 3 x 416 x 416
        self.layer1 = nn.Sequential(
            ConvBNLeaky(3, 32, 3, 1),
            ConvBNLeaky(32, 64, 3, 2),       # downsample
            ResidualBlock(64)
        )

        self.layer2 = nn.Sequential(
            ConvBNLeaky(64, 128, 3, 2),
            ResidualBlock(128),
            ResidualBlock(128)
        )

        self.layer3 = nn.Sequential(
            ConvBNLeaky(128, 256, 3, 2),
            *[ResidualBlock(256) for _ in range(8)]
        )

        self.layer4 = nn.Sequential(
            ConvBNLeaky(256, 512, 3, 2),
            *[ResidualBlock(512) for _ in range(8)]
        )

        self.layer5 = nn.Sequential(
            ConvBNLeaky(512, 1024, 3, 2),
            *[ResidualBlock(1024) for _ in range(4)]
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x3 = self.layer3(x)   # 52x52 (small objects용)
        x4 = self.layer4(x3)  # 26x26
        x5 = self.layer5(x4)  # 13x13
        return x3, x4, x5
