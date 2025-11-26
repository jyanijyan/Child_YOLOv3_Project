

import torch
import torch.nn as nn
from .layers import ConvBNLeaky, ResidualBlock


class Darknet53(nn.Module):
    """
    Darknet-53 Backbone (YOLOv3 논문 기반)
    
    입력:
        x: B x 3 x 416 x 416
        
    출력:
        C3: B x 256 x 52 x 52   (small objects)
        C4: B x 512 x 26 x 26   (medium)
        C5: B x 1024 x 13 x 13  (large)
    
    구조:
        layer1:  1 residual
        layer2:  2 residual
        layer3:  8 residual
        layer4:  8 residual
        layer5:  4 residual
    """

    def __init__(self):
        super().__init__()

        # 입력: (3, 416, 416)
        self.layer1 = nn.Sequential(
            ConvBNLeaky(3, 32, 3, 1),
            ConvBNLeaky(32, 64, 3, 2),  # stride=2 → downsample
            ResidualBlock(64),
        )

        self.layer2 = nn.Sequential(
            ConvBNLeaky(64, 128, 3, 2),
            ResidualBlock(128),
            ResidualBlock(128),
        )

        self.layer3 = nn.Sequential(
            ConvBNLeaky(128, 256, 3, 2),
            *[ResidualBlock(256) for _ in range(8)],
        )

        self.layer4 = nn.Sequential(
            ConvBNLeaky(256, 512, 3, 2),
            *[ResidualBlock(512) for _ in range(8)],
        )

        self.layer5 = nn.Sequential(
            ConvBNLeaky(512, 1024, 3, 2),
            *[ResidualBlock(1024) for _ in range(4)],
        )

    def forward(self, x):
        x = self.layer1(x)  # B x 64 x 208 x 208
        x = self.layer2(x)  # B x 128 x 104 x 104

        c3 = self.layer3(x)  # B x 256 x 52 x 52
        c4 = self.layer4(c3) # B x 512 x 26 x 26
        c5 = self.layer5(c4) # B x 1024 x 13 x 13

        return c3, c4, c5
