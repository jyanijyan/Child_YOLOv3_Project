import torch
import torch.nn as nn

from .layers import ConvBNLeaky
from .darknet53 import Darknet53


class YoloV3Neck(nn.Module):
    """
    YOLOv3 FPN Neck
    - 입력:  Darknet53에서 나온 C3, C4, C5 feature maps
        C3: B x 256 x 52 x 52
        C4: B x 512 x 26 x 26
        C5: B x 1024 x 13 x 13
    - 출력: P1, P2, P3
        P1: B x 128 x 52 x 52  (small objects)
        P2: B x 256 x 26 x 26  (medium)
        P3: B x 512 x 13 x 13  (large)
    """

    def __init__(self):
        super().__init__()

        # Top: from C5 (1024 -> 512 tower)
        self.conv_c5 = nn.Sequential(
            ConvBNLeaky(1024, 512, 1),
            ConvBNLeaky(512, 1024, 3),
            ConvBNLeaky(1024, 512, 1),
            ConvBNLeaky(512, 1024, 3),
            ConvBNLeaky(1024, 512, 1),  # output: 512 (P3 features)
        )

        # P3 -> upsample & C4(512)와 concat
        self.conv_p3_to_p2 = ConvBNLeaky(512, 256, 1)  # 512 -> 256
        # concat 후 채널: 256(업샘플) + 512(C4) = 768
        self.conv_p2 = nn.Sequential(
            ConvBNLeaky(768, 256, 1),
            ConvBNLeaky(256, 512, 3),
            ConvBNLeaky(512, 256, 1),
            ConvBNLeaky(256, 512, 3),
            ConvBNLeaky(512, 256, 1),  # output: 256 (P2 features)
        )

        # P2 -> upsample & C3(256)와 concat
        self.conv_p2_to_p1 = ConvBNLeaky(256, 128, 1)  # 256 -> 128
        # concat 후 채널: 128 + 256 = 384
        self.conv_p1 = nn.Sequential(
            ConvBNLeaky(384, 128, 1),
            ConvBNLeaky(128, 256, 3),
            ConvBNLeaky(256, 128, 1),
            ConvBNLeaky(128, 256, 3),
            ConvBNLeaky(256, 128, 1),  # output: 128 (P1 features)
        )

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, c3, c4, c5):
        # top scale (13x13)
        p3 = self.conv_c5(c5)  # B x 512 x 13 x 13

        # middle scale (26x26)
        up_p3 = self.upsample(self.conv_p3_to_p2(p3))  # B x 256 x 26 x 26
        p3_c4 = torch.cat([up_p3, c4], dim=1)          # B x 768 x 26 x 26
        p2 = self.conv_p2(p3_c4)                       # B x 256 x 26 x 26

        # small-object scale (52x52)
        up_p2 = self.upsample(self.conv_p2_to_p1(p2))  # B x 128 x 52 x 52
        p2_c3 = torch.cat([up_p2, c3], dim=1)          # B x 384 x 52 x 52
        p1 = self.conv_p1(p2_c3)                       # B x 128 x 52 x 52

        # 52x52, 26x26, 13x13 feature maps
        return p1, p2, p3


class YoloV3Body(nn.Module):
    """
    YOLOv3 Body = Backbone(Darknet53) + Neck(FPN)

    input:
        x: B x 3 x 416 x 416

    output:
        P1, P2, P3 feature maps (Neck에서 설명한 형태)
    """

    def __init__(self):
        super().__init__()
        self.backbone = Darknet53()
        self.neck = YoloV3Neck()

    def forward(self, x):
        # Darknet53에서 C3, C4, C5 추출
        c3, c4, c5 = self.backbone(x)
        # Neck으로 FPN feature 생성
        p1, p2, p3 = self.neck(c3, c4, c5)
        return p1, p2, p3
