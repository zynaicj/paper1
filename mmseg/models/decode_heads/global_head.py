import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead


@MODELS.register_module()
class GlobalSemanticHead(BaseDecodeHead):
    """Global Semantic Auxiliary Head (Module B).

    Operates on deep backbone features (c3, c4) to provide global semantic
    supervision. Complements Module A's BoundaryHead which operates on
    shallow features (c1, c2) for local boundary details.

    Innovation: Shallow-Deep Dual Auxiliary Supervision.
      - Module A (BoundaryHead): c1+c2 → boundary where-to-look
      - Module B (GlobalHead):   c3+c4 → semantic what-it-is

    Pipeline:
      c3 [H/16] + c4 [H/32] → upsample c4 → concat → conv → cls_seg
    """

    def __init__(self, **kwargs):
        super().__init__(
            input_transform='multiple_select',
            **kwargs
        )
        C = self.channels

        self.align_c3 = nn.Conv2d(self.in_channels[0], C, 1)
        self.align_c4 = nn.Conv2d(self.in_channels[1], C, 1)

        self.conv = ConvModule(C * 2, C, 3, padding=1)

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        c3, c4 = inputs

        c4 = F.interpolate(c4, size=c3.shape[2:],
                           mode='bilinear', align_corners=False)

        c3 = self.align_c3(c3)
        c4 = self.align_c4(c4)

        x = torch.cat([c3, c4], dim=1)
        x = self.conv(x)
        x = self.cls_seg(x)
        return x
