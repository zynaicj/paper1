import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead


class LGAG(nn.Module):
    """Large-Kernel Grouped Attention Gate (from EMCAD, CVPR 2024).

    Gates a skip-connection feature using a higher-level signal.
    Args:
        F_g: Gate signal channels.
        F_l: Skip feature channels.
    """

    def __init__(self, F_g, F_l, F_int, kernel_size=3, groups=1):
        super().__init__()
        if kernel_size == 1:
            groups = 1
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size, padding=kernel_size // 2,
                      groups=groups, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size, padding=kernel_size // 2,
                      groups=groups, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


@MODELS.register_module()
class LGAGBoundaryHead(BaseDecodeHead):
    """Boundary Head with LGAG fusion (EMCAD, CVPR 2024).

    c1 (gate) + c2 (skip) → LGAG → gated c2 → concat(c1, gated c2) → cls_seg
    """

    def __init__(self, lgag_kernel=3, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        C = self.channels

        self.align_c1 = nn.Conv2d(self.in_channels[0], C, 1)
        self.align_c2 = nn.Conv2d(self.in_channels[1], C, 1)

        self.lgag = LGAG(F_g=C, F_l=C, F_int=C // 2,
                         kernel_size=lgag_kernel, groups=C // 2)

        self.conv = ConvModule(C * 2, C, 3, padding=1)

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        c1, c2 = inputs

        c2 = F.interpolate(c2, size=c1.shape[2:],
                           mode='bilinear', align_corners=False)

        c1 = self.align_c1(c1)
        c2 = self.align_c2(c2)

        c2 = self.lgag(g=c1, x=c2)

        x = torch.cat([c1, c2], dim=1)
        x = self.conv(x)
        return self.cls_seg(x)

    def loss_by_feat(self, seg_logits, batch_data_samples):
        gt_boundary = []
        for ds in batch_data_samples:
            gt_boundary.append(ds.gt_boundary_map.data)
        gt_boundary = torch.stack(gt_boundary).float()
        seg_logits = F.interpolate(seg_logits, size=gt_boundary.shape[2:],
                                   mode='bilinear', align_corners=False)
        loss = self.loss_decode(seg_logits, gt_boundary.squeeze(1))
        return dict(loss_boundary=loss)
