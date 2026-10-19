# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from ..utils import resize


class sSE(nn.Module):
    """Spatial squeeze and excitation."""

    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.conv(x))


class cSE(nn.Module):
    """Channel squeeze and excitation."""

    def __init__(self, in_channels):
        super().__init__()
        mid_channels = max(in_channels // 2, 1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        return x * self.excitation(self.avgpool(x))


class scSE(nn.Module):
    """Concurrent spatial and channel squeeze and excitation."""

    def __init__(self, in_channels):
        super().__init__()
        self.cse = cSE(in_channels)
        self.sse = sSE(in_channels)

    def forward(self, x):
        return self.cse(x) + self.sse(x)


@MODELS.register_module()
class EncoderFeatureAuxHead(BaseDecodeHead):
    """Auxiliary semantic supervision for MIT encoder c2/c3/c4 features.

    The head is intended for training-time deep supervision only. It projects
    c2, c3, and c4 to a shared channel width, aligns them to c2 resolution,
    fuses them, and predicts auxiliary semantic logits.
    """

    def __init__(self, use_scse=False, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        self.use_scse = use_scse

        self.proj_convs = nn.ModuleList()
        for in_channel in self.in_channels:
            self.proj_convs.append(
                ConvModule(
                    in_channels=in_channel,
                    out_channels=self.channels,
                    kernel_size=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.fusion_conv = ConvModule(
            in_channels=self.channels * len(self.in_channels),
            out_channels=self.channels,
            kernel_size=3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.scse = scSE(self.channels) if use_scse else None

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        target_size = inputs[0].shape[2:]

        outs = []
        for idx, x in enumerate(inputs):
            x = self.proj_convs[idx](x)
            if x.shape[2:] != target_size:
                x = resize(
                    input=x,
                    size=target_size,
                    mode='bilinear',
                    align_corners=self.align_corners)
            outs.append(x)

        out = self.fusion_conv(torch.cat(outs, dim=1))
        if self.scse is not None:
            out = self.scse(out)
        return self.cls_seg(out)
