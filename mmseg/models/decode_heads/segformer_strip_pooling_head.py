# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from ..utils import resize


class StripPooling(nn.Module):
    """Strip pooling context block for decoder feature refinement.

    The block captures square multi-scale context and long horizontal/vertical
    strip context. It returns a residual feature with the same shape as input.
    """

    def __init__(self,
                 in_channels,
                 pool_sizes=(20, 12),
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        assert len(pool_sizes) == 2
        inter_channels = max(in_channels // 4, 1)
        self.pool_sizes = pool_sizes

        self.conv1_1 = ConvModule(
            in_channels,
            inter_channels,
            kernel_size=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv1_2 = ConvModule(
            in_channels,
            inter_channels,
            kernel_size=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.conv2_0 = ConvModule(
            inter_channels,
            inter_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.conv2_1 = ConvModule(
            inter_channels,
            inter_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.conv2_2 = ConvModule(
            inter_channels,
            inter_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.conv2_3 = ConvModule(
            inter_channels,
            inter_channels,
            kernel_size=(1, 3),
            padding=(0, 1),
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.conv2_4 = ConvModule(
            inter_channels,
            inter_channels,
            kernel_size=(3, 1),
            padding=(1, 0),
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.conv2_5 = ConvModule(
            inter_channels,
            inter_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2_6 = ConvModule(
            inter_channels,
            inter_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv3 = ConvModule(
            inter_channels * 2,
            in_channels,
            kernel_size=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=None)

    @staticmethod
    def _resize(feat, size, align_corners):
        return resize(
            input=feat,
            size=size,
            mode='bilinear',
            align_corners=align_corners)

    def forward(self, x, align_corners=False):
        _, _, h, w = x.size()
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)

        context_local = self.conv2_0(x1)
        context_pool1 = self._resize(
            self.conv2_1(F.adaptive_avg_pool2d(x1, self.pool_sizes[0])),
            (h, w), align_corners)
        context_pool2 = self._resize(
            self.conv2_2(F.adaptive_avg_pool2d(x1, self.pool_sizes[1])),
            (h, w), align_corners)

        context_h = self._resize(
            self.conv2_3(F.adaptive_avg_pool2d(x2, (1, w))), (h, w),
            align_corners)
        context_w = self._resize(
            self.conv2_4(F.adaptive_avg_pool2d(x2, (h, 1))), (h, w),
            align_corners)

        branch1 = self.conv2_5(
            F.relu(context_local + context_pool1 + context_pool2, inplace=True))
        branch2 = self.conv2_6(F.relu(context_h + context_w, inplace=True))
        return self.conv3(torch.cat([branch1, branch2], dim=1))


class ResidualStripPooling(nn.Module):
    """Residual wrapper that starts as an identity mapping."""

    def __init__(self,
                 channels,
                 pool_sizes=(20, 12),
                 init_gamma=0.0,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        self.strip_pooling = StripPooling(
            channels, pool_sizes=pool_sizes, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.gamma = nn.Parameter(torch.tensor(float(init_gamma)))

    def forward(self, x, align_corners=False):
        residual = self.strip_pooling(x, align_corners=align_corners)
        return F.relu(x + self.gamma * residual, inplace=True)


@MODELS.register_module()
class SegformerStripPoolingHead(BaseDecodeHead):
    """SegFormer head with residual strip-pooling enhancement.

    It keeps SegFormerHead's original multi-scale feature fusion and inserts a
    residual strip-pooling block before the final classifier. This is intended
    to complement boundary supervision with long-range regional context.
    """

    def __init__(self,
                 interpolate_mode='bilinear',
                 strip_pool_sizes=(20, 12),
                 strip_init_gamma=0.0,
                 **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)
        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        for in_channel in self.in_channels:
            self.convs.append(
                ConvModule(
                    in_channels=in_channel,
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

        self.strip_pooling = ResidualStripPooling(
            self.channels,
            pool_sizes=strip_pool_sizes,
            init_gamma=strip_init_gamma,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, inputs):
        # Receive 4 stage backbone feature maps: 1/4, 1/8, 1/16, 1/32.
        inputs = self._transform_inputs(inputs)
        outs = []
        for idx, x in enumerate(inputs):
            outs.append(
                resize(
                    input=self.convs[idx](x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out = self.fusion_conv(torch.cat(outs, dim=1))
        out = self.strip_pooling(out, align_corners=self.align_corners)
        out = self.cls_seg(out)
        return out
