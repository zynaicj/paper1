# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils.freq_fusion import FreqFusion
from mmseg.registry import MODELS
from ..utils import resize


@MODELS.register_module()
class SegformerFreqFusionHead(BaseDecodeHead):
    """SegFormer head with one local FreqFusion insertion.

    The overall SegFormerHead concat-fusion path is kept unchanged. FreqFusion
    only replaces one adjacent-scale interaction, e.g. c4 -> c3, and the fused
    high-resolution feature is then fed back into the normal concat path.
    """

    _FUSION_PAIRS = {
        'c4_c3': (2, 3),
        'c3_c2': (1, 2),
        'c2_c1': (0, 1),
    }

    def __init__(self,
                 interpolate_mode='bilinear',
                 freqfusion_pair='c4_c3',
                 compressed_channels=64,
                 **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        assert len(self.in_channels) == len(self.in_index)
        assert freqfusion_pair in self._FUSION_PAIRS
        self.interpolate_mode = interpolate_mode
        self.freqfusion_pair = freqfusion_pair
        self.compressed_channels = compressed_channels

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

        self.freq_fusion = FreqFusion(
            hr_channels=self.channels,
            lr_channels=self.channels,
            scale_factor=1,
            lowpass_kernel=5,
            highpass_kernel=3,
            up_group=1,
            encoder_kernel=3,
            encoder_dilation=1,
            compressed_channels=compressed_channels,
            align_corners=self.align_corners,
            upsample_mode='nearest',
            feature_resample=False,
            comp_feat_upsample=True,
            use_high_pass=True,
            use_low_pass=True,
            hr_residual=True,
            semi_conv=True,
            hamming_window=True)

        self.fusion_conv = ConvModule(
            in_channels=self.channels * len(self.in_channels),
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

    @staticmethod
    def _resize_like(feat, ref):
        if feat.shape[2:] != ref.shape[2:]:
            feat = F.interpolate(feat, size=ref.shape[2:], mode='nearest')
        return feat

    def forward(self, inputs):
        # Receive 4 stage backbone feature maps: c1, c2, c3, c4.
        inputs = self._transform_inputs(inputs)
        feats = [conv(x) for conv, x in zip(self.convs, inputs)]

        hr_idx, lr_idx = self._FUSION_PAIRS[self.freqfusion_pair]
        _, hr_feat, lr_feat = self.freq_fusion(
            hr_feat=feats[hr_idx], lr_feat=feats[lr_idx])
        lr_feat = self._resize_like(lr_feat, hr_feat)
        feats[hr_idx] = hr_feat + lr_feat

        outs = [
            resize(
                input=feat,
                size=inputs[0].shape[2:],
                mode=self.interpolate_mode,
                align_corners=self.align_corners) for feat in feats
        ]

        out = self.fusion_conv(torch.cat(outs, dim=1))
        out = self.cls_seg(out)
        return out
